//! Metal command buffer recording.
//!
//! Maps the facade's command interface onto a `MTLCommandBuffer` plus a current
//! `MTLRenderCommandEncoder`. Presentation is deferred: `transition_to_present`
//! stashes the drawable and [`MetalCommandBuffer::commit`] (called from the queue
//! submit) records `presentDrawable` before committing.
//!
//! M0 implements the clear path (begin → begin_rendering(clear) → end_rendering →
//! present → submit). Draw / pipeline / resource methods land in M2+.

use std::cell::{Cell, RefCell};
use std::ptr::NonNull;
use std::rc::Rc;

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLEvent;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePassDescriptor, MTLDevice, MTLFence,
    MTLIndexType, MTLLoadAction, MTLOrigin, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassColorAttachmentDescriptor, MTLRenderPassDepthAttachmentDescriptor,
    MTLRenderPassDescriptor, MTLRenderStages, MTLResource, MTLResourceUsage, MTLScissorRect,
    MTLSize, MTLStoreAction, MTLTexture, MTLViewport,
};
use objc2_quartz_core::CAMetalDrawable;
use rhi_types::{ClearColor, Extent2D, Rect2D};

use crate::device::DeviceShared;
use crate::resources::{
    BINDLESS_BUFFER_INDEX, BINDLESS_BUFFER_INDEX_WITH_GLOBALS, GLOBALS_BUFFER_INDEX, MetalBuffer,
    MetalComputePipeline, MetalCubemap, MetalDepthBuffer, MetalGraphicsPipeline, MetalRenderTarget,
    MetalStorageBuffer, PUSH_CONSTANT_INDEX, VERTEX_BUFFER_INDEX,
};
use crate::rt_pipeline::MetalRaytracingPipeline;
use crate::swapchain::MetalSwapchain;
use crate::{Result, rhi_err};

/// A bound index buffer plus its width flag (`true` = 32-bit indices).
type BoundIndexBuffer = (Retained<ProtocolObject<dyn MTLBuffer>>, bool);

pub struct MetalCommandBuffer {
    shared: Rc<DeviceShared>,
    /// The queue this buffer records onto (graphics or the dedicated compute queue).
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    cmd: RefCell<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    encoder: RefCell<Option<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>>>,
    /// The current compute command encoder (a compute pass / dispatch). Only one of
    /// `encoder` / `compute_encoder` is ever open at a time.
    compute_encoder: RefCell<Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>>,
    present: RefCell<Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>>,
    /// Index buffer + width bound by `bind_index_buffer`; consumed by
    /// `draw_indexed` (Metal takes the index buffer at draw time, not bind time).
    index_buffer: RefCell<Option<BoundIndexBuffer>>,
    /// Byte offset into the globals UBO selected by `set_globals`; applied when a
    /// `uses_globals` pipeline is bound.
    globals_offset: Cell<u32>,
    /// Threadgroup size of the bound compute pipeline, used by `dispatch` to turn
    /// threadgroup counts into `dispatchThreadgroups:threadsPerThreadgroup:`.
    pipeline_threads: Cell<MTLSize>,
    rt_push_constants: RefCell<Vec<u8>>,
    /// Cross-encoder hazard fence. Metal only auto-tracks read-after-write hazards for
    /// resources reached *directly* by an encoder; a transient written as a render
    /// attachment (e.g. the G-buffer depth) and then read in a *compute* pass through the
    /// bindless argument buffer is NOT auto-synchronized, so the compute pass can sample the
    /// freshly-cleared texture (the cause of the Sponza GI/reflection temporal shimmer on
    /// Metal). We serialize the main-queue encoder chain explicitly: every encoder updates
    /// this fence on close and waits on it at open, mirroring the explicit barriers the
    /// Vulkan/D3D12 backends already emit. One fence is reused; `fence_pending` gates the
    /// first encoder of each command buffer (nothing to wait on yet).
    fence: Retained<ProtocolObject<dyn MTLFence>>,
    fence_pending: Cell<bool>,
}

impl MetalCommandBuffer {
    pub(crate) fn new(
        shared: Rc<DeviceShared>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Self {
        let fence = shared
            .device
            .newFence()
            .expect("MTLDevice::newFence returned nil");
        Self {
            shared,
            queue,
            fence,
            fence_pending: Cell::new(false),
            cmd: RefCell::new(None),
            encoder: RefCell::new(None),
            compute_encoder: RefCell::new(None),
            present: RefCell::new(None),
            index_buffer: RefCell::new(None),
            globals_offset: Cell::new(0),
            pipeline_threads: Cell::new(MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            }),
            rt_push_constants: RefCell::new(Vec::new()),
        }
    }

    /// End whichever encoder (render or compute) is currently open. Compute passes
    /// have no explicit `end_rendering`, so the next encoder start closes them.
    ///
    /// Before ending, the encoder updates the cross-encoder hazard [`Self::fence`] so the
    /// next encoder can wait on its writes (see the `fence` field). `fence_pending` is set
    /// whenever a real encoder closed, so the *first* encoder of a command buffer (which has
    /// nothing to wait for) skips the wait.
    fn end_any_encoder(&self) {
        if let Some(enc) = self.encoder.borrow_mut().take() {
            enc.updateFence_afterStages(
                &self.fence,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
            enc.endEncoding();
            self.fence_pending.set(true);
        }
        if let Some(enc) = self.compute_encoder.borrow_mut().take() {
            enc.updateFence(&self.fence);
            enc.endEncoding();
            self.fence_pending.set(true);
        }
    }

    /// Wait the hazard fence on a freshly opened render encoder (before any stage runs), so
    /// it sees the writes of every prior encoder in this command buffer. No-op for the first
    /// encoder (nothing recorded the fence yet).
    fn wait_fence_render(&self, enc: &ProtocolObject<dyn MTLRenderCommandEncoder>) {
        if self.fence_pending.get() {
            enc.waitForFence_beforeStages(
                &self.fence,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
        }
    }

    /// Wait the hazard fence on a freshly opened compute encoder.
    fn wait_fence_compute(&self, enc: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        if self.fence_pending.get() {
            enc.waitForFence(&self.fence);
        }
    }

    /// Create a render command encoder for `pass` and stash it as the current one.
    fn start_encoder(&self, pass: &MTLRenderPassDescriptor) {
        self.end_any_encoder();
        let cmd = self.cmd.borrow();
        let cmd = cmd.as_ref().expect("begin_rendering* without begin");
        let enc = cmd
            .renderCommandEncoderWithDescriptor(pass)
            .expect("failed to create render command encoder");
        self.wait_fence_render(&enc);
        *self.encoder.borrow_mut() = Some(enc);
    }

    pub fn begin(&self) -> Result<()> {
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| rhi_err("commandBuffer() returned nil"))?;
        *self.cmd.borrow_mut() = Some(cb);
        *self.encoder.borrow_mut() = None;
        *self.compute_encoder.borrow_mut() = None;
        *self.present.borrow_mut() = None;
        *self.index_buffer.borrow_mut() = None;
        self.rt_push_constants.borrow_mut().clear();
        // Fresh command buffer: the fence carries no in-buffer producer yet.
        self.fence_pending.set(false);
        Ok(())
    }

    pub fn end(&self) -> Result<()> {
        self.end_any_encoder();
        Ok(())
    }

    /// Timestamp query recording (Phase 9 M1).
    ///
    /// `reset_queries` / `resolve_queries` are no-ops on Metal: an
    /// `MTLCounterSampleBuffer` needs no per-frame reset (unlike a Vulkan query
    /// pool), and it is resolved on the *host* via `resolveCounterRange` (see
    /// [`crate::query::MetalQueryHeap::read`]), not with a GPU resolve command
    /// (unlike D3D12's `ResolveQueryData`).
    pub fn reset_queries(&self, _heap: &crate::query::MetalQueryHeap, _first: u32, _count: u32) {}
    pub fn resolve_queries(&self, _heap: &crate::query::MetalQueryHeap, _count: u32) {}

    /// Sample a GPU timestamp into slot `index` of the heap's counter sample
    /// buffer, at this point in the command stream.
    ///
    /// The render graph calls this at pass *boundaries* — between encoders, when
    /// none is open. Apple-family GPUs support only **stage-boundary** counter
    /// sampling (not blit/dispatch mid-encoder `sampleCountersInBuffer`), so we
    /// open a tiny **empty compute pass** whose sample-buffer attachment records
    /// slot `index` at its *start-of-encoder* stage boundary, then close it
    /// immediately (no dispatch). That start boundary is the current point in the
    /// command stream, giving the same "timestamp at a point" semantics as the
    /// Vulkan `vkCmdWriteTimestamp` the render graph brackets passes with:
    /// successive samples bracket the encoders (passes) recorded between them.
    ///
    /// We first close (and fence) any open encoder, and the new pass waits the
    /// hazard fence, so the boundary orders after all prior encoders' work (the
    /// empty encoder's start boundary won't fire until the GPU reaches it).
    ///
    /// No-op when the device lacks a timestamp counter set (heap has no sample
    /// buffer): the profiler then reads 0 ticks, exactly as before.
    pub fn write_timestamp(&self, heap: &crate::query::MetalQueryHeap, index: u32) {
        let Some(sample_buffer) = heap.sample_buffer() else {
            return;
        };
        self.end_any_encoder();

        // Compute pass descriptor: sample the timestamp at the start-of-encoder
        // stage boundary into slot `index`; omit the end sample (DONT_SAMPLE).
        let desc = MTLComputePassDescriptor::new();
        let attachments = desc.sampleBufferAttachments();
        // SAFETY: index 0 is a valid attachment slot on the array; `index` is within
        // the heap's `count` slots (the render graph writes indices `0..=passes`);
        // `MTLCounterDontSample` (usize::MAX) omits the end-of-encoder sample.
        let attach = unsafe { attachments.objectAtIndexedSubscript(0) };
        attach.setSampleBuffer(Some(sample_buffer));
        unsafe {
            attach.setStartOfEncoderSampleIndex(index as usize);
            attach.setEndOfEncoderSampleIndex(crate::query::COUNTER_DONT_SAMPLE);
        }

        let cmd = self.cmd.borrow();
        let cmd = cmd.as_ref().expect("write_timestamp without begin");
        let enc = cmd
            .computeCommandEncoderWithDescriptor(&desc)
            .expect("failed to create timestamp compute encoder");
        if self.fence_pending.get() {
            enc.waitForFence(&self.fence);
        }
        // No dispatch: the start-of-encoder stage boundary is the sample point.
        enc.endEncoding();
    }

    /// Debug-marker regions (Phase 9 M2) — no-ops on the Metal stub (Metal uses
    /// per-encoder `pushDebugGroup`, not a command-buffer-level label).
    pub fn begin_debug_label(&self, _name: &str) {}
    pub fn end_debug_label(&self) {}

    /// End any open encoder, signal `event` to `value` (cross-queue ordering for
    /// async compute), and commit. Used by [`crate::device::MetalComputeQueue`].
    pub(crate) fn commit_signaling(
        &self,
        event: &ProtocolObject<dyn MTLEvent>,
        value: u64,
    ) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        self.end_any_encoder();
        let cb = self
            .cmd
            .borrow_mut()
            .take()
            .expect("commit_signaling() without begin()");
        cb.encodeSignalEvent_value(event, value);
        cb.commit();
        cb
    }

    /// Commit the recorded work (ending any open encoder and recording the
    /// deferred drawable present), returning the committed command buffer so a
    /// fence can block on it. Called from the queue submit paths.
    pub(crate) fn commit(&self) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        self.end_any_encoder();
        let cb = self
            .cmd
            .borrow_mut()
            .take()
            .expect("commit() without begin()");
        if let Some(drawable) = self.present.borrow_mut().take() {
            // presentDrawable takes id<MTLDrawable>; pass the drawable object
            // pointer directly (ObjC ignores the Rust protocol type).
            let _: () = unsafe { msg_send![&*cb, presentDrawable: &*drawable] };
        }
        cb.commit();
        cb
    }

    // ---- Swapchain transitions / present ----------------------------------

    pub fn transition_to_render_target(&self, _swapchain: &MetalSwapchain, _image_index: u32) {
        // Drawable acquisition handles the transition on Metal.
    }

    pub fn transition_to_present(&self, swapchain: &MetalSwapchain, _image_index: u32) {
        *self.present.borrow_mut() = swapchain.take_current_drawable();
    }

    // ---- Render passes -----------------------------------------------------

    pub fn begin_rendering(
        &self,
        swapchain: &MetalSwapchain,
        _image_index: u32,
        color_clear: Option<ClearColor>,
        depth: Option<&MetalDepthBuffer>,
    ) {
        let drawable = swapchain
            .current_drawable()
            .expect("acquire_next_image before begin_rendering");
        let texture = drawable.texture();

        let pass = MTLRenderPassDescriptor::new();
        let attachments = pass.colorAttachments();
        let attach = unsafe { attachments.objectAtIndexedSubscript(0) };
        attach.setTexture(Some(&texture));
        match color_clear {
            Some(c) => {
                attach.setLoadAction(MTLLoadAction::Clear);
                attach.setClearColor(MTLClearColor {
                    red: c.r as f64,
                    green: c.g as f64,
                    blue: c.b as f64,
                    alpha: c.a as f64,
                });
            }
            None => attach.setLoadAction(MTLLoadAction::Load),
        }
        attach.setStoreAction(MTLStoreAction::Store);

        // Depth attachment: clear to far (1.0) and discard after the pass (M3 does
        // not sample the depth buffer; the shadow pass keeps it in M4).
        if let Some(depth) = depth {
            let da = pass.depthAttachment();
            da.setTexture(Some(&depth.texture));
            da.setLoadAction(MTLLoadAction::Clear);
            da.setClearDepth(1.0);
            da.setStoreAction(MTLStoreAction::DontCare);
        }

        self.end_any_encoder();
        let cmd = self.cmd.borrow();
        let cmd = cmd.as_ref().expect("begin_rendering without begin");
        let enc = cmd
            .renderCommandEncoderWithDescriptor(&pass)
            .expect("failed to create render command encoder");
        self.wait_fence_render(&enc);
        *self.encoder.borrow_mut() = Some(enc);
    }

    pub fn end_rendering(&self) {
        // Route through end_any_encoder so the render encoder records the hazard fence (a
        // following compute pass that samples this pass's attachments must wait on it).
        self.end_any_encoder();
    }

    pub fn set_viewport_scissor(&self, swapchain: &MetalSwapchain) {
        self.set_viewport_scissor_extent(swapchain.extent_2d());
    }

    pub fn set_viewport_scissor_extent(&self, extent: Extent2D) {
        if let Some(enc) = self.encoder.borrow().as_ref() {
            enc.setViewport(MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: extent.width as f64,
                height: extent.height as f64,
                znear: 0.0,
                zfar: 1.0,
            });
            enc.setScissorRect(MTLScissorRect {
                x: 0,
                y: 0,
                width: extent.width as usize,
                height: extent.height as usize,
            });
        }
    }

    /// Set the viewport + scissor to an arbitrary sub-rect of the bound target
    /// (shadow-atlas tiling: each cascade / light slot renders into its own tile).
    pub fn set_viewport_scissor_rect(&self, rect: Rect2D) {
        if let Some(enc) = self.encoder.borrow().as_ref() {
            enc.setViewport(MTLViewport {
                originX: rect.x as f64,
                originY: rect.y as f64,
                width: rect.width as f64,
                height: rect.height as f64,
                znear: 0.0,
                zfar: 1.0,
            });
            enc.setScissorRect(MTLScissorRect {
                x: rect.x as usize,
                y: rect.y as usize,
                width: rect.width as usize,
                height: rect.height as usize,
            });
        }
    }

    // ---- Offscreen render targets / MRT / shadow / cubemaps (M4) -----------

    /// Begin rendering into one offscreen color target (+ optional depth). `depth_clear`
    /// clears the depth (first writer) else loads it. Depth is always **stored**: Metal
    /// tile memory discards a `DontCare` depth, so the G-buffer depth the deferred SW-RT
    /// passes sample (GDF AO / GI / reflections, which reconstruct from it) must be kept;
    /// storing is safe under transient aliasing (the store lands at pass-end in memory the
    /// plan has already consumed or has yet to rewrite, never a live resource).
    pub fn begin_rendering_target(
        &self,
        target: &MetalRenderTarget,
        color_clear: Option<ClearColor>,
        depth: Option<&MetalDepthBuffer>,
        depth_clear: bool,
    ) {
        let pass = MTLRenderPassDescriptor::new();
        let attach = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        attach.setTexture(Some(&target.texture));
        config_color(&attach, color_clear);
        if let Some(d) = depth {
            config_depth(
                &pass.depthAttachment(),
                &d.texture,
                depth_clear,
                MTLStoreAction::Store,
            );
        }
        self.start_encoder(&pass);
    }

    /// Begin rendering into N offscreen color targets (MRT, e.g. the 4-attachment
    /// G-buffer) + optional depth. Each target clears if `Some`, else loads; `depth_clear`
    /// likewise clears (first writer) vs loads the depth, which is **stored** (see
    /// [`Self::begin_rendering_target`]).
    pub fn begin_rendering_targets(
        &self,
        targets: &[(&MetalRenderTarget, Option<ClearColor>)],
        depth: Option<&MetalDepthBuffer>,
        depth_clear: bool,
    ) {
        let pass = MTLRenderPassDescriptor::new();
        let attachments = pass.colorAttachments();
        for (i, (target, clear)) in targets.iter().enumerate() {
            let attach = unsafe { attachments.objectAtIndexedSubscript(i) };
            attach.setTexture(Some(&target.texture));
            config_color(&attach, *clear);
        }
        if let Some(d) = depth {
            config_depth(
                &pass.depthAttachment(),
                &d.texture,
                depth_clear,
                MTLStoreAction::Store,
            );
        }
        self.start_encoder(&pass);
    }

    /// Select the per-frame globals byte offset for the next `uses_globals` bind.
    pub fn set_globals(&self, offset: u32) {
        self.globals_offset.set(offset);
    }

    /// Begin a depth-only pass into a shadow map: no color attachments, depth
    /// cleared and **stored** so the lighting pass can sample it.
    pub fn begin_rendering_depth_only(&self, depth: &MetalDepthBuffer) {
        let pass = MTLRenderPassDescriptor::new();
        config_depth(
            &pass.depthAttachment(),
            &depth.texture,
            true, // shadow map: clear then store for the lighting pass to sample
            MTLStoreAction::Store,
        );
        self.start_encoder(&pass);
    }

    /// Render-graph transition hooks. On Metal these toggle bindless residency:
    /// `*_to_render_target` drops a resource from the resident set before it is
    /// written as an attachment; `*_to_sampled` makes it resident before a sampling
    /// pass. Hazards across encoders (write→sample, aliasing) are tracked by Metal
    /// automatically (tracked placement heap), so no explicit barrier is needed.
    pub fn depth_to_render_target(&self, depth: &MetalDepthBuffer) {
        self.shared.set_resident(&depth.texture, false);
    }
    pub fn depth_to_sampled(&self, depth: &MetalDepthBuffer) {
        self.shared.set_resident(&depth.texture, true);
    }
    pub fn cube_to_color(&self, cube: &MetalCubemap) {
        self.shared.set_resident(&cube.texture, false);
    }
    pub fn cube_to_sampled(&self, cube: &MetalCubemap) {
        self.shared.set_resident(&cube.texture, true);
    }

    /// Begin rendering into one (face, mip) of a cubemap (color only). Metal selects
    /// the subresource via the attachment's slice (face) + level (mip), so no
    /// per-(face, mip) view is needed (unlike Vulkan).
    pub fn begin_rendering_cube_face(
        &self,
        cube: &MetalCubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
    ) {
        let pass = MTLRenderPassDescriptor::new();
        let attach = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        attach.setTexture(Some(&cube.texture));
        attach.setSlice(face as usize);
        attach.setLevel(mip as usize);
        config_color(&attach, clear);
        self.start_encoder(&pass);
    }

    /// Begin rendering into one (face, mip) of a cubemap **with a depth buffer**
    /// (for capturing scene geometry with correct occlusion). Depth is cleared and
    /// discarded.
    pub fn begin_rendering_cube_face_depth(
        &self,
        cube: &MetalCubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
        depth: &MetalDepthBuffer,
    ) {
        let pass = MTLRenderPassDescriptor::new();
        let attach = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        attach.setTexture(Some(&cube.texture));
        attach.setSlice(face as usize);
        attach.setLevel(mip as usize);
        config_color(&attach, clear);
        config_depth(
            &pass.depthAttachment(),
            &depth.texture,
            true, // cube-face capture: clear depth; not sampled afterwards
            MTLStoreAction::DontCare,
        );
        self.start_encoder(&pass);
    }

    /// Blit the current drawable's texture into `buffer` (tightly packed BGRA8) so
    /// the host can read the rendered frame back. Recorded onto this frame's command
    /// buffer; the readback is valid once the buffer completes (the submit fence).
    /// Requires the layer to be non-`framebufferOnly` (see `swapchain.rs`).
    pub fn copy_swapchain_to_buffer(
        &self,
        swapchain: &MetalSwapchain,
        _image_index: u32,
        buffer: &MetalBuffer,
    ) {
        // The render graph transitions the backbuffer to present before the app
        // optionally records its screenshot blit. At that point ownership has
        // already moved out of the swapchain into `self.present`, so accept either
        // location without adding a long-lived second owner.
        let drawable = swapchain
            .current_drawable()
            .or_else(|| self.present.borrow().clone())
            .expect("copy_swapchain_to_buffer without an acquired drawable");
        let texture = drawable.texture();
        // Close (and fence) any open encoder so this blit sees the render pass that drew the
        // swapchain image it is copying.
        self.end_any_encoder();
        let cmd = self.cmd.borrow();
        let cmd = cmd
            .as_ref()
            .expect("copy_swapchain_to_buffer without begin");
        let blit = cmd
            .blitCommandEncoder()
            .expect("failed to create blit command encoder");
        if self.fence_pending.get() {
            blit.waitForFence(&self.fence);
        }
        let width = texture.width();
        let height = texture.height();
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &texture,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width, height, depth: 1 },
                &buffer.buffer,
                0,
                width * 4,
                width * height * 4,
            );
        }
        blit.endEncoding();
    }

    pub fn rt_to_render_target(&self, target: &MetalRenderTarget) {
        self.shared.set_resident(&target.texture, false);
    }
    pub fn rt_to_sampled(&self, target: &MetalRenderTarget) {
        self.shared.set_resident(&target.texture, true);
    }
    pub fn aliasing_barrier(&self, target: &MetalRenderTarget) {
        // The target is about to be written; drop it from the resident set (tracked
        // placement heap handles the aliasing hazard automatically).
        self.shared.set_resident(&target.texture, false);
    }

    pub fn bind_graphics_pipeline(&self, pipeline: &MetalGraphicsPipeline) {
        if let Some(enc) = self.encoder.borrow().as_ref() {
            enc.setRenderPipelineState(&pipeline.state);
            if let Some(ds) = pipeline.depth_stencil.as_ref() {
                enc.setDepthStencilState(Some(ds));
            }
            // Per-frame globals UBO (camera/lights/shadow/IBL) for the PBR lighting
            // pass: bound at GLOBALS_BUFFER_INDEX with the offset from `set_globals`.
            if pipeline.uses_globals
                && let Some(globals) = self.shared.globals_buffer()
            {
                let offset = self.globals_offset.get() as usize;
                unsafe {
                    enc.setVertexBuffer_offset_atIndex(
                        Some(&globals),
                        offset,
                        GLOBALS_BUFFER_INDEX,
                    );
                    enc.setFragmentBuffer_offset_atIndex(
                        Some(&globals),
                        offset,
                        GLOBALS_BUFFER_INDEX,
                    );
                }
            }
            if pipeline.bindless {
                // The globals UBO (if any) sits at buffer(1), so the bindless block
                // shifts to buffer(2) for `uses_globals` pipelines; otherwise it is
                // at buffer(1). Bound to both stages (the vertex stage may index it).
                let bindless_index = if pipeline.uses_globals {
                    BINDLESS_BUFFER_INDEX_WITH_GLOBALS
                } else {
                    BINDLESS_BUFFER_INDEX
                };
                unsafe {
                    enc.setVertexBuffer_offset_atIndex(
                        Some(&self.shared.arg_buffer),
                        0,
                        bindless_index,
                    );
                    enc.setFragmentBuffer_offset_atIndex(
                        Some(&self.shared.arg_buffer),
                        0,
                        bindless_index,
                    );
                }
                // Resources referenced indirectly through an argument buffer must be
                // made resident explicitly, or the GPU may not have them mapped.
                for tex in self.shared.resident_textures().iter() {
                    let res: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**tex);
                    enc.useResource_usage_stages(
                        res,
                        MTLResourceUsage::Read,
                        MTLRenderStages::Fragment,
                    );
                }
                // Storage buffers the vertex stage pulls from (particle / cull draw
                // read `g.storage_buffers[i]` in `vsMain`); compute wrote them.
                for buf in self.shared.storage_buffers().iter() {
                    let res: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**buf);
                    enc.useResource_usage_stages(
                        res,
                        MTLResourceUsage::Read,
                        MTLRenderStages::Vertex | MTLRenderStages::Fragment,
                    );
                }
            }
        }
    }

    pub fn draw(&self, vertex_count: u32, instance_count: u32) {
        if let Some(enc) = self.encoder.borrow().as_ref() {
            unsafe {
                enc.drawPrimitives_vertexStart_vertexCount_instanceCount(
                    MTLPrimitiveType::Triangle,
                    0,
                    vertex_count as usize,
                    instance_count as usize,
                );
            }
        }
    }

    /// Begin a compute dispatch: end any open encoder, open a fresh compute command
    /// encoder, set the pipeline, and (for bindless pipelines) bind the argument
    /// buffer + make the storage resources resident. One encoder per
    /// `bind_compute_pipeline` call means consecutive compute passes (e.g. cull
    /// reset → cull) sit on separate encoders, so Metal's automatic hazard tracking
    /// orders their reads/writes (the `storage_buffer_*` barriers stay no-ops).
    /// `threads_per_group` is stashed for the following `dispatch`.
    pub fn bind_compute_pipeline(&self, pipeline: &MetalComputePipeline) {
        self.end_any_encoder();
        let enc = {
            let cmd = self.cmd.borrow();
            let cmd = cmd.as_ref().expect("bind_compute_pipeline without begin");
            cmd.computeCommandEncoder()
                .expect("failed to create compute command encoder")
        };
        self.wait_fence_compute(&enc);
        enc.setComputePipelineState(&pipeline.state);
        // Per-frame globals UBO (Stage C7 SSR reprojection reads `globals.prev_view_proj`):
        // bound at GLOBALS_BUFFER_INDEX with the `set_globals` offset, mirroring the
        // graphics path. The bindless block then shifts past it to buffer(2).
        if pipeline.uses_globals
            && let Some(globals) = self.shared.globals_buffer()
        {
            unsafe {
                enc.setBuffer_offset_atIndex(
                    Some(&globals),
                    self.globals_offset.get() as usize,
                    GLOBALS_BUFFER_INDEX,
                );
            }
        }
        if pipeline.bindless {
            // The globals UBO (if any) sits at buffer(1), so the bindless argument
            // buffer shifts to buffer(2) for `uses_globals` pipelines; otherwise it
            // is at buffer(1) — identical to the graphics convention.
            let bindless_index = if pipeline.uses_globals {
                BINDLESS_BUFFER_INDEX_WITH_GLOBALS
            } else {
                BINDLESS_BUFFER_INDEX
            };
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.shared.arg_buffer), 0, bindless_index);
            }
            // Make argument-buffer-referenced resources resident: sampled inputs
            // (Read), storage images being written (Read | Write), and the storage
            // buffers (Read | Write).
            for tex in self.shared.resident_textures().iter() {
                let res: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**tex);
                enc.useResource_usage(res, MTLResourceUsage::Read);
            }
            for tex in self.shared.storage_resident_textures().iter() {
                let res: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**tex);
                enc.useResource_usage(res, MTLResourceUsage::Read | MTLResourceUsage::Write);
            }
            for buf in self.shared.storage_buffers().iter() {
                let res: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**buf);
                enc.useResource_usage(res, MTLResourceUsage::Read | MTLResourceUsage::Write);
            }
            // Acceleration structures traced through the bindless `g.tlas` (the
            // inline path tracer): the TLAS *and* every referenced BLAS must be
            // resident, since they are reached indirectly via the argument buffer.
            for accel in self.shared.rt_acceleration_structures().iter() {
                let res: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**accel);
                enc.useResource_usage(res, MTLResourceUsage::Read);
            }
        }
        self.pipeline_threads.set(pipeline.threads_per_group);
        *self.compute_encoder.borrow_mut() = Some(enc);
    }

    /// Dispatch `x * y * z` threadgroups of the bound pipeline's threadgroup size.
    pub fn dispatch(&self, x: u32, y: u32, z: u32) {
        if let Some(enc) = self.compute_encoder.borrow().as_ref() {
            let groups = MTLSize {
                width: x as usize,
                height: y as usize,
                depth: z as usize,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(groups, self.pipeline_threads.get());
        }
    }

    /// Upload compute push constants at [`PUSH_CONSTANT_INDEX`] (padded up to 16,
    /// like the graphics path).
    pub fn push_constants_compute(&self, data: &[u8]) {
        if let Some(enc) = self.compute_encoder.borrow().as_ref() {
            let mut buf = [0u8; 256];
            let len = data.len();
            assert!(
                len <= buf.len(),
                "compute push constant block too large for Metal"
            );
            buf[..len].copy_from_slice(data);
            let padded = (len + 15) & !15;
            let ptr = NonNull::new(buf.as_ptr() as *mut std::ffi::c_void)
                .expect("push_constants_compute data pointer is null");
            unsafe {
                enc.setBytes_length_atIndex(ptr, padded, PUSH_CONSTANT_INDEX);
            }
        }
    }

    /// Bind a Metal Shader Converter RT pipeline. The converter raygen is a compute
    /// kernel, so this opens a compute encoder just like [`Self::bind_compute_pipeline`].
    pub fn bind_raytracing_pipeline(&self, pipeline: &MetalRaytracingPipeline) {
        self.end_any_encoder();
        let enc = {
            let cmd = self.cmd.borrow();
            let cmd = cmd
                .as_ref()
                .expect("bind_raytracing_pipeline without begin");
            cmd.computeCommandEncoder()
                .expect("failed to create RT compute command encoder")
        };
        self.wait_fence_compute(&enc);
        enc.setComputePipelineState(pipeline.state());
        self.pipeline_threads.set(pipeline.threads_per_group());
        *self.compute_encoder.borrow_mut() = Some(enc);
    }

    pub fn push_constants_rt(&self, data: &[u8]) {
        let mut push = self.rt_push_constants.borrow_mut();
        push.clear();
        push.extend_from_slice(data);
    }

    pub fn trace_rays(&self, pipeline: &MetalRaytracingPipeline, width: u32, height: u32) {
        if let Some(enc) = self.compute_encoder.borrow().as_ref() {
            pipeline
                .encode_dispatch(
                    &self.shared,
                    enc,
                    &self.rt_push_constants.borrow(),
                    width,
                    height,
                )
                .expect("Metal trace_rays encode failed");
        }
    }

    /// Render-graph storage transitions. On a single queue Metal tracks the
    /// compute-write → graphics-read and compute → compute hazards across encoder
    /// boundaries automatically, so the buffer barriers are no-ops; the storage
    /// *image* hooks only need to toggle which residency set the target is in
    /// (sampled `Read` vs UAV `Read | Write`).
    pub fn rt_to_storage(&self, target: &MetalRenderTarget) {
        self.shared.set_resident(&target.texture, false);
        self.shared.set_storage_resident(&target.texture, true);
    }
    pub fn storage_to_sampled(&self, target: &MetalRenderTarget) {
        self.shared.set_storage_resident(&target.texture, false);
        self.shared.set_resident(&target.texture, true);
    }
    /// Phase 11 Stage B volume transitions. Like the 2D storage hooks above, a 3D
    /// volume is both compute-written (the SDF bake / GDF merge) and sampled (the SW
    /// ray marcher), so residency toggles between the UAV `Read | Write` set and the
    /// sampled `Read` set; Metal's encoder-boundary hazard tracking orders the
    /// write → sample on the single queue, so there is no explicit barrier.
    pub fn volume_to_storage(&self, volume: &crate::resources::MetalVolume) {
        self.shared.set_resident(&volume.texture, false);
        self.shared.set_storage_resident(&volume.texture, true);
    }
    pub fn volume_to_sampled(&self, volume: &crate::resources::MetalVolume) {
        self.shared.set_storage_resident(&volume.texture, false);
        self.shared.set_resident(&volume.texture, true);
    }
    pub fn storage_buffer_barrier(&self, _buffer: &MetalStorageBuffer) {}
    pub fn storage_buffer_to_indirect(&self, _buffer: &MetalStorageBuffer) {}
    pub fn storage_buffer_to_storage(&self, _buffer: &MetalStorageBuffer) {}

    /// Issue `draw_count` indexed draws sourced from `buffer` at `offset`. Metal's
    /// indirect-args struct (`MTLDrawIndexedPrimitivesIndirectArguments`) matches
    /// the Vulkan / D3D12 5×u32 layout the cull compute shader writes; one Metal
    /// call draws one command, so loop for `draw_count > 1`.
    pub fn draw_indexed_indirect(&self, buffer: &MetalStorageBuffer, offset: u64, draw_count: u32) {
        let enc = self.encoder.borrow();
        let Some(enc) = enc.as_ref() else { return };
        let index_buffer = self.index_buffer.borrow();
        let (ibuf, wide) = index_buffer
            .as_ref()
            .expect("draw_indexed_indirect without bind_index_buffer");
        let index_type = if *wide {
            MTLIndexType::UInt32
        } else {
            MTLIndexType::UInt16
        };
        for i in 0..draw_count as u64 {
            unsafe {
                enc.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                    MTLPrimitiveType::Triangle,
                    index_type,
                    ibuf,
                    0,
                    &buffer.buffer,
                    (offset + i * 20) as usize,
                );
            }
        }
    }

    pub fn set_scissor(&self, rect: Rect2D) {
        if let Some(enc) = self.encoder.borrow().as_ref() {
            enc.setScissorRect(MTLScissorRect {
                x: rect.x.max(0) as usize,
                y: rect.y.max(0) as usize,
                width: rect.width as usize,
                height: rect.height as usize,
            });
        }
    }

    /// Bind the vertex buffer at [`VERTEX_BUFFER_INDEX`]. `stride` is unused (the
    /// pipeline's vertex descriptor carries it) — present for facade parity.
    pub fn bind_vertex_buffer(&self, buffer: &MetalBuffer, _stride: u32) {
        if let Some(enc) = self.encoder.borrow().as_ref() {
            unsafe {
                enc.setVertexBuffer_offset_atIndex(Some(&buffer.buffer), 0, VERTEX_BUFFER_INDEX);
            }
        }
    }

    /// Stash the index buffer for the next `draw_indexed` (`wide` = 32-bit indices).
    pub fn bind_index_buffer(&self, buffer: &MetalBuffer, wide: bool) {
        *self.index_buffer.borrow_mut() = Some((buffer.buffer.clone(), wide));
    }

    /// Upload push constants to both stages at [`PUSH_CONSTANT_INDEX`]. Setting the
    /// fragment slot too is harmless when only the vertex stage declares the block.
    ///
    /// Metal validates `setBytes` length against the shader argument's
    /// *alignment-padded* size: e.g. the ImGui `{float2, float2, uint}` block is 20
    /// bytes of data but Metal rounds it to 24 (8-byte alignment) and rejects a
    /// 20-byte upload. Copy into a zero-padded buffer rounded up to 16 so the length
    /// always covers any trailing padding — over-providing is allowed, under is not.
    pub fn push_constants(&self, data: &[u8]) {
        if let Some(enc) = self.encoder.borrow().as_ref() {
            let mut buf = [0u8; 256];
            let len = data.len();
            assert!(len <= buf.len(), "push constant block too large for Metal");
            buf[..len].copy_from_slice(data);
            let padded = (len + 15) & !15;
            let ptr = NonNull::new(buf.as_ptr() as *mut std::ffi::c_void)
                .expect("push_constants data pointer is null");
            unsafe {
                enc.setVertexBytes_length_atIndex(ptr, padded, PUSH_CONSTANT_INDEX);
                enc.setFragmentBytes_length_atIndex(ptr, padded, PUSH_CONSTANT_INDEX);
            }
        }
    }

    pub fn draw_indexed(&self, index_count: u32, first_index: u32, vertex_offset: i32) {
        let enc = self.encoder.borrow();
        let Some(enc) = enc.as_ref() else { return };
        let index_buffer = self.index_buffer.borrow();
        let (buffer, wide) = index_buffer
            .as_ref()
            .expect("draw_indexed without bind_index_buffer");
        let (index_type, index_size) = if *wide {
            (MTLIndexType::UInt32, 4)
        } else {
            (MTLIndexType::UInt16, 2)
        };
        unsafe {
            enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                MTLPrimitiveType::Triangle,
                index_count as usize,
                index_type,
                buffer,
                first_index as usize * index_size,
                1,
                vertex_offset as isize,
                0,
            );
        }
    }
}

/// Configure a color attachment: `Some(clear)` clears it, `None` loads; always
/// stored (every offscreen/cube/MRT target is read by a later pass).
fn config_color(attach: &MTLRenderPassColorAttachmentDescriptor, clear: Option<ClearColor>) {
    match clear {
        Some(c) => {
            attach.setLoadAction(MTLLoadAction::Clear);
            attach.setClearColor(MTLClearColor {
                red: c.r as f64,
                green: c.g as f64,
                blue: c.b as f64,
                alpha: c.a as f64,
            });
        }
        None => attach.setLoadAction(MTLLoadAction::Load),
    }
    attach.setStoreAction(MTLStoreAction::Store);
}

/// Configure a depth attachment: always cleared to far (1.0); `store` is `Store`
/// for the shadow map (sampled later) or `DontCare` otherwise.
fn config_depth(
    da: &MTLRenderPassDepthAttachmentDescriptor,
    texture: &Retained<ProtocolObject<dyn MTLTexture>>,
    clear: bool,
    store: MTLStoreAction,
) {
    da.setTexture(Some(texture));
    // `clear` clears to far (the first writer of this depth); otherwise LOAD preserves an
    // existing depth so a later pass (the decal pass) tests against — and the deferred SW-RT
    // passes sample — the real G-buffer depth, not a freshly-cleared one.
    da.setLoadAction(if clear {
        MTLLoadAction::Clear
    } else {
        MTLLoadAction::Load
    });
    da.setClearDepth(1.0);
    da.setStoreAction(store);
}
