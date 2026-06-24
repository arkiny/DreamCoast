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
    MTLCommandQueue, MTLComputeCommandEncoder, MTLIndexType, MTLLoadAction, MTLOrigin,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassColorAttachmentDescriptor,
    MTLRenderPassDepthAttachmentDescriptor, MTLRenderPassDescriptor, MTLRenderStages, MTLResource,
    MTLResourceUsage, MTLScissorRect, MTLSize, MTLStoreAction, MTLTexture, MTLViewport,
};
use objc2_quartz_core::CAMetalDrawable;
use rhi_types::{ClearColor, Extent2D, Rect2D};

use crate::device::DeviceShared;
use crate::resources::{
    BINDLESS_BUFFER_INDEX, BINDLESS_BUFFER_INDEX_WITH_GLOBALS, GLOBALS_BUFFER_INDEX, MetalBuffer,
    MetalComputePipeline, MetalCubemap, MetalDepthBuffer, MetalGraphicsPipeline, MetalRenderTarget,
    MetalStorageBuffer, PUSH_CONSTANT_INDEX, VERTEX_BUFFER_INDEX,
};
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
}

impl MetalCommandBuffer {
    pub(crate) fn new(
        shared: Rc<DeviceShared>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Self {
        Self {
            shared,
            queue,
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
        }
    }

    /// End whichever encoder (render or compute) is currently open. Compute passes
    /// have no explicit `end_rendering`, so the next encoder start closes them.
    fn end_any_encoder(&self) {
        if let Some(enc) = self.encoder.borrow_mut().take() {
            enc.endEncoding();
        }
        if let Some(enc) = self.compute_encoder.borrow_mut().take() {
            enc.endEncoding();
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
        Ok(())
    }

    pub fn end(&self) -> Result<()> {
        self.end_any_encoder();
        Ok(())
    }

    /// End any open encoder, signal `event` to `value` (cross-queue ordering for
    /// async compute), and commit. Used by [`crate::device::MetalComputeQueue`].
    pub(crate) fn commit_signaling(&self, event: &ProtocolObject<dyn MTLEvent>, value: u64) {
        self.end_any_encoder();
        let cb = self
            .cmd
            .borrow_mut()
            .take()
            .expect("commit_signaling() without begin()");
        cb.encodeSignalEvent_value(event, value);
        cb.commit();
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
        *self.encoder.borrow_mut() = Some(enc);
    }

    pub fn end_rendering(&self) {
        if let Some(enc) = self.encoder.borrow_mut().take() {
            enc.endEncoding();
        }
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

    // ---- Offscreen render targets / MRT / shadow / cubemaps (M4) -----------

    /// Begin rendering into one offscreen color target (+ optional depth). Depth is
    /// cleared and discarded (`DontCare`) — only the shadow pass stores depth.
    pub fn begin_rendering_target(
        &self,
        target: &MetalRenderTarget,
        color_clear: Option<ClearColor>,
        depth: Option<&MetalDepthBuffer>,
    ) {
        let pass = MTLRenderPassDescriptor::new();
        let attach = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        attach.setTexture(Some(&target.texture));
        config_color(&attach, color_clear);
        if let Some(d) = depth {
            config_depth(
                &pass.depthAttachment(),
                &d.texture,
                MTLStoreAction::DontCare,
            );
        }
        self.start_encoder(&pass);
    }

    /// Begin rendering into N offscreen color targets (MRT, e.g. the 4-attachment
    /// G-buffer) + optional depth. Each target clears if `Some`, else loads.
    pub fn begin_rendering_targets(
        &self,
        targets: &[(&MetalRenderTarget, Option<ClearColor>)],
        depth: Option<&MetalDepthBuffer>,
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
                MTLStoreAction::DontCare,
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
        let cmd = self.cmd.borrow();
        let cmd = cmd
            .as_ref()
            .expect("copy_swapchain_to_buffer without begin");
        let blit = cmd
            .blitCommandEncoder()
            .expect("failed to create blit command encoder");
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
        enc.setComputePipelineState(&pipeline.state);
        if pipeline.bindless {
            unsafe {
                enc.setBuffer_offset_atIndex(
                    Some(&self.shared.arg_buffer),
                    0,
                    BINDLESS_BUFFER_INDEX,
                );
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
    store: MTLStoreAction,
) {
    da.setTexture(Some(texture));
    da.setLoadAction(MTLLoadAction::Clear);
    da.setClearDepth(1.0);
    da.setStoreAction(store);
}
