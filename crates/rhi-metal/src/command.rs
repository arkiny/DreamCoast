//! Metal command buffer recording.
//!
//! Maps the facade's command interface onto a `MTLCommandBuffer` plus a current
//! `MTLRenderCommandEncoder`. Presentation is deferred: `transition_to_present`
//! stashes the drawable and [`MetalCommandBuffer::commit`] (called from the queue
//! submit) records `presentDrawable` before committing.
//!
//! M0 implements the clear path (begin → begin_rendering(clear) → end_rendering →
//! present → submit). Draw / pipeline / resource methods land in M2+.

use std::cell::RefCell;
use std::ptr::NonNull;
use std::rc::Rc;

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLIndexType, MTLLoadAction, MTLOrigin, MTLPrimitiveType, MTLRenderStages,
    MTLResource, MTLResourceUsage, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLScissorRect, MTLSize, MTLStoreAction, MTLTexture, MTLViewport,
};
use objc2_quartz_core::CAMetalDrawable;
use rhi_types::{ClearColor, Extent2D, Rect2D};

use crate::device::DeviceShared;
use crate::resources::{
    BINDLESS_BUFFER_INDEX, MetalBuffer, MetalComputePipeline, MetalCubemap, MetalDepthBuffer,
    MetalGraphicsPipeline, MetalRenderTarget, MetalStorageBuffer, PUSH_CONSTANT_INDEX,
    VERTEX_BUFFER_INDEX,
};
use crate::swapchain::MetalSwapchain;
use crate::{Result, rhi_err};

/// A bound index buffer plus its width flag (`true` = 32-bit indices).
type BoundIndexBuffer = (Retained<ProtocolObject<dyn MTLBuffer>>, bool);

pub struct MetalCommandBuffer {
    shared: Rc<DeviceShared>,
    cmd: RefCell<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    encoder: RefCell<Option<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>>>,
    present: RefCell<Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>>,
    /// Index buffer + width bound by `bind_index_buffer`; consumed by
    /// `draw_indexed` (Metal takes the index buffer at draw time, not bind time).
    index_buffer: RefCell<Option<BoundIndexBuffer>>,
}

impl MetalCommandBuffer {
    pub(crate) fn new(shared: Rc<DeviceShared>) -> Self {
        Self {
            shared,
            cmd: RefCell::new(None),
            encoder: RefCell::new(None),
            present: RefCell::new(None),
            index_buffer: RefCell::new(None),
        }
    }

    pub fn begin(&self) -> Result<()> {
        let cb = self
            .shared
            .queue
            .commandBuffer()
            .ok_or_else(|| rhi_err("commandBuffer() returned nil"))?;
        *self.cmd.borrow_mut() = Some(cb);
        *self.encoder.borrow_mut() = None;
        *self.present.borrow_mut() = None;
        *self.index_buffer.borrow_mut() = None;
        Ok(())
    }

    pub fn end(&self) -> Result<()> {
        if let Some(enc) = self.encoder.borrow_mut().take() {
            enc.endEncoding();
        }
        Ok(())
    }

    /// Commit the recorded work (ending any open encoder and recording the
    /// deferred drawable present), returning the committed command buffer so a
    /// fence can block on it. Called from the queue submit paths.
    pub(crate) fn commit(&self) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        if let Some(enc) = self.encoder.borrow_mut().take() {
            enc.endEncoding();
        }
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
        *self.present.borrow_mut() = swapchain.current_drawable();
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

    // ---- Implemented in later milestones (M2+) -----------------------------

    pub fn begin_rendering_target(
        &self,
        _target: &MetalRenderTarget,
        _color_clear: Option<ClearColor>,
        _depth: Option<&MetalDepthBuffer>,
    ) {
        unimplemented!("Metal offscreen targets: milestone M4")
    }

    pub fn begin_rendering_targets(
        &self,
        _targets: &[(&MetalRenderTarget, Option<ClearColor>)],
        _depth: Option<&MetalDepthBuffer>,
    ) {
        unimplemented!("Metal MRT: milestone M4")
    }

    pub fn set_globals(&self, _offset: u32) {
        unimplemented!("Metal globals binding: milestone M4")
    }

    pub fn begin_rendering_depth_only(&self, _depth: &MetalDepthBuffer) {
        unimplemented!("Metal shadow pass: milestone M4")
    }

    pub fn depth_to_render_target(&self, _depth: &MetalDepthBuffer) {}
    pub fn depth_to_sampled(&self, _depth: &MetalDepthBuffer) {}
    pub fn cube_to_color(&self, _cube: &MetalCubemap) {}
    pub fn cube_to_sampled(&self, _cube: &MetalCubemap) {}

    pub fn begin_rendering_cube_face(
        &self,
        _cube: &MetalCubemap,
        _face: u32,
        _mip: u32,
        _clear: Option<ClearColor>,
    ) {
        unimplemented!("Metal cubemap rendering: milestone M4")
    }

    pub fn begin_rendering_cube_face_depth(
        &self,
        _cube: &MetalCubemap,
        _face: u32,
        _mip: u32,
        _clear: Option<ClearColor>,
        _depth: &MetalDepthBuffer,
    ) {
        unimplemented!("Metal cubemap rendering: milestone M4")
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
        let drawable = swapchain
            .current_drawable()
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

    pub fn rt_to_render_target(&self, _target: &MetalRenderTarget) {}
    pub fn rt_to_sampled(&self, _target: &MetalRenderTarget) {}
    pub fn aliasing_barrier(&self, _target: &MetalRenderTarget) {}

    pub fn bind_graphics_pipeline(&self, pipeline: &MetalGraphicsPipeline) {
        if let Some(enc) = self.encoder.borrow().as_ref() {
            enc.setRenderPipelineState(&pipeline.state);
            if let Some(ds) = pipeline.depth_stencil.as_ref() {
                enc.setDepthStencilState(Some(ds));
            }
            if pipeline.bindless {
                // Bind the bindless argument buffer at [[buffer(1)]] for both stages
                // (the vertex stage may index it too on other shaders).
                unsafe {
                    enc.setVertexBuffer_offset_atIndex(
                        Some(&self.shared.arg_buffer),
                        0,
                        BINDLESS_BUFFER_INDEX,
                    );
                    enc.setFragmentBuffer_offset_atIndex(
                        Some(&self.shared.arg_buffer),
                        0,
                        BINDLESS_BUFFER_INDEX,
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

    pub fn bind_compute_pipeline(&self, _pipeline: &MetalComputePipeline) {
        unimplemented!("Metal compute: milestone M5")
    }

    pub fn dispatch(&self, _x: u32, _y: u32, _z: u32) {
        unimplemented!("Metal compute: milestone M5")
    }

    pub fn push_constants_compute(&self, _data: &[u8]) {
        unimplemented!("Metal compute: milestone M5")
    }

    pub fn rt_to_storage(&self, _target: &MetalRenderTarget) {}
    pub fn storage_to_sampled(&self, _target: &MetalRenderTarget) {}
    pub fn storage_buffer_barrier(&self, _buffer: &MetalStorageBuffer) {}
    pub fn storage_buffer_to_indirect(&self, _buffer: &MetalStorageBuffer) {}
    pub fn storage_buffer_to_storage(&self, _buffer: &MetalStorageBuffer) {}

    pub fn draw_indexed_indirect(
        &self,
        _buffer: &MetalStorageBuffer,
        _offset: u64,
        _draw_count: u32,
    ) {
        unimplemented!("Metal indirect draw: milestone M5")
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
