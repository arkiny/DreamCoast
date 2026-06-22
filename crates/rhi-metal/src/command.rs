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
use std::rc::Rc;

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLLoadAction,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLStoreAction,
};
use objc2_quartz_core::CAMetalDrawable;
use rhi_types::{ClearColor, Extent2D, Rect2D};

use crate::device::DeviceShared;
use crate::resources::{
    MetalBuffer, MetalComputePipeline, MetalCubemap, MetalDepthBuffer, MetalGraphicsPipeline,
    MetalRenderTarget, MetalStorageBuffer,
};
use crate::swapchain::MetalSwapchain;
use crate::{Result, rhi_err};

pub struct MetalCommandBuffer {
    shared: Rc<DeviceShared>,
    cmd: RefCell<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    encoder: RefCell<Option<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>>>,
    present: RefCell<Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>>,
}

impl MetalCommandBuffer {
    pub(crate) fn new(shared: Rc<DeviceShared>) -> Self {
        Self {
            shared,
            cmd: RefCell::new(None),
            encoder: RefCell::new(None),
            present: RefCell::new(None),
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
        _depth: Option<&MetalDepthBuffer>,
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

    pub fn set_viewport_scissor(&self, _swapchain: &MetalSwapchain) {
        // Encoder defaults to the full attachment; explicit viewport lands in M2.
    }

    pub fn set_viewport_scissor_extent(&self, _extent: Extent2D) {}

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

    pub fn copy_swapchain_to_buffer(
        &self,
        _swapchain: &MetalSwapchain,
        _image_index: u32,
        _buffer: &MetalBuffer,
    ) {
        unimplemented!("Metal screenshot readback: milestone M6")
    }

    pub fn rt_to_render_target(&self, _target: &MetalRenderTarget) {}
    pub fn rt_to_sampled(&self, _target: &MetalRenderTarget) {}
    pub fn aliasing_barrier(&self, _target: &MetalRenderTarget) {}

    pub fn bind_graphics_pipeline(&self, _pipeline: &MetalGraphicsPipeline) {
        unimplemented!("Metal graphics pipelines: milestone M2")
    }

    pub fn draw(&self, _vertex_count: u32, _instance_count: u32) {
        unimplemented!("Metal draw: milestone M2")
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

    pub fn set_scissor(&self, _rect: Rect2D) {
        unimplemented!("Metal scissor: milestone M2")
    }

    pub fn bind_vertex_buffer(&self, _buffer: &MetalBuffer, _stride: u32) {
        unimplemented!("Metal vertex buffers: milestone M2")
    }

    pub fn bind_index_buffer(&self, _buffer: &MetalBuffer, _wide: bool) {
        unimplemented!("Metal index buffers: milestone M2")
    }

    pub fn push_constants(&self, _data: &[u8]) {
        unimplemented!("Metal push constants: milestone M2")
    }

    pub fn draw_indexed(&self, _index_count: u32, _first_index: u32, _vertex_offset: i32) {
        unimplemented!("Metal indexed draw: milestone M2")
    }
}
