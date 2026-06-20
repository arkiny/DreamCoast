//! Command buffer recording for the triangle frame.

use std::cell::Cell;
use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::{ClearColor, Extent2D, Rect2D};

use crate::buffer::VulkanBuffer;
use crate::depth::VulkanDepthBuffer;
use crate::device::DeviceShared;
use crate::pipeline::VulkanGraphicsPipeline;
use crate::render_target::VulkanRenderTarget;
use crate::swapchain::VulkanSwapchain;
use crate::{color_subresource_range, vk_err};

/// A primary command buffer, reset and re-recorded each frame.
pub struct VulkanCommandBuffer {
    device: Arc<DeviceShared>,
    cmd: vk::CommandBuffer,
    // Layout of the currently bound pipeline (for push constants).
    current_layout: Cell<vk::PipelineLayout>,
}

impl VulkanCommandBuffer {
    pub(crate) fn new(device: Arc<DeviceShared>) -> Result<Self, EngineError> {
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(device.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe {
            device
                .device
                .allocate_command_buffers(&alloc)
                .map_err(vk_err)?
        }[0];
        Ok(Self {
            device,
            cmd,
            current_layout: Cell::new(vk::PipelineLayout::null()),
        })
    }

    pub(crate) fn raw(&self) -> vk::CommandBuffer {
        self.cmd
    }

    /// Reset and begin recording (one-time submit).
    pub fn begin(&self) -> Result<(), EngineError> {
        unsafe {
            self.device
                .device
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
                .map_err(vk_err)?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .device
                .begin_command_buffer(self.cmd, &begin)
                .map_err(vk_err)
        }
    }

    /// Finish recording.
    pub fn end(&self) -> Result<(), EngineError> {
        unsafe {
            self.device
                .device
                .end_command_buffer(self.cmd)
                .map_err(vk_err)
        }
    }

    /// Transition a swapchain image `UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL`.
    pub fn transition_to_render_target(&self, swapchain: &VulkanSwapchain, image_index: u32) {
        self.image_barrier(
            swapchain.image(image_index),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );
    }

    /// Transition a swapchain image `COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC`.
    pub fn transition_to_present(&self, swapchain: &VulkanSwapchain, image_index: u32) {
        self.image_barrier(
            swapchain.image(image_index),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        );
    }

    /// Begin dynamic rendering. `color_clear = Some` clears the color attachment,
    /// `None` loads it (overlay pass). `depth = Some` attaches + clears depth.
    pub fn begin_rendering(
        &self,
        swapchain: &VulkanSwapchain,
        image_index: u32,
        color_clear: Option<ClearColor>,
        depth: Option<&VulkanDepthBuffer>,
    ) {
        let (load_op, clear_value) = match color_clear {
            Some(c) => (
                vk::AttachmentLoadOp::CLEAR,
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [c.r, c.g, c.b, c.a],
                    },
                },
            ),
            None => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
        };
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(swapchain.view(image_index))
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_value);
        let attachments = [color_attachment];
        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.extent(),
            })
            .layer_count(1)
            .color_attachments(&attachments);

        let depth_attachment;
        if let Some(d) = depth {
            depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(d.view())
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });
            rendering_info = rendering_info.depth_attachment(&depth_attachment);
        }

        unsafe {
            self.device
                .device
                .cmd_begin_rendering(self.cmd, &rendering_info)
        };
    }

    /// Begin dynamic rendering into an offscreen color target (+ optional depth).
    /// The target must already be in `COLOR_ATTACHMENT_OPTIMAL` (see
    /// [`Self::rt_to_render_target`]).
    pub fn begin_rendering_target(
        &self,
        target: &VulkanRenderTarget,
        color_clear: Option<ClearColor>,
        depth: Option<&VulkanDepthBuffer>,
    ) {
        let (load_op, clear_value) = match color_clear {
            Some(c) => (
                vk::AttachmentLoadOp::CLEAR,
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [c.r, c.g, c.b, c.a],
                    },
                },
            ),
            None => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
        };
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(target.view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_value);
        let attachments = [color_attachment];
        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: target.extent(),
            })
            .layer_count(1)
            .color_attachments(&attachments);

        let depth_attachment;
        if let Some(d) = depth {
            depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(d.view())
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });
            rendering_info = rendering_info.depth_attachment(&depth_attachment);
        }

        unsafe {
            self.device
                .device
                .cmd_begin_rendering(self.cmd, &rendering_info)
        };
    }

    /// End dynamic rendering.
    pub fn end_rendering(&self) {
        unsafe { self.device.device.cmd_end_rendering(self.cmd) };
    }

    /// Set viewport and scissor to cover the swapchain extent.
    pub fn set_viewport_scissor(&self, swapchain: &VulkanSwapchain) {
        let extent = swapchain.extent();
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        unsafe {
            self.device
                .device
                .cmd_set_viewport(self.cmd, 0, &[viewport]);
            self.device.device.cmd_set_scissor(self.cmd, 0, &[scissor]);
        }
    }

    /// Set viewport and scissor to cover an arbitrary extent (offscreen target).
    pub fn set_viewport_scissor_extent(&self, extent: Extent2D) {
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: extent.width,
                height: extent.height,
            },
        };
        unsafe {
            self.device
                .device
                .cmd_set_viewport(self.cmd, 0, &[viewport]);
            self.device.device.cmd_set_scissor(self.cmd, 0, &[scissor]);
        }
    }

    /// Transition an offscreen target into `COLOR_ATTACHMENT_OPTIMAL` for writing.
    pub fn rt_to_render_target(&self, target: &VulkanRenderTarget) {
        let old = target.layout();
        if old == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            return;
        }
        let (src_access, src_stage) = if old == vk::ImageLayout::UNDEFINED {
            (
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
            )
        } else {
            (
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            )
        };
        self.image_barrier(
            target.image(),
            old,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_access,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            src_stage,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );
        target.set_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }

    /// Prepare an aliased target for writing into shared heap memory: a single
    /// `UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL` barrier that discards whatever the
    /// previous tenant left and waits for that tenant's reads/writes to finish.
    pub fn aliasing_barrier(&self, target: &VulkanRenderTarget) {
        self.image_barrier(
            target.image(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );
        target.set_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }

    /// Transition an offscreen target into `SHADER_READ_ONLY_OPTIMAL` for sampling.
    pub fn rt_to_sampled(&self, target: &VulkanRenderTarget) {
        if target.layout() == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            return;
        }
        self.image_barrier(
            target.image(),
            target.layout(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
        target.set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    /// Bind a graphics pipeline (and its bindless descriptor set, if any).
    pub fn bind_graphics_pipeline(&self, pipeline: &VulkanGraphicsPipeline) {
        unsafe {
            self.device.device.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.raw(),
            );
            self.current_layout.set(pipeline.layout());
            if pipeline.is_bindless() {
                self.device.device.cmd_bind_descriptor_sets(
                    self.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.layout(),
                    0,
                    &[self.device.bindless_set],
                    &[],
                );
            }
        }
    }

    /// Issue a non-indexed draw.
    pub fn draw(&self, vertex_count: u32, instance_count: u32) {
        unsafe {
            self.device
                .device
                .cmd_draw(self.cmd, vertex_count, instance_count, 0, 0);
        }
    }

    /// Set the scissor rectangle.
    pub fn set_scissor(&self, rect: Rect2D) {
        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: rect.x,
                y: rect.y,
            },
            extent: vk::Extent2D {
                width: rect.width,
                height: rect.height,
            },
        };
        unsafe { self.device.device.cmd_set_scissor(self.cmd, 0, &[scissor]) };
    }

    /// Bind a vertex buffer at binding 0. `stride` is unused (Vulkan takes it
    /// from the pipeline's vertex layout) — present for facade parity with D3D12.
    pub fn bind_vertex_buffer(&self, buffer: &VulkanBuffer, _stride: u32) {
        unsafe {
            self.device
                .device
                .cmd_bind_vertex_buffers(self.cmd, 0, &[buffer.raw()], &[0]);
        }
    }

    /// Bind an index buffer (`wide` selects 32-bit indices, else 16-bit).
    pub fn bind_index_buffer(&self, buffer: &VulkanBuffer, wide: bool) {
        let ty = if wide {
            vk::IndexType::UINT32
        } else {
            vk::IndexType::UINT16
        };
        unsafe {
            self.device
                .device
                .cmd_bind_index_buffer(self.cmd, buffer.raw(), 0, ty);
        }
    }

    /// Upload push constants (visible to both stages) for the bound pipeline.
    pub fn push_constants(&self, data: &[u8]) {
        unsafe {
            self.device.device.cmd_push_constants(
                self.cmd,
                self.current_layout.get(),
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                data,
            );
        }
    }

    /// Issue an indexed draw.
    pub fn draw_indexed(&self, index_count: u32, first_index: u32, vertex_offset: i32) {
        unsafe {
            self.device.device.cmd_draw_indexed(
                self.cmd,
                index_count,
                1,
                first_index,
                vertex_offset,
                0,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn image_barrier(
        &self,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color_subresource_range());
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                self.cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }
}

impl Drop for VulkanCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device
                .free_command_buffers(self.device.command_pool, &[self.cmd]);
        }
    }
}
