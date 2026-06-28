//! Command buffer recording for the triangle frame.

use std::cell::Cell;
use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::{ClearColor, Extent2D, Rect2D};

use crate::buffer::{VulkanBuffer, VulkanStorageBuffer};
use crate::cubemap::VulkanCubemap;
use crate::depth::{VulkanDepthBuffer, depth_subresource_range};
use crate::device::DeviceShared;
use crate::pipeline::{VulkanComputePipeline, VulkanGraphicsPipeline};
use crate::query::VulkanQueryHeap;
use crate::render_target::VulkanRenderTarget;
use crate::swapchain::VulkanSwapchain;
use crate::volume::VulkanVolume;
use crate::{color_subresource_range, vk_err};

/// A primary command buffer, reset and re-recorded each frame.
pub struct VulkanCommandBuffer {
    device: Arc<DeviceShared>,
    cmd: vk::CommandBuffer,
    // Pool this buffer was allocated from (graphics or async-compute).
    pool: vk::CommandPool,
    // Layout of the currently bound pipeline (for push constants).
    current_layout: Cell<vk::PipelineLayout>,
    // Dynamic offset into the globals buffer for the next PBR pipeline bind.
    globals_offset: Cell<u32>,
}

impl VulkanCommandBuffer {
    pub(crate) fn new(device: Arc<DeviceShared>) -> Result<Self, EngineError> {
        let pool = device.command_pool;
        Self::from_pool(device, pool)
    }

    /// Allocate a command buffer on the async-compute family's pool (Phase 7).
    pub(crate) fn new_compute(device: Arc<DeviceShared>) -> Result<Self, EngineError> {
        let pool = device.compute_command_pool;
        Self::from_pool(device, pool)
    }

    fn from_pool(device: Arc<DeviceShared>, pool: vk::CommandPool) -> Result<Self, EngineError> {
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
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
            pool,
            current_layout: Cell::new(vk::PipelineLayout::null()),
            globals_offset: Cell::new(0),
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

    /// Reset `count` timestamp queries before they are (re)written this frame.
    /// Must be recorded outside a render pass (this is called at graph start).
    pub fn reset_queries(&self, heap: &VulkanQueryHeap, first: u32, count: u32) {
        unsafe {
            self.device
                .device
                .cmd_reset_query_pool(self.cmd, heap.raw(), first, count);
        }
    }

    /// Write a timestamp into query `index` at the bottom of the pipe (records the
    /// completion point of all prior work — pass boundary timing).
    pub fn write_timestamp(&self, heap: &VulkanQueryHeap, index: u32) {
        unsafe {
            self.device.device.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                heap.raw(),
                index,
            );
        }
    }

    /// Resolve written queries into their readback buffer. No-op on Vulkan
    /// (results are read directly from the pool); present for backend symmetry.
    pub fn resolve_queries(&self, _heap: &VulkanQueryHeap, _count: u32) {}

    /// Open a named debug-marker region (shown as a group in RenderDoc/NSight
    /// captures). No-op unless the debug-utils loader is active (debug build +
    /// validation). Must be balanced with [`Self::end_debug_label`].
    pub fn begin_debug_label(&self, name: &str) {
        if let Some(du) = &self.device.debug_utils {
            let cname = std::ffi::CString::new(name).unwrap_or_default();
            let label = vk::DebugUtilsLabelEXT::default().label_name(&cname);
            unsafe { du.cmd_begin_debug_utils_label(self.cmd, &label) };
        }
    }

    /// Close the most recently opened debug-marker region.
    pub fn end_debug_label(&self) {
        if let Some(du) = &self.device.debug_utils {
            unsafe { du.cmd_end_debug_utils_label(self.cmd) };
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

    /// Begin dynamic rendering into N offscreen color targets (MRT) plus optional
    /// depth. Each target's `Some(clear)` clears it, `None` loads. All targets must
    /// already be in `COLOR_ATTACHMENT_OPTIMAL`. The render area is taken from the
    /// first color target. `targets` must be non-empty.
    pub fn begin_rendering_targets(
        &self,
        targets: &[(&VulkanRenderTarget, Option<ClearColor>)],
        depth: Option<&VulkanDepthBuffer>,
    ) {
        let extent = targets[0].0.extent();
        let color_attachments: Vec<vk::RenderingAttachmentInfo> = targets
            .iter()
            .map(|(t, clear)| {
                let (load_op, clear_value) = match clear {
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
                vk::RenderingAttachmentInfo::default()
                    .image_view(t.view())
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(load_op)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(clear_value)
            })
            .collect();
        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments);

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

    /// Begin dynamic rendering into a depth-only target (a shadow map): no color
    /// attachments, depth is cleared and **stored** so a later pass can sample it.
    /// The depth image must already be in `DEPTH_ATTACHMENT_OPTIMAL` (see
    /// [`Self::depth_to_render_target`]).
    pub fn begin_rendering_depth_only(&self, depth: &VulkanDepthBuffer) {
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(depth.view())
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: depth.extent(),
            })
            .layer_count(1)
            .depth_attachment(&depth_attachment);
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

    /// Transition a depth buffer into `DEPTH_ATTACHMENT_OPTIMAL` for writing (a
    /// shadow map reused across frames may currently be in shader-read).
    pub fn depth_to_render_target(&self, depth: &VulkanDepthBuffer) {
        let old = depth.layout();
        if old == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
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
        self.depth_image_barrier(
            depth.image(),
            old,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            src_access,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            src_stage,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        );
        depth.set_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);
    }

    /// Transition a whole cubemap into `COLOR_ATTACHMENT_OPTIMAL` for writing its
    /// faces/mips (the IBL generation passes).
    pub fn cube_to_color(&self, cube: &VulkanCubemap) {
        let old = cube.layout();
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
        self.cube_image_barrier(
            cube,
            old,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_access,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            src_stage,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );
        cube.set_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }

    /// Transition a whole cubemap into `SHADER_READ_ONLY_OPTIMAL` for sampling.
    pub fn cube_to_sampled(&self, cube: &VulkanCubemap) {
        if cube.layout() == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            return;
        }
        self.cube_image_barrier(
            cube,
            cube.layout(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
        cube.set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    /// Begin dynamic rendering into one (face, mip) of a cubemap. The cubemap must
    /// already be in `COLOR_ATTACHMENT_OPTIMAL` (see [`Self::cube_to_color`]).
    pub fn begin_rendering_cube_face(
        &self,
        cube: &VulkanCubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
    ) {
        let (load_op, clear_value) = match clear {
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
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(cube.render_view(face, mip))
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_value);
        let attachments = [attachment];
        let size = cube.mip_size(mip);
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: size,
                    height: size,
                },
            })
            .layer_count(1)
            .color_attachments(&attachments);
        unsafe {
            self.device
                .device
                .cmd_begin_rendering(self.cmd, &rendering_info)
        };
    }

    /// Begin rendering into one (face, mip) of a cubemap **with a depth buffer**
    /// (clears depth), for capturing scene geometry. Color is loaded if
    /// `clear = None`. The cube must be in `COLOR_ATTACHMENT_OPTIMAL` and the
    /// depth in `DEPTH_ATTACHMENT_OPTIMAL` (see `cube_to_color`/`depth_to_render_target`).
    pub fn begin_rendering_cube_face_depth(
        &self,
        cube: &VulkanCubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
        depth: &VulkanDepthBuffer,
    ) {
        let (load_op, clear_value) = match clear {
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
        let color = vk::RenderingAttachmentInfo::default()
            .image_view(cube.render_view(face, mip))
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_value);
        let colors = [color];
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(depth.view())
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });
        let size = cube.mip_size(mip);
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: size,
                    height: size,
                },
            })
            .layer_count(1)
            .color_attachments(&colors)
            .depth_attachment(&depth_attachment);
        unsafe {
            self.device
                .device
                .cmd_begin_rendering(self.cmd, &rendering_info)
        };
    }

    /// Transition a depth buffer into `SHADER_READ_ONLY_OPTIMAL` for sampling.
    pub fn depth_to_sampled(&self, depth: &VulkanDepthBuffer) {
        if depth.layout() == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            return;
        }
        self.depth_image_barrier(
            depth.image(),
            depth.layout(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
        depth.set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    /// Copy a rendered swapchain image into a host-visible readback buffer (for
    /// screenshots). The image must be in `PRESENT_SRC_KHR` (the state the render
    /// graph leaves it in); it is restored to that layout afterward. Rows are
    /// tightly packed (`row_pitch = width * 4`).
    pub fn copy_swapchain_to_buffer(
        &self,
        swapchain: &VulkanSwapchain,
        image_index: u32,
        buffer: &VulkanBuffer,
    ) {
        let image = swapchain.image(image_index);
        let extent = swapchain.extent();

        self.image_barrier(
            image,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER,
        );

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0) // tightly packed
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            });
        unsafe {
            self.device.device.cmd_copy_image_to_buffer(
                self.cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.raw(),
                &[region],
            );
        }

        self.image_barrier(
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        );
    }

    /// Select the per-frame globals slice (dynamic offset) used by the next PBR
    /// pipeline bind.
    pub fn set_globals(&self, offset: u32) {
        self.globals_offset.set(offset);
    }

    /// Bind a graphics pipeline (and its bindless set 0 + globals set 1, if any).
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
            if pipeline.uses_uniform() {
                self.device.device.cmd_bind_descriptor_sets(
                    self.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.layout(),
                    1,
                    &[self.device.globals_set],
                    &[self.globals_offset.get()],
                );
            }
        }
    }

    /// Pipeline stages that read/write storage resources: compute always, plus the
    /// ray-tracing stages on RT-capable devices (the path-tracer pipeline writes its
    /// output image + accumulation buffer from the raygen stage, Phase 8 M5).
    /// Widening a barrier's stage scope is always safe; it just covers the RT case.
    fn storage_stages(&self) -> vk::PipelineStageFlags {
        let mut s = vk::PipelineStageFlags::COMPUTE_SHADER;
        if self.device.has_raytracing {
            s |= vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR;
        }
        s
    }

    /// Transition a storage render target into `GENERAL` for compute writes.
    pub fn rt_to_storage(&self, target: &VulkanRenderTarget) {
        let old = target.layout();
        if old == vk::ImageLayout::GENERAL {
            return;
        }
        let (src_access, src_stage) = match old {
            vk::ImageLayout::UNDEFINED => (
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
            ),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => (
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
        };
        self.image_barrier(
            target.image(),
            old,
            vk::ImageLayout::GENERAL,
            src_access,
            vk::AccessFlags::SHADER_WRITE,
            src_stage,
            self.storage_stages(),
        );
        target.set_layout(vk::ImageLayout::GENERAL);
    }

    /// Transition a storage image from `GENERAL` (compute write) into
    /// `SHADER_READ_ONLY_OPTIMAL` for sampling by a later graphics pass.
    pub fn storage_to_sampled(&self, target: &VulkanRenderTarget) {
        if target.layout() == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            return;
        }
        self.image_barrier(
            target.image(),
            target.layout(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ,
            self.storage_stages(),
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
        target.set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    /// Transition a 3D volume into `GENERAL` so a compute bake can write it
    /// (Phase 11 Stage B). Mirrors `rt_to_storage` for the volume tables.
    pub fn volume_to_storage(&self, volume: &VulkanVolume) {
        let old = volume.layout();
        if old == vk::ImageLayout::GENERAL {
            return;
        }
        let (src_access, src_stage) = match old {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                vk::AccessFlags::SHADER_READ,
                self.storage_stages() | vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => (
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
            ),
        };
        self.image_barrier(
            volume.image(),
            old,
            vk::ImageLayout::GENERAL,
            src_access,
            vk::AccessFlags::SHADER_WRITE,
            src_stage,
            self.storage_stages(),
        );
        volume.set_layout(vk::ImageLayout::GENERAL);
    }

    /// Transition a 3D volume from `GENERAL` (compute bake) into
    /// `SHADER_READ_ONLY_OPTIMAL` for trilinear sampling by a later pass.
    pub fn volume_to_sampled(&self, volume: &VulkanVolume) {
        if volume.layout() == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            return;
        }
        self.image_barrier(
            volume.image(),
            volume.layout(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ,
            self.storage_stages(),
            self.storage_stages() | vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
        volume.set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    /// UAV barrier on a storage buffer: order a compute write before any later
    /// shader read (compute / vertex / fragment).
    pub fn storage_buffer_barrier(&self, buffer: &VulkanStorageBuffer) {
        self.buffer_memory_barrier(
            buffer.raw(),
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            self.storage_stages(),
            self.storage_stages()
                | vk::PipelineStageFlags::VERTEX_SHADER
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
    }

    /// UAV barrier on a storage buffer, COMPUTE-stage only — for use on the async-compute queue
    /// (whose family does not support the vertex/fragment stages of `storage_buffer_barrier`).
    pub fn storage_buffer_barrier_compute(&self, buffer: &VulkanStorageBuffer) {
        self.buffer_memory_barrier(
            buffer.raw(),
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
    }

    /// Order a compute write to an indirect-args buffer before its consumption by
    /// `draw_indexed_indirect`.
    pub fn storage_buffer_to_indirect(&self, buffer: &VulkanStorageBuffer) {
        self.buffer_memory_barrier(
            buffer.raw(),
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::INDIRECT_COMMAND_READ,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::DRAW_INDIRECT,
        );
    }

    /// Order a buffer's prior reads (indirect / vertex / compute) before the next
    /// frame's compute write back to it.
    pub fn storage_buffer_to_storage(&self, buffer: &VulkanStorageBuffer) {
        self.buffer_memory_barrier(
            buffer.raw(),
            vk::AccessFlags::INDIRECT_COMMAND_READ | vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::DRAW_INDIRECT
                | vk::PipelineStageFlags::VERTEX_SHADER
                | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
    }

    fn buffer_memory_barrier(
        &self,
        buffer: vk::Buffer,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) {
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                self.cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }
    }

    /// Bind a compute pipeline (and its bindless set 0, if any).
    pub fn bind_compute_pipeline(&self, pipeline: &VulkanComputePipeline) {
        unsafe {
            self.device.device.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.raw(),
            );
            self.current_layout.set(pipeline.layout());
            if pipeline.is_bindless() {
                self.device.device.cmd_bind_descriptor_sets(
                    self.cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout(),
                    0,
                    &[self.device.bindless_set],
                    &[],
                );
            }
            // Bind the per-frame globals UBO (set 1) at the dynamic offset selected by
            // `set_globals`, mirroring the graphics path (Stage C7 reflection passes).
            if pipeline.uses_uniform() {
                self.device.device.cmd_bind_descriptor_sets(
                    self.cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout(),
                    1,
                    &[self.device.globals_set],
                    &[self.globals_offset.get()],
                );
            }
        }
    }

    /// Dispatch the bound compute pipeline over `(x, y, z)` workgroups.
    pub fn dispatch(&self, x: u32, y: u32, z: u32) {
        unsafe { self.device.device.cmd_dispatch(self.cmd, x, y, z) };
    }

    /// Upload push constants for the bound compute pipeline (COMPUTE stage).
    pub fn push_constants_compute(&self, data: &[u8]) {
        unsafe {
            self.device.device.cmd_push_constants(
                self.cmd,
                self.current_layout.get(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                data,
            );
        }
    }

    /// Bind a ray-tracing pipeline and the shared bindless set (Phase 8 M5).
    pub fn bind_raytracing_pipeline(
        &self,
        pipeline: &crate::rt_pipeline::VulkanRaytracingPipeline,
    ) {
        unsafe {
            self.device.device.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.raw(),
            );
            self.current_layout.set(pipeline.layout());
            self.device.device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.layout(),
                0,
                &[self.device.bindless_set],
                &[],
            );
        }
    }

    /// Upload push constants for the bound ray-tracing pipeline (RT stages).
    pub fn push_constants_rt(&self, data: &[u8]) {
        unsafe {
            self.device.device.cmd_push_constants(
                self.cmd,
                self.current_layout.get(),
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                    | vk::ShaderStageFlags::MISS_KHR,
                0,
                data,
            );
        }
    }

    /// Trace a `width` x `height` grid of rays through the bound RT pipeline's SBT.
    pub fn trace_rays(
        &self,
        pipeline: &crate::rt_pipeline::VulkanRaytracingPipeline,
        width: u32,
        height: u32,
    ) {
        let loader = self
            .device
            .rt_pipeline_loader
            .as_ref()
            .expect("ray tracing pipeline loader (RT-capable device)");
        let (raygen, miss, hit, callable) = pipeline.regions();
        unsafe {
            loader.cmd_trace_rays(self.cmd, raygen, miss, hit, callable, width, height, 1);
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

    /// Issue `draw_count` indexed indirect draws reading `VkDrawIndexedIndirectCommand`
    /// (20-byte) records from `buffer` starting at `offset`. The buffer must be in
    /// indirect-read state (see [`Self::storage_buffer_to_indirect`]).
    pub fn draw_indexed_indirect(
        &self,
        buffer: &VulkanStorageBuffer,
        offset: u64,
        draw_count: u32,
    ) {
        unsafe {
            self.device.device.cmd_draw_indexed_indirect(
                self.cmd,
                buffer.raw(),
                offset,
                draw_count,
                20,
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

    #[allow(clippy::too_many_arguments)]
    fn cube_image_barrier(
        &self,
        cube: &VulkanCubemap,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) {
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: cube.mip_levels(),
            base_array_layer: 0,
            layer_count: 6,
        };
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(cube.image())
            .subresource_range(range);
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

    #[allow(clippy::too_many_arguments)]
    fn depth_image_barrier(
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
            .subresource_range(depth_subresource_range());
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
                .free_command_buffers(self.pool, &[self.cmd]);
        }
    }
}
