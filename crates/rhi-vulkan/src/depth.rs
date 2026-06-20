//! Depth buffer (D32_SFLOAT) for the mesh pass.
//!
//! Created with `SAMPLED` usage and registered in the bindless table so it can
//! double as a shadow map: a depth-only pass writes it, then a later pass samples
//! it. Its current layout is tracked so the render graph can emit the
//! depth-attachment <-> shader-read barriers.

use std::cell::Cell;
use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::Extent2D;

use crate::device::DeviceShared;
use crate::vk_err;

/// Depth subresource range (D32_SFLOAT has only a depth aspect).
pub(crate) fn depth_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::DEPTH,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// A device-local depth image + view, registered as a bindless sampled texture.
pub struct VulkanDepthBuffer {
    device: Arc<DeviceShared>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    index: u32,
    extent: vk::Extent2D,
    /// Current image layout, updated by the barrier helpers.
    layout: Cell<vk::ImageLayout>,
}

impl VulkanDepthBuffer {
    pub(crate) fn new(device: Arc<DeviceShared>, extent: Extent2D) -> Result<Self, EngineError> {
        unsafe {
            let format = vk::Format::D32_SFLOAT;
            let vk_extent = vk::Extent2D {
                width: extent.width.max(1),
                height: extent.height.max(1),
            };
            let image_ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: vk_extent.width,
                    height: vk_extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = device
                .device
                .create_image(&image_ci, None)
                .map_err(vk_err)?;
            let req = device.device.get_image_memory_requirements(image);
            let mem_type = device
                .find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type);
            let memory = device
                .device
                .allocate_memory(&alloc, None)
                .map_err(vk_err)?;
            device
                .device
                .bind_image_memory(image, memory, 0)
                .map_err(vk_err)?;

            let view_ci = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(depth_subresource_range());
            let view = device
                .device
                .create_image_view(&view_ci, None)
                .map_err(vk_err)?;

            // The bindless descriptor records SHADER_READ_ONLY_OPTIMAL; the graph
            // transitions the image to that layout before a sampling pass.
            let index = device.register_sampled_image(view);

            Ok(Self {
                device,
                image,
                memory,
                view,
                index,
                extent: vk_extent,
                // Realized lazily as an attachment on first use.
                layout: Cell::new(vk::ImageLayout::UNDEFINED),
            })
        }
    }

    pub(crate) fn view(&self) -> vk::ImageView {
        self.view
    }

    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }

    pub(crate) fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub(crate) fn layout(&self) -> vk::ImageLayout {
        self.layout.get()
    }

    pub(crate) fn set_layout(&self, layout: vk::ImageLayout) {
        self.layout.set(layout);
    }

    /// Index of this depth buffer in the device's bindless table (shadow map).
    pub fn bindless_index(&self) -> u32 {
        self.index
    }
}

impl Drop for VulkanDepthBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_image_view(self.view, None);
            self.device.device.destroy_image(self.image, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}
