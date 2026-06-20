//! A render-target cubemap: a 6-layer, optionally mipped color image usable both
//! as a per-face/-mip color attachment (the IBL generation passes write into it)
//! and as a bindless `TextureCube` (shaders sample it by direction). Used for the
//! environment map and its derived irradiance / prefilter maps.

use std::cell::Cell;
use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::CubemapDesc;

use crate::device::DeviceShared;
use crate::{to_vk_format, vk_err};

/// A device-local cube image (6 faces × `mip_levels`), registered as a bindless
/// cubemap and carrying a 2D render view per (face, mip).
pub struct VulkanCubemap {
    device: Arc<DeviceShared>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// CUBE view over all faces/mips, used for sampling.
    sample_view: vk::ImageView,
    /// One 2D view per (face, mip), row-major `face * mip_levels + mip`.
    face_views: Vec<vk::ImageView>,
    index: u32,
    size: u32,
    mip_levels: u32,
    /// Current layout of the whole image, updated by the barrier helpers.
    layout: Cell<vk::ImageLayout>,
}

impl VulkanCubemap {
    pub(crate) fn new(device: Arc<DeviceShared>, desc: &CubemapDesc) -> Result<Self, EngineError> {
        unsafe {
            let format = to_vk_format(desc.format);
            let size = desc.size.max(1);
            let mip_levels = desc.mip_levels.max(1);

            let image_ci = vk::ImageCreateInfo::default()
                .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: size,
                    height: size,
                    depth: 1,
                })
                .mip_levels(mip_levels)
                .array_layers(6)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = device.device.create_image(&image_ci, None).map_err(vk_err)?;

            let req = device.device.get_image_memory_requirements(image);
            let mem_type = device
                .find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type);
            let memory = device.device.allocate_memory(&alloc, None).map_err(vk_err)?;
            device
                .device
                .bind_image_memory(image, memory, 0)
                .map_err(vk_err)?;

            // CUBE sampling view (all faces, all mips).
            let sample_view = device
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::CUBE)
                        .format(format)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: mip_levels,
                            base_array_layer: 0,
                            layer_count: 6,
                        }),
                    None,
                )
                .map_err(vk_err)?;

            // One 2D render view per (face, mip).
            let mut face_views = Vec::with_capacity(6 * mip_levels as usize);
            for face in 0..6 {
                for mip in 0..mip_levels {
                    let view = device
                        .device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(image)
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .format(format)
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    base_mip_level: mip,
                                    level_count: 1,
                                    base_array_layer: face,
                                    layer_count: 1,
                                }),
                            None,
                        )
                        .map_err(vk_err)?;
                    face_views.push(view);
                }
            }

            let index = device.register_sampled_cube(sample_view);

            Ok(Self {
                device,
                image,
                memory,
                sample_view,
                face_views,
                index,
                size,
                mip_levels,
                layout: Cell::new(vk::ImageLayout::UNDEFINED),
            })
        }
    }

    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }

    pub(crate) fn render_view(&self, face: u32, mip: u32) -> vk::ImageView {
        self.face_views[face as usize * self.mip_levels as usize + mip as usize]
    }

    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    /// Edge length of `mip` (`size >> mip`, at least 1).
    pub fn mip_size(&self, mip: u32) -> u32 {
        (self.size >> mip).max(1)
    }

    pub(crate) fn layout(&self) -> vk::ImageLayout {
        self.layout.get()
    }

    pub(crate) fn set_layout(&self, layout: vk::ImageLayout) {
        self.layout.set(layout);
    }

    /// Index of this cubemap in the bindless cube table.
    pub fn bindless_index(&self) -> u32 {
        self.index
    }
}

impl Drop for VulkanCubemap {
    fn drop(&mut self) {
        unsafe {
            for &v in &self.face_views {
                self.device.device.destroy_image_view(v, None);
            }
            self.device.device.destroy_image_view(self.sample_view, None);
            self.device.device.destroy_image(self.image, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}
