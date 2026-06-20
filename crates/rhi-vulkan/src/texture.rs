//! Sampled 2D textures: device-local image + staging upload + bindless registration.

use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::TextureDesc;

use crate::device::DeviceShared;
use crate::{color_subresource_range, to_vk_format, vk_err};

/// A device-local sampled texture, registered in the bindless table.
pub struct VulkanTexture {
    device: Arc<DeviceShared>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    index: u32,
}

impl VulkanTexture {
    pub(crate) fn new(
        device: Arc<DeviceShared>,
        desc: &TextureDesc,
        pixels: &[u8],
    ) -> Result<Self, EngineError> {
        unsafe {
            let format = to_vk_format(desc.format);
            let extent = vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: 1,
            };

            // Device-local image.
            let image_ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
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

            // Staging buffer with the pixel data.
            let (staging, staging_mem) = create_staging(&device, pixels)?;

            // Upload: transition, copy, transition to shader-read.
            device.immediate_submit(|cmd| {
                image_barrier(
                    &device.device,
                    cmd,
                    image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                );
                let region = vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .image_extent(extent);
                device.device.cmd_copy_buffer_to_image(
                    cmd,
                    staging,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
                image_barrier(
                    &device.device,
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                );
            })?;

            device.device.destroy_buffer(staging, None);
            device.device.free_memory(staging_mem, None);

            let view_ci = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(color_subresource_range());
            let view = device
                .device
                .create_image_view(&view_ci, None)
                .map_err(vk_err)?;

            let index = device.register_sampled_image(view);

            Ok(Self {
                device,
                image,
                memory,
                view,
                index,
            })
        }
    }

    /// The texture's index in the bindless table.
    pub fn bindless_index(&self) -> u32 {
        self.index
    }
}

impl Drop for VulkanTexture {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_image_view(self.view, None);
            self.device.device.destroy_image(self.image, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}

/// Create a host-visible staging buffer pre-filled with `pixels`.
fn create_staging(
    device: &DeviceShared,
    pixels: &[u8],
) -> Result<(vk::Buffer, vk::DeviceMemory), EngineError> {
    unsafe {
        let ci = vk::BufferCreateInfo::default()
            .size(pixels.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = device.device.create_buffer(&ci, None).map_err(vk_err)?;
        let req = device.device.get_buffer_memory_requirements(buffer);
        let mem_type = device.find_memory_type(
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let memory = device
            .device
            .allocate_memory(&alloc, None)
            .map_err(vk_err)?;
        device
            .device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(vk_err)?;
        let ptr = device
            .device
            .map_memory(memory, 0, pixels.len() as u64, vk::MemoryMapFlags::empty())
            .map_err(vk_err)? as *mut u8;
        std::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr, pixels.len());
        device.device.unmap_memory(memory);
        Ok((buffer, memory))
    }
}

#[allow(clippy::too_many_arguments)]
fn image_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old)
        .new_layout(new)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(color_subresource_range());
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}
