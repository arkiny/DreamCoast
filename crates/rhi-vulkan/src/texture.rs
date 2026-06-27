//! Sampled 2D textures: device-local image + staging upload + bindless registration.

use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::TextureDesc;

use crate::device::DeviceShared;
use crate::{to_vk_format, vk_err};

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

            // CPU-generated mip chain (identical bytes across backends — the
            // cross-backend-parity rule; see rhi_types::generate_mip_chain).
            let levels =
                rhi_types::generate_mip_chain(pixels, desc.width, desc.height, desc.format);
            let mip_levels = levels.len() as u32;
            let full_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            };

            // Device-local image with the full mip chain.
            let image_ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(extent)
                .mip_levels(mip_levels)
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

            // One staging buffer holding all mip levels back-to-back; one copy region
            // per level at its byte offset.
            let mut staging_bytes: Vec<u8> = Vec::new();
            let mut regions: Vec<vk::BufferImageCopy> = Vec::with_capacity(levels.len());
            for (mip, level) in levels.iter().enumerate() {
                let offset = staging_bytes.len() as u64;
                staging_bytes.extend_from_slice(level);
                let w = (desc.width >> mip).max(1);
                let h = (desc.height >> mip).max(1);
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(mip as u32)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: w,
                            height: h,
                            depth: 1,
                        }),
                );
            }
            let (staging, staging_mem) = create_staging(&device, &staging_bytes)?;

            // Upload: transition all levels, copy each, transition to shader-read.
            device.immediate_submit(|cmd| {
                image_barrier(
                    &device.device,
                    cmd,
                    image,
                    full_range,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                );
                device.device.cmd_copy_buffer_to_image(
                    cmd,
                    staging,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                );
                image_barrier(
                    &device.device,
                    cmd,
                    image,
                    full_range,
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
                .subresource_range(full_range);
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
pub(crate) fn create_staging(
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
pub(crate) fn image_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
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
        .subresource_range(range);
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
