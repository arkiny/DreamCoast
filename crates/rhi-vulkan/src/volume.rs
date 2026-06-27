//! A 3D (volume) texture, usable as both a compute-writable storage volume
//! (bindless `storage_volumes[]`, binding 7) and a trilinear-sampled volume
//! (bindless `volumes[]`, binding 6). Phase 11 Stage B distance fields. Its current
//! layout is tracked so the caller can barrier between baking (GENERAL) and sampling
//! (SHADER_READ_ONLY_OPTIMAL), mirroring the 2D storage render target.

use std::cell::Cell;
use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::VolumeDesc;

use crate::device::DeviceShared;
use crate::{color_subresource_range, to_vk_format, vk_err};

/// A device-local 3D texture registered in both bindless volume tables.
pub struct VulkanVolume {
    device: Arc<DeviceShared>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// `volumes[]` (SRV) index — trilinear sampling.
    sampled_index: u32,
    /// `storage_volumes[]` (UAV) index — compute writes.
    storage_index: u32,
    /// Current image layout, updated by the barrier helpers.
    layout: Cell<vk::ImageLayout>,
}

impl VulkanVolume {
    pub(crate) fn new(device: Arc<DeviceShared>, desc: &VolumeDesc) -> Result<Self, EngineError> {
        unsafe {
            let format = to_vk_format(desc.format);
            let extent = vk::Extent3D {
                width: desc.width.max(1),
                height: desc.height.max(1),
                depth: desc.depth.max(1),
            };
            let image_ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_3D)
                .format(format)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
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
                .view_type(vk::ImageViewType::TYPE_3D)
                .format(format)
                .subresource_range(color_subresource_range());
            let view = device
                .device
                .create_image_view(&view_ci, None)
                .map_err(vk_err)?;

            // One view, registered in both tables: SHADER_READ_ONLY for sampling,
            // GENERAL for storage writes (the layout is tracked per use).
            let sampled_index = device.register_volume(view);
            let storage_index = device.register_storage_volume(view);

            Ok(Self {
                device,
                image,
                memory,
                view,
                sampled_index,
                storage_index,
                layout: Cell::new(vk::ImageLayout::UNDEFINED),
            })
        }
    }

    /// Create a volume seeded with host `data` (Phase 12 M2: a CPU-baked SDF
    /// uploaded verbatim instead of a GPU bake). `data` is `width*height*depth`
    /// voxels in `x + dim*(y + dim*z)` order, matching the bake's linear buffer.
    /// Adds `TRANSFER_DST` usage, stages the bytes, and leaves the image in
    /// `SHADER_READ_ONLY_OPTIMAL` ready to sample.
    pub(crate) fn new_init(
        device: Arc<DeviceShared>,
        desc: &VolumeDesc,
        data: &[u8],
    ) -> Result<Self, EngineError> {
        unsafe {
            let format = to_vk_format(desc.format);
            let extent = vk::Extent3D {
                width: desc.width.max(1),
                height: desc.height.max(1),
                depth: desc.depth.max(1),
            };
            let image_ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_3D)
                .format(format)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_DST,
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

            // Stage the voxels and copy the whole 3D extent in one region, then
            // transition to shader-read.
            let (staging, staging_mem) = crate::texture::create_staging(&device, data)?;
            let range = color_subresource_range();
            let region = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_extent(extent);
            device.immediate_submit(|cmd| {
                crate::texture::image_barrier(
                    &device.device,
                    cmd,
                    image,
                    range,
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
                    &[region],
                );
                crate::texture::image_barrier(
                    &device.device,
                    cmd,
                    image,
                    range,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER
                        | vk::PipelineStageFlags::FRAGMENT_SHADER,
                );
            })?;
            device.device.destroy_buffer(staging, None);
            device.device.free_memory(staging_mem, None);

            let view_ci = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_3D)
                .format(format)
                .subresource_range(color_subresource_range());
            let view = device
                .device
                .create_image_view(&view_ci, None)
                .map_err(vk_err)?;

            let sampled_index = device.register_volume(view);
            let storage_index = device.register_storage_volume(view);

            Ok(Self {
                device,
                image,
                memory,
                view,
                sampled_index,
                storage_index,
                layout: Cell::new(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            })
        }
    }

    /// Read the volume back to host memory (Phase 12 item 3) as `w*h*d*bpp`
    /// tightly-packed bytes in `x + dim*(y + dim*z)` order — the inverse of
    /// `new_init`. Synchronous: copies the image into a host-visible buffer via a
    /// one-shot submit, then maps it. Vulkan packs the copy tightly (no row pad),
    /// so the buffer is returned as-is. Leaves the image back in its prior layout.
    pub(crate) fn read_back(
        &self,
        w: u32,
        h: u32,
        d: u32,
        bpp: u32,
    ) -> Result<Vec<u8>, EngineError> {
        let device = &self.device;
        let size = (w as u64) * (h as u64) * (d as u64) * (bpp as u64);
        unsafe {
            // Host-visible TRANSFER_DST buffer for the readback.
            let ci = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
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

            let prior = self.layout.get();
            let range = color_subresource_range();
            let region = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: d,
                });
            device.immediate_submit(|cmd| {
                crate::texture::image_barrier(
                    &device.device,
                    cmd,
                    self.image,
                    range,
                    prior,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::SHADER_READ,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::PipelineStageFlags::COMPUTE_SHADER
                        | vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                );
                device.device.cmd_copy_image_to_buffer(
                    cmd,
                    self.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    buffer,
                    &[region],
                );
                crate::texture::image_barrier(
                    &device.device,
                    cmd,
                    self.image,
                    range,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    prior,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER
                        | vk::PipelineStageFlags::FRAGMENT_SHADER,
                );
            })?;

            let ptr = device
                .device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(vk_err)? as *const u8;
            let mut out = vec![0u8; size as usize];
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), size as usize);
            device.device.unmap_memory(memory);
            device.device.destroy_buffer(buffer, None);
            device.device.free_memory(memory, None);
            Ok(out)
        }
    }

    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }

    pub(crate) fn layout(&self) -> vk::ImageLayout {
        self.layout.get()
    }

    pub(crate) fn set_layout(&self, layout: vk::ImageLayout) {
        self.layout.set(layout);
    }

    /// `volumes[]` (SRV) index for trilinear sampling.
    pub fn sampled_index(&self) -> u32 {
        self.sampled_index
    }

    /// `storage_volumes[]` (UAV) index for compute writes.
    pub fn storage_index(&self) -> u32 {
        self.storage_index
    }
}

impl Drop for VulkanVolume {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_image_view(self.view, None);
            self.device.device.destroy_image(self.image, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}
