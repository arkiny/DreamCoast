//! Host-visible buffers for dynamic per-frame upload (vertex/index).

use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::{BufferDesc, BufferUsage, StorageBufferDesc};

use crate::device::DeviceShared;
use crate::vk_err;

/// A persistently-mapped, host-visible buffer.
pub struct VulkanBuffer {
    device: Arc<DeviceShared>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    size: u64,
}

impl VulkanBuffer {
    pub(crate) fn new(device: Arc<DeviceShared>, desc: &BufferDesc) -> Result<Self, EngineError> {
        unsafe {
            let usage = match desc.usage {
                BufferUsage::Vertex => vk::BufferUsageFlags::VERTEX_BUFFER,
                BufferUsage::Index => vk::BufferUsageFlags::INDEX_BUFFER,
                BufferUsage::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER,
                BufferUsage::Readback => vk::BufferUsageFlags::TRANSFER_DST,
            };
            let ci = vk::BufferCreateInfo::default()
                .size(desc.size)
                .usage(usage)
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

            let mapped = device
                .device
                .map_memory(memory, 0, desc.size, vk::MemoryMapFlags::empty())
                .map_err(vk_err)? as *mut u8;

            Ok(Self {
                device,
                buffer,
                memory,
                mapped,
                size: desc.size,
            })
        }
    }

    /// Copy `data` into the buffer (clamped to its size). Host-coherent, so no
    /// explicit flush is needed.
    pub fn write(&self, data: &[u8]) -> Result<(), EngineError> {
        let n = data.len().min(self.size as usize);
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped, n) };
        Ok(())
    }

    /// Copy `data` into the buffer at `offset` (for per-frame slices).
    pub fn write_at(&self, offset: u64, data: &[u8]) -> Result<(), EngineError> {
        if offset + data.len() as u64 > self.size {
            return Err(EngineError::Rhi("buffer write_at out of bounds".into()));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.mapped.add(offset as usize),
                data.len(),
            )
        };
        Ok(())
    }

    /// Copy out of the buffer into `dst` (clamped to its size), for readback.
    /// Host-coherent, so no explicit invalidate is needed.
    pub fn read_into(&self, dst: &mut [u8]) {
        let n = dst.len().min(self.size as usize);
        unsafe { std::ptr::copy_nonoverlapping(self.mapped, dst.as_mut_ptr(), n) };
    }

    pub(crate) fn raw(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.unmap_memory(self.memory);
            self.device.device.destroy_buffer(self.buffer, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}

/// A device-local read-write storage buffer (UAV / `STORAGE_BUFFER`), registered
/// in the bindless storage-buffer table. Written by compute, optionally used as
/// an indirect-draw argument buffer (Phase 7).
pub struct VulkanStorageBuffer {
    device: Arc<DeviceShared>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    index: u32,
}

impl VulkanStorageBuffer {
    pub(crate) fn new(
        device: Arc<DeviceShared>,
        desc: &StorageBufferDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC;
            if desc.indirect {
                usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
            }
            let ci = vk::BufferCreateInfo::default()
                .size(desc.size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = device.device.create_buffer(&ci, None).map_err(vk_err)?;

            let req = device.device.get_buffer_memory_requirements(buffer);
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
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(vk_err)?;

            let index = device.register_storage_buffer(buffer, desc.size);
            Ok(Self {
                device,
                buffer,
                memory,
                index,
            })
        }
    }

    /// Index of this buffer in the bindless storage-buffer table.
    pub fn storage_index(&self) -> u32 {
        self.index
    }

    pub(crate) fn raw(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Drop for VulkanStorageBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_buffer(self.buffer, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}
