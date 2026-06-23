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
            let mut usage = match desc.usage {
                BufferUsage::Vertex => vk::BufferUsageFlags::VERTEX_BUFFER,
                BufferUsage::Index => vk::BufferUsageFlags::INDEX_BUFFER,
                BufferUsage::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER,
                BufferUsage::Readback => vk::BufferUsageFlags::TRANSFER_DST,
            };
            // When hardware ray tracing is available, vertex/index buffers double as
            // bottom-level acceleration-structure build inputs (Phase 8): the BLAS
            // build reads their device addresses, so they need the device-address +
            // AS-build-input usages and a device-address-flagged allocation. Additive
            // and gated, so non-RT devices are unaffected.
            let rt_geometry = device.has_raytracing
                && matches!(desc.usage, BufferUsage::Vertex | BufferUsage::Index);
            if rt_geometry {
                usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
            }
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
            let mut flags_info = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type);
            if rt_geometry {
                alloc = alloc.push_next(&mut flags_info);
            }
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

    /// GPU device address of this buffer (Phase 8 BLAS build input). Valid only
    /// when the buffer was created on an RT-capable device with a vertex/index
    /// usage (see `new`); otherwise the address query is invalid.
    pub(crate) fn device_address(&self) -> u64 {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe { self.device.device.get_buffer_device_address(&info) }
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
            // Storage buffers may be touched by both the graphics and async-compute
            // queues; CONCURRENT sharing across the two families avoids per-use queue
            // ownership transfers (only meaningful when the families differ).
            let families = [device.graphics_family, device.compute_family];
            let mut ci = vk::BufferCreateInfo::default().size(desc.size).usage(usage);
            ci = if device.has_dedicated_compute {
                ci.sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families)
            } else {
                ci.sharing_mode(vk::SharingMode::EXCLUSIVE)
            };
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

    /// Create a device-local storage buffer seeded with host `data` via a staging
    /// copy (Phase 8: ray-tracing geometry + instance table read by the path
    /// tracer). Uploaded once at scene setup through a one-shot transfer.
    pub(crate) fn new_init(
        device: Arc<DeviceShared>,
        desc: &StorageBufferDesc,
        data: &[u8],
    ) -> Result<Self, EngineError> {
        let sb = Self::new(device.clone(), desc)?;
        unsafe {
            // Host-visible staging buffer, copied into the device-local target.
            let ci = vk::BufferCreateInfo::default()
                .size(desc.size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let staging = device.device.create_buffer(&ci, None).map_err(vk_err)?;
            let req = device.device.get_buffer_memory_requirements(staging);
            let mem_type = device.find_memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type);
            let mem = device
                .device
                .allocate_memory(&alloc, None)
                .map_err(vk_err)?;
            device
                .device
                .bind_buffer_memory(staging, mem, 0)
                .map_err(vk_err)?;
            let ptr = device
                .device
                .map_memory(mem, 0, desc.size, vk::MemoryMapFlags::empty())
                .map_err(vk_err)? as *mut u8;
            let n = data.len().min(desc.size as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, n);
            device.device.unmap_memory(mem);

            device.immediate_submit(|cmd| {
                let region = vk::BufferCopy::default().size(desc.size);
                device
                    .device
                    .cmd_copy_buffer(cmd, staging, sb.buffer, &[region]);
            })?;

            device.device.destroy_buffer(staging, None);
            device.device.free_memory(mem, None);
        }
        Ok(sb)
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
