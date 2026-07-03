//! Host-visible UPLOAD-heap buffers for dynamic per-frame data (vertex/index).

use std::cell::Cell;
use std::ffi::c_void;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::{BufferDesc, BufferUsage, StorageBufferDesc};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_HEAP_FLAG_NONE,
    D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_CUSTOM, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD, D3D12_MEMORY_POOL_L0,
    D3D12_MEMORY_POOL_UNKNOWN, D3D12_RANGE, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    D3D12_RESOURCE_STATES, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

use crate::device::DeviceShared;
use crate::instance::d3d_err;

/// A persistently-mapped UPLOAD-heap buffer.
pub struct D3d12Buffer {
    #[allow(dead_code)]
    device: Rc<DeviceShared>,
    resource: ID3D12Resource,
    mapped: *mut u8,
    size: u64,
}

impl D3d12Buffer {
    pub(crate) fn new(device: Rc<DeviceShared>, desc: &BufferDesc) -> Result<Self, EngineError> {
        unsafe {
            // Readback uses a READBACK heap (GPU writes via copy, CPU reads);
            // everything else uses an UPLOAD heap (CPU writes, GPU reads).
            let (heap_type, initial_state) = match desc.usage {
                BufferUsage::Readback => (D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST),
                _ => (D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ),
            };
            let heap = D3D12_HEAP_PROPERTIES {
                Type: heap_type,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            };
            let res_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: 0,
                Width: desc.size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut res: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &res_desc,
                    initial_state,
                    None,
                    &mut res,
                )
                .map_err(d3d_err)?;
            let resource = res.ok_or_else(|| EngineError::Rhi("buffer was null".into()))?;

            let mut ptr: *mut c_void = std::ptr::null_mut();
            resource.Map(0, None, Some(&mut ptr)).map_err(d3d_err)?;

            Ok(Self {
                device,
                resource,
                mapped: ptr as *mut u8,
                size: desc.size,
            })
        }
    }

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
    /// `Map` with a full read range makes the GPU-written contents CPU-visible.
    pub fn read_into(&self, dst: &mut [u8]) -> Result<(), EngineError> {
        let n = dst.len().min(self.size as usize);
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let range = D3D12_RANGE {
                Begin: 0,
                End: self.size as usize,
            };
            self.resource
                .Map(0, Some(&range), Some(&mut ptr))
                .map_err(d3d_err)?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, dst.as_mut_ptr(), n);
            // Written range empty: the CPU did not modify the buffer.
            self.resource
                .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        }
        Ok(())
    }

    pub fn gpu_va(&self) -> u64 {
        unsafe { self.resource.GetGPUVirtualAddress() }
    }

    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    pub(crate) fn size(&self) -> u64 {
        self.size
    }
}

/// A DEFAULT-heap read-write storage buffer (UAV), registered in the bindless
/// storage-buffer table. Written by compute, optionally used as an indirect-draw
/// argument buffer (Phase 7). Its resource state is tracked for UAV/indirect
/// transitions.
pub struct D3d12StorageBuffer {
    #[allow(dead_code)]
    device: Rc<DeviceShared>,
    resource: ID3D12Resource,
    index: u32,
    state: Cell<D3D12_RESOURCE_STATES>,
    /// Persistently-mapped host pointer for the host-visible (`new_host`) variant — the per-frame
    /// GPU-skinning palette (animation Stage B.2c); null for the default DEFAULT-heap buffer.
    mapped: *mut u8,
    size: u64,
}

impl D3d12StorageBuffer {
    pub(crate) fn new(
        device: Rc<DeviceShared>,
        desc: &StorageBufferDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            };
            let res_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: 0,
                Width: desc.size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            };
            let initial = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            let mut res: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &res_desc,
                    initial,
                    None,
                    &mut res,
                )
                .map_err(d3d_err)?;
            let resource = res.ok_or_else(|| EngineError::Rhi("storage buffer null".into()))?;

            let index = device.register_storage_buffer(&resource, desc.size);
            Ok(Self {
                device,
                resource,
                index,
                state: Cell::new(initial),
                mapped: std::ptr::null_mut(),
                size: desc.size,
            })
        }
    }

    /// Host-visible, persistently-mapped storage buffer (animation Stage B.2c): the per-frame
    /// GPU-skinning joint palette the vertex shader reads from `g.storage_buffers[]`. A DEFAULT
    /// (UPLOAD) heap can't be both CPU-writable and a UAV, so this uses a CUSTOM **L0** (system
    /// memory) **write-combine** heap, which is CPU-mappable AND UAV-capable — `write()` is a plain
    /// memcpy and the GPU reads the UAV directly (over PCIe), matching the VK HOST_COHERENT /
    /// Metal Shared semantics. Registered in the same bindless UAV table as every storage buffer.
    pub(crate) fn new_host(
        device: Rc<DeviceShared>,
        desc: &StorageBufferDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            };
            let res_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: 0,
                Width: desc.size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            };
            let initial = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            let mut res: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &res_desc,
                    initial,
                    None,
                    &mut res,
                )
                .map_err(d3d_err)?;
            let resource =
                res.ok_or_else(|| EngineError::Rhi("host storage buffer null".into()))?;
            let mut ptr: *mut c_void = std::ptr::null_mut();
            resource.Map(0, None, Some(&mut ptr)).map_err(d3d_err)?;
            let index = device.register_storage_buffer(&resource, desc.size);
            Ok(Self {
                device,
                resource,
                index,
                state: Cell::new(initial),
                mapped: ptr as *mut u8,
                size: desc.size,
            })
        }
    }

    /// Create a DEFAULT-heap storage buffer seeded with host `data` via an UPLOAD
    /// staging copy (Phase 8: ray-tracing geometry + instance table). Uploaded
    /// once at scene setup through a one-shot transfer; ends in UAV state.
    pub(crate) fn new_init(
        device: Rc<DeviceShared>,
        desc: &StorageBufferDesc,
        data: &[u8],
    ) -> Result<Self, EngineError> {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_FLAG_NONE,
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_TRANSITION_BARRIER,
        };
        let sb = Self::new(device.clone(), desc)?;
        unsafe {
            // UPLOAD staging buffer (CPU writes, GPU copies to the DEFAULT target).
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_UPLOAD,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            };
            let res_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: 0,
                Width: desc.size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAG_NONE,
            };
            let mut staging: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &res_desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut staging,
                )
                .map_err(d3d_err)?;
            let staging = staging.ok_or_else(|| EngineError::Rhi("staging null".into()))?;
            let mut ptr: *mut c_void = std::ptr::null_mut();
            staging.Map(0, None, Some(&mut ptr)).map_err(d3d_err)?;
            let n = data.len().min(desc.size as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, n);
            staging.Unmap(0, None);

            let transition = |from, to| D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(Some(std::mem::transmute_copy(
                            &sb.resource,
                        ))),
                        Subresource: 0,
                        StateBefore: from,
                        StateAfter: to,
                    }),
                },
            };
            device.immediate_submit(|list| {
                list.ResourceBarrier(&[transition(
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
                list.CopyBufferRegion(&sb.resource, 0, &staging, 0, desc.size);
                list.ResourceBarrier(&[transition(
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            })?;
        }
        Ok(sb)
    }

    /// Index of this buffer in the bindless storage-buffer table.
    pub fn storage_index(&self) -> u32 {
        self.index
    }

    /// Host-write the buffer (GPU-skinning joint palette). Valid only for the host-visible
    /// `new_host` variant (CUSTOM L0 write-combine heap, persistently mapped → plain memcpy,
    /// immediately GPU-visible). The default DEFAULT-heap buffer is not host-writable.
    pub fn write(&self, data: &[u8]) -> Result<(), EngineError> {
        if self.mapped.is_null() {
            return Err(EngineError::Rhi(
                "host-write of a DEFAULT-heap D3D12 storage buffer (use create_storage_buffer_host)"
                    .into(),
            ));
        }
        let n = (data.len() as u64).min(self.size) as usize;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped, n) };
        Ok(())
    }

    /// Host-read the buffer into `dst` (HZB cull stats). Valid only for the host-visible
    /// `new_host` variant. The L0 heap is write-combined, so CPU reads are slow — this is
    /// a diagnostic-only path (tiny buffer, off by default). Caller syncs GPU writes
    /// first. (DX/VK parity pending Windows verification.)
    pub fn read_into(&self, dst: &mut [u8]) -> Result<(), EngineError> {
        if self.mapped.is_null() {
            return Err(EngineError::Rhi(
                "host-read of a DEFAULT-heap D3D12 storage buffer (use create_storage_buffer_host)"
                    .into(),
            ));
        }
        let n = (dst.len() as u64).min(self.size) as usize;
        unsafe { std::ptr::copy_nonoverlapping(self.mapped as *const u8, dst.as_mut_ptr(), n) };
        Ok(())
    }

    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    pub(crate) fn state(&self) -> D3D12_RESOURCE_STATES {
        self.state.get()
    }

    pub(crate) fn set_state(&self, state: D3D12_RESOURCE_STATES) {
        self.state.set(state);
    }
}

impl Drop for D3d12StorageBuffer {
    fn drop(&mut self) {
        // Return the bindless slot to the free-list; the COM `resource` releases itself. Safe
        // because the handoff contract defers this Drop until the referencing frames retire.
        self.device.free_storage_buffer(self.index);
    }
}
