//! Host-visible UPLOAD-heap buffers for dynamic per-frame data (vertex/index).

use std::ffi::c_void;
use std::rc::Rc;

use engine_core::EngineError;
use rhi_types::BufferDesc;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_TYPE_UPLOAD, D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12Resource,
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
                ..Default::default()
            };
            let mut res: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &res_desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
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

    pub(crate) fn gpu_va(&self) -> u64 {
        unsafe { self.resource.GetGPUVirtualAddress() }
    }

    pub(crate) fn size(&self) -> u64 {
        self.size
    }
}
