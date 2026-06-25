//! A 3D (volume) texture, usable as both a compute-writable storage volume
//! (bindless `storage_volumes[]`) and a trilinear-sampled volume (bindless
//! `volumes[]`). Phase 11 Stage B distance fields. Its current resource state is
//! tracked so the caller can barrier between baking (UNORDERED_ACCESS) and sampling
//! (NON_PIXEL_SHADER_RESOURCE), mirroring the 2D storage render target.

use std::cell::Cell;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::VolumeDesc;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_TEXTURE3D,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATES, D3D12_TEXTURE_LAYOUT_UNKNOWN, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use crate::device::DeviceShared;
use crate::instance::d3d_err;
use crate::to_dxgi_format;

/// A device-local 3D texture registered in both bindless volume tables.
pub struct D3d12Volume {
    #[allow(dead_code)] // keeps the GPU resource alive while its views are bound
    resource: ID3D12Resource,
    sampled_index: u32,
    storage_index: u32,
    state: Cell<D3D12_RESOURCE_STATES>,
}

impl D3d12Volume {
    pub(crate) fn new(device: Rc<DeviceShared>, desc: &VolumeDesc) -> Result<Self, EngineError> {
        unsafe {
            let heap_props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CPUPageProperty: Default::default(),
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            };
            let res_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
                Alignment: 0,
                Width: desc.width.max(1) as u64,
                Height: desc.height.max(1),
                DepthOrArraySize: desc.depth.max(1) as u16,
                MipLevels: 1,
                Format: to_dxgi_format(desc.format),
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            };
            // Created shader-readable; the caller transitions to UNORDERED_ACCESS
            // before the bake pass and back to NON_PIXEL_SHADER_RESOURCE to sample.
            let initial = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            let mut res: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &heap_props,
                    D3D12_HEAP_FLAG_NONE,
                    &res_desc,
                    initial,
                    None,
                    &mut res,
                )
                .map_err(d3d_err)?;
            let resource = res.ok_or_else(|| EngineError::Rhi("volume null".into()))?;

            let sampled_index = device.register_volume(&resource, desc.format);
            let storage_index =
                device.register_storage_volume(&resource, desc.format, desc.depth.max(1));
            Ok(Self {
                resource,
                sampled_index,
                storage_index,
                state: Cell::new(initial),
            })
        }
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

    /// `volumes[]` (SRV) index for trilinear sampling.
    pub fn sampled_index(&self) -> u32 {
        self.sampled_index
    }

    /// `storage_volumes[]` (UAV) index for compute writes.
    pub fn storage_index(&self) -> u32 {
        self.storage_index
    }
}
