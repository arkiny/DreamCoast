//! A render-target cubemap: a 6-face, optionally mipped color texture usable as
//! a per-(face, mip) render target (the IBL generation passes write into it) and
//! as a bindless `TextureCube` (shaders sample it by direction). Backs the
//! environment map and its derived irradiance / prefilter maps.

use std::cell::Cell;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::CubemapDesc;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CLEAR_VALUE, D3D12_CLEAR_VALUE_0, D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_TYPE_DEFAULT, D3D12_MEMORY_POOL_UNKNOWN, D3D12_RENDER_TARGET_VIEW_DESC,
    D3D12_RENDER_TARGET_VIEW_DESC_0, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATES, D3D12_RTV_DIMENSION_TEXTURE2DARRAY, D3D12_TEX2D_ARRAY_RTV,
    D3D12_TEXTURE_LAYOUT_UNKNOWN, ID3D12DescriptorHeap, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use crate::device::DeviceShared;
use crate::instance::d3d_err;
use crate::to_dxgi_format;

/// A cube texture (6 faces × `mip_levels`) + per-(face, mip) RTVs + bindless cube
/// SRV. Its resource state is tracked for the render-target <-> shader-read
/// transitions during IBL generation.
pub struct D3d12Cubemap {
    #[allow(dead_code)]
    device: Rc<DeviceShared>,
    resource: ID3D12Resource,
    #[allow(dead_code)] // owns the RTV descriptor storage
    rtv_heap: ID3D12DescriptorHeap,
    /// First RTV CPU handle; per-(face, mip) handles are `start + (face*mips+mip)*size`.
    rtv_start: D3D12_CPU_DESCRIPTOR_HANDLE,
    rtv_size: u32,
    index: u32,
    size: u32,
    mip_levels: u32,
    state: Cell<D3D12_RESOURCE_STATES>,
}

impl D3d12Cubemap {
    pub(crate) fn new(device: Rc<DeviceShared>, desc: &CubemapDesc) -> Result<Self, EngineError> {
        unsafe {
            let size = desc.size.max(1);
            let mip_levels = desc.mip_levels.max(1);
            let format = to_dxgi_format(desc.format);

            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            };
            let res_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: size as u64,
                Height: size,
                DepthOrArraySize: 6, // cube = 6-layer 2D array
                MipLevels: mip_levels as u16,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            };
            let clear = D3D12_CLEAR_VALUE {
                Format: format,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    Color: [0.0, 0.0, 0.0, 1.0],
                },
            };
            // Created shader-readable; generation transitions it to RENDER_TARGET.
            let initial = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            let mut res: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &res_desc,
                    initial,
                    Some(&clear),
                    &mut res,
                )
                .map_err(d3d_err)?;
            let resource = res.ok_or_else(|| EngineError::Rhi("cubemap resource null".into()))?;

            // RTV per (face, mip).
            let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 6 * mip_levels,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            };
            let rtv_heap: ID3D12DescriptorHeap = device
                .device
                .CreateDescriptorHeap(&heap_desc)
                .map_err(d3d_err)?;
            let rtv_size = device
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            let rtv_start = rtv_heap.GetCPUDescriptorHandleForHeapStart();
            for face in 0..6u32 {
                for mip in 0..mip_levels {
                    let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
                        Format: format,
                        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2DARRAY,
                        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                            Texture2DArray: D3D12_TEX2D_ARRAY_RTV {
                                MipSlice: mip,
                                FirstArraySlice: face,
                                ArraySize: 1,
                                PlaneSlice: 0,
                            },
                        },
                    };
                    let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: rtv_start.ptr
                            + ((face * mip_levels + mip) as usize) * (rtv_size as usize),
                    };
                    device
                        .device
                        .CreateRenderTargetView(&resource, Some(&rtv_desc), handle);
                }
            }

            let index = device.register_sampled_cube(&resource, desc.format, mip_levels);

            Ok(Self {
                device,
                resource,
                rtv_heap,
                rtv_start,
                rtv_size,
                index,
                size,
                mip_levels,
                state: Cell::new(initial),
            })
        }
    }

    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    pub(crate) fn rtv_handle(&self, face: u32, mip: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.rtv_start.ptr
                + ((face * self.mip_levels + mip) as usize) * (self.rtv_size as usize),
        }
    }

    pub(crate) fn state(&self) -> D3D12_RESOURCE_STATES {
        self.state.get()
    }

    pub(crate) fn set_state(&self, state: D3D12_RESOURCE_STATES) {
        self.state.set(state);
    }

    /// Edge length of `mip` (`size >> mip`, at least 1).
    pub fn mip_size(&self, mip: u32) -> u32 {
        (self.size >> mip).max(1)
    }

    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    /// Index of this cubemap in the bindless cube table.
    pub fn bindless_index(&self) -> u32 {
        self.index
    }

    /// Tag this cubemap's resource with a debug name (Phase 9 M2).
    pub fn set_name(&self, name: &str) {
        if !cfg!(debug_assertions) {
            return;
        }
        let wide = windows::core::HSTRING::from(name);
        unsafe {
            let _ = self.resource.SetName(&wide);
        }
    }
}
