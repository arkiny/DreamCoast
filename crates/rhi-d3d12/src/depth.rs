//! Depth buffer + DSV for the mesh pass.
//!
//! The resource is created `R32_TYPELESS` so it can carry both a `D32_FLOAT`
//! depth-stencil view (writing) and an `R32_FLOAT` shader-resource view
//! (sampling), letting it double as a shadow map. Its current resource state is
//! tracked so the render graph can emit the DEPTH_WRITE <-> shader-read barriers.

use std::cell::Cell;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::Extent2D;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CLEAR_VALUE, D3D12_CLEAR_VALUE_0, D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_DEPTH_STENCIL_VALUE, D3D12_DEPTH_STENCIL_VIEW_DESC,
    D3D12_DEPTH_STENCIL_VIEW_DESC_0, D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    D3D12_DESCRIPTOR_HEAP_TYPE_DSV, D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_DSV_FLAG_NONE,
    D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE,
    D3D12_RESOURCE_STATES, D3D12_TEX2D_DSV, D3D12_TEXTURE_LAYOUT_UNKNOWN, ID3D12DescriptorHeap,
    ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC,
};

use crate::device::DeviceShared;
use crate::instance::d3d_err;

/// A depth resource + DSV, registered as a bindless `R32_FLOAT` sampled texture.
pub struct D3d12DepthBuffer {
    #[allow(dead_code)]
    device: Rc<DeviceShared>,
    resource: ID3D12Resource,
    #[allow(dead_code)] // owns the DSV descriptor storage
    dsv_heap: ID3D12DescriptorHeap,
    dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    index: u32,
    /// Current resource state, updated by the barrier helpers.
    state: Cell<D3D12_RESOURCE_STATES>,
}

impl D3d12DepthBuffer {
    pub(crate) fn new(device: Rc<DeviceShared>, extent: Extent2D) -> Result<Self, EngineError> {
        unsafe {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: extent.width.max(1) as u64,
                Height: extent.height.max(1),
                DepthOrArraySize: 1,
                MipLevels: 1,
                // Typeless so it carries both a D32 DSV and an R32 SRV.
                Format: DXGI_FORMAT_R32_TYPELESS,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            };
            // Optimized clear value uses the typed depth format.
            let clear = D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_D32_FLOAT,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                        Depth: 1.0,
                        Stencil: 0,
                    },
                },
            };
            let initial = D3D12_RESOURCE_STATE_DEPTH_WRITE;
            let mut res: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    initial,
                    Some(&clear),
                    &mut res,
                )
                .map_err(d3d_err)?;
            let resource = res.ok_or_else(|| EngineError::Rhi("depth resource null".into()))?;

            let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            };
            let dsv_heap: ID3D12DescriptorHeap = device
                .device
                .CreateDescriptorHeap(&heap_desc)
                .map_err(d3d_err)?;
            let dsv = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            // Typeless resource needs an explicit DSV format.
            let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: DXGI_FORMAT_D32_FLOAT,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                Flags: D3D12_DSV_FLAG_NONE,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_DSV { MipSlice: 0 },
                },
            };
            device
                .device
                .CreateDepthStencilView(&resource, Some(&dsv_desc), dsv);

            // R32_FLOAT shader-resource view in the bindless heap (shadow map).
            let index = device.register_sampled_depth(&resource);

            Ok(Self {
                device,
                resource,
                dsv_heap,
                dsv,
                index,
                state: Cell::new(initial),
            })
        }
    }

    pub(crate) fn dsv(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.dsv
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

    /// Index of this depth buffer in the device's bindless SRV heap (shadow map).
    pub fn bindless_index(&self) -> u32 {
        self.index
    }

    /// Tag this depth buffer's resource with a debug name (Phase 9 M2).
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
