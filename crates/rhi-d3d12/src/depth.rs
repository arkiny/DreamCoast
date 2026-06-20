//! Depth buffer (D32_FLOAT) + DSV for the mesh pass.

use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::Extent2D;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CLEAR_VALUE, D3D12_CLEAR_VALUE_0, D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_DEPTH_STENCIL_VALUE, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_NONE, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, D3D12_HEAP_FLAG_NONE,
    D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
    D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_TEXTURE_LAYOUT_UNKNOWN, ID3D12DescriptorHeap,
    ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_D32_FLOAT, DXGI_SAMPLE_DESC};

use crate::device::DeviceShared;
use crate::instance::d3d_err;

/// A depth resource + DSV, sized to the swapchain.
pub struct D3d12DepthBuffer {
    #[allow(dead_code)]
    device: Rc<DeviceShared>,
    #[allow(dead_code)] // kept alive; referenced by the DSV
    resource: ID3D12Resource,
    #[allow(dead_code)] // owns the DSV descriptor storage
    dsv_heap: ID3D12DescriptorHeap,
    dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
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
                Format: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            };
            let clear = D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_D32_FLOAT,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                        Depth: 1.0,
                        Stencil: 0,
                    },
                },
            };
            let mut res: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
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
            device.device.CreateDepthStencilView(&resource, None, dsv);

            Ok(Self {
                device,
                resource,
                dsv_heap,
                dsv,
            })
        }
    }

    pub(crate) fn dsv(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.dsv
    }
}
