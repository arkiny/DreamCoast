//! Offscreen color render target: a render-target + bindless-sampled texture.
//! Its current state is tracked so the render graph can emit RT<->shader-read
//! transitions.
//!
//! A target's storage is either a dedicated **committed** resource or a
//! **placed** resource inside a [`D3d12TransientHeap`] at a graph-computed
//! offset, so transient targets with non-overlapping lifetimes share memory.

use std::cell::Cell;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::{MemoryRequirements, RenderTargetDesc};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_NONE, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_HEAP_DESC,
    D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_TYPE_DEFAULT, D3D12_MEMORY_POOL_UNKNOWN, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAGS,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET,
    D3D12_RESOURCE_STATES, D3D12_TEXTURE_LAYOUT_UNKNOWN, ID3D12DescriptorHeap, ID3D12Heap,
    ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use crate::device::DeviceShared;
use crate::instance::d3d_err;
use crate::to_dxgi_format;

/// A color render target (resource + RTV + bindless SRV).
pub struct D3d12RenderTarget {
    #[allow(dead_code)]
    device: Rc<DeviceShared>,
    resource: ID3D12Resource,
    #[allow(dead_code)] // owns the RTV descriptor storage
    rtv_heap: ID3D12DescriptorHeap,
    #[allow(dead_code)] // keeps the placed-resource heap alive (None if committed)
    heap: Option<ID3D12Heap>,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    index: u32,
    /// Bindless storage-image (UAV) index, when created with `storage` (Phase 7).
    storage_index: Option<u32>,
    /// Current resource state, updated by the barrier helpers.
    state: Cell<D3D12_RESOURCE_STATES>,
}

impl D3d12RenderTarget {
    /// Create a target backed by its own dedicated (committed) allocation.
    pub(crate) fn new(
        device: Rc<DeviceShared>,
        desc: &RenderTargetDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let heap_props = default_heap();
            let res_desc = resource_desc(desc);
            // Created shader-readable; the graph transitions to RENDER_TARGET
            // before the first writing pass.
            let initial = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
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
            let resource = res.ok_or_else(|| EngineError::Rhi("render target null".into()))?;
            Self::finish(device, resource, None, desc, initial)
        }
    }

    /// Create a target as a placed resource inside `heap` at `offset`.
    pub(crate) fn new_aliased(
        device: Rc<DeviceShared>,
        heap: &D3d12TransientHeap,
        offset: u64,
        desc: &RenderTargetDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let res_desc = resource_desc(desc);
            // Placed targets start in RENDER_TARGET; an aliasing barrier discards
            // the previous tenant's content before each frame's first write.
            let initial = D3D12_RESOURCE_STATE_RENDER_TARGET;
            let mut res: Option<ID3D12Resource> = None;
            device
                .device
                .CreatePlacedResource(&heap.heap, offset, &res_desc, initial, None, &mut res)
                .map_err(d3d_err)?;
            let resource = res.ok_or_else(|| EngineError::Rhi("placed target null".into()))?;
            Self::finish(device, resource, Some(heap.heap.clone()), desc, initial)
        }
    }

    fn finish(
        device: Rc<DeviceShared>,
        resource: ID3D12Resource,
        heap: Option<ID3D12Heap>,
        desc: &RenderTargetDesc,
        initial: D3D12_RESOURCE_STATES,
    ) -> Result<Self, EngineError> {
        unsafe {
            let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            };
            let rtv_heap: ID3D12DescriptorHeap = device
                .device
                .CreateDescriptorHeap(&heap_desc)
                .map_err(d3d_err)?;
            let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
            device.device.CreateRenderTargetView(&resource, None, rtv);
            let index = device.register_texture(&resource, desc.format);
            let storage_index = if desc.storage {
                Some(device.register_storage_image(&resource, desc.format))
            } else {
                None
            };
            Ok(Self {
                device,
                resource,
                rtv_heap,
                heap,
                rtv,
                index,
                storage_index,
                state: Cell::new(initial),
            })
        }
    }

    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    pub(crate) fn rtv_handle(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.rtv
    }

    pub(crate) fn state(&self) -> D3D12_RESOURCE_STATES {
        self.state.get()
    }

    pub(crate) fn set_state(&self, state: D3D12_RESOURCE_STATES) {
        self.state.set(state);
    }

    /// Index of this target in the device's bindless SRV heap.
    pub fn bindless_index(&self) -> u32 {
        self.index
    }

    /// Tag this target's resource with a debug name (Phase 9 M2).
    pub fn set_name(&self, name: &str) {
        if !cfg!(debug_assertions) {
            return;
        }
        let wide = windows::core::HSTRING::from(name);
        unsafe {
            let _ = self.resource.SetName(&wide);
        }
    }

    /// Bindless storage-image (UAV) index, if created with `storage`.
    pub fn storage_index(&self) -> Option<u32> {
        self.storage_index
    }
}

fn default_heap() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    }
}

fn resource_desc(desc: &RenderTargetDesc) -> D3D12_RESOURCE_DESC {
    let mut flags: D3D12_RESOURCE_FLAGS = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if desc.storage {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: desc.width.max(1) as u64,
        Height: desc.height.max(1),
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: to_dxgi_format(desc.format),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
    }
}

/// Query the placed-resource memory footprint of a render target.
pub(crate) fn render_target_memory(
    device: &DeviceShared,
    desc: &RenderTargetDesc,
) -> Result<MemoryRequirements, EngineError> {
    unsafe {
        let info = device
            .device
            .GetResourceAllocationInfo(0, &[resource_desc(desc)]);
        Ok(MemoryRequirements {
            size: info.SizeInBytes,
            alignment: info.Alignment,
        })
    }
}

/// A DEFAULT heap that transient render targets are placed into at graph-computed
/// offsets.
pub struct D3d12TransientHeap {
    #[allow(dead_code)]
    device: Rc<DeviceShared>,
    heap: ID3D12Heap,
}

impl D3d12TransientHeap {
    pub(crate) fn new(device: Rc<DeviceShared>, size: u64) -> Result<Self, EngineError> {
        unsafe {
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: size.max(1),
                Properties: default_heap(),
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES,
            };
            let mut heap: Option<ID3D12Heap> = None;
            device
                .device
                .CreateHeap(&desc, &mut heap)
                .map_err(d3d_err)?;
            let heap = heap.ok_or_else(|| EngineError::Rhi("transient heap null".into()))?;
            Ok(Self { device, heap })
        }
    }
}
