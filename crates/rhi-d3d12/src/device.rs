//! Logical device, command queue, and resource creation.

use std::cell::Cell;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::{
    BufferDesc, CubemapDesc, Extent2D, Format, GraphicsPipelineDesc, MemoryRequirements,
    ReadbackLayout, RenderTargetDesc, SwapchainDesc, TextureDesc,
};
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUFFER_UAV, D3D12_BUFFER_UAV_FLAG_RAW, D3D12_COMMAND_LIST_TYPE_COMPUTE,
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE,
    D3D12_COMMAND_SIGNATURE_DESC, D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    D3D12_FEATURE_D3D12_OPTIONS5, D3D12_FEATURE_DATA_D3D12_OPTIONS5, D3D12_FENCE_FLAG_NONE,
    D3D12_GPU_DESCRIPTOR_HANDLE, D3D12_INDIRECT_ARGUMENT_DESC,
    D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    D3D12_RAYTRACING_TIER_1_1, D3D12_SHADER_RESOURCE_VIEW_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC_0,
    D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_SRV_DIMENSION_TEXTURE2D,
    D3D12_SRV_DIMENSION_TEXTURE3D, D3D12_SRV_DIMENSION_TEXTURECUBE, D3D12_TEX2D_SRV,
    D3D12_TEX2D_UAV, D3D12_TEX3D_SRV, D3D12_TEX3D_UAV, D3D12_TEXCUBE_SRV,
    D3D12_UAV_DIMENSION_BUFFER, D3D12_UAV_DIMENSION_TEXTURE2D, D3D12_UAV_DIMENSION_TEXTURE3D,
    D3D12_UNORDERED_ACCESS_VIEW_DESC, D3D12_UNORDERED_ACCESS_VIEW_DESC_0, D3D12CreateDevice,
    ID3D12CommandList, ID3D12CommandQueue, ID3D12CommandSignature, ID3D12DescriptorHeap,
    ID3D12Device, ID3D12Fence, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_UNKNOWN,
};
use windows::Win32::Graphics::Dxgi::IDXGIFactory6;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCWSTR};

use crate::command::D3d12CommandBuffer;
use crate::cubemap::D3d12Cubemap;
use crate::depth::D3d12DepthBuffer;
use crate::instance::{D3d12Instance, d3d_err};
use crate::pipeline::D3d12GraphicsPipeline;
use crate::render_target::{self, D3d12RenderTarget, D3d12TransientHeap};
use crate::swapchain::D3d12Swapchain;
use crate::sync::{D3d12Fence, D3d12Semaphore};
use crate::texture::D3d12Texture;
use crate::{D3d12Buffer, to_dxgi_format};

/// Size of the bindless 2D-texture SRV table (heap slots `0..BINDLESS_COUNT`).
pub(crate) const BINDLESS_COUNT: u32 = 1024;
/// Size of the bindless cubemap SRV table (heap slots
/// `BINDLESS_COUNT..BINDLESS_COUNT+CUBE_COUNT`, register space 1). Its index
/// space is separate from the 2D table and starts at 0.
pub(crate) const CUBE_COUNT: u32 = 64;
/// Size of the bindless storage-image UAV table (Phase 7), in the heap region
/// right after the cubes. Separate 0-based index space (`u0, space0`).
pub(crate) const STORAGE_IMAGE_COUNT: u32 = 64;
/// Size of the bindless storage-buffer UAV table (Phase 7), after the storage
/// images. Separate 0-based index space (`u0, space1`).
pub(crate) const STORAGE_BUFFER_COUNT: u32 = 64;
/// Heap offset where the storage-image UAV region begins.
pub(crate) const STORAGE_IMAGE_BASE: u32 = BINDLESS_COUNT + CUBE_COUNT;
/// Heap offset where the storage-buffer UAV region begins.
pub(crate) const STORAGE_BUFFER_BASE: u32 = STORAGE_IMAGE_BASE + STORAGE_IMAGE_COUNT;
/// Heap offset of the single scene-TLAS SRV (Phase 8). One slot after the
/// storage-buffer region; the shader sees it at `t{BINDLESS_COUNT+CUBE_COUNT},
/// space1` (a `RaytracingAccelerationStructure`).
pub(crate) const TLAS_SLOT: u32 = STORAGE_BUFFER_BASE + STORAGE_BUFFER_COUNT;
/// Size of the bindless sampled 3D-volume SRV table (Phase 11 Stage B), after the
/// TLAS slot. Shader sees it at `t{BINDLESS_COUNT+CUBE_COUNT+1}, space1`.
pub(crate) const VOLUME_COUNT: u32 = 64;
/// Size of the bindless storage 3D-volume UAV table (Phase 11 Stage B), after the
/// sampled volumes. Shader sees it at `u{STORAGE_IMAGE_COUNT+STORAGE_BUFFER_COUNT},
/// space1`.
pub(crate) const STORAGE_VOLUME_COUNT: u32 = 64;
/// Heap offset where the sampled-volume SRV region begins.
pub(crate) const VOLUME_BASE: u32 = TLAS_SLOT + 1;
/// Heap offset where the storage-volume UAV region begins.
pub(crate) const STORAGE_VOLUME_BASE: u32 = VOLUME_BASE + VOLUME_COUNT;
/// Total descriptors in the shader-visible bindless heap.
pub(crate) const HEAP_DESCRIPTORS: u32 = STORAGE_VOLUME_BASE + STORAGE_VOLUME_COUNT;

/// Device-level objects shared (via `Rc`) by every GPU resource.
pub(crate) struct DeviceShared {
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    pub factory: IDXGIFactory6,
    pub hwnd: HWND,
    // Shader-visible CBV_SRV_UAV heap holding bindless texture SRVs.
    pub srv_heap: ID3D12DescriptorHeap,
    pub srv_size: u32,
    srv_next: Cell<u32>,
    cube_next: Cell<u32>,
    storage_image_next: Cell<u32>,
    storage_buffer_next: Cell<u32>,
    volume_next: Cell<u32>,
    storage_volume_next: Cell<u32>,
    // Command signature for indexed indirect draws (`ExecuteIndirect`, Phase 7).
    pub indirect_draw_signature: ID3D12CommandSignature,
    // Async compute (Phase 7): a COMPUTE-type queue overlapping the DIRECT queue,
    // and a fence the compute queue signals / the graphics queue waits on (GPU
    // cross-queue sync). `async_value` holds the last value signaled.
    pub compute_queue: ID3D12CommandQueue,
    pub async_fence: ID3D12Fence,
    async_value: Cell<u64>,
    // A dedicated fence for `wait_idle`.
    idle_fence: ID3D12Fence,
    idle_event: HANDLE,
    idle_value: Cell<u64>,
    // Hardware ray tracing (Phase 8): true when DXR Tier >= 1.1 is supported.
    has_raytracing: bool,
}

impl DeviceShared {
    pub(crate) fn new(instance: &D3d12Instance) -> Result<Self, EngineError> {
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(
                &instance.shared.adapter,
                D3D_FEATURE_LEVEL_12_0,
                &mut device,
            )
            .map_err(d3d_err)?;
            let device =
                device.ok_or_else(|| EngineError::Rhi("CreateDevice returned null".into()))?;

            // Hardware ray tracing (Phase 8): DXR needs RaytracingTier >= 1.1
            // (inline ray query path) and ID3D12Device5 / GraphicsCommandList4,
            // which are queried/cast where used. Gate so non-RT devices still work.
            // `DREAMCOAST_NO_RAYTRACING` forces RT off (parity with Vulkan) so a
            // capture tool that lacks DXR support can grab the raster path.
            let force_no_rt = std::env::var_os("DREAMCOAST_NO_RAYTRACING").is_some();
            let mut options5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
            let has_raytracing = !force_no_rt
                && device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS5,
                        &mut options5 as *mut _ as *mut core::ffi::c_void,
                        std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() as u32,
                    )
                    .is_ok()
                && options5.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0;

            let queue_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                Priority: 0,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                NodeMask: 0,
            };
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&queue_desc).map_err(d3d_err)?;

            // Async-compute queue (COMPUTE type) + a cross-queue sync fence.
            let compute_queue_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                Priority: 0,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                NodeMask: 0,
            };
            let compute_queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&compute_queue_desc)
                .map_err(d3d_err)?;
            let async_fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(d3d_err)?;

            let idle_fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(d3d_err)?;
            let idle_event = CreateEventW(None, false, false, PCWSTR::null()).map_err(d3d_err)?;

            // Shader-visible bindless SRV heap.
            let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: HEAP_DESCRIPTORS,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            let srv_heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&heap_desc).map_err(d3d_err)?;
            let srv_size =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            // Command signature: one DRAW_INDEXED argument, 20-byte stride (matches
            // VkDrawIndexedIndirectCommand so one compute shader fills both APIs).
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
                ..Default::default()
            };
            let sig_desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: 20,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut indirect_draw_signature: Option<ID3D12CommandSignature> = None;
            device
                .CreateCommandSignature(&sig_desc, None, &mut indirect_draw_signature)
                .map_err(d3d_err)?;
            let indirect_draw_signature = indirect_draw_signature
                .ok_or_else(|| EngineError::Rhi("command signature null".into()))?;
            tracing::debug!("D3D12 device + queue ready");

            Ok(Self {
                device,
                queue,
                factory: instance.shared.factory.clone(),
                hwnd: instance.shared.hwnd,
                srv_heap,
                srv_size,
                srv_next: Cell::new(0),
                cube_next: Cell::new(0),
                storage_image_next: Cell::new(0),
                storage_buffer_next: Cell::new(0),
                volume_next: Cell::new(0),
                storage_volume_next: Cell::new(0),
                indirect_draw_signature,
                compute_queue,
                async_fence,
                async_value: Cell::new(0),
                idle_fence,
                idle_event,
                idle_value: Cell::new(0),
                has_raytracing,
            })
        }
    }

    /// GPU handle to the start of the bindless SRV heap (for the root table).
    pub(crate) fn srv_gpu_start(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        unsafe { self.srv_heap.GetGPUDescriptorHandleForHeapStart() }
    }

    /// Create an SRV for `resource` at the next bindless slot; returns its index.
    pub(crate) fn register_texture(&self, resource: &ID3D12Resource, format: Format) -> u32 {
        let index = self.srv_next.get();
        self.srv_next.set(index + 1);
        let cpu = unsafe { self.srv_heap.GetCPUDescriptorHandleForHeapStart() };
        let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + (index as usize) * (self.srv_size as usize),
        };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: to_dxgi_format(format),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    // -1 = expose all mip levels the resource has (the full chain for
                    // material textures; still 1 for single-mip resources).
                    MipLevels: u32::MAX,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        unsafe {
            self.device
                .CreateShaderResourceView(resource, Some(&srv), handle)
        };
        index
    }

    /// Create an `R32_FLOAT` SRV for a typeless depth `resource` at the next
    /// bindless slot (so a depth buffer can be sampled as a shadow map); returns
    /// its index.
    pub(crate) fn register_sampled_depth(&self, resource: &ID3D12Resource) -> u32 {
        let index = self.srv_next.get();
        self.srv_next.set(index + 1);
        let cpu = unsafe { self.srv_heap.GetCPUDescriptorHandleForHeapStart() };
        let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + (index as usize) * (self.srv_size as usize),
        };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        unsafe {
            self.device
                .CreateShaderResourceView(resource, Some(&srv), handle)
        };
        index
    }

    /// Create a `TEXTURECUBE` SRV for `resource` in the reserved cube heap region
    /// (slot `BINDLESS_COUNT + index`); returns the 0-based cube index. The cube
    /// root range is offset to `BINDLESS_COUNT`, so the shader indexes it 0-based.
    pub(crate) fn register_sampled_cube(
        &self,
        resource: &ID3D12Resource,
        format: Format,
        mip_levels: u32,
    ) -> u32 {
        let index = self.cube_next.get();
        self.cube_next.set(index + 1);
        let slot = BINDLESS_COUNT + index;
        let cpu = unsafe { self.srv_heap.GetCPUDescriptorHandleForHeapStart() };
        let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + (slot as usize) * (self.srv_size as usize),
        };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: to_dxgi_format(format),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                TextureCube: D3D12_TEXCUBE_SRV {
                    MostDetailedMip: 0,
                    MipLevels: mip_levels,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        unsafe {
            self.device
                .CreateShaderResourceView(resource, Some(&srv), handle)
        };
        index
    }

    /// Create the scene-TLAS SRV (`RaytracingAccelerationStructure`) at the
    /// reserved [`TLAS_SLOT`], referencing the TLAS by GPU virtual address
    /// (an AS SRV binds no resource — `None` — and carries the VA in the desc).
    /// Phase 8 M3.
    pub(crate) fn register_tlas(&self, gpu_va: u64) {
        let handle = self.cpu_handle(TLAS_SLOT);
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                RaytracingAccelerationStructure:
                    windows::Win32::Graphics::Direct3D12::D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV {
                        Location: gpu_va,
                    },
            },
        };
        unsafe {
            self.device
                .CreateShaderResourceView(None, Some(&srv), handle)
        };
    }

    /// CPU handle for bindless heap `slot`.
    fn cpu_handle(&self, slot: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let cpu = unsafe { self.srv_heap.GetCPUDescriptorHandleForHeapStart() };
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + (slot as usize) * (self.srv_size as usize),
        }
    }

    /// Create a Texture2D UAV for `resource` in the reserved storage-image heap
    /// region; returns the 0-based storage-image index (Phase 7).
    pub(crate) fn register_storage_image(&self, resource: &ID3D12Resource, format: Format) -> u32 {
        let index = self.storage_image_next.get();
        self.storage_image_next.set(index + 1);
        let handle = self.cpu_handle(STORAGE_IMAGE_BASE + index);
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: to_dxgi_format(format),
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        };
        unsafe {
            self.device
                .CreateUnorderedAccessView(resource, None, Some(&uav), handle);
        }
        index
    }

    /// Create a `Texture3D` SRV for a volume in the reserved sampled-volume heap
    /// region; returns the 0-based volume index (Phase 11 Stage B).
    pub(crate) fn register_volume(&self, resource: &ID3D12Resource, format: Format) -> u32 {
        let index = self.volume_next.get();
        self.volume_next.set(index + 1);
        let handle = self.cpu_handle(VOLUME_BASE + index);
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: to_dxgi_format(format),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        unsafe {
            self.device
                .CreateShaderResourceView(resource, Some(&srv), handle)
        };
        index
    }

    /// Create a `Texture3D` UAV for a volume in the reserved storage-volume heap
    /// region; returns the 0-based storage-volume index. `depth` = the volume's W
    /// extent so the UAV covers every slice (Phase 11 Stage B).
    pub(crate) fn register_storage_volume(
        &self,
        resource: &ID3D12Resource,
        format: Format,
        depth: u32,
    ) -> u32 {
        let index = self.storage_volume_next.get();
        self.storage_volume_next.set(index + 1);
        let handle = self.cpu_handle(STORAGE_VOLUME_BASE + index);
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: to_dxgi_format(format),
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_UAV {
                    MipSlice: 0,
                    FirstWSlice: 0,
                    WSize: depth,
                },
            },
        };
        unsafe {
            self.device
                .CreateUnorderedAccessView(resource, None, Some(&uav), handle);
        }
        index
    }

    /// Create a raw (byte-address) UAV for `resource` in the reserved storage-buffer
    /// heap region; returns the 0-based storage-buffer index. Raw views let one
    /// bindless array hold heterogeneous data (particles, instances, indirect args,
    /// counters) addressed by byte offset (Phase 7). `size_bytes` must be a
    /// multiple of 4.
    pub(crate) fn register_storage_buffer(
        &self,
        resource: &ID3D12Resource,
        size_bytes: u64,
    ) -> u32 {
        let index = self.storage_buffer_next.get();
        self.storage_buffer_next.set(index + 1);
        let handle = self.cpu_handle(STORAGE_BUFFER_BASE + index);
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: (size_bytes / 4) as u32,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        unsafe {
            self.device
                .CreateUnorderedAccessView(resource, None, Some(&uav), handle);
        }
        index
    }

    /// Record + submit a one-time command list and wait for completion.
    pub(crate) fn immediate_submit(
        &self,
        record: impl FnOnce(&windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList),
    ) -> Result<(), EngineError> {
        use windows::Win32::Graphics::Direct3D12::{
            ID3D12CommandAllocator, ID3D12GraphicsCommandList,
        };
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(d3d_err)?;
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .map_err(d3d_err)?;
            record(&list);
            list.Close().map_err(d3d_err)?;
            let cl: ID3D12CommandList = list.cast().map_err(d3d_err)?;
            self.queue.ExecuteCommandLists(&[Some(cl)]);
            self.wait_idle()?;
            Ok(())
        }
    }

    /// Block until the GPU has finished all previously submitted work.
    pub(crate) fn wait_idle(&self) -> Result<(), EngineError> {
        unsafe {
            let value = self.idle_value.get() + 1;
            self.idle_value.set(value);
            self.queue
                .Signal(&self.idle_fence, value)
                .map_err(d3d_err)?;
            if self.idle_fence.GetCompletedValue() < value {
                self.idle_fence
                    .SetEventOnCompletion(value, self.idle_event)
                    .map_err(d3d_err)?;
                WaitForSingleObject(self.idle_event, INFINITE);
            }
            Ok(())
        }
    }
}

impl Drop for DeviceShared {
    fn drop(&mut self) {
        // Ensure the GPU is idle before COM objects release.
        let _ = self.wait_idle();
        unsafe {
            let _ = CloseHandle(self.idle_event);
        }
    }
}

/// A logical D3D12 device: the factory for swapchains, pipelines, command
/// buffers, and synchronization primitives.
pub struct D3d12Device {
    pub(crate) shared: Rc<DeviceShared>,
}

impl D3d12Device {
    pub fn create_swapchain(&self, desc: &SwapchainDesc) -> Result<D3d12Swapchain, EngineError> {
        D3d12Swapchain::new(self.shared.clone(), desc)
    }

    pub fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc,
    ) -> Result<D3d12GraphicsPipeline, EngineError> {
        D3d12GraphicsPipeline::new(self.shared.clone(), desc)
    }

    pub fn create_compute_pipeline(
        &self,
        desc: &rhi_types::ComputePipelineDesc,
    ) -> Result<crate::pipeline::D3d12ComputePipeline, EngineError> {
        crate::pipeline::D3d12ComputePipeline::new(self.shared.clone(), desc)
    }

    pub fn create_command_buffer(&self) -> Result<D3d12CommandBuffer, EngineError> {
        D3d12CommandBuffer::new(self.shared.clone())
    }

    /// Create a timestamp query heap of `count` queries (Phase 9 profiling).
    pub fn create_query_heap(
        &self,
        count: u32,
    ) -> Result<crate::query::D3d12QueryHeap, EngineError> {
        crate::query::D3d12QueryHeap::new(self.shared.clone(), count)
    }

    /// Create a hardware ray-tracing pipeline (state object) + SBT (Phase 8 M5).
    pub fn create_raytracing_pipeline(
        &self,
        desc: &rhi_types::RaytracingPipelineDesc,
    ) -> Result<crate::rt_pipeline::D3d12RaytracingPipeline, EngineError> {
        crate::rt_pipeline::D3d12RaytracingPipeline::new(self.shared.clone(), desc)
    }

    /// Build the scene's acceleration structures (BLAS per mesh + one TLAS) in a
    /// one-shot DIRECT-queue submission (static scene, Phase 8 M2).
    pub fn build_raytracing_scene(
        &self,
        geometries: &[(&D3d12Buffer, &D3d12Buffer, rhi_types::BlasGeometry)],
        instances: &[rhi_types::TlasInstance],
    ) -> Result<crate::accel::D3d12RaytracingScene, EngineError> {
        crate::accel::D3d12RaytracingScene::build(self.shared.clone(), geometries, instances)
    }

    /// Register the scene TLAS in the bindless heap so shaders can trace it
    /// (Phase 8 M3). Call once after building a static scene.
    pub fn bind_tlas(&self, scene: &crate::accel::D3d12RaytracingScene) {
        self.shared.register_tlas(scene.tlas_gpu_va());
    }

    /// Allocate a COMPUTE-type command buffer for the async-compute queue (Phase 7).
    pub fn create_compute_command_buffer(&self) -> Result<D3d12CommandBuffer, EngineError> {
        D3d12CommandBuffer::new_compute(self.shared.clone())
    }

    /// The async-compute queue (Phase 7).
    pub fn compute_queue(&self) -> D3d12ComputeQueue {
        D3d12ComputeQueue {
            shared: self.shared.clone(),
        }
    }

    /// D3D12 always exposes a separate COMPUTE queue, so async compute is available.
    pub fn has_async_compute(&self) -> bool {
        true
    }

    /// Whether hardware ray tracing (DXR Tier >= 1.1) is available (Phase 8).
    pub fn has_raytracing(&self) -> bool {
        self.shared.has_raytracing
    }

    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<D3d12Buffer, EngineError> {
        D3d12Buffer::new(self.shared.clone(), desc)
    }

    /// Create a device-local storage buffer (UAV) for compute (Phase 7).
    pub fn create_storage_buffer(
        &self,
        desc: &rhi_types::StorageBufferDesc,
    ) -> Result<crate::buffer::D3d12StorageBuffer, EngineError> {
        crate::buffer::D3d12StorageBuffer::new(self.shared.clone(), desc)
    }

    /// Create a storage buffer seeded with host data (Phase 8: RT geometry +
    /// instance table read by the path tracer).
    pub fn create_storage_buffer_init(
        &self,
        desc: &rhi_types::StorageBufferDesc,
        data: &[u8],
    ) -> Result<crate::buffer::D3d12StorageBuffer, EngineError> {
        crate::buffer::D3d12StorageBuffer::new_init(self.shared.clone(), desc, data)
    }

    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        pixels: &[u8],
    ) -> Result<D3d12Texture, EngineError> {
        D3d12Texture::new(self.shared.clone(), desc, pixels)
    }

    pub fn create_depth_buffer(&self, extent: Extent2D) -> Result<D3d12DepthBuffer, EngineError> {
        D3d12DepthBuffer::new(self.shared.clone(), extent)
    }

    pub fn create_cubemap(&self, desc: &CubemapDesc) -> Result<D3d12Cubemap, EngineError> {
        D3d12Cubemap::new(self.shared.clone(), desc)
    }

    /// CPU memory layout for reading a swapchain image back to the host. D3D12
    /// pads each row to 256 bytes (`GetCopyableFootprints`).
    pub fn swapchain_readback_layout(&self, swapchain: &D3d12Swapchain) -> ReadbackLayout {
        unsafe {
            let desc = swapchain.buffer(0).GetDesc();
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut num_rows = 0u32;
            let mut row_size = 0u64;
            let mut total = 0u64;
            self.shared.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut footprint),
                Some(&mut num_rows),
                Some(&mut row_size),
                Some(&mut total),
            );
            ReadbackLayout {
                width: desc.Width as u32,
                height: desc.Height,
                row_pitch: footprint.Footprint.RowPitch,
                size: total,
            }
        }
    }

    pub fn create_render_target(
        &self,
        desc: &RenderTargetDesc,
    ) -> Result<D3d12RenderTarget, EngineError> {
        D3d12RenderTarget::new(self.shared.clone(), desc)
    }

    /// Create a 3D (volume) texture, registered in the bindless sampled + storage
    /// volume tables (Phase 11 Stage B).
    pub fn create_volume(
        &self,
        desc: &rhi_types::VolumeDesc,
    ) -> Result<crate::volume::D3d12Volume, EngineError> {
        crate::volume::D3d12Volume::new(self.shared.clone(), desc)
    }

    pub fn render_target_memory(
        &self,
        desc: &RenderTargetDesc,
    ) -> Result<MemoryRequirements, EngineError> {
        render_target::render_target_memory(&self.shared, desc)
    }

    pub fn create_transient_heap(&self, size: u64) -> Result<D3d12TransientHeap, EngineError> {
        D3d12TransientHeap::new(self.shared.clone(), size)
    }

    pub fn create_aliased_target(
        &self,
        heap: &D3d12TransientHeap,
        offset: u64,
        desc: &RenderTargetDesc,
    ) -> Result<D3d12RenderTarget, EngineError> {
        D3d12RenderTarget::new_aliased(self.shared.clone(), heap, offset, desc)
    }

    pub fn create_fence(&self, signaled: bool) -> Result<D3d12Fence, EngineError> {
        D3d12Fence::new(self.shared.clone(), signaled)
    }

    pub fn create_semaphore(&self) -> Result<D3d12Semaphore, EngineError> {
        Ok(D3d12Semaphore::new())
    }

    pub fn queue(&self) -> D3d12Queue {
        D3d12Queue {
            shared: self.shared.clone(),
        }
    }

    pub fn wait_idle(&self) -> Result<(), EngineError> {
        self.shared.wait_idle()
    }
}

/// The device's async-compute (COMPUTE-type) queue (Phase 7).
pub struct D3d12ComputeQueue {
    pub(crate) shared: Rc<DeviceShared>,
}

impl D3d12ComputeQueue {
    /// Execute compute work on the compute queue and signal the cross-queue fence;
    /// the graphics queue's `submit_async` GPU-waits on this. `signal` (a no-op
    /// D3D12 semaphore) is for facade parity with Vulkan.
    pub fn submit(
        &self,
        cmd: &D3d12CommandBuffer,
        _signal: &D3d12Semaphore,
    ) -> Result<(), EngineError> {
        unsafe {
            let list: ID3D12CommandList = cmd.list().cast().map_err(d3d_err)?;
            self.shared.compute_queue.ExecuteCommandLists(&[Some(list)]);
            let value = self.shared.async_value.get() + 1;
            self.shared.async_value.set(value);
            self.shared
                .compute_queue
                .Signal(&self.shared.async_fence, value)
                .map_err(d3d_err)?;
            Ok(())
        }
    }
}

/// The device's DIRECT queue.
pub struct D3d12Queue {
    pub(crate) shared: Rc<DeviceShared>,
}

impl D3d12Queue {
    /// Execute a command list, then signal `fence` (semaphores are ignored on
    /// D3D12 — see crate docs).
    pub fn submit(
        &self,
        cmd: &D3d12CommandBuffer,
        _wait: &D3d12Semaphore,
        _signal: &D3d12Semaphore,
        fence: &D3d12Fence,
    ) -> Result<(), EngineError> {
        unsafe {
            let list: ID3D12CommandList = cmd.list().cast().map_err(d3d_err)?;
            self.shared.queue.ExecuteCommandLists(&[Some(list)]);
            let value = fence.next_value();
            self.shared
                .queue
                .Signal(fence.raw(), value)
                .map_err(d3d_err)?;
            fence.set_target(value);
            Ok(())
        }
    }

    /// Execute on the graphics queue, first GPU-waiting on the async-compute
    /// queue's last signal (so the particle draw sees the compute-written buffer),
    /// then signaling `fence` (Phase 7). Semaphores are D3D12 no-ops.
    pub fn submit_async(
        &self,
        cmd: &D3d12CommandBuffer,
        _wait: &D3d12Semaphore,
        _signal: &D3d12Semaphore,
        fence: &D3d12Fence,
    ) -> Result<(), EngineError> {
        unsafe {
            // GPU-side wait: the graphics queue blocks until the compute queue has
            // signaled its latest value.
            self.shared
                .queue
                .Wait(&self.shared.async_fence, self.shared.async_value.get())
                .map_err(d3d_err)?;
            let list: ID3D12CommandList = cmd.list().cast().map_err(d3d_err)?;
            self.shared.queue.ExecuteCommandLists(&[Some(list)]);
            let value = fence.next_value();
            self.shared
                .queue
                .Signal(fence.raw(), value)
                .map_err(d3d_err)?;
            fence.set_target(value);
            Ok(())
        }
    }

    /// Execute a command list with no semaphore sync, signaling `fence`. For
    /// one-off startup work (e.g. IBL cubemap generation).
    pub fn submit_oneshot(
        &self,
        cmd: &D3d12CommandBuffer,
        fence: &D3d12Fence,
    ) -> Result<(), EngineError> {
        unsafe {
            let list: ID3D12CommandList = cmd.list().cast().map_err(d3d_err)?;
            self.shared.queue.ExecuteCommandLists(&[Some(list)]);
            let value = fence.next_value();
            self.shared
                .queue
                .Signal(fence.raw(), value)
                .map_err(d3d_err)?;
            fence.set_target(value);
            Ok(())
        }
    }

    /// Present the swapchain (vsync). Returns `true` if it should be recreated.
    pub fn present(
        &self,
        swapchain: &D3d12Swapchain,
        _image_index: u32,
        _wait: &D3d12Semaphore,
    ) -> Result<bool, EngineError> {
        swapchain.present()
    }
}
