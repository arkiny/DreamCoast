//! Logical device, command queue, and resource creation.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::{
    BufferDesc, CubemapDesc, Extent2D, Format, GraphicsPipelineDesc, MemoryRequirements,
    ReadbackLayout, RenderTargetDesc, SwapchainDesc, TextureDesc,
};
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D_SHADER_MODEL_6_6, D3D12_BUFFER_UAV, D3D12_BUFFER_UAV_FLAG_RAW,
    D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
    D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMMAND_SIGNATURE_DESC, D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    D3D12_FEATURE_D3D12_OPTIONS5, D3D12_FEATURE_D3D12_OPTIONS7, D3D12_FEATURE_D3D12_OPTIONS11,
    D3D12_FEATURE_DATA_D3D12_OPTIONS5, D3D12_FEATURE_DATA_D3D12_OPTIONS7,
    D3D12_FEATURE_DATA_D3D12_OPTIONS11, D3D12_FEATURE_DATA_SHADER_MODEL,
    D3D12_FEATURE_SHADER_MODEL, D3D12_FENCE_FLAG_NONE, D3D12_GPU_DESCRIPTOR_HANDLE,
    D3D12_INDIRECT_ARGUMENT_DESC, D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
    D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
    D3D12_MESH_SHADER_TIER_1, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_RAYTRACING_TIER_1_1,
    D3D12_SHADER_RESOURCE_VIEW_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC_0,
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
/// right after the cubes. Separate 0-based index space (`u0, space0`). Raised 64→256 for Phase 14:
/// the per-FIF vgeo HZB pyramid + grid HZB pyramid + transient compute targets allocate one UAV
/// slot PER MIP, so a GI-heavy scene at high resolution neared the old 64-slot ceiling (the crash
/// on window resize before slot reclamation landed). Generous like `STORAGE_BUFFER_COUNT`, matching
/// UE's large-bindless-heap approach; downstream bases + the root-signature range derive from it, so
/// the layout shifts consistently — must match `Bindless.storage_images[N]` in bindless.slang and
/// rhi-vulkan / rhi-metal.
pub(crate) const STORAGE_IMAGE_COUNT: u32 = 256;
/// Size of the bindless storage-buffer UAV table (Phase 7), after the storage
/// images. Separate 0-based index space (`u0, space1`). Raised 64→128 for Phase 14 virtual
/// geometry: the GI-heavy default scene already fills all 64 slots, leaving no room for vgeo's
/// cluster-page / visibility / cut buffers. Downstream heap offsets (`TLAS_SLOT`, `VOLUME_BASE`,
/// …) and the root-signature ranges derive from this, so the whole layout shifts consistently;
/// must match `Bindless.storage_buffers[N]` in bindless.slang and rhi-vulkan / rhi-metal.
pub(crate) const STORAGE_BUFFER_COUNT: u32 = 2048;
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
    // Free-list of bindless SRV (sampled 2D-texture) slots returned by dropped render targets /
    // depth buffers, mirroring the storage-image/-buffer reclaim. Static textures never free (they
    // live the app's lifetime), so this only recycles the transient targets recreated on window
    // resize — otherwise those slowly leaked the 1024-slot SRV table.
    srv_free: RefCell<Vec<u32>>,
    cube_next: Cell<u32>,
    // High-water mark for the bindless storage-IMAGE (UAV) table + a free-list of slots returned by
    // dropped render targets, mirroring the storage-buffer reclaim below. Without it, recreating
    // storage render targets (e.g. the per-FIF vgeo HZB pyramid on window resize) walked the table
    // monotonically past its 64-slot end into the adjacent storage-buffer region → descriptor
    // corruption → device removal on resize. Reclaim is safe: a target only Drops after the frames
    // referencing it retire (resize does `wait_idle` first).
    storage_image_next: Cell<u32>,
    storage_image_free: RefCell<Vec<u32>>,
    // High-water mark for the bindless storage-buffer table + a free-list of slots returned by
    // dropped buffers. `register_storage_buffer` reuses a freed slot before bumping the mark, so a
    // long run creating + destroying storage buffers (resize, subsystem teardown) no longer walks
    // the table monotonically into the reserved TLAS/volume regions. Reclaim is safe: a buffer only
    // Drops after the frames referencing it retire (the handoff contract), so the reused slot is
    // idle. Single-threaded (`Rc`), hence `RefCell` not a lock.
    storage_buffer_next: Cell<u32>,
    storage_buffer_free: RefCell<Vec<u32>>,
    volume_next: Cell<u32>,
    storage_volume_next: Cell<u32>,
    // Command signature for indexed indirect draws (`ExecuteIndirect`, Phase 7).
    pub indirect_draw_signature: ID3D12CommandSignature,
    // Command signature for indirect compute dispatch (`ExecuteIndirect` over a
    // `D3D12_DISPATCH_ARGUMENTS`, Phase 14). 12-byte stride matches `VkDispatchIndirectCommand`
    // so one compute shader fills args for both APIs.
    pub indirect_dispatch_signature: ID3D12CommandSignature,
    // Command signature for indirect mesh dispatch (`ExecuteIndirect` over a
    // `D3D12_DISPATCH_MESH_ARGUMENTS`, Phase 14 Track B). 12-byte stride matches
    // `VkDrawMeshTasksIndirectCommandEXT` so one LOD-cut compute fills args for both APIs.
    pub indirect_dispatch_mesh_signature: ID3D12CommandSignature,
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
    // 64-bit buffer atomics (Phase 14 virtual geometry): true when the device supports
    // Shader Model 6.6 AND `AtomicInt64OnDescriptorHeapResourceSupported` (OPTIONS11). The
    // SW-raster visibility buffer is a bindless (descriptor-heap-indexed) `RWByteAddressBuffer`
    // doing `InterlockedMax64`, so descriptor-heap atomics are exactly the required cap.
    has_atomic_int64: bool,
    // Mesh + amplification shaders (Phase 14 virtual geometry Track B): true when
    // OPTIONS7 `MeshShaderTier` >= TIER_1. Gates `create_mesh_pipeline` and the HW-path smokes.
    has_mesh_shader: bool,
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

            // 64-bit buffer atomics (Phase 14 virtual geometry). The visibility buffer is a
            // bindless `RWByteAddressBuffer` doing `InterlockedMax64`, which needs Shader Model 6.6
            // and — because it is reached through the shader-visible descriptor heap —
            // `AtomicInt64OnDescriptorHeapResourceSupported` (OPTIONS11). Probe both; report the
            // capability so the vgeo path gates cleanly on adapters that lack it.
            let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            let sm_ok = device
                .CheckFeatureSupport(
                    D3D12_FEATURE_SHADER_MODEL,
                    &mut sm as *mut _ as *mut core::ffi::c_void,
                    std::mem::size_of::<D3D12_FEATURE_DATA_SHADER_MODEL>() as u32,
                )
                .is_ok()
                && sm.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            let mut options11 = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
            let atomic64_heap = device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS11,
                    &mut options11 as *mut _ as *mut core::ffi::c_void,
                    std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS11>() as u32,
                )
                .is_ok()
                && options11
                    .AtomicInt64OnDescriptorHeapResourceSupported
                    .as_bool();
            let has_atomic_int64 = sm_ok && atomic64_heap;

            // Mesh + amplification shaders (Phase 14 Track B): OPTIONS7 `MeshShaderTier` >= TIER_1.
            let mut options7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let has_mesh_shader = device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS7,
                    &mut options7 as *mut _ as *mut core::ffi::c_void,
                    std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS7>() as u32,
                )
                .is_ok()
                && options7.MeshShaderTier.0 >= D3D12_MESH_SHADER_TIER_1.0;

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

            // Indirect-dispatch signature: one DISPATCH argument, 12-byte stride (matches
            // VkDispatchIndirectCommand's three u32 groupCount so one compute shader fills both
            // APIs). No root arguments referenced, so no root signature (like the draw one).
            let dispatch_arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
                ..Default::default()
            };
            let dispatch_sig_desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: 12,
                NumArgumentDescs: 1,
                pArgumentDescs: &dispatch_arg,
                NodeMask: 0,
            };
            let mut indirect_dispatch_signature: Option<ID3D12CommandSignature> = None;
            device
                .CreateCommandSignature(&dispatch_sig_desc, None, &mut indirect_dispatch_signature)
                .map_err(d3d_err)?;
            let indirect_dispatch_signature = indirect_dispatch_signature
                .ok_or_else(|| EngineError::Rhi("dispatch command signature null".into()))?;

            // Indirect mesh-dispatch signature: one DISPATCH_MESH argument, 12-byte stride
            // (matches `D3D12_DISPATCH_MESH_ARGUMENTS` = `VkDrawMeshTasksIndirectCommandEXT`'s
            // three u32 groupCount, so one LOD-cut compute fills both APIs). No root arguments
            // referenced → no root signature (like the draw/dispatch ones). Only built when the
            // adapter has mesh shaders, since `DISPATCH_MESH` is otherwise an invalid argument.
            let indirect_dispatch_mesh_signature = if has_mesh_shader {
                let mesh_arg = D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH,
                    ..Default::default()
                };
                let mesh_sig_desc = D3D12_COMMAND_SIGNATURE_DESC {
                    ByteStride: 12,
                    NumArgumentDescs: 1,
                    pArgumentDescs: &mesh_arg,
                    NodeMask: 0,
                };
                let mut sig: Option<ID3D12CommandSignature> = None;
                device
                    .CreateCommandSignature(&mesh_sig_desc, None, &mut sig)
                    .map_err(d3d_err)?;
                sig.ok_or_else(|| EngineError::Rhi("dispatch-mesh command signature null".into()))?
            } else {
                // Reuse the dispatch signature as an inert placeholder; never referenced without
                // mesh shaders (the smokes gate on `capabilities().mesh_shader`).
                indirect_dispatch_signature.clone()
            };
            tracing::debug!("D3D12 device + queue ready");

            Ok(Self {
                device,
                queue,
                factory: instance.shared.factory.clone(),
                hwnd: instance.shared.hwnd,
                srv_heap,
                srv_size,
                srv_next: Cell::new(0),
                srv_free: RefCell::new(Vec::new()),
                cube_next: Cell::new(0),
                storage_image_next: Cell::new(0),
                storage_image_free: RefCell::new(Vec::new()),
                storage_buffer_next: Cell::new(0),
                storage_buffer_free: RefCell::new(Vec::new()),
                volume_next: Cell::new(0),
                storage_volume_next: Cell::new(0),
                indirect_draw_signature,
                indirect_dispatch_signature,
                indirect_dispatch_mesh_signature,
                compute_queue,
                async_fence,
                async_value: Cell::new(0),
                idle_fence,
                idle_event,
                idle_value: Cell::new(0),
                has_raytracing,
                has_atomic_int64,
                has_mesh_shader,
            })
        }
    }

    /// GPU handle to the start of the bindless SRV heap (for the root table).
    pub(crate) fn srv_gpu_start(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        unsafe { self.srv_heap.GetGPUDescriptorHandleForHeapStart() }
    }

    /// Allocate a bindless SRV slot: reuse a freed one (dropped render target / depth buffer) before
    /// bumping the high-water mark. Shared by `register_texture` / `register_sampled_depth`.
    fn next_srv_slot(&self) -> u32 {
        let index = self.srv_free.borrow_mut().pop().unwrap_or_else(|| {
            let i = self.srv_next.get();
            self.srv_next.set(i + 1);
            i
        });
        assert!(
            index < BINDLESS_COUNT,
            "bindless SRV table overflow (> {BINDLESS_COUNT})"
        );
        index
    }

    /// Return an SRV slot to the free-list (called from `D3d12RenderTarget` / `D3d12DepthBuffer`
    /// drop). The stale descriptor is left in place — no shader indexes a freed slot until reuse
    /// overwrites it. Safe: resize `wait_idle`s before dropping.
    pub(crate) fn free_texture(&self, index: u32) {
        self.srv_free.borrow_mut().push(index);
    }

    /// Create an SRV for `resource` at the next bindless slot; returns its index.
    pub(crate) fn register_texture(&self, resource: &ID3D12Resource, format: Format) -> u32 {
        let index = self.next_srv_slot();
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
        let index = self.next_srv_slot();
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
        // Reuse a freed slot before bumping the high-water mark (see the field docs).
        let index = self
            .storage_image_free
            .borrow_mut()
            .pop()
            .unwrap_or_else(|| {
                let i = self.storage_image_next.get();
                self.storage_image_next.set(i + 1);
                i
            });
        assert!(
            index < STORAGE_IMAGE_COUNT,
            "bindless storage-image table overflow (> {STORAGE_IMAGE_COUNT}); raise \
             STORAGE_IMAGE_COUNT across bindless.slang + all three backends"
        );
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
        // Reuse a freed slot before bumping the high-water mark (see the field docs).
        let index = self
            .storage_buffer_free
            .borrow_mut()
            .pop()
            .unwrap_or_else(|| {
                let i = self.storage_buffer_next.get();
                self.storage_buffer_next.set(i + 1);
                i
            });
        assert!(
            index < STORAGE_BUFFER_COUNT,
            "bindless storage-buffer table overflow (> {STORAGE_BUFFER_COUNT}); raise \
             STORAGE_BUFFER_COUNT across bindless.slang + all three backends"
        );
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

    /// Return a storage-buffer slot to the free-list (called from `D3d12StorageBuffer::drop`).
    /// The stale UAV at `index` is left in place — no shader indexes a freed slot, and the next
    /// reuse overwrites it.
    pub(crate) fn free_storage_buffer(&self, index: u32) {
        self.storage_buffer_free.borrow_mut().push(index);
    }

    /// Return a storage-image slot to the free-list (called from `D3d12RenderTarget::drop`).
    pub(crate) fn free_storage_image(&self, index: u32) {
        self.storage_image_free.borrow_mut().push(index);
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

    /// Drain the D3D12 debug-layer info queue into the tracing log and return the number of
    /// ERROR/CORRUPTION messages seen (Phase 15 M4 B3 verification: the D3D12 debug layer writes
    /// to OutputDebugString, invisible to our logs — this bridges it to tracing like the Vulkan
    /// validation messenger, so threading violations surface where we can see them). No-op (0)
    /// when the debug layer isn't active (the device doesn't expose an `ID3D12InfoQueue`).
    pub fn drain_debug_messages(&self) -> u64 {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_MESSAGE, D3D12_MESSAGE_SEVERITY_CORRUPTION, D3D12_MESSAGE_SEVERITY_ERROR,
            D3D12_MESSAGE_SEVERITY_WARNING, ID3D12InfoQueue,
        };
        let Ok(iq) = self.shared.device.cast::<ID3D12InfoQueue>() else {
            return 0;
        };
        let mut errors = 0u64;
        unsafe {
            let n = iq.GetNumStoredMessages();
            for i in 0..n {
                let mut len: usize = 0;
                if iq.GetMessage(i, None, &mut len).is_err() || len == 0 {
                    continue;
                }
                let mut buf = vec![0u8; len];
                let msg = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
                if iq.GetMessage(i, Some(msg), &mut len).is_err() {
                    continue;
                }
                let m = &*msg;
                let desc = if m.pDescription.is_null() {
                    String::new()
                } else {
                    std::ffi::CStr::from_ptr(m.pDescription as *const i8)
                        .to_string_lossy()
                        .into_owned()
                };
                match m.Severity {
                    D3D12_MESSAGE_SEVERITY_CORRUPTION | D3D12_MESSAGE_SEVERITY_ERROR => {
                        tracing::error!("D3D12 debug layer: {desc}");
                        errors += 1;
                    }
                    // WARNING/INFO are demoted to debug: real workloads emit benign perf
                    // warnings (e.g. a clear with no committed clear value) every frame, which
                    // would flood the log. Only ERROR/CORRUPTION (the threading/correctness
                    // signal) is surfaced by default; raise the log level to see the rest.
                    D3D12_MESSAGE_SEVERITY_WARNING => {
                        tracing::debug!("D3D12 debug layer (warn): {desc}");
                    }
                    _ => tracing::trace!("D3D12 debug layer: {desc}"),
                }
            }
            iq.ClearStoredMessages();
        }
        errors
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

    /// Backend-agnostic device identity, surfaced up through the facade for platform-default
    /// quality-tier selection (macOS perf, axis A). Additive + read-only. A D3D12 adapter is never
    /// an Apple GPU in this engine (Metal owns macOS), so the tier only needs a non-Apple marker
    /// here. Reporting a stable `"D3D12"` name keeps this change from touching device selection.
    pub fn device_info(&self) -> rhi_types::DeviceInfo {
        rhi_types::DeviceInfo {
            name: "D3D12".to_string(),
            unified_memory: false,
            low_power: false,
        }
    }

    /// Compile a mesh-shader pipeline (Phase 14 Track B): an SM6.5 `MS`+`PS` (optionally `AS`)
    /// PSO built through a `D3D12_PIPELINE_STATE_STREAM`. Requires
    /// [`DeviceCapabilities::mesh_shader`] (callers gate on it).
    pub fn create_mesh_pipeline(
        &self,
        desc: &rhi_types::MeshPipelineDesc,
    ) -> Result<crate::D3d12MeshPipeline, EngineError> {
        crate::D3d12MeshPipeline::new(self.shared.clone(), desc)
    }

    /// Optional GPU capabilities (Phase 14 virtual geometry). `atomic_int64` reflects the probed
    /// SM6.6 + `AtomicInt64OnDescriptorHeapResourceSupported` (OPTIONS11) support (the SW-raster
    /// visibility-buffer path); `mesh_shader` reflects OPTIONS7 `MeshShaderTier >= TIER_1` (Track B
    /// HW path). `dispatch_indirect` is always available (the DISPATCH command signature is created
    /// at device init).
    pub fn capabilities(&self) -> rhi_types::DeviceCapabilities {
        rhi_types::DeviceCapabilities {
            mesh_shader: self.shared.has_mesh_shader,
            atomic_int64: self.shared.has_atomic_int64,
            dispatch_indirect: true,
        }
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

    /// Host-visible storage buffer for per-frame host writes (GPU-skinning palette, B.2c).
    pub fn create_storage_buffer_host(
        &self,
        desc: &rhi_types::StorageBufferDesc,
    ) -> Result<crate::buffer::D3d12StorageBuffer, EngineError> {
        crate::buffer::D3d12StorageBuffer::new_host(self.shared.clone(), desc)
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

    /// Create a sampled texture from pre-compressed BCn mip levels (Phase 12 M3).
    pub fn create_texture_compressed(
        &self,
        desc: &TextureDesc,
        levels: &[Vec<u8>],
    ) -> Result<D3d12Texture, EngineError> {
        D3d12Texture::new_compressed(self.shared.clone(), desc, levels)
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

    /// Create a 3D volume seeded with host `data` (Phase 12 M2: CPU-baked SDF
    /// uploaded instead of a GPU bake), left ready to sample.
    pub fn create_volume_init(
        &self,
        desc: &rhi_types::VolumeDesc,
        data: &[u8],
    ) -> Result<crate::volume::D3d12Volume, EngineError> {
        crate::volume::D3d12Volume::new_init(self.shared.clone(), desc, data)
    }

    /// Read a 3D volume back to host memory (Phase 12 item 3): `w*h*d*bpp` tightly
    /// packed bytes (`x + dim*(y + dim*z)` order).
    pub fn read_volume(
        &self,
        volume: &crate::volume::D3d12Volume,
        w: u32,
        h: u32,
        d: u32,
        bytes_per_voxel: u32,
    ) -> Result<Vec<u8>, EngineError> {
        volume.read_back(&self.shared, w, h, d, bytes_per_voxel)
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

    /// Submit async-compute work, signaling the cross-queue `async_fence` (the graphics queue waits
    /// it) AND `fence` (so the CPU knows the compute command list is free to re-record). The
    /// `_signal` semaphore is a D3D12 no-op (facade parity with Vulkan).
    pub fn submit_fenced(
        &self,
        cmd: &D3d12CommandBuffer,
        _signal: &D3d12Semaphore,
        fence: &D3d12Fence,
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
            let fv = fence.next_value();
            self.shared
                .compute_queue
                .Signal(fence.raw(), fv)
                .map_err(d3d_err)?;
            fence.set_target(fv);
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
