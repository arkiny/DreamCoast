//! The engine RHI facade: enum-dispatch over graphics backends.
//!
//! Each GPU object is an `enum` with one variant per backend available on the
//! target OS. Methods match on the variant and forward to the backend — no
//! vtable, all backends for the target compiled in, backend chosen at runtime via
//! [`Instance::new`]. Consumers (e.g. `sandbox`) depend only on this crate, never
//! on a backend directly.
//!
//! The active backend set is platform-gated: on Windows the variants are `Vulkan`
//! (ash) and `D3d12` (windows-rs); on macOS the variant is `Metal` (objc2). The
//! per-arm `#[cfg]`s select which forwarding code exists on each OS.
//!
//! All objects in a frame come from the same [`Device`], so cross-backend
//! argument combinations are impossible; those match arms are `unreachable!`.
//!
//! Backend-agnostic descriptors and enums are re-exported from [`rhi_types`].

pub use rhi_types::*;

use dreamcoast_core::EngineError;
use dreamcoast_platform::Window;

mod command_list;
pub use command_list::{CommandList, Recorder, RhiCommand};

type Result<T> = std::result::Result<T, EngineError>;

/// Panic message for impossible cross-backend argument mixes.
#[cfg(windows)]
const MIXED: &str = "RHI objects from different backends were mixed";

macro_rules! backend_enum {
    ($(#[$m:meta])* $name:ident => $vk:ty, $dx:ty, $mtl:ty) => {
        $(#[$m])*
        pub enum $name {
            #[cfg(windows)]
            Vulkan($vk),
            #[cfg(windows)]
            D3d12($dx),
            #[cfg(target_os = "macos")]
            Metal($mtl),
        }
    };
}

backend_enum!(/// A graphics instance bound to a window surface.
    Instance => rhi_vulkan::VulkanInstance, rhi_d3d12::D3d12Instance, rhi_metal::MetalInstance);
backend_enum!(/// A logical device: the factory for GPU resources.
    Device => rhi_vulkan::VulkanDevice, rhi_d3d12::D3d12Device, rhi_metal::MetalDevice);
backend_enum!(/// A submission/present queue.
    Queue => rhi_vulkan::VulkanQueue, rhi_d3d12::D3d12Queue, rhi_metal::MetalQueue);
backend_enum!(/// An async-compute queue overlapping the graphics queue (Phase 7).
    ComputeQueue => rhi_vulkan::VulkanComputeQueue, rhi_d3d12::D3d12ComputeQueue, rhi_metal::MetalComputeQueue);
backend_enum!(/// A window swapchain.
    Swapchain => rhi_vulkan::VulkanSwapchain, rhi_d3d12::D3d12Swapchain, rhi_metal::MetalSwapchain);
backend_enum!(/// A graphics pipeline.
    GraphicsPipeline => rhi_vulkan::VulkanGraphicsPipeline, rhi_d3d12::D3d12GraphicsPipeline, rhi_metal::MetalGraphicsPipeline);
backend_enum!(/// A compute pipeline (Phase 7).
    ComputePipeline => rhi_vulkan::VulkanComputePipeline, rhi_d3d12::D3d12ComputePipeline, rhi_metal::MetalComputePipeline);
backend_enum!(/// A primary command buffer.
    CommandBuffer => rhi_vulkan::VulkanCommandBuffer, rhi_d3d12::D3d12CommandBuffer, rhi_metal::MetalCommandBuffer);
backend_enum!(/// A CPU-GPU fence.
    Fence => rhi_vulkan::VulkanFence, rhi_d3d12::D3d12Fence, rhi_metal::MetalFence);
backend_enum!(/// A GPU-GPU binary semaphore (no-op on D3D12 / single-queue Metal).
    Semaphore => rhi_vulkan::VulkanSemaphore, rhi_d3d12::D3d12Semaphore, rhi_metal::MetalSemaphore);
backend_enum!(/// A host-visible buffer (vertex/index).
    Buffer => rhi_vulkan::VulkanBuffer, rhi_d3d12::D3d12Buffer, rhi_metal::MetalBuffer);
backend_enum!(/// A device-local storage buffer (UAV) for compute (Phase 7).
    StorageBuffer => rhi_vulkan::VulkanStorageBuffer, rhi_d3d12::D3d12StorageBuffer, rhi_metal::MetalStorageBuffer);
backend_enum!(/// A sampled 2D texture registered in the bindless table.
    Texture => rhi_vulkan::VulkanTexture, rhi_d3d12::D3d12Texture, rhi_metal::MetalTexture);
backend_enum!(/// A depth buffer for the mesh pass.
    DepthBuffer => rhi_vulkan::VulkanDepthBuffer, rhi_d3d12::D3d12DepthBuffer, rhi_metal::MetalDepthBuffer);
backend_enum!(/// An offscreen color render target (attachment + bindless sampled).
    RenderTarget => rhi_vulkan::VulkanRenderTarget, rhi_d3d12::D3d12RenderTarget, rhi_metal::MetalRenderTarget);
backend_enum!(/// A render-target cubemap (6 faces + bindless `TextureCube`), for IBL.
    Cubemap => rhi_vulkan::VulkanCubemap, rhi_d3d12::D3d12Cubemap, rhi_metal::MetalCubemap);
backend_enum!(/// A 3D volume texture: compute-writable storage + trilinear sampled (Phase 11).
    Volume => rhi_vulkan::VulkanVolume, rhi_d3d12::D3d12Volume, rhi_metal::MetalVolume);
backend_enum!(/// A heap that transient render targets alias into at graph-computed offsets.
    TransientHeap => rhi_vulkan::VulkanTransientHeap, rhi_d3d12::D3d12TransientHeap, rhi_metal::MetalTransientHeap);
backend_enum!(/// A built ray-tracing scene: BLAS per mesh + one TLAS (Phase 8).
    RaytracingScene => rhi_vulkan::VulkanRaytracingScene, rhi_d3d12::D3d12RaytracingScene, rhi_metal::MetalRaytracingScene);
backend_enum!(/// A hardware ray-tracing pipeline + shader binding table (Phase 8 M5).
    RaytracingPipeline => rhi_vulkan::VulkanRaytracingPipeline, rhi_d3d12::D3d12RaytracingPipeline, rhi_metal::MetalRaytracingPipeline);
backend_enum!(/// A GPU timestamp query heap for per-pass profiling (Phase 9 M1).
    QueryHeap => rhi_vulkan::VulkanQueryHeap, rhi_d3d12::D3d12QueryHeap, rhi_metal::MetalQueryHeap);

// Phase 15 M4 B3 — the RHI-thread handoff boundary.
//
// These objects are not `Send` by default: the backend types hold an
// `Rc<DeviceShared>` (non-atomic refcount) + `RefCell`/`Cell` interior mutability
// (Metal), and objc2/ash/COM handles. The B3 design (`P15_RHI_THREAD`) moves a
// fixed set of them — the graphics/compute queues, the swapchain, the per-fif
// command buffers, and the frame fences/semaphores — onto a single RHI thread that
// *solely owns* them for the program's lifetime.
//
// SAFETY: soundness rests on the M4 handoff contract, not on these types becoming
// thread-safe:
//   * Single-owner handoff — each boundary object is moved to the RHI thread once
//     at boot and is never touched by the record thread again; at most one thread
//     accesses it at any instant, so the inner `RefCell`/`Cell` is never aliased.
//   * No concurrent `Rc` traffic — the record thread keeps `Device` (sharing the
//     same `Rc<DeviceShared>`), but the handoff guarantees no `Rc` *clone/drop* of
//     the moved objects happens off-thread during a frame: the RHI thread only
//     *borrows* through them, and teardown drops them after the RHI thread has
//     joined (single-threaded again).
//   * objc2 backing — `MTLCommandQueue`/`MTLDevice` are documented thread-safe;
//     `MTLCommandBuffer` encoding is single-thread but each per-fif buffer lives on
//     exactly one thread here, and commit from another thread is allowed.
// VK/DX `Send` soundness (ash handles / D3D12 COM) is asserted by the same
// single-owner contract but verified only on the Windows box (parity pending).
unsafe impl Send for Queue {}
unsafe impl Send for ComputeQueue {}
unsafe impl Send for Swapchain {}
unsafe impl Send for CommandBuffer {}
unsafe impl Send for Fence {}
unsafe impl Send for Semaphore {}

/// One mesh's geometry for a BLAS build: its vertex + index buffers plus the
/// plain shape data (Phase 8). Pairs facade [`Buffer`] handles with [`BlasGeometry`].
pub struct RtGeometry<'a> {
    pub vertex_buffer: &'a Buffer,
    pub index_buffer: &'a Buffer,
    pub geometry: BlasGeometry,
}

impl Buffer {
    /// Copy bytes into the buffer (host-visible).
    pub fn write(&self, data: &[u8]) -> Result<()> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(b) => b.write(data),
            #[cfg(windows)]
            Self::D3d12(b) => b.write(data),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => b.write(data),
        }
    }

    /// Copy bytes into the buffer at `offset` (for per-frame uniform slices).
    pub fn write_at(&self, offset: u64, data: &[u8]) -> Result<()> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(b) => b.write_at(offset, data),
            #[cfg(windows)]
            Self::D3d12(b) => b.write_at(offset, data),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => b.write_at(offset, data),
        }
    }

    /// Copy bytes out of the buffer into `dst` (for `Readback` buffers).
    pub fn read_into(&self, dst: &mut [u8]) -> Result<()> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(b) => {
                b.read_into(dst);
                Ok(())
            }
            #[cfg(windows)]
            Self::D3d12(b) => b.read_into(dst),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => b.read_into(dst),
        }
    }
}

impl Texture {
    /// Index of this texture in the device's bindless table.
    pub fn bindless_index(&self) -> u32 {
        match self {
            #[cfg(windows)]
            Self::Vulkan(t) => t.bindless_index(),
            #[cfg(windows)]
            Self::D3d12(t) => t.bindless_index(),
            #[cfg(target_os = "macos")]
            Self::Metal(t) => t.bindless_index(),
        }
    }
}

impl StorageBuffer {
    /// Index of this buffer in the device's bindless storage-buffer table.
    pub fn storage_index(&self) -> u32 {
        match self {
            #[cfg(windows)]
            Self::Vulkan(b) => b.storage_index(),
            #[cfg(windows)]
            Self::D3d12(b) => b.storage_index(),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => b.storage_index(),
        }
    }
}

impl RenderTarget {
    /// Bindless storage-image (UAV) index, if created with `storage`.
    pub fn storage_index(&self) -> Option<u32> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(t) => t.storage_index(),
            #[cfg(windows)]
            Self::D3d12(t) => t.storage_index(),
            #[cfg(target_os = "macos")]
            Self::Metal(t) => t.storage_index(),
        }
    }

    /// Index of this render target in the device's bindless table.
    pub fn bindless_index(&self) -> u32 {
        match self {
            #[cfg(windows)]
            Self::Vulkan(t) => t.bindless_index(),
            #[cfg(windows)]
            Self::D3d12(t) => t.bindless_index(),
            #[cfg(target_os = "macos")]
            Self::Metal(t) => t.bindless_index(),
        }
    }

    /// Tag this target with a debug name for GPU captures (Phase 9 M2).
    pub fn set_name(&self, name: &str) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(t) => t.set_name(name),
            #[cfg(windows)]
            Self::D3d12(t) => t.set_name(name),
            #[cfg(target_os = "macos")]
            Self::Metal(t) => t.set_name(name),
        }
    }
}

impl Volume {
    /// `volumes[]` (SRV) bindless index for trilinear sampling.
    pub fn sampled_index(&self) -> u32 {
        match self {
            #[cfg(windows)]
            Self::Vulkan(v) => v.sampled_index(),
            #[cfg(windows)]
            Self::D3d12(v) => v.sampled_index(),
            #[cfg(target_os = "macos")]
            Self::Metal(v) => v.sampled_index(),
        }
    }

    /// `storage_volumes[]` (UAV) bindless index for compute bakes.
    pub fn storage_index(&self) -> u32 {
        match self {
            #[cfg(windows)]
            Self::Vulkan(v) => v.storage_index(),
            #[cfg(windows)]
            Self::D3d12(v) => v.storage_index(),
            #[cfg(target_os = "macos")]
            Self::Metal(v) => v.storage_index(),
        }
    }
}

impl DepthBuffer {
    /// Index of this depth buffer in the device's bindless table (shadow map).
    pub fn bindless_index(&self) -> u32 {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => d.bindless_index(),
            #[cfg(windows)]
            Self::D3d12(d) => d.bindless_index(),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => d.bindless_index(),
        }
    }

    /// Tag this depth buffer with a debug name for GPU captures (Phase 9 M2).
    pub fn set_name(&self, name: &str) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => d.set_name(name),
            #[cfg(windows)]
            Self::D3d12(d) => d.set_name(name),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => d.set_name(name),
        }
    }
}

impl Cubemap {
    /// Index of this cubemap in the device's bindless cube table.
    pub fn bindless_index(&self) -> u32 {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.bindless_index(),
            #[cfg(windows)]
            Self::D3d12(c) => c.bindless_index(),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.bindless_index(),
        }
    }

    /// Number of mip levels.
    pub fn mip_levels(&self) -> u32 {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.mip_levels(),
            #[cfg(windows)]
            Self::D3d12(c) => c.mip_levels(),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.mip_levels(),
        }
    }

    /// Edge length of `mip` (`size >> mip`, at least 1).
    pub fn mip_size(&self, mip: u32) -> u32 {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.mip_size(mip),
            #[cfg(windows)]
            Self::D3d12(c) => c.mip_size(mip),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.mip_size(mip),
        }
    }

    /// Tag this cubemap with a debug name for GPU captures (Phase 9 M2).
    pub fn set_name(&self, name: &str) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.set_name(name),
            #[cfg(windows)]
            Self::D3d12(c) => c.set_name(name),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.set_name(name),
        }
    }
}

impl QueryHeap {
    /// Number of timestamp slots in the heap.
    pub fn count(&self) -> u32 {
        match self {
            #[cfg(windows)]
            Self::Vulkan(h) => h.count(),
            #[cfg(windows)]
            Self::D3d12(h) => h.count(),
            #[cfg(target_os = "macos")]
            Self::Metal(h) => h.count(),
        }
    }

    /// Nanoseconds per timestamp tick (multiply tick deltas by this for ns).
    pub fn period_ns(&self) -> f32 {
        match self {
            #[cfg(windows)]
            Self::Vulkan(h) => h.period_ns(),
            #[cfg(windows)]
            Self::D3d12(h) => h.period_ns(),
            #[cfg(target_os = "macos")]
            Self::Metal(h) => h.period_ns(),
        }
    }

    /// Read all raw timestamp ticks. Call only after the submission that wrote
    /// them has completed (e.g. after the frame fence).
    pub fn read(&self) -> Vec<u64> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(h) => h.read(),
            #[cfg(windows)]
            Self::D3d12(h) => h.read(),
            #[cfg(target_os = "macos")]
            Self::Metal(h) => h.read(),
        }
    }
}

impl Instance {
    /// Create an instance for the requested backend. Returns an error if the
    /// backend is not available on the current platform.
    pub fn new(backend: BackendKind, window: &Window, desc: &InstanceDesc) -> Result<Self> {
        match backend {
            #[cfg(windows)]
            BackendKind::Vulkan => Ok(Self::Vulkan(rhi_vulkan::VulkanInstance::new(window, desc)?)),
            #[cfg(windows)]
            BackendKind::D3d12 => Ok(Self::D3d12(rhi_d3d12::D3d12Instance::new(window, desc)?)),
            #[cfg(target_os = "macos")]
            BackendKind::Metal => Ok(Self::Metal(rhi_metal::MetalInstance::new(window, desc)?)),
            #[allow(unreachable_patterns)]
            other => Err(EngineError::Rhi(format!(
                "backend {other:?} is not available on this platform"
            ))),
        }
    }

    /// Create a logical device.
    pub fn create_device(&self) -> Result<Device> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(i) => Ok(Device::Vulkan(i.create_device()?)),
            #[cfg(windows)]
            Self::D3d12(i) => Ok(Device::D3d12(i.create_device()?)),
            #[cfg(target_os = "macos")]
            Self::Metal(i) => Ok(Device::Metal(i.create_device()?)),
        }
    }

    /// The backend kind in use.
    pub fn backend(&self) -> BackendKind {
        match self {
            #[cfg(windows)]
            Self::Vulkan(i) => i.backend(),
            #[cfg(windows)]
            Self::D3d12(i) => i.backend(),
            #[cfg(target_os = "macos")]
            Self::Metal(i) => i.backend(),
        }
    }
}

impl Device {
    pub fn create_swapchain(&self, desc: &SwapchainDesc) -> Result<Swapchain> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(Swapchain::Vulkan(d.create_swapchain(desc)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(Swapchain::D3d12(d.create_swapchain(desc)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(Swapchain::Metal(d.create_swapchain(desc)?)),
        }
    }

    pub fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc,
    ) -> Result<GraphicsPipeline> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(GraphicsPipeline::Vulkan(d.create_graphics_pipeline(desc)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(GraphicsPipeline::D3d12(d.create_graphics_pipeline(desc)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(GraphicsPipeline::Metal(d.create_graphics_pipeline(desc)?)),
        }
    }

    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> Result<ComputePipeline> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(ComputePipeline::Vulkan(d.create_compute_pipeline(desc)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(ComputePipeline::D3d12(d.create_compute_pipeline(desc)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(ComputePipeline::Metal(d.create_compute_pipeline(desc)?)),
        }
    }

    pub fn create_command_buffer(&self) -> Result<CommandBuffer> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(CommandBuffer::Vulkan(d.create_command_buffer()?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(CommandBuffer::D3d12(d.create_command_buffer()?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(CommandBuffer::Metal(d.create_command_buffer()?)),
        }
    }

    /// Create a GPU timestamp query heap of `count` queries for per-pass profiling
    /// (Phase 9 M1). Pair with [`RenderGraph::execute`]'s profiler argument.
    pub fn create_query_heap(&self, count: u32) -> Result<QueryHeap> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(QueryHeap::Vulkan(d.create_query_heap(count)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(QueryHeap::D3d12(d.create_query_heap(count)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(QueryHeap::Metal(d.create_query_heap(count)?)),
        }
    }

    /// Allocate a command buffer for the async-compute queue (Phase 7).
    pub fn create_compute_command_buffer(&self) -> Result<CommandBuffer> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(CommandBuffer::Vulkan(d.create_compute_command_buffer()?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(CommandBuffer::D3d12(d.create_compute_command_buffer()?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(CommandBuffer::Metal(d.create_compute_command_buffer()?)),
        }
    }

    /// The async-compute queue, for work that overlaps the graphics queue (Phase 7).
    pub fn compute_queue(&self) -> ComputeQueue {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => ComputeQueue::Vulkan(d.compute_queue()),
            #[cfg(windows)]
            Self::D3d12(d) => ComputeQueue::D3d12(d.compute_queue()),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => ComputeQueue::Metal(d.compute_queue()),
        }
    }

    /// Whether a dedicated async-compute queue is available (else compute work
    /// would alias the graphics queue with no real overlap). D3D12 always exposes
    /// a COMPUTE queue; Vulkan depends on a dedicated compute family (Phase 7);
    /// Metal gains it in milestone M5.
    pub fn has_async_compute(&self) -> bool {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => d.has_async_compute(),
            #[cfg(windows)]
            Self::D3d12(d) => d.has_async_compute(),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => d.has_async_compute(),
        }
    }

    /// Whether hardware ray tracing is available (Vulkan KHR ray-tracing
    /// extensions / D3D12 DXR Tier >= 1.1) (Phase 8). Always false on Metal for
    /// now (Phase 8 deferred).
    pub fn has_raytracing(&self) -> bool {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => d.has_raytracing(),
            #[cfg(windows)]
            Self::D3d12(d) => d.has_raytracing(),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => d.has_raytracing(),
        }
    }

    /// Whether a DXR-style ray-tracing *pipeline* (raygen/miss/closesthit + SBT,
    /// [`CommandBuffer::trace_rays`]) is available. True on Vulkan/D3D12 when ray
    /// tracing is supported; always false on Metal, whose hardware ray tracing is
    /// inline-only (`DispatchRays` has no equivalent — the path tracer runs through
    /// the inline `rt_path` compute path). Callers gate the RT-pipeline path on this.
    pub fn supports_rt_pipeline(&self) -> bool {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => d.has_raytracing(),
            #[cfg(windows)]
            Self::D3d12(d) => d.has_raytracing(),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => d.supports_rt_pipeline(),
        }
    }

    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<Buffer> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(Buffer::Vulkan(d.create_buffer(desc)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(Buffer::D3d12(d.create_buffer(desc)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(Buffer::Metal(d.create_buffer(desc)?)),
        }
    }

    /// Build the scene's acceleration structures (BLAS per mesh + one TLAS) for a
    /// static scene (Phase 8). Requires [`Self::has_raytracing`]. `instances`
    /// reference geometries by index (`blas_index`). Metal returns an error
    /// (hardware ray tracing deferred).
    pub fn build_raytracing_scene(
        &self,
        geometries: &[RtGeometry],
        instances: &[TlasInstance],
    ) -> Result<RaytracingScene> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => {
                let geos: Vec<_> = geometries
                    .iter()
                    .map(|g| match (g.vertex_buffer, g.index_buffer) {
                        (Buffer::Vulkan(v), Buffer::Vulkan(i)) => (v, i, g.geometry),
                        _ => unreachable!("{MIXED}"),
                    })
                    .collect();
                Ok(RaytracingScene::Vulkan(
                    d.build_raytracing_scene(&geos, instances)?,
                ))
            }
            #[cfg(windows)]
            Self::D3d12(d) => {
                let geos: Vec<_> = geometries
                    .iter()
                    .map(|g| match (g.vertex_buffer, g.index_buffer) {
                        (Buffer::D3d12(v), Buffer::D3d12(i)) => (v, i, g.geometry),
                        _ => unreachable!("{MIXED}"),
                    })
                    .collect();
                Ok(RaytracingScene::D3d12(
                    d.build_raytracing_scene(&geos, instances)?,
                ))
            }
            #[cfg(target_os = "macos")]
            Self::Metal(d) => {
                let geos: Vec<_> = geometries
                    .iter()
                    .map(|g| {
                        let Buffer::Metal(v) = g.vertex_buffer;
                        let Buffer::Metal(i) = g.index_buffer;
                        (v, i, g.geometry)
                    })
                    .collect();
                Ok(RaytracingScene::Metal(
                    d.build_raytracing_scene(&geos, instances)?,
                ))
            }
        }
    }

    /// Create a hardware ray-tracing pipeline + shader binding table (Phase 8 M5).
    /// Requires [`Self::has_raytracing`]. On Metal this is backed by Metal Shader
    /// Converter's kernel raygen + visible-function-table ABI.
    pub fn create_raytracing_pipeline(
        &self,
        desc: &RaytracingPipelineDesc,
    ) -> Result<RaytracingPipeline> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(RaytracingPipeline::Vulkan(
                d.create_raytracing_pipeline(desc)?,
            )),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(RaytracingPipeline::D3d12(
                d.create_raytracing_pipeline(desc)?,
            )),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(RaytracingPipeline::Metal(
                d.create_raytracing_pipeline(desc)?,
            )),
        }
    }

    /// Register a built scene's TLAS in the bindless table so shaders can trace it
    /// (Phase 8 M3). Call once after [`Self::build_raytracing_scene`].
    pub fn bind_tlas(&self, scene: &RaytracingScene) {
        match (self, scene) {
            #[cfg(windows)]
            (Self::Vulkan(d), RaytracingScene::Vulkan(s)) => d.bind_tlas(s),
            #[cfg(windows)]
            (Self::D3d12(d), RaytracingScene::D3d12(s)) => d.bind_tlas(s),
            #[cfg(target_os = "macos")]
            (Self::Metal(d), RaytracingScene::Metal(s)) => d.bind_tlas(s),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Create a device-local storage buffer (UAV) for compute (Phase 7).
    pub fn create_storage_buffer(&self, desc: &StorageBufferDesc) -> Result<StorageBuffer> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(StorageBuffer::Vulkan(d.create_storage_buffer(desc)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(StorageBuffer::D3d12(d.create_storage_buffer(desc)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(StorageBuffer::Metal(d.create_storage_buffer(desc)?)),
        }
    }

    /// Create a device-local storage buffer seeded with host `data` (Phase 8: RT
    /// geometry + per-instance table read by the path tracer).
    pub fn create_storage_buffer_init(
        &self,
        desc: &StorageBufferDesc,
        data: &[u8],
    ) -> Result<StorageBuffer> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(StorageBuffer::Vulkan(
                d.create_storage_buffer_init(desc, data)?,
            )),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(StorageBuffer::D3d12(
                d.create_storage_buffer_init(desc, data)?,
            )),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(StorageBuffer::Metal(
                d.create_storage_buffer_init(desc, data)?,
            )),
        }
    }

    /// Register the per-frame globals uniform buffer. `slice_size` is one frame's
    /// slice (selected per-frame via [`CommandBuffer::set_globals`]). On D3D12 this
    /// is a no-op (the globals are bound as a root CBV by GPU address per draw).
    pub fn set_globals_buffer(&self, buffer: &Buffer, slice_size: u64) {
        match (self, buffer) {
            #[cfg(windows)]
            (Self::Vulkan(d), Buffer::Vulkan(b)) => d.set_globals_buffer(b, slice_size),
            #[cfg(windows)]
            (Self::D3d12(_), Buffer::D3d12(_)) => {}
            #[cfg(target_os = "macos")]
            (Self::Metal(d), Buffer::Metal(b)) => d.set_globals_buffer(b, slice_size),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn create_texture(&self, desc: &TextureDesc, pixels: &[u8]) -> Result<Texture> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(Texture::Vulkan(d.create_texture(desc, pixels)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(Texture::D3d12(d.create_texture(desc, pixels)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(Texture::Metal(d.create_texture(desc, pixels)?)),
        }
    }

    /// Create a sampled texture from pre-compressed BCn mip levels (Phase 12 M3).
    /// `desc.format` must be a block-compressed format; `levels[i]` holds mip `i`'s
    /// blocks (level 0 full-res). The GPU samples the blocks natively — there is no
    /// decompression at load, and both backends upload identical bytes.
    pub fn create_texture_compressed(
        &self,
        desc: &TextureDesc,
        levels: &[Vec<u8>],
    ) -> Result<Texture> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(Texture::Vulkan(d.create_texture_compressed(desc, levels)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(Texture::D3d12(d.create_texture_compressed(desc, levels)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(Texture::Metal(d.create_texture_compressed(desc, levels)?)),
        }
    }

    pub fn create_depth_buffer(&self, extent: Extent2D) -> Result<DepthBuffer> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(DepthBuffer::Vulkan(d.create_depth_buffer(extent)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(DepthBuffer::D3d12(d.create_depth_buffer(extent)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(DepthBuffer::Metal(d.create_depth_buffer(extent)?)),
        }
    }

    pub fn create_render_target(&self, desc: &RenderTargetDesc) -> Result<RenderTarget> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(RenderTarget::Vulkan(d.create_render_target(desc)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(RenderTarget::D3d12(d.create_render_target(desc)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(RenderTarget::Metal(d.create_render_target(desc)?)),
        }
    }

    /// Create a 3D (volume) texture: compute-writable storage volume + trilinear
    /// sampled volume, registered in the bindless volume tables (Phase 11 Stage B).
    pub fn create_volume(&self, desc: &VolumeDesc) -> Result<Volume> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(Volume::Vulkan(d.create_volume(desc)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(Volume::D3d12(d.create_volume(desc)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(Volume::Metal(d.create_volume(desc)?)),
        }
    }

    /// Create a 3D volume seeded with host `data` — a deterministic CPU bake
    /// uploaded verbatim instead of a GPU compute bake (Phase 12 M2). `data` is
    /// `width*height*depth` voxels in `x + dim*(y + dim*z)` order. The volume is
    /// left ready to sample (trilinear). Both backends upload identical bytes, so
    /// the field is byte-identical across Vulkan and D3D12 by construction.
    pub fn create_volume_init(&self, desc: &VolumeDesc, data: &[u8]) -> Result<Volume> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(Volume::Vulkan(d.create_volume_init(desc, data)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(Volume::D3d12(d.create_volume_init(desc, data)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(Volume::Metal(d.create_volume_init(desc, data)?)),
        }
    }

    /// Read a 3D volume back to host memory (Phase 12 item 3) — the inverse of
    /// [`Device::create_volume_init`]. Returns `w*h*d*bytes_per_voxel` tightly
    /// packed bytes in `x + dim*(y + dim*z)` order. Synchronous (one-shot copy +
    /// map). Lets a GPU-produced volume be cooked / verified at the data level.
    pub fn read_volume(
        &self,
        volume: &Volume,
        w: u32,
        h: u32,
        d: u32,
        bytes_per_voxel: u32,
    ) -> Result<Vec<u8>> {
        match (self, volume) {
            #[cfg(windows)]
            (Self::Vulkan(dev), Volume::Vulkan(v)) => dev.read_volume(v, w, h, d, bytes_per_voxel),
            #[cfg(windows)]
            (Self::D3d12(dev), Volume::D3d12(v)) => dev.read_volume(v, w, h, d, bytes_per_voxel),
            #[cfg(target_os = "macos")]
            (Self::Metal(dev), Volume::Metal(v)) => dev.read_volume(v, w, h, d, bytes_per_voxel),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Create a render-target cubemap (6 faces, `mip_levels` each) for IBL.
    pub fn create_cubemap(&self, desc: &CubemapDesc) -> Result<Cubemap> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(Cubemap::Vulkan(d.create_cubemap(desc)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(Cubemap::D3d12(d.create_cubemap(desc)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(Cubemap::Metal(d.create_cubemap(desc)?)),
        }
    }

    /// CPU memory layout for reading a swapchain image back to the host (for
    /// screenshots). Use it to size a [`BufferUsage::Readback`] buffer and to skip
    /// per-row padding after [`CommandBuffer::copy_swapchain_to_buffer`].
    pub fn swapchain_readback_layout(&self, swapchain: &Swapchain) -> ReadbackLayout {
        match (self, swapchain) {
            #[cfg(windows)]
            (Self::Vulkan(d), Swapchain::Vulkan(s)) => d.swapchain_readback_layout(s),
            #[cfg(windows)]
            (Self::D3d12(d), Swapchain::D3d12(s)) => d.swapchain_readback_layout(s),
            #[cfg(target_os = "macos")]
            (Self::Metal(d), Swapchain::Metal(s)) => d.swapchain_readback_layout(s),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Memory footprint of an aliasable render target (for graph alias planning).
    pub fn render_target_memory(&self, desc: &RenderTargetDesc) -> Result<MemoryRequirements> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => d.render_target_memory(desc),
            #[cfg(windows)]
            Self::D3d12(d) => d.render_target_memory(desc),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => d.render_target_memory(desc),
        }
    }

    /// Create a transient heap of `size` bytes for aliased render targets.
    pub fn create_transient_heap(&self, size: u64) -> Result<TransientHeap> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(TransientHeap::Vulkan(d.create_transient_heap(size)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(TransientHeap::D3d12(d.create_transient_heap(size)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(TransientHeap::Metal(d.create_transient_heap(size)?)),
        }
    }

    /// Create a render target aliased into `heap` at `offset`.
    pub fn create_aliased_target(
        &self,
        heap: &TransientHeap,
        offset: u64,
        desc: &RenderTargetDesc,
    ) -> Result<RenderTarget> {
        match (self, heap) {
            #[cfg(windows)]
            (Self::Vulkan(d), TransientHeap::Vulkan(h)) => Ok(RenderTarget::Vulkan(
                d.create_aliased_target(h, offset, desc)?,
            )),
            #[cfg(windows)]
            (Self::D3d12(d), TransientHeap::D3d12(h)) => Ok(RenderTarget::D3d12(
                d.create_aliased_target(h, offset, desc)?,
            )),
            #[cfg(target_os = "macos")]
            (Self::Metal(d), TransientHeap::Metal(h)) => Ok(RenderTarget::Metal(
                d.create_aliased_target(h, offset, desc)?,
            )),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn create_fence(&self, signaled: bool) -> Result<Fence> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(Fence::Vulkan(d.create_fence(signaled)?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(Fence::D3d12(d.create_fence(signaled)?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(Fence::Metal(d.create_fence(signaled)?)),
        }
    }

    pub fn create_semaphore(&self) -> Result<Semaphore> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Ok(Semaphore::Vulkan(d.create_semaphore()?)),
            #[cfg(windows)]
            Self::D3d12(d) => Ok(Semaphore::D3d12(d.create_semaphore()?)),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Ok(Semaphore::Metal(d.create_semaphore()?)),
        }
    }

    pub fn queue(&self) -> Queue {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => Queue::Vulkan(d.queue()),
            #[cfg(windows)]
            Self::D3d12(d) => Queue::D3d12(d.queue()),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => Queue::Metal(d.queue()),
        }
    }

    pub fn wait_idle(&self) -> Result<()> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(d) => d.wait_idle(),
            #[cfg(windows)]
            Self::D3d12(d) => d.wait_idle(),
            #[cfg(target_os = "macos")]
            Self::Metal(d) => d.wait_idle(),
        }
    }

    /// The backend this device dispatches to.
    pub fn backend(&self) -> BackendKind {
        match self {
            #[cfg(windows)]
            Self::Vulkan(_) => BackendKind::Vulkan,
            #[cfg(windows)]
            Self::D3d12(_) => BackendKind::D3d12,
            #[cfg(target_os = "macos")]
            Self::Metal(_) => BackendKind::Metal,
        }
    }
}

impl Swapchain {
    /// Acquire the next image; `Some(index)` to render, `None` if it must be
    /// recreated first.
    pub fn acquire_next_image(&self, signal: &Semaphore) -> Result<Option<u32>> {
        match (self, signal) {
            #[cfg(windows)]
            (Self::Vulkan(s), Semaphore::Vulkan(sem)) => s.acquire_next_image(sem),
            #[cfg(windows)]
            (Self::D3d12(s), Semaphore::D3d12(sem)) => s.acquire_next_image(sem),
            #[cfg(target_os = "macos")]
            (Self::Metal(s), Semaphore::Metal(sem)) => s.acquire_next_image(sem),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn recreate(&mut self, desc: &SwapchainDesc) -> Result<()> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(s) => s.recreate(desc),
            #[cfg(windows)]
            Self::D3d12(s) => s.recreate(desc),
            #[cfg(target_os = "macos")]
            Self::Metal(s) => s.recreate(desc),
        }
    }

    pub fn format(&self) -> Format {
        match self {
            #[cfg(windows)]
            Self::Vulkan(s) => s.format(),
            #[cfg(windows)]
            Self::D3d12(s) => s.format(),
            #[cfg(target_os = "macos")]
            Self::Metal(s) => s.format(),
        }
    }

    pub fn extent_2d(&self) -> Extent2D {
        match self {
            #[cfg(windows)]
            Self::Vulkan(s) => s.extent_2d(),
            #[cfg(windows)]
            Self::D3d12(s) => s.extent_2d(),
            #[cfg(target_os = "macos")]
            Self::Metal(s) => s.extent_2d(),
        }
    }

    pub fn image_count(&self) -> u32 {
        match self {
            #[cfg(windows)]
            Self::Vulkan(s) => s.image_count(),
            #[cfg(windows)]
            Self::D3d12(s) => s.image_count(),
            #[cfg(target_os = "macos")]
            Self::Metal(s) => s.image_count(),
        }
    }
}

impl CommandBuffer {
    pub fn begin(&self) -> Result<()> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.begin(),
            #[cfg(windows)]
            Self::D3d12(c) => c.begin(),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.begin(),
        }
    }

    pub fn end(&self) -> Result<()> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.end(),
            #[cfg(windows)]
            Self::D3d12(c) => c.end(),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.end(),
        }
    }

    /// Reset `count` timestamp queries (from `first`) before they are rewritten
    /// this frame. Record once at frame start, outside any render pass. No-op on
    /// D3D12 (Phase 9 M1).
    pub fn reset_queries(&self, heap: &QueryHeap, first: u32, count: u32) {
        match (self, heap) {
            #[cfg(windows)]
            (Self::Vulkan(c), QueryHeap::Vulkan(h)) => c.reset_queries(h, first, count),
            #[cfg(windows)]
            (Self::D3d12(c), QueryHeap::D3d12(h)) => c.reset_queries(h, first, count),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), QueryHeap::Metal(h)) => c.reset_queries(h, first, count),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Write a timestamp into query `index` at the current pipeline tail.
    pub fn write_timestamp(&self, heap: &QueryHeap, index: u32) {
        match (self, heap) {
            #[cfg(windows)]
            (Self::Vulkan(c), QueryHeap::Vulkan(h)) => c.write_timestamp(h, index),
            #[cfg(windows)]
            (Self::D3d12(c), QueryHeap::D3d12(h)) => c.write_timestamp(h, index),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), QueryHeap::Metal(h)) => c.write_timestamp(h, index),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Resolve `count` written timestamp queries into the heap's readback buffer
    /// (after the last write, before submit). No-op on Vulkan.
    pub fn resolve_queries(&self, heap: &QueryHeap, count: u32) {
        match (self, heap) {
            #[cfg(windows)]
            (Self::Vulkan(c), QueryHeap::Vulkan(h)) => c.resolve_queries(h, count),
            #[cfg(windows)]
            (Self::D3d12(c), QueryHeap::D3d12(h)) => c.resolve_queries(h, count),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), QueryHeap::Metal(h)) => c.resolve_queries(h, count),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Open a named debug-marker region for GPU captures (RenderDoc/PIX/NSight).
    /// Debug builds only; balance with [`Self::end_debug_label`] (Phase 9 M2).
    pub fn begin_debug_label(&self, name: &str) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.begin_debug_label(name),
            #[cfg(windows)]
            Self::D3d12(c) => c.begin_debug_label(name),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.begin_debug_label(name),
        }
    }

    /// Close the most recently opened debug-marker region.
    pub fn end_debug_label(&self) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.end_debug_label(),
            #[cfg(windows)]
            Self::D3d12(c) => c.end_debug_label(),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.end_debug_label(),
        }
    }

    pub fn transition_to_render_target(&self, swapchain: &Swapchain, image_index: u32) {
        match (self, swapchain) {
            #[cfg(windows)]
            (Self::Vulkan(c), Swapchain::Vulkan(s)) => {
                c.transition_to_render_target(s, image_index)
            }
            #[cfg(windows)]
            (Self::D3d12(c), Swapchain::D3d12(s)) => c.transition_to_render_target(s, image_index),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Swapchain::Metal(s)) => c.transition_to_render_target(s, image_index),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn transition_to_present(&self, swapchain: &Swapchain, image_index: u32) {
        match (self, swapchain) {
            #[cfg(windows)]
            (Self::Vulkan(c), Swapchain::Vulkan(s)) => c.transition_to_present(s, image_index),
            #[cfg(windows)]
            (Self::D3d12(c), Swapchain::D3d12(s)) => c.transition_to_present(s, image_index),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Swapchain::Metal(s)) => c.transition_to_present(s, image_index),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Begin a render pass. `color_clear = Some` clears the color attachment,
    /// `None` loads it (overlay pass). `depth = Some` attaches + clears depth.
    pub fn begin_rendering(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        color_clear: Option<ClearColor>,
        depth: Option<&DepthBuffer>,
    ) {
        match (self, swapchain, depth) {
            #[cfg(windows)]
            (Self::Vulkan(c), Swapchain::Vulkan(s), None) => {
                c.begin_rendering(s, image_index, color_clear, None)
            }
            #[cfg(windows)]
            (Self::Vulkan(c), Swapchain::Vulkan(s), Some(DepthBuffer::Vulkan(d))) => {
                c.begin_rendering(s, image_index, color_clear, Some(d))
            }
            #[cfg(windows)]
            (Self::D3d12(c), Swapchain::D3d12(s), None) => {
                c.begin_rendering(s, image_index, color_clear, None)
            }
            #[cfg(windows)]
            (Self::D3d12(c), Swapchain::D3d12(s), Some(DepthBuffer::D3d12(d))) => {
                c.begin_rendering(s, image_index, color_clear, Some(d))
            }
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Swapchain::Metal(s), None) => {
                c.begin_rendering(s, image_index, color_clear, None)
            }
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Swapchain::Metal(s), Some(DepthBuffer::Metal(d))) => {
                c.begin_rendering(s, image_index, color_clear, Some(d))
            }
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Begin a render pass into an offscreen color target. `color_clear = Some`
    /// clears it, `None` loads it. `depth = Some` attaches + clears depth. The
    /// target must be in render-target state (see [`Self::rt_to_render_target`]).
    pub fn begin_rendering_target(
        &self,
        target: &RenderTarget,
        color_clear: Option<ClearColor>,
        depth: Option<&DepthBuffer>,
    ) {
        match (self, target, depth) {
            #[cfg(windows)]
            (Self::Vulkan(c), RenderTarget::Vulkan(t), None) => {
                c.begin_rendering_target(t, color_clear, None)
            }
            #[cfg(windows)]
            (Self::Vulkan(c), RenderTarget::Vulkan(t), Some(DepthBuffer::Vulkan(d))) => {
                c.begin_rendering_target(t, color_clear, Some(d))
            }
            #[cfg(windows)]
            (Self::D3d12(c), RenderTarget::D3d12(t), None) => {
                c.begin_rendering_target(t, color_clear, None)
            }
            #[cfg(windows)]
            (Self::D3d12(c), RenderTarget::D3d12(t), Some(DepthBuffer::D3d12(d))) => {
                c.begin_rendering_target(t, color_clear, Some(d))
            }
            #[cfg(target_os = "macos")]
            (Self::Metal(c), RenderTarget::Metal(t), None) => {
                c.begin_rendering_target(t, color_clear, None)
            }
            #[cfg(target_os = "macos")]
            (Self::Metal(c), RenderTarget::Metal(t), Some(DepthBuffer::Metal(d))) => {
                c.begin_rendering_target(t, color_clear, Some(d))
            }
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Begin a render pass into N offscreen color targets (MRT) + optional depth.
    /// Each `Some(clear)` clears its target, `None` loads. All targets must be in
    /// render-target state (see [`Self::rt_to_render_target`]). `targets` must be
    /// non-empty and all from the same backend as this command buffer.
    pub fn begin_rendering_targets(
        &self,
        targets: &[(&RenderTarget, Option<ClearColor>)],
        depth: Option<&DepthBuffer>,
    ) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => {
                let vk_targets: Vec<_> = targets
                    .iter()
                    .map(|(t, clear)| match t {
                        RenderTarget::Vulkan(t) => (t, *clear),
                        _ => unreachable!("{MIXED}"),
                    })
                    .collect();
                match depth {
                    None => c.begin_rendering_targets(&vk_targets, None),
                    Some(DepthBuffer::Vulkan(d)) => c.begin_rendering_targets(&vk_targets, Some(d)),
                    _ => unreachable!("{MIXED}"),
                }
            }
            #[cfg(windows)]
            Self::D3d12(c) => {
                let dx_targets: Vec<_> = targets
                    .iter()
                    .map(|(t, clear)| match t {
                        RenderTarget::D3d12(t) => (t, *clear),
                        _ => unreachable!("{MIXED}"),
                    })
                    .collect();
                match depth {
                    None => c.begin_rendering_targets(&dx_targets, None),
                    Some(DepthBuffer::D3d12(d)) => c.begin_rendering_targets(&dx_targets, Some(d)),
                    _ => unreachable!("{MIXED}"),
                }
            }
            #[cfg(target_os = "macos")]
            Self::Metal(c) => {
                let mtl_targets: Vec<_> = targets
                    .iter()
                    .map(|(t, clear)| match t {
                        RenderTarget::Metal(t) => (t, *clear),
                    })
                    .collect();
                match depth {
                    None => c.begin_rendering_targets(&mtl_targets, None),
                    Some(DepthBuffer::Metal(d)) => c.begin_rendering_targets(&mtl_targets, Some(d)),
                }
            }
        }
    }

    /// Select the per-frame globals slice for the next PBR pipeline bind. `offset`
    /// is the byte offset of this frame's slice within the globals buffer
    /// registered via [`Device::set_globals_buffer`].
    pub fn set_globals(&self, buffer: &Buffer, offset: u64) {
        match (self, buffer) {
            #[cfg(windows)]
            (Self::Vulkan(c), Buffer::Vulkan(_)) => c.set_globals(offset as u32),
            #[cfg(windows)]
            (Self::D3d12(c), Buffer::D3d12(b)) => c.set_globals(b.gpu_va() + offset),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Buffer::Metal(_)) => c.set_globals(offset as u32),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Begin a depth-only render pass into `depth` (a shadow map): no color
    /// targets, depth cleared + stored. The depth must already be in
    /// depth-attachment state (see [`Self::depth_to_render_target`]).
    pub fn begin_rendering_depth_only(&self, depth: &DepthBuffer) {
        match (self, depth) {
            #[cfg(windows)]
            (Self::Vulkan(c), DepthBuffer::Vulkan(d)) => c.begin_rendering_depth_only(d),
            #[cfg(windows)]
            (Self::D3d12(c), DepthBuffer::D3d12(d)) => c.begin_rendering_depth_only(d),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), DepthBuffer::Metal(d)) => c.begin_rendering_depth_only(d),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a depth buffer into depth-attachment state for writing.
    pub fn depth_to_render_target(&self, depth: &DepthBuffer) {
        match (self, depth) {
            #[cfg(windows)]
            (Self::Vulkan(c), DepthBuffer::Vulkan(d)) => c.depth_to_render_target(d),
            #[cfg(windows)]
            (Self::D3d12(c), DepthBuffer::D3d12(d)) => c.depth_to_render_target(d),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), DepthBuffer::Metal(d)) => c.depth_to_render_target(d),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a depth buffer into shader-read state for sampling.
    pub fn depth_to_sampled(&self, depth: &DepthBuffer) {
        match (self, depth) {
            #[cfg(windows)]
            (Self::Vulkan(c), DepthBuffer::Vulkan(d)) => c.depth_to_sampled(d),
            #[cfg(windows)]
            (Self::D3d12(c), DepthBuffer::D3d12(d)) => c.depth_to_sampled(d),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), DepthBuffer::Metal(d)) => c.depth_to_sampled(d),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a whole cubemap into render-target state for writing its faces.
    pub fn cube_to_color(&self, cube: &Cubemap) {
        match (self, cube) {
            #[cfg(windows)]
            (Self::Vulkan(c), Cubemap::Vulkan(m)) => c.cube_to_color(m),
            #[cfg(windows)]
            (Self::D3d12(c), Cubemap::D3d12(m)) => c.cube_to_color(m),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Cubemap::Metal(m)) => c.cube_to_color(m),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a whole cubemap into shader-read state for sampling.
    pub fn cube_to_sampled(&self, cube: &Cubemap) {
        match (self, cube) {
            #[cfg(windows)]
            (Self::Vulkan(c), Cubemap::Vulkan(m)) => c.cube_to_sampled(m),
            #[cfg(windows)]
            (Self::D3d12(c), Cubemap::D3d12(m)) => c.cube_to_sampled(m),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Cubemap::Metal(m)) => c.cube_to_sampled(m),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Begin a render pass into one (face, mip) of a cubemap. The cubemap must
    /// already be in render-target state (see [`Self::cube_to_color`]).
    pub fn begin_rendering_cube_face(
        &self,
        cube: &Cubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
    ) {
        match (self, cube) {
            #[cfg(windows)]
            (Self::Vulkan(c), Cubemap::Vulkan(m)) => {
                c.begin_rendering_cube_face(m, face, mip, clear)
            }
            #[cfg(windows)]
            (Self::D3d12(c), Cubemap::D3d12(m)) => c.begin_rendering_cube_face(m, face, mip, clear),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Cubemap::Metal(m)) => c.begin_rendering_cube_face(m, face, mip, clear),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Begin rendering into one (face, mip) of a cubemap with a depth buffer
    /// (clears depth), for capturing scene geometry. The cube must be in
    /// render-target state, the depth in depth-attachment state.
    pub fn begin_rendering_cube_face_depth(
        &self,
        cube: &Cubemap,
        face: u32,
        mip: u32,
        clear: Option<ClearColor>,
        depth: &DepthBuffer,
    ) {
        match (self, cube, depth) {
            #[cfg(windows)]
            (Self::Vulkan(c), Cubemap::Vulkan(m), DepthBuffer::Vulkan(d)) => {
                c.begin_rendering_cube_face_depth(m, face, mip, clear, d)
            }
            #[cfg(windows)]
            (Self::D3d12(c), Cubemap::D3d12(m), DepthBuffer::D3d12(d)) => {
                c.begin_rendering_cube_face_depth(m, face, mip, clear, d)
            }
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Cubemap::Metal(m), DepthBuffer::Metal(d)) => {
                c.begin_rendering_cube_face_depth(m, face, mip, clear, d)
            }
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn end_rendering(&self) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.end_rendering(),
            #[cfg(windows)]
            Self::D3d12(c) => c.end_rendering(),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.end_rendering(),
        }
    }

    /// Copy a rendered swapchain image into a `Readback` buffer for screenshots.
    /// Call right after the render graph executes (the backbuffer is in present
    /// state); the image is restored to present state afterward. Submit the
    /// command buffer, wait for the fence, then [`Buffer::read_into`] and decode
    /// using [`Device::swapchain_readback_layout`].
    pub fn copy_swapchain_to_buffer(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        buffer: &Buffer,
    ) {
        match (self, swapchain, buffer) {
            #[cfg(windows)]
            (Self::Vulkan(c), Swapchain::Vulkan(s), Buffer::Vulkan(b)) => {
                c.copy_swapchain_to_buffer(s, image_index, b)
            }
            #[cfg(windows)]
            (Self::D3d12(c), Swapchain::D3d12(s), Buffer::D3d12(b)) => {
                c.copy_swapchain_to_buffer(s, image_index, b)
            }
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Swapchain::Metal(s), Buffer::Metal(b)) => {
                c.copy_swapchain_to_buffer(s, image_index, b)
            }
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn set_viewport_scissor(&self, swapchain: &Swapchain) {
        match (self, swapchain) {
            #[cfg(windows)]
            (Self::Vulkan(c), Swapchain::Vulkan(s)) => c.set_viewport_scissor(s),
            #[cfg(windows)]
            (Self::D3d12(c), Swapchain::D3d12(s)) => c.set_viewport_scissor(s),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Swapchain::Metal(s)) => c.set_viewport_scissor(s),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Set viewport and scissor to cover an arbitrary extent (offscreen target).
    pub fn set_viewport_scissor_extent(&self, extent: Extent2D) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.set_viewport_scissor_extent(extent),
            #[cfg(windows)]
            Self::D3d12(c) => c.set_viewport_scissor_extent(extent),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.set_viewport_scissor_extent(extent),
        }
    }

    /// Transition an offscreen target into render-target state for writing.
    pub fn rt_to_render_target(&self, target: &RenderTarget) {
        match (self, target) {
            #[cfg(windows)]
            (Self::Vulkan(c), RenderTarget::Vulkan(t)) => c.rt_to_render_target(t),
            #[cfg(windows)]
            (Self::D3d12(c), RenderTarget::D3d12(t)) => c.rt_to_render_target(t),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), RenderTarget::Metal(t)) => c.rt_to_render_target(t),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition an offscreen target into shader-read state for sampling.
    pub fn rt_to_sampled(&self, target: &RenderTarget) {
        match (self, target) {
            #[cfg(windows)]
            (Self::Vulkan(c), RenderTarget::Vulkan(t)) => c.rt_to_sampled(t),
            #[cfg(windows)]
            (Self::D3d12(c), RenderTarget::D3d12(t)) => c.rt_to_sampled(t),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), RenderTarget::Metal(t)) => c.rt_to_sampled(t),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Discard an aliased target's shared memory and ready it for writing (issued
    /// before the first write of a target that reuses another's heap region).
    pub fn aliasing_barrier(&self, target: &RenderTarget) {
        match (self, target) {
            #[cfg(windows)]
            (Self::Vulkan(c), RenderTarget::Vulkan(t)) => c.aliasing_barrier(t),
            #[cfg(windows)]
            (Self::D3d12(c), RenderTarget::D3d12(t)) => c.aliasing_barrier(t),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), RenderTarget::Metal(t)) => c.aliasing_barrier(t),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn bind_graphics_pipeline(&self, pipeline: &GraphicsPipeline) {
        match (self, pipeline) {
            #[cfg(windows)]
            (Self::Vulkan(c), GraphicsPipeline::Vulkan(p)) => c.bind_graphics_pipeline(p),
            #[cfg(windows)]
            (Self::D3d12(c), GraphicsPipeline::D3d12(p)) => c.bind_graphics_pipeline(p),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), GraphicsPipeline::Metal(p)) => c.bind_graphics_pipeline(p),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn draw(&self, vertex_count: u32, instance_count: u32) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.draw(vertex_count, instance_count),
            #[cfg(windows)]
            Self::D3d12(c) => c.draw(vertex_count, instance_count),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.draw(vertex_count, instance_count),
        }
    }

    /// Bind a compute pipeline (and its bindless tables, if any).
    pub fn bind_compute_pipeline(&self, pipeline: &ComputePipeline) {
        match (self, pipeline) {
            #[cfg(windows)]
            (Self::Vulkan(c), ComputePipeline::Vulkan(p)) => c.bind_compute_pipeline(p),
            #[cfg(windows)]
            (Self::D3d12(c), ComputePipeline::D3d12(p)) => c.bind_compute_pipeline(p),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), ComputePipeline::Metal(p)) => c.bind_compute_pipeline(p),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Dispatch the bound compute pipeline over `(x, y, z)` workgroups.
    pub fn dispatch(&self, x: u32, y: u32, z: u32) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.dispatch(x, y, z),
            #[cfg(windows)]
            Self::D3d12(c) => c.dispatch(x, y, z),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.dispatch(x, y, z),
        }
    }

    /// Upload push/root constants for the bound **compute** pipeline.
    pub fn push_constants_compute(&self, data: &[u8]) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.push_constants_compute(data),
            #[cfg(windows)]
            Self::D3d12(c) => c.push_constants_compute(data),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.push_constants_compute(data),
        }
    }

    /// Bind a ray-tracing pipeline + its shader binding table and the bindless
    /// tables (Phase 8 M5).
    pub fn bind_raytracing_pipeline(&self, pipeline: &RaytracingPipeline) {
        match (self, pipeline) {
            #[cfg(windows)]
            (Self::Vulkan(c), RaytracingPipeline::Vulkan(p)) => c.bind_raytracing_pipeline(p),
            #[cfg(windows)]
            (Self::D3d12(c), RaytracingPipeline::D3d12(p)) => c.bind_raytracing_pipeline(p),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), RaytracingPipeline::Metal(p)) => c.bind_raytracing_pipeline(p),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Upload push/root constants for the bound **ray-tracing** pipeline.
    pub fn push_constants_rt(&self, data: &[u8]) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.push_constants_rt(data),
            #[cfg(windows)]
            Self::D3d12(c) => c.push_constants_rt(data),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.push_constants_rt(data),
        }
    }

    /// Trace a `width` x `height` grid of rays through the bound RT pipeline's SBT.
    pub fn trace_rays(&self, pipeline: &RaytracingPipeline, width: u32, height: u32) {
        match (self, pipeline) {
            #[cfg(windows)]
            (Self::Vulkan(c), RaytracingPipeline::Vulkan(p)) => c.trace_rays(p, width, height),
            #[cfg(windows)]
            (Self::D3d12(c), RaytracingPipeline::D3d12(p)) => c.trace_rays(p, width, height),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), RaytracingPipeline::Metal(p)) => c.trace_rays(p, width, height),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a storage render target into compute-writable state.
    pub fn rt_to_storage(&self, target: &RenderTarget) {
        match (self, target) {
            #[cfg(windows)]
            (Self::Vulkan(c), RenderTarget::Vulkan(t)) => c.rt_to_storage(t),
            #[cfg(windows)]
            (Self::D3d12(c), RenderTarget::D3d12(t)) => c.rt_to_storage(t),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), RenderTarget::Metal(t)) => c.rt_to_storage(t),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a 3D volume into compute-writable storage for a bake pass (Phase 11).
    pub fn volume_to_storage(&self, volume: &Volume) {
        match (self, volume) {
            #[cfg(windows)]
            (Self::Vulkan(c), Volume::Vulkan(v)) => c.volume_to_storage(v),
            #[cfg(windows)]
            (Self::D3d12(c), Volume::D3d12(v)) => c.volume_to_storage(v),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Volume::Metal(v)) => c.volume_to_storage(v),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a 3D volume from compute-write into shader-read for sampling (Phase 11).
    pub fn volume_to_sampled(&self, volume: &Volume) {
        match (self, volume) {
            #[cfg(windows)]
            (Self::Vulkan(c), Volume::Vulkan(v)) => c.volume_to_sampled(v),
            #[cfg(windows)]
            (Self::D3d12(c), Volume::D3d12(v)) => c.volume_to_sampled(v),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Volume::Metal(v)) => c.volume_to_sampled(v),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a storage image from compute-write into shader-read for sampling.
    pub fn storage_to_sampled(&self, target: &RenderTarget) {
        match (self, target) {
            #[cfg(windows)]
            (Self::Vulkan(c), RenderTarget::Vulkan(t)) => c.storage_to_sampled(t),
            #[cfg(windows)]
            (Self::D3d12(c), RenderTarget::D3d12(t)) => c.storage_to_sampled(t),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), RenderTarget::Metal(t)) => c.storage_to_sampled(t),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// UAV barrier ordering a compute write to a storage buffer before later reads.
    pub fn storage_buffer_barrier(&self, buffer: &StorageBuffer) {
        match (self, buffer) {
            #[cfg(windows)]
            (Self::Vulkan(c), StorageBuffer::Vulkan(b)) => c.storage_buffer_barrier(b),
            #[cfg(windows)]
            (Self::D3d12(c), StorageBuffer::D3d12(b)) => c.storage_buffer_barrier(b),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), StorageBuffer::Metal(b)) => c.storage_buffer_barrier(b),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// COMPUTE-stage-only UAV barrier, for command buffers recorded on the async-compute queue
    /// (whose family can't reference the vertex/fragment stages of `storage_buffer_barrier`). On
    /// D3D12/Metal a UAV barrier is queue-agnostic, so it falls back to the regular barrier.
    pub fn storage_buffer_barrier_compute(&self, buffer: &StorageBuffer) {
        match (self, buffer) {
            #[cfg(windows)]
            (Self::Vulkan(c), StorageBuffer::Vulkan(b)) => c.storage_buffer_barrier_compute(b),
            #[cfg(windows)]
            (Self::D3d12(c), StorageBuffer::D3d12(b)) => c.storage_buffer_barrier(b),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), StorageBuffer::Metal(b)) => c.storage_buffer_barrier(b),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a storage buffer (compute write) into indirect-args state for
    /// `draw_indexed_indirect`.
    pub fn storage_buffer_to_indirect(&self, buffer: &StorageBuffer) {
        match (self, buffer) {
            #[cfg(windows)]
            (Self::Vulkan(c), StorageBuffer::Vulkan(b)) => c.storage_buffer_to_indirect(b),
            #[cfg(windows)]
            (Self::D3d12(c), StorageBuffer::D3d12(b)) => c.storage_buffer_to_indirect(b),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), StorageBuffer::Metal(b)) => c.storage_buffer_to_indirect(b),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a storage buffer back into compute-writable state (next frame).
    pub fn storage_buffer_to_storage(&self, buffer: &StorageBuffer) {
        match (self, buffer) {
            #[cfg(windows)]
            (Self::Vulkan(c), StorageBuffer::Vulkan(b)) => c.storage_buffer_to_storage(b),
            #[cfg(windows)]
            (Self::D3d12(c), StorageBuffer::D3d12(b)) => c.storage_buffer_to_storage(b),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), StorageBuffer::Metal(b)) => c.storage_buffer_to_storage(b),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Issue an indexed indirect draw reading args from `buffer` at `offset`
    /// (a `draw_count`-element array of `[index_count, instance_count, first_index,
    /// vertex_offset, first_instance]`).
    pub fn draw_indexed_indirect(&self, buffer: &StorageBuffer, offset: u64, draw_count: u32) {
        match (self, buffer) {
            #[cfg(windows)]
            (Self::Vulkan(c), StorageBuffer::Vulkan(b)) => {
                c.draw_indexed_indirect(b, offset, draw_count)
            }
            #[cfg(windows)]
            (Self::D3d12(c), StorageBuffer::D3d12(b)) => {
                c.draw_indexed_indirect(b, offset, draw_count)
            }
            #[cfg(target_os = "macos")]
            (Self::Metal(c), StorageBuffer::Metal(b)) => {
                c.draw_indexed_indirect(b, offset, draw_count)
            }
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn set_scissor(&self, rect: Rect2D) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.set_scissor(rect),
            #[cfg(windows)]
            Self::D3d12(c) => c.set_scissor(rect),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.set_scissor(rect),
        }
    }

    pub fn bind_vertex_buffer(&self, buffer: &Buffer, stride: u32) {
        match (self, buffer) {
            #[cfg(windows)]
            (Self::Vulkan(c), Buffer::Vulkan(b)) => c.bind_vertex_buffer(b, stride),
            #[cfg(windows)]
            (Self::D3d12(c), Buffer::D3d12(b)) => c.bind_vertex_buffer(b, stride),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Buffer::Metal(b)) => c.bind_vertex_buffer(b, stride),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn bind_index_buffer(&self, buffer: &Buffer, wide: bool) {
        match (self, buffer) {
            #[cfg(windows)]
            (Self::Vulkan(c), Buffer::Vulkan(b)) => c.bind_index_buffer(b, wide),
            #[cfg(windows)]
            (Self::D3d12(c), Buffer::D3d12(b)) => c.bind_index_buffer(b, wide),
            #[cfg(target_os = "macos")]
            (Self::Metal(c), Buffer::Metal(b)) => c.bind_index_buffer(b, wide),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn push_constants(&self, data: &[u8]) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.push_constants(data),
            #[cfg(windows)]
            Self::D3d12(c) => c.push_constants(data),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.push_constants(data),
        }
    }

    pub fn draw_indexed(&self, index_count: u32, first_index: u32, vertex_offset: i32) {
        match self {
            #[cfg(windows)]
            Self::Vulkan(c) => c.draw_indexed(index_count, first_index, vertex_offset),
            #[cfg(windows)]
            Self::D3d12(c) => c.draw_indexed(index_count, first_index, vertex_offset),
            #[cfg(target_os = "macos")]
            Self::Metal(c) => c.draw_indexed(index_count, first_index, vertex_offset),
        }
    }
}

impl Queue {
    pub fn submit(
        &self,
        cmd: &CommandBuffer,
        wait: &Semaphore,
        signal: &Semaphore,
        fence: &Fence,
    ) -> Result<()> {
        match (self, cmd, wait, signal, fence) {
            #[cfg(windows)]
            (
                Self::Vulkan(q),
                CommandBuffer::Vulkan(c),
                Semaphore::Vulkan(w),
                Semaphore::Vulkan(s),
                Fence::Vulkan(f),
            ) => q.submit(c, w, s, f),
            #[cfg(windows)]
            (
                Self::D3d12(q),
                CommandBuffer::D3d12(c),
                Semaphore::D3d12(w),
                Semaphore::D3d12(s),
                Fence::D3d12(f),
            ) => q.submit(c, w, s, f),
            #[cfg(target_os = "macos")]
            (
                Self::Metal(q),
                CommandBuffer::Metal(c),
                Semaphore::Metal(w),
                Semaphore::Metal(s),
                Fence::Metal(f),
            ) => q.submit(c, w, s, f),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Submit graphics work that consumes async-compute output (Phase 7). The
    /// graphics queue GPU-waits on the compute queue's completion (`compute_wait`
    /// on Vulkan; a cross-queue fence on D3D12, where the semaphores are no-ops)
    /// before running, so the draw sees the compute-written buffer. Also waits on
    /// `wait` (image-acquired) and signals `signal`/`fence` like `submit`.
    pub fn submit_async(
        &self,
        cmd: &CommandBuffer,
        wait: &Semaphore,
        compute_wait: &Semaphore,
        signal: &Semaphore,
        fence: &Fence,
    ) -> Result<()> {
        match (self, cmd, wait, compute_wait, signal, fence) {
            #[cfg(windows)]
            (
                Self::Vulkan(q),
                CommandBuffer::Vulkan(c),
                Semaphore::Vulkan(w),
                Semaphore::Vulkan(cw),
                Semaphore::Vulkan(s),
                Fence::Vulkan(f),
            ) => q.submit_async(c, w, cw, s, f),
            #[cfg(windows)]
            (
                Self::D3d12(q),
                CommandBuffer::D3d12(c),
                Semaphore::D3d12(w),
                Semaphore::D3d12(_cw),
                Semaphore::D3d12(s),
                Fence::D3d12(f),
            ) => q.submit_async(c, w, s, f),
            #[cfg(target_os = "macos")]
            (
                Self::Metal(q),
                CommandBuffer::Metal(c),
                Semaphore::Metal(w),
                Semaphore::Metal(cw),
                Semaphore::Metal(s),
                Fence::Metal(f),
            ) => q.submit_async(c, w, cw, s, f),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Submit one command buffer with no semaphore sync, signaling `fence`. For
    /// one-off startup work (e.g. IBL cubemap generation).
    pub fn submit_oneshot(&self, cmd: &CommandBuffer, fence: &Fence) -> Result<()> {
        match (self, cmd, fence) {
            #[cfg(windows)]
            (Self::Vulkan(q), CommandBuffer::Vulkan(c), Fence::Vulkan(f)) => q.submit_oneshot(c, f),
            #[cfg(windows)]
            (Self::D3d12(q), CommandBuffer::D3d12(c), Fence::D3d12(f)) => q.submit_oneshot(c, f),
            #[cfg(target_os = "macos")]
            (Self::Metal(q), CommandBuffer::Metal(c), Fence::Metal(f)) => q.submit_oneshot(c, f),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Present a swapchain image; returns `true` if it needs recreation.
    pub fn present(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        wait: &Semaphore,
    ) -> Result<bool> {
        match (self, swapchain, wait) {
            #[cfg(windows)]
            (Self::Vulkan(q), Swapchain::Vulkan(s), Semaphore::Vulkan(w)) => {
                q.present(s, image_index, w)
            }
            #[cfg(windows)]
            (Self::D3d12(q), Swapchain::D3d12(s), Semaphore::D3d12(w)) => {
                q.present(s, image_index, w)
            }
            #[cfg(target_os = "macos")]
            (Self::Metal(q), Swapchain::Metal(s), Semaphore::Metal(w)) => {
                q.present(s, image_index, w)
            }
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }
}

impl ComputeQueue {
    /// Submit async-compute work, signaling `signal` on completion. The graphics
    /// queue's `submit_async` waits on `signal` before reading the compute output.
    /// On D3D12 the semaphore is a no-op (a cross-queue fence carries the sync).
    pub fn submit(&self, cmd: &CommandBuffer, signal: &Semaphore) -> Result<()> {
        match (self, cmd, signal) {
            #[cfg(windows)]
            (Self::Vulkan(q), CommandBuffer::Vulkan(c), Semaphore::Vulkan(s)) => q.submit(c, s),
            #[cfg(windows)]
            (Self::D3d12(q), CommandBuffer::D3d12(c), Semaphore::D3d12(s)) => q.submit(c, s),
            #[cfg(target_os = "macos")]
            (Self::Metal(q), CommandBuffer::Metal(c), Semaphore::Metal(s)) => q.submit(c, s),
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Submit async-compute work, signaling `signal` (graphics waits it) AND `fence` (so the CPU
    /// knows the compute command buffer is free to re-record). For the cross-frame cache relight,
    /// where the graphics fence does not transitively cover this frame's compute submission.
    pub fn submit_fenced(
        &self,
        cmd: &CommandBuffer,
        signal: &Semaphore,
        fence: &Fence,
    ) -> Result<()> {
        match (self, cmd, signal, fence) {
            #[cfg(windows)]
            (Self::Vulkan(q), CommandBuffer::Vulkan(c), Semaphore::Vulkan(s), Fence::Vulkan(f)) => {
                q.submit_fenced(c, s, f)
            }
            #[cfg(windows)]
            (Self::D3d12(q), CommandBuffer::D3d12(c), Semaphore::D3d12(s), Fence::D3d12(f)) => {
                q.submit_fenced(c, s, f)
            }
            #[cfg(target_os = "macos")]
            (Self::Metal(q), CommandBuffer::Metal(c), Semaphore::Metal(s), Fence::Metal(f)) => {
                q.submit_fenced(c, s, f)
            }
            #[cfg(windows)]
            _ => unreachable!("{MIXED}"),
        }
    }
}

impl Fence {
    pub fn wait(&self) -> Result<()> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(f) => f.wait(),
            #[cfg(windows)]
            Self::D3d12(f) => f.wait(),
            #[cfg(target_os = "macos")]
            Self::Metal(f) => f.wait(),
        }
    }

    pub fn reset(&self) -> Result<()> {
        match self {
            #[cfg(windows)]
            Self::Vulkan(f) => f.reset(),
            #[cfg(windows)]
            Self::D3d12(f) => f.reset(),
            #[cfg(target_os = "macos")]
            Self::Metal(f) => f.reset(),
        }
    }
}
