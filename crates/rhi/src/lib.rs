//! The engine RHI facade: enum-dispatch over graphics backends.
//!
//! Each GPU object is an `enum` with one variant per backend. Methods match on
//! the variant and forward to the backend — no vtable, both backends compiled
//! in, backend chosen at runtime via [`Instance::new`]. Consumers (e.g.
//! `sandbox`) depend only on this crate, never on a backend directly.
//!
//! All objects in a frame come from the same [`Device`], so cross-backend
//! argument combinations are impossible; those match arms are `unreachable!`.
//!
//! Backend-agnostic descriptors and enums are re-exported from [`rhi_types`].

pub use rhi_types::*;

use dreamcoast_core::EngineError;
use dreamcoast_platform::Window;

type Result<T> = std::result::Result<T, EngineError>;

/// Panic message for impossible cross-backend argument mixes.
const MIXED: &str = "RHI objects from different backends were mixed";

macro_rules! backend_enum {
    ($(#[$m:meta])* $name:ident => $vk:ty, $dx:ty) => {
        $(#[$m])*
        pub enum $name {
            Vulkan($vk),
            D3d12($dx),
        }
    };
}

backend_enum!(/// A graphics instance bound to a window surface.
    Instance => rhi_vulkan::VulkanInstance, rhi_d3d12::D3d12Instance);
backend_enum!(/// A logical device: the factory for GPU resources.
    Device => rhi_vulkan::VulkanDevice, rhi_d3d12::D3d12Device);
backend_enum!(/// A submission/present queue.
    Queue => rhi_vulkan::VulkanQueue, rhi_d3d12::D3d12Queue);
backend_enum!(/// An async-compute queue overlapping the graphics queue (Phase 7).
    ComputeQueue => rhi_vulkan::VulkanComputeQueue, rhi_d3d12::D3d12ComputeQueue);
backend_enum!(/// A window swapchain.
    Swapchain => rhi_vulkan::VulkanSwapchain, rhi_d3d12::D3d12Swapchain);
backend_enum!(/// A graphics pipeline.
    GraphicsPipeline => rhi_vulkan::VulkanGraphicsPipeline, rhi_d3d12::D3d12GraphicsPipeline);
backend_enum!(/// A compute pipeline (Phase 7).
    ComputePipeline => rhi_vulkan::VulkanComputePipeline, rhi_d3d12::D3d12ComputePipeline);
backend_enum!(/// A primary command buffer.
    CommandBuffer => rhi_vulkan::VulkanCommandBuffer, rhi_d3d12::D3d12CommandBuffer);
backend_enum!(/// A CPU-GPU fence.
    Fence => rhi_vulkan::VulkanFence, rhi_d3d12::D3d12Fence);
backend_enum!(/// A GPU-GPU binary semaphore (no-op on D3D12).
    Semaphore => rhi_vulkan::VulkanSemaphore, rhi_d3d12::D3d12Semaphore);
backend_enum!(/// A host-visible buffer (vertex/index).
    Buffer => rhi_vulkan::VulkanBuffer, rhi_d3d12::D3d12Buffer);
backend_enum!(/// A device-local storage buffer (UAV) for compute (Phase 7).
    StorageBuffer => rhi_vulkan::VulkanStorageBuffer, rhi_d3d12::D3d12StorageBuffer);
backend_enum!(/// A sampled 2D texture registered in the bindless table.
    Texture => rhi_vulkan::VulkanTexture, rhi_d3d12::D3d12Texture);
backend_enum!(/// A depth buffer for the mesh pass.
    DepthBuffer => rhi_vulkan::VulkanDepthBuffer, rhi_d3d12::D3d12DepthBuffer);
backend_enum!(/// An offscreen color render target (attachment + bindless sampled).
    RenderTarget => rhi_vulkan::VulkanRenderTarget, rhi_d3d12::D3d12RenderTarget);
backend_enum!(/// A render-target cubemap (6 faces + bindless `TextureCube`), for IBL.
    Cubemap => rhi_vulkan::VulkanCubemap, rhi_d3d12::D3d12Cubemap);
backend_enum!(/// A heap that transient render targets alias into at graph-computed offsets.
    TransientHeap => rhi_vulkan::VulkanTransientHeap, rhi_d3d12::D3d12TransientHeap);

impl Buffer {
    /// Copy bytes into the buffer (host-visible).
    pub fn write(&self, data: &[u8]) -> Result<()> {
        match self {
            Self::Vulkan(b) => b.write(data),
            Self::D3d12(b) => b.write(data),
        }
    }

    /// Copy bytes into the buffer at `offset` (for per-frame uniform slices).
    pub fn write_at(&self, offset: u64, data: &[u8]) -> Result<()> {
        match self {
            Self::Vulkan(b) => b.write_at(offset, data),
            Self::D3d12(b) => b.write_at(offset, data),
        }
    }

    /// Copy bytes out of the buffer into `dst` (for `Readback` buffers).
    pub fn read_into(&self, dst: &mut [u8]) -> Result<()> {
        match self {
            Self::Vulkan(b) => {
                b.read_into(dst);
                Ok(())
            }
            Self::D3d12(b) => b.read_into(dst),
        }
    }
}

impl Texture {
    /// Index of this texture in the device's bindless table.
    pub fn bindless_index(&self) -> u32 {
        match self {
            Self::Vulkan(t) => t.bindless_index(),
            Self::D3d12(t) => t.bindless_index(),
        }
    }
}

impl StorageBuffer {
    /// Index of this buffer in the device's bindless storage-buffer table.
    pub fn storage_index(&self) -> u32 {
        match self {
            Self::Vulkan(b) => b.storage_index(),
            Self::D3d12(b) => b.storage_index(),
        }
    }
}

impl RenderTarget {
    /// Bindless storage-image (UAV) index, if created with `storage`.
    pub fn storage_index(&self) -> Option<u32> {
        match self {
            Self::Vulkan(t) => t.storage_index(),
            Self::D3d12(t) => t.storage_index(),
        }
    }
}

impl RenderTarget {
    /// Index of this render target in the device's bindless table.
    pub fn bindless_index(&self) -> u32 {
        match self {
            Self::Vulkan(t) => t.bindless_index(),
            Self::D3d12(t) => t.bindless_index(),
        }
    }
}

impl DepthBuffer {
    /// Index of this depth buffer in the device's bindless table (shadow map).
    pub fn bindless_index(&self) -> u32 {
        match self {
            Self::Vulkan(d) => d.bindless_index(),
            Self::D3d12(d) => d.bindless_index(),
        }
    }
}

impl Cubemap {
    /// Index of this cubemap in the device's bindless cube table.
    pub fn bindless_index(&self) -> u32 {
        match self {
            Self::Vulkan(c) => c.bindless_index(),
            Self::D3d12(c) => c.bindless_index(),
        }
    }

    /// Number of mip levels.
    pub fn mip_levels(&self) -> u32 {
        match self {
            Self::Vulkan(c) => c.mip_levels(),
            Self::D3d12(c) => c.mip_levels(),
        }
    }

    /// Edge length of `mip` (`size >> mip`, at least 1).
    pub fn mip_size(&self, mip: u32) -> u32 {
        match self {
            Self::Vulkan(c) => c.mip_size(mip),
            Self::D3d12(c) => c.mip_size(mip),
        }
    }
}

impl Instance {
    /// Create an instance for the requested backend.
    pub fn new(backend: BackendKind, window: &Window, desc: &InstanceDesc) -> Result<Self> {
        match backend {
            BackendKind::Vulkan => Ok(Self::Vulkan(rhi_vulkan::VulkanInstance::new(window, desc)?)),
            BackendKind::D3d12 => Ok(Self::D3d12(rhi_d3d12::D3d12Instance::new(window, desc)?)),
        }
    }

    /// Create a logical device.
    pub fn create_device(&self) -> Result<Device> {
        match self {
            Self::Vulkan(i) => Ok(Device::Vulkan(i.create_device()?)),
            Self::D3d12(i) => Ok(Device::D3d12(i.create_device()?)),
        }
    }

    /// The backend kind in use.
    pub fn backend(&self) -> BackendKind {
        match self {
            Self::Vulkan(i) => i.backend(),
            Self::D3d12(i) => i.backend(),
        }
    }
}

impl Device {
    pub fn create_swapchain(&self, desc: &SwapchainDesc) -> Result<Swapchain> {
        match self {
            Self::Vulkan(d) => Ok(Swapchain::Vulkan(d.create_swapchain(desc)?)),
            Self::D3d12(d) => Ok(Swapchain::D3d12(d.create_swapchain(desc)?)),
        }
    }

    pub fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc,
    ) -> Result<GraphicsPipeline> {
        match self {
            Self::Vulkan(d) => Ok(GraphicsPipeline::Vulkan(d.create_graphics_pipeline(desc)?)),
            Self::D3d12(d) => Ok(GraphicsPipeline::D3d12(d.create_graphics_pipeline(desc)?)),
        }
    }

    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> Result<ComputePipeline> {
        match self {
            Self::Vulkan(d) => Ok(ComputePipeline::Vulkan(d.create_compute_pipeline(desc)?)),
            Self::D3d12(d) => Ok(ComputePipeline::D3d12(d.create_compute_pipeline(desc)?)),
        }
    }

    pub fn create_command_buffer(&self) -> Result<CommandBuffer> {
        match self {
            Self::Vulkan(d) => Ok(CommandBuffer::Vulkan(d.create_command_buffer()?)),
            Self::D3d12(d) => Ok(CommandBuffer::D3d12(d.create_command_buffer()?)),
        }
    }

    /// Allocate a command buffer for the async-compute queue (Phase 7).
    pub fn create_compute_command_buffer(&self) -> Result<CommandBuffer> {
        match self {
            Self::Vulkan(d) => Ok(CommandBuffer::Vulkan(d.create_compute_command_buffer()?)),
            Self::D3d12(d) => Ok(CommandBuffer::D3d12(d.create_compute_command_buffer()?)),
        }
    }

    /// The async-compute queue, for work that overlaps the graphics queue (Phase 7).
    pub fn compute_queue(&self) -> ComputeQueue {
        match self {
            Self::Vulkan(d) => ComputeQueue::Vulkan(d.compute_queue()),
            Self::D3d12(d) => ComputeQueue::D3d12(d.compute_queue()),
        }
    }

    /// Whether a dedicated async-compute queue is available (else compute work
    /// would alias the graphics queue with no real overlap). D3D12 always exposes
    /// a COMPUTE queue; Vulkan depends on a dedicated compute family (Phase 7).
    pub fn has_async_compute(&self) -> bool {
        match self {
            Self::Vulkan(d) => d.has_async_compute(),
            Self::D3d12(d) => d.has_async_compute(),
        }
    }

    /// Whether hardware ray tracing is available (Vulkan KHR ray-tracing
    /// extensions / D3D12 DXR Tier >= 1.1) (Phase 8).
    pub fn has_raytracing(&self) -> bool {
        match self {
            Self::Vulkan(d) => d.has_raytracing(),
            Self::D3d12(d) => d.has_raytracing(),
        }
    }

    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<Buffer> {
        match self {
            Self::Vulkan(d) => Ok(Buffer::Vulkan(d.create_buffer(desc)?)),
            Self::D3d12(d) => Ok(Buffer::D3d12(d.create_buffer(desc)?)),
        }
    }

    /// Create a device-local storage buffer (UAV) for compute (Phase 7).
    pub fn create_storage_buffer(&self, desc: &StorageBufferDesc) -> Result<StorageBuffer> {
        match self {
            Self::Vulkan(d) => Ok(StorageBuffer::Vulkan(d.create_storage_buffer(desc)?)),
            Self::D3d12(d) => Ok(StorageBuffer::D3d12(d.create_storage_buffer(desc)?)),
        }
    }

    /// Register the per-frame globals uniform buffer. `slice_size` is one frame's
    /// slice (selected per-frame via [`CommandBuffer::set_globals`]). On D3D12 this
    /// is a no-op (the globals are bound as a root CBV by GPU address per draw).
    pub fn set_globals_buffer(&self, buffer: &Buffer, slice_size: u64) {
        match (self, buffer) {
            (Self::Vulkan(d), Buffer::Vulkan(b)) => d.set_globals_buffer(b, slice_size),
            (Self::D3d12(_), Buffer::D3d12(_)) => {}
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn create_texture(&self, desc: &TextureDesc, pixels: &[u8]) -> Result<Texture> {
        match self {
            Self::Vulkan(d) => Ok(Texture::Vulkan(d.create_texture(desc, pixels)?)),
            Self::D3d12(d) => Ok(Texture::D3d12(d.create_texture(desc, pixels)?)),
        }
    }

    pub fn create_depth_buffer(&self, extent: Extent2D) -> Result<DepthBuffer> {
        match self {
            Self::Vulkan(d) => Ok(DepthBuffer::Vulkan(d.create_depth_buffer(extent)?)),
            Self::D3d12(d) => Ok(DepthBuffer::D3d12(d.create_depth_buffer(extent)?)),
        }
    }

    pub fn create_render_target(&self, desc: &RenderTargetDesc) -> Result<RenderTarget> {
        match self {
            Self::Vulkan(d) => Ok(RenderTarget::Vulkan(d.create_render_target(desc)?)),
            Self::D3d12(d) => Ok(RenderTarget::D3d12(d.create_render_target(desc)?)),
        }
    }

    /// Create a render-target cubemap (6 faces, `mip_levels` each) for IBL.
    pub fn create_cubemap(&self, desc: &CubemapDesc) -> Result<Cubemap> {
        match self {
            Self::Vulkan(d) => Ok(Cubemap::Vulkan(d.create_cubemap(desc)?)),
            Self::D3d12(d) => Ok(Cubemap::D3d12(d.create_cubemap(desc)?)),
        }
    }

    /// CPU memory layout for reading a swapchain image back to the host (for
    /// screenshots). Use it to size a [`BufferUsage::Readback`] buffer and to skip
    /// per-row padding after [`CommandBuffer::copy_swapchain_to_buffer`].
    pub fn swapchain_readback_layout(&self, swapchain: &Swapchain) -> ReadbackLayout {
        match (self, swapchain) {
            (Self::Vulkan(d), Swapchain::Vulkan(s)) => d.swapchain_readback_layout(s),
            (Self::D3d12(d), Swapchain::D3d12(s)) => d.swapchain_readback_layout(s),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Memory footprint of an aliasable render target (for graph alias planning).
    pub fn render_target_memory(&self, desc: &RenderTargetDesc) -> Result<MemoryRequirements> {
        match self {
            Self::Vulkan(d) => d.render_target_memory(desc),
            Self::D3d12(d) => d.render_target_memory(desc),
        }
    }

    /// Create a transient heap of `size` bytes for aliased render targets.
    pub fn create_transient_heap(&self, size: u64) -> Result<TransientHeap> {
        match self {
            Self::Vulkan(d) => Ok(TransientHeap::Vulkan(d.create_transient_heap(size)?)),
            Self::D3d12(d) => Ok(TransientHeap::D3d12(d.create_transient_heap(size)?)),
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
            (Self::Vulkan(d), TransientHeap::Vulkan(h)) => Ok(RenderTarget::Vulkan(
                d.create_aliased_target(h, offset, desc)?,
            )),
            (Self::D3d12(d), TransientHeap::D3d12(h)) => Ok(RenderTarget::D3d12(
                d.create_aliased_target(h, offset, desc)?,
            )),
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn create_fence(&self, signaled: bool) -> Result<Fence> {
        match self {
            Self::Vulkan(d) => Ok(Fence::Vulkan(d.create_fence(signaled)?)),
            Self::D3d12(d) => Ok(Fence::D3d12(d.create_fence(signaled)?)),
        }
    }

    pub fn create_semaphore(&self) -> Result<Semaphore> {
        match self {
            Self::Vulkan(d) => Ok(Semaphore::Vulkan(d.create_semaphore()?)),
            Self::D3d12(d) => Ok(Semaphore::D3d12(d.create_semaphore()?)),
        }
    }

    pub fn queue(&self) -> Queue {
        match self {
            Self::Vulkan(d) => Queue::Vulkan(d.queue()),
            Self::D3d12(d) => Queue::D3d12(d.queue()),
        }
    }

    pub fn wait_idle(&self) -> Result<()> {
        match self {
            Self::Vulkan(d) => d.wait_idle(),
            Self::D3d12(d) => d.wait_idle(),
        }
    }

    /// The backend this device dispatches to.
    pub fn backend(&self) -> BackendKind {
        match self {
            Self::Vulkan(_) => BackendKind::Vulkan,
            Self::D3d12(_) => BackendKind::D3d12,
        }
    }
}

impl Swapchain {
    /// Acquire the next image; `Some(index)` to render, `None` if it must be
    /// recreated first.
    pub fn acquire_next_image(&self, signal: &Semaphore) -> Result<Option<u32>> {
        match (self, signal) {
            (Self::Vulkan(s), Semaphore::Vulkan(sem)) => s.acquire_next_image(sem),
            (Self::D3d12(s), Semaphore::D3d12(sem)) => s.acquire_next_image(sem),
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn recreate(&mut self, desc: &SwapchainDesc) -> Result<()> {
        match self {
            Self::Vulkan(s) => s.recreate(desc),
            Self::D3d12(s) => s.recreate(desc),
        }
    }

    pub fn format(&self) -> Format {
        match self {
            Self::Vulkan(s) => s.format(),
            Self::D3d12(s) => s.format(),
        }
    }

    pub fn extent_2d(&self) -> Extent2D {
        match self {
            Self::Vulkan(s) => s.extent_2d(),
            Self::D3d12(s) => s.extent_2d(),
        }
    }

    pub fn image_count(&self) -> u32 {
        match self {
            Self::Vulkan(s) => s.image_count(),
            Self::D3d12(s) => s.image_count(),
        }
    }
}

impl CommandBuffer {
    pub fn begin(&self) -> Result<()> {
        match self {
            Self::Vulkan(c) => c.begin(),
            Self::D3d12(c) => c.begin(),
        }
    }

    pub fn end(&self) -> Result<()> {
        match self {
            Self::Vulkan(c) => c.end(),
            Self::D3d12(c) => c.end(),
        }
    }

    pub fn transition_to_render_target(&self, swapchain: &Swapchain, image_index: u32) {
        match (self, swapchain) {
            (Self::Vulkan(c), Swapchain::Vulkan(s)) => {
                c.transition_to_render_target(s, image_index)
            }
            (Self::D3d12(c), Swapchain::D3d12(s)) => c.transition_to_render_target(s, image_index),
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn transition_to_present(&self, swapchain: &Swapchain, image_index: u32) {
        match (self, swapchain) {
            (Self::Vulkan(c), Swapchain::Vulkan(s)) => c.transition_to_present(s, image_index),
            (Self::D3d12(c), Swapchain::D3d12(s)) => c.transition_to_present(s, image_index),
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
            (Self::Vulkan(c), Swapchain::Vulkan(s), None) => {
                c.begin_rendering(s, image_index, color_clear, None)
            }
            (Self::Vulkan(c), Swapchain::Vulkan(s), Some(DepthBuffer::Vulkan(d))) => {
                c.begin_rendering(s, image_index, color_clear, Some(d))
            }
            (Self::D3d12(c), Swapchain::D3d12(s), None) => {
                c.begin_rendering(s, image_index, color_clear, None)
            }
            (Self::D3d12(c), Swapchain::D3d12(s), Some(DepthBuffer::D3d12(d))) => {
                c.begin_rendering(s, image_index, color_clear, Some(d))
            }
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
            (Self::Vulkan(c), RenderTarget::Vulkan(t), None) => {
                c.begin_rendering_target(t, color_clear, None)
            }
            (Self::Vulkan(c), RenderTarget::Vulkan(t), Some(DepthBuffer::Vulkan(d))) => {
                c.begin_rendering_target(t, color_clear, Some(d))
            }
            (Self::D3d12(c), RenderTarget::D3d12(t), None) => {
                c.begin_rendering_target(t, color_clear, None)
            }
            (Self::D3d12(c), RenderTarget::D3d12(t), Some(DepthBuffer::D3d12(d))) => {
                c.begin_rendering_target(t, color_clear, Some(d))
            }
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
        }
    }

    /// Select the per-frame globals slice for the next PBR pipeline bind. `offset`
    /// is the byte offset of this frame's slice within the globals buffer
    /// registered via [`Device::set_globals_buffer`].
    pub fn set_globals(&self, buffer: &Buffer, offset: u64) {
        match (self, buffer) {
            (Self::Vulkan(c), Buffer::Vulkan(_)) => c.set_globals(offset as u32),
            (Self::D3d12(c), Buffer::D3d12(b)) => c.set_globals(b.gpu_va() + offset),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Begin a depth-only render pass into `depth` (a shadow map): no color
    /// targets, depth cleared + stored. The depth must already be in
    /// depth-attachment state (see [`Self::depth_to_render_target`]).
    pub fn begin_rendering_depth_only(&self, depth: &DepthBuffer) {
        match (self, depth) {
            (Self::Vulkan(c), DepthBuffer::Vulkan(d)) => c.begin_rendering_depth_only(d),
            (Self::D3d12(c), DepthBuffer::D3d12(d)) => c.begin_rendering_depth_only(d),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a depth buffer into depth-attachment state for writing.
    pub fn depth_to_render_target(&self, depth: &DepthBuffer) {
        match (self, depth) {
            (Self::Vulkan(c), DepthBuffer::Vulkan(d)) => c.depth_to_render_target(d),
            (Self::D3d12(c), DepthBuffer::D3d12(d)) => c.depth_to_render_target(d),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a depth buffer into shader-read state for sampling.
    pub fn depth_to_sampled(&self, depth: &DepthBuffer) {
        match (self, depth) {
            (Self::Vulkan(c), DepthBuffer::Vulkan(d)) => c.depth_to_sampled(d),
            (Self::D3d12(c), DepthBuffer::D3d12(d)) => c.depth_to_sampled(d),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a whole cubemap into render-target state for writing its faces.
    pub fn cube_to_color(&self, cube: &Cubemap) {
        match (self, cube) {
            (Self::Vulkan(c), Cubemap::Vulkan(m)) => c.cube_to_color(m),
            (Self::D3d12(c), Cubemap::D3d12(m)) => c.cube_to_color(m),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a whole cubemap into shader-read state for sampling.
    pub fn cube_to_sampled(&self, cube: &Cubemap) {
        match (self, cube) {
            (Self::Vulkan(c), Cubemap::Vulkan(m)) => c.cube_to_sampled(m),
            (Self::D3d12(c), Cubemap::D3d12(m)) => c.cube_to_sampled(m),
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
            (Self::Vulkan(c), Cubemap::Vulkan(m)) => {
                c.begin_rendering_cube_face(m, face, mip, clear)
            }
            (Self::D3d12(c), Cubemap::D3d12(m)) => c.begin_rendering_cube_face(m, face, mip, clear),
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
            (Self::Vulkan(c), Cubemap::Vulkan(m), DepthBuffer::Vulkan(d)) => {
                c.begin_rendering_cube_face_depth(m, face, mip, clear, d)
            }
            (Self::D3d12(c), Cubemap::D3d12(m), DepthBuffer::D3d12(d)) => {
                c.begin_rendering_cube_face_depth(m, face, mip, clear, d)
            }
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn end_rendering(&self) {
        match self {
            Self::Vulkan(c) => c.end_rendering(),
            Self::D3d12(c) => c.end_rendering(),
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
            (Self::Vulkan(c), Swapchain::Vulkan(s), Buffer::Vulkan(b)) => {
                c.copy_swapchain_to_buffer(s, image_index, b)
            }
            (Self::D3d12(c), Swapchain::D3d12(s), Buffer::D3d12(b)) => {
                c.copy_swapchain_to_buffer(s, image_index, b)
            }
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn set_viewport_scissor(&self, swapchain: &Swapchain) {
        match (self, swapchain) {
            (Self::Vulkan(c), Swapchain::Vulkan(s)) => c.set_viewport_scissor(s),
            (Self::D3d12(c), Swapchain::D3d12(s)) => c.set_viewport_scissor(s),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Set viewport and scissor to cover an arbitrary extent (offscreen target).
    pub fn set_viewport_scissor_extent(&self, extent: Extent2D) {
        match self {
            Self::Vulkan(c) => c.set_viewport_scissor_extent(extent),
            Self::D3d12(c) => c.set_viewport_scissor_extent(extent),
        }
    }

    /// Transition an offscreen target into render-target state for writing.
    pub fn rt_to_render_target(&self, target: &RenderTarget) {
        match (self, target) {
            (Self::Vulkan(c), RenderTarget::Vulkan(t)) => c.rt_to_render_target(t),
            (Self::D3d12(c), RenderTarget::D3d12(t)) => c.rt_to_render_target(t),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition an offscreen target into shader-read state for sampling.
    pub fn rt_to_sampled(&self, target: &RenderTarget) {
        match (self, target) {
            (Self::Vulkan(c), RenderTarget::Vulkan(t)) => c.rt_to_sampled(t),
            (Self::D3d12(c), RenderTarget::D3d12(t)) => c.rt_to_sampled(t),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Discard an aliased target's shared memory and ready it for writing (issued
    /// before the first write of a target that reuses another's heap region).
    pub fn aliasing_barrier(&self, target: &RenderTarget) {
        match (self, target) {
            (Self::Vulkan(c), RenderTarget::Vulkan(t)) => c.aliasing_barrier(t),
            (Self::D3d12(c), RenderTarget::D3d12(t)) => c.aliasing_barrier(t),
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn bind_graphics_pipeline(&self, pipeline: &GraphicsPipeline) {
        match (self, pipeline) {
            (Self::Vulkan(c), GraphicsPipeline::Vulkan(p)) => c.bind_graphics_pipeline(p),
            (Self::D3d12(c), GraphicsPipeline::D3d12(p)) => c.bind_graphics_pipeline(p),
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn draw(&self, vertex_count: u32, instance_count: u32) {
        match self {
            Self::Vulkan(c) => c.draw(vertex_count, instance_count),
            Self::D3d12(c) => c.draw(vertex_count, instance_count),
        }
    }

    /// Bind a compute pipeline (and its bindless tables, if any).
    pub fn bind_compute_pipeline(&self, pipeline: &ComputePipeline) {
        match (self, pipeline) {
            (Self::Vulkan(c), ComputePipeline::Vulkan(p)) => c.bind_compute_pipeline(p),
            (Self::D3d12(c), ComputePipeline::D3d12(p)) => c.bind_compute_pipeline(p),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Dispatch the bound compute pipeline over `(x, y, z)` workgroups.
    pub fn dispatch(&self, x: u32, y: u32, z: u32) {
        match self {
            Self::Vulkan(c) => c.dispatch(x, y, z),
            Self::D3d12(c) => c.dispatch(x, y, z),
        }
    }

    /// Upload push/root constants for the bound **compute** pipeline.
    pub fn push_constants_compute(&self, data: &[u8]) {
        match self {
            Self::Vulkan(c) => c.push_constants_compute(data),
            Self::D3d12(c) => c.push_constants_compute(data),
        }
    }

    /// Transition a storage render target into compute-writable state.
    pub fn rt_to_storage(&self, target: &RenderTarget) {
        match (self, target) {
            (Self::Vulkan(c), RenderTarget::Vulkan(t)) => c.rt_to_storage(t),
            (Self::D3d12(c), RenderTarget::D3d12(t)) => c.rt_to_storage(t),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a storage image from compute-write into shader-read for sampling.
    pub fn storage_to_sampled(&self, target: &RenderTarget) {
        match (self, target) {
            (Self::Vulkan(c), RenderTarget::Vulkan(t)) => c.storage_to_sampled(t),
            (Self::D3d12(c), RenderTarget::D3d12(t)) => c.storage_to_sampled(t),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// UAV barrier ordering a compute write to a storage buffer before later reads.
    pub fn storage_buffer_barrier(&self, buffer: &StorageBuffer) {
        match (self, buffer) {
            (Self::Vulkan(c), StorageBuffer::Vulkan(b)) => c.storage_buffer_barrier(b),
            (Self::D3d12(c), StorageBuffer::D3d12(b)) => c.storage_buffer_barrier(b),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a storage buffer (compute write) into indirect-args state for
    /// `draw_indexed_indirect`.
    pub fn storage_buffer_to_indirect(&self, buffer: &StorageBuffer) {
        match (self, buffer) {
            (Self::Vulkan(c), StorageBuffer::Vulkan(b)) => c.storage_buffer_to_indirect(b),
            (Self::D3d12(c), StorageBuffer::D3d12(b)) => c.storage_buffer_to_indirect(b),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Transition a storage buffer back into compute-writable state (next frame).
    pub fn storage_buffer_to_storage(&self, buffer: &StorageBuffer) {
        match (self, buffer) {
            (Self::Vulkan(c), StorageBuffer::Vulkan(b)) => c.storage_buffer_to_storage(b),
            (Self::D3d12(c), StorageBuffer::D3d12(b)) => c.storage_buffer_to_storage(b),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Issue an indexed indirect draw reading args from `buffer` at `offset`
    /// (a `draw_count`-element array of `[index_count, instance_count, first_index,
    /// vertex_offset, first_instance]`).
    pub fn draw_indexed_indirect(&self, buffer: &StorageBuffer, offset: u64, draw_count: u32) {
        match (self, buffer) {
            (Self::Vulkan(c), StorageBuffer::Vulkan(b)) => {
                c.draw_indexed_indirect(b, offset, draw_count)
            }
            (Self::D3d12(c), StorageBuffer::D3d12(b)) => {
                c.draw_indexed_indirect(b, offset, draw_count)
            }
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn set_scissor(&self, rect: Rect2D) {
        match self {
            Self::Vulkan(c) => c.set_scissor(rect),
            Self::D3d12(c) => c.set_scissor(rect),
        }
    }

    pub fn bind_vertex_buffer(&self, buffer: &Buffer, stride: u32) {
        match (self, buffer) {
            (Self::Vulkan(c), Buffer::Vulkan(b)) => c.bind_vertex_buffer(b, stride),
            (Self::D3d12(c), Buffer::D3d12(b)) => c.bind_vertex_buffer(b, stride),
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn bind_index_buffer(&self, buffer: &Buffer, wide: bool) {
        match (self, buffer) {
            (Self::Vulkan(c), Buffer::Vulkan(b)) => c.bind_index_buffer(b, wide),
            (Self::D3d12(c), Buffer::D3d12(b)) => c.bind_index_buffer(b, wide),
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn push_constants(&self, data: &[u8]) {
        match self {
            Self::Vulkan(c) => c.push_constants(data),
            Self::D3d12(c) => c.push_constants(data),
        }
    }

    pub fn draw_indexed(&self, index_count: u32, first_index: u32, vertex_offset: i32) {
        match self {
            Self::Vulkan(c) => c.draw_indexed(index_count, first_index, vertex_offset),
            Self::D3d12(c) => c.draw_indexed(index_count, first_index, vertex_offset),
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
            (
                Self::Vulkan(q),
                CommandBuffer::Vulkan(c),
                Semaphore::Vulkan(w),
                Semaphore::Vulkan(s),
                Fence::Vulkan(f),
            ) => q.submit(c, w, s, f),
            (
                Self::D3d12(q),
                CommandBuffer::D3d12(c),
                Semaphore::D3d12(w),
                Semaphore::D3d12(s),
                Fence::D3d12(f),
            ) => q.submit(c, w, s, f),
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
            (
                Self::Vulkan(q),
                CommandBuffer::Vulkan(c),
                Semaphore::Vulkan(w),
                Semaphore::Vulkan(cw),
                Semaphore::Vulkan(s),
                Fence::Vulkan(f),
            ) => q.submit_async(c, w, cw, s, f),
            (
                Self::D3d12(q),
                CommandBuffer::D3d12(c),
                Semaphore::D3d12(w),
                Semaphore::D3d12(_cw),
                Semaphore::D3d12(s),
                Fence::D3d12(f),
            ) => q.submit_async(c, w, s, f),
            _ => unreachable!("{MIXED}"),
        }
    }

    /// Submit one command buffer with no semaphore sync, signaling `fence`. For
    /// one-off startup work (e.g. IBL cubemap generation).
    pub fn submit_oneshot(&self, cmd: &CommandBuffer, fence: &Fence) -> Result<()> {
        match (self, cmd, fence) {
            (Self::Vulkan(q), CommandBuffer::Vulkan(c), Fence::Vulkan(f)) => q.submit_oneshot(c, f),
            (Self::D3d12(q), CommandBuffer::D3d12(c), Fence::D3d12(f)) => q.submit_oneshot(c, f),
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
            (Self::Vulkan(q), Swapchain::Vulkan(s), Semaphore::Vulkan(w)) => {
                q.present(s, image_index, w)
            }
            (Self::D3d12(q), Swapchain::D3d12(s), Semaphore::D3d12(w)) => {
                q.present(s, image_index, w)
            }
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
            (Self::Vulkan(q), CommandBuffer::Vulkan(c), Semaphore::Vulkan(s)) => q.submit(c, s),
            (Self::D3d12(q), CommandBuffer::D3d12(c), Semaphore::D3d12(s)) => q.submit(c, s),
            _ => unreachable!("{MIXED}"),
        }
    }
}

impl Fence {
    pub fn wait(&self) -> Result<()> {
        match self {
            Self::Vulkan(f) => f.wait(),
            Self::D3d12(f) => f.wait(),
        }
    }

    pub fn reset(&self) -> Result<()> {
        match self {
            Self::Vulkan(f) => f.reset(),
            Self::D3d12(f) => f.reset(),
        }
    }
}
