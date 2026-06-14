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

use engine_core::EngineError;
use engine_platform::Window;

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
backend_enum!(/// A window swapchain.
    Swapchain => rhi_vulkan::VulkanSwapchain, rhi_d3d12::D3d12Swapchain);
backend_enum!(/// A graphics pipeline.
    GraphicsPipeline => rhi_vulkan::VulkanGraphicsPipeline, rhi_d3d12::D3d12GraphicsPipeline);
backend_enum!(/// A primary command buffer.
    CommandBuffer => rhi_vulkan::VulkanCommandBuffer, rhi_d3d12::D3d12CommandBuffer);
backend_enum!(/// A CPU-GPU fence.
    Fence => rhi_vulkan::VulkanFence, rhi_d3d12::D3d12Fence);
backend_enum!(/// A GPU-GPU binary semaphore (no-op on D3D12).
    Semaphore => rhi_vulkan::VulkanSemaphore, rhi_d3d12::D3d12Semaphore);
backend_enum!(/// A host-visible buffer (vertex/index).
    Buffer => rhi_vulkan::VulkanBuffer, rhi_d3d12::D3d12Buffer);
backend_enum!(/// A sampled 2D texture registered in the bindless table.
    Texture => rhi_vulkan::VulkanTexture, rhi_d3d12::D3d12Texture);

impl Buffer {
    /// Copy bytes into the buffer (host-visible).
    pub fn write(&self, data: &[u8]) -> Result<()> {
        match self {
            Self::Vulkan(b) => b.write(data),
            Self::D3d12(b) => b.write(data),
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

    pub fn create_command_buffer(&self) -> Result<CommandBuffer> {
        match self {
            Self::Vulkan(d) => Ok(CommandBuffer::Vulkan(d.create_command_buffer()?)),
            Self::D3d12(d) => Ok(CommandBuffer::D3d12(d.create_command_buffer()?)),
        }
    }

    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<Buffer> {
        match self {
            Self::Vulkan(d) => Ok(Buffer::Vulkan(d.create_buffer(desc)?)),
            Self::D3d12(d) => Ok(Buffer::D3d12(d.create_buffer(desc)?)),
        }
    }

    pub fn create_texture(&self, desc: &TextureDesc, pixels: &[u8]) -> Result<Texture> {
        match self {
            Self::Vulkan(d) => Ok(Texture::Vulkan(d.create_texture(desc, pixels)?)),
            Self::D3d12(d) => Ok(Texture::D3d12(d.create_texture(desc, pixels)?)),
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

    pub fn begin_rendering(&self, swapchain: &Swapchain, image_index: u32, clear: ClearColor) {
        match (self, swapchain) {
            (Self::Vulkan(c), Swapchain::Vulkan(s)) => c.begin_rendering(s, image_index, clear),
            (Self::D3d12(c), Swapchain::D3d12(s)) => c.begin_rendering(s, image_index, clear),
            _ => unreachable!("{MIXED}"),
        }
    }

    pub fn end_rendering(&self) {
        match self {
            Self::Vulkan(c) => c.end_rendering(),
            Self::D3d12(c) => c.end_rendering(),
        }
    }

    pub fn set_viewport_scissor(&self, swapchain: &Swapchain) {
        match (self, swapchain) {
            (Self::Vulkan(c), Swapchain::Vulkan(s)) => c.set_viewport_scissor(s),
            (Self::D3d12(c), Swapchain::D3d12(s)) => c.set_viewport_scissor(s),
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
