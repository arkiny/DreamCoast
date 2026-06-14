//! The engine RHI facade: enum-dispatch over graphics backends.
//!
//! Each GPU object is an `enum` with one variant per backend (`Vulkan` today;
//! `D3d12` in Phase 2). Methods match on the variant and forward to the backend
//! — no vtable, both backends compiled in, backend chosen at runtime. Consumers
//! (e.g. `sandbox`) depend only on this crate, never on a backend directly, so
//! adding D3D12 is a new variant + arms rather than a churn of call sites.
//!
//! Backend-agnostic descriptors and enums are re-exported from [`rhi_types`].

pub use rhi_types::*;

use engine_core::EngineError;
use engine_platform::Window;

type Result<T> = std::result::Result<T, EngineError>;

/// A graphics instance bound to a window surface.
pub enum Instance {
    Vulkan(rhi_vulkan::VulkanInstance),
}

/// A logical device: the factory for GPU resources.
pub enum Device {
    Vulkan(rhi_vulkan::VulkanDevice),
}

/// A submission/present queue.
pub enum Queue {
    Vulkan(rhi_vulkan::VulkanQueue),
}

/// A window swapchain.
pub enum Swapchain {
    Vulkan(rhi_vulkan::VulkanSwapchain),
}

/// A graphics pipeline.
pub enum GraphicsPipeline {
    Vulkan(rhi_vulkan::VulkanGraphicsPipeline),
}

/// A primary command buffer.
pub enum CommandBuffer {
    Vulkan(rhi_vulkan::VulkanCommandBuffer),
}

/// A CPU↔GPU fence.
pub enum Fence {
    Vulkan(rhi_vulkan::VulkanFence),
}

/// A GPU↔GPU binary semaphore.
pub enum Semaphore {
    Vulkan(rhi_vulkan::VulkanSemaphore),
}

impl Instance {
    /// Create an instance for the requested backend.
    pub fn new(backend: BackendKind, window: &Window, desc: &InstanceDesc) -> Result<Self> {
        match backend {
            BackendKind::Vulkan => Ok(Self::Vulkan(rhi_vulkan::VulkanInstance::new(window, desc)?)),
            BackendKind::D3d12 => Err(EngineError::Rhi(
                "D3D12 backend not implemented (Phase 2)".into(),
            )),
        }
    }

    /// Create a logical device.
    pub fn create_device(&self) -> Result<Device> {
        match self {
            Self::Vulkan(i) => Ok(Device::Vulkan(i.create_device()?)),
        }
    }

    /// The backend kind in use.
    pub fn backend(&self) -> BackendKind {
        match self {
            Self::Vulkan(i) => i.backend(),
        }
    }
}

impl Device {
    pub fn create_swapchain(&self, desc: &SwapchainDesc) -> Result<Swapchain> {
        match self {
            Self::Vulkan(d) => Ok(Swapchain::Vulkan(d.create_swapchain(desc)?)),
        }
    }

    pub fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc,
    ) -> Result<GraphicsPipeline> {
        match self {
            Self::Vulkan(d) => Ok(GraphicsPipeline::Vulkan(d.create_graphics_pipeline(desc)?)),
        }
    }

    pub fn create_command_buffer(&self) -> Result<CommandBuffer> {
        match self {
            Self::Vulkan(d) => Ok(CommandBuffer::Vulkan(d.create_command_buffer()?)),
        }
    }

    pub fn create_fence(&self, signaled: bool) -> Result<Fence> {
        match self {
            Self::Vulkan(d) => Ok(Fence::Vulkan(d.create_fence(signaled)?)),
        }
    }

    pub fn create_semaphore(&self) -> Result<Semaphore> {
        match self {
            Self::Vulkan(d) => Ok(Semaphore::Vulkan(d.create_semaphore()?)),
        }
    }

    pub fn queue(&self) -> Queue {
        match self {
            Self::Vulkan(d) => Queue::Vulkan(d.queue()),
        }
    }

    pub fn wait_idle(&self) -> Result<()> {
        match self {
            Self::Vulkan(d) => d.wait_idle(),
        }
    }
}

impl Swapchain {
    /// Acquire the next image; `Some(index)` to render, `None` if it must be
    /// recreated first.
    pub fn acquire_next_image(&self, signal: &Semaphore) -> Result<Option<u32>> {
        match (self, signal) {
            (Self::Vulkan(s), Semaphore::Vulkan(sem)) => s.acquire_next_image(sem),
        }
    }

    pub fn recreate(&mut self, desc: &SwapchainDesc) -> Result<()> {
        match self {
            Self::Vulkan(s) => s.recreate(desc),
        }
    }

    pub fn format(&self) -> Format {
        match self {
            Self::Vulkan(s) => s.format(),
        }
    }

    pub fn extent_2d(&self) -> Extent2D {
        match self {
            Self::Vulkan(s) => s.extent_2d(),
        }
    }

    pub fn image_count(&self) -> u32 {
        match self {
            Self::Vulkan(s) => s.image_count(),
        }
    }
}

impl CommandBuffer {
    pub fn begin(&self) -> Result<()> {
        match self {
            Self::Vulkan(c) => c.begin(),
        }
    }

    pub fn end(&self) -> Result<()> {
        match self {
            Self::Vulkan(c) => c.end(),
        }
    }

    pub fn transition_to_render_target(&self, swapchain: &Swapchain, image_index: u32) {
        match (self, swapchain) {
            (Self::Vulkan(c), Swapchain::Vulkan(s)) => {
                c.transition_to_render_target(s, image_index)
            }
        }
    }

    pub fn transition_to_present(&self, swapchain: &Swapchain, image_index: u32) {
        match (self, swapchain) {
            (Self::Vulkan(c), Swapchain::Vulkan(s)) => c.transition_to_present(s, image_index),
        }
    }

    pub fn begin_rendering(&self, swapchain: &Swapchain, image_index: u32, clear: ClearColor) {
        match (self, swapchain) {
            (Self::Vulkan(c), Swapchain::Vulkan(s)) => c.begin_rendering(s, image_index, clear),
        }
    }

    pub fn end_rendering(&self) {
        match self {
            Self::Vulkan(c) => c.end_rendering(),
        }
    }

    pub fn set_viewport_scissor(&self, swapchain: &Swapchain) {
        match (self, swapchain) {
            (Self::Vulkan(c), Swapchain::Vulkan(s)) => c.set_viewport_scissor(s),
        }
    }

    pub fn bind_graphics_pipeline(&self, pipeline: &GraphicsPipeline) {
        match (self, pipeline) {
            (Self::Vulkan(c), GraphicsPipeline::Vulkan(p)) => c.bind_graphics_pipeline(p),
        }
    }

    pub fn draw(&self, vertex_count: u32, instance_count: u32) {
        match self {
            Self::Vulkan(c) => c.draw(vertex_count, instance_count),
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
        }
    }
}

impl Fence {
    pub fn wait(&self) -> Result<()> {
        match self {
            Self::Vulkan(f) => f.wait(),
        }
    }

    pub fn reset(&self) -> Result<()> {
        match self {
            Self::Vulkan(f) => f.reset(),
        }
    }
}
