//! Backend-agnostic RHI types: plain data shared by every backend and the
//! enum-dispatch facade.
//!
//! This crate has no dependencies (not even on a backend), which lets both the
//! backend crates and the `rhi` facade depend on it without a dependency cycle.
//! It carries only descriptors and enums — no GPU handles, no logic.

/// Which graphics backend a facade object is dispatching to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// Vulkan via `ash`.
    Vulkan,
    /// Direct3D 12 via `windows` (Phase 2).
    D3d12,
}

/// A 2D size in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Extent2D {
    pub width: u32,
    pub height: u32,
}

impl Extent2D {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// Pixel formats used by the minimal Phase 1 slice. Extended as later phases
/// need more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// 8-bit BGRA, unorm.
    Bgra8Unorm,
    /// 8-bit BGRA, sRGB.
    Bgra8Srgb,
    /// 8-bit RGBA, unorm.
    Rgba8Unorm,
    /// 8-bit RGBA, sRGB.
    Rgba8Srgb,
}

/// Swapchain present/pacing mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentMode {
    /// VSync, always supported (Vulkan FIFO).
    Fifo,
    /// Low-latency triple buffering when available.
    Mailbox,
    /// No VSync; may tear.
    Immediate,
}

/// Primitive assembly topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveTopology {
    TriangleList,
}

/// An RGBA clear color (linear, 0..1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl ClearColor {
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
}

/// Instance/device creation options.
#[derive(Clone, Debug)]
pub struct InstanceDesc {
    /// Application name reported to the API.
    pub app_name: String,
    /// Request validation/debug layers when available.
    pub validation: bool,
}

impl Default for InstanceDesc {
    fn default() -> Self {
        Self {
            app_name: "engine".to_string(),
            validation: true,
        }
    }
}

/// Swapchain creation parameters.
#[derive(Clone, Copy, Debug)]
pub struct SwapchainDesc {
    pub extent: Extent2D,
    pub format: Format,
    pub present_mode: PresentMode,
    /// Desired image count; the backend clamps to driver limits.
    pub image_count: u32,
}

/// Graphics pipeline parameters for the minimal slice (no vertex input; the
/// shader synthesizes vertices from the vertex index).
#[derive(Clone, Copy, Debug)]
pub struct GraphicsPipelineDesc<'a> {
    /// Vertex stage SPIR-V (Vulkan) / DXIL (D3D12, later) bytes.
    pub vertex_bytes: &'a [u8],
    /// Fragment/pixel stage bytecode.
    pub fragment_bytes: &'a [u8],
    /// Vertex entry-point name.
    pub vertex_entry: &'a str,
    /// Fragment entry-point name.
    pub fragment_entry: &'a str,
    /// Color attachment format (matches the swapchain).
    pub color_format: Format,
    pub topology: PrimitiveTopology,
}
