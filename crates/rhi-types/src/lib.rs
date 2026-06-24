//! Backend-agnostic RHI types: plain data shared by every backend and the
//! enum-dispatch facade.
//!
//! This crate has no dependencies (not even on a backend), which lets both the
//! backend crates and the `rhi` facade depend on it without a dependency cycle.
//! It carries only descriptors and enums — no GPU handles, no logic.

/// Which graphics backend a facade object is dispatching to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// Vulkan via `ash` (Windows).
    Vulkan,
    /// Direct3D 12 via `windows` (Phase 2, Windows).
    D3d12,
    /// Metal via `objc2-metal` (macOS).
    Metal,
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
    /// 16-bit float RGBA (HDR color, world-space normals).
    Rgba16Float,
    /// 16-bit float RG (BRDF integration LUT).
    Rg16Float,
    /// 32-bit float depth.
    Depth32Float,
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
    /// Request validation/debug layers when available. Honored only in
    /// development builds — shipping (release) builds compile validation out
    /// regardless of this flag (see the Vulkan backend's instance setup).
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

/// Vertex input layout. Phase 3 only needs the fixed ImGui layout; a general
/// layout system can come later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexLayout {
    /// No vertex buffers; the shader synthesizes vertices from the vertex index.
    None,
    /// Dear ImGui's `ImDrawVert`: position f32x2, uv f32x2, color unorm8x4 (20-byte stride).
    ImGui,
    /// Mesh vertex: position f32x3, normal f32x3, uv f32x2 (32-byte stride).
    Mesh,
    /// Mesh vertex buffer, but only the position attribute is consumed (32-byte
    /// stride). Used by the depth-only shadow pass, whose vertex shader reads
    /// just `POSITION` — declaring only what the shader consumes keeps the
    /// Vulkan validation layer quiet.
    MeshPosition,
    /// Mesh vertex buffer, position + normal consumed, uv skipped (32-byte
    /// stride). Used by the environment-capture forward pass.
    MeshPosNormal,
}

/// Color blending mode for the single color attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// Opaque (no blending).
    Opaque,
    /// Standard src-alpha / one-minus-src-alpha blending (UI).
    AlphaBlend,
}

/// Graphics pipeline parameters.
#[derive(Clone, Copy, Debug)]
pub struct GraphicsPipelineDesc<'a> {
    /// Vertex stage SPIR-V (Vulkan) / DXIL (D3D12) bytes.
    pub vertex_bytes: &'a [u8],
    /// Fragment/pixel stage bytecode.
    pub fragment_bytes: &'a [u8],
    /// Vertex entry-point name.
    pub vertex_entry: &'a str,
    /// Fragment entry-point name.
    pub fragment_entry: &'a str,
    /// Color attachment formats, in attachment order. A single element is the
    /// common case (swapchain/offscreen); multiple drive MRT (G-buffer). An
    /// empty slice is a depth-only pipeline (shadow pass).
    pub color_formats: &'a [Format],
    pub topology: PrimitiveTopology,
    /// Vertex input layout.
    pub vertex_layout: VertexLayout,
    /// Color blending.
    pub blend: BlendMode,
    /// Size in bytes of the push/root constant block (0 = none). Visible to both stages.
    pub push_constant_size: u32,
    /// Whether the pipeline binds the device's bindless texture table.
    pub bindless: bool,
    /// Whether the pipeline binds the per-frame globals uniform buffer (camera,
    /// lights, shadow, IBL). Only the deferred PBR passes opt in.
    pub uniform_buffer: bool,
    /// Enable depth test + write (compare LESS).
    pub depth_test: bool,
    /// Depth attachment format the pipeline renders against (`None` = no depth).
    pub depth_format: Option<Format>,
}

/// Compute pipeline parameters (Phase 7). A single compute stage; binds the
/// bindless table (sampled + storage) when `bindless`, plus optional push
/// constants. No vertex input, no attachments.
#[derive(Clone, Copy, Debug)]
pub struct ComputePipelineDesc<'a> {
    /// Compute stage SPIR-V (Vulkan) / DXIL (D3D12) bytes.
    pub compute_bytes: &'a [u8],
    /// Compute entry-point name.
    pub compute_entry: &'a str,
    /// Size in bytes of the push/root constant block (0 = none).
    pub push_constant_size: u32,
    /// Whether the pipeline binds the device's bindless tables.
    pub bindless: bool,
    /// Threads per threadgroup (the shader's `[numthreads(x, y, z)]`). Vulkan and
    /// D3D12 bake this into the shader, so they ignore it; Metal's MSL kernels do
    /// not, so the backend needs it to turn a `dispatch(x, y, z)` (threadgroup
    /// counts) into `dispatchThreadgroups:threadsPerThreadgroup:`.
    pub threads_per_group: [u32; 3],
}

/// A hardware ray-tracing pipeline (Phase 8 M5): one raygen, one miss, and one
/// closest-hit shader, assembled into a single shader binding table (one raygen
/// record / one miss record / one hit group). Bytes are SPIR-V (Vulkan) or a DXIL
/// library (D3D12); the three entry points come from the same `.slang` source.
pub struct RaytracingPipelineDesc<'a> {
    /// Ray-generation shader bytes (SPIR-V) / DXIL library bytes (D3D12).
    pub raygen_bytes: &'a [u8],
    pub raygen_entry: &'a str,
    /// Miss shader bytes (SPIR-V) / DXIL library bytes (D3D12).
    pub miss_bytes: &'a [u8],
    pub miss_entry: &'a str,
    /// Closest-hit shader bytes (SPIR-V) / DXIL library bytes (D3D12).
    pub closesthit_bytes: &'a [u8],
    pub closesthit_entry: &'a str,
    /// Metal Shader Converter synthesized indirect ray-dispatch kernel.
    /// Other backends ignore this optional wrapper.
    pub metal_ray_dispatch_bytes: Option<&'a [u8]>,
    pub metal_ray_dispatch_entry: Option<&'a str>,
    /// Metal Shader Converter synthesized indirect triangle-intersection function.
    /// Other backends ignore this optional wrapper.
    pub metal_intersection_bytes: Option<&'a [u8]>,
    pub metal_intersection_entry: Option<&'a str>,
    /// Size in bytes of the push/root constant block (0 = none).
    pub push_constant_size: u32,
    /// Maximum ray payload size in bytes (D3D12 shader config; Vulkan derives it).
    pub max_payload_size: u32,
    /// Maximum hit-attribute size in bytes (barycentrics = 8).
    pub max_attribute_size: u32,
}

/// Intended use of a buffer. All these buffers are host-visible (mappable) for
/// per-frame dynamic upload or host readback. GPU-local read-write storage lives
/// in the dedicated [`StorageBufferDesc`] type (Phase 7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferUsage {
    Vertex,
    Index,
    /// Per-frame globals (constant/uniform buffer).
    Uniform,
    /// GPU-writable, CPU-readable staging buffer for reading rendered images back
    /// to the host (e.g. saving a screenshot).
    Readback,
}

/// A GPU-local read-write storage buffer (UAV / `STORAGE_BUFFER`), registered in
/// the bindless storage-buffer table and addressed by index (Phase 7). Written by
/// compute and (for particles) read by the vertex stage; seeded on the GPU, not
/// from the host. `indirect` additionally allows use as a `draw_indexed_indirect`
/// argument buffer.
#[derive(Clone, Copy, Debug)]
pub struct StorageBufferDesc {
    /// Total size in bytes.
    pub size: u64,
    /// Element stride in bytes (for the structured-buffer view); the buffer holds
    /// `size / stride` elements.
    pub stride: u32,
    /// Also usable as an indirect draw-argument buffer.
    pub indirect: bool,
}

/// Buffer creation parameters.
#[derive(Clone, Copy, Debug)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: BufferUsage,
}

/// CPU-side memory layout of an image copied into a readback buffer. `row_pitch`
/// may exceed `width * 4` because backends pad rows to an alignment (D3D12 needs
/// 256-byte rows); consumers must skip the padding per row. Pixels are 8-bit
/// BGRA in the swapchain's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadbackLayout {
    pub width: u32,
    pub height: u32,
    /// Bytes per row in the readback buffer (>= `width * 4`).
    pub row_pitch: u32,
    /// Total buffer size needed, in bytes.
    pub size: u64,
}

/// Render-target cubemap creation parameters (6 faces, `mip_levels` each). Used
/// for the IBL environment map and its derived irradiance / prefilter maps.
#[derive(Clone, Copy, Debug)]
pub struct CubemapDesc {
    /// Edge length of each face at mip 0.
    pub size: u32,
    pub format: Format,
    /// Number of mip levels (e.g. roughness levels for a prefilter map).
    pub mip_levels: u32,
}

/// 2D sampled texture creation parameters.
#[derive(Clone, Copy, Debug)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub format: Format,
}

/// Offscreen color render-target creation parameters. The target is usable both
/// as a color attachment and as a bindless sampled texture (render-graph passes
/// write it, later passes sample it). When `storage` is set, it additionally gets
/// a UAV + a bindless storage-image index so a compute pass can write it (Phase 7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderTargetDesc {
    pub width: u32,
    pub height: u32,
    pub format: Format,
    /// Also create an unordered-access view (compute-writable storage image).
    pub storage: bool,
}

/// GPU memory footprint of a resource, used by the render graph to plan transient
/// aliasing (placing non-overlapping-lifetime targets at shared heap offsets).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRequirements {
    /// Required size in bytes.
    pub size: u64,
    /// Required start alignment in bytes.
    pub alignment: u64,
}

/// One bottom-level acceleration-structure (BLAS) input: a single opaque
/// triangle mesh described by its vertex/index counts and layout (Phase 8). The
/// actual vertex/index buffers are passed as backend handles at build time; this
/// carries only the plain shape data. Positions are assumed `f32x3` at byte
/// offset 0 of a `vertex_stride`-byte vertex (the engine's `Mesh` layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlasGeometry {
    /// Number of vertices in the vertex buffer.
    pub vertex_count: u32,
    /// Vertex stride in bytes (positions read from offset 0).
    pub vertex_stride: u32,
    /// Number of 32-bit indices (triangle list, so a multiple of 3).
    pub index_count: u32,
}

/// One top-level acceleration-structure (TLAS) instance (Phase 8): references a
/// BLAS (by its index in the scene's geometry list) and places it in the world
/// with a row-major 3x4 transform. `custom_index` is readable in the hit shader
/// (`InstanceID`) to look up per-instance geometry/material; `mask` is the ray
/// visibility mask (0xFF = visible to all rays).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TlasInstance {
    /// Index into the scene's geometry/BLAS list.
    pub blas_index: u32,
    /// Row-major 3x4 object-to-world transform (12 floats).
    pub transform: [f32; 12],
    /// 24-bit value exposed to the hit shader as `InstanceID`/instance custom index.
    pub custom_index: u32,
    /// 8-bit ray visibility mask.
    pub mask: u8,
}

/// GPU memory footprint of an acceleration structure plus the scratch buffer its
/// build needs, returned by the backend's prebuild query (Phase 8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccelSizes {
    /// Bytes for the acceleration-structure buffer.
    pub accel_size: u64,
    /// Bytes for the transient build scratch buffer.
    pub scratch_size: u64,
}

/// A scissor / sub-rect in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect2D {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}
