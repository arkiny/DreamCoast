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

/// Backend-agnostic GPU device identity, surfaced up through the facade so app code can pick a
/// platform-appropriate default quality tier (macOS/M3 perf, axis A). Additive + read-only: it
/// carries no GPU handles, so exposing it does not change any backend's rendering behavior.
///
/// `name` is the raw adapter string (e.g. `"Apple M3"`, `"NVIDIA GeForce RTX 2070 SUPER"`).
/// `unified_memory` / `low_power` are cheap capability flags where the backend exposes them (Metal
/// `hasUnifiedMemory` / `isLowPower`); backends that don't expose them report `false`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct DeviceInfo {
    /// Adapter / device name string, as reported by the driver.
    pub name: String,
    /// Whether the GPU shares memory with the CPU (Apple Silicon / most iGPUs). `false` if unknown.
    pub unified_memory: bool,
    /// Whether the driver flags this as a low-power / integrated GPU. `false` if unknown.
    pub low_power: bool,
}

impl DeviceInfo {
    /// Heuristic: does this look like an Apple GPU? Matches the vendor name substring (`"Apple"`,
    /// e.g. `"Apple M3"`) as the primary signal, with the unified-memory flag as a secondary hint.
    /// Used only to select an aggressive Apple default quality tier; non-Apple / unknown stays on
    /// the honest `Med` fallback. Case-insensitive so a driver-formatting change can't defeat it.
    pub fn is_apple_gpu(&self) -> bool {
        self.name.to_ascii_lowercase().contains("apple")
    }
}

/// Optional GPU capabilities a backend/adapter may or may not expose, probed once at
/// device creation. Phase 14 (virtual geometry) is the first consumer: its full path
/// needs a mesh-shader pipeline, 64-bit buffer atomics (the visibility-buffer `atomicMax`),
/// and indirect compute dispatch — none of which the earlier phases required, and each of
/// which a given adapter can lack. Features gate opt-in code paths that must **fail
/// clearly** (not silently mis-render) when unsupported; nothing in the default render
/// path reads these, so a `false` here never affects an existing scene.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceCapabilities {
    /// Mesh/task (object) shader pipelines (`draw_mesh_tasks`). Metal: Apple7+/Metal 3;
    /// Vulkan: `VK_EXT_mesh_shader`; D3D12: SM6.5 mesh shaders.
    pub mesh_shader: bool,
    /// 64-bit buffer atomics (`InterlockedMax64` / `atomic_max_explicit`). Metal: Apple8+;
    /// Vulkan: `shaderBufferInt64Atomics`; D3D12: SM6.6 64-bit atomics.
    pub atomic_int64: bool,
    /// Indirect compute dispatch (`dispatch_indirect`). Metal:
    /// `dispatchThreadgroupsWithIndirectBuffer`; Vulkan: `vkCmdDispatchIndirect`;
    /// D3D12: `ExecuteIndirect` over a DISPATCH signature. Near-universal, tracked for symmetry.
    pub dispatch_indirect: bool,
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
    /// 32-bit float R (single-channel; signed distance fields, Phase 11).
    R32Float,
    /// 32-bit float depth.
    Depth32Float,
    /// BC1 (DXT1) block-compressed RGB, sRGB. 8 bytes / 4×4 block (Phase 12 M3).
    Bc1Srgb,
    /// BC1 (DXT1) block-compressed RGB, unorm. 8 bytes / 4×4 block.
    Bc1Unorm,
    /// BC5 block-compressed two-channel (RG) unorm, for normals. 16 bytes / 4×4
    /// block.
    Bc5Unorm,
    /// BC3 (DXT5) block-compressed RGBA, sRGB. 16 bytes / 4×4 block (colour + alpha).
    Bc3Srgb,
    /// BC3 (DXT5) block-compressed RGBA, unorm. 16 bytes / 4×4 block.
    Bc3Unorm,
    /// BC4 block-compressed single-channel (R) unorm. 8 bytes / 4×4 block.
    Bc4Unorm,
    /// BC7 block-compressed RGBA, sRGB (high quality). 16 bytes / 4×4 block.
    Bc7Srgb,
    /// BC7 block-compressed RGBA, unorm. 16 bytes / 4×4 block.
    Bc7Unorm,
}

impl Format {
    /// True for sRGB-encoded color formats (their RGB channels must be downsampled
    /// in linear space).
    pub fn is_srgb(self) -> bool {
        matches!(
            self,
            Format::Bgra8Srgb
                | Format::Rgba8Srgb
                | Format::Bc1Srgb
                | Format::Bc3Srgb
                | Format::Bc7Srgb
        )
    }

    /// Compressed bytes per 4×4 block for a block-compressed format, else `None`
    /// (an uncompressed format). The block edge is always 4 texels.
    pub fn block_bytes(self) -> Option<usize> {
        match self {
            Format::Bc1Srgb | Format::Bc1Unorm | Format::Bc4Unorm => Some(8),
            Format::Bc5Unorm
            | Format::Bc3Srgb
            | Format::Bc3Unorm
            | Format::Bc7Srgb
            | Format::Bc7Unorm => Some(16),
            _ => None,
        }
    }

    /// Whether this is a block-compressed (BCn) format.
    pub fn is_block_compressed(self) -> bool {
        self.block_bytes().is_some()
    }

    /// Row pitch (bytes) and row count for one mip of `width×height` in this format:
    /// for BCn a "row" is a row of 4×4 blocks; for uncompressed it is a pixel row.
    /// Used to size staging copies. Assumes 4 bytes/pixel for uncompressed (the only
    /// uncompressed textures uploaded via `create_texture`).
    pub fn upload_pitch(self, width: u32, height: u32) -> (usize, usize) {
        match self.block_bytes() {
            Some(bb) => (width.div_ceil(4) as usize * bb, height.div_ceil(4) as usize),
            None => (width as usize * 4, height as usize),
        }
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Box-downsample a tightly-packed 8-bit/4-channel `mip0` image into a full mip
/// chain (levels `0..floor(log2(max(w,h)))+1`; level 0 is a copy of the input).
///
/// Shared by every backend's `create_texture` so the generated mips are *identical*
/// across Vulkan / D3D12 / Metal (the engine's hard cross-backend-parity rule — a
/// GPU `generateMipmaps` would differ per driver). For sRGB formats the RGB channels
/// are averaged in linear space (alpha stays linear) so mips don't darken.
pub fn generate_mip_chain(mip0: &[u8], width: u32, height: u32, format: Format) -> Vec<Vec<u8>> {
    let srgb = format.is_srgb();
    let mut levels = vec![mip0.to_vec()];
    let (mut w, mut h) = (width.max(1), height.max(1));
    let mut src = mip0.to_vec();
    while w > 1 || h > 1 {
        let nw = (w / 2).max(1);
        let nh = (h / 2).max(1);
        let mut dst = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let x0 = (x * 2).min(w - 1);
                let x1 = (x * 2 + 1).min(w - 1);
                let y0 = (y * 2).min(h - 1);
                let y1 = (y * 2 + 1).min(h - 1);
                for c in 0..4u32 {
                    let fetch = |px: u32, py: u32| -> f32 {
                        let v = src[((py * w + px) * 4 + c) as usize] as f32 / 255.0;
                        if srgb && c < 3 { srgb_to_linear(v) } else { v }
                    };
                    let avg =
                        (fetch(x0, y0) + fetch(x1, y0) + fetch(x0, y1) + fetch(x1, y1)) * 0.25;
                    let out = if srgb && c < 3 {
                        linear_to_srgb(avg)
                    } else {
                        avg
                    };
                    dst[((y * nw + x) * 4 + c) as usize] =
                        (out * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
        levels.push(dst.clone());
        src = dst;
        w = nw;
        h = nh;
    }
    levels
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
    /// Mesh vertex buffer, position + uv consumed, normal skipped (32-byte
    /// stride). The shadow pass VS reads `POSITION` for depth and `TEXCOORD` for
    /// alpha-cutout but omits `NORMAL` from its input struct, so each backend's
    /// input signature is just those two (uv packs to SPIR-V location 1, no gap).
    /// Declaring only what the shader consumes keeps the Vulkan validation layer
    /// quiet (no "attribute not consumed" warning).
    MeshPositionUv,
}

/// Color blending mode. `Opaque`/`AlphaBlend` apply uniformly to every color attachment;
/// `DecalAlbedo` is a per-attachment **G-buffer decal** preset (see its docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// Opaque (no blending), write all channels.
    Opaque,
    /// Standard src-alpha / one-minus-src-alpha blending on every attachment (UI).
    AlphaBlend,
    /// Deferred surface-decal preset for the deferred **G-buffer** MRT layout
    /// (`RT0` albedo+AO, `RT1` normal, `RT2` material[r metallic, g roughness, b AO],
    /// `RT3` world-pos): attachment **0** alpha-blends its RGB into the albedo with a write
    /// mask of **RGB only** (so the baked AO in `RT0.a` is preserved), attachment **2**
    /// alpha-blends the decal roughness into **G only** (a dusty decal raises the covered
    /// surface's roughness; metallic `r` + AO `b` stay the surface's — A4), and `RT1`/`RT3`
    /// are write-masked **off**. So the decal only tints albedo and nudges roughness; it
    /// never overwrites metallic / normal / world-pos (the Intel Sponza `dirt_decal` fix).
    /// Only meaningful for a pipeline rendering the G-buffer MRT set.
    DecalAlbedo,
}

/// Depth-comparison function for a depth-testing pipeline. Kept separate from
/// `depth_test`/`depth_write` so a pass can pick how it compares against the bound
/// depth (the load behaviour is a graph attachment property; this is pipeline state).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DepthCompare {
    /// Standard closer-passes test (Vulkan/D3D12 `LESS`; Metal `LessEqual` — the
    /// pre-existing per-backend mapping, kept so every current pipeline stays
    /// byte-identical). This is the default for all opaque geometry.
    #[default]
    Less,
    /// `EQUAL` on every backend: the depth-pre-pass base-pass mode — the G-buffer
    /// fill only shades fragments whose depth equals the pre-pass depth already in
    /// the buffer (Early-Z overdraw elimination). Requires the pre-pass and base
    /// pass to compute clip-space position with the identical instruction sequence
    /// (same `mvp`, same shader math) so the depths match bit-exactly.
    Equal,
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
    /// Enable depth testing (compare LESS on Vulkan/D3D12, LESS_EQUAL on Metal).
    pub depth_test: bool,
    /// Enable depth writes. Normally equal to `depth_test`; a deferred **decal** pass sets
    /// `depth_test: true, depth_write: false` so it is occluded by closer geometry but does
    /// not perturb the opaque depth buffer that downstream passes read.
    pub depth_write: bool,
    /// Depth-comparison function (only consulted when `depth_test`). `Less` (default)
    /// keeps every existing pipeline unchanged; `Equal` is the depth-pre-pass base-pass
    /// mode (shade only fragments matching the pre-pass depth).
    pub depth_compare: DepthCompare,
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
    /// Whether the pipeline binds the per-frame globals uniform buffer (set 1 / b1),
    /// the same one the deferred PBR passes use. Lets a compute pass read structured
    /// per-frame camera data (e.g. the reflection reprojection matrices) instead of
    /// overflowing the push-constant budget. Bound via [`CommandBuffer::set_globals`].
    pub uniform_buffer: bool,
    /// Threads per threadgroup (the shader's `[numthreads(x, y, z)]`). Vulkan and
    /// D3D12 bake this into the shader, so they ignore it; Metal's MSL kernels do
    /// not, so the backend needs it to turn a `dispatch(x, y, z)` (threadgroup
    /// counts) into `dispatchThreadgroups:threadsPerThreadgroup:`.
    pub threads_per_group: [u32; 3],
}

/// A mesh-shader pipeline (Phase 14 virtual geometry): an optional task/object stage that
/// amplifies into a mesh stage which emits primitives, then the fragment stage. Drawn with
/// [`CommandBuffer::draw_mesh_tasks`] instead of a vertex draw. Requires
/// [`DeviceCapabilities::mesh_shader`]. M0 exercises the mesh+fragment path (one hardcoded
/// triangle, no object stage, no bindless); the object stage + per-cluster payload arrive in
/// M2. `object_threads` / `mesh_threads` are the per-stage `[numthreads]` — Metal needs them
/// at draw time (like compute); Vulkan/D3D12 bake them into the shader and ignore them.
#[derive(Clone, Copy, Debug)]
pub struct MeshPipelineDesc<'a> {
    /// Task/object stage bytecode, or `None` for a mesh-only pipeline (M0).
    pub object_bytes: Option<&'a [u8]>,
    /// Task/object entry-point name (ignored when `object_bytes` is `None`).
    pub object_entry: &'a str,
    /// Mesh stage bytecode.
    pub mesh_bytes: &'a [u8],
    /// Mesh entry-point name.
    pub mesh_entry: &'a str,
    /// Fragment/pixel stage bytecode.
    pub fragment_bytes: &'a [u8],
    /// Fragment entry-point name.
    pub fragment_entry: &'a str,
    /// Color attachment formats, in attachment order.
    pub color_formats: &'a [Format],
    /// Depth attachment format (`None` = no depth).
    pub depth_format: Option<Format>,
    /// Size in bytes of the push/root constant block (0 = none). Visible to all stages.
    pub push_constant_size: u32,
    /// Whether the pipeline binds the device's bindless table (mesh/object stages).
    pub bindless: bool,
    /// Whether the pipeline binds the per-frame globals UBO.
    pub uniform_buffer: bool,
    /// Object (task) stage `[numthreads]`; `[1,1,1]` when there is no object stage.
    pub object_threads: [u32; 3],
    /// Mesh stage `[numthreads]`.
    pub mesh_threads: [u32; 3],
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

/// A 3D (volume) texture, created as both a UAV (compute-writable storage volume,
/// bindless `storage_volumes[]`) and an SRV (trilinear-sampled `volumes[]`) so a
/// compute pass can bake into it and a later pass can ray-march/sample it (Phase 11
/// Stage B distance fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VolumeDesc {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: Format,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_chain_count_and_sizes() {
        // 4x4 RGBA8 -> levels 4x4, 2x2, 1x1 (floor(log2(4))+1 = 3).
        let mip0 = vec![0u8; 4 * 4 * 4];
        let levels = generate_mip_chain(&mip0, 4, 4, Format::Rgba8Unorm);
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0].len(), 4 * 4 * 4);
        assert_eq!(levels[1].len(), 2 * 2 * 4);
        assert_eq!(levels[2].len(), 4);
    }

    #[test]
    fn mip_chain_box_average_unorm() {
        // 2x2 with values 0,100,200,255 in R -> mip1 R = round(mean) = 139.
        let mut mip0 = vec![0u8; 2 * 2 * 4];
        mip0[0] = 0; // (0,0) R
        mip0[4] = 100; // (1,0) R
        mip0[8] = 200; // (0,1) R
        mip0[12] = 255; // (1,1) R
        let levels = generate_mip_chain(&mip0, 2, 2, Format::Rgba8Unorm);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[1][0], 139); // (0+100+200+255)/4 = 138.75 -> 139
    }

    #[test]
    fn mip_chain_srgb_averages_in_linear() {
        // Averaging 0 and 255 in sRGB space differs from naive byte mean (127/128).
        // Linear average of black+white sRGB is ~188 after re-encoding.
        let mut mip0 = vec![0u8; 2 * 2 * 4];
        for px in 0..4 {
            let v = if px % 2 == 0 { 0 } else { 255 };
            for c in 0..3 {
                mip0[px * 4 + c] = v;
            }
            mip0[px * 4 + 3] = 255;
        }
        let levels = generate_mip_chain(&mip0, 2, 2, Format::Rgba8Srgb);
        // Two black + two white pixels: linear mean 0.5 -> sRGB ~0.7353 -> ~188.
        assert!(
            levels[1][0] >= 185 && levels[1][0] <= 192,
            "got {}",
            levels[1][0]
        );
    }
}
