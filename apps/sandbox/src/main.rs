//! Sandbox: the playground executable.
//!
//! Builds a deferred-PBR render graph each frame: a G-buffer fill pass
//! rasterizes the glTF mesh (or procedural cube fallback) into four MRT targets
//! (albedo, world normal, metallic/roughness/AO, world position) with depth; a
//! full-screen lighting pass reads the G-buffer and shades it with a
//! Cook-Torrance BRDF (one directional sun + a few point lights) into a linear
//! HDR target; a tonemap pass maps that to the backbuffer; and a Dear ImGui
//! overlay exposes the lighting controls and a G-buffer debug view. Runs on
//! either backend (`--backend vulkan|d3d12`).

use std::time::Instant;

use dreamcoast_asset::MeshData;
use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_core::init_logging;
use dreamcoast_gui::{Gui, imgui};
use dreamcoast_platform::Window;
use dreamcoast_render::{ComputePassInfo, GraphProfiler, PassInfo, RenderGraph, ResourcePool};
use rhi::{
    BackendKind, BlasGeometry, BlendMode, Buffer, BufferDesc, BufferUsage, ClearColor,
    ComputePipelineDesc, CubemapDesc, Extent2D, Format, GraphicsPipelineDesc, Instance,
    InstanceDesc, PresentMode, PrimitiveTopology, RaytracingPipelineDesc, RtGeometry,
    StorageBufferDesc, SwapchainDesc, Texture, TlasInstance, VertexLayout,
};
use tracing::info;

mod app;
mod cull;
mod gdf;
mod ibl;
mod mesh;
mod particle;
mod push;
mod smoketest;
use app::*;
use cull::*;
use gdf::*;
use ibl::*;
use mesh::*;
use particle::*;
use push::*;
use smoketest::*;

const FRAMES_IN_FLIGHT: usize = 2;
// Swapchain / backbuffer. UNORM, not sRGB: GPU capture & overlay layers
// (RenderDoc, OBS, …) force VK_IMAGE_USAGE_STORAGE onto swapchain images, which an
// sRGB surface format can never support — an sRGB backbuffer makes RenderDoc
// hard-crash at swapchain creation. The final passes encode sRGB in-shader
// instead (see `linear_to_srgb` in bindless.slang).
const COLOR_FORMAT: Format = Format::Bgra8Unorm;
const DEPTH_FORMAT: Format = Format::Depth32Float;
const HDR_FORMAT: Format = Format::Rgba16Float; // linear HDR lighting target
// G-buffer attachment formats.
const GB_ALBEDO_FMT: Format = Format::Rgba8Unorm;
const GB_NORMAL_FMT: Format = Format::Rgba16Float;
const GB_MATERIAL_FMT: Format = Format::Rgba8Unorm;
const GB_POSITION_FMT: Format = Format::Rgba16Float;
/// Per-frame globals slice size (256-byte aligned for D3D12 root CBV / Vulkan
/// dynamic UBO offset). 512 holds the lighting globals plus the light
/// view-projection matrix.
const GLOBALS_SLICE: u64 = 512;
/// Directional shadow map resolution (square).
const SHADOW_SIZE: u32 = 2048;
/// GPU particle count (Phase 7 fountain demo); 32 bytes each.
const PARTICLE_COUNT: usize = 4096;
/// GPU-culling instance grid: `GRID_DIM x GRID_DIM` cube instances (Phase 7).
const GRID_DIM: u32 = 16;
const GRID_COUNT: u32 = GRID_DIM * GRID_DIM;
/// Environment cubemap face resolution (square) and its full mip chain. The
/// prefilter convolution samples these mips so a roughness lookup needs only
/// ~32-64 samples instead of hundreds.
const ENV_SIZE: u32 = 256;
const ENV_MIPS: u32 = ENV_SIZE.ilog2() + 1;
/// Diffuse irradiance cubemap face resolution (small — it is very low frequency,
/// and kept cheap for per-frame real-time capture).
const IRRADIANCE_SIZE: u32 = 16;
/// Specular prefilter cubemap face resolution (mip 0) and roughness mip count.
const PREFILTER_SIZE: u32 = 128;
const PREFILTER_MIPS: u32 = 5;
/// BRDF integration LUT resolution.
const BRDF_SIZE: u32 = 256;
/// Sentinel for "this material texture is absent — use the scalar factor".
const NO_TEXTURE: u32 = u32::MAX;
const MODEL_PATH: &str = "assets/model.glb";

const DEBUG_VIEWS: [&str; 9] = [
    "Lit",
    "Albedo",
    "Normal",
    "Metallic",
    "Roughness",
    "Position",
    "AO",
    "Direct",
    "IBL",
];
const POST_EFFECTS: [&str; 3] = ["None", "Grayscale", "Vignette"];

/// One drawable in the sample scene: a mesh + world transform + PBR material.
pub(crate) struct SceneObject {
    pub(crate) vbuf: Buffer,
    pub(crate) ibuf: Buffer,
    pub(crate) index_count: u32,
    /// Vertex count (for the BLAS build's `max_vertex`, Phase 8).
    pub(crate) vertex_count: u32,
    pub(crate) transform: Mat4,
    pub(crate) base_color: [f32; 4],
    pub(crate) metallic: f32,
    pub(crate) roughness: f32,
    /// base color, metallic-roughness, normal, emissive bindless indices
    /// (`NO_TEXTURE` if absent).
    pub(crate) tex: [u32; 4],
    pub(crate) casts_shadow: bool,
}

/// Per-frame globals, mirrored by `Globals` in pbr.slang. All members are 16-byte
/// vectors so the std140 (Vulkan) and cbuffer (D3D12) layouts agree.
#[repr(C)]
#[derive(Clone, Copy)]
struct Globals {
    camera_pos: [f32; 4],
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    ambient: [f32; 4],
    counts: [i32; 4], // x point count, y debug view, w shadows enabled
    point_pos: [[f32; 4]; 4],
    point_color: [[f32; 4]; 4],
    light_view_proj: [f32; 16], // world -> light clip (shadow lookup)
    shadow: [f32; 4],           // x depth bias, y texel size (1 / SHADOW_SIZE)
    inv_view_proj: [f32; 16],   // clip -> world (skybox ray reconstruction)
    ibl: [i32; 4],              // x env, y irradiance, z prefilter, w BRDF (-1 = none)
    probe: [f32; 4],            // xyz reflection-probe capture centre, w parallax on (1) / off (0)
    probe_box_min: [f32; 4],    // xyz reflection proxy AABB min corner
    probe_box_max: [f32; 4],    // xyz reflection proxy AABB max corner
}

fn swapchain_desc(extent: Extent2D) -> SwapchainDesc {
    SwapchainDesc {
        extent,
        format: COLOR_FORMAT,
        present_mode: PresentMode::Fifo,
        image_count: 3,
    }
}

fn main() -> anyhow::Result<()> {
    // `--log-file <path>` mirrors logs to a file (the logging layer reads the env
    // var). It's a CLI flag, not just the env var, because GPU capture launchers
    // (RenderDoc's UI env editor) mangle environment values but pass command-line
    // arguments through cleanly. SAFE: set once at startup before any threads.
    if let Some(path) = log_file_path() {
        unsafe { std::env::set_var("DREAMCOAST_LOG_FILE", path) };
    }
    init_logging();
    // Log any fatal error before it propagates: under a GPU capture tool
    // (RenderDoc) stdout/stderr are redirected away, so a bare `Err` return would
    // vanish. With `DREAMCOAST_LOG_FILE` set this lands the real cause in the file.
    let result = run();
    if let Err(e) = &result {
        tracing::error!("fatal: {e:?}");
    }
    result
}

fn run() -> anyhow::Result<()> {
    let backend = select_backend();
    info!("requested backend: {backend:?}");

    // Screenshot mode: `--screenshot[/-clean] <path>` renders a few frames then
    // captures + exits; otherwise F2 captures interactively.
    let captures = screenshot_captures();
    let screenshot_mode = !captures.is_empty();
    const VK_F2: u16 = 0x71;
    const SCREENSHOT_WARMUP: u64 = 3;

    // Load a glTF model if present, else fall back to a procedural cube.
    let model_path = model_path();
    let mut model = match dreamcoast_asset::load_gltf(&model_path) {
        Ok(m) => {
            info!(
                "loaded {model_path}: {} verts, {} indices",
                m.vertices.len(),
                m.indices.len()
            );
            m
        }
        Err(e) => {
            info!("no glTF at {model_path} ({e}); using procedural cube");
            dreamcoast_asset::unit_cube()
        }
    };
    // Normalize the model (recenter on origin, base at y=0, unit bounding radius)
    // so framing, ground, lights, and the shadow box use model-independent units.
    let bounds = normalize_on_ground(&mut model);
    let model_radius = bounds.radius;

    let title = format!("DreamCoast Sandbox — {backend:?}");
    let mut window = Window::new(&title, 1280, 720)?;
    let (w, h) = window.size();

    // Validation is a launch-time choice (instance-level): on by default, off via
    // `--no-validation`. In release builds the backend compiles validation out
    // regardless, so `validation_on` is only meaningful in debug builds.
    let validation_on = validation_enabled();
    let instance = Instance::new(
        backend,
        &window,
        &InstanceDesc {
            app_name: "dreamcoast-sandbox".into(),
            validation: validation_on,
        },
    )?;
    let device = instance.create_device()?;
    let queue = device.queue();
    info!(
        "device capabilities: async_compute={}, raytracing={}",
        device.has_async_compute(),
        device.has_raytracing()
    );
    // Phase-7 compute (post blur / GPU particles / GPU culling) is implemented on
    // all three backends (Metal compute landed in M5).
    let compute_supported = true;

    let mut swapchain = device.create_swapchain(&swapchain_desc(Extent2D::new(w, h)))?;

    // M0 backend bring-up: a minimal acquire→clear→present loop that needs no
    // pipelines or shaders. The Metal backend defaults through this until the
    // triangle/PBR milestones implement pipelines.
    if clear_test_enabled() {
        return run_clear_test(&mut window, &device, &mut swapchain);
    }

    // M2 backend bring-up: clear + a single hardcoded-triangle pipeline (no vertex
    // buffers, push constants, or bindless). Validates pipeline creation + draw.
    if triangle_test_enabled() {
        return run_triangle_test(backend, &mut window, &device, &mut swapchain);
    }

    // M3 backend bring-up: textured bindless mesh (depth-tested) + an ImGui overlay.
    // Exercises the bindless argument buffer, sampled textures, depth, and ImGui on
    // the Metal backend; cross-backend like the other *_test loops.
    if mesh_test_enabled() {
        return run_mesh_test(
            backend,
            &mut window,
            &device,
            &mut swapchain,
            &model,
            model_radius,
        );
    }

    // G-buffer fill pipeline: mesh -> 4 MRT (+ depth).
    let (gb_vs, gb_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::gbuffer_vs_spirv,
        dreamcoast_shader::gbuffer_fs_spirv,
        dreamcoast_shader::gbuffer_vs_dxil,
        dreamcoast_shader::gbuffer_fs_dxil,
        dreamcoast_shader::gbuffer_vs_metallib,
        dreamcoast_shader::gbuffer_fs_metallib,
        "gbuffer",
    )?;
    let gbuffer_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: gb_vs,
        fragment_bytes: gb_fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_formats: &[
            GB_ALBEDO_FMT,
            GB_NORMAL_FMT,
            GB_MATERIAL_FMT,
            GB_POSITION_FMT,
        ],
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::Mesh,
        blend: BlendMode::Opaque,
        push_constant_size: 112, // mat4(64) + base_color(16) + mr(16) + tex u32x4(16)
        bindless: true,
        uniform_buffer: false,
        depth_test: true,
        depth_format: Some(DEPTH_FORMAT),
    })?;

    // Shadow pipeline: depth-only, rasterizes the mesh from the light's POV.
    let (shadow_vs, shadow_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::shadow_vs_spirv,
        dreamcoast_shader::shadow_fs_spirv,
        dreamcoast_shader::shadow_vs_dxil,
        dreamcoast_shader::shadow_fs_dxil,
        dreamcoast_shader::shadow_vs_metallib,
        dreamcoast_shader::shadow_fs_metallib,
        "shadow",
    )?;
    let shadow_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: shadow_vs,
        fragment_bytes: shadow_fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_formats: &[], // depth-only
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::MeshPosition,
        blend: BlendMode::Opaque,
        push_constant_size: 64, // light_mvp mat4
        bindless: true,         // for the root-constants param (push constants)
        uniform_buffer: false,
        depth_test: true,
        depth_format: Some(DEPTH_FORMAT),
    })?;

    // Deferred lighting pipeline: full-screen, reads G-buffer + globals -> HDR.
    let (pbr_vs, pbr_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::pbr_vs_spirv,
        dreamcoast_shader::pbr_fs_spirv,
        dreamcoast_shader::pbr_vs_dxil,
        dreamcoast_shader::pbr_fs_dxil,
        dreamcoast_shader::pbr_vs_metallib,
        dreamcoast_shader::pbr_fs_metallib,
        "pbr",
    )?;
    let pbr_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: pbr_vs,
        fragment_bytes: pbr_fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_formats: &[HDR_FORMAT],
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::None,
        blend: BlendMode::Opaque,
        push_constant_size: 24, // 4 G-buffer indices + flip_y + shadow_index
        bindless: true,
        uniform_buffer: true,
        depth_test: false,
        depth_format: None,
    })?;

    // Tonemap pipeline: HDR -> backbuffer (encodes sRGB in-shader; UNORM target).
    let (post_vs, post_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::post_vs_spirv,
        dreamcoast_shader::post_fs_spirv,
        dreamcoast_shader::post_vs_dxil,
        dreamcoast_shader::post_fs_dxil,
        dreamcoast_shader::post_vs_metallib,
        dreamcoast_shader::post_fs_metallib,
        "post",
    )?;
    let post_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: post_vs,
        fragment_bytes: post_fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_formats: &[swapchain.format()],
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::None,
        blend: BlendMode::Opaque,
        push_constant_size: 16, // hdr_index + mode + flip_y + pad
        bindless: true,
        uniform_buffer: false,
        depth_test: false,
        depth_format: None,
    })?;
    // Compute post-process pipeline (Phase 7): blurs the HDR target into a
    // storage image between the lighting and tonemap passes.
    let post_compute_cs = load_compute_shader(
        backend,
        dreamcoast_shader::post_compute_cs_spirv,
        dreamcoast_shader::post_compute_cs_dxil,
        dreamcoast_shader::post_compute_cs_metallib,
        "post_compute",
    )?;
    let post_compute_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
        compute_bytes: post_compute_cs,
        compute_entry: "csMain",
        push_constant_size: 16, // hdr_index + out_index + width + height
        bindless: true,
        threads_per_group: [8, 8, 1],
    })?;

    // Phase 11 software ray tracing + global distance field (Stage A analytic trace,
    // Stage B volumes / SDF bake / GDF merge / GDF trace). See `gdf.rs`.
    let gdf = GdfSystem::new(&device, backend, compute_supported)?;

    // Phase 8 M3: inline ray-query trace pipeline (compute + `RayQuery`). Only on
    // RT-capable devices; the bindless block then carries the scene TLAS (binding
    // 5 / `t1088,space1`).
    let rt_trace_pipeline = if device.has_raytracing() {
        let rt_trace_cs = load_compute_shader(
            backend,
            dreamcoast_shader::rt_trace_cs_spirv,
            dreamcoast_shader::rt_trace_cs_dxil,
            dreamcoast_shader::rt_trace_cs_metallib,
            "rt_trace",
        )?;
        Some(device.create_compute_pipeline(&ComputePipelineDesc {
            compute_bytes: rt_trace_cs,
            compute_entry: "csMain",
            push_constant_size: 112, // inv_view_proj + cam_pos + sun_dir + out/w/h/pad
            bindless: true,
            threads_per_group: [8, 8, 1],
        })?)
    } else {
        None
    };

    // Phase 8 M4: inline path tracer (diffuse GI bounce loop + progressive
    // accumulation). Shares the bindless TLAS + geometry storage buffers.
    let rt_path_pipeline = if device.has_raytracing() {
        let rt_path_cs = load_compute_shader(
            backend,
            dreamcoast_shader::rt_path_cs_spirv,
            dreamcoast_shader::rt_path_cs_dxil,
            dreamcoast_shader::rt_path_cs_metallib,
            "rt_path",
        )?;
        Some(device.create_compute_pipeline(&ComputePipelineDesc {
            compute_bytes: rt_path_cs,
            compute_entry: "csMain",
            push_constant_size: 128, // inv_view_proj + cam_pos + sun + 2x uint4
            bindless: true,
            threads_per_group: [8, 8, 1],
        })?)
    } else {
        None
    };

    let rt_pipeline_shaders_available = match backend {
        BackendKind::Metal => {
            dreamcoast_shader::rt_pipeline_rgen_metallib().is_some()
                && dreamcoast_shader::rt_pipeline_miss_metallib().is_some()
                && dreamcoast_shader::rt_pipeline_chit_metallib().is_some()
                && dreamcoast_shader::rt_pipeline_dispatch_metallib().is_some()
                && dreamcoast_shader::rt_pipeline_isect_metallib().is_some()
        }
        _ => true,
    };

    let rt_pipeline_requested = std::env::var_os("P8_PATHTRACE_PIPELINE").is_some();

    // Phase 8 M5: the same path tracer via the hardware ray-tracing *pipeline*
    // (raygen / miss / closest-hit + shader binding table). Reproduces the inline
    // tracer's image so the two RT abstractions can be cross-checked. Gated on
    // `supports_rt_pipeline()`; on Metal the shader bytes are optional because the
    // converter/DXC toolchain may not be installed on every development machine.
    let rt_pt_pipeline = if rt_pipeline_requested
        && device.supports_rt_pipeline()
        && rt_pipeline_shaders_available
    {
        let rgen = load_compute_shader(
            backend,
            dreamcoast_shader::rt_pipeline_rgen_spirv,
            dreamcoast_shader::rt_pipeline_rgen_dxil,
            dreamcoast_shader::rt_pipeline_rgen_metallib,
            "rt_pipeline_rgen",
        )?;
        let miss = load_compute_shader(
            backend,
            dreamcoast_shader::rt_pipeline_miss_spirv,
            dreamcoast_shader::rt_pipeline_miss_dxil,
            dreamcoast_shader::rt_pipeline_miss_metallib,
            "rt_pipeline_miss",
        )?;
        let chit = load_compute_shader(
            backend,
            dreamcoast_shader::rt_pipeline_chit_spirv,
            dreamcoast_shader::rt_pipeline_chit_dxil,
            dreamcoast_shader::rt_pipeline_chit_metallib,
            "rt_pipeline_chit",
        )?;
        Some(device.create_raytracing_pipeline(&RaytracingPipelineDesc {
            raygen_bytes: rgen,
            raygen_entry: "rgMain",
            miss_bytes: miss,
            miss_entry: "msMain",
            closesthit_bytes: chit,
            closesthit_entry: "chMain",
            metal_ray_dispatch_bytes: if backend == BackendKind::Metal {
                dreamcoast_shader::rt_pipeline_dispatch_metallib()
            } else {
                None
            },
            metal_ray_dispatch_entry: Some("RaygenIndirection"),
            metal_intersection_bytes: if backend == BackendKind::Metal {
                dreamcoast_shader::rt_pipeline_isect_metallib()
            } else {
                None
            },
            metal_intersection_entry: Some("irconverter.wrapper.intersection.function.triangle"),
            push_constant_size: 128, // matches rt_path / rt_pipeline PushConstants
            // Payload = float3 x4 (48) + uint x3 (12) + float x2 cone state (8) = 68,
            // rounded up to a multiple of 8. Must be >= the shader payload or D3D12
            // CreateStateObject rejects the SHADER_CONFIG with E_INVALIDARG.
            max_payload_size: 72,
            max_attribute_size: 8, // barycentrics (float2)
        })?)
    } else {
        None
    };

    // GPU particle system (Phase 7): a persistent ping-pong buffer pair advanced by
    // a compute pass and drawn as instanced billboards (see `particle.rs`). Seeds
    // both buffers on construction.
    let mut particles =
        ParticleSystem::new(&device, backend, compute_supported, swapchain.format())?;

    // GPU frustum culling (Phase 7): a compute pass tests a cube instance grid
    // against the frustum and writes an indirect draw; the draw renders only the
    // visible instances (see `cull.rs`).
    let cull = CullSystem::new(&device, backend, compute_supported, swapchain.format())?;

    // Sky pipeline: renders the procedural sky into each environment cube face.
    let (sky_vs, sky_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::sky_vs_spirv,
        dreamcoast_shader::sky_fs_spirv,
        dreamcoast_shader::sky_vs_dxil,
        dreamcoast_shader::sky_fs_dxil,
        dreamcoast_shader::sky_vs_metallib,
        dreamcoast_shader::sky_fs_metallib,
        "sky",
    )?;
    let sky_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: sky_vs,
        fragment_bytes: sky_fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_formats: &[Format::Rgba16Float],
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::None,
        blend: BlendMode::Opaque,
        push_constant_size: 32, // sun float4 + face + flip_y + pad
        bindless: true,         // for the root-constants param (push constants)
        uniform_buffer: false,
        depth_test: false,
        depth_format: None,
    })?;

    // Capture pipeline: forward-renders scene geometry into the env cube faces
    // (camera-based real-time capture), simple direct lighting only.
    let (cap_vs, cap_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::capture_vs_spirv,
        dreamcoast_shader::capture_fs_spirv,
        dreamcoast_shader::capture_vs_dxil,
        dreamcoast_shader::capture_fs_dxil,
        dreamcoast_shader::capture_vs_metallib,
        dreamcoast_shader::capture_fs_metallib,
        "capture",
    )?;
    let capture_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: cap_vs,
        fragment_bytes: cap_fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_formats: &[HDR_FORMAT],
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::MeshPosNormal,
        blend: BlendMode::Opaque,
        push_constant_size: 208, // mvp+model(128) + base_color(16) + sun(16) + misc(16) + eye(16) + ibl(16)
        bindless: true,
        uniform_buffer: false,
        depth_test: true, // occlusion when capturing the scene into the cube
        depth_format: Some(DEPTH_FORMAT),
    })?;

    // Irradiance pipeline: convolves the env cube into a diffuse-irradiance cube.
    let (irr_vs, irr_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::irradiance_vs_spirv,
        dreamcoast_shader::irradiance_fs_spirv,
        dreamcoast_shader::irradiance_vs_dxil,
        dreamcoast_shader::irradiance_fs_dxil,
        dreamcoast_shader::irradiance_vs_metallib,
        dreamcoast_shader::irradiance_fs_metallib,
        "irradiance",
    )?;
    let irradiance_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: irr_vs,
        fragment_bytes: irr_fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_formats: &[HDR_FORMAT],
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::None,
        blend: BlendMode::Opaque,
        push_constant_size: 16, // face + flip_y + env_index + pad
        bindless: true,
        uniform_buffer: false,
        depth_test: false,
        depth_format: None,
    })?;

    // Prefilter pipeline: GGX-prefilters the env cube per roughness mip.
    let (pre_vs, pre_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::prefilter_vs_spirv,
        dreamcoast_shader::prefilter_fs_spirv,
        dreamcoast_shader::prefilter_vs_dxil,
        dreamcoast_shader::prefilter_fs_dxil,
        dreamcoast_shader::prefilter_vs_metallib,
        dreamcoast_shader::prefilter_fs_metallib,
        "prefilter",
    )?;
    let prefilter_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: pre_vs,
        fragment_bytes: pre_fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_formats: &[HDR_FORMAT],
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::None,
        blend: BlendMode::Opaque,
        push_constant_size: 20, // face + flip_y + env_index + roughness + env_mips
        bindless: true,
        uniform_buffer: false,
        depth_test: false,
        depth_format: None,
    })?;

    // BRDF LUT pipeline: integrates the environment-BRDF terms into an Rg16Float 2D.
    let (brdf_vs, brdf_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::brdf_vs_spirv,
        dreamcoast_shader::brdf_fs_spirv,
        dreamcoast_shader::brdf_vs_dxil,
        dreamcoast_shader::brdf_fs_dxil,
        dreamcoast_shader::brdf_vs_metallib,
        dreamcoast_shader::brdf_fs_metallib,
        "brdf",
    )?;
    let brdf_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: brdf_vs,
        fragment_bytes: brdf_fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_formats: &[Format::Rg16Float],
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::None,
        blend: BlendMode::Opaque,
        push_constant_size: 16, // flip_y + pad
        bindless: true,
        uniform_buffer: false,
        depth_test: false,
        depth_format: None,
    })?;

    // Clip-space Y orientation for the full-screen passes (Vulkan = 1, D3D12 /
    // Metal = 0; both have a Y-up NDC with a top-left framebuffer origin).
    let flip_y: u32 = match backend {
        BackendKind::Vulkan => 1,
        BackendKind::D3d12 => 0,
        BackendKind::Metal => 0,
    };

    // Upload material textures for the loaded model (bindless). Base color is
    // sRGB-encoded; the metallic-roughness and normal maps carry linear data.
    let mut _textures: Vec<Texture> = Vec::new();
    let base_index = match &model.material.base_color {
        Some(im) => upload_texture(&device, &mut _textures, im, Format::Rgba8Srgb)?,
        None => {
            let t = make_checker_texture(&device)?;
            let i = t.bindless_index();
            _textures.push(t);
            i
        }
    };
    let mr_index = match &model.material.metallic_roughness {
        Some(im) => upload_texture(&device, &mut _textures, im, Format::Rgba8Unorm)?,
        None => NO_TEXTURE,
    };
    let normal_index = match &model.material.normal {
        Some(im) => upload_texture(&device, &mut _textures, im, Format::Rgba8Unorm)?,
        None => NO_TEXTURE,
    };
    let emissive_index = match &model.material.emissive {
        Some(im) => upload_texture(&device, &mut _textures, im, Format::Rgba8Srgb)?,
        None => NO_TEXTURE,
    };

    // Build the sample scene: the loaded model at the origin plus a few procedural
    // objects with varied materials (showing PBR + image-based reflections). A
    // ground plane (kept separate — it's also the environment-capture geometry)
    // catches the shadows and grounds the reflections.
    let r = model_radius;
    let sphere = dreamcoast_asset::uv_sphere(48, 32);
    let cube = dreamcoast_asset::unit_cube();
    let trs =
        |pos: Vec3, scale: f32| Mat4::from_translation(pos) * Mat4::from_scale(Vec3::splat(scale));
    let mut scene: Vec<SceneObject> = Vec::new();
    // Loaded model (its glTF material).
    let (vbuf, ibuf, index_count) = upload_mesh(&device, &model)?;
    scene.push(SceneObject {
        vbuf,
        ibuf,
        index_count,
        vertex_count: model.vertices.len() as u32,
        transform: Mat4::IDENTITY,
        base_color: model.material.base_color_factor,
        metallic: model.material.metallic_factor,
        roughness: model.material.roughness_factor,
        tex: [base_index, mr_index, normal_index, emissive_index],
        casts_shadow: true,
    });
    // Polished chrome sphere.
    let (sv, si, sc) = upload_mesh(&device, &sphere)?;
    scene.push(SceneObject {
        vbuf: sv,
        ibuf: si,
        index_count: sc,
        vertex_count: sphere.vertices.len() as u32,
        transform: trs(Vec3::new(-r * 1.7, r * 0.75, r * 0.5), r * 0.75),
        base_color: [0.95, 0.96, 0.97, 1.0],
        metallic: 1.0,
        roughness: 0.08,
        tex: [NO_TEXTURE; 4],
        casts_shadow: true,
    });
    // Brushed-copper sphere.
    let (sv2, si2, sc2) = upload_mesh(&device, &sphere)?;
    scene.push(SceneObject {
        vbuf: sv2,
        ibuf: si2,
        index_count: sc2,
        vertex_count: sphere.vertices.len() as u32,
        transform: trs(Vec3::new(r * 1.9, r * 0.5, -r * 0.4), r * 0.5),
        base_color: [0.95, 0.64, 0.54, 1.0],
        metallic: 1.0,
        roughness: 0.35,
        tex: [NO_TEXTURE; 4],
        casts_shadow: true,
    });
    // Red dielectric cube.
    let (cv, ci, cc) = upload_mesh(&device, &cube)?;
    scene.push(SceneObject {
        vbuf: cv,
        ibuf: ci,
        index_count: cc,
        vertex_count: cube.vertices.len() as u32,
        transform: Mat4::from_translation(Vec3::new(0.0, r * 0.45, -r * 2.0))
            * Mat4::from_scale(Vec3::splat(r * 0.45)),
        base_color: [0.85, 0.25, 0.2, 1.0],
        metallic: 0.0,
        roughness: 0.5,
        tex: [NO_TEXTURE; 4],
        casts_shadow: true,
    });
    let scene_radius = r * 3.0;

    // Ground plane (separate handle: also used by the environment capture).
    let ground = ground_mesh(scene_radius * 1.3, 0.0);
    let (ground_vbuf, ground_ibuf, ground_count) = upload_mesh(&device, &ground)?;

    // Hardware ray tracing (Phase 8): build one BLAS per scene mesh + ground and a
    // TLAS over their instances, then register the TLAS in the bindless table so
    // the inline-trace compute pass (M3) can trace it. `rt_scene` outlives the
    // frame loop (the TLAS must stay alive while it is bound).
    let rt_scene = if device.has_raytracing() {
        let mut geoms: Vec<RtGeometry> = scene
            .iter()
            .map(|o| RtGeometry {
                vertex_buffer: &o.vbuf,
                index_buffer: &o.ibuf,
                geometry: BlasGeometry {
                    vertex_count: o.vertex_count,
                    vertex_stride: 32,
                    index_count: o.index_count,
                },
            })
            .collect();
        geoms.push(RtGeometry {
            vertex_buffer: &ground_vbuf,
            index_buffer: &ground_ibuf,
            geometry: BlasGeometry {
                vertex_count: ground.vertices.len() as u32,
                vertex_stride: 32,
                index_count: ground_count,
            },
        });
        let instances: Vec<TlasInstance> = (0..geoms.len())
            .map(|i| TlasInstance {
                blas_index: i as u32,
                transform: mat4_to_3x4(if i < scene.len() {
                    scene[i].transform
                } else {
                    Mat4::IDENTITY
                }),
                custom_index: i as u32,
                mask: 0xFF,
            })
            .collect();
        match device.build_raytracing_scene(&geoms, &instances) {
            Ok(s) => {
                device.bind_tlas(&s);
                info!("ray-tracing scene built: {} BLAS + 1 TLAS", geoms.len());
                Some(s)
            }
            Err(e) => {
                tracing::error!("ray-tracing scene build failed: {e}");
                None
            }
        }
    } else {
        None
    };

    // Phase 8 M4: per-instance geometry storage buffers + instance table for the
    // path tracer's hit shading. One vertex + one index storage buffer per
    // instance (read as raw byte-address buffers in the shader), plus a table
    // mapping InstanceID -> { vertex SB index, index SB index, albedo }. The order
    // MUST match the TLAS instance custom_index order (scene objects, then ground).
    // `_rt_geometry` keeps the geometry buffers alive for the program's lifetime.
    let (rt_instance_table, _rt_geometry, rt_instance_count) = if rt_scene.is_some() {
        // (mesh, material) per instance, in TLAS instance order (objects, then
        // ground). Materials mirror the rasterizer's so the path tracer shades with
        // the same metallic-roughness PBR model.
        // base_color.a is the path tracer's emissive scale (the Cornell light uses
        // it). The sample-scene objects are NOT emitters — their .a is just opacity —
        // so zero it, else e.g. the chrome sphere emits its own base color and reads
        // as a glowing white ball instead of a mirror.
        let mat_of = |o: &SceneObject| PtMaterial {
            base_color: [o.base_color[0], o.base_color[1], o.base_color[2], 0.0],
            metallic: o.metallic,
            roughness: o.roughness,
            ao: 1.0,
            tex: o.tex,
        };
        let entries: [(&MeshData, PtMaterial); 5] = [
            (&model, mat_of(&scene[0])),
            (&sphere, mat_of(&scene[1])),
            (&sphere, mat_of(&scene[2])),
            (&cube, mat_of(&scene[3])),
            (&ground, PtMaterial::diffuse([0.8, 0.8, 0.8, 0.0])),
        ];
        let (table, geometry) = build_pt_instance_table(&device, &entries)?;
        info!("path-tracer instance table: {} instances", entries.len());
        (Some(table), geometry, entries.len() as u32)
    } else {
        (None, Vec::new(), 0u32)
    };

    // Phase 8 M4: an alternate Cornell-box scene for the path tracer (strong color
    // bleeding from the red/green walls, area-light GI). Built once: its own BLAS
    // per quad/box + TLAS + instance table. The host-visible vertex/index buffers
    // are only needed during the BLAS build, so they drop at the end of this block.
    let (cornell_scene, cornell_table, _cornell_geometry, cornell_instance_count) =
        if device.has_raytracing() {
            let meshes = dreamcoast_asset::cornell_box();
            let mut hostbufs: Vec<(Buffer, Buffer, u32, u32)> = Vec::with_capacity(meshes.len());
            for (m, _) in &meshes {
                let (vb, ib, ic) = upload_mesh(&device, m)?;
                hostbufs.push((vb, ib, ic, m.vertices.len() as u32));
            }
            let geoms: Vec<RtGeometry> = hostbufs
                .iter()
                .map(|(vb, ib, ic, vc)| RtGeometry {
                    vertex_buffer: vb,
                    index_buffer: ib,
                    geometry: BlasGeometry {
                        vertex_count: *vc,
                        vertex_stride: 32,
                        index_count: *ic,
                    },
                })
                .collect();
            let instances: Vec<TlasInstance> = (0..geoms.len() as u32)
                .map(|i| TlasInstance {
                    blas_index: i,
                    transform: mat4_to_3x4(Mat4::IDENTITY), // geometry already world-space
                    custom_index: i,
                    mask: 0xFF,
                })
                .collect();
            let scene = device.build_raytracing_scene(&geoms, &instances)?;
            // The Cornell box is all matte diffuse (emissive ceiling via base_color.a).
            let entries: Vec<(&MeshData, PtMaterial)> = meshes
                .iter()
                .map(|(m, a)| (m, PtMaterial::diffuse(*a)))
                .collect();
            let (table, geometry) = build_pt_instance_table(&device, &entries)?;
            info!("cornell-box scene built: {} instances", meshes.len());
            (Some(scene), Some(table), geometry, meshes.len() as u32)
        } else {
            (None, None, Vec::new(), 0u32)
        };

    // Per-frame globals uniform buffer (one 256-byte slice per frame-in-flight).
    let globals_buffer = device.create_buffer(&BufferDesc {
        size: GLOBALS_SLICE * FRAMES_IN_FLIGHT as u64,
        usage: BufferUsage::Uniform,
    })?;
    device.set_globals_buffer(&globals_buffer, GLOBALS_SLICE);

    let mut gui = Gui::new(&device, swapchain.format(), FRAMES_IN_FLIGHT)?;

    // One render-graph transient pool per frame-in-flight (reused only after the
    // frame slot's fence has signaled — no cross-frame hazards).
    let mut pools: Vec<ResourcePool> = (0..FRAMES_IN_FLIGHT).map(|_| ResourcePool::new()).collect();

    // UI-controlled lighting state.
    let mut sun_dir = [0.4f32, 0.8, 0.4];
    let mut sun_intensity = 3.0f32;
    let mut ambient = 0.04f32;
    let mut exposure = 0.6f32;
    // On by default; `NO_POINT_LIGHTS=1` disables them (the path tracer has no point
    // lights, so a fair raster-vs-ground-truth comparison turns these off).
    let mut point_lights_on = std::env::var_os("NO_POINT_LIGHTS").is_none();
    let mut shadows_on = true;
    let mut shadow_bias = 0.0015f32;
    // Override the model's metallic/roughness (to inspect IBL on the avocado).
    let mut override_material = false;
    let mut metallic_override = 1.0f32;
    let mut roughness_override = 0.15f32;
    let mut debug_view: usize = 0;
    let mut post_mode: usize = 0;
    let mut aliasing = true;
    // Phase 7: route the HDR result through a compute post-process (3x3 blur into
    // a storage image) before tonemapping. Initial state seedable via env var so
    // headless screenshots can exercise each demo (`P7_COMPUTE_POST=1`, etc.).
    let mut compute_post = compute_supported && std::env::var_os("P7_COMPUTE_POST").is_some();
    // Phase 7: GPU particle simulation (compute-updated buffer, instanced draw).
    let mut particles_on = compute_supported && std::env::var_os("P7_PARTICLES").is_some();
    // Run the particle sim on the async-compute queue (overlapping graphics) when a
    // dedicated compute queue exists. Off / unsupported -> the sim runs as a graph
    // compute pass on the graphics queue (single-queue path), with identical output.
    let async_compute_supported = device.has_async_compute();
    let mut async_compute_on = async_compute_supported
        && (std::env::var_os("ASYNC_COMPUTE").is_some() || !screenshot_mode);
    // Phase 7: GPU frustum culling -> indirect draw of a cube instance grid.
    let mut gpu_cull = compute_supported && std::env::var_os("P7_CULL").is_some();
    // Phase 8 M3: replace the rasterized image with an inline ray-query trace
    // (primary hit instance color modulated by a hardware shadow ray). Requires a
    // built RT scene; headless toggle via `P8_PATHTRACE`.
    let mut path_trace = rt_trace_pipeline.is_some()
        && rt_scene.is_some()
        && std::env::var_os("P8_PATHTRACE").is_some();
    // Debug mode: show the M3 single-bounce trace viz (instance color + RT shadow)
    // instead of the M4 path tracer. Headless toggle via `P8_RT_DEBUG`.
    let mut rt_debug = device.has_raytracing() && std::env::var_os("P8_RT_DEBUG").is_some();
    // Cornell-box scene for the path tracer (strong color bleeding). Headless
    // toggle via `P8_CORNELL`; uses a fixed front-facing camera.
    let mut cornell = cornell_scene.is_some() && std::env::var_os("P8_CORNELL").is_some();
    // Phase 11 Stage A: replace the rasterized image with a compute software ray
    // trace of the analytic SDF scene (sphere tracing, no hardware RT). Headless
    // toggle via `P11_SDF`.
    let mut sdf_trace = gdf.has_sdf_trace() && std::env::var_os("P11_SDF").is_some();
    // Phase 11 Stage B (B1): 3D volume texture smoke test — fill a volume then view a
    // trilinear-sampled Z slice. Headless toggle via `P11_VOLUME_TEST`.
    let mut volume_test = gdf.has_volume() && std::env::var_os("P11_VOLUME_TEST").is_some();
    // Phase 11 Stage B (B2): bake a mesh's SDF into the volume, then view a slice.
    // Headless toggle via `P11_SDF_BAKE`. The bake is expensive (O(voxels*tris)) so
    // it runs once (`sdf_bake_done`), and later frames just view the baked volume.
    let mut sdf_bake = gdf.has_bake() && std::env::var_os("P11_SDF_BAKE").is_some();
    let mut sdf_bake_done = false;
    // Phase 11 Stage B (B3): merge per-mesh SDF instances into the global distance
    // field, then view a slice. Headless toggle via `P11_GDF_MERGE`. Like the bake it
    // builds the field once (`gdf_merge_done`); the merge depends on the B2 bake, so it
    // also drives the bake when enabled.
    let mut gdf_merge = gdf.has_merge() && std::env::var_os("P11_GDF_MERGE").is_some();
    let mut gdf_merge_done = false;
    // Phase 11 Stage B (B4): SW ray trace the merged GDF. Builds the GDF (bake + merge,
    // once) then sphere-traces it from a fixed camera framing the unit-cube scene.
    // Headless toggle via `P11_GDF_TRACE`; `P11_GDF_ANALYTIC` swaps the GDF sample for
    // the analytic field it was baked from (the B4 correctness reference).
    let mut gdf_trace = gdf.has_gdf_trace() && std::env::var_os("P11_GDF_TRACE").is_some();
    let gdf_trace_analytic = std::env::var_os("P11_GDF_ANALYTIC").is_some();
    // Phase 8 M5: drive the path tracer through the full RT pipeline + SBT instead
    // of the inline compute ray query (same image). Headless toggle via
    // `P8_PATHTRACE_PIPELINE`; ignored when no RT pipeline is available.
    let mut path_trace_pipeline = rt_pt_pipeline.is_some() && rt_pipeline_requested;
    // Samples per path-trace dispatch (accumulated progressively across frames).
    let path_spp: u32 = 8;
    // Real-time environment capture: re-capture the env chain every frame (so the
    // sky/IBL track the live sun); when off, re-capture only when the sun changes.
    let mut realtime_env = true;
    // Multi-bounce: shade captured surfaces with IBL from the previous frame's
    // cube set, so reflective surfaces appear reflective inside reflections.
    let mut multibounce = true;
    let mut env_captured = false;
    let mut last_sun = (sun_dir, sun_intensity);
    // Ping-pong parity for the two cube sets; advances only when a capture
    // actually happens, so a skipped frame keeps sampling the last written set.
    let mut env_parity = 0usize;
    let mut last_written = 0usize;

    let mut command_buffers = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut image_available = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut in_flight = Vec::with_capacity(FRAMES_IN_FLIGHT);
    // Async-compute resources: a command buffer on the compute queue per frame,
    // plus a semaphore the compute submit signals and the graphics submit waits on
    // (so the particle draw sees the compute-written buffer). Only used when a
    // dedicated compute queue exists and the async toggle is on.
    let compute_queue = device.compute_queue();
    let mut compute_command_buffers = Vec::with_capacity(FRAMES_IN_FLIGHT);
    let mut compute_done = Vec::with_capacity(FRAMES_IN_FLIGHT);
    // GPU profiler: one timestamp query heap per frame in flight (read back after
    // that slot's fence, so the results never stall the GPU). MAX_QUERIES covers
    // (scheduled passes + 1) boundaries with headroom (Phase 9 M1).
    const MAX_QUERIES: u32 = 32;
    let mut query_heaps = Vec::with_capacity(FRAMES_IN_FLIGHT);
    for _ in 0..FRAMES_IN_FLIGHT {
        command_buffers.push(device.create_command_buffer()?);
        image_available.push(device.create_semaphore()?);
        in_flight.push(device.create_fence(true)?);
        compute_command_buffers.push(device.create_compute_command_buffer()?);
        compute_done.push(device.create_semaphore()?);
        query_heaps.push(device.create_query_heap(MAX_QUERIES)?);
    }
    // Profiler state: per-slot recorded pass names (to interpret the next readback)
    // + the most recent per-pass GPU milliseconds (for the ImGui table). Off by
    // default; toggled in the UI.
    let mut profiler_on = std::env::var("PROFILE_GPU").is_ok();
    let mut slot_pass_names: Vec<Vec<String>> = vec![Vec::new(); FRAMES_IN_FLIGHT];
    let mut gpu_timings: Vec<(String, f32)> = Vec::new();
    let mut render_finished = build_render_finished(&device, swapchain.image_count())?;

    // IBL resource set. The environment cube holds the procedural sky; the
    // irradiance / prefilter cubes and BRDF LUT are derived from it. The env +
    // irradiance + prefilter generation is reusable (call again to re-capture
    // when the sky changes — a future real-time atmospheric model); the BRDF LUT
    // is sky-independent and generated once.
    // Double-buffered environment cube sets for multi-bounce reflections. Each
    // frame captures the scene into the "write" set while shading those captured
    // surfaces with IBL from the "read" set (the previous frame), so reflective
    // surfaces reflect other reflective surfaces. The sets ping-pong; the main
    // lighting pass always samples the freshly written set. The BRDF LUT is
    // sky-independent so it stays single.
    let make_cube_set = || -> anyhow::Result<CubeSet> {
        Ok(CubeSet {
            env: device.create_cubemap(&CubemapDesc {
                size: ENV_SIZE,
                format: HDR_FORMAT,
                mip_levels: ENV_MIPS,
            })?,
            irradiance: device.create_cubemap(&CubemapDesc {
                size: IRRADIANCE_SIZE,
                format: HDR_FORMAT,
                mip_levels: 1,
            })?,
            prefilter: device.create_cubemap(&CubemapDesc {
                size: PREFILTER_SIZE,
                format: HDR_FORMAT,
                mip_levels: PREFILTER_MIPS,
            })?,
        })
    };
    let cube_sets = [make_cube_set()?, make_cube_set()?];
    // Depth buffer for capturing scene geometry into the env cube faces.
    let capture_depth = device.create_depth_buffer(Extent2D::new(ENV_SIZE, ENV_SIZE))?;
    let brdf_lut = device.create_render_target(&rhi::RenderTargetDesc {
        width: BRDF_SIZE,
        height: BRDF_SIZE,
        format: Format::Rg16Float,
        storage: false,
    })?;
    let brdf_index = brdf_lut.bindless_index() as i32;
    // Name the persistent IBL resources so GPU captures (RenderDoc/PIX) show
    // readable identifiers instead of anonymous "Texture N" (Phase 9 M2; debug
    // builds only — the backends no-op these in release).
    brdf_lut.set_name("ibl_brdf_lut");
    capture_depth.set_name("ibl_capture_depth");
    for (i, set) in cube_sets.iter().enumerate() {
        set.env.set_name(&format!("ibl_env_cube[{i}]"));
        set.irradiance
            .set_name(&format!("ibl_irradiance_cube[{i}]"));
        set.prefilter.set_name(&format!("ibl_prefilter_cube[{i}]"));
    }
    let ibl = IblResources {
        sky_pipeline: &sky_pipeline,
        capture_pipeline: &capture_pipeline,
        irradiance_pipeline: &irradiance_pipeline,
        prefilter_pipeline: &prefilter_pipeline,
        ground_vbuf: &ground_vbuf,
        ground_ibuf: &ground_ibuf,
        ground_count,
    };
    {
        // The BRDF LUT is sky-independent — generate it once. The environment
        // chain is (re)captured per frame inside the render loop.
        let gen_cmd = device.create_command_buffer()?;
        let gen_fence = device.create_fence(false)?;
        generate_brdf_lut(
            &queue,
            &gen_cmd,
            &gen_fence,
            &brdf_pipeline,
            &brdf_lut,
            flip_y,
        )?;
    }
    // Initialize both cube sets once (single-bounce, no previous environment) so
    // the first multi-bounce frame reads valid data instead of uninitialized
    // memory. Uses an approximate camera; the render loop immediately recaptures
    // with the live camera.
    {
        let boot_eye = Vec3::new(0.0, model_radius * 0.6, 0.0)
            + Vec3::new(scene_radius * 1.6, scene_radius * 0.55, 0.0);
        let init_cmd = device.create_command_buffer()?;
        let init_fence = device.create_fence(false)?;
        init_cmd.begin()?;
        for set in &cube_sets {
            record_environment_capture(
                &init_cmd,
                &ibl,
                set,
                None,
                brdf_index,
                &scene,
                &capture_depth,
                boot_eye,
                sun_dir,
                sun_intensity,
                ambient,
                flip_y,
                backend == BackendKind::Vulkan,
            );
        }
        init_cmd.end()?;
        queue.submit_oneshot(&init_cmd, &init_fence)?;
        init_fence.wait()?;
    }

    let _ = window.take_resized();
    info!("entering render loop");
    let mut frame = 0usize;
    let mut frame_no = 0u64;
    let mut f2_prev = false;
    let mut needs_recreate = false;
    let mut last = Instant::now();
    // Accumulated wall-clock time, for the particle simulation's respawn hashing.
    let mut elapsed = 0.0f32;
    // Fixed view in screenshot mode for reproducible output.
    let mut angle = if screenshot_mode { 0.7 } else { 0.0 };
    // Path-tracer progressive accumulation (Phase 8 M4): a persistent float4-per-
    // pixel sum buffer, reset when the view/lighting/resolution changes. Extra
    // headless warmup frames let the static-camera screenshot converge.
    let mut path_accum: Option<rhi::StorageBuffer> = None;
    let mut accum_extent = (0u32, 0u32);
    let mut accum_frame = 0u32;
    let mut last_pt_key: Option<[u32; 8]> = None;
    // Which scene's TLAS is currently bound to the bindless slot (None = open
    // scene, the startup default). Switching rebinds (wait_idle) the TLAS.
    let mut bound_cornell = false;
    const PATHTRACE_WARMUP: u64 = 64;

    while !window.should_close() {
        window.pump_events();
        if window.take_resized() {
            needs_recreate = true;
        }
        let (ww, wh) = window.size();
        if ww == 0 || wh == 0 {
            std::thread::sleep(std::time::Duration::from_millis(16));
            continue;
        }
        if needs_recreate {
            device.wait_idle()?;
            swapchain.recreate(&swapchain_desc(Extent2D::new(ww, wh)))?;
            for p in &mut pools {
                p.clear(); // transient extents changed; drop cached targets
            }
            render_finished = build_render_finished(&device, swapchain.image_count())?;
            needs_recreate = false;
        }

        // Wait for this frame slot's previous submission to finish BEFORE the
        // acquire below. The acquire reuses `image_available[frame]`, and Vulkan
        // forbids acquiring with a semaphore that still has a pending wait from
        // that earlier submit (VUID-vkAcquireNextImageKHR-semaphore-01779). This is
        // the standard frames-in-flight order: wait → reset → acquire → record →
        // submit. (D3D12/Metal ignore the semaphore, but the wait still gates reuse
        // of this slot's command buffer / query heap below.)
        let fence = &in_flight[frame];
        fence.wait()?;

        // Acquire the drawable up front: its *actual* pixel size is the single
        // source of truth for this whole frame (ImGui display size, camera aspect,
        // render extent, viewport). A failed acquire skips here, BEFORE the ImGui
        // frame is started, so NewFrame/Render stay balanced (skipping after
        // new_frame() trips an ImGui assertion).
        let image_index = match swapchain.acquire_next_image(&image_available[frame])? {
            Some(i) => i,
            None => {
                needs_recreate = true;
                continue;
            }
        };
        let (cw, ch) = {
            let e = swapchain.extent_2d();
            (e.width, e.height)
        };

        let now = Instant::now();
        let dt = (now - last).as_secs_f32();
        last = now;
        elapsed += dt;
        // Clamp the sim step so a long stall (e.g. resize) can't explode particles.
        let sim_dt = dt.clamp(0.0, 1.0 / 30.0);
        if !screenshot_mode {
            angle += dt * 0.6; // hold a fixed view when capturing
        }

        // Path-trace screenshots need a long warmup so the static-camera
        // accumulation converges before the frame is captured.
        let warmup = if path_trace && !rt_debug {
            PATHTRACE_WARMUP
        } else {
            SCREENSHOT_WARMUP
        };

        // Decide whether this frame produces a screenshot: a scheduled capture in
        // screenshot mode (after warmup), or an F2 rising edge interactively.
        let f2 = window.input().key_down(VK_F2);
        let f2_pressed = f2 && !f2_prev;
        f2_prev = f2;
        let capture_this_frame: Option<Capture> = if screenshot_mode {
            frame_no
                .checked_sub(warmup)
                .and_then(|i| captures.get(i as usize).cloned())
        } else if f2_pressed {
            Some(Capture {
                path: interactive_screenshot_path(),
                include_ui: true,
            })
        } else {
            None
        };
        let include_ui = capture_this_frame
            .as_ref()
            .map(|c| c.include_ui)
            .unwrap_or(true);

        // Orbiting camera framing the whole sample scene.
        let focus = Vec3::new(0.0, model_radius * 0.6, 0.0);
        let dist = scene_radius * 1.6;
        let eye = focus + Vec3::new(angle.cos() * dist, scene_radius * 0.55, angle.sin() * dist);
        let view = Mat4::look_at_rh(eye, focus, Vec3::Y);
        let proj_noflip =
            Mat4::perspective_rh(60f32.to_radians(), cw as f32 / ch as f32, 0.05, 100.0);
        let mut proj = proj_noflip;
        if backend == BackendKind::Vulkan {
            proj.y_axis.y *= -1.0; // Vulkan clip-space Y points down
        }
        let view_proj = proj * view;
        // Frustum culling uses the Y-flip-free matrix so the visible set (and the
        // indirect draw count) is identical on both backends.
        let cull_view_proj = proj_noflip * view;
        // Camera basis in world space (for billboarding the GPU particles).
        let inv_view = view.inverse();
        let cam_right = inv_view.x_axis.truncate();
        let cam_up = inv_view.y_axis.truncate();
        // For reconstructing background view rays (skybox) in the lighting pass.
        let inv_view_proj = view_proj.inverse().to_cols_array();

        // Skip the whole ImGui frame (NewFrame/Render must stay balanced) when
        // capturing a clean screenshot.
        if include_ui {
            let ui = gui.new_frame(dt, [cw as f32, ch as f32], window.input());
            ui.window("DreamCoast")
                .size([320.0, 320.0], imgui::Condition::FirstUseEver)
                .build(|| {
                    ui.text(format!("backend: {backend:?}"));
                    ui.text(format!(
                        "{:.1} FPS ({:.2} ms)",
                        1.0 / dt.max(1e-4),
                        dt * 1000.0
                    ));
                    ui.text(format!("scene: {} objects + ground", scene.len()));
                    ui.text(format!(
                        "validation: {}",
                        if validation_on { "on" } else { "off" }
                    ));
                    ui.separator();

                    // Sample browser: each technique group is a collapsing section,
                    // so the sandbox reads as a catalog of techniques rather than a
                    // flat wall of toggles. Core groups default open.
                    use imgui::TreeNodeFlags;
                    let open = TreeNodeFlags::DEFAULT_OPEN;

                    if ui.collapsing_header("Lighting", open) {
                        ui.combo_simple_string("Debug view", &mut debug_view, &DEBUG_VIEWS);
                        ui.input_float3("Sun dir", &mut sun_dir).build();
                        ui.slider("Sun intensity", 0.0, 10.0, &mut sun_intensity);
                        ui.slider("Ambient", 0.0, 0.5, &mut ambient);
                        ui.slider("Exposure", 0.1, 4.0, &mut exposure);
                        ui.checkbox("Point lights", &mut point_lights_on);
                        ui.checkbox("Shadows", &mut shadows_on);
                        ui.slider("Shadow bias", 0.0, 0.01, &mut shadow_bias);
                    }

                    if ui.collapsing_header("Material override", TreeNodeFlags::empty()) {
                        ui.checkbox("Override material", &mut override_material);
                        ui.slider("Metallic", 0.0, 1.0, &mut metallic_override);
                        ui.slider("Roughness", 0.0, 1.0, &mut roughness_override);
                    }

                    if ui.collapsing_header("IBL / Environment", TreeNodeFlags::empty()) {
                        ui.checkbox("Real-time env capture", &mut realtime_env);
                        ui.checkbox("Multi-bounce reflections", &mut multibounce);
                        ui.combo_simple_string("Post effect", &mut post_mode, &POST_EFFECTS);
                        ui.checkbox("Transient aliasing", &mut aliasing);
                    }

                    if ui.collapsing_header("Compute / GPGPU (Phase 7)", TreeNodeFlags::empty()) {
                        ui.checkbox("Compute post (blur)", &mut compute_post);
                        ui.checkbox("GPU particles", &mut particles_on);
                        if async_compute_supported {
                            ui.checkbox("  - async compute queue", &mut async_compute_on);
                        } else {
                            ui.text_disabled("  - async compute (no dedicated queue)");
                        }
                        ui.checkbox("GPU culling (indirect)", &mut gpu_cull);
                    }

                    if rt_path_pipeline.is_some() && rt_scene.is_some() {
                        if ui.collapsing_header("Ray tracing (Phase 8)", TreeNodeFlags::empty()) {
                            ui.checkbox("Path trace (inline ray query)", &mut path_trace);
                            if path_trace {
                                ui.checkbox("  - debug: instance + shadow viz", &mut rt_debug);
                                if !rt_debug {
                                    if cornell_scene.is_some() {
                                        ui.checkbox("  - Cornell box", &mut cornell);
                                    }
                                    if rt_pt_pipeline.is_some() {
                                        ui.checkbox(
                                            "  - pipeline + SBT (vs inline)",
                                            &mut path_trace_pipeline,
                                        );
                                    }
                                    ui.text(format!(
                                        "  - {} spp accumulated ({})",
                                        accum_frame.saturating_mul(path_spp),
                                        if path_trace_pipeline {
                                            "pipeline"
                                        } else {
                                            "inline"
                                        }
                                    ));
                                }
                            }
                        }
                    } else {
                        ui.text_disabled("Ray tracing (unsupported)");
                    }

                    if gdf.has_sdf_trace()
                        && ui.collapsing_header(
                            "Software ray tracing (Phase 11)",
                            TreeNodeFlags::empty(),
                        )
                    {
                        ui.checkbox("SDF sphere trace (compute, no HW RT)", &mut sdf_trace);
                        if sdf_trace {
                            ui.text_disabled("  - analytic SDF scene (Stage A)");
                        }
                        if gdf.has_volume() {
                            ui.checkbox("3D volume test (fill + slice view)", &mut volume_test);
                            if volume_test {
                                ui.text_disabled("  - Stage B RHI smoke test");
                            }
                        }
                        if gdf.has_bake() {
                            if ui.checkbox("SDF bake (per-mesh, slice view)", &mut sdf_bake) {
                                sdf_bake_done = false; // re-bake when re-enabled
                            }
                            if sdf_bake {
                                ui.text_disabled("  - Stage B2: baked sphere ≈ analytic");
                            }
                        }
                        if gdf.has_merge() {
                            if ui.checkbox("GDF merge (instances, slice view)", &mut gdf_merge) {
                                gdf_merge_done = false; // re-merge when re-enabled
                            }
                            if gdf_merge {
                                ui.text_disabled("  - Stage B3: min-merged instances");
                            }
                        }
                        if gdf.has_gdf_trace() {
                            ui.checkbox("GDF SW ray trace (compute)", &mut gdf_trace);
                            if gdf_trace {
                                ui.text_disabled("  - Stage B4: sphere-march baked GDF");
                            }
                        }
                    }

                    if ui.collapsing_header("Profiling & debug (Phase 9)", open) {
                        ui.checkbox("GPU profiler", &mut profiler_on);
                        if profiler_on {
                            if gpu_timings.is_empty() {
                                ui.text_disabled("  (measuring…)");
                            } else {
                                let mut total = 0.0;
                                for (name, ms) in &gpu_timings {
                                    ui.text(format!("  {name:<9} {ms:6.3} ms"));
                                    total += ms;
                                }
                                ui.text(format!("  {:<9} {total:6.3} ms", "total"));
                            }
                        }
                    }
                });
        }

        // This slot's previous submission is complete (waited on `fence` above), so
        // its timestamp queries are ready: read them back and turn the tick
        // boundaries into per-pass GPU milliseconds for the profiler UI (shown next
        // frame).
        if profiler_on && !slot_pass_names[frame].is_empty() {
            let heap = &query_heaps[frame];
            let ticks = heap.read();
            let period_ns = heap.period_ns();
            let names = &slot_pass_names[frame];
            gpu_timings = names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let dt = ticks[i + 1].saturating_sub(ticks[i]);
                    (name.clone(), dt as f32 * period_ns * 1e-6)
                })
                .collect();
        }

        fence.reset()?;

        let cmd = &command_buffers[frame];
        cmd.begin()?;

        // (Re)capture the environment into the "write" cube set before the main
        // graph samples it: every frame when real-time is on, otherwise only the
        // first frame and whenever the sun changes. With multi-bounce on, the
        // captured surfaces are shaded with IBL from the "read" set (the previous
        // frame), so reflective surfaces reflect each other; the parity advances
        // only on an actual capture, so a skipped frame keeps the last written set.
        let sun_changed = (sun_dir, sun_intensity) != last_sun;
        if realtime_env || !env_captured || sun_changed {
            let write = env_parity % 2;
            let read = 1 - write;
            let prev = if multibounce && env_captured {
                Some(&cube_sets[read])
            } else {
                None
            };
            record_environment_capture(
                cmd,
                &ibl,
                &cube_sets[write],
                prev,
                brdf_index,
                &scene,
                &capture_depth,
                // Capture the reflection probe at the scene centre, NOT the camera:
                // a camera-anchored probe gives every reflective surface a parallax
                // error (the reflected ground/horizon slides up the spheres as the
                // camera moves). A fixed probe near the objects keeps reflections
                // stable and roughly correct for surfaces around the centre.
                focus,
                sun_dir,
                sun_intensity,
                ambient,
                flip_y,
                backend == BackendKind::Vulkan,
            );
            last_written = write;
            env_parity += 1;
            env_captured = true;
            last_sun = (sun_dir, sun_intensity);
        }

        // The main lighting pass samples the most recently written set.
        let write_set = &cube_sets[last_written];
        let ibl_indices = [
            write_set.env.bindless_index() as i32,
            write_set.irradiance.bindless_index() as i32,
            write_set.prefilter.bindless_index() as i32,
            brdf_index,
        ];

        // Directional light view-projection: an orthographic box covering the
        // whole scene, looking from the sun toward it. Backend-neutral (the pbr
        // shader handles the Vulkan/D3D12 shadow-UV flip).
        let shadow_center = Vec3::new(0.0, model_radius * 0.5, 0.0);
        let light_vp = light_view_proj(sun_dir, shadow_center, scene_radius);

        // Write this frame's globals slice.
        let r = model_radius;
        let point_intensity = r * r * 8.0;
        let globals = Globals {
            camera_pos: [eye.x, eye.y, eye.z, 0.0],
            sun_direction: normalize3(sun_dir),
            sun_color: [1.0, 1.0, 1.0, sun_intensity],
            ambient: [ambient, ambient, ambient, exposure],
            counts: [
                if point_lights_on { 2 } else { 0 },
                debug_view as i32,
                (PREFILTER_MIPS - 1) as i32, // prefilter max LOD
                shadows_on as i32,
            ],
            point_pos: [
                [r * 2.0, r * 1.5, 0.0, 0.0],
                [-r * 2.0, r * 1.0, r * 1.5, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            point_color: [
                [1.0, 0.35, 0.2, point_intensity],
                [0.3, 0.5, 1.0, point_intensity],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            light_view_proj: light_vp.to_cols_array(),
            shadow: [shadow_bias, 1.0 / SHADOW_SIZE as f32, 0.0, 0.0],
            inv_view_proj,
            ibl: ibl_indices,
            // Reflection-probe centre (matches the env-capture eye) + a box proxy
            // for parallax-corrected specular IBL. The box floor sits on the ground
            // plane (y = 0) and its walls/ceiling match the captured ground extent,
            // so reflected-floor rays re-anchor onto the actual flat ground instead
            // of a sphere that bent them up to the (darker) horizon.
            probe: [focus.x, focus.y, focus.z, 1.0],
            probe_box_min: [-scene_radius * 1.3, 0.0, -scene_radius * 1.3, 0.0],
            probe_box_max: [
                scene_radius * 1.3,
                scene_radius * 2.0,
                scene_radius * 1.3,
                0.0,
            ],
        };
        let globals_offset = frame as u64 * GLOBALS_SLICE;
        globals_buffer.write_at(globals_offset, globals_bytes(&globals))?;

        // Build the deferred render graph:
        //   gbuffer (4 MRT + depth) -> lighting (HDR) -> tonemap (backbuffer) -> ui
        // Phase 8 M4: manage the path tracer's persistent accumulation buffer and
        // reset key BEFORE building the render graph — the fallible buffer
        // (re)allocation must not sit on a `?` early-return path while the graph
        // holds borrows of transient resources.
        let pt_active =
            path_trace && !rt_debug && rt_path_pipeline.is_some() && rt_instance_table.is_some();
        // The path tracer uses the Cornell scene (fixed front camera) when toggled,
        // else the orbiting open scene. `pt_eye` / `pt_inv_vp` feed the trace rays.
        let use_cornell = pt_active && cornell && cornell_scene.is_some();
        let (pt_eye, pt_inv_vp) = if use_cornell {
            let c_eye = Vec3::new(0.0, 1.0, 3.2);
            let c_view = Mat4::look_at_rh(c_eye, Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
            let mut c_proj =
                Mat4::perspective_rh(40f32.to_radians(), cw as f32 / ch as f32, 0.05, 100.0);
            if backend == BackendKind::Vulkan {
                c_proj.y_axis.y *= -1.0;
            }
            (c_eye, (c_proj * c_view).inverse().to_cols_array())
        } else {
            (eye, inv_view_proj)
        };
        if pt_active {
            // Switch the bound TLAS when toggling scenes (rare → wait_idle is fine).
            if bound_cornell != use_cornell {
                device.wait_idle()?;
                if use_cornell {
                    device.bind_tlas(cornell_scene.as_ref().unwrap());
                } else if let Some(s) = rt_scene.as_ref() {
                    device.bind_tlas(s);
                }
                bound_cornell = use_cornell;
                accum_frame = 0;
            }
            if accum_extent != (cw, ch) {
                device.wait_idle()?;
                path_accum = Some(device.create_storage_buffer(&StorageBufferDesc {
                    size: (cw as u64) * (ch as u64) * 16,
                    stride: 16,
                    indirect: false,
                })?);
                accum_extent = (cw, ch);
                accum_frame = 0;
            }
            let key = [
                pt_eye.x.to_bits(),
                pt_eye.y.to_bits(),
                pt_eye.z.to_bits(),
                sun_dir[0].to_bits(),
                sun_dir[1].to_bits(),
                sun_dir[2].to_bits(),
                sun_intensity.to_bits(),
                (cw.wrapping_mul(0x9E37_79B1).wrapping_add(ch)) ^ (use_cornell as u32),
            ];
            if last_pt_key != Some(key) {
                accum_frame = 0;
                last_pt_key = Some(key);
            }
        }

        let extent = Extent2D::new(cw, ch);
        let mut graph = RenderGraph::new();
        let backbuffer = graph.import_backbuffer(swapchain.format(), extent);
        let g_albedo = graph.create_color("g_albedo", GB_ALBEDO_FMT, extent);
        let g_normal = graph.create_color("g_normal", GB_NORMAL_FMT, extent);
        let g_material = graph.create_color("g_material", GB_MATERIAL_FMT, extent);
        let g_position = graph.create_color("g_position", GB_POSITION_FMT, extent);
        let g_depth = graph.create_depth("g_depth", extent);
        let shadow_map = graph.create_depth("shadow_map", Extent2D::new(SHADOW_SIZE, SHADOW_SIZE));
        let hdr = graph.create_color("hdr", HDR_FORMAT, extent);
        // Phase 7: compute post writes the blurred HDR into a storage image that
        // the tonemap pass samples instead of the raw `hdr` target.
        let hdr_post = if compute_post {
            Some(graph.create_storage_image("hdr_post", HDR_FORMAT, extent))
        } else {
            None
        };

        let sky = ClearColor {
            r: ambient,
            g: ambient,
            b: ambient,
            a: 1.0,
        };
        let zero = ClearColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        };
        // Shadow pass: rasterize the mesh from the light's POV into the depth-only
        // shadow map (the lighting pass samples it).
        graph.add_pass(
            PassInfo {
                name: "shadow",
                colors: vec![],
                depth: Some(shadow_map),
                reads: vec![],
            },
            |ctx| {
                // Scene objects are the shadow casters (the ground is a flat
                // receiver). Each draws with its own light-space MVP.
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&shadow_pipeline);
                for obj in &scene {
                    if !obj.casts_shadow {
                        continue;
                    }
                    let lmvp = (light_vp * obj.transform).to_cols_array();
                    cmd.push_constants(&mat4_bytes(&lmvp));
                    cmd.bind_vertex_buffer(&obj.vbuf, 32);
                    cmd.bind_index_buffer(&obj.ibuf, true);
                    cmd.draw_indexed(obj.index_count, 0, 0);
                }
                Ok(())
            },
        );
        graph.add_pass(
            PassInfo {
                name: "gbuffer",
                colors: vec![
                    (g_albedo, Some(sky)),
                    (g_normal, Some(ClearColor::BLACK)),
                    (g_material, Some(ClearColor::BLACK)),
                    (g_position, Some(zero)), // alpha 0 marks "no geometry"
                ],
                depth: Some(g_depth),
                reads: vec![],
            },
            |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&gbuffer_pipeline);
                // Each scene object with its own transform + PBR material. The UI
                // material override (for IBL inspection) replaces metallic/roughness
                // and drops the m/r texture so the factors apply directly.
                for obj in &scene {
                    let obj_mvp = (view_proj * obj.transform).to_cols_array();
                    let (m, rgh, mr_tex) = if override_material {
                        (metallic_override, roughness_override, NO_TEXTURE)
                    } else {
                        (obj.metallic, obj.roughness, obj.tex[1])
                    };
                    cmd.push_constants(&gbuffer_push(
                        obj_mvp,
                        obj.base_color,
                        m,
                        rgh,
                        [obj.tex[0], mr_tex, obj.tex[2], obj.tex[3]],
                    ));
                    cmd.bind_vertex_buffer(&obj.vbuf, 32);
                    cmd.bind_index_buffer(&obj.ibuf, true);
                    cmd.draw_indexed(obj.index_count, 0, 0);
                }
                // Ground plane (plain matte material, no textures).
                cmd.push_constants(&gbuffer_push(
                    view_proj.to_cols_array(),
                    [0.8, 0.8, 0.8, 1.0],
                    0.0,
                    0.9,
                    [NO_TEXTURE; 4],
                ));
                cmd.bind_vertex_buffer(&ground_vbuf, 32);
                cmd.bind_index_buffer(&ground_ibuf, true);
                cmd.draw_indexed(ground_count, 0, 0);
                Ok(())
            },
        );
        graph.add_pass(
            PassInfo {
                name: "lighting",
                colors: vec![(hdr, Some(ClearColor::BLACK))],
                depth: None,
                reads: vec![g_albedo, g_normal, g_material, g_position, shadow_map],
            },
            |ctx| {
                let indices = [
                    ctx.sampled_index(g_albedo),
                    ctx.sampled_index(g_normal),
                    ctx.sampled_index(g_material),
                    ctx.sampled_index(g_position),
                ];
                let shadow_index = ctx.sampled_index(shadow_map);
                let cmd = ctx.cmd();
                cmd.set_globals(&globals_buffer, globals_offset);
                cmd.bind_graphics_pipeline(&pbr_pipeline);
                cmd.push_constants(&pbr_push(indices, flip_y, shadow_index));
                cmd.draw(3, 1);
                Ok(())
            },
        );
        // Phase 7 compute post: blur `hdr` into the `hdr_post` storage image.
        if let Some(hdr_post) = hdr_post {
            let post_cp = &post_compute_pipeline;
            graph.add_compute_pass(
                ComputePassInfo {
                    name: "post_compute",
                    storage_writes: vec![hdr_post],
                    reads: vec![hdr],
                },
                move |ctx| {
                    let in_index = ctx.sampled_index(hdr);
                    let out_index = ctx.storage_index(hdr_post);
                    let cmd = ctx.cmd();
                    cmd.bind_compute_pipeline(post_cp);
                    cmd.push_constants_compute(&post_compute_push(in_index, out_index, cw, ch));
                    cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                    Ok(())
                },
            );
        }
        // Phase 7 GPU particles: a compute pass advances the persistent particle
        // buffer; an external graph resource sequences it before the draw pass.
        let particles_ext = if particles_on {
            Some(graph.import_external("particles"))
        } else {
            None
        };
        // This frame's ping-pong buffer indices (read the previous write). Captured
        // before the end-of-frame `advance()` so the async-submit path and the draw
        // pass all reference the same pair.
        let particle_read = particles.read_index();
        let particle_write = particles.write_index();
        // Run the sim on the async-compute queue this frame? (Else it's a graph
        // compute pass on the graphics queue, below.) Decided once here so the
        // compute-cmd recording + submit path after the graph match. The async
        // recording itself happens after `graph.execute` (the compute command
        // buffer is independent of the graphics graph).
        let async_sim = particles_on && async_compute_supported && async_compute_on;
        if let (false, Some(particles_ext)) = (async_sim, particles_ext) {
            particles.record_sim(&mut graph, particles_ext, sim_dt, elapsed);
        }

        // Phase 7 GPU culling: reset the indirect args, frustum-cull the instance
        // grid into a visible list + draw count, then draw indirectly. The args and
        // visible buffers are external resources sequencing the three passes.
        // A compact grid floating above the scene, so the scene stays visible and
        // orbiting the camera culls cubes off the frustum edges.
        let grid = CullGrid {
            spacing: scene_radius * 0.14,
            height: scene_radius * 1.15,
            cube_scale: scene_radius * 0.045,
            cube_radius: scene_radius * 0.045 * 0.5 * 3.0_f32.sqrt(),
        };
        let cull_res = if gpu_cull {
            Some(CullSystem::import(&mut graph))
        } else {
            None
        };
        if let Some((args_ext, visible_ext)) = cull_res {
            cull.record_cull(
                &mut graph,
                args_ext,
                visible_ext,
                frustum_planes(cull_view_proj),
                &grid,
            );
        }

        // Phase 8 ray tracing: M4 inline path tracer (default) or the M3 trace viz
        // (debug). The chosen compute pass writes a storage image the tonemap pass
        // displays in place of the rasterized HDR. (`pt_active` + the accumulation
        // buffer were prepared above, before the graph was built.)
        let rt_on = path_trace && (rt_path_pipeline.is_some() || rt_trace_pipeline.is_some());
        let rt_out = if rt_on {
            Some(graph.create_storage_image("rt_out", HDR_FORMAT, extent))
        } else {
            None
        };
        if let Some(rt_out) = rt_out {
            if pt_active {
                let rt_pipe = rt_path_pipeline.as_ref().unwrap();
                // M5: when enabled, drive the same path tracer through the RT pipeline
                // + SBT instead of the inline compute ray query (`None` = inline).
                let rt_pt = if path_trace_pipeline {
                    rt_pt_pipeline.as_ref()
                } else {
                    None
                };
                // Index only (no borrow held into the graph closure — that would
                // over-extend the graph's lifetime vs. the transient resources).
                let accum_index = path_accum.as_ref().unwrap().storage_index();
                let inst_index = if use_cornell {
                    cornell_table.as_ref().unwrap().storage_index()
                } else {
                    rt_instance_table.as_ref().unwrap().storage_index()
                };
                let inst_count = if use_cornell {
                    cornell_instance_count
                } else {
                    rt_instance_count
                };
                // External resource so the graph orders the accumulation write (and
                // inserts a barrier before the next frame's read).
                let accum_ext = graph.import_external("rt_accum");
                let inv_vp = pt_inv_vp;
                let cam = pt_eye;
                let sun = sun_dir;
                let sun_i = sun_intensity;
                // bit0 = Vulkan Y-flip, bit1 = Cornell env mode (no sun, black bg).
                let flip = flip_y | if use_cornell { 2 } else { 0 };
                let frame_idx = accum_frame;
                let spp = path_spp;
                graph.add_compute_pass(
                    ComputePassInfo {
                        name: "rt_path",
                        storage_writes: vec![rt_out, accum_ext],
                        reads: vec![],
                    },
                    move |ctx| {
                        let out_index = ctx.storage_index(rt_out);
                        let cmd = ctx.cmd();
                        let push = rt_path_push(
                            &inv_vp,
                            cam,
                            sun,
                            sun_i,
                            out_index,
                            accum_index,
                            inst_index,
                            inst_count,
                            frame_idx,
                            cw,
                            ch,
                            flip,
                            spp,
                        );
                        if let Some(rt_pt) = rt_pt {
                            // Full RT pipeline path (raygen/miss/hit + SBT).
                            cmd.bind_raytracing_pipeline(rt_pt);
                            cmd.push_constants_rt(&push);
                            cmd.trace_rays(rt_pt, cw, ch);
                        } else {
                            // Inline ray-query compute path.
                            cmd.bind_compute_pipeline(rt_pipe);
                            cmd.push_constants_compute(&push);
                            cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                        }
                        Ok(())
                    },
                );
                accum_frame += 1;
            } else if let Some(rt_pipe) = rt_trace_pipeline.as_ref() {
                let inv_vp = inv_view_proj;
                let cam = eye;
                let sun = sun_dir;
                let flip = flip_y;
                graph.add_compute_pass(
                    ComputePassInfo {
                        name: "rt_trace",
                        storage_writes: vec![rt_out],
                        reads: vec![],
                    },
                    move |ctx| {
                        let out_index = ctx.storage_index(rt_out);
                        let cmd = ctx.cmd();
                        cmd.bind_compute_pipeline(rt_pipe);
                        cmd.push_constants_compute(&rt_trace_push(
                            &inv_vp, cam, sun, out_index, cw, ch, flip,
                        ));
                        cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                        Ok(())
                    },
                );
            }
        }

        // Phase 11 Stage A: compute software ray trace of the analytic SDF scene,
        // written to a storage image the tonemap pass displays in place of the HDR
        // (mirrors the M3 `rt_trace` viz path). Only when the HW-RT path is off.
        let sdf_out = if sdf_trace && rt_out.is_none() {
            Some(gdf.record_sdf_trace(
                &mut graph,
                extent,
                inv_view_proj,
                eye,
                sun_dir,
                sun_intensity,
                cw,
                ch,
                flip_y,
            ))
        } else {
            None
        };

        // Phase 11 Stage B (B1): 3D volume smoke test — fill a storage volume, then
        // view a trilinear-sampled Z slice. Only when the other replacements are off.
        let vol_out = if volume_test && rt_out.is_none() && sdf_out.is_none() {
            Some(gdf.record_volume_test(&mut graph, extent, cw, ch))
        } else {
            None
        };

        // Phase 11 Stage B (B2): bake a mesh's signed-distance field into the volume,
        // then view a slice through the same `volume_view` pass B1 uses — so the baked
        // sphere is pixel-comparable against B1's analytic fill (and VK ≡ DX). The bake
        // is O(voxels*tris): run it once (`sdf_bake_done`) and only re-view afterwards.
        let bake_out = if sdf_bake && rt_out.is_none() && sdf_out.is_none() && vol_out.is_none() {
            let out = gdf.record_bake_view(&mut graph, extent, cw, ch, !sdf_bake_done);
            sdf_bake_done = true;
            Some(out)
        } else {
            None
        };

        // Phase 11 Stage B (B3): bake the per-mesh SDF, merge its instances into the
        // global distance field, then view a slice — all through reused passes (B2 bake,
        // this merge, B1 view). Bake + merge run once (`gdf_merge_done`); later frames
        // re-view the persistent GDF. Pixel-comparable across backends (VK ≡ DX).
        let gdf_out = if gdf_merge
            && rt_out.is_none()
            && sdf_out.is_none()
            && vol_out.is_none()
            && bake_out.is_none()
        {
            let out = gdf.record_gdf_view(&mut graph, extent, cw, ch, !gdf_merge_done);
            gdf_merge_done = true;
            Some(out)
        } else {
            None
        };

        // Phase 11 Stage B (B4): SW ray trace the merged GDF. Ensures the GDF is built
        // (bake + merge, once — shared `gdf_merge_done` with the B3 view), then sphere-
        // traces it from a fixed camera over the unit-cube scene. `P11_GDF_ANALYTIC`
        // swaps in the analytic field for the correctness reference. VK ≡ DX.
        let gdf_trace_out = if gdf_trace
            && rt_out.is_none()
            && sdf_out.is_none()
            && vol_out.is_none()
            && bake_out.is_none()
            && gdf_out.is_none()
        {
            let out = gdf.record_gdf_trace(
                &mut graph,
                extent,
                cw,
                ch,
                sun_dir,
                sun_intensity,
                flip_y,
                backend == BackendKind::Vulkan,
                gdf_trace_analytic,
                !gdf_merge_done,
            );
            gdf_merge_done = true;
            Some(out)
        } else {
            None
        };

        // Tonemap samples the RT output (M4 path trace / M3 trace viz) if active,
        // else the SW-RT SDF trace, else the Stage-B volume slice, else compute-post,
        // else HDR.
        let tonemap_src = rt_out
            .or(sdf_out)
            .or(vol_out)
            .or(bake_out)
            .or(gdf_out)
            .or(gdf_trace_out)
            .or(hdr_post)
            .unwrap_or(hdr);
        // The rasterized HDR already bakes exposure into the lighting pass; the
        // path-traced + SW-RT outputs carry raw scene radiance, so apply the camera
        // exposure here before the filmic curve (else the bright sky + sun blow out).
        let tm_exposure = if pt_active || sdf_out.is_some() || gdf_trace_out.is_some() {
            exposure
        } else {
            1.0
        };
        graph.add_pass(
            PassInfo {
                name: "tonemap",
                colors: vec![(backbuffer, Some(ClearColor::BLACK))],
                depth: None,
                reads: vec![tonemap_src],
            },
            |ctx| {
                let hdr_index = ctx.sampled_index(tonemap_src);
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&post_pipeline);
                cmd.push_constants(&post_push(hdr_index, post_mode as u32, flip_y, tm_exposure));
                cmd.draw(3, 1);
                Ok(())
            },
        );
        // Phase 7 GPU-culling draw: indirect, instanced render of the visible cube
        // grid over the tonemapped image, with its own depth buffer.
        if let Some((args_ext, visible_ext)) = cull_res {
            cull.record_draw(
                &mut graph,
                backbuffer,
                extent,
                args_ext,
                visible_ext,
                view_proj.to_cols_array(),
                sun_dir,
                &grid,
            );
        }

        // Phase 7 particle draw: instanced billboards composited over the tonemapped
        // image (alpha blend), reading the compute-updated buffer in the vertex
        // stage. Declared after tonemap so the WAW on the backbuffer orders it last.
        if let Some(particles_ext) = particles_ext {
            particles.record_draw(
                &mut graph,
                backbuffer,
                particles_ext,
                view_proj.to_cols_array(),
                cam_right,
                cam_up,
            );
        }

        if include_ui {
            graph.add_pass(
                PassInfo {
                    name: "ui",
                    colors: vec![(backbuffer, None)],
                    depth: None,
                    reads: vec![],
                },
                |ctx| gui.render(&device, ctx.cmd(), frame),
            );
        }
        let mut profiler = profiler_on.then(|| GraphProfiler::new(&query_heaps[frame]));
        graph.execute(
            &device,
            &mut pools[frame],
            cmd,
            &swapchain,
            image_index,
            aliasing,
            profiler.as_mut(),
        )?;
        // Remember this slot's scheduled pass names so the next readback (after
        // this frame's fence) can pair them with the timestamp boundaries.
        slot_pass_names[frame] = match &profiler {
            Some(p) => p.names.clone(),
            None => Vec::new(),
        };

        // For a screenshot, copy the just-rendered backbuffer into a readback
        // buffer in the same command buffer (before it ends).
        let readback = if capture_this_frame.is_some() {
            let layout = device.swapchain_readback_layout(&swapchain);
            let buf = device.create_buffer(&BufferDesc {
                size: layout.size,
                usage: BufferUsage::Readback,
            })?;
            cmd.copy_swapchain_to_buffer(&swapchain, image_index, &buf);
            Some((buf, layout))
        } else {
            None
        };

        cmd.end()?;

        let signal = &render_finished[image_index as usize];
        if async_sim {
            // Record the particle sim into this frame's compute command buffer and
            // run it on the compute queue (overlapping graphics), signaling
            // `compute_done`; the graphics submit GPU-waits on it so the particle
            // draw's vertex-stage read sees the freshly written buffer.
            let ccmd = &compute_command_buffers[frame];
            ccmd.begin()?;
            ccmd.bind_compute_pipeline(particles.sim_pipeline());
            ccmd.push_constants_compute(&particle_sim_push(
                particles.buffer_storage_index(particle_read),
                particles.buffer_storage_index(particle_write),
                PARTICLE_COUNT as u32,
                sim_dt,
                elapsed,
                0,
            ));
            ccmd.dispatch((PARTICLE_COUNT as u32).div_ceil(64), 1, 1);
            ccmd.end()?;
            compute_queue.submit(ccmd, &compute_done[frame])?;
            queue.submit_async(
                cmd,
                &image_available[frame],
                &compute_done[frame],
                signal,
                fence,
            )?;
        } else {
            queue.submit(cmd, &image_available[frame], signal, fence)?;
        }
        // Swap the particle ping-pong parity for the next simulated frame (deferred to
        // here so the graph's `&self` borrows have ended — see `particle.rs`).
        if particles_on {
            particles.advance();
        }

        // Wait for the GPU (copy included), read the buffer back, and save a PNG.
        if let (Some(cap), Some((buf, layout))) = (capture_this_frame.as_ref(), readback.as_ref()) {
            fence.wait()?;
            let mut bytes = vec![0u8; layout.size as usize];
            buf.read_into(&mut bytes)?;
            save_screenshot(&cap.path, &bytes, layout)?;
            info!(
                "saved screenshot {} ({}x{}, ui={})",
                cap.path, layout.width, layout.height, cap.include_ui
            );
        }

        if queue.present(&swapchain, image_index, signal)? {
            needs_recreate = true;
        }
        frame = (frame + 1) % FRAMES_IN_FLIGHT;
        frame_no += 1;

        // In screenshot mode, exit once every requested capture is saved.
        if screenshot_mode && frame_no >= warmup + captures.len() as u64 {
            break;
        }
    }

    device.wait_idle()?;
    info!("shutting down");
    Ok(())
}

/// View the globals struct as bytes for upload.
fn globals_bytes(g: &Globals) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            g as *const Globals as *const u8,
            std::mem::size_of::<Globals>(),
        )
    }
}

fn normalize3(v: [f32; 3]) -> [f32; 4] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-5);
    [v[0] / len, v[1] / len, v[2] / len, 0.0]
}

/// Pack the G-buffer push block: mvp(64) + base_color(16) + metallic/roughness(16)
/// + texture indices u32x4 (16) = 112 bytes.
fn gbuffer_push(
    mvp: [f32; 16],
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    tex: [u32; 4],
) -> [u8; 112] {
    let mut pc = [0u8; 112];
    for (i, f) in mvp.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    for (i, f) in base_color.iter().enumerate() {
        let o = 64 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    pc[80..84].copy_from_slice(&metallic.to_le_bytes());
    pc[84..88].copy_from_slice(&roughness.to_le_bytes());
    for (i, t) in tex.iter().enumerate() {
        let o = 96 + i * 4;
        pc[o..o + 4].copy_from_slice(&t.to_le_bytes());
    }
    pc
}

/// Pack the lighting push block: 4 G-buffer indices + flip_y + shadow_index
/// (24 bytes).
fn pbr_push(indices: [u32; 4], flip_y: u32, shadow_index: u32) -> [u8; 24] {
    let mut pc = [0u8; 24];
    for (i, v) in indices.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[16..20].copy_from_slice(&flip_y.to_le_bytes());
    pc[20..24].copy_from_slice(&shadow_index.to_le_bytes());
    pc
}

/// View a column-major 4x4 matrix as push-constant bytes.
fn mat4_bytes(m: &[f32; 16]) -> [u8; 64] {
    let mut pc = [0u8; 64];
    for (i, f) in m.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    pc
}

/// Directional-light view-projection: an orthographic box centered on `center`,
/// looking from the sun's direction toward it. Returned column-major (glam's
/// `to_cols_array`), matching the shader's `mul(M, v)` convention. No Vulkan
/// Y-flip — the pbr shader handles the per-backend shadow-UV flip.
fn light_view_proj(sun_dir: [f32; 3], center: Vec3, radius: f32) -> Mat4 {
    let dir = Vec3::new(sun_dir[0], sun_dir[1], sun_dir[2]).normalize_or_zero();
    let dir = if dir == Vec3::ZERO { Vec3::Y } else { dir };
    let dist = radius * 4.0;
    let light_pos = center + dir * dist;
    // Avoid a degenerate up vector when the light is near-vertical.
    let up = if dir.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(light_pos, center, up);
    let half = radius * 1.6;
    let proj = Mat4::orthographic_rh(-half, half, -half, half, 0.1, dist + radius * 2.0);
    proj * view
}
