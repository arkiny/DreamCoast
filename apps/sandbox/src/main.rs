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

use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_core::init_logging;
use dreamcoast_gui::{Gui, imgui};
use dreamcoast_platform::Window;
use dreamcoast_render::{GraphProfiler, PassInfo, RenderGraph, ResourcePool};
use rhi::{
    BackendKind, Buffer, BufferDesc, BufferUsage, Extent2D, Format, Instance, InstanceDesc,
    PresentMode, SwapchainDesc, Texture,
};
use tracing::info;

mod app;
mod cull;
mod deferred;
mod gdf;
mod ibl;
mod mesh;
mod particle;
mod push;
mod rt;
mod smoketest;
use app::*;
use cull::*;
use deferred::*;
use gdf::*;
use ibl::*;
use mesh::*;
use particle::*;
use push::*;
use rt::*;
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

    // Deferred-PBR backbone (see `deferred.rs`): the shadow / G-buffer / lighting /
    // tonemap graphics pipelines, the compute post-process pipeline, and the per-frame
    // globals uniform buffer.
    let deferred = DeferredRenderer::new(&device, backend, swapchain.format())?;

    // Phase 11 software ray tracing + global distance field (Stage A analytic trace,
    // Stage B volumes / SDF bake / GDF merge / GDF trace). See `gdf.rs`.
    let gdf = GdfSystem::new(&device, backend, compute_supported)?;

    // GPU particle system (Phase 7): a persistent ping-pong buffer pair advanced by
    // a compute pass and drawn as instanced billboards (see `particle.rs`). Seeds
    // both buffers on construction.
    let mut particles =
        ParticleSystem::new(&device, backend, compute_supported, swapchain.format())?;

    // GPU frustum culling (Phase 7): a compute pass tests a cube instance grid
    // against the frustum and writes an indirect draw; the draw renders only the
    // visible instances (see `cull.rs`).
    let cull = CullSystem::new(&device, backend, compute_supported, swapchain.format())?;

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

    // Hardware ray tracing (Phase 8): BLAS/TLAS over the sample scene + ground, the
    // path tracer's per-instance geometry table, and the alternate Cornell-box scene
    // — plus the M3/M4/M5 RT pipelines. See `rt.rs`. The sample scene's TLAS is bound
    // on construction (the startup default). The instance table's mesh order MUST
    // match the TLAS custom_index order (scene objects, then ground).
    let mut rt = RtSystem::new(
        &device,
        backend,
        &scene,
        &[&model, &sphere, &sphere, &cube],
        &ground,
        &ground_vbuf,
        &ground_ibuf,
        ground_count,
    )?;

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
    let mut path_trace =
        rt.has_trace() && rt.has_scene() && std::env::var_os("P8_PATHTRACE").is_some();
    // Debug mode: show the M3 single-bounce trace viz (instance color + RT shadow)
    // instead of the M4 path tracer. Headless toggle via `P8_RT_DEBUG`.
    let mut rt_debug = device.has_raytracing() && std::env::var_os("P8_RT_DEBUG").is_some();
    // Cornell-box scene for the path tracer (strong color bleeding). Headless
    // toggle via `P8_CORNELL`; uses a fixed front-facing camera.
    let mut cornell = rt.has_cornell() && std::env::var_os("P8_CORNELL").is_some();
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
    // `pt_pipeline` is only built when the pipeline was requested, so its presence
    // alone is the default-on condition.
    let mut path_trace_pipeline = rt.has_pt_pipeline();
    // Samples per path-trace dispatch (accumulated progressively across frames).
    let path_spp: u32 = 8;
    // Real-time environment capture: re-capture the env chain every frame (so the
    // sky/IBL track the live sun); when off, re-capture only when the sun changes.
    let mut realtime_env = true;
    // Multi-bounce: shade captured surfaces with IBL from the previous frame's
    // cube set, so reflective surfaces appear reflective inside reflections.
    let mut multibounce = true;

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

    // Image-based lighting (see `ibl.rs`): the procedural-sky / capture / irradiance
    // / prefilter / BRDF pipelines, the ping-pong environment cube sets, the capture
    // depth and the BRDF LUT (generated once on construction). The env chain is
    // (re)captured per frame inside the render loop via `maybe_capture`.
    let mut ibl = IblSystem::new(&device, backend, &queue, flip_y, sun_dir, sun_intensity)?;
    // Seed both cube sets once (single-bounce, no previous environment) so the first
    // multi-bounce frame reads valid data instead of uninitialized memory. Uses an
    // approximate camera; the render loop immediately recaptures with the live one.
    let boot_eye = Vec3::new(0.0, model_radius * 0.6, 0.0)
        + Vec3::new(scene_radius * 1.6, scene_radius * 0.55, 0.0);
    ibl.seed(
        &device,
        &queue,
        &scene,
        &ground_vbuf,
        &ground_ibuf,
        ground_count,
        boot_eye,
        sun_dir,
        sun_intensity,
        ambient,
        flip_y,
        backend == BackendKind::Vulkan,
    )?;

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
    // Path-tracer progressive accumulation (Phase 8 M4) lives in `rt`; extra headless
    // warmup frames let the static-camera screenshot converge.
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

                    if rt.has_path() && rt.has_scene() {
                        if ui.collapsing_header("Ray tracing (Phase 8)", TreeNodeFlags::empty()) {
                            ui.checkbox("Path trace (inline ray query)", &mut path_trace);
                            if path_trace {
                                ui.checkbox("  - debug: instance + shadow viz", &mut rt_debug);
                                if !rt_debug {
                                    if rt.has_cornell() {
                                        ui.checkbox("  - Cornell box", &mut cornell);
                                    }
                                    if rt.has_pt_pipeline() {
                                        ui.checkbox(
                                            "  - pipeline + SBT (vs inline)",
                                            &mut path_trace_pipeline,
                                        );
                                    }
                                    ui.text(format!(
                                        "  - {} spp accumulated ({})",
                                        rt.accum_frame().saturating_mul(path_spp),
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

        // (Re)capture the environment into the "write" cube set before the main graph
        // samples it (see `ibl.rs`). The reflection probe is fixed at the scene centre
        // (`focus`), NOT the camera, to avoid per-surface parallax error.
        ibl.maybe_capture(
            cmd,
            realtime_env,
            multibounce,
            &scene,
            &ground_vbuf,
            &ground_ibuf,
            ground_count,
            focus,
            sun_dir,
            sun_intensity,
            ambient,
            flip_y,
            backend == BackendKind::Vulkan,
        );

        // The main lighting pass samples the most recently written set.
        let ibl_indices = ibl.lighting_indices();

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
        deferred.write_globals(globals_offset, globals_bytes(&globals))?;

        // Build the deferred render graph:
        //   gbuffer (4 MRT + depth) -> lighting (HDR) -> tonemap (backbuffer) -> ui
        // Phase 8 M4: manage the path tracer's persistent accumulation buffer and
        // reset key BEFORE building the render graph — the fallible buffer
        // (re)allocation must not sit on a `?` early-return path while the graph
        // holds borrows of transient resources.
        let pt_active =
            path_trace && !rt_debug && rt.has_path() && rt.has_instance_table();
        // The path tracer uses the Cornell scene (fixed front camera) when toggled,
        // else the orbiting open scene. `pt_eye` / `pt_inv_vp` feed the trace rays.
        let use_cornell = pt_active && cornell && rt.has_cornell();
        let (pt_eye, pt_inv_vp) = if use_cornell {
            RtSystem::cornell_camera(cw, ch, backend == BackendKind::Vulkan)
        } else {
            (eye, inv_view_proj)
        };
        rt.prepare(
            &device,
            pt_active,
            use_cornell,
            cw,
            ch,
            pt_eye,
            sun_dir,
            sun_intensity,
        )?;

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
        let gbuf = GBufferTargets {
            albedo: g_albedo,
            normal: g_normal,
            material: g_material,
            position: g_position,
            depth: g_depth,
        };

        // Deferred backbone (see `deferred.rs`): shadow -> gbuffer -> lighting (HDR),
        // then the optional compute-post blur.
        deferred.record_shadow(&mut graph, shadow_map, &scene, light_vp);
        deferred.record_gbuffer(
            &mut graph,
            gbuf,
            &scene,
            &ground_vbuf,
            &ground_ibuf,
            ground_count,
            view_proj,
            ambient,
            override_material,
            metallic_override,
            roughness_override,
        );
        deferred.record_lighting(&mut graph, hdr, gbuf, shadow_map, globals_offset, flip_y);
        if let Some(hdr_post) = hdr_post {
            deferred.record_compute_post(&mut graph, hdr, hdr_post, cw, ch);
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
        let rt_on = path_trace && (rt.has_path() || rt.has_trace());
        let rt_out = if rt_on {
            Some(graph.create_storage_image("rt_out", HDR_FORMAT, extent))
        } else {
            None
        };
        if let Some(rt_out) = rt_out {
            if pt_active {
                rt.record_path(
                    &mut graph,
                    rt_out,
                    use_cornell,
                    path_trace_pipeline,
                    pt_inv_vp,
                    pt_eye,
                    sun_dir,
                    sun_intensity,
                    cw,
                    ch,
                    flip_y,
                    path_spp,
                );
            } else if rt.has_trace() {
                rt.record_trace(&mut graph, rt_out, inv_view_proj, eye, sun_dir, cw, ch, flip_y);
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
        deferred.record_tonemap(
            &mut graph,
            backbuffer,
            tonemap_src,
            post_mode as u32,
            flip_y,
            tm_exposure,
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
        // Bump the path tracer's progressive-accumulation counter (deferred here for the
        // same reason: the `record_path` pass borrowed `&rt` for the graph's lifetime).
        if pt_active {
            rt.advance_accum();
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
