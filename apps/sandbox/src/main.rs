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
//!
//! The per-feature GPU work lives in focused bundles (`deferred`, `ibl`, `rt`,
//! `gdf`, `particle`, `cull`); `App` owns those bundles plus the device / swapchain
//! / per-frame sync, with `App::new` doing setup and `App::frame` running one frame
//! of the loop. `run()` shrinks to window + device bring-up + `App::new` + the loop.

use std::time::Instant;

use dreamcoast_asset::MeshData;
use dreamcoast_core::glam::{Mat4, Quat, Vec3};
use dreamcoast_core::init_logging;
use dreamcoast_gui::{Gui, imgui};
use dreamcoast_platform::Window;
use dreamcoast_render::{GraphProfiler, PassInfo, RenderGraph, ResourceId, ResourcePool};
use rhi::{
    BackendKind, Buffer, BufferDesc, BufferUsage, CommandBuffer, ComputeQueue, Device, Extent2D,
    Fence, Format, Instance, InstanceDesc, PresentMode, QueryHeap, Queue, Semaphore, Swapchain,
    SwapchainDesc, Texture,
};
use tracing::info;

mod app;
mod camera;
mod clipmap;
mod cull;
mod deferred;
mod fuse;
mod gdf;
mod gi;
mod ibl;
mod level;
mod mesh;
mod particle;
mod push;
mod quality;
mod reflect;
mod registry;
mod rt;
mod smoketest;
mod taau;
mod world;
use app::*;
use cull::*;
use deferred::*;
use dreamcoast_scene::{LocalTransform, MeshInstance, World};
use gdf::*;
use gi::*;
use ibl::*;
use mesh::*;
use particle::*;
use push::*;
use reflect::*;
use registry::{GpuMesh, MaterialDesc, MaterialRegistry, MeshRegistry, build_scene};
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
/// The ground plane's linear albedo — the single source of truth shared by the
/// G-buffer ground draw (direct view) and the SW-RT GI / reflection re-light passes.
/// The ground is analytic (not in the per-voxel albedo volume), so those passes must
/// be told its material explicitly; sourcing it here keeps them from drifting (and
/// stops `albedo_at()` returning the nearest *object's* colour for a floor hit).
pub(crate) const GROUND_ALBEDO: [f32; 3] = [0.8, 0.8, 0.8];
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

const DEBUG_VIEWS: [&str; 11] = [
    "Lit",
    "Albedo",
    "Normal",
    "Metallic",
    "Roughness",
    "Position",
    "AO",
    "Direct",
    "IBL",
    "GDF AO",
    "GDF GI",
];
const POST_EFFECTS: [&str; 3] = ["None", "Grayscale", "Vignette"];

/// One materialized drawable in the sample scene: a shared GPU mesh + world
/// transform + resolved PBR material. Produced from the ECS draw list by
/// [`registry::build_scene`]; consumed by the rasterizer, RT, and GDF passes.
pub(crate) struct SceneObject {
    /// Shared uploaded geometry (vertex/index buffers + counts).
    pub(crate) mesh: std::rc::Rc<GpuMesh>,
    pub(crate) transform: Mat4,
    pub(crate) base_color: [f32; 4],
    pub(crate) metallic: f32,
    pub(crate) roughness: f32,
    /// base color, metallic-roughness, normal, emissive bindless indices
    /// (`NO_TEXTURE` if absent).
    pub(crate) tex: [u32; 4],
    pub(crate) casts_shadow: bool,
}

/// A level's lighting (sun + point lights), applied in the `Globals` assembly in
/// place of the gallery's hardcoded code-default lights. `None` keeps the gallery
/// defaults (byte-identical baseline).
struct LevelLighting {
    sun_dir: [f32; 3],
    sun_intensity: f32,
    point_pos: [[f32; 4]; 4],
    point_color: [[f32; 4]; 4],
    point_count: i32,
}

/// Build a [`LevelLighting`] from a level's environment + lights: the environment sun
/// (overridden by an explicit directional light), and up to 4 point lights.
fn level_lighting(level: &dreamcoast_asset::LevelData) -> LevelLighting {
    use dreamcoast_asset::level::LightKind;
    // A level authors a directional light's `vec` as the direction the light *travels*
    // (the glTF convention — "the sun shines down" = a downward vector). The renderer's
    // `sun_direction` is the direction *toward* the sun, so negate.
    let toward_sun = |v: [f32; 3]| [-v[0], -v[1], -v[2]];
    let env = level.environment;
    let mut sun_dir = toward_sun(env.sun_dir);
    let mut sun_intensity = env.sun_intensity;
    let mut point_pos = [[0.0f32; 4]; 4];
    let mut point_color = [[0.0f32; 4]; 4];
    let mut count = 0usize;
    for l in &level.lights {
        match l.kind {
            LightKind::Directional => {
                sun_dir = toward_sun(l.vec);
                sun_intensity = l.intensity;
            }
            LightKind::Point if count < 4 => {
                point_pos[count] = [l.vec[0], l.vec[1], l.vec[2], 0.0];
                point_color[count] = [l.color[0], l.color[1], l.color[2], l.intensity];
                count += 1;
            }
            LightKind::Point => {}
        }
    }
    LevelLighting {
        sun_dir,
        sun_intensity,
        point_pos,
        point_color,
        point_count: count as i32,
    }
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
    prev_view_proj: [f32; 16],  // world -> previous clip (Stage C7 SSR history reprojection)
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

    // Load the model via the cooked-asset pipeline (Phase 12 M1): a fresh
    // `.dcasset` loads directly (no glTF parse / texture decode); a miss cooks from
    // glTF and caches. Falls back to a procedural cube when neither exists. The
    // source path is resolved relative to the executable (not the cwd) so it loads
    // when launched from anywhere, not just the repo root.
    let model_ref = model_path();
    let model_path = app::resolve_asset_path(&model_ref);
    let cache_dir = app::cooked_cache_dir();
    // Phase 12 M3: opt-in BCn texture compression in the cook (default off keeps the
    // render byte-for-byte; GPU-native so there is no decompression cost at load).
    // `P12_TEX_COMPRESS=1|fast` → BC1/BC3 (size-first), `=high|bc7` → BC7
    // (quality-first). Data textures stay uncompressed either way.
    use dreamcoast_asset::cook::TexCompress;
    let compress_tex = match std::env::var("P12_TEX_COMPRESS").ok().as_deref() {
        Some("1") | Some("fast") => TexCompress::Fast,
        Some("high") | Some("bc7") => TexCompress::High,
        _ => TexCompress::Off,
    };
    let mut model = match dreamcoast_asset::cook::load_cooked(
        &model_path,
        &model_ref,
        &cache_dir,
        compress_tex,
    ) {
        Ok((m, outcome)) => {
            info!(
                "loaded {} ({outcome:?}): {} verts, {} indices",
                model_path.display(),
                m.vertices.len(),
                m.indices.len()
            );
            m
        }
        Err(e) => {
            info!(
                "no model at {} ({e}); using procedural cube",
                model_path.display()
            );
            dreamcoast_asset::unit_cube()
        }
    };
    // Normalize the model (recenter on origin, base at y=0, unit bounding radius)
    // so framing, ground, lights, and the shadow box use model-independent units.
    let bounds = normalize_on_ground(&mut model);
    let model_radius = bounds.radius;

    let title = format!("DreamCoast Sandbox — {backend:?}");
    // The window client size IS the final (present) resolution: render_scale fractions it for the
    // internal scene render, and the upscale (TAAU / tonemap) reconstructs back to this size.
    // `WINDOW_RES=WxH` opens at the display size for QHD/UHD output (clamped by the OS work area);
    // default HD 1280x720 keeps the headless screenshot baselines unchanged.
    let (win_w, win_h) = std::env::var("WINDOW_RES")
        .ok()
        .and_then(|s| {
            let (a, b) = s.split_once(['x', 'X', ','])?;
            Some((a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?))
        })
        .map(|(w, h)| (w.clamp(320, 7680), h.clamp(240, 4320)))
        .unwrap_or((1280, 720));
    let mut window = Window::new(&title, win_w, win_h)?;
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
    info!(
        "device capabilities: async_compute={}, raytracing={}",
        device.has_async_compute(),
        device.has_raytracing()
    );

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

    let mut app = App::new(
        window,
        instance,
        device,
        swapchain,
        backend,
        &model,
        model_radius,
        screenshot_mode,
        captures,
        validation_on,
    )?;
    app.run()
}

/// The full deferred-PBR application: owns the device / swapchain / per-frame sync
/// and every feature bundle, plus the UI + loop state. `new` does setup; `frame`
/// runs one iteration of the render loop.
struct App {
    // Window + device bring-up. `_instance` is kept alive only so the device (and
    // the window-derived surface) outlive it.
    window: Window,
    _instance: Instance,
    device: Device,
    queue: Queue,
    compute_queue: ComputeQueue,
    swapchain: Swapchain,
    backend: BackendKind,
    gui: Gui,

    // Feature bundles (see the per-module docs).
    deferred: DeferredRenderer,
    gdf: GdfSystem,
    gi: GiSystem,
    reflect: ReflectSystem,
    particles: ParticleSystem,
    cull: CullSystem,
    rt: RtSystem,
    ibl: IblSystem,

    // Scene. The ECS `world` (+ registries) is the single source of truth; the flat
    // `SceneObject` draw list is materialized from it each frame via `build_scene`.
    // `_textures` keeps the model's bindless textures alive.
    _textures: Vec<Texture>,
    world: World,
    mesh_registry: MeshRegistry,
    material_registry: MaterialRegistry,
    // Stage C level hot-swap: the discovered `.level` files, the loaded index, and a
    // pending selection from the UI (applied at the next frame's start). Empty unless
    // started in level mode (`LEVEL`).
    level_paths: Vec<String>,
    current_level: usize,
    pending_level: Option<usize>,
    // Stage D streaming: present in world mode (`WORLD`). Owns the level graph + the
    // resident chunk arenas; the per-frame draw list comes from it instead of `world`.
    streaming: Option<world::Streaming>,
    ground_vbuf: Buffer,
    ground_ibuf: Buffer,
    ground_count: u32,

    // Per-frame-in-flight resources + GPU profiler heaps.
    pools: Vec<ResourcePool>,
    command_buffers: Vec<CommandBuffer>,
    image_available: Vec<Semaphore>,
    in_flight: Vec<Fence>,
    compute_command_buffers: Vec<CommandBuffer>,
    compute_done: Vec<Semaphore>,
    /// Async-compute surface-cache relight: opt-in toggle + the two cross-frame "relight done"
    /// semaphores (compute frame N signals `cache_done[N%2]`; graphics frame N waits the previous,
    /// `cache_done[(N-1)%2]`, so the consumer reads of last frame's radiance are GPU-ordered).
    async_cache_on: bool,
    cache_done: Vec<Semaphore>,
    /// Per-fif fence the async relight signals, so its compute command buffer isn't re-recorded
    /// while still pending (the graphics fence only transitively covers the PREVIOUS frame's
    /// relight, not this frame's, on the cross-frame path).
    cache_compute_fence: Vec<Fence>,
    query_heaps: Vec<QueryHeap>,
    render_finished: Vec<Semaphore>,

    // Launch-time constants.
    flip_y: u32,
    model_radius: f32,
    scene_radius: f32,
    /// World-space focus point the orbit camera frames (scene AABB centre at native
    /// scale; the gallery's legacy focus otherwise).
    scene_center: Vec3,
    /// A level's authored camera (eye, target), applied as the initial view when the
    /// level defines a non-default camera. `None` falls back to the orbit framing.
    level_view: Option<(Vec3, Vec3)>,
    /// A level's lighting (sun + point lights), replacing the gallery's code-default
    /// lights. `None` keeps the gallery defaults (byte-identical).
    level_lighting: Option<LevelLighting>,
    /// True only for the hardcoded gallery (its legacy shadow framing is byte-identical;
    /// other modes frame the shadow box on the scene AABB).
    is_gallery: bool,
    screenshot_mode: bool,
    captures: Vec<Capture>,
    /// QHD/UHD measurement: `CAPTURE_SEQ=N` dumps N consecutive frames with the camera
    /// advancing a fixed deterministic step each frame (temporal-stability frame-to-frame
    /// diff). `None` = the normal fixed-camera capture. Measurement-only (never the parity
    /// baseline path).
    capture_seq: Option<u32>,
    validation_on: bool,
    async_compute_supported: bool,
    path_spp: u32,
    gdf_trace_analytic: bool,

    // UI-controlled lighting / feature state.
    sun_dir: [f32; 3],
    sun_intensity: f32,
    ambient: f32,
    exposure: f32,
    point_lights_on: bool,
    shadows_on: bool,
    shadow_bias: f32,
    // PCSS-lite penumbra scale (max soft-shadow radius in shadow-map UV); 0 = hard 3x3 PCF.
    shadow_softness: f32,
    /// Soft-shadow blocker/PCF tap count, written to `globals.shadow.w` (RenderQuality knob).
    /// Only the soft path (softness > 0) reads it; the shader clamps to [1, 16].
    shadow_taps: u32,
    /// Active RenderQuality tier (Stage D). The single selector that seeded the knob defaults
    /// below; shown in the UI. Individual env vars still override per knob.
    quality: quality::RenderQuality,
    override_material: bool,
    metallic_override: f32,
    roughness_override: f32,
    debug_view: usize,
    post_mode: usize,
    aliasing: bool,
    compute_post: bool,
    particles_on: bool,
    async_compute_on: bool,
    gpu_cull: bool,
    path_trace: bool,
    rt_debug: bool,
    cornell: bool,
    sdf_trace: bool,
    volume_test: bool,
    sdf_bake: bool,
    sdf_bake_done: bool,
    gdf_merge: bool,
    gdf_merge_done: bool,
    gdf_trace: bool,
    scene_gdf: bool,
    /// C2: GDF AO multiplied into the deferred ambient term (the first GDF feature in
    /// the real render path; C1's `scene_gdf` is a standalone trace viz).
    gdf_ao: bool,
    /// C3: GDF 1-bounce diffuse GI added to the deferred ambient term.
    gdf_gi: bool,
    /// C3 hemisphere rays per pixel.
    gi_spp: u32,
    /// Stage D2: surface-cache amortized-relight period (round-robin card budget; 1 = legacy
    /// every-frame, forced for the gallery anchor). Higher = cheaper `sdf_cache_light`.
    cache_relight_period: u32,
    /// Stage D2b: drive the relight budget by per-card camera-frustum visibility (off-screen
    /// cards relit far less). Off for the gallery anchor. Pure perf (on-screen image invariant).
    cache_feedback: bool,
    /// Stage D3: surface-cache relight indirect-gather rays/texel (gallery forced to legacy 8).
    cache_relight_spp: u32,
    /// Stage D3: C3 GI bounce-ray march step cap (gallery forced to legacy 64).
    gi_max_steps: u32,
    /// Stage D3: GGX reflection-ray march step cap (gallery forced to legacy 96).
    reflect_max_steps: u32,
    /// Stage D3: trace the GGX reflection at half resolution + bilateral upsample (gallery off).
    reflect_half_res: bool,
    /// Stage D1: trace the C3 GI at half resolution + joint-bilateral upsample (1/4 the rays).
    /// Forced off for the gallery anchor (full-res = byte-identical). Content scenes opt in by tier.
    gi_half_res: bool,
    /// C4: spatio-temporal denoise of the noisy C3 GI.
    gi_denoise: bool,
    /// Previous frame's view-projection (world -> clip) for C4 temporal reprojection.
    prev_view_proj: [f32; 16],
    /// QHD/UHD TAAU: previous frame's UNJITTERED view-projection (the stable grid the TAAU history
    /// lives on; the per-frame jitter must not enter the history reprojection).
    prev_view_proj_taau: [f32; 16],
    /// C5: screen-space reflections (viz; C7 will composite into lighting).
    gdf_ssr: bool,
    /// C6: GDF reflections (off-screen fallback viz; C7 composites SSR→GDF→sky).
    gdf_reflect: bool,
    /// C7: hybrid reflection composite (SSR over GDF / sky), viz toward IBL-specular replacement.
    gdf_hybrid: bool,
    /// C7c: feed the hybrid composite into the lighting specular (replaces the prefilter-cube
    /// IBL specular). The toggle that compares legacy captured-cube IBL vs the new SW-RT path.
    swrt_reflect: bool,
    /// C8a: use the per-voxel albedo volumes (real surface color) in the GDF GI / reflection
    /// re-light instead of a constant albedo. Off => achromatic (pre-C8a), for no-reg compare.
    gdf_color: bool,
    /// C8b1: viz the captured mesh-card surface-cache atlas (validation of card capture).
    cache_viz: bool,
    /// C8b3: GDF GI / reflection consumers sample the lit surface cache (multibounce radiance)
    /// instead of per-ray re-lighting. Drives the per-frame cache lighting too.
    surface_cache: bool,
    /// C8g: use the surface cache as the GDF reflection hit radiance (default on); ground hits
    /// (no cards) fall back to the per-ray re-light. Cheaper than the full GI cache above.
    reflect_cache: bool,
    /// Firefly clamp: bound the per-sample radiance in the reflection source / composite / GI
    /// gather so a bright specular pixel can't become a speckle. Off => unbounded (pre-clamp).
    firefly_clamp: bool,
    /// C8d reflection max-roughness threshold: the screen-space mirror SSR (accurate on-screen
    /// reflection) is used below this roughness and fades to the GDF prefilter above it (the
    /// mirror can't blur; a stochastic trace goes dark on sharp metals). `P11_REFLECT_MAX_ROUGHNESS`.
    reflect_max_roughness: f32,
    /// C8d: SSR trace mode. Default = full-res screen mirror (accurate on-screen reflection).
    /// `P11_SSR_STOCHASTIC=1` selects the half-res GGX-jittered trace + ratio-estimator resolve
    /// (the glossy path — cheaper, but it goes dark on sharp metals, so it is not the default).
    ssr_stochastic: bool,
    /// Shared bake-once latch for the world scene GDF (C1 trace + C2 AO + C3 GI read it).
    scene_gdf_baked: bool,
    /// C8a bake-once latch for the per-voxel albedo volumes (GI + reflection consumers share).
    scene_albedo_baked: bool,
    /// C8b1 capture-once latch for the surface cache (static geometry capture).
    scene_cache_captured: bool,
    /// C8b2 temporal reset for the cache lighting (true until the first lit frame).
    scene_cache_reset: bool,
    path_trace_pipeline: bool,
    realtime_env: bool,
    multibounce: bool,
    /// Deprecated legacy captured-cube IBL path (prefilter-cube specular + scene capture).
    /// Off by default — the SW-RT hybrid reflection + GDF GI are the default ambient.
    legacy_ibl: bool,

    /// QHD/UHD track: offscreen render extent override (`RENDER_RES=WxH`), decoupled from the
    /// window/swapchain. `None` => render at the swapchain extent (default, byte-identical).
    /// The scene passes (g-buffer → GDF → lighting → HDR) render here; tonemap downscales to the
    /// swapchain backbuffer. Lets headless perf measure true QHD/UHD regardless of display size.
    render_res: Option<(u32, u32)>,
    /// QHD/UHD track: internal render scale (fraction of the display extent), the production knob
    /// for dynamic resolution. `1.0` = native (byte-identical). Ignored when `render_res` (absolute)
    /// is set. `RENDER_SCALE` env / `quality.rs` tier.
    render_scale: f32,
    /// QHD/UHD track: temporal upsampler (TAAU) — reconstructs full-res from jittered low-res
    /// frames when the internal render extent is smaller than the output. `P_TAAU=0` disables.
    taau: taau::TaauSystem,
    taau_on: bool,
    /// QHD/UHD: camera sub-pixel jitter for TAAU. Default ON in the upscale path — the jitter is the
    /// super-sampling signal that reconstructs full-res detail (Halton(2,3), ±0.5px). It is now
    /// coordinated across every screen-space temporal accumulator: TAAU + GI denoiser + reflection
    /// resolve all reproject history sub-pixel-accurately (B1/B2), so the jitter resolves into sharp
    /// detail instead of shimmer. Only active when TAAU is (cw<sw); `P_TAAU_JITTER=0` forces it off.
    taau_jitter: bool,
    // Profiler UI state.
    profiler_on: bool,
    slot_pass_names: Vec<Vec<String>>,
    gpu_timings: Vec<(String, f32)>,

    // Loop bookkeeping.
    fif: usize,
    frame_no: u64,
    f2_prev: bool,
    needs_recreate: bool,
    last: Instant,
    elapsed: f32,
    angle: f32,
    // Diagnostic: tight orbit centred on one scene object (by index) for inspecting it
    // from all sides. `None` = normal whole-scene framing. `diag_pitch` = elevation.
    diag_obj: Option<usize>,
    diag_pitch: Option<f32>,
    // Stage 0 free-fly camera. `cam_mode` defaults to Orbit (the screenshot/parity
    // baseline); Tab toggles to Fly interactively. `fly` is lazily seeded from the
    // current orbit view on first switch so there is no jump.
    cam_mode: camera::CameraMode,
    fly: Option<camera::FlyCamera>,
    tab_prev: bool,
}

const VK_F2: u16 = 0x71;
const VK_TAB: u16 = 0x09;
const SCREENSHOT_WARMUP: u64 = 3;
// Path-trace screenshots need a long warmup so the static-camera accumulation
// converges before the frame is captured.
const PATHTRACE_WARMUP: u64 = 64;
// GI temporal accumulation likewise needs several frames to converge for a clean
// screenshot (the camera is held fixed while capturing).
const GI_DENOISE_WARMUP: u64 = 64;
/// TAAU sub-pixel jitter sequence length (Halton(2,3)); the history accumulates over this many
/// jittered frames to reconstruct full-res detail.
const TAAU_JITTER_LEN: u64 = 8;

/// Halton low-discrepancy sample (1-indexed) for the TAAU jitter sequence — uniform sub-pixel
/// coverage so the temporal accumulation resolves detail the low-res frame lacks.
fn halton(mut i: u32, base: u32) -> f32 {
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

impl App {
    #[allow(clippy::too_many_arguments)]
    fn new(
        window: Window,
        instance: Instance,
        device: Device,
        swapchain: Swapchain,
        backend: BackendKind,
        model: &MeshData,
        model_radius: f32,
        screenshot_mode: bool,
        captures: Vec<Capture>,
        validation_on: bool,
    ) -> anyhow::Result<Self> {
        let queue = device.queue();
        // Phase-7 compute (post blur / GPU particles / GPU culling) is implemented
        // on all three backends (Metal compute landed in M5).
        let compute_supported = true;

        // Deferred-PBR backbone (see `deferred.rs`): the shadow / G-buffer / lighting
        // / tonemap graphics pipelines, the compute post-process pipeline, and the
        // per-frame globals uniform buffer.
        let deferred = DeferredRenderer::new(&device, backend, swapchain.format())?;

        // Phase 11 software ray tracing + global distance field (Stage A analytic
        // trace, Stage B volumes / SDF bake / GDF merge / GDF trace, Stage C1 world
        // scene GDF). See `gdf.rs`. The scene GDF is registered after the scene is built.
        let mut gdf = GdfSystem::new(&device, backend, compute_supported)?;
        // Stage C GDF-lighting consumers (C2 AO, C3 GI, C4 denoise). See `gi.rs`.
        let gi = GiSystem::new(&device, backend, compute_supported)?;
        // Stage C reflection track (C5 SSR; C6/C7 later). See `reflect.rs`.
        let reflect = ReflectSystem::new(&device, backend, compute_supported)?;
        // QHD/UHD track: temporal upsampler. See `taau.rs`.
        let taau = taau::TaauSystem::new(&device, backend, compute_supported)?;

        // GPU particle system (Phase 7): a persistent ping-pong buffer pair advanced
        // by a compute pass and drawn as instanced billboards (see `particle.rs`).
        let particles =
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
        let mut textures: Vec<Texture> = Vec::new();
        let base_index = match &model.material.base_color {
            Some(im) => upload_texture(&device, &mut textures, im, Format::Rgba8Srgb)?,
            None => {
                let t = make_checker_texture(&device)?;
                let i = t.bindless_index();
                textures.push(t);
                i
            }
        };
        let mr_index = match &model.material.metallic_roughness {
            Some(im) => upload_texture(&device, &mut textures, im, Format::Rgba8Unorm)?,
            None => NO_TEXTURE,
        };
        let normal_index = match &model.material.normal {
            Some(im) => upload_texture(&device, &mut textures, im, Format::Rgba8Unorm)?,
            None => NO_TEXTURE,
        };
        let emissive_index = match &model.material.emissive {
            Some(im) => upload_texture(&device, &mut textures, im, Format::Rgba8Srgb)?,
            None => NO_TEXTURE,
        };

        // Build the scene as ECS entities: the procedural gallery (default), a full
        // glTF scene (`SCENE_GLTF=<path>`, Stage B), a declarative level (`LEVEL=<name>`,
        // Stage C), or a streaming world of chunks (`WORLD`, Stage D). A ground plane is
        // kept separate (it's also the environment-capture geometry). `sphere`/`cube`
        // are always built — the gallery uses them, and they're cheap.
        let r = model_radius;
        let sphere = dreamcoast_asset::uv_sphere(48, 32);
        let cube = dreamcoast_asset::unit_cube();
        // Scene-mode env vars (precedence: WORLD > LEVEL > SCENE_GLTF > gallery).
        let world_mode = std::env::var_os("WORLD").is_some();
        let level_select = if world_mode {
            None
        } else {
            std::env::var("LEVEL").ok()
        };
        let scene_gltf_path = if world_mode || level_select.is_some() {
            None
        } else {
            std::env::var("SCENE_GLTF").ok()
        };
        // The gallery is the only scene with the GDF/HW-RT path; glTF + levels + worlds
        // use the captured-cube IBL (forced via `legacy_ibl` below).
        let gallery_scene = !world_mode && level_select.is_none() && scene_gltf_path.is_none();
        let levels_dir = std::path::PathBuf::from("apps/sandbox/levels");

        // Registries own the GPU meshes + material descriptors the scene's handles
        // point at (P2). Unique geometry uploads once — the two spheres share a handle.
        let mut mesh_registry = MeshRegistry::new();
        let mut material_registry = MaterialRegistry::new();
        let mut world = World::new();
        // Level-mode hot-swap state: the discovered `.level` files + the loaded index.
        let mut level_paths: Vec<String> = Vec::new();
        let mut current_level = 0usize;
        // Stage D: the streaming manager (world mode only). Chunks load on demand from
        // the camera, so `world`/registries above stay empty in this mode.
        let mut streaming: Option<world::Streaming> = None;
        // World-space AABB of the placed scene (metres), used to frame the camera at the
        // scene's native scale. `None` keeps the legacy gallery framing.
        let mut scene_bounds: Option<level::Bounds> = None;
        // A level's authored camera (applied as the initial view if non-default).
        let mut level_view: Option<(Vec3, Vec3)> = None;
        // A level's lighting, replacing the gallery's code-default sun + point lights.
        let mut level_lighting_override: Option<LevelLighting> = None;

        if world_mode {
            // Stage D: load the level graph + the level files its chunks reference.
            level::ensure_level_files(&levels_dir)?;
            let world_path = world::ensure_world_file(&levels_dir)?;
            let graph = dreamcoast_asset::LevelGraph::load_ron(&world_path)?;
            info!(
                "world '{}': {} chunks, stream_radius {}",
                world_path.display(),
                graph.chunks.len(),
                graph.stream_radius
            );
            streaming = Some(world::Streaming::new(graph, levels_dir.clone()));
        } else if let Some(select) = &level_select {
            // Stage C: load a declarative level (auto-writing the built-in levels first).
            level_paths = level::ensure_level_files(&levels_dir)?;
            current_level = level_paths
                .iter()
                .position(|p| {
                    std::path::Path::new(p)
                        .file_stem()
                        .is_some_and(|s| s.eq_ignore_ascii_case(select))
                })
                .unwrap_or(0);
            // Stage E: load through the cook (RON → cooked .dcasset, cache-keyed).
            let level = level::load(std::path::Path::new(&level_paths[current_level]))?;
            level_view = level::level_camera(&level);
            level_lighting_override = Some(level_lighting(&level));
            scene_bounds = level::build_level(
                &device,
                &level,
                &mut world,
                &mut mesh_registry,
                &mut material_registry,
                &mut textures,
                Vec3::ZERO,
            )?;
        } else if let Some(path) = &scene_gltf_path {
            // Stage B: import the whole node hierarchy + every primitive/material/image.
            let gscene = dreamcoast_asset::load_gltf_scene(path)?;
            info!(
                "glTF scene '{path}': {} nodes, {} primitives, {} materials, {} images",
                gscene.nodes.len(),
                gscene.primitive_count(),
                gscene.materials.len(),
                gscene.images.len()
            );
            let prim_handles = registry::upload_gltf_scene(
                &device,
                &gscene,
                &mut mesh_registry,
                &mut material_registry,
                &mut textures,
            )?;
            let imported = dreamcoast_scene::instantiate_gltf(&mut world, &gscene, &prim_handles);
            // Place at native (1 unit = 1 m) scale under a wrapper root (so the whole
            // import can be spun/inspected); the camera frames it from its AABB below.
            let scene_root = world.spawn();
            world.insert(scene_root, LocalTransform::IDENTITY);
            world.insert(scene_root, dreamcoast_scene::Name("scene-root".to_owned()));
            world.insert(imported, dreamcoast_scene::Parent(scene_root));
            // Optionally spin the import to prove `propagate_transforms` moves the whole
            // hierarchy (Stage B verification).
            let spin = std::env::var("GLTF_SPIN")
                .ok()
                .and_then(|v| v.parse::<f32>().ok());
            if let (Some(deg), Some(lt)) = (spin, world.get_mut::<LocalTransform>(scene_root)) {
                lt.rotation = Quat::from_rotation_y(deg.to_radians());
            }
            scene_bounds = registry::gltf_bounds(&gscene);
        } else {
            // The procedural gallery (default) — byte-identical to Stage A.
            let mesh_model = mesh_registry.upload(&device, model)?;
            let mesh_sphere = mesh_registry.upload(&device, &sphere)?;
            let mesh_cube = mesh_registry.upload(&device, &cube)?;
            // The loaded model is textured: its representative GI albedo is the base-color
            // texture's linear average × factor (the procedural objects use their factor's
            // RGB). `representative_albedo` is the one definition the fuse later reads.
            let mat_model = material_registry.add(MaterialDesc {
                base_color: model.material.base_color_factor,
                metallic: model.material.metallic_factor,
                roughness: model.material.roughness_factor,
                tex: [base_index, mr_index, normal_index, emissive_index],
                albedo: registry::representative_albedo(
                    model
                        .material
                        .base_color
                        .as_ref()
                        .map(|t| t.average_linear()),
                    model.material.base_color_factor,
                ),
            });
            let mat_chrome = material_registry.add(MaterialDesc {
                base_color: [0.95, 0.96, 0.97, 1.0],
                metallic: 1.0,
                roughness: 0.08,
                tex: [NO_TEXTURE; 4],
                albedo: registry::representative_albedo(None, [0.95, 0.96, 0.97, 1.0]),
            });
            let mat_copper = material_registry.add(MaterialDesc {
                base_color: [0.95, 0.64, 0.54, 1.0],
                metallic: 1.0,
                roughness: 0.35,
                tex: [NO_TEXTURE; 4],
                albedo: registry::representative_albedo(None, [0.95, 0.64, 0.54, 1.0]),
            });
            let mat_red = material_registry.add(MaterialDesc {
                base_color: [0.85, 0.25, 0.2, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                tex: [NO_TEXTURE; 4],
                albedo: registry::representative_albedo(None, [0.85, 0.25, 0.2, 1.0]),
            });
            // Spawn order defines the deterministic draw / TLAS-instance order (model,
            // chrome, copper, cube) — the order the legacy flat list used.
            world
                .spawn_node()
                .named("model")
                .with(MeshInstance::new(mesh_model, mat_model))
                .with(LocalTransform::IDENTITY);
            world
                .spawn_node()
                .named("chrome-sphere")
                .with(MeshInstance::new(mesh_sphere, mat_chrome))
                .with(LocalTransform::trs(
                    Vec3::new(-r * 1.7, r * 0.75, r * 0.5),
                    r * 0.75,
                ));
            world
                .spawn_node()
                .named("copper-sphere")
                .with(MeshInstance::new(mesh_sphere, mat_copper))
                .with(LocalTransform::trs(
                    Vec3::new(r * 1.9, r * 0.5, -r * 0.4),
                    r * 0.5,
                ));
            world
                .spawn_node()
                .named("red-cube")
                .with(MeshInstance::new(mesh_cube, mat_red))
                .with(LocalTransform::trs(
                    Vec3::new(0.0, r * 0.45, -r * 2.0),
                    r * 0.45,
                ));
        }
        dreamcoast_scene::propagate_transforms(&mut world);

        // Materialize the ECS draw list into the flat `SceneObject` list the GPU passes
        // consume. (Static scene → built once; later stages rebuild on scene change. In
        // world mode `world` is empty — the per-frame list comes from the streamer.)
        let scene = build_scene(&world, &mesh_registry, &material_registry);
        // Frame the camera at the scene's native scale: derive the centre + radius from
        // the placed-geometry AABB (Sponza fills its real ~20 m, lanterns their ~2 m).
        // The gallery keeps its exact legacy framing (byte-identical baseline); world
        // mode has no single AABB (streaming), so it uses a fixed extent.
        let (scene_center, scene_radius) = match scene_bounds {
            Some((min, max)) => {
                let center = (min + max) * 0.5;
                let radius = (0.5 * (max - min).length()).max(0.5);
                (center, radius)
            }
            None if world_mode => (Vec3::new(0.0, 2.0, 0.0), 28.0),
            None => (Vec3::new(0.0, r * 0.6, 0.0), r * 3.0), // gallery legacy framing
        };
        if let Some((min, max)) = scene_bounds {
            let s = max - min;
            info!(
                "scene bounds (m): size [{:.2}, {:.2}, {:.2}], centre [{:.2}, {:.2}, {:.2}]",
                s.x, s.y, s.z, scene_center.x, scene_center.y, scene_center.z
            );
        }
        // World mode drives streaming from a free-fly camera. Seed its eye from
        // `WORLD_CAM="x,y,z"` (default above the chunk row, looking along it) so a
        // headless capture can position the camera; interactively, WASD flies it.
        let world_fly = world_mode.then(|| {
            let eye = parse_vec3_env("WORLD_CAM").unwrap_or(Vec3::new(0.0, 3.0, 9.0));
            camera::FlyCamera::from_look(eye, Vec3::new(eye.x, 2.0, 0.0), scene_radius * 0.4)
        });
        // RT instance-table mesh sources, aligned 1:1 with the draw list (TLAS order).
        // Only the gallery builds the HW-RT scene accel; glTF/level paths pass nothing
        // (RtSystem::new skips the table when `build_scene_accel` is false).
        let scene_meshes: Vec<&MeshData> = if gallery_scene {
            vec![model, &sphere, &sphere, &cube]
        } else {
            Vec::new()
        };

        // Ground plane (separate handle: also used by the environment capture).
        let ground = ground_mesh(scene_radius * 1.3, 0.0);
        let (ground_vbuf, ground_ibuf, ground_count) = upload_mesh(&device, &ground)?;

        // Hardware ray tracing (Phase 8): BLAS/TLAS over the sample scene + ground,
        // the path tracer's per-instance geometry table, and the alternate Cornell-box
        // scene — plus the M3/M4/M5 RT pipelines. See `rt.rs`. The sample scene's TLAS
        // is bound on construction (the startup default). The instance table's mesh
        // order MUST match the TLAS custom_index order (scene objects, then ground).
        let rt = RtSystem::new(
            &device,
            backend,
            &scene,
            &scene_meshes,
            &ground,
            &ground_vbuf,
            &ground_ibuf,
            ground_count,
            gallery_scene,
        )?;

        // Scalable-GI Stage 0: fuse the opaque draw list into one world-space triangle
        // soup and register it as the scene GDF (baked once on the graph). Ground is
        // handled analytically at trace time; disjoint objects give the union SDF via the
        // closest-triangle sign convention. Geometry + albedo come from the registries
        // (`fuse::fuse_scene` — the single fuse path), so the same routine fuses the
        // gallery, an imported glTF scene, or a level. The byte layout matches the legacy
        // gallery fuse, so the gallery's baked field is byte-identical.
        //
        // Stage D: build the scene GDF for ANY non-streaming scene with geometry — the
        // gallery, an imported glTF, or a level (Sponza). `fuse_scene` + the Stage A grid
        // bake + the clipmap make this affordable. World streaming stays out of scope (no
        // single AABB). The gallery is byte-identical (same fuse → same bake → same cards).
        let build_scene_gdf = !world_mode && gdf.has_gdf_trace() && !world.draw_list().is_empty();
        if build_scene_gdf {
            let fused = fuse::fuse_scene(&world, &mesh_registry, &material_registry);
            let fused_v = fused.vtx;
            let fused_i = fused.idx;
            let tri_albedo = fused.tri_albedo;
            let amin = fused.aabb_min;
            let amax = fused.aabb_max;
            let tri_count = fused.tri_count;
            // Per-drawable world AABBs (for the surface-cache mesh cards).
            let obj_aabb = fused.drawable_aabb;
            // Phase 12 M2: cook the scene SDF (deterministic CPU bake, cached as a
            // `.dcasset` keyed on the fused geometry + grid) and upload it, replacing
            // the one-time GPU bake. A fresh cache loads directly; a miss bakes + saves.
            let sdf_dim = gdf.scene_dim();
            // Stage B (clipmap): plan the camera-centered level scheme. The gallery is the
            // byte-identical regression reference, so it stays single-level by default
            // (= the legacy 48³ volume). `P11_GDF_CLIP_LEVELS=N` opts into an N-level clipmap
            // (B3 multi-level path verification); the finer levels are cooked over their
            // sub-AABBs (Stage A grid bake, cached) and installed. Default activation for
            // content scenes (Sponza) lands in Stage D.
            let clip_center = [
                (amin[0] + amax[0]) * 0.5,
                (amin[1] + amax[1]) * 0.5,
                (amin[2] + amax[2]) * 0.5,
            ];
            // The gallery stays single-level (byte-identical reference); content scenes
            // (Sponza) default to a 4-level clipmap (auto-trimmed by extent in plan_levels)
            // — the camera-centered clipmap is the default for content, per the design.
            let clip_max_levels = std::env::var("P11_GDF_CLIP_LEVELS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(if gallery_scene { 1 } else { 4 })
                .max(1);
            let clip = clipmap::plan_levels(amin, amax, clip_center, sdf_dim, 0.1, clip_max_levels);
            info!("GDF clipmap: {} level(s)", clip.level_count());
            let (sdf_vol, sdf_outcome) = dreamcoast_asset::cook::load_or_bake_scene_sdf(
                &fused_v,
                &fused_i,
                sdf_dim,
                amin,
                amax,
                &app::cooked_cache_dir(),
            );
            info!("scene SDF {sdf_dim}^3 ({sdf_outcome:?})");
            let sdf_bytes = sdf_vol.to_le_bytes();
            // C8a per-voxel albedo volumes: cooked the same way (CPU bake, cached),
            // uploaded so the one-time GPU albedo bake is skipped too.
            let (albedo_vol, alb_outcome) = dreamcoast_asset::cook::load_or_bake_scene_albedo(
                &fused_v,
                &fused_i,
                &tri_albedo,
                sdf_dim,
                amin,
                amax,
                &app::cooked_cache_dir(),
            );
            info!("scene albedo {sdf_dim}^3 ({alb_outcome:?})");
            let alb = [
                albedo_vol.channel_le_bytes(0),
                albedo_vol.channel_le_bytes(1),
                albedo_vol.channel_le_bytes(2),
            ];
            gdf.build_scene_sdf(
                &device,
                &fused_v,
                &fused_i,
                &tri_albedo,
                tri_count,
                amin,
                amax,
                Some(&sdf_bytes),
                Some([&alb[0], &alb[1], &alb[2]]),
            )?;
            // Stage D: the gallery's floor is analytic (y = 0, no floor geometry); content
            // scenes carry their floor as real geometry, so disable the analytic ground
            // (a very low Y) to avoid a spurious second floor in the SW-RT march.
            gdf.set_scene_ground_y(if gallery_scene { 0.0 } else { -1.0e9 });
            // Stage B3: cook + install the finer clipmap levels (every level but the
            // coarsest, which `build_scene_sdf` just created). Each is keyed on its own
            // sub-AABB so the cache stores them separately; off unless P11_GDF_CLIP_LEVELS>1.
            if clip.level_count() > 1 {
                let finer = &clip.levels[..clip.level_count() - 1];
                let mut sdf_store: Vec<Vec<u8>> = Vec::new();
                let mut alb_store: Vec<[Vec<u8>; 3]> = Vec::new();
                for (lmin, lmax) in finer {
                    let (v, _) = dreamcoast_asset::cook::load_or_bake_scene_sdf(
                        &fused_v,
                        &fused_i,
                        sdf_dim,
                        *lmin,
                        *lmax,
                        &app::cooked_cache_dir(),
                    );
                    sdf_store.push(v.to_le_bytes());
                    let (av, _) = dreamcoast_asset::cook::load_or_bake_scene_albedo(
                        &fused_v,
                        &fused_i,
                        &tri_albedo,
                        sdf_dim,
                        *lmin,
                        *lmax,
                        &app::cooked_cache_dir(),
                    );
                    alb_store.push([
                        av.channel_le_bytes(0),
                        av.channel_le_bytes(1),
                        av.channel_le_bytes(2),
                    ]);
                }
                let level_data: Vec<crate::gdf::ClipLevelData> = finer
                    .iter()
                    .enumerate()
                    .map(|(i, (lmin, lmax))| crate::gdf::ClipLevelData {
                        aabb_min: *lmin,
                        aabb_max: *lmax,
                        sdf: &sdf_store[i],
                        albedo: Some([&alb_store[i][0], &alb_store[i][1], &alb_store[i][2]]),
                    })
                    .collect();
                gdf.set_clip_levels(&device, &level_data)?;
            }
            // Phase 12 item 3: optional GPU→CPU volume-readback round-trip check. Reads
            // the just-uploaded scene SDF back and confirms it equals the bytes we
            // uploaded — validating `Device::read_volume` on the live backend.
            if std::env::var_os("P12_VERIFY_VOLUME").is_some()
                && let Some(vol) = gdf.scene_gdf_volume()
            {
                let back = device.read_volume(vol, sdf_dim, sdf_dim, sdf_dim, 4)?;
                let mismatches = back.iter().zip(&sdf_bytes).filter(|(a, b)| a != b).count();
                info!(
                    "volume readback round-trip ({sdf_dim}^3): {} bytes, {mismatches} mismatch(es)",
                    back.len()
                );
            }
            // Stage C/D: the surface-cache atlas (cards + per-card texel buffers, re-lit each
            // frame) feeds the SW-RT reflection/GI. It is the default ambient for any GDF
            // scene now, so build it unless the IBL escape hatch is forced (then it would be
            // unused — skip the ~67 MB atlas + per-frame relight). MAX_CARDS (fuse.rs) bounds
            // it; cards are draw-list-driven.
            let build_cache = std::env::var_os("P11_LEGACY_IBL").is_none();
            if build_cache {
                let cards = fuse::build_surface_cards(&obj_aabb);
                let num_cards = (cards.len() / 64) as u32;
                // QHD/UHD track: the surface-cache atlas tile is runtime-tunable (`P11_CACHE_TILE`)
                // so content can trade cache cost + atlas memory for reflection-cache sharpness.
                // Default 32 = unchanged (byte-identical). Measured: tile 16 cuts the relight only
                // ~30% (the relight isn't purely texel-bound at spp1/period40) while blurring
                // reflections (max ~94 LSB) — a poor default, so it stays opt-in. Built once here.
                let cache_tile = std::env::var("P11_CACHE_TILE")
                    .ok()
                    .and_then(|v| v.trim().parse::<u32>().ok())
                    .unwrap_or(32)
                    .clamp(4, 64);
                gdf.build_surface_cache(&device, &cards, num_cards, cache_tile)?;
            }
        }

        let gui = Gui::new(&device, swapchain.format(), FRAMES_IN_FLIGHT)?;

        // One render-graph transient pool per frame-in-flight (reused only after the
        // frame slot's fence has signaled — no cross-frame hazards).
        let pools: Vec<ResourcePool> = (0..FRAMES_IN_FLIGHT).map(|_| ResourcePool::new()).collect();

        // UI-controlled lighting state defaults.
        let sun_dir = [0.4f32, 0.8, 0.4];
        let sun_intensity = 3.0f32;
        let ambient = 0.04f32;
        // On by default; `NO_POINT_LIGHTS=1` disables them (the path tracer has no
        // point lights, so a fair raster-vs-ground-truth comparison turns these off).
        let point_lights_on = std::env::var_os("NO_POINT_LIGHTS").is_none();
        // RenderQuality tier (Stage D): `RENDER_QUALITY=low|med|high` (unset => Med = the legacy
        // defaults, no-reg). `qp` is the single tier→knob table; each knob below reads its own env
        // first and falls back to `qp.*`, so an explicit `P11_*`/`SHADOW_*` override always wins.
        let quality = quality::RenderQuality::from_env();
        let qp = quality::preset(quality);
        info!("RenderQuality tier: {} (RENDER_QUALITY)", quality.label());
        // Phase 7: route the HDR result through a compute post-process (3x3 blur into
        // a storage image) before tonemapping. Initial state seedable via env var so
        // headless screenshots can exercise each demo (`P7_COMPUTE_POST=1`, etc.).
        let compute_post = compute_supported && std::env::var_os("P7_COMPUTE_POST").is_some();
        let particles_on = compute_supported && std::env::var_os("P7_PARTICLES").is_some();
        // Run the particle sim on the async-compute queue (overlapping graphics) when
        // a dedicated compute queue exists. Off / unsupported -> the sim runs as a
        // graph compute pass on the graphics queue (single-queue path), identical out.
        let async_compute_supported = device.has_async_compute();
        let async_compute_on = async_compute_supported
            && (std::env::var_os("ASYNC_COMPUTE").is_some() || !screenshot_mode);
        // Async-compute surface-cache relight (QHD/UHD track): the resolution-independent
        // `sdf_cache_light` pass (the biggest cost in the VK frame) runs on the compute queue
        // overlapping the graphics frame; consumers read the previous frame's radiance (1-frame
        // latency, hidden by the cache's existing amortization + EMA). Opt-in (`P_ASYNC_CACHE`);
        // needs a dedicated compute queue + the cache-lighting pipeline. Particles fall back to the
        // graph sim when this is on (the two would contend for the single compute submission).
        let async_cache_on = async_compute_supported
            && gdf.has_cache_lighting()
            && std::env::var_os("P_ASYNC_CACHE").is_some();
        gdf.set_cache_async(async_cache_on);
        let gpu_cull = compute_supported && std::env::var_os("P7_CULL").is_some();
        // Phase 8 M3: replace the rasterized image with an inline ray-query trace.
        let path_trace =
            rt.has_trace() && rt.has_scene() && std::env::var_os("P8_PATHTRACE").is_some();
        let rt_debug = device.has_raytracing() && std::env::var_os("P8_RT_DEBUG").is_some();
        let cornell = rt.has_cornell() && std::env::var_os("P8_CORNELL").is_some();
        let sdf_trace = gdf.has_sdf_trace() && std::env::var_os("P11_SDF").is_some();
        let volume_test = gdf.has_volume() && std::env::var_os("P11_VOLUME_TEST").is_some();
        let sdf_bake = gdf.has_bake() && std::env::var_os("P11_SDF_BAKE").is_some();
        let gdf_merge = gdf.has_merge() && std::env::var_os("P11_GDF_MERGE").is_some();
        let gdf_trace = gdf.has_gdf_trace() && std::env::var_os("P11_GDF_TRACE").is_some();
        let gdf_trace_analytic = std::env::var_os("P11_GDF_ANALYTIC").is_some();
        // Stage C1: trace the world-space scene GDF from the live camera.
        let scene_gdf = gdf.has_scene_sdf() && std::env::var_os("P11_SCENE_GDF").is_some();
        // Stage C2: GDF AO multiplied into the deferred ambient term.
        let gdf_ao =
            gi.has_ao() && gdf.has_scene_sdf() && quality::env_bool("P11_GDF_AO", qp.gdf_ao);
        // Deprecate the legacy captured-cube IBL: by default the deferred ambient is the
        // SW-RT hybrid reflection (specular) + GDF GI (diffuse scene bounce) + sky irradiance.
        // `P11_LEGACY_IBL` restores the captured-cube path (prefilter-cube specular + scene
        // capture) for comparison.
        // Stage D lighting flip: any scene WITH a scene GDF (the gallery and now content
        // levels/glTF, via the clipmap) uses the SW-RT GDF ambient by default — the camera-
        // centered clipmap is the default for content. `P11_LEGACY_IBL` restores the captured-
        // cube IBL (the escape hatch / comparison). Scenes without a scene GDF (world
        // streaming) always use the IBL.
        let legacy_ibl = !gdf.has_scene_sdf() || std::env::var_os("P11_LEGACY_IBL").is_some();
        let swrt_ok = reflect.has_ssr()
            && reflect.has_gdf_reflect()
            && reflect.has_composite()
            && reflect.has_lit_history()
            && gdf.has_scene_sdf();
        // Stage C3: GDF 1-bounce diffuse GI — part of the default ambient now (on unless
        // legacy IBL); `P11_GDF_GI` still force-enables it under legacy.
        let gdf_gi = gi.has_gi()
            && gdf.has_scene_sdf()
            && (!legacy_ibl || std::env::var_os("P11_GDF_GI").is_some());
        // Stage D3: gallery forced to the legacy 8 (byte-identical anchor); content takes the tier.
        let gi_spp = std::env::var("P11_GI_SPP")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(if gallery_scene { 8 } else { qp.gi_spp })
            .clamp(1, 256);
        // Stage D3: C3 bounce-ray march step cap. Gallery forced to the legacy 64 (byte-identical);
        // content takes the tier value. `P11_GI_MAX_STEPS` overrides.
        let gi_max_steps = std::env::var("P11_GI_MAX_STEPS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(if gallery_scene { 64 } else { qp.gi_max_steps })
            .clamp(1, 256);
        // Stage D2: surface-cache amortized-relight period. The gallery is the byte-identical
        // regression anchor, so it is forced to 1 (every-frame relight = legacy) just like the
        // clipmap level count above; content scenes (Sponza) take the tier default and amortize.
        // `P11_CACHE_RELIGHT_PERIOD` overrides either way.
        let cache_relight_period = std::env::var("P11_CACHE_RELIGHT_PERIOD")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(if gallery_scene {
                1
            } else {
                qp.cache_relight_period
            })
            .max(1);
        // Stage D2b: camera-visibility feedback drives the relight budget (off-screen cards relit
        // far less). Pure perf optimization — invariant for the on-screen image — so default on for
        // content; forced off for the gallery anchor (uniform period = byte-identical). Needs the
        // visibility pipeline (capability-gated). `P11_CACHE_FEEDBACK` overrides.
        let cache_feedback =
            gdf.has_cache_visibility() && quality::env_bool("P11_CACHE_FEEDBACK", !gallery_scene);
        // Stage D3: relight gather rays/texel. Gallery forced to the legacy 8 (byte-identical);
        // content takes the tier value. `P11_CACHE_RELIGHT_SPP` overrides.
        let cache_relight_spp = std::env::var("P11_CACHE_RELIGHT_SPP")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(if gallery_scene {
                8
            } else {
                qp.cache_relight_spp
            })
            .max(1);
        // Stage D1: half-res GI trace + bilateral upsample. Gallery stays full-res (the
        // byte-identical anchor); content scenes take the tier default. `P11_GI_HALF_RES`
        // overrides. Needs the upsample pipeline (capability-gated).
        let gi_half_res = gi.has_upsample()
            && quality::env_bool("P11_GI_HALF_RES", !gallery_scene && qp.gi_half_res);
        // C4 denoise: on by default whenever GI runs (P11_GI_DENOISE=0 to see raw GI).
        let gi_denoise = gi.has_denoise() && quality::env_bool("P11_GI_DENOISE", qp.gi_denoise);
        // C5 screen-space reflections (viz toggle).
        let gdf_ssr = reflect.has_ssr() && std::env::var_os("P11_SSR").is_some();
        // C6 GDF reflections (off-screen fallback viz toggle).
        let gdf_reflect = reflect.has_gdf_reflect()
            && gdf.has_scene_sdf()
            && std::env::var_os("P11_GDF_REFLECT").is_some();
        // C7 hybrid reflection composite (viz toggle): needs SSR + GDF reflect + composite.
        let gdf_hybrid = reflect.has_ssr()
            && reflect.has_gdf_reflect()
            && reflect.has_composite()
            && gdf.has_scene_sdf()
            && std::env::var_os("P11_HYBRID").is_some();
        // C7c: feed the hybrid composite into the lighting specular — the DEFAULT specular
        // now (replaces the prefilter-cube IBL); `P11_LEGACY_IBL` falls back to the cube.
        let swrt_reflect = swrt_ok && !legacy_ibl;
        // C8a colored GDF re-light (per-voxel albedo). On by default when the albedo volumes
        // exist; `P11_GDF_COLOR=0` forces the achromatic constant-albedo path (no-reg compare).
        let gdf_color = gdf.has_scene_albedo()
            && std::env::var("P11_GDF_COLOR")
                .map(|v| v != "0")
                .unwrap_or(true);
        // C8b1 surface-cache atlas viz (validation toggle).
        let cache_viz = gdf.has_surface_cache() && std::env::var_os("P11_CACHE_VIZ").is_some();
        // C8b3 surface-cache consumers (multibounce radiance lookup in GI / reflections).
        let surface_cache = gdf.has_surface_cache()
            && gdf.has_cache_lighting()
            && quality::env_bool("P11_SURFACE_CACHE", qp.surface_cache);
        // C8g: use the surface cache as the GDF REFLECTION hit radiance by default (accurate lit
        // colour for reflected objects — fixes the grazing avocado smear; ground hits have no
        // cards and fall back to the per-ray re-light). Cheap (only the per-frame cache-light pass
        // + a reflect-side lookup); the expensive per-ray GI cache lookup stays opt-in above.
        // `P11_REFLECT_CACHE=0` disables (reflections then use the C8a per-ray re-light).
        let reflect_cache = swrt_reflect
            && gdf.has_surface_cache()
            && gdf.has_cache_lighting()
            && quality::env_bool("P11_REFLECT_CACHE", qp.reflect_cache);
        // Firefly clamp on by default (P11_FIREFLY_CLAMP=0 to disable / compare).
        let firefly_clamp = quality::env_bool("P11_FIREFLY_CLAMP", qp.firefly_clamp);
        // C8d: roughness above which screen-mirror SSR stops contributing (GDF takes over).
        let reflect_max_roughness = std::env::var("P11_REFLECT_MAX_ROUGHNESS")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(qp.reflect_max_roughness);
        // Stage D3: reflection-ray march step cap. Gallery forced to the legacy 96 (byte-identical);
        // content takes the tier value. `P11_REFLECT_MAX_STEPS` overrides.
        let reflect_max_steps = std::env::var("P11_REFLECT_MAX_STEPS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(if gallery_scene {
                96
            } else {
                qp.reflect_max_steps
            })
            .clamp(1, 256);
        // Stage D3: half-res reflection trace + bilateral upsample (reuses the GI upsample).
        // Gallery forced off (full-res = byte-identical anchor); content takes the tier value.
        let reflect_half_res = gi.has_upsample()
            && quality::env_bool(
                "P11_REFLECT_HALF_RES",
                !gallery_scene && qp.reflect_half_res,
            );
        // C8d: default to the full-res mirror SSR; opt into the stochastic glossy path to compare.
        let ssr_stochastic = quality::env_bool("P11_SSR_STOCHASTIC", qp.ssr_stochastic);
        // Diagnostic single-object orbit: frame one scene object tightly so it can be
        // inspected from every side. `DIAG_OBJ=<index>` selects it (2 = copper sphere,
        // 3 = red cube); `DIAG_COPPER=1` / `DIAG_CUBE=1` are shortcuts. `DIAG_ANGLE=<deg>`
        // pins the orbit azimuth and `DIAG_PITCH=<deg>` the elevation (90 = straight down).
        let diag_obj = std::env::var("DIAG_OBJ")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .or_else(|| std::env::var_os("DIAG_COPPER").map(|_| 2))
            .or_else(|| std::env::var_os("DIAG_CUBE").map(|_| 3));
        let diag_angle = std::env::var("DIAG_ANGLE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(f32::to_radians);
        let diag_pitch = std::env::var("DIAG_PITCH")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(f32::to_radians);
        // Phase 8 M5: `pt_pipeline` is only built when the pipeline was requested, so
        // its presence alone is the default-on condition.
        let path_trace_pipeline = rt.has_pt_pipeline();

        let mut command_buffers = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut image_available = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(FRAMES_IN_FLIGHT);
        // Async-compute resources: a command buffer on the compute queue per frame,
        // plus a semaphore the compute submit signals and the graphics submit waits
        // on. Only used when a dedicated compute queue exists and the toggle is on.
        let compute_queue = device.compute_queue();
        let mut compute_command_buffers = Vec::with_capacity(FRAMES_IN_FLIGHT);
        let mut compute_done = Vec::with_capacity(FRAMES_IN_FLIGHT);
        // Two cross-frame "cache relight done" semaphores (indexed by frame parity, NOT fif): the
        // async relight signals one, next frame's graphics waits it (1-frame latency).
        let cache_done = vec![device.create_semaphore()?, device.create_semaphore()?];
        // Per-fif compute-completion fences (created signaled, like in_flight) gating reuse of the
        // async relight's compute command buffer.
        let mut cache_compute_fence = Vec::with_capacity(FRAMES_IN_FLIGHT);
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
            cache_compute_fence.push(device.create_fence(true)?);
            query_heaps.push(device.create_query_heap(MAX_QUERIES)?);
        }
        // QHD/UHD track: parse the offscreen render-extent override (`RENDER_RES=WxH`).
        let render_res = std::env::var("RENDER_RES").ok().and_then(|s| {
            let (a, b) = s.split_once(['x', 'X', ','])?;
            let w = a.trim().parse::<u32>().ok()?.clamp(320, 7680);
            let h = b.trim().parse::<u32>().ok()?.clamp(240, 4320);
            Some((w, h))
        });
        // QHD/UHD track: internal render scale (production knob). `RENDER_SCALE` overrides the tier.
        // Min internal scale clamped to 0.6667 (2/3): below that the TAAU upscale can't reconstruct
        // enough detail for a sharp result (per the QHD/UHD plan).
        let render_scale = std::env::var("RENDER_SCALE")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(qp.render_scale)
            // Floor at 1/3 (DLSS "ultra performance" territory): below that even a temporal
            // reconstruction can't hold up. The TAAU jitter reconstruction (B-track) makes the
            // 0.4–0.6 range viable, which is what QHD/UHD high-fps needs; 1.0 stays the default
            // (byte-identical native).
            .clamp(0.3333, 1.0);
        let profiler_on = std::env::var("PROFILE_GPU").is_ok();
        let slot_pass_names: Vec<Vec<String>> = vec![Vec::new(); FRAMES_IN_FLIGHT];
        let render_finished = build_render_finished(&device, swapchain.image_count())?;

        // Image-based lighting (see `ibl.rs`): the procedural-sky / capture /
        // irradiance / prefilter / BRDF pipelines, the ping-pong environment cube
        // sets, the capture depth and the BRDF LUT (generated once on construction).
        let ibl = IblSystem::new(&device, backend, &queue, flip_y, sun_dir, sun_intensity)?;
        // Seed both cube sets once (single-bounce, no previous environment) so the
        // first multi-bounce frame reads valid data instead of uninitialized memory.
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

        let mut window = window;
        let _ = window.take_resized();
        info!("entering render loop");

        // Read before `gdf` is moved into the struct (Phase 12 M2): the cooked SDF /
        // albedo were uploaded, so their one-time GPU bakes are pre-satisfied.
        let scene_gdf_cooked = gdf.scene_sdf_is_cooked();
        let scene_albedo_cooked = gdf.scene_albedo_is_cooked();

        Ok(Self {
            window,
            _instance: instance,
            device,
            queue,
            compute_queue,
            swapchain,
            backend,
            gui,
            deferred,
            gdf,
            gi,
            reflect,
            particles,
            cull,
            rt,
            ibl,
            _textures: textures,
            world,
            mesh_registry,
            material_registry,
            level_paths,
            current_level,
            pending_level: None,
            streaming,
            ground_vbuf,
            ground_ibuf,
            // The hardcoded flat ground is a gallery-only code default. Every other mode
            // brings its own floor (a level's asset geometry, or a "ground" entity — so a
            // streamed chunk carries its own ground patch). Drawing 0 indices keeps the
            // buffers valid but renders nothing.
            ground_count: if gallery_scene { ground_count } else { 0 },
            pools,
            command_buffers,
            image_available,
            in_flight,
            compute_command_buffers,
            compute_done,
            async_cache_on,
            cache_done,
            cache_compute_fence,
            query_heaps,
            render_finished,
            flip_y,
            model_radius,
            scene_radius,
            scene_center,
            level_view,
            level_lighting: level_lighting_override,
            is_gallery: gallery_scene,
            screenshot_mode,
            captures,
            capture_seq: std::env::var("CAPTURE_SEQ")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|&n| n > 0),
            validation_on,
            async_compute_supported,
            path_spp: 8,
            gdf_trace_analytic,
            sun_dir,
            sun_intensity,
            ambient,
            exposure: 0.6,
            point_lights_on,
            shadows_on: true,
            shadow_bias: 0.0015,
            // PCSS-lite soft shadows: an opt-in quality tier (the scalability seam).
            // Default 0 = hard 3x3 PCF — cheapest AND the closest match to the path
            // tracer, whose sun disk (SUN_COS_MAX ~1.15deg) is near-sharp, so a wide
            // penumbra actually diverges from PT. `SHADOW_SOFTNESS=<f>` (or the UI slider)
            // turns it on; the PT-calibrated factor is ~0.0375, larger = softer/aesthetic.
            shadow_softness: std::env::var("SHADOW_SOFTNESS")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(qp.shadow_softness),
            // Soft-shadow tap count (RenderQuality knob, written to globals.shadow.w). Only the
            // soft path reads it; the shader clamps to [1, 16] (POISSON16). `SHADOW_TAPS` overrides.
            shadow_taps: std::env::var("SHADOW_TAPS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(qp.shadow_taps)
                .clamp(1, 16),
            quality,
            override_material: false,
            metallic_override: 1.0,
            roughness_override: 0.15,
            debug_view: std::env::var("DEBUG_VIEW")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0),
            post_mode: 0,
            aliasing: true,
            compute_post,
            particles_on,
            async_compute_on,
            gpu_cull,
            path_trace,
            rt_debug,
            cornell,
            sdf_trace,
            volume_test,
            sdf_bake,
            sdf_bake_done: false,
            gdf_merge,
            gdf_merge_done: false,
            gdf_trace,
            scene_gdf,
            gdf_ao,
            gdf_gi,
            gi_spp,
            cache_relight_period,
            cache_feedback,
            cache_relight_spp,
            gi_max_steps,
            reflect_max_steps,
            reflect_half_res,
            gi_half_res,
            gi_denoise,
            prev_view_proj: Mat4::IDENTITY.to_cols_array(),
            prev_view_proj_taau: Mat4::IDENTITY.to_cols_array(),
            gdf_ssr,
            gdf_reflect,
            gdf_hybrid,
            swrt_reflect,
            gdf_color,
            cache_viz,
            surface_cache,
            reflect_cache,
            firefly_clamp,
            reflect_max_roughness,
            ssr_stochastic,
            // Phase 12 M2: a cooked SDF was uploaded into the scene GDF, so the
            // one-time GPU bake is already satisfied — latch it as baked.
            scene_gdf_baked: scene_gdf_cooked,
            scene_albedo_baked: scene_albedo_cooked,
            scene_cache_captured: false,
            scene_cache_reset: true,
            path_trace_pipeline,
            realtime_env: true,
            multibounce: true,
            legacy_ibl,
            render_res,
            render_scale,
            taau,
            taau_on: quality::env_bool("P_TAAU", true),
            taau_jitter: quality::env_bool("P_TAAU_JITTER", true),
            profiler_on,
            slot_pass_names,
            gpu_timings: Vec::new(),
            fif: 0,
            frame_no: 0,
            f2_prev: false,
            needs_recreate: false,
            last: Instant::now(),
            elapsed: 0.0,
            // Fixed view in screenshot mode for reproducible output; `DIAG_ANGLE`
            // overrides it (degrees) for capturing the chosen object from a fixed side.
            angle: diag_angle.unwrap_or(if screenshot_mode { 0.7 } else { 0.0 }),
            diag_obj,
            diag_pitch,
            cam_mode: if world_mode {
                camera::CameraMode::Fly
            } else {
                camera::CameraMode::Orbit
            },
            fly: world_fly,
            tab_prev: false,
        })
    }

    /// Run the render loop until the window closes (or, in screenshot mode, every
    /// requested capture is saved).
    fn run(&mut self) -> anyhow::Result<()> {
        while !self.window.should_close() {
            if !self.frame()? {
                break;
            }
        }
        self.device.wait_idle()?;
        info!("shutting down");
        Ok(())
    }

    /// Hot-swap to level `idx` (Stage C): rebuild the ECS world + registries +
    /// textures from the level file. Waits for the GPU to idle first so the resources
    /// the previous frames referenced are safe to drop. The per-frame draw list is
    /// materialized from `self.world`, so the next frame picks up the new scene.
    fn load_level(&mut self, idx: usize) -> anyhow::Result<()> {
        self.device.wait_idle()?;
        let path = self.level_paths[idx].clone();
        let level = level::load(std::path::Path::new(&path))?;
        self.level_view = level::level_camera(&level);
        self.level_lighting = Some(level_lighting(&level));
        let mut world = World::new();
        let mut mesh_registry = MeshRegistry::new();
        let mut material_registry = MaterialRegistry::new();
        let mut textures: Vec<Texture> = Vec::new();
        let bounds = level::build_level(
            &self.device,
            &level,
            &mut world,
            &mut mesh_registry,
            &mut material_registry,
            &mut textures,
            Vec3::ZERO,
        )?;
        dreamcoast_scene::propagate_transforms(&mut world);
        self.world = world;
        self.mesh_registry = mesh_registry;
        self.material_registry = material_registry;
        self._textures = textures;
        self.current_level = idx;
        // Re-frame the camera for the new level's native-scale bounds.
        if let Some((min, max)) = bounds {
            self.scene_center = (min + max) * 0.5;
            self.scene_radius = (0.5 * (max - min).length()).max(0.5);
        }
        info!(
            "hot-swapped to level '{path}' ({} entities)",
            level.entities.len()
        );
        Ok(())
    }

    /// One iteration of the render loop. Returns `false` when the loop should stop
    /// (screenshot mode done); `true` to continue (including the skip-this-frame
    /// cases — zero-size window, failed acquire).
    fn frame(&mut self) -> anyhow::Result<bool> {
        // Apply a pending level hot-swap requested from the UI last frame.
        if let Some(idx) = self.pending_level.take() {
            self.load_level(idx)?;
        }
        self.window.pump_events();
        if self.window.take_resized() {
            self.needs_recreate = true;
        }
        self.window.pump_events();
        if self.window.take_resized() {
            self.needs_recreate = true;
        }
        let (ww, wh) = self.window.size();
        if ww == 0 || wh == 0 {
            std::thread::sleep(std::time::Duration::from_millis(16));
            return Ok(true);
        }
        if self.needs_recreate {
            self.device.wait_idle()?;
            self.swapchain
                .recreate(&swapchain_desc(Extent2D::new(ww, wh)))?;
            for p in &mut self.pools {
                p.clear(); // transient extents changed; drop cached targets
            }
            self.render_finished =
                build_render_finished(&self.device, self.swapchain.image_count())?;
            self.needs_recreate = false;
        }

        // Wait for this frame slot's previous submission to finish BEFORE the acquire
        // below. The acquire reuses `image_available[fif]`, and Vulkan forbids
        // acquiring with a semaphore that still has a pending wait from that earlier
        // submit (VUID-vkAcquireNextImageKHR-semaphore-01779). This is the standard
        // frames-in-flight order: wait → reset → acquire → record → submit.
        let fif = self.fif;
        self.in_flight[fif].wait()?;

        // Acquire the drawable up front: its *actual* pixel size is the single source
        // of truth for this whole frame (ImGui display size, camera aspect, render
        // extent, viewport). A failed acquire skips here, BEFORE the ImGui frame is
        // started, so NewFrame/Render stay balanced.
        let image_index = match self
            .swapchain
            .acquire_next_image(&self.image_available[fif])?
        {
            Some(i) => i,
            None => {
                self.needs_recreate = true;
                return Ok(true);
            }
        };
        // The swapchain (display) extent presents the final image; the *render* extent is where
        // the scene passes run. They are equal by default (byte-identical), but `RENDER_RES`
        // decouples them so headless can render the scene at QHD/UHD offscreen and the tonemap
        // downscales to the (display-bound) swapchain backbuffer. QHD/UHD perf track.
        let (sw, sh) = {
            let e = self.swapchain.extent_2d();
            (e.width, e.height)
        };
        let (cw, ch) = match self.render_res {
            Some(r) => r, // absolute override (headless QHD/UHD measurement)
            None if (self.render_scale - 1.0).abs() < 1e-4 => (sw, sh), // native (byte-identical)
            None => (
                ((sw as f32 * self.render_scale).round() as u32).max(64),
                ((sh as f32 * self.render_scale).round() as u32).max(64),
            ),
        };
        // QHD/UHD track: TAAU is active when the scene renders below the output resolution (upscale)
        // and isn't a path-trace/debug capture. It jitters the camera + reconstructs full-res from
        // history. When render == output (default), it's inactive (byte-identical).
        let taau_active = self.taau_on && self.taau.has_taau() && cw < sw && ch < sh;
        // When TAAU runs with sub-pixel jitter, the jitter IS the anti-aliasing (super-sampling
        // reconstruction); the spatial FXAA pre-pass then only blurs and — because it smooths each
        // jittered frame's edges differently — adds temporal variance, so it is skipped in the
        // jitter path. The non-jittered upscale (no temporal reconstruction) keeps FXAA to soften
        // the bilinear aliasing.
        let taau_jitter_active = taau_active && self.taau_jitter;
        // B2: when the camera is sub-pixel jittered, the screen-space temporal passes (GI denoiser,
        // reflection resolve) must reproject history with sub-pixel (bilinear) accuracy or the
        // jitter blurs/destabilizes their accumulation. bit1 of the flip word selects that path in
        // gdf_temporal/reflect_temporal; cleared (no jitter) = integer-floor fetch = byte-identical
        // to the legacy path. Computed once here, applied at both call sites.
        let temporal_flip = self.flip_y | if taau_jitter_active { 2 } else { 0 };

        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32();
        self.last = now;
        self.elapsed += dt;
        // Clamp the sim step so a long stall (e.g. resize) can't explode particles.
        let sim_dt = dt.clamp(0.0, 1.0 / 30.0);
        if !self.screenshot_mode {
            self.angle += dt * 0.6; // hold a fixed view when capturing
        } else if self.capture_seq.is_some() {
            // CAPTURE_SEQ: advance the camera a fixed deterministic step per frame so the
            // dumped sequence exercises the temporal passes under motion (stability diff).
            // `CAPTURE_SEQ_STEP` (radians/frame, default 0.015) tunes it; 0 = static (a
            // shimmer/convergence test — the sequence should diff to ~0 when stable).
            let step = std::env::var("CAPTURE_SEQ_STEP")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.015);
            self.angle += step;
        }

        // Stage 0: Tab toggles the free-fly camera (interactive only — never during a
        // headless capture, so the parity baseline stays the fixed orbit). Re-seed the
        // fly camera from the current orbit view each time it is entered.
        if !self.screenshot_mode {
            let tab = self.window.input().key_down(VK_TAB);
            if tab && !self.tab_prev {
                self.cam_mode = match self.cam_mode {
                    camera::CameraMode::Orbit => camera::CameraMode::Fly,
                    camera::CameraMode::Fly => camera::CameraMode::Orbit,
                };
                if self.cam_mode == camera::CameraMode::Fly {
                    self.fly = None; // force re-seed from the orbit view below
                }
            }
            self.tab_prev = tab;
        }

        // Path-trace + GI-denoise screenshots need a long warmup so the static-camera
        // accumulation converges before the frame is captured.
        let warmup = if self.path_trace && !self.rt_debug {
            PATHTRACE_WARMUP
        } else if (self.gdf_gi && self.gi_denoise && self.gi.has_denoise())
            || self.cache_viz
            || self.surface_cache
            || self.reflect_cache
            || taau_active
            || (self.swrt_reflect && self.reflect.has_reflect_temporal())
        {
            // The surface cache / stochastic GGX reflection accrue a sample per frame + temporally
            // accumulate, like the GI denoiser — warm them up before the static screenshot.
            GI_DENOISE_WARMUP
        } else {
            SCREENSHOT_WARMUP
        };

        // Decide whether this frame produces a screenshot: a scheduled capture in
        // screenshot mode (after warmup), or an F2 rising edge interactively.
        let f2 = self.window.input().key_down(VK_F2);
        let f2_pressed = f2 && !self.f2_prev;
        self.f2_prev = f2;
        let capture_this_frame: Option<Capture> = if self.screenshot_mode {
            match self.capture_seq {
                // CAPTURE_SEQ: frames [warmup, warmup+N) each dump to `<path>.NNNN.png`.
                Some(n) => self
                    .frame_no
                    .checked_sub(warmup)
                    .filter(|&i| i < n as u64)
                    .map(|i| Capture {
                        path: seq_capture_path(&self.captures[0].path, i),
                        include_ui: self.captures[0].include_ui,
                    }),
                None => self
                    .frame_no
                    .checked_sub(warmup)
                    .and_then(|i| self.captures.get(i as usize).cloned()),
            }
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

        // Materialize this frame's draw list from the ECS world (the single source of
        // truth). Static scene → identical every frame; the rebuild is cheap (no GPU
        // re-upload, just handle resolution + Rc clones). In world mode the list comes
        // from the streamer instead (rebuilt after the camera moves, below).
        let mut scene = if self.streaming.is_some() {
            Vec::new()
        } else {
            build_scene(&self.world, &self.mesh_registry, &self.material_registry)
        };

        // Orbiting camera framing the whole sample scene — or, in single-object
        // diagnostic mode, a tight orbit centred on one scene object so it can be
        // inspected from every side (azimuth = self.angle, elevation = diag_pitch).
        let (focus, eye) = if let Some(oi) = self.diag_obj.filter(|&i| i < scene.len()) {
            let center = scene[oi].transform.w_axis.truncate();
            let radius = scene[oi].transform.x_axis.truncate().length(); // uniform scale
            let dist = radius * 4.5;
            let pitch = self.diag_pitch.unwrap_or(0.18); // slight elevation by default
            let (sp, cp) = (pitch.sin(), pitch.cos());
            let eye = center + dist * Vec3::new(cp * self.angle.cos(), sp, cp * self.angle.sin());
            (center, eye)
        } else if let Some((eye, target)) = self.level_view {
            // A level's authored camera (e.g. the Sponza demo angle). Headless captures
            // hold it fixed (the byte-identical parity baseline); interactively, orbit it
            // around the level focus so the scene can be inspected from any side. Seeded
            // from the authored camera so `self.angle == 0` reproduces it exactly (no jump
            // on launch), then the per-frame `self.angle += dt*0.6` spins it. Tab still
            // switches to the free-fly camera (seeded from this resolved pose) below.
            if self.screenshot_mode {
                (target, eye)
            } else {
                let offset = eye - target;
                let rh = (offset.x * offset.x + offset.z * offset.z).sqrt();
                let base = offset.z.atan2(offset.x);
                let a = base + self.angle;
                (
                    target,
                    target + Vec3::new(rh * a.cos(), offset.y, rh * a.sin()),
                )
            }
        } else {
            let focus = self.scene_center;
            let dist = self.scene_radius * 1.6;
            let eye = focus
                + Vec3::new(
                    self.angle.cos() * dist,
                    self.scene_radius * 0.55,
                    self.angle.sin() * dist,
                );
            (focus, eye)
        };
        // Stage 0: in fly mode, override the orbit framing with the free camera. Seed
        // it from the orbit view on first entry so the switch is seamless. Headless
        // captures stay in Orbit for the gallery baseline; world mode (Stage D) is the
        // exception — it flies even headless (static at `WORLD_CAM`) so streaming can be
        // positioned and captured.
        let fly_active = self.cam_mode == camera::CameraMode::Fly
            && (!self.screenshot_mode || self.streaming.is_some());
        let (focus, eye) = if fly_active {
            let seed_speed = self.scene_radius * 0.8;
            let fly = self
                .fly
                .get_or_insert_with(|| camera::FlyCamera::from_look(eye, focus, seed_speed));
            if !self.screenshot_mode {
                fly.update(self.window.input(), dt);
            }
            (fly.focus(), fly.position)
        } else {
            (focus, eye)
        };
        // Diagnostic camera override: `CAM_EYE="x,y,z"` (+ optional `CAM_TARGET`) places
        // the camera at a fixed pose for headless inspection of any scene (e.g. flying
        // inside an imported environment like Sponza). Applies before streaming so it
        // can also drive chunk loading.
        let (focus, eye) = match (parse_vec3_env("CAM_EYE"), parse_vec3_env("CAM_TARGET")) {
            (Some(e), Some(t)) => (t, e),
            (Some(e), None) => (focus, e),
            _ => (focus, eye),
        };

        // Stage D: stream chunks in/out around the camera, then rebuild the draw list
        // from the resident chunks (each chunk's transforms already include its origin).
        if let Some(streaming) = &mut self.streaming {
            if streaming.update(&self.device, eye)? {
                info!(
                    "streaming: resident chunks {:?}",
                    streaming.loaded_indices()
                );
            }
            scene = streaming.build_scene();
        }
        let view = Mat4::look_at_rh(eye, focus, Vec3::Y);
        let proj_noflip =
            Mat4::perspective_rh(60f32.to_radians(), cw as f32 / ch as f32, 0.05, 100.0);
        let mut proj = proj_noflip;
        if self.backend == BackendKind::Vulkan {
            proj.y_axis.y *= -1.0; // Vulkan clip-space Y points down
        }
        // The unjittered (but Y-flipped) view-proj — the stable grid the TAAU history lives on.
        let view_proj_stable = proj * view;
        // QHD/UHD TAAU: sub-pixel camera jitter (Halton(2,3)) so successive low-res frames sample
        // different sub-pixel positions; the TAAU history reconstructs full-res detail from them.
        // Applied to the scene projection only (cull_view_proj stays unjittered = stable culling);
        // the GI/reflect denoisers reproject in world space, so the consistent jitter cancels.
        let mut taau_jitter_uv = [0.0f32, 0.0f32];
        if taau_active && self.taau_jitter {
            let j = ((self.frame_no % TAAU_JITTER_LEN) + 1) as u32;
            let jx = (halton(j, 2) - 0.5) * 2.0 / cw as f32; // NDC offset (±1 px in internal res)
            let jy = (halton(j, 3) - 0.5) * 2.0 / ch as f32;
            // clip.xy += offset * clip.w  ⇒  row0/row1 += offset * row3 (row3 = (0,0,-1,0) RH).
            // Negate Y on Vulkan so the screen-space jitter direction matches D3D12 (DX≡VK).
            let sy = if self.backend == BackendKind::Vulkan {
                -1.0
            } else {
                1.0
            };
            proj.z_axis.x -= jx;
            proj.z_axis.y -= jy * sy;
            // UV-space shift the jitter gives the on-screen content, so the TAAU can sample the
            // internal frame at `uv + jitter_uv` to fetch the content that landed on the stable
            // output pixel. Working it through both conventions (jitter NDC shift -> rendered
            // screen UV -> the shader's reconstruct() UV->NDC, which uses sy = flip_y?1:-1):
            //   Δuv.x = +jx/2,  Δuv.y = -jy/2   — identical on D3D12 AND Vulkan (the two Y flips
            // cancel to a single net negation). The previous +jy/2 left a ~1px vertical
            // reprojection error, so the history fetch missed and the jitter degraded to shimmer.
            taau_jitter_uv = [jx * 0.5, -jy * 0.5];
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
            let validation_on = self.validation_on;
            let backend = self.backend;
            let path_spp = self.path_spp;
            // ImGui's display size, mouse hit-testing and clip rects all live in WINDOW
            // (backbuffer) pixels — the UI pass renders into the swapchain backbuffer
            // `(sw,sh)`, and `input.mouse_position()` returns client-area pixels in that
            // same space. With an internal render scale `(cw,ch)` decoupled from the
            // window, using the internal extent here stretched the UI vertices to the
            // backbuffer while leaving the mouse + scissor in the smaller space — widgets
            // were clipped away when dragged and the hit-test was off. Always feed the
            // output extent so all three agree (cw==sw on the default path = unchanged).
            let ui = self
                .gui
                .new_frame(dt, [sw as f32, sh as f32], self.window.input());
            let App {
                gdf,
                gi,
                reflect,
                rt,
                debug_view,
                sun_dir,
                sun_intensity,
                ambient,
                exposure,
                point_lights_on,
                shadows_on,
                shadow_bias,
                shadow_softness,
                shadow_taps,
                quality,
                override_material,
                metallic_override,
                roughness_override,
                realtime_env,
                multibounce,
                legacy_ibl,
                post_mode,
                aliasing,
                compute_post,
                particles_on,
                async_compute_on,
                gpu_cull,
                path_trace,
                rt_debug,
                cornell,
                path_trace_pipeline,
                sdf_trace,
                volume_test,
                sdf_bake,
                sdf_bake_done,
                gdf_merge,
                gdf_merge_done,
                gdf_trace,
                scene_gdf,
                gdf_ao,
                gdf_gi,
                gi_spp,
                cache_relight_period,
                gi_half_res,
                gi_denoise,
                gdf_ssr,
                gdf_reflect,
                gdf_hybrid,
                swrt_reflect,
                gdf_color,
                cache_viz,
                surface_cache,
                reflect_cache,
                firefly_clamp,
                reflect_max_roughness,
                ssr_stochastic,
                profiler_on,
                gpu_timings,
                async_compute_supported,
                level_paths,
                current_level,
                pending_level,
                streaming,
                is_gallery,
                ..
            } = self;
            let async_compute_supported = *async_compute_supported;
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
                    // Stage D: streaming chunk readout (world mode). Fly (WASD) across the
                    // chunk row to stream them in/out.
                    if let Some(s) = streaming.as_ref() {
                        let loaded = s.loaded_indices();
                        ui.text(format!(
                            "world: {}/{} chunks (r={:.0})",
                            loaded.len(),
                            s.chunk_count(),
                            s.stream_radius()
                        ));
                        let names: Vec<&str> = loaded.iter().map(|&i| s.chunk_name(i)).collect();
                        ui.text(format!("  loaded: [{}]", names.join(", ")));
                    }
                    // Stage C: level hot-swap dropdown (level mode only). Selecting a level
                    // requests a rebuild applied at the next frame's start (deferred so the
                    // GPU can idle first). The file names (stems) label the entries.
                    if !level_paths.is_empty() {
                        let names: Vec<&str> = level_paths
                            .iter()
                            .map(|p| {
                                std::path::Path::new(p)
                                    .file_stem()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or(p)
                            })
                            .collect();
                        let mut sel = *current_level;
                        if ui.combo_simple_string("level", &mut sel, &names)
                            && sel != *current_level
                        {
                            *pending_level = Some(sel);
                        }
                    }
                    // RenderQuality tier (Stage D): switching re-applies the preset to the live
                    // knobs below (capability-gated). A manual pick supersedes any startup env
                    // override — the env seam only seeds the initial state. The graph is rebuilt
                    // every frame, so the new tier takes effect immediately.
                    let mut tier_idx = match *quality {
                        quality::RenderQuality::Low => 0usize,
                        quality::RenderQuality::Med => 1,
                        quality::RenderQuality::High => 2,
                    };
                    if ui.combo_simple_string(
                        "RenderQuality",
                        &mut tier_idx,
                        &["low", "med", "high"],
                    ) {
                        let nq = [
                            quality::RenderQuality::Low,
                            quality::RenderQuality::Med,
                            quality::RenderQuality::High,
                        ][tier_idx];
                        *quality = nq;
                        let p = quality::preset(nq);
                        // Re-derive each knob from the preset, preserving the same capability gates
                        // used at construction so a tier can't enable a feature the device lacks.
                        *gi_spp = p.gi_spp.clamp(1, 256);
                        // Gallery stays the byte-identical anchor (period 1); content amortizes.
                        *cache_relight_period = if *is_gallery {
                            1
                        } else {
                            p.cache_relight_period.max(1)
                        };
                        // Half-res GI: content-only (gallery full-res = byte-identical anchor).
                        *gi_half_res = gi.has_upsample() && !*is_gallery && p.gi_half_res;
                        *gi_denoise = gi.has_denoise() && p.gi_denoise;
                        *reflect_cache = *swrt_reflect
                            && gdf.has_surface_cache()
                            && gdf.has_cache_lighting()
                            && p.reflect_cache;
                        *surface_cache =
                            gdf.has_surface_cache() && gdf.has_cache_lighting() && p.surface_cache;
                        *ssr_stochastic = p.ssr_stochastic;
                        *reflect_max_roughness = p.reflect_max_roughness;
                        *gdf_ao = gi.has_ao() && gdf.has_scene_sdf() && p.gdf_ao;
                        *firefly_clamp = p.firefly_clamp;
                        *shadow_softness = p.shadow_softness;
                        *shadow_taps = p.shadow_taps.clamp(1, 16);
                    }
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
                        ui.combo_simple_string("Debug view", debug_view, &DEBUG_VIEWS);
                        ui.input_float3("Sun dir", sun_dir).build();
                        ui.slider("Sun intensity", 0.0, 10.0, sun_intensity);
                        ui.slider("Ambient", 0.0, 0.5, ambient);
                        ui.slider("Exposure", 0.1, 4.0, exposure);
                        ui.checkbox("Point lights", point_lights_on);
                        ui.checkbox("Shadows", shadows_on);
                        ui.slider("Shadow bias", 0.0, 0.01, shadow_bias);
                        ui.slider("Shadow softness (0=hard)", 0.0, 0.1, shadow_softness);
                    }

                    if ui.collapsing_header("Material override", TreeNodeFlags::empty()) {
                        ui.checkbox("Override material", override_material);
                        ui.slider("Metallic", 0.0, 1.0, metallic_override);
                        ui.slider("Roughness", 0.0, 1.0, roughness_override);
                    }

                    if ui.collapsing_header("IBL / Environment", TreeNodeFlags::empty()) {
                        ui.checkbox("Legacy captured-cube IBL (deprecated)", legacy_ibl);
                        if *legacy_ibl {
                            ui.text_disabled("  - prefilter-cube specular + scene capture");
                            ui.checkbox("Real-time env capture", realtime_env);
                            ui.checkbox("Multi-bounce reflections", multibounce);
                        } else {
                            ui.text_disabled("  - default: SW-RT specular + GDF GI diffuse");
                        }
                        ui.combo_simple_string("Post effect", post_mode, &POST_EFFECTS);
                        ui.checkbox("Transient aliasing", aliasing);
                    }

                    if ui.collapsing_header("Compute / GPGPU (Phase 7)", TreeNodeFlags::empty()) {
                        ui.checkbox("Compute post (blur)", compute_post);
                        ui.checkbox("GPU particles", particles_on);
                        if async_compute_supported {
                            ui.checkbox("  - async compute queue", async_compute_on);
                        } else {
                            ui.text_disabled("  - async compute (no dedicated queue)");
                        }
                        ui.checkbox("GPU culling (indirect)", gpu_cull);
                    }

                    if rt.has_path() && rt.has_scene() {
                        if ui.collapsing_header("Ray tracing (Phase 8)", TreeNodeFlags::empty()) {
                            ui.checkbox("Path trace (inline ray query)", path_trace);
                            if *path_trace {
                                ui.checkbox("  - debug: instance + shadow viz", rt_debug);
                                if !*rt_debug {
                                    if rt.has_cornell() {
                                        ui.checkbox("  - Cornell box", cornell);
                                    }
                                    if rt.has_pt_pipeline() {
                                        ui.checkbox(
                                            "  - pipeline + SBT (vs inline)",
                                            path_trace_pipeline,
                                        );
                                    }
                                    ui.text(format!(
                                        "  - {} spp accumulated ({})",
                                        rt.accum_frame().saturating_mul(path_spp),
                                        if *path_trace_pipeline {
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
                        ui.checkbox("SDF sphere trace (compute, no HW RT)", sdf_trace);
                        if *sdf_trace {
                            ui.text_disabled("  - analytic SDF scene (Stage A)");
                        }
                        if gdf.has_volume() {
                            ui.checkbox("3D volume test (fill + slice view)", volume_test);
                            if *volume_test {
                                ui.text_disabled("  - Stage B RHI smoke test");
                            }
                        }
                        if gdf.has_bake() {
                            if ui.checkbox("SDF bake (per-mesh, slice view)", sdf_bake) {
                                *sdf_bake_done = false; // re-bake when re-enabled
                            }
                            if *sdf_bake {
                                ui.text_disabled("  - Stage B2: baked sphere ≈ analytic");
                            }
                        }
                        if gdf.has_merge() {
                            if ui.checkbox("GDF merge (instances, slice view)", gdf_merge) {
                                *gdf_merge_done = false; // re-merge when re-enabled
                            }
                            if *gdf_merge {
                                ui.text_disabled("  - Stage B3: min-merged instances");
                            }
                        }
                        if gdf.has_gdf_trace() {
                            ui.checkbox("GDF SW ray trace (compute)", gdf_trace);
                            if *gdf_trace {
                                ui.text_disabled("  - Stage B4: sphere-march baked GDF");
                            }
                        }
                        if gdf.has_scene_sdf() {
                            ui.checkbox("Scene GDF (world, live camera)", scene_gdf);
                            if *scene_gdf {
                                ui.text_disabled("  - Stage C1: fused world scene SDF");
                            }
                        }
                        if gi.has_ao() && gdf.has_scene_sdf() {
                            ui.checkbox("GDF ambient occlusion (deferred)", gdf_ao);
                            if *gdf_ao {
                                ui.text_disabled("  - Stage C2: GDF AO into ambient");
                            }
                        }
                        if gi.has_gi() && gdf.has_scene_sdf() {
                            ui.checkbox("GDF diffuse GI (deferred)", gdf_gi);
                            if *gdf_gi {
                                ui.text_disabled("  - Stage C3: 1-bounce stochastic");
                                if gi.has_denoise() {
                                    ui.checkbox("  - C4 spatio-temporal denoise", gi_denoise);
                                }
                            }
                        }
                        if reflect.has_ssr() {
                            ui.checkbox("Screen-space reflections (viz)", gdf_ssr);
                            if *gdf_ssr {
                                ui.text_disabled("  - Stage C5: SSR buffer (C7 composites)");
                            }
                        }
                        if reflect.has_gdf_reflect() && gdf.has_scene_sdf() {
                            ui.checkbox("GDF reflections (viz)", gdf_reflect);
                            if *gdf_reflect {
                                ui.text_disabled("  - Stage C6: SSR-miss fallback (sky on escape)");
                            }
                        }
                        if reflect.has_composite() && gdf.has_scene_sdf() {
                            ui.checkbox("Hybrid reflections (viz)", gdf_hybrid);
                            if *gdf_hybrid {
                                ui.text_disabled("  - Stage C7: SSR over GDF / sky composite");
                            }
                        }
                        if reflect.has_lit_history() && gdf.has_scene_sdf() {
                            ui.checkbox("SW-RT reflections in lighting", swrt_reflect);
                            if *swrt_reflect {
                                ui.text_disabled("  - Stage C7c: replaces IBL prefilter specular");
                            }
                        }
                        if gdf.has_scene_albedo() {
                            ui.checkbox("Colored GDF re-light (C8a)", gdf_color);
                            if *gdf_color {
                                ui.text_disabled("  - per-voxel albedo: colored GI + reflections");
                            }
                        }
                        if gdf.has_surface_cache() {
                            ui.checkbox("Surface-cache atlas (C8b1/2 viz)", cache_viz);
                            if *cache_viz {
                                ui.text_disabled("  - mesh cards: lit radiance (multibounce)");
                            }
                            if gdf.has_cache_lighting() {
                                ui.checkbox(
                                    "Surface cache in GI/reflections (C8b3)",
                                    surface_cache,
                                );
                                if *surface_cache {
                                    ui.text_disabled("  - cached multibounce radiance lookup");
                                }
                            }
                        }
                        ui.checkbox("Firefly clamp (reflections/GI)", firefly_clamp);
                        if *swrt_reflect {
                            ui.slider(
                                "Reflection max roughness (C8d)",
                                0.0,
                                1.0,
                                reflect_max_roughness,
                            );
                            ui.text_disabled("  - screen mirror below, GDF prefilter above");
                            ui.checkbox(
                                "Stochastic glossy SSR (else full-res mirror)",
                                ssr_stochastic,
                            );
                        }
                    }

                    if ui.collapsing_header("Profiling & debug (Phase 9)", open) {
                        ui.checkbox("GPU profiler", profiler_on);
                        if *profiler_on {
                            if gpu_timings.is_empty() {
                                ui.text_disabled("  (measuring…)");
                            } else {
                                let mut total = 0.0;
                                for (name, ms) in gpu_timings.iter() {
                                    ui.text(format!("  {name:<9} {ms:6.3} ms"));
                                    total += ms;
                                }
                                ui.text(format!("  {:<9} {total:6.3} ms", "total"));
                            }
                        }
                    }
                });
        }

        // This slot's previous submission is complete (waited on its fence above), so
        // its timestamp queries are ready: read them back and turn the tick
        // boundaries into per-pass GPU milliseconds for the profiler UI (next frame).
        if self.profiler_on && !self.slot_pass_names[fif].is_empty() {
            let heap = &self.query_heaps[fif];
            let ticks = heap.read();
            let period_ns = heap.period_ns();
            self.gpu_timings = self.slot_pass_names[fif]
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let dt = ticks[i + 1].saturating_sub(ticks[i]);
                    (name.clone(), dt as f32 * period_ns * 1e-6)
                })
                .collect();
            // Headless dump (screenshot mode has no UI): log per-pass GPU ms so PROFILE_GPU
            // is useful for measuring without the interactive table.
            if self.screenshot_mode {
                let total: f32 = self.gpu_timings.iter().map(|(_, ms)| ms).sum();
                let rows: String = self
                    .gpu_timings
                    .iter()
                    .map(|(n, ms)| format!("  {n:<20} {ms:.4} ms"))
                    .collect::<Vec<_>>()
                    .join("\n");
                tracing::info!("GPU profile (total {total:.4} ms):\n{rows}");
            }
        }

        self.in_flight[fif].reset()?;

        let cmd = &self.command_buffers[fif];
        cmd.begin()?;

        // Lighting: a level supplies its own sun + point lights; otherwise the gallery's
        // code-default sun + two coloured point lights (preserved exactly = byte-identical).
        let r = self.model_radius;
        let point_intensity = r * r * 8.0;
        let (sun_dir, sun_intensity, point_count, point_pos, point_color) =
            match &self.level_lighting {
                Some(ll) => (
                    ll.sun_dir,
                    ll.sun_intensity,
                    ll.point_count,
                    ll.point_pos,
                    ll.point_color,
                ),
                None => (
                    self.sun_dir,
                    self.sun_intensity,
                    2,
                    [
                        [r * 2.0, r * 1.5, 0.0, 0.0],
                        [-r * 2.0, r * 1.0, r * 1.5, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [1.0, 0.35, 0.2, point_intensity],
                        [0.3, 0.5, 1.0, point_intensity],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                ),
            };

        // (Re)capture the environment into the "write" cube set before the main graph
        // samples it (see `ibl.rs`). The reflection probe is fixed at the scene centre
        // (`focus`), NOT the camera, to avoid per-surface parallax error.
        // Deprecation: with the SW-RT specular default, the prefilter cube (its only consumer
        // of the scene-in-cube capture) is unused — so demote the capture to sky-only (empty
        // scene + single bounce). The diffuse irradiance (mip 2) and skybox (mip 1) are sky
        // anyway, so this is behavior-preserving for the default path, just cheaper. Legacy
        // IBL keeps the full scene capture for the prefilter-cube specular.
        // Demote only when SW-RT actually feeds the specular (so the no-compute / legacy
        // fallback, where pbr still samples the prefilter cube, keeps its scene capture).
        let (cap_scene, cap_multibounce): (&[SceneObject], bool) = if self.swrt_reflect {
            (&[], false)
        } else {
            (&scene, self.multibounce)
        };
        self.ibl.maybe_capture(
            cmd,
            self.realtime_env,
            cap_multibounce,
            cap_scene,
            &self.ground_vbuf,
            &self.ground_ibuf,
            self.ground_count,
            focus,
            sun_dir,
            sun_intensity,
            self.ambient,
            self.flip_y,
            self.backend == BackendKind::Vulkan,
        );

        // The main lighting pass samples the most recently written set.
        let ibl_indices = self.ibl.lighting_indices();

        // Directional light view-projection: an orthographic box covering the whole
        // scene, looking from the sun toward it. Backend-neutral (the pbr shader
        // handles the Vulkan/D3D12 shadow-UV flip).
        let shadow_center = if self.is_gallery {
            Vec3::new(0.0, self.model_radius * 0.5, 0.0)
        } else {
            self.scene_center
        };
        let light_vp = light_view_proj(sun_dir, shadow_center, self.scene_radius);

        // Write this frame's globals slice.
        let globals = Globals {
            camera_pos: [eye.x, eye.y, eye.z, 0.0],
            sun_direction: normalize3(sun_dir),
            sun_color: [1.0, 1.0, 1.0, sun_intensity],
            ambient: [self.ambient, self.ambient, self.ambient, self.exposure],
            counts: [
                if self.point_lights_on { point_count } else { 0 },
                self.debug_view as i32,
                (PREFILTER_MIPS - 1) as i32, // prefilter max LOD
                self.shadows_on as i32,
            ],
            point_pos,
            point_color,
            light_view_proj: light_vp.to_cols_array(),
            shadow: [
                self.shadow_bias,
                1.0 / SHADOW_SIZE as f32,
                self.shadow_softness, // z: PCSS-lite penumbra scale (0 = hard PCF)
                self.shadow_taps as f32, // w: soft-shadow tap count (RenderQuality; soft path only)
            ],
            inv_view_proj,
            ibl: ibl_indices,
            // Reflection-probe centre (matches the env-capture eye) + a box proxy for
            // parallax-corrected specular IBL. The box floor sits on the ground plane
            // (y = 0) and its walls/ceiling match the captured ground extent, so
            // reflected-floor rays re-anchor onto the actual flat ground instead of a
            // sphere that bent them up to the (darker) horizon.
            probe: [focus.x, focus.y, focus.z, 1.0],
            probe_box_min: [-self.scene_radius * 1.3, 0.0, -self.scene_radius * 1.3, 0.0],
            probe_box_max: [
                self.scene_radius * 1.3,
                self.scene_radius * 2.0,
                self.scene_radius * 1.3,
                0.0,
            ],
            // Last frame's view-projection (updated end-of-frame) so the SSR history
            // sample reprojects the world hit point into the previous frame (Stage C7b).
            prev_view_proj: self.prev_view_proj,
        };
        let globals_offset = fif as u64 * GLOBALS_SLICE;
        // Firefly clamp ceiling (raw radiance, max component). ~8 keeps diffuse + moderate
        // gloss but caps blown-out specular spikes; 1e30 = effectively off (byte-identical).
        let firefly_max = if self.firefly_clamp { 8.0f32 } else { 1e30 };
        self.deferred
            .write_globals(globals_offset, globals_bytes(&globals))?;

        // Phase 8 M4: manage the path tracer's persistent accumulation buffer and
        // reset key BEFORE building the render graph — the fallible buffer
        // (re)allocation must not sit on a `?` early-return path while the graph holds
        // borrows of transient resources.
        let pt_active =
            self.path_trace && !self.rt_debug && self.rt.has_path() && self.rt.has_instance_table();
        // The path tracer uses the Cornell scene (fixed front camera) when toggled,
        // else the orbiting open scene. `pt_eye` / `pt_inv_vp` feed the trace rays.
        let use_cornell = pt_active && self.cornell && self.rt.has_cornell();
        let (pt_eye, pt_inv_vp) = if use_cornell {
            RtSystem::cornell_camera(cw, ch, self.backend == BackendKind::Vulkan)
        } else {
            (eye, inv_view_proj)
        };
        self.rt.prepare(
            &self.device,
            pt_active,
            use_cornell,
            cw,
            ch,
            pt_eye,
            self.sun_dir,
            self.sun_intensity,
        )?;

        // C4: (re)allocate the GI denoiser history + reset accumulation on a lighting/
        // quality change (NOT camera — the temporal pass reprojects). Runs before the
        // graph, like the path-tracer's accumulation prepare.
        let gi_denoise_active = self.gdf_gi && self.gi_denoise && self.gi.has_denoise();
        if gi_denoise_active {
            let mut key = 0u64;
            for b in self.sun_dir.iter() {
                key = key
                    .wrapping_mul(0x100_0000_01b3)
                    .wrapping_add(b.to_bits() as u64);
            }
            key = key
                .wrapping_mul(0x100_0000_01b3)
                .wrapping_add(self.sun_intensity.to_bits() as u64);
            key = key
                .wrapping_mul(0x100_0000_01b3)
                .wrapping_add(self.gi_spp as u64);
            self.gi.prepare_denoise(&self.device, cw, ch, key)?;
        }
        // C7b: (re)allocate the lit-color history buffers for the hybrid reflection path
        // (the standalone viz or the C7c lighting feedback).
        if self.gdf_hybrid || self.swrt_reflect {
            self.reflect.prepare_history(&self.device, cw, ch)?;
        }
        // Stochastic SSR runs at half-res; (re)allocate its temporal accumulation buffers.
        let (hcw, hch) = (cw.div_ceil(2), ch.div_ceil(2));
        if self.swrt_reflect && self.ssr_stochastic && self.reflect.has_ssr_resolve() {
            self.reflect.prepare_ssr_accum(&self.device, hcw, hch)?;
        }
        // C8j: (re)allocate the stochastic GDF-reflection temporal accumulation buffers (full-res).
        if self.swrt_reflect && self.reflect.has_reflect_temporal() {
            self.reflect.prepare_reflect_accum(&self.device, cw, ch)?;
        }
        // QHD/UHD TAAU: (re)allocate the full-res (output) history.
        if taau_active {
            self.taau.prepare(&self.device, sw, sh, 0)?;
        }

        let extent = Extent2D::new(cw, ch); // scene render extent (RENDER_RES or swapchain)
        let swap_extent = Extent2D::new(sw, sh); // display/backbuffer extent
        // Stage B3: the finer clipmap-level volumes each GDF pass transitions to sampled
        // (empty for the single-level gallery). Bound before the graph so it outlives the
        // pass closures that borrow it.
        let scene_clip_vols = self.gdf.clip_level_volumes();
        let mut graph = RenderGraph::new();
        // The backbuffer is the actual swapchain image (display extent); tonemap samples the
        // render-extent HDR by UV, so a render≠display extent just means a downscale at present.
        let backbuffer = graph.import_backbuffer(self.swapchain.format(), swap_extent);
        let g_albedo = graph.create_color("g_albedo", GB_ALBEDO_FMT, extent);
        let g_normal = graph.create_color("g_normal", GB_NORMAL_FMT, extent);
        let g_material = graph.create_color("g_material", GB_MATERIAL_FMT, extent);
        let g_position = graph.create_color("g_position", GB_POSITION_FMT, extent);
        let g_depth = graph.create_depth("g_depth", extent);
        let shadow_map = graph.create_depth("shadow_map", Extent2D::new(SHADOW_SIZE, SHADOW_SIZE));
        let hdr = graph.create_color("hdr", HDR_FORMAT, extent);
        // Phase 7: compute post writes the blurred HDR into a storage image that the
        // tonemap pass samples instead of the raw `hdr` target.
        let hdr_post = if self.compute_post {
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
        self.deferred
            .record_shadow(&mut graph, shadow_map, &scene, light_vp);
        self.deferred.record_gbuffer(
            &mut graph,
            gbuf,
            &scene,
            &self.ground_vbuf,
            &self.ground_ibuf,
            self.ground_count,
            view_proj,
            self.ambient,
            self.override_material,
            self.metallic_override,
            self.roughness_override,
        );
        // Stage C2/C3 (GDF-lighting consumers, see `gi.rs`) share the world scene GDF:
        // import its handle once + record the one-time fused-scene bake (the volume is
        // owned by `GdfSystem`), then AO + GI read it. Recorded before lighting so the
        // graph orders gbuffer -> AO/GI -> lighting. The bake latch is shared with the C1
        // trace (whichever runs first bakes).
        let scene_gdf_vol = self.gdf.scene_gdf_volume();
        let (scene_aabb_min, scene_aabb_max) = self.gdf.scene_aabb();
        // Stage B: the clipmap descriptor the SW-RT shaders sample the scene field through
        // (single level today = the legacy volume). `(0, 1)` is an inert fallback never used
        // (the GDF passes gate on `scene_gdf_vol.is_some()`, which implies a descriptor).
        let scene_clip = self.gdf.clip_descriptor().unwrap_or((0, 1));
        let scene_gdf_ext = if (self.gdf_ao
            || self.gdf_gi
            || self.gdf_reflect
            || self.gdf_hybrid
            || self.swrt_reflect
            || self.cache_viz
            || self.surface_cache)
            && scene_gdf_vol.is_some()
        {
            let ext = graph.import_external("scene_gdf");
            if !self.scene_gdf_baked {
                self.gdf.record_scene_bake(&mut graph, ext);
                self.scene_gdf_baked = true;
            }
            Some(ext)
        } else {
            None
        };
        // Stage C8a: the per-voxel albedo volumes (colored GI + reflection re-light). Import
        // the shared handle once + bake once (separate from the distance bake so that stays
        // bit-identical); the re-light consumers (C3 GI, C6/C7 reflection) read it. Gated on
        // `gdf_color` so `P11_GDF_COLOR=0` is the achromatic pre-C8a path.
        let scene_albedo_ext = if self.gdf_color
            && (self.gdf_gi
                || self.gdf_reflect
                || self.gdf_hybrid
                || self.swrt_reflect
                || self.cache_viz
                || self.surface_cache)
            && self.gdf.has_scene_albedo()
        {
            let ext = graph.import_external("scene_albedo");
            if !self.scene_albedo_baked {
                self.gdf.record_scene_albedo_bake(&mut graph, ext);
                self.scene_albedo_baked = true;
            }
            Some(ext)
        } else {
            None
        };
        // The (volumes, handle) pair the re-light consumers take; `None` => constant albedo.
        let scene_albedo = match (self.gdf.scene_albedo(), scene_albedo_ext) {
            (Some(vols), Some(ext)) => Some((vols, ext)),
            _ => None,
        };
        // C8b1: capture the mesh-card surface cache once (static geometry + albedo), reading
        // the scene GDF (+ the C8a albedo volumes for the captured color). C8b2 then re-lights
        // it every frame into a radiance atlas (multibounce gather from the previous frame).
        let cache_active = (self.cache_viz || self.surface_cache || self.reflect_cache)
            && self.gdf.has_surface_cache();
        let scene_cache_ext = match (cache_active, scene_gdf_ext) {
            (true, Some(gdf_ext)) => {
                let ext = graph.import_external("scene_cache");
                if !self.scene_cache_captured {
                    self.gdf
                        .record_cache_capture(&mut graph, gdf_ext, scene_albedo_ext, ext);
                    self.scene_cache_captured = true;
                }
                Some(ext)
            }
            _ => None,
        };
        // Stage D3: period-aware temporal alpha keeps the visible cards converged within the
        // screenshot warmup as the period rises; gallery (feedback off) keeps the legacy 0.35.
        let relight_alpha = if self.cache_feedback {
            (0.35 * (self.cache_relight_period as f32 / 8.0)).clamp(0.35, 0.8)
        } else {
            0.35
        };
        // This frame's reset flag, captured for the async relight (recorded in the submit section);
        // the in-graph (sync) path consumes it inline below.
        let cache_reset_this_frame = self.scene_cache_reset;
        // C8b2: re-light the cache (direct sun + sky + multibounce gather from last frame). Async:
        // the relight runs on the compute queue (submit section) and consumers read the previous
        // frame's radiance — the graph handle is only a placeholder for the consumer reads (the
        // cross-queue semaphore orders the data). Sync: record visibility + relight in-graph.
        let scene_cache_lit_ext = match (scene_gdf_ext, scene_cache_ext) {
            (Some(gdf_ext), Some(cache_ext)) if self.gdf.has_cache_lighting() => {
                let ext = graph.import_external("scene_cache_lit");
                if !self.async_cache_on {
                    // Stage D2b: per-card camera visibility (Y-flip-free planes => DX≡VK).
                    let card_vis_ext = if self.cache_feedback {
                        self.gdf
                            .record_cache_visibility(&mut graph, frustum_planes(cull_view_proj))
                    } else {
                        None
                    };
                    self.gdf.record_cache_light(
                        &mut graph,
                        gdf_ext,
                        cache_ext,
                        ext,
                        self.sun_dir,
                        self.sun_intensity,
                        self.cache_relight_spp,
                        self.frame_no as u32,
                        cache_reset_this_frame,
                        self.cache_relight_period,
                        card_vis_ext,
                        relight_alpha,
                    );
                    self.scene_cache_reset = false;
                }
                Some(ext)
            }
            _ => None,
        };
        // C8b3/C8g: the (indices, lit-handle) the consumers use to read the cached multibounce
        // radiance at a hit (instead of a per-ray re-light). `None` => per-ray. Split so the
        // reflection cache (C8g, default) is independent of the heavier per-ray GI cache (opt-in).
        let cache_read = self.gdf.surface_cache_read();
        let gi_cache_arg: Option<([u32; 5], ResourceId)> = match scene_cache_lit_ext {
            Some(ext) if self.surface_cache => {
                cache_read.map(|(c, p, r, n, t)| ([c, p, r, n, t], ext))
            }
            _ => None,
        };
        let reflect_cache_arg: Option<([u32; 5], ResourceId)> = match scene_cache_lit_ext {
            Some(ext) if self.reflect_cache => {
                cache_read.map(|(c, p, r, n, t)| ([c, p, r, n, t], ext))
            }
            _ => None,
        };
        // Stage C2: GDF ambient occlusion, multiplied into the lighting ambient term.
        let gdf_ao_out = match (self.gdf_ao, scene_gdf_vol, scene_gdf_ext) {
            (true, Some(vol), Some(ext)) => Some(self.gi.record_ao(
                &mut graph,
                vol,
                ext,
                scene_aabb_min,
                scene_aabb_max,
                g_depth,
                g_normal,
                extent,
                inv_view_proj,
                cw,
                ch,
                self.flip_y,
                scene_clip,
                &scene_clip_vols,
            )),
            _ => None,
        };
        // Stage C3: 1-bounce diffuse GI added to the ambient term, optionally denoised (C4).
        let gdf_gi_out = match (self.gdf_gi, scene_gdf_vol, scene_gdf_ext) {
            (true, Some(vol), Some(ext)) => {
                // Stage D1: trace at half res (1/4 the rays) when enabled, then joint-bilateral
                // upsample to full res before the denoiser. gdf_gi.slang samples the G-buffer by
                // normalized UV, so a half extent + half dims trace correctly with no shader change.
                let half_gi = self.gi_half_res;
                let (gw, gh) = if half_gi { (hcw, hch) } else { (cw, ch) };
                let gi_extent = if half_gi {
                    Extent2D::new(gw, gh)
                } else {
                    extent
                };
                let traced = self.gi.record_gi(
                    &mut graph,
                    vol,
                    ext,
                    scene_aabb_min,
                    scene_aabb_max,
                    g_depth,
                    g_normal,
                    gi_extent,
                    inv_view_proj,
                    self.sun_dir,
                    self.sun_intensity,
                    gw,
                    gh,
                    self.flip_y,
                    self.gi_spp,
                    self.frame_no as u32,
                    scene_albedo,
                    gi_cache_arg,
                    firefly_max,
                    scene_clip,
                    &scene_clip_vols,
                    self.gi_max_steps,
                );
                let raw = if half_gi {
                    self.gi.record_upsample(
                        &mut graph,
                        traced,
                        g_depth,
                        g_normal,
                        extent,
                        inv_view_proj,
                        scene_aabb_min,
                        scene_aabb_max,
                        cw,
                        ch,
                        gw,
                        gh,
                        self.flip_y,
                    )
                } else {
                    traced
                };
                let out = if gi_denoise_active {
                    self.gi.record_denoise(
                        &mut graph,
                        raw,
                        g_depth,
                        g_normal,
                        extent,
                        inv_view_proj,
                        self.prev_view_proj,
                        scene_aabb_min,
                        scene_aabb_max,
                        cw,
                        ch,
                        temporal_flip,
                        self.firefly_clamp,
                    )
                } else {
                    raw
                };
                Some(out)
            }
            _ => None,
        };
        // Stage C7c: hybrid SW-RT reflection feeding the lighting specular. Built BEFORE
        // lighting (it replaces the prefilter-cube IBL specular). SSR runs in history mode
        // (reprojected previous-frame raw radiance) so it never reads this frame's
        // not-yet-written HDR; the GDF reflect + composite are already raw radiance, so the
        // composite is a drop-in for the cube `prefiltered` (pbr applies exposure once). The
        // lit-color history for the NEXT frame's SSR is captured after lighting, below.
        let swrt_reflect_out = match (
            self.swrt_reflect && self.reflect.has_lit_history(),
            scene_gdf_vol,
            scene_gdf_ext,
        ) {
            (true, Some(vol), Some(ext)) => {
                // C8d: the SSR that feeds the composite. Default = full-res screen MIRROR — the
                // accurate on-screen source (real neighbour pixels via the reprojected lit-history;
                // correct colour + geometry the GDF sphere-trace can't match). The composite uses
                // it below `reflect_max_roughness` and the GDF prefilter above, with a luminance
                // gate that routes off-screen / grazing bad hits to the GDF (also keeping the march
                // VK≡DX-stable). `P11_SSR_STOCHASTIC` selects the half-res GGX trace + ratio-
                // estimator resolve instead (the glossy path; it goes dark on sharp metals).
                let ssr = if self.ssr_stochastic {
                    let half = Extent2D::new(hcw, hch);
                    let (ssr_a, ssr_b) = self.reflect.record_ssr(
                        &mut graph,
                        self.deferred.globals_buffer(),
                        globals_offset,
                        hdr,
                        g_depth,
                        g_normal,
                        g_material,
                        half,
                        view_proj.to_cols_array(),
                        inv_view_proj,
                        eye,
                        hcw,
                        hch,
                        cw,
                        ch,
                        self.flip_y,
                        self.frame_no as u32,
                        self.scene_radius * 1.5,
                        self.scene_radius * 0.06,
                        true,
                        self.firefly_clamp,
                        true, // stochastic GGX jitter
                    );
                    self.reflect.record_ssr_resolve(
                        &mut graph,
                        ssr_a,
                        ssr_b,
                        g_depth,
                        g_normal,
                        g_material,
                        half,
                        inv_view_proj,
                        self.prev_view_proj,
                        eye,
                        hcw,
                        hch,
                        self.flip_y,
                        self.scene_radius * 0.02,
                        firefly_max,
                        2.0,
                    )
                } else {
                    self.reflect
                        .record_ssr(
                            &mut graph,
                            self.deferred.globals_buffer(),
                            globals_offset,
                            hdr,
                            g_depth,
                            g_normal,
                            g_material,
                            extent,
                            view_proj.to_cols_array(),
                            inv_view_proj,
                            eye,
                            cw,
                            ch,
                            cw,
                            ch,
                            self.flip_y,
                            self.frame_no as u32,
                            self.scene_radius * 1.5,
                            self.scene_radius * 0.06,
                            true, // history mode: reprojected raw-radiance previous frame
                            self.firefly_clamp,
                            false, // mirror ray (composite handles roughness via the GDF prefilter)
                        )
                        .0
                };
                // Stage D3: trace the reflection at half res (1/4 the rays) + bilateral upsample
                // to full res before the temporal resolve. gdf_reflect.slang samples the G-buffer
                // by normalized UV, so a half extent + half dims trace correctly (no shader change).
                let refl_half = self.reflect_half_res;
                let (rw, rh) = if refl_half { (hcw, hch) } else { (cw, ch) };
                let refl_extent = if refl_half {
                    Extent2D::new(rw, rh)
                } else {
                    extent
                };
                let refl_traced = self.reflect.record_gdf_reflect(
                    &mut graph,
                    vol,
                    ext,
                    scene_aabb_min,
                    scene_aabb_max,
                    g_depth,
                    g_normal,
                    g_material,
                    refl_extent,
                    inv_view_proj,
                    eye,
                    self.sun_dir,
                    self.sun_intensity,
                    rw,
                    rh,
                    self.flip_y,
                    self.frame_no as u32,
                    scene_albedo,
                    reflect_cache_arg,
                    scene_clip,
                    &scene_clip_vols,
                    self.reflect_max_steps,
                );
                let gdf_refl = if refl_half {
                    self.gi.record_upsample(
                        &mut graph,
                        refl_traced,
                        g_depth,
                        g_normal,
                        extent,
                        inv_view_proj,
                        scene_aabb_min,
                        scene_aabb_max,
                        cw,
                        ch,
                        rw,
                        rh,
                        self.flip_y,
                    )
                } else {
                    refl_traced
                };
                // C8j: temporally resolve the stochastic GGX GDF reflection (UE-style; the rough
                // lobe is sampled by real rays + denoised, so it's correctly blurred without an
                // image-space prefilter that over-brightens rough metals).
                let gdf_resolved = self.reflect.record_reflect_temporal(
                    &mut graph,
                    gdf_refl,
                    g_depth,
                    g_material,
                    extent,
                    cw,
                    ch,
                    inv_view_proj,
                    self.prev_view_proj,
                    eye,
                    temporal_flip,
                    self.scene_radius * 0.02,
                    64.0,
                    firefly_max,
                    0.25, // tonemap-space range for stable HDR accumulation
                );
                Some(self.reflect.record_composite(
                    &mut graph,
                    ssr,
                    gdf_resolved,
                    g_material,
                    extent,
                    cw,
                    ch,
                    1.0,
                    firefly_max,
                    self.reflect_max_roughness,
                ))
            }
            _ => None,
        };
        self.deferred.record_lighting(
            &mut graph,
            hdr,
            gbuf,
            shadow_map,
            gdf_ao_out,
            gdf_gi_out,
            swrt_reflect_out,
            globals_offset,
            self.flip_y,
            // Two-sided shading for imported scenes (single-sided walls seen from inside);
            // the gallery stays single-sided so its baseline is byte-identical.
            !self.is_gallery,
        );
        // C7c: capture this frame's lit HDR (as raw radiance) for next frame's SSR history.
        // Reads the lit `hdr` (not the post-blur), so it sequences after the lighting pass.
        if swrt_reflect_out.is_some() {
            self.reflect.record_lit_history(
                &mut graph,
                hdr,
                cw,
                ch,
                1.0 / self.exposure.max(1e-4),
                firefly_max,
            );
        }
        if let Some(hdr_post) = hdr_post {
            self.deferred
                .record_compute_post(&mut graph, hdr, hdr_post, cw, ch);
        }

        // Phase 7 GPU particles: a compute pass advances the persistent particle
        // buffer; an external graph resource sequences it before the draw pass.
        let particles_ext = if self.particles_on {
            Some(graph.import_external("particles"))
        } else {
            None
        };
        // This frame's ping-pong buffer indices (read the previous write). Captured
        // before the end-of-frame `advance()` so the async-submit path and the draw
        // pass all reference the same pair.
        let particle_read = self.particles.read_index();
        let particle_write = self.particles.write_index();
        // Run the sim on the async-compute queue this frame? (Else it's a graph
        // compute pass on the graphics queue, below.)
        // Async cache relight owns the per-frame compute submission; particles fall back to the
        // graph sim when it's on (avoids two consumers of the single compute command buffer).
        let async_sim = self.particles_on
            && self.async_compute_supported
            && self.async_compute_on
            && !self.async_cache_on;
        if let (false, Some(particles_ext)) = (async_sim, particles_ext) {
            self.particles
                .record_sim(&mut graph, particles_ext, sim_dt, self.elapsed);
        }

        // Phase 7 GPU culling: reset the indirect args, frustum-cull the instance grid
        // into a visible list + draw count, then draw indirectly. A compact grid
        // floating above the scene, so the scene stays visible and orbiting the camera
        // culls cubes off the frustum edges.
        let grid = CullGrid {
            spacing: self.scene_radius * 0.14,
            height: self.scene_radius * 1.15,
            cube_scale: self.scene_radius * 0.045,
            cube_radius: self.scene_radius * 0.045 * 0.5 * 3.0_f32.sqrt(),
        };
        let cull_res = if self.gpu_cull {
            Some(CullSystem::import(&mut graph))
        } else {
            None
        };
        if let Some((args_ext, visible_ext)) = cull_res {
            self.cull.record_cull(
                &mut graph,
                args_ext,
                visible_ext,
                frustum_planes(cull_view_proj),
                &grid,
            );
        }

        // Phase 8 ray tracing: M4 inline path tracer (default) or the M3 trace viz
        // (debug). The chosen compute pass writes a storage image the tonemap pass
        // displays in place of the rasterized HDR.
        let rt_on = self.path_trace && (self.rt.has_path() || self.rt.has_trace());
        let rt_out = if rt_on {
            Some(graph.create_storage_image("rt_out", HDR_FORMAT, extent))
        } else {
            None
        };
        if let Some(rt_out) = rt_out {
            if pt_active {
                self.rt.record_path(
                    &mut graph,
                    rt_out,
                    use_cornell,
                    self.path_trace_pipeline,
                    pt_inv_vp,
                    pt_eye,
                    self.sun_dir,
                    self.sun_intensity,
                    cw,
                    ch,
                    self.flip_y,
                    self.path_spp,
                );
            } else if self.rt.has_trace() {
                self.rt.record_trace(
                    &mut graph,
                    rt_out,
                    inv_view_proj,
                    eye,
                    self.sun_dir,
                    cw,
                    ch,
                    self.flip_y,
                );
            }
        }

        // Phase 11 Stage A: compute software ray trace of the analytic SDF scene,
        // written to a storage image the tonemap pass displays in place of the HDR
        // (mirrors the M3 `rt_trace` viz path). Only when the HW-RT path is off.
        let sdf_out = if self.sdf_trace && rt_out.is_none() {
            Some(self.gdf.record_sdf_trace(
                &mut graph,
                extent,
                inv_view_proj,
                eye,
                self.sun_dir,
                self.sun_intensity,
                cw,
                ch,
                self.flip_y,
            ))
        } else {
            None
        };

        // Phase 11 Stage B (B1): 3D volume smoke test — fill a storage volume, then
        // view a trilinear-sampled Z slice. Only when the other replacements are off.
        let vol_out = if self.volume_test && rt_out.is_none() && sdf_out.is_none() {
            Some(self.gdf.record_volume_test(&mut graph, extent, cw, ch))
        } else {
            None
        };

        // Phase 11 Stage B (B2): bake a mesh's signed-distance field into the volume,
        // then view a slice through the same `volume_view` pass B1 uses. The bake is
        // O(voxels*tris): run it once (`sdf_bake_done`) and only re-view afterwards.
        let bake_out =
            if self.sdf_bake && rt_out.is_none() && sdf_out.is_none() && vol_out.is_none() {
                let out =
                    self.gdf
                        .record_bake_view(&mut graph, extent, cw, ch, !self.sdf_bake_done);
                self.sdf_bake_done = true;
                Some(out)
            } else {
                None
            };

        // Phase 11 Stage B (B3): bake the per-mesh SDF, merge its instances into the
        // global distance field, then view a slice. Bake + merge run once
        // (`gdf_merge_done`); later frames re-view the persistent GDF. VK ≡ DX.
        let gdf_out = if self.gdf_merge
            && rt_out.is_none()
            && sdf_out.is_none()
            && vol_out.is_none()
            && bake_out.is_none()
        {
            let out = self
                .gdf
                .record_gdf_view(&mut graph, extent, cw, ch, !self.gdf_merge_done);
            self.gdf_merge_done = true;
            Some(out)
        } else {
            None
        };

        // Phase 11 Stage B (B4): SW ray trace the merged GDF. Ensures the GDF is built
        // (bake + merge, once — shared `gdf_merge_done` with the B3 view), then
        // sphere-traces it from a fixed camera over the unit-cube scene.
        // `P11_GDF_ANALYTIC` swaps in the analytic field for the reference. VK ≡ DX.
        let gdf_trace_out = if self.gdf_trace
            && rt_out.is_none()
            && sdf_out.is_none()
            && vol_out.is_none()
            && bake_out.is_none()
            && gdf_out.is_none()
        {
            let out = self.gdf.record_gdf_trace(
                &mut graph,
                extent,
                cw,
                ch,
                self.sun_dir,
                self.sun_intensity,
                self.flip_y,
                self.backend == BackendKind::Vulkan,
                self.gdf_trace_analytic,
                !self.gdf_merge_done,
            );
            self.gdf_merge_done = true;
            Some(out)
        } else {
            None
        };

        // Phase 11 Stage C1: world-space scene GDF traced from the live camera (build
        // the fused scene SDF once, then SW ray-trace it) — validates the world GDF
        // matches the rasterized scene. Only when the other replacements are off.
        let scene_gdf_out = if self.scene_gdf
            && rt_out.is_none()
            && sdf_out.is_none()
            && vol_out.is_none()
            && bake_out.is_none()
            && gdf_out.is_none()
            && gdf_trace_out.is_none()
        {
            let out = self.gdf.record_scene_gdf(
                &mut graph,
                extent,
                eye,
                inv_view_proj,
                self.sun_dir,
                self.sun_intensity,
                cw,
                ch,
                self.flip_y,
                !self.scene_gdf_baked,
            );
            self.scene_gdf_baked = true;
            Some(out)
        } else {
            None
        };

        // Phase 11 Stage C5: screen-space reflections. Runs after lighting (samples the
        // lit HDR) and replaces the tonemap source as a standalone viz of the reflection
        // buffer; C7 will instead composite it into the lighting's specular term. Only
        // when the other full-screen replacements are off.
        let ssr_out = if self.gdf_ssr
            && !self.gdf_hybrid
            && !self.swrt_reflect
            && rt_out.is_none()
            && sdf_out.is_none()
            && vol_out.is_none()
            && bake_out.is_none()
            && gdf_out.is_none()
            && gdf_trace_out.is_none()
            && scene_gdf_out.is_none()
        {
            Some(
                self.reflect
                    .record_ssr(
                        &mut graph,
                        self.deferred.globals_buffer(),
                        globals_offset,
                        hdr,
                        g_depth,
                        g_normal,
                        g_material,
                        extent,
                        view_proj.to_cols_array(),
                        inv_view_proj,
                        eye,
                        cw,
                        ch,
                        cw,
                        ch,
                        self.flip_y,
                        self.frame_no as u32,
                        self.scene_radius * 1.5,
                        self.scene_radius * 0.06,
                        false, // standalone C5 viz: sample the current lit HDR
                        false, // (viz uses the current HDR, no reprojected history to clamp)
                        false, // mirror (no stochastic jitter) for the raw-SSR viz
                    )
                    .0,
            )
        } else {
            None
        };

        // Phase 11 Stage C6: GDF reflections (off-screen fallback). A standalone viz of
        // the GDF-traced reflection buffer (sky on escape), raw radiance like the C1
        // trace; C7 will composite it under SSR. Only when the other replacements are off.
        let reflect_out = match (
            self.gdf_reflect
                && !self.gdf_hybrid
                && !self.swrt_reflect
                && rt_out.is_none()
                && sdf_out.is_none()
                && vol_out.is_none()
                && bake_out.is_none()
                && gdf_out.is_none()
                && gdf_trace_out.is_none()
                && scene_gdf_out.is_none()
                && ssr_out.is_none(),
            scene_gdf_vol,
            scene_gdf_ext,
        ) {
            (true, Some(vol), Some(ext)) => Some(self.reflect.record_gdf_reflect(
                &mut graph,
                vol,
                ext,
                scene_aabb_min,
                scene_aabb_max,
                g_depth,
                g_normal,
                g_material,
                extent,
                inv_view_proj,
                eye,
                self.sun_dir,
                self.sun_intensity,
                cw,
                ch,
                self.flip_y,
                self.frame_no as u32,
                scene_albedo,
                reflect_cache_arg,
                scene_clip,
                &scene_clip_vols,
                self.reflect_max_steps,
            )),
            _ => None,
        };

        // Phase 11 Stage C7: hybrid reflection composite. Runs both reflection sources and
        // blends them by SSR confidence into one raw-radiance image — the reflection that
        // will replace the IBL prefilter-cube specular (C7c). SSR samples the previous
        // frame's raw-radiance lit-color history (reprojected) so it can feed back into
        // lighting without a read-before-write cycle; the GDF reflect is already raw, so the
        // composite stays in raw radiance (gdf_scale = 1.0) and the tonemap applies exposure
        // (like the other SW-RT viz). A copy pass then captures this frame's lit HDR into the
        // history for the next frame. Only when the other full-screen replacements are off.
        let hybrid_out = match (
            self.gdf_hybrid
                && !self.swrt_reflect
                && self.reflect.has_lit_history()
                && rt_out.is_none()
                && sdf_out.is_none()
                && vol_out.is_none()
                && bake_out.is_none()
                && gdf_out.is_none()
                && gdf_trace_out.is_none()
                && scene_gdf_out.is_none(),
            scene_gdf_vol,
            scene_gdf_ext,
        ) {
            (true, Some(vol), Some(ext)) => {
                let ssr = self
                    .reflect
                    .record_ssr(
                        &mut graph,
                        self.deferred.globals_buffer(),
                        globals_offset,
                        hdr,
                        g_depth,
                        g_normal,
                        g_material,
                        extent,
                        view_proj.to_cols_array(),
                        inv_view_proj,
                        eye,
                        cw,
                        ch,
                        cw,
                        ch,
                        self.flip_y,
                        self.frame_no as u32,
                        self.scene_radius * 1.5,
                        self.scene_radius * 0.06,
                        true, // history mode: reprojected raw-radiance previous frame
                        self.firefly_clamp,
                        false, // mirror for the hybrid viz (no temporal resolve here)
                    )
                    .0;
                let gdf_refl = self.reflect.record_gdf_reflect(
                    &mut graph,
                    vol,
                    ext,
                    scene_aabb_min,
                    scene_aabb_max,
                    g_depth,
                    g_normal,
                    g_material,
                    extent,
                    inv_view_proj,
                    eye,
                    self.sun_dir,
                    self.sun_intensity,
                    cw,
                    ch,
                    self.flip_y,
                    self.frame_no as u32,
                    scene_albedo,
                    reflect_cache_arg,
                    scene_clip,
                    &scene_clip_vols,
                    self.reflect_max_steps,
                );
                // Standalone viz: no temporal resolve buffers here, so feed the GDF reflection
                // straight into the composite (the resolve runs only in the lighting-fed path).
                let composite = self.reflect.record_composite(
                    &mut graph,
                    ssr,
                    gdf_refl,
                    g_material,
                    extent,
                    cw,
                    ch,
                    1.0,
                    firefly_max,
                    self.reflect_max_roughness,
                );
                // Capture this frame's lit HDR (as raw radiance) for next frame's SSR history.
                self.reflect.record_lit_history(
                    &mut graph,
                    hdr,
                    cw,
                    ch,
                    1.0 / self.exposure.max(1e-4),
                    firefly_max,
                );
                Some(composite)
            }
            _ => None,
        };

        // Stage C8b1/2: surface-cache atlas viz — tiles the cards across the screen, showing
        // the lit radiance (C8b2) when lighting ran, else the captured albedo (C8b1).
        let cache_out = match (self.cache_viz, scene_cache_lit_ext, scene_cache_ext) {
            (true, Some(lit), _) => {
                let rad = self
                    .gdf
                    .surface_cache_read()
                    .map(|t| t.2)
                    .unwrap_or(u32::MAX);
                Some(
                    self.gdf
                        .record_cache_view(&mut graph, lit, rad, extent, cw, ch),
                )
            }
            (true, None, Some(cap)) => {
                let alb = self.gdf.cache_albedo_index();
                Some(
                    self.gdf
                        .record_cache_view(&mut graph, cap, alb, extent, cw, ch),
                )
            }
            _ => None,
        };

        // Tonemap samples the RT output (M4 path trace / M3 trace viz) if active, else
        // the SW-RT SDF trace, else the Stage-B volume slice, else the Stage-C1 scene
        // GDF, else the Stage-C5 SSR / C6 GDF-reflection viz, else the C8b1 cache atlas,
        // else compute-post, else HDR.
        // QHD/UHD TAAU: reconstruct the full-res (output) HDR from the jittered internal-res lit
        // image + history, before tonemap. Only the main lit path (not the debug/RT viz outputs).
        let taau_out = if taau_active {
            let main_lit = hdr_post.unwrap_or(hdr);
            // Decima FXAA→TAA: spatially anti-alias the jittered internal frame first so its edges
            // don't flicker frame to frame, then temporally upsample. Stabilizes the jitter.
            let taau_in = if self.taau.has_fxaa() && !taau_jitter_active {
                self.taau.record_fxaa(&mut graph, main_lit, extent, cw, ch)
            } else {
                main_lit
            };
            Some(self.taau.record(
                &mut graph,
                taau_in,
                g_depth,
                swap_extent,
                sw,
                sh,
                cw,
                ch,
                inv_view_proj,
                self.prev_view_proj_taau,
                self.flip_y,
                self.scene_radius * 2.0,
                taau_jitter_uv,
                false,
            ))
        } else {
            None
        };
        let tonemap_src = rt_out
            .or(sdf_out)
            .or(vol_out)
            .or(bake_out)
            .or(gdf_out)
            .or(gdf_trace_out)
            .or(scene_gdf_out)
            .or(ssr_out)
            .or(reflect_out)
            .or(hybrid_out)
            .or(cache_out)
            .or(taau_out)
            .or(hdr_post)
            .unwrap_or(hdr);
        // The rasterized HDR already bakes exposure into the lighting pass; the
        // path-traced + SW-RT outputs carry raw scene radiance, so apply the camera
        // exposure here before the filmic curve (else the bright sky + sun blow out).
        let tm_exposure = if pt_active
            || sdf_out.is_some()
            || gdf_trace_out.is_some()
            || scene_gdf_out.is_some()
            || reflect_out.is_some()
            || hybrid_out.is_some()
            || cache_out.is_some()
        {
            self.exposure
        } else {
            1.0
        };
        // QHD/UHD: sharpen only when the TAAU upscale produced this frame (recover crispness lost
        // in temporal upsampling); native/debug paths get 0 = byte-identical.
        let (sharpen, inv_w, inv_h) = if taau_active && taau_out.is_some() {
            (0.25, 1.0 / sw as f32, 1.0 / sh as f32)
        } else {
            (0.0, 0.0, 0.0)
        };
        self.deferred.record_tonemap(
            &mut graph,
            backbuffer,
            tonemap_src,
            self.post_mode as u32,
            self.flip_y,
            tm_exposure,
            sharpen,
            inv_w,
            inv_h,
        );

        // Phase 7 GPU-culling draw: indirect, instanced render of the visible cube
        // grid over the tonemapped image, with its own depth buffer.
        if let Some((args_ext, visible_ext)) = cull_res {
            self.cull.record_draw(
                &mut graph,
                backbuffer,
                swap_extent,
                args_ext,
                visible_ext,
                view_proj.to_cols_array(),
                self.sun_dir,
                &grid,
            );
        }

        // Phase 7 particle draw: instanced billboards composited over the tonemapped
        // image (alpha blend), reading the compute-updated buffer in the vertex stage.
        // Declared after tonemap so the WAW on the backbuffer orders it last.
        if let Some(particles_ext) = particles_ext {
            self.particles.record_draw(
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
                |ctx| self.gui.render(&self.device, ctx.cmd(), fif),
            );
        }
        let mut profiler = self
            .profiler_on
            .then(|| GraphProfiler::new(&self.query_heaps[fif]));
        graph.execute(
            &self.device,
            &mut self.pools[fif],
            cmd,
            &self.swapchain,
            image_index,
            self.aliasing,
            profiler.as_mut(),
        )?;
        // Remember this slot's scheduled pass names so the next readback (after this
        // frame's fence) can pair them with the timestamp boundaries.
        self.slot_pass_names[fif] = match &profiler {
            Some(p) => p.names.clone(),
            None => Vec::new(),
        };

        // For a screenshot, copy the just-rendered backbuffer into a readback buffer
        // in the same command buffer (before it ends).
        let readback = if capture_this_frame.is_some() {
            let layout = self.device.swapchain_readback_layout(&self.swapchain);
            let buf = self.device.create_buffer(&BufferDesc {
                size: layout.size,
                usage: BufferUsage::Readback,
            })?;
            cmd.copy_swapchain_to_buffer(&self.swapchain, image_index, &buf);
            Some((buf, layout))
        } else {
            None
        };

        cmd.end()?;

        let signal = &self.render_finished[image_index as usize];
        if async_sim {
            // Record the particle sim into this frame's compute command buffer and run
            // it on the compute queue (overlapping graphics), signaling `compute_done`;
            // the graphics submit GPU-waits on it so the particle draw's vertex-stage
            // read sees the freshly written buffer.
            let ccmd = &self.compute_command_buffers[fif];
            ccmd.begin()?;
            ccmd.bind_compute_pipeline(self.particles.sim_pipeline());
            ccmd.push_constants_compute(&particle_sim_push(
                self.particles.buffer_storage_index(particle_read),
                self.particles.buffer_storage_index(particle_write),
                PARTICLE_COUNT as u32,
                sim_dt,
                self.elapsed,
                0,
            ));
            ccmd.dispatch((PARTICLE_COUNT as u32).div_ceil(64), 1, 1);
            ccmd.end()?;
            self.compute_queue.submit(ccmd, &self.compute_done[fif])?;
            self.queue.submit_async(
                cmd,
                &self.image_available[fif],
                &self.compute_done[fif],
                signal,
                &self.in_flight[fif],
            )?;
        } else if self.async_cache_on && scene_cache_lit_ext.is_some() {
            // Async surface-cache relight: record visibility + relight onto the compute command
            // buffer and run it on the compute queue, overlapping this graphics frame. The graphics
            // queue GPU-waits on the PREVIOUS frame's relight (1-frame latency) so its consumer
            // reads of last frame's radiance are ordered; the 3-slot ring guarantees the slot this
            // frame writes is not one an in-flight graphics frame still reads (no WAR). Submit
            // graphics BEFORE the compute signal so D3D12's fence wait targets the previous value.
            // The compute command buffer for this fif may still be pending from 2 frames ago (the
            // graphics fence doesn't cover it on the cross-frame path) — wait its own fence first.
            self.cache_compute_fence[fif].wait()?;
            self.cache_compute_fence[fif].reset()?;
            let ccmd = &self.compute_command_buffers[fif];
            ccmd.begin()?;
            self.gdf.record_cache_async(
                ccmd,
                frustum_planes(cull_view_proj),
                self.sun_dir,
                self.sun_intensity,
                self.cache_relight_spp,
                self.frame_no as u32,
                cache_reset_this_frame,
                self.cache_relight_period,
                relight_alpha,
                self.cache_feedback,
            );
            ccmd.end()?;
            let cur = (self.frame_no % 2) as usize;
            if self.frame_no == 0 {
                // No previous relight to wait on; graphics submits normally, the relight still
                // signals so frame 1's wait pairs up.
                self.queue.submit(
                    cmd,
                    &self.image_available[fif],
                    signal,
                    &self.in_flight[fif],
                )?;
            } else {
                let prev = ((self.frame_no + 1) % 2) as usize; // (N-1) mod 2
                self.queue.submit_async(
                    cmd,
                    &self.image_available[fif],
                    &self.cache_done[prev],
                    signal,
                    &self.in_flight[fif],
                )?;
            }
            self.compute_queue.submit_fenced(
                ccmd,
                &self.cache_done[cur],
                &self.cache_compute_fence[fif],
            )?;
            self.scene_cache_reset = false;
        } else {
            self.queue.submit(
                cmd,
                &self.image_available[fif],
                signal,
                &self.in_flight[fif],
            )?;
        }
        // Swap the particle ping-pong parity for the next simulated frame (deferred to
        // here so the graph's `&self` borrows have ended — see `particle.rs`).
        if self.particles_on {
            self.particles.advance();
        }
        // Bump the path tracer's progressive-accumulation counter (deferred here for
        // the same reason: `record_path` borrowed `&rt` for the graph's lifetime).
        if pt_active {
            self.rt.advance_accum();
        }
        // C4: advance the GI denoiser accumulation (ping-pong swap) and stash this
        // frame's view-projection for the next frame's temporal reprojection.
        if gi_denoise_active {
            self.gi.advance_denoise();
        }
        // C7b: advance the lit-color history ping-pong so next frame reads this frame's write.
        if self.gdf_hybrid || self.swrt_reflect {
            self.reflect.advance_history();
        }
        // Advance the stochastic-SSR temporal accumulation ping-pong (stochastic mode only).
        if self.swrt_reflect && self.ssr_stochastic && self.reflect.has_ssr_resolve() {
            self.reflect.advance_ssr_accum();
        }
        // C8j: advance the stochastic GDF-reflection temporal accumulation ping-pong.
        if self.swrt_reflect && self.reflect.has_reflect_temporal() {
            self.reflect.advance_reflect_accum();
        }
        // C8b2: advance the surface-cache radiance ping-pong (next frame reads this frame's).
        if scene_cache_lit_ext.is_some() {
            self.gdf.advance_cache();
        }
        // QHD/UHD TAAU: advance the history ping-pong (next frame reprojects this frame's).
        if taau_active {
            self.taau.advance();
        }
        self.prev_view_proj = view_proj.to_cols_array();
        self.prev_view_proj_taau = view_proj_stable.to_cols_array();

        // Wait for the GPU (copy included), read the buffer back, and save a PNG.
        if let (Some(cap), Some((buf, layout))) = (capture_this_frame.as_ref(), readback.as_ref()) {
            self.in_flight[fif].wait()?;
            let mut bytes = vec![0u8; layout.size as usize];
            buf.read_into(&mut bytes)?;
            save_screenshot(&cap.path, &bytes, layout)?;
            info!(
                "saved screenshot {} ({}x{}, ui={})",
                cap.path, layout.width, layout.height, cap.include_ui
            );
        }

        if self.queue.present(&self.swapchain, image_index, signal)? {
            self.needs_recreate = true;
        }
        self.fif = (self.fif + 1) % FRAMES_IN_FLIGHT;
        self.frame_no += 1;

        // In screenshot mode, stop once every requested capture is saved (CAPTURE_SEQ
        // dumps N frames, else one per requested path).
        let total_captures = self
            .capture_seq
            .map(|n| n as u64)
            .unwrap_or(self.captures.len() as u64);
        if self.screenshot_mode && self.frame_no >= warmup + total_captures {
            return Ok(false);
        }
        Ok(true)
    }
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
/// Parse a `"x,y,z"` environment variable into a `Vec3` (for the diagnostic camera).
fn parse_vec3_env(name: &str) -> Option<Vec3> {
    let v = std::env::var(name).ok()?;
    let n: Vec<f32> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    (n.len() == 3).then(|| Vec3::new(n[0], n[1], n[2]))
}

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
