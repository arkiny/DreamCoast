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
    BackendKind, Buffer, BufferDesc, BufferUsage, CommandBuffer, CommandList, ComputeQueue, Device,
    Extent2D, Fence, Format, Instance, InstanceDesc, PresentMode, QueryHeap, Queue, ReadbackLayout,
    Recorder, Semaphore, Swapchain, SwapchainDesc, Texture,
};
use tracing::info;

mod app;
mod atmosphere;
mod camera;
mod clipmap;
mod cluster;
mod compose;
mod csm;
mod cull;
mod deferred;
mod fuse;
mod gdf;
mod gi;
mod gtao;
mod hzb;
mod ibl;
mod level;
mod mesh;
mod mesh_sdf;
mod morph;
mod particle;
mod push;
mod quality;
mod reflect;
mod registry;
mod rhi_thread;
mod rt;
mod skin;
mod smoketest;
mod taau;
mod translucent;
mod velocity;
mod world;
use app::*;
use cluster::{ClusterLight, ClusterSystem};
use cull::*;
use deferred::*;
use dreamcoast_scene::{LocalTransform, MeshInstance, World};
use gdf::*;
use gi::*;
use hzb::*;
use ibl::*;
use mesh::*;
use particle::*;
use push::*;
use reflect::*;
use registry::{GpuMesh, MaterialDesc, MaterialRegistry, MeshRegistry, build_scene};
use rt::*;
use smoketest::*;

const FRAMES_IN_FLIGHT: usize = 2;
// CPU/GPU-bound diagnosis (`PROFILE_CPU`): microseconds spent in the per-frame fence wait (blocking
// on the fif's previous GPU submission) and the whole-frame CPU wall time. wait≈0 ⇒ CPU-bound (the
// GPU already finished, the CPU is the long pole); wait large ⇒ GPU-bound. Logged in screenshot mode.
static LAST_WAIT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAST_FRAME_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
// CPU record time: frame entry → just before `present` (excludes the display-paced present block),
// so it isolates the real CPU work (graph build + command recording + submit) from vsync pacing.
static LAST_CPU_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
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
/// dynamic UBO offset). Holds the lighting globals, the light view-projection matrix,
/// the (PR-6) clustered-lighting view-Z row + froxel params, and the (PR-7) CSM cascade
/// block (up to 4 cascade view-projections + splits/tiles). 1280 = next 256-aligned size
/// that fits both blocks; the OFF paths leave their fields zeroed so the shader takes the
/// legacy branches and the gallery anchor stays byte-identical.
const GLOBALS_SLICE: u64 = 1280;

/// Clustered light culling froxel grid (PR-6). Single source of truth mirrored by
/// `CLUSTER_X/Y/Z` in `light_cluster_common.slang`. 16×9 tiles matches 16:9; 24
/// exponential Z slices is the DOOM-2016-era default. Bump for a higher RenderQuality
/// tier (also grows the grid/index storage buffers via `CLUSTER_COUNT`).
const CLUSTER_X: u32 = 16;
const CLUSTER_Y: u32 = 9;
const CLUSTER_Z: u32 = 24;
const CLUSTER_COUNT: u32 = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;
/// Max lights binned per cluster (mirrors `MAX_LIGHTS_PER_CLUSTER` in the shader).
const MAX_LIGHTS_PER_CLUSTER: u32 = 128;
/// Camera near/far — single source of truth for the scene projection AND the clustered
/// froxel Z slicing (they must agree so cluster slices span the actual view frustum).
const CLUSTER_Z_NEAR: f32 = 0.05;
const CLUSTER_Z_FAR: f32 = 100.0;
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

const DEBUG_VIEWS: [&str; 12] = [
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
    "Velocity", // 11 — motion vectors (needs P_VELOCITY=1)
];
/// DEBUG_VIEW=11 velocity viz amplification: a small NDC motion (a few 1e-3) is scaled into a
/// visible colour range. Tuned so a spinning object's edge motion is clearly readable.
const VELOCITY_VIZ_SCALE: f32 = 40.0;
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
    /// Alpha-test cutoff for `alphaMode: MASK` (0.0 = opaque, no test). Drives the G-buffer
    /// cutout discard and the masked-shadow discard from one value.
    pub(crate) alpha_cutoff: f32,
    /// Renderer routing tag from material classification (deferred-decal pass split). `Opaque`
    /// for procedural/level drawables; glTF imports may be `Decal`/`Transparent`.
    pub(crate) kind: dreamcoast_asset::MaterialKind,
    pub(crate) casts_shadow: bool,
    /// GPU-skinning storage-buffer indices `[joints, weights, palette, joint_count]`
    /// when this drawable is skinned (animation Stage B.2); `None` = static. Set by the
    /// per-frame skinning patch; the G-buffer pass draws these with the skinned pipeline
    /// + the bind-pose vertex buffer (the vertex shader does the deform).
    pub(crate) skin: Option<[u32; 4]>,
    /// GPU-morph indices `[deltas, weights, target_count, vertex_count]` when this drawable
    /// is GPU-morphed (animation Stage C optimization); `None` = not morphed (or CPU-morphed,
    /// which instead swaps the vertex buffer). Set by the per-frame morph patch; the
    /// G-buffer/shadow passes draw these with the morph pipeline + the bind-pose buffer.
    pub(crate) morph: Option<[u32; 4]>,
}

/// A level's lighting (sun + point lights), applied in the `Globals` assembly in
/// place of the gallery's hardcoded code-default lights. `None` keeps the gallery
/// defaults (byte-identical baseline).
struct LevelLighting {
    sun_dir: [f32; 3],
    sun_intensity: f32,
    /// The directional (sun) light's RGB color — drives the analytic sun tint so a level can author
    /// a warm sun (e.g. `[1.0, 0.96, 0.9]`). White `[1,1,1]` if the level has no directional light.
    sun_color: [f32; 3],
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
    let mut sun_color = [1.0f32, 1.0, 1.0];
    let mut point_pos = [[0.0f32; 4]; 4];
    let mut point_color = [[0.0f32; 4]; 4];
    let mut count = 0usize;
    for l in &level.lights {
        match l.kind {
            LightKind::Directional => {
                sun_dir = toward_sun(l.vec);
                sun_intensity = l.intensity;
                sun_color = l.color;
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
        sun_color,
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
    // Clustered light culling (PR-6). `cluster_view_z_row` is row 2 of the world->view matrix
    // (positive linear view depth = -dot(row, [world,1])); `cluster_params` = (z_near, z_far, 0, 0)
    // for the froxel Z slicing. Unused on the brute-force path (CLUSTERED_LIGHTS off).
    cluster_view_z_row: [f32; 4],
    cluster_params: [f32; 4],
    // --- PR-7 shadow atlas / CSM. All zero on the legacy single-map path (csm_params.x == 0),
    // which makes the shader take the `light_view_proj` branch → byte-identical anchor.
    csm_params: [i32; 4], // x cascade count (0 = off), y debug-cascade viz, z tile texels, w atlas texels
    csm_split: [f32; 4],  // per-cascade view-space far distance (up to MAX_CASCADES)
    csm_opts: [f32; 4],   // x cross-cascade blend fraction (of the cascade depth range)
    csm_view_proj: [[f32; 16]; 4], // per-cascade world -> atlas-tile light clip
    csm_atlas_uv: [[f32; 4]; 4], // per-cascade atlas UV sub-rect: xy offset, zw scale
}

fn swapchain_desc(extent: Extent2D) -> SwapchainDesc {
    SwapchainDesc {
        extent,
        format: COLOR_FORMAT,
        // `NO_VSYNC=1` runs the presentation uncapped (no display-refresh pacing) so the true
        // CPU+GPU frame time can be benchmarked past the 60Hz vsync ceiling. Default = Fifo (vsync).
        present_mode: if std::env::var_os("NO_VSYNC").is_some() {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo
        },
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
    // Size the job-system worker pool from the machine's core count before any
    // parallel work (propagate / parallel record / morph) touches the global pool.
    // Default = `available_parallelism() - 1` worker threads (the main thread is the
    // other participant); `JOBS_THREADS=<n>` overrides for benchmarking.
    let job_threads = std::env::var("JOBS_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    dreamcoast_jobs::init_global(job_threads);
    info!(
        "job system: {} worker threads (+ main)",
        dreamcoast_jobs::global().num_workers()
    );
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
    // `queue`/`swapchain` are `None` while the RHI thread owns them (P15_RHI_THREAD,
    // M4 B3): they're moved into the worker at spawn and reclaimed at join. In the
    // default inline path they're always `Some` — use [`App::queue`]/[`App::swapchain`].
    queue: Option<Queue>,
    compute_queue: ComputeQueue,
    swapchain: Option<Swapchain>,
    backend: BackendKind,
    gui: Gui,

    // Feature bundles (see the per-module docs).
    deferred: DeferredRenderer,
    gdf: GdfSystem,
    gi: GiSystem,
    gtao: gtao::GtaoSystem,
    /// PR-4 (render-pipeline re-baseline track): the sky/atmosphere composite slot
    /// (today: opt-in analytic height fog, `P_HEIGHT_FOG`). See `atmosphere.rs`.
    atmosphere: atmosphere::AtmosphereSystem,
    /// PR-3 (render-pipeline re-baseline track): the forward translucency slot (sorted
    /// alpha-blend after opaque lighting + fog, before post). See `translucent.rs`.
    translucency: translucent::TranslucencySystem,
    /// Translucent drawables sorted + drawn by the translucency pass. Empty by default
    /// (the slot then adds no pass → byte-identical). `P_TRANSLUCENT_TEST=1` seeds demo
    /// glass panes; glTF `Transparent` (BLEND) materials also route here.
    translucents: Vec<translucent::TranslucentObject>,
    reflect: ReflectSystem,
    particles: ParticleSystem,
    cull: CullSystem,
    /// Clustered light culling (PR-6). `None` where compute is unavailable; the feature
    /// stays off. Gated on `clustered_lights` (`CLUSTERED_LIGHTS=1`).
    cluster: Option<ClusterSystem>,
    /// HZB occlusion culling (PR-8), `None` when compute is unavailable. Layered on
    /// top of `cull`'s frustum test behind `HZB_CULL=1`.
    hzb: Option<HzbSystem>,
    rt: RtSystem,
    ibl: IblSystem,

    // Scene. The ECS `world` (+ registries) is the single source of truth; the flat
    // `SceneObject` draw list is materialized from it each frame via `build_scene`.
    // `_textures` keeps the model's bindless textures alive.
    _textures: Vec<Texture>,
    world: World,
    mesh_registry: MeshRegistry,
    material_registry: MaterialRegistry,
    /// CPU-skinned primitives (animation Stage B). Empty unless an imported glTF scene
    /// has skins; each is re-skinned + uploaded per frame (inline path only).
    skinned: Vec<skin::SkinnedMesh>,
    /// Morph-target primitives (animation Stage C). Empty unless an imported glTF scene
    /// has morph targets; GPU primitives write a per-frame weights buffer (the VS blends),
    /// CPU ones re-blend + upload a vertex ring each frame (inline path only).
    morphed: morph::MorphSet,
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
    /// Phase 15 M4 B3: the RHI (submit) thread. `Some` when `P15_RHI_THREAD` is set —
    /// it solely owns `queue`/`swapchain`/`command_buffers`/`image_available`/
    /// `in_flight`/`render_finished` (moved at spawn). The record thread builds a
    /// `Send` `CommandList` per frame and ships it; the worker acquires + translates +
    /// submits + presents, overlapping the record thread's next frame (record N+1 ∥
    /// submit N). `None` = the default single-thread inline path (byte-identical).
    rhi_thread: Option<rhi_thread::RhiThread>,
    /// M4 B4: record the render graph's passes in parallel on the job system (each
    /// pass builds its own IR bucket, concatenated in schedule order). Opt-in
    /// (`P15_PARALLEL_RECORD`) and only on the threaded path (profiler-free); default
    /// off = sequential recording (byte-identical).
    parallel_record: bool,
    /// Cached swapchain extent/format/readback-layout, kept in sync at construction
    /// and on resize. The record thread reads these (instead of the swapchain) while
    /// the RHI thread owns the swapchain.
    swap_extent_cached: Extent2D,
    swap_format_cached: Format,
    readback_layout_cached: ReadbackLayout,

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
    /// Analytic sun RGB tint (the level's directional-light color, or white). Multiplies the direct
    /// sun in the lighting pass so a warm authored sun reads warm. `SUN_COLOR` env / level override.
    sun_color: [f32; 3],
    ambient: f32,
    exposure: f32,
    /// Atmosphere inscatter→radiance gain — the sun:sky illuminance ratio knob fed to the env
    /// capture (`sky.slang`). Legacy 6.0 (gallery anchor); content lowers it so the direct sun
    /// dominates the sky physically, interiors filled by multibounce GI. `SKY_GAIN` overrides.
    sky_gain: f32,
    /// Sky white balance — per-channel gain on the procedural sky radiance fed to the env capture
    /// (`sky.slang`). Warms/neutralises the IBL + SW-RT GI ambient (the sky-sourced fill) without
    /// tinting the direct sun. `[1,1,1]` = neutral (no-op). `SKY_WB`/level `sky_white_balance` set it.
    sky_wb: [f32; 3],
    /// Physical-camera auto-exposure: meter the lit HDR each frame and adapt the exposure (vs the
    /// fixed EV100). Opt-in (`AUTO_EXPOSURE`); off → the lighting uses the static `exposure` and is
    /// byte-identical (the gallery anchor never auto-exposes).
    auto_exposure: bool,
    /// 레퍼런스식 multi-bounce energy compensation strength for the GDF GI (0 = off = gallery anchor;
    /// ~0.6 content). Written to `globals.probe_box_max.w`; see pbr.slang. `P_GI_MULTIBOUNCE`.
    gi_multibounce: f32,
    point_lights_on: bool,
    /// Clustered light culling on (PR-6, `CLUSTERED_LIGHTS=1`). Off = brute-force loop.
    clustered_lights: bool,
    /// A/B baseline (`CLUSTERED_BRUTE=1`): upload the light buffer but loop all lights (no
    /// froxel list) — for profiling clustered vs brute-force on the same light set.
    clustered_brute: bool,
    /// Deterministic test-light count (`TEST_LIGHTS=N`, PR-6 scale proof). 0 = off.
    test_lights: u32,
    shadows_on: bool,
    shadow_bias: f32,
    // PCSS-lite penumbra scale (max soft-shadow radius in shadow-map UV); 0 = hard 3x3 PCF.
    shadow_softness: f32,
    /// Soft-shadow blocker/PCF tap count, written to `globals.shadow.w` (RenderQuality knob).
    /// Only the soft path (softness > 0) reads it; the shader clamps to [1, 16].
    shadow_taps: u32,
    /// PR-7 shadow atlas / CSM config (opt-in `CSM=1` / `CSM=<N>`). `enabled == false` is the
    /// legacy single directional map — byte-identical anchor.
    csm: csm::CsmConfig,
    /// Cascade-index debug overlay: 1 tints each cascade's coverage by index so the split
    /// boundaries are visible. Written to `globals.csm_params.y`. Off = 0.
    csm_debug: bool,
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
    /// Depth pre-pass active (pipeline rebaseline PR-1, opt-in `DEPTH_PREPASS=1`): render an
    /// opaque depth-only pass before the G-buffer so the base pass runs EQUAL-test + write-off
    /// (Early-Z overdraw elimination) and the screen-space passes sample a completed depth.
    /// Off by default = the pre-pass-less path (byte-identical golden anchor).
    depth_prepass: bool,
    particles_on: bool,
    async_compute_on: bool,
    gpu_cull: bool,
    /// HZB occlusion culling on top of GPU frustum culling (PR-8, `HZB_CULL=1`; needs
    /// `P7_CULL=1`). Runtime stats of the last cull are read back into `hzb_stats`.
    hzb_cull: bool,
    hzb_stats: std::cell::Cell<(u32, u32)>, // (survived, culled_by_occlusion) last frame
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
    /// Screen-space near-field AO (HBAO-lite), composed with the GDF AO. `SSAO` env / tier.
    ssao: bool,
    /// SSAO tuning: world radius (m), intensity, contact bias (m), contrast power.
    ssao_params: [f32; 4],
    /// C3: GDF 1-bounce diffuse GI added to the deferred ambient term.
    gdf_gi: bool,
    /// F3 (HW-RT high-fidelity path, first increment): route the GI gather rays through the scene
    /// TLAS via an inline RayQuery (hardware ray tracing) instead of the SW sphere-march. High-tier
    /// opt-in (`P_HWRT_GI=1`); default off keeps the scalable SW path (gallery byte-identical). This
    /// first increment returns a hardware-traced VISIBILITY term only (hit lighting is a later
    /// increment) and requires the BLAS/TLAS built by `rt.rs` (currently the gallery scene only).
    hwrt_gi: bool,
    /// 레퍼런스 엔진 GI-fidelity: world irradiance volume (DDGI-lite radiance cache). When on, the GI pass
    /// samples a multibounce-propagating world volume instead of a single-bounce ray march — the
    /// real fix for deep-interior darkness. Content-only (`P_GI_VOLUME`); gallery forced off.
    gi_volume: bool,
    /// 레퍼런스식 indoor skylight occlusion (occludes the IBL diffuse skylight by the GI volume's
    /// directional sky-visibility): neutral leak fraction (`P_SKYVIS_TINT`) + min-occlusion floor
    /// (`P_SKYVIS_MIN_OCC`). Only active when `gi_volume` is on (content); gallery passes the
    /// no-op sentinel image so it stays byte-identical.
    skyvis_tint: f32,
    skyvis_min_occ: f32,
    /// PR-4 (render-pipeline re-baseline track): opt-in analytic exponential height fog
    /// (`P_HEIGHT_FOG=1`), composited in the atmosphere slot after opaque lighting +
    /// reflections, before TAAU/tonemap. Off by default (byte-identical gallery anchor —
    /// the slot itself is always present in the graph wiring, but unscheduled when off).
    height_fog: bool,
    /// Height-fog density at world height 0 (`a` in `d(y) = a·exp(-b·y)`, 1/world-unit).
    /// `P_FOG_DENSITY` overrides; scaled by `scene_radius` at the call site so the default
    /// reads sensibly across scene scales (unit-cube gallery vs. the larger Sponza level).
    fog_density: f32,
    /// Height-fog falloff rate `b` (1/world-unit; larger = fog thins out faster with height).
    /// `P_FOG_HEIGHT_FALLOFF` overrides; also scaled by `scene_radius`.
    fog_height_falloff: f32,
    /// Height-fog inscatter→radiance gain fed to the shared `procedural_sky` (mirrors
    /// `sky_gain`'s role for the env-cube capture — same single-source atmosphere model,
    /// not a duplicated constant). `P_FOG_INSCATTER_GAIN` overrides; defaults to `sky_gain`.
    fog_inscatter_gain: f32,
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
    /// P3 (SW-RT GI 레퍼런스급): cone-trace LOD march slope for the SW-RT march loops (gallery forced 0 =
    /// legacy linear march = byte-identical). Content takes the tier value.
    gdf_cone_k: f32,
    /// Stage D3: trace the GGX reflection at half resolution + bilateral upsample (gallery off).
    reflect_half_res: bool,
    /// macOS/M3 perf (M3-C): reflection trace divisor when `reflect_half_res` is on (2 = legacy half,
    /// 4 = quarter). The one lever that cuts `gdf_reflect` (measured resolution-only). `P_REFLECT_RES_DIV`.
    reflect_res_div: u32,
    /// macOS/M3 perf: GDF AO trace divisor (1 = full-res, 2 = half). Traced at 1/div + bilateral
    /// upsample; the Apple tier uses 2 (gdf_ao is the top pass after quarter-res reflection). `P_AO_RES_DIV`.
    ao_res_div: u32,
    /// macOS/M3 perf: à-trous GI-denoise iteration count (2 = legacy, Apple = 1). `P_GI_ATROUS_STEPS`.
    gi_atrous_steps: u32,
    /// Stage D1: trace the C3 GI at half resolution + joint-bilateral upsample (1/4 the rays).
    /// Forced off for the gallery anchor (full-res = byte-identical). Content scenes opt in by tier.
    gi_half_res: bool,
    /// P1 (SW-RT GI 레퍼런스급): GI trace divisor when `gi_half_res` is on (2 = half, 4 = quarter probes).
    gi_res_div: u32,
    /// Screen-space radiance probe GI: replace the world-volume / ray-march GI consumption with
    /// per-tile screen probes traced into an octahedral atlas + a per-pixel gather. Content-only
    /// (`SCREEN_PROBE`); the gallery keeps its current GI path (byte-identical anchor).
    screen_probe: bool,
    /// P4 world radiance cache fallback for escaped screen-probe rays (`P_WRC`, on with probes).
    wrc: bool,
    /// GI-on-distance-field visualization: march the camera into the GDF and paint hits. Replaces
    /// the tonemap source. `wrc_viz` = the view pass is active; `sc_viz` = shade from the high-res
    /// surface cache (`P_SC_VIZ`) instead of the coarse world radiance cache (`P_WRC_VIZ`).
    wrc_viz: bool,
    sc_viz: bool,
    /// Reflection temporal history clamp (0 off / 1 hard / 2 variance; gallery forced 0) + variance γ.
    reflect_history_clamp: u32,
    reflect_clamp_gamma: f32,
    /// GI temporal denoiser history-clamp (gdf_temporal params.w): 0 off (content; fixes shimmer),
    /// 1 hard (gallery legacy byte-identical), >1.5 variance γ.
    gi_temporal_clamp: f32,
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
    /// Per-frame dynamic sun (time-of-day): when on, the sun arcs across the sky from
    /// `elapsed`, so the physical atmosphere + IBL recapture each frame (already per-frame
    /// via `realtime_env`) visibly drive a moving sky. Off by default (static, deterministic
    /// screenshots). `TIME_OF_DAY` env / UI toggle. Infrastructure for future time-of-day.
    time_of_day: bool,
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
    /// `P_TAAU_FORCE=1`: run TAAU even at native resolution (internal == output) — i.e. temporal
    /// anti-aliasing (jitter + accumulation, no upscale). Opt-in so the default native path stays
    /// byte-identical (TAAU off when render==output).
    taau_force: bool,
    /// QHD/UHD Stage 8: TAA-aware texture LOD bias added when jitter is active (the primary
    /// distant-texture sharpness lever). Resolved once from `quality::TAA_MIP_BIAS` with a
    /// `TAA_MIP_BIAS` env override for sweeps. See `quality.rs` for the rationale.
    taa_mip_bias: f32,
    /// Velocity (motion-vector) G-buffer channel (pipeline re-baseline PR-2). Owns the opaque
    /// velocity pass + the DEBUG_VIEW=11 viz. `velocity_on` gates the whole feature (`P_VELOCITY=1`);
    /// default off = no velocity target, camera-only TAAU reprojection, byte-identical.
    velocity: velocity::VelocitySystem,
    velocity_on: bool,
    /// Single prev-pose source (PR-2): each drawable's PREVIOUS-frame unjittered world transform,
    /// keyed by the frame's stable draw-list index (deterministic insertion order). The velocity
    /// pass reads it to compute per-object screen motion for static / Spin / node-animated draws
    /// (skinning / morph add their prev palette / weights on top). Rebuilt after each frame from the
    /// current transforms.
    prev_transforms: Vec<Mat4>,
    // Profiler UI state.
    profiler_on: bool,
    slot_pass_names: Vec<Vec<&'static str>>,
    gpu_timings: Vec<(&'static str, f32)>,

    // Loop bookkeeping.
    fif: usize,
    frame_no: u64,
    f2_prev: bool,
    needs_recreate: bool,
    last: Instant,
    elapsed: f32,
    angle: f32,
    /// Previous frame's orbit `angle`, kept so the rendered camera can interpolate
    /// between the last and current fixed-timestep sim states (M2 fixed timestep).
    prev_angle: f32,
    /// Leftover real time not yet consumed by a whole fixed sim step. Carries the
    /// fractional remainder frame-to-frame; `remainder / FIXED_DT` is the render
    /// interpolation alpha. Interactive only — headless capture bypasses it.
    sim_accumulator: f32,
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
        // Screen-space near-field AO (composed with the GDF AO). See `gtao.rs`.
        let gtao = gtao::GtaoSystem::new(&device, backend, compute_supported)?;
        // PR-4 (render-pipeline re-baseline track): the sky/atmosphere composite slot
        // (opt-in height fog today). See `atmosphere.rs`.
        let atmosphere = atmosphere::AtmosphereSystem::new(&device, backend, HDR_FORMAT)?;
        // PR-3 (render-pipeline re-baseline track): the forward translucency slot. Built
        // unconditionally (the shader is exercised by every backend's build); the pass is
        // only recorded when there are translucent objects. See `translucent.rs`.
        let translucency =
            translucent::TranslucencySystem::new(&device, backend, HDR_FORMAT, DEPTH_FORMAT)?;
        // Stage C reflection track (C5 SSR; C6/C7 later). See `reflect.rs`.
        let reflect = ReflectSystem::new(&device, backend, compute_supported)?;
        // QHD/UHD track: temporal upsampler. See `taau.rs`.
        let taau = taau::TaauSystem::new(&device, backend, compute_supported)?;
        // Velocity (motion-vector) channel (pipeline re-baseline PR-2). See `velocity.rs`.
        let velocity = velocity::VelocitySystem::new(&device, backend)?;

        // GPU particle system (Phase 7): a persistent ping-pong buffer pair advanced
        // by a compute pass and drawn as instanced billboards (see `particle.rs`).
        // PR-3 side-effect: the draw now composites in the HDR translucency slot (before
        // tonemap), not over the tonemapped LDR — so it's built with HDR_FORMAT. Default-off
        // (`P7_PARTICLES`), so the default gallery/anchor output is unchanged.
        let particles = ParticleSystem::new(&device, backend, compute_supported, HDR_FORMAT)?;

        // GPU frustum culling (Phase 7): a compute pass tests a cube instance grid
        // against the frustum and writes an indirect draw; the draw renders only the
        // visible instances (see `cull.rs`). PR-3 side-effect: draws into the HDR slot
        // (before tonemap), so built with HDR_FORMAT. Default-off (`P7_CULL`).
        let cull = CullSystem::new(&device, backend, compute_supported, HDR_FORMAT)?;

        // Clustered light culling (PR-6): a compute pass bins the scene's point lights into a
        // view-frustum froxel grid so the lighting pass loops only its cluster's lights. `None`
        // where compute is unavailable; opt-in via `CLUSTERED_LIGHTS=1` (see `cluster.rs`).
        let cluster = ClusterSystem::new(&device, backend, compute_supported)?;

        // HZB occlusion culling (PR-8): a max-reduced Hi-Z pyramid built from the scene
        // depth feeds an occlusion-aware variant of the cull compute. Compute-only; the
        // pyramid is sized to the render extent and resized in-frame if it changes.
        let hzb = if compute_supported {
            Some(HzbSystem::new(&device, backend, swapchain.extent_2d())?)
        } else {
            None
        };

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
        // Content scenes (levels / glTF imports / worlds) carry large multi-material assets —
        // e.g. Sponza + foliage needs ~7.8 GB of uncompressed textures, right at an 8 GB card's
        // edge (Vulkan OOMs where D3D12's VRAM oversubscription barely fits). Block-compress
        // their textures by DEFAULT (BC1 colour / BC5 normals, ~4× smaller, GPU-native = no
        // decompress cost). The gallery anchor stays `Off` (byte-identical regression scene).
        let content_compress = level::content_tex_compress();

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
        let mut gltf_skinned: Vec<skin::SkinnedMesh> = Vec::new();
        let mut gltf_morphed = morph::MorphSet::default();
        // A level's authored camera (applied as the initial view if non-default).
        let mut level_view: Option<(Vec3, Vec3)> = None;
        // A level's lighting, replacing the gallery's code-default sun + point lights.
        let mut level_lighting_override: Option<LevelLighting> = None;
        // A level's sky white balance (per-channel sky-radiance gain), captured for the IBL.
        let mut level_sky_wb: Option<[f32; 3]> = None;

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
            level_sky_wb = Some(level.environment.sky_white_balance);
            scene_bounds = level::build_level(
                &device,
                &level,
                &mut world,
                &mut mesh_registry,
                &mut material_registry,
                &mut textures,
                Vec3::ZERO,
                content_compress,
            )?;
        } else if let Some(path) = &scene_gltf_path {
            // Stage B: import the whole node hierarchy + every primitive/material/image,
            // through the cooked, block-compressed `.dcasset` (a hit skips glTF parse +
            // decode + BCn encode). `content_compress` sets the tier.
            let (gscene, outcome) = dreamcoast_asset::cook::load_or_cook_gltf_scene(
                std::path::Path::new(path),
                path,
                &app::cooked_cache_dir(),
                content_compress,
            )?;
            info!(
                "glTF scene '{path}' ({outcome:?}): {} nodes, {} primitives, {} materials, {} images",
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
            let (imported, node_map) =
                dreamcoast_scene::instantiate_gltf_mapped(&mut world, &gscene, &prim_handles);
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
            // Animation Stage A: play one of the glTF's clips. `GLTF_ANIM[=<index>]`
            // (default clip 0) attaches an `AnimationPlayer` whose channels target the
            // imported node entities; the frame loop's `advance_animation` drives it.
            // No-op without `GLTF_ANIM` or without node-TRS clips (byte-identical).
            if let Ok(sel) = std::env::var("GLTF_ANIM") {
                let idx = sel.trim().parse::<usize>().unwrap_or(0);
                match gscene.animations.get(idx) {
                    Some(anim) => {
                        let clip = dreamcoast_scene::AnimationClip::from_gltf(anim, &node_map);
                        if clip.is_empty() {
                            info!("animation: clip {idx} has no node-TRS channels for this scene");
                        } else {
                            let dur = clip.duration;
                            let player = world.spawn();
                            world.insert(player, dreamcoast_scene::AnimationPlayer::new(clip));
                            info!(
                                "animation: playing clip {idx} '{}' ({} channels, {dur:.2}s)",
                                anim.name.as_deref().unwrap_or("<unnamed>"),
                                anim.channels.len(),
                            );
                        }
                    }
                    None => info!(
                        "animation: no clip {idx} ({} available)",
                        gscene.animations.len()
                    ),
                }
            }
            // Animation Stage B: CPU-skin any skinned primitives each frame.
            gltf_skinned = skin::build_skinned_meshes(
                &device,
                &gscene,
                &prim_handles,
                &node_map,
                &mesh_registry,
            )?;
            if !gltf_skinned.is_empty() {
                info!(
                    "skinning: {} skinned primitive(s) (GPU)",
                    gltf_skinned.len()
                );
            }
            // Animation Stage C: blend any morph-target primitives each frame (GPU
            // vertex-pulling where host storage is available, else CPU).
            gltf_morphed = morph::build_morph_meshes(
                &device,
                &gscene,
                &prim_handles,
                &node_map,
                &mesh_registry,
            )?;
            if !gltf_morphed.is_empty() {
                info!(
                    "morph: {} GPU + {} CPU morph primitive(s)",
                    gltf_morphed.gpu_count(),
                    gltf_morphed.cpu_count()
                );
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
                alpha_cutoff: 0.0,
                kind: dreamcoast_asset::MaterialKind::Opaque,
            });
            let mat_chrome = material_registry.add(MaterialDesc {
                base_color: [0.95, 0.96, 0.97, 1.0],
                metallic: 1.0,
                roughness: 0.08,
                tex: [NO_TEXTURE; 4],
                albedo: registry::representative_albedo(None, [0.95, 0.96, 0.97, 1.0]),
                alpha_cutoff: 0.0,
                kind: dreamcoast_asset::MaterialKind::Opaque,
            });
            let mat_copper = material_registry.add(MaterialDesc {
                base_color: [0.95, 0.64, 0.54, 1.0],
                metallic: 1.0,
                roughness: 0.35,
                tex: [NO_TEXTURE; 4],
                albedo: registry::representative_albedo(None, [0.95, 0.64, 0.54, 1.0]),
                alpha_cutoff: 0.0,
                kind: dreamcoast_asset::MaterialKind::Opaque,
            });
            let mat_red = material_registry.add(MaterialDesc {
                base_color: [0.85, 0.25, 0.2, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                tex: [NO_TEXTURE; 4],
                albedo: registry::representative_albedo(None, [0.85, 0.25, 0.2, 1.0]),
                alpha_cutoff: 0.0,
                kind: dreamcoast_asset::MaterialKind::Opaque,
            });
            // Spawn order defines the deterministic draw / TLAS-instance order (model,
            // chrome, copper, cube) — the order the legacy flat list used.
            let model_e = world
                .spawn_node()
                .named("model")
                .with(MeshInstance::new(mesh_model, mat_model))
                .with(LocalTransform::IDENTITY)
                .id();
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
            let cube_e = world
                .spawn_node()
                .named("red-cube")
                .with(MeshInstance::new(mesh_cube, mat_red))
                .with(LocalTransform::trs(
                    Vec3::new(0.0, r * 0.45, -r * 2.0),
                    r * 0.45,
                ))
                .id();

            // Phase 15 verification: `P15_SPIN[=<rad/s>]` attaches a `Spin` to the
            // (asymmetric, so visibly rotating) model + cube. The fixed-timestep
            // loop advances them each step and the per-frame parallel propagate +
            // draw renders the motion. Default (unset) = no Spin → byte-identical
            // gallery, so the parity baseline is untouched.
            if let Ok(raw) = std::env::var("P15_SPIN") {
                let speed = raw.parse::<f32>().ok().filter(|s| *s != 0.0).unwrap_or(1.0);
                world.insert(
                    model_e,
                    dreamcoast_scene::Spin {
                        axis: Vec3::Y,
                        speed,
                    },
                );
                world.insert(
                    cube_e,
                    dreamcoast_scene::Spin {
                        axis: Vec3::new(0.3, 1.0, 0.0),
                        speed: speed * 1.5,
                    },
                );
            }
        }
        dreamcoast_scene::propagate_transforms_parallel(&mut world, dreamcoast_jobs::global());

        // Materialize the ECS draw list into the flat `SceneObject` list the GPU passes
        // consume. (Static scene → built once; later stages rebuild on scene change. In
        // world mode `world` is empty — the per-frame list comes from the streamer.)
        let mut scene = build_scene(&world, &mesh_registry, &material_registry);
        // Decal/transparent census. Decals are tinted into the G-buffer by the deferred decal
        // pass (`record_decals`); transparents still fall back to opaque (track B).
        {
            use dreamcoast_asset::MaterialKind;
            let decals = scene
                .iter()
                .filter(|o| o.kind == MaterialKind::Decal)
                .count();
            let transparents = scene
                .iter()
                .filter(|o| o.kind == MaterialKind::Transparent)
                .count();
            if decals + transparents > 0 {
                info!(
                    "material kinds: {decals} decal (deferred decal pass), {transparents} \
                     transparent (opaque fallback) of {} drawables",
                    scene.len()
                );
            }
        }
        // Foliage soft edges (opt-in, approach C). By default `Transparent` foliage renders as a
        // crisp alpha-tested cutout (positive `alpha_cutoff`). `FOLIAGE_HASHED=1` flips that cutoff
        // NEGATIVE for foliage drawables, which switches gbuffer.slang to a world-space hashed
        // (stochastic) alpha test; the camera's sub-pixel TAA jitter then resolves the dither into
        // soft, translucent leaf edges — so pair it with `P_TAAU_FORCE=1` (TAA at native res). The
        // magnitude is unchanged, so shadows (which use |cutoff|) stay crisp. Default off keeps the
        // crisp cutout and the byte-identical non-foliage baseline (no `Transparent` is touched).
        if quality::env_bool("FOLIAGE_HASHED", false) {
            let mut n = 0;
            for o in scene.iter_mut() {
                if o.kind == dreamcoast_asset::MaterialKind::Transparent && o.alpha_cutoff > 0.0 {
                    o.alpha_cutoff = -o.alpha_cutoff;
                    n += 1;
                }
            }
            if n > 0 {
                info!(
                    "foliage: hashed alpha on {n} transparent drawable(s) — needs TAA \
                     (P_TAAU_FORCE=1 at native res) to resolve the dither into soft edges"
                );
            }
        }
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
            // Per-drawable world AABBs + representative albedo (for the surface-cache cards).
            let obj_aabb = fused.drawable_aabb;
            let obj_albedo = fused.drawable_albedo;
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
            // Stage S1 (per-mesh-distance-fields.md): composite per-mesh DFs (baked once per unique
            // mesh at ~5 cm target voxels, cached/instanced) into each clip level instead of the
            // fused whole-scene triangle-soup bake. Per-mesh is now the **DEFAULT for content**: the
            // fused bake is DEPRECATED — its coarse whole-scene voxels (~0.76 m on a 37 m scene) lose
            // thin features (reliefs, thin walls, tracery), so DF-based passes (GI/AO/reflection +
            // the debug view) march straight through them. The gallery keeps the fused bake (it is
            // the byte-identical anchor and a simple scene where per-mesh buys nothing). The first
            // cook of a non-instanced scene (Intel Sponza ~426 unique meshes) is slower but cached;
            // the win compounds on instanced content (a unique asset bakes once, reused per
            // placement). `P11_PERMESH_GDF=0` forces the deprecated fused path (fallback / A-B).
            // `scene_diag` is the "open space" distance for voxels no object covers.
            let use_permesh = !gallery_scene && quality::env_bool("P11_PERMESH_GDF", true);
            if !use_permesh && !gallery_scene {
                tracing::warn!(
                    "GDF: using the DEPRECATED fused whole-scene distance field (P11_PERMESH_GDF=0). \
                     Thin features (reliefs, thin walls) are lost below the coarse voxel size."
                );
            }
            let scene_diag = {
                let d = [amax[0] - amin[0], amax[1] - amin[1], amax[2] - amin[2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            };
            let mut mesh_sdfs: Vec<dreamcoast_asset::sdf::SdfVolume> = Vec::new();
            // F5 (gi-fidelity-phases.md): one per-mesh albedo volume per unique mesh, parallel to
            // `mesh_sdfs` (same order / dedup), baked over the SAME grid so the albedo tile aligns
            // 1:1 with the SDF tile. Opt-in (heavy: per-mesh bake + 3 atlas volumes); off ⇒ dense.
            let permesh_albedo = use_permesh && quality::env_bool("F5_PERMESH_ALBEDO", false);
            let mut mesh_albedos: Vec<dreamcoast_asset::sdf::AlbedoVolumes> = Vec::new();
            let mut compose_objects: Vec<crate::compose::ComposeObject> = Vec::new();
            if use_permesh {
                use std::collections::HashMap;
                let cache_dir = app::cooked_cache_dir();
                // G1 (gdf-reference-alignment.md): small-mesh radius cull — drop tiny drawables
                // from the composite (they barely move low-frequency GI/AO but each is a full
                // per-mesh bake). `P11_GDF_MIN_RADIUS` (m); 0 disables.
                let min_radius = std::env::var("P11_GDF_MIN_RADIUS")
                    .ok()
                    .and_then(|v| v.trim().parse::<f32>().ok())
                    .unwrap_or(crate::compose::DEFAULT_MIN_MESH_RADIUS);
                let mut mesh_index: HashMap<u32, usize> = HashMap::new();
                let mut culled = 0u32;
                let mut negated = 0u32;
                for d in world.draw_list() {
                    let cpu = mesh_registry.cpu(d.mesh);
                    let (mn, mx) = dreamcoast_asset::sdf::mesh_local_aabb_padded(&cpu.vertices);
                    if min_radius > 0.0
                        && crate::compose::mesh_world_radius(d.world, mn, mx) < min_radius
                    {
                        culled += 1;
                        continue;
                    }
                    let mi = if let Some(&i) = mesh_index.get(&d.mesh.0) {
                        i
                    } else {
                        let mdim = dreamcoast_asset::sdf::mesh_sdf_dim(mn, mx);
                        let mvtx = dreamcoast_asset::sdf::encode_vertices_fused(&cpu.vertices);
                        let midx = dreamcoast_asset::sdf::encode_indices(&cpu.indices);
                        let (mut vol, _) = dreamcoast_asset::cook::load_or_bake_mesh_sdf(
                            &mvtx, &midx, mdim, mn, mx, &cache_dir,
                        );
                        // A per-mesh DF that is *mostly* negative has globally-inverted normals
                        // (it reads open space as "inside"): negate it so the sign is correct.
                        // This removes the compose poisoning *and* the spurious floor-occlusion
                        // blotches those meshes cause via AO/GI. A correct solid is mostly
                        // positive outside and a correct thin sheet ~50 %, so the 60 % threshold
                        // only flips clearly-inverted meshes.
                        let neg = vol.voxels.iter().filter(|&&d| d < 0.0).count();
                        if neg * 5 > vol.voxels.len() * 3 {
                            for v in &mut vol.voxels {
                                *v = -*v;
                            }
                            negated += 1;
                        }
                        mesh_sdfs.push(vol);
                        // F5: bake this unique mesh's albedo over the same grid. Albedo depends on
                        // the drawable's material too, so — matching the mesh-only dedup here — we
                        // key it on the mesh + this first drawable's representative material colour
                        // (uniform per drawable in these scenes), repeated per triangle.
                        if permesh_albedo {
                            let alb = material_registry.get(d.material).albedo;
                            let tri_count = cpu.indices.len() / 3;
                            let mut tri_albedo = Vec::with_capacity(tri_count * 12);
                            for _ in 0..tri_count {
                                for c in alb {
                                    tri_albedo.extend_from_slice(&c.to_le_bytes());
                                }
                            }
                            let (av, _) = dreamcoast_asset::cook::load_or_bake_mesh_albedo(
                                &mvtx,
                                &midx,
                                &tri_albedo,
                                mdim,
                                mn,
                                mx,
                                &cache_dir,
                            );
                            mesh_albedos.push(av);
                        }
                        let i = mesh_sdfs.len() - 1;
                        mesh_index.insert(d.mesh.0, i);
                        i
                    };
                    compose_objects.push(crate::compose::ComposeObject::new(
                        d.world,
                        mi,
                        &mesh_sdfs[mi],
                    ));
                }
                info!(
                    "per-mesh DF: {} unique meshes, {} instances ({} culled < {:.2} m radius, \
                     {} inverted-meshes negated)",
                    mesh_sdfs.len(),
                    compose_objects.len(),
                    culled,
                    min_radius,
                    negated
                );
            }
            let sdf_bytes = if !use_permesh {
                let (sdf_vol, sdf_outcome) = dreamcoast_asset::cook::load_or_bake_scene_sdf(
                    &fused_v,
                    &fused_i,
                    sdf_dim,
                    amin,
                    amax,
                    &app::cooked_cache_dir(),
                );
                info!("scene SDF {sdf_dim}^3 ({sdf_outcome:?})");
                sdf_vol.to_le_bytes()
            } else {
                let vol = crate::compose::compose_sdf_level(
                    &compose_objects,
                    &mesh_sdfs,
                    amin,
                    amax,
                    sdf_dim,
                    scene_diag,
                );
                info!(
                    "scene SDF {sdf_dim}^3 (composed from {} per-mesh DFs)",
                    mesh_sdfs.len()
                );
                vol.to_le_bytes()
            };
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
                    // S1: finer levels compose from per-mesh DFs when opt-in, else the fused
                    // bake (this loop only runs for content — the gallery is single-level).
                    let sdf_le = if use_permesh {
                        crate::compose::compose_sdf_level(
                            &compose_objects,
                            &mesh_sdfs,
                            *lmin,
                            *lmax,
                            sdf_dim,
                            scene_diag,
                        )
                        .to_le_bytes()
                    } else {
                        dreamcoast_asset::cook::load_or_bake_scene_sdf(
                            &fused_v,
                            &fused_i,
                            sdf_dim,
                            *lmin,
                            *lmax,
                            &app::cooked_cache_dir(),
                        )
                        .0
                        .to_le_bytes()
                    };
                    sdf_store.push(sdf_le);
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
            // P3 (per-mesh-sdf-direct-sample-plan.md): pack every unique mesh's field into one
            // atlas volume + build the instance table / cell grid, then switch the SW-RT field
            // source to direct per-mesh sampling — the **content default** (dense loses per-mesh
            // resolution → thin-geo penetration + surface-cache checkerboard). `P11_DIRECT_SDF=0`
            // opts out to the dense-only composite (kept above as the hybrid's coarse field, and
            // as the A/B fallback). Content-only; the gallery keeps the dense anchor untouched.
            let direct_sdf = use_permesh && quality::env_bool("P11_DIRECT_SDF", true);
            if use_permesh && !direct_sdf {
                info!(
                    "GDF: per-mesh SDF direct sampling DISABLED (P11_DIRECT_SDF=0) — dense \
                     composite only (loses per-mesh resolution)"
                );
            }
            if direct_sdf {
                // Atlas memory cap: tiles are dense `dim³`, so downsampling the largest meshes
                // (whose extra resolution is low-frequency, covered by the coarse dense field)
                // trims the atlas a lot while thin features — resolved by their tight AABB, not
                // the cube dim — survive. `P11_ATLAS_MAX_DIM` tunes it (native = 48).
                let atlas_cap = std::env::var("P11_ATLAS_MAX_DIM")
                    .ok()
                    .and_then(|v| v.trim().parse::<u32>().ok())
                    .map(|d| d.clamp(dreamcoast_asset::sdf::MESH_SDF_MIN_DIM, 48))
                    .unwrap_or(32);
                let atlas =
                    dreamcoast_asset::sdf_atlas::SdfAtlas::pack_capped(&mesh_sdfs, atlas_cap);
                let res = crate::mesh_sdf::grid_res_for(compose_objects.len());
                let build = crate::mesh_sdf::build(&compose_objects, &atlas, amin, amax, res);
                // F5: pack the per-mesh albedo volumes into the SAME tile geometry as the SDF
                // atlas (one `tile_uvw` maps both), so the shader reads hit colour at the hit
                // instance with per-mesh precision. Opt-in (`F5_PERMESH_ALBEDO`); off ⇒ dense.
                let albedo_atlas = if permesh_albedo && mesh_albedos.len() == mesh_sdfs.len() {
                    Some(dreamcoast_asset::sdf_atlas::AlbedoAtlas::pack_like(
                        &atlas,
                        &mesh_albedos,
                        [0.7, 0.7, 0.7],
                    ))
                } else {
                    None
                };
                info!(
                    "per-mesh SDF direct sample: atlas {}x{}x{} ({:.1} MB), {} instances, {}^3 cell grid{}",
                    atlas.dim[0],
                    atlas.dim[1],
                    atlas.dim[2],
                    (atlas.voxels.len() * 4) as f32 / 1.0e6,
                    build.instance_count,
                    res,
                    if albedo_atlas.is_some() {
                        " + per-mesh albedo atlas (F5)"
                    } else {
                        ""
                    },
                );
                let alb_ch = albedo_atlas.as_ref().map(|a| {
                    [
                        a.channel_le_bytes(0),
                        a.channel_le_bytes(1),
                        a.channel_le_bytes(2),
                    ]
                });
                let alb_ref = alb_ch
                    .as_ref()
                    .map(|c| [c[0].as_slice(), c[1].as_slice(), c[2].as_slice()]);
                gdf.install_mesh_sdf(&device, &atlas.to_le_bytes(), atlas.dim, &build, alb_ref)?;
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
                // F1 (surface-cache virtualization): rank drawables for card residency from a
                // static reference camera resolved once here, matching the per-frame camera's
                // eye/focus precedence (`CAM_EYE`/`CAM_TARGET` → authored level view → orbit
                // framing). Within-budget scenes (the gallery) keep every drawable regardless of
                // this pose, so the anchor stays byte-identical; over-budget scenes select the
                // camera-relevant subset deterministically and mark the rest coarse fallback.
                let (ref_focus, ref_eye) =
                    match (parse_vec3_env("CAM_EYE"), parse_vec3_env("CAM_TARGET")) {
                        (Some(e), Some(t)) => (t, e),
                        (Some(e), None) => (scene_center, e),
                        _ => match level_view {
                            Some((e, t)) => (t, e),
                            None => (
                                scene_center,
                                scene_center
                                    + Vec3::new(scene_radius * 1.6, scene_radius * 0.55, 0.0),
                            ),
                        },
                    };
                let card_cam = fuse::CardCamera::from_look(ref_eye, ref_focus);
                let (cards, card_albedo, _residency) =
                    fuse::build_surface_cards(&obj_aabb, &obj_albedo, &card_cam);
                let num_cards = (cards.len() / 64) as u32;
                // C: content stamps the drawable's true albedo onto its cards (fine color);
                // the gallery keeps the legacy voxel-volume albedo (byte-identical anchor).
                // `P11_CARD_ALBEDO=0` forces the legacy path (A/B isolation of the cache color).
                let card_albedo = if gallery_scene || !quality::env_bool("P11_CARD_ALBEDO", true) {
                    None
                } else {
                    Some(card_albedo.as_slice())
                };
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
                gdf.build_surface_cache(&device, &cards, num_cards, cache_tile, card_albedo)?;
            }
        }

        let gui = Gui::new(&device, swapchain.format(), FRAMES_IN_FLIGHT)?;

        // One render-graph transient pool per frame-in-flight (reused only after the
        // frame slot's fence has signaled — no cross-frame hazards).
        let pools: Vec<ResourcePool> = (0..FRAMES_IN_FLIGHT).map(|_| ResourcePool::new()).collect();

        // Physically-based directional "sun" lighting. Content scenes author the sun in real
        // photometric units — **illuminance in lux** (clear-sky noon sun ≈ 100,000 lx) — and map
        // it to display with a physical-camera **EV100** exposure (`exposure = 1/(1.2·2^EV100)`,
        // sunny-16 ≈ EV15). The gallery keeps its legacy arbitrary 3.0 / 0.6 (the byte-identical
        // regression anchor — a synthetic test scene, its look is not a target). `SUN_LUX` /
        // `EV100` override the content values. Because the whole atmosphere scales with the sun
        // and the exposure compensates, the absolute lux is meaningful (not just relative): it is
        // what a light meter would read, and EV100 is what a camera would dial.
        // Direction TO the sun. ONE source of truth, resolved in priority order:
        //   1. `SUN_DIR="x,y,z"` env — explicit override, always wins.
        //   2. A loaded level (`.dclevel`) — its authored sun drives the sky/atmosphere/IBL/GI
        //      *and* the direct lighting + shadows, so the sky's sun, the cast shadows, and the
        //      shaded surfaces all agree (no drift between the lit image and the visible sun).
        //      The per-frame direct path reads the same `level_lighting` sun, so this unifies them.
        //   3. The code default — gallery keeps its overhead [0.4,0.8,0.4] (byte-identical anchor);
        //      a code-built content scene (e.g. `SCENE_GLTF`) takes the ~68° nave-clearing angle.
        // (auto-normalized in the push packers). Both direction AND intensity resolve from the SAME
        // source so the sky's sun, the cast shadows, the direct shading, the IBL/GI, and the camera
        // exposure all agree — a level's lighting is in physical lux, so it drives every consumer.
        let sun_dir = parse_vec3_env("SUN_DIR")
            .map(|v| [v.x, v.y, v.z])
            .or_else(|| level_lighting_override.as_ref().map(|ll| ll.sun_dir))
            .unwrap_or(if gallery_scene {
                [0.4, 0.8, 0.4]
            } else {
                // ~68° elevation, slightly off-axis. A narrow nave with ~12 m walls needs a HIGH sun
                // to clear the walls and put direct light on the floor (a low sun is fully blocked —
                // the wall shadow covers the whole nave); the slight X/Z tilt gives the columns
                // raking shadows. The roofed side aisles stay indirect (bounce/ambient) as in reality.
                [0.3, 0.9, 0.2]
            });
        // Sun illuminance (lux). Unified with the direction: `SUN_LUX`/`SUN_INTENSITY` > the loaded
        // level's directional intensity > the code default. So `self.sun_intensity` (which drives the
        // sky/atmosphere/IBL capture + the GI/reflection bounce) equals the level's authored sun — no
        // drift between the direct light and the sky/indirect.
        let sun_intensity = std::env::var("SUN_LUX")
            .or_else(|_| std::env::var("SUN_INTENSITY"))
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .or_else(|| level_lighting_override.as_ref().map(|ll| ll.sun_intensity))
            .unwrap_or(if gallery_scene { 3.0 } else { 100_000.0 })
            .max(0.0);
        // Analytic sun tint. Resolved like the direction: `SUN_COLOR="r,g,b"` env wins, else the
        // loaded level's directional-light color (so a level can author a warm sun — New Sponza's is
        // [1.0, 0.96, 0.9]), else white. White keeps the gallery/code-default scenes byte-identical;
        // a warm sun also takes some of the blue out of the daylight read (the sky ambient is blue).
        let sun_color = parse_vec3_env("SUN_COLOR")
            .map(|v| [v.x, v.y, v.z])
            .or_else(|| level_lighting_override.as_ref().map(|ll| ll.sun_color))
            .unwrap_or([1.0, 1.0, 1.0]);
        // Sun:sky ratio fed to the env capture (see the `sky_gain` field). Kept at 6.0 by default:
        // measurement showed that lowering it for "physical sun dominance" darkens open-roof
        // interiors (Sponza's atrium legitimately receives strong skylight), regressing exactly the
        // interior brightness we want. It is exposed as the `SKY_GAIN` knob for closed scenes that
        // want the sun to dominate, with the interior then filled by multibounce GI instead.
        let sky_gain = std::env::var("SKY_GAIN")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(6.0)
            .max(0.0);
        // Sky white balance — a per-channel gain on the procedural sky radiance (env capture), so it
        // warms/neutralises the IBL + SW-RT GI ambient without tinting the direct sun. Resolved like
        // the sun: `SKY_WB="r,g,b"` env wins, else the loaded level's `environment.sky_white_balance`,
        // else neutral `[1,1,1]` (a no-op → byte-identical to the pre-knob capture).
        let sky_wb = parse_vec3_env("SKY_WB")
            .map(|v| [v.x, v.y, v.z])
            .or(level_sky_wb)
            .unwrap_or([1.0, 1.0, 1.0]);
        // Physical-camera auto-exposure (see the field). Opt-in, never the gallery (fixed-exposure
        // anchor), and only when the metering pipeline built.
        let auto_exposure = std::env::var_os("AUTO_EXPOSURE").is_some()
            && !gallery_scene
            && deferred.exposure_buf_index().is_some();
        let ambient = 0.04f32;
        // On by default; `NO_POINT_LIGHTS=1` disables them (the path tracer has no
        // point lights, so a fair raster-vs-ground-truth comparison turns these off).
        let point_lights_on = std::env::var_os("NO_POINT_LIGHTS").is_none();
        // Clustered light culling (PR-6): opt-in seam. Default off = the brute-force point-light
        // loop (byte-identical anchor); on routes lighting through the froxel light list. Only
        // when the cluster compute system built (compute available).
        let clustered_lights = std::env::var_os("CLUSTERED_LIGHTS").is_some() && cluster.is_some();
        // A/B baseline (`CLUSTERED_BRUTE=1`): upload the same light buffer but loop ALL lights in
        // the shader (no froxel list) so brute-force vs clustered PROFILE_GPU can be compared on the
        // identical light set at scale. Implies the clustered light upload path (needs the buffer).
        let clustered_brute = std::env::var_os("CLUSTERED_BRUTE").is_some() && cluster.is_some();
        // Deterministic stress spawner (PR-6 scale proof): `TEST_LIGHTS=N` places N point lights
        // on a fixed grid across the scene bounds (no animation, fixed layout) so brute-force vs
        // clustered can be profiled at scale. 0 = off (level/gallery lights only).
        let test_lights: u32 = std::env::var("TEST_LIGHTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        // RenderQuality tier (Stage D): `RENDER_QUALITY=low|med|high` (unset => platform default).
        // The platform default consults the GPU identity (macOS perf, axis A): an Apple GPU maps to
        // the aggressive `Apple` tier, every other / unknown GPU (incl. all VK/D3D12 adapters) stays
        // on `Med` = the legacy no-reg baseline. An explicit `RENDER_QUALITY` value always wins over
        // the platform default. `qp` is the single tier→knob table; each knob below reads its own env
        // first and falls back to `qp.*`, so an explicit `P11_*`/`SHADOW_*`/`RENDER_SCALE`/`SSAO`
        // override always wins over the tier.
        let device_info = device.device_info();
        let quality = quality::RenderQuality::from_env_for_device(&device_info);
        // Scalability resolution base (single source of truth): the gallery is the byte-identical
        // path-tracer anchor, so its knobs resolve against a fixed legacy preset rather than the
        // active tier — structurally, so a new tier knob can't break the anchor by forgetting a
        // per-site `if gallery_scene { .. }` force. Content scenes resolve against the tier preset.
        // Each knob below is `env_override.unwrap_or(base.<knob>).clamp(..)`; the UI live-swap uses
        // the same `base` (see the RenderQuality combo) so the two paths can never drift.
        let base = if gallery_scene {
            quality::gallery_preset()
        } else {
            quality::preset(quality)
        };
        info!(
            "RenderQuality tier: {} (RENDER_QUALITY; GPU \"{}\")",
            quality.label(),
            device_info.name
        );
        // Phase 7: route the HDR result through a compute post-process (3x3 blur into
        // a storage image) before tonemapping. Initial state seedable via env var so
        // headless screenshots can exercise each demo (`P7_COMPUTE_POST=1`, etc.).
        let compute_post = compute_supported && std::env::var_os("P7_COMPUTE_POST").is_some();
        // Pipeline rebaseline PR-1: opt-in depth pre-pass (`DEPTH_PREPASS=1`). Off by default so
        // the frame graph is identical to the pre-pass-less path (byte-identical golden anchor).
        let depth_prepass = std::env::var_os("DEPTH_PREPASS").is_some();
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
        // HZB occlusion culling (PR-8) is layered on the GPU frustum cull; it requires
        // P7_CULL (the cull grid it operates on). If HZB_CULL is set without P7_CULL,
        // log and ignore — the base cull path stays byte-identical (default OFF).
        let hzb_cull = {
            let want = std::env::var_os("HZB_CULL").is_some();
            if want && !gpu_cull {
                eprintln!(
                    "[hzb] HZB_CULL=1 ignored: requires P7_CULL=1 (the GPU cull grid it culls). \
                     Set both to enable occlusion culling."
                );
            }
            want && gpu_cull && hzb.is_some()
        };
        // Hardware ray tracing (DXR / VK_KHR) path tracer — the explicit `--raytracing` option
        // (or the legacy `P8_PATHTRACE` env). Separate from the SW-RT RenderQuality tiers, which
        // all use the GDF software path; this swaps the whole render for the HW-RT ground truth.
        let path_trace = rt.has_trace() && rt.has_scene() && crate::app::raytracing_enabled();
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
        // Stage C2: GDF contact AO into the deferred ambient term. Gallery forced off (byte-
        // identical anchor); content takes the tier (Med now on — the contact-scale reach fix
        // makes AO add depth, not crush the interior). `P11_GDF_AO` overrides either way.
        let gdf_ao =
            gi.has_ao() && gdf.has_scene_sdf() && quality::env_bool("P11_GDF_AO", base.gdf_ao);
        // Screen-space near-field AO (HBAO-lite), composed with the GDF AO for crevice/contact
        // definition the coarse GDF can't resolve. Gallery off (byte-identical anchor); content
        // takes the tier default (`qp.ssao` — on for Med/High/Low, OFF for the Apple tier where
        // gdf_ao already covers contact AO and this reclaims the ~13 ms 2nd AO pass). `SSAO`
        // overrides either way; `SSAO_RADIUS/INTENSITY/BIAS/POWER` tune. World radius 0.5 m (contact
        // scale), intensity 1.5, bias 2 cm, power 1.5 (contrast).
        let ssao = compute_supported && quality::env_bool("SSAO", base.ssao);
        let env_f32 = |name: &str, default: f32| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(default)
        };
        let ssao_params = [
            env_f32("SSAO_RADIUS", 0.5),
            env_f32("SSAO_INTENSITY", 2.0),
            env_f32("SSAO_BIAS", 0.02),
            env_f32("SSAO_POWER", 1.5),
        ];
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
        // F3 (HW-RT high-fidelity path, first increment): opt-in `P_HWRT_GI=1`. Requires an RT
        // device AND a built scene TLAS (rt.rs builds BLAS/TLAS only for the gallery scene today —
        // the glTF/level path skips it because its per-primitive vertex/index storage buffers would
        // overflow the 64-slot bindless storage table). Gated here so a content scene without a TLAS
        // silently stays on the SW march instead of tracing an empty/stale acceleration structure.
        let hwrt_gi = std::env::var_os("P_HWRT_GI").is_some()
            && device.has_raytracing()
            && rt.has_scene()
            && gdf_gi;
        // 레퍼런스 엔진 GI-fidelity: the world irradiance volume (DDGI-lite) = our world-space RADIANCE CACHE,
        // the same idea 레퍼런스 SW-RT GI uses. It replaces the per-pixel 1-spp GI march with a smooth, stable
        // volume sample — so high-variance lighting (e.g. point lights) doesn't produce the firefly
        // speckle a single-bounce stochastic march leaves behind (which the temporal denoiser can't
        // fully clear). Default ON for content, never the gallery (the byte-identical anchor stays on
        // the legacy march); `P_GI_VOLUME=0` forces the march back.
        let gi_volume =
            quality::env_bool("P_GI_VOLUME", !gallery_scene) && gdf_gi && gi.has_gi_volume();
        // 레퍼런스식 indoor skylight occlusion knobs (only effective on the volume GI path, content):
        // `P_SKYVIS_TINT` = neutral OcclusionTint leak as a fraction of the occluded skylight
        // luminance (the occluded floor keeps brightness but loses the blue cast); `P_SKYVIS_MIN_OCC`
        // = min sky-visibility floor (=1.0 disables the occlusion entirely → SH-L1 baseline).
        let skyvis_tint = std::env::var("P_SKYVIS_TINT")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let skyvis_min_occ = std::env::var("P_SKYVIS_MIN_OCC")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        // PR-4 (render-pipeline re-baseline track): opt-in analytic height fog. Off by default
        // (`P_HEIGHT_FOG=1` to enable) so the byte-identical gallery/regression anchors are
        // untouched — the atmosphere slot exists in the graph wiring unconditionally, but the
        // pass itself is only added when this is true (see the `run()` call site).
        let height_fog = quality::env_bool("P_HEIGHT_FOG", false);
        // Density/falloff default to a gentle, scene-scale-relative haze: `1/scene_radius` puts
        // the characteristic falloff height and the "full extinction" distance both on the order
        // of the scene's own size, so the same defaults look sensible on the unit-radius gallery
        // and a much larger Sponza level alike. `P_FOG_DENSITY`/`P_FOG_HEIGHT_FALLOFF` override.
        let fog_density = std::env::var("P_FOG_DENSITY")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.15 / scene_radius.max(1e-3))
            .max(0.0);
        let fog_height_falloff = std::env::var("P_FOG_HEIGHT_FALLOFF")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.0 / scene_radius.max(1e-3))
            .max(0.0);
        // Inscatter gain defaults to the same sun:sky ratio the env-cube capture uses
        // (`sky_gain`) — single source, not a duplicated constant. `P_FOG_INSCATTER_GAIN`
        // overrides independently (e.g. to desaturate/dim the fog relative to the sky).
        let fog_inscatter_gain = std::env::var("P_FOG_INSCATTER_GAIN")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(sky_gain);
        // Stage D3: gallery forced to the legacy 8 (byte-identical anchor); content takes the tier.
        let gi_spp = std::env::var("P11_GI_SPP")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(base.gi_spp)
            .clamp(1, 256);
        // Stage D3: C3 bounce-ray march step cap. Gallery forced to the legacy 64 (byte-identical);
        // content takes the tier value. `P11_GI_MAX_STEPS` overrides.
        let gi_max_steps = std::env::var("P11_GI_MAX_STEPS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(base.gi_max_steps)
            .clamp(1, 256);
        // Stage D2: surface-cache amortized-relight period. The gallery is the byte-identical
        // regression anchor, so it is forced to 1 (every-frame relight = legacy) just like the
        // clipmap level count above; content scenes (Sponza) take the tier default and amortize.
        // `P11_CACHE_RELIGHT_PERIOD` overrides either way.
        let cache_relight_period = std::env::var("P11_CACHE_RELIGHT_PERIOD")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(base.cache_relight_period)
            .max(1);
        // Stage D2b: camera-visibility feedback drives the relight budget (off-screen cards relit
        // far less). Pure perf optimization — invariant for the on-screen image — so default on for
        // GI-on-distance-field visualization: `P_SC_VIZ=1` shades hits from the high-res surface
        // cache (2D mesh cards, final lit radiance — like the reference "scene" view); `P_WRC_VIZ=1`
        // shades from the coarse world radiance cache. Defined here (before `cache_feedback`) because
        // the surface-cache view forces the visibility gating off (below).
        let sc_viz = gi.has_wrc_view()
            && gdf.has_surface_cache()
            && gdf.has_cache_lighting()
            && std::env::var_os("P_SC_VIZ").is_some();
        // content; forced off for the gallery anchor (uniform period = byte-identical). Needs the
        // visibility pipeline (capability-gated). `P11_CACHE_FEEDBACK` overrides. The surface-cache
        // VIEW forces it OFF: the camera-visibility priority relights "hidden" cards 8x slower, but
        // the flyable view's DF march reaches those hidden-card surfaces and would show them stale
        // (black). Uniform relight fills every card so the view has no dead cards.
        let cache_feedback = gdf.has_cache_visibility()
            && !sc_viz
            && quality::env_bool("P11_CACHE_FEEDBACK", !gallery_scene);
        // Stage D3: relight gather rays/texel. Gallery forced to the legacy 8 (byte-identical);
        // content takes the tier value. `P11_CACHE_RELIGHT_SPP` overrides.
        let cache_relight_spp = std::env::var("P11_CACHE_RELIGHT_SPP")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(base.cache_relight_spp)
            .max(1);
        // Stage D1: half-res GI trace + bilateral upsample. Gallery stays full-res (the
        // byte-identical anchor); content scenes take the tier default. `P11_GI_HALF_RES`
        // overrides. Needs the upsample pipeline (capability-gated).
        let gi_half_res =
            gi.has_upsample() && quality::env_bool("P11_GI_HALF_RES", base.gi_half_res);
        // P1 (SW-RT GI 레퍼런스급): GI trace-resolution divisor used when `gi_half_res` is on (2 = legacy
        // half, 4 = quarter = sparser screen-space probes). Only affects content (the gallery traces
        // full-res). `P_GI_RES_DIV` overrides the tier.
        let gi_res_div = std::env::var("P_GI_RES_DIV")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(base.gi_res_div)
            .clamp(1, 16);
        // macOS/M3 perf (M3-C): reflection trace-resolution divisor used when `reflect_half_res` is on
        // (2 = legacy half = the old `div_ceil(2)`, 4 = quarter). Only affects content (the gallery
        // traces full-res = byte-identical). `P_REFLECT_RES_DIV` overrides the tier.
        let reflect_res_div = std::env::var("P_REFLECT_RES_DIV")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(base.reflect_res_div)
            .clamp(1, 16);
        // macOS/M3 perf: GDF AO trace-resolution divisor (1 = full-res = byte-identical; 2 = half).
        // Traced at 1/div then joint-bilateral upsampled; the gallery never runs gdf_ao so the anchor
        // is unaffected. `P_AO_RES_DIV` overrides the tier.
        let ao_res_div = std::env::var("P_AO_RES_DIV")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(base.ao_res_div)
            .clamp(1, 16);
        // macOS/M3 perf: à-trous spatial GI-denoise iteration count (2 = legacy byte-identical; the
        // Apple tier uses 1). The gallery forces the legacy 2 (it runs GI denoise, so the count would
        // otherwise shift the byte-identical anchor). `P_GI_ATROUS_STEPS` overrides the tier.
        let gi_atrous_steps = std::env::var("P_GI_ATROUS_STEPS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(base.gi_atrous_steps)
            .clamp(1, 5);
        // Screen-space radiance probe GI (P1+): opt-in. Replaces the GI consumption (world-volume
        // sample / per-pixel ray march) with per-tile screen probes + a per-pixel gather. Default
        // OFF, so the gallery anchor (no env) stays byte-identical; an explicit `SCREEN_PROBE=1`
        // turns it on for any scene — including the gallery, so the technique can be measured
        // against the path-traced ground truth (the only path-traceable scene).
        let screen_probe = gi.has_screen_probe() && quality::env_bool("SCREEN_PROBE", false);
        // P4 world radiance cache: an off-screen / far-field / infinite-bounce fallback escaped
        // screen-probe rays sample. Opt-in (`P_WRC=1`), default OFF: measurement shows it is
        // inert-to-slightly-negative on our architecture because the full-scene GDF clipmap
        // already covers all on/off-screen geometry (screen-probe rays hit real geometry instead
        // of escaping), so the cache's primary role is subsumed. Kept as correct infrastructure
        // for the multi-bounce-at-hits refinement (docs/world-radiance-cache.md). See the finding.
        let wrc = screen_probe && gi.has_wrc() && quality::env_bool("P_WRC", false);
        // `wrc_viz` (the GI-on-DF view pass) is enabled by either source flag; `sc_viz` (defined
        // earlier, above `cache_feedback`) selects the high-res surface-cache source.
        let wrc_viz = gi.has_wrc_view() && (std::env::var_os("P_WRC_VIZ").is_some() || sc_viz);
        // C4 denoise: on by default whenever GI runs (P11_GI_DENOISE=0 to see raw GI).
        let gi_denoise = gi.has_denoise() && quality::env_bool("P11_GI_DENOISE", base.gi_denoise);
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
        // Skipped when a full-screen GI-on-DF view replaces the output: the reflection only feeds
        // the scene HDR that the view discards, so it is ~21 ms of wasted work in that mode.
        let swrt_reflect = swrt_ok && !legacy_ibl && !wrc_viz;
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
            && quality::env_bool("P11_SURFACE_CACHE", base.surface_cache);
        // C8g: use the surface cache as the GDF REFLECTION hit radiance by default (accurate lit
        // colour for reflected objects — fixes the grazing avocado smear; ground hits have no
        // cards and fall back to the per-ray re-light). Cheap (only the per-frame cache-light pass
        // + a reflect-side lookup); the expensive per-ray GI cache lookup stays opt-in above.
        // `P11_REFLECT_CACHE=0` disables (reflections then use the C8a per-ray re-light).
        let reflect_cache = swrt_reflect
            && gdf.has_surface_cache()
            && gdf.has_cache_lighting()
            && quality::env_bool("P11_REFLECT_CACHE", base.reflect_cache);
        // Firefly clamp on by default (P11_FIREFLY_CLAMP=0 to disable / compare).
        let firefly_clamp = quality::env_bool("P11_FIREFLY_CLAMP", base.firefly_clamp);
        // C8d: roughness above which screen-mirror SSR stops contributing (GDF takes over). Gallery
        // forced to the legacy 0.5 (byte-identical anchor; a no-op under Med where qp is already 0.5)
        // so the Apple tier's lower 0.4 cutoff never shifts the gallery's reflections.
        let reflect_max_roughness = std::env::var("P11_REFLECT_MAX_ROUGHNESS")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(base.reflect_max_roughness);
        // Stage D3: reflection-ray march step cap. Gallery forced to the legacy 96 (byte-identical);
        // content takes the tier value. `P11_REFLECT_MAX_STEPS` overrides.
        let reflect_max_steps = std::env::var("P11_REFLECT_MAX_STEPS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(base.reflect_max_steps)
            .clamp(1, 256);
        // P3 (SW-RT GI 레퍼런스급 SW-RT): cone-trace LOD march slope. Gallery forced to 0 (legacy linear
        // march = byte-identical anchor); content takes the tier value. `P_CONE_K` overrides.
        let gdf_cone_k = std::env::var("P_CONE_K")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(base.gdf_cone_k)
            .clamp(0.0, 1.0);
        // Reflection history clamp permutation (0 off / 1 hard / 2 variance). Gallery forced to 0
        // (byte-identical legacy resolve = regression anchor); content takes the tier. `P_REFL_CLAMP`
        // + `P_REFL_CLAMP_GAMMA` override.
        let reflect_history_clamp = std::env::var("P_REFL_CLAMP")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(base.reflect_history_clamp)
            .min(2);
        let reflect_clamp_gamma = std::env::var("P_REFL_CLAMP_GAMMA")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(base.reflect_clamp_gamma)
            .clamp(0.0, 8.0);
        // GI temporal history clamp: gallery forced to 1.0 (hard 3x3 = legacy byte-identical anchor);
        // content takes the tier (0.0 = off = the static-shimmer fix). `P_GI_TEMPORAL_CLAMP` override.
        let gi_temporal_clamp = std::env::var("P_GI_TEMPORAL_CLAMP")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(base.gi_temporal_clamp)
            .clamp(0.0, 16.0);
        // Stage D3: half-res reflection trace + bilateral upsample (reuses the GI upsample).
        // Gallery forced off (full-res = byte-identical anchor); content takes the tier value.
        let reflect_half_res =
            gi.has_upsample() && quality::env_bool("P11_REFLECT_HALF_RES", base.reflect_half_res);
        // C8d: default to the full-res mirror SSR; opt into the stochastic glossy path to compare.
        let ssr_stochastic = quality::env_bool("P11_SSR_STOCHASTIC", base.ssr_stochastic);
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
        // The gallery is forced native (1.0) regardless of tier so the byte-identical anchor holds
        // even when the platform default is the Apple tier (render_scale 0.67) — mirrors the
        // `if gallery_scene { legacy } else { qp.* }` gate every other tier knob uses.
        let render_scale = std::env::var("RENDER_SCALE")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(base.render_scale)
            // Floor at 1/3 (DLSS "ultra performance" territory): below that even a temporal
            // reconstruction can't hold up. The TAAU jitter reconstruction (B-track) makes the
            // 0.4–0.6 range viable, which is what QHD/UHD high-fps needs; 1.0 stays the default
            // (byte-identical native).
            .clamp(0.3333, 1.0);
        let profiler_on = std::env::var("PROFILE_GPU").is_ok();
        let slot_pass_names: Vec<Vec<&'static str>> = vec![Vec::new(); FRAMES_IN_FLIGHT];
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
            sky_gain,
            sky_wb,
        )?;

        let mut window = window;
        let _ = window.take_resized();
        info!("entering render loop");

        // Read before `gdf` is moved into the struct (Phase 12 M2): the cooked SDF /
        // albedo were uploaded, so their one-time GPU bakes are pre-satisfied.
        let scene_gdf_cooked = gdf.scene_sdf_is_cooked();
        let scene_albedo_cooked = gdf.scene_albedo_is_cooked();

        // Snapshot the swapchain's extent/format/readback-layout before it is moved
        // into the struct, so the record thread can read them while the RHI thread
        // owns the swapchain (M4 B3). Kept in sync on resize.
        let swap_extent_cached = swapchain.extent_2d();
        let swap_format_cached = swapchain.format();
        let readback_layout_cached = device.swapchain_readback_layout(&swapchain);

        // PR-3: seed the translucency slot. `P_TRANSLUCENT_TEST=1` spawns two overlapping
        // tinted glass panes in the gallery (there is no translucent sample asset yet), to
        // verify sorted alpha compositing / depth-test occlusion / fog interaction. Empty by
        // default → the pass adds no work → byte-identical anchor.
        //
        let mut translucents: Vec<translucent::TranslucentObject> = Vec::new();
        if gallery_scene && quality::env_bool("P_TRANSLUCENT_TEST", false) {
            translucents =
                translucent::translucent_test_planes(&device, scene_radius, scene_center)?;
            info!(
                "P_TRANSLUCENT_TEST: spawned {} translucent glass pane(s)",
                translucents.len()
            );
        }
        // glTF `Transparent` (BLEND, non-decal) routing skeleton. The census below is where
        // production BLEND drawables would be lifted OUT of the opaque G-buffer list and turned
        // into `TranslucentObject`s for this slot — `translucent::TranslucentObject::from_scene`
        // does the conversion (mesh/transform/material → forward material). It is intentionally
        // NOT wired to run by default: doing so must also SUPPRESS the object's opaque G-buffer
        // draw (else it renders twice), and — critically — the foliage path depends on
        // `Transparent` drawables staying in the opaque G-buffer with a positive `alpha_cutoff`
        // (crisp cutout / hashed soft edges). So enabling gltf routing is deferred to the Phase
        // 20 work that also adds the G-buffer suppression; foliage behaviour is unchanged here.
        // See `docs/translucency-pass.md` (§ glTF routing).

        Ok(Self {
            window,
            _instance: instance,
            device,
            queue: Some(queue),
            compute_queue,
            swapchain: Some(swapchain),
            backend,
            gui,
            deferred,
            gdf,
            gi,
            gtao,
            atmosphere,
            translucency,
            translucents,
            reflect,
            particles,
            cull,
            cluster,
            hzb,
            rt,
            ibl,
            _textures: textures,
            world,
            mesh_registry,
            material_registry,
            skinned: gltf_skinned,
            morphed: gltf_morphed,
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
            rhi_thread: None,
            parallel_record: std::env::var_os("P15_PARALLEL_RECORD").is_some(),
            swap_extent_cached,
            swap_format_cached,
            readback_layout_cached,
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
            sun_color,
            ambient,
            // Camera exposure (pre-filmic, applied as `(ambient+lo)·exposure`). Gallery keeps the
            // legacy 0.6 (anchor). Content derives it from a physical-camera **EV100** so the lux
            // sun exposes correctly: `exposure = 1/(1.2·2^EV100)`. Default EV100 ≈ 14 (a touch under
            // sunny-16 to favour the bounce-lit interior, as a metered interior shot would). `EV100`
            // sets the stop directly; `EXPOSURE` still overrides the raw multiplier if given.
            exposure: std::env::var("EXPOSURE")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or_else(|| {
                    if gallery_scene {
                        0.6
                    } else {
                        let ev100 = std::env::var("EV100")
                            .ok()
                            .and_then(|v| v.parse::<f32>().ok())
                            .unwrap_or(14.0);
                        ev100_to_exposure(ev100)
                    }
                })
                .max(0.0),
            sky_gain,
            sky_wb,
            auto_exposure,
            gi_multibounce: std::env::var("P_GI_MULTIBOUNCE")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(if gallery_scene { 0.0 } else { 0.6 }),
            point_lights_on,
            clustered_lights,
            clustered_brute,
            test_lights,
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
                .unwrap_or(base.shadow_softness),
            // Soft-shadow tap count (RenderQuality knob, written to globals.shadow.w). Only the
            // soft path reads it; the shader clamps to [1, 16] (POISSON16). `SHADOW_TAPS` overrides.
            shadow_taps: std::env::var("SHADOW_TAPS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(base.shadow_taps)
                .clamp(1, 16),
            // PR-7 CSM/atlas: opt-in via `CSM` (default off = legacy single map = anchor).
            csm: csm::CsmConfig::from_env(),
            // `CSM_DEBUG=1` overlays the per-cascade index color so splits are inspectable.
            csm_debug: std::env::var("CSM_DEBUG").is_ok_and(|v| v != "0"),
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
            depth_prepass,
            particles_on,
            async_compute_on,
            gpu_cull,
            hzb_cull,
            hzb_stats: std::cell::Cell::new((0, 0)),
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
            ssao,
            ssao_params,
            gdf_gi,
            hwrt_gi,
            gi_volume,
            skyvis_tint,
            skyvis_min_occ,
            height_fog,
            fog_density,
            fog_height_falloff,
            fog_inscatter_gain,
            gi_spp,
            cache_relight_period,
            cache_feedback,
            cache_relight_spp,
            gi_max_steps,
            reflect_max_steps,
            gdf_cone_k,
            reflect_half_res,
            reflect_res_div,
            ao_res_div,
            gi_atrous_steps,
            gi_half_res,
            gi_res_div,
            screen_probe,
            wrc,
            wrc_viz,
            sc_viz,
            reflect_history_clamp,
            reflect_clamp_gamma,
            gi_temporal_clamp,
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
            time_of_day: std::env::var_os("TIME_OF_DAY").is_some(),
            multibounce: true,
            legacy_ibl,
            render_res,
            render_scale,
            taau,
            taau_on: quality::env_bool("P_TAAU", true),
            taau_jitter: quality::env_bool("P_TAAU_JITTER", true),
            taau_force: quality::env_bool("P_TAAU_FORCE", false),
            taa_mip_bias: std::env::var("TAA_MIP_BIAS")
                .ok()
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(quality::TAA_MIP_BIAS),
            velocity,
            velocity_on: quality::env_bool("P_VELOCITY", false),
            prev_transforms: Vec::new(),
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
            prev_angle: diag_angle.unwrap_or(if screenshot_mode { 0.7 } else { 0.0 }),
            sim_accumulator: 0.0,
            diag_obj,
            diag_pitch,
            // Non-gallery scenes (level / glTF / world) start in the free-fly camera: static at
            // the authored pose, then WASD + Q/E (down/up) move and right-mouse-drag looks. The
            // gallery keeps the auto-orbit demo view. Tab toggles between the two. Headless captures
            // ignore this (fly is gated off in screenshot mode -> the fixed parity pose).
            cam_mode: if gallery_scene {
                camera::CameraMode::Orbit
            } else {
                camera::CameraMode::Fly
            },
            fly: world_fly,
            tab_prev: false,
        })
    }

    /// The graphics queue (inline path). Panics if the RHI thread owns it.
    fn queue(&self) -> &Queue {
        self.queue
            .as_ref()
            .expect("queue is owned by the RHI thread")
    }

    /// The swapchain (inline path). Panics if the RHI thread owns it.
    fn swapchain(&self) -> &Swapchain {
        self.swapchain
            .as_ref()
            .expect("swapchain is owned by the RHI thread")
    }

    /// Mutable swapchain (inline path / resize). Panics if the RHI thread owns it.
    fn swapchain_mut(&mut self) -> &mut Swapchain {
        self.swapchain
            .as_mut()
            .expect("swapchain is owned by the RHI thread")
    }

    /// Swapchain extent — from the live swapchain when inline, else the cached value
    /// (the RHI thread owns the swapchain).
    fn swap_extent(&self) -> Extent2D {
        self.swapchain
            .as_ref()
            .map(|s| s.extent_2d())
            .unwrap_or(self.swap_extent_cached)
    }

    /// Swapchain format — live when inline, else cached (RHI thread owns it).
    fn swap_format(&self) -> Format {
        self.swapchain
            .as_ref()
            .map(|s| s.format())
            .unwrap_or(self.swap_format_cached)
    }

    /// Run the render loop until the window closes (or, in screenshot mode, every
    /// requested capture is saved).
    fn run(&mut self) -> anyhow::Result<()> {
        // M4 B3: opt into the separate RHI (submit) thread. Default off = the inline
        // single-thread path (byte-identical). Async-compute paths stay on the inline
        // path, so the worker is only spawned for the normal submit config.
        if std::env::var_os("P15_RHI_THREAD").is_some() {
            if self.async_cache_on || self.particles_on {
                info!(
                    "P15_RHI_THREAD ignored: async-compute (cache/particle) paths use the \
                     inline submitter"
                );
            } else {
                self.spawn_rhi_thread()?;
            }
        }
        // Debug: trigger one RenderDoc frame capture (RDOC_CAPTURE=1, under
        // `renderdoccmd capture`). No-op when not running under RenderDoc.
        #[cfg(windows)]
        let mut rdoc = std::env::var_os("RDOC_CAPTURE")
            .and_then(|_| renderdoc::RenderDoc::<renderdoc::V141>::new().ok());
        #[cfg(windows)]
        let mut rdoc_triggered = false;
        while !self.window.should_close() {
            #[cfg(windows)]
            if let Some(rd) = rdoc.as_mut()
                && !rdoc_triggered
                && self.frame_no >= 2
            {
                rd.trigger_capture();
                rdoc_triggered = true;
                info!("RenderDoc: frame capture triggered");
            }
            if !self.frame()? {
                break;
            }
        }
        // Reclaim the boundary objects from the worker before shutdown so they (and
        // device-idle/teardown) run single-threaded again.
        if let Some(rhi) = self.rhi_thread.take() {
            self.reclaim_rhi_objects(rhi.join());
        }
        self.device.wait_idle()?;
        info!("shutting down");
        Ok(())
    }

    /// Move the boundary objects (queue/swapchain/per-fif command buffers + frame
    /// fences/semaphores) into a freshly spawned RHI thread, plus persistent per-fif
    /// readback buffers it copies captures into (created here, dropped on the record
    /// thread at reclaim — never on the worker).
    fn spawn_rhi_thread(&mut self) -> anyhow::Result<()> {
        let layout = self.readback_layout_cached;
        let fif = self.command_buffers.len();
        let mut readback = Vec::with_capacity(fif);
        for _ in 0..fif {
            readback.push(self.device.create_buffer(&BufferDesc {
                size: layout.size,
                usage: BufferUsage::Readback,
            })?);
        }
        let objects = rhi_thread::ThreadObjects {
            queue: self.queue.take().expect("queue present before spawn"),
            swapchain: self
                .swapchain
                .take()
                .expect("swapchain present before spawn"),
            command_buffers: std::mem::take(&mut self.command_buffers),
            image_available: std::mem::take(&mut self.image_available),
            in_flight: std::mem::take(&mut self.in_flight),
            render_finished: std::mem::take(&mut self.render_finished),
            readback,
            readback_layout: layout,
        };
        self.rhi_thread = Some(rhi_thread::RhiThread::spawn(objects));
        info!("P15_RHI_THREAD: render-graph \u{2194} RHI thread split active");
        Ok(())
    }

    /// Put the worker's reclaimed objects back into `self` (after `join`). The
    /// readback buffers in `o` drop here, on the record thread (Rc-safe).
    fn reclaim_rhi_objects(&mut self, o: rhi_thread::ThreadObjects) {
        self.queue = Some(o.queue);
        self.swapchain = Some(o.swapchain);
        self.command_buffers = o.command_buffers;
        self.image_available = o.image_available;
        self.in_flight = o.in_flight;
        self.render_finished = o.render_finished;
    }

    /// Recreate the swapchain (resize) on the inline path: idle, recreate, drop
    /// cached transients, rebuild the per-image present semaphores, and refresh the
    /// extent/format/readback-layout caches.
    fn recreate_swapchain(&mut self, ww: u32, wh: u32) -> anyhow::Result<()> {
        self.device.wait_idle()?;
        self.swapchain_mut()
            .recreate(&swapchain_desc(Extent2D::new(ww, wh)))?;
        for p in &mut self.pools {
            p.clear(); // transient extents changed; drop cached targets
        }
        let count = self.swapchain().image_count();
        self.render_finished = build_render_finished(&self.device, count)?;
        self.swap_extent_cached = self.swapchain().extent_2d();
        self.swap_format_cached = self.swapchain().format();
        self.readback_layout_cached = self.device.swapchain_readback_layout(self.swapchain());
        self.needs_recreate = false;
        Ok(())
    }

    /// Resize under the RHI thread: reclaim the objects (join), recreate inline, then
    /// respawn the worker (which rebuilds its readback buffers for the new size).
    fn recreate_threaded(&mut self, ww: u32, wh: u32) -> anyhow::Result<()> {
        let rhi = self.rhi_thread.take().expect("rhi thread present");
        self.reclaim_rhi_objects(rhi.join());
        self.recreate_swapchain(ww, wh)?;
        self.spawn_rhi_thread()
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
            level::content_tex_compress(),
        )?;
        dreamcoast_scene::propagate_transforms_parallel(&mut world, dreamcoast_jobs::global());
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
        let _t_cpu = Instant::now(); // CPU record timer (stored right before present)
        // Apply a pending level hot-swap requested from the UI last frame.
        if let Some(idx) = self.pending_level.take() {
            self.load_level(idx)?;
        }
        // Pump Win32 messages ONCE per frame. A second pump_events here re-ran begin_frame (which
        // latches frame_start_pos = current cursor pos and clears the wheel) AFTER the first pump had
        // already drained every WM_MOUSEMOVE — so mouse_delta()/wheel collapsed to 0 every frame and
        // right-mouse look (and wheel speed) silently did nothing, while held keys (WASD) still worked.
        self.window.pump_events();
        if self.window.take_resized() {
            self.needs_recreate = true;
        }
        // The worker flags an out-of-date swapchain (present/acquire) here so the
        // resize below runs on the record thread (it owns the swapchain after join).
        if let Some(rhi) = &self.rhi_thread
            && rhi.take_recreate()
        {
            self.needs_recreate = true;
        }
        let (ww, wh) = self.window.size();
        if ww == 0 || wh == 0 {
            std::thread::sleep(std::time::Duration::from_millis(16));
            return Ok(true);
        }
        if self.needs_recreate {
            if self.rhi_thread.is_some() {
                self.recreate_threaded(ww, wh)?;
            } else {
                self.recreate_swapchain(ww, wh)?;
            }
        }

        // Wait for this frame slot's previous submission to finish BEFORE the acquire
        // below. The acquire reuses `image_available[fif]`, and Vulkan forbids
        // acquiring with a semaphore that still has a pending wait from that earlier
        // submit (VUID-vkAcquireNextImageKHR-semaphore-01779). This is the standard
        // frames-in-flight order: wait → reset → acquire → record → submit.
        //
        // M4 B3: when the RHI thread owns the swapchain it does wait+acquire itself;
        // the record half is backbuffer-relative (the IR carries no image index), so
        // it needs none here. `image_index` is then only used on the inline submit
        // path (the threaded path ships the IR and lets the worker resolve it).
        let fif = self.fif;
        let image_index = if self.rhi_thread.is_some() {
            0
        } else {
            let _t_wait = Instant::now();
            self.in_flight[fif].wait()?;
            // Acquire the drawable up front: its *actual* pixel size is the single
            // source of truth for this whole frame (ImGui display size, camera aspect,
            // render extent, viewport). A failed acquire skips here, BEFORE the ImGui
            // frame is started, so NewFrame/Render stay balanced. NOTE: `nextDrawable`
            // blocks here until the compositor releases a drawable — the 60Hz frame pace
            // in windowed mode — so fold it into the "wait" so `cpu-record` isolates the
            // true compute (record minus fence-wait minus this acquire-pacing block).
            let img = match self
                .swapchain()
                .acquire_next_image(&self.image_available[fif])?
            {
                Some(i) => i,
                None => {
                    self.needs_recreate = true;
                    return Ok(true);
                }
            };
            LAST_WAIT_US.store(
                _t_wait.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            img
        };
        // The swapchain (display) extent presents the final image; the *render* extent is where
        // the scene passes run. They are equal by default (byte-identical), but `RENDER_RES`
        // decouples them so headless can render the scene at QHD/UHD offscreen and the tonemap
        // downscales to the (display-bound) swapchain backbuffer. QHD/UHD perf track.
        let (sw, sh) = {
            let e = self.swap_extent();
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
        // HZB (PR-8): the Hi-Z pyramid must match the scene-depth (render) extent. Rebuild
        // it here if the render extent changed (RENDER_RES / render_scale / window resize).
        // `device` and `hzb` are disjoint fields, so borrow them independently.
        if self.hzb_cull {
            let App { device, hzb, .. } = self;
            if let Some(hzb) = hzb {
                hzb.resize(device, Extent2D::new(cw, ch))?;
            }
        }
        // QHD/UHD track: TAAU is active when the scene renders below the output resolution (upscale)
        // and isn't a path-trace/debug capture. It jitters the camera + reconstructs full-res from
        // history. When render == output (default), it's inactive (byte-identical).
        // TAAU runs when the scene renders below the output (upscale), or — with P_TAAU_FORCE — at
        // native (internal == output) as plain temporal AA (jitter + accumulation, no upscale).
        let taau_active = self.taau_on
            && self.taau.has_taau()
            && cw <= sw
            && ch <= sh
            && (cw < sw || ch < sh || self.taau_force);
        // One-time readout of the resolved resolution path so it's obvious at a glance what the
        // scene is actually rendering at vs the output (and whether TAAU upscaling is even on).
        if self.frame_no == 0 {
            let pct = if sw > 0 {
                100.0 * cw as f32 / sw as f32
            } else {
                100.0
            };
            info!(
                "render: internal {}x{} -> output {}x{} ({:.1}% scale, {}x area), TAAU={}, jitter={}",
                cw,
                ch,
                sw,
                sh,
                pct,
                if cw > 0 {
                    (sw as f32 / cw as f32).powi(2)
                } else {
                    1.0
                },
                if taau_active { "on" } else { "off (native)" },
                if taau_active && self.taau_jitter {
                    "on"
                } else {
                    "off"
                },
            );
        }
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
        LAST_FRAME_US.store((dt * 1e6) as u64, std::sync::atomic::Ordering::Relaxed);

        // --- Fixed-timestep simulation (M2) ---------------------------------------
        // The sim advances in whole `FIXED_DT` steps so motion is framerate-
        // independent and deterministic given the same dt sequence; the renderer
        // interpolates between the previous and current sim state by `render_alpha`.
        // **Headless capture is byte-identical by construction**: it is already
        // frame-counted and deterministic, so it bypasses the accumulator entirely
        // and keeps the exact legacy per-frame advance (fixed angle, real-dt
        // `elapsed`, CAPTURE_SEQ step).
        const FIXED_DT: f32 = 1.0 / 60.0;
        const MAX_STEPS: u32 = 5; // backlog cap — avoids the spiral of death after a stall
        self.prev_angle = self.angle;
        let render_alpha: f32;
        let sim_dt; // step length handed to per-frame GPU sim (particles)
        if self.screenshot_mode {
            // Unchanged legacy capture path (see above).
            self.elapsed += dt;
            sim_dt = dt.clamp(0.0, 1.0 / 30.0);
            render_alpha = 1.0;
            if self.capture_seq.is_some() {
                // CAPTURE_SEQ: advance the camera a fixed deterministic step per frame so the
                // dumped sequence exercises the temporal passes under motion (stability diff).
                // `CAPTURE_SEQ_STEP` (radians/frame, default 0.015) tunes it; 0 = static (a
                // shimmer/convergence test — the sequence should diff to ~0 when stable).
                let step = std::env::var("CAPTURE_SEQ_STEP")
                    .ok()
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.015);
                self.angle += step;
                // Advance animated objects one deterministic step per captured frame
                // so a CAPTURE_SEQ dump shows (reproducible) object motion, not just
                // camera motion. No-op when no entity carries `Spin` / `AnimationPlayer`.
                dreamcoast_scene::advance_spin(&mut self.world, FIXED_DT);
                dreamcoast_scene::advance_animation(&mut self.world, FIXED_DT);
            }
            self.prev_angle = self.angle; // no interpolation when capturing
        } else {
            self.sim_accumulator += dt;
            let mut steps = 0u32;
            while self.sim_accumulator >= FIXED_DT && steps < MAX_STEPS {
                self.sim_accumulator -= FIXED_DT;
                steps += 1;
                self.elapsed += FIXED_DT;
                self.angle += FIXED_DT * 0.6;
                // Animated objects advance one fixed step (framerate-independent,
                // deterministic). No-op when nothing carries `Spin` / `AnimationPlayer`.
                dreamcoast_scene::advance_spin(&mut self.world, FIXED_DT);
                dreamcoast_scene::advance_animation(&mut self.world, FIXED_DT);
            }
            if steps == MAX_STEPS {
                // Stalled longer than the cap: drop the backlog rather than chasing it.
                self.sim_accumulator = 0.0;
            }
            render_alpha = self.sim_accumulator / FIXED_DT;
            // Particles step by the sim time actually consumed this frame (0 when no
            // whole step elapsed → they hold still, the correct fixed-step behavior).
            sim_dt = steps as f32 * FIXED_DT;
        }
        // Rendered orbit angle interpolates between the last and current sim step.
        let render_angle = self.prev_angle + (self.angle - self.prev_angle) * render_alpha;

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
            || self.gi_volume
            || self.auto_exposure
            || self.wrc
            || self.wrc_viz
            || (self.swrt_reflect && self.reflect.has_reflect_temporal())
        {
            // The surface cache / stochastic GGX reflection accrue a sample per frame + temporally
            // accumulate, like the GI denoiser — warm them up before the static screenshot.
            GI_DENOISE_WARMUP
        } else {
            SCREENSHOT_WARMUP
        };
        // `WARMUP_FRAMES=<n>` overrides the headless warmup — e.g. to let an amortized cache
        // (surface-cache re-light) fully converge before the capture.
        let warmup = std::env::var("WARMUP_FRAMES")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(warmup);

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

        // Re-derive world transforms from the (possibly just-animated) locals via the
        // parallel ECS pass, then materialize the draw list. For a static scene this
        // recomputes identical matrices (byte-identical), so the gallery baseline is
        // unaffected; for animated objects it pushes their new pose to the draw list.
        // World/streaming mode keeps its own path (the world is empty here).
        let mut scene = if self.streaming.is_some() {
            Vec::new()
        } else {
            dreamcoast_scene::propagate_transforms_parallel(
                &mut self.world,
                dreamcoast_jobs::global(),
            );
            build_scene(&self.world, &self.mesh_registry, &self.material_registry)
        };

        // Velocity (PR-2) single prev-pose source: seed each drawable's previous transform from
        // last frame's stored transforms (same stable draw-list order). A newly-appeared drawable
        // (or first frame) has no history → default (identity + current), i.e. zero object motion.
        // Skinning / morph overwrite their entries with the prev palette / weights below. Built
        // only when velocity is on (else an empty vec — no per-frame cost on the default path).
        let mut prev_scene: Vec<velocity::PrevPose> = if self.velocity_on {
            scene
                .iter()
                .enumerate()
                .map(|(i, obj)| velocity::PrevPose {
                    transform: self
                        .prev_transforms
                        .get(i)
                        .copied()
                        .unwrap_or(obj.transform),
                    skin_palette: 0,
                    morph_weights: 0,
                })
                .collect()
        } else {
            Vec::new()
        };

        // Animation Stage B: CPU-skin the skinned primitives and swap each skinned
        // drawable to this frame-in-flight's vertex buffer. Inline path only — the
        // per-frame vertex write relies on this slot's fence having been waited at
        // frame start (the RHI thread defers that wait; skinning + P15_RHI_THREAD is
        // not supported in B.1).
        if !self.skinned.is_empty() && self.rhi_thread.is_none() {
            skin::update_palettes(&mut self.skinned, &self.world, fif)?;
            skin::patch_scene(&self.skinned, &mut scene, &mut prev_scene, fif);
        }
        // Animation Stage C: blend morph targets (GPU = write the per-frame weights
        // buffer; CPU = re-blend the vertex ring) + patch this frame's drawables. Inline
        // path only (the per-frame storage/vertex write relies on the frame-start fence).
        if !self.morphed.is_empty() && self.rhi_thread.is_none() {
            morph::apply_morph(&mut self.morphed, &self.world, fif)?;
            morph::patch_scene(&self.morphed, &mut scene, &mut prev_scene, fif);
        }

        // Orbiting camera framing the whole sample scene — or, in single-object
        // diagnostic mode, a tight orbit centred on one scene object so it can be
        // inspected from every side (azimuth = self.angle, elevation = diag_pitch).
        let (focus, eye) = if let Some(oi) = self.diag_obj.filter(|&i| i < scene.len()) {
            let center = scene[oi].transform.w_axis.truncate();
            let radius = scene[oi].transform.x_axis.truncate().length(); // uniform scale
            let dist = radius * 4.5;
            let pitch = self.diag_pitch.unwrap_or(0.18); // slight elevation by default
            let (sp, cp) = (pitch.sin(), pitch.cos());
            let eye =
                center + dist * Vec3::new(cp * render_angle.cos(), sp, cp * render_angle.sin());
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
                let a = base + render_angle;
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
                    render_angle.cos() * dist,
                    self.scene_radius * 0.55,
                    render_angle.sin() * dist,
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
        let proj_noflip = Mat4::perspective_rh(
            60f32.to_radians(),
            cw as f32 / ch as f32,
            CLUSTER_Z_NEAR,
            CLUSTER_Z_FAR,
        );
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
            let has_auto_exposure = self.deferred.exposure_buf_index().is_some();
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
                auto_exposure,
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
                time_of_day,
                multibounce,
                legacy_ibl,
                post_mode,
                aliasing,
                compute_post,
                particles_on,
                async_compute_on,
                gpu_cull,
                hzb_cull,
                hzb_stats,
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
                gi_max_steps,
                cache_relight_period,
                cache_relight_spp,
                gi_half_res,
                gi_res_div,
                reflect_history_clamp,
                reflect_clamp_gamma,
                gi_temporal_clamp,
                gi_denoise,
                reflect_max_steps,
                reflect_res_div,
                reflect_half_res,
                ao_res_div,
                gi_atrous_steps,
                gdf_cone_k,
                render_scale,
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
                    // The `apple` platform tier is auto-selected on Apple GPUs (not via env); it is
                    // listed here so the live tier displays correctly and a manual re-pick works.
                    let mut tier_idx = match *quality {
                        quality::RenderQuality::Low => 0usize,
                        quality::RenderQuality::Med => 1,
                        quality::RenderQuality::High => 2,
                        quality::RenderQuality::Apple => 3,
                    };
                    if ui.combo_simple_string(
                        "RenderQuality",
                        &mut tier_idx,
                        &["low", "med", "high", "apple"],
                    ) {
                        let nq = [
                            quality::RenderQuality::Low,
                            quality::RenderQuality::Med,
                            quality::RenderQuality::High,
                            quality::RenderQuality::Apple,
                        ][tier_idx];
                        *quality = nq;
                        // Re-derive each knob from the SAME base the construction path uses: the
                        // gallery resolves against the fixed legacy `gallery_preset()`, content
                        // against the tier — so the gallery-lock is structural (no scattered
                        // `if *is_gallery` here) and the two resolution paths can never drift.
                        // Capability gates stay so a tier can't enable a feature the device lacks.
                        let base = if *is_gallery {
                            quality::gallery_preset()
                        } else {
                            quality::preset(nq)
                        };
                        *gi_spp = base.gi_spp.clamp(1, 256);
                        *cache_relight_period = base.cache_relight_period.max(1);
                        *gi_half_res = gi.has_upsample() && base.gi_half_res;
                        *gi_res_div = base.gi_res_div.clamp(1, 16);
                        *reflect_history_clamp = base.reflect_history_clamp.min(2);
                        *reflect_clamp_gamma = base.reflect_clamp_gamma;
                        *gi_temporal_clamp = base.gi_temporal_clamp;
                        *gi_denoise = gi.has_denoise() && base.gi_denoise;
                        *reflect_cache = *swrt_reflect
                            && gdf.has_surface_cache()
                            && gdf.has_cache_lighting()
                            && base.reflect_cache;
                        *surface_cache = gdf.has_surface_cache()
                            && gdf.has_cache_lighting()
                            && base.surface_cache;
                        *ssr_stochastic = base.ssr_stochastic;
                        *reflect_max_roughness = base.reflect_max_roughness;
                        *gdf_ao = gi.has_ao() && gdf.has_scene_sdf() && base.gdf_ao;
                        *firefly_clamp = base.firefly_clamp;
                        *shadow_softness = base.shadow_softness;
                        *shadow_taps = base.shadow_taps.clamp(1, 16);
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

                    // Live scalability panel (docs/scalability-system.md): the individual knobs the
                    // RenderQuality tier resolves, exposed as live sliders/toggles. The render graph
                    // rebuilds every frame, so a change here takes effect immediately — no restart.
                    // The gallery scene is the byte-identical path-tracer-parity anchor, so its
                    // controls are disabled here (the headless screenshot path has no UI at all and
                    // is unaffected regardless; this is purely to stop an interactive session from
                    // perturbing the anchor image by hand).
                    if ui.collapsing_header("Scalability", open) {
                        if *is_gallery {
                            ui.text_disabled(
                                "gallery scene: scalability knobs locked (byte-identical anchor)",
                            );
                        }
                        // Coarse scalability-group profile of the active tier (read-only): the six
                        // 0-3 group levels this tier corresponds to (quality::ScalabilityGroup). A
                        // glance-level summary; the fine sliders below are the precise controls.
                        ui.text(format!("tier '{}' group levels:", quality.label()));
                        for (g, lvl) in quality::groups(*quality) {
                            ui.same_line();
                            ui.text_disabled(format!("{}={}", g.label(), lvl));
                        }
                        ui.disabled(*is_gallery, || {
                            ui.slider("Render scale", 0.33, 1.0, render_scale);
                            *render_scale = render_scale.clamp(0.3333, 1.0);
                            if ui.slider("GI res divisor", 1u32, 8, gi_res_div) {
                                *gi_res_div = (*gi_res_div).clamp(1, 8);
                            }
                            if ui.slider("Reflect res divisor", 1u32, 8, reflect_res_div) {
                                *reflect_res_div = (*reflect_res_div).clamp(1, 8);
                            }
                            if ui.slider("AO res divisor", 1u32, 4, ao_res_div) {
                                *ao_res_div = (*ao_res_div).clamp(1, 4);
                            }
                            if ui.slider("GI a-trous steps", 1u32, 5, gi_atrous_steps) {
                                *gi_atrous_steps = (*gi_atrous_steps).clamp(1, 5);
                            }
                            if ui.slider("GI samples/px", 1u32, 32, gi_spp) {
                                *gi_spp = (*gi_spp).clamp(1, 256);
                            }
                            if ui.slider("GI max march steps", 8u32, 128, gi_max_steps) {
                                *gi_max_steps = (*gi_max_steps).clamp(1, 256);
                            }
                            if ui.slider("Reflect max march steps", 8u32, 256, reflect_max_steps) {
                                *reflect_max_steps = (*reflect_max_steps).clamp(1, 256);
                            }
                            if ui.slider("Cache relight period", 1u32, 256, cache_relight_period) {
                                *cache_relight_period = (*cache_relight_period).max(1);
                            }
                            ui.slider("Reflect max roughness", 0.0, 1.0, reflect_max_roughness);
                            ui.slider("GDF cone-trace slope", 0.0, 0.2, gdf_cone_k);
                            *gdf_cone_k = gdf_cone_k.clamp(0.0, 1.0);
                            ui.slider("Shadow softness (0=hard)", 0.0, 0.1, shadow_softness);
                            if ui.slider("Shadow taps", 1u32, 16, shadow_taps) {
                                *shadow_taps = (*shadow_taps).clamp(1, 16);
                            }

                            ui.checkbox("Stochastic SSR", ssr_stochastic);
                            if ui.checkbox("GI half-res trace", gi_half_res) {
                                *gi_half_res = gi.has_upsample() && *gi_half_res;
                            }
                            if ui.checkbox("Reflect half-res trace", reflect_half_res) {
                                *reflect_half_res = gi.has_upsample() && *reflect_half_res;
                            }
                            if ui.checkbox("GDF ambient occlusion", gdf_ao) {
                                *gdf_ao = gi.has_ao() && gdf.has_scene_sdf() && *gdf_ao;
                            }
                            ui.checkbox("Firefly clamp", firefly_clamp);
                            if ui.checkbox("GI denoise", gi_denoise) {
                                *gi_denoise = gi.has_denoise() && *gi_denoise;
                            }

                            let mut clamp_idx = (*reflect_history_clamp).min(2) as usize;
                            if ui.combo_simple_string(
                                "Reflect history clamp",
                                &mut clamp_idx,
                                &["off", "hard", "variance"],
                            ) {
                                *reflect_history_clamp = clamp_idx as u32;
                            }

                            if ui.button("Reset to tier default") {
                                let base = if *is_gallery {
                                    quality::gallery_preset()
                                } else {
                                    quality::preset(*quality)
                                };
                                *render_scale = base.render_scale.clamp(0.3333, 1.0);
                                *gi_res_div = base.gi_res_div.clamp(1, 16);
                                *reflect_res_div = base.reflect_res_div.clamp(1, 16);
                                *ao_res_div = base.ao_res_div.clamp(1, 16);
                                *gi_atrous_steps = base.gi_atrous_steps.clamp(1, 5);
                                *gi_spp = base.gi_spp.clamp(1, 256);
                                *gi_max_steps = base.gi_max_steps.clamp(1, 256);
                                *reflect_max_steps = base.reflect_max_steps.clamp(1, 256);
                                *cache_relight_period = base.cache_relight_period.max(1);
                                *cache_relight_spp = base.cache_relight_spp.max(1);
                                *reflect_max_roughness = base.reflect_max_roughness;
                                *gdf_cone_k = base.gdf_cone_k.clamp(0.0, 1.0);
                                *shadow_softness = base.shadow_softness;
                                *shadow_taps = base.shadow_taps.clamp(1, 16);
                                *ssr_stochastic = base.ssr_stochastic;
                                *gi_half_res = gi.has_upsample() && base.gi_half_res;
                                *reflect_half_res = gi.has_upsample() && base.reflect_half_res;
                                *gdf_ao = gi.has_ao() && gdf.has_scene_sdf() && base.gdf_ao;
                                *firefly_clamp = base.firefly_clamp;
                                *gi_denoise = gi.has_denoise() && base.gi_denoise;
                                *reflect_history_clamp = base.reflect_history_clamp.min(2);
                                *reflect_clamp_gamma = base.reflect_clamp_gamma;
                                *gi_temporal_clamp = base.gi_temporal_clamp;
                                *reflect_cache = *swrt_reflect
                                    && gdf.has_surface_cache()
                                    && gdf.has_cache_lighting()
                                    && base.reflect_cache;
                                *surface_cache = gdf.has_surface_cache()
                                    && gdf.has_cache_lighting()
                                    && base.surface_cache;
                            }
                        });
                    }

                    if ui.collapsing_header("Lighting", open) {
                        ui.combo_simple_string("Debug view", debug_view, &DEBUG_VIEWS);
                        ui.input_float3("Sun dir", sun_dir).build();
                        ui.slider("Sun intensity", 0.0, 32.0, sun_intensity);
                        ui.slider("Ambient", 0.0, 0.5, ambient);
                        ui.slider("Exposure", 0.1, 4.0, exposure);
                        if has_auto_exposure {
                            ui.checkbox("Auto exposure", auto_exposure);
                        }
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
                            ui.checkbox("Time-of-day (dynamic sun)", time_of_day);
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
                        if *gpu_cull {
                            ui.checkbox("  - HZB occlusion cull", hzb_cull);
                            if *hzb_cull {
                                let (survived, culled) = hzb_stats.get();
                                ui.text(format!(
                                    "    {} total / {} vis / {} occluded",
                                    GRID_COUNT, survived, culled
                                ));
                            }
                        }
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
                .map(|(i, &name)| {
                    let dt = ticks[i + 1].saturating_sub(ticks[i]);
                    (name, dt as f32 * period_ns * 1e-6)
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
                let frame_ms =
                    LAST_FRAME_US.load(std::sync::atomic::Ordering::Relaxed) as f32 / 1000.0;
                let wait_ms =
                    LAST_WAIT_US.load(std::sync::atomic::Ordering::Relaxed) as f32 / 1000.0;
                // frame = full wall time; wait = fence stall on the GPU; cpu ≈ frame − wait is the
                // CPU record/build long pole. wait≈0 with cpu>gpu ⇒ CPU-bound (the real bottleneck).
                let cpu_ms = LAST_CPU_US.load(std::sync::atomic::Ordering::Relaxed) as f32 / 1000.0;
                // record = CPU time between fence-wait-end and present (pure graph build + encode +
                // submit, no present pacing). This is the real CPU long pole on a CPU-bound frame.
                tracing::info!(
                    "GPU profile (total {total:.4} ms):\n{rows}\n  --- frame {frame_ms:.3} ms | fence-wait {wait_ms:.3} ms | cpu-record {:.3} ms | gpu-passes {total:.3} ms",
                    (cpu_ms - wait_ms).max(0.0)
                );
            }
        }

        // Inline path: reset this slot's fence and open its command buffer. When the
        // RHI thread owns the command buffers (`cmd` is `None`), the worker does the
        // reset + begin/translate/end itself; the record thread builds `frame_list`
        // instead and ships it.
        let cmd: Option<&CommandBuffer> = if self.rhi_thread.is_some() {
            None
        } else {
            self.in_flight[fif].reset()?;
            let cmd = &self.command_buffers[fif];
            cmd.begin()?;
            Some(cmd)
        };
        // The whole frame's IR (capture + graph, in record order) when threaded.
        let frame_list = CommandList::new();

        // Lighting: a level supplies its own sun + point lights; otherwise the gallery's
        // code-default sun + two coloured point lights (preserved exactly = byte-identical).
        let r = self.model_radius;
        let point_intensity = r * r * 8.0;
        let (mut sun_dir, sun_intensity, point_count, point_pos, point_color) =
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

        // Time-of-day: arc the sun across the sky from elapsed time. The atmosphere + IBL
        // already recapture per frame (`realtime_env`), and `maybe_capture` re-marches the sky
        // on the sun change, so the physical sky moves with the sun. Off by default (static).
        if self.time_of_day {
            let t = self.elapsed * 0.15; // ~40 s/day; future: bind to a day-length setting
            let height = (t.sin() * 0.5 + 0.5) * 0.85 + 0.06; // stay above the horizon
            let horiz = (1.0 - height * height).max(0.0).sqrt();
            sun_dir = [t.cos() * horiz, height, t.sin() * horiz * 0.4];
        }

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
        // Record the (occasional) environment capture into the same target the frame
        // uses: the real command buffer inline, or the shipped IR list when threaded.
        let capture_rec: &dyn Recorder = match cmd {
            Some(c) => c,
            None => &frame_list,
        };
        self.ibl.maybe_capture(
            capture_rec,
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
            self.sky_gain,
            self.sky_wb,
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

        // PR-7 CSM: fit N cascades to the view frustum when the seam is on (single source of
        // all shadow matrices/splits). The perspective params match the scene camera above
        // (fov 60deg, near 0.05, far 100.0) so the cascades tile the depth range the view
        // samples. Empty on the legacy path (byte-identical anchor).
        let csm_slots = if self.csm.enabled {
            csm::compute_cascades(
                &self.csm,
                &csm::ViewCamera {
                    eye,
                    target: focus,
                    fov_y_rad: 60f32.to_radians(),
                    aspect: cw as f32 / ch as f32,
                    near: 0.05,
                    far: 100.0,
                },
                sun_dir,
            )
        } else {
            Vec::new()
        };

        // Write this frame's globals slice.
        let globals = Globals {
            camera_pos: [eye.x, eye.y, eye.z, 0.0],
            sun_direction: normalize3(sun_dir),
            sun_color: [
                self.sun_color[0],
                self.sun_color[1],
                self.sun_color[2],
                sun_intensity,
            ],
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
                self.gi_multibounce, // w: GI multi-bounce energy compensation strength (pbr.slang)
            ],
            // Last frame's view-projection (updated end-of-frame) so the SSR history
            // sample reprojects the world hit point into the previous frame (Stage C7b).
            prev_view_proj: self.prev_view_proj,
            // Clustered lighting (PR-6): row 2 of the world->view matrix so the shader can
            // recover positive linear view depth from a G-buffer world position, plus the
            // camera near/far for the froxel Z slicing. `view` is column-major (glam), so
            // row 2 = the .z component of each column.
            cluster_view_z_row: [view.x_axis.z, view.y_axis.z, view.z_axis.z, view.w_axis.z],
            cluster_params: [CLUSTER_Z_NEAR, CLUSTER_Z_FAR, 0.0, 0.0],
            // PR-7 CSM: filled from the fitted cascade slots when the seam is on; zero here =
            // the legacy single-map path (the shader branches on csm_params.x == 0 → anchor).
            csm_params: {
                let atlas = self.csm.atlas_size as i32;
                let tile = self.csm.tile_size() as i32;
                [csm_slots.len() as i32, self.csm_debug as i32, tile, atlas]
            },
            csm_split: {
                let mut s = [0.0f32; 4];
                for (i, slot) in csm_slots.iter().take(4).enumerate() {
                    s[i] = slot.split_far;
                }
                s
            },
            csm_opts: [self.csm.blend_frac, 0.0, 0.0, 0.0],
            csm_view_proj: {
                let mut m = [[0.0f32; 16]; 4];
                for (i, slot) in csm_slots.iter().take(4).enumerate() {
                    m[i] = slot.view_proj.to_cols_array();
                }
                m
            },
            csm_atlas_uv: {
                // Per-cascade atlas UV sub-rect (xy offset, zw scale). The sampler maps the
                // cascade clip → [0,1] tile UV → this sub-rect in the shared atlas texture.
                let atlas = self.csm.atlas_size.max(1) as f32;
                let mut uv = [[0.0f32; 4]; 4];
                for (i, slot) in csm_slots.iter().take(4).enumerate() {
                    uv[i] = [
                        slot.rect.x as f32 / atlas,
                        slot.rect.y as f32 / atlas,
                        slot.rect.width as f32 / atlas,
                        slot.rect.height as f32 / atlas,
                    ];
                }
                uv
            },
        };
        let globals_offset = fif as u64 * GLOBALS_SLICE;
        // Firefly clamp ceiling (raw radiance, max component). ~8 keeps diffuse + moderate
        // gloss but caps blown-out specular spikes; 1e30 = effectively off (byte-identical).
        let firefly_max = if self.firefly_clamp { 8.0f32 } else { 1e30 };
        self.deferred
            .write_globals(globals_offset, globals_bytes(&globals))?;

        // Clustered light culling (PR-6): assemble this frame's point-light list and upload it
        // to the cluster/light buffer, returning the bindless indices the build + lighting passes
        // read. Done BEFORE the graph is built (host write + possible realloc mutate `self.cluster`;
        // the graph then borrows it immutably for the record closures).
        //
        // Parity note: the scene's authored point lights get an EFFECTIVELY INFINITE radius so
        // every cluster bins them — the shader then accumulates the SAME lights in the SAME order
        // as the brute-force loop (which applies no distance cutoff), so the result is byte-identical
        // for scenes with few lights. The deterministic TEST_LIGHTS stress grid gets a finite radius
        // (that's where per-cluster culling actually pays off).
        let cluster_indices: Option<(u32, u32, u32, u32)> = if let (true, Some(cluster_sys)) = (
            self.clustered_lights || self.clustered_brute,
            self.cluster.as_mut(),
        ) {
            let mut lights: Vec<ClusterLight> = Vec::new();
            let n = if self.point_lights_on {
                point_count as usize
            } else {
                0
            };
            for i in 0..n.min(4) {
                lights.push(ClusterLight {
                    position: [point_pos[i][0], point_pos[i][1], point_pos[i][2]],
                    radius: CLUSTER_Z_FAR * 4.0, // "infinite": covers the whole frustum
                    color: [point_color[i][0], point_color[i][1], point_color[i][2]],
                    intensity: point_color[i][3],
                });
            }
            if self.test_lights > 0 {
                lights.extend(test_light_grid(
                    self.test_lights,
                    self.scene_center,
                    self.scene_radius,
                ));
            }
            Some(cluster_sys.upload(&self.device, fif, &lights)?)
        } else {
            None
        };

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
        // P4: (re)allocate the world radiance cache atlas for the scene's clipmap level count.
        if self.wrc || self.wrc_viz {
            let levels = self.gdf.clip_descriptor().map(|(_, c)| c).unwrap_or(1);
            self.gi.prepare_wrc(&self.device, levels)?;
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
        let backbuffer = graph.import_backbuffer(self.swap_format(), swap_extent);
        let g_albedo = graph.create_color("g_albedo", GB_ALBEDO_FMT, extent);
        let g_normal = graph.create_color("g_normal", GB_NORMAL_FMT, extent);
        let g_material = graph.create_color("g_material", GB_MATERIAL_FMT, extent);
        let g_position = graph.create_color("g_position", GB_POSITION_FMT, extent);
        let g_depth = graph.create_depth("g_depth", extent);
        // Shadow depth target. Legacy path: a single SHADOW_SIZE square map. CSM path: the
        // atlas texture (CSM_ATLAS square, tiled into per-cascade slots). Same resource id
        // feeds the lighting pass either way, so the sampling side re-wire is zero.
        let shadow_map = if self.csm.enabled {
            let s = self.csm.atlas_size;
            graph.create_depth("shadow_atlas", Extent2D::new(s, s))
        } else {
            graph.create_depth("shadow_map", Extent2D::new(SHADOW_SIZE, SHADOW_SIZE))
        };
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
        // then the optional compute-post blur. CSM path fills the atlas (one tile per
        // cascade via per-tile viewports); the legacy path fills the single directional map.
        if self.csm.enabled {
            self.deferred
                .record_shadow_atlas(&mut graph, shadow_map, &scene, &csm_slots);
        } else {
            self.deferred
                .record_shadow(&mut graph, shadow_map, &scene, light_vp);
        }
        // TAA-aware texture LOD bias for the G-buffer texture fetches (Stage 7 + Stage 8). Two terms:
        //   1. log2(internal/output): the DLSS/FSR2 resolution term — at a reduced internal
        //      resolution the screen-space derivatives are ~2x larger and fetch blurry mips, so this
        //      negative offset pulls them back to the full-res mip the upscaler reconstructs to.
        //   2. taa_mip_bias (~-1, Stage 8): the PRIMARY distant-sharpness lever. With sub-pixel
        //      jitter the temporal accumulation super-samples, so we can bias toward sharper mips and
        //      let TAA resolve the aliasing — this is 레퍼런스 엔진/DLSS's primary approach to distant-texture
        //      sharpness (레퍼런스 엔진 also uses hardware anisotropic filtering, but that's the grazing-surface
        //      lever — see Stage 9 / P_ANISO — not the distance one). Applies even at native (term 1
        //      = 0) under forced TAA. Only added with jitter (no jitter => no supersampling to hide it).
        // Gallery (TAA off => taau_active false) keeps bias 0 => SampleBias(.,0)==Sample() => byte
        // identical. Driver-independent LOD offset on the existing trilinear sampler (no DX≡VK risk).
        let mip_bias = if taau_active {
            let scale_bias = (cw as f32 / sw as f32).log2();
            let taa_bias = if taau_jitter_active {
                self.taa_mip_bias
            } else {
                0.0
            };
            scale_bias + taa_bias
        } else {
            0.0
        };
        // Depth pre-pass (pipeline rebaseline PR-1, opt-in `DEPTH_PREPASS=1`): render an opaque
        // depth-only pass into `g_depth` BEFORE the G-buffer so the base pass runs EQUAL-test +
        // write-off (Early-Z overdraw elimination) and the screen-space passes (AO/GI/SSR) sample
        // a completed depth whose producer is now explicitly the pre-pass (the render graph orders
        // pre-pass → G-buffer via the shared `g_depth` write, and the AO/GI/SSR reads of `g_depth`
        // chain after it). Off by default = no pre-pass pass at all (byte-identical golden anchor).
        if self.depth_prepass {
            self.deferred.record_prepass(
                &mut graph,
                g_depth,
                &scene,
                &self.ground_vbuf,
                &self.ground_ibuf,
                self.ground_count,
                view_proj,
                self.override_material,
                self.metallic_override,
                self.roughness_override,
                mip_bias,
            );
        }
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
            mip_bias,
            self.depth_prepass,
        );
        // Deferred surface-decal pass (decals A3): tint the G-buffer albedo for `kind == Decal`
        // drawables after the opaque fill, before lighting. No-op (no pass) when the scene has
        // no decals, so the gallery / non-decal scenes stay byte-identical.
        self.deferred
            .record_decals(&mut graph, gbuf, &scene, view_proj, mip_bias);
        // Velocity (motion-vector) channel (pipeline re-baseline PR-2, opt-in `P_VELOCITY=1`): a
        // separate opaque pass into an RG16Float target holding per-pixel screen motion. Uses the
        // UNJITTERED current + previous camera matrices (`view_proj_stable` / `prev_view_proj_taau`)
        // so no TAA jitter leaks into the motion; per-object prev pose from `prev_scene`. Consumed by
        // the velocity-aware TAAU reprojection + the DEBUG_VIEW=11 viz. Off = no target, no pass.
        let velocity_target = if self.velocity_on {
            let vt = graph.create_color("velocity", velocity::VELOCITY_FMT, extent);
            self.velocity.record(
                &mut graph,
                vt,
                g_depth,
                &scene,
                &prev_scene,
                &self.ground_vbuf,
                &self.ground_ibuf,
                self.ground_count,
                view_proj_stable,
                Mat4::from_cols_array(&self.prev_view_proj_taau),
            );
            Some(vt)
        } else {
            None
        };
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
            || self.surface_cache
            || self.wrc_viz)
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
        let cache_active =
            (self.cache_viz || self.surface_cache || self.reflect_cache || self.sc_viz)
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
        // The surface-cache VIEW uses a much SLOWER alpha: the multibounce gather is stochastic
        // (random rays per frame), so a fast EMA leaves per-frame noise that reads as flicker
        // ("new colours every frame"); a slow alpha temporally averages the gather over ~20 frames,
        // so the (static-camera) view converges to a stable image. `P_SC_VIZ_ALPHA` overrides.
        let relight_alpha = if self.sc_viz {
            std::env::var("P_SC_VIZ_ALPHA")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.05)
                .clamp(0.005, 1.0)
        } else if self.cache_feedback {
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
                        self.gdf_cone_k,
                        self.sky_gain,
                        self.sky_wb,
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
            (true, Some(vol), Some(ext)) => {
                // macOS/M3 perf: trace the AO at 1/div (Apple tier = half) + joint-bilateral upsample.
                // gdf_ao.slang samples the G-buffer by normalized UV, so a coarser extent + dims trace
                // correctly (no shader change); div=1 keeps the legacy full-res path byte-identical.
                let adiv = self.ao_res_div.max(1);
                let (aw, ah) = (cw.div_ceil(adiv), ch.div_ceil(adiv));
                let ao_extent = if adiv > 1 {
                    Extent2D::new(aw, ah)
                } else {
                    extent
                };
                let ao_traced = self.gi.record_ao(
                    &mut graph,
                    vol,
                    ext,
                    scene_aabb_min,
                    scene_aabb_max,
                    g_depth,
                    g_normal,
                    ao_extent,
                    inv_view_proj,
                    aw,
                    ah,
                    self.flip_y,
                    scene_clip,
                    &scene_clip_vols,
                );
                Some(if adiv > 1 {
                    self.gi.record_upsample(
                        &mut graph,
                        ao_traced,
                        g_depth,
                        g_normal,
                        extent,
                        inv_view_proj,
                        scene_aabb_min,
                        scene_aabb_max,
                        cw,
                        ch,
                        aw,
                        ah,
                        self.flip_y,
                    )
                } else {
                    ao_traced
                })
            }
            _ => None,
        };
        // Screen-space near-field AO (HBAO-lite), composed with the GDF AO in the lighting pass.
        // proj_scale = 0.5/tan(fovY/2) with the fixed 60° vertical FOV (perspective_rh above).
        let ssao_out = if self.ssao {
            self.gtao.record(
                &mut graph,
                g_depth,
                g_normal,
                extent,
                inv_view_proj,
                [eye.x, eye.y, eye.z],
                cw,
                ch,
                self.flip_y,
                self.ssao_params[0],
                self.ssao_params[1],
                self.ssao_params[2],
                0.5 / (30f32.to_radians().tan()),
                self.ssao_params[3],
            )
        } else {
            None
        };
        // Stage C3: 1-bounce diffuse GI added to the ambient term, optionally denoised (C4).
        // Indoor skylight-occlusion image (directional sky-visibility), produced on the volume GI
        // path and consumed by the lighting (occludes the IBL diffuse skylight). `None` otherwise.
        let mut gi_skyvis_out: Option<ResourceId> = None;
        let gdf_gi_out = match (self.gdf_gi, scene_gdf_vol, scene_gdf_ext) {
            (true, Some(vol), Some(ext)) if self.screen_probe => {
                // Screen-space radiance probes (P1+): per-tile probe trace into an octahedral
                // atlas + a per-pixel gather replace the GI consumption (world-volume sample /
                // ray march). Full-res output; denoise (temporal + à-trous) for stability like the
                // ray-march path. The gather also builds the indoor skylight occlusion (sky-vis)
                // from the probes' per-ray sky visibility, fed to lighting like the volume path.
                // P4: update the world radiance cache first, so escaped probe rays sample this
                // frame's cache (off-screen / far-field / infinite bounce). `None` when disabled.
                let wrc_arg = if self.wrc {
                    self.gi.record_wrc_update(
                        &mut graph,
                        vol,
                        ext,
                        scene_aabb_min,
                        scene_aabb_max,
                        self.sun_dir,
                        self.sun_intensity,
                        scene_clip,
                        &scene_clip_vols,
                        scene_albedo,
                        gi_cache_arg,
                        self.gi_max_steps,
                        self.gdf_cone_k,
                        0.1, // EMA alpha
                    )
                } else {
                    None
                };
                let (traced, sp_skyvis) = self.gi.record_screen_probe(
                    &mut graph,
                    vol,
                    ext,
                    scene_aabb_min,
                    scene_aabb_max,
                    g_depth,
                    g_normal,
                    extent,
                    inv_view_proj,
                    self.sun_dir,
                    self.sun_intensity,
                    cw,
                    ch,
                    self.flip_y,
                    self.frame_no as u32,
                    scene_albedo,
                    gi_cache_arg,
                    firefly_max,
                    scene_clip,
                    &scene_clip_vols,
                    self.gi_max_steps,
                    self.gdf_cone_k,
                    // P2 spatial cross-probe filter half-kernel (1 = 3x3; `P_SP_FILTER=0` disables).
                    std::env::var("P_SP_FILTER")
                        .ok()
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(1)
                        .min(4),
                    wrc_arg,
                );
                gi_skyvis_out = Some(sp_skyvis);
                let out = if gi_denoise_active {
                    self.gi.record_denoise(
                        &mut graph,
                        traced,
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
                        self.gi_temporal_clamp,
                        self.gi_atrous_steps,
                    )
                } else {
                    traced
                };
                Some(out)
            }
            (true, Some(vol), Some(ext)) => {
                // Stage D1: trace at half res (1/4 the rays) when enabled, then joint-bilateral
                // upsample to full res before the denoiser. gdf_gi.slang samples the G-buffer by
                // normalized UV, so a half extent + half dims trace correctly with no shader change.
                // P1: trace at 1/div of the render extent (2 = legacy half, 4 = quarter probes),
                // then the bilateral upsample (gdf_gi_upsample.slang, generic over the source dims)
                // reconstructs full res. Content-only; the gallery is full-res (byte-identical).
                let half_gi = self.gi_half_res;
                let div = if half_gi { self.gi_res_div.max(1) } else { 1 };
                let (gw, gh) = (cw.div_ceil(div), ch.div_ceil(div));
                let gi_extent = Extent2D::new(gw, gh);
                // 레퍼런스 엔진 GI-fidelity: update + bind the world irradiance volume (DDGI-lite). The update
                // propagates last frame's volume into hits (multibounce), so deep interiors fill in
                // over frames. When bound, the GI pass samples the volume instead of marching.
                let gi_volume_arg = if self.gi_volume {
                    self.gi
                        .record_gi_volume(
                            &mut graph,
                            vol,
                            ext,
                            scene_aabb_min,
                            scene_aabb_max,
                            self.sun_dir,
                            self.sun_intensity,
                            scene_clip,
                            &scene_clip_vols,
                            scene_albedo,
                            crate::GROUND_ALBEDO,
                            self.frame_no as u32,
                            self.gi_spp,
                            0.1,
                        )
                        .zip(self.gi.gi_volume_sampled())
                        .map(|(vext, (rad_base, skyvis_base))| (rad_base, skyvis_base, vext))
                } else {
                    None
                };
                let (traced, skyvis) = self.gi.record_gi(
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
                    self.gdf_cone_k,
                    gi_volume_arg,
                    // F3: HW-RT gather only on the ray-march path (the volume path samples the field,
                    // not rays). Default off (`P_HWRT_GI` unset) -> SW march -> gallery byte-identical.
                    self.hwrt_gi && gi_volume_arg.is_none(),
                );
                gi_skyvis_out = skyvis;
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
                        self.gi_temporal_clamp,
                        self.gi_atrous_steps,
                    )
                } else {
                    raw
                };
                Some(out)
            }
            _ => None,
        };
        // GI-on-distance-field visualization: update the world radiance cache, then march the
        // camera into the GDF and paint each hit with the cache's stored indirect irradiance.
        // Replaces the tonemap source (added to the `tonemap_src` chain below). `P_WRC_VIZ_MODE`
        // 0 = irradiance grayscale, 1 = irradiance × clay; `P_WRC_VIZ_GAIN` lifts the dim indirect.
        let wrc_view_out = match (self.wrc_viz, scene_gdf_vol, scene_gdf_ext) {
            (true, Some(vol), Some(ext)) => self
                .gi
                .record_wrc_update(
                    &mut graph,
                    vol,
                    ext,
                    scene_aabb_min,
                    scene_aabb_max,
                    self.sun_dir,
                    self.sun_intensity,
                    scene_clip,
                    &scene_clip_vols,
                    scene_albedo,
                    gi_cache_arg,
                    self.gi_max_steps,
                    self.gdf_cone_k,
                    0.1, // EMA alpha
                )
                .map(|(wrc_atlas, wrc_ext)| {
                    let mode = std::env::var("P_WRC_VIZ_MODE")
                        .ok()
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(1);
                    let gain = std::env::var("P_WRC_VIZ_GAIN")
                        .ok()
                        .and_then(|v| v.parse::<f32>().ok())
                        .unwrap_or(1.0);
                    self.gi.record_wrc_view(
                        &mut graph,
                        vol,
                        extent,
                        eye.into(),
                        inv_view_proj,
                        scene_aabb_min,
                        scene_aabb_max,
                        cw,
                        ch,
                        self.flip_y,
                        scene_clip,
                        &scene_clip_vols,
                        wrc_atlas,
                        wrc_ext,
                        mode,
                        gain,
                        // Surface-cache source (P_SC_VIZ): shade from the high-res mesh cards' final
                        // lit radiance where a card covers the hit; else fall back to the world cache.
                        if self.sc_viz { 1 } else { 0 },
                        if self.sc_viz {
                            cache_read
                                .zip(scene_cache_lit_ext)
                                .map(|((c, p, r, n, t), ext)| ([c, p, r, n, t], ext))
                        } else {
                            None
                        },
                    )
                }),
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
                // Stage D3 / M3-C: trace the reflection at 1/div res + bilateral upsample to full res
                // before the temporal resolve. gdf_reflect.slang samples the G-buffer by normalized UV,
                // so any coarser extent + dims trace correctly (no shader change). `div = 2` reproduces
                // the legacy half-res exactly (`cw.div_ceil(2)` == the old `hcw/hch`); the Apple tier
                // uses `div = 4` (quarter-res) — the measured single lever on gdf_reflect. Gallery keeps
                // `reflect_half_res` off ⇒ div collapses to 1 (full-res, byte-identical anchor).
                let refl_half = self.reflect_half_res;
                let rdiv = if refl_half {
                    self.reflect_res_div.max(1)
                } else {
                    1
                };
                let (rw, rh) = (cw.div_ceil(rdiv), ch.div_ceil(rdiv));
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
                    // Content: fixed frame (0) → temporally stable GGX jitter (no reflection
                    // sparkle). Gallery: real frame → byte-identical legacy anchor.
                    if self.is_gallery {
                        self.frame_no as u32
                    } else {
                        0
                    },
                    scene_albedo,
                    reflect_cache_arg,
                    scene_clip,
                    &scene_clip_vols,
                    self.reflect_max_steps,
                    self.gdf_cone_k,
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
                // C8j: temporally resolve the stochastic GGX GDF reflection (레퍼런스식; the rough
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
                    self.reflect_history_clamp,
                    self.reflect_clamp_gamma,
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
        // Clustered light culling (PR-6): build this frame's froxel light lists, then hand the
        // buffer ids to the lighting pass. The build pass writes the grid/index externals, so the
        // lighting pass (which reads them) sequences after it. `None` = brute-force point loop.
        let cluster_lighting = match (cluster_indices, self.cluster.as_ref()) {
            (Some((grid_idx, index_idx, light_idx, light_count)), Some(cluster_sys))
                if self.clustered_lights =>
            {
                let (grid_ext, index_ext) = ClusterSystem::import(&mut graph);
                let view_z_row = [view.x_axis.z, view.y_axis.z, view.z_axis.z, view.w_axis.z];
                cluster_sys.record_build(
                    &mut graph,
                    grid_ext,
                    index_ext,
                    fif,
                    light_count,
                    view_z_row,
                    inv_view_proj,
                    [eye.x, eye.y, eye.z],
                    CLUSTER_Z_NEAR,
                    CLUSTER_Z_FAR,
                    cw,
                    ch,
                );
                Some((
                    grid_ext,
                    index_ext,
                    grid_idx,
                    index_idx,
                    light_idx,
                    light_count,
                ))
            }
            // Brute-force A/B: upload the light buffer but pass index_buf = MAX (loop all lights,
            // no froxel list). Import a dummy external so the tuple shape holds; no build pass.
            (Some((grid_idx, _, light_idx, light_count)), Some(_)) if self.clustered_brute => {
                let (grid_ext, index_ext) = ClusterSystem::import(&mut graph);
                Some((
                    grid_ext,
                    index_ext,
                    grid_idx,
                    u32::MAX,
                    light_idx,
                    light_count,
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
            ssao_out,
            gdf_gi_out,
            swrt_reflect_out,
            gi_skyvis_out,
            self.skyvis_tint,
            self.skyvis_min_occ,
            globals_offset,
            self.flip_y,
            // Two-sided shading for imported scenes (single-sided walls seen from inside);
            // the gallery stays single-sided so its baseline is byte-identical.
            !self.is_gallery,
            // Auto-exposure: when on, the lighting reads the adapted exposure from the metering
            // buffer; off → sentinel (use the static globals.ambient.a, byte-identical anchor).
            if self.auto_exposure {
                self.deferred.exposure_buf_index().unwrap_or(u32::MAX)
            } else {
                u32::MAX
            },
            cluster_lighting,
        );
        // Auto-exposure metering: read this frame's lit HDR, adapt the exposure for next frame.
        // After lighting (the `hdr` read orders it). `adapt` = 1-exp(-dt·speed) (eye/iris speed).
        if self.auto_exposure {
            let speed = 2.5f32;
            let adapt = 1.0 - (-dt * speed).exp();
            self.deferred
                .record_auto_exposure(&mut graph, hdr, cw, ch, 0.12, adapt, 1.0e-6, 4.0);
        }
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
                if self.auto_exposure {
                    self.deferred.exposure_buf_index().unwrap_or(u32::MAX)
                } else {
                    u32::MAX
                },
            );
        }
        if let Some(hdr_post) = hdr_post {
            self.deferred
                .record_compute_post(&mut graph, hdr, hdr_post, cw, ch);
        }

        // PR-4 (render-pipeline re-baseline track, `docs/render-pipeline-reference.md` §3):
        // the sky/atmosphere composite slot. Sits right after the opaque scene color is
        // final (lighting + reflections + the optional compute-post blur all done) and
        // before TAAU/tonemap — matching the reference pipeline's "opaque complete -> sky/
        // fog -> translucency -> post" ordering (§1.6). The slot is unconditional in the
        // graph's *shape*; only the pass itself is opt-in (`P_HEIGHT_FOG=1`), so leaving it
        // off costs nothing and the gallery/regression anchors stay byte-identical.
        let fog_src = hdr_post.unwrap_or(hdr);
        let fog_out = if self.height_fog {
            let hdr_fog = graph.create_color("hdr_fog", HDR_FORMAT, extent);
            self.atmosphere.record_fog(
                &mut graph,
                fog_src,
                hdr_fog,
                g_position,
                [eye.x, eye.y, eye.z],
                self.fog_density,
                self.fog_height_falloff,
                sun_dir,
                sun_intensity,
                self.sky_wb,
                self.fog_inscatter_gain,
                // `procedural_sky` returns raw (unexposed) radiance; `hdr` is already exposed
                // (baked in by `record_lighting`), so the inscatter needs the same treatment.
                // Uses the static EV100 exposure (not the auto-exposure adapted value) — a
                // documented simplification; the fog is a slow-varying ambient term so a
                // one-frame-stale exposure has no visible impact.
                self.exposure,
            );
            Some(hdr_fog)
        } else {
            None
        };

        // PR-3 (render-pipeline re-baseline track, `docs/render-pipeline-reference.md` §1.7
        // #12 / §3): the forward translucency slot. Draws sorted alpha-blended translucent
        // geometry over the finished opaque+fog HDR (blend in place), depth-testing against the
        // opaque `g_depth` (occluded behind solid geometry) with depth-write off (overlapping
        // panes all blend). Sits AFTER fog and BEFORE TAAU/tonemap — reference ordering. Adds
        // no pass when `translucents` is empty (zero cost, byte-identical anchor).
        let translucency_target = fog_out.unwrap_or(fog_src);
        self.translucency.record(
            &mut graph,
            translucency_target,
            g_depth,
            &self.translucents,
            view_proj,
            eye,
            self.deferred.globals_buffer(),
            globals_offset,
            self.flip_y,
            shadow_map,
        );

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
        // HZB occlusion culling (PR-8): when enabled, the frustum-only `record_cull` is
        // replaced by reset + an occlusion-aware cull that also tests each instance's
        // screen AABB against LAST frame's Hi-Z pyramid, and a build pass regenerates
        // the pyramid from THIS frame's scene depth (for next frame). The build declares
        // the HZB external as a WRITE and the cull as a READ, so the graph's WAR edge
        // runs cull (last frame's data) before build (overwrite) — the canonical
        // prev-frame-HZB scheme (conservative; no reprojection). Off => the original
        // frustum-only path (byte-identical).
        let hzb_active = self.hzb_cull && self.hzb.is_some();
        if let Some((args_ext, visible_ext)) = cull_res {
            if hzb_active {
                let hzb = self.hzb.as_ref().unwrap();
                let hzb_ext = HzbSystem::import(&mut graph);
                self.cull.record_reset(
                    &mut graph,
                    args_ext,
                    visible_ext,
                    frustum_planes(cull_view_proj),
                    &grid,
                );
                let (args_buf, visible_buf) = self.cull.buffers();
                // Occlusion test disabled on the first frame (no pyramid yet), and only
                // when the pyramid's bindless slots are consecutive (the shader indexes
                // hzb_base + mip). Either way the frustum cull still runs.
                let occlude = self.frame_no >= 1 && hzb.slots_are_consecutive();
                hzb.record_cull(
                    &mut graph,
                    args_buf,
                    visible_buf,
                    args_ext,
                    visible_ext,
                    hzb_ext,
                    frustum_planes(cull_view_proj),
                    cull_view_proj.to_cols_array(),
                    &grid,
                    self.cull.index_count(),
                    occlude,
                );
                hzb.record_build(&mut graph, g_depth, hzb_ext, extent);
            } else {
                self.cull.record_cull(
                    &mut graph,
                    args_ext,
                    visible_ext,
                    frustum_planes(cull_view_proj),
                    &grid,
                );
            }
        }

        // PR-3 side-effect: the Phase-7 particle + GPU-culling draws move into the HDR
        // translucency slot (alpha-blend over `translucency_target`, BEFORE tonemap), instead
        // of drawing over the tonemapped LDR backbuffer. Both are default-off demo features
        // (`P7_PARTICLES` / `P7_CULL`), so the default gallery/anchor output is unchanged; this
        // only fixes the HDR-composite ordering the reference pipeline expects (§2.1-3, #21).
        // Declared after the sim/cull compute passes so the WAR/WAW deps order them correctly;
        // the graph schedules by resource dependency, not declaration order.
        if let Some((args_ext, visible_ext)) = cull_res {
            self.cull.record_draw(
                &mut graph,
                translucency_target,
                extent,
                args_ext,
                visible_ext,
                view_proj.to_cols_array(),
                self.sun_dir,
                &grid,
                g_depth,
                extent,
            );
        }
        if let Some(particles_ext) = particles_ext {
            self.particles.record_draw(
                &mut graph,
                translucency_target,
                particles_ext,
                view_proj.to_cols_array(),
                cam_right,
                cam_up,
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
                    self.sky_gain,
                    self.sky_wb,
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
                // Content: fixed frame (0) → stable GGX jitter; gallery: real frame (anchor).
                if self.is_gallery {
                    self.frame_no as u32
                } else {
                    0
                },
                scene_albedo,
                reflect_cache_arg,
                scene_clip,
                &scene_clip_vols,
                self.reflect_max_steps,
                self.gdf_cone_k,
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
                    // Content: fixed frame (0) → temporally stable GGX jitter (no reflection
                    // sparkle). Gallery: real frame → byte-identical legacy anchor.
                    if self.is_gallery {
                        self.frame_no as u32
                    } else {
                        0
                    },
                    scene_albedo,
                    reflect_cache_arg,
                    scene_clip,
                    &scene_clip_vols,
                    self.reflect_max_steps,
                    self.gdf_cone_k,
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
                    if self.auto_exposure {
                        self.deferred.exposure_buf_index().unwrap_or(u32::MAX)
                    } else {
                        u32::MAX
                    },
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
            let main_lit = fog_out.unwrap_or(fog_src);
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
                velocity_target,
            ))
        } else {
            None
        };
        // DEBUG_VIEW=11: colour-code the velocity target (needs `P_VELOCITY=1`). Takes precedence
        // over the lit/TAAU output so the motion vectors are visualized directly.
        let velocity_view_out = if self.debug_view == 11 {
            velocity_target.map(|vt| {
                self.velocity
                    .record_viz(&mut graph, vt, extent, cw, ch, VELOCITY_VIZ_SCALE)
            })
        } else {
            None
        };
        let tonemap_src = velocity_view_out
            .or(rt_out)
            .or(wrc_view_out)
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
            .or(fog_out)
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
        // Profiler is inline-only: the threaded path's query-heap readback would need
        // to cross the RHI-thread fence, so it's disabled under P15_RHI_THREAD.
        let mut profiler = self
            .profiler_on
            .then(|| GraphProfiler::new(&self.query_heaps[fif]));
        let threaded = cmd.is_none();
        // Inline-path readback buffer (None when threaded — the worker owns its own).
        let mut readback: Option<(Buffer, ReadbackLayout)> = None;
        if let Some(cmd) = cmd {
            graph.execute(
                &self.device,
                &mut self.pools[fif],
                cmd,
                self.swapchain.as_ref().expect("inline swapchain"),
                image_index,
                self.aliasing,
                profiler.as_mut(),
            )?;
            // Remember this slot's scheduled pass names so the next readback (after
            // this frame's fence) can pair them with the timestamp boundaries.
            self.slot_pass_names[fif] = match &profiler {
                Some(p) => p.names.clone(),
                None => Vec::new(),
            };

            // For a screenshot, copy the just-rendered backbuffer into a readback
            // buffer in the same command buffer (before it ends).
            if capture_this_frame.is_some() {
                let layout = self
                    .device
                    .swapchain_readback_layout(self.swapchain.as_ref().expect("inline swapchain"));
                let buf = self.device.create_buffer(&BufferDesc {
                    size: layout.size,
                    usage: BufferUsage::Readback,
                })?;
                cmd.copy_swapchain_to_buffer(
                    self.swapchain.as_ref().expect("inline swapchain"),
                    image_index,
                    &buf,
                );
                readback = Some((buf, layout));
            }

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
                self.queue().submit_async(
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
                    self.gdf_cone_k,
                    self.sky_gain,
                    self.sky_wb,
                );
                ccmd.end()?;
                let cur = (self.frame_no % 2) as usize;
                if self.frame_no == 0 {
                    // No previous relight to wait on; graphics submits normally, the relight still
                    // signals so frame 1's wait pairs up.
                    self.queue().submit(
                        cmd,
                        &self.image_available[fif],
                        signal,
                        &self.in_flight[fif],
                    )?;
                } else {
                    let prev = ((self.frame_no + 1) % 2) as usize; // (N-1) mod 2
                    self.queue().submit_async(
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
                self.queue().submit(
                    cmd,
                    &self.image_available[fif],
                    signal,
                    &self.in_flight[fif],
                )?;
            }
        } else {
            // Threaded: append the graph onto the frame IR (after any IBL capture) and
            // ship it. The RHI thread acquires + translates + submits + presents, and
            // copies/saves the capture itself, overlapping this thread's next frame.
            // M4 B4: optionally record the graph's passes in parallel on the job
            // workers (each builds its own IR bucket, concatenated in schedule order).
            let jobs = self.parallel_record.then(dreamcoast_jobs::global);
            graph.record_into(
                &frame_list,
                &self.device,
                &mut self.pools[fif],
                self.aliasing,
                None,
                jobs,
            )?;
            self.slot_pass_names[fif] = Vec::new();
            let capture = capture_this_frame.as_ref().map(|c| rhi_thread::CaptureReq {
                path: c.path.clone(),
                include_ui: c.include_ui,
            });
            self.rhi_thread
                .as_ref()
                .expect("rhi thread")
                .submit(frame_list, fif, capture);
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
        // P4: advance the world radiance cache ping-pong so next frame reads this frame's write.
        if self.wrc || self.wrc_viz {
            self.gi.advance_wrc();
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
        // 레퍼런스 엔진 GI-fidelity: advance the world irradiance volume ping-pong (next frame reads this).
        if self.gi_volume {
            self.gi.advance_gi_volume();
        }
        // QHD/UHD TAAU: advance the history ping-pong (next frame reprojects this frame's).
        if taau_active {
            self.taau.advance();
        }
        self.prev_view_proj = view_proj.to_cols_array();
        self.prev_view_proj_taau = view_proj_stable.to_cols_array();
        // Velocity (PR-2): stash this frame's per-object world transforms as next frame's prev pose
        // (single source; stable draw-list order). Only when velocity is on (else no cost / no state
        // churn). Uses the pre-skin transform for static/Spin draws; skinned draws carry identity
        // here and their motion comes from the palette history instead.
        if self.velocity_on {
            self.prev_transforms.clear();
            self.prev_transforms
                .extend(scene.iter().map(|o| o.transform));
        }

        // Inline present + capture readback. Threaded: the RHI thread did both (its
        // capture readback waits the same frame fence → byte-identical + deterministic).
        if !threaded {
            // Wait for the GPU (copy included), read the buffer back, and save a PNG.
            if let (Some(cap), Some((buf, layout))) =
                (capture_this_frame.as_ref(), readback.as_ref())
            {
                self.in_flight[fif].wait()?;
                let mut bytes = vec![0u8; layout.size as usize];
                buf.read_into(&mut bytes)?;
                save_screenshot(&cap.path, &bytes, layout)?;
                info!(
                    "saved screenshot {} ({}x{}, ui={})",
                    cap.path, layout.width, layout.height, cap.include_ui
                );
            }

            LAST_CPU_US.store(
                _t_cpu.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            let signal = &self.render_finished[image_index as usize];
            if self.queue().present(
                self.swapchain.as_ref().expect("inline swapchain"),
                image_index,
                signal,
            )? {
                self.needs_recreate = true;
            }
        }
        self.fif = (self.fif + 1) % FRAMES_IN_FLIGHT;
        self.frame_no += 1;

        // HZB cull stats (PR-8): read back the (survived, occlusion-culled) counters and
        // log them periodically (and on the last screenshot frame). Frames-in-flight give
        // this a small latency; it is a diagnostic, so a recent frame's numbers are fine.
        if self.hzb_cull
            && let Some(hzb) = self.hzb.as_ref()
        {
            let (survived, culled) = hzb.read_stats();
            self.hzb_stats.set((survived, culled));
            // Log periodically, and every frame once past the warmup in screenshot mode
            // (so a capture run always prints the final numbers).
            if self.frame_no.is_multiple_of(60) || (self.screenshot_mode && self.frame_no >= warmup)
            {
                println!(
                    "[hzb] instances: {} total, {} survived, {} occlusion-culled",
                    GRID_COUNT, survived, culled
                );
            }
        }

        // Bridge the D3D12 debug layer into the log (it otherwise only reaches OutputDebugString).
        // Catches validation/threading violations — e.g. the Phase 15 M4 B3 RHI submit thread's
        // cross-thread queue submit / present. No-op on Vulkan (already bridged) / Metal.
        self.device.drain_debug_messages();

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
/// Deterministic stress-test point lights (PR-6 `TEST_LIGHTS=N`): a fixed cubic-ish grid of
/// `n` lights filling the scene's bounding volume, with a fixed rotating palette and a finite
/// influence radius so per-cluster culling actually excludes most of them per cluster. No time
/// dependence — the layout is a pure function of (n, center, radius), so runs are reproducible
/// and A/B comparable (brute-force vs clustered) at identical light sets.
fn test_light_grid(n: u32, center: Vec3, radius: f32) -> Vec<ClusterLight> {
    // Cube-root grid dimensions (as close to equal as possible), filling [-1,1]^3 of the scene
    // bounds scaled up a bit so lights sit among the geometry.
    let side = (n as f32).cbrt().ceil() as u32;
    let extent = radius * 1.4;
    // Per-light influence radius: a couple of grid cells so neighbours overlap but distant
    // clusters cull the light. Independent of n's exact value (deterministic).
    let cell = (2.0 * extent) / side.max(1) as f32;
    let light_radius = cell * 1.5;
    // Fixed candela so N lights of this radius stay visible but bounded.
    let intensity = radius * radius * 2.0;
    let palette = [
        [1.0, 0.4, 0.3],
        [0.3, 0.6, 1.0],
        [0.5, 1.0, 0.4],
        [1.0, 0.9, 0.4],
        [0.8, 0.4, 1.0],
    ];
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let gx = i % side;
        let gy = (i / side) % side;
        let gz = i / (side * side);
        let f = |g: u32| -> f32 {
            if side <= 1 {
                0.0
            } else {
                (g as f32 / (side - 1) as f32) * 2.0 - 1.0
            }
        };
        out.push(ClusterLight {
            position: [
                center.x + f(gx) * extent,
                center.y + f(gy) * extent,
                center.z + f(gz) * extent,
            ],
            radius: light_radius,
            color: palette[(i as usize) % palette.len()],
            intensity,
        });
    }
    out
}

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

/// Physical-camera exposure multiplier from an EV100 stop. `EV100 = log2(N²/t)` at ISO 100;
/// the linear exposure that maps scene luminance to [0,1] before the filmic curve is
/// `1 / (1.2 · 2^EV100)` (the 1.2 is the standard ISO 100 saturation-based constant, q·S/K).
/// Higher EV100 = brighter scene / darker image (shorter exposure); sunny-16 ≈ EV15.
fn ev100_to_exposure(ev100: f32) -> f32 {
    1.0 / (1.2 * 2f32.powf(ev100))
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
