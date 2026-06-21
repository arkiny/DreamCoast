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

use anyhow::anyhow;
use dreamcoast_asset::{ImageData, MeshData, MeshVertex};
use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_core::init_logging;
use dreamcoast_gui::{Gui, imgui};
use dreamcoast_platform::Window;
use dreamcoast_render::{ComputePassInfo, PassInfo, RenderGraph, ResourcePool};
use rhi::{
    BackendKind, BlendMode, Buffer, BufferDesc, BufferUsage, ClearColor, CommandBuffer,
    ComputePipelineDesc, Cubemap, CubemapDesc, Device, Extent2D, Fence, Format, GraphicsPipeline,
    GraphicsPipelineDesc, Instance, InstanceDesc, PresentMode, PrimitiveTopology, Queue,
    ReadbackLayout, Semaphore, StorageBufferDesc, SwapchainDesc, Texture, TextureDesc,
    VertexLayout,
};
use tracing::info;

const FRAMES_IN_FLIGHT: usize = 2;
const COLOR_FORMAT: Format = Format::Bgra8Srgb; // swapchain / backbuffer
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
struct SceneObject {
    vbuf: Buffer,
    ibuf: Buffer,
    index_count: u32,
    transform: Mat4,
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    /// base color, metallic-roughness, normal, emissive bindless indices
    /// (`NO_TEXTURE` if absent).
    tex: [u32; 4],
    casts_shadow: bool,
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
    init_logging();

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

    let instance = Instance::new(
        backend,
        &window,
        &InstanceDesc {
            app_name: "dreamcoast-sandbox".into(),
            validation: true,
        },
    )?;
    let device = instance.create_device()?;
    let queue = device.queue();

    let mut swapchain = device.create_swapchain(&swapchain_desc(Extent2D::new(w, h)))?;

    // G-buffer fill pipeline: mesh -> 4 MRT (+ depth).
    let (gb_vs, gb_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::gbuffer_vs_spirv,
        dreamcoast_shader::gbuffer_fs_spirv,
        dreamcoast_shader::gbuffer_vs_dxil,
        dreamcoast_shader::gbuffer_fs_dxil,
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

    // Tonemap pipeline: HDR -> backbuffer (sRGB).
    let (post_vs, post_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::post_vs_spirv,
        dreamcoast_shader::post_fs_spirv,
        dreamcoast_shader::post_vs_dxil,
        dreamcoast_shader::post_fs_dxil,
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
        "post_compute",
    )?;
    let post_compute_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
        compute_bytes: post_compute_cs,
        compute_entry: "csMain",
        push_constant_size: 16, // hdr_index + out_index + width + height
        bindless: true,
    })?;

    // GPU particle simulation (Phase 7): a persistent storage buffer the sim
    // compute pass updates in-place each frame and the draw pass reads in its
    // vertex stage. Seeded once via a one-shot compute dispatch.
    let particle_sim_cs = load_compute_shader(
        backend,
        dreamcoast_shader::particle_sim_cs_spirv,
        dreamcoast_shader::particle_sim_cs_dxil,
        "particle_sim",
    )?;
    let particle_sim_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
        compute_bytes: particle_sim_cs,
        compute_entry: "csMain",
        push_constant_size: 20, // buffer_index + count + dt + time + init
        bindless: true,
    })?;
    let (pd_vs, pd_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::particle_draw_vs_spirv,
        dreamcoast_shader::particle_draw_fs_spirv,
        dreamcoast_shader::particle_draw_vs_dxil,
        dreamcoast_shader::particle_draw_fs_dxil,
        "particle_draw",
    )?;
    let particle_draw_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_bytes: pd_vs,
        fragment_bytes: pd_fs,
        vertex_entry: "vsMain",
        fragment_entry: "fsMain",
        color_formats: &[swapchain.format()],
        topology: PrimitiveTopology::TriangleList,
        vertex_layout: VertexLayout::None, // vertex-pull from the storage buffer
        blend: BlendMode::AlphaBlend,
        push_constant_size: 112, // view_proj + cam_right + cam_up + buffer/count/size/pad
        bindless: true,
        uniform_buffer: false,
        depth_test: false,
        depth_format: None,
    })?;
    let particle_buffer = device.create_storage_buffer(&StorageBufferDesc {
        size: (PARTICLE_COUNT * 32) as u64,
        stride: 32,
        indirect: false,
    })?;
    // Seed the particle buffer once (init dispatch).
    {
        let init_cmd = device.create_command_buffer()?;
        init_cmd.begin()?;
        init_cmd.bind_compute_pipeline(&particle_sim_pipeline);
        init_cmd.push_constants_compute(&particle_sim_push(
            particle_buffer.storage_index(),
            PARTICLE_COUNT as u32,
            0.0,
            0.0,
            1,
        ));
        init_cmd.dispatch((PARTICLE_COUNT as u32).div_ceil(64), 1, 1);
        init_cmd.end()?;
        let fence = device.create_fence(false)?;
        device.queue().submit_oneshot(&init_cmd, &fence)?;
        fence.wait()?;
    }

    // Sky pipeline: renders the procedural sky into each environment cube face.
    let (sky_vs, sky_fs) = load_shader_pair(
        backend,
        dreamcoast_shader::sky_vs_spirv,
        dreamcoast_shader::sky_fs_spirv,
        dreamcoast_shader::sky_vs_dxil,
        dreamcoast_shader::sky_fs_dxil,
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

    // Clip-space Y orientation for the full-screen passes (Vulkan = 1, D3D12 = 0).
    let flip_y: u32 = match backend {
        BackendKind::Vulkan => 1,
        BackendKind::D3d12 => 0,
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
        transform: Mat4::IDENTITY,
        base_color: model.material.base_color_factor,
        metallic: model.material.metallic_factor,
        roughness: model.material.roughness_factor,
        tex: [base_index, mr_index, normal_index, NO_TEXTURE],
        casts_shadow: true,
    });
    // Polished chrome sphere.
    let (sv, si, sc) = upload_mesh(&device, &sphere)?;
    scene.push(SceneObject {
        vbuf: sv,
        ibuf: si,
        index_count: sc,
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
    let mut exposure = 1.0f32;
    let mut point_lights_on = true;
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
    let mut compute_post = std::env::var_os("P7_COMPUTE_POST").is_some();
    // Phase 7: GPU particle simulation (compute-updated buffer, instanced draw).
    let mut particles_on = std::env::var_os("P7_PARTICLES").is_some();
    // Phase 7: GPU frustum culling -> indirect draw of a cube instance grid.
    let mut gpu_cull = std::env::var_os("P7_CULL").is_some();
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
    for _ in 0..FRAMES_IN_FLIGHT {
        command_buffers.push(device.create_command_buffer()?);
        image_available.push(device.create_semaphore()?);
        in_flight.push(device.create_fence(true)?);
    }
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

    while !window.should_close() {
        window.pump_events();
        if window.take_resized() {
            needs_recreate = true;
        }
        let (cw, ch) = window.size();
        if cw == 0 || ch == 0 {
            std::thread::sleep(std::time::Duration::from_millis(16));
            continue;
        }
        if needs_recreate {
            device.wait_idle()?;
            swapchain.recreate(&swapchain_desc(Extent2D::new(cw, ch)))?;
            for p in &mut pools {
                p.clear(); // transient extents changed; drop cached targets
            }
            render_finished = build_render_finished(&device, swapchain.image_count())?;
            needs_recreate = false;
        }

        let now = Instant::now();
        let dt = (now - last).as_secs_f32();
        last = now;
        elapsed += dt;
        // Clamp the sim step so a long stall (e.g. resize) can't explode particles.
        let sim_dt = dt.clamp(0.0, 1.0 / 30.0);
        if !screenshot_mode {
            angle += dt * 0.6; // hold a fixed view when capturing
        }

        // Decide whether this frame produces a screenshot: a scheduled capture in
        // screenshot mode (after warmup), or an F2 rising edge interactively.
        let f2 = window.input().key_down(VK_F2);
        let f2_pressed = f2 && !f2_prev;
        f2_prev = f2;
        let capture_this_frame: Option<Capture> = if screenshot_mode {
            frame_no
                .checked_sub(SCREENSHOT_WARMUP)
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
        let mut proj = Mat4::perspective_rh(60f32.to_radians(), cw as f32 / ch as f32, 0.05, 100.0);
        if backend == BackendKind::Vulkan {
            proj.y_axis.y *= -1.0; // Vulkan clip-space Y points down
        }
        let view_proj = proj * view;
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
                    ui.separator();
                    ui.combo_simple_string("Debug view", &mut debug_view, &DEBUG_VIEWS);
                    ui.input_float3("Sun dir", &mut sun_dir).build();
                    ui.slider("Sun intensity", 0.0, 10.0, &mut sun_intensity);
                    ui.slider("Ambient", 0.0, 0.5, &mut ambient);
                    ui.slider("Exposure", 0.1, 4.0, &mut exposure);
                    ui.checkbox("Point lights", &mut point_lights_on);
                    ui.checkbox("Shadows", &mut shadows_on);
                    ui.slider("Shadow bias", 0.0, 0.01, &mut shadow_bias);
                    ui.separator();
                    ui.checkbox("Override material", &mut override_material);
                    ui.slider("Metallic", 0.0, 1.0, &mut metallic_override);
                    ui.slider("Roughness", 0.0, 1.0, &mut roughness_override);
                    ui.separator();
                    ui.checkbox("Real-time env capture", &mut realtime_env);
                    ui.checkbox("Multi-bounce reflections", &mut multibounce);
                    ui.combo_simple_string("Post effect", &mut post_mode, &POST_EFFECTS);
                    ui.checkbox("Transient aliasing", &mut aliasing);
                    ui.separator();
                    ui.text("Compute / GPGPU (Phase 7)");
                    ui.checkbox("Compute post (blur)", &mut compute_post);
                    ui.checkbox("GPU particles", &mut particles_on);
                    ui.checkbox("GPU culling (indirect)", &mut gpu_cull);
                });
        }

        let fence = &in_flight[frame];
        fence.wait()?;
        let image_index = match swapchain.acquire_next_image(&image_available[frame])? {
            Some(i) => i,
            None => {
                needs_recreate = true;
                continue;
            }
        };
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
                eye,
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
        };
        let globals_offset = frame as u64 * GLOBALS_SLICE;
        globals_buffer.write_at(globals_offset, globals_bytes(&globals))?;

        // Build the deferred render graph:
        //   gbuffer (4 MRT + depth) -> lighting (HDR) -> tonemap (backbuffer) -> ui
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
        if let Some(particles_ext) = particles_ext {
            let sim = &particle_sim_pipeline;
            let buf = &particle_buffer;
            graph.add_compute_pass(
                ComputePassInfo {
                    name: "particle_sim",
                    storage_writes: vec![particles_ext],
                    reads: vec![],
                },
                move |ctx| {
                    let cmd = ctx.cmd();
                    cmd.bind_compute_pipeline(sim);
                    cmd.push_constants_compute(&particle_sim_push(
                        buf.storage_index(),
                        PARTICLE_COUNT as u32,
                        sim_dt,
                        elapsed,
                        0,
                    ));
                    cmd.dispatch((PARTICLE_COUNT as u32).div_ceil(64), 1, 1);
                    // Order the write before the draw pass's vertex-stage read.
                    cmd.storage_buffer_barrier(buf);
                    Ok(())
                },
            );
        }

        // Tonemap samples the compute-post output when enabled, else raw HDR.
        let tonemap_src = hdr_post.unwrap_or(hdr);
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
                cmd.push_constants(&post_push(hdr_index, post_mode as u32, flip_y));
                cmd.draw(3, 1);
                Ok(())
            },
        );
        // Phase 7 particle draw: instanced billboards composited over the tonemapped
        // image (alpha blend), reading the compute-updated buffer in the vertex
        // stage. Declared after tonemap so the WAW on the backbuffer orders it last.
        if let Some(particles_ext) = particles_ext {
            let draw = &particle_draw_pipeline;
            let buf = &particle_buffer;
            let vp = view_proj.to_cols_array();
            graph.add_pass(
                PassInfo {
                    name: "particle_draw",
                    colors: vec![(backbuffer, None)],
                    depth: None,
                    reads: vec![particles_ext],
                },
                move |ctx| {
                    let cmd = ctx.cmd();
                    cmd.bind_graphics_pipeline(draw);
                    cmd.push_constants(&particle_draw_push(
                        &vp,
                        cam_right,
                        cam_up,
                        buf.storage_index(),
                        PARTICLE_COUNT as u32,
                        0.05,
                    ));
                    cmd.draw(6, PARTICLE_COUNT as u32);
                    Ok(())
                },
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
        graph.execute(
            &device,
            &mut pools[frame],
            cmd,
            &swapchain,
            image_index,
            aliasing,
        )?;

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
        queue.submit(cmd, &image_available[frame], signal, fence)?;

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
        if screenshot_mode && frame_no >= SCREENSHOT_WARMUP + captures.len() as u64 {
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

/// The pipelines + cubemaps used to (re)generate the IBL environment chain, plus
/// the scene geometry captured into the env cube (camera-based reflections).
/// One double-buffered environment: the captured cube plus its diffuse and
/// specular convolutions. Two of these ping-pong each frame for multi-bounce.
struct CubeSet {
    env: Cubemap,
    irradiance: Cubemap,
    prefilter: Cubemap,
}

struct IblResources<'a> {
    sky_pipeline: &'a GraphicsPipeline,
    capture_pipeline: &'a GraphicsPipeline,
    irradiance_pipeline: &'a GraphicsPipeline,
    prefilter_pipeline: &'a GraphicsPipeline,
    /// Ground plane (a shadow/reflection receiver) captured into env mip 0.
    ground_vbuf: &'a Buffer,
    ground_ibuf: &'a Buffer,
    ground_count: u32,
}

/// Record the environment chain into an already-open command buffer (no submit):
/// procedural sky → env cube (full mip chain), then convolve into the
/// diffuse-irradiance cube and the per-roughness specular prefilter cube (each
/// left shader-readable). Recorded each frame before the main graph, so the
/// lighting pass samples a fresh environment (real-time capture). The BRDF LUT is
/// sky-independent and generated once (see [`generate_brdf_lut`]).
#[allow(clippy::too_many_arguments)]
fn record_environment_capture(
    cmd: &CommandBuffer,
    ibl: &IblResources,
    write: &CubeSet,
    prev: Option<&CubeSet>,
    brdf_index: i32,
    scene: &[SceneObject],
    capture_depth: &rhi::DepthBuffer,
    camera_pos: Vec3,
    sun_dir: [f32; 3],
    sun_intensity: f32,
    ambient: f32,
    flip_y: u32,
    vulkan: bool,
) {
    let env_index = write.env.bindless_index();
    let env_mips = write.env.mip_levels();
    let prefilter_max_lod = (PREFILTER_MIPS - 1) as f32;
    // Previous frame's convolved cubes (multi-bounce IBL source); -1 = single
    // bounce (capture surfaces with flat ambient only).
    let prev_ibl = match prev {
        Some(p) => [
            p.irradiance.bindless_index() as i32,
            p.prefilter.bindless_index() as i32,
            brdf_index,
        ],
        None => [-1, -1, -1],
    };

    // 1. Procedural sky -> environment cube, every mip (the sky is procedural and
    // position-independent, so each mip is just a lower-res render — no
    // downsample/self-sample hazard; the prefilter samples this mip chain).
    cmd.cube_to_color(&write.env);
    for mip in 0..env_mips {
        let size = (ENV_SIZE >> mip).max(1);
        for face in 0..6u32 {
            cmd.begin_rendering_cube_face(&write.env, face, mip, Some(ClearColor::BLACK));
            cmd.set_viewport_scissor_extent(Extent2D::new(size, size));
            cmd.bind_graphics_pipeline(ibl.sky_pipeline);
            cmd.push_constants(&sky_push(sun_dir, sun_intensity, face, flip_y));
            cmd.draw(3, 1);
            cmd.end_rendering();
        }
    }

    // 1b. Scene (ground + objects) into env mip 0 from the camera position, with
    // a depth buffer for correct occlusion, so reflective surfaces reflect the
    // live scene. Captured surfaces are shaded with direct sun + IBL from the
    // previous frame's cubes (multi-bounce) — never the cube being written, so
    // there is no recursion.
    let face_vp = cube_face_view_proj(camera_pos, vulkan);
    cmd.depth_to_render_target(capture_depth);
    for face in 0..6u32 {
        cmd.begin_rendering_cube_face_depth(&write.env, face, 0, None, capture_depth);
        cmd.set_viewport_scissor_extent(Extent2D::new(ENV_SIZE, ENV_SIZE));
        cmd.bind_graphics_pipeline(ibl.capture_pipeline);
        // Ground (matte receiver; identity model).
        cmd.push_constants(&capture_push(
            face_vp[face as usize].to_cols_array(),
            Mat4::IDENTITY.to_cols_array(),
            [0.8, 0.8, 0.8, 1.0],
            0.0,
            0.9,
            sun_dir,
            sun_intensity,
            ambient,
            camera_pos,
            prefilter_max_lod,
            prev_ibl,
        ));
        cmd.bind_vertex_buffer(ibl.ground_vbuf, 32);
        cmd.bind_index_buffer(ibl.ground_ibuf, true);
        cmd.draw_indexed(ibl.ground_count, 0, 0);
        // Scene objects (their real metallic/roughness so reflective surfaces
        // appear reflective inside the reflection).
        for obj in scene {
            let mvp = (face_vp[face as usize] * obj.transform).to_cols_array();
            cmd.push_constants(&capture_push(
                mvp,
                obj.transform.to_cols_array(),
                obj.base_color,
                obj.metallic,
                obj.roughness,
                sun_dir,
                sun_intensity,
                ambient,
                camera_pos,
                prefilter_max_lod,
                prev_ibl,
            ));
            cmd.bind_vertex_buffer(&obj.vbuf, 32);
            cmd.bind_index_buffer(&obj.ibuf, true);
            cmd.draw_indexed(obj.index_count, 0, 0);
        }
        cmd.end_rendering();
    }
    cmd.cube_to_sampled(&write.env);

    // 2. Env -> diffuse irradiance cube.
    cmd.cube_to_color(&write.irradiance);
    for face in 0..6u32 {
        cmd.begin_rendering_cube_face(&write.irradiance, face, 0, Some(ClearColor::BLACK));
        cmd.set_viewport_scissor_extent(Extent2D::new(IRRADIANCE_SIZE, IRRADIANCE_SIZE));
        cmd.bind_graphics_pipeline(ibl.irradiance_pipeline);
        cmd.push_constants(&cube_gen_push(face, flip_y, env_index, 0.0));
        cmd.draw(3, 1);
        cmd.end_rendering();
    }
    cmd.cube_to_sampled(&write.irradiance);

    // 3. Env -> specular prefilter cube (one roughness per mip).
    cmd.cube_to_color(&write.prefilter);
    for mip in 0..PREFILTER_MIPS {
        let roughness = if PREFILTER_MIPS > 1 {
            mip as f32 / (PREFILTER_MIPS - 1) as f32
        } else {
            0.0
        };
        let size = (PREFILTER_SIZE >> mip).max(1);
        for face in 0..6u32 {
            cmd.begin_rendering_cube_face(&write.prefilter, face, mip, Some(ClearColor::BLACK));
            cmd.set_viewport_scissor_extent(Extent2D::new(size, size));
            cmd.bind_graphics_pipeline(ibl.prefilter_pipeline);
            cmd.push_constants(&prefilter_push(
                face, flip_y, env_index, roughness, env_mips,
            ));
            cmd.draw(3, 1);
            cmd.end_rendering();
        }
    }
    cmd.cube_to_sampled(&write.prefilter);
}

/// Integrate the environment-BRDF LUT (sky-independent; generate once).
fn generate_brdf_lut(
    queue: &Queue,
    cmd: &CommandBuffer,
    fence: &Fence,
    brdf_pipeline: &GraphicsPipeline,
    brdf_lut: &rhi::RenderTarget,
    flip_y: u32,
) -> anyhow::Result<()> {
    cmd.begin()?;
    cmd.rt_to_render_target(brdf_lut);
    cmd.begin_rendering_target(brdf_lut, Some(ClearColor::BLACK), None);
    cmd.set_viewport_scissor_extent(Extent2D::new(BRDF_SIZE, BRDF_SIZE));
    cmd.bind_graphics_pipeline(brdf_pipeline);
    let mut push = [0u8; 16];
    push[0..4].copy_from_slice(&flip_y.to_le_bytes());
    cmd.push_constants(&push);
    cmd.draw(3, 1);
    cmd.end_rendering();
    cmd.rt_to_sampled(brdf_lut);
    cmd.end()?;
    queue.submit_oneshot(cmd, fence)?;
    fence.wait()?;
    fence.reset()?;
    Ok(())
}

/// The 6 cube-face view-projections from `eye` (90° FOV, aspect 1), matching the
/// `TextureCube` face convention. The Vulkan clip-space Y flip keeps the captured
/// faces oriented the same as the procedural sky on both backends.
fn cube_face_view_proj(eye: Vec3, vulkan: bool) -> [Mat4; 6] {
    let dirs = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
    let ups = [-Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z, -Vec3::Y, -Vec3::Y];
    let mut proj = Mat4::perspective_rh(90f32.to_radians(), 1.0, 0.05, 100.0);
    if vulkan {
        proj.y_axis.y *= -1.0;
    }
    let mut out = [Mat4::IDENTITY; 6];
    for i in 0..6 {
        let view = Mat4::look_at_rh(eye, eye + dirs[i], ups[i]);
        out[i] = proj * view;
    }
    out
}

/// Pack the capture push block (208 bytes). Layout: mvp(64), model(64),
/// base_color(16), sun(16 — xyz dir, w intensity), misc(16 — x ambient,
/// y roughness, z metallic, w prefilter max LOD), eye(16 — xyz), ibl(16 — int4
/// irradiance/prefilter/BRDF indices, -1 = no previous environment).
#[allow(clippy::too_many_arguments)]
fn capture_push(
    mvp: [f32; 16],
    model: [f32; 16],
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    sun_dir: [f32; 3],
    sun_intensity: f32,
    ambient: f32,
    eye: Vec3,
    prefilter_max_lod: f32,
    ibl: [i32; 3],
) -> [u8; 208] {
    let mut pc = [0u8; 208];
    for (i, f) in mvp.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    for (i, f) in model.iter().enumerate() {
        let o = 64 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    for (i, f) in base_color.iter().enumerate() {
        let o = 128 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    let n = normalize3(sun_dir);
    for (i, f) in n.iter().take(3).enumerate() {
        let o = 144 + i * 4;
        pc[o..o + 4].copy_from_slice(&f.to_le_bytes());
    }
    pc[156..160].copy_from_slice(&sun_intensity.to_le_bytes());
    // misc: x ambient, y roughness, z metallic, w prefilter max LOD.
    pc[160..164].copy_from_slice(&ambient.to_le_bytes());
    pc[164..168].copy_from_slice(&roughness.to_le_bytes());
    pc[168..172].copy_from_slice(&metallic.to_le_bytes());
    pc[172..176].copy_from_slice(&prefilter_max_lod.to_le_bytes());
    // eye: xyz capture/camera position.
    pc[176..180].copy_from_slice(&eye.x.to_le_bytes());
    pc[180..184].copy_from_slice(&eye.y.to_le_bytes());
    pc[184..188].copy_from_slice(&eye.z.to_le_bytes());
    // ibl: int4 previous-frame irradiance / prefilter / BRDF indices.
    pc[192..196].copy_from_slice(&ibl[0].to_le_bytes());
    pc[196..200].copy_from_slice(&ibl[1].to_le_bytes());
    pc[200..204].copy_from_slice(&ibl[2].to_le_bytes());
    pc
}

/// Pack the sky push block: sun float4 (xyz dir, w intensity) + face + flip_y +
/// pad (32 bytes).
fn sky_push(sun_dir: [f32; 3], intensity: f32, face: u32, flip_y: u32) -> [u8; 32] {
    let n = normalize3(sun_dir);
    let mut pc = [0u8; 32];
    for (i, v) in n.iter().take(3).enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[12..16].copy_from_slice(&intensity.to_le_bytes());
    pc[16..20].copy_from_slice(&face.to_le_bytes());
    pc[20..24].copy_from_slice(&flip_y.to_le_bytes());
    pc
}

/// Pack the irradiance push block: face + flip_y + env_index + pad (16 bytes).
fn cube_gen_push(face: u32, flip_y: u32, env_index: u32, roughness: f32) -> [u8; 16] {
    let mut pc = [0u8; 16];
    pc[0..4].copy_from_slice(&face.to_le_bytes());
    pc[4..8].copy_from_slice(&flip_y.to_le_bytes());
    pc[8..12].copy_from_slice(&env_index.to_le_bytes());
    pc[12..16].copy_from_slice(&roughness.to_le_bytes());
    pc
}

/// Pack the prefilter push block: face + flip_y + env_index + roughness +
/// env_mips (20 bytes — env_mips drives the mip-based importance sampling).
fn prefilter_push(
    face: u32,
    flip_y: u32,
    env_index: u32,
    roughness: f32,
    env_mips: u32,
) -> [u8; 20] {
    let mut pc = [0u8; 20];
    pc[0..4].copy_from_slice(&face.to_le_bytes());
    pc[4..8].copy_from_slice(&flip_y.to_le_bytes());
    pc[8..12].copy_from_slice(&env_index.to_le_bytes());
    pc[12..16].copy_from_slice(&roughness.to_le_bytes());
    pc[16..20].copy_from_slice(&env_mips.to_le_bytes());
    pc
}

/// Pack the tonemap push block: hdr_index + mode + flip_y + pad (16 bytes).
fn post_push(hdr_index: u32, mode: u32, flip_y: u32) -> [u8; 16] {
    let mut pc = [0u8; 16];
    pc[0..4].copy_from_slice(&hdr_index.to_le_bytes());
    pc[4..8].copy_from_slice(&mode.to_le_bytes());
    pc[8..12].copy_from_slice(&flip_y.to_le_bytes());
    pc
}

/// Pack the particle-sim push block: buffer_index + count + dt + time + init.
fn particle_sim_push(buffer_index: u32, count: u32, dt: f32, time: f32, init: u32) -> [u8; 20] {
    let mut pc = [0u8; 20];
    pc[0..4].copy_from_slice(&buffer_index.to_le_bytes());
    pc[4..8].copy_from_slice(&count.to_le_bytes());
    pc[8..12].copy_from_slice(&dt.to_le_bytes());
    pc[12..16].copy_from_slice(&time.to_le_bytes());
    pc[16..20].copy_from_slice(&init.to_le_bytes());
    pc
}

/// Pack the particle-draw push block: view_proj(64) + cam_right(16) + cam_up(16)
/// + buffer_index + count + size + pad (16) = 112 bytes.
fn particle_draw_push(
    view_proj: &[f32; 16],
    cam_right: Vec3,
    cam_up: Vec3,
    buffer_index: u32,
    count: u32,
    size: f32,
) -> [u8; 112] {
    let mut pc = [0u8; 112];
    for (i, v) in view_proj.iter().enumerate() {
        pc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    pc[64..68].copy_from_slice(&cam_right.x.to_le_bytes());
    pc[68..72].copy_from_slice(&cam_right.y.to_le_bytes());
    pc[72..76].copy_from_slice(&cam_right.z.to_le_bytes());
    pc[80..84].copy_from_slice(&cam_up.x.to_le_bytes());
    pc[84..88].copy_from_slice(&cam_up.y.to_le_bytes());
    pc[88..92].copy_from_slice(&cam_up.z.to_le_bytes());
    pc[96..100].copy_from_slice(&buffer_index.to_le_bytes());
    pc[100..104].copy_from_slice(&count.to_le_bytes());
    pc[104..108].copy_from_slice(&size.to_le_bytes());
    pc
}

/// Pack the compute-post push block: hdr_index + out_index + width + height.
fn post_compute_push(hdr_index: u32, out_index: u32, width: u32, height: u32) -> [u8; 16] {
    let mut pc = [0u8; 16];
    pc[0..4].copy_from_slice(&hdr_index.to_le_bytes());
    pc[4..8].copy_from_slice(&out_index.to_le_bytes());
    pc[8..12].copy_from_slice(&width.to_le_bytes());
    pc[12..16].copy_from_slice(&height.to_le_bytes());
    pc
}

fn upload_mesh(device: &Device, model: &MeshData) -> anyhow::Result<(Buffer, Buffer, u32)> {
    let vbytes = unsafe {
        std::slice::from_raw_parts(
            model.vertices.as_ptr() as *const u8,
            std::mem::size_of_val(model.vertices.as_slice()),
        )
    };
    let ibytes = unsafe {
        std::slice::from_raw_parts(
            model.indices.as_ptr() as *const u8,
            std::mem::size_of_val(model.indices.as_slice()),
        )
    };
    let vbuf = device.create_buffer(&BufferDesc {
        size: vbytes.len() as u64,
        usage: BufferUsage::Vertex,
    })?;
    vbuf.write(vbytes)?;
    let ibuf = device.create_buffer(&BufferDesc {
        size: ibytes.len() as u64,
        usage: BufferUsage::Index,
    })?;
    ibuf.write(ibytes)?;
    Ok((vbuf, ibuf, model.indices.len() as u32))
}

/// Create a sampled texture from decoded image data and return its bindless index,
/// keeping the texture alive in `store`.
fn upload_texture(
    device: &Device,
    store: &mut Vec<Texture>,
    img: &ImageData,
    format: Format,
) -> anyhow::Result<u32> {
    let t = device.create_texture(
        &TextureDesc {
            width: img.width,
            height: img.height,
            format,
        },
        &img.rgba8,
    )?;
    let idx = t.bindless_index();
    store.push(t);
    Ok(idx)
}

/// A large horizontal quad at height `y` (normal up, +Y), used as a shadow
/// receiver. `half` is half its side length. Built on the mesh vertex layout so
/// it shares the G-buffer / shadow pipelines.
fn ground_mesh(half: f32, y: f32) -> MeshData {
    let v = |x: f32, z: f32, u: f32, w: f32| MeshVertex {
        pos: [x, y, z],
        normal: [0.0, 1.0, 0.0],
        uv: [u, w],
    };
    MeshData {
        vertices: vec![
            v(-half, -half, 0.0, 0.0),
            v(half, -half, 1.0, 0.0),
            v(half, half, 1.0, 1.0),
            v(-half, half, 0.0, 1.0),
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
        material: dreamcoast_asset::Material::default(),
    }
}

/// Framing bounds of the normalized model.
struct ModelBounds {
    /// Bounding-sphere radius (always 1.0 after normalization) — the unit the
    /// camera, ground, lights, and shadow box are sized in.
    radius: f32,
}

/// Normalize a mesh into canonical units: recenter its footprint on the origin,
/// rest its base on `y = 0`, and uniformly scale so its bounding-sphere radius is
/// 1.0. glTF models vary wildly in authored scale/placement (this avocado is
/// sub-0.1 units, off the origin); normalizing keeps the camera/near-far planes,
/// ground, lights, and shadow box in comfortable, model-independent units.
fn normalize_on_ground(model: &mut MeshData) -> ModelBounds {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for v in &model.vertices {
        for i in 0..3 {
            min[i] = min[i].min(v.pos[i]);
            max[i] = max[i].max(v.pos[i]);
        }
    }
    let cx = (min[0] + max[0]) * 0.5;
    let cz = (min[2] + max[2]) * 0.5;
    let base = min[1];
    let (sx, sy, sz) = (max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    let radius = (0.5 * (sx * sx + sy * sy + sz * sz).sqrt()).max(1e-6);
    let s = 1.0 / radius; // normalize the bounding-sphere radius to 1.0
    for v in &mut model.vertices {
        v.pos[0] = (v.pos[0] - cx) * s;
        v.pos[1] = (v.pos[1] - base) * s;
        v.pos[2] = (v.pos[2] - cz) * s;
    }
    ModelBounds { radius: 1.0 }
}

/// 8x8 magenta/grey checker (fallback base color).
fn make_checker_texture(device: &Device) -> anyhow::Result<Texture> {
    const N: u32 = 8;
    let mut pixels = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            let on = (x + y) % 2 == 0;
            pixels.extend_from_slice(if on {
                &[220, 60, 200, 255]
            } else {
                &[40, 40, 48, 255]
            });
        }
    }
    Ok(device.create_texture(
        &TextureDesc {
            width: N,
            height: N,
            format: Format::Rgba8Srgb,
        },
        &pixels,
    )?)
}

/// Fetch the (vertex, fragment) bytecode for `backend` from a shader's four
/// generated accessors, erroring if unavailable.
fn load_shader_pair(
    backend: BackendKind,
    vs_spirv: fn() -> Option<&'static [u8]>,
    fs_spirv: fn() -> Option<&'static [u8]>,
    vs_dxil: fn() -> Option<&'static [u8]>,
    fs_dxil: fn() -> Option<&'static [u8]>,
    name: &str,
) -> anyhow::Result<(&'static [u8], &'static [u8])> {
    let (vs, fs) = match backend {
        BackendKind::Vulkan => (vs_spirv(), fs_spirv()),
        BackendKind::D3d12 => (vs_dxil(), fs_dxil()),
    };
    let vs = vs.ok_or_else(|| anyhow!("{name} vertex shader unavailable for {backend:?}"))?;
    let fs = fs.ok_or_else(|| anyhow!("{name} fragment shader unavailable for {backend:?}"))?;
    Ok((vs, fs))
}

/// Fetch single-stage (compute) bytecode for `backend`, erroring if unavailable.
fn load_compute_shader(
    backend: BackendKind,
    cs_spirv: fn() -> Option<&'static [u8]>,
    cs_dxil: fn() -> Option<&'static [u8]>,
    name: &str,
) -> anyhow::Result<&'static [u8]> {
    let cs = match backend {
        BackendKind::Vulkan => cs_spirv(),
        BackendKind::D3d12 => cs_dxil(),
    };
    cs.ok_or_else(|| anyhow!("{name} compute shader unavailable for {backend:?}"))
}

fn build_render_finished(device: &Device, count: u32) -> anyhow::Result<Vec<Semaphore>> {
    (0..count)
        .map(|_| device.create_semaphore().map_err(Into::into))
        .collect()
}

/// A requested screenshot: output path + whether to include the ImGui overlay.
#[derive(Clone)]
struct Capture {
    path: String,
    include_ui: bool,
}

/// Parse `--screenshot <path>` (with UI overlay) and `--screenshot-clean <path>`
/// (3D only) flags into capture requests, in argument order. Presence of any
/// puts the app in headless screenshot mode (render a few frames, capture, exit).
fn screenshot_captures() -> Vec<Capture> {
    let mut out = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let include_ui = match arg.as_str() {
            "--screenshot" => true,
            "--screenshot-clean" => false,
            _ => continue,
        };
        if let Some(path) = args.next() {
            out.push(Capture { path, include_ui });
        }
    }
    out
}

/// Auto-generated path for an interactive (F2) screenshot.
fn interactive_screenshot_path() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("screenshot_{secs}.png")
}

/// Save BGRA readback bytes (rows padded to `layout.row_pitch`) as a PNG. The
/// swapchain stores sRGB-encoded bytes, so they map straight to a PNG after the
/// B<->R channel swap; padding is dropped per row.
fn save_screenshot(path: &str, data: &[u8], layout: &ReadbackLayout) -> anyhow::Result<()> {
    let w = layout.width as usize;
    let h = layout.height as usize;
    let pitch = layout.row_pitch as usize;
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        let src = &data[y * pitch..y * pitch + w * 4];
        let dst = &mut rgba[y * w * 4..(y + 1) * w * 4];
        for x in 0..w {
            dst[x * 4] = src[x * 4 + 2]; // R <- B
            dst[x * 4 + 1] = src[x * 4 + 1]; // G
            dst[x * 4 + 2] = src[x * 4]; // B <- R
            dst[x * 4 + 3] = src[x * 4 + 3]; // A
        }
    }
    let img = image::RgbaImage::from_raw(layout.width, layout.height, rgba)
        .ok_or_else(|| anyhow!("screenshot buffer size mismatch"))?;
    img.save(path)?;
    Ok(())
}

/// Model path: `--model <path>` or the default `assets/model.glb`.
fn model_path() -> String {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--model"
            && let Some(p) = args.next()
        {
            return p;
        }
    }
    MODEL_PATH.to_string()
}

fn select_backend() -> BackendKind {
    let mut backend = if cfg!(windows) {
        BackendKind::D3d12
    } else {
        BackendKind::Vulkan
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--backend" {
            match args.next().as_deref() {
                Some("vulkan") => backend = BackendKind::Vulkan,
                Some("d3d12") => backend = BackendKind::D3d12,
                other => tracing::warn!("unknown --backend value {other:?}; using default"),
            }
        }
    }
    backend
}
