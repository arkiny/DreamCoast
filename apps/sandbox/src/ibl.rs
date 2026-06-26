//! Image-based lighting — extracted from `run()` as R5 of the render-loop
//! decomposition (see docs/refactor-sandbox.md). `IblSystem` owns the sky / capture
//! / irradiance / prefilter / BRDF pipelines, the two ping-pong environment cube
//! sets, the capture depth buffer and the BRDF LUT, plus the per-frame ping-pong /
//! capture-decision state. The actual GPU recording stays in the `record_*` /
//! `generate_*` helpers below (the bake logic the bundle wraps).
//!
//! The environment capture records into the raw command buffer *before* the render
//! graph (so the lighting pass samples a fresh environment), not as a graph pass, so
//! `IblSystem` exposes `maybe_capture(&mut self, cmd, …)` (the per-frame ping-pong)
//! and `lighting_indices()` (the bindless indices the lighting pass reads), rather
//! than the `record_*` graph methods the other bundles use.

use dreamcoast_core::glam::{Mat4, Vec3};
use rhi::{
    BackendKind, BlendMode, Buffer, ClearColor, CommandBuffer, Cubemap, CubemapDesc, DepthBuffer,
    Device, Extent2D, Fence, Format, GraphicsPipeline, GraphicsPipelineDesc, PrimitiveTopology,
    Queue, RenderTarget, RenderTargetDesc, VertexLayout,
};

use crate::app::load_shader_pair;
use crate::push::{capture_push, cube_face_view_proj, cube_gen_push, prefilter_push, sky_push};
use crate::{
    BRDF_SIZE, DEPTH_FORMAT, ENV_MIPS, ENV_SIZE, HDR_FORMAT, IRRADIANCE_SIZE, PREFILTER_MIPS,
    PREFILTER_SIZE, SceneObject,
};

/// One double-buffered environment: the captured cube plus its diffuse and
/// specular convolutions. Two of these ping-pong each frame for multi-bounce.
struct CubeSet {
    env: Cubemap,
    irradiance: Cubemap,
    prefilter: Cubemap,
}

/// Transient borrow of the capture pipelines + ground geometry, assembled per call
/// from the owning [`IblSystem`] to feed [`record_environment_capture`].
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

/// Pipelines + persistent resources for image-based lighting: the procedural-sky /
/// scene-capture / irradiance / prefilter / BRDF pipelines, the two ping-pong
/// environment cube sets, the capture depth buffer and the (sky-independent) BRDF
/// LUT. Also tracks the per-frame ping-pong parity + capture-decision state.
pub(crate) struct IblSystem {
    sky_pipeline: GraphicsPipeline,
    capture_pipeline: GraphicsPipeline,
    irradiance_pipeline: GraphicsPipeline,
    prefilter_pipeline: GraphicsPipeline,
    /// Double-buffered environment cube sets for multi-bounce reflections.
    cube_sets: [CubeSet; 2],
    /// Depth buffer for capturing scene geometry into the env cube faces.
    capture_depth: DepthBuffer,
    /// Kept alive so its bindless slot (`brdf_index`, sampled by the lighting pass)
    /// stays valid; not otherwise read after construction.
    _brdf_lut: RenderTarget,
    brdf_index: i32,
    /// False until the first capture; the boot warm-up does not flip it, so the
    /// first loop frame always captures.
    env_captured: bool,
    last_sun: ([f32; 3], f32),
    /// Ping-pong parity for the two cube sets; advances only on an actual capture,
    /// so a skipped frame keeps sampling the last written set.
    env_parity: usize,
    last_written: usize,
}

impl IblSystem {
    /// Build the IBL pipelines + cube sets + BRDF LUT, naming the persistent
    /// resources for GPU captures, and integrate the (sky-independent) BRDF LUT once.
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        queue: &Queue,
        flip_y: u32,
        sun_dir: [f32; 3],
        sun_intensity: f32,
    ) -> anyhow::Result<Self> {
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
        let capture_depth = device.create_depth_buffer(Extent2D::new(ENV_SIZE, ENV_SIZE))?;
        let brdf_lut = device.create_render_target(&RenderTargetDesc {
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

        // The BRDF LUT is sky-independent — generate it once. The environment chain
        // is (re)captured per frame inside the render loop.
        {
            let gen_cmd = device.create_command_buffer()?;
            let gen_fence = device.create_fence(false)?;
            generate_brdf_lut(
                queue,
                &gen_cmd,
                &gen_fence,
                &brdf_pipeline,
                &brdf_lut,
                flip_y,
            )?;
        }

        Ok(Self {
            sky_pipeline,
            capture_pipeline,
            irradiance_pipeline,
            prefilter_pipeline,
            cube_sets,
            capture_depth,
            _brdf_lut: brdf_lut,
            brdf_index,
            env_captured: false,
            last_sun: (sun_dir, sun_intensity),
            env_parity: 0,
            last_written: 0,
        })
    }

    /// Transient borrow of the capture pipelines + ground geometry for a recording.
    fn resources<'a>(
        &'a self,
        ground_vbuf: &'a Buffer,
        ground_ibuf: &'a Buffer,
        ground_count: u32,
    ) -> IblResources<'a> {
        IblResources {
            sky_pipeline: &self.sky_pipeline,
            capture_pipeline: &self.capture_pipeline,
            irradiance_pipeline: &self.irradiance_pipeline,
            prefilter_pipeline: &self.prefilter_pipeline,
            ground_vbuf,
            ground_ibuf,
            ground_count,
        }
    }

    /// Seed both cube sets once (single-bounce, no previous environment) so the first
    /// multi-bounce frame reads valid data instead of uninitialized memory. Uses an
    /// approximate camera; the render loop immediately recaptures with the live one.
    /// Does not flip `env_captured`, so the first loop frame still captures.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn seed(
        &self,
        device: &Device,
        queue: &Queue,
        scene: &[SceneObject],
        ground_vbuf: &Buffer,
        ground_ibuf: &Buffer,
        ground_count: u32,
        boot_eye: Vec3,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        ambient: f32,
        flip_y: u32,
        vulkan: bool,
    ) -> anyhow::Result<()> {
        let res = self.resources(ground_vbuf, ground_ibuf, ground_count);
        let init_cmd = device.create_command_buffer()?;
        let init_fence = device.create_fence(false)?;
        init_cmd.begin()?;
        for set in &self.cube_sets {
            record_environment_capture(
                &init_cmd,
                &res,
                set,
                None,
                self.brdf_index,
                scene,
                &self.capture_depth,
                boot_eye,
                sun_dir,
                sun_intensity,
                ambient,
                flip_y,
                vulkan,
            );
        }
        init_cmd.end()?;
        queue.submit_oneshot(&init_cmd, &init_fence)?;
        init_fence.wait()?;
        Ok(())
    }

    /// (Re)capture the environment into the "write" cube set before the main graph
    /// samples it: every frame when `realtime_env`, otherwise only the first frame
    /// and whenever the sun changes. With `multibounce`, captured surfaces are shaded
    /// with IBL from the "read" set (the previous frame), so reflective surfaces
    /// reflect each other; the parity advances only on an actual capture.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn maybe_capture(
        &mut self,
        cmd: &CommandBuffer,
        realtime_env: bool,
        multibounce: bool,
        scene: &[SceneObject],
        ground_vbuf: &Buffer,
        ground_ibuf: &Buffer,
        ground_count: u32,
        focus: Vec3,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        ambient: f32,
        flip_y: u32,
        vulkan: bool,
    ) {
        let sun_changed = (sun_dir, sun_intensity) != self.last_sun;
        if !(realtime_env || !self.env_captured || sun_changed) {
            return;
        }
        let write = self.env_parity % 2;
        let read = 1 - write;
        let res = self.resources(ground_vbuf, ground_ibuf, ground_count);
        let prev = if multibounce && self.env_captured {
            Some(&self.cube_sets[read])
        } else {
            None
        };
        record_environment_capture(
            cmd,
            &res,
            &self.cube_sets[write],
            prev,
            self.brdf_index,
            scene,
            &self.capture_depth,
            // Capture the reflection probe at the scene centre, NOT the camera: a
            // camera-anchored probe gives every reflective surface a parallax error
            // (the reflected ground/horizon slides up the spheres as the camera
            // moves). A fixed probe near the objects keeps reflections stable and
            // roughly correct for surfaces around the centre.
            focus,
            sun_dir,
            sun_intensity,
            ambient,
            flip_y,
            vulkan,
        );
        self.last_written = write;
        self.env_parity += 1;
        self.env_captured = true;
        self.last_sun = (sun_dir, sun_intensity);
    }

    /// The bindless indices the main lighting pass samples (env, irradiance,
    /// prefilter of the most recently written cube set, plus the BRDF LUT).
    pub(crate) fn lighting_indices(&self) -> [i32; 4] {
        let write_set = &self.cube_sets[self.last_written];
        [
            write_set.env.bindless_index() as i32,
            write_set.irradiance.bindless_index() as i32,
            write_set.prefilter.bindless_index() as i32,
            self.brdf_index,
        ]
    }
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
