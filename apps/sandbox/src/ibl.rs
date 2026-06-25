//! Image-based-lighting bake extracted from `main.rs`: the per-frame environment
//! capture (procedural sky + scene into a cubemap, then diffuse-irradiance and
//! specular-prefilter convolutions) and the one-time BRDF LUT. These record GPU
//! commands but own no persistent state, so they sit apart from the render loop.

use dreamcoast_core::glam::{Mat4, Vec3};
use rhi::{Buffer, ClearColor, CommandBuffer, Cubemap, Extent2D, Fence, GraphicsPipeline, Queue};

use crate::push::{capture_push, cube_face_view_proj, cube_gen_push, prefilter_push, sky_push};
use crate::{BRDF_SIZE, ENV_SIZE, IRRADIANCE_SIZE, PREFILTER_MIPS, PREFILTER_SIZE, SceneObject};

/// The pipelines + cubemaps used to (re)generate the IBL environment chain, plus
/// the scene geometry captured into the env cube (camera-based reflections).
/// One double-buffered environment: the captured cube plus its diffuse and
/// specular convolutions. Two of these ping-pong each frame for multi-bounce.
pub(crate) struct CubeSet {
    pub(crate) env: Cubemap,
    pub(crate) irradiance: Cubemap,
    pub(crate) prefilter: Cubemap,
}

pub(crate) struct IblResources<'a> {
    pub(crate) sky_pipeline: &'a GraphicsPipeline,
    pub(crate) capture_pipeline: &'a GraphicsPipeline,
    pub(crate) irradiance_pipeline: &'a GraphicsPipeline,
    pub(crate) prefilter_pipeline: &'a GraphicsPipeline,
    /// Ground plane (a shadow/reflection receiver) captured into env mip 0.
    pub(crate) ground_vbuf: &'a Buffer,
    pub(crate) ground_ibuf: &'a Buffer,
    pub(crate) ground_count: u32,
}

/// Record the environment chain into an already-open command buffer (no submit):
/// procedural sky → env cube (full mip chain), then convolve into the
/// diffuse-irradiance cube and the per-roughness specular prefilter cube (each
/// left shader-readable). Recorded each frame before the main graph, so the
/// lighting pass samples a fresh environment (real-time capture). The BRDF LUT is
/// sky-independent and generated once (see [`generate_brdf_lut`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn record_environment_capture(
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
pub(crate) fn generate_brdf_lut(
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
