//! PR-4 (render-pipeline re-baseline track, `docs/render-pipeline-reference.md` §3) —
//! the sky/atmosphere composite slot: a full-screen pass point sitting after opaque
//! lighting + reflections are fully composited and before the (future) transparency
//! pass / post chain (§1.6 #10-11 in the reference doc).
//!
//! The slot is opt-in end to end: `AtmosphereSystem::new` still builds the pipeline
//! (so the shader is exercised by every backend's build), but `record_fog` is only
//! ever CALLED from `main.rs` when `P_HEIGHT_FOG=1` — the call site wraps it in an
//! `Option`, so the pass is never added to the graph (and costs nothing, byte-
//! identical) unless the flag is on. See `docs/atmosphere-fog-slot.md`.
//!
//! Today's one feature: analytic exponential height fog (`atmosphere.slang`), whose
//! inscatter color reuses the single-source procedural sky (`sky_common.slang`) —
//! the same function `sky.slang` (env-cube capture) and the path tracer call — fed
//! the SAME Rust-side sun/sky-gain/white-balance values already threaded everywhere
//! else in `main.rs`, so there is no duplicated ambient/sky constant.

use dreamcoast_render::{PassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, BlendMode, DepthCompare, Device, Format, GraphicsPipeline, GraphicsPipelineDesc,
};

use crate::app::load_shader_pair;
use crate::push::atmosphere_push;

/// The atmosphere/height-fog full-screen composite pipeline.
pub(crate) struct AtmosphereSystem {
    pipeline: GraphicsPipeline,
}

impl AtmosphereSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        hdr_format: Format,
    ) -> anyhow::Result<Self> {
        let (vs, fs) = load_shader_pair(
            backend,
            dreamcoast_shader::atmosphere_vs_spirv,
            dreamcoast_shader::atmosphere_fs_spirv,
            dreamcoast_shader::atmosphere_vs_dxil,
            dreamcoast_shader::atmosphere_fs_dxil,
            dreamcoast_shader::atmosphere_vs_metallib,
            dreamcoast_shader::atmosphere_fs_metallib,
            "atmosphere",
        )?;
        let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex_bytes: vs,
            fragment_bytes: fs,
            vertex_entry: "vsMain",
            fragment_entry: "fsMain",
            color_formats: &[hdr_format],
            topology: rhi::PrimitiveTopology::TriangleList,
            vertex_layout: rhi::VertexLayout::None,
            blend: BlendMode::Opaque,
            push_constant_size: 80, // 4 uints + 4 float4 rows (see push::atmosphere_push)
            bindless: true,
            uniform_buffer: false,
            depth_test: false,
            depth_write: false,
            depth_compare: DepthCompare::Less,
            depth_format: None,
        })?;
        Ok(Self { pipeline })
    }

    /// Record the height-fog composite: reads `hdr_in` (the finished opaque HDR scene
    /// color, post-lighting/reflection) + `position` (the G-buffer world-position MRT,
    /// for per-pixel camera distance + the background/sky mask), writes a NEW `hdr_out`
    /// color target (same format/extent as `hdr_in`) the caller threads forward in place
    /// of `hdr_in` — the same "read old, write new, rethread" convention
    /// `record_compute_post`/`record_tonemap` use elsewhere in the graph.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_fog<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        hdr_in: ResourceId,
        hdr_out: ResourceId,
        position: ResourceId,
        camera_pos: [f32; 3],
        density: f32,
        height_falloff: f32,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        sky_wb: [f32; 3],
        inscatter_gain: f32,
        exposure: f32,
        flip_y: u32,
    ) {
        graph.add_pass(
            PassInfo {
                name: "atmosphere_fog",
                colors: vec![(hdr_out, None)],
                depth: None,
                reads: vec![hdr_in, position],
            },
            move |ctx| {
                let hdr_index = ctx.sampled_index(hdr_in);
                let position_index = ctx.sampled_index(position);
                let cmd = ctx.cmd();
                cmd.bind_graphics_pipeline(&self.pipeline);
                cmd.push_constants(&atmosphere_push(
                    hdr_index,
                    position_index,
                    camera_pos,
                    density,
                    sun_dir,
                    sun_intensity,
                    sky_wb,
                    inscatter_gain,
                    height_falloff,
                    exposure,
                    flip_y,
                ));
                cmd.draw(3, 1);
                Ok(())
            },
        );
    }
}
