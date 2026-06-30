//! Screen-space ambient occlusion (HBAO-lite obscurance) — the near-field AO that the coarse
//! GDF AO (`gi.rs` / `gdf_ao.slang`) can't resolve. A single compute pass reads the G-buffer
//! depth + world normal and accumulates horizon obscurance in a depth-scaled screen disk; the
//! lighting pass composes it with the GDF AO (`pbr.slang`: `ambient * gdf_ao * ssao`). Output is
//! an R-channel-significant HDR image (matches the GDF AO target). See `gtao.slang`.
//!
//! Opt-in via `quality.rs` (`SSAO`); absent → the lighting binds `0xFFFFFFFF` and multiplies by
//! 1.0 (byte-identical baseline). The dither rotation is a pure integer hash (DX≡VK-deterministic).

use dreamcoast_render::{ComputePassInfo, RenderGraph, ResourceId};
use rhi::{BackendKind, ComputePipeline, ComputePipelineDesc, Device, Extent2D};

use crate::HDR_FORMAT;
use crate::app::load_compute_shader;
use crate::push::gtao_push;

/// The screen-space AO compute pipeline (the `gtao.slang` `csMain` obscurance kernel).
pub(crate) struct GtaoSystem {
    pipeline: Option<ComputePipeline>,
}

impl GtaoSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        compute_supported: bool,
    ) -> anyhow::Result<Self> {
        let pipeline = if compute_supported {
            let cs = load_compute_shader(
                backend,
                dreamcoast_shader::gtao_cs_spirv,
                dreamcoast_shader::gtao_cs_dxil,
                dreamcoast_shader::gtao_cs_metallib,
                "gtao",
            )?;
            Some(device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: cs,
                compute_entry: "csMain",
                push_constant_size: 144,
                bindless: true,
                uniform_buffer: false,
                threads_per_group: [8, 8, 1],
            })?)
        } else {
            None
        };
        Ok(Self { pipeline })
    }

    /// Record the SSAO pass: reconstruct world position from `depth`, read `normal`, accumulate
    /// obscurance, write the AO factor to a new storage image (returned). `None` if compute is
    /// unsupported. `radius`/`intensity`/`bias`/`power` are the tuning knobs; `proj_scale` =
    /// `0.5/tan(fovY/2)` scales the world radius to a depth-correct screen footprint.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        depth: ResourceId,
        normal: ResourceId,
        extent: Extent2D,
        inv_view_proj: [f32; 16],
        camera_pos: [f32; 3],
        cw: u32,
        ch: u32,
        flip_y: u32,
        radius: f32,
        intensity: f32,
        bias: f32,
        proj_scale: f32,
        power: f32,
    ) -> Option<ResourceId> {
        let pipe = self.pipeline.as_ref()?;
        let out = graph.create_storage_image("ssao_out", HDR_FORMAT, extent);
        let aspect = cw as f32 / ch.max(1) as f32;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "ssao",
                storage_writes: vec![out],
                reads: vec![depth, normal],
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&gtao_push(
                    &inv_view_proj,
                    camera_pos,
                    depth_index,
                    normal_index,
                    out_index,
                    cw,
                    ch,
                    flip_y,
                    0,
                    u32::MAX,
                    radius,
                    intensity,
                    bias,
                    proj_scale,
                    aspect,
                    power,
                    0.0,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        Some(out)
    }
}
