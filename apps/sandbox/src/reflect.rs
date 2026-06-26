//! Phase 11 Stage C reflection track — split from `gdf.rs` because it is screen-space,
//! not GDF-based, and is its own growing cluster: screen-space reflections (C5) and,
//! later, the GDF reflection fallback (C6) + hybrid composite (C7) that together replace
//! the captured-cube IBL specular. Each `record_*` adds one pass and returns its output
//! image, borrowing `&'a self` for the graph's lifetime like the other render bundles.

use dreamcoast_core::glam::Vec3;
use dreamcoast_render::{ComputePassInfo, RenderGraph, ResourceId};
use rhi::{BackendKind, ComputePipeline, ComputePipelineDesc, Device, Extent2D};

use crate::HDR_FORMAT;
use crate::app::load_compute_shader;
use crate::push::ssr_push;

pub(crate) struct ReflectSystem {
    ssr_pipeline: Option<ComputePipeline>, // C5 screen-space reflections
}

impl ReflectSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        compute_supported: bool,
    ) -> anyhow::Result<Self> {
        let ssr_pipeline = if compute_supported {
            let cs = load_compute_shader(
                backend,
                dreamcoast_shader::ssr_cs_spirv,
                dreamcoast_shader::ssr_cs_dxil,
                dreamcoast_shader::ssr_cs_metallib,
                "ssr",
            )?;
            Some(device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: cs,
                compute_entry: "csMain",
                push_constant_size: 192,
                bindless: true,
                threads_per_group: [8, 8, 1],
            })?)
        } else {
            None
        };
        Ok(Self { ssr_pipeline })
    }

    pub(crate) fn has_ssr(&self) -> bool {
        self.ssr_pipeline.is_some()
    }

    /// C5: screen-space reflections. A full-screen compute pass reflects the view ray
    /// about each surface normal and marches it through the depth buffer, sampling the
    /// shaded HDR at the hit so reflective surfaces show real neighbouring geometry.
    /// Reads the G-buffer + the (already-lit) HDR; returns the reflection image
    /// (rgb = reflected color, a = confidence; misses are zero for the C6/C7 fallback).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_ssr<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        hdr: ResourceId,
        depth: ResourceId,
        normal: ResourceId,
        material: ResourceId,
        extent: Extent2D,
        view_proj: [f32; 16],
        inv_view_proj: [f32; 16],
        eye: Vec3,
        cw: u32,
        ch: u32,
        flip_y: u32,
        max_dist: f32,
        thickness: f32,
    ) -> ResourceId {
        let pipe = self.ssr_pipeline.as_ref().expect("ssr pipeline");
        let out = graph.create_storage_image("ssr_out", HDR_FORMAT, extent);
        graph.add_compute_pass(
            ComputePassInfo {
                name: "ssr",
                storage_writes: vec![out],
                reads: vec![hdr, depth, normal, material],
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let material_index = ctx.sampled_index(material);
                let color_index = ctx.sampled_index(hdr);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&ssr_push(
                    &view_proj,
                    &inv_view_proj,
                    eye,
                    depth_index,
                    normal_index,
                    material_index,
                    color_index,
                    out_index,
                    cw,
                    ch,
                    flip_y,
                    max_dist,
                    thickness,
                    96.0, // march steps (finer = less stair-step speckle)
                    0.1,  // edge-fade width (fraction of half-screen)
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }
}
