//! Phase 11 Stage C reflection track — split from `gdf.rs` because it is screen-space,
//! not GDF-based, and is its own growing cluster: screen-space reflections (C5) and,
//! later, the GDF reflection fallback (C6) + hybrid composite (C7) that together replace
//! the captured-cube IBL specular. Each `record_*` adds one pass and returns its output
//! image, borrowing `&'a self` for the graph's lifetime like the other render bundles.

use dreamcoast_core::glam::Vec3;
use dreamcoast_render::{ComputePassInfo, RenderGraph, ResourceId};
use rhi::{BackendKind, ComputePipeline, ComputePipelineDesc, Device, Extent2D, Volume};

use crate::HDR_FORMAT;
use crate::app::load_compute_shader;
use crate::push::{gdf_reflect_push, reflect_composite_push, ssr_push};

pub(crate) struct ReflectSystem {
    ssr_pipeline: Option<ComputePipeline>, // C5 screen-space reflections
    reflect_pipeline: Option<ComputePipeline>, // C6 GDF reflection fallback
    composite_pipeline: Option<ComputePipeline>, // C7 hybrid composite
}

impl ReflectSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        compute_supported: bool,
    ) -> anyhow::Result<Self> {
        let compute = |spirv: fn() -> Option<&'static [u8]>,
                       dxil: fn() -> Option<&'static [u8]>,
                       metallib: fn() -> Option<&'static [u8]>,
                       name: &str,
                       pcsize: u32|
         -> anyhow::Result<Option<ComputePipeline>> {
            if !compute_supported {
                return Ok(None);
            }
            let cs = load_compute_shader(backend, spirv, dxil, metallib, name)?;
            Ok(Some(device.create_compute_pipeline(
                &ComputePipelineDesc {
                    compute_bytes: cs,
                    compute_entry: "csMain",
                    push_constant_size: pcsize,
                    bindless: true,
                    threads_per_group: [8, 8, 1],
                },
            )?))
        };
        let ssr_pipeline = compute(
            dreamcoast_shader::ssr_cs_spirv,
            dreamcoast_shader::ssr_cs_dxil,
            dreamcoast_shader::ssr_cs_metallib,
            "ssr",
            192,
        )?;
        let reflect_pipeline = compute(
            dreamcoast_shader::gdf_reflect_cs_spirv,
            dreamcoast_shader::gdf_reflect_cs_dxil,
            dreamcoast_shader::gdf_reflect_cs_metallib,
            "gdf_reflect",
            176,
        )?;
        let composite_pipeline = compute(
            dreamcoast_shader::reflect_composite_cs_spirv,
            dreamcoast_shader::reflect_composite_cs_dxil,
            dreamcoast_shader::reflect_composite_cs_metallib,
            "reflect_composite",
            32,
        )?;
        Ok(Self {
            ssr_pipeline,
            reflect_pipeline,
            composite_pipeline,
        })
    }

    pub(crate) fn has_ssr(&self) -> bool {
        self.ssr_pipeline.is_some()
    }
    pub(crate) fn has_gdf_reflect(&self) -> bool {
        self.reflect_pipeline.is_some()
    }
    pub(crate) fn has_composite(&self) -> bool {
        self.composite_pipeline.is_some()
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

    /// C6: GDF reflections — the off-screen fallback for the C5 SSR misses. A full-screen
    /// compute pass reflects the view ray about each surface normal and sphere-traces it
    /// through the world scene GDF (re-lighting the hit with constant albedo + sun + sky,
    /// like the C3 GI; escapes return the procedural sky, NOT 0, since a specular miss
    /// shows the sky). `scene_gdf` / `scene_gdf_ext` are the volume + its imported graph
    /// handle (its one-time bake is recorded by the caller via `GdfSystem`). Output is raw
    /// radiance (the tonemap applies exposure). Returns the reflection image.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_gdf_reflect<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        scene_gdf: &'a Volume,
        scene_gdf_ext: ResourceId,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        depth: ResourceId,
        normal: ResourceId,
        extent: Extent2D,
        inv_view_proj: [f32; 16],
        eye: Vec3,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        cw: u32,
        ch: u32,
        flip_y: u32,
    ) -> ResourceId {
        let pipe = self
            .reflect_pipeline
            .as_ref()
            .expect("gdf reflect pipeline");
        let out = graph.create_storage_image("gdf_reflect_out", HDR_FORMAT, extent);
        let sampled = scene_gdf.sampled_index();
        let diag = {
            let d = [
                aabb_max[0] - aabb_min[0],
                aabb_max[1] - aabb_min[1],
                aabb_max[2] - aabb_min[2],
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        let bias = diag * 0.01;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gdf_reflect",
                storage_writes: vec![out],
                reads: vec![depth, normal, scene_gdf_ext],
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(scene_gdf);
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&gdf_reflect_push(
                    &inv_view_proj,
                    eye,
                    sun_dir,
                    sun_intensity,
                    depth_index,
                    normal_index,
                    sampled,
                    out_index,
                    cw,
                    ch,
                    flip_y,
                    aabb_min,
                    aabb_max,
                    0.0,  // world ground plane at y = 0
                    diag, // sample distance clamp
                    diag, // reflection ray max distance
                    0.7,  // constant hit albedo (no material in the GDF)
                    0.25, // sky fill at the reflected hit
                    bias,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    /// C7: hybrid reflection composite. A full-screen compute pass blends the C5 SSR image
    /// (`ssr`, rgb = reflected color, a = confidence) over the C6 GDF reflection image
    /// (`gdf_reflect`, sky baked in on a ray escape) by the SSR confidence — SSR where it
    /// is confident, the GDF / sky fallback elsewhere. The result is the single reflection
    /// radiance that replaces the prefilter-cube IBL specular (C7c). `gdf_scale` lifts the
    /// raw GDF radiance into the SSR's post-exposure space for the standalone viz; it is
    /// 1.0 once both sources are raw radiance (C7b). Returns the composite image.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_composite<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        ssr: ResourceId,
        gdf_reflect: ResourceId,
        extent: Extent2D,
        cw: u32,
        ch: u32,
        gdf_scale: f32,
    ) -> ResourceId {
        let pipe = self
            .composite_pipeline
            .as_ref()
            .expect("composite pipeline");
        let out = graph.create_storage_image("reflect_composite_out", HDR_FORMAT, extent);
        graph.add_compute_pass(
            ComputePassInfo {
                name: "reflect_composite",
                storage_writes: vec![out],
                reads: vec![ssr, gdf_reflect],
            },
            move |ctx| {
                let ssr_index = ctx.sampled_index(ssr);
                let gdf_index = ctx.sampled_index(gdf_reflect);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&reflect_composite_push(
                    ssr_index, gdf_index, out_index, cw, ch, gdf_scale,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }
}
