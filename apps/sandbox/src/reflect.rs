//! Phase 11 Stage C reflection track — split from `gdf.rs` because it is screen-space,
//! not GDF-based, and is its own growing cluster: screen-space reflections (C5) and,
//! later, the GDF reflection fallback (C6) + hybrid composite (C7) that together replace
//! the captured-cube IBL specular. Each `record_*` adds one pass and returns its output
//! image, borrowing `&'a self` for the graph's lifetime like the other render bundles.

use dreamcoast_core::glam::Vec3;
use dreamcoast_render::{ComputePassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, Buffer, ComputePipeline, ComputePipelineDesc, Device, Extent2D, StorageBuffer,
    StorageBufferDesc, Volume,
};

use crate::HDR_FORMAT;
use crate::app::load_compute_shader;
use crate::push::{gdf_reflect_push, lit_history_push, reflect_composite_push, ssr_push};

pub(crate) struct ReflectSystem {
    ssr_pipeline: Option<ComputePipeline>, // C5 screen-space reflections
    reflect_pipeline: Option<ComputePipeline>, // C6 GDF reflection fallback
    composite_pipeline: Option<ComputePipeline>, // C7 hybrid composite
    lit_history_pipeline: Option<ComputePipeline>, // C7b lit-color history capture
    /// C7b lit-color history: ping-pong byte-address storage buffers (float4/pixel, rgb =
    /// raw radiance, a = 1), (re)allocated to the render extent. The SSR reads the previous
    /// frame's buffer (reprojected); the copy pass writes this frame's.
    lit_hist: [Option<StorageBuffer>; 2],
    lit_hist_extent: (u32, u32),
    /// Frames since the last history (re)allocation; selects the ping-pong read/write pair.
    lit_hist_frame: u32,
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
                       pcsize: u32,
                       uniform_buffer: bool|
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
                    uniform_buffer,
                    threads_per_group: [8, 8, 1],
                },
            )?))
        };
        // SSR binds the per-frame globals UBO (set 1 / b1) for the C7b reprojection
        // matrices (prev_view_proj) that don't fit the push-constant budget.
        let ssr_pipeline = compute(
            dreamcoast_shader::ssr_cs_spirv,
            dreamcoast_shader::ssr_cs_dxil,
            dreamcoast_shader::ssr_cs_metallib,
            "ssr",
            192,
            true,
        )?;
        let reflect_pipeline = compute(
            dreamcoast_shader::gdf_reflect_cs_spirv,
            dreamcoast_shader::gdf_reflect_cs_dxil,
            dreamcoast_shader::gdf_reflect_cs_metallib,
            "gdf_reflect",
            224,
            false,
        )?;
        let composite_pipeline = compute(
            dreamcoast_shader::reflect_composite_cs_spirv,
            dreamcoast_shader::reflect_composite_cs_dxil,
            dreamcoast_shader::reflect_composite_cs_metallib,
            "reflect_composite",
            32,
            false,
        )?;
        let lit_history_pipeline = compute(
            dreamcoast_shader::lit_history_cs_spirv,
            dreamcoast_shader::lit_history_cs_dxil,
            dreamcoast_shader::lit_history_cs_metallib,
            "lit_history",
            32,
            false,
        )?;
        Ok(Self {
            ssr_pipeline,
            reflect_pipeline,
            composite_pipeline,
            lit_history_pipeline,
            lit_hist: [None, None],
            lit_hist_extent: (0, 0),
            lit_hist_frame: 0,
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
    pub(crate) fn has_lit_history(&self) -> bool {
        self.lit_history_pipeline.is_some()
    }

    /// C7b: (re)allocate the lit-color history buffers on a resize (resetting the ping-pong
    /// counter). Runs before the graph is built (its `wait_idle` + fallible alloc stay off
    /// the graph borrow path), mirroring `GiSystem::prepare_denoise`. No-op without the
    /// history pipeline.
    pub(crate) fn prepare_history(
        &mut self,
        device: &Device,
        cw: u32,
        ch: u32,
    ) -> anyhow::Result<()> {
        if self.lit_history_pipeline.is_none() {
            return Ok(());
        }
        if self.lit_hist_extent != (cw, ch) {
            device.wait_idle()?;
            let make = || -> anyhow::Result<Option<StorageBuffer>> {
                Ok(Some(device.create_storage_buffer(&StorageBufferDesc {
                    size: (cw as u64) * (ch as u64) * 16,
                    stride: 16,
                    indirect: false,
                })?))
            };
            self.lit_hist = [make()?, make()?];
            self.lit_hist_extent = (cw, ch);
            self.lit_hist_frame = 0;
        }
        Ok(())
    }

    /// Bump the history ping-pong counter (end-of-frame, after submit) so the next frame
    /// reads the buffer this frame wrote.
    pub(crate) fn advance_history(&mut self) {
        self.lit_hist_frame = self.lit_hist_frame.saturating_add(1);
    }

    /// C5/C7b: screen-space reflections. A full-screen compute pass reflects the view ray
    /// about each surface normal and marches it through the depth buffer. The color source
    /// depends on `use_history`:
    ///   * `false` (standalone C5 viz): samples this frame's lit HDR at the hit (post-
    ///     exposure), so reflective surfaces show real neighbouring geometry.
    ///   * `true` (C7b, feeds lighting): reprojects the world hit into the previous frame
    ///     (via `globals.prev_view_proj`) and samples the raw-radiance lit-color history,
    ///     so SSR can feed back into the lighting specular (C7c) without a read-before-write
    ///     cycle. `prepare_history` must have run this frame.
    ///
    /// Binds the per-frame `globals` UBO (for the reprojection matrix) via `set_globals`.
    /// Returns the reflection image (rgb = reflected color, a = confidence; misses are 0).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_ssr<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        globals: &'a Buffer,
        globals_offset: u64,
        hdr: ResourceId,
        depth: ResourceId,
        normal: ResourceId,
        extent: Extent2D,
        view_proj: [f32; 16],
        inv_view_proj: [f32; 16],
        eye: Vec3,
        cw: u32,
        ch: u32,
        flip_y: u32,
        max_dist: f32,
        thickness: f32,
        use_history: bool,
    ) -> ResourceId {
        let pipe = self.ssr_pipeline.as_ref().expect("ssr pipeline");
        let out = graph.create_storage_image("ssr_out", HDR_FORMAT, extent);
        // History mode: read the previous frame's buffer (ping-pong), set the history flag
        // bit. The buffer was written last frame (cross-frame sync via the frame fence), so
        // it is bound by index like the C4 denoiser history — not a graph resource.
        let hist_index = if use_history {
            let read = ((self.lit_hist_frame + 1) % 2) as usize;
            self.lit_hist[read]
                .as_ref()
                .map(|b| b.storage_index())
                .unwrap_or(u32::MAX)
        } else {
            u32::MAX
        };
        let flags = if use_history { flip_y | 2 } else { flip_y };
        // History mode reads the buffer (not the HDR), so the HDR is only a graph read in
        // the current-frame viz path.
        let reads = if use_history {
            vec![depth, normal]
        } else {
            vec![hdr, depth, normal]
        };
        graph.add_compute_pass(
            ComputePassInfo {
                name: "ssr",
                storage_writes: vec![out],
                reads,
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let color_index = if use_history {
                    u32::MAX
                } else {
                    ctx.sampled_index(hdr)
                };
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.set_globals(globals, globals_offset);
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&ssr_push(
                    &view_proj,
                    &inv_view_proj,
                    eye,
                    depth_index,
                    normal_index,
                    hist_index,
                    color_index,
                    out_index,
                    cw,
                    ch,
                    flags,
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

    /// C7b: capture this frame's lit HDR into the ping-pong history buffer (raw radiance =
    /// `hdr * inv_exposure`) so the next frame's history-mode SSR can sample it. Runs after
    /// the lighting pass. `prepare_history` must have run this frame.
    pub(crate) fn record_lit_history<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        hdr: ResourceId,
        cw: u32,
        ch: u32,
        inv_exposure: f32,
    ) {
        let pipe = self
            .lit_history_pipeline
            .as_ref()
            .expect("lit history pipeline");
        let write = (self.lit_hist_frame % 2) as usize;
        let out_buffer = self.lit_hist[write]
            .as_ref()
            .map(|b| b.storage_index())
            .unwrap_or(u32::MAX);
        let hist_w_ext = graph.import_external("lit_hist_w");
        graph.add_compute_pass(
            ComputePassInfo {
                name: "lit_history",
                storage_writes: vec![hist_w_ext],
                reads: vec![hdr],
            },
            move |ctx| {
                let hdr_index = ctx.sampled_index(hdr);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&lit_history_push(
                    hdr_index,
                    out_buffer,
                    cw,
                    ch,
                    inv_exposure,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
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
        albedo: Option<(&'a [Volume; 3], ResourceId)>,
        cache: Option<([u32; 5], ResourceId)>,
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
        // C8a: read the per-voxel albedo volumes (colored reflections) when present; else
        // the shader's constant `hit_albedo` fallback (sentinel indices).
        let mut reads = vec![depth, normal, scene_gdf_ext];
        if let Some((_, ext)) = albedo {
            reads.push(ext);
        }
        if let Some((_, ext)) = cache {
            reads.push(ext);
        }
        let cache_idx = cache.map(|(idx, _)| idx).unwrap_or([u32::MAX; 5]);
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gdf_reflect",
                storage_writes: vec![out],
                reads,
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(scene_gdf);
                let albedo_rgb = if let Some((vols, _)) = albedo {
                    for v in vols.iter() {
                        cmd.volume_to_sampled(v);
                    }
                    [
                        vols[0].sampled_index(),
                        vols[1].sampled_index(),
                        vols[2].sampled_index(),
                    ]
                } else {
                    [u32::MAX; 3]
                };
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
                    0.7,  // constant hit-albedo fallback (sentinel albedo => achromatic, pre-C8a)
                    0.25, // sky fill at the reflected hit
                    bias,
                    albedo_rgb,
                    cache_idx,
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
