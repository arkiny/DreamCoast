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
use crate::push::{
    gdf_reflect_push, lit_history_push, reflect_composite_push, reflect_temporal_push, ssr_push,
    ssr_resolve_push,
};

pub(crate) struct ReflectSystem {
    ssr_pipeline: Option<ComputePipeline>, // C5 screen-space reflections (stochastic half-res)
    ssr_resolve_pipeline: Option<ComputePipeline>, // stochastic SSR temporal resolve
    reflect_pipeline: Option<ComputePipeline>, // C6 GDF reflection fallback
    composite_pipeline: Option<ComputePipeline>, // C7 hybrid composite
    lit_history_pipeline: Option<ComputePipeline>, // C7b lit-color history capture
    temporal_pipeline: Option<ComputePipeline>, // C8j stochastic-GDF-reflection temporal resolve
    /// C8j stochastic GDF reflection temporal accumulation: ping-pong byte-address buffers —
    /// `accum` (tonemap-space rgb + history len) and `pos` (surface world point + valid), at the
    /// full render extent. The resolve reprojects the surface into the previous frame.
    refl_accum: [Option<StorageBuffer>; 2],
    refl_pos: [Option<StorageBuffer>; 2],
    refl_accum_extent: (u32, u32),
    refl_accum_frame: u32,
    /// C7b lit-color history: ping-pong byte-address storage buffers (float4/pixel, rgb =
    /// raw radiance, a = 1), (re)allocated to the render extent. The SSR reads the previous
    /// frame's buffer (reprojected); the copy pass writes this frame's.
    lit_hist: [Option<StorageBuffer>; 2],
    lit_hist_extent: (u32, u32),
    /// Frames since the last history (re)allocation; selects the ping-pong read/write pair.
    lit_hist_frame: u32,
    /// Stochastic SSR temporal accumulation (half-res): ping-pong byte-address buffers —
    /// `accum` (rgb + confidence) and `pos` (surface world point + valid), (re)allocated to
    /// the half-res extent. The resolve reprojects the surface into the previous frame.
    ssr_accum: [Option<StorageBuffer>; 2],
    ssr_pos: [Option<StorageBuffer>; 2],
    ssr_accum_extent: (u32, u32),
    ssr_accum_frame: u32,
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
            224,
            true,
        )?;
        let ssr_resolve_pipeline = compute(
            dreamcoast_shader::ssr_resolve_cs_spirv,
            dreamcoast_shader::ssr_resolve_cs_dxil,
            dreamcoast_shader::ssr_resolve_cs_metallib,
            "ssr_resolve",
            224,
            false,
        )?;
        let reflect_pipeline = compute(
            dreamcoast_shader::gdf_reflect_cs_spirv,
            dreamcoast_shader::gdf_reflect_cs_dxil,
            dreamcoast_shader::gdf_reflect_cs_metallib,
            "gdf_reflect",
            240,
            false,
        )?;
        let composite_pipeline = compute(
            dreamcoast_shader::reflect_composite_cs_spirv,
            dreamcoast_shader::reflect_composite_cs_dxil,
            dreamcoast_shader::reflect_composite_cs_metallib,
            "reflect_composite",
            48,
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
        // C8j: temporal resolve of the stochastic GGX GDF reflection.
        let temporal_pipeline = compute(
            dreamcoast_shader::reflect_temporal_cs_spirv,
            dreamcoast_shader::reflect_temporal_cs_dxil,
            dreamcoast_shader::reflect_temporal_cs_metallib,
            "reflect_temporal",
            224,
            false,
        )?;
        Ok(Self {
            ssr_pipeline,
            ssr_resolve_pipeline,
            reflect_pipeline,
            composite_pipeline,
            lit_history_pipeline,
            temporal_pipeline,
            refl_accum: [None, None],
            refl_pos: [None, None],
            refl_accum_extent: (0, 0),
            refl_accum_frame: 0,
            lit_hist: [None, None],
            lit_hist_extent: (0, 0),
            lit_hist_frame: 0,
            ssr_accum: [None, None],
            ssr_pos: [None, None],
            ssr_accum_extent: (0, 0),
            ssr_accum_frame: 0,
        })
    }

    pub(crate) fn has_ssr_resolve(&self) -> bool {
        self.ssr_resolve_pipeline.is_some()
    }

    /// (Re)allocate the half-res stochastic-SSR accumulation buffers on a resize (resetting
    /// the ping-pong counter). Runs before the graph, like `prepare_history`. No-op without
    /// the resolve pipeline.
    pub(crate) fn prepare_ssr_accum(
        &mut self,
        device: &Device,
        hw: u32,
        hh: u32,
    ) -> anyhow::Result<()> {
        if self.ssr_resolve_pipeline.is_none() {
            return Ok(());
        }
        if self.ssr_accum_extent != (hw, hh) {
            device.wait_idle()?;
            let make = || -> anyhow::Result<Option<StorageBuffer>> {
                Ok(Some(device.create_storage_buffer(&StorageBufferDesc {
                    size: (hw as u64) * (hh as u64) * 16,
                    stride: 16,
                    indirect: false,
                })?))
            };
            self.ssr_accum = [make()?, make()?];
            self.ssr_pos = [make()?, make()?];
            self.ssr_accum_extent = (hw, hh);
            self.ssr_accum_frame = 0;
        }
        Ok(())
    }

    pub(crate) fn advance_ssr_accum(&mut self) {
        self.ssr_accum_frame = self.ssr_accum_frame.saturating_add(1);
    }

    pub(crate) fn has_reflect_temporal(&self) -> bool {
        self.temporal_pipeline.is_some()
    }

    /// C8j: (re)allocate the stochastic-GDF-reflection temporal accumulation buffers on a resize.
    /// Runs before the graph, like `prepare_history`. No-op without the temporal pipeline.
    pub(crate) fn prepare_reflect_accum(
        &mut self,
        device: &Device,
        cw: u32,
        ch: u32,
    ) -> anyhow::Result<()> {
        if self.temporal_pipeline.is_none() {
            return Ok(());
        }
        if self.refl_accum_extent != (cw, ch) {
            device.wait_idle()?;
            let make = || -> anyhow::Result<Option<StorageBuffer>> {
                Ok(Some(device.create_storage_buffer(&StorageBufferDesc {
                    size: (cw as u64) * (ch as u64) * 16,
                    stride: 16,
                    indirect: false,
                })?))
            };
            self.refl_accum = [make()?, make()?];
            self.refl_pos = [make()?, make()?];
            self.refl_accum_extent = (cw, ch);
            self.refl_accum_frame = 0;
        }
        Ok(())
    }

    pub(crate) fn advance_reflect_accum(&mut self) {
        self.refl_accum_frame = self.refl_accum_frame.saturating_add(1);
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
        material: ResourceId,
        extent: Extent2D,
        view_proj: [f32; 16],
        inv_view_proj: [f32; 16],
        eye: Vec3,
        cw: u32,
        ch: u32,
        full_cw: u32,
        full_ch: u32,
        flip_y: u32,
        frame: u32,
        max_dist: f32,
        thickness: f32,
        use_history: bool,
        neighborhood_clamp: bool,
        stochastic: bool,
    ) -> (ResourceId, ResourceId) {
        let pipe = self.ssr_pipeline.as_ref().expect("ssr pipeline");
        let out = graph.create_storage_image("ssr_out", HDR_FORMAT, extent);
        // 2nd output (stochastic ratio estimator): the ray direction + GGX pdf per pixel.
        let out_b = graph.create_storage_image("ssr_dir", HDR_FORMAT, extent);
        let hist_index = if use_history {
            let read = ((self.lit_hist_frame + 1) % 2) as usize;
            self.lit_hist[read]
                .as_ref()
                .map(|b| b.storage_index())
                .unwrap_or(u32::MAX)
        } else {
            u32::MAX
        };
        // bit1 = history mode; bit2 = neighborhood-clamp the reprojected history; bit3 = GGX
        // stochastic jitter (the temporal resolve accumulates the per-frame rays).
        let mut flags = if use_history { flip_y | 2 } else { flip_y };
        if use_history && neighborhood_clamp {
            flags |= 4;
        }
        if stochastic {
            flags |= 8;
        }
        let reads = if use_history {
            vec![depth, normal, material]
        } else {
            vec![hdr, depth, normal, material]
        };
        graph.add_compute_pass(
            ComputePassInfo {
                name: "ssr",
                storage_writes: vec![out, out_b],
                reads,
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let material_index = ctx.sampled_index(material);
                let color_index = if use_history {
                    u32::MAX
                } else {
                    ctx.sampled_index(hdr)
                };
                let out_index = ctx.storage_index(out);
                let out_b_index = ctx.storage_index(out_b);
                let cmd = ctx.cmd();
                cmd.set_globals(globals, globals_offset);
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&ssr_push(
                    &view_proj,
                    &inv_view_proj,
                    eye,
                    depth_index,
                    normal_index,
                    material_index,
                    hist_index,
                    color_index,
                    out_index,
                    cw,
                    ch,
                    full_cw,
                    full_ch,
                    flags,
                    frame,
                    max_dist,
                    thickness,
                    256.0, // screen-space DDA step cap (actual steps = min(ray screen length, cap))
                    0.1,   // edge-fade width (fraction of half-screen)
                    out_b_index,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        (out, out_b)
    }

    /// Stochastic SSR resolve (Frostbite ratio estimator): gather the half-res neighbour rays
    /// (`ssr_a` colour+conf, `ssr_b` dir+pdf), reweight each by `pdf_p(dir)/pdf_q` so the
    /// centre pixel borrows them under its own GGX lobe (roughness-adaptive, low variance per
    /// frame), then a light temporal EMA + firefly clamp. Returns the resolved half-res
    /// reflection (the composite samples it bilinearly). `prepare_ssr_accum` must have run.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_ssr_resolve<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        ssr_a: ResourceId,
        ssr_b: ResourceId,
        depth: ResourceId,
        normal: ResourceId,
        material: ResourceId,
        extent: Extent2D,
        inv_view_proj: [f32; 16],
        prev_view_proj: [f32; 16],
        eye: Vec3,
        hw: u32,
        hh: u32,
        flip_y: u32,
        reject_dist: f32,
        clamp_max: f32,
        kernel_radius: f32,
    ) -> ResourceId {
        let pipe = self
            .ssr_resolve_pipeline
            .as_ref()
            .expect("ssr resolve pipeline");
        let out = graph.create_storage_image("ssr_resolved", HDR_FORMAT, extent);
        let reset = self.ssr_accum_frame == 0;
        let read = ((self.ssr_accum_frame + 1) % 2) as usize;
        let write = (self.ssr_accum_frame % 2) as usize;
        let accum_r = self.ssr_accum[read]
            .as_ref()
            .expect("accum r")
            .storage_index();
        let accum_w = self.ssr_accum[write]
            .as_ref()
            .expect("accum w")
            .storage_index();
        let pos_r = self.ssr_pos[read].as_ref().expect("pos r").storage_index();
        let pos_w = self.ssr_pos[write].as_ref().expect("pos w").storage_index();
        let accum_w_ext = graph.import_external("ssr_accum_w");
        let pos_w_ext = graph.import_external("ssr_pos_w");
        graph.add_compute_pass(
            ComputePassInfo {
                name: "ssr_resolve",
                storage_writes: vec![out, accum_w_ext, pos_w_ext],
                reads: vec![ssr_a, ssr_b, depth, normal, material],
            },
            move |ctx| {
                let ssr_a_index = ctx.sampled_index(ssr_a);
                let ssr_b_index = ctx.sampled_index(ssr_b);
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let material_index = ctx.sampled_index(material);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&ssr_resolve_push(
                    &inv_view_proj,
                    &prev_view_proj,
                    eye,
                    ssr_a_index,
                    ssr_b_index,
                    depth_index,
                    normal_index,
                    material_index,
                    out_index,
                    accum_r,
                    accum_w,
                    pos_r,
                    pos_w,
                    hw,
                    hh,
                    flip_y,
                    u32::from(reset),
                    reject_dist,
                    0.15, // temporal EMA alpha (spatial resolve already cut the variance)
                    clamp_max,
                    kernel_radius,
                ));
                cmd.dispatch(hw.div_ceil(8), hh.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    /// C8j: temporally resolve the stochastic GGX GDF reflection. Reprojects each surface into
    /// the previous frame, EMA-accumulates the noisy single-ray sample (in tonemap space), and
    /// disocclusion-rejects. `prepare_reflect_accum` must have run this frame.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_reflect_temporal<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        refl: ResourceId,
        depth: ResourceId,
        material: ResourceId,
        extent: Extent2D,
        cw: u32,
        ch: u32,
        inv_view_proj: [f32; 16],
        prev_view_proj: [f32; 16],
        eye: Vec3,
        flip_y: u32,
        reject_dist: f32,
        max_len: f32,
        firefly_clamp: f32,
        tonemap_range: f32,
        clamp_mode: u32,
        clamp_gamma: f32,
    ) -> ResourceId {
        let pipe = self
            .temporal_pipeline
            .as_ref()
            .expect("reflect temporal pipeline");
        let out = graph.create_storage_image("reflect_resolved", HDR_FORMAT, extent);
        let reset = self.refl_accum_frame == 0;
        let read = ((self.refl_accum_frame + 1) % 2) as usize;
        let write = (self.refl_accum_frame % 2) as usize;
        let accum_r = self.refl_accum[read]
            .as_ref()
            .expect("accum r")
            .storage_index();
        let accum_w = self.refl_accum[write]
            .as_ref()
            .expect("accum w")
            .storage_index();
        let pos_r = self.refl_pos[read].as_ref().expect("pos r").storage_index();
        let pos_w = self.refl_pos[write]
            .as_ref()
            .expect("pos w")
            .storage_index();
        let accum_w_ext = graph.import_external("refl_accum_w");
        let pos_w_ext = graph.import_external("refl_pos_w");
        graph.add_compute_pass(
            ComputePassInfo {
                name: "reflect_temporal",
                storage_writes: vec![out, accum_w_ext, pos_w_ext],
                reads: vec![refl, depth, material],
            },
            move |ctx| {
                let refl_index = ctx.sampled_index(refl);
                let depth_index = ctx.sampled_index(depth);
                let material_index = ctx.sampled_index(material);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&reflect_temporal_push(
                    &inv_view_proj,
                    &prev_view_proj,
                    eye,
                    refl_index,
                    depth_index,
                    out_index,
                    accum_r,
                    accum_w,
                    pos_r,
                    pos_w,
                    cw,
                    ch,
                    flip_y,
                    u32::from(reset),
                    material_index,
                    reject_dist,
                    max_len,
                    firefly_clamp,
                    tonemap_range,
                    clamp_mode,
                    clamp_gamma,
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_lit_history<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        hdr: ResourceId,
        cw: u32,
        ch: u32,
        inv_exposure: f32,
        clamp_max: f32,
        exposure_buf: u32,
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
                    clamp_max,
                    exposure_buf,
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
        material: ResourceId,
        extent: Extent2D,
        inv_view_proj: [f32; 16],
        eye: Vec3,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        cw: u32,
        ch: u32,
        flip_y: u32,
        frame: u32,
        albedo: Option<(&'a [Volume; 3], ResourceId)>,
        cache: Option<([u32; 5], ResourceId)>,
        clip: (u32, u32),
        clip_vols: &'a [&'a Volume],
        max_steps: u32,
        cone_k: f32,
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
        let mut reads = vec![depth, normal, material, scene_gdf_ext];
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
                let material_index = ctx.sampled_index(material);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(scene_gdf);
                for v in clip_vols {
                    cmd.volume_to_sampled(v);
                }
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
                    material_index,
                    aabb_min,
                    aabb_max,
                    0.0,  // world ground plane at y = 0
                    diag, // sample distance clamp
                    diag, // reflection ray max distance
                    0.7,  // constant hit-albedo fallback (sentinel albedo => achromatic, pre-C8a)
                    0.25, // sky fill at the reflected hit
                    bias,
                    albedo_rgb,
                    frame,
                    cache_idx,
                    clip.0,
                    clip.1,
                    crate::GROUND_ALBEDO, // analytic ground material (floor reflection hits)
                    max_steps,            // D3: reflection-ray march step cap
                    cone_k,               // P3: cone-trace LOD slope (0 = legacy)
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
        material: ResourceId,
        extent: Extent2D,
        cw: u32,
        ch: u32,
        gdf_scale: f32,
        clamp_max: f32,
        max_roughness: f32,
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
                reads: vec![ssr, gdf_reflect, material],
            },
            move |ctx| {
                let ssr_index = ctx.sampled_index(ssr);
                let gdf_index = ctx.sampled_index(gdf_reflect);
                let material_index = ctx.sampled_index(material);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&reflect_composite_push(
                    ssr_index,
                    gdf_index,
                    out_index,
                    cw,
                    ch,
                    gdf_scale,
                    clamp_max,
                    material_index,
                    max_roughness,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }
}
