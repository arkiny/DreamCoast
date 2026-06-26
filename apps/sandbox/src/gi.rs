//! Phase 11 Stage C GDF-lighting consumers — split from `gdf.rs` so the distance field
//! (build + debug viz, in `GdfSystem`) is separate from the real-render features that
//! *consume* it. `GiSystem` owns the ambient-occlusion (C2), 1-bounce diffuse GI (C3),
//! and spatio-temporal denoise (C4) pipelines + the denoiser's ping-pong history. Its
//! `record_*` read the world scene GDF (passed in by the caller as a borrowed `Volume`
//! with its imported graph handle and AABB — the volume itself stays owned by
//! `GdfSystem`, which also records the one-time bake) plus the deferred G-buffer, and
//! feed the lighting pass's ambient term. Each `record_*` borrows `&'a self` for the
//! graph's lifetime, like the other bundles.

use dreamcoast_render::{ComputePassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, ComputePipeline, ComputePipelineDesc, Device, Extent2D, StorageBuffer,
    StorageBufferDesc, Volume,
};

use crate::HDR_FORMAT;
use crate::app::load_compute_shader;
use crate::push::{gdf_ao_push, gdf_atrous_push, gdf_gi_push, gdf_temporal_push};

pub(crate) struct GiSystem {
    ao_pipeline: Option<ComputePipeline>, // C2 GDF ambient occlusion
    gi_pipeline: Option<ComputePipeline>, // C3 GDF 1-bounce diffuse GI
    temporal_pipeline: Option<ComputePipeline>, // C4 temporal reprojection
    atrous_pipeline: Option<ComputePipeline>, // C4 spatial à-trous
    /// C4 GI denoiser history: ping-pong float4/pixel storage buffers — `gi_hist`
    /// (rgb = accumulated irradiance, a = history length) + `gi_pos` (xyz = the world
    /// point the sample belongs to, w = valid), (re)allocated to the render extent.
    gi_hist: [Option<StorageBuffer>; 2],
    gi_pos: [Option<StorageBuffer>; 2],
    gi_denoise_extent: (u32, u32),
    /// Frames since the last denoiser reset (0 = reset this frame, ignore history).
    gi_denoise_frame: u32,
    /// Lighting/quality key; a change (sun, spp, …) resets the accumulation.
    gi_denoise_key: Option<u64>,
}

impl GiSystem {
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
                    uniform_buffer: false,
                    threads_per_group: [8, 8, 1],
                },
            )?))
        };
        let ao_pipeline = compute(
            dreamcoast_shader::gdf_ao_cs_spirv,
            dreamcoast_shader::gdf_ao_cs_dxil,
            dreamcoast_shader::gdf_ao_cs_metallib,
            "gdf_ao",
            144,
        )?;
        let gi_pipeline = compute(
            dreamcoast_shader::gdf_gi_cs_spirv,
            dreamcoast_shader::gdf_gi_cs_dxil,
            dreamcoast_shader::gdf_gi_cs_metallib,
            "gdf_gi",
            176,
        )?;
        let temporal_pipeline = compute(
            dreamcoast_shader::gdf_temporal_cs_spirv,
            dreamcoast_shader::gdf_temporal_cs_dxil,
            dreamcoast_shader::gdf_temporal_cs_metallib,
            "gdf_temporal",
            192,
        )?;
        let atrous_pipeline = compute(
            dreamcoast_shader::gdf_atrous_cs_spirv,
            dreamcoast_shader::gdf_atrous_cs_dxil,
            dreamcoast_shader::gdf_atrous_cs_metallib,
            "gdf_atrous",
            112,
        )?;
        Ok(Self {
            ao_pipeline,
            gi_pipeline,
            temporal_pipeline,
            atrous_pipeline,
            gi_hist: [None, None],
            gi_pos: [None, None],
            gi_denoise_extent: (0, 0),
            gi_denoise_frame: 0,
            gi_denoise_key: None,
        })
    }

    pub(crate) fn has_ao(&self) -> bool {
        self.ao_pipeline.is_some()
    }
    pub(crate) fn has_gi(&self) -> bool {
        self.gi_pipeline.is_some()
    }
    pub(crate) fn has_denoise(&self) -> bool {
        self.temporal_pipeline.is_some() && self.atrous_pipeline.is_some()
    }

    /// Scene-GDF AABB diagonal — the world-unit scale for the AO reach / GI bias /
    /// denoiser sigmas.
    fn diag(aabb_min: [f32; 3], aabb_max: [f32; 3]) -> f32 {
        let d = [
            aabb_max[0] - aabb_min[0],
            aabb_max[1] - aabb_min[1],
            aabb_max[2] - aabb_min[2],
        ];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    /// Stage C2: GDF ambient occlusion. A full-screen compute pass reconstructs each
    /// pixel's world surface point from the depth G-buffer, marches the world scene GDF
    /// along the world normal, and writes an AO factor [0,1] the lighting pass multiplies
    /// into its ambient term. World position comes from depth (not the object-space
    /// position MRT) so transformed objects line up with the world GDF. `scene_gdf` /
    /// `scene_gdf_ext` are the volume + its imported graph handle (its one-time bake is
    /// recorded by the caller via `GdfSystem`). Returns the AO storage image.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_ao<'a>(
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
        cw: u32,
        ch: u32,
        flip_y: u32,
    ) -> ResourceId {
        let aop = self.ao_pipeline.as_ref().expect("gdf ao pipeline");
        let out = graph.create_storage_image("gdf_ao_out", HDR_FORMAT, extent);
        let sampled = scene_gdf.sampled_index();
        // The scene extent sets the world-unit AO scale: a fraction of the AABB diagonal
        // for the sampling reach + a small surface bias, with the clamp = full diagonal
        // (exceeds the field's true max, so a query never wrongly clamps).
        let diag = Self::diag(aabb_min, aabb_max);
        let reach = diag * 0.07;
        let bias = diag * 0.004;
        let strength = 1.6;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gdf_ao",
                storage_writes: vec![out],
                reads: vec![depth, normal, scene_gdf_ext],
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(scene_gdf);
                cmd.bind_compute_pipeline(aop);
                cmd.push_constants_compute(&gdf_ao_push(
                    &inv_view_proj,
                    depth_index,
                    normal_index,
                    sampled,
                    out_index,
                    cw,
                    ch,
                    flip_y,
                    aabb_min,
                    aabb_max,
                    0.0, // world ground plane at y = 0
                    diag,
                    reach,
                    strength,
                    bias,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    /// Stage C3: stochastic 1-bounce diffuse GI. A full-screen compute pass reconstructs
    /// each pixel's world surface from depth, casts `spp` cosine-hemisphere rays into the
    /// world scene GDF, shades the hits (constant albedo + sun + sky), and writes the mean
    /// incoming radiance (indirect irradiance) the lighting pass adds to the ambient term
    /// (× surface albedo × 1-metallic). Returns the GI storage image.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_gi<'a>(
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
        sun_dir: [f32; 3],
        sun_intensity: f32,
        cw: u32,
        ch: u32,
        flip_y: u32,
        spp: u32,
        frame: u32,
        albedo: Option<(&'a [Volume; 3], ResourceId)>,
        cache: Option<([u32; 5], ResourceId)>,
        clamp_max: f32,
    ) -> ResourceId {
        let gip = self.gi_pipeline.as_ref().expect("gdf gi pipeline");
        let out = graph.create_storage_image("gdf_gi_out", HDR_FORMAT, extent);
        let sampled = scene_gdf.sampled_index();
        let diag = Self::diag(aabb_min, aabb_max);
        let bias = diag * 0.004;
        // C8a: read the per-voxel albedo volumes (colored bounce) when present; else fall
        // back to the constant `hit_albedo` in the shader (sentinel indices). C8b3: when the
        // surface cache is bound, a hit reads its cached multibounce radiance instead.
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
                name: "gdf_gi",
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
                cmd.bind_compute_pipeline(gip);
                cmd.push_constants_compute(&gdf_gi_push(
                    &inv_view_proj,
                    sun_dir,
                    sun_intensity,
                    depth_index,
                    normal_index,
                    sampled,
                    out_index,
                    cw,
                    ch,
                    flip_y,
                    spp,
                    frame,
                    albedo_rgb,
                    aabb_min,
                    aabb_max,
                    0.0,  // world ground plane at y = 0
                    diag, // sample distance clamp
                    diag, // ray max distance (bounce reach = scene diagonal)
                    bias,
                    0.25, // sky fill radiance at the bounce hit
                    0.7,  // constant hit-albedo fallback (sentinel albedo => achromatic, pre-C8a)
                    cache_idx,
                    clamp_max,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    /// C4: (re)allocate the GI denoiser history buffers on a resize and reset the
    /// accumulation counter on a resize or lighting/quality change. Runs before the
    /// graph is built (its `wait_idle` + fallible alloc stay off the graph borrow path),
    /// mirroring `RtSystem::prepare`. No-op without the denoise pipelines.
    pub(crate) fn prepare_denoise(
        &mut self,
        device: &Device,
        cw: u32,
        ch: u32,
        reset_key: u64,
    ) -> anyhow::Result<()> {
        if self.temporal_pipeline.is_none() {
            return Ok(());
        }
        if self.gi_denoise_extent != (cw, ch) {
            device.wait_idle()?;
            let make = || -> anyhow::Result<Option<StorageBuffer>> {
                Ok(Some(device.create_storage_buffer(&StorageBufferDesc {
                    size: (cw as u64) * (ch as u64) * 16,
                    stride: 16,
                    indirect: false,
                })?))
            };
            self.gi_hist = [make()?, make()?];
            self.gi_pos = [make()?, make()?];
            self.gi_denoise_extent = (cw, ch);
            self.gi_denoise_frame = 0;
        }
        if self.gi_denoise_key != Some(reset_key) {
            self.gi_denoise_frame = 0;
            self.gi_denoise_key = Some(reset_key);
        }
        Ok(())
    }

    /// Bump the denoiser accumulation counter (end-of-frame, after submit) so the next
    /// frame reprojects history and swaps the ping-pong buffers.
    pub(crate) fn advance_denoise(&mut self) {
        self.gi_denoise_frame = self.gi_denoise_frame.saturating_add(1);
    }

    /// C4: spatio-temporal denoise of the noisy C3 GI image. A temporal pass reprojects
    /// and accumulates `gi_raw` into the ping-pong history (validated by world position),
    /// then two edge-aware à-trous passes clean the residual. Returns the denoised image
    /// the lighting pass consumes in place of the raw GI. `prepare_denoise` must have run
    /// this frame.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_denoise<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        gi_raw: ResourceId,
        depth: ResourceId,
        normal: ResourceId,
        extent: Extent2D,
        inv_view_proj: [f32; 16],
        prev_view_proj: [f32; 16],
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        cw: u32,
        ch: u32,
        flip_y: u32,
    ) -> ResourceId {
        let tempp = self.temporal_pipeline.as_ref().expect("temporal pipeline");
        let atrousp = self.atrous_pipeline.as_ref().expect("atrous pipeline");
        let frame = self.gi_denoise_frame;
        let reset = u32::from(frame == 0);
        let read = ((frame + 1) % 2) as usize;
        let write = (frame % 2) as usize;
        let hist_r = self.gi_hist[read].as_ref().expect("hist r").storage_index();
        let hist_w = self.gi_hist[write]
            .as_ref()
            .expect("hist w")
            .storage_index();
        let pos_r = self.gi_pos[read].as_ref().expect("pos r").storage_index();
        let pos_w = self.gi_pos[write].as_ref().expect("pos w").storage_index();
        let hist_w_ext = graph.import_external("gi_hist_w");
        let pos_w_ext = graph.import_external("gi_pos_w");

        let diag = Self::diag(aabb_min, aabb_max);
        let reject_dist = diag * 0.01;
        let max_hist = 64.0_f32;

        let temporal_out = graph.create_storage_image("gi_temporal", HDR_FORMAT, extent);
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gdf_temporal",
                storage_writes: vec![temporal_out, hist_w_ext, pos_w_ext],
                reads: vec![gi_raw, depth, normal],
            },
            move |ctx| {
                let gi_raw_index = ctx.sampled_index(gi_raw);
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let out_index = ctx.storage_index(temporal_out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(tempp);
                cmd.push_constants_compute(&gdf_temporal_push(
                    &inv_view_proj,
                    &prev_view_proj,
                    gi_raw_index,
                    depth_index,
                    normal_index,
                    out_index,
                    hist_r,
                    hist_w,
                    pos_r,
                    pos_w,
                    cw,
                    ch,
                    flip_y,
                    reset,
                    reject_dist,
                    max_hist,
                    1.0 / max_hist,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );

        // Two à-trous iterations (step 1 then 2): a wide edge-aware blur at low cost.
        let pos_sigma = diag * 0.03;
        let normal_power = 32.0_f32;
        let mut cur = temporal_out;
        for (i, step) in [1u32, 2u32].into_iter().enumerate() {
            let out = graph.create_storage_image(
                if i == 0 { "gi_atrous0" } else { "gi_atrous1" },
                HDR_FORMAT,
                extent,
            );
            let src = cur;
            graph.add_compute_pass(
                ComputePassInfo {
                    name: "gdf_atrous",
                    storage_writes: vec![out],
                    reads: vec![src, depth, normal],
                },
                move |ctx| {
                    let in_index = ctx.sampled_index(src);
                    let depth_index = ctx.sampled_index(depth);
                    let normal_index = ctx.sampled_index(normal);
                    let out_index = ctx.storage_index(out);
                    let cmd = ctx.cmd();
                    cmd.bind_compute_pipeline(atrousp);
                    cmd.push_constants_compute(&gdf_atrous_push(
                        &inv_view_proj,
                        in_index,
                        depth_index,
                        normal_index,
                        out_index,
                        cw,
                        ch,
                        step,
                        flip_y,
                        pos_sigma,
                        normal_power,
                    ));
                    cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                    Ok(())
                },
            );
            cur = out;
        }
        cur
    }
}
