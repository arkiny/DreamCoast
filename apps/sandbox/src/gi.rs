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
    BackendKind, ComputePipeline, ComputePipelineDesc, Device, Extent2D, Format, StorageBuffer,
    StorageBufferDesc, Volume, VolumeDesc,
};

use crate::HDR_FORMAT;
use crate::app::load_compute_shader;
use crate::push::{
    gdf_ao_push, gdf_atrous_push, gdf_gi_push, gdf_gi_upsample_push, gdf_temporal_push,
    gi_volume_push, screen_probe_filter_push, screen_probe_integrate_push,
    screen_probe_irradiance_push, screen_probe_trace_push, wrc_update_push, wrc_view_push,
};

/// Screen-space radiance probe density: one probe per `SP_DOWNSAMPLE`x`SP_DOWNSAMPLE` screen
/// tile (reference uses ~16). Tunable later via a `RenderQuality` tier.
const SP_DOWNSAMPLE: u32 = 16;
/// Octahedral radiance-tile resolution per probe (texels per side). Reference starts 8.
const SP_OCT_RES: u32 = 8;

/// World radiance cache (P4) probe grid resolution per clipmap level (probes per side).
const WRC_GRID: u32 = 16;
/// World radiance cache octahedral tile resolution (texels per side).
const WRC_OCT: u32 = 8;

/// World-space directional-irradiance volume (radiance-cache) probe-grid resolution.
const GI_VOL_DIM: u32 = 32;

/// SH band-0/1 coefficient count per probe channel-set: 4 coeffs × RGB = 12 R32F volumes per
/// ping-pong slot (slot index = channel*4 + coeff). Allocated contiguously so the shaders take
/// only the base bindless index. See `gi_volume.slang`.
const GI_VOL_SH: usize = 12;

/// Scalar sky-visibility SH band-0/1 coefficient count per ping-pong slot (4 R32F volumes,
/// contiguous; base index only). 레퍼런스식 indoor skylight occlusion. See `gi_volume.slang`.
const GI_SKYVIS_SH: usize = 4;

pub(crate) struct GiSystem {
    ao_pipeline: Option<ComputePipeline>, // C2 GDF ambient occlusion
    gi_pipeline: Option<ComputePipeline>, // C3 GDF 1-bounce diffuse GI (SW march — the default)
    /// F3: the `gdf_gi_hwrt` permutation (hardware-ray-traced gather compiled in). Built only on
    /// RT-capable devices; bound in place of `gi_pipeline` when HW-RT GI is opted in. The default
    /// `gi_pipeline` above has no acceleration-structure reference, so the SW path is RT-independent.
    gi_hwrt_pipeline: Option<ComputePipeline>,
    upsample_pipeline: Option<ComputePipeline>, // D1 half-res GI joint-bilateral upsample
    temporal_pipeline: Option<ComputePipeline>, // C4 temporal reprojection
    atrous_pipeline: Option<ComputePipeline>,   // C4 spatial à-trous
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
    /// GI-fidelity track: world-space directional-irradiance volume (radiance cache). Update
    /// pipeline + 24 R32F volumes (ping-pong [read|write] × 12 SH coefficients). Allocated
    /// contiguously per slot so only the base bindless index is passed to the shaders.
    gi_vol_pipeline: Option<ComputePipeline>,
    gi_vol: [[Option<Volume>; GI_VOL_SH]; 2],
    /// Screen-space radiance probes (P1+): per-tile probe trace into an octahedral radiance
    /// atlas, then a per-pixel gather of that atlas into indirect irradiance. Replaces the
    /// world-volume / ray-march GI consumption on content scenes (opt-in `SCREEN_PROBE`).
    sp_trace_pipeline: Option<ComputePipeline>,
    sp_integrate_pipeline: Option<ComputePipeline>,
    /// P2 spatial cross-probe joint-bilateral filter of the radiance atlas (optional).
    sp_filter_pipeline: Option<ComputePipeline>,
    /// P5 per-probe radiance->irradiance pre-integration (makes the per-pixel gather a cheap lookup).
    sp_irradiance_pipeline: Option<ComputePipeline>,
    /// P4 world radiance cache: update pipeline + ping-pong atlas buffers (octahedral radiance
    /// tiles for `levels * WRC_GRID^3` clipmap probes, 16 B/texel). Persists across frames for
    /// EMA accumulation + infinite bounce; (re)allocated when the clipmap level count changes.
    wrc_pipeline: Option<ComputePipeline>,
    wrc_atlas: [Option<StorageBuffer>; 2],
    wrc_levels: u32,
    wrc_frame: u32,
    /// GI-on-distance-field visualization: march the camera into the GDF, paint hits with the
    /// world radiance cache's stored indirect irradiance.
    wrc_view_pipeline: Option<ComputePipeline>,
    /// 레퍼런스식 indoor skylight occlusion: directional sky-visibility SH (4 scalar coeffs/slot,
    /// ping-pong = 8 volumes), filled in the same `gi_volume` pass. Contiguous per slot (base only).
    gi_skyvis: [[Option<Volume>; GI_SKYVIS_SH]; 2],
    gi_vol_frame: u32,
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
            160,
        )?;
        let gi_pipeline = compute(
            dreamcoast_shader::gdf_gi_cs_spirv,
            dreamcoast_shader::gdf_gi_cs_dxil,
            dreamcoast_shader::gdf_gi_cs_metallib,
            "gdf_gi",
            256, // +16B row: F3 `hwrt` @240 (SW variant ignores it); default = SW march.
        )?;
        // F3: the HW-RT permutation, built only on RT-capable devices (its inline RayQuery /
        // acceleration-structure use can't be created without RT support). Bound in place of
        // `gi_pipeline` when HW-RT GI is opted in; absent ⇒ the SW default is always used.
        let gi_hwrt_pipeline = if compute_supported && device.has_raytracing() {
            compute(
                dreamcoast_shader::gdf_gi_hwrt_cs_spirv,
                dreamcoast_shader::gdf_gi_hwrt_cs_dxil,
                dreamcoast_shader::gdf_gi_hwrt_cs_metallib,
                "gdf_gi_hwrt",
                256,
            )?
        } else {
            None
        };
        let upsample_pipeline = compute(
            dreamcoast_shader::gdf_gi_upsample_cs_spirv,
            dreamcoast_shader::gdf_gi_upsample_cs_dxil,
            dreamcoast_shader::gdf_gi_upsample_cs_metallib,
            "gdf_gi_upsample",
            128,
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
        let gi_vol_pipeline = compute(
            dreamcoast_shader::gi_volume_cs_spirv,
            dreamcoast_shader::gi_volume_cs_dxil,
            dreamcoast_shader::gi_volume_cs_metallib,
            "gi_volume",
            160,
        )?;
        let sp_trace_pipeline = compute(
            dreamcoast_shader::screen_probe_trace_cs_spirv,
            dreamcoast_shader::screen_probe_trace_cs_dxil,
            dreamcoast_shader::screen_probe_trace_cs_metallib,
            "screen_probe_trace",
            240, // ProbeTracePush = 240B (wrc_atlas/grid/oct/pad0 row @224); matches screen_probe_trace_push.
        )?;
        let sp_integrate_pipeline = compute(
            dreamcoast_shader::screen_probe_integrate_cs_spirv,
            dreamcoast_shader::screen_probe_integrate_cs_dxil,
            dreamcoast_shader::screen_probe_integrate_cs_metallib,
            "screen_probe_integrate",
            128,
        )?;
        let sp_filter_pipeline = compute(
            dreamcoast_shader::screen_probe_filter_cs_spirv,
            dreamcoast_shader::screen_probe_filter_cs_dxil,
            dreamcoast_shader::screen_probe_filter_cs_metallib,
            "screen_probe_filter",
            128,
        )?;
        let sp_irradiance_pipeline = compute(
            dreamcoast_shader::screen_probe_irradiance_cs_spirv,
            dreamcoast_shader::screen_probe_irradiance_cs_dxil,
            dreamcoast_shader::screen_probe_irradiance_cs_metallib,
            "screen_probe_irradiance",
            32,
        )?;
        let wrc_pipeline = compute(
            dreamcoast_shader::wrc_update_cs_spirv,
            dreamcoast_shader::wrc_update_cs_dxil,
            dreamcoast_shader::wrc_update_cs_metallib,
            "wrc_update",
            128,
        )?;
        let wrc_view_pipeline = compute(
            dreamcoast_shader::wrc_view_cs_spirv,
            dreamcoast_shader::wrc_view_cs_dxil,
            dreamcoast_shader::wrc_view_cs_metallib,
            "wrc_view",
            192,
        )?;
        // 24 R32F volumes: ping-pong [read|write] × 12 SH coefficients. Empty (zero) at start = no
        // fill until the update converges; the lighting falls back gracefully (e = 0). Allocated
        // back-to-back so each slot's 12 volumes are contiguous in the bindless sampled AND storage
        // tables (`create_volume` bumps both index counters by one) — the shaders address them as
        // `base + channel*4 + coeff`, so only the base index is pushed.
        let mut gi_vol: [[Option<Volume>; GI_VOL_SH]; 2] = Default::default();
        let mut gi_skyvis: [[Option<Volume>; GI_SKYVIS_SH]; 2] = Default::default();
        if gi_vol_pipeline.is_some() {
            let vd = VolumeDesc {
                width: GI_VOL_DIM,
                height: GI_VOL_DIM,
                depth: GI_VOL_DIM,
                format: Format::R32Float,
            };
            for set in gi_vol.iter_mut() {
                for ch in set.iter_mut() {
                    *ch = Some(device.create_volume(&vd)?);
                }
            }
            for set in gi_skyvis.iter_mut() {
                for ch in set.iter_mut() {
                    *ch = Some(device.create_volume(&vd)?);
                }
            }
            // The base-index addressing is only valid if each slot's volumes are contiguous in both
            // the sampled and storage bindless tables; assert it so a future interleaving allocation
            // can't silently break it.
            let check = |sets: &[&[Option<Volume>]]| {
                for set in sets {
                    let base_s = set[0].as_ref().unwrap().sampled_index();
                    let base_u = set[0].as_ref().unwrap().storage_index();
                    for (i, ch) in set.iter().enumerate() {
                        let v = ch.as_ref().unwrap();
                        debug_assert_eq!(v.sampled_index(), base_s + i as u32);
                        debug_assert_eq!(v.storage_index(), base_u + i as u32);
                    }
                }
            };
            check(&[&gi_vol[0], &gi_vol[1], &gi_skyvis[0], &gi_skyvis[1]]);
        }
        Ok(Self {
            ao_pipeline,
            gi_pipeline,
            gi_hwrt_pipeline,
            upsample_pipeline,
            temporal_pipeline,
            atrous_pipeline,
            gi_hist: [None, None],
            gi_pos: [None, None],
            gi_denoise_extent: (0, 0),
            gi_denoise_frame: 0,
            gi_denoise_key: None,
            gi_vol_pipeline,
            gi_vol,
            gi_skyvis,
            gi_vol_frame: 0,
            sp_trace_pipeline,
            sp_integrate_pipeline,
            sp_filter_pipeline,
            sp_irradiance_pipeline,
            wrc_pipeline,
            wrc_atlas: [None, None],
            wrc_levels: 0,
            wrc_frame: 0,
            wrc_view_pipeline,
        })
    }

    pub(crate) fn has_gi_volume(&self) -> bool {
        self.gi_vol_pipeline.is_some()
    }

    /// The sampled base indices the GI pass should READ this frame — the slot the update pass wrote
    /// (write slot = frame % 2). Returns `(radiance_base, skyvis_base)`; each set is contiguous from
    /// its base. `None` if the volume isn't built.
    pub(crate) fn gi_volume_sampled(&self) -> Option<(u32, u32)> {
        let w = (self.gi_vol_frame % 2) as usize;
        Some((
            self.gi_vol[w][0].as_ref()?.sampled_index(),
            self.gi_skyvis[w][0].as_ref()?.sampled_index(),
        ))
    }

    /// Advance the volume ping-pong (end-of-frame, after submit), like the denoiser counter.
    pub(crate) fn advance_gi_volume(&mut self) {
        self.gi_vol_frame = self.gi_vol_frame.saturating_add(1);
    }

    /// The PREVIOUS (completed) frame's sky-visibility SH base, for consumers recorded BEFORE this
    /// frame's volume update — the surface-cache relight's deferred-parity skylight. That slot was
    /// transitioned back to sampled at the end of its update and is not written this frame, so it
    /// reads cleanly with one frame of latency (hidden by the cache's EMA, like the async relight).
    /// `None` until the first update has landed (the slot's contents are undefined before then).
    pub(crate) fn gi_skyvis_prev_sampled(&self) -> Option<u32> {
        if self.gi_vol_frame == 0 {
            return None;
        }
        let prev = ((self.gi_vol_frame + 1) % 2) as usize;
        self.gi_skyvis[prev][0].as_ref().map(|v| v.sampled_index())
    }

    /// 레퍼런스 엔진 GI-fidelity track: update the world irradiance volume (DDGI-lite). Each probe casts
    /// sphere rays into the scene GDF, shades hits (direct + the PREVIOUS volume = multibounce),
    /// EMA-accumulates into the write slot. Returns the write graph handle (a read dep for the GI
    /// pass that samples it). `None` without the pipeline/volumes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_gi_volume<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        scene_gdf: &'a Volume,
        scene_gdf_ext: ResourceId,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        sun_dir: [f32; 3],
        sun_intensity: f32,
        sky_gain: f32,
        clip: (u32, u32),
        clip_vols: &'a [&'a Volume],
        albedo: Option<(&'a [Volume; 3], ResourceId)>,
        ground_albedo: [f32; 3],
        frame: u32,
        spp: u32,
        alpha: f32,
    ) -> Option<ResourceId> {
        let pipe = self.gi_vol_pipeline.as_ref()?;
        let read = ((self.gi_vol_frame + 1) % 2) as usize;
        let write = (self.gi_vol_frame % 2) as usize;
        let rv = &self.gi_vol[read];
        let wv = &self.gi_vol[write];
        let sv_r = &self.gi_skyvis[read];
        let sv_w = &self.gi_skyvis[write];
        // Contiguous bases: the previous slot's sampled base (multibounce + EMA read) and this
        // slot's storage base (write). The SH volumes follow each base in order.
        let read_base = rv[0].as_ref()?.sampled_index();
        let write_base = wv[0].as_ref()?.storage_index();
        let skyvis_read_base = sv_r[0].as_ref()?.sampled_index();
        let skyvis_write_base = sv_w[0].as_ref()?.storage_index();
        let diag = Self::diag(aabb_min, aabb_max);
        let reset = u32::from(self.gi_vol_frame == 0);
        let vol_ext = graph.import_external("gi_volume_w");
        let mut reads = vec![scene_gdf_ext];
        if let Some((_, ext)) = albedo {
            reads.push(ext);
        }
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gi_volume",
                storage_writes: vec![vol_ext],
                reads,
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(scene_gdf);
                for v in clip_vols {
                    cmd.volume_to_sampled(v);
                }
                for ch in rv.iter().flatten() {
                    cmd.volume_to_sampled(ch);
                }
                for ch in sv_r.iter().flatten() {
                    cmd.volume_to_sampled(ch);
                }
                let albedo_idx = if let Some((vols, _)) = albedo {
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
                for ch in wv.iter().flatten() {
                    cmd.volume_to_storage(ch);
                }
                for ch in sv_w.iter().flatten() {
                    cmd.volume_to_storage(ch);
                }
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&gi_volume_push(
                    aabb_min,
                    0.0, // ground plane y = 0
                    aabb_max,
                    diag, // sample distance clamp
                    sun_dir,
                    sun_intensity,
                    [GI_VOL_DIM, GI_VOL_DIM, GI_VOL_DIM],
                    frame,
                    [read_base, skyvis_read_base, 0], // x = radiance SH base, y = sky-vis SH base
                    reset,
                    [write_base, skyvis_write_base, 0], // x = radiance SH base, y = sky-vis SH base
                    albedo_idx,
                    clip.0,
                    clip.1,
                    spp as f32,
                    diag,     // ray max distance = scene diagonal
                    sky_gain, // sky gain -> procedural_sky fill at bounce hits (was flat 0.4)
                    alpha,    // EMA alpha
                    ground_albedo,
                    diag * 0.01, // surface bias
                ));
                let g = GI_VOL_DIM.div_ceil(4);
                cmd.dispatch(g, g, g);
                // Transition the just-written volumes back to sampled so the GI pass can read them.
                for ch in wv.iter().flatten() {
                    cmd.volume_to_sampled(ch);
                }
                for ch in sv_w.iter().flatten() {
                    cmd.volume_to_sampled(ch);
                }
                Ok(())
            },
        );
        Some(vol_ext)
    }

    pub(crate) fn has_ao(&self) -> bool {
        self.ao_pipeline.is_some()
    }
    pub(crate) fn has_gi(&self) -> bool {
        self.gi_pipeline.is_some()
    }
    pub(crate) fn has_upsample(&self) -> bool {
        self.upsample_pipeline.is_some()
    }
    pub(crate) fn has_screen_probe(&self) -> bool {
        self.sp_trace_pipeline.is_some() && self.sp_integrate_pipeline.is_some()
    }
    pub(crate) fn has_wrc(&self) -> bool {
        self.wrc_pipeline.is_some()
    }
    pub(crate) fn has_wrc_view(&self) -> bool {
        self.wrc_view_pipeline.is_some() && self.wrc_pipeline.is_some()
    }

    /// GI-on-distance-field visualization: a full-screen pass that marches the camera ray into
    /// the scene GDF and paints each hit with the world radiance cache's stored indirect
    /// irradiance (reconstructed for the hit normal). `wrc_atlas`/`wrc_ext` are the cache the
    /// update wrote this frame (the handle orders this pass after it). `mode` 0 = irradiance
    /// grayscale, 1 = irradiance × clay albedo. Returns the raw-radiance image (host tonemaps).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_wrc_view<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        scene_gdf: &'a Volume,
        extent: Extent2D,
        cam_pos: [f32; 3],
        inv_view_proj: [f32; 16],
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        cw: u32,
        ch: u32,
        flip_y: u32,
        clip: (u32, u32),
        clip_vols: &'a [&'a Volume],
        wrc_atlas: u32,
        wrc_ext: ResourceId,
        mode: u32,
        gain: f32,
        // Shading source: 0 = world radiance cache (coarse probes); 1 = surface cache (high-res
        // mesh cards, final lit radiance). `surface_cache` = its `(indices, lit graph handle)`; the
        // handle orders the view after the cache re-light. `None` => world-cache source only.
        source: u32,
        surface_cache: Option<([u32; 5], ResourceId)>,
    ) -> ResourceId {
        let pipe = self.wrc_view_pipeline.as_ref().expect("wrc view pipeline");
        let diag = Self::diag(aabb_min, aabb_max);
        let out = graph.create_storage_image("wrc_view_out", HDR_FORMAT, extent);
        let sc = surface_cache.map(|(idx, _)| idx).unwrap_or([u32::MAX; 5]);
        let mut reads = vec![wrc_ext];
        if let Some((_, ext)) = surface_cache {
            reads.push(ext); // order the view after this frame's surface-cache re-light
        }
        graph.add_compute_pass(
            ComputePassInfo {
                name: "wrc_view",
                storage_writes: vec![out],
                reads,
            },
            move |ctx| {
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(scene_gdf);
                for v in clip_vols {
                    cmd.volume_to_sampled(v);
                }
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&wrc_view_push(
                    &inv_view_proj,
                    cam_pos,
                    aabb_min,
                    0.0, // world ground plane y = 0
                    aabb_max,
                    diag,            // GDF sample distance clamp
                    [0.5, 0.5, 0.5], // neutral clay albedo for the lit-clay look (mode 1)
                    gain,
                    out_index,
                    cw,
                    ch,
                    flip_y,
                    clip.0,
                    clip.1,
                    wrc_atlas,
                    WRC_GRID,
                    WRC_OCT,
                    mode,
                    source,
                    sc,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    /// (Re)allocate the world radiance cache ping-pong atlas buffers for `levels` clipmap levels
    /// (each `WRC_GRID^3` probes of `WRC_OCT^2` octahedral texels, 16 B/texel). Runs before the
    /// graph (its `wait_idle` + alloc stay off the graph borrow), like `prepare_denoise`. No-op
    /// without the pipeline or when the level count is unchanged.
    pub(crate) fn prepare_wrc(&mut self, device: &Device, levels: u32) -> anyhow::Result<()> {
        if self.wrc_pipeline.is_none() || levels == 0 {
            return Ok(());
        }
        if self.wrc_levels != levels {
            device.wait_idle()?;
            let atlas_w = WRC_GRID * WRC_OCT;
            let atlas_h = levels * WRC_GRID * WRC_GRID * WRC_OCT;
            let bytes = (atlas_w as u64) * (atlas_h as u64) * 16;
            let make = || -> anyhow::Result<Option<StorageBuffer>> {
                Ok(Some(device.create_storage_buffer(&StorageBufferDesc {
                    size: bytes,
                    stride: 16,
                    indirect: false,
                })?))
            };
            self.wrc_atlas = [make()?, make()?];
            self.wrc_levels = levels;
            self.wrc_frame = 0;
        }
        Ok(())
    }

    /// Advance the world-cache ping-pong (end-of-frame, after submit).
    pub(crate) fn advance_wrc(&mut self) {
        self.wrc_frame = self.wrc_frame.saturating_add(1);
    }

    /// P4: update the world radiance cache. Every clipmap probe re-traces its octahedral
    /// directions into the scene GDF (shared bounce tracer) and EMA-accumulates into the write
    /// atlas; escaped rays sample the previous atlas (infinite bounce + far-field). Returns the
    /// write atlas `(storage_buffers[] index, graph write handle)` for the screen probes to
    /// sample (the handle orders their trace after this update). `None` without the cache.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_wrc_update<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        scene_gdf: &'a Volume,
        scene_gdf_ext: ResourceId,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        sun_dir: [f32; 3],
        sun_intensity: f32,
        clip: (u32, u32),
        clip_vols: &'a [&'a Volume],
        albedo: Option<(&'a [Volume; 3], ResourceId)>,
        cache: Option<([u32; 5], ResourceId)>,
        max_steps: u32,
        cone_k: f32,
        alpha: f32,
    ) -> Option<(u32, ResourceId)> {
        let pipe = self.wrc_pipeline.as_ref()?;
        let levels = clip.1.max(1);
        let write = (self.wrc_frame % 2) as usize;
        let prev = ((self.wrc_frame + 1) % 2) as usize;
        let write_idx = self.wrc_atlas[write].as_ref()?.storage_index();
        let prev_idx = if self.wrc_frame == 0 {
            u32::MAX
        } else {
            self.wrc_atlas[prev].as_ref()?.storage_index()
        };
        let reset = u32::from(self.wrc_frame == 0);
        let diag = Self::diag(aabb_min, aabb_max);
        let bias = diag * 0.004;
        let cache_idx = cache.map(|(idx, _)| idx).unwrap_or([u32::MAX; 5]);
        let cache4 = [cache_idx[0], cache_idx[1], cache_idx[2], cache_idx[3]];
        let cache_tile = cache_idx[4];
        let atlas_w = WRC_GRID * WRC_OCT;
        let atlas_h = levels * WRC_GRID * WRC_GRID * WRC_OCT;
        let write_ext = graph.import_external("wrc_atlas_w");
        let mut reads = vec![scene_gdf_ext];
        if let Some((_, ext)) = albedo {
            reads.push(ext);
        }
        if let Some((_, ext)) = cache {
            reads.push(ext);
        }
        graph.add_compute_pass(
            ComputePassInfo {
                name: "wrc_update",
                storage_writes: vec![write_ext],
                reads,
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(scene_gdf);
                for v in clip_vols {
                    cmd.volume_to_sampled(v);
                }
                if let Some((vols, _)) = albedo {
                    for v in vols.iter() {
                        cmd.volume_to_sampled(v);
                    }
                }
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&wrc_update_push(
                    sun_dir,
                    sun_intensity,
                    diag, // ray max distance
                    bias,
                    0.25, // sky fill radiance at the bounce hit
                    0.7,  // constant hit-albedo fallback
                    crate::GROUND_ALBEDO,
                    cone_k,
                    cache4,
                    clip.0,
                    levels,
                    WRC_GRID,
                    WRC_OCT,
                    write_idx,
                    prev_idx,
                    cache_tile,
                    max_steps,
                    self.wrc_frame,
                    reset,
                    alpha,
                    diag, // GDF sample distance clamp
                    0.0,  // world ground plane y = 0
                ));
                cmd.dispatch(atlas_w.div_ceil(8), atlas_h.div_ceil(8), 1);
                Ok(())
            },
        );
        Some((write_idx, write_ext))
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
        clip: (u32, u32),
        clip_vols: &'a [&'a Volume],
    ) -> ResourceId {
        let aop = self.ao_pipeline.as_ref().expect("gdf ao pipeline");
        let out = graph.create_storage_image("gdf_ao_out", HDR_FORMAT, extent);
        let sampled = scene_gdf.sampled_index();
        // AO is a LOCAL contact effect at a fixed physical scale (1 unit = 1 m), not a fraction of
        // the whole scene, so the reach is a metric distance (capped) regardless of scene size —
        // `diag * 0.07` alone would make a big scene (Sponza diag ~37 -> 2.6 m) treat every wall
        // within 2.6 m as an occluder. The exponential falloff (gdf_ao.slang) never crushes to pure
        // black. Strength `1.5`: the earlier `3.0` was pushed up while the AO was unreliable (the
        // G-buffer depth it samples was being discarded/cleared — see `fix/gbuffer-depth-store`);
        // now that it applies correctly every frame, 3.0 over-darkened the interior, so it is dialled
        // back to a subtle, natural contact occlusion. `AO_STRENGTH` / `AO_REACH` tune it.
        let diag = Self::diag(aabb_min, aabb_max);
        // Reach capped at 0.5 m: a contact-scale band so the occlusion hugs the contact line
        // (a column meeting the floor) instead of smearing ~1 m up the column.
        let reach = std::env::var("AO_REACH")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or((diag * 0.07).min(0.5));
        let bias = (diag * 0.004).min(0.02);
        // Strength `2.0`: `1.5` (set when the depth fix first made AO reliable) read too faint in the
        // interiors, so it is nudged up for a more present contact shade — still well below the `3.0`
        // that over-darkened the scene. (AO is slated for a `RenderQuality`/UI parameterization; this
        // is just a better default in the meantime.) `AO_STRENGTH` overrides.
        let strength = std::env::var("AO_STRENGTH")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(2.0);
        // AO floor: deep contacts keep at least this fraction of the ambient (gdf_ao.slang remaps
        // to [floor, 1]) so column bases / recesses read as soft shade, not near-black. Lowered
        // `0.4 -> 0.3` so the deepest contacts can read a touch darker without crushing to black.
        let floor = std::env::var("AO_FLOOR")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.3);
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
                for v in clip_vols {
                    cmd.volume_to_sampled(v);
                }
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
                    floor,
                    clip.0,
                    clip.1,
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
        clip: (u32, u32),
        clip_vols: &'a [&'a Volume],
        max_steps: u32,
        cone_k: f32,
        // GI-fidelity: when present, the GI pass SAMPLES this directional-irradiance volume
        // `(radiance_SH_base, skyvis_SH_base, update-pass write handle)` and reconstructs E(n) +
        // the sky-visibility instead of marching rays.
        gi_volume: Option<(u32, u32, ResourceId)>,
        // F3 (HW-RT high-fidelity path, first increment): when true (High tier, `P_HWRT_GI=1`) the
        // GI gather rays trace the scene TLAS with an inline RayQuery and return a hardware-traced
        // visibility term instead of the SW sphere-march. Requires the BLAS/TLAS built by `rt.rs`
        // (currently the gallery scene only). Ignored on the volume path (that samples the field).
        // Default false keeps the SW march -> gallery byte-identical.
        hwrt: bool,
        // Returns `(gi_image, skyvis_image)`; the sky-vis image (indoor skylight occlusion) is only
        // produced on the volume path (None on the ray-march/gallery path).
    ) -> (ResourceId, Option<ResourceId>) {
        // F3: bind the HW-RT permutation only when opted in AND it was built (RT-capable device);
        // otherwise the SW-default variant (which contains no acceleration-structure reference).
        let gip = if hwrt {
            self.gi_hwrt_pipeline
                .as_ref()
                .or(self.gi_pipeline.as_ref())
                .expect("gdf gi pipeline")
        } else {
            self.gi_pipeline.as_ref().expect("gdf gi pipeline")
        };
        let out = graph.create_storage_image("gdf_gi_out", HDR_FORMAT, extent);
        // Sky-visibility output: produced only on the volume path. NOT denoised (the volume field is
        // already smooth/stable) — the lighting samples it directly at this (possibly half) res.
        let skyvis_out =
            gi_volume.map(|_| graph.create_storage_image("gi_skyvis", HDR_FORMAT, extent));
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
        if let Some((_, _, ext)) = gi_volume {
            reads.push(ext); // barrier the GI sample after the volume update
        }
        // vol_r = radiance SH base, vol_g = sky-vis SH base; vol_b (sky-vis output storage index) is
        // resolved inside the closure (it's a graph resource).
        let (vol_r, vol_g) = gi_volume
            .map(|(rb, sb, _)| (rb, sb))
            .unwrap_or((u32::MAX, u32::MAX));
        let cache_idx = cache.map(|(idx, _)| idx).unwrap_or([u32::MAX; 5]);
        let mut storage_writes = vec![out];
        if let Some(sv) = skyvis_out {
            storage_writes.push(sv);
        }
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gdf_gi",
                storage_writes,
                reads,
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let out_index = ctx.storage_index(out);
                let vol_b = skyvis_out
                    .map(|sv| ctx.storage_index(sv))
                    .unwrap_or(u32::MAX);
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
                    clip.0,               // clipmap descriptor index
                    clip.1,               // clipmap level count
                    crate::GROUND_ALBEDO, // analytic ground material (floor bounce hits)
                    max_steps,            // D3: bounce-ray march step cap
                    cone_k,               // P3: cone-trace LOD slope (0 = legacy)
                    // vol_r = radiance SH base, vol_g = sky-vis SH base, vol_b = sky-vis out image.
                    [vol_r, vol_g, vol_b],
                    u32::from(hwrt), // F3: HW-RT gather toggle (0 = SW march, default & anchor)
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        (out, skyvis_out)
    }

    /// Screen-space radiance probes (P1 trace + P3 integrate). A probe is placed on the
    /// representative G-buffer surface of every `SP_DOWNSAMPLE` screen tile; the trace pass
    /// fills each probe's octahedral radiance tile (atlas) by marching the shared bounce tracer
    /// into the scene GDF; the integrate pass gathers the surrounding probes per pixel into
    /// indirect irradiance E (the lighting multiplies by albedo). Returns the full-res E image
    /// (a drop-in for `record_gi`'s output). Same GDF / albedo / cache / clip inputs as
    /// `record_gi`. `None` without the pipelines.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_screen_probe<'a>(
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
        frame: u32,
        albedo: Option<(&'a [Volume; 3], ResourceId)>,
        cache: Option<([u32; 5], ResourceId)>,
        clamp_max: f32,
        clip: (u32, u32),
        clip_vols: &'a [&'a Volume],
        max_steps: u32,
        cone_k: f32,
        // P2: apply the spatial cross-probe joint-bilateral filter to the radiance atlas before
        // the gather (probe-neighborhood half kernel; 0 disables). Content quality knob.
        filter_half_kernel: u32,
        // P4: world radiance cache sampled by escaped probe rays for off-screen / far-field /
        // infinite bounce. `(storage_buffers[] index, graph write handle)`; the handle orders the
        // trace after this frame's cache update. `None` = unbound (no cache fallback).
        wrc: Option<(u32, ResourceId)>,
        // Returns `(gi_image, skyvis_image)`; the sky-vis image (indoor skylight occlusion) is
        // built from the probes' per-ray sky visibility, mirroring the volume GI path's output.
    ) -> (ResourceId, ResourceId) {
        let trace = self.sp_trace_pipeline.as_ref().expect("sp trace pipeline");
        let integrate = self
            .sp_integrate_pipeline
            .as_ref()
            .expect("sp integrate pipeline");
        let diag = Self::diag(aabb_min, aabb_max);
        let bias = diag * 0.004;

        // Probe density + octahedral resolution are the two scalability knobs (a future
        // RenderQuality tier). `P_SP_DOWNSAMPLE` (screen px / probe) and `P_SP_OCT` (octahedral
        // texels / side) override the defaults for A/B tuning.
        let ds = std::env::var("P_SP_DOWNSAMPLE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(SP_DOWNSAMPLE)
            .clamp(4, 64);
        let oct = std::env::var("P_SP_OCT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(SP_OCT_RES)
            .clamp(4, 32);

        let (wrc_atlas, wrc_ext) = match wrc {
            Some((idx, ext)) => (idx, Some(ext)),
            None => (u32::MAX, None),
        };

        let probes_x = cw.div_ceil(ds);
        let probes_y = ch.div_ceil(ds);
        let atlas_w = probes_x * oct;
        let atlas_h = probes_y * oct;
        let atlas = graph.create_storage_image(
            "sp_radiance_atlas",
            HDR_FORMAT,
            Extent2D::new(atlas_w, atlas_h),
        );
        let out = graph.create_storage_image("sp_gi_out", HDR_FORMAT, extent);
        let skyvis = graph.create_storage_image("sp_skyvis", HDR_FORMAT, extent);

        let cache_idx = cache.map(|(idx, _)| idx).unwrap_or([u32::MAX; 5]);
        let cache4 = [cache_idx[0], cache_idx[1], cache_idx[2], cache_idx[3]];
        let cache_tile = cache_idx[4];

        // --- P1: trace every probe's octahedral radiance tile into the atlas. ---
        let mut trace_reads = vec![depth, normal, scene_gdf_ext];
        if let Some((_, ext)) = albedo {
            trace_reads.push(ext);
        }
        if let Some((_, ext)) = cache {
            trace_reads.push(ext);
        }
        // Order the trace after this frame's world-cache update (escaped rays sample it).
        if let Some(ext) = wrc_ext {
            trace_reads.push(ext);
        }
        graph.add_compute_pass(
            ComputePassInfo {
                name: "screen_probe_trace",
                storage_writes: vec![atlas],
                reads: trace_reads,
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let atlas_index = ctx.storage_index(atlas);
                let cmd = ctx.cmd();
                cmd.volume_to_sampled(scene_gdf);
                for v in clip_vols {
                    cmd.volume_to_sampled(v);
                }
                if let Some((vols, _)) = albedo {
                    for v in vols.iter() {
                        cmd.volume_to_sampled(v);
                    }
                }
                cmd.bind_compute_pipeline(trace);
                cmd.push_constants_compute(&screen_probe_trace_push(
                    &inv_view_proj,
                    sun_dir,
                    sun_intensity,
                    aabb_min,
                    0.0, // world ground plane y = 0
                    aabb_max,
                    diag, // sample distance clamp
                    diag, // ray max distance
                    bias,
                    0.25, // sky fill radiance at the bounce hit
                    0.7,  // constant hit-albedo fallback (sentinel albedo)
                    crate::GROUND_ALBEDO,
                    cone_k,
                    cache4,
                    depth_index,
                    normal_index,
                    atlas_index,
                    cache_tile,
                    cw,
                    ch,
                    probes_x,
                    probes_y,
                    ds,
                    oct,
                    flip_y,
                    frame,
                    max_steps,
                    clip.0,
                    clip.1,
                    clamp_max,
                    wrc_atlas,
                    WRC_GRID,
                    WRC_OCT,
                ));
                cmd.dispatch(atlas_w.div_ceil(8), atlas_h.div_ceil(8), 1);
                Ok(())
            },
        );

        let pos_sigma = diag * 0.01;
        let normal_power = 8.0_f32;

        // --- P2: spatial cross-probe joint-bilateral filter of the radiance atlas. Smooths
        // probe-to-probe variation on shared surfaces (blockiness) without blurring across
        // silhouettes; skipped when disabled or the pipeline is absent. ---
        let gather_atlas = match (filter_half_kernel > 0, self.sp_filter_pipeline.as_ref()) {
            (true, Some(filter)) => {
                let filtered = graph.create_storage_image(
                    "sp_radiance_atlas_filtered",
                    HDR_FORMAT,
                    Extent2D::new(atlas_w, atlas_h),
                );
                graph.add_compute_pass(
                    ComputePassInfo {
                        name: "screen_probe_filter",
                        storage_writes: vec![filtered],
                        reads: vec![atlas, depth, normal],
                    },
                    move |ctx| {
                        let depth_index = ctx.sampled_index(depth);
                        let normal_index = ctx.sampled_index(normal);
                        let atlas_in = ctx.sampled_index(atlas);
                        let atlas_out = ctx.storage_index(filtered);
                        let cmd = ctx.cmd();
                        cmd.bind_compute_pipeline(filter);
                        cmd.push_constants_compute(&screen_probe_filter_push(
                            &inv_view_proj,
                            depth_index,
                            normal_index,
                            atlas_in,
                            atlas_out,
                            cw,
                            ch,
                            probes_x,
                            probes_y,
                            ds,
                            oct,
                            flip_y,
                            filter_half_kernel,
                            pos_sigma,
                            normal_power,
                        ));
                        cmd.dispatch(atlas_w.div_ceil(8), atlas_h.div_ceil(8), 1);
                        Ok(())
                    },
                );
                filtered
            }
            _ => atlas,
        };

        // --- P5: pre-integrate each probe's octahedral RADIANCE tile into an IRRADIANCE tile
        // ONCE, so the per-pixel gather is a cheap directional lookup (~4-probe bilinear) instead
        // of a full hemisphere integral per pixel (~oct^2 taps). The reference's default. When
        // enabled the integrate runs in lookup `mode = 1`; `P_SP_IRRADIANCE=0` keeps the direct
        // per-pixel integral (`mode = 0`). ---
        let irradiance_on = std::env::var("P_SP_IRRADIANCE")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(true);
        let (integrate_atlas, integrate_mode) =
            match (irradiance_on, self.sp_irradiance_pipeline.as_ref()) {
                (true, Some(irr)) => {
                    let irr_atlas = graph.create_storage_image(
                        "sp_irradiance_atlas",
                        HDR_FORMAT,
                        Extent2D::new(atlas_w, atlas_h),
                    );
                    graph.add_compute_pass(
                        ComputePassInfo {
                            name: "screen_probe_irradiance",
                            storage_writes: vec![irr_atlas],
                            reads: vec![gather_atlas],
                        },
                        move |ctx| {
                            let atlas_in = ctx.sampled_index(gather_atlas);
                            let atlas_out = ctx.storage_index(irr_atlas);
                            let cmd = ctx.cmd();
                            cmd.bind_compute_pipeline(irr);
                            cmd.push_constants_compute(&screen_probe_irradiance_push(
                                atlas_in, atlas_out, probes_x, probes_y, oct,
                            ));
                            cmd.dispatch(atlas_w.div_ceil(8), atlas_h.div_ceil(8), 1);
                            Ok(())
                        },
                    );
                    (irr_atlas, 1u32)
                }
                _ => (gather_atlas, 0u32),
            };

        // --- P3: gather the probe atlas per pixel into indirect irradiance (lookup in mode 1). ---
        graph.add_compute_pass(
            ComputePassInfo {
                name: "screen_probe_integrate",
                storage_writes: vec![out, skyvis],
                reads: vec![integrate_atlas, depth, normal],
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let atlas_index = ctx.sampled_index(integrate_atlas);
                let out_index = ctx.storage_index(out);
                let skyvis_index = ctx.storage_index(skyvis);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(integrate);
                cmd.push_constants_compute(&screen_probe_integrate_push(
                    &inv_view_proj,
                    depth_index,
                    normal_index,
                    atlas_index,
                    out_index,
                    cw,
                    ch,
                    probes_x,
                    probes_y,
                    ds,
                    oct,
                    flip_y,
                    skyvis_index,
                    pos_sigma,
                    normal_power,
                    integrate_mode,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        (out, skyvis)
    }

    /// Stage D1: joint-bilateral upsample of the half-res GI to full resolution. The C3 trace
    /// (the dominant Sponza cost) runs at half res (1/4 the rays); this reconstructs the full-res
    /// indirect irradiance with a depth/normal-aware guided upscale before the full-res denoiser
    /// and lighting consume it. `gi_half` is the half-res GI; `depth`/`normal` are the full-res
    /// G-buffer. Returns the full-res GI image. See `gdf_gi_upsample.slang`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_upsample<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        gi_half: ResourceId,
        depth: ResourceId,
        normal: ResourceId,
        full_extent: Extent2D,
        inv_view_proj: [f32; 16],
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        cw: u32,
        ch: u32,
        hw: u32,
        hh: u32,
        flip_y: u32,
    ) -> ResourceId {
        let up = self
            .upsample_pipeline
            .as_ref()
            .expect("gi upsample pipeline");
        let out = graph.create_storage_image("gdf_gi_full", HDR_FORMAT, full_extent);
        // Same edge-stopping scale as the à-trous denoiser so the upsample preserves the
        // same silhouettes (world-position sigma = a small fraction of the scene diagonal).
        let pos_sigma = Self::diag(aabb_min, aabb_max) * 0.03;
        let normal_power = 32.0_f32;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gdf_gi_upsample",
                storage_writes: vec![out],
                reads: vec![gi_half, depth, normal],
            },
            move |ctx| {
                let gi_half_index = ctx.sampled_index(gi_half);
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(up);
                cmd.push_constants_compute(&gdf_gi_upsample_push(
                    &inv_view_proj,
                    gi_half_index,
                    depth_index,
                    normal_index,
                    out_index,
                    cw,
                    ch,
                    hw,
                    hh,
                    flip_y,
                    pos_sigma,
                    normal_power,
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
        // Temporal history-clamp mode (params.w): 0 = off (let the EMA converge — kills the static
        // shimmer the hard clamp caused on the noisy spp1 GI), ~1.0 = hard 3x3 min/max (legacy,
        // gallery byte-identical anchor), > 1.5 = variance clamp with gamma = this value.
        temporal_clamp: f32,
        // Number of edge-aware à-trous iterations (macOS/M3 perf): 2 = legacy (steps 1,2); 1 = a single
        // wide blur (Apple tier — the sparse GI is temporally denoised + upsampled). Clamped to [1,5].
        atrous_steps: u32,
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
                    temporal_clamp,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );

        // Edge-aware à-trous iterations (dilation 1, 2, 4, … per step): a wide blur at low cost. The
        // count is tier-driven (`atrous_steps`, clamped [1,5]) — 2 legacy, 1 on the Apple perf tier.
        let pos_sigma = diag * 0.03;
        let normal_power = 32.0_f32;
        let mut cur = temporal_out;
        let steps = atrous_steps.clamp(1, 5);
        const ATROUS_NAMES: [&str; 5] = [
            "gi_atrous0",
            "gi_atrous1",
            "gi_atrous2",
            "gi_atrous3",
            "gi_atrous4",
        ];
        for i in 0..steps {
            let step = 1u32 << i; // 1, 2, 4, … (à-trous hole dilation)
            let out = graph.create_storage_image(ATROUS_NAMES[i as usize], HDR_FORMAT, extent);
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
