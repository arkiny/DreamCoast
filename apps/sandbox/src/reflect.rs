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
    gdf_reflect_push, lit_history_push, reflect_compact_push, reflect_composite_push,
    reflect_resolve_push, reflect_spatial_push, reflect_temporal_push, ssr_push, ssr_resolve_push,
};

pub(crate) struct ReflectSystem {
    ssr_pipeline: Option<ComputePipeline>, // C5 screen-space reflections (stochastic half-res)
    ssr_resolve_pipeline: Option<ComputePipeline>, // stochastic SSR temporal resolve
    reflect_pipeline: Option<ComputePipeline>, // C6 GDF reflection fallback
    /// Phase 16 HWRT hybrid: the reflection shader's `HWRT_REFLECT` permutation (traces the scene
    /// TLAS instead of the GDF march, then shades the hit from the surface cache). Built only on
    /// RT-capable devices; bound in place of `reflect_pipeline` when `P_HWRT` is opted in.
    reflect_hwrt_pipeline: Option<ComputePipeline>,
    /// B2' screen-hit early-out: the `SCREEN_HIT` permutation (SW march with a screen-trace
    /// prepass per ray — validated on-screen hits read the prev-frame full-res lit history).
    reflect_screen_pipeline: Option<ComputePipeline>,
    /// B2 mirror compaction: the `REFLECT_COMPACT` permutation (SCREEN_HIT shading re-traced
    /// over the compacted near-mirror pixel list, one thread per LISTED pixel) plus its
    /// classify/reset/args passes and app-owned list/args buffers at the refine extent. The
    /// near-mirror band shows the sparse trace's texels directly (no denoiser can reconstruct
    /// a mirror image the trace never resolved), so those pixels — and only those — are
    /// re-traced dense; cost scales with the on-screen mirror area. See reflect_compact.slang.
    reflect_compact_pipeline: Option<ComputePipeline>,
    /// HWRT variant of the compacted re-trace (TLAS + screen-color/hit-lighting shading): true
    /// material colours for the off-screen reflected content the surface cache approximates.
    /// Built only on RT-capable devices.
    reflect_compact_hwrt_pipeline: Option<ComputePipeline>,
    compact_reset_pipeline: Option<ComputePipeline>,
    compact_classify_pipeline: Option<ComputePipeline>,
    compact_args_pipeline: Option<ComputePipeline>,
    /// Compacted pixel list (u32/texel: x | y<<16) + dispatch args ([gx, gy, gz, count], the
    /// `indirect` buffer `dispatch_indirect` consumes).
    compact_list: Option<StorageBuffer>,
    compact_args: Option<StorageBuffer>,
    compact_extent: (u32, u32),
    composite_pipeline: Option<ComputePipeline>, // C7 hybrid composite
    lit_history_pipeline: Option<ComputePipeline>, // C7b lit-color history capture
    resolve_pipeline: Option<ComputePipeline>,   // A1 spatial ratio-estimator resolve (trace res)
    spatial_pipeline: Option<ComputePipeline>, // A4b variance-guided bilateral denoiser (full res)
    temporal_pipeline: Option<ComputePipeline>, // C8j stochastic-GDF-reflection temporal resolve
    /// C8j stochastic GDF reflection temporal accumulation: ping-pong byte-address buffers —
    /// `accum` (tonemap-space rgb + history len) and `pos` (surface world point + valid), at the
    /// full render extent. The resolve reprojects the surface into the previous frame.
    refl_accum: [Option<StorageBuffer>; 2],
    refl_pos: [Option<StorageBuffer>; 2],
    /// A4a: per-pixel 2nd-moment (luminance²) accumulation ping-pong (scalar in .x), for the A4b
    /// variance-guided spatial denoiser. Allocated alongside `refl_accum`; used only when denoise is on.
    refl_moment: [Option<StorageBuffer>; 2],
    refl_accum_extent: (u32, u32),
    refl_accum_frame: u32,
    /// A3 adaptive temporal skip (docs/lossless-opt-ledger.md): a half-res ping-pong (32 B/px —
    /// float4 world_pos+valid, float4 traced radiance) at the gdf_reflect trace extent. gdf_reflect
    /// reuses last frame's radiance for a pixel whose surface point is unchanged (world-pos gate),
    /// skipping the sphere-march. Separate from `refl_accum` (which is full-res, tonemap-space, and
    /// owned by reflect_temporal) because the trace runs half-res.
    refl_skip: [Option<StorageBuffer>; 2],
    refl_skip_extent: (u32, u32),
    refl_skip_frame: u32,
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
            240,
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
        // Phase 16 HWRT hybrid permutation — same push layout (240B), traces the TLAS. Built only on
        // RT-capable devices (it references the acceleration structure); absent ⇒ SW march is used.
        // `uniform_buffer: true` binds the globals UBO (set 1/b1) for the B.2 screen-color-at-hit
        // reprojection (`prev_view_proj`), like SSR — the matrix doesn't fit the 240 B push.
        let reflect_hwrt_pipeline = if device.has_raytracing() {
            compute(
                dreamcoast_shader::gdf_reflect_hwrt_cs_spirv,
                dreamcoast_shader::gdf_reflect_hwrt_cs_dxil,
                dreamcoast_shader::gdf_reflect_hwrt_cs_metallib,
                "gdf_reflect_hwrt",
                240,
                true,
            )?
        } else {
            None
        };
        // B2' screen-hit early-out permutation — same push layout (240B), marches the reflection
        // ray against the depth buffer first and takes the previous frame's full-res lit radiance
        // on a validated on-screen hit (skipping the GDF march + cache shade). `uniform_buffer:
        // true` binds the globals UBO for the reprojection (`prev_view_proj`), like the HWRT B.2
        // screen path whose validation law it reuses.
        let reflect_screen_pipeline = compute(
            dreamcoast_shader::gdf_reflect_screen_cs_spirv,
            dreamcoast_shader::gdf_reflect_screen_cs_dxil,
            dreamcoast_shader::gdf_reflect_screen_cs_metallib,
            "gdf_reflect_screen",
            240,
            true,
        )?;
        // B2 mirror compaction: the list-driven re-trace runs [numthreads(64,1,1)] (flat list
        // index), so it can't share the [8,8,1] `compute` closure; same 240 B push + globals UBO
        // as the SCREEN_HIT permutation whose shading path it compiles in.
        let reflect_compact_pipeline = if compute_supported {
            let cs = load_compute_shader(
                backend,
                dreamcoast_shader::gdf_reflect_compact_cs_spirv,
                dreamcoast_shader::gdf_reflect_compact_cs_dxil,
                dreamcoast_shader::gdf_reflect_compact_cs_metallib,
                "gdf_reflect_compact",
            )?;
            Some(device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: cs,
                compute_entry: "csMain",
                push_constant_size: 240,
                bindless: true,
                uniform_buffer: true,
                threads_per_group: [64, 1, 1],
            })?)
        } else {
            None
        };
        // The classify/reset/args entries are NOT `csMain` (three entries in one file), so they
        // can't share the `compute` closure; explicit descs with matching threadgroup sizes.
        let compact_pass = |spirv: fn() -> Option<&'static [u8]>,
                            dxil: fn() -> Option<&'static [u8]>,
                            metallib: fn() -> Option<&'static [u8]>,
                            name: &str,
                            entry: &'static str,
                            threads: [u32; 3]|
         -> anyhow::Result<Option<ComputePipeline>> {
            if !compute_supported {
                return Ok(None);
            }
            let cs = load_compute_shader(backend, spirv, dxil, metallib, name)?;
            Ok(Some(device.create_compute_pipeline(
                &ComputePipelineDesc {
                    compute_bytes: cs,
                    compute_entry: entry,
                    push_constant_size: 32,
                    bindless: true,
                    uniform_buffer: false,
                    threads_per_group: threads,
                },
            )?))
        };
        let reflect_compact_hwrt_pipeline = if compute_supported && device.has_raytracing() {
            let cs = load_compute_shader(
                backend,
                dreamcoast_shader::gdf_reflect_compact_hwrt_cs_spirv,
                dreamcoast_shader::gdf_reflect_compact_hwrt_cs_dxil,
                dreamcoast_shader::gdf_reflect_compact_hwrt_cs_metallib,
                "gdf_reflect_compact_hwrt",
            )?;
            Some(device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: cs,
                compute_entry: "csMain",
                push_constant_size: 240,
                bindless: true,
                uniform_buffer: true,
                threads_per_group: [64, 1, 1],
            })?)
        } else {
            None
        };
        let compact_reset_pipeline = compact_pass(
            dreamcoast_shader::reflect_compact_reset_cs_spirv,
            dreamcoast_shader::reflect_compact_reset_cs_dxil,
            dreamcoast_shader::reflect_compact_reset_cs_metallib,
            "reflect_compact_reset",
            "csReset",
            [1, 1, 1],
        )?;
        let compact_classify_pipeline = compact_pass(
            dreamcoast_shader::reflect_compact_classify_cs_spirv,
            dreamcoast_shader::reflect_compact_classify_cs_dxil,
            dreamcoast_shader::reflect_compact_classify_cs_metallib,
            "reflect_compact_classify",
            "csClassify",
            [8, 8, 1],
        )?;
        let compact_args_pipeline = compact_pass(
            dreamcoast_shader::reflect_compact_args_cs_spirv,
            dreamcoast_shader::reflect_compact_args_cs_dxil,
            dreamcoast_shader::reflect_compact_args_cs_metallib,
            "reflect_compact_args",
            "csArgs",
            [1, 1, 1],
        )?;
        let composite_pipeline = compute(
            dreamcoast_shader::reflect_composite_cs_spirv,
            dreamcoast_shader::reflect_composite_cs_dxil,
            dreamcoast_shader::reflect_composite_cs_metallib,
            "reflect_composite",
            64, // +B2 mirror-compaction row (refine target + grid + roughness gate)
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
        // A1: spatial ratio-estimator resolve of the stochastic GGX GDF reflection (trace res).
        let resolve_pipeline = compute(
            dreamcoast_shader::reflect_resolve_cs_spirv,
            dreamcoast_shader::reflect_resolve_cs_dxil,
            dreamcoast_shader::reflect_resolve_cs_metallib,
            "reflect_resolve",
            128,
            false,
        )?;
        // A4b: variance-guided bilateral reflection denoiser (post-temporal, full res).
        let spatial_pipeline = compute(
            dreamcoast_shader::reflect_spatial_cs_spirv,
            dreamcoast_shader::reflect_spatial_cs_dxil,
            dreamcoast_shader::reflect_spatial_cs_metallib,
            "reflect_spatial",
            128,
            false,
        )?;
        // C8j: temporal resolve of the stochastic GGX GDF reflection.
        let temporal_pipeline = compute(
            dreamcoast_shader::reflect_temporal_cs_spirv,
            dreamcoast_shader::reflect_temporal_cs_dxil,
            dreamcoast_shader::reflect_temporal_cs_metallib,
            "reflect_temporal",
            240,
            false,
        )?;
        Ok(Self {
            ssr_pipeline,
            ssr_resolve_pipeline,
            reflect_pipeline,
            reflect_hwrt_pipeline,
            reflect_screen_pipeline,
            reflect_compact_pipeline,
            reflect_compact_hwrt_pipeline,
            compact_reset_pipeline,
            compact_classify_pipeline,
            compact_args_pipeline,
            compact_list: None,
            compact_args: None,
            compact_extent: (0, 0),
            composite_pipeline,
            lit_history_pipeline,
            resolve_pipeline,
            spatial_pipeline,
            temporal_pipeline,
            refl_accum: [None, None],
            refl_pos: [None, None],
            refl_moment: [None, None],
            refl_accum_extent: (0, 0),
            refl_accum_frame: 0,
            refl_skip: [None, None],
            refl_skip_extent: (0, 0),
            refl_skip_frame: 0,
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
            self.refl_moment = [make()?, make()?];
            self.refl_accum_extent = (cw, ch);
            self.refl_accum_frame = 0;
        }
        Ok(())
    }

    pub(crate) fn advance_reflect_accum(&mut self) {
        self.refl_accum_frame = self.refl_accum_frame.saturating_add(1);
    }

    pub(crate) fn has_reflect_skip(&self) -> bool {
        self.refl_skip[0].is_some()
    }

    /// A3: (re)allocate the half-res adaptive-skip ping-pong at the gdf_reflect trace extent
    /// (`rw`×`rh`). No-op without the GDF reflect pipeline. Resets the ping-pong on a resize.
    pub(crate) fn prepare_reflect_skip(
        &mut self,
        device: &Device,
        rw: u32,
        rh: u32,
    ) -> anyhow::Result<()> {
        if self.reflect_pipeline.is_none() {
            return Ok(());
        }
        if self.refl_skip_extent != (rw, rh) {
            device.wait_idle()?;
            let make = || -> anyhow::Result<Option<StorageBuffer>> {
                Ok(Some(device.create_storage_buffer(&StorageBufferDesc {
                    size: (rw as u64) * (rh as u64) * 32,
                    stride: 32,
                    indirect: false,
                })?))
            };
            self.refl_skip = [make()?, make()?];
            self.refl_skip_extent = (rw, rh);
            self.refl_skip_frame = 0;
        }
        Ok(())
    }

    pub(crate) fn advance_reflect_skip(&mut self) {
        self.refl_skip_frame = self.refl_skip_frame.saturating_add(1);
    }

    /// B2 mirror compaction: every pipeline of the classify → args → compacted-re-trace chain
    /// is available (compute-gated like the rest of the reflection stack).
    pub(crate) fn has_reflect_compact(&self) -> bool {
        self.reflect_compact_pipeline.is_some()
            && self.compact_reset_pipeline.is_some()
            && self.compact_classify_pipeline.is_some()
            && self.compact_args_pipeline.is_some()
    }

    /// B2: (re)allocate the compacted pixel list (u32 per refine texel — capacity one entry per
    /// texel, so the classify append can never overflow) and the 16 B indirect-args buffer at
    /// the refine extent.
    pub(crate) fn prepare_reflect_compact(
        &mut self,
        device: &Device,
        rw: u32,
        rh: u32,
    ) -> anyhow::Result<()> {
        if !self.has_reflect_compact() {
            return Ok(());
        }
        if self.compact_extent != (rw, rh) {
            device.wait_idle()?;
            // Header u32 (count mirror) + capacity for every refine texel.
            self.compact_list = Some(device.create_storage_buffer(&StorageBufferDesc {
                size: 4 + (rw as u64) * (rh as u64) * 4,
                stride: 4,
                indirect: false,
            })?);
            self.compact_args = Some(device.create_storage_buffer(&StorageBufferDesc {
                size: 16,
                stride: 4,
                indirect: true,
            })?);
            self.compact_extent = (rw, rh);
        }
        Ok(())
    }

    /// A3: the (read, write) storage indices for this frame's skip ping-pong, or `None` when the
    /// buffers don't exist. `read` is last frame's write (the reuse source); `write` is this frame's.
    pub(crate) fn reflect_skip_indices(&self) -> Option<(u32, u32)> {
        let read = ((self.refl_skip_frame + 1) % 2) as usize;
        let write = (self.refl_skip_frame % 2) as usize;
        Some((
            self.refl_skip[read].as_ref()?.storage_index(),
            self.refl_skip[write].as_ref()?.storage_index(),
        ))
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

    /// Phase 16 B.2: the bindless storage index of the PREVIOUS frame's lit-radiance history (the
    /// same buffer SSR reprojects into), for the HWRT reflection's screen-color-at-hit. `0x7FFFFFFF`
    /// = no history bound (the shader then keeps the surface-cache shading). Capped to 31 bits since
    /// it rides the reflection push's march-cap field (bit 31 is the content flag).
    pub(crate) fn lit_hist_read_index(&self) -> u32 {
        let read = ((self.lit_hist_frame + 1) % 2) as usize;
        self.lit_hist[read]
            .as_ref()
            .map(|b| b.storage_index() & 0x7FFF_FFFF)
            .unwrap_or(0x7FFF_FFFF)
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

    /// Stochastic SSR resolve (다른 레퍼런스 엔진 ratio estimator): gather the half-res neighbour rays
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
        clamp_mode: u32,
        clamp_gamma: f32,
        // Temporal blend factor: > 0 = legacy fixed-alpha EMA (0.15); < 0 = CONVERGE running mean
        // with K = -alpha (reference NumFramesAccumulated-style, damps the lit-history feedback).
        ema_alpha: f32,
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
                    ema_alpha,
                    clamp_max,
                    kernel_radius,
                    clamp_mode,
                    clamp_gamma,
                ));
                cmd.dispatch(hw.div_ceil(8), hh.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    pub(crate) fn has_reflect_resolve(&self) -> bool {
        self.resolve_pipeline.is_some()
    }

    /// A1: spatial ratio-estimator resolve of the stochastic GGX GDF reflection, at the TRACE
    /// resolution (before the bilateral upsample). Reconstructs each neighbour's ray direction from
    /// the same deterministic per-pixel jitter and reweights the borrowed sample by `pdf_p/pdf_q`
    /// (Stachowiak's ratio estimator — the same scheme `record_ssr_resolve` uses). Stateless (no
    /// history). Near-mirror pixels pass through. Returns the resolved trace-res reflection.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_reflect_resolve<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        refl: ResourceId,
        depth: ResourceId,
        normal: ResourceId,
        material: ResourceId,
        extent: Extent2D,
        rw: u32,
        rh: u32,
        inv_view_proj: [f32; 16],
        eye: Vec3,
        flip_y: u32,
        frame: u32,
        mirror_thresh: f32,
        kernel_radius: f32,
        // Packed flags (mirrors ResolvePush.stochastic): bits 0..7 = A5 sampler select (0 white,
        // 1 blue-noise — must match the sampler gdf_reflect used this frame); bits 16..23 = the B2'
        // rough-prefilter threshold (roughness*255, 0 = off) so prefiltered pixels pass through.
        stochastic: u32,
    ) -> ResourceId {
        let pipe = self
            .resolve_pipeline
            .as_ref()
            .expect("reflect resolve pipeline");
        let out = graph.create_storage_image("reflect_resolved_spatial", HDR_FORMAT, extent);
        graph.add_compute_pass(
            ComputePassInfo {
                name: "reflect_resolve",
                storage_writes: vec![out],
                reads: vec![refl, depth, normal, material],
            },
            move |ctx| {
                let refl_index = ctx.sampled_index(refl);
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let material_index = ctx.sampled_index(material);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&reflect_resolve_push(
                    &inv_view_proj,
                    eye,
                    refl_index,
                    depth_index,
                    normal_index,
                    material_index,
                    out_index,
                    rw,
                    rh,
                    flip_y,
                    frame,
                    mirror_thresh,
                    kernel_radius,
                    stochastic,
                ));
                cmd.dispatch(rw.div_ceil(8), rh.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    pub(crate) fn has_reflect_spatial(&self) -> bool {
        self.spatial_pipeline.is_some()
    }

    /// A4b: variance-guided bilateral reflection denoiser (post-temporal, full resolution). Reads the
    /// temporal output (rgb + per-pixel StdDev in .a from A4a) + G-buffer, and blurs only the noisy
    /// pixels with depth/normal/variance-driven edge stops (mirror = passthrough). Returns the
    /// denoised reflection (linear, for the composite).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_reflect_spatial<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        refl: ResourceId,
        depth: ResourceId,
        normal: ResourceId,
        material: ResourceId,
        extent: Extent2D,
        cw: u32,
        ch: u32,
        inv_view_proj: [f32; 16],
        eye: Vec3,
        flip_y: u32,
        kernel_radius: f32,
        tonemap_range: f32,
    ) -> ResourceId {
        let pipe = self
            .spatial_pipeline
            .as_ref()
            .expect("reflect spatial pipeline");
        let out = graph.create_storage_image("reflect_denoised", HDR_FORMAT, extent);
        graph.add_compute_pass(
            ComputePassInfo {
                name: "reflect_spatial",
                storage_writes: vec![out],
                reads: vec![refl, depth, normal, material],
            },
            move |ctx| {
                let refl_index = ctx.sampled_index(refl);
                let depth_index = ctx.sampled_index(depth);
                let normal_index = ctx.sampled_index(normal);
                let material_index = ctx.sampled_index(material);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&reflect_spatial_push(
                    &inv_view_proj,
                    eye,
                    refl_index,
                    depth_index,
                    normal_index,
                    material_index,
                    out_index,
                    cw,
                    ch,
                    flip_y,
                    kernel_radius,
                    tonemap_range,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
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
        // A1: 1 = the spatial ratio-estimator resolve already ran → skip this pass's box average.
        spatial_off: bool,
        // A4a: accumulate the per-pixel 2nd moment + emit StdDev in out.a for the A4b spatial denoiser.
        denoise: bool,
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
        // A4a: 2nd-moment ping-pong indices (sentinel when denoise off ⇒ shader skips all M2 work).
        let (moment_r, moment_w) = if denoise {
            (
                self.refl_moment[read]
                    .as_ref()
                    .map(|b| b.storage_index())
                    .unwrap_or(u32::MAX),
                self.refl_moment[write]
                    .as_ref()
                    .map(|b| b.storage_index())
                    .unwrap_or(u32::MAX),
            )
        } else {
            (u32::MAX, u32::MAX)
        };
        let denoise_on = denoise && moment_r != u32::MAX && moment_w != u32::MAX;
        let accum_w_ext = graph.import_external("refl_accum_w");
        let pos_w_ext = graph.import_external("refl_pos_w");
        let mut writes = vec![out, accum_w_ext, pos_w_ext];
        if denoise_on {
            writes.push(graph.import_external("refl_moment_w"));
        }
        graph.add_compute_pass(
            ComputePassInfo {
                name: "reflect_temporal",
                storage_writes: writes,
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
                    u32::from(spatial_off),
                    moment_r,
                    moment_w,
                    u32::from(denoise_on),
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
        // GI irradiance volume (radiance cache): (radiance_base, skyvis_base, fine-AABB storage
        // buffer or u32::MAX, write-order handle), same tuple the GI pass gets. Sampled at
        // reflection hits for the GI-lit indirect term so shadowed reflected surfaces aren't
        // black; the fine buffer opens the finest-first fall-through in the sample helpers
        // (F4B). `None` (gallery) -> legacy analytic fill.
        gi_volume: Option<(u32, u32, u32, ResourceId)>,
        // IBL diffuse-irradiance cube index (physical-units skylight). For content (gi_volume present)
        // it rides the shader-unused `sky_fill` push slot; a reflected uncovered/shadowed surface and
        // an escaped (sky) ray floor to this instead of near-black. `u32::MAX` / gallery -> legacy.
        irradiance_index: u32,
        clip: (u32, u32),
        clip_vols: &'a [&'a Volume],
        max_steps: u32,
        cone_k: f32,
        // A3 adaptive temporal skip: (read_idx, write_idx, K stagger, real frame). read == u32::MAX
        // disables reuse; write == u32::MAX disables the persist. Both sentinel = legacy full trace.
        skip: [u32; 4],
        // Phase 16 HWRT hybrid: when true, trace the reflection ray against the scene TLAS (the
        // `HWRT_REFLECT` permutation) instead of the GDF sphere-march. Same push layout; the hit is
        // still shaded from the surface cache. Falls back to the SW pipeline if the permutation is
        // absent (non-RT device). Default false ⇒ SW march (gallery byte-identical).
        hwrt: bool,
        // B2' screen-hit early-out: run the `SCREEN_HIT` SW permutation (per-ray screen-trace
        // prepass; validated on-screen hits read the prev-frame full-res lit history and skip the
        // GDF march + cache shade). SW only (ignored when `hwrt` wins); needs the lit history —
        // sentinel index ⇒ the shader falls through to the plain march.
        screen_hit: bool,
        // Phase 16 B.2 screen-color-at-hit: the globals UBO (for `prev_view_proj`) + the PREVIOUS
        // frame's lit-radiance history index. Consumed on the HWRT path (the index rides the
        // march-cap push field, unused there) AND the SCREEN_HIT path (the index rides the
        // constant-hit-albedo push slot, dead for content). Ignored by the plain SW path.
        globals: &'a Buffer,
        globals_offset: u64,
        lit_hist: u32,
        // Phase 16 E (Hit Lighting): consolidated content geometry/material `(vtx, idx, table)`
        // bindless indices for shading an OFF-SCREEN HW hit with the real material. `Some` only on the
        // HWRT path with the table built; rides the (HWRT-unused) coarse-albedo push slots + frame
        // bit31 (the enable), so no push growth. `None` ⇒ off-screen hits keep the surface cache.
        hit_lighting: Option<(u32, u32, u32)>,
    ) -> ResourceId {
        let use_hwrt = hwrt && self.reflect_hwrt_pipeline.is_some();
        let use_screen = !use_hwrt && screen_hit && self.reflect_screen_pipeline.is_some();
        let hit_lighting = if use_hwrt { hit_lighting } else { None };
        let pipe = if use_hwrt {
            self.reflect_hwrt_pipeline.as_ref()
        } else if use_screen {
            self.reflect_screen_pipeline.as_ref()
        } else {
            self.reflect_pipeline.as_ref()
        }
        .expect("gdf reflect pipeline");
        // The lit-history index rides the march-cap field on the HWRT path (bit 31 stays the content
        // flag); the SW path keeps the real step cap. So screen-color-at-hit needs no push growth.
        let max_steps = if use_hwrt {
            (max_steps & 0x8000_0000) | (lit_hist & 0x7FFF_FFFF)
        } else {
            max_steps
        };
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
        if let Some((_, _, _, ext)) = gi_volume {
            reads.push(ext); // barrier the reflection sample after this frame's volume update
        }
        // vol_r = radiance SH base; vol_g = sky-vis SH base (the fallback occludes its skylight
        // top-up / miss-sky by V — single source with the deferred skylight occlusion).
        let gi_vol_base = gi_volume.map(|(rb, ..)| rb).unwrap_or(u32::MAX);
        let gi_skyvis_base = gi_volume.map(|(_, sb, ..)| sb).unwrap_or(u32::MAX);
        let gi_fine_buf = gi_volume.map(|(_, _, fb, _)| fb).unwrap_or(u32::MAX);
        let cache_idx = cache.map(|(idx, _)| idx).unwrap_or([u32::MAX; 5]);
        // A3: order this frame's skip-buffer writes (imported external; read is last frame's slot,
        // covered by the frame fence, so it isn't graph-tracked — same pattern as refl_accum).
        let mut writes = vec![out];
        if skip[1] != u32::MAX {
            writes.push(graph.import_external("refl_skip_w"));
        }
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gdf_reflect",
                storage_writes: writes,
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
                // Phase 16 E: in Hit Lighting mode the coarse-albedo volume slots (unused — hit
                // lighting replaces the analytic fallback) instead carry the consolidated geometry/
                // material `(vtx, idx, table)` indices, and `frame` bit31 flags the mode. Zero push
                // growth (avoids a D3D12 root-CBV spill on the shared reflect push).
                let (albedo_rgb, frame) = match hit_lighting {
                    Some((v, i, t)) => ([v, i, t], frame | 0x8000_0000),
                    None => (albedo_rgb, frame),
                };
                // B.2 / B2': the HWRT and SCREEN_HIT pipelines bind the globals UBO for the
                // screen reprojection (`prev_view_proj`); the plain SW pipeline has no globals.
                if use_hwrt || use_screen {
                    cmd.set_globals(globals, globals_offset);
                }
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
                    gi_vol_base, // GI irradiance volume base (u32::MAX = off, legacy fill)
                    gi_skyvis_base,
                    gi_fine_buf, // F4B fine-AABB buffer (u32::MAX = coarse-half remap, legacy)
                    material_index,
                    aabb_min,
                    aabb_max,
                    0.0,  // world ground plane at y = 0
                    diag, // sample distance clamp
                    diag, // reflection ray max distance
                    // Constant hit-albedo fallback (sentinel albedo => achromatic, pre-C8a). The
                    // SCREEN_HIT permutation overloads this dead-for-content slot with the lit-
                    // history index (its shader hard-codes 0.7 for the fallback instead).
                    if use_screen {
                        f32::from_bits(lit_hist & 0x7FFF_FFFF)
                    } else {
                        0.7
                    },
                    // sky_fill slot: content (vol on) overloads it with the IBL irradiance cube index
                    // (the shader reads it only on the vol_on skylight-fill path); gallery keeps 0.25.
                    if gi_vol_base != u32::MAX {
                        f32::from_bits(irradiance_index)
                    } else {
                        0.25
                    },
                    bias,
                    albedo_rgb,
                    frame,
                    cache_idx,
                    clip.0,
                    clip.1,
                    crate::GROUND_ALBEDO, // analytic ground material (floor reflection hits)
                    max_steps,            // D3: reflection-ray march step cap
                    cone_k,               // P3: cone-trace LOD slope (0 = legacy)
                    skip,                 // A3: adaptive temporal skip (read, write, K, frame)
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    /// B2 mirror compaction: classify the near-mirror pixels of the `rw`×`rh` refine grid, turn
    /// the count into indirect-dispatch args, and re-trace exactly those pixels dense with the
    /// `REFLECT_COMPACT` permutation (SCREEN_HIT shading path, deterministic mirror rays).
    /// Returns the refine target (rgb radiance, a = hit distance; un-traced texels zeroed) that
    /// `record_composite` prefers for near-mirror pixels. Cost scales with the on-screen mirror
    /// area, not the frame — a mirror-free view pays two single-thread passes, a classify at the
    /// refine res, and an empty indirect dispatch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_reflect_compact<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        scene_gdf: &'a Volume,
        scene_gdf_ext: ResourceId,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        depth: ResourceId,
        normal: ResourceId,
        material: ResourceId,
        refine_extent: Extent2D,
        inv_view_proj: [f32; 16],
        eye: Vec3,
        sun_dir: [f32; 3],
        sun_intensity: f32,
        rw: u32,
        rh: u32,
        flip_y: u32,
        frame: u32,
        albedo: Option<(&'a [Volume; 3], ResourceId)>,
        cache: Option<([u32; 5], ResourceId)>,
        gi_volume: Option<(u32, u32, u32, ResourceId)>,
        irradiance_index: u32,
        clip: (u32, u32),
        clip_vols: &'a [&'a Volume],
        max_steps: u32,
        cone_k: f32,
        globals: &'a Buffer,
        globals_offset: u64,
        lit_hist: u32,
        mirror_thresh: f32,
        // HWRT refine: trace the listed pixels against the scene TLAS and shade off-screen hits
        // with the real material (`hit_lighting` = the consolidated (vtx, idx, table) indices) —
        // the surface cache stops being the mirror's primary colour source. Falls back to the SW
        // permutation when the pipeline/TLAS is absent.
        hwrt: bool,
        hit_lighting: Option<(u32, u32, u32)>,
        // Compact screen fetch (max_steps bit30): serve near-pixel-footprint on-screen hits from
        // the full-res lit history instead of the hybrid cache cone. Only viable with the
        // deferred-parity cache skylight (the two sources must agree on tone or the footprint
        // gate reads as a material seam).
        screen_fetch: bool,
    ) -> ResourceId {
        let reset = self.compact_reset_pipeline.as_ref().expect("compact reset");
        let classify = self
            .compact_classify_pipeline
            .as_ref()
            .expect("compact classify");
        let args_pipe = self.compact_args_pipeline.as_ref().expect("compact args");
        let use_hwrt = hwrt && self.reflect_compact_hwrt_pipeline.is_some();
        let hit_lighting = if use_hwrt { hit_lighting } else { None };
        let trace = if use_hwrt {
            self.reflect_compact_hwrt_pipeline.as_ref()
        } else {
            self.reflect_compact_pipeline.as_ref()
        }
        .expect("compact trace");
        let list_buf = self.compact_list.as_ref().expect("compact list");
        let args_buf = self.compact_args.as_ref().expect("compact args buf");
        let list_idx = list_buf.storage_index();
        let args_idx = args_buf.storage_index();

        let out = graph.create_storage_image("reflect_refine", HDR_FORMAT, refine_extent);
        let list_ext = graph.import_external("refl_compact_list");
        let args_ext = graph.import_external("refl_compact_args");

        // 1) Reset the append counter / dispatch args (and flip the args buffer back from last
        //    frame's INDIRECT state into UAV).
        graph.add_compute_pass(
            ComputePassInfo {
                name: "refl_compact_reset",
                storage_writes: vec![args_ext],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.storage_buffer_to_storage(args_buf); // INDIRECT (prev frame) -> UAV
                cmd.bind_compute_pipeline(reset);
                cmd.push_constants_compute(&reflect_compact_push(
                    0,
                    0,
                    list_idx,
                    args_idx,
                    0,
                    rw,
                    rh,
                    mirror_thresh,
                ));
                cmd.dispatch(1, 1, 1);
                cmd.storage_buffer_barrier(args_buf);
                Ok(())
            },
        );
        // 2) Classify: append every near-mirror refine texel; zero the refine target (validity
        //    key for the composite fetch).
        graph.add_compute_pass(
            ComputePassInfo {
                name: "refl_compact_classify",
                storage_writes: vec![out, list_ext, args_ext],
                reads: vec![depth, material],
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(depth);
                let material_index = ctx.sampled_index(material);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(classify);
                cmd.push_constants_compute(&reflect_compact_push(
                    depth_index,
                    material_index,
                    list_idx,
                    args_idx,
                    out_index,
                    rw,
                    rh,
                    mirror_thresh,
                ));
                cmd.dispatch(rw.div_ceil(8), rh.div_ceil(8), 1);
                cmd.storage_buffer_barrier(args_buf);
                cmd.storage_buffer_barrier(list_buf);
                Ok(())
            },
        );
        // 3) Args: count -> ceil(count/64) groups (+ the count mirrored into the list header),
        //    then flip the args buffer to indirect-args state for the dispatch.
        graph.add_compute_pass(
            ComputePassInfo {
                name: "refl_compact_args",
                storage_writes: vec![args_ext, list_ext],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(args_pipe);
                cmd.push_constants_compute(&reflect_compact_push(
                    0,
                    0,
                    list_idx,
                    args_idx,
                    0,
                    rw,
                    rh,
                    mirror_thresh,
                ));
                cmd.dispatch(1, 1, 1);
                cmd.storage_buffer_barrier(list_buf);
                cmd.storage_buffer_to_indirect(args_buf);
                Ok(())
            },
        );

        // 4) Compacted dense re-trace of the listed pixels (indirect; same push layout as the
        //    main trace, with the A3 skip slot carrying the list/args indices — see the
        //    REFLECT_COMPACT block in gdf_reflect.slang).
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
        let mut reads = vec![depth, normal, material, scene_gdf_ext, list_ext, args_ext];
        if let Some((_, ext)) = albedo {
            reads.push(ext);
        }
        if let Some((_, ext)) = cache {
            reads.push(ext);
        }
        if let Some((_, _, _, ext)) = gi_volume {
            reads.push(ext);
        }
        let gi_vol_base = gi_volume.map(|(rb, ..)| rb).unwrap_or(u32::MAX);
        let gi_skyvis_base = gi_volume.map(|(_, sb, ..)| sb).unwrap_or(u32::MAX);
        let gi_fine_buf = gi_volume.map(|(_, _, fb, _)| fb).unwrap_or(u32::MAX);
        let cache_idx = cache.map(|(idx, _)| idx).unwrap_or([u32::MAX; 5]);
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gdf_reflect_refine",
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
                // Phase 16 E overloads, exactly as record_gdf_reflect's HWRT path: hit-lighting
                // rides the (HWRT-unused) coarse-albedo slots + frame bit31; the lit-history
                // index rides the (HWRT-unused) march-cap field, keeping bit31 = content.
                let (albedo_rgb, frame) = match hit_lighting {
                    Some((v, i, t)) => ([v, i, t], frame | 0x8000_0000),
                    None => (albedo_rgb, frame),
                };
                let max_steps = if use_hwrt {
                    (max_steps & 0x8000_0000)
                        | if screen_fetch { 0x4000_0000 } else { 0 }
                        | (lit_hist & 0x3FFF_FFFF)
                } else {
                    max_steps
                };
                cmd.set_globals(globals, globals_offset);
                cmd.bind_compute_pipeline(trace);
                cmd.push_constants_compute(&gdf_reflect_push(
                    &inv_view_proj,
                    eye,
                    sun_dir,
                    sun_intensity,
                    depth_index,
                    normal_index,
                    sampled,
                    out_index,
                    rw,
                    rh,
                    flip_y,
                    gi_vol_base,
                    gi_skyvis_base,
                    gi_fine_buf, // F4B fine-AABB buffer (u32::MAX = coarse-half remap, legacy)
                    material_index,
                    aabb_min,
                    aabb_max,
                    0.0,
                    diag,
                    diag,
                    // SCREEN_HIT overloads the dead hit-albedo slot with the lit-history index;
                    // the HWRT variant carries it in max_steps instead and keeps the legacy
                    // constant-albedo fallback here (mirrors record_gdf_reflect).
                    if use_hwrt {
                        0.7
                    } else {
                        f32::from_bits(lit_hist & 0x7FFF_FFFF)
                    },
                    if gi_vol_base != u32::MAX {
                        f32::from_bits(irradiance_index)
                    } else {
                        0.25
                    },
                    bias,
                    albedo_rgb,
                    frame,
                    cache_idx,
                    clip.0,
                    clip.1,
                    crate::GROUND_ALBEDO,
                    max_steps,
                    cone_k,
                    // REFLECT_COMPACT overload: the A3 skip slot carries the list/args bindless
                    // indices, byte-split so the packer's per-byte masking reassembles
                    // `list | args << 16` exactly (indices are < 2048 = 11 bits each).
                    [
                        list_idx & 0xFF,
                        (list_idx >> 8) & 0xFF,
                        args_idx & 0xFF,
                        (args_idx >> 8) & 0xFF,
                    ],
                ));
                cmd.dispatch_indirect(args_buf, 0);
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
        // Content: drop SSR on NEAR-MIRROR surfaces and use the GDF/surface-cache reflection instead.
        // Our SSR has no hit-validation (uncertain/depth-thickness), so on a convex SW-RT mirror (the
        // chrome sphere) its screen march finds unreliable glancing hits that a partial blend smears in
        // as a messy edge. The reference engine keeps SSR only as a validated hard-switch upgrade; until
        // that validation exists, the (HQ) GDF/cache is the cleaner mirror source. Gallery passes false
        // → byte-identical anchor.
        skip_mirror_ssr: bool,
        // Roughness-scaled blur radius (texels) that smooths the low-res reflection's blocky
        // "sparkle" on rough surfaces while keeping its correct local colour. 0 = off (gallery).
        rough_blur: f32,
        // B1-lite hard handoff: zero the SSR blend entirely — the GDF image (SCREEN_HIT trace)
        // already carries validated on-screen colours per ray, so blending the unvalidated SSR
        // would double-count them and re-introduce its feedback wiggle. The `ssr` input is then
        // a stand-in (never sampled).
        ssr_cut: bool,
        // B2 mirror compaction: the dense near-mirror re-trace `(refine target, grid w, grid h)`
        // the composite prefers for roughness < `refine_thresh` pixels. `None` = off (legacy).
        refine: Option<(ResourceId, u32, u32)>,
        refine_thresh: f32,
    ) -> ResourceId {
        let pipe = self
            .composite_pipeline
            .as_ref()
            .expect("composite pipeline");
        let out = graph.create_storage_image("reflect_composite_out", HDR_FORMAT, extent);
        let mut reads = vec![ssr, gdf_reflect, material];
        if let Some((r, _, _)) = refine {
            reads.push(r);
        }
        graph.add_compute_pass(
            ComputePassInfo {
                name: "reflect_composite",
                storage_writes: vec![out],
                reads,
            },
            move |ctx| {
                let ssr_index = ctx.sampled_index(ssr);
                let gdf_index = ctx.sampled_index(gdf_reflect);
                let material_index = ctx.sampled_index(material);
                let out_index = ctx.storage_index(out);
                let refine_arg = refine.map(|(r, w, h)| (ctx.sampled_index(r), w, h));
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
                    skip_mirror_ssr,
                    rough_blur,
                    ssr_cut,
                    refine_arg,
                    refine_thresh,
                ));
                cmd.dispatch(cw.div_ceil(8), ch.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }
}
