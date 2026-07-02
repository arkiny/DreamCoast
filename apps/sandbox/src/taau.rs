//! QHD/UHD track — Temporal Anti-aliasing Upsampling (TAAU).
//!
//! Owns the TAAU compute pipeline + its full-resolution history (ping-pong `hist` rgb/length and
//! `pos` world-point buffers). The scene renders at a reduced internal resolution with a per-frame
//! sub-pixel jitter; `record` reprojects + accumulates those jittered low-res frames into a
//! full-res HDR the tonemap consumes. Mirrors the GI denoiser's history management
//! (`GiSystem::prepare_denoise` / `record_denoise` / `advance_denoise`). See `taau.slang`.

use dreamcoast_render::{ComputePassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, ComputePipeline, ComputePipelineDesc, Device, Extent2D, StorageBuffer,
    StorageBufferDesc,
};

use crate::HDR_FORMAT;
use crate::app::load_compute_shader;
use crate::push::{fxaa_push, taau_push};

pub(crate) struct TaauSystem {
    pipeline: Option<ComputePipeline>,
    /// Decima FXAA→TAA pre-pass: spatial edge AA on the current frame before temporal accumulation.
    fxaa_pipeline: Option<ComputePipeline>,
    /// Ping-pong history at OUTPUT resolution: `hist` (rgb accumulated + length), `pos`
    /// (world point + valid) — the disocclusion validation, like the GI denoiser.
    hist: [Option<StorageBuffer>; 2],
    pos: [Option<StorageBuffer>; 2],
    extent: (u32, u32),
    frame: u32,
}

impl TaauSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        compute_supported: bool,
    ) -> anyhow::Result<Self> {
        let pipeline = if compute_supported {
            let cs = load_compute_shader(
                backend,
                dreamcoast_shader::taau_cs_spirv,
                dreamcoast_shader::taau_cs_dxil,
                dreamcoast_shader::taau_cs_metallib,
                "taau",
            )?;
            Some(device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: cs,
                compute_entry: "csMain",
                push_constant_size: 240,
                bindless: true,
                uniform_buffer: false,
                threads_per_group: [8, 8, 1],
            })?)
        } else {
            None
        };
        let fxaa_pipeline = if compute_supported {
            let cs = load_compute_shader(
                backend,
                dreamcoast_shader::fxaa_cs_spirv,
                dreamcoast_shader::fxaa_cs_dxil,
                dreamcoast_shader::fxaa_cs_metallib,
                "fxaa",
            )?;
            Some(device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: cs,
                compute_entry: "csMain",
                push_constant_size: 16,
                bindless: true,
                uniform_buffer: false,
                threads_per_group: [8, 8, 1],
            })?)
        } else {
            None
        };
        Ok(Self {
            pipeline,
            fxaa_pipeline,
            hist: [None, None],
            pos: [None, None],
            extent: (0, 0),
            frame: 0,
        })
    }

    pub(crate) fn has_taau(&self) -> bool {
        self.pipeline.is_some()
    }
    pub(crate) fn has_fxaa(&self) -> bool {
        self.fxaa_pipeline.is_some()
    }

    /// Decima FXAA→TAA pre-pass: spatial HDR-aware FXAA on the current internal frame (same extent)
    /// before the temporal accumulation, so per-frame edge aliasing doesn't destabilize the history.
    pub(crate) fn record_fxaa<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        hdr: ResourceId,
        extent: Extent2D,
        w: u32,
        h: u32,
    ) -> ResourceId {
        let pipe = self.fxaa_pipeline.as_ref().expect("fxaa pipeline");
        let out = graph.create_storage_image("fxaa_out", HDR_FORMAT, extent);
        graph.add_compute_pass(
            ComputePassInfo {
                name: "fxaa",
                storage_writes: vec![out],
                reads: vec![hdr],
            },
            move |ctx| {
                let in_index = ctx.sampled_index(hdr);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&fxaa_push(in_index, out_index, w, h));
                cmd.dispatch(w.div_ceil(8), h.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }

    /// (Re)allocate the full-res history on a resize and reset accumulation on a resize / key
    /// change. Runs before the graph (its `wait_idle` + fallible alloc stay off the borrow path).
    pub(crate) fn prepare(
        &mut self,
        device: &Device,
        ow: u32,
        oh: u32,
        reset_key: u64,
    ) -> anyhow::Result<()> {
        if self.pipeline.is_none() {
            return Ok(());
        }
        let _ = reset_key;
        if self.extent != (ow, oh) {
            device.wait_idle()?;
            let make = || -> anyhow::Result<Option<StorageBuffer>> {
                Ok(Some(device.create_storage_buffer(&StorageBufferDesc {
                    size: (ow as u64) * (oh as u64) * 16,
                    stride: 16,
                    indirect: false,
                })?))
            };
            self.hist = [make()?, make()?];
            self.pos = [make()?, make()?];
            self.extent = (ow, oh);
            self.frame = 0;
        }
        Ok(())
    }

    pub(crate) fn advance(&mut self) {
        self.frame = self.frame.saturating_add(1);
    }

    /// Record the TAAU upsample: `hdr` is the internal-res lit HDR, `depth` the internal-res
    /// depth; returns the full-res accumulated HDR. `out_extent` is the full (output) resolution,
    /// `iw`/`ih` the internal resolution. `force_reset` ignores history this frame.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        hdr: ResourceId,
        depth: ResourceId,
        out_extent: Extent2D,
        ow: u32,
        oh: u32,
        iw: u32,
        ih: u32,
        inv_view_proj: [f32; 16],
        prev_view_proj: [f32; 16],
        flip_y: u32,
        scene_diag: f32,
        jitter_uv: [f32; 2],
        force_reset: bool,
        velocity: Option<ResourceId>,
    ) -> ResourceId {
        let pipe = self.pipeline.as_ref().expect("taau pipeline");
        let frame = self.frame;
        let reset = u32::from(frame == 0 || force_reset);
        let read = ((frame + 1) % 2) as usize;
        let write = (frame % 2) as usize;
        let hist_r = self.hist[read].as_ref().expect("hist r").storage_index();
        let hist_w = self.hist[write].as_ref().expect("hist w").storage_index();
        let pos_r = self.pos[read].as_ref().expect("pos r").storage_index();
        let pos_w = self.pos[write].as_ref().expect("pos w").storage_index();
        let hist_w_ext = graph.import_external("taau_hist_w");
        let pos_w_ext = graph.import_external("taau_pos_w");
        let out = graph.create_storage_image("taau_out", HDR_FORMAT, out_extent);
        let reject_dist = scene_diag * 0.01;
        // 16-frame history balances stability (jitter hidden) against gather-accumulation softening;
        // the FXAA pre-pass removes the per-frame edge aliasing that long history used to mask, so
        // it need not run longer. gamma = variance-box half-width (γσ), ~1 = standard.
        let max_hist = 32.0_f32;
        let gamma = 1.0_f32;
        let mut reads = vec![hdr, depth];
        if let Some(v) = velocity {
            reads.push(v);
        }
        graph.add_compute_pass(
            ComputePassInfo {
                name: "taau",
                storage_writes: vec![out, hist_w_ext, pos_w_ext],
                reads,
            },
            move |ctx| {
                let hdr_index = ctx.sampled_index(hdr);
                let depth_index = ctx.sampled_index(depth);
                let velocity_index = velocity.map(|v| ctx.sampled_index(v)).unwrap_or(u32::MAX);
                let out_index = ctx.storage_index(out);
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(pipe);
                cmd.push_constants_compute(&taau_push(
                    &inv_view_proj,
                    &prev_view_proj,
                    hdr_index,
                    depth_index,
                    out_index,
                    hist_r,
                    hist_w,
                    pos_r,
                    pos_w,
                    ow,
                    oh,
                    iw,
                    ih,
                    flip_y,
                    reset,
                    reject_dist,
                    max_hist,
                    gamma,
                    jitter_uv,
                    velocity_index,
                ));
                cmd.dispatch(ow.div_ceil(8), oh.div_ceil(8), 1);
                Ok(())
            },
        );
        out
    }
}
