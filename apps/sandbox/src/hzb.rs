//! Hi-Z (HZB) occlusion culling (PR-8, `docs/hzb-occlusion-culling.md`).
//!
//! A max-reduced depth pyramid built from the scene depth, plus an HZB-aware variant
//! of the P7 GPU cull pass that rejects grid instances whose whole screen AABB is
//! behind the nearest occluder. The pyramid is app-owned (persistent) so this frame's
//! build feeds *next* frame's cull — the canonical prev-frame-HZB scheme that stays
//! conservative for a static/slow camera (no reprojection; reprojecting last-frame
//! depth is non-conservative and pops — see the doc). Opt-in behind `HZB_CULL=1`;
//! when off the frustum-only `CullSystem` path is byte-identical.
//!
//! Standard-Z (near = 0, far = 1) is in use, so the pyramid stores the FARTHEST
//! occluder depth per region (`max`) and an instance is culled iff its nearest
//! screen-space depth exceeds the HZB max over its footprint (see the shaders).

use dreamcoast_render::{ComputePassInfo, RenderGraph, ResourceId};
use rhi::{
    BackendKind, ComputePipeline, ComputePipelineDesc, Device, Extent2D, Format, RenderTarget,
    RenderTargetDesc, StorageBuffer, StorageBufferDesc,
};

use crate::app::load_compute_shader;
use crate::push::{cull_hzb_push, hzb_build_push};
use crate::{GRID_COUNT, GRID_DIM};

/// Base of the HZB relative to the render extent. Level 0 is half the render
/// resolution: the depth is downsampled once on copy, which halves the pyramid's
/// memory/bandwidth while a single coarse tap still bounds any instance footprint.
const HZB_BASE_DIVISOR: u32 = 2;

/// One mip level's render target (R32Float, storage = UAV + sampled) and its size.
struct HzbLevel {
    target: RenderTarget,
    width: u32,
    height: u32,
}

pub(crate) struct HzbSystem {
    levels: Vec<HzbLevel>,
    copy_pipeline: ComputePipeline,
    reduce_pipeline: ComputePipeline,
    cull_pipeline: ComputePipeline,
    /// Zeros the stats counters on the GPU (1-thread), ordered before the cull so the
    /// per-frame count is correct with frames in flight.
    stats_clear_pipeline: ComputePipeline,
    /// Host-visible stats buffer: [0] = survived instances, [1] = occlusion-culled.
    /// Cleared on the GPU each frame before the cull; read back after the frame.
    stats: StorageBuffer,
    /// Render extent the current pyramid was sized for (rebuilt on resize).
    extent: Extent2D,
}

impl HzbSystem {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        extent: Extent2D,
    ) -> anyhow::Result<Self> {
        let copy_cs = load_compute_shader(
            backend,
            dreamcoast_shader::hzb_copy_cs_spirv,
            dreamcoast_shader::hzb_copy_cs_dxil,
            dreamcoast_shader::hzb_copy_cs_metallib,
            "hzb_copy",
        )?;
        let copy_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            compute_bytes: copy_cs,
            compute_entry: "csCopy",
            push_constant_size: 32,
            bindless: true,
            uniform_buffer: false,
            threads_per_group: [8, 8, 1],
        })?;
        let reduce_cs = load_compute_shader(
            backend,
            dreamcoast_shader::hzb_reduce_cs_spirv,
            dreamcoast_shader::hzb_reduce_cs_dxil,
            dreamcoast_shader::hzb_reduce_cs_metallib,
            "hzb_reduce",
        )?;
        let reduce_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            compute_bytes: reduce_cs,
            compute_entry: "csReduce",
            push_constant_size: 32,
            bindless: true,
            uniform_buffer: false,
            threads_per_group: [8, 8, 1],
        })?;
        let cull_cs = load_compute_shader(
            backend,
            dreamcoast_shader::cull_hzb_cs_spirv,
            dreamcoast_shader::cull_hzb_cs_dxil,
            dreamcoast_shader::cull_hzb_cs_metallib,
            "cull_hzb",
        )?;
        let cull_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            compute_bytes: cull_cs,
            compute_entry: "csCullHzb",
            push_constant_size: 224, // frustum block (128) + occlusion block (96)
            bindless: true,
            uniform_buffer: false,
            threads_per_group: [64, 1, 1],
        })?;
        let clear_cs = load_compute_shader(
            backend,
            dreamcoast_shader::cull_stats_clear_cs_spirv,
            dreamcoast_shader::cull_stats_clear_cs_dxil,
            dreamcoast_shader::cull_stats_clear_cs_metallib,
            "cull_stats_clear",
        )?;
        let stats_clear_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            compute_bytes: clear_cs,
            compute_entry: "csClearStats",
            push_constant_size: 224, // same push as the cull (reads stats_index)
            bindless: true,
            uniform_buffer: false,
            threads_per_group: [1, 1, 1],
        })?;
        let levels = Self::make_levels(device, extent)?;
        // 2 u32 counters (survived, occlusion-culled); host-visible so the CPU can zero
        // it before each cull and read it back after (diagnostics only).
        let stats = device.create_storage_buffer_host(&StorageBufferDesc {
            size: 8,
            stride: 4,
            indirect: false,
        })?;
        Ok(Self {
            levels,
            copy_pipeline,
            reduce_pipeline,
            cull_pipeline,
            stats_clear_pipeline,
            stats,
            extent,
        })
    }

    /// Read back (survived, occlusion-culled) from the previous cull. The caller must
    /// have synced the frame (e.g. after `wait_idle` / fence). Returns (0, 0) on error.
    pub(crate) fn read_stats(&self) -> (u32, u32) {
        let mut bytes = [0u8; 8];
        if self.stats.read_into(&mut bytes).is_ok() {
            (
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            )
        } else {
            (0, 0)
        }
    }

    /// Allocate the R32Float mip chain: level 0 at `extent / HZB_BASE_DIVISOR`, then
    /// halving to 1x1. Each level is a storage render target (UAV write + sampled read).
    fn make_levels(device: &Device, extent: Extent2D) -> anyhow::Result<Vec<HzbLevel>> {
        let mut w = (extent.width / HZB_BASE_DIVISOR).max(1);
        let mut h = (extent.height / HZB_BASE_DIVISOR).max(1);
        let mut levels = Vec::new();
        loop {
            let target = device.create_render_target(&RenderTargetDesc {
                width: w,
                height: h,
                format: Format::R32Float,
                storage: true,
            })?;
            target.set_name("hzb_level");
            levels.push(HzbLevel {
                target,
                width: w,
                height: h,
            });
            if w == 1 && h == 1 {
                break;
            }
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        Ok(levels)
    }

    /// Rebuild the pyramid for a new render extent (window resize).
    pub(crate) fn resize(&mut self, device: &Device, extent: Extent2D) -> anyhow::Result<()> {
        if extent == self.extent {
            return Ok(());
        }
        self.levels = Self::make_levels(device, extent)?;
        self.extent = extent;
        Ok(())
    }

    /// Number of mip levels (for the cull push / stats).
    pub(crate) fn level_count(&self) -> u32 {
        self.levels.len() as u32
    }

    /// Bindless sampled index of level 0 (the base; higher levels are consecutive —
    /// asserted at record time). Used by the cull shader as `hzb_base + mip`.
    pub(crate) fn base_sampled_index(&self) -> u32 {
        self.levels[0].target.bindless_index()
    }

    pub(crate) fn base_dims(&self) -> (u32, u32) {
        (self.levels[0].width, self.levels[0].height)
    }

    /// Whether the HZB mip chain occupies consecutive bindless sampled slots (the
    /// shader indexes `hzb_base + mip`). Bindless allocation is sequential here, but
    /// the caller verifies this before enabling the occlusion test so a future
    /// allocator change fails loud rather than sampling a wrong texture.
    pub(crate) fn slots_are_consecutive(&self) -> bool {
        let base = self.levels[0].target.bindless_index();
        self.levels
            .iter()
            .enumerate()
            .all(|(i, lvl)| lvl.target.bindless_index() == base + i as u32)
    }

    /// Import the HZB as an external graph edge so the build passes sequence after the
    /// G-buffer (which produces the depth they read) and before next frame's cull.
    pub(crate) fn import(graph: &mut RenderGraph) -> ResourceId {
        graph.import_external("hzb")
    }

    /// Record the pyramid build from the finished scene depth: `csCopy` writes level 0
    /// (half-res max-downsample), then `csReduce` halves down to 1x1. Reads `g_depth`
    /// (sampled) via the graph; writes the app-owned levels via explicit UAV barriers.
    pub(crate) fn record_build<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        g_depth: ResourceId,
        hzb_ext: ResourceId,
        depth_extent: Extent2D,
    ) {
        let copy = &self.copy_pipeline;
        let reduce = &self.reduce_pipeline;
        let levels = &self.levels;
        graph.add_compute_pass(
            ComputePassInfo {
                name: "hzb_build",
                storage_writes: vec![hzb_ext],
                reads: vec![g_depth],
            },
            move |ctx| {
                let depth_index = ctx.sampled_index(g_depth);
                let cmd = ctx.cmd();
                // Level 0 <- scene depth. The source extent is passed in explicitly:
                // `ctx.extent()` would report the pass's first storage write — the
                // zero-sized external hzb edge — not the depth texture (a src of 0x0
                // made every level-0 texel read depth texel (0,0) and false-cull the
                // whole grid; caught by the sponza look-up conservatism gate).
                let (src_w, src_h) = (depth_extent.width, depth_extent.height);
                let l0 = &levels[0];
                cmd.rt_to_storage(&l0.target);
                cmd.bind_compute_pipeline(copy);
                cmd.push_constants_compute(&hzb_build_push(
                    depth_index,
                    l0.target.storage_index().expect("hzb level is storage"),
                    l0.width,
                    l0.height,
                    src_w,
                    src_h,
                    (src_w / l0.width).clamp(1, 2),
                    (src_h / l0.height).clamp(1, 2),
                ));
                cmd.dispatch(l0.width.div_ceil(8), l0.height.div_ceil(8), 1);
                // Reduce level n -> n+1 (2x2 max, +odd guard).
                cmd.bind_compute_pipeline(reduce);
                for n in 1..levels.len() {
                    let src = &levels[n - 1];
                    let dst = &levels[n];
                    cmd.storage_to_sampled(&src.target); // finish writing src as UAV -> read
                    cmd.rt_to_storage(&dst.target);
                    let tap_x = if (src.width & 1) != 0 { 3 } else { 2 };
                    let tap_y = if (src.height & 1) != 0 { 3 } else { 2 };
                    cmd.push_constants_compute(&hzb_build_push(
                        src.target.bindless_index(),
                        dst.target.storage_index().expect("hzb level is storage"),
                        dst.width,
                        dst.height,
                        src.width,
                        src.height,
                        tap_x,
                        tap_y,
                    ));
                    cmd.dispatch(dst.width.div_ceil(8), dst.height.div_ceil(8), 1);
                }
                // Leave every level readable for next frame's cull.
                for lvl in levels {
                    cmd.storage_to_sampled(&lvl.target);
                }
                Ok(())
            },
        );
    }

    /// Record the HZB-aware cull: same reset + frustum test as `CullSystem`, plus the
    /// occlusion test against last frame's pyramid. `enabled` gates the occlusion test
    /// (false on the very first frame, before any pyramid exists). Reuses the caller's
    /// `args`/`visible` buffers so the indirect draw is unchanged.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_cull<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        args: &'a StorageBuffer,
        visible: &'a StorageBuffer,
        args_ext: ResourceId,
        visible_ext: ResourceId,
        hzb_ext: ResourceId,
        planes: [[f32; 4]; 6],
        view_proj: [f32; 16],
        grid: &crate::cull::CullGrid,
        index_count: u32,
        enabled: bool,
    ) {
        let cull = &self.cull_pipeline;
        let clear = &self.stats_clear_pipeline;
        let stats = &self.stats;
        let hzb_base = self.base_sampled_index();
        let hzb_levels = self.level_count();
        let (hzb_w, hzb_h) = self.base_dims();
        // Stats index only when the occlusion test runs (0xffffffff = no stats).
        let stats_index = if enabled {
            self.stats.storage_index()
        } else {
            0xffff_ffffu32
        };
        let (spacing, radius, height) = (grid.spacing, grid.cube_radius, grid.height);
        graph.add_compute_pass(
            ComputePassInfo {
                name: "cull_hzb",
                storage_writes: vec![args_ext, visible_ext],
                reads: vec![hzb_ext],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                let push = cull_hzb_push(
                    &planes,
                    args.storage_index(),
                    visible.storage_index(),
                    GRID_COUNT,
                    GRID_DIM,
                    spacing,
                    radius,
                    height,
                    index_count,
                    &view_proj,
                    hzb_base,
                    hzb_levels,
                    hzb_w,
                    hzb_h,
                    enabled,
                    stats_index,
                );
                // Zero the stats on the GPU first (ordered before the atomic adds).
                if enabled {
                    cmd.bind_compute_pipeline(clear);
                    cmd.push_constants_compute(&push);
                    cmd.dispatch(1, 1, 1);
                    cmd.storage_buffer_barrier(stats);
                }
                cmd.bind_compute_pipeline(cull);
                cmd.push_constants_compute(&push);
                cmd.dispatch(GRID_COUNT.div_ceil(64), 1, 1);
                cmd.storage_buffer_barrier(visible);
                cmd.storage_buffer_to_indirect(args);
                Ok(())
            },
        );
    }
}
