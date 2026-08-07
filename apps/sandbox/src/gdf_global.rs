//! Global distance field — reference-engine parity (docs/gdf-scale-follow-plan.md U1,
//! mechanisms: docs/research/gdf_global.txt).
//!
//! The OBJECT stage (per-mesh SDF atlas + instance table) stays world-anchored; this
//! module owns the derived cache above it: camera-centered clipmap levels over a
//! page-granular toroidal window, GPU-composited from the culled instance list and
//! sampled with the WRAP trilinear sampler.
//!
//! Consumer seam: `mesh_sdf_sample.slang`'s dense fallback reads an INDIRECTION buffer
//! (header offset 116) whose single word names the live consts slot — flipping the
//! 2-slot consts ring is one host u32 store, and either value names a valid, complete
//! descriptor (the latch pattern; recenters are dead-zone-spaced, far beyond the
//! frames-in-flight window).
//!
//! Dynamic contract (the standing directive): the instance records re-upload through a
//! host ring on every composite frame; U2 turns that into the every-frame draw-list
//! rebuild that lets movers reach the field the frame they move.

use dreamcoast_core::glam::Vec3;
use dreamcoast_render::{ComputePassInfo, RenderGraph};
use rhi::{
    BackendKind, ComputePipeline, ComputePipelineDesc, Device, Format, StorageBuffer,
    StorageBufferDesc, Volume, VolumeDesc,
};

use crate::app::load_compute_shader;
use crate::push::{gdf_global_composite_push, gdf_global_cull_push};

/// Clipmap level count (reference default 4, exponent-spaced).
pub(crate) const GLOBAL_LEVELS: usize = 4;
/// Voxels per level axis (the reference runs 128 — a tier dial once U3 lands).
pub(crate) const GLOBAL_RES: u32 = 48;
/// Voxels per page axis: recenter deltas snap to pages, so scrolled-in content arrives
/// in whole page rows.
pub(crate) const PAGE_SIZE: u32 = 8;
#[allow(dead_code)] // U3's sparse page tables index by this; the planner test asserts it now.
pub(crate) const PAGES_PER_AXIS: u32 = GLOBAL_RES / PAGE_SIZE;
/// Finest level half-extent in metres: voxel 1 m at 48³; levels double (2/4/8 m voxels,
/// coarsest box 384 m — covers the 160-tile stress floor).
pub(crate) const LEVEL0_HALF: f32 = 24.0;
/// Per-level culled-list capacity (indices; +1 count word). 512 covers the stress
/// floor's 397 instances with headroom; the cull drops beyond it.
pub(crate) const CULL_CAP: u32 = 512;

/// One planned level: a page-snapped camera-centered box plus its toroidal scroll.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GlobalLevel {
    pub(crate) min: [f32; 3],
    pub(crate) max: [f32; 3],
    /// Accumulated content scroll in VOXELS (world voxel j lives at memory voxel
    /// `(j + scroll) mod GLOBAL_RES`), kept in `[0, GLOBAL_RES)`.
    pub(crate) scroll: [i32; 3],
}

impl GlobalLevel {
    pub(crate) fn half(i: usize) -> f32 {
        LEVEL0_HALF * (1u32 << i) as f32
    }
    pub(crate) fn voxel(i: usize) -> f32 {
        2.0 * Self::half(i) / GLOBAL_RES as f32
    }
    pub(crate) fn page_world(i: usize) -> f32 {
        Self::voxel(i) * PAGE_SIZE as f32
    }
}

/// Plan the level boxes around `eye`, page-snapped per level. Pure (unit-tested); the
/// runtime diffs successive plans into page deltas.
pub(crate) fn plan_levels(eye: Vec3) -> [GlobalLevel; GLOBAL_LEVELS] {
    std::array::from_fn(|i| {
        let half = GlobalLevel::half(i);
        let page = GlobalLevel::page_world(i);
        let snap = |v: f32| (v / page).round() * page;
        let c = [snap(eye.x), snap(eye.y), snap(eye.z)];
        GlobalLevel {
            min: [c[0] - half, c[1] - half, c[2] - half],
            max: [c[0] + half, c[1] + half, c[2] + half],
            scroll: [0; 3],
        }
    })
}

/// A pending composite job: refresh `ext` world-voxels from `box0` on `level`.
#[derive(Clone, Copy, Debug)]
struct CompositeJob {
    level: usize,
    box0: [u32; 3],
    ext: [u32; 3],
}

pub(crate) struct GdfGlobal {
    cull_pipeline: ComputePipeline,
    composite_pipeline: ComputePipeline,
    /// SDF level volumes (dense pages in U1: the volume IS the level's page set).
    sdf: [Volume; GLOBAL_LEVELS],
    /// Albedo level volumes (R/G/B per level).
    albedo: [[Volume; 3]; GLOBAL_LEVELS],
    /// Level descriptors, 64 B/level {min+voxel, max+border, uint4 volume idx,
    /// uint4 scroll+flags} + 16 B tail {active, res, levels, seq} — 2-slot host ring.
    consts: [StorageBuffer; 2],
    consts_live: usize,
    /// The one-word latch the mesh-SDF header points at: the live consts slot's
    /// bindless index. Host-rewritten on flip (either value names a valid slot).
    indirection: StorageBuffer,
    /// Instance ring (host): the retained record bytes re-upload before every
    /// composite batch (U2: rebuilt from the draw list every frame).
    instances: Vec<StorageBuffer>,
    instances_next: usize,
    /// Per-level culled index lists: [count, idx...] × levels. Host-visible so the
    /// count words zero without a clear pass (job frames are dead-zone spaced).
    culled: StorageBuffer,
    /// Retained object-stage 112 B instance records.
    record_bytes: Vec<u8>,
    record_count: u32,
    /// Atlas sampled indices for the composite push.
    atlas_idx: u32,
    alb_atlas_idx: [u32; 3],
    /// Open-space distance (the scene diagonal — matches compose.rs semantics).
    empty: f32,
    levels: [GlobalLevel; GLOBAL_LEVELS],
    pending: Vec<(usize, [i32; 3])>,
    jobs: Vec<CompositeJob>,
    /// This frame's promoted batch: (instance ring slot, jobs). Built by [`Self::apply`]
    /// (all mutation pre-graph), read immutably by [`Self::record`] — the borrow split
    /// every follow system here uses (a &mut record would pin the graph's shared borrows).
    batch: Option<(usize, Vec<CompositeJob>)>,
    /// Diagnostic: voxels recomposited by the last job batch.
    pub(crate) last_recomposite_voxels: u64,
}

impl GdfGlobal {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        eye: Vec3,
        frames_in_flight: usize,
    ) -> anyhow::Result<Self> {
        let pipeline = |spirv: fn() -> Option<&'static [u8]>,
                        dxil: fn() -> Option<&'static [u8]>,
                        metallib: fn() -> Option<&'static [u8]>,
                        name: &str,
                        pcsize: u32|
         -> anyhow::Result<ComputePipeline> {
            let cs = load_compute_shader(backend, spirv, dxil, metallib, name)?;
            Ok(device.create_compute_pipeline(&ComputePipelineDesc {
                compute_bytes: cs,
                compute_entry: "csMain",
                push_constant_size: pcsize,
                bindless: true,
                uniform_buffer: false,
                threads_per_group: dreamcoast_shader::compute_group_size(&format!("{name}_cs")),
            })?)
        };
        let cull_pipeline = pipeline(
            dreamcoast_shader::gdf_global_cull_cs_spirv,
            dreamcoast_shader::gdf_global_cull_cs_dxil,
            dreamcoast_shader::gdf_global_cull_cs_metallib,
            "gdf_global_cull",
            32,
        )?;
        let composite_pipeline = pipeline(
            dreamcoast_shader::gdf_global_composite_cs_spirv,
            dreamcoast_shader::gdf_global_composite_cs_dxil,
            dreamcoast_shader::gdf_global_composite_cs_metallib,
            "gdf_global_composite",
            112,
        )?;

        let vd = VolumeDesc {
            width: GLOBAL_RES,
            height: GLOBAL_RES,
            depth: GLOBAL_RES,
            format: Format::R32Float,
        };
        let mk = || device.create_volume(&vd);
        let sdf = [mk()?, mk()?, mk()?, mk()?];
        let albedo = [
            [mk()?, mk()?, mk()?],
            [mk()?, mk()?, mk()?],
            [mk()?, mk()?, mk()?],
            [mk()?, mk()?, mk()?],
        ];
        let consts_size = (GLOBAL_LEVELS * 64 + 16) as u64;
        let mk_consts = || {
            device.create_storage_buffer_host(&StorageBufferDesc {
                size: consts_size,
                stride: 16,
                indirect: false,
            })
        };
        let consts = [mk_consts()?, mk_consts()?];
        let indirection = device.create_storage_buffer_host(&StorageBufferDesc {
            size: 16,
            stride: 16,
            indirect: false,
        })?;
        let culled = device.create_storage_buffer_host(&StorageBufferDesc {
            size: (GLOBAL_LEVELS as u64) * (CULL_CAP as u64 + 1) * 4,
            stride: 4,
            indirect: false,
        })?;
        let mut instances = Vec::with_capacity(frames_in_flight.max(1));
        for _ in 0..frames_in_flight.max(1) {
            instances.push(device.create_storage_buffer_host(&StorageBufferDesc {
                size: crate::mesh_sdf::INSTANCE_STRIDE as u64,
                stride: crate::mesh_sdf::INSTANCE_STRIDE,
                indirect: false,
            })?);
        }
        Ok(GdfGlobal {
            cull_pipeline,
            composite_pipeline,
            sdf,
            albedo,
            consts,
            consts_live: 0,
            indirection,
            instances,
            instances_next: 0,
            culled,
            record_bytes: Vec::new(),
            record_count: 0,
            atlas_idx: u32::MAX,
            alb_atlas_idx: [u32::MAX; 3],
            empty: 100.0,
            levels: plan_levels(eye),
            pending: Vec::new(),
            jobs: Vec::new(),
            batch: None,
            last_recomposite_voxels: 0,
        })
    }

    /// The bindless index the mesh-SDF header stores (offset 116) — the latch word.
    pub(crate) fn indirection_index(&self) -> u32 {
        self.indirection.storage_index()
    }

    /// Install the object-stage products: the 112 B instance records (retained; the
    /// composite re-uploads them through the ring), the atlas sampled indices, and the
    /// open-space distance. Queues a FULL rebuild of every level.
    pub(crate) fn set_content(
        &mut self,
        device: &Device,
        records: Vec<u8>,
        count: u32,
        atlas_idx: u32,
        alb_atlas_idx: [u32; 3],
        empty: f32,
    ) -> anyhow::Result<()> {
        let size = (records.len() as u64).max(crate::mesh_sdf::INSTANCE_STRIDE as u64);
        for slot in self.instances.iter_mut() {
            *slot = device.create_storage_buffer_host(&StorageBufferDesc {
                size,
                stride: crate::mesh_sdf::INSTANCE_STRIDE,
                indirect: false,
            })?;
        }
        self.record_bytes = records;
        self.record_count = count;
        self.atlas_idx = atlas_idx;
        self.alb_atlas_idx = alb_atlas_idx;
        self.empty = empty;
        self.write_consts(self.consts_live)?;
        self.write_consts(1 - self.consts_live)?;
        self.write_latch()?;
        self.jobs.clear();
        for level in 0..GLOBAL_LEVELS {
            self.jobs.push(CompositeJob {
                level,
                box0: [0; 3],
                ext: [GLOBAL_RES; 3],
            });
        }
        Ok(())
    }

    /// Latch layout: {live consts idx, active, res, levels} — one host u32-row store
    /// per flip; the consumer shader derives everything else from the consts rows.
    fn write_latch(&self) -> anyhow::Result<()> {
        let mut b = Vec::with_capacity(16);
        for v in [
            self.consts[self.consts_live].storage_index(),
            1,
            GLOBAL_RES,
            GLOBAL_LEVELS as u32,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.indirection.write(&b)?;
        Ok(())
    }

    fn write_consts(&self, slot: usize) -> anyhow::Result<()> {
        let mut b = Vec::with_capacity(GLOBAL_LEVELS * 64 + 16);
        for (i, lv) in self.levels.iter().enumerate() {
            for v in [lv.min[0], lv.min[1], lv.min[2], GlobalLevel::voxel(i)] {
                b.extend_from_slice(&v.to_le_bytes());
            }
            // Border = half a texel: the wrap sampler must never blend the opposite
            // WORLD edge (the memory wrap is where world rows ARE adjacent; the world
            // border is where they are not).
            for v in [lv.max[0], lv.max[1], lv.max[2], 0.5 / GLOBAL_RES as f32] {
                b.extend_from_slice(&v.to_le_bytes());
            }
            for v in [
                self.sdf[i].sampled_index(),
                self.albedo[i][0].sampled_index(),
                self.albedo[i][1].sampled_index(),
                self.albedo[i][2].sampled_index(),
            ] {
                b.extend_from_slice(&v.to_le_bytes());
            }
            for v in [
                lv.scroll[0].rem_euclid(GLOBAL_RES as i32) as u32,
                lv.scroll[1].rem_euclid(GLOBAL_RES as i32) as u32,
                lv.scroll[2].rem_euclid(GLOBAL_RES as i32) as u32,
                1u32,
            ] {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in [1u32, GLOBAL_RES, GLOBAL_LEVELS as u32, 0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.consts[slot].write(&b)?;
        Ok(())
    }

    /// Dead-zone check (quarter half-extent per level — the codebase's follow
    /// contract). Arms page-snapped deltas; a fixed camera never arms — the
    /// gate-recipe invariant. New arms wait while a batch is still queued.
    pub(crate) fn arm(&mut self, eye: Vec3) {
        if self.record_count == 0 || !self.pending.is_empty() || !self.jobs.is_empty() {
            return;
        }
        for (i, lv) in self.levels.iter().enumerate() {
            let half = GlobalLevel::half(i);
            let page = GlobalLevel::page_world(i);
            let c = [
                0.5 * (lv.min[0] + lv.max[0]),
                0.5 * (lv.min[1] + lv.max[1]),
                0.5 * (lv.min[2] + lv.max[2]),
            ];
            let dead = half * 0.5;
            let d = [eye.x - c[0], eye.y - c[1], eye.z - c[2]];
            if d[0].abs() > dead || d[1].abs() > dead || d[2].abs() > dead {
                let pages = [
                    (d[0] / page).round() as i32,
                    (d[1] / page).round() as i32,
                    (d[2] / page).round() as i32,
                ];
                if pages != [0; 3] {
                    self.pending.push((i, pages));
                }
            }
        }
    }

    /// Pre-graph point: consume armed recenters — move box + scroll together, rewrite
    /// the inactive consts slot, flip the latch — then promote any queued jobs into
    /// this frame's batch (instance upload + culled-count zeroing happen HERE, so
    /// [`Self::record`] is purely immutable).
    pub(crate) fn apply(&mut self) -> anyhow::Result<()> {
        self.batch = None;
        if !self.pending.is_empty() {
            self.apply_pending()?;
        }
        if !self.jobs.is_empty() && self.record_count > 0 {
            let jobs = std::mem::take(&mut self.jobs);
            self.last_recomposite_voxels = jobs
                .iter()
                .map(|j| j.ext[0] as u64 * j.ext[1] as u64 * j.ext[2] as u64)
                .sum();
            let slot = self.instances_next;
            self.instances_next = (self.instances_next + 1) % self.instances.len();
            self.instances[slot].write(&self.record_bytes)?;
            // Zero the whole culled buffer (8 KB): the count words must reset and the
            // stale indices past each count are never read. Job frames are dead-zone
            // spaced, so no in-flight frame reads these words while we write.
            let zeros = vec![0u8; GLOBAL_LEVELS * (CULL_CAP as usize + 1) * 4];
            self.culled.write(&zeros)?;
            self.batch = Some((slot, jobs));
        }
        Ok(())
    }

    fn apply_pending(&mut self) -> anyhow::Result<()> {
        let pending = std::mem::take(&mut self.pending);
        for (i, pages) in pending {
            let vox_delta = [
                pages[0] * PAGE_SIZE as i32,
                pages[1] * PAGE_SIZE as i32,
                pages[2] * PAGE_SIZE as i32,
            ];
            let vsize = GlobalLevel::voxel(i);
            let lv = &mut self.levels[i];
            for (a, dv) in vox_delta.iter().enumerate() {
                let w = *dv as f32 * vsize;
                lv.min[a] += w;
                lv.max[a] += w;
                lv.scroll[a] = (lv.scroll[a] + dv).rem_euclid(GLOBAL_RES as i32);
            }
            // Newly exposed WORLD rows per axis: +delta exposes the high side, −delta
            // the low. Corner overlap between axes recomposites twice — harmless.
            for a in 0..3 {
                let d = vox_delta[a].unsigned_abs().min(GLOBAL_RES);
                if d == 0 {
                    continue;
                }
                let mut box0 = [0u32; 3];
                let mut ext = [GLOBAL_RES; 3];
                ext[a] = d;
                box0[a] = if vox_delta[a] > 0 { GLOBAL_RES - d } else { 0 };
                self.jobs.push(CompositeJob {
                    level: i,
                    box0,
                    ext,
                });
            }
        }
        let next = 1 - self.consts_live;
        self.write_consts(next)?;
        self.consts_live = next;
        self.write_latch()?;
        tracing::info!(
            "GDF global recenter: {} level(s) scrolled (toroidal, pages recomposite)",
            self.jobs
                .iter()
                .map(|j| j.level)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        );
        Ok(())
    }

    /// Record this frame's cull + composite passes from the batch [`Self::apply`]
    /// promoted. Writes ride the `scene_gdf` external so every existing GDF consumer
    /// orders after them.
    pub(crate) fn record<'a>(&'a self, graph: &mut RenderGraph<'a>) {
        let Some((slot, jobs)) = &self.batch else {
            return;
        };
        let slot = *slot;
        let inst_idx = self.instances[slot].storage_index();
        let culled_idx = self.culled.storage_index();
        let consts_idx = self.consts[self.consts_live].storage_index();
        let count = self.record_count;
        let cull_pipe = &self.cull_pipeline;
        let comp_pipe = &self.composite_pipeline;
        let ext_tok = graph.import_external("scene_gdf");
        graph.add_compute_pass(
            ComputePassInfo {
                name: "gdf_global_cull",
                storage_writes: vec![ext_tok],
                reads: vec![],
            },
            move |ctx| {
                let cmd = ctx.cmd();
                cmd.bind_compute_pipeline(cull_pipe);
                cmd.push_constants_compute(&gdf_global_cull_push(
                    inst_idx,
                    count,
                    culled_idx,
                    GLOBAL_LEVELS as u32,
                    consts_idx,
                    CULL_CAP,
                ));
                cmd.dispatch(count.div_ceil(64), 1, 1);
                Ok(())
            },
        );

        let atlas = self.atlas_idx;
        let alb_atlas = self.alb_atlas_idx;
        let empty = self.empty;
        for job in jobs.iter().copied() {
            let lv = job.level;
            let dst = [
                self.sdf[lv].storage_index(),
                self.albedo[lv][0].storage_index(),
                self.albedo[lv][1].storage_index(),
                self.albedo[lv][2].storage_index(),
            ];
            let scroll = [
                self.levels[lv].scroll[0].rem_euclid(GLOBAL_RES as i32) as u32,
                self.levels[lv].scroll[1].rem_euclid(GLOBAL_RES as i32) as u32,
                self.levels[lv].scroll[2].rem_euclid(GLOBAL_RES as i32) as u32,
            ];
            let sdf_vol = &self.sdf[lv];
            let alb_vols = &self.albedo[lv];
            let ext_tok = graph.import_external("scene_gdf");
            graph.add_compute_pass(
                ComputePassInfo {
                    name: "gdf_global_composite",
                    storage_writes: vec![ext_tok],
                    reads: vec![],
                },
                move |ctx| {
                    let cmd = ctx.cmd();
                    cmd.volume_to_storage(sdf_vol);
                    for v in alb_vols {
                        cmd.volume_to_storage(v);
                    }
                    cmd.bind_compute_pipeline(comp_pipe);
                    cmd.push_constants_compute(&gdf_global_composite_push(
                        [inst_idx, culled_idx, consts_idx, lv as u32],
                        dst,
                        [atlas, alb_atlas[0], alb_atlas[1], alb_atlas[2]],
                        [job.box0[0], job.box0[1], job.box0[2], CULL_CAP],
                        [job.ext[0], job.ext[1], job.ext[2], GLOBAL_RES],
                        scroll,
                        [empty, 0.7, 0.0, 0.0],
                    ));
                    cmd.dispatch(
                        job.ext[0].div_ceil(4),
                        job.ext[1].div_ceil(4),
                        job.ext[2].div_ceil(4),
                    );
                    Ok(())
                },
            );
        }
    }

    /// The level volumes a consumer pass transitions to sampled before marching.
    pub(crate) fn sampled_volumes(&self) -> Vec<&Volume> {
        let mut v: Vec<&Volume> = self.sdf.iter().collect();
        for set in &self.albedo {
            v.extend(set.iter());
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_are_exponent_spaced_and_page_snapped() {
        let l = plan_levels(Vec3::new(13.3, 2.0, -7.9));
        for (i, lv) in l.iter().enumerate() {
            let half = GlobalLevel::half(i);
            assert!((lv.max[0] - lv.min[0] - 2.0 * half).abs() < 1e-3);
            let page = GlobalLevel::page_world(i);
            for a in 0..3 {
                let c = 0.5 * (lv.min[a] + lv.max[a]);
                let snapped = (c / page).round() * page;
                assert!(
                    (c - snapped).abs() < 1e-3,
                    "level {i} axis {a} not page-snapped"
                );
            }
        }
        assert_eq!(GlobalLevel::half(1), 2.0 * GlobalLevel::half(0));
        assert_eq!(GlobalLevel::half(3), 8.0 * GlobalLevel::half(0));
        assert!(2.0 * GlobalLevel::half(3) >= 320.0);
    }

    #[test]
    fn page_math_is_consistent() {
        assert_eq!(PAGES_PER_AXIS * PAGE_SIZE, GLOBAL_RES);
        assert!(GlobalLevel::voxel(0) <= 80.0 / 48.0);
    }
}
