//! Global distance field — reference-engine parity (docs/gdf-scale-follow-plan.md U1/U2,
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
//! U2 dual layer (the standing dynamic directive): instances split into a MOSTLY-STATIC
//! block (composited into a cached static layer, refreshed only on scroll / membership
//! change) and a MOVABLE block (an instance is promoted the first frame its ECS world
//! transform changes, demoted never — sticky, like the reference's cached-primitive
//! promotion). The consumer-visible MERGED volumes rebuild a dirty page box by seeding
//! the static cache's value and unioning only the movable list on top, so a mover costs
//! its own footprint, not the static content's. Movement/despawn is detected per frame
//! against the ECS draw set keyed by ENTITY (draw order is not stable); the dirty
//! footprint is the union of the OLD and NEW world AABBs (the VSM double-footprint
//! rule), padded one voxel and kept VOXEL-granular (see `dirty_aabb`), then greedily
//! coalesced per level. An instance's influence is already bounded by its padded tile
//! AABB (the composite skips `t ∉ [0,1]`), so that pad is exact — no distance-band
//! clamp is needed for bounded invalidation.

use std::collections::HashMap;

use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_render::{ComputePassInfo, RenderGraph};
use dreamcoast_scene::Entity;
use rhi::{
    BackendKind, ComputePipeline, ComputePipelineDesc, Device, Format, StorageBuffer,
    StorageBufferDesc, Volume, VolumeDesc,
};

use crate::app::load_compute_shader;
use crate::compose::ComposeObject;
use crate::mesh_sdf::TileMap;
use crate::push::{gdf_global_composite_push, gdf_global_cull_push};

/// Clipmap level count (reference default 4, exponent-spaced).
pub(crate) const GLOBAL_LEVELS: usize = 4;
/// Default voxels per level axis — the U3 tier dial (`gdf_global_res` in
/// scalability.ron, `P11_GDF_GLOBAL_RES` env override; the reference runs 128).
pub(crate) const GLOBAL_RES_DEFAULT: u32 = 48;
/// Voxels per page axis: recenter deltas snap to pages, so scrolled-in content arrives
/// in whole page rows.
pub(crate) const PAGE_SIZE: u32 = 8;
/// Finest level half-extent in metres: voxel 1 m at 48³; levels double (2/4/8 m voxels,
/// coarsest box 384 m — covers the 160-tile stress floor).
pub(crate) const LEVEL0_HALF: f32 = 24.0;
/// Per-level culled-list capacity (indices; +1 count word). 512 covers the stress
/// floor's 397 instances with headroom; the cull drops beyond it.
pub(crate) const CULL_CAP: u32 = 512;
/// Bytes per world-AABB side-table row (min.xyz pad + max.xyz pad) — the cull's box
/// source; the 112 B record's bounds rows are the tile's LOCAL frame.
const AABB_STRIDE: u32 = 32;

/// Clamp a tier/env resolution to the page grid and a sane band.
pub(crate) fn clamp_res(res: u32) -> u32 {
    (res.clamp(2 * PAGE_SIZE, 128) / PAGE_SIZE) * PAGE_SIZE
}

/// One planned level: a page-snapped camera-centered box plus its toroidal scroll.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GlobalLevel {
    pub(crate) min: [f32; 3],
    pub(crate) max: [f32; 3],
    /// Accumulated content scroll in VOXELS (world voxel j lives at memory voxel
    /// `(j + scroll) mod res`), kept in `[0, res)`.
    pub(crate) scroll: [i32; 3],
}

impl GlobalLevel {
    pub(crate) fn half(i: usize) -> f32 {
        LEVEL0_HALF * (1u32 << i) as f32
    }
    pub(crate) fn voxel(i: usize, res: u32) -> f32 {
        2.0 * Self::half(i) / res as f32
    }
    pub(crate) fn page_world(i: usize, res: u32) -> f32 {
        Self::voxel(i, res) * PAGE_SIZE as f32
    }
}

/// Plan the level boxes around `eye`, page-snapped per level. Pure (unit-tested); the
/// runtime diffs successive plans into page deltas.
pub(crate) fn plan_levels(eye: Vec3, res: u32) -> [GlobalLevel; GLOBAL_LEVELS] {
    std::array::from_fn(|i| {
        let half = GlobalLevel::half(i);
        let page = GlobalLevel::page_world(i, res);
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

impl CompositeJob {
    fn voxels(&self) -> u64 {
        self.ext[0] as u64 * self.ext[1] as u64 * self.ext[2] as u64
    }
}

/// Greedy per-level box coalescing with a bounded-waste rule: a job folds into an
/// existing cluster when the union's volume is at most twice the pair's summed volume.
/// A character's 8–17 animating rig parts (each an old+new footprint per level) would
/// otherwise be hundreds of small dispatches per frame; their tight overlap collapses
/// them to roughly one box per character per level, while far-apart movers stay
/// separate (a spanning union fails the waste rule). Overlapping survivors are fine —
/// the composite is deterministic, so double-written voxels get the same value.
fn coalesce_jobs(jobs: Vec<CompositeJob>) -> Vec<CompositeJob> {
    let mut out: Vec<CompositeJob> = Vec::with_capacity(jobs.len().min(32));
    'jobs: for j in jobs {
        for c in out.iter_mut().filter(|c| c.level == j.level) {
            let mut u0 = [0u32; 3];
            let mut u1 = [0u32; 3];
            for a in 0..3 {
                u0[a] = c.box0[a].min(j.box0[a]);
                u1[a] = (c.box0[a] + c.ext[a]).max(j.box0[a] + j.ext[a]);
            }
            let union = CompositeJob {
                level: j.level,
                box0: u0,
                ext: [u1[0] - u0[0], u1[1] - u0[1], u1[2] - u0[2]],
            };
            if union.voxels() <= 2 * (c.voxels() + j.voxels()) {
                *c = union;
                continue 'jobs;
            }
        }
        out.push(j);
    }
    out
}

/// One tracked content instance — the CPU mirror the per-frame sync diffs the ECS
/// against. `world` is the last world transform whose record was encoded.
struct Tracked {
    entity: Entity,
    mesh: usize,
    world: Mat4,
    wmin: [f32; 3],
    wmax: [f32; 3],
    alive: bool,
    /// Sticky promotion: set the first frame the transform changes (or on despawn, so
    /// the record leaves the static layer), never cleared.
    movable: bool,
}

/// This frame's promoted batch (built by [`GdfGlobal::apply`], read by
/// [`GdfGlobal::record`] — the pre-graph/record borrow split).
struct Batch {
    slot: usize,
    static_jobs: Vec<CompositeJob>,
    merge_jobs: Vec<CompositeJob>,
    static_count: u32,
    movable_count: u32,
}

pub(crate) struct GdfGlobal {
    cull_pipeline: ComputePipeline,
    composite_pipeline: ComputePipeline,
    res: u32,
    /// MERGED level volumes — what every consumer samples (static ∪ movable).
    sdf: [Volume; GLOBAL_LEVELS],
    albedo: [[Volume; 3]; GLOBAL_LEVELS],
    /// Static-layer cache: recomposited only on scroll / static membership change;
    /// the merge pass seeds from it. Never sampled by consumers.
    cache_sdf: [Volume; GLOBAL_LEVELS],
    cache_albedo: [[Volume; 3]; GLOBAL_LEVELS],
    /// Level descriptors, 64 B/level {min+voxel, max+border, uint4 volume idx,
    /// uint4 scroll+flags} + 16 B tail {active, res, levels, seq} — 2-slot host ring.
    consts: [StorageBuffer; 2],
    consts_live: usize,
    /// The one-word latch the mesh-SDF header points at: the live consts slot's
    /// bindless index. Host-rewritten on flip (either value names a valid slot).
    indirection: StorageBuffer,
    /// Instance record ring (host, 112 B rows, [static..][movable..] partition) + the
    /// parallel world-AABB side table ring (32 B rows) the cull boxes against.
    instances: Vec<StorageBuffer>,
    aabbs: Vec<StorageBuffer>,
    instances_next: usize,
    /// Per-level culled index lists, TWO sections ([static | movable]), each
    /// [count, idx...] × levels. Host-visible so the count words zero without a
    /// clear pass.
    culled: StorageBuffer,
    /// Tracked instances (CPU mirror) + per-unique-mesh tile maps for re-encoding.
    tracked: Vec<Tracked>,
    tiles: Vec<TileMap>,
    /// Encoded record/AABB bytes in partition order, rebuilt when `records_dirty`.
    record_bytes: Vec<u8>,
    aabb_bytes: Vec<u8>,
    static_count: u32,
    movable_count: u32,
    records_dirty: bool,
    /// Per-frame dynamic sync enable (`P11_GDF_DYNAMIC`, default on — the directive).
    dynamic_on: bool,
    /// `P11_GDF_REFRESH=<frames>`: periodic FULL recomposite of both layers (0 = off).
    /// A divergence probe first (incremental-update rot shows as a sawtooth instead of
    /// a climb) and a stopgap self-heal while a divergence is being hunted.
    refresh_period: u64,
    frame_no: u64,
    /// Atlas sampled indices for the composite push.
    atlas_idx: u32,
    alb_atlas_idx: [u32; 3],
    /// Open-space distance (the scene diagonal — matches compose.rs semantics).
    empty: f32,
    levels: [GlobalLevel; GLOBAL_LEVELS],
    pending: Vec<(usize, [i32; 3])>,
    jobs_static: Vec<CompositeJob>,
    jobs_merge: Vec<CompositeJob>,
    batch: Option<Batch>,
    /// Diagnostics: voxels recomposited by the last batch (merge / static layers).
    pub(crate) last_recomposite_voxels: u64,
    pub(crate) last_static_voxels: u64,
}

impl GdfGlobal {
    pub(crate) fn new(
        device: &Device,
        backend: BackendKind,
        eye: Vec3,
        frames_in_flight: usize,
        res: u32,
    ) -> anyhow::Result<Self> {
        let res = clamp_res(res);
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
            128,
        )?;

        let vd = VolumeDesc {
            width: res,
            height: res,
            depth: res,
            format: Format::R32Float,
        };
        let mk = || device.create_volume(&vd);
        let mk3 = || -> anyhow::Result<[Volume; 3]> { Ok([mk()?, mk()?, mk()?]) };
        let sdf = [mk()?, mk()?, mk()?, mk()?];
        let albedo = [mk3()?, mk3()?, mk3()?, mk3()?];
        let cache_sdf = [mk()?, mk()?, mk()?, mk()?];
        let cache_albedo = [mk3()?, mk3()?, mk3()?, mk3()?];
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
            size: 2 * (GLOBAL_LEVELS as u64) * (CULL_CAP as u64 + 1) * 4,
            stride: 4,
            indirect: false,
        })?;
        let fif = frames_in_flight.max(1);
        let mut instances = Vec::with_capacity(fif);
        let mut aabbs = Vec::with_capacity(fif);
        for _ in 0..fif {
            instances.push(device.create_storage_buffer_host(&StorageBufferDesc {
                size: crate::mesh_sdf::INSTANCE_STRIDE as u64,
                stride: crate::mesh_sdf::INSTANCE_STRIDE,
                indirect: false,
            })?);
            aabbs.push(device.create_storage_buffer_host(&StorageBufferDesc {
                size: AABB_STRIDE as u64,
                stride: AABB_STRIDE,
                indirect: false,
            })?);
        }
        Ok(GdfGlobal {
            cull_pipeline,
            composite_pipeline,
            res,
            sdf,
            albedo,
            cache_sdf,
            cache_albedo,
            consts,
            consts_live: 0,
            indirection,
            instances,
            aabbs,
            instances_next: 0,
            culled,
            tracked: Vec::new(),
            tiles: Vec::new(),
            record_bytes: Vec::new(),
            aabb_bytes: Vec::new(),
            static_count: 0,
            movable_count: 0,
            records_dirty: false,
            dynamic_on: crate::quality::env_bool("P11_GDF_DYNAMIC", true),
            refresh_period: std::env::var("P11_GDF_REFRESH")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(0),
            frame_no: 0,
            atlas_idx: u32::MAX,
            alb_atlas_idx: [u32::MAX; 3],
            empty: 100.0,
            levels: plan_levels(eye, res),
            pending: Vec::new(),
            jobs_static: Vec::new(),
            jobs_merge: Vec::new(),
            batch: None,
            last_recomposite_voxels: 0,
            last_static_voxels: 0,
        })
    }

    /// The bindless index the mesh-SDF header stores (offset 116) — the latch word.
    pub(crate) fn indirection_index(&self) -> u32 {
        self.indirection.storage_index()
    }

    pub(crate) fn res(&self) -> u32 {
        self.res
    }

    /// Whether the per-frame sync has anything to diff (content installed + the
    /// dynamic seam on) — the caller skips building the entity→world map otherwise.
    pub(crate) fn wants_sync(&self) -> bool {
        self.dynamic_on && !self.tracked.is_empty()
    }

    /// Install the object-stage content: one tracked instance per surviving drawable
    /// (entity + world + unique-mesh index, in build order) and the per-mesh tile
    /// maps. Records are self-encoded through the SAME `mesh_sdf::encode_record` the
    /// load-time build used (`expect_records` cross-checks byte identity in debug).
    /// Queues a FULL rebuild of both layers on every level.
    #[allow(clippy::too_many_arguments)] // load-time install: one argument per object-stage product
    pub(crate) fn set_content(
        &mut self,
        device: &Device,
        tracked: Vec<(Entity, Mat4, usize)>,
        tiles: Vec<TileMap>,
        expect_records: &[u8],
        atlas_idx: u32,
        alb_atlas_idx: [u32; 3],
        empty: f32,
    ) -> anyhow::Result<()> {
        self.tiles = tiles;
        self.tracked = tracked
            .into_iter()
            .map(|(entity, world, mesh)| {
                let t = &self.tiles[mesh];
                let o = ComposeObject::from_local_aabb(world, mesh, t.aabb_min, t.aabb_max);
                Tracked {
                    entity,
                    mesh,
                    world,
                    wmin: o.wmin,
                    wmax: o.wmax,
                    alive: true,
                    movable: false,
                }
            })
            .collect();
        self.encode_records();
        debug_assert_eq!(
            self.record_bytes, expect_records,
            "global-field self-encoded records must match the load-time build"
        );
        let _ = expect_records;
        let size = (self.record_bytes.len() as u64).max(crate::mesh_sdf::INSTANCE_STRIDE as u64);
        let asize = (self.aabb_bytes.len() as u64).max(AABB_STRIDE as u64);
        for slot in self.instances.iter_mut() {
            *slot = device.create_storage_buffer_host(&StorageBufferDesc {
                size,
                stride: crate::mesh_sdf::INSTANCE_STRIDE,
                indirect: false,
            })?;
        }
        for slot in self.aabbs.iter_mut() {
            *slot = device.create_storage_buffer_host(&StorageBufferDesc {
                size: asize,
                stride: AABB_STRIDE,
                indirect: false,
            })?;
        }
        self.atlas_idx = atlas_idx;
        self.alb_atlas_idx = alb_atlas_idx;
        self.empty = empty;
        self.write_consts(self.consts_live)?;
        self.write_consts(1 - self.consts_live)?;
        self.write_latch()?;
        self.jobs_static.clear();
        self.jobs_merge.clear();
        for level in 0..GLOBAL_LEVELS {
            let full = CompositeJob {
                level,
                box0: [0; 3],
                ext: [self.res; 3],
            };
            self.jobs_static.push(full);
            self.jobs_merge.push(full);
        }
        Ok(())
    }

    /// Re-encode the record + AABB tables in partition order ([alive static..]
    /// [alive movable..]) — runs at install and on any event frame (`records_dirty`);
    /// a full re-encode of a few hundred records is trivia next to the upload.
    fn encode_records(&mut self) {
        self.record_bytes.clear();
        self.aabb_bytes.clear();
        let mut static_count = 0u32;
        let mut movable_count = 0u32;
        for movable_pass in [false, true] {
            for t in self
                .tracked
                .iter()
                .filter(|t| t.alive && t.movable == movable_pass)
            {
                let tile = &self.tiles[t.mesh];
                let o =
                    ComposeObject::from_local_aabb(t.world, t.mesh, tile.aabb_min, tile.aabb_max);
                crate::mesh_sdf::encode_record(&mut self.record_bytes, &o, tile);
                for v in [t.wmin[0], t.wmin[1], t.wmin[2], 0.0] {
                    self.aabb_bytes.extend_from_slice(&v.to_le_bytes());
                }
                for v in [t.wmax[0], t.wmax[1], t.wmax[2], 0.0] {
                    self.aabb_bytes.extend_from_slice(&v.to_le_bytes());
                }
                if movable_pass {
                    movable_count += 1;
                } else {
                    static_count += 1;
                }
            }
        }
        self.static_count = static_count;
        self.movable_count = movable_count;
        self.records_dirty = false;
    }

    /// Per-frame dynamic sync (pre-graph, before [`Self::apply`]): diff the tracked
    /// set against the ECS entity→world map. A changed transform promotes the instance
    /// to the movable block and dirties BOTH footprints; a missing entity retires the
    /// record. Untracked entities (spawned after install) are outside the field until
    /// the next content install — the documented U-follow-up.
    pub(crate) fn sync_dynamic(&mut self, worlds: &HashMap<Entity, Mat4>) {
        if !self.wants_sync() {
            return;
        }
        // A recenter is armed for THIS frame: sit the sync out. Mover boxes queued
        // now would need the scroll translation (and any residual coordinate-frame
        // subtlety there compounds into field rot over long play); skipping costs
        // nothing — next frame's transform diff spans both footprints, and a mover
        // travels well under the one-voxel dirty pad per frame.
        if !self.pending.is_empty() {
            return;
        }
        for i in 0..self.tracked.len() {
            let t = &self.tracked[i];
            if !t.alive {
                continue;
            }
            match worlds.get(&t.entity) {
                None => {
                    let (was_static, wmin, wmax) = (!t.movable, t.wmin, t.wmax);
                    let t = &mut self.tracked[i];
                    t.alive = false;
                    t.movable = true;
                    self.records_dirty = true;
                    // The retired record leaves whichever layer held it.
                    self.dirty_aabb(wmin, wmax, was_static);
                }
                Some(w) if *w != t.world => {
                    let was_static = !t.movable;
                    let (old_min, old_max) = (t.wmin, t.wmax);
                    let tile = self.tiles[t.mesh];
                    let o =
                        ComposeObject::from_local_aabb(*w, t.mesh, tile.aabb_min, tile.aabb_max);
                    let t = &mut self.tracked[i];
                    t.world = *w;
                    t.wmin = o.wmin;
                    t.wmax = o.wmax;
                    t.movable = true;
                    self.records_dirty = true;
                    // Double footprint: the old pages lose the instance (and, on
                    // promotion, the static layer loses it there too), the new pages
                    // gain it.
                    self.dirty_aabb(old_min, old_max, was_static);
                    self.dirty_aabb(o.wmin, o.wmax, false);
                }
                _ => {}
            }
        }
    }

    /// Queue dirty jobs for a world AABB: a merge-layer refresh on every overlapping
    /// level, plus a static-layer recomposite of the same box when the STATIC layer's
    /// membership changed there.
    ///
    /// Dirty boxes are VOXEL-granular (padded one voxel each side — the exact influence
    /// bound, see the module doc), NOT page-snapped: the composite kernel is
    /// box-granular, and snapping a sub-metre mover out to 8³-voxel pages per level per
    /// footprint measured ~600 K recomposited voxels/frame on the dungeon's 63
    /// animating rig parts (page granularity belongs to scroll rows and U3 sparse
    /// residency, where memory — not dispatch extent — is what's page-shaped).
    fn dirty_aabb(&mut self, wmin: [f32; 3], wmax: [f32; 3], also_static: bool) {
        for i in 0..GLOBAL_LEVELS {
            let lv = self.levels[i];
            let vox = GlobalLevel::voxel(i, self.res);
            let mut box0 = [0u32; 3];
            let mut ext = [0u32; 3];
            let mut hit = true;
            for a in 0..3 {
                if wmax[a] < lv.min[a] - vox || wmin[a] > lv.max[a] + vox {
                    hit = false;
                    break;
                }
                let lo = (((wmin[a] - lv.min[a]) / vox).floor() as i64 - 1).max(0) as u32;
                let hi = ((((wmax[a] - lv.min[a]) / vox).floor() as i64) + 1)
                    .clamp(0, self.res as i64 - 1) as u32;
                box0[a] = lo;
                ext[a] = hi - lo + 1;
            }
            if !hit {
                continue;
            }
            let job = CompositeJob {
                level: i,
                box0,
                ext,
            };
            self.jobs_merge.push(job);
            if also_static {
                self.jobs_static.push(job);
            }
        }
    }

    /// Latch layout: {live consts idx, active, res, levels} — one host u32-row store
    /// per flip; the consumer shader derives everything else from the consts rows.
    fn write_latch(&self) -> anyhow::Result<()> {
        let mut b = Vec::with_capacity(16);
        for v in [
            self.consts[self.consts_live].storage_index(),
            1,
            self.res,
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
            for v in [
                lv.min[0],
                lv.min[1],
                lv.min[2],
                GlobalLevel::voxel(i, self.res),
            ] {
                b.extend_from_slice(&v.to_le_bytes());
            }
            // Border = half a texel: the wrap sampler must never blend the opposite
            // WORLD edge (the memory wrap is where world rows ARE adjacent; the world
            // border is where they are not).
            for v in [lv.max[0], lv.max[1], lv.max[2], 0.5 / self.res as f32] {
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
                lv.scroll[0].rem_euclid(self.res as i32) as u32,
                lv.scroll[1].rem_euclid(self.res as i32) as u32,
                lv.scroll[2].rem_euclid(self.res as i32) as u32,
                1u32,
            ] {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in [1u32, self.res, GLOBAL_LEVELS as u32, 0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.consts[slot].write(&b)?;
        Ok(())
    }

    /// Dead-zone check (quarter half-extent per level — the codebase's follow
    /// contract). Arms page-snapped deltas; a fixed camera never arms — the
    /// gate-recipe invariant. New arms wait while a batch is still queued.
    pub(crate) fn arm(&mut self, eye: Vec3) {
        if self.record_bytes.is_empty()
            || !self.pending.is_empty()
            || !self.jobs_static.is_empty()
            || !self.jobs_merge.is_empty()
        {
            return;
        }
        for (i, lv) in self.levels.iter().enumerate() {
            let half = GlobalLevel::half(i);
            let page = GlobalLevel::page_world(i, self.res);
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
    /// this frame's batch (record re-encode + upload + culled-count zeroing happen
    /// HERE, so [`Self::record`] is purely immutable).
    pub(crate) fn apply(&mut self) -> anyhow::Result<()> {
        self.batch = None;
        self.frame_no += 1;
        if self.refresh_period > 0
            && self.frame_no.is_multiple_of(self.refresh_period)
            && !self.tracked.is_empty()
        {
            for level in 0..GLOBAL_LEVELS {
                let full = CompositeJob {
                    level,
                    box0: [0; 3],
                    ext: [self.res; 3],
                };
                self.jobs_static.push(full);
                self.jobs_merge.push(full);
            }
        }
        if !self.pending.is_empty() {
            self.apply_pending()?;
        }
        if (!self.jobs_static.is_empty() || !self.jobs_merge.is_empty()) && !self.tracked.is_empty()
        {
            if self.records_dirty {
                self.encode_records();
            }
            let static_jobs = coalesce_jobs(std::mem::take(&mut self.jobs_static));
            let merge_jobs = coalesce_jobs(std::mem::take(&mut self.jobs_merge));
            let vox = |jobs: &[CompositeJob]| {
                jobs.iter()
                    .map(|j| j.ext[0] as u64 * j.ext[1] as u64 * j.ext[2] as u64)
                    .sum()
            };
            self.last_static_voxels = vox(&static_jobs);
            self.last_recomposite_voxels = vox(&merge_jobs);
            let slot = self.instances_next;
            self.instances_next = (self.instances_next + 1) % self.instances.len();
            self.instances[slot].write(&self.record_bytes)?;
            self.aabbs[slot].write(&self.aabb_bytes)?;
            // Zero both culled sections (16 KB): the count words must reset and the
            // stale indices past each count are never read. Batch frames re-run the
            // culls they need before any composite reads a section.
            let zeros = vec![0u8; 2 * GLOBAL_LEVELS * (CULL_CAP as usize + 1) * 4];
            self.culled.write(&zeros)?;
            if crate::quality::env_bool("DIAG_GDF_GLOBAL", false) {
                tracing::info!(
                    "GDF global batch: static {} voxels / merge {} voxels, {} static + {} \
                     movable records",
                    self.last_static_voxels,
                    self.last_recomposite_voxels,
                    self.static_count,
                    self.movable_count,
                );
            }
            self.batch = Some(Batch {
                slot,
                static_jobs,
                merge_jobs,
                static_count: self.static_count,
                movable_count: self.movable_count,
            });
        }
        Ok(())
    }

    fn apply_pending(&mut self) -> anyhow::Result<()> {
        let pending = std::mem::take(&mut self.pending);
        let scrolled = pending
            .iter()
            .map(|(i, _)| *i)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let res = self.res;
        for (i, pages) in pending {
            let vox_delta = [
                pages[0] * PAGE_SIZE as i32,
                pages[1] * PAGE_SIZE as i32,
                pages[2] * PAGE_SIZE as i32,
            ];
            // Jobs queued EARLIER this frame (the dynamic sync runs before the
            // recenter) hold box coords relative to the PRE-scroll window. The window
            // is about to move by `vox_delta`, so translate them into the new frame —
            // compositing them un-shifted would refresh a region offset by the whole
            // scroll (stale content left where the mover actually is, plus a wrongly
            // rewritten patch elsewhere), and a long walk with per-frame movers turns
            // every recenter into another corrupt blotch (observed in play as
            // accumulating colour/field corruption + runaway march cost).
            for jobs in [&mut self.jobs_static, &mut self.jobs_merge] {
                jobs.retain_mut(|job| {
                    if job.level != i {
                        return true;
                    }
                    for (a, dv) in vox_delta.iter().enumerate() {
                        let lo = job.box0[a] as i64 - *dv as i64;
                        let hi = lo + job.ext[a] as i64; // exclusive
                        let lo_c = lo.clamp(0, res as i64);
                        let hi_c = hi.clamp(0, res as i64);
                        if hi_c <= lo_c {
                            return false; // scrolled fully out of the window
                        }
                        job.box0[a] = lo_c as u32;
                        job.ext[a] = (hi_c - lo_c) as u32;
                    }
                    true
                });
            }
            let vsize = GlobalLevel::voxel(i, res);
            let lv = &mut self.levels[i];
            for (a, dv) in vox_delta.iter().enumerate() {
                let w = *dv as f32 * vsize;
                lv.min[a] += w;
                lv.max[a] += w;
                lv.scroll[a] = (lv.scroll[a] + dv).rem_euclid(res as i32);
            }
            // Newly exposed WORLD rows per axis: +delta exposes the high side, −delta
            // the low. Corner overlap between axes recomposites twice — harmless.
            // Both layers scroll together (they share the mem mapping), so each
            // exposed row refreshes static THEN merge.
            for a in 0..3 {
                let d = vox_delta[a].unsigned_abs().min(res);
                if d == 0 {
                    continue;
                }
                let mut box0 = [0u32; 3];
                let mut ext = [res; 3];
                ext[a] = d;
                box0[a] = if vox_delta[a] > 0 { res - d } else { 0 };
                let job = CompositeJob {
                    level: i,
                    box0,
                    ext,
                };
                self.jobs_static.push(job);
                self.jobs_merge.push(job);
            }
        }
        let next = 1 - self.consts_live;
        self.write_consts(next)?;
        self.consts_live = next;
        self.write_latch()?;
        tracing::info!(
            "GDF global recenter: {scrolled} level(s) scrolled (toroidal, pages recomposite)"
        );
        Ok(())
    }

    /// Record this frame's cull + composite passes from the batch [`Self::apply`]
    /// promoted: static cull → static-layer composites (into the cache) → movable
    /// cull → merge composites (cache-seeded, movable list only). Writes ride the
    /// `scene_gdf` external so every existing GDF consumer orders after them.
    pub(crate) fn record<'a>(&'a self, graph: &mut RenderGraph<'a>) {
        // `this` (a Copy `&'a Self`) is what the pass closures capture — references
        // derived through it keep the full `'a` lifetime the graph requires.
        let this: &'a Self = self;
        let Some(batch) = &this.batch else {
            return;
        };
        let aabb_idx = this.aabbs[batch.slot].storage_index();
        let inst_idx = this.instances[batch.slot].storage_index();
        let culled_idx = this.culled.storage_index();
        let consts_idx = this.consts[this.consts_live].storage_index();
        let cull_pipe = &this.cull_pipeline;
        let comp_pipe = &this.composite_pipeline;
        let movable_section = GLOBAL_LEVELS as u32 * (CULL_CAP + 1);

        let cull = |graph: &mut RenderGraph<'a>, count: u32, base: u32, section: u32| {
            if count == 0 {
                return;
            }
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
                        aabb_idx,
                        count,
                        culled_idx,
                        GLOBAL_LEVELS as u32,
                        consts_idx,
                        CULL_CAP,
                        base,
                        section,
                    ));
                    cmd.dispatch(count.div_ceil(64), 1, 1);
                    Ok(())
                },
            );
        };

        let atlas = this.atlas_idx;
        let alb_atlas = this.alb_atlas_idx;
        let empty = this.empty;
        let res = this.res;
        let composite = |graph: &mut RenderGraph<'a>,
                         job: CompositeJob,
                         dst_set: (&'a Volume, &'a [Volume; 3]),
                         seed: [u32; 4],
                         section: u32| {
            let lv = job.level;
            let (sdf_vol, alb_vols) = dst_set;
            let dst = [
                sdf_vol.storage_index(),
                alb_vols[0].storage_index(),
                alb_vols[1].storage_index(),
                alb_vols[2].storage_index(),
            ];
            let scroll = [
                this.levels[lv].scroll[0].rem_euclid(res as i32) as u32,
                this.levels[lv].scroll[1].rem_euclid(res as i32) as u32,
                this.levels[lv].scroll[2].rem_euclid(res as i32) as u32,
                section,
            ];
            let seed_vols: Option<(&'a Volume, &'a [Volume; 3])> =
                (seed[0] != u32::MAX).then(|| (&this.cache_sdf[lv], &this.cache_albedo[lv]));
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
                    if let Some((s, sa)) = seed_vols {
                        cmd.volume_to_storage(s);
                        for v in sa {
                            cmd.volume_to_storage(v);
                        }
                    }
                    cmd.bind_compute_pipeline(comp_pipe);
                    cmd.push_constants_compute(&gdf_global_composite_push(
                        [inst_idx, culled_idx, consts_idx, lv as u32],
                        dst,
                        [atlas, alb_atlas[0], alb_atlas[1], alb_atlas[2]],
                        [job.box0[0], job.box0[1], job.box0[2], CULL_CAP],
                        [job.ext[0], job.ext[1], job.ext[2], res],
                        scroll,
                        [empty, 0.7, 0.0, 0.0],
                        seed,
                    ));
                    cmd.dispatch(
                        job.ext[0].div_ceil(4),
                        job.ext[1].div_ceil(4),
                        job.ext[2].div_ceil(4),
                    );
                    Ok(())
                },
            );
        };

        // Static layer: only when membership/scroll changed there. The cull re-lists
        // the whole static block (a few hundred threads) into section 0.
        if !batch.static_jobs.is_empty() {
            cull(graph, batch.static_count, 0, 0);
            for job in batch.static_jobs.iter().copied() {
                composite(
                    graph,
                    job,
                    (&this.cache_sdf[job.level], &this.cache_albedo[job.level]),
                    [u32::MAX; 4],
                    0,
                );
            }
        }
        // Merged layer: seed from the cache, union the movable block on top. With no
        // movables the pass degenerates to a cache→merged copy of the box.
        if !batch.merge_jobs.is_empty() {
            cull(
                graph,
                batch.movable_count,
                batch.static_count,
                movable_section,
            );
            for job in batch.merge_jobs.iter().copied() {
                let seed = [
                    this.cache_sdf[job.level].storage_index(),
                    this.cache_albedo[job.level][0].storage_index(),
                    this.cache_albedo[job.level][1].storage_index(),
                    this.cache_albedo[job.level][2].storage_index(),
                ];
                composite(
                    graph,
                    job,
                    (&this.sdf[job.level], &this.albedo[job.level]),
                    seed,
                    movable_section,
                );
            }
        }
    }

    /// The level volumes a consumer pass transitions to sampled before marching (the
    /// MERGED set only — the static cache is never sampled).
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
        let res = GLOBAL_RES_DEFAULT;
        let l = plan_levels(Vec3::new(13.3, 2.0, -7.9), res);
        for (i, lv) in l.iter().enumerate() {
            let half = GlobalLevel::half(i);
            assert!((lv.max[0] - lv.min[0] - 2.0 * half).abs() < 1e-3);
            let page = GlobalLevel::page_world(i, res);
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
    fn coalesce_folds_overlaps_and_keeps_distant_boxes() {
        let j = |level, box0, ext| CompositeJob { level, box0, ext };
        // Two overlapping character-part boxes fold into one; a far box survives; a
        // different level never folds.
        let out = coalesce_jobs(vec![
            j(0, [10, 10, 10], [4, 6, 4]),
            j(0, [11, 12, 11], [4, 5, 4]),
            j(0, [40, 40, 40], [3, 3, 3]),
            j(1, [10, 10, 10], [4, 6, 4]),
        ]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].box0, [10, 10, 10]);
        assert_eq!(out[0].ext, [5, 7, 5]);
        // Full-level rebuild jobs absorb anything on their level.
        let out = coalesce_jobs(vec![
            j(2, [0, 0, 0], [48, 48, 48]),
            j(2, [10, 10, 10], [4, 4, 4]),
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].ext, [48, 48, 48]);
    }

    #[test]
    fn page_math_is_consistent() {
        assert_eq!(GLOBAL_RES_DEFAULT % PAGE_SIZE, 0);
        assert!(GlobalLevel::voxel(0, GLOBAL_RES_DEFAULT) <= 80.0 / 48.0);
        assert_eq!(clamp_res(47), 40);
        assert_eq!(clamp_res(64), 64);
        assert_eq!(clamp_res(9999), 128);
    }
}
