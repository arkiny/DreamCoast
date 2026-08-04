//! Global distance field — reference-engine parity (docs/gdf-scale-follow-plan.md U0,
//! mechanisms: docs/research/gdf_global.txt).
//!
//! The OBJECT stage (per-mesh SDF atlas + instance table) stays world-anchored; this
//! module owns the derived cache above it: **camera-centered clipmap levels over a
//! page-granular toroidal window**, composited on the GPU from the culled instance
//! list, sampled through a page table with a WRAP trilinear sampler.
//!
//! U0 scope (this file): the pure level planner + the GPU resources + the per-frame
//! instance upload seam (the dynamic-game directive: transforms re-upload every frame
//! so movers can enter the field). Nothing consumes the field yet — U1 wires the cull
//! and composite kernels, U2 the static/movable dual layer, U3 sparsity + mips.

// As `compose.rs`/`items.rs`: authored complete for the U1 integration wave — the
// resources and planner exist before their first GPU consumer, so the wave that wires
// the cull/composite kernels changes no interfaces. This silences "never used", not
// "never checked" (the planner is unit-tested below).
#![allow(dead_code)]

use dreamcoast_core::glam::Vec3;
use rhi::{Device, Format, StorageBuffer, StorageBufferDesc, Volume, VolumeDesc};

/// Clipmap level count (reference default 4, exponent-spaced).
pub(crate) const GLOBAL_LEVELS: usize = 4;
/// Voxels per level axis. 48 keeps parity with the dense scene volume the field
/// replaces; the reference runs 128 — a `RenderQuality` tier dial once U3 lands.
pub(crate) const GLOBAL_RES: u32 = 48;
/// Voxels per page axis: 48 = 6 pages of 8 — page granularity for scroll/dirty
/// updates without the reference's 128-page machinery at our resolution.
pub(crate) const PAGE_SIZE: u32 = 8;
/// Pages per level axis.
pub(crate) const PAGES_PER_AXIS: u32 = GLOBAL_RES / PAGE_SIZE;
/// Finest level half-extent in metres: voxel 1 m at 48³. Each coarser level doubles
/// (2/4/8 m voxels), so four levels cover a 384 m box — the 160-tile stress floor
/// (320 m) fits inside the coarsest.
pub(crate) const LEVEL0_HALF: f32 = 24.0;

/// One planned level: a page-snapped camera-centered box plus its toroidal scroll.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GlobalLevel {
    pub(crate) min: [f32; 3],
    pub(crate) max: [f32; 3],
    /// Content scroll in PAGES (world page j lives at memory page (j + scroll) mod
    /// PAGES_PER_AXIS) — page granularity keeps scroll updates page-aligned.
    pub(crate) scroll_pages: [i32; 3],
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

/// Plan the level boxes around `eye`, page-snapped per level (the reference snaps each
/// clipmap to its own page grid so cached pages survive camera motion). Pure — unit
/// tested below; the runtime diffs successive plans to derive scroll deltas.
pub(crate) fn plan_levels(eye: Vec3) -> [GlobalLevel; GLOBAL_LEVELS] {
    std::array::from_fn(|i| {
        let half = GlobalLevel::half(i);
        let page = GlobalLevel::page_world(i);
        let snap = |v: f32| (v / page).round() * page;
        let c = [snap(eye.x), snap(eye.y), snap(eye.z)];
        GlobalLevel {
            min: [c[0] - half, c[1] - half, c[2] - half],
            max: [c[0] + half, c[1] + half, c[2] + half],
            scroll_pages: [0; 3],
        }
    })
}

/// Per-frame instance record the cull/composite kernels read (48 B, three float4 rows
/// of the world→local affine + broad-phase AABB + atlas tile id — mirrors the static
/// `mesh_sdf` table but re-uploaded EVERY frame from the draw list, which is what lets
/// a mover's transform reach the field the frame it moves).
pub(crate) const INSTANCE_STRIDE: usize = 64;

/// The GPU resources: one page atlas volume per level set (dense pages in U1 — the
/// atlas IS the level volume at 48³ until U3 makes residency sparse), the page tables,
/// and the per-frame instance ring.
pub(crate) struct GdfGlobal {
    /// SDF page atlas per level (R32F 48³ each in U1's dense bring-up).
    pub(crate) sdf: [Volume; GLOBAL_LEVELS],
    /// Albedo pages per level (R/G/B channel volumes).
    pub(crate) albedo: [[Volume; 3]; GLOBAL_LEVELS],
    /// Page tables: u32 per page, [level][page_linear] — U1 writes identity (dense),
    /// U3 turns them into real allocations. One buffer holds all levels.
    pub(crate) page_table: StorageBuffer,
    /// Level constants (min/extent/scroll per level + counts) — 2-slot host ring, the
    /// descriptor-flip pattern every follow system in this codebase uses.
    pub(crate) consts: [StorageBuffer; 2],
    pub(crate) consts_live: usize,
    /// Per-frame instance table ring (host-visible, FRAMES_IN_FLIGHT slots): rebuilt
    /// from the draw list every frame — the dynamic-game seam.
    pub(crate) instances: Vec<StorageBuffer>,
    pub(crate) instance_capacity: u32,
    /// The current plan (world boxes + accumulated page scrolls).
    pub(crate) levels: [GlobalLevel; GLOBAL_LEVELS],
}

impl GdfGlobal {
    /// Allocate for up to `max_instances`. Never consumed unless `P11_GDF_GLOBAL` wires
    /// U1's kernels, so construction is content-gated by the caller.
    pub(crate) fn new(
        device: &Device,
        eye: Vec3,
        max_instances: u32,
        frames_in_flight: usize,
    ) -> anyhow::Result<Self> {
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
        let pages = (PAGES_PER_AXIS * PAGES_PER_AXIS * PAGES_PER_AXIS) as u64;
        let page_table = device.create_storage_buffer(&StorageBufferDesc {
            size: pages * GLOBAL_LEVELS as u64 * 4,
            stride: 4,
            indirect: false,
        })?;
        let consts_size = (GLOBAL_LEVELS * 48 + 16) as u64;
        let mk_consts = || {
            device.create_storage_buffer_host(&StorageBufferDesc {
                size: consts_size,
                stride: 16,
                indirect: false,
            })
        };
        let consts = [mk_consts()?, mk_consts()?];
        let cap = max_instances.max(64);
        let mut instances = Vec::with_capacity(frames_in_flight);
        for _ in 0..frames_in_flight {
            instances.push(device.create_storage_buffer_host(&StorageBufferDesc {
                size: cap as u64 * INSTANCE_STRIDE as u64,
                stride: INSTANCE_STRIDE as u32,
                indirect: false,
            })?);
        }
        Ok(GdfGlobal {
            sdf,
            albedo,
            page_table,
            consts,
            consts_live: 0,
            instances,
            instance_capacity: cap,
            levels: plan_levels(eye),
        })
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
        // Exponent coverage: each level doubles the previous.
        assert_eq!(GlobalLevel::half(1), 2.0 * GlobalLevel::half(0));
        assert_eq!(GlobalLevel::half(3), 8.0 * GlobalLevel::half(0));
        // The stress floor (320 m) fits in the coarsest box (384 m).
        assert!(2.0 * GlobalLevel::half(3) >= 320.0);
    }

    #[test]
    fn page_math_is_consistent() {
        assert_eq!(PAGES_PER_AXIS * PAGE_SIZE, GLOBAL_RES);
        // Finest voxel must not be coarser than today's small-floor dense field
        // (80 m / 48 ≈ 1.67 m) — the whole point of the near level.
        assert!(GlobalLevel::voxel(0) <= 80.0 / 48.0);
    }
}
