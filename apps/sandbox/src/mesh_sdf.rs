//! Per-mesh SDF **direct sampling** — the CPU broad-phase + GPU record packing
//! (per-mesh-sdf-direct-sample-plan.md, P1).
//!
//! The dense-composite path (`compose.rs`) resamples every per-mesh field into one whole-scene
//! grid, discarding the per-mesh resolution. Direct sampling instead keeps each mesh's field in
//! the shared atlas (`dreamcoast_asset::sdf_atlas`) and, at query time, samples only the handful
//! of instances near the point. This module builds the data the shader needs:
//!
//! * an **instance table** (one record per drawable): the world→local affine, the mesh's local
//!   AABB, the mesh's atlas UVW mapping, and the world→local distance scale; and
//! * a **uniform cell grid** over the scene AABB: each cell holds a contiguous slice of the flat
//!   instance-index array listing every instance whose world AABB overlaps that cell, so the
//!   shader turns a world point into a short candidate list with one grid lookup.
//!
//! Both are built on the CPU (deterministic, backend-independent) and uploaded as bindless
//! storage buffers. Content-scene only — the gallery keeps the dense single-level field (the
//! byte-identical anchor).
//!
//! Staged ahead of its consumer: this CPU builder + its unit tests land first, then the P1 GPU
//! wiring (`gdf.rs` descriptor upload) + P2 direct-sample shader call `build` and read its
//! fields. `allow(dead_code)` covers the gap until that wiring lands.
#![allow(dead_code)]

use dreamcoast_asset::sdf_atlas::SdfAtlas;
use dreamcoast_core::glam::Mat4;

use crate::compose::ComposeObject;

/// Bytes per instance record (7 × float4). Matches `MeshSdfInstance` in `mesh_sdf_sample.slang`.
pub(crate) const INSTANCE_STRIDE: u64 = 112;
/// Bytes per cell record (`uint2` = offset, count).
pub(crate) const CELL_STRIDE: u64 = 8;

/// The packed direct-sample data ready to upload as bindless storage buffers.
pub(crate) struct MeshSdfBuild {
    /// One `INSTANCE_STRIDE`-byte record per instance, in `objects` order.
    pub(crate) instances: Vec<u8>,
    pub(crate) instance_count: u32,
    /// `res³` `CELL_STRIDE`-byte `(offset, count)` records, `cell = x + res*(y + res*z)`.
    pub(crate) cell_ranges: Vec<u8>,
    /// Flat instance indices grouped by cell (`cell_ranges` slices into it).
    pub(crate) indices: Vec<u8>,
    /// The grid's world AABB + edge resolution (the shader maps a point to a cell with these).
    pub(crate) grid_min: [f32; 3],
    pub(crate) grid_max: [f32; 3],
    pub(crate) res: u32,
}

/// A cell-grid edge resolution for `n` instances: roughly `cbrt(n)` so each cell holds a handful,
/// clamped to a sane band (a tiny scene still gets a 1³ grid; a big one is capped so the ranges
/// buffer stays small).
pub(crate) fn grid_res_for(n: usize) -> u32 {
    ((n as f32).cbrt().ceil() as u32).clamp(1, 64)
}

/// Push little-endian `f32`s.
fn push_f32(out: &mut Vec<u8>, vs: &[f32]) {
    for v in vs {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// The three rows of the upper 3×4 of an affine matrix (row `r` = `[m(r,0), m(r,1), m(r,2),
/// m(r,3)]`), so the shader forms `local = M * float4(p, 1)` as three dot products. glam is
/// column-major; `to_cols_array()[c*4 + r]` is element `(r, c)`.
fn affine_rows(m: Mat4) -> [[f32; 4]; 3] {
    let a = m.to_cols_array();
    let el = |r: usize, c: usize| a[c * 4 + r];
    [
        [el(0, 0), el(0, 1), el(0, 2), el(0, 3)],
        [el(1, 0), el(1, 1), el(1, 2), el(1, 3)],
        [el(2, 0), el(2, 1), el(2, 2), el(2, 3)],
    ]
}

/// Pack one instance record (see `INSTANCE_STRIDE`): inv_world rows, local AABB + dist scale,
/// atlas UVW bias + scale.
fn push_instance(out: &mut Vec<u8>, o: &ComposeObject, atlas: &SdfAtlas) {
    let (bias, scale) = atlas.tile_uvw(o.mesh);
    let t = &atlas.tiles[o.mesh];
    let rows = affine_rows(o.inv_world);
    let start = out.len();
    push_f32(out, &rows[0]);
    push_f32(out, &rows[1]);
    push_f32(out, &rows[2]);
    push_f32(out, &[t.aabb_min[0], t.aabb_min[1], t.aabb_min[2], o.scale]);
    push_f32(out, &[t.aabb_max[0], t.aabb_max[1], t.aabb_max[2], 0.0]);
    push_f32(out, &[bias[0], bias[1], bias[2], 0.0]);
    push_f32(out, &[scale[0], scale[1], scale[2], 0.0]);
    debug_assert_eq!(out.len() - start, INSTANCE_STRIDE as usize);
}

/// The inclusive `[lo, hi]` cell range a world AABB `[mn, mx]` overlaps (clamped to the grid).
fn cell_range(
    mn: [f32; 3],
    mx: [f32; 3],
    grid_min: [f32; 3],
    grid_max: [f32; 3],
    res: u32,
) -> ([u32; 3], [u32; 3]) {
    let mut lo = [0u32; 3];
    let mut hi = [0u32; 3];
    for a in 0..3 {
        let ext = (grid_max[a] - grid_min[a]).max(1e-8);
        let to_cell = |v: f32| {
            (((v - grid_min[a]) / ext) * res as f32)
                .floor()
                .clamp(0.0, res as f32 - 1.0) as u32
        };
        lo[a] = to_cell(mn[a]);
        hi[a] = to_cell(mx[a]);
    }
    (lo, hi)
}

/// Build the instance table + cell grid for a content scene. `objects` are the same broad-phase
/// records `compose.rs` uses (already small-mesh-culled); `atlas` is their packed per-mesh field.
/// The grid spans `[grid_min, grid_max]` (the scene GDF AABB) at `res³` cells.
pub(crate) fn build(
    objects: &[ComposeObject],
    atlas: &SdfAtlas,
    grid_min: [f32; 3],
    grid_max: [f32; 3],
    res: u32,
) -> MeshSdfBuild {
    let res = res.max(1);
    let cell_count = (res * res * res) as usize;

    // Pass 1: count instances per cell.
    let mut counts = vec![0u32; cell_count];
    for o in objects {
        let (lo, hi) = cell_range(o.wmin, o.wmax, grid_min, grid_max, res);
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    counts[(x + res * (y + res * z)) as usize] += 1;
                }
            }
        }
    }

    // Prefix-sum → per-cell offsets into the flat index array.
    let mut offsets = vec![0u32; cell_count];
    let mut running = 0u32;
    for c in 0..cell_count {
        offsets[c] = running;
        running += counts[c];
    }
    let total = running as usize;

    // Pass 2: scatter each instance index into its cells (a per-cell cursor keeps it stable).
    let mut cursor = offsets.clone();
    let mut indices = vec![0u32; total];
    for (i, o) in objects.iter().enumerate() {
        let (lo, hi) = cell_range(o.wmin, o.wmax, grid_min, grid_max, res);
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let c = (x + res * (y + res * z)) as usize;
                    indices[cursor[c] as usize] = i as u32;
                    cursor[c] += 1;
                }
            }
        }
    }

    // Serialize.
    let mut instances = Vec::with_capacity(objects.len() * INSTANCE_STRIDE as usize);
    for o in objects {
        push_instance(&mut instances, o, atlas);
    }
    let mut cell_ranges = Vec::with_capacity(cell_count * CELL_STRIDE as usize);
    for c in 0..cell_count {
        cell_ranges.extend_from_slice(&offsets[c].to_le_bytes());
        cell_ranges.extend_from_slice(&counts[c].to_le_bytes());
    }
    let mut index_bytes = Vec::with_capacity(total * 4);
    for i in &indices {
        index_bytes.extend_from_slice(&i.to_le_bytes());
    }

    MeshSdfBuild {
        instances,
        instance_count: objects.len() as u32,
        cell_ranges,
        indices: index_bytes,
        grid_min,
        grid_max,
        res,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dreamcoast_asset::sdf::SdfVolume;
    use dreamcoast_core::glam::Vec3;

    fn unit_mesh(dim: u32) -> SdfVolume {
        let n = (dim * dim * dim) as usize;
        SdfVolume {
            dim,
            aabb_min: [-1.0; 3],
            aabb_max: [1.0; 3],
            voxels: vec![0.0; n],
        }
    }

    /// Decode a cell record `(offset, count)`.
    fn cell(build: &MeshSdfBuild, x: u32, y: u32, z: u32) -> (u32, u32) {
        let c = (x + build.res * (y + build.res * z)) as usize;
        let b = &build.cell_ranges[c * 8..c * 8 + 8];
        (
            u32::from_le_bytes(b[0..4].try_into().unwrap()),
            u32::from_le_bytes(b[4..8].try_into().unwrap()),
        )
    }

    fn index_at(build: &MeshSdfBuild, i: u32) -> u32 {
        let b = &build.indices[i as usize * 4..i as usize * 4 + 4];
        u32::from_le_bytes(b.try_into().unwrap())
    }

    /// Two well-separated boxes in a 2³ grid must land in distinct cells, each listing exactly
    /// its own instance, and the grid must reference every instance exactly once (no drop/dup).
    #[test]
    fn separated_instances_land_in_distinct_cells() {
        let mesh = unit_mesh(8);
        let atlas = SdfAtlas::pack(std::slice::from_ref(&mesh));
        // Object 0 near the min corner, object 1 near the max corner of a [0,4]³ grid.
        let o0 = ComposeObject::new(Mat4::from_translation(Vec3::new(0.5, 0.5, 0.5)), 0, &mesh);
        let o1 = ComposeObject::new(Mat4::from_translation(Vec3::new(3.5, 3.5, 3.5)), 0, &mesh);
        let build = build(&[o0, o1], &atlas, [0.0; 3], [4.0; 3], 2);

        let (o_lo, c_lo) = cell(&build, 0, 0, 0);
        let (o_hi, c_hi) = cell(&build, 1, 1, 1);
        assert_eq!(c_lo, 1, "min-corner cell lists one instance");
        assert_eq!(c_hi, 1, "max-corner cell lists one instance");
        assert_eq!(index_at(&build, o_lo), 0);
        assert_eq!(index_at(&build, o_hi), 1);

        // Every reference across all cells sums to (instances × cells-they-span). Here each box
        // (unit mesh, tight AABB) sits in exactly one cell, so total refs == instance count.
        let total: u32 = (0..8)
            .map(|c| {
                let b = &build.cell_ranges[c * 8 + 4..c * 8 + 8];
                u32::from_le_bytes(b.try_into().unwrap())
            })
            .sum();
        assert_eq!(total, 2);
        assert_eq!(build.indices.len() as u32, total * 4);
    }

    /// A big instance spanning the whole grid must be listed in every cell (broad-phase is
    /// conservative), and offsets+counts must tile the index array with no gaps/overlaps.
    #[test]
    fn spanning_instance_is_in_every_cell() {
        // A large mesh whose padded AABB covers the whole grid.
        let big = SdfVolume {
            dim: 8,
            aabb_min: [-10.0; 3],
            aabb_max: [10.0; 3],
            voxels: vec![0.0; 512],
        };
        let atlas = SdfAtlas::pack(std::slice::from_ref(&big));
        let o = ComposeObject::new(Mat4::IDENTITY, 0, &big);
        let build = build(std::slice::from_ref(&o), &atlas, [0.0; 3], [4.0; 3], 2);
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let (_, count) = cell(&build, x, y, z);
                    assert_eq!(count, 1, "spanning instance missing from cell {x},{y},{z}");
                }
            }
        }
        // Offsets are a valid prefix-sum: cell k starts where cell k-1 ended.
        let mut expect = 0u32;
        for c in 0..8u32 {
            let (off, cnt) = cell(&build, c % 2, (c / 2) % 2, c / 4);
            assert_eq!(off, expect);
            expect += cnt;
        }
    }

    /// The instance record's inv_world rows must transform a world point back to the mesh's
    /// local frame (the shader relies on this to form the atlas coordinate).
    #[test]
    fn instance_inv_world_maps_world_to_local() {
        let mesh = unit_mesh(8);
        let atlas = SdfAtlas::pack(std::slice::from_ref(&mesh));
        let world = Mat4::from_translation(Vec3::new(2.0, -3.0, 5.0));
        let o = ComposeObject::new(world, 0, &mesh);
        let build = build(std::slice::from_ref(&o), &atlas, [-8.0; 3], [8.0; 3], 1);

        // Decode the three inv_world rows from record 0 and apply to a world point.
        let f = |off: usize| f32::from_le_bytes(build.instances[off..off + 4].try_into().unwrap());
        let p = [2.5, -2.5, 5.5, 1.0]; // world point = local (0.5, 0.5, 0.5)
        let mut local = [0.0f32; 3];
        for (r, lv) in local.iter_mut().enumerate() {
            let base = r * 16;
            *lv = f(base) * p[0] + f(base + 4) * p[1] + f(base + 8) * p[2] + f(base + 12) * p[3];
        }
        for (a, lv) in local.iter().enumerate() {
            assert!((lv - 0.5).abs() < 1e-5, "axis {a}: {lv}");
        }
    }
}
