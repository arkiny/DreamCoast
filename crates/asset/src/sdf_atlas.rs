//! Variable-tile per-mesh SDF **atlas** (per-mesh-sdf-direct-sample-plan.md, P0).
//!
//! Direct-sampling per-mesh distance fields would want one bindless volume slot per unique
//! mesh, but the sampled-volume table is only 64 slots wide on every backend while a content
//! scene has hundreds of unique meshes. So instead of composing the per-mesh fields into a
//! dense whole-scene grid (which throws away the per-mesh resolution — the root cause of the
//! thin-geometry penetration + surface-card-registration noise), we pack every unique mesh's
//! `dim³` field into **one** atlas volume as a variable-size tile and sample it directly at
//! query time. One volume slot (+ later one albedo atlas) covers the whole scene.
//!
//! The pack is a deterministic 3D shelf: tiles are placed footprint-first along X, wrapping to
//! new Z rows and then new Y layers, so the same mesh set always produces byte-identical atlas
//! bytes (backend-independent, cache-friendly). Each tile carries a 1-voxel **gutter** whose
//! voxels replicate the tile's edge (clamp addressing), so a hardware-trilinear tap inside the
//! tile interior never bleeds into a neighbour tile — reproducing [`SdfVolume::sample`]'s
//! clamp-to-edge convention on the GPU.
//!
//! ## GPU sampling contract
//! For a query point mapped into a mesh's local frame, the caller forms the mesh-local
//! normalized coordinate `t = saturate((lp - aabb_min) / (aabb_max - aabb_min))` (per axis),
//! then samples the atlas at `uvw = tile.uvw_bias + t * tile.uvw_scale`. Because the atlas
//! stores mesh voxel `i` of a `d`-wide tile at atlas voxel `origin_inner + i`, and the GPU
//! sampler reads at continuous texel coord `uvw * A - 0.5`, matching [`SdfVolume::sample`]
//! (continuous mesh index `t*d - 0.5`, clamped) requires exactly
//! `uvw = (origin_inner + t*d) / A = origin_inner/A + t*(d/A)` — i.e. `uvw_bias =
//! origin_inner/A`, `uvw_scale = d/A`. [`SdfAtlas::tile_uvw`] returns that pair.

use crate::sdf::SdfVolume;

/// One packed tile: where a mesh's `dim³` field lives in the atlas, plus the mesh's local AABB
/// (so the sampler can form the mesh-local normalized coordinate `t`).
#[derive(Clone, Debug)]
pub struct AtlasTile {
    /// Inner-block origin in atlas voxels (the mesh's voxel `0` — past the gutter).
    pub origin: [u32; 3],
    /// Inner cube edge (voxels) = the mesh's `SdfVolume::dim`.
    pub dim: u32,
    /// The mesh's padded local AABB (matches the source `SdfVolume`).
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

/// A set of per-mesh SDFs packed into one atlas volume, `tiles[i]` describing mesh `i`.
pub struct SdfAtlas {
    /// Atlas volume dimensions (voxels): `[x, y, z]`. May be non-cubic.
    pub dim: [u32; 3],
    /// `dim.x * dim.y * dim.z` signed distances, `idx = x + dim.x*(y + dim.y*z)` — the linear
    /// order [`crate::sdf::SdfVolume`] / the GPU volume upload use.
    pub voxels: Vec<f32>,
    /// One entry per input mesh, in input order.
    pub tiles: Vec<AtlasTile>,
}

/// Gutter thickness (voxels) around every tile. One replicated-edge voxel is enough: a sample
/// with `t` saturated to `[0,1]` reaches continuous atlas coord `origin_inner + d`, whose
/// trilinear neighbour is the single gutter voxel (weight 0 at the exact edge), never the
/// adjacent tile.
const GUTTER: u32 = 1;

impl SdfAtlas {
    /// Pack `meshes` into one atlas. Deterministic: the placement depends only on the meshes'
    /// dims and their input order. An empty input yields a 1³ empty atlas.
    pub fn pack(meshes: &[SdfVolume]) -> SdfAtlas {
        if meshes.is_empty() {
            return SdfAtlas {
                dim: [1, 1, 1],
                voxels: vec![0.0],
                tiles: Vec::new(),
            };
        }

        // Footprint side (voxels) of each tile including the gutter on both sides.
        let side = |m: &SdfVolume| m.dim + 2 * GUTTER;
        let max_side = meshes.iter().map(side).max().unwrap();
        // Roughly cubic atlas: a square X/Z footprint sized so the shelf grows to ~its own
        // extent in Y. Bump 15 % for shelf waste and clamp up to the largest single tile.
        let total: u64 = meshes.iter().map(|m| (side(m) as u64).pow(3)).sum();
        let foot = ((total as f64).cbrt() * 1.15).ceil() as u32;
        let foot = foot.max(max_side);

        // 3D shelf placement: advance X within a Z-row (row depth = tallest tile in the row),
        // wrap to a new Z-row, then a new Y-layer (layer height = tallest tile in the layer).
        let mut tiles = Vec::with_capacity(meshes.len());
        let (mut x, mut y, mut z) = (0u32, 0u32, 0u32);
        let (mut row_depth, mut layer_height) = (0u32, 0u32);
        let mut atlas_w = 0u32;
        let mut atlas_d = 0u32;
        for m in meshes {
            let s = side(m);
            if x + s > foot && x > 0 {
                x = 0;
                z += row_depth;
                row_depth = 0;
            }
            if z + s > foot && z > 0 {
                z = 0;
                y += layer_height;
                layer_height = 0;
            }
            tiles.push(AtlasTile {
                origin: [x + GUTTER, y + GUTTER, z + GUTTER],
                dim: m.dim,
                aabb_min: m.aabb_min,
                aabb_max: m.aabb_max,
            });
            x += s;
            row_depth = row_depth.max(s);
            layer_height = layer_height.max(s);
            atlas_w = atlas_w.max(x);
            atlas_d = atlas_d.max(z + row_depth);
        }
        let atlas_h = y + layer_height;
        let dim = [atlas_w.max(1), atlas_h.max(1), atlas_d.max(1)];

        // Fill: each tile's footprint (inner + gutter) reads the mesh with clamp-to-edge, so the
        // gutter replicates the boundary voxel. Untouched atlas voxels stay 0 (never sampled —
        // no tile maps there — but a benign value keeps the field finite).
        let (ax, ay) = (dim[0] as usize, dim[1] as usize);
        let mut voxels = vec![0.0f32; ax * ay * dim[2] as usize];
        for (m, t) in meshes.iter().zip(&tiles) {
            let d = m.dim as i32;
            let base = [
                t.origin[0] as i32 - GUTTER as i32,
                t.origin[1] as i32 - GUTTER as i32,
                t.origin[2] as i32 - GUTTER as i32,
            ];
            let s = side(m) as i32;
            for fz in 0..s {
                let mz = (fz - GUTTER as i32).clamp(0, d - 1);
                let az = (base[2] + fz) as usize;
                for fy in 0..s {
                    let my = (fy - GUTTER as i32).clamp(0, d - 1);
                    let ay_ = (base[1] + fy) as usize;
                    for fx in 0..s {
                        let mx = (fx - GUTTER as i32).clamp(0, d - 1);
                        let ax_ = (base[0] + fx) as usize;
                        let src = (mx + d * (my + d * mz)) as usize;
                        voxels[ax_ + ax * (ay_ + ay * az)] = m.voxels[src];
                    }
                }
            }
        }

        SdfAtlas { dim, voxels, tiles }
    }

    /// The `(uvw_bias, uvw_scale)` for tile `i`: atlas UVW = `uvw_bias + t * uvw_scale`, where
    /// `t` is the mesh-local normalized coordinate. See the module docs for the derivation.
    pub fn tile_uvw(&self, i: usize) -> ([f32; 3], [f32; 3]) {
        let t = &self.tiles[i];
        let a = [self.dim[0] as f32, self.dim[1] as f32, self.dim[2] as f32];
        let bias = [
            t.origin[0] as f32 / a[0],
            t.origin[1] as f32 / a[1],
            t.origin[2] as f32 / a[2],
        ];
        let scale = [
            t.dim as f32 / a[0],
            t.dim as f32 / a[1],
            t.dim as f32 / a[2],
        ];
        (bias, scale)
    }

    /// The atlas voxels as little-endian f32 bytes — the layout `Device::create_volume_init`
    /// (and the `.dcasset` SDF chunk) expect.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.voxels.len() * 4);
        for v in &self.voxels {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `dim³` `SdfVolume` over `[min,max]` whose voxel value is a deterministic function of
    /// its voxel index, so the atlas round-trip can be checked exactly.
    fn ramp_volume(dim: u32, min: [f32; 3], max: [f32; 3], seed: f32) -> SdfVolume {
        let n = (dim * dim * dim) as usize;
        let voxels = (0..n).map(|i| seed + i as f32 * 0.01).collect();
        SdfVolume {
            dim,
            aabb_min: min,
            aabb_max: max,
            voxels,
        }
    }

    /// Reference trilinear tap on the atlas (voxel-center clamp addressing, GPU convention).
    fn atlas_sample(atlas: &SdfAtlas, uvw: [f32; 3]) -> f32 {
        let a = [
            atlas.dim[0] as f32,
            atlas.dim[1] as f32,
            atlas.dim[2] as f32,
        ];
        let mut g = [0.0f32; 3];
        for k in 0..3 {
            g[k] = (uvw[k] * a[k] - 0.5).clamp(0.0, a[k] - 1.0);
        }
        let i0 = [g[0] as u32, g[1] as u32, g[2] as u32];
        let i1 = [
            (i0[0] + 1).min(atlas.dim[0] - 1),
            (i0[1] + 1).min(atlas.dim[1] - 1),
            (i0[2] + 1).min(atlas.dim[2] - 1),
        ];
        let f = [
            g[0] - i0[0] as f32,
            g[1] - i0[1] as f32,
            g[2] - i0[2] as f32,
        ];
        let (dx, dy) = (atlas.dim[0], atlas.dim[1]);
        let at = |x: u32, y: u32, z: u32| atlas.voxels[(x + dx * (y + dy * z)) as usize];
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(at(i0[0], i0[1], i0[2]), at(i1[0], i0[1], i0[2]), f[0]);
        let c10 = lerp(at(i0[0], i1[1], i0[2]), at(i1[0], i1[1], i0[2]), f[0]);
        let c01 = lerp(at(i0[0], i0[1], i1[2]), at(i1[0], i0[1], i1[2]), f[0]);
        let c11 = lerp(at(i0[0], i1[1], i1[2]), at(i1[0], i1[1], i1[2]), f[0]);
        lerp(lerp(c00, c10, f[1]), lerp(c01, c11, f[1]), f[2])
    }

    #[test]
    fn empty_input_is_benign() {
        let atlas = SdfAtlas::pack(&[]);
        assert_eq!(atlas.dim, [1, 1, 1]);
        assert!(atlas.tiles.is_empty());
    }

    #[test]
    fn pack_is_deterministic() {
        let meshes = [
            ramp_volume(8, [-1.0; 3], [1.0; 3], 0.0),
            ramp_volume(16, [0.0; 3], [2.0; 3], 5.0),
            ramp_volume(48, [-3.0; 3], [3.0; 3], -2.0),
        ];
        let a = SdfAtlas::pack(&meshes);
        let b = SdfAtlas::pack(&meshes);
        assert_eq!(a.dim, b.dim);
        assert_eq!(a.voxels, b.voxels);
    }

    #[test]
    fn tiles_do_not_overlap() {
        let meshes: Vec<SdfVolume> = (0..20)
            .map(|i| ramp_volume(8 + (i % 5) * 8, [-1.0; 3], [1.0; 3], i as f32))
            .collect();
        let atlas = SdfAtlas::pack(&meshes);
        // Footprints (inner + gutter) must be disjoint: mark every footprint voxel once.
        let (ax, ay, az) = (atlas.dim[0], atlas.dim[1], atlas.dim[2]);
        let mut used = vec![false; (ax * ay * az) as usize];
        for t in &atlas.tiles {
            for dz in 0..t.dim + 2 * GUTTER {
                for dy in 0..t.dim + 2 * GUTTER {
                    for dx in 0..t.dim + 2 * GUTTER {
                        let x = t.origin[0] - GUTTER + dx;
                        let y = t.origin[1] - GUTTER + dy;
                        let z = t.origin[2] - GUTTER + dz;
                        let idx = (x + ax * (y + ay * z)) as usize;
                        assert!(!used[idx], "tile footprints overlap at {x},{y},{z}");
                        used[idx] = true;
                    }
                }
            }
        }
    }

    /// The atlas sampled through `tile_uvw` must reproduce the source mesh's own trilinear
    /// sample — the whole point of the UVW contract. Checks interior + edge points, where the
    /// gutter's clamp-to-edge matters.
    #[test]
    fn atlas_reproduces_mesh_sample() {
        let meshes = [
            ramp_volume(8, [-1.0, -2.0, 0.5], [1.0, 0.0, 3.0], 0.0),
            ramp_volume(32, [0.0; 3], [4.0, 2.0, 1.0], 10.0),
            ramp_volume(48, [-5.0; 3], [5.0; 3], -1.0),
        ];
        let atlas = SdfAtlas::pack(&meshes);
        for (i, m) in meshes.iter().enumerate() {
            let (bias, scale) = atlas.tile_uvw(i);
            // Sweep normalized coords, including the exact edges (t=0, t=1) the gutter guards.
            for &tx in &[0.0f32, 0.1, 0.5, 0.9, 1.0] {
                for &ty in &[0.0f32, 0.37, 1.0] {
                    for &tz in &[0.0f32, 0.63, 1.0] {
                        let t = [tx, ty, tz];
                        // Map t -> a world point in the mesh's AABB, then let SdfVolume::sample
                        // (which re-derives the same t) give the reference.
                        let p = [
                            m.aabb_min[0] + t[0] * (m.aabb_max[0] - m.aabb_min[0]),
                            m.aabb_min[1] + t[1] * (m.aabb_max[1] - m.aabb_min[1]),
                            m.aabb_min[2] + t[2] * (m.aabb_max[2] - m.aabb_min[2]),
                        ];
                        let want = m.sample(p);
                        let uvw = [
                            bias[0] + t[0] * scale[0],
                            bias[1] + t[1] * scale[1],
                            bias[2] + t[2] * scale[2],
                        ];
                        let got = atlas_sample(&atlas, uvw);
                        assert!(
                            (got - want).abs() < 1e-4,
                            "mesh {i} t={t:?}: atlas {got} vs mesh {want}"
                        );
                    }
                }
            }
        }
    }
}
