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

use crate::sdf::{AlbedoVolumes, SdfVolume};

/// One packed tile: where a mesh's `dims`-sized field lives in the atlas, plus the mesh's
/// local AABB (so the sampler can form the mesh-local normalized coordinate `t`).
#[derive(Clone, Debug)]
pub struct AtlasTile {
    /// Inner-block origin in atlas voxels (the mesh's voxel `0` — past the gutter).
    pub origin: [u32; 3],
    /// Inner block edge per axis (voxels) = the mesh's `SdfVolume::dims` (F2 S2a).
    pub dims: [u32; 3],
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

/// Trilinearly resample `src` to `new_dims` over the *same* AABB (a box downsample on any
/// axis where `new_dims[a] < src.dims[a]`). Averaging signed distances preserves the
/// zero-isosurface to first order, so a coarser tile still resolves a mesh's thin features
/// (whose voxel size is set by the tight AABB) while shedding long-axis over-resolution.
/// Deterministic.
fn resample(src: &SdfVolume, new_dims: [u32; 3]) -> SdfVolume {
    let n = [new_dims[0].max(1), new_dims[1].max(1), new_dims[2].max(1)];
    let mut voxels = vec![0.0f32; (n[0] * n[1] * n[2]) as usize];
    let ext = [
        src.aabb_max[0] - src.aabb_min[0],
        src.aabb_max[1] - src.aabb_min[1],
        src.aabb_max[2] - src.aabb_min[2],
    ];
    let inv = [1.0 / n[0] as f32, 1.0 / n[1] as f32, 1.0 / n[2] as f32];
    for z in 0..n[2] {
        for y in 0..n[1] {
            for x in 0..n[0] {
                let p = [
                    src.aabb_min[0] + ext[0] * (x as f32 + 0.5) * inv[0],
                    src.aabb_min[1] + ext[1] * (y as f32 + 0.5) * inv[1],
                    src.aabb_min[2] + ext[2] * (z as f32 + 0.5) * inv[2],
                ];
                voxels[(x + n[0] * (y + n[1] * z)) as usize] = src.sample(p);
            }
        }
    }
    SdfVolume {
        dims: n,
        aabb_min: src.aabb_min,
        aabb_max: src.aabb_max,
        voxels,
    }
}

impl SdfAtlas {
    /// Pack `meshes` into one atlas at their native resolution (no cap). Deterministic.
    pub fn pack(meshes: &[SdfVolume]) -> SdfAtlas {
        Self::pack_capped(meshes, u32::MAX)
    }

    /// Pack `meshes`, first downsampling any axis whose `dims[a] > max_dim` to `max_dim`
    /// (a memory cap on the largest tiles — their extra resolution is low-frequency and
    /// covered by the coarse dense field). `max_dim = u32::MAX` packs at native
    /// resolution. Deterministic.
    pub fn pack_capped(meshes: &[SdfVolume], max_dim: u32) -> SdfAtlas {
        // Downsample over-cap tiles up front, then pack exactly (borrow the capped set).
        let capped: Vec<SdfVolume> = meshes
            .iter()
            .map(|m| {
                if m.dims.iter().any(|&d| d > max_dim) {
                    resample(
                        m,
                        [
                            m.dims[0].min(max_dim),
                            m.dims[1].min(max_dim),
                            m.dims[2].min(max_dim),
                        ],
                    )
                } else {
                    SdfVolume {
                        dims: m.dims,
                        aabb_min: m.aabb_min,
                        aabb_max: m.aabb_max,
                        voxels: m.voxels.clone(),
                    }
                }
            })
            .collect();
        Self::pack_native(&capped)
    }

    /// Pack `meshes` at exactly their current resolution (internal; `pack_capped` handles caps).
    fn pack_native(meshes: &[SdfVolume]) -> SdfAtlas {
        if meshes.is_empty() {
            return SdfAtlas {
                dim: [1, 1, 1],
                voxels: vec![0.0],
                tiles: Vec::new(),
            };
        }

        // Footprint sides (voxels) of each tile per axis, including the gutter on both sides.
        let sides = |m: &SdfVolume| {
            [
                m.dims[0] + 2 * GUTTER,
                m.dims[1] + 2 * GUTTER,
                m.dims[2] + 2 * GUTTER,
            ]
        };
        let max_side = meshes
            .iter()
            .flat_map(|m| sides(m).into_iter())
            .max()
            .unwrap();
        // Roughly cubic atlas: a square X/Z footprint sized so the shelf grows to ~its own
        // extent in Y. Bump 15 % for shelf waste and clamp up to the largest single tile.
        let total: u64 = meshes
            .iter()
            .map(|m| {
                let s = sides(m);
                s[0] as u64 * s[1] as u64 * s[2] as u64
            })
            .sum();
        let foot = ((total as f64).cbrt() * 1.15).ceil() as u32;
        let foot = foot.max(max_side);

        // 3D shelf placement: advance X within a Z-row (row depth = deepest tile in the row),
        // wrap to a new Z-row, then a new Y-layer (layer height = tallest tile in the layer).
        // Per-axis tile sides (F2 S2a): X-advance uses the tile's own width; the row/layer
        // extents track the max depth/height seen, exactly as the cubic shelf did.
        let mut tiles = Vec::with_capacity(meshes.len());
        let (mut x, mut y, mut z) = (0u32, 0u32, 0u32);
        let (mut row_depth, mut layer_height) = (0u32, 0u32);
        let mut atlas_w = 0u32;
        let mut atlas_d = 0u32;
        for m in meshes {
            let s = sides(m);
            if x + s[0] > foot && x > 0 {
                x = 0;
                z += row_depth;
                row_depth = 0;
            }
            if z + s[2] > foot && z > 0 {
                z = 0;
                y += layer_height;
                layer_height = 0;
            }
            tiles.push(AtlasTile {
                origin: [x + GUTTER, y + GUTTER, z + GUTTER],
                dims: m.dims,
                aabb_min: m.aabb_min,
                aabb_max: m.aabb_max,
            });
            x += s[0];
            row_depth = row_depth.max(s[2]);
            layer_height = layer_height.max(s[1]);
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
            let d = [m.dims[0] as i32, m.dims[1] as i32, m.dims[2] as i32];
            let base = [
                t.origin[0] as i32 - GUTTER as i32,
                t.origin[1] as i32 - GUTTER as i32,
                t.origin[2] as i32 - GUTTER as i32,
            ];
            let s = sides(m);
            for fz in 0..s[2] as i32 {
                let mz = (fz - GUTTER as i32).clamp(0, d[2] - 1);
                let az = (base[2] + fz) as usize;
                for fy in 0..s[1] as i32 {
                    let my = (fy - GUTTER as i32).clamp(0, d[1] - 1);
                    let ay_ = (base[1] + fy) as usize;
                    for fx in 0..s[0] as i32 {
                        let mx = (fx - GUTTER as i32).clamp(0, d[0] - 1);
                        let ax_ = (base[0] + fx) as usize;
                        let src = (mx + d[0] * (my + d[1] * mz)) as usize;
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
            t.dims[0] as f32 / a[0],
            t.dims[1] as f32 / a[1],
            t.dims[2] as f32 / a[2],
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

    /// The atlas voxels as little-endian IEEE binary16 bytes (`Format::R16Float` upload) —
    /// the compact storage path (F2 S2b). Distances are mesh-local (metres, bounded by the
    /// mesh diagonal), so half precision holds the sphere-march bound to ~2^-11 of the
    /// stored value; the per-instance `dist_scale` is applied at sample time unchanged.
    pub fn to_le_bytes_f16(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.voxels.len() * 2);
        for v in &self.voxels {
            out.extend_from_slice(&f32_to_f16_bits(*v).to_le_bytes());
        }
        out
    }
}

/// One f32 → IEEE-754 binary16 bit pattern, round-to-nearest-even. Pure integer ops —
/// deterministic and backend/platform-independent (the f16 atlas bytes are part of the
/// run-to-run byte-identity gate). Overflow clamps to the largest finite half (±65504)
/// instead of infinity so a downstream `min()` union always sees a finite distance;
/// the SDF bake never produces values near that range.
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x007f_ffff;

    if exp == 255 {
        // Inf / NaN (not produced by the bake; preserved for correctness).
        return sign | 0x7c00 | (((man != 0) as u16) << 9);
    }
    let e = exp - 112; // re-bias 127 -> 15
    if e >= 31 {
        return sign | 0x7bff; // clamp to max finite half
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflows to signed zero
        }
        // Subnormal half: restore the implicit leading 1, shift into place, RNE.
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32; // 13 mantissa bits dropped + (1 - e) extra
        let halfway = 1u32 << (shift - 1);
        let rounded = (m + halfway - 1 + ((m >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    // Normal: drop 13 mantissa bits with round-half-to-even; a mantissa carry
    // propagates into the exponent via the addition below.
    let m = (man + 0x0fff + ((man >> 13) & 1)) >> 13;
    sign | (((e as u32) << 10) + m) as u16
}

/// Trilinearly resample one `src_dims`-sized channel to `new_dims` over the *same*
/// normalized `[0,1]³` box (voxel-center clamp addressing, the GPU convention). Used to
/// match an albedo tile to the SDF tile's (possibly capped) dims so a single UVW mapping
/// addresses both. Deterministic; a box downsample on shrinking axes, identity when equal.
fn resample_channel(src: &[f32], src_dims: [u32; 3], new_dims: [u32; 3]) -> Vec<f32> {
    let sd = [src_dims[0].max(1), src_dims[1].max(1), src_dims[2].max(1)];
    let n = [new_dims[0].max(1), new_dims[1].max(1), new_dims[2].max(1)];
    if sd == n {
        return src.to_vec();
    }
    let sample = |t: [f32; 3]| -> f32 {
        // Continuous voxel-center coord, clamped (matches SdfVolume::sample / the GPU sampler).
        let mut g = [0.0f32; 3];
        for a in 0..3 {
            g[a] = (t[a] * sd[a] as f32 - 0.5).clamp(0.0, sd[a] as f32 - 1.0);
        }
        let i0 = [g[0] as u32, g[1] as u32, g[2] as u32];
        let i1 = [
            (i0[0] + 1).min(sd[0] - 1),
            (i0[1] + 1).min(sd[1] - 1),
            (i0[2] + 1).min(sd[2] - 1),
        ];
        let f = [
            g[0] - i0[0] as f32,
            g[1] - i0[1] as f32,
            g[2] - i0[2] as f32,
        ];
        let at = |x: u32, y: u32, z: u32| src[(x + sd[0] * (y + sd[1] * z)) as usize];
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(at(i0[0], i0[1], i0[2]), at(i1[0], i0[1], i0[2]), f[0]);
        let c10 = lerp(at(i0[0], i1[1], i0[2]), at(i1[0], i1[1], i0[2]), f[0]);
        let c01 = lerp(at(i0[0], i0[1], i1[2]), at(i1[0], i0[1], i1[2]), f[0]);
        let c11 = lerp(at(i0[0], i1[1], i1[2]), at(i1[0], i1[1], i1[2]), f[0]);
        lerp(lerp(c00, c10, f[1]), lerp(c01, c11, f[1]), f[2])
    };
    let inv = [1.0 / n[0] as f32, 1.0 / n[1] as f32, 1.0 / n[2] as f32];
    let mut out = vec![0.0f32; (n[0] * n[1] * n[2]) as usize];
    for z in 0..n[2] {
        for y in 0..n[1] {
            for x in 0..n[0] {
                let t = [
                    (x as f32 + 0.5) * inv[0],
                    (y as f32 + 0.5) * inv[1],
                    (z as f32 + 0.5) * inv[2],
                ];
                out[(x + n[0] * (y + n[1] * z)) as usize] = sample(t);
            }
        }
    }
    out
}

/// A set of per-mesh **albedo** fields packed into three atlas channel volumes (R/G/B),
/// reusing an [`SdfAtlas`]'s exact tile geometry (gi-fidelity-phases.md, F5 S2).
///
/// The whole point of F5 is that hit *colour* gets the same per-mesh precision the SDF already
/// has: instead of the coarse whole-scene albedo grid (which blurs colour across neighbouring
/// meshes), each mesh's albedo lives in a tile sampled through the *same* `tile_uvw` mapping the
/// SDF uses — so `mesh_sdf_sample.slang` reads the hit instance's own colour at the same UVW it
/// used for the distance. Because the tiles share origins/dims/atlas-dims with the SDF atlas,
/// there is exactly one tile contract for both (single source).
pub struct AlbedoAtlas {
    /// Atlas volume dimensions (voxels) — **identical** to the paired [`SdfAtlas::dim`].
    pub dim: [u32; 3],
    /// Three `dim.x*dim.y*dim.z` channels (R/G/B), `idx = x + dim.x*(y + dim.y*z)`.
    pub channels: [Vec<f32>; 3],
}

impl AlbedoAtlas {
    /// Pack `albedos[i]` (the albedo of mesh `i`, baked over the same local grid as its SDF)
    /// into the layout of `sdf_atlas` — same tile origins, same tile `dim`, same atlas `dim`.
    /// Each albedo, baked at the mesh's native `dim`, is resampled to the (possibly capped) SDF
    /// tile `dim` so the two tiles coincide voxel-for-voxel. A missing/mismatched albedo fills
    /// its tile with `fallback` (a benign neutral) so the atlas is always well-formed.
    /// Deterministic — a pure function of the SDF atlas + albedos.
    pub fn pack_like(sdf_atlas: &SdfAtlas, albedos: &[AlbedoVolumes], fallback: [f32; 3]) -> Self {
        let dim = sdf_atlas.dim;
        let (ax, ay) = (dim[0] as usize, dim[1] as usize);
        let n = ax * ay * dim[2] as usize;
        let mut channels = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];

        for (i, tile) in sdf_atlas.tiles.iter().enumerate() {
            let d = tile.dims;
            // The per-channel source at the tile's dims (resampled from the albedo's native
            // dims, exactly as the SDF tile was capped), or the neutral fallback when absent.
            let src: [Vec<f32>; 3] = match albedos.get(i) {
                Some(a) => [
                    resample_channel(&a.channels[0], a.dims, d),
                    resample_channel(&a.channels[1], a.dims, d),
                    resample_channel(&a.channels[2], a.dims, d),
                ],
                None => {
                    let m = (d[0] * d[1] * d[2]) as usize;
                    [
                        vec![fallback[0]; m],
                        vec![fallback[1]; m],
                        vec![fallback[2]; m],
                    ]
                }
            };
            let di = [d[0] as i32, d[1] as i32, d[2] as i32];
            let base = [
                tile.origin[0] as i32 - GUTTER as i32,
                tile.origin[1] as i32 - GUTTER as i32,
                tile.origin[2] as i32 - GUTTER as i32,
            ];
            let s = [
                (d[0] + 2 * GUTTER) as i32,
                (d[1] + 2 * GUTTER) as i32,
                (d[2] + 2 * GUTTER) as i32,
            ];
            for fz in 0..s[2] {
                let mz = (fz - GUTTER as i32).clamp(0, di[2] - 1);
                let az = (base[2] + fz) as usize;
                for fy in 0..s[1] {
                    let my = (fy - GUTTER as i32).clamp(0, di[1] - 1);
                    let ay_ = (base[1] + fy) as usize;
                    for fx in 0..s[0] {
                        let mx = (fx - GUTTER as i32).clamp(0, di[0] - 1);
                        let ax_ = (base[0] + fx) as usize;
                        let sidx = (mx + di[0] * (my + di[1] * mz)) as usize;
                        let aidx = ax_ + ax * (ay_ + ay * az);
                        for c in 0..3 {
                            channels[c][aidx] = src[c][sidx];
                        }
                    }
                }
            }
        }
        AlbedoAtlas { dim, channels }
    }

    /// Channel `c` (0=R,1=G,2=B) as little-endian f32 bytes for `Device::create_volume_init`.
    pub fn channel_le_bytes(&self, c: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.channels[c].len() * 4);
        for v in &self.channels[c] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Channel `c` as little-endian binary16 bytes (`Format::R16Float`, F2 S2b). Albedo is
    /// [0,1] colour — half precision is ~3 decimal digits there, far past display precision.
    pub fn channel_le_bytes_f16(&self, c: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.channels[c].len() * 2);
        for v in &self.channels[c] {
            out.extend_from_slice(&f32_to_f16_bits(*v).to_le_bytes());
        }
        out
    }
}

// --- F2 S1: sparse-brick occupancy analysis (gi-fidelity-phases.md, F2) ---------------
//
// A dense `dim³` tile spends most of its voxels on far-from-surface distance, which is pure
// low-frequency filler: the coarse whole-scene field already carries long-range distance, and
// the sampler only needs the per-mesh field *near the surface* to pull the iso-surface onto the
// mesh's thin geometry. So a mesh's field can be split into small **bricks** and only the bricks
// the surface passes through need be stored; the rest are an "empty" marker that falls back to
// the coarse field. This module is the CPU **analysis + measurement** for that split — it
// classifies bricks and reports the projected memory win, so S2 (the GPU brick atlas +
// indirection volume) can be built against a measured, unit-tested data structure. No GPU wiring
// here; the pack contract above is untouched.

/// Brick edge (voxels). A mesh field is split into `ceil(dim / BRICK_DIM)³` bricks. 8 keeps
/// bricks small enough that empty ones are a real saving, while large enough that the
/// per-brick indirection overhead (S2) stays modest.
pub const BRICK_DIM: u32 = 8;

/// Per-brick occupancy classification of one mesh field: which bricks the surface passes
/// through (within a distance band) versus empty. Deterministic — a pure function of the field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrickOccupancy {
    /// The mesh's `SdfVolume::dims`.
    pub dims: [u32; 3],
    /// Brick-grid resolution per axis = `dims[a].div_ceil(BRICK_DIM)`.
    pub bricks_per_axis: [u32; 3],
    /// One flag per brick, `bidx = bx + bpa[0]*(by + bpa[1]*bz)`: `true` if the brick is
    /// occupied (holds a voxel with `|distance| < band`), `false` if empty.
    pub occupied: Vec<bool>,
}

impl BrickOccupancy {
    /// Total brick count (`bpa[0]*bpa[1]*bpa[2]`).
    pub fn brick_count(&self) -> u32 {
        self.bricks_per_axis[0] * self.bricks_per_axis[1] * self.bricks_per_axis[2]
    }

    /// How many bricks are occupied.
    pub fn occupied_count(&self) -> u32 {
        self.occupied.iter().filter(|&&b| b).count() as u32
    }
}

/// A per-mesh band width (world distance units) below which a voxel counts as "near the
/// surface", derived from the mesh's own voxel size: `band = margin * voxel_edge`, where
/// `voxel_edge` is the largest per-axis voxel spacing (`extent / dim`). Measuring in voxels
/// makes the band mesh-relative — a coarse mesh gets a proportionally wider band — so a brick
/// straddling the surface is classified occupied regardless of the mesh's scale. `margin`
/// controls how many voxels of shell around the zero-isosurface are kept resident; a band of a
/// few voxels captures the trilinear neighbourhood the sampler actually reads.
pub fn brick_band_for(vol: &SdfVolume, margin: f32) -> f32 {
    let voxel_edge = (0..3)
        .map(|a| (vol.aabb_max[a] - vol.aabb_min[a]).abs() / vol.dims[a].max(1) as f32)
        .fold(0.0f32, f32::max);
    margin * voxel_edge.max(f32::MIN_POSITIVE)
}

/// Classify each `BRICK_DIM³` brick of `vol` as occupied (`|distance| < band` for some voxel it
/// covers) or empty. `band` is a world-space distance (see [`brick_band_for`]). Deterministic:
/// a pure scan of the voxels, so the same field always yields the same flags.
///
/// A brick covers voxels `[bx*BRICK_DIM .. min((bx+1)*BRICK_DIM, dim))` per axis (the last brick
/// along each axis is partial when `dim` is not a multiple of `BRICK_DIM`). Any covered voxel
/// within the band marks the whole brick occupied — conservative, so the surface is never
/// dropped from residency.
pub fn classify_bricks(vol: &SdfVolume, band: f32) -> BrickOccupancy {
    let dims = vol.dims;
    let bpa = [
        dims[0].div_ceil(BRICK_DIM).max(1),
        dims[1].div_ceil(BRICK_DIM).max(1),
        dims[2].div_ceil(BRICK_DIM).max(1),
    ];
    let mut occupied = vec![false; (bpa[0] * bpa[1] * bpa[2]) as usize];
    let at = |x: u32, y: u32, z: u32| vol.voxels[(x + dims[0] * (y + dims[1] * z)) as usize];
    for bz in 0..bpa[2] {
        for by in 0..bpa[1] {
            for bx in 0..bpa[0] {
                let mut hit = false;
                let z0 = bz * BRICK_DIM;
                let z1 = ((bz + 1) * BRICK_DIM).min(dims[2]);
                let y0 = by * BRICK_DIM;
                let y1 = ((by + 1) * BRICK_DIM).min(dims[1]);
                let x0 = bx * BRICK_DIM;
                let x1 = ((bx + 1) * BRICK_DIM).min(dims[0]);
                'scan: for z in z0..z1 {
                    for y in y0..y1 {
                        for x in x0..x1 {
                            if at(x, y, z).abs() < band {
                                hit = true;
                                break 'scan;
                            }
                        }
                    }
                }
                occupied[(bx + bpa[0] * (by + bpa[1] * bz)) as usize] = hit;
            }
        }
    }
    BrickOccupancy {
        dims,
        bricks_per_axis: bpa,
        occupied,
    }
}

/// The measured sparse-brick win for a set of meshes, versus the current dense atlas.
///
/// All byte figures count only the f32 distance payload of the *tiles* (the gutter/shelf waste
/// of a concrete atlas pack is not modelled here — this is an apples-to-apples payload
/// comparison of dense vs. brick storage, the quantity S2 actually trades). "Sparse" adds a
/// 1-voxel gutter *per brick* (so a brick tap clamps to its own edge, exactly as the dense
/// tiles do today), which is the real cost the S2 brick atlas will pay.
#[derive(Clone, Debug, PartialEq)]
pub struct BrickAnalysis {
    /// Number of meshes analysed.
    pub mesh_count: usize,
    /// Sum over meshes of `bricks_per_axis³`.
    pub total_bricks: u64,
    /// Sum over meshes of occupied bricks.
    pub occupied_bricks: u64,
    /// `occupied_bricks / total_bricks` (0 when there are no bricks).
    pub occupied_fraction: f32,
    /// Dense payload: sum of `dim³` voxels × 4 bytes (the tile payload the current atlas stores).
    pub dense_bytes: u64,
    /// Sparse payload: sum over occupied bricks of `(BRICK_DIM + 2)³` voxels × 4 bytes (each
    /// brick padded by a 1-voxel gutter, matching the atlas's clamp-to-edge tap convention).
    pub sparse_bytes: u64,
}

impl BrickAnalysis {
    /// `sparse_bytes / dense_bytes` (1.0 when dense is empty) — the memory fraction the brick
    /// scheme projects. Lower is better; a value above 1 means the per-brick gutter overhead
    /// outweighs the emptiness (only for meshes that are almost entirely surface).
    pub fn memory_ratio(&self) -> f32 {
        if self.dense_bytes == 0 {
            1.0
        } else {
            self.sparse_bytes as f32 / self.dense_bytes as f32
        }
    }
}

/// Analyse the sparse-brick occupancy over `meshes`, deriving each mesh's band from its own
/// voxel size (`margin` voxels of shell; see [`brick_band_for`]). Returns the aggregate
/// occupied-brick fraction and the projected dense-vs-sparse payload bytes. Deterministic.
///
/// This is the F2 measurement primitive: it quantifies "how much of the dense tile is actually
/// near the surface" for a real mesh set, so the S2 brick atlas can be justified and sized
/// before any GPU code is written.
pub fn analyze_bricks(meshes: &[SdfVolume], margin: f32) -> BrickAnalysis {
    let gutter_side = (BRICK_DIM + 2 * GUTTER) as u64;
    let brick_payload = gutter_side * gutter_side * gutter_side * 4;

    let mut total_bricks = 0u64;
    let mut occupied_bricks = 0u64;
    let mut dense_bytes = 0u64;
    let mut sparse_bytes = 0u64;
    for m in meshes {
        let band = brick_band_for(m, margin);
        let occ = classify_bricks(m, band);
        total_bricks += occ.brick_count() as u64;
        let occ_n = occ.occupied_count() as u64;
        occupied_bricks += occ_n;
        dense_bytes += m.voxel_count() as u64 * 4;
        sparse_bytes += occ_n * brick_payload;
    }
    let occupied_fraction = if total_bricks == 0 {
        0.0
    } else {
        occupied_bricks as f32 / total_bricks as f32
    };
    BrickAnalysis {
        mesh_count: meshes.len(),
        total_bricks,
        occupied_bricks,
        occupied_fraction,
        dense_bytes,
        sparse_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference binary16 → f32 (test-only), for round-trip checks of `f32_to_f16_bits`.
    fn f16_bits_to_f32(h: u16) -> f32 {
        let sign = ((h >> 15) & 1) as u32;
        let exp = ((h >> 10) & 0x1f) as u32;
        let man = (h & 0x3ff) as u32;
        let bits = if exp == 0 {
            if man == 0 {
                sign << 31
            } else {
                // Subnormal: value = man * 2^-24.
                return (if sign == 1 { -1.0 } else { 1.0 }) * man as f32 * (-24f32).exp2();
            }
        } else if exp == 31 {
            (sign << 31) | 0x7f80_0000 | (man << 13)
        } else {
            (sign << 31) | ((exp + 112) << 23) | (man << 13)
        };
        f32::from_bits(bits)
    }

    /// Exactly-representable values must convert losslessly; everything in the SDF's
    /// working range must round-trip within half-precision epsilon (2^-11 relative).
    #[test]
    fn f16_conversion_roundtrip() {
        for v in [0.0f32, -0.0, 1.0, -1.0, 0.5, -2.0, 65504.0, -65504.0, 0.25] {
            assert_eq!(f16_bits_to_f32(f32_to_f16_bits(v)), v, "exact {v}");
        }
        // Signed-zero encodings.
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        // Working-range values (mesh-local distances, metres): relative error <= 2^-11.
        let mut v = 1e-3f32;
        while v < 100.0 {
            for s in [v, -v] {
                let rt = f16_bits_to_f32(f32_to_f16_bits(s));
                assert!(
                    (rt - s).abs() <= s.abs() * (2f32).powi(-11) + 1e-7,
                    "{s} -> {rt}"
                );
            }
            v *= 1.7;
        }
        // Overflow clamps to the largest finite half (never Inf — min() unions stay finite).
        assert_eq!(f32_to_f16_bits(1e9), 0x7bff);
        assert_eq!(f32_to_f16_bits(-1e9), 0xfbff);
    }

    /// The f16 byte stream is exactly the per-voxel conversion of the f32 stream, half the
    /// size, and deterministic across calls.
    #[test]
    fn f16_bytes_match_voxels() {
        let mesh = ramp_volume(8, [-1.0; 3], [1.0; 3], -0.37);
        let atlas = SdfAtlas::pack(std::slice::from_ref(&mesh));
        let b32 = atlas.to_le_bytes();
        let b16 = atlas.to_le_bytes_f16();
        assert_eq!(b16.len() * 2, b32.len());
        assert_eq!(b16, atlas.to_le_bytes_f16(), "deterministic");
        for (i, v) in atlas.voxels.iter().enumerate() {
            let got = u16::from_le_bytes([b16[i * 2], b16[i * 2 + 1]]);
            assert_eq!(got, f32_to_f16_bits(*v), "voxel {i}");
        }
    }

    /// A `dim³` `SdfVolume` over `[min,max]` whose voxel value is a deterministic function of
    /// its voxel index, so the atlas round-trip can be checked exactly.
    fn ramp_volume(dim: u32, min: [f32; 3], max: [f32; 3], seed: f32) -> SdfVolume {
        let n = (dim * dim * dim) as usize;
        let voxels = (0..n).map(|i| seed + i as f32 * 0.01).collect();
        SdfVolume {
            dims: [dim; 3],
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
    fn capped_pack_downsamples_large_tiles() {
        // A 48³ mesh capped to 16 must occupy a 16³ tile; a 12³ mesh (< cap) is untouched.
        let big = ramp_volume(48, [-1.0; 3], [1.0; 3], 0.0);
        let small = ramp_volume(12, [0.0; 3], [1.0; 3], 3.0);
        let atlas = SdfAtlas::pack_capped(&[big, small], 16);
        assert_eq!(
            atlas.tiles[0].dims, [16; 3],
            "large tile downsampled to the cap"
        );
        assert_eq!(
            atlas.tiles[1].dims, [12; 3],
            "small tile left at native dim"
        );
    }

    #[test]
    fn resample_preserves_a_linear_field() {
        // A field linear in world space resamples exactly (trilinear is exact on linear data),
        // so the cap can't distort a smooth SDF beyond its own discretisation.
        let dim = 32u32;
        let (mn, mx) = ([-2.0f32; 3], [2.0f32; 3]);
        let n = (dim * dim * dim) as usize;
        let mut voxels = vec![0.0f32; n];
        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    // world x at this voxel center
                    let wx = mn[0] + (mx[0] - mn[0]) * (x as f32 + 0.5) / dim as f32;
                    voxels[(x + dim * (y + dim * z)) as usize] = wx;
                }
            }
        }
        let src = SdfVolume {
            dims: [dim; 3],
            aabb_min: mn,
            aabb_max: mx,
            voxels,
        };
        let ds = resample(&src, [16; 3]);
        // A voxel center in the coarse grid maps to the same world x → same value.
        for i in 0..16usize {
            // row y=z=0, so the linear index is just i
            let wx = mn[0] + (mx[0] - mn[0]) * (i as f32 + 0.5) / 16.0;
            let got = ds.voxels[i];
            assert!((got - wx).abs() < 1e-3, "i={i}: {got} vs {wx}");
        }
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
            for dz in 0..t.dims[2] + 2 * GUTTER {
                for dy in 0..t.dims[1] + 2 * GUTTER {
                    for dx in 0..t.dims[0] + 2 * GUTTER {
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

    // --- F2 S1: sparse-brick occupancy analysis ---------------------------------------

    /// A `dim³` field whose voxel value is the signed distance (in AABB-normalized units) to a
    /// planar surface at normalized `x = plane_t` — a controlled "surface" whose occupied bricks
    /// are analytically known. Distance is `(x_center - plane_t)` scaled to the AABB extent, so
    /// the band test behaves like a real world-space SDF over `[0,1]³`.
    fn plane_field(dim: u32, plane_t: f32) -> SdfVolume {
        let n = (dim * dim * dim) as usize;
        let mut voxels = vec![0.0f32; n];
        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    let xc = (x as f32 + 0.5) / dim as f32; // voxel-center normalized x
                    voxels[(x + dim * (y + dim * z)) as usize] = xc - plane_t;
                }
            }
        }
        SdfVolume {
            dims: [dim; 3],
            aabb_min: [0.0; 3],
            aabb_max: [1.0; 3],
            voxels,
        }
    }

    #[test]
    fn classify_occupies_only_bricks_the_surface_crosses() {
        // 16³ field → 2 bricks/axis (8-wide bricks). A plane at x=0.5 sits exactly on the brick
        // boundary; a band of one voxel keeps it out of both x-slabs' far ends. Use a plane at
        // x=0.30 so only the low-x brick column straddles it, with a tight band.
        let dim = 16u32;
        let vol = plane_field(dim, 0.30);
        // voxel_edge = 1/16 = 0.0625; band = 1.5 voxels = 0.09375 in normalized units.
        let band = brick_band_for(&vol, 1.5);
        let occ = classify_bricks(&vol, band);
        assert_eq!(occ.bricks_per_axis, [2; 3]);
        assert_eq!(occ.brick_count(), 8);
        // The surface at x≈0.30 lives in the low-x bricks (voxels x=0..7 span 0.03..0.47), and
        // band 0.094 reaches those bricks only. All 4 low-x bricks (by,bz any) are occupied; the
        // 4 high-x bricks (x=8..15 span 0.53..0.97, min |dist| = 0.53-0.30 = 0.23 > band) empty.
        for bz in 0..2 {
            for by in 0..2 {
                let lo = occ.occupied[(2 * (by + 2 * bz)) as usize];
                let hi = occ.occupied[(1 + 2 * (by + 2 * bz)) as usize];
                assert!(lo, "low-x brick (by={by},bz={bz}) must be occupied");
                assert!(!hi, "high-x brick (by={by},bz={bz}) must be empty");
            }
        }
        assert_eq!(occ.occupied_count(), 4);
    }

    #[test]
    fn classify_marks_all_bricks_for_a_surface_through_the_centre() {
        // A plane at x=0.5 with a wide band (3 voxels) reaches into every x-column, so with the
        // surface plus band spanning the middle, only the outermost bricks could be empty. Check
        // a mid plane with a generous band occupies at least the central column and is symmetric.
        let dim = 24u32; // 3 bricks/axis
        let vol = plane_field(dim, 0.5);
        let band = brick_band_for(&vol, 2.0); // ~2 voxels
        let occ = classify_bricks(&vol, band);
        assert_eq!(occ.bricks_per_axis, [3; 3]);
        // The middle x-brick (bx=1, voxels x=8..15, centers 0.354..0.646) straddles x=0.5 → all
        // 9 of its (by,bz) entries occupied; the outer x-bricks are far → empty.
        let mut mid = 0;
        for bz in 0..3 {
            for by in 0..3 {
                if occ.occupied[(1 + 3 * (by + 3 * bz)) as usize] {
                    mid += 1;
                }
            }
        }
        assert_eq!(mid, 9, "entire central x-brick column occupied");
    }

    #[test]
    fn classify_is_deterministic() {
        let vol = plane_field(24, 0.42);
        let band = brick_band_for(&vol, 1.7);
        let a = classify_bricks(&vol, band);
        let b = classify_bricks(&vol, band);
        assert_eq!(a, b, "brick classification must be deterministic");
    }

    #[test]
    fn band_scales_with_voxel_size() {
        // Two fields, same dim but 2× AABB extent → 2× voxel edge → 2× band for the same margin.
        let small = SdfVolume {
            dims: [8; 3],
            aabb_min: [0.0; 3],
            aabb_max: [1.0; 3],
            voxels: vec![0.0; 512],
        };
        let big = SdfVolume {
            dims: [8; 3],
            aabb_min: [0.0; 3],
            aabb_max: [2.0; 3],
            voxels: vec![0.0; 512],
        };
        let bs = brick_band_for(&small, 2.0);
        let bb = brick_band_for(&big, 2.0);
        assert!(
            (bb - 2.0 * bs).abs() < 1e-6,
            "band ∝ voxel edge: {bs} vs {bb}"
        );
        // margin 2 voxels over a 1/8 = 0.125 edge → 0.25.
        assert!(
            (bs - 0.25).abs() < 1e-6,
            "small band = 2 * 0.125 = 0.25, got {bs}"
        );
    }

    #[test]
    fn analysis_projects_a_memory_win_for_a_mostly_empty_field() {
        // A single plane field: most of the volume is far-from-surface, so most bricks are empty
        // and the sparse payload is a fraction of the dense payload despite the per-brick gutter.
        let vol = plane_field(48, 0.5); // 6 bricks/axis = 216 bricks
        let report = analyze_bricks(std::slice::from_ref(&vol), 2.0);
        assert_eq!(report.mesh_count, 1);
        assert_eq!(report.total_bricks, 6 * 6 * 6);
        // A single planar surface occupies only the x-columns it crosses → a slab of bricks, well
        // under half the volume.
        assert!(
            report.occupied_fraction < 0.5,
            "a plane should leave most bricks empty, got {}",
            report.occupied_fraction
        );
        assert!(
            report.memory_ratio() < 1.0,
            "sparse payload must beat dense for a mostly-empty field, got {}",
            report.memory_ratio()
        );
        // Dense payload is exactly dim³ * 4.
        assert_eq!(report.dense_bytes, 48u64.pow(3) * 4);
        // Sparse payload is occupied_bricks * (8+2)³ * 4.
        assert_eq!(report.sparse_bytes, report.occupied_bricks * 1000 * 4);
    }

    #[test]
    fn analysis_is_deterministic_over_a_mesh_set() {
        let meshes = [
            plane_field(16, 0.3),
            plane_field(32, 0.6),
            plane_field(48, 0.5),
        ];
        let a = analyze_bricks(&meshes, 1.5);
        let b = analyze_bricks(&meshes, 1.5);
        assert_eq!(a, b, "aggregate brick analysis must be deterministic");
    }

    #[test]
    fn analysis_of_empty_set_is_benign() {
        let report = analyze_bricks(&[], 2.0);
        assert_eq!(report.mesh_count, 0);
        assert_eq!(report.total_bricks, 0);
        assert_eq!(report.occupied_fraction, 0.0);
        assert_eq!(report.memory_ratio(), 1.0);
    }

    /// Measurement (ignored — prints numbers, no asset needed). Bakes a representative mix of
    /// real per-mesh SDFs spanning the two regimes the brick win depends on — surface-dominated
    /// shapes (thin sheet, wall) where nearly every brick touches the surface, and
    /// volume-dominated shapes (a big sphere with a far interior, a sparse two-part assembly
    /// with empty space in its AABB) where interior/empty bricks are droppable — through the
    /// production `bake_mesh_sdf`, then reports the occupied-brick fraction and dense-vs-sparse
    /// payload. Run:
    ///   `cargo test -p dreamcoast-asset brick_analysis_report -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement: prints occupied-brick fraction + projected memory"]
    fn brick_analysis_report() {
        use crate::MeshVertex;
        use crate::sdf::bake_mesh_sdf;

        let quad = |sx: f32, sy: f32| -> (Vec<MeshVertex>, Vec<u32>) {
            let mk = |x: f32, y: f32| MeshVertex {
                pos: [x, y, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
            };
            (
                vec![mk(-sx, -sy), mk(sx, -sy), mk(sx, sy), mk(-sx, sy)],
                vec![0u32, 1, 2, 0, 2, 3],
            )
        };
        let sphere_at = |r: f32, c: [f32; 3]| -> (Vec<MeshVertex>, Vec<u32>) {
            let mut s = crate::uv_sphere(32, 24);
            for v in &mut s.vertices {
                v.pos = [
                    v.pos[0] * r + c[0],
                    v.pos[1] * r + c[1],
                    v.pos[2] * r + c[2],
                ];
            }
            (s.vertices, s.indices)
        };

        // Surface-dominated: a 2×2 m thin curtain sheet.
        let (sv, si) = quad(1.0, 1.0);
        let sheet = bake_mesh_sdf(&sv, &si);

        // Surface-dominated: a thin 3×2×0.15 m box wall.
        let cube = crate::unit_cube();
        let mut wv = cube.vertices.clone();
        for v in &mut wv {
            v.pos = [v.pos[0] * 1.5, v.pos[1] * 1.0, v.pos[2] * 0.075];
        }
        let wall = bake_mesh_sdf(&wv, &cube.indices);

        // Volume-dominated: a big solid sphere (r=1.5 m) — the interior is far from the surface,
        // so its inner bricks carry only large-magnitude distance (droppable).
        let (bv, bi) = sphere_at(1.5, [0.0; 3]);
        let big_sphere = bake_mesh_sdf(&bv, &bi);

        // Sparse assembly: two small spheres at opposite corners of one AABB — the empty middle
        // is far from either surface (the classic sparse-brick win).
        let (mut av, ai0) = sphere_at(0.35, [-1.6, -1.6, -1.6]);
        let (bv2, bi2) = sphere_at(0.35, [1.6, 1.6, 1.6]);
        let off = av.len() as u32;
        av.extend(bv2);
        let mut asm_i = ai0;
        asm_i.extend(bi2.iter().map(|&i| i + off));
        let assembly = bake_mesh_sdf(&av, &asm_i);

        let meshes = [sheet, wall, big_sphere, assembly];
        for margin in [1.0f32, 2.0, 3.0] {
            let r = analyze_bricks(&meshes, margin);
            eprintln!(
                "margin={margin:.0} vox | meshes={} bricks={} occupied={} \
                 frac={:.1}% | dense={:.1} KB sparse={:.1} KB ratio={:.2}",
                r.mesh_count,
                r.total_bricks,
                r.occupied_bricks,
                r.occupied_fraction * 100.0,
                r.dense_bytes as f64 / 1024.0,
                r.sparse_bytes as f64 / 1024.0,
                r.memory_ratio(),
            );
        }
        for (i, m) in meshes.iter().enumerate() {
            let band = brick_band_for(m, 2.0);
            let occ = classify_bricks(m, band);
            eprintln!(
                "  mesh[{i}] dims={:?} bricks/axis={:?} occupied {}/{}",
                m.dims,
                occ.bricks_per_axis,
                occ.occupied_count(),
                occ.brick_count(),
            );
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

    // --- F5 S1/S2: per-mesh albedo atlas ------------------------------------------------

    /// A `dim³` albedo field whose per-channel value is a deterministic function of the voxel
    /// index (distinct per channel), so an atlas round-trip can be checked exactly.
    fn ramp_albedo(dim: u32, seed: f32) -> AlbedoVolumes {
        let n = (dim * dim * dim) as usize;
        let ch = |k: f32| {
            (0..n)
                .map(|i| seed + k + i as f32 * 0.003)
                .collect::<Vec<f32>>()
        };
        AlbedoVolumes {
            dims: [dim; 3],
            channels: [ch(0.0), ch(0.11), ch(0.29)],
        }
    }

    /// Reference trilinear tap on one albedo channel (voxel-center clamp, GPU convention).
    fn albedo_sample(atlas: &AlbedoAtlas, c: usize, uvw: [f32; 3]) -> f32 {
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
        let at = |x: u32, y: u32, z: u32| atlas.channels[c][(x + dx * (y + dy * z)) as usize];
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(at(i0[0], i0[1], i0[2]), at(i1[0], i0[1], i0[2]), f[0]);
        let c10 = lerp(at(i0[0], i1[1], i0[2]), at(i1[0], i1[1], i0[2]), f[0]);
        let c01 = lerp(at(i0[0], i0[1], i1[2]), at(i1[0], i0[1], i1[2]), f[0]);
        let c11 = lerp(at(i0[0], i1[1], i1[2]), at(i1[0], i1[1], i1[2]), f[0]);
        lerp(lerp(c00, c10, f[1]), lerp(c01, c11, f[1]), f[2])
    }

    /// Trilinear tap on a raw `dim³` channel (the reference the resampled atlas must match).
    fn channel_sample(ch: &[f32], dim: u32, t: [f32; 3]) -> f32 {
        let d = dim as f32;
        let mut g = [0.0f32; 3];
        for k in 0..3 {
            g[k] = (t[k] * d - 0.5).clamp(0.0, d - 1.0);
        }
        let i0 = [g[0] as u32, g[1] as u32, g[2] as u32];
        let i1 = [
            (i0[0] + 1).min(dim - 1),
            (i0[1] + 1).min(dim - 1),
            (i0[2] + 1).min(dim - 1),
        ];
        let f = [
            g[0] - i0[0] as f32,
            g[1] - i0[1] as f32,
            g[2] - i0[2] as f32,
        ];
        let at = |x: u32, y: u32, z: u32| ch[(x + dim * (y + dim * z)) as usize];
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(at(i0[0], i0[1], i0[2]), at(i1[0], i0[1], i0[2]), f[0]);
        let c10 = lerp(at(i0[0], i1[1], i0[2]), at(i1[0], i1[1], i0[2]), f[0]);
        let c01 = lerp(at(i0[0], i0[1], i1[2]), at(i1[0], i0[1], i1[2]), f[0]);
        let c11 = lerp(at(i0[0], i1[1], i1[2]), at(i1[0], i1[1], i1[2]), f[0]);
        lerp(lerp(c00, c10, f[1]), lerp(c01, c11, f[1]), f[2])
    }

    /// The albedo atlas, sampled through the paired SDF atlas's `tile_uvw`, must reproduce each
    /// mesh's own albedo channels — the F5 contract: **one** UVW mapping addresses both fields.
    /// Native dims (no cap) so the round-trip is exact to trilinear tolerance.
    #[test]
    fn albedo_atlas_reproduces_mesh_sample_through_sdf_uvw() {
        let dims = [8u32, 32, 48];
        let meshes: Vec<SdfVolume> = dims
            .iter()
            .enumerate()
            .map(|(i, &d)| ramp_volume(d, [-1.0; 3], [1.0; 3], i as f32))
            .collect();
        let albedos: Vec<AlbedoVolumes> = dims.iter().map(|&d| ramp_albedo(d, d as f32)).collect();

        let sdf_atlas = SdfAtlas::pack(&meshes);
        let alb_atlas = AlbedoAtlas::pack_like(&sdf_atlas, &albedos, [0.5; 3]);
        assert_eq!(alb_atlas.dim, sdf_atlas.dim, "albedo atlas shares SDF dims");

        for (i, a) in albedos.iter().enumerate() {
            let (bias, scale) = sdf_atlas.tile_uvw(i);
            for &tx in &[0.0f32, 0.1, 0.5, 0.9, 1.0] {
                for &ty in &[0.0f32, 0.37, 1.0] {
                    for &tz in &[0.0f32, 0.63, 1.0] {
                        let t = [tx, ty, tz];
                        let uvw = [
                            bias[0] + t[0] * scale[0],
                            bias[1] + t[1] * scale[1],
                            bias[2] + t[2] * scale[2],
                        ];
                        for c in 0..3 {
                            let want = channel_sample(&a.channels[c], a.dims[0], t);
                            let got = albedo_sample(&alb_atlas, c, uvw);
                            assert!(
                                (got - want).abs() < 1e-4,
                                "mesh {i} ch{c} t={t:?}: atlas {got} vs mesh {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// When the SDF atlas is capped (large tiles downsampled), the albedo tile is resampled to
    /// the SAME capped dim, so the two tiles stay voxel-aligned and one `tile_uvw` maps both.
    #[test]
    fn albedo_atlas_matches_capped_sdf_tile_dims() {
        let big_sdf = ramp_volume(48, [-1.0; 3], [1.0; 3], 0.0);
        let big_alb = ramp_albedo(48, 7.0);
        let sdf_atlas = SdfAtlas::pack_capped(std::slice::from_ref(&big_sdf), 16);
        assert_eq!(sdf_atlas.tiles[0].dims, [16; 3], "SDF tile capped to 16");
        let alb_atlas =
            AlbedoAtlas::pack_like(&sdf_atlas, std::slice::from_ref(&big_alb), [0.5; 3]);
        // A center sample must read the albedo resampled to 16³ (not the native 48³), matching
        // the SDF's capped tile — i.e. no dim mismatch between the paired atlases.
        let (bias, scale) = sdf_atlas.tile_uvw(0);
        let t = [0.5f32, 0.5, 0.5];
        let uvw = [
            bias[0] + t[0] * scale[0],
            bias[1] + t[1] * scale[1],
            bias[2] + t[2] * scale[2],
        ];
        let capped = resample_channel(&big_alb.channels[0], big_alb.dims, [16; 3]);
        let want = channel_sample(&capped, 16, t);
        let got = albedo_sample(&alb_atlas, 0, uvw);
        assert!((got - want).abs() < 1e-4, "capped albedo {got} vs {want}");
    }

    /// A missing albedo (fewer albedos than tiles) fills its tile with the neutral fallback,
    /// so the atlas is always well-formed and the shader's fallback path is exercisable.
    #[test]
    fn albedo_atlas_fallback_fills_absent_tiles() {
        let meshes = [
            ramp_volume(8, [-1.0; 3], [1.0; 3], 0.0),
            ramp_volume(16, [0.0; 3], [1.0; 3], 1.0),
        ];
        let albedos = [ramp_albedo(8, 3.0)]; // only mesh 0 has albedo
        let sdf_atlas = SdfAtlas::pack(&meshes);
        let fb = [0.42f32, 0.13, 0.77];
        let alb_atlas = AlbedoAtlas::pack_like(&sdf_atlas, &albedos, fb);
        // Mesh 1's tile center reads the fallback on every channel.
        let (bias, scale) = sdf_atlas.tile_uvw(1);
        let t = [0.5f32, 0.5, 0.5];
        let uvw = [
            bias[0] + t[0] * scale[0],
            bias[1] + t[1] * scale[1],
            bias[2] + t[2] * scale[2],
        ];
        for (c, &f) in fb.iter().enumerate() {
            let got = albedo_sample(&alb_atlas, c, uvw);
            assert!((got - f).abs() < 1e-5, "fallback ch{c}: {got} vs {f}");
        }
    }

    #[test]
    fn albedo_atlas_is_deterministic() {
        let meshes = [
            ramp_volume(8, [-1.0; 3], [1.0; 3], 0.0),
            ramp_volume(16, [0.0; 3], [2.0; 3], 5.0),
        ];
        let albedos = [ramp_albedo(8, 1.0), ramp_albedo(16, 2.0)];
        let sdf_atlas = SdfAtlas::pack(&meshes);
        let a = AlbedoAtlas::pack_like(&sdf_atlas, &albedos, [0.5; 3]);
        let b = AlbedoAtlas::pack_like(&sdf_atlas, &albedos, [0.5; 3]);
        assert_eq!(a.dim, b.dim);
        assert_eq!(a.channels, b.channels);
    }
}
