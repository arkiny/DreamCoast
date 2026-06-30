//! Per-mesh distance-field → clipmap composition (per-mesh-distance-fields.md, Stage S1).
//!
//! Instead of re-baking the whole-scene fused triangle soup over every clipmap level, each
//! unique mesh bakes its own local-space DF once (cached and instanced via `crate::registry`
//! and `cook::load_or_bake_mesh_sdf`), and this module composites those into a clipmap
//! level's signed-distance volume: per voxel, the minimum over nearby objects of that
//! object's DF sampled in its local frame (scaled to world). The output is the same R32F
//! `dim³` volume the existing upload and `clipmap.slang cm_geo_*` march already consume, so
//! there is no shader / GPU-resource change (color stays the S2 concern).
//!
//! **Content-scene only.** The gallery keeps the fused bake (the byte-identical anchor):
//! composition is a `min` of trilinearly-resampled per-object fields, an approximation that
//! is not bit-identical to the fused exact-nearest-triangle bake.
//!
//! Why this is the win: a thin curtain bakes its DF over its *own* tight AABB (resolved at
//! its own scale), and far-from-everything voxels (e.g. the finer levels' mostly-empty
//! mid-air) cost ~nothing (AABB-culled) — the empty-voxel ring-search blow-up of the fused
//! finer-level bake disappears.

use dreamcoast_asset::sdf::SdfVolume;
use dreamcoast_core::glam::{Mat4, Vec3};

/// UE `MinMeshSDFRadius` analogue: the smallest drawable (world bounding-sphere radius) that
/// contributes to the GDF composite. Tiny props (bolts, small detail) barely occlude/bounce
/// the low-frequency GI/AO the distance field feeds, yet each is a full per-mesh DF bake +
/// composite — so on a non-instanced scene they dominate the cost for ~no quality. Culling
/// them concentrates the field on the large occluders (walls, columns, curtains). Knob:
/// `P11_GDF_MIN_RADIUS` (metres); `0` disables the cull.
pub(crate) const DEFAULT_MIN_MESH_RADIUS: f32 = 0.2;

/// World bounding-sphere radius of a mesh whose padded local AABB is `(mn, mx)` placed by
/// `world` (translation + uniform scale): half the local diagonal times the world scale.
pub(crate) fn mesh_world_radius(world: Mat4, mn: [f32; 3], mx: [f32; 3]) -> f32 {
    let scale = world.x_axis.truncate().length();
    let half =
        0.5 * ((mx[0] - mn[0]).powi(2) + (mx[1] - mn[1]).powi(2) + (mx[2] - mn[2]).powi(2)).sqrt();
    half * scale
}

/// One instance to composite: the inverse world transform (world→local), the uniform world
/// scale (local distance → world distance), its world-space AABB (the padded per-mesh DF
/// bounds, transformed — for the broad-phase cull), and the index of its per-mesh DF.
pub(crate) struct ComposeObject {
    pub(crate) inv_world: Mat4,
    pub(crate) scale: f32,
    pub(crate) wmin: [f32; 3],
    pub(crate) wmax: [f32; 3],
    pub(crate) mesh: usize,
}

impl ComposeObject {
    /// Build an object from its world transform + its per-mesh DF (whose `aabb` is the
    /// padded local bounds). Transforms (assumed translation + uniform scale, like the
    /// scene fuse) the DF's 8 local-AABB corners to get the world broad-phase AABB and
    /// reads the uniform scale from the matrix' first column.
    pub(crate) fn new(world: Mat4, mesh: usize, df: &SdfVolume) -> Self {
        let scale = world.x_axis.truncate().length().max(1e-8);
        let (mn, mx) = (df.aabb_min, df.aabb_max);
        let mut wmin = [f32::MAX; 3];
        let mut wmax = [f32::MIN; 3];
        for cx in [mn[0], mx[0]] {
            for cy in [mn[1], mx[1]] {
                for cz in [mn[2], mx[2]] {
                    let w = world.transform_point3(Vec3::new(cx, cy, cz));
                    for a in 0..3 {
                        wmin[a] = wmin[a].min(w[a]);
                        wmax[a] = wmax[a].max(w[a]);
                    }
                }
            }
        }
        Self {
            inv_world: world.inverse(),
            scale,
            wmin,
            wmax,
            mesh,
        }
    }
}

/// Compose a clipmap level's SDF: per voxel, the min over objects whose world AABB contains
/// the voxel of that object's per-mesh DF (local-sampled, scaled). Voxels covered by no
/// object get `empty` (a large positive distance, so the marcher steps through open space).
/// Parallel over Z slabs (the bake's threading pattern).
pub(crate) fn compose_sdf_level(
    objects: &[ComposeObject],
    mesh_sdfs: &[SdfVolume],
    level_min: [f32; 3],
    level_max: [f32; 3],
    dim: u32,
    empty: f32,
) -> SdfVolume {
    let slab = (dim * dim) as usize;
    let mut voxels = vec![0.0f32; slab * dim as usize];
    let inv_dim = 1.0 / dim as f32;
    let ext = [
        level_max[0] - level_min[0],
        level_max[1] - level_min[1],
        level_max[2] - level_min[2],
    ];

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(dim as usize)
        .max(1);
    let per = dim.div_ceil(threads as u32);

    std::thread::scope(|scope| {
        let mut rest = voxels.as_mut_slice();
        let mut z0 = 0u32;
        while z0 < dim {
            let z1 = (z0 + per).min(dim);
            let take = ((z1 - z0) as usize) * slab;
            let (head, tail) = rest.split_at_mut(take.min(rest.len()));
            rest = tail;
            scope.spawn(move || {
                for z in z0..z1 {
                    for y in 0..dim {
                        for x in 0..dim {
                            let p = [
                                level_min[0] + ext[0] * (x as f32 + 0.5) * inv_dim,
                                level_min[1] + ext[1] * (y as f32 + 0.5) * inv_dim,
                                level_min[2] + ext[2] * (z as f32 + 0.5) * inv_dim,
                            ];
                            let mut best = empty;
                            for o in objects {
                                if p[0] < o.wmin[0]
                                    || p[0] > o.wmax[0]
                                    || p[1] < o.wmin[1]
                                    || p[1] > o.wmax[1]
                                    || p[2] < o.wmin[2]
                                    || p[2] > o.wmax[2]
                                {
                                    continue;
                                }
                                let lp = o.inv_world.transform_point3(Vec3::new(p[0], p[1], p[2]));
                                let d = mesh_sdfs[o.mesh].sample([lp.x, lp.y, lp.z]) * o.scale;
                                if d < best {
                                    best = d;
                                }
                            }
                            let local = ((z - z0) * dim * dim + y * dim + x) as usize;
                            head[local] = best;
                        }
                    }
                }
            });
            z0 = z1;
        }
    });

    SdfVolume {
        dim,
        aabb_min: level_min,
        aabb_max: level_max,
        voxels,
    }
}
