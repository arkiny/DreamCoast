//! CPU signed-distance-field bake (Phase 12 M2) — a faithful port of
//! `crates/shader/shaders/sdf_bake.slang` (`bakeMain`).
//!
//! The GPU bake brute-forces, per voxel, the minimum distance to every triangle
//! (closest-point-on-triangle, Ericson) and signs it by the closest triangle's
//! averaged vertex normals (negative inside). Running the identical algorithm on
//! the CPU makes the baked field a **deterministic, backend-independent cook**: the
//! same bytes are produced on any machine and uploaded verbatim to the GPU volume,
//! so the per-mesh SDF persists in a `.dcasset` (M2) instead of being re-baked every
//! launch, and Vulkan/D3D12 read byte-identical fields.
//!
//! Voxel order is `idx = x + dim*(y + dim*z)` (x fastest), matching the linear
//! buffer the volume upload copies into the 3D texture. The mesh is read straight
//! from the **fused 32-byte vertex records** (`pos` at byte 0, `normal` at byte 12)
//! the rasterizer / HW path tracer / GPU bake all share — one layout, no second
//! source.

/// A baked signed-distance volume: `dim³` R32F voxels over `[aabb_min, aabb_max]`.
pub struct SdfVolume {
    pub dim: u32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    /// `dim³` signed distances, `idx = x + dim*(y + dim*z)`.
    pub voxels: Vec<f32>,
}

impl SdfVolume {
    /// The voxels as little-endian f32 bytes — the layout the GPU volume upload
    /// (`Device::create_volume_init`) and the `.dcasset` SDF chunk both store.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.voxels.len() * 4);
        for v in &self.voxels {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// Vertex stride of the fused buffer (pos[3] + normal[3] + uv[2], all f32).
const VTX_STRIDE: usize = 32;

#[inline]
fn read_vec3(bytes: &[u8], off: usize) -> [f32; 3] {
    [
        f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()),
        f32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()),
        f32::from_le_bytes(bytes[off + 8..off + 12].try_into().unwrap()),
    ]
}

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Closest point on triangle `abc` to `p` (Ericson, *Real-Time Collision
/// Detection*) — the exact branch structure of the shader's `closest_on_triangle`.
fn closest_on_triangle(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = sub(c, b);
        return [b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]];
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    [
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ]
}

/// The bake inputs shared by every Z-slab: the voxel grid + the decoded mesh.
/// Grouped so the per-slab worker takes a single borrow (and to keep the argument
/// count sane).
struct BakeCtx<'a> {
    dim: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    positions: &'a [[f32; 3]],
    normals: &'a [[f32; 3]],
    indices: &'a [u32],
}

/// Bake one Z-slab `[z0, z1)` of the volume into `out` (the matching slice).
/// Extracted so the bake can run the slabs across threads.
fn bake_slab(out: &mut [f32], z0: u32, z1: u32, ctx: &BakeCtx) {
    let BakeCtx {
        dim,
        aabb_min,
        aabb_max,
        positions,
        normals,
        indices,
    } = *ctx;
    let tri_count = indices.len() / 3;
    let inv_dim = 1.0 / dim as f32;
    for z in z0..z1 {
        for y in 0..dim {
            for x in 0..dim {
                let t = [
                    (x as f32 + 0.5) * inv_dim,
                    (y as f32 + 0.5) * inv_dim,
                    (z as f32 + 0.5) * inv_dim,
                ];
                let p = [
                    aabb_min[0] + (aabb_max[0] - aabb_min[0]) * t[0],
                    aabb_min[1] + (aabb_max[1] - aabb_min[1]) * t[1],
                    aabb_min[2] + (aabb_max[2] - aabb_min[2]) * t[2],
                ];
                let mut best = 1e30_f32;
                let mut sign_d = 1.0_f32;
                for tri in 0..tri_count {
                    let i0 = indices[tri * 3] as usize;
                    let i1 = indices[tri * 3 + 1] as usize;
                    let i2 = indices[tri * 3 + 2] as usize;
                    let a = positions[i0];
                    let b = positions[i1];
                    let c = positions[i2];
                    let q = closest_on_triangle(p, a, b, c);
                    let dp = sub(p, q);
                    let d = dot(dp, dp).sqrt();
                    if d < best {
                        best = d;
                        // Sign by the closest triangle's averaged (outward) vertex
                        // normals: dot(p-q, n) < 0 ⇒ inside (negative). Winding-
                        // independent, matching the shader.
                        let n0 = normals[i0];
                        let n1 = normals[i1];
                        let n2 = normals[i2];
                        let n = [
                            n0[0] + n1[0] + n2[0],
                            n0[1] + n1[1] + n2[1],
                            n0[2] + n1[2] + n2[2],
                        ];
                        sign_d = if dot(dp, n) < 0.0 { -1.0 } else { 1.0 };
                    }
                }
                let local = ((z - z0) * dim * dim + y * dim + x) as usize;
                out[local] = best * sign_d;
            }
        }
    }
}

/// Bake a `dim³` signed-distance volume over `[aabb_min, aabb_max]` from the fused
/// 32-byte vertex buffer (`pos`@0, `normal`@12) and u32 index buffer — the same
/// bytes the GPU bake reads. Brute force O(voxels·triangles), parallelized over Z
/// slabs (one-time cook; the result is cached as a `.dcasset` SDF chunk).
pub fn bake_sdf_from_fused(
    vtx_bytes: &[u8],
    idx_bytes: &[u8],
    dim: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> SdfVolume {
    let vtx_count = vtx_bytes.len() / VTX_STRIDE;
    let mut positions = Vec::with_capacity(vtx_count);
    let mut normals = Vec::with_capacity(vtx_count);
    for v in 0..vtx_count {
        let base = v * VTX_STRIDE;
        positions.push(read_vec3(vtx_bytes, base));
        normals.push(read_vec3(vtx_bytes, base + 12));
    }
    let indices: Vec<u32> = idx_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let slab = dim * dim;
    let mut voxels = vec![0.0_f32; (slab * dim) as usize];

    // Parallelize over Z slabs. Each slab writes a disjoint, contiguous region, so
    // the output splits cleanly with no locking. std::thread only (no rayon dep).
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(dim as usize)
        .max(1);
    let per = dim.div_ceil(threads as u32);

    let ctx = BakeCtx {
        dim,
        aabb_min,
        aabb_max,
        positions: &positions,
        normals: &normals,
        indices: &indices,
    };
    std::thread::scope(|scope| {
        let mut rest = voxels.as_mut_slice();
        let mut z0 = 0u32;
        let ctx = &ctx;
        while z0 < dim {
            let z1 = (z0 + per).min(dim);
            let take = ((z1 - z0) * slab) as usize;
            let (head, tail) = rest.split_at_mut(take.min(rest.len()));
            rest = tail;
            scope.spawn(move || bake_slab(head, z0, z1, ctx));
            z0 = z1;
        }
    });

    SdfVolume {
        dim,
        aabb_min,
        aabb_max,
        voxels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uv_sphere;

    /// Build the B2 verification mesh: a unit uv-sphere scaled to radius 0.3,
    /// centred at (0.5,0.5,0.5) — the exact bake mesh of `gdf.rs` — as the fused
    /// 32-byte vertex buffer (pos@0, normal@12, uv@24) + u32 index buffer.
    fn sphere_fused() -> (Vec<u8>, Vec<u8>) {
        let mut sphere = uv_sphere(48, 32);
        for v in &mut sphere.vertices {
            v.pos = [
                v.pos[0] * 0.3 + 0.5,
                v.pos[1] * 0.3 + 0.5,
                v.pos[2] * 0.3 + 0.5,
            ];
        }
        let mut vtx = Vec::with_capacity(sphere.vertices.len() * VTX_STRIDE);
        for v in &sphere.vertices {
            for f in v.pos.iter().chain(&v.normal).chain(&v.uv) {
                vtx.extend_from_slice(&f.to_le_bytes());
            }
        }
        let mut idx = Vec::with_capacity(sphere.indices.len() * 4);
        for &i in &sphere.indices {
            idx.extend_from_slice(&i.to_le_bytes());
        }
        (vtx, idx)
    }

    #[inline]
    fn at(vol: &SdfVolume, x: u32, y: u32, z: u32) -> f32 {
        vol.voxels[(x + vol.dim * (y + vol.dim * z)) as usize] as _
    }

    #[test]
    fn sphere_field_matches_analytic() {
        let (vtx, idx) = sphere_fused();
        let dim = 32;
        let vol = bake_sdf_from_fused(&vtx, &idx, dim, [0.0; 3], [1.0; 3]);
        assert_eq!(vol.voxels.len(), (dim * dim * dim) as usize);

        // Centre voxel (≈ world 0.5): inside, distance ≈ -radius (0.3). Faceting makes
        // the triangulated radius a hair under 0.3, so allow a small tolerance.
        let cc = dim / 2;
        let centre = at(&vol, cc, cc, cc);
        assert!(
            (centre + 0.3).abs() < 0.03,
            "centre {centre} should be ≈ -0.3"
        );

        // A corner voxel (≈ world ~0.0) is well outside: large positive distance.
        assert!(at(&vol, 0, 0, 0) > 0.4, "corner should be positive/outside");

        // Sign flips across the surface along +X from the centre.
        assert!(centre < 0.0, "centre inside");
        assert!(at(&vol, dim - 1, cc, cc) > 0.0, "+X edge outside");
    }

    #[test]
    fn bake_is_deterministic() {
        let (vtx, idx) = sphere_fused();
        let a = bake_sdf_from_fused(&vtx, &idx, 16, [0.0; 3], [1.0; 3]);
        let b = bake_sdf_from_fused(&vtx, &idx, 16, [0.0; 3], [1.0; 3]);
        assert_eq!(a.voxels, b.voxels, "two CPU bakes must be byte-identical");
    }
}
