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

use crate::MeshVertex;

/// A baked signed-distance volume: `dims[0]·dims[1]·dims[2]` R32F voxels over
/// `[aabb_min, aabb_max]`. Per-axis dims (F2 S2a): the voxel count follows each axis'
/// extent, so a thin wall no longer spends a full cube edge across its 0.2 m axis.
pub struct SdfVolume {
    pub dims: [u32; 3],
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    /// `dims[0]*dims[1]*dims[2]` signed distances, `idx = x + dims[0]*(y + dims[1]*z)`.
    pub voxels: Vec<f32>,
}

impl SdfVolume {
    /// Total voxel count (`dims[0]*dims[1]*dims[2]`).
    pub fn voxel_count(&self) -> usize {
        self.dims[0] as usize * self.dims[1] as usize * self.dims[2] as usize
    }

    /// The voxels as little-endian f32 bytes — the layout the GPU volume upload
    /// (`Device::create_volume_init`) and the `.dcasset` SDF chunk both store.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.voxels.len() * 4);
        for v in &self.voxels {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Trilinear sample at a point in the volume's own world space (`[aabb_min, aabb_max]`),
    /// matching the GPU `SampleLevel` convention: voxel centers at `(i+0.5)/dims[a]`, clamp-
    /// to-edge outside the center grid. Used by the per-mesh→clipmap compositor
    /// (per-mesh-distance-fields.md, S1) to read a mesh's local DF at a world voxel mapped
    /// into the mesh's local frame. The caller decides containment; this never returns NaN.
    pub fn sample(&self, p: [f32; 3]) -> f32 {
        let mut g = [0.0f32; 3]; // continuous voxel-center coords
        for a in 0..3 {
            let dim = self.dims[a] as f32;
            let ext = self.aabb_max[a] - self.aabb_min[a];
            let t = if ext > 0.0 {
                (p[a] - self.aabb_min[a]) / ext
            } else {
                0.5
            };
            // map normalized [0,1] to voxel-center index space [-0.5 .. dim-0.5], then clamp
            // so edge samples replicate (GPU clamp addressing).
            g[a] = (t * dim - 0.5).clamp(0.0, dim - 1.0);
        }
        let i0 = [g[0] as u32, g[1] as u32, g[2] as u32];
        let i1 = [
            (i0[0] + 1).min(self.dims[0] - 1),
            (i0[1] + 1).min(self.dims[1] - 1),
            (i0[2] + 1).min(self.dims[2] - 1),
        ];
        let f = [
            g[0] - i0[0] as f32,
            g[1] - i0[1] as f32,
            g[2] - i0[2] as f32,
        ];
        let at = |x: u32, y: u32, z: u32| -> f32 {
            self.voxels[(x + self.dims[0] * (y + self.dims[1] * z)) as usize]
        };
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(at(i0[0], i0[1], i0[2]), at(i1[0], i0[1], i0[2]), f[0]);
        let c10 = lerp(at(i0[0], i1[1], i0[2]), at(i1[0], i1[1], i0[2]), f[0]);
        let c01 = lerp(at(i0[0], i0[1], i1[2]), at(i1[0], i0[1], i1[2]), f[0]);
        let c11 = lerp(at(i0[0], i1[1], i1[2]), at(i1[0], i1[1], i1[2]), f[0]);
        lerp(lerp(c00, c10, f[1]), lerp(c01, c11, f[1]), f[2])
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

/// The result of a nearest-triangle query: the closest triangle's index, the distance
/// to it, and `p - q` (closest point) so the SDF sign can be derived without recomputing
/// the closest point. Defaults match a zero-triangle mesh (distance ∞, outward sign).
struct Hit {
    dist: f32,
    tri: usize,
    dp: [f32; 3],
}

/// A uniform spatial grid over the bake AABB binning triangles by the cells their AABB
/// overlaps (Scalable-GI Stage A). It turns the per-voxel nearest-triangle search from
/// O(triangles) into O(near cells), which is what lets the bake scale to Sponza-sized
/// meshes. CSR layout: `cell_tris[cell_start[c]..cell_start[c+1]]` are cell `c`'s tris.
///
/// **Determinism:** the grid is a pure function of the inputs, and the query
/// (ring-expanding with a conservative stop and a lowest-triangle-index tiebreak)
/// returns the *same* winner as the brute-force scan — so the baked field is byte-for-
/// byte identical with or without the grid (`grid_matches_brute` proves it).
struct TriGrid {
    res: [usize; 3],
    /// `res[a] / extent[a]` (0 on a degenerate axis → everything maps to cell 0).
    inv_cell: [f32; 3],
    origin: [f32; 3],
    /// Smallest cell edge — the per-shell distance bound's conservative unit.
    min_cell: f32,
    cell_start: Vec<u32>,
    cell_tris: Vec<u32>,
}

impl TriGrid {
    /// Pick a per-axis resolution: ~`cbrt(tri_count)` cells along the longest axis,
    /// others proportional to their extent (so cells stay roughly cubic), each clamped
    /// to `[1, MAX_GRID_RES]`. Resolution only affects speed, never the result.
    fn choose_res(ext: [f32; 3], tri_count: usize) -> [usize; 3] {
        const MAX_GRID_RES: f32 = 128.0;
        let n = tri_count.max(1) as f32;
        let max_ext = ext[0].max(ext[1]).max(ext[2]).max(1e-6);
        let base = n.cbrt().clamp(1.0, MAX_GRID_RES);
        let axis = |e: f32| ((e / max_ext) * base).ceil().clamp(1.0, MAX_GRID_RES) as usize;
        [axis(ext[0]), axis(ext[1]), axis(ext[2])]
    }

    /// The `[lo, hi]` inclusive cell range an AABB `[mn, mx]` overlaps (clamped to grid).
    fn cell_range(&self, mn: [f32; 3], mx: [f32; 3]) -> ([usize; 3], [usize; 3]) {
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for a in 0..3 {
            let l = ((mn[a] - self.origin[a]) * self.inv_cell[a]).floor();
            let h = ((mx[a] - self.origin[a]) * self.inv_cell[a]).floor();
            lo[a] = (l.max(0.0) as usize).min(self.res[a] - 1);
            hi[a] = (h.max(0.0) as usize).min(self.res[a] - 1);
        }
        (lo, hi)
    }

    #[inline]
    fn cell_index(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.res[0] * (y + self.res[1] * z)
    }

    /// Build the grid from the mesh's triangles.
    fn build(
        positions: &[[f32; 3]],
        indices: &[u32],
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
    ) -> Self {
        let tri_count = indices.len() / 3;
        let ext = [
            aabb_max[0] - aabb_min[0],
            aabb_max[1] - aabb_min[1],
            aabb_max[2] - aabb_min[2],
        ];
        let res = Self::choose_res(ext, tri_count);
        let inv_cell = [
            if ext[0] > 0.0 {
                res[0] as f32 / ext[0]
            } else {
                0.0
            },
            if ext[1] > 0.0 {
                res[1] as f32 / ext[1]
            } else {
                0.0
            },
            if ext[2] > 0.0 {
                res[2] as f32 / ext[2]
            } else {
                0.0
            },
        ];
        let min_cell = (0..3)
            .map(|a| {
                if res[a] > 0 {
                    ext[a] / res[a] as f32
                } else {
                    0.0
                }
            })
            .filter(|&c| c > 0.0)
            .fold(f32::MAX, f32::min);
        let min_cell = if min_cell == f32::MAX { 1e-6 } else { min_cell };
        let ncells = res[0] * res[1] * res[2];

        let mut grid = TriGrid {
            res,
            inv_cell,
            origin: aabb_min,
            min_cell,
            cell_start: vec![0u32; ncells + 1],
            cell_tris: Vec::new(),
        };

        // Per-triangle AABB → overlapping cell range. Count, prefix-sum, then fill.
        let tri_aabb = |tri: usize| -> ([f32; 3], [f32; 3]) {
            let a = positions[indices[tri * 3] as usize];
            let b = positions[indices[tri * 3 + 1] as usize];
            let c = positions[indices[tri * 3 + 2] as usize];
            let mut mn = a;
            let mut mx = a;
            for v in [b, c] {
                for k in 0..3 {
                    mn[k] = mn[k].min(v[k]);
                    mx[k] = mx[k].max(v[k]);
                }
            }
            (mn, mx)
        };
        for tri in 0..tri_count {
            let (mn, mx) = tri_aabb(tri);
            let (lo, hi) = grid.cell_range(mn, mx);
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    for x in lo[0]..=hi[0] {
                        let ci = grid.cell_index(x, y, z);
                        grid.cell_start[ci + 1] += 1;
                    }
                }
            }
        }
        for i in 0..ncells {
            grid.cell_start[i + 1] += grid.cell_start[i];
        }
        grid.cell_tris = vec![0u32; grid.cell_start[ncells] as usize];
        let mut cursor: Vec<u32> = grid.cell_start[..ncells].to_vec();
        for tri in 0..tri_count {
            let (mn, mx) = tri_aabb(tri);
            let (lo, hi) = grid.cell_range(mn, mx);
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    for x in lo[0]..=hi[0] {
                        let c = grid.cell_index(x, y, z);
                        grid.cell_tris[cursor[c] as usize] = tri as u32;
                        cursor[c] += 1;
                    }
                }
            }
        }
        grid
    }

    /// Nearest triangle to `p`, expanding Chebyshev shells of cells until no unsearched
    /// shell can hold anything closer. `visited` is a per-thread scratch stamp (sized to
    /// the triangle count) so a triangle binned into several cells is tested once per
    /// query; `stamp` is bumped by the caller per voxel.
    ///
    /// Tiebreak: among triangles at the exact minimum distance, the lowest index wins —
    /// matching the brute scan's `if d < best` (strict), so the winner (hence the SDF
    /// sign and albedo) is byte-identical.
    fn nearest(
        &self,
        p: [f32; 3],
        positions: &[[f32; 3]],
        indices: &[u32],
        visited: &mut [u32],
        stamp: u32,
    ) -> Hit {
        let mut best = Hit {
            dist: 1e30,
            tri: 0,
            dp: [0.0; 3],
        };
        let pc: [i32; 3] = std::array::from_fn(|a| {
            let c = ((p[a] - self.origin[a]) * self.inv_cell[a]).floor() as i32;
            c.clamp(0, self.res[a] as i32 - 1)
        });
        let max_r = (0..3)
            .map(|a| pc[a].max(self.res[a] as i32 - 1 - pc[a]))
            .max()
            .unwrap_or(0);

        let mut r = 0i32;
        loop {
            // Process every cell at Chebyshev distance exactly `r` from `pc`.
            for dz in -r..=r {
                let z = pc[2] + dz;
                if z < 0 || z >= self.res[2] as i32 {
                    continue;
                }
                for dy in -r..=r {
                    let y = pc[1] + dy;
                    if y < 0 || y >= self.res[1] as i32 {
                        continue;
                    }
                    for dx in -r..=r {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                            continue; // already covered by an inner shell
                        }
                        let x = pc[0] + dx;
                        if x < 0 || x >= self.res[0] as i32 {
                            continue;
                        }
                        let c = self.cell_index(x as usize, y as usize, z as usize);
                        for &t in &self.cell_tris
                            [self.cell_start[c] as usize..self.cell_start[c + 1] as usize]
                        {
                            let tri = t as usize;
                            if visited[tri] == stamp {
                                continue;
                            }
                            visited[tri] = stamp;
                            let a = positions[indices[tri * 3] as usize];
                            let b = positions[indices[tri * 3 + 1] as usize];
                            let cc = positions[indices[tri * 3 + 2] as usize];
                            let q = closest_on_triangle(p, a, b, cc);
                            let dp = sub(p, q);
                            let d = dot(dp, dp).sqrt();
                            // Strict-less keeps the first (lowest-index) at the min; the
                            // equal-distance branch lets an even lower index win on ties.
                            if d < best.dist || (d == best.dist && tri < best.tri) {
                                best = Hit { dist: d, tri, dp };
                            }
                        }
                    }
                }
            }
            // Distance from `p` to any cell in shell r+1 is ≥ r·min_cell. Stop (strict)
            // once nothing unsearched can match `best` — equal-distance ties in the next
            // shell are kept by the non-strict bound.
            if best.dist < r as f32 * self.min_cell {
                break;
            }
            r += 1;
            if r > max_r {
                break;
            }
        }
        best
    }
}

/// The bake inputs shared by every Z-slab: the voxel grid params, the decoded mesh, and
/// the acceleration grid (built once, borrowed read-only by every slab thread).
/// Grouped so the per-slab worker takes a single borrow (and to keep the argument
/// count sane).
struct BakeCtx<'a> {
    dims: [u32; 3],
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    positions: &'a [[f32; 3]],
    normals: &'a [[f32; 3]],
    indices: &'a [u32],
    grid: &'a TriGrid,
}

/// Bake one Z-slab `[z0, z1)` of the volume into `out` (the matching slice).
/// Extracted so the bake can run the slabs across threads.
fn bake_slab(out: &mut [f32], z0: u32, z1: u32, ctx: &BakeCtx) {
    let BakeCtx {
        dims,
        aabb_min,
        aabb_max,
        positions,
        normals,
        indices,
        grid,
    } = *ctx;
    let tri_count = indices.len() / 3;
    let inv = [
        1.0 / dims[0] as f32,
        1.0 / dims[1] as f32,
        1.0 / dims[2] as f32,
    ];
    // Per-thread visit stamps so the grid query tests each triangle once per voxel.
    let mut visited = vec![0u32; tri_count];
    let mut stamp = 0u32;
    for z in z0..z1 {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let t = [
                    (x as f32 + 0.5) * inv[0],
                    (y as f32 + 0.5) * inv[1],
                    (z as f32 + 0.5) * inv[2],
                ];
                let p = [
                    aabb_min[0] + (aabb_max[0] - aabb_min[0]) * t[0],
                    aabb_min[1] + (aabb_max[1] - aabb_min[1]) * t[1],
                    aabb_min[2] + (aabb_max[2] - aabb_min[2]) * t[2],
                ];
                stamp += 1;
                let hit = grid.nearest(p, positions, indices, &mut visited, stamp);
                let mut sign_d = 1.0_f32;
                if tri_count > 0 {
                    // Sign by the closest triangle's averaged (outward) vertex normals:
                    // dot(p-q, n) < 0 ⇒ inside (negative). Winding-independent, matching
                    // the shader.
                    let i0 = indices[hit.tri * 3] as usize;
                    let i1 = indices[hit.tri * 3 + 1] as usize;
                    let i2 = indices[hit.tri * 3 + 2] as usize;
                    let n0 = normals[i0];
                    let n1 = normals[i1];
                    let n2 = normals[i2];
                    let n = [
                        n0[0] + n1[0] + n2[0],
                        n0[1] + n1[1] + n2[1],
                        n0[2] + n1[2] + n2[2],
                    ];
                    sign_d = if dot(hit.dp, n) < 0.0 { -1.0 } else { 1.0 };
                }
                let local = ((z - z0) * dims[0] * dims[1] + y * dims[0] + x) as usize;
                out[local] = hit.dist * sign_d;
            }
        }
    }
}

/// Bake a `dims[0]·dims[1]·dims[2]` signed-distance volume over `[aabb_min, aabb_max]`
/// from the fused 32-byte vertex buffer (`pos`@0, `normal`@12) and u32 index buffer —
/// the same bytes the GPU bake reads. Brute force O(voxels·triangles), parallelized
/// over Z slabs (one-time cook; the result is cached as a `.dcasset` SDF chunk).
pub fn bake_sdf_from_fused(
    vtx_bytes: &[u8],
    idx_bytes: &[u8],
    dims: [u32; 3],
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

    let slab = dims[0] * dims[1];
    let mut voxels = vec![0.0_f32; (slab * dims[2]) as usize];

    // Stage A: one acceleration grid over the bake AABB, built once and shared by every
    // slab thread (read-only) — the per-voxel search is then O(near cells), not O(tris).
    let grid = TriGrid::build(&positions, &indices, aabb_min, aabb_max);

    // Parallelize over Z slabs. Each slab writes a disjoint, contiguous region, so
    // the output splits cleanly with no locking. std::thread only (no rayon dep).
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(dims[2] as usize)
        .max(1);
    let per = dims[2].div_ceil(threads as u32);

    let ctx = BakeCtx {
        dims,
        aabb_min,
        aabb_max,
        positions: &positions,
        normals: &normals,
        indices: &indices,
        grid: &grid,
    };
    std::thread::scope(|scope| {
        let mut rest = voxels.as_mut_slice();
        let mut z0 = 0u32;
        let ctx = &ctx;
        while z0 < dims[2] {
            let z1 = (z0 + per).min(dims[2]);
            let take = ((z1 - z0) * slab) as usize;
            let (head, tail) = rest.split_at_mut(take.min(rest.len()));
            rest = tail;
            scope.spawn(move || bake_slab(head, z0, z1, ctx));
            z0 = z1;
        }
    });

    SdfVolume {
        dims,
        aabb_min,
        aabb_max,
        voxels,
    }
}

/// A baked per-voxel albedo field: three `dim³` R32F channels (R/G/B) over the same
/// grid as the scene SDF (Phase 11 Stage C8a). Each voxel holds the linear albedo of
/// the nearest triangle. Used by the GI / reflection re-lighting to recover surface
/// colour at a ray hit.
pub struct AlbedoVolumes {
    pub dims: [u32; 3],
    /// `[r, g, b]`, each `dims[0]*dims[1]*dims[2]` values in
    /// `idx = x + dims[0]*(y + dims[1]*z)` order.
    pub channels: [Vec<f32>; 3],
}

impl AlbedoVolumes {
    /// Channel `c` (0=R,1=G,2=B) as little-endian f32 bytes for `create_volume_init`.
    pub fn channel_le_bytes(&self, c: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.channels[c].len() * 4);
        for v in &self.channels[c] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// Bake one Z-slab of the albedo field: per voxel, the nearest triangle's albedo
/// (RGB interleaved into `out`, 3 floats per voxel). The nearest-triangle search is
/// identical to [`bake_slab`] so the winner matches the distance field's.
fn bake_albedo_slab(out: &mut [f32], z0: u32, z1: u32, ctx: &BakeCtx, tri_albedo: &[[f32; 3]]) {
    let BakeCtx {
        dims,
        aabb_min,
        aabb_max,
        positions,
        indices,
        grid,
        ..
    } = *ctx;
    let tri_count = indices.len() / 3;
    let inv = [
        1.0 / dims[0] as f32,
        1.0 / dims[1] as f32,
        1.0 / dims[2] as f32,
    ];
    let mut visited = vec![0u32; tri_count];
    let mut stamp = 0u32;
    for z in z0..z1 {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let t = [
                    (x as f32 + 0.5) * inv[0],
                    (y as f32 + 0.5) * inv[1],
                    (z as f32 + 0.5) * inv[2],
                ];
                let p = [
                    aabb_min[0] + (aabb_max[0] - aabb_min[0]) * t[0],
                    aabb_min[1] + (aabb_max[1] - aabb_min[1]) * t[1],
                    aabb_min[2] + (aabb_max[2] - aabb_min[2]) * t[2],
                ];
                stamp += 1;
                // Same nearest-triangle search as the SDF (sqrt-distance ordering picks
                // the same argmin as the brute squared-distance scan), so the winning
                // triangle — hence its albedo — is byte-identical.
                let hit = grid.nearest(p, positions, indices, &mut visited, stamp);
                let albedo = if tri_count > 0 {
                    tri_albedo[hit.tri]
                } else {
                    [0.7, 0.7, 0.7]
                };
                let local = (((z - z0) * dims[0] * dims[1] + y * dims[0] + x) * 3) as usize;
                out[local] = albedo[0];
                out[local + 1] = albedo[1];
                out[local + 2] = albedo[2];
            }
        }
    }
}

/// Bake the per-voxel albedo volumes from the fused geometry + a per-triangle linear
/// albedo buffer (12 bytes / triangle, the same bytes the GPU albedo bake reads).
/// Brute force O(voxels·triangles), parallelized over Z slabs — a deterministic CPU
/// cook persisted as a `.dcasset` albedo chunk (Phase 12 M2 extension).
pub fn bake_albedo_from_fused(
    vtx_bytes: &[u8],
    idx_bytes: &[u8],
    tri_albedo_bytes: &[u8],
    dims: [u32; 3],
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> AlbedoVolumes {
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
    let tri_albedo: Vec<[f32; 3]> = tri_albedo_bytes
        .chunks_exact(12)
        .map(|c| {
            [
                f32::from_le_bytes(c[0..4].try_into().unwrap()),
                f32::from_le_bytes(c[4..8].try_into().unwrap()),
                f32::from_le_bytes(c[8..12].try_into().unwrap()),
            ]
        })
        .collect();

    let slab = dims[0] * dims[1];
    // RGB interleaved (3 floats / voxel) so the parallel split stays one contiguous
    // output; deinterleaved into channels afterwards.
    let mut interleaved = vec![0.0_f32; (slab * dims[2] * 3) as usize];
    // Stage A: the same acceleration grid as the SDF bake (built from the identical
    // fused geometry), so the nearest-triangle winner — hence each voxel's albedo — is
    // byte-identical with the brute scan.
    let grid = TriGrid::build(&positions, &indices, aabb_min, aabb_max);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(dims[2] as usize)
        .max(1);
    let per = dims[2].div_ceil(threads as u32);
    let ctx = BakeCtx {
        dims,
        aabb_min,
        aabb_max,
        positions: &positions,
        normals: &normals,
        indices: &indices,
        grid: &grid,
    };
    std::thread::scope(|scope| {
        let mut rest = interleaved.as_mut_slice();
        let mut z0 = 0u32;
        let ctx = &ctx;
        let tri_albedo = &tri_albedo;
        while z0 < dims[2] {
            let z1 = (z0 + per).min(dims[2]);
            let take = ((z1 - z0) * slab * 3) as usize;
            let (head, tail) = rest.split_at_mut(take.min(rest.len()));
            rest = tail;
            scope.spawn(move || bake_albedo_slab(head, z0, z1, ctx, tri_albedo));
            z0 = z1;
        }
    });

    let n = (slab * dims[2]) as usize;
    let mut channels = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for voxel in interleaved.chunks_exact(3) {
        channels[0].push(voxel[0]);
        channels[1].push(voxel[1]);
        channels[2].push(voxel[2]);
    }
    AlbedoVolumes { dims, channels }
}

// --- Per-mesh distance fields (per-mesh-distance-fields.md, Stage S0) ---------------
//
// A mesh's SDF is baked over its **own padded local AABB** at a resolution scaled to the
// mesh size, so thin geometry (e.g. curtain cloth) is resolved at its own scale — unlike
// the whole-scene fused field, where a 36 m bound coarsens it away. It reuses the same
// deterministic grid-accelerated `bake_sdf_from_fused`, so the result is bit-identical and
// backend-independent, and is content-hash cached per *unique* mesh — shared across all
// instances / levels / scenes (see `cook::load_or_bake_mesh_sdf`). The runtime composites
// these into the camera clipmap; this module is just the bake + grid sizing.

/// Target voxel edge (metres) for a per-mesh SDF — a per-mesh "resolution scale".
/// Each axis' voxel count scales with that axis' extent, clamped to
/// [`MESH_SDF_MIN_DIM`, `MESH_SDF_MAX_DIM`]: 0.05 m gives fine local detail; tiny axes
/// clamp up, long axes clamp down (a big wall's long axis at the cap is still fine — it
/// has no thin features there, and the coarse dense field covers the low frequency).
pub const MESH_SDF_TARGET_VOXEL: f32 = 0.05;
/// Minimum per-mesh grid edge per axis (small meshes still get a usable field).
pub const MESH_SDF_MIN_DIM: u32 = 8;
/// Maximum per-mesh grid edge per axis. Capped low (there are *many* unique meshes in a
/// non-instanced scene): a thin sheet's thin axis is resolved by its own small extent,
/// so a higher cap only burns bake time and atlas bytes for no visible gain.
pub const MESH_SDF_MAX_DIM: u32 = 48;

/// Per-axis voxel-grid dims for a mesh of the given **padded** local extent: each axis'
/// extent / the target voxel size, clamped (F2 S2a). The old cubic grid spent a full
/// cube edge across every axis, over-resolving a thin wall's 0.2 m axis ~50× (voxels
/// are `extent/dim` per axis); per-axis dims keep voxels near-uniform at
/// [`MESH_SDF_TARGET_VOXEL`] instead, so the same atlas budget carries a higher
/// long-axis cap (probe table: phase-f2-mesh-sdf-scalability-plan.md §2).
/// Fraction of a mesh's unique edges with exactly ONE incident triangle (boundary
/// edges). 0 for a watertight mesh; an open sheet (cloth, banner) approaches its
/// perimeter/area ratio. A mesh with a significant open fraction cannot carry a
/// meaningful inside/outside sign — its closest-triangle sign paints the entire
/// half-space behind the sheet "inside" (F6H).
pub fn mesh_open_fraction(index_bytes: &[u8]) -> f32 {
    use std::collections::HashMap;
    let mut edges: HashMap<(u32, u32), u32> = HashMap::with_capacity(index_bytes.len() / 4);
    let idx = |i: usize| u32::from_le_bytes(index_bytes[i * 4..i * 4 + 4].try_into().unwrap());
    let tri_count = index_bytes.len() / 12;
    for t in 0..tri_count {
        let tri = [idx(t * 3), idx(t * 3 + 1), idx(t * 3 + 2)];
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = (a.min(b), a.max(b));
            *edges.entry(key).or_insert(0) += 1;
        }
    }
    if edges.is_empty() {
        return 0.0;
    }
    let boundary = edges.values().filter(|&&c| c == 1).count();
    boundary as f32 / edges.len() as f32
}

/// Robust in/out re-sign for OPEN meshes (F6J): the closest-triangle sign paints
/// half-spaces "inside" behind open sheets, and no |d|-side strategy can express a
/// perforated shell (the F6I frontier). Instead, vote the sign per voxel with THREE
/// axis-aligned crossing parities: along each axis column, count triangle crossings
/// below the voxel centre — odd = that axis votes "inside" — and take the majority.
/// A sheet's back half-space gets one odd vote (the axis crossing the sheet) and two
/// even votes -> outside (the phantom dies); a solid wall's interior is crossed once
/// from every side -> inside (the zero-crossing survives). Magnitudes keep |d|.
/// Degenerate projections (triangles edge-on to an axis) are skipped — the other
/// axes cover them, and the majority absorbs stragglers.
pub fn parity_resign(vol: &mut SdfVolume, vtx_bytes: &[u8], idx_bytes: &[u8]) {
    let [dx, dy, dz] = vol.dims;
    let dims = [dx as usize, dy as usize, dz as usize];
    let n = dims[0] * dims[1] * dims[2];
    let tri_count = idx_bytes.len() / 12;
    if n == 0 || tri_count == 0 {
        return;
    }
    let idx_at =
        |i: usize| u32::from_le_bytes(idx_bytes[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
    let pos = |v: usize| read_vec3(vtx_bytes, v * VTX_STRIDE);
    let ext = [
        vol.aabb_max[0] - vol.aabb_min[0],
        vol.aabb_max[1] - vol.aabb_min[1],
        vol.aabb_max[2] - vol.aabb_min[2],
    ];
    let vox = [
        ext[0] / dims[0] as f32,
        ext[1] / dims[1] as f32,
        ext[2] / dims[2] as f32,
    ];
    // votes[i] bit a = axis a's parity for voxel i (built from sorted column crossings).
    let mut votes = vec![0u8; n];
    for axis in 0..3usize {
        let (a1, a2) = ((axis + 1) % 3, (axis + 2) % 3);
        if vox[a1] <= 0.0 || vox[a2] <= 0.0 || vox[axis] <= 0.0 {
            continue;
        }
        // Crossing depths along `axis`, per (a1, a2) column.
        let cols = dims[a1] * dims[a2];
        let mut crossings: Vec<Vec<f32>> = vec![Vec::new(); cols];
        for t in 0..tri_count {
            let p0 = pos(idx_at(t * 3));
            let p1 = pos(idx_at(t * 3 + 1));
            let p2 = pos(idx_at(t * 3 + 2));
            // Projected 2D triangle in the (a1, a2) plane.
            let q0 = [p0[a1], p0[a2]];
            let q1 = [p1[a1], p1[a2]];
            let q2 = [p2[a1], p2[a2]];
            let det = (q1[0] - q0[0]) * (q2[1] - q0[1]) - (q2[0] - q0[0]) * (q1[1] - q0[1]);
            if det.abs() < 1e-12 {
                continue; // edge-on to this axis: the other axes vote
            }
            let inv = 1.0 / det;
            // Column index ranges the projected AABB touches.
            // Deterministic sub-voxel jitter: authored geometry sits at round
            // coordinates, so exact voxel centres land on shared quad edges and
            // double-count (even parity). Fixed irrational offsets break every such
            // tie identically on every machine (the bake stays a deterministic cook).
            let jit = [0.0f32, 0.257_896, 0.413_729];
            let centre =
                |a: usize, i: usize, j: f32| vol.aabb_min[a] + (i as f32 + 0.5 + j) * vox[a];
            let col_range = |a: usize, lo: f32, hi: f32| -> (usize, usize) {
                let s = ((lo - vol.aabb_min[a]) / vox[a] - 0.5).ceil().max(0.0) as usize;
                let e = ((hi - vol.aabb_min[a]) / vox[a] - 0.5).floor();
                if e < 0.0 {
                    return (1, 0);
                }
                (s, (e as usize).min(dims[a] - 1))
            };
            let (i1s, i1e) =
                col_range(a1, q0[0].min(q1[0]).min(q2[0]), q0[0].max(q1[0]).max(q2[0]));
            let (i2s, i2e) =
                col_range(a2, q0[1].min(q1[1]).min(q2[1]), q0[1].max(q1[1]).max(q2[1]));
            for i2 in i2s..=i2e.min(dims[a2].saturating_sub(1)) {
                for i1 in i1s..=i1e.min(dims[a1].saturating_sub(1)) {
                    let c = [centre(a1, i1, jit[1]), centre(a2, i2, jit[2])];
                    // Barycentric point-in-triangle in the projected plane.
                    let w1 =
                        ((c[0] - q0[0]) * (q2[1] - q0[1]) - (c[1] - q0[1]) * (q2[0] - q0[0])) * inv;
                    let w2 =
                        ((c[1] - q0[1]) * (q1[0] - q0[0]) - (c[0] - q0[0]) * (q1[1] - q0[1])) * inv;
                    if w1 < 0.0 || w2 < 0.0 || w1 + w2 > 1.0 {
                        continue;
                    }
                    let h = p0[axis] + w1 * (p1[axis] - p0[axis]) + w2 * (p2[axis] - p0[axis]);
                    crossings[i1 + dims[a1] * i2].push(h);
                }
            }
        }
        // Walk each column: parity at a voxel centre = crossings strictly below it.
        for i2 in 0..dims[a2] {
            for i1 in 0..dims[a1] {
                let col = &mut crossings[i1 + dims[a1] * i2];
                if col.is_empty() {
                    continue;
                }
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mut k = 0usize;
                for ia in 0..dims[axis] {
                    let zc = vol.aabb_min[axis] + (ia as f32 + 0.5) * vox[axis];
                    while k < col.len() && col[k] < zc {
                        k += 1;
                    }
                    if k & 1 == 1 {
                        let mut c3 = [0usize; 3];
                        c3[axis] = ia;
                        c3[a1] = i1;
                        c3[a2] = i2;
                        votes[c3[0] + dims[0] * (c3[1] + dims[1] * c3[2])] |= 1 << axis;
                    }
                }
            }
        }
    }
    for (v, &vote) in vol.voxels.iter_mut().zip(votes.iter()) {
        let a = v.abs();
        *v = if vote.count_ones() >= 2 { -a } else { a };
    }
}

/// Re-sign a baked field whose closest-triangle sign is contaminated (non-watertight
/// meshes paint half-spaces "inside" — F6H/F6I): flood-fill AIR from the tile boundary
/// (the bake padding guarantees the boundary shell is air) through cells with
/// `|d| >= band`, treating the surface band (`|d| < band`) as the only barrier — the
/// phantom sign boundaries sit in open air where `|d|` is large, so the flood crosses
/// and re-signs them. Reached cells become `+|d|`; unreached cells (true wall
/// interiors, sealed cavities) become `-|d|`; band cells stay positive when they touch
/// reached air (the outside skin) and go negative otherwise (the inside skin), which
/// preserves a detectable zero-crossing for thick shells at any sampling resolution.
/// A sheet with no enclosed volume floods around its open rim and comes out all-
/// positive — the caller detects that (negative fraction ~0) and applies the
/// resolution-honest erosion instead. `band` ~ 0.75x the smallest voxel edge.
pub fn flood_resign(vol: &mut SdfVolume, band: f32) {
    let [dx, dy, dz] = vol.dims;
    let (dx, dy, dz) = (dx as usize, dy as usize, dz as usize);
    let n = dx * dy * dz;
    if n == 0 {
        return;
    }
    let idx = |x: usize, y: usize, z: usize| x + dx * (y + dy * z);
    let mut reached = vec![false; n];
    let mut queue = std::collections::VecDeque::new();
    // Seed every boundary cell that is clear of the surface band.
    for z in 0..dz {
        for y in 0..dy {
            for x in 0..dx {
                if x == 0 || y == 0 || z == 0 || x == dx - 1 || y == dy - 1 || z == dz - 1 {
                    let i = idx(x, y, z);
                    if vol.voxels[i].abs() >= band && !reached[i] {
                        reached[i] = true;
                        queue.push_back((x, y, z));
                    }
                }
            }
        }
    }
    while let Some((x, y, z)) = queue.pop_front() {
        let push =
            |x: usize,
             y: usize,
             z: usize,
             reached: &mut Vec<bool>,
             queue: &mut std::collections::VecDeque<(usize, usize, usize)>| {
                let i = idx(x, y, z);
                if !reached[i] && vol.voxels[i].abs() >= band {
                    reached[i] = true;
                    queue.push_back((x, y, z));
                }
            };
        if x > 0 {
            push(x - 1, y, z, &mut reached, &mut queue);
        }
        if x + 1 < dx {
            push(x + 1, y, z, &mut reached, &mut queue);
        }
        if y > 0 {
            push(x, y - 1, z, &mut reached, &mut queue);
        }
        if y + 1 < dy {
            push(x, y + 1, z, &mut reached, &mut queue);
        }
        if z > 0 {
            push(x, y, z - 1, &mut reached, &mut queue);
        }
        if z + 1 < dz {
            push(x, y, z + 1, &mut reached, &mut queue);
        }
    }
    // Band cells: positive iff they touch reached air (the outside skin).
    let mut band_pos = vec![false; n];
    for z in 0..dz {
        for y in 0..dy {
            for x in 0..dx {
                let i = idx(x, y, z);
                if vol.voxels[i].abs() < band {
                    let mut touch = false;
                    let mut probe = |x: usize, y: usize, z: usize| {
                        if reached[idx(x, y, z)] {
                            touch = true;
                        }
                    };
                    if x > 0 {
                        probe(x - 1, y, z);
                    }
                    if x + 1 < dx {
                        probe(x + 1, y, z);
                    }
                    if y > 0 {
                        probe(x, y - 1, z);
                    }
                    if y + 1 < dy {
                        probe(x, y + 1, z);
                    }
                    if z > 0 {
                        probe(x, y, z - 1);
                    }
                    if z + 1 < dz {
                        probe(x, y, z + 1);
                    }
                    band_pos[i] = touch;
                }
            }
        }
    }
    for i in 0..n {
        let a = vol.voxels[i].abs();
        vol.voxels[i] = if a >= band {
            if reached[i] { a } else { -a }
        } else if band_pos[i] {
            a
        } else {
            -a
        };
    }
}

pub fn mesh_sdf_dims(aabb_min: [f32; 3], aabb_max: [f32; 3]) -> [u32; 3] {
    let dim = |a: usize| {
        let ext = (aabb_max[a] - aabb_min[a]).max(0.0);
        ((ext / MESH_SDF_TARGET_VOXEL).ceil() as u32).clamp(MESH_SDF_MIN_DIM, MESH_SDF_MAX_DIM)
    };
    [dim(0), dim(1), dim(2)]
}

/// A mesh's local AABB, padded 10 % per axis (≥0.05 m) so the zero-isosurface isn't
/// clipped at the volume edge — identical padding to the scene fuse (`fuse.rs`). An empty
/// mesh yields a zero AABB.
pub fn mesh_local_aabb_padded(vertices: &[MeshVertex]) -> ([f32; 3], [f32; 3]) {
    if vertices.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    let mut mn = [f32::MAX; 3];
    let mut mx = [f32::MIN; 3];
    for v in vertices {
        for a in 0..3 {
            mn[a] = mn[a].min(v.pos[a]);
            mx[a] = mx[a].max(v.pos[a]);
        }
    }
    for a in 0..3 {
        let pad = ((mx[a] - mn[a]) * 0.1).max(0.05);
        mn[a] -= pad;
        mx[a] += pad;
    }
    (mn, mx)
}

/// Encode mesh vertices into the fused 32-byte records (`pos`@0, `normal`@12, `uv`@24)
/// the SDF/albedo bakes read, so a single mesh bakes through the same path as the fused
/// scene (and the bytes are the content-hash cache key).
pub fn encode_vertices_fused(vertices: &[MeshVertex]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vertices.len() * VTX_STRIDE);
    for v in vertices {
        for f in v.pos.iter().chain(&v.normal).chain(&v.uv) {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    out
}

/// Encode u32 indices little-endian (the index-buffer layout the bakes read).
pub fn encode_indices(indices: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(indices.len() * 4);
    for &i in indices {
        out.extend_from_slice(&i.to_le_bytes());
    }
    out
}

/// Bake a mesh's local-space SDF: pick the size-scaled per-axis grid over the mesh's
/// padded local AABB, then run the shared deterministic bake. The returned volume's
/// `aabb_min/max` are the padded local bounds the runtime composites with.
pub fn bake_mesh_sdf(vertices: &[MeshVertex], indices: &[u32]) -> SdfVolume {
    let (mn, mx) = mesh_local_aabb_padded(vertices);
    let dims = mesh_sdf_dims(mn, mx);
    let vtx = encode_vertices_fused(vertices);
    let idx = encode_indices(indices);
    bake_sdf_from_fused(&vtx, &idx, dims, mn, mx)
}

/// Bake a mesh's local-space **albedo** volumes (gi-fidelity-phases.md, F5 S1) over the
/// *exact same* grid as [`bake_mesh_sdf`] — the size-scaled per-axis `dims` on the mesh's padded
/// local AABB — so the albedo tile aligns 1:1 with the SDF tile in the atlas and a single
/// tile UVW mapping addresses both. `tri_albedo_bytes` is the per-triangle linear albedo
/// (12 B / triangle, same order as `indices`), mirroring the dense `bake_albedo_from_fused`
/// but bounded to this one mesh's frame so a hit reads the mesh's own colour at its own
/// resolution instead of the coarse whole-scene albedo grid (which blurs across meshes).
///
/// `dims`/`aabb` match `bake_mesh_sdf` by construction (both call `mesh_sdf_dims` /
/// `mesh_local_aabb_padded`), so the SDF atlas's `tile_uvw` is the albedo atlas's too.
pub fn bake_mesh_albedo(
    vertices: &[MeshVertex],
    indices: &[u32],
    tri_albedo_bytes: &[u8],
) -> AlbedoVolumes {
    let (mn, mx) = mesh_local_aabb_padded(vertices);
    let dims = mesh_sdf_dims(mn, mx);
    let vtx = encode_vertices_fused(vertices);
    let idx = encode_indices(indices);
    bake_albedo_from_fused(&vtx, &idx, tri_albedo_bytes, dims, mn, mx)
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
        vol.voxels[(x + vol.dims[0] * (y + vol.dims[1] * z)) as usize] as _
    }

    fn vert(pos: [f32; 3], normal: [f32; 3]) -> MeshVertex {
        MeshVertex {
            pos,
            normal,
            uv: [0.0, 0.0],
        }
    }

    fn parity_vtx(ps: &[[f32; 3]]) -> Vec<u8> {
        let mut out = Vec::new();
        for p in ps {
            for c in p {
                out.extend_from_slice(&c.to_le_bytes());
            }
            out.extend_from_slice(&[0u8; 20]); // normal + uv padding (32-byte stride)
        }
        out
    }

    fn parity_idx(is: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for i in is {
            out.extend_from_slice(&i.to_le_bytes());
        }
        out
    }

    #[test]
    fn parity_resign_box_interior_negative_air_positive() {
        // A closed box [0.4,1.2]^3 in a 16^3 tile over [0,1.6]^3, with GARBAGE signs
        // (everything positive): the 3-axis parity majority restores inside/outside.
        let (b0, b1) = (0.4f32, 1.2f32);
        let v = [
            [b0, b0, b0],
            [b1, b0, b0],
            [b1, b1, b0],
            [b0, b1, b0],
            [b0, b0, b1],
            [b1, b0, b1],
            [b1, b1, b1],
            [b0, b1, b1],
        ];
        let quads: [[u32; 4]; 6] = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 3, 7, 4],
            [1, 2, 6, 5],
        ];
        let mut idx = Vec::new();
        for q in quads {
            idx.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
        }
        let dims = [16u32, 16, 16];
        let voxels = vec![0.1f32; 16 * 16 * 16]; // garbage: all "outside"
        let mut vol = SdfVolume {
            dims,
            aabb_min: [0.0; 3],
            aabb_max: [1.6; 3],
            voxels,
        };
        parity_resign(&mut vol, &parity_vtx(&v), &parity_idx(&idx));
        let at = |x: usize, y: usize, z: usize| vol.voxels[x + 16 * (y + 16 * z)];
        assert!(at(8, 8, 8) < 0.0, "box interior must vote inside");
        assert!(at(1, 8, 8) > 0.0, "air outside stays positive");
        assert!(at(14, 14, 14) > 0.0, "far corner air stays positive");
    }

    #[test]
    fn parity_resign_open_sheet_back_half_space_positive() {
        // A single open quad at x = 0.8 with the classic contamination (back half
        // negative): only the x axis sees a crossing -> 1/3 votes -> outside
        // everywhere; the phantom half-space dies.
        let v = [
            [0.8f32, 0.2, 0.2],
            [0.8, 1.4, 0.2],
            [0.8, 1.4, 1.4],
            [0.8, 0.2, 1.4],
        ];
        let idx = [0u32, 1, 2, 0, 2, 3];
        let dims = [16u32, 16, 16];
        let mut voxels = vec![0.0f32; 16 * 16 * 16];
        for z in 0..16usize {
            for y in 0..16usize {
                for x in 0..16usize {
                    let wx = (x as f32 + 0.5) * 0.1;
                    let sign = if wx > 0.8 { -1.0 } else { 1.0 };
                    voxels[x + 16 * (y + 16 * z)] = sign * (wx - 0.8).abs().max(0.01);
                }
            }
        }
        let mut vol = SdfVolume {
            dims,
            aabb_min: [0.0; 3],
            aabb_max: [1.6; 3],
            voxels,
        };
        parity_resign(&mut vol, &parity_vtx(&v), &parity_idx(&idx));
        assert!(
            vol.voxels.iter().all(|&d| d > 0.0),
            "an open sheet must have no interior"
        );
    }

    #[test]
    fn flood_resign_sheet_uncontaminates_both_sides() {
        // A zero-thickness sheet at x = 0.8 in a 16^3 tile over [0, 1.6]^3, with the
        // classic contamination: the whole back half-space baked negative. The flood
        // goes around the open rim -> everything re-signs positive (the caller then
        // erodes for band detectability).
        let dims = [16u32, 16, 16];
        let mut voxels = vec![0.0f32; 16 * 16 * 16];
        for z in 0..16usize {
            for y in 0..16usize {
                for x in 0..16usize {
                    let wx = (x as f32 + 0.5) * 0.1;
                    let d = (wx - 0.8).abs();
                    let sign = if wx > 0.8 { -1.0 } else { 1.0 }; // contaminated back side
                    voxels[x + 16 * (y + 16 * z)] = sign * d;
                }
            }
        }
        let mut vol = SdfVolume {
            dims,
            aabb_min: [0.0; 3],
            aabb_max: [1.6; 3],
            voxels,
        };
        flood_resign(&mut vol, 0.075);
        assert!(
            vol.voxels.iter().all(|&d| d >= 0.0),
            "sheet must flood all-positive"
        );
    }

    #[test]
    fn flood_resign_thick_wall_keeps_interior_negative() {
        // A thick finite plate (box [0.6,1.0]x[0.3,1.3]x[0.3,1.3] — clear of the tile
        // boundary, as the bake padding guarantees) with the back air contaminated
        // negative: after the flood the back air re-signs positive (the flood goes
        // around the plate) while the plate interior stays negative (the
        // zero-crossing survives at any resolution).
        let dims = [16u32, 16, 16];
        let bmin = [0.6f32, 0.3, 0.3];
        let bmax = [1.0f32, 1.3, 1.3];
        let mut voxels = vec![0.0f32; 16 * 16 * 16];
        for z in 0..16usize {
            for y in 0..16usize {
                for x in 0..16usize {
                    let w = [
                        (x as f32 + 0.5) * 0.1,
                        (y as f32 + 0.5) * 0.1,
                        (z as f32 + 0.5) * 0.1,
                    ];
                    // Exact box SDF (negative inside).
                    let mut out2 = 0.0f32;
                    let mut inner = f32::MIN;
                    for a in 0..3 {
                        let q = (bmin[a] - w[a]).max(w[a] - bmax[a]);
                        if q > 0.0 {
                            out2 += q * q;
                        }
                        inner = inner.max(q);
                    }
                    let d = if out2 > 0.0 { out2.sqrt() } else { inner };
                    // Contaminate: the half-space behind the plate reads negative.
                    let sign = if d < 0.0 || w[0] >= 1.0 { -1.0 } else { 1.0 };
                    voxels[x + 16 * (y + 16 * z)] = sign * d.abs();
                }
            }
        }
        let mut vol = SdfVolume {
            dims,
            aabb_min: [0.0; 3],
            aabb_max: [1.6; 3],
            voxels,
        };
        flood_resign(&mut vol, 0.075);
        let at = |x: usize| vol.voxels[x + 16 * (8 + 16 * 8)];
        assert!(
            at(14) > 0.0,
            "back air must re-sign positive, got {}",
            at(14)
        );
        assert!(at(1) > 0.0, "front air stays positive");
        assert!(at(7) < 0.0, "plate interior stays negative, got {}", at(7));
    }

    #[test]
    fn mesh_sdf_dims_scale_and_clamp_per_axis() {
        // Tiny mesh clamps every axis to the floor; a huge axis clamps to the ceiling
        // while its short axes clamp to the floor (the F2 S2a win: no cubic coupling).
        assert_eq!(
            mesh_sdf_dims([0.0; 3], [0.01, 0.01, 0.01]),
            [MESH_SDF_MIN_DIM; 3]
        );
        assert_eq!(
            mesh_sdf_dims([0.0; 3], [100.0, 1.0, 1.0]),
            [MESH_SDF_MAX_DIM, 20, 20]
        );
        // 2 m / 0.05 = 40; 0.5 m = 10; 0.1 m clamps up to the floor.
        assert_eq!(
            mesh_sdf_dims([0.0; 3], [2.0, 0.5, 0.1]),
            [40, 10, MESH_SDF_MIN_DIM]
        );
    }

    #[test]
    fn mesh_local_aabb_pads_thin_axis() {
        // A flat 2x2 quad in XY (zero Z extent) pads the thin axis by the 0.05 m floor.
        let verts = [
            vert([-1.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
            vert([1.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
            vert([1.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            vert([-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ];
        let (mn, mx) = mesh_local_aabb_padded(&verts);
        assert!(
            (mn[2] + 0.05).abs() < 1e-6 && (mx[2] - 0.05).abs() < 1e-6,
            "thin Z padded ±0.05"
        );
        // Wide axes padded 10 %: 2 m → ±0.2.
        assert!((mn[0] + 1.2).abs() < 1e-5 && (mx[0] - 1.2).abs() < 1e-5);
    }

    /// THE point of per-mesh DF: a thin sheet (a curtain proxy) resolves a clean signed
    /// zero-crossing in its OWN local field — the whole-scene fused field at 48³/36 m
    /// cannot. We bake a flat quad and check the SDF flips sign across the thin axis with
    /// a near-zero magnitude at the sheet (so a marching ray registers a hit on it).
    #[test]
    fn mesh_sdf_resolves_thin_sheet() {
        let verts = [
            vert([-1.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
            vert([1.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
            vert([1.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            vert([-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ];
        let indices = [0u32, 1, 2, 0, 2, 3];
        let vol = bake_mesh_sdf(&verts, &indices);
        // Centre column in XY; the sheet sits at world z = 0, between two adjacent Z
        // voxels. Find the sign flip and confirm the straddling values are small.
        let c = vol.dims[0] / 2;
        let mut flipped = false;
        for z in 0..vol.dims[2] - 1 {
            let below = at(&vol, c, c, z);
            let above = at(&vol, c, c, z + 1);
            if below < 0.0 && above > 0.0 {
                flipped = true;
                // Both straddling voxels are within ~one voxel of the sheet (z extent is
                // 0.1 m over `dim` voxels), i.e. the thin sheet is finely resolved.
                let vox = 0.1 / vol.dims[2] as f32;
                assert!(
                    below.abs() < 2.0 * vox && above.abs() < 2.0 * vox,
                    "straddling SDF should be sub-voxel: {below} {above} (vox {vox})"
                );
            }
        }
        assert!(
            flipped,
            "thin sheet must produce a signed zero-crossing in its local DF"
        );
    }

    #[test]
    fn sphere_field_matches_analytic() {
        let (vtx, idx) = sphere_fused();
        let dim = 32;
        let vol = bake_sdf_from_fused(&vtx, &idx, [dim; 3], [0.0; 3], [1.0; 3]);
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
        let a = bake_sdf_from_fused(&vtx, &idx, [16; 3], [0.0; 3], [1.0; 3]);
        let b = bake_sdf_from_fused(&vtx, &idx, [16; 3], [0.0; 3], [1.0; 3]);
        assert_eq!(a.voxels, b.voxels, "two CPU bakes must be byte-identical");
    }

    #[test]
    fn albedo_bake_picks_nearest_triangle_colour() {
        // One sphere whose every triangle carries the same albedo → every voxel
        // (only one mesh to be nearest to) gets that colour.
        let (vtx, idx) = sphere_fused();
        let tri_count = idx.len() / 4 / 3;
        let colour = [0.2_f32, 0.6, 0.9];
        let mut tri_albedo = Vec::with_capacity(tri_count * 12);
        for _ in 0..tri_count {
            for c in colour {
                tri_albedo.extend_from_slice(&c.to_le_bytes());
            }
        }
        let dim = 16;
        let alb = bake_albedo_from_fused(&vtx, &idx, &tri_albedo, [dim; 3], [0.0; 3], [1.0; 3]);
        assert_eq!(alb.dims, [dim; 3]);
        for (ch, (channel, &expected)) in alb.channels.iter().zip(&colour).enumerate() {
            assert_eq!(channel.len(), (dim * dim * dim) as usize);
            assert!(
                channel.iter().all(|&v| (v - expected).abs() < 1e-6),
                "channel {ch} should be uniformly {expected}"
            );
        }
    }

    #[test]
    fn albedo_bake_is_deterministic() {
        let (vtx, idx) = sphere_fused();
        let tri_count = idx.len() / 4 / 3;
        let tri_albedo: Vec<u8> = (0..tri_count * 12).map(|i| (i % 251) as u8).collect();
        let a = bake_albedo_from_fused(&vtx, &idx, &tri_albedo, [12; 3], [0.0; 3], [1.0; 3]);
        let b = bake_albedo_from_fused(&vtx, &idx, &tri_albedo, [12; 3], [0.0; 3], [1.0; 3]);
        assert_eq!(a.channels, b.channels);
    }

    /// Two unit spheres (radius 0.18) at different positions, fused like the engine —
    /// a multi-object mesh so the nearest-triangle search crosses grid cells and picks
    /// between objects, exercising the ring expansion (not just one convex blob).
    fn two_spheres_fused() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let centres = [[0.3_f32, 0.5, 0.5], [0.75, 0.45, 0.6]];
        let mut vtx = Vec::new();
        let mut idx = Vec::new();
        let mut tri_albedo = Vec::new();
        let mut base = 0u32;
        for (oi, c) in centres.into_iter().enumerate() {
            let mut s = uv_sphere(20, 14);
            for v in &mut s.vertices {
                v.pos = [
                    v.pos[0] * 0.18 + c[0],
                    v.pos[1] * 0.18 + c[1],
                    v.pos[2] * 0.18 + c[2],
                ];
            }
            for v in &s.vertices {
                for f in v.pos.iter().chain(&v.normal).chain(&v.uv) {
                    vtx.extend_from_slice(&f.to_le_bytes());
                }
            }
            for &i in &s.indices {
                idx.extend_from_slice(&(i + base).to_le_bytes());
            }
            let colour = if oi == 0 {
                [0.8_f32, 0.2, 0.2]
            } else {
                [0.2, 0.4, 0.9]
            };
            for _ in 0..(s.indices.len() / 3) {
                for ch in colour {
                    tri_albedo.extend_from_slice(&ch.to_le_bytes());
                }
            }
            base += s.vertices.len() as u32;
        }
        (vtx, idx, tri_albedo)
    }

    /// Decode the fused buffers the same way the bakes do (for the brute references).
    fn decode(vtx: &[u8], idx: &[u8]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
        let vc = vtx.len() / VTX_STRIDE;
        let mut pos = Vec::with_capacity(vc);
        let mut nrm = Vec::with_capacity(vc);
        for v in 0..vc {
            pos.push(read_vec3(vtx, v * VTX_STRIDE));
            nrm.push(read_vec3(vtx, v * VTX_STRIDE + 12));
        }
        let indices = idx
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (pos, nrm, indices)
    }

    /// Brute-force SDF (the pre-Stage-A O(voxel·triangle) scan) — the reference the
    /// grid-accelerated bake must reproduce bit-for-bit.
    fn brute_sdf(vtx: &[u8], idx: &[u8], dim: u32, amin: [f32; 3], amax: [f32; 3]) -> Vec<f32> {
        let (pos, nrm, indices) = decode(vtx, idx);
        let tri_count = indices.len() / 3;
        let inv = 1.0 / dim as f32;
        let mut out = vec![0.0f32; (dim * dim * dim) as usize];
        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    // Match the slab's exact float grouping (t first, then scale) so the
                    // reference position is bit-identical to the production bake.
                    let t = [
                        (x as f32 + 0.5) * inv,
                        (y as f32 + 0.5) * inv,
                        (z as f32 + 0.5) * inv,
                    ];
                    let p = [
                        amin[0] + (amax[0] - amin[0]) * t[0],
                        amin[1] + (amax[1] - amin[1]) * t[1],
                        amin[2] + (amax[2] - amin[2]) * t[2],
                    ];
                    let mut best = 1e30f32;
                    let mut sign_d = 1.0f32;
                    for tri in 0..tri_count {
                        let i0 = indices[tri * 3] as usize;
                        let i1 = indices[tri * 3 + 1] as usize;
                        let i2 = indices[tri * 3 + 2] as usize;
                        let q = closest_on_triangle(p, pos[i0], pos[i1], pos[i2]);
                        let dp = sub(p, q);
                        let d = dot(dp, dp).sqrt();
                        if d < best {
                            best = d;
                            let n = [
                                nrm[i0][0] + nrm[i1][0] + nrm[i2][0],
                                nrm[i0][1] + nrm[i1][1] + nrm[i2][1],
                                nrm[i0][2] + nrm[i1][2] + nrm[i2][2],
                            ];
                            sign_d = if dot(dp, n) < 0.0 { -1.0 } else { 1.0 };
                        }
                    }
                    out[(x + dim * (y + dim * z)) as usize] = best * sign_d;
                }
            }
        }
        out
    }

    /// Brute-force albedo (pre-Stage-A) — picks the nearest triangle's colour.
    fn brute_albedo(
        vtx: &[u8],
        idx: &[u8],
        tri_albedo: &[u8],
        dim: u32,
        amin: [f32; 3],
        amax: [f32; 3],
    ) -> [Vec<f32>; 3] {
        let (pos, _nrm, indices) = decode(vtx, idx);
        let tri_count = indices.len() / 3;
        let tri_col: Vec<[f32; 3]> = tri_albedo
            .chunks_exact(12)
            .map(|c| {
                [
                    f32::from_le_bytes(c[0..4].try_into().unwrap()),
                    f32::from_le_bytes(c[4..8].try_into().unwrap()),
                    f32::from_le_bytes(c[8..12].try_into().unwrap()),
                ]
            })
            .collect();
        let inv = 1.0 / dim as f32;
        let n = (dim * dim * dim) as usize;
        let mut ch = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
        for z in 0..dim {
            for y in 0..dim {
                for x in 0..dim {
                    let t = [
                        (x as f32 + 0.5) * inv,
                        (y as f32 + 0.5) * inv,
                        (z as f32 + 0.5) * inv,
                    ];
                    let p = [
                        amin[0] + (amax[0] - amin[0]) * t[0],
                        amin[1] + (amax[1] - amin[1]) * t[1],
                        amin[2] + (amax[2] - amin[2]) * t[2],
                    ];
                    let mut best = 1e30f32;
                    let mut best_tri = 0usize;
                    for tri in 0..tri_count {
                        let q = closest_on_triangle(
                            p,
                            pos[indices[tri * 3] as usize],
                            pos[indices[tri * 3 + 1] as usize],
                            pos[indices[tri * 3 + 2] as usize],
                        );
                        let dp = sub(p, q);
                        let d = dot(dp, dp);
                        if d < best {
                            best = d;
                            best_tri = tri;
                        }
                    }
                    let col = if tri_count > 0 {
                        tri_col[best_tri]
                    } else {
                        [0.7, 0.7, 0.7]
                    };
                    let i = (x + dim * (y + dim * z)) as usize;
                    ch[0][i] = col[0];
                    ch[1][i] = col[1];
                    ch[2][i] = col[2];
                }
            }
        }
        ch
    }

    #[test]
    fn grid_sdf_matches_brute() {
        // Multi-object mesh + a padded, non-cubic AABB (like a real scene) so the grid's
        // ring expansion and tiebreak are genuinely exercised.
        let (vtx, idx, _) = two_spheres_fused();
        let dim = 24;
        let amin = [0.0f32, 0.1, 0.2];
        let amax = [1.1f32, 0.9, 1.0];
        let grid = bake_sdf_from_fused(&vtx, &idx, [dim; 3], amin, amax);
        let brute = brute_sdf(&vtx, &idx, dim, amin, amax);
        assert_eq!(
            grid.voxels, brute,
            "grid-accelerated SDF must be byte-identical to the brute scan"
        );
    }

    #[test]
    fn grid_albedo_matches_brute() {
        let (vtx, idx, tri_albedo) = two_spheres_fused();
        let dim = 24;
        let amin = [0.0f32, 0.1, 0.2];
        let amax = [1.1f32, 0.9, 1.0];
        let grid = bake_albedo_from_fused(&vtx, &idx, &tri_albedo, [dim; 3], amin, amax);
        let brute = brute_albedo(&vtx, &idx, &tri_albedo, dim, amin, amax);
        assert_eq!(
            grid.channels, brute,
            "grid-accelerated albedo must be byte-identical to the brute scan"
        );
    }

    /// Stage A measurement (ignored by default — needs the local-only Sponza asset).
    /// Run with `cargo test -p dreamcoast-asset --release bench_sponza -- --ignored
    /// --nocapture`. Loads Sponza, fuses its whole hierarchy to world space, then times
    /// the grid (production, parallel) vs the brute (serial reference) SDF bake at 48³ —
    /// and `assert_eq!`s them, so it doubles as the bit-identity proof on a real
    /// 260k-triangle mesh.
    #[test]
    #[ignore = "benchmark: requires local assets/Sponza/Sponza.gltf"]
    fn bench_sponza_bake() {
        use dreamcoast_core::glam::{Mat4, Quat, Vec3};
        use std::time::Instant;

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/Sponza/Sponza.gltf"
        );
        if !std::path::Path::new(path).exists() {
            eprintln!("Sponza absent at {path} — skipping benchmark");
            return;
        }
        let scene = crate::load_gltf_scene(path).expect("load Sponza");

        // Fuse the whole node hierarchy to world space (the engine's native 1u=1m scale).
        let mut vtx: Vec<u8> = Vec::new();
        let mut idx: Vec<u8> = Vec::new();
        let mut base = 0u32;
        let mut amin = [f32::MAX; 3];
        let mut amax = [f32::MIN; 3];
        let mut stack: Vec<(usize, Mat4)> =
            scene.roots.iter().map(|&r| (r, Mat4::IDENTITY)).collect();
        while let Some((ni, parent)) = stack.pop() {
            let n = &scene.nodes[ni];
            let world = parent
                * Mat4::from_scale_rotation_translation(
                    Vec3::from(n.scale),
                    Quat::from_array(n.rotation),
                    Vec3::from(n.translation),
                );
            if let Some(mi) = n.mesh {
                for prim in &scene.meshes[mi] {
                    for v in &prim.vertices {
                        let p = world.transform_point3(Vec3::from(v.pos));
                        let nn = world
                            .transform_vector3(Vec3::from(v.normal))
                            .normalize_or_zero();
                        amin = [amin[0].min(p.x), amin[1].min(p.y), amin[2].min(p.z)];
                        amax = [amax[0].max(p.x), amax[1].max(p.y), amax[2].max(p.z)];
                        for f in [p.x, p.y, p.z, nn.x, nn.y, nn.z, v.uv[0], v.uv[1]] {
                            vtx.extend_from_slice(&f.to_le_bytes());
                        }
                    }
                    for &i in &prim.indices {
                        idx.extend_from_slice(&(i + base).to_le_bytes());
                    }
                    base += prim.vertices.len() as u32;
                }
            }
            for &c in &n.children {
                stack.push((c, world));
            }
        }
        for i in 0..3 {
            let pad = ((amax[i] - amin[i]) * 0.1).max(0.05);
            amin[i] -= pad;
            amax[i] += pad;
        }
        let tris = idx.len() / 4 / 3;
        eprintln!(
            "Sponza fused: {} verts, {tris} tris, AABB size [{:.1}, {:.1}, {:.1}] m",
            vtx.len() / VTX_STRIDE,
            amax[0] - amin[0],
            amax[1] - amin[1],
            amax[2] - amin[2],
        );

        let dim = 48;
        let t0 = Instant::now();
        let grid = bake_sdf_from_fused(&vtx, &idx, [dim; 3], amin, amax);
        let grid_ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("grid bake (parallel) {dim}^3: {grid_ms:.0} ms");

        let t1 = Instant::now();
        let brute = brute_sdf(&vtx, &idx, dim, amin, amax);
        let brute_ms = t1.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "brute bake (serial) {dim}^3: {brute_ms:.0} ms  ({:.0}x slower than parallel grid)",
            brute_ms / grid_ms
        );

        assert_eq!(
            grid.voxels, brute,
            "Sponza grid bake must be byte-identical to the brute scan"
        );
    }
}
