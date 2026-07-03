//! Virtual-geometry cluster data model + offline builder (Phase 14 M1).
//!
//! A mesh is split into **clusters** (meshlets, ~128 triangles) that reference a shared
//! vertex pool through a compact local window — the meshopt-style `(vertex window, u8 triangle
//! indices)` layout the GPU mesh-shader path consumes. Each cluster carries the culling bounds
//! (bounding sphere + normal cone) the runtime needs. This M1a step produces a single LOD
//! (the finest); the LOD DAG (group → simplify → parent error bounds) lands in M1c and bumps
//! the serialization version.
//!
//! We build our own clusterizer (referencing meshoptimizer's approach, no FFI): a greedy
//! sweep fills a cluster until it would exceed the vertex/triangle caps, then starts a new one.
//! Order-based greedy is spatially naive but produces valid, deterministic clusters; a
//! shared-edge locality pass is a later refinement (it only changes cluster *shape*, not the
//! format or the runtime contract).

use dreamcoast_core::glam::Vec3;

use crate::MeshVertex;

/// Max unique vertices per cluster. Bounded to 255 so triangle indices fit in a `u8` local
/// window (mesh-shader friendly); 64 is the common sweet spot for GPU occupancy.
pub const MAX_CLUSTER_VERTICES: usize = 64;
/// Max triangles per cluster (~128; 124 keeps the `u8` triple count a multiple of 4).
pub const MAX_CLUSTER_TRIANGLES: usize = 124;

/// One meshlet-style cluster: a small triangle patch referencing the shared vertex pool via a
/// local vertex window, plus the culling bounds the virtual-geometry runtime needs. Triangles
/// are stored as `u8` triples indexing into `[vertex_offset .. vertex_offset+vertex_count)` of
/// [`MeshClusters::cluster_vertices`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cluster {
    /// Start of this cluster's window into [`MeshClusters::cluster_vertices`].
    pub vertex_offset: u32,
    /// Number of unique vertices this cluster references (`<= MAX_CLUSTER_VERTICES`).
    pub vertex_count: u32,
    /// Start of this cluster's triangles in [`MeshClusters::cluster_triangles`] (in `u8`s;
    /// `triangle_count * 3` bytes follow).
    pub triangle_offset: u32,
    /// Number of triangles (`<= MAX_CLUSTER_TRIANGLES`).
    pub triangle_count: u32,
    /// Bounding-sphere center (local mesh space) — frustum cull + screen-space error project.
    pub bounds_center: [f32; 3],
    /// Bounding-sphere radius.
    pub bounds_radius: f32,
    /// Normal-cone axis (normalized average face normal) for backface cluster culling.
    pub cone_axis: [f32; 3],
    /// `cos(half-angle)` of the cone containing every face normal: the min face-normal·axis.
    /// `<= 0` means the spread is too wide to backface-cull; stored as `-1.0` ("never cull").
    pub cone_cutoff: f32,
    /// Material id (carried through from the source mesh; one material per mesh in M1).
    pub material: u32,
}

/// A mesh's clusterized geometry — the per-mesh **cluster page stream** serialized to
/// `.dcasset`. Self-contained: it owns the source mesh's vertex pool (`vertices`), so a page
/// carries everything the runtime needs to upload + draw without a separate mesh chunk.
/// `cluster_vertices` remaps into `vertices`; `cluster_triangles` holds every cluster's `u8`
/// local triangle indices back to back.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MeshClusters {
    /// The source mesh's vertex pool (position/normal/uv), in original order. Owned so the
    /// cluster page is self-describing; `cluster_vertices` and the drawn geometry index into it.
    pub vertices: Vec<MeshVertex>,
    /// Remap: `cluster_vertices[cluster.vertex_offset + local]` is a `vertices` index.
    pub cluster_vertices: Vec<u32>,
    /// Local `u8` triangle indices (3 per triangle), addressing a cluster's vertex window.
    pub cluster_triangles: Vec<u8>,
    /// The clusters, in build order.
    pub clusters: Vec<Cluster>,
}

/// Build a mesh's finest-LOD clusters (Phase 14 M1a). `indices` is the triangle list into
/// `vertices`; `material` is stamped on every cluster. Deterministic: the greedy sweep visits
/// triangles in input order, so the same mesh always produces the same clusters (cook-cache
/// friendly, cross-platform byte-identical).
pub fn build_clusters(vertices: &[MeshVertex], indices: &[u32], material: u32) -> MeshClusters {
    let mut out = MeshClusters {
        vertices: vertices.to_vec(),
        ..Default::default()
    };
    let tri_count = indices.len() / 3;

    // Current cluster accumulation state.
    let mut window: Vec<u32> = Vec::with_capacity(MAX_CLUSTER_VERTICES);
    // Map source vertex index -> local window slot, for the current cluster only.
    let mut local_of: std::collections::HashMap<u32, u8> = std::collections::HashMap::new();
    let mut tris: Vec<[u8; 3]> = Vec::with_capacity(MAX_CLUSTER_TRIANGLES);

    let flush = |out: &mut MeshClusters,
                 window: &mut Vec<u32>,
                 local_of: &mut std::collections::HashMap<u32, u8>,
                 tris: &mut Vec<[u8; 3]>| {
        if tris.is_empty() {
            return;
        }
        let (center, radius) = bounding_sphere(vertices, window);
        let (axis, cutoff) = normal_cone(vertices, window, tris);
        let cluster = Cluster {
            vertex_offset: out.cluster_vertices.len() as u32,
            vertex_count: window.len() as u32,
            triangle_offset: out.cluster_triangles.len() as u32,
            triangle_count: tris.len() as u32,
            bounds_center: center.to_array(),
            bounds_radius: radius,
            cone_axis: axis.to_array(),
            cone_cutoff: cutoff,
            material,
        };
        out.cluster_vertices.extend_from_slice(window);
        for t in tris.iter() {
            out.cluster_triangles.extend_from_slice(t);
        }
        out.clusters.push(cluster);
        window.clear();
        local_of.clear();
        tris.clear();
    };

    for t in 0..tri_count {
        let tri = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];
        // How many NEW unique vertices would this triangle add to the current window?
        let new_verts = tri.iter().filter(|v| !local_of.contains_key(v)).count();
        let would_overflow = window.len() + new_verts > MAX_CLUSTER_VERTICES
            || tris.len() + 1 > MAX_CLUSTER_TRIANGLES;
        if would_overflow {
            flush(&mut out, &mut window, &mut local_of, &mut tris);
        }
        // Add the triangle's vertices to the window and record the local u8 triple.
        let mut local = [0u8; 3];
        for (i, &v) in tri.iter().enumerate() {
            let slot = *local_of.entry(v).or_insert_with(|| {
                let s = window.len() as u8;
                window.push(v);
                s
            });
            local[i] = slot;
        }
        tris.push(local);
    }
    flush(&mut out, &mut window, &mut local_of, &mut tris);
    out
}

/// Loose bounding sphere for a cluster's vertex window: centroid + max radius. Cheap and
/// conservative (a Ritter/Welzl fit would be tighter, a later refinement); correctness for
/// culling only needs it to *contain* every vertex, which centroid+max-dist guarantees.
fn bounding_sphere(vertices: &[MeshVertex], window: &[u32]) -> (Vec3, f32) {
    if window.is_empty() {
        return (Vec3::ZERO, 0.0);
    }
    let mut center = Vec3::ZERO;
    for &v in window {
        center += Vec3::from(vertices[v as usize].pos);
    }
    center /= window.len() as f32;
    let mut r2 = 0.0f32;
    for &v in window {
        r2 = r2.max(center.distance_squared(Vec3::from(vertices[v as usize].pos)));
    }
    (center, r2.sqrt())
}

/// Normal cone for backface cluster culling: axis = normalized sum of geometric face normals;
/// cutoff = the minimum face-normal·axis (`cos` of the cone half-angle that contains every
/// face normal). A non-positive cutoff (spread too wide, or a degenerate axis) is stored as
/// `-1.0`, meaning "never backface-cull this cluster".
fn normal_cone(vertices: &[MeshVertex], window: &[u32], tris: &[[u8; 3]]) -> (Vec3, f32) {
    let pos = |slot: u8| Vec3::from(vertices[window[slot as usize] as usize].pos);
    let mut face_normals: Vec<Vec3> = Vec::with_capacity(tris.len());
    let mut sum = Vec3::ZERO;
    for t in tris {
        let n = (pos(t[1]) - pos(t[0])).cross(pos(t[2]) - pos(t[0]));
        let n = n.normalize_or_zero();
        face_normals.push(n);
        sum += n;
    }
    let axis = sum.normalize_or_zero();
    if axis == Vec3::ZERO {
        return (Vec3::Z, -1.0);
    }
    let mut cutoff = 1.0f32;
    for n in &face_normals {
        if *n != Vec3::ZERO {
            cutoff = cutoff.min(n.dot(axis));
        }
    }
    if cutoff <= 0.0 {
        (axis, -1.0)
    } else {
        (axis, cutoff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_mesh(n: u32) -> (Vec<MeshVertex>, Vec<u32>) {
        // An n×n quad grid → 2*(n-1)² triangles, enough to force several clusters.
        let mut verts = Vec::new();
        for y in 0..n {
            for x in 0..n {
                verts.push(MeshVertex {
                    pos: [x as f32, y as f32, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [x as f32, y as f32],
                });
            }
        }
        let mut idx = Vec::new();
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let a = y * n + x;
                let b = a + 1;
                let c = a + n;
                let d = c + 1;
                idx.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        (verts, idx)
    }

    #[test]
    fn clusters_respect_caps_and_cover_all_triangles() {
        let (verts, idx) = grid_mesh(20);
        let mc = build_clusters(&verts, &idx, 7);
        assert!(
            mc.clusters.len() > 1,
            "grid should split into many clusters"
        );
        let mut total_tris = 0u32;
        for c in &mc.clusters {
            assert!(c.vertex_count as usize <= MAX_CLUSTER_VERTICES);
            assert!(c.triangle_count as usize <= MAX_CLUSTER_TRIANGLES);
            assert_eq!(c.material, 7);
            total_tris += c.triangle_count;
            // Every local triangle index is within the cluster's vertex window.
            let base = c.triangle_offset as usize;
            for k in 0..c.triangle_count as usize * 3 {
                assert!((mc.cluster_triangles[base + k] as u32) < c.vertex_count);
            }
        }
        assert_eq!(
            total_tris as usize,
            idx.len() / 3,
            "all triangles clustered"
        );
    }

    #[test]
    fn flat_grid_cone_is_tight() {
        let (verts, idx) = grid_mesh(8);
        let mc = build_clusters(&verts, &idx, 0);
        // A planar grid: every face normal is parallel (±Z by winding), so the cone axis is
        // aligned to Z and the cutoff (min normal·axis) is ~1 (a tight, cullable cone).
        for c in &mc.clusters {
            let axis = Vec3::from(c.cone_axis);
            assert!(
                axis.dot(Vec3::Z).abs() > 0.99,
                "flat cluster cone should align to Z"
            );
            assert!(c.cone_cutoff > 0.99, "flat cluster cone should be tight");
        }
    }
}
