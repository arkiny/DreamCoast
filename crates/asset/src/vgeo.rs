//! Virtual-geometry cluster data model + offline builder (Phase 14 M1).
//!
//! A mesh is split into **clusters** (meshlets, ~128 triangles) that reference a shared
//! vertex pool through a compact local window — the meshopt-style `(vertex window, u8 triangle
//! indices)` layout the GPU mesh-shader path consumes. Each cluster carries the culling bounds
//! (bounding sphere + normal cone) the runtime needs.
//!
//! [`build_clusters`] produces the finest LOD; [`build_lod_dag`] builds the full **LOD DAG** —
//! recursively group clusters, merge + simplify each group (locking its boundary so seams stay
//! crack-free, via [`crate::simplify`]), and re-clusterize into a coarser level, recording the
//! monotone `self_error`/`parent_error` bounds the runtime projects for continuous LOD.
//!
//! We build our own clusterizer + grouping (referencing meshoptimizer / METIS, no FFI): a greedy
//! sweep fills a cluster to the vertex/triangle caps; a greedy shared-edge partitioner forms
//! groups. Order-based greedy is spatially naive but valid and deterministic; a shared-edge
//! locality pass on the clusterizer is a later refinement (it only changes cluster *shape*, not
//! the format or the runtime contract).

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

    // ── LOD DAG (Phase 14 M1d) ──────────────────────────────────────────────────────────
    // Two error/sphere pairs drive crack-free continuous LOD. `self_*` is the error incurred to
    // create THIS cluster's LOD (0 at the finest); `parent_*` is the error of coarsening one
    // level further. Both are GROUP-UNIFORM (every cluster simplified together shares them), so a
    // group makes one cut decision → no cracks. Monotone: `parent_error >= self_error`. Runtime
    // cut (M3): draw when `project(parent_*) > τ && project(self_*) <= τ`.
    /// LOD level (0 = finest / full detail).
    pub lod_level: u32,
    /// Group id at this cluster's LOD (clusters simplified together share it).
    pub group: u32,
    /// Error introduced to build this cluster's LOD (world-space; 0 at the finest LOD).
    pub self_error: f32,
    /// Bounding-sphere center the `self_error` projects against (the group's merged sphere).
    pub self_center: [f32; 3],
    /// Bounding-sphere radius for the `self_error` projection.
    pub self_radius: f32,
    /// Error of coarsening one level past this cluster (monotone ≥ `self_error`; `f32::MAX` at
    /// the root, i.e. "always fine enough to stop here").
    pub parent_error: f32,
    /// Bounding-sphere center the `parent_error` projects against (the parent group's sphere).
    pub parent_center: [f32; 3],
    /// Bounding-sphere radius for the `parent_error` projection.
    pub parent_radius: f32,
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

/// A group id meaning "not yet grouped for coarsening" (set by the DAG build once a cluster is
/// assigned to a group).
pub const UNGROUPED: u32 = u32::MAX;

/// Greedy meshlet sweep over `indices` (into `vertices`), producing a self-contained chunk of
/// clusters (`cluster_vertices`, `cluster_triangles`, `clusters`) with offsets relative to the
/// returned arrays — the caller rebases them when appending to a [`MeshClusters`]. LOD metadata:
/// `lod_level` / `self_error`; `self_sphere` is the group-uniform LOD sphere for coarse levels,
/// or `None` at LOD0 (each cluster uses its own bounds, since `self_error` is 0 there). `group`
/// and the `parent_*` fields default to "ungrouped / root" and are filled in by the DAG build.
/// Deterministic (input triangle order).
#[allow(clippy::type_complexity)]
fn clusterize(
    vertices: &[MeshVertex],
    indices: &[u32],
    material: u32,
    lod_level: u32,
    self_error: f32,
    self_sphere: Option<(Vec3, f32)>,
) -> (Vec<u32>, Vec<u8>, Vec<Cluster>) {
    let mut cv: Vec<u32> = Vec::new();
    let mut ct: Vec<u8> = Vec::new();
    let mut clusters: Vec<Cluster> = Vec::new();
    let tri_count = indices.len() / 3;

    let mut window: Vec<u32> = Vec::with_capacity(MAX_CLUSTER_VERTICES);
    let mut local_of: std::collections::HashMap<u32, u8> = std::collections::HashMap::new();
    let mut tris: Vec<[u8; 3]> = Vec::with_capacity(MAX_CLUSTER_TRIANGLES);

    let mut flush = |window: &mut Vec<u32>,
                     local_of: &mut std::collections::HashMap<u32, u8>,
                     tris: &mut Vec<[u8; 3]>| {
        if tris.is_empty() {
            return;
        }
        let (center, radius) = bounding_sphere(vertices, window);
        let (axis, cutoff) = normal_cone(vertices, window, tris);
        let (sc, sr) = self_sphere.unwrap_or((center, radius));
        clusters.push(Cluster {
            vertex_offset: cv.len() as u32,
            vertex_count: window.len() as u32,
            triangle_offset: ct.len() as u32,
            triangle_count: tris.len() as u32,
            bounds_center: center.to_array(),
            bounds_radius: radius,
            cone_axis: axis.to_array(),
            cone_cutoff: cutoff,
            material,
            lod_level,
            group: UNGROUPED,
            self_error,
            self_center: sc.to_array(),
            self_radius: sr,
            parent_error: f32::MAX,
            parent_center: [0.0; 3],
            parent_radius: 0.0,
        });
        cv.extend_from_slice(window);
        for t in tris.iter() {
            ct.extend_from_slice(t);
        }
        window.clear();
        local_of.clear();
        tris.clear();
    };

    for t in 0..tri_count {
        let tri = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];
        let new_verts = tri.iter().filter(|v| !local_of.contains_key(v)).count();
        let would_overflow = window.len() + new_verts > MAX_CLUSTER_VERTICES
            || tris.len() + 1 > MAX_CLUSTER_TRIANGLES;
        if would_overflow {
            flush(&mut window, &mut local_of, &mut tris);
        }
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
    flush(&mut window, &mut local_of, &mut tris);
    (cv, ct, clusters)
}

/// Append a [`clusterize`] chunk to `mc`, rebasing its vertex/triangle offsets. Returns the
/// range of appended cluster indices.
fn append_chunk(
    mc: &mut MeshClusters,
    chunk: (Vec<u32>, Vec<u8>, Vec<Cluster>),
) -> std::ops::Range<usize> {
    let (cv, ct, clusters) = chunk;
    let vbase = mc.cluster_vertices.len() as u32;
    let tbase = mc.cluster_triangles.len() as u32;
    let start = mc.clusters.len();
    for mut c in clusters {
        c.vertex_offset += vbase;
        c.triangle_offset += tbase;
        mc.clusters.push(c);
    }
    mc.cluster_vertices.extend_from_slice(&cv);
    mc.cluster_triangles.extend_from_slice(&ct);
    start..mc.clusters.len()
}

/// Build a mesh's finest-LOD clusters (Phase 14 M1a). One LOD (level 0, `self_error` 0). For
/// the full LOD DAG use [`build_lod_dag`]. Deterministic: the greedy sweep visits triangles in
/// input order, so the same mesh always produces the same clusters (cook-cache friendly).
pub fn build_clusters(vertices: &[MeshVertex], indices: &[u32], material: u32) -> MeshClusters {
    let mut mc = MeshClusters {
        vertices: vertices.to_vec(),
        ..Default::default()
    };
    let chunk = clusterize(&mc.vertices, indices, material, 0, 0.0, None);
    append_chunk(&mut mc, chunk);
    mc
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

/// Target clusters per group when grouping for LOD simplification (~8-32; the plan's midpoint).
/// A group's triangles are merged and simplified together, so bigger groups amortize the locked
/// boundary but coarsen granularity.
pub const TARGET_GROUP_SIZE: usize = 16;

/// An undirected edge as a sorted global-vertex pair (source-mesh indices).
type Edge = (u32, u32);

fn edge(a: u32, b: u32) -> Edge {
    if a < b { (a, b) } else { (b, a) }
}

/// Map each cluster's local triangle vertices back to source-mesh (global) vertex indices, so
/// two clusters that meet share the *same* edge key even though their `u8` local indices differ.
fn cluster_global_edges(mc: &MeshClusters, ci: usize) -> Vec<Edge> {
    let c = &mc.clusters[ci];
    let vbase = c.vertex_offset as usize;
    let tbase = c.triangle_offset as usize;
    let g = |local: u8| mc.cluster_vertices[vbase + local as usize];
    let mut seen = std::collections::HashSet::new();
    let mut edges = Vec::new();
    for t in 0..c.triangle_count as usize {
        let tri = [
            mc.cluster_triangles[tbase + t * 3],
            mc.cluster_triangles[tbase + t * 3 + 1],
            mc.cluster_triangles[tbase + t * 3 + 2],
        ];
        for k in 0..3 {
            let e = edge(g(tri[k]), g(tri[(k + 1) % 3]));
            if seen.insert(e) {
                edges.push(e);
            }
        }
    }
    edges
}

/// Shared-edge adjacency among a **subset** of clusters (a single LOD level), keyed by cluster
/// index: `adj[i][j]` = the number of edges clusters `i` and `j` share. Only edges between two
/// subset clusters count, so grouping stays within a level.
fn adjacency_subset(
    mc: &MeshClusters,
    subset: &[usize],
) -> std::collections::HashMap<usize, std::collections::HashMap<usize, u32>> {
    let mut edge_clusters: std::collections::HashMap<Edge, Vec<usize>> =
        std::collections::HashMap::new();
    for &ci in subset {
        for e in cluster_global_edges(mc, ci) {
            edge_clusters.entry(e).or_default().push(ci);
        }
    }
    let mut adj: std::collections::HashMap<usize, std::collections::HashMap<usize, u32>> =
        subset.iter().map(|&c| (c, Default::default())).collect();
    for cls in edge_clusters.values() {
        for i in 0..cls.len() {
            for j in i + 1..cls.len() {
                *adj.get_mut(&cls[i]).unwrap().entry(cls[j]).or_insert(0) += 1;
                *adj.get_mut(&cls[j]).unwrap().entry(cls[i]).or_insert(0) += 1;
            }
        }
    }
    adj
}

/// Partition a subset of clusters (one LOD level) into groups of up to `target` by greedy
/// region growing: seed the lowest unassigned cluster, then repeatedly annex the unassigned
/// cluster sharing the most edges with the current group (ties → lowest index) until full or
/// nothing is adjacent. Our own partitioner (referencing METIS's shared-boundary objective, no
/// FFI). Deterministic. Returns groups as cluster-index lists.
pub fn group_subset(mc: &MeshClusters, subset: &[usize], target: usize) -> Vec<Vec<usize>> {
    let adj = adjacency_subset(mc, subset);
    let mut order: Vec<usize> = subset.to_vec();
    order.sort_unstable();
    let mut assigned: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut groups = Vec::new();

    for &seed in &order {
        if assigned.contains(&seed) {
            continue;
        }
        assigned.insert(seed);
        let mut group = vec![seed];
        while group.len() < target {
            let mut weight: std::collections::HashMap<usize, u32> =
                std::collections::HashMap::new();
            for &c in &group {
                for (&nb, &w) in &adj[&c] {
                    if !assigned.contains(&nb) {
                        *weight.entry(nb).or_insert(0) += w;
                    }
                }
            }
            let best = weight
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)));
            match best {
                Some((c, _)) => {
                    assigned.insert(c);
                    group.push(c);
                }
                None => break,
            }
        }
        groups.push(group);
    }
    groups
}

/// The group's triangle soup as global (`mc.vertices`) indices — concatenating every cluster's
/// triangles, remapped from `u8` local windows. The unit the simplifier coarsens.
fn group_global_indices(mc: &MeshClusters, group: &[usize]) -> Vec<u32> {
    let mut out = Vec::new();
    for &ci in group {
        let c = &mc.clusters[ci];
        let vbase = c.vertex_offset as usize;
        let tbase = c.triangle_offset as usize;
        for k in 0..c.triangle_count as usize * 3 {
            out.push(mc.cluster_vertices[vbase + mc.cluster_triangles[tbase + k] as usize]);
        }
    }
    out
}

/// Bounding sphere (centroid + max radius) of every vertex referenced by a group's clusters.
fn group_bounds(mc: &MeshClusters, group: &[usize]) -> (Vec3, f32) {
    let mut verts: Vec<u32> = Vec::new();
    for &ci in group {
        let c = &mc.clusters[ci];
        let vbase = c.vertex_offset as usize;
        for i in 0..c.vertex_count as usize {
            verts.push(mc.cluster_vertices[vbase + i]);
        }
    }
    bounding_sphere(&mc.vertices, &verts)
}

/// Reconstruct the global (`vertices`) triangle index buffer for one LOD level — every cluster
/// at `lod_level`, its `u8` local triangles remapped through its vertex window. Used by the
/// debug viewer to render a chosen LOD and by the crack-free test to inspect its topology.
pub fn lod_indices(mc: &MeshClusters, lod_level: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for c in &mc.clusters {
        if c.lod_level != lod_level {
            continue;
        }
        let vbase = c.vertex_offset as usize;
        let tbase = c.triangle_offset as usize;
        for k in 0..c.triangle_count as usize * 3 {
            out.push(mc.cluster_vertices[vbase + mc.cluster_triangles[tbase + k] as usize]);
        }
    }
    out
}

/// The set of distinct LOD levels present, ascending (0 = finest).
pub fn lod_levels(mc: &MeshClusters) -> Vec<u32> {
    let mut ls: Vec<u32> = mc.clusters.iter().map(|c| c.lod_level).collect();
    ls.sort_unstable();
    ls.dedup();
    ls
}

/// Build the full LOD DAG for a mesh (Phase 14 M1d): the finest clusters (LOD 0) plus
/// recursively coarser levels. Each level groups the current clusters, merges + simplifies each
/// group (locking its boundary so shared seams stay crack-free), and re-clusterizes the result
/// as the parent level. Records the monotone `self_error` / `parent_error` bounds the runtime
/// projects for continuous LOD. Recurses until a level stops shrinking or reaches one cluster.
/// Deterministic; single material (Phase 14 M1). Uses [`crate::simplify`].
pub fn build_lod_dag(vertices: &[MeshVertex], indices: &[u32], material: u32) -> MeshClusters {
    let mut mc = MeshClusters {
        vertices: vertices.to_vec(),
        ..Default::default()
    };
    let chunk = clusterize(&mc.vertices, indices, material, 0, 0.0, None);
    let mut current: Vec<usize> = append_chunk(&mut mc, chunk).collect();

    // Positions for the simplifier (shared across levels; vertices never change).
    let positions: Vec<Vec3> = mc.vertices.iter().map(|v| Vec3::from(v.pos)).collect();
    // Seam vertices (position shared by another index — UV/normal splits) are locked at every
    // level: index-based collapse can't tell the two sides coincide, so simplifying one side but
    // not the other would tear the seam. Conservative but crack-free on real (seamed) assets.
    let seam_lock = crate::simplify::duplicate_position_vertices(&positions);

    let mut lod_level = 1u32;
    let mut next_group_id = 0u32;
    while current.len() > 1 {
        let groups = group_subset(&mc, &current, TARGET_GROUP_SIZE);
        let mut next: Vec<usize> = Vec::new();

        for group in &groups {
            let group_tris = group_global_indices(&mc, group);
            let tri_n = group_tris.len() / 3;
            // Lock the group's boundary (its open-boundary edges = borders + edges shared with
            // OTHER groups, so adjacent groups coarsen a shared seam identically) plus global
            // position-seams. `locked_edges` additionally forbids deleting a border edge.
            let locked_edges = crate::simplify::open_boundary_edges(&group_tris);
            let mut locked = crate::simplify::open_boundary_vertices(&group_tris);
            locked.extend(seam_lock.iter().copied());
            let target = (tri_n / 2).max(1);
            let (simplified, err) = crate::simplify::simplify_subset(
                &positions,
                &group_tris,
                target,
                &locked,
                &locked_edges,
            );

            // Monotone group error: the worst child's error plus this level's simplification.
            let child_max = group
                .iter()
                .map(|&c| mc.clusters[c].self_error)
                .fold(0.0f32, f32::max);
            let group_error = child_max + err as f32;
            let (gc, gr) = group_bounds(&mc, group);
            let gid = next_group_id;
            next_group_id += 1;

            // Link the children into this group (their coarsening bound).
            for &c in group {
                mc.clusters[c].group = gid;
                mc.clusters[c].parent_error = group_error;
                mc.clusters[c].parent_center = gc.to_array();
                mc.clusters[c].parent_radius = gr;
            }

            // No reduction (e.g. everything on a locked boundary) → the group can't coarsen; its
            // clusters stay terminal (parent bound above still applies). Don't emit a parent LOD.
            if simplified.len() / 3 >= tri_n {
                continue;
            }
            let chunk = clusterize(
                &mc.vertices,
                &simplified,
                material,
                lod_level,
                group_error,
                Some((gc, gr)),
            );
            next.extend(append_chunk(&mut mc, chunk));
        }

        if next.is_empty() {
            break; // no group coarsened → the current level is the root set
        }
        current = next;
        lod_level += 1;
    }
    mc
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
    fn grouping_covers_all_clusters_and_bounds_size() {
        let (verts, idx) = grid_mesh(40);
        let mc = build_clusters(&verts, &idx, 0);
        assert!(
            mc.clusters.len() > 20,
            "need enough clusters to form groups"
        );

        let all: Vec<usize> = (0..mc.clusters.len()).collect();
        let groups = group_subset(&mc, &all, TARGET_GROUP_SIZE);
        // Every cluster is in exactly one group.
        let mut seen = vec![false; mc.clusters.len()];
        let mut total = 0;
        for g in &groups {
            assert!(!g.is_empty());
            assert!(g.len() <= TARGET_GROUP_SIZE, "group exceeds target size");
            for &c in g {
                assert!(!seen[c], "cluster {c} in two groups");
                seen[c] = true;
                total += 1;
            }
        }
        assert_eq!(total, mc.clusters.len(), "all clusters grouped");
        assert!(seen.iter().all(|&s| s), "no cluster left ungrouped");

        // Determinism: same input → same partition.
        assert_eq!(groups, group_subset(&mc, &all, TARGET_GROUP_SIZE));
    }

    #[test]
    fn adjacency_is_symmetric() {
        let (verts, idx) = grid_mesh(24);
        let mc = build_clusters(&verts, &idx, 0);
        let all: Vec<usize> = (0..mc.clusters.len()).collect();
        let adj = adjacency_subset(&mc, &all);
        for (&i, nbrs) in &adj {
            for (&j, &w) in nbrs {
                assert_eq!(adj[&j].get(&i), Some(&w), "adjacency asymmetric");
            }
        }
    }

    /// A grid displaced by a smooth height field — non-planar, so simplification incurs real
    /// (nonzero) QEM error, unlike the flat `grid_mesh`.
    fn bumpy_grid(n: u32) -> (Vec<MeshVertex>, Vec<u32>) {
        let (mut verts, idx) = grid_mesh(n);
        for v in &mut verts {
            let (x, y) = (v.pos[0], v.pos[1]);
            v.pos[2] = (x * 0.4).sin() * 1.5 + (y * 0.3).cos() * 1.2;
        }
        (verts, idx)
    }

    #[test]
    fn lod_dag_builds_monotone_shrinking_levels() {
        // A denser bumpy grid so several LOD levels form and carry real error.
        let (verts, idx) = bumpy_grid(48);
        let mc = build_lod_dag(&verts, &idx, 5);

        let max_lod = mc.clusters.iter().map(|c| c.lod_level).max().unwrap();
        assert!(
            max_lod >= 1,
            "expected at least one coarser LOD, got {max_lod}"
        );

        // Per-level triangle totals must strictly shrink as LOD coarsens.
        let mut per_level = vec![0u32; max_lod as usize + 1];
        for c in &mc.clusters {
            per_level[c.lod_level as usize] += c.triangle_count;
        }
        for l in 1..=max_lod as usize {
            assert!(
                per_level[l] < per_level[l - 1],
                "LOD {l} ({}) not coarser than LOD {} ({})",
                per_level[l],
                l - 1,
                per_level[l - 1]
            );
        }

        // Monotone error bounds: parent_error >= self_error on every cluster; coarser LODs have
        // larger self_error than the finest (which is 0).
        for c in &mc.clusters {
            assert!(c.parent_error >= c.self_error, "parent_error < self_error");
            if c.lod_level == 0 {
                assert_eq!(c.self_error, 0.0, "LOD0 self_error must be 0");
            }
        }
        assert!(
            mc.clusters
                .iter()
                .any(|c| c.lod_level > 0 && c.self_error > 0.0),
            "coarser LODs should carry a nonzero error"
        );

        // Determinism.
        let mc2 = build_lod_dag(&verts, &idx, 5);
        assert_eq!(mc, mc2);
    }

    /// A torus: parametric `u×v` grid wrapped in both directions → a genuinely closed,
    /// index-watertight mesh (no seam, no poles, no duplicated positions). The clean case for a
    /// topological crack-free check.
    fn torus(u: u32, v: u32) -> (Vec<MeshVertex>, Vec<u32>) {
        use std::f32::consts::TAU;
        let (big_r, small_r) = (3.0f32, 1.0f32);
        let mut verts = Vec::new();
        for i in 0..u {
            for j in 0..v {
                let (t, p) = (TAU * i as f32 / u as f32, TAU * j as f32 / v as f32);
                let pos = [
                    (big_r + small_r * p.cos()) * t.cos(),
                    (big_r + small_r * p.cos()) * t.sin(),
                    small_r * p.sin(),
                ];
                verts.push(MeshVertex {
                    pos,
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                });
            }
        }
        let idx_of = |i: u32, j: u32| (i % u) * v + (j % v);
        let mut idx = Vec::new();
        for i in 0..u {
            for j in 0..v {
                let (a, b, c, d) = (
                    idx_of(i, j),
                    idx_of(i + 1, j),
                    idx_of(i, j + 1),
                    idx_of(i + 1, j + 1),
                );
                idx.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }
        (verts, idx)
    }

    #[test]
    fn lod_levels_are_crack_free_on_a_closed_mesh() {
        // A torus is closed and index-watertight: no open boundary. If simplification tore a
        // seam, that LOD would gain boundary edges. Assert every LOD stays closed → crack-free.
        // This is the M1 gate, checked topologically (stronger than eyeballing a screenshot).
        let (verts, idx) = torus(64, 32);
        assert!(
            crate::simplify::open_boundary_vertices(&idx).is_empty(),
            "test mesh must be closed"
        );
        assert!(
            crate::simplify::duplicate_position_vertices(
                &verts.iter().map(|v| Vec3::from(v.pos)).collect::<Vec<_>>()
            )
            .is_empty(),
            "torus should have no duplicated positions"
        );

        let mc = build_lod_dag(&verts, &idx, 0);
        let levels = lod_levels(&mc);
        assert!(levels.len() >= 2, "expected multiple LODs, got {levels:?}");
        for &l in &levels {
            let lidx = lod_indices(&mc, l);
            assert!(!lidx.is_empty(), "LOD {l} has no geometry");
            let open = crate::simplify::open_boundary_vertices(&lidx);
            assert!(
                open.is_empty(),
                "LOD {l} introduced {} boundary vertices (a crack)",
                open.len()
            );
        }
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
