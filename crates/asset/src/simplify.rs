//! Quadric-error-metric (QEM) triangle-mesh simplification (Phase 14 M1c).
//!
//! Our own edge-collapse simplifier (referencing Garland–Heckbert, no FFI), used by the
//! virtual-geometry LOD DAG to coarsen a cluster group. It is a **subset** simplifier: a
//! collapse merges vertex `u` into a surviving vertex `v` and `v` never moves, so survivors are
//! a subset of the input vertices with their positions (and hence normals/uv) preserved exactly.
//! That keeps attributes consistent for free and lets us pin group boundaries **crack-free** —
//! a `locked` vertex is never removed, so a shared boundary is simplified identically no matter
//! which group's simplification touches it.
//!
//! The collapse cost is the Garland–Heckbert quadric evaluated at the survivor: the sum of
//! squared distances to the planes of the faces around the removed vertex. `simplify_subset`
//! returns the coarsened index buffer (into the *same* vertex array) and the maximum geometric
//! error (world-space distance) it introduced — the monotone error the DAG projects at runtime.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use dreamcoast_core::glam::{DVec3, Vec3};

/// An undirected edge as a sorted vertex-index pair.
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// A symmetric 4×4 quadric `Q` (Garland–Heckbert) storing the 10 unique upper-triangle terms
/// `[a², ab, ac, ad, b², bc, bd, c², cd, d²]` for a plane `(a,b,c,d)`. `f64` for stability.
#[derive(Clone, Copy, Default)]
struct Quadric {
    m: [f64; 10],
}

impl Quadric {
    /// Quadric of a plane `n·x + d = 0` (`n` unit): the outer product `p pᵀ`, `p = (a,b,c,d)`.
    fn from_plane(n: DVec3, d: f64) -> Self {
        let (a, b, c) = (n.x, n.y, n.z);
        Quadric {
            m: [
                a * a,
                a * b,
                a * c,
                a * d,
                b * b,
                b * c,
                b * d,
                c * c,
                c * d,
                d * d,
            ],
        }
    }
    fn add(&mut self, o: &Quadric) {
        for i in 0..10 {
            self.m[i] += o.m[i];
        }
    }
    /// `[x 1]ᵀ Q [x 1]` — the sum of squared plane distances at point `p` (clamped ≥ 0).
    fn eval(&self, p: DVec3) -> f64 {
        let (x, y, z) = (p.x, p.y, p.z);
        let m = &self.m;
        let e = m[0] * x * x
            + 2.0 * m[1] * x * y
            + 2.0 * m[2] * x * z
            + 2.0 * m[3] * x
            + m[4] * y * y
            + 2.0 * m[5] * y * z
            + 2.0 * m[6] * y
            + m[7] * z * z
            + 2.0 * m[8] * z
            + m[9];
        e.max(0.0)
    }
}

/// A pending collapse `u → v`, ordered by ascending cost (min-heap via reversed `Ord`).
/// `gen_u`/`gen_v` snapshot the endpoints' generation so a stale entry (an endpoint changed
/// since it was pushed) is detected and skipped when popped — lazy deletion, no heap updates.
struct Collapse {
    cost: f64,
    u: u32,
    v: u32,
    gen_u: u32,
    gen_v: u32,
}
impl PartialEq for Collapse {
    fn eq(&self, o: &Self) -> bool {
        self.cost == o.cost
    }
}
impl Eq for Collapse {}
impl PartialOrd for Collapse {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Collapse {
    fn cmp(&self, o: &Self) -> Ordering {
        // Reversed so `BinaryHeap` (a max-heap) yields the LOWEST cost first; tie-break on
        // indices keeps the pop order deterministic.
        o.cost
            .total_cmp(&self.cost)
            .then(o.u.cmp(&self.u))
            .then(o.v.cmp(&self.v))
    }
}

/// Simplify a triangle mesh to at most `target_tris` triangles by QEM edge collapses. Two
/// constraints keep shared boundaries crack-free: a `locked` vertex is never removed, and a
/// `locked_edges` edge is never deleted — even by collapsing the interior apex of the triangle
/// that carries it (which would otherwise degenerate the triangle and tear the border). Pass the
/// group's open-boundary edges as `locked_edges` (and their endpoints, plus seams, as `locked`).
/// `positions` is the vertex pool; `indices` are triangles into it. Returns the coarsened index
/// buffer (into the SAME `positions`) and the max geometric error introduced. Deterministic.
pub fn simplify_subset(
    positions: &[Vec3],
    indices: &[u32],
    target_tris: usize,
    locked: &HashSet<u32>,
    locked_edges: &HashSet<(u32, u32)>,
) -> (Vec<u32>, f64) {
    let n = positions.len();
    let pos: Vec<DVec3> = positions.iter().map(|p| p.as_dvec3()).collect();

    // Live triangles (None = collapsed away) and per-vertex incident-triangle sets.
    let mut tris: Vec<Option<[u32; 3]>> = indices
        .chunks_exact(3)
        .map(|t| Some([t[0], t[1], t[2]]))
        .collect();
    let mut incident: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for (ti, t) in tris.iter().enumerate() {
        if let Some(t) = t {
            for &v in t {
                incident[v as usize].insert(ti);
            }
        }
    }

    // Per-vertex accumulated quadric + a generation counter for lazy heap invalidation.
    let mut quad: Vec<Quadric> = vec![Quadric::default(); n];
    for t in tris.iter().flatten() {
        let (v0, v1, v2) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        let nrm = (v1 - v0).cross(v2 - v0);
        let len = nrm.length();
        if len > 0.0 {
            let nrm = nrm / len;
            let q = Quadric::from_plane(nrm, -nrm.dot(v0));
            for &v in t {
                quad[v as usize].add(&q);
            }
        }
    }
    let mut generation = vec![0u32; n];
    let mut alive = vec![true; n];
    let mut live_tris = tris.iter().filter(|t| t.is_some()).count();

    // Collapse cost of removing `u` onto `v` (v survives, unmoved): the pair quadric at v.pos.
    let cost = |quad: &[Quadric], u: u32, v: u32| -> f64 {
        let mut q = quad[u as usize];
        q.add(&quad[v as usize]);
        q.eval(pos[v as usize])
    };

    // Seed the heap with every undirected edge, oriented so the survivor is valid vs `locked`.
    let mut heap = BinaryHeap::new();
    let push_edge =
        |heap: &mut BinaryHeap<Collapse>, quad: &[Quadric], a: u32, b: u32, generation: &[u32]| {
            // Orientation: remove the non-locked endpoint. Both locked → not collapsible.
            let (u, v) = match (locked.contains(&a), locked.contains(&b)) {
                (true, true) => return,
                (false, true) => (a, b),
                (true, false) => (b, a),
                (false, false) => (a, b), // remove `a` onto `b`; deterministic by edge orientation
            };
            heap.push(Collapse {
                cost: cost(quad, u, v),
                u,
                v,
                gen_u: generation[u as usize],
                gen_v: generation[v as usize],
            });
        };
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    for t in tris.iter().flatten() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let e = if a < b { (a, b) } else { (b, a) };
            if edges.insert(e) {
                push_edge(&mut heap, &quad, e.0, e.1, &generation);
            }
        }
    }

    let mut max_err = 0.0f64;
    while live_tris > target_tris {
        let Some(c) = heap.pop() else { break };
        // Stale? (an endpoint changed generation since this was pushed) → skip.
        if !alive[c.u as usize]
            || !alive[c.v as usize]
            || generation[c.u as usize] != c.gen_u
            || generation[c.v as usize] != c.gen_v
        {
            continue;
        }
        if locked.contains(&c.u) {
            continue; // never remove a locked vertex
        }
        // Edge-preservation guard: collapsing `u` onto `v` degenerates any triangle (u,v,w),
        // which would delete the border edge (v,w). If (v,w) is locked, forbid this collapse so
        // the shared boundary survives identically on both sides (crack-free).
        let deletes_locked_edge = incident[c.u as usize].iter().any(|&ti| {
            let Some(t) = tris[ti] else { return false };
            if !t.contains(&c.v) {
                return false;
            }
            // The third vertex of the triangle (the one that isn't u or v).
            let w = t.iter().copied().find(|&x| x != c.u && x != c.v);
            w.map(|w| locked_edges.contains(&edge_key(c.v, w)))
                .unwrap_or(false)
        });
        if deletes_locked_edge {
            continue;
        }
        // Apply: rewrite u→v in every incident triangle, dropping degenerates.
        let u_tris: Vec<usize> = incident[c.u as usize].iter().copied().collect();
        let mut removed = 0usize;
        for ti in &u_tris {
            let Some(t) = tris[*ti] else { continue };
            let nt = t.map(|x| if x == c.u { c.v } else { x });
            if nt[0] == nt[1] || nt[1] == nt[2] || nt[0] == nt[2] {
                // Degenerate → drop; detach from its vertices.
                for &v in &t {
                    incident[v as usize].remove(ti);
                }
                tris[*ti] = None;
                removed += 1;
            } else {
                tris[*ti] = Some(nt);
                incident[c.v as usize].insert(*ti);
                incident[c.u as usize].remove(ti);
            }
        }
        alive[c.u as usize] = false;
        let qu = quad[c.u as usize];
        quad[c.v as usize].add(&qu);
        generation[c.v as usize] += 1;
        // QEM (plane distance) is PLANAR-LOSSLESS: a collapse that fans an interior vertex out into
        // long, thin, COPLANAR triangles bridging across an opening costs ≈0 QEM, yet it changes
        // COVERAGE (the stretched triangle fills the opening, occluding what's behind). QEM cannot
        // see that. So also bound the error by the geometric SIZE the collapse introduced — the
        // longest edge among the survivor's resulting triangles that grew past its pre-collapse
        // extent. A coarse cluster's error is then ≥ its triangle size, so the runtime LOD cut only
        // selects it once those triangles are sub-`tau` on screen (distance-appropriate) — matching
        // the always-LOD0 mesh fill up close, where a stretched bridge would otherwise pop forward.
        let mut span = 0.0f64;
        for &ti in &incident[c.v as usize] {
            if let Some(t) = tris[ti] {
                for k in 0..3 {
                    span = span.max((pos[t[k] as usize] - pos[t[(k + 1) % 3] as usize]).length());
                }
            }
        }
        max_err = max_err.max(c.cost.sqrt()).max(span);
        live_tris -= removed;

        // Re-price edges around the survivor.
        let neigh: HashSet<u32> = incident[c.v as usize]
            .iter()
            .flat_map(|ti| tris[*ti].unwrap())
            .filter(|&w| w != c.v && alive[w as usize])
            .collect();
        for w in neigh {
            push_edge(&mut heap, &quad, c.v, w, &generation);
        }
    }

    // Compact surviving triangles into a flat index buffer (still into `positions`).
    let mut out = Vec::with_capacity(live_tris * 3);
    for t in tris.iter().flatten() {
        out.extend_from_slice(t);
    }
    (out, max_err)
}

/// Edges on the mesh's **open boundary** — used by exactly one triangle. For a group's triangle
/// soup these are the group's border: true mesh borders plus edges shared with another group
/// (whose other triangle lives in that group). The LOD DAG passes these as `locked_edges` so a
/// shared border is coarsened identically on both sides (crack-free).
pub fn open_boundary_edges(indices: &[u32]) -> HashSet<(u32, u32)> {
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for t in indices.chunks_exact(3) {
        for k in 0..3 {
            *edge_count
                .entry(edge_key(t[k], t[(k + 1) % 3]))
                .or_insert(0) += 1;
        }
    }
    edge_count
        .into_iter()
        .filter(|&(_, n)| n == 1)
        .map(|(e, _)| e)
        .collect()
}

/// Endpoints of every open-boundary edge (see [`open_boundary_edges`]) — the vertices the LOD
/// DAG locks so a border vertex is never removed.
pub fn open_boundary_vertices(indices: &[u32]) -> HashSet<u32> {
    let mut locked = HashSet::new();
    for (a, b) in open_boundary_edges(indices) {
        locked.insert(a);
        locked.insert(b);
    }
    locked
}

/// Vertices whose position is shared by another distinct index — **seam** vertices (a UV/normal
/// split, or a longitude seam). Index-based edge collapse can't see that the two sides coincide,
/// so collapsing one side but not the other tears a visible crack. The LOD DAG locks these so a
/// seam is preserved identically on both sides. Positions are keyed bit-exactly (the cook is
/// deterministic), so only truly-coincident vertices weld.
pub fn duplicate_position_vertices(positions: &[Vec3]) -> HashSet<u32> {
    let mut first: HashMap<[u32; 3], u32> = HashMap::new();
    let mut dup = HashSet::new();
    for (i, p) in positions.iter().enumerate() {
        let key = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
        match first.get(&key) {
            Some(&j) => {
                dup.insert(j);
                dup.insert(i as u32);
            }
            None => {
                first.insert(key, i as u32);
            }
        }
    }
    dup
}

#[cfg(test)]
mod tests {
    use super::*;

    /// n×n planar grid of vertices → 2·(n-1)² triangles in the z=0 plane.
    fn grid(n: u32) -> (Vec<Vec3>, Vec<u32>) {
        let mut pos = Vec::new();
        for y in 0..n {
            for x in 0..n {
                pos.push(Vec3::new(x as f32, y as f32, 0.0));
            }
        }
        let mut idx = Vec::new();
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let a = y * n + x;
                idx.extend_from_slice(&[a, a + n, a + 1, a + 1, a + n, a + n + 1]);
            }
        }
        (pos, idx)
    }

    #[test]
    fn planar_grid_simplifies_with_bounded_span_error() {
        let (pos, idx) = grid(9); // 8×8 quads, unit spacing → extent 8
        let tri_count = idx.len() / 3;
        let target = tri_count / 2;
        let (out, err) = simplify_subset(&pos, &idx, target, &HashSet::new(), &HashSet::new());
        assert!(out.len() / 3 <= target, "reached target triangle count");
        assert!(out.len() / 3 > 0, "did not collapse to nothing");
        // The QEM (plane-distance) part is 0 (all collapses stay in z=0), but the reported error now
        // ALSO bounds the geometric SPAN the collapse introduces — the longest resulting triangle
        // edge — so a coarse LOD is only selected once its triangles are sub-`tau` on screen (the
        // fix for near-coplanar bridge collapses that QEM alone rates as lossless). So the error is
        // NOT ~0 but is bounded by the coarsened triangle size (≤ the grid's 8-unit extent here).
        assert!(
            err > 0.0,
            "span error should be non-zero for a coarsened grid"
        );
        assert!(
            err <= 8.0 * std::f64::consts::SQRT_2 + 1e-6,
            "span error must stay bounded by the grid diagonal, got {err}"
        );
    }

    #[test]
    fn locked_boundary_vertices_survive() {
        let (pos, idx) = grid(9);
        let locked = open_boundary_vertices(&idx);
        assert!(!locked.is_empty());
        // Aggressive target to force collapsing everything collapsible.
        let (out, _) = simplify_subset(&pos, &idx, 1, &locked, &open_boundary_edges(&idx));
        let used: HashSet<u32> = out.iter().copied().collect();
        // Boundary ring vertices must all still be referenced (never removed).
        for v in &locked {
            assert!(used.contains(v), "locked boundary vertex {v} was removed");
        }
    }

    #[test]
    fn deterministic() {
        let (pos, idx) = grid(12);
        let t = idx.len() / 3 / 2;
        let a = simplify_subset(&pos, &idx, t, &HashSet::new(), &HashSet::new());
        let b = simplify_subset(&pos, &idx, t, &HashSet::new(), &HashSet::new());
        assert_eq!(a.0, b.0);
        assert_eq!(a.1.to_bits(), b.1.to_bits());
    }
}
