# Phase 14 — Nanite-aligned cluster build (`build_lod_dag` rework)

Reworks the virtual-geometry offline cluster builder (`crates/asset/src/vgeo.rs`) from a
naive **in-order greedy sweep** to a **spatially-partitioned, position-welded** build that
mirrors Unreal Nanite's `NaniteBuilder`. Motivated by the Sponza interior validation: the
in-order sweep gives loose cluster bounds (weak culling) and leaves coincident/overlapping
faces that z-fight in the R64 visibility buffer (the arch-soffit streak).

## Reference: Nanite's build (studied from UE 5.x source)

- **Position-welded edge adjacency** — `ClusterDAG.cpp:42–103`. Every triangle edge is hashed
  by its two endpoint **positions** (`FEdgeHash`/`HashPosition`), not vertex indices. Opposite
  matching edges link the two triangles → a triangle adjacency graph. Welds split vertices (UV
  seams / duplicated indices) and flags non-manifold edges shared by >2 triangles (`-2`). A
  `FDisjointSet` finds connected islands.
- **Locality links** — `ClusterDAG.cpp:148` `BuildLocalityLinks`: Morton/bounds/material-aware
  spatial links so disconnected islands that are physically near still cluster together.
- **Graph partition** — `GraphPartitioner.cpp:34–79` `METIS_PartGraphKway`: split the
  adjacency+locality graph into ~128-tri clusters **minimizing cut edges** → compact clusters,
  tight bounds, few external edges, balanced sizes.
- **External-edge locking** — `Cluster.cpp:239–251`: each cluster records edges shared with
  *other* clusters; the simplifier locks them → crack-free LOD.

## Our current build (`crates/asset/src/vgeo.rs`)

`clusterize` sweeps triangles in **input order**, packing 124-tri / 64-vert chunks. No
adjacency, no position weld, no spatial partition, no external-edge locking (crack-freeness
relies solely on the group-uniform-error trick). Consequences measured on Sponza:

- Cluster bounds can span the whole mesh → frustum/normal-cone cull far weaker than Nanite's.
- Coincident/overlapping opaque faces survive → visibility-buffer z-fight (soffit streak) that
  a depth-buffer forward renderer hides via draw order.

## Design (no new FFI — keep the "minimal installs" constraint)

METIS is a C library; we do **not** add it. Instead a pure-Rust spatial partition that captures
the dominant Nanite win (spatial compactness) deterministically:

### M-A · Position weld + adjacency (fixes coincident faces)
1. **Position weld**: build a `HashMap<QuantizedPos, u32>` (positions quantized to a small grid
   to fold float noise) → a canonical vertex id per position. This is the weld Nanite's edge
   hash performs implicitly.
2. **Degenerate + exact-duplicate triangle removal**: drop triangles whose three welded ids are
   not distinct (degenerate), and drop triangles whose welded-id triple (as an unordered set
   with winding) already appears — the true source of the soffit z-fight when Sponza authored a
   face twice. **Distinct** coplanar faces (different positions/UVs) are kept; only exact
   duplicates are removed, so no real surface is lost.
3. **Edge adjacency** over welded ids: `HashMap<(min,max welded id), Vec<tri>>` → per-triangle
   neighbor list. Non-manifold edges (>2 tris) recorded but not culled.

### M-B · Spatial clustering (fixes culling)
Replace the in-order sweep with **Morton-ordered greedy growth**:
1. Morton-encode each triangle centroid (quantized to the mesh AABB).
2. Sort triangles by Morton code (Nanite's locality links are themselves Morton-based).
3. Greedy-grow clusters along the sorted order, preferring adjacency-connected neighbors first
   (a light shared-edge locality pass), capped at 124 tris / 64 verts.

This yields spatially-compact clusters (tight bounds) without a graph-partition library. Full
METIS-quality edge-cut minimization is a later refinement (M-D) if culling still under-performs.

### M-C · External-edge locking (crack-free, quality)
Carry the adjacency into the group simplify: mark edges shared across cluster/group boundaries
and lock them in `crate::simplify` (already boundary-aware; feed it the real external-edge set
instead of the current heuristic). Keeps the group-uniform-error contract.

### M-D · (optional) graph-partition refinement
If tight-bounds culling still lags, add recursive spatial bisection (median/SAH, à la Nanite's
`FBVHCluster` fallback) minimizing cut edges. Deferred until measured.

## Verification (per phase)
- **Gallery anchor**: features-off byte-identical (build changes are opt-in behind `P14_VGEO`).
- **Determinism**: same input → same clusters (Morton + stable sort; no `HashMap` iteration
  order in the output path).
- **DX≡VK**: gallery `P14_VGEO` ≤ 0.001/ch.
- **Soffit**: Sponza interior `DEBUG_VIEW=2` normal — the arch soffit matches the mesh baseline
  (no cyan/streak) after M-A dedup.
- **Culling**: log cluster-bounds tightness (avg radius / mesh radius) before/after M-B; expect a
  large drop. `PROFILE_GPU` cut-pass cost on Sponza.
- **Cook**: `.dcasset` re-bake is keyed on the geometry hash; the new build changes cluster bytes
  → cache miss + re-cook is expected once.

## Rollout
M-B (spatial clustering, the culling win) is the primary value. M-C/M-D as tracked follow-ups.

## Empirical findings (2026-07-04, on `assets/Sponza/Sponza.gltf`)

Prototyped M-A and a draw-order visibility tiebreak specifically to kill the **arch-soffit
streak**; both were reverted after measuring — the soffit is **not** a build-side dedup or
tiebreak problem:

- **No removable duplicates.** A position-welded pass (exact position bits) over every Sponza
  mesh found **0 degenerate + 0 same-winding-duplicate** triangles, and only ~3–4 *back-to-back*
  (opposite-winding coplanar) pairs per mesh — far too few to explain the broad soffit dither.
  So the fighting faces are **distinct near-coplanar** geometry, not redundant faces to remove.
- **Not the tiebreak.** Inverting the payload so `atomicMax` breaks equal-depth ties toward the
  first-drawn (lowest) index — matching the forward depth buffer — left the soffit **unchanged**.
  Since neither tiebreak direction moves the winner, the winner is **depth-determined**: an
  up-facing surface has a genuinely nearer depth key in vgeo, yet the mesh baseline (same
  geometry at `VGEO_TAU=0.02`, same projection on D3D12) keeps the soffit. That contradiction
  (identical depth inputs, opposite winner) can only be resolved with **pixel-level ground
  truth** of the visibility buffer + depth.
- **RenderDoc is blocked**: the sandbox exits right after Vulkan device creation under
  `renderdoccmd` injection (RT/mesh-shader extensions) and `renderdoc::new()` returns `None`, so
  no `.rdc` is captured. Cracking the soffit needs either that injection fixed or a GPU→CPU
  visibility-buffer readback for offline inspection — tracked separately from this clustering work.

**Conclusion:** the soffit streak is orthogonal to clustering; do M-B for culling on its own
merits, and debug the soffit with real ground truth as a separate task.
