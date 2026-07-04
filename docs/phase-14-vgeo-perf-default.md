# Phase 14 — vgeo: DAG cache, default-on, perf, and the cone-cull follow-up

Status: items 1 / 2 / 4 **DONE (2026-07-04)**; item 3 **planned**. This is the "make vgeo the
default renderer" track that follows the correctness fixes in
[`docs/phase-14-vgeo-lod-soffit-fix.md`](phase-14-vgeo-lod-soffit-fix.md).

## 1. Per-mesh LOD-DAG cache (DONE) — the startup blocker

`VgeoSystem::new` rebuilt the LOD DAG (`build_lod_dag`) for every mesh at load. That is cheap for
the gallery but a **17-minute wall** for Intel "New Sponza" (452 cluster pages, 564k clusters).

Fix: cache each mesh's DAG to `cache/dcasset/vgeo/<geom-hash>.dcasset` via the existing standalone
cluster page (`dcasset::write_clusters`/`read_clusters`), keyed on the geometry hash + the format
version. Load on a hit, build + write on a miss; a read/write error just falls back to building
(the cache is an optimization, never a correctness source). `build_lod_dag` is cross-process
deterministic (verified: two build processes produce byte-identical cache files) and the cluster
serialization is exact (`assert_eq!(mc, back)` round-trip test), so the cached DAG *is* the built
DAG.

Measured (RTX 2070 SUPER, D3D12):

| Scene | build (cold) | cached load | render diff (fixed exposure) |
|-------|-------------:|------------:|:----------------------------:|
| Sponza (103 meshes) | ~6.4 s start | ~4.3 s | 0.0 |
| Intel New Sponza (452 pages) | **1036 s** | **17 s** (~60×) | **0.0** |

> Debug lesson: **auto-exposure is a temporal accumulator** whose adaptation is perturbed by the
> multi-second DAG-build delay on the *build* run, so a build-vs-load screenshot differs ~0.0015
> with it on — an exposure artifact, not the DAG. Use `AUTO_EXPOSURE=0` for byte comparisons
> (like the earlier tonemap / `saturate` lessons).

## 2. Virtual geometry default-ON for non-gallery scenes (DONE)

`P14_VGEO` now defaults **ON** for levels / glTF scenes / worlds on a 64-bit-atomics adapter, and
**OFF** for the gallery (the fixed-exposure byte-identical DX≡VK anchor). `P14_VGEO=0/1` overrides.
Binning (the HW mesh-shader large-triangle path) likewise defaults ON when the adapter has mesh
shaders (`P14_VGEO_BIN=0` forces SW-only). **Skinned / morphed objects route to the mesh fill** —
their cluster page is baked from the static bind pose, so the mesh fill (which runs the GPU
skinning/morph vertex shader) must draw them or the animation freezes.

Verified: Sponza default-on == explicit `P14_VGEO=1` (0.0 avg/ch); gallery default == `P14_VGEO=0`
(0.0, anchor preserved). Fallback to the mesh fill is per-object (`page_for` → `None`) and whole-
system (`VgeoSystem::new` failure logs and leaves the mesh fill), so an unsupported adapter or a
mesh that can't be clustered degrades cleanly.

## 4. Runtime perf (DONE, measured)

The Sponza interior frame is **GDF software-RT bound**: ~25.6 ms of GPU passes, of which the mesh
G-buffer fill is only **0.39 ms** and the rest is the GDF AO / GI / reflection + surface-cache
compute. vgeo reuses that same GDF (it runs on the fused CPU triangle soup, not the cluster page),
so its G-buffer production (cut → clear → raster/hwvis → resolve, sub-millisecond) is negligible to
the frame — **vgeo does not regress runtime** on this GDF-heavy content; the visibility-buffer path
is a G-buffer *producer* swap, not a lighting change.

Profiler note: `PROFILE_GPU`'s headless dump is skipped under vgeo because vgeo records passes
**per object** (≈103 × 5 passes on Sponza), overflowing the fixed GPU timer-query heap. A per-frame
consolidated vgeo pass (single cut/raster/resolve over all objects) would both fix profiling and cut
per-object overhead — tracked with item 3.

## 3. Re-enable the normal-cone cluster cull (PLANNED)

The normal-cone backface **cluster** cull is disabled (see the soffit-fix doc §2) because the naive
greedy in-order clusterizer produces **loose, inaccurate cones** that over-cull silhouette clusters.
Re-enabling it (recovering that culling perf) needs **tight per-cluster bounds**, i.e. the planned
Nanite-style build (`docs/phase-14-vgeo-clustering.md`, "M-B"):

1. **Spatial clustering** — replace the in-order greedy sweep with a Morton-order spatial partition
   (position-welded, no METIS FFI) so a cluster is a compact spatial patch → a tight normal cone +
   bounding sphere.
2. **Tight cone build** — recompute `cone_axis` / `cone_cutoff` (and the bounds sphere) from the
   spatially-coherent cluster; the corrected radius-aware `cone_backfacing` (already in
   `vgeo_cut.slang`) then culls reliably.
3. **Flip `CONE_CULL_ENABLED` on** and re-verify: cone-on ≡ cone-off render (perfectly conservative),
   DX≡VK ≤ 0.001, and measure the cut/raster saving.

Adjacent follow-ups surfaced by items 1/2/4: a **per-frame consolidated vgeo pass** (one cut/raster/
resolve over all objects instead of per-object serialization) to cut overhead + fix `PROFILE_GPU`,
and **HZB same-frame occlusion** (attempted + reverted in M4b, needs a genuinely-occluded scene like
Intel Sponza to validate the cull direction DX≡VK).
