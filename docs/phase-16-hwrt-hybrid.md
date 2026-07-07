# Phase 16 — Hybrid SW/HW ray-traced reflections (reference-engine style)

Move reflections from the software distance-field trace toward a hybrid that traces the **real
triangle BVH** with hardware ray tracing when available, shading the hit cheaply from the surface
cache — the reference engine's "SurfaceCache lighting mode". Opt-in (`P_HWRT`); default off keeps
the SW-RT path and the gallery byte-identical. Trademark-free: "reference engine" only.

## Why

The SW reflection traces a baked distance field at reduced resolution → it approximates geometry
(leaking / see-through thin surfaces) and delivers low-res hits. HW rays hit exact triangles. The
reference engine combines them: a HW BVH ray finds the hit; the hit is shaded from the surface cache
(cheap) or, when the cache is invalid, from full material evaluation ("Hit Lighting"). We are
well-aligned: HWRT infra already exists (BLAS/TLAS, inline `RayQuery`, all backends incl. Metal),
we already shade reflection hits with `sample_surface_cache_cone`, and `reflect_temporal` already
does the reference's Karis tonemap + variance clamp + hit-distance carry.

## What landed (Phase A + B — SurfaceCache mode)

- **Phase A — content BLAS/TLAS** (`rt.rs::build_content_accel`): per-unique-mesh BLAS from the
  `MeshRegistry` + one world TLAS from the ECS draw list, bound as `g.tlas` (1 bindless slot). No
  per-primitive geometry table (that would overflow the 64-slot bindless limit and is only needed
  for Hit Lighting). Opt-in `P_HWRT`, RT-capable device, content only. Cost: sponza_intel ≈ 448 BLAS
  built once in ~140 ms on M3 (static scene; not per-frame).
- **Phase B — HWRT reflection trace** (`gdf_reflect.slang` `HWRT_REFLECT` permutation, mirroring
  `HWRT_GI`): the reflection ray is traced against `g.tlas` with an inline `RayQuery` (closest-hit,
  `rt_trace.slang` pattern) instead of the GDF march. The hit is shaded by the SAME downstream code
  (`sample_surface_cache_cone` at the GDF-gradient normal), so only the trace changes. Output flows
  unchanged through `reflect_temporal` → `reflect_composite` → `ambient_ibl`. Selected by `hwrt` in
  `record_gdf_reflect`; falls back to the SW pipeline on a non-RT device.

Verified (Metal): gallery golden `af70c1a5` byte-identical (default off); content TLAS builds; the
HW reflection produces geometrically **more accurate** reflection structure (the chrome ball's
reflected hall/curtains sit in the right places).

## Key finding — the sharpness bottleneck is the surface cache, not the trace

In SurfaceCache mode the HW hit is shaded from the low-res surface-cache atlas, so a near-mirror
(chrome ball) stays blurry and the rough floor is unchanged — HWRT fixes hit *accuracy* but shares
the cache's *resolution*. Measured: floor sparkle ≈ unchanged (0.013 → 0.016 cloud-fraction). So the
visible win needs one of:

- **Phase B.2 — screen-color-at-hit ✔ (LANDED)** (reference `SampleSceneColorAtHit`): at the HW hit,
  reproject to the previous frame via `globals.prev_view_proj`, validate against the current depth
  (reconstruct the world point at the reprojected UV and compare — rejects occluders), and sample the
  full-res **lit-radiance history** (`lit_hist`, the same buffer SSR reprojects into) for a SHARP
  reflection; the surface cache stays the off-screen / occluded fallback. HWRT supplies the validated
  hit that our SSR lacked (why content mirrors skip SSR). Plumbing: the HWRT reflect pipeline binds
  the globals UBO (`uniform_buffer: true`) for `prev_view_proj`; the lit-history index rides the
  march-cap push field (unused in the HW variant) so the 240 B push doesn't grow; `depth.GetDimensions`
  gives the full-res dims to index the history at a half-res trace. **Result (Metal): the chrome ball
  now reflects the real scene with correct lit colours (red/teal drapes, columns, arch, floor tiles)
  — a clear sharpness win over the milky SurfaceCache blob; floor sparkle unchanged (0.0135 ≈ SW).**
  Gallery `af70c1a5` byte-identical (off). Residual blockiness = the half-res trace (a later tier knob).
## What else landed

- **Phase C — full-res HWRT reflection** (`P_HWRT_FULLRES`): traces the reflection at full resolution
  (no half-res + upsample) for a crisp mirror — the chrome ball shows sharp columns, drapes, arch and
  floor tiles. ~4x cost (a quality/screenshot mode); default `P_HWRT` keeps the half-res trace.
- **Phase D — HWRT GI shaded from the surface cache** (`P_HWRT_GI` + content TLAS): the HWRT GI
  permutation now casts the cosine-hemisphere bounce rays against the BVH (closest hit) and shades
  the hit from the surface cache (`bs_shade_hit`, shared with the SW march) — actual bounced radiance,
  not the visibility-only first increment. Active on content with `P_HWRT=1 P_HWRT_GI=1 P_GI_VOLUME=0`
  (the volume path returns before the HWRT block). Gallery byte-identical (the `bs_shade_hit` extract
  is a pure refactor of the SW path).

## Phase E — Hit Lighting (LANDED, inline + consolidated geometry)

Off-screen reflection HW hits (where screen-color-at-hit can't reach) now shade with the REAL material
instead of the low-res surface cache. Opt-in `P_HWRT_HITLIGHTING`.

**Chosen approach (b) — inline + consolidated geometry, NOT the reference's RT-pipeline + SBT.** We
have a SINGLE metallic-roughness PBR model, so the reference's per-material closest-hit shaders (the
one thing the SBT buys) gain us nothing; consolidated geometry + one PBR eval scales fine and reuses
the proven inline RayQuery path (avoids the less-tested Metal RT-pipeline).
- **E1** `mesh.rs::build_content_hit_table`: ALL unique meshes packed into ONE vertex buffer + ONE
  index buffer (indices rebased absolute into the shared vertex buffer) + ONE per-drawable
  `{idx_base, prim_count, material}` record buffer — three bindless slots total regardless of mesh
  count (dodges the 64-slot per-primitive overflow; a reusable asset that also unblocks a content HW
  path tracer). Built alongside the content TLAS; `rt.rs` `content_hit` / `content_hit_indices()`.
- **E2** `gdf_reflect.slang` HWRT_REFLECT: capture the hit's instance/primitive/barycentrics/
  object-to-world; when enabled (`frame` bit31; the vtx/idx/table indices ride the HWRT-unused
  coarse-albedo push slots → no push growth / no D3D12 CBV spill), fetch the triangle's interpolated
  normal + UV, sample the albedo texture (wrap sampler), re-light with sun (HW shadow ray) + GI
  radiance cache + IBL sky (same energy convention as the analytic path). Reuses the B.2 screen-color
  path for on-screen hits (the reference's two-pass bookmark idea, collapsed into one inline trace).

Result (Metal, chromeball): the ball reflects the FULL scene with real materials — the off-screen ring
shows real curtains/columns/floor tiles, not the blurry cache. Gallery `af70c1a5` byte-identical (off).

## Reference — how the reference engine does Hit Lighting (extracted verbatim, for the record)

We deliberately deviated from this (see approach (b) above). **How the reference engine does it:** Hit Lighting is **RT-PIPELINE ONLY — inline
RayQuery is explicitly forbidden** (`LumenReflectionHardwareRayTracing.cpp:225`: `if (Inline &&
HitLighting) return false;`). It relies on the **Shader Binding Table**: each mesh/material has its own
closest-hit shader (`MaterialCHS`, `RayTracingMaterialHitShaders.usf:639`); at a hit the HW invokes that
material's shader, which **automatically** interpolates the triangle's vertex attributes
(`CalcInterpolants`) and evaluates the material (`GetMaterialPixelParameters`) into a full material
payload (`FPackedMaterialClosestHitPayload`) — **no manual geometry/material buffer fetch**. Lighting is
then `CalculateLightingAtHit` → `CalculateDirectLighting` → `AccumulateResults` (a **shadow ray per
light against the TLAS**) + skylight; secondary bounces stay on the surface cache. A **two-pass bookmark**
(`FLumenRayHitBookmark`) lets the cheap SurfaceCache DEFAULT pass record the hit so the HitLighting pass
re-shades the SAME hit **without re-traversing the BVH**, and only for cache-miss pixels
(`!Result.bIsRadianceCompleted`).

**Two approaches for our engine (a genuine fork):**
- **(a) Reference-faithful — RT pipeline + SBT.** Route reflection rays through our RT pipeline (M5/M7,
  Metal Shader Converter) with per-material closest-hit shaders; geometry/material fetch is automatic per
  hit-group. Scales to many materials, but needs SBT management across ~448 meshes + material hit shaders,
  and leans on the Metal RT-pipeline path (less battle-tested than our inline RayQuery).
- **(b) Inline + consolidated geometry.** Keep the inline compute reflection and fetch geometry/material
  manually: pack all content verts→1 storage buffer, indices→1, + a per-instance offset/material table
  (vgeo "4 buffers" precedent, [[dreamcoast-vgeo-metal-atomic64]]), dodging the 64-slot per-primitive
  bindless overflow. At a hit: `InstanceID → offset → fetch 3 verts by PrimitiveIndex → barycentric
  normal/UV → material tex → sun (HW shadow ray) + IBL`. Fits our current inline architecture (no SBT),
  but is a deviation from how the reference does it.

**Reusable now:** the reference's two-pass **bookmark** structure maps directly onto ours — keep B
(SurfaceCache) as pass 1, run E (HitLighting) only on cache-miss pixels as pass 2. Deferred as its own
phase (either approach is multi-part) rather than rushed, since A→D already deliver the visible wins.

## Roadmap

A (content accel) ✔ → B (HWRT trace + surface-cache shade) ✔ → B.2 (screen-color-at-hit) ✔ → C
(full-res reflection mode) ✔ → D (HWRT GI + cache shading) ✔ → E (Hit Lighting, inline + consolidated
geometry) ✔. **All phases complete.** Follow-ups: DX≡VK Windows parity (Metal-only line); a Hit
Lighting perf pass (it re-shades every off-screen pixel — the reference's cache-miss-only gating is a
possible optimization); wiring the consolidated geometry into a content HW path tracer.

## Gates

Gallery `af70c1a5` byte-identical (off). Path-tracer parity: our HW path tracer is ground truth;
HWRT reflections should converge toward it. Determinism (RayQuery deterministic) + DX≡VK follow-up
(Metal-verified here). Perf: one-time BLAS build + per-frame HW-trace ms via `PROFILE_GPU`.
