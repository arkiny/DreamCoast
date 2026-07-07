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

## Remaining — Phase E: Hit Lighting (deferred, large)

Full material + shadow-ray evaluation at the HW hit (reference `HitLighting` / `HitLightingForReflections`).
Value: sharp/accurate **off-screen** reflections (on-screen is already sharp via B.2/C; off-screen
currently uses the surface cache). Requires new infrastructure that the SurfaceCache path deliberately
avoids: **consolidated per-instance geometry + material buffers** — pack all content vertices into ONE
storage buffer + all indices into ONE + a per-instance offset/material table (the vgeo "4 buffers"
precedent, [[dreamcoast-vgeo-metal-atomic64]]), dodging the 64-slot per-primitive bindless overflow.
At a HW hit: `InstanceID → offset table → fetch 3 verts by PrimitiveIndex → barycentric-interpolate
normal/UV → sample the material texture (bindless) → sun (HW shadow ray) + IBL`. This is a separate
multi-part feature (Rust consolidated-buffer build + shader fetch/interp/material/shadow); deferred as
its own phase rather than rushed, since A→D already deliver the visible reflection/GI wins.

## Roadmap

A (content accel) ✔ → B (HWRT trace + surface-cache shade) ✔ → B.2 (screen-color-at-hit — the sharp
win) ✔ → C (full-res reflection mode) ✔ → D (HWRT GI + cache shading) ✔ → **E (Hit Lighting —
deferred, needs consolidated geometry)**.

## Gates

Gallery `af70c1a5` byte-identical (off). Path-tracer parity: our HW path tracer is ground truth;
HWRT reflections should converge toward it. Determinism (RayQuery deterministic) + DX≡VK follow-up
(Metal-verified here). Perf: one-time BLAS build + per-frame HW-trace ms via `PROFILE_GPU`.
