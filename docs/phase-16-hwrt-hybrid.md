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

- **Phase B.2 — screen-color-at-hit** (reference `SampleSceneColorAtHit`): project the HW hit to the
  previous frame's screen and sample the **lit HDR history** (sharp, full-res) when the reprojected
  depth validates; fall back to the surface cache when off-screen. HWRT supplies the validated hit
  that our SSR lacked (the reason content mirrors skip SSR), so this is the natural sharp-reflection
  path. Needs the reflection pass to bind the globals UBO (for `prev_view_proj`) like SSR does — the
  push can't hold another matrix (240 B already).
- **Phase D — Hit Lighting**: full material + shadow-ray evaluation at the hit (highest quality),
  needs consolidated per-instance geometry/material buffers to dodge the 64-slot bindless overflow.

## Roadmap

A (content accel) ✔ → B (HWRT trace + surface-cache shade) ✔ → **B.2 (screen-color-at-hit — the
sharp win)** → C (HWRT GI, extend `P_HWRT_GI` to content + cache shading) → D (Hit Lighting).

## Gates

Gallery `af70c1a5` byte-identical (off). Path-tracer parity: our HW path tracer is ground truth;
HWRT reflections should converge toward it. Determinism (RayQuery deterministic) + DX≡VK follow-up
(Metal-verified here). Perf: one-time BLAS build + per-frame HW-trace ms via `PROFILE_GPU`.
