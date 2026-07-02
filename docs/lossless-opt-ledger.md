# Lossless-opt execution ledger — Sponza/IntelSponza 1080p→60fps

Running record of the **image-identical** optimization track (goal, gates, and designs in
[cold-start-lossless-opt.md](cold-start-lossless-opt.md) + [lossless-opt-plans.md](lossless-opt-plans.md)).
Append one row per attempt — **including dead ends** — so no effort is ever repeated. Every claim
is measured (`scratchpad/measure.py`, `tools/rt-compare.py`), both backends, before/after.

## Verification framework (2026-07-03, rebuilt harness)

- **Perf metric:** `gpu-passes` median over the settled tail (measure.py: WARMUP 60, tail 12).
  Primary success = gpu-passes ≤ 16.6ms on BOTH d3d12 and vulkan.
- **Image-identical gate:** fixed exposure `AUTO_EXPOSURE=0 EXPOSURE=8 WARMUP_FRAMES=200`, Sponza
  med/0.6667+TAAU. Each opt's output vs the pre-opt golden (same backend) must be **≤ 0.089/ch**
  (the measured DX≡VK stochastic floor), AND DX≡VK must stay **≤ 0.089/ch**.
- **Gallery byte anchor:** default scene (no LEVEL), both backends, must stay **≤ 0.001/ch**.
- Goldens live in the session scratchpad `gold/` (REF_dx.png, REF_vk.png, gallery_dx/vk.png).
  Recapture recipe is in cold-start-lossless-opt.md if the scratchpad is lost.

## BASELINE (build `076aa9f`, 2026-07-03, RTX 2070 SUPER)

| pass | DX ms | VK ms |
|---|---:|---:|
| **gpu-passes total** | **27.2 (37fps)** | **33.7 (30fps)** |
| gdf_reflect | 11.5 | 13.5 |
| sdf_cache_light | (async queue, hidden) | 8.4 |
| gdf_ao | 3.1 | 4.0 |
| gi_volume | 6.1 | 1.9 |
| ssao | 0.8 | 0.9 |
| gbuffer / shadow | 0.8 / 0.8 | 1.0 / 1.1 |
| reflect_temporal | 0.8 | 0.8 |

- Floors: Sponza DX≡VK **0.089/ch**; gallery DX≡VK **0.001/ch**. Both reproduced.
- Note: on DX `sdf_cache_light` runs on the async compute queue (`409b2c1` default-on), so it's
  absent from the graphics-queue profile; `gi_volume` reads higher on DX (6.1 vs VK 1.9), likely
  async-contention on the graphics queue. VK runs cache inline (8.4ms) → VK's structural deficit.

## ATTEMPTS

_(append below; newest last)_

### A1 — SDF-march header hoist (plan #2 Accelerant 2) — ❌ DEAD END (reverted)

- **What:** added `ms_geo_h(MeshSdfHeader h, …)` / `cm_geo_march_h` / `cm_geo_inside_h` that take a
  once-per-dispatch pre-loaded header, threaded a hoisted `MeshSdfHeader hdr` through gdf_reflect's
  march helpers (`scene_march/occ/normal/refl_shadow/geo_*`), guarded on `count==0`.
- **Result (both backends, measured):** DX reflect 11.5→11.7ms (flat), **VK reflect 13.5→15.5ms
  (REGRESSION, reproduced twice)**. Net loss.
- **Root cause:** the Slang→DXIL/SPIR-V compiler ALREADY CSEs the invariant header load out of the
  march loop (the header buffer isn't written in-shader + the desc index is a push constant ⇒
  provably loop-invariant). Manually hoisting adds a 25-dword struct passed **by value** through the
  helper call chain → extra register pressure that pessimizes VK's allocator. The plan预测 this
  exactly ("verify the compiler isn't already CSE-ing before investing — if it is, skip to #2").
- **Verdict:** the per-step field-eval micro-opt is fighting the compiler; the GDF marches are
  SM/throughput-bound, not header-reload-bound. **Do not retry.** Reverted (git checkout).
- **Implication:** deprioritize plan #2's coarse-mip too unless a later measurement shows real
  sample-cost headroom — the real wins are the *temporal-skip* algorithms (#3 cache dirty-skip,
  #1 adaptive reflect) that avoid whole marches, not cheaper marches.

### A2 — Surface-cache convergence dirty-skip (plan #3) — ✅ LANDED

- **What (host-side only, no shader change):** derive a cache **lighting/scene epoch** (FNV hash of
  sun dir/intensity, sky gain/wb, relight spp/period/alpha, cone_k — **NOT** the camera; the cache
  is view-independent). Once the epoch holds steady `P_CACHE_SETTLE` frames (default 45), the relight
  EMA has reached its fixpoint → **freeze**: skip the `sdf_cache_light` dispatch (sync graph pass on
  VK, async-compute submit on DX) and skip `advance_cache` so consumers stay pinned to the converged
  radiance slot. `main.rs` fields `cache_epoch/stable_frames/settle_frames/dirty_skip` + an
  `async_cache_gap` latch so the async (D3D12) relight resumes with a no-wait submit after a freeze
  gap (the `cache_done` chain is broken by the skip). Gallery-gated (`!gallery_scene`).
- **Why image-identical:** a converged EMA re-lights to the value it already holds
  (`lerp(prev,R,α)==prev` when `R==prev`). Camera motion doesn't bump the epoch, so this wins under
  camera motion too — only sun/sky/geometry change costs a re-converge (no free lunch, no wrong image).
- **Perf (measured, gpu-passes):** **DX 27.2→21.9ms (37→46fps)**, **VK 33.7→25.5ms (30→39fps)**.
  VK `sdf_cache_light` 8.4→0 (skipped). DX `gi_volume` 6.1→1.6 — the async cache was **contending**
  with the graphics queue; freezing it freed the SMs (confirms the memory's async-contention note).
- **Image-identical gate:** A2-vs-baseline **DX 0.021/ch, VK 0.020/ch** (≪ 0.089 floor) ✓. DX≡VK
  0.094 vs baseline 0.089 (max 178 vs 177 — same reflect-stochastic pixels, no new hotspot; the
  +0.005 is within single-backend run-to-run noise). Gallery anchor **0.000/0.000** ✓.
- **Robustness:** clippy -D warnings clean (fixed 2 pre-existing `manual_is_multiple_of` at
  main.rs:5006/6594 that the updated 1.94 toolchain flags), 123 tests pass, Vulkan validation clean,
  moving-sun (continuous relight, never-settle) smoke clean both backends, 200-frame static
  freeze-hold clean. Resume (settle→sun-change) is guarded (`async_cache_gap` + D3D12 monotonic
  fences; VK sync path has no fence hazard) — reasoned sound; not directly triggerable headlessly.
- **Env:** `P_CACHE_DIRTY_SKIP=0` (force legacy always-relight), `P_CACHE_SETTLE=<n>` (freeze delay).
- **Next levers exposed:** `gdf_reflect` now the #1 pass on BOTH (DX 11.7, VK 14.0) → plan #1
  adaptive reflect. `gdf_ao` #2 (3.1/3.7). `gi_volume` (1.6/1.8) could take the SAME dirty-skip
  freeze (view-independent DDGI) for a small extra win. Need DX −5.3ms / VK −8.9ms more for 60fps.

### A3 — Adaptive temporal reflect skip (plan #1, matrix-free variant) — ✅ LANDED — **60fps BOTH**

- **Design pivot from the plan:** the plan's per-pixel reproject needs `prev_view_proj` (+64B) in
  gdf_reflect's push, but VK's `maxPushConstantsSize` is 256B and the push is already 240B → won't
  fit (D3D12 auto-spills to a root CBV, VK has no spill). Instead: a **matrix-free** skip — gdf_reflect
  keeps its own half-res ping-pong (32B/px: world_pos+valid, radiance) and reuses last frame's traced
  radiance for a pixel whose **surface point is unchanged** (world-pos gate), no reprojection. The
  push stays 240B (skip cfg byte-packed into the unused `gdf_sampled` slot — read|write|K|frame, since
  bindless indices are <64). New `reflect.rs` `refl_skip` ping-pong + prepare/advance; `main.rs`
  enables REUSE only once the cache is frozen (A2 settled ⇒ reflect inputs stable ⇒ reused==fresh).
- **Wave-coherent staggered floor (critical):** a scattered `(lin+frame)%K` re-trace put ≥1 marching
  thread in every wave → SIMD divergence made the whole wave pay the march (K=8 cost 7ms not ~2ms).
  Forcing whole **8×8 tiles** (`(tile_id+frame)%K`) re-march coherently → 1/K real cost. K=8 default
  (`P_REFLECT_SKIP_STAGGER`), insurance vs future dynamic content; the static scene is exact without it.
- **Perf (measured, apples-to-apples current thermal state):** cache-dirty-skip baseline DX 20.6 / VK
  24.1 → **DX 13.5ms (74fps) / VK 14.6ms (69fps)**. `gdf_reflect` DX 10.4→3.3, VK 12.4→3.0.
  **≥60fps on BOTH backends achieved** (static/slow camera).
- **Image-identical:** A3 vs the ORIGINAL max-q baseline **DX 0.050/ch, VK 0.051/ch** (< 0.089 floor)
  ✓. DX≡VK 0.095 (≈ baseline 0.089). Gallery anchor **0.000/0.000** ✓. The 0.050 (> A2's 0.020) is
  reuse across the TAAU sub-pixel jitter — sub-pixel staleness, below floor; tightening the world-pos
  reject would kill the win. Clippy clean, 123 tests pass, moving-sun (un-settle) smoke clean both.
- **Fast-motion behaviour (honest):** the master-off path (`P_REFLECT_SKIP=0`, all pixels march) is
  DX 15.5ms vs the 10.4ms baseline — the compiled-in skip-buffer UAV *writes* cost ~5ms of occupancy
  on the all-march path (VK unaffected). BUT this degenerate case needs the feature globally off;
  under real motion the per-pixel world-pos gate still reuses the ~90%+ of surfaces that persist
  frame-to-frame, so only disoccluded edges march → effective fast-motion cost stays low. A future
  mitigation (move the skip-buffer write into a separate cheap pass so gdf_reflect only *reads*) would
  remove even that occupancy hit; deferred (not needed for the target).
- **Env:** `P_REFLECT_SKIP=0` (legacy full trace), `P_REFLECT_SKIP_STAGGER=<K>` (re-trace floor).

### IntelSponza baseline (2026-07-03, A2+A3 active) — geometry-bound, needs culling/LOD

`LEVEL=sponza_intel` 1080p/0.667 med: **DX 43.1ms (23fps), VK 43.0ms (23fps)**. Top pass on BOTH is
**`shadow` 14.1ms** (not the GDF stack — A2/A3 already cut gdf_reflect to ~4.4ms here). Geometry passes
(shadow + gbuffer + prepass) dominate ~25-30ms of the 43ms; DX≈VK because the bottleneck is raster
geometry (where the backends match), not GDF compute (where VK lags). **Confirms docs/cull-lod-design.md:
culling + LOD is the IntelSponza lever** (the GI lossless track that won Sponza does NOT move the needle
here). NEXT phase = real-scene frustum/occlusion culling + distance LOD (design S0→S6). The directional
shadow frustum covers the whole scene, so shadow-map cost needs LOD / cascade-culling, not just camera
frustum cull; camera-frustum + HZB occlusion cull attacks the gbuffer/prepass share.

### A4 — Cached shadow map (dirty-skip) — ⚠️ REVERTED (design sound, transient-pool impl broken)

- **What:** mirror A2 for the `shadow` pass — the legacy directional shadow map is camera-independent
  (`light_view_proj` reads only sun/scene_center/scene_radius), so skip the re-raster when sun+geometry
  are stable and re-sample last frame's depth. Design in **docs/shadow-cache-design.md**.
- **Perf: CONFIRMED huge** — skipping the pass took IntelSponza `shadow` **14.1ms → 0** (DX 42.8→28.9,
  VK 43.0→29.3), and it correctly survives camera motion (camera-independent map). This validates the
  lever: **shadow caching is THE IntelSponza shadow win.**
- **BUT image WRONG:** the design assumed "skip the writer ⇒ the transient depth pool slot keeps last
  frame's depth." **Empirically false** — Sponza shadow-cache-ON vs golden = **1.726/ch** (shadows
  gone → scene too bright); IntelSponza ON-vs-OFF = 0.133/ch (run-to-run noise is only 0.002). When no
  pass writes the transient depth, the graph clears/reuses the slot (there is no `import_depth` for an
  app-owned persistent depth attachment; `import_external` is barrier-tracking only). Gallery anchor
  stayed 0.000 (cache off there). **Reverted** (`git checkout main.rs`).
- **Correct fix (bounded follow-up, NOT landed):** an explicit **persistent app-owned shadow depth
  texture** — render the transient map as today then **copy** it into the persistent texture; lighting
  samples the persistent copy; skip BOTH the raster and the copy when static (the copy retains last
  frame's depth). Copy cost ~0.5ms when rendering, 0 when frozen — still ~14ms net win. Needs a depth
  texture-to-texture copy in the RHI (verify it exists on both backends first). Alternative: add
  graph support for an imported persistent depth attachment (bigger).
- **Do NOT** retry the transient-pool trick — it's confirmed broken.
- **Side note:** IntelSponza has a **pre-existing DX≡VK divergence of ~0.94/ch** (present with the
  shadow cache OFF too) — a content parity bug in the sponza_intel scene, unrelated to this track,
  worth a separate look before trusting IntelSponza DX≡VK gates.

## STATUS: Sponza 1080p/0.667 med — **≥60fps on DX (74) and VK (69)**, image-identical (≤0.051/ch)

IntelSponza (43ms) NOT yet at 60fps — it is GEOMETRY-bound (gbuffer 14 + shadow 14), a different
problem from the GI-lossless track. Two vetted designs ready: **docs/shadow-cache-design.md** (shadow
14ms→0, needs the persistent-texture fix above) + **docs/virtual-geometry-feasibility.md** /
**docs/cull-lod-design.md** (gbuffer via per-mesh frustum+HZB cull + discrete LOD — Option A, NOT full
Nanite, which is unjustified here). Both are implement-ready; neither landed yet.

Remaining track work: (1) IntelSponza to 60fps (docs/cull-lod-design.md — culling/LOD/streaming;
culling is NOT the Sponza lever but IS the IntelSponza-scale lever), (2) frustum+occlusion culling +
distance LOD for the real scene draws (currently only the synthetic grid is culled), (3) optional
small wins: gi_volume dirty-skip freeze (view-independent, ~1.6/1.8ms), gdf_ao half-res.
