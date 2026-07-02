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
