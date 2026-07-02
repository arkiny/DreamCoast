# Lossless (image-identical) optimization plans — Sponza 1080p/0.667 Med → 60fps both backends

Goal: DX **and** VK ≥60fps at 1920×1080 / 0.6667 internal + TAAU, RENDER_QUALITY=med,
**MAXIMUM quality — image-identical** (no quality-reducing scalability knobs). Identical DX/VK
settings. Baseline (max quality, no knobs): **DX 25ms (40fps), VK 32ms (31fps)**. Dominant cost:
`gdf_reflect` 10–12ms, `sdf_cache_light` 7–9ms, `gdf_ao` 3–4ms, `gi_volume` 2–5ms.

Verification golden: max-q Med, fixed EXPOSURE=8 (auto-exposure isolated), 200-frame warmup;
DX≡VK baseline stochastic gap = **0.089/ch** (this is the image-identical tolerance floor).
`scratchpad/measure.py` for perf; `tools/rt-compare.py` for image diff.

These four plans came from parallel read-only R&D design agents (2026-07-03). Each is a real
feature needing careful image-identical + DX≡VK + gallery-anchor verification. **Not yet
implemented** — this doc preserves the designs for focused implementation.

---

## 1. Adaptive / temporally-amortized `gdf_reflect` (biggest lever: 10–12ms)

Cost is ~100% in the per-pixel GGX sphere-march + per-hit 48-step shadow march; `reflect_temporal`
already holds a converged, reprojectable per-pixel history but the trace re-runs every pixel every
frame. **No `dispatch_indirect` in the RHI** (only `draw_indexed_indirect`) → ray-compaction is out;
use an **in-shader early-out mask** instead.

- **Algorithm (in `gdf_reflect.slang` csMain, before the march at :243):** reproject world_pos into
  prev frame; read the temporal `refl_accum`/`refl_pos` history; validity test (identical to
  `reflect_temporal.slang:65,88`: `pos.w>0.5 && dist<reject_dist && puv∈[0,1] && hist_len≥converged`).
  Disoccluded/new/short-history → **must trace** (graceful degradation, never stale). Converged+clean
  → **skip the march, write a=0 sentinel, reuse history**. A **staggered 1/K full-trace floor**
  (`(lin+frame)%K==0`) re-traces a rotating subset so nothing goes stale (K≤8, ≤ cache relight period).
- **`reflect_temporal.slang`:** on `a<=0.5` (skipped) carry reprojected history unchanged (no EMA
  blend); on `a==1` blend as today. Image-identical because a converged pixel's reused value = the EMA
  fixpoint; disoccluded always traced; staggered floor bounds staleness.
- **Expected:** gdf_reflect 10.3/12.4 → **~2/2.7ms** best case (static camera); degrades gracefully to
  baseline+~0.3ms on fast motion (never stale).
- **RISK — push-layout change:** `gdf_reflect_push` is 240B fully packed (cone_k at +236 in a float4.w).
  Adding `prev_view_proj` (64B) + scalars must go on 16-aligned rows (offset 256+), re-verify DXIL/
  SPIR-V/MSL packing (the classic float3-pad parity bug). Y-flip reproject must mirror
  `reflect_temporal.slang:43,211` exactly. Gallery forced K=1 (byte-identical anchor).
- Files: gdf_reflect.slang, reflect_temporal.slang, reflect.rs, push.rs, quality.rs, main.rs.

## 2. SDF-march acceleration — cheaper per-step field eval (shared by ALL marches)

**Correction to the premise:** every sphere-march already steps by the SDF distance (`t += max(d,floor)`),
so empty space is NOT traversed step-by-step — a "skip empty in one big step" occupancy is a no-op
(this is why `cone_k` gave only −8%). The real cost is the **per-step SAMPLE cost**: each `ms_geo`
(`mesh_sdf_sample.slang:77`) reloads a 112B header (7×Load4) + dense tap + a per-instance candidate
loop (Load4×5 + transform + AABB + atlas SampleLevel) — **every step**.

- **Accelerant 1 — coarse conservative min-distance mip (Lumen page-validity/mip split):** bake (cooked,
  CPU, in `compose.rs`) a low-res R32F volume where each coarse voxel = `min(true dist over region) −
  half coarse-diagonal` (a conservative SDF: `coarse(p) ≤ true(p)`, safe sphere-trace step). In `ms_geo`:
  `d_coarse = coarse_mip(p); if d_coarse>BAND return min(d_coarse,clamp)` (one cheap tap, skip dense+
  candidate loop) `else ms_geo_exact(...)` (legacy near-band). **Image-identical**: far steps can't hit
  (`true ≥ d_coarse−halfdiag > HIT_EPS`) and never overstep; near band runs exact code → identical hit.
  Content-gated (sentinel `coarse_mip_idx==0xFFFFFFFF` for gallery = legacy exact = byte-identical anchor).
- **Accelerant 2 — hoist `ms_load_header` out of the per-step march (bit-identical CSE, zero risk):**
  the 112B header is invariant per dispatch but re-read every step. Load once per thread, pass a
  `MeshSdfHeader` into an `ms_geo_h(h,...)` variant. Sponza uses `count==0` direct-sample
  (`clipmap.slang:47` delegates to a single `ms_geo(c.desc,...)`) → header is constant per thread, hoist
  is clean. Benefits reflect/ao/gi/cache/shadow. **Both sdf-accel AND ao-cache agents independently
  flagged this as the top safe win.**
- **Expected:** large per-step-cost cut on the far/mid steps (majority of reflection/shadow rays);
  single-digit ms on DX, proportionally more on VK (binding). Must confirm with PROFILE_GPU.
- Files: mesh_sdf_sample.slang (ms_geo two-tier + header hoist), clipmap.slang (delegation), gdf.rs
  (upload cooked mip + MeshSdfHeader field), compose.rs (bake mip reduction), quality.rs (gallery seam).

## 3. `sdf_cache_light` static-convergence dirty-skip + `gdf_ao` in-march CSE

- **Cache dirty-skip (image-identical, unlike period-amortization):** EMA fixpoint — if fresh `R==prev`
  (converged), `lerp(prev,R,α)==prev` for any α, so skipping relight and carrying `prev` is exact. Can't
  test `R==prev` per-frame (stochastic gather), so gate on an **invariant precondition**: a card is
  skippable iff (geometry static — captured once) AND (lighting static — a global `lighting_epoch`
  reusing the GI-denoise sun key at main.rs:~4443, bumped on sun/sky change → clears settled) AND
  (multibounce: MVP = global "whole cache converged for N frames" AND-reduction latch; per-card
  reachability is a later refinement). Settled cards skip the ~100-march relight (keep the cheap carry-
  forward copy first = trivially lossless; no-copy single-buffer is the bigger but subtler follow-up).
  **Expected:** static Sponza cache ~7ms → <1ms after warmup; moving-sun bumps epoch → no skip = no
  free lunch, no image change. Gallery (period==1) forces settled=0 sentinel = every-frame = anchor safe.
- **`gdf_ao` — NO lossless dedup with GI** (volume sky-vis ≠ the 5-tap 0.5m contact march; honest). The
  wins are in-march: (a) header hoist (same as #2), (b) exact ground-plane short-circuit in `scene_occ`
  (VERIFY the algebra: `scene_occ=min(geo_inside,ground)`; only skip the SDF tap where it provably can't
  lower the min below `h` — the agent's claim needs a correctness re-check before trusting). Expected
  gdf_ao 3.2 → ~2–2.5ms.
- Files: sdf_cache_light.slang, gdf.rs (card_settled/epoch buffers + a Y-flip-free settle-reduction
  pass like sdf_cache_visibility), mesh_sdf_sample.slang (header hoist), gdf_ao.slang, main.rs (epoch).

## 4. Async-compute overlap architecture — (async-arch agent, see its transcript)

The existing async cache (`P_ASYNC_CACHE`, default-on D3D12 content) helps DX but HURTS VK at max
quality because the GDF graphics-queue compute is already SM-saturated → the async cache only contends,
no idle time to fill. The lever is to overlap the view-independent work into the **raster window**
(gbuffer+shadow ~1.7ms, which leaves compute units idle) via: (2a) fork the graphics cmd buffer into a
raster submission + a lighting submission and issue the compute-queue relight submit RIGHT AFTER the
raster submission (not after the whole frame is recorded — the current coarse whole-frame-vs-whole-frame
race is why VK's scheduler gets no overlap); (2b) chunk the relight dispatch so the scheduler can find
the idle pockets; (2c) lower the VK compute queue `queuePriorities` to 0.5 (a real VK-only lever; D3D12
has no usable background tier for COMPUTE queues). No timeline semaphores needed (same-queue FIFO order).
**HONEST CEILING:** at max quality the GDF compute is SM-saturated (RTX 2070S async queues draw from the
same SM pool), so only the ~1.7ms raster window is truly idle — expect VK ~1–1.5ms recovery, DX flat
(already fine). Stage 6's +33% VK was at RENDER_SCALE=0.5 where real idle time existed; that headroom is
gone at med/0.6667. Image-identical (only reorders submission + chunks dispatch, per-texel math unchanged).
Files: main.rs (submit split ~6362), gdf.rs (chunk cache_light_push), rhi-vulkan/device.rs (queue prio),
rhi-d3d12/device.rs. gi_volume-async is a separate opt-in follow-up (own 1-frame-latency proof).

## Combined potential (best case, static/slow camera; degrades to baseline on fast motion)
adaptive reflect −8ms + cache dirty-skip −6ms + SDF march accel (several ms across passes) + async ~1.5ms
⇒ plausibly DX 25→~12ms, VK 32→~16–18ms at ZERO quality loss. The lossless path can realistically reach
~60fps on a static/slow camera at max quality; fast motion falls back to full-quality baseline (never
stale, never wrong). This validates the "real optimization, max quality" direction.

---

## Recommended implementation order (safest/highest-certainty first)
1. **#2 Accelerant 2 (header hoist)** — bit-identical, broad, zero risk. Prove the pattern + measure.
2. **#2 Accelerant 1 (coarse mip)** — the main per-step-cost win, image-identical, content-gated.
3. **#3 cache dirty-skip** — big on static scenes; land the copy-preserving lossless MVP first.
4. **#1 adaptive reflect** — biggest single-pass win but push-layout parity risk; verify carefully.
5. **#4 async raster-overlap** — bounded by idle time; the VK-specific lever.

Each gate: image-identical vs the max-q golden (within 0.089/ch), DX≡VK ≤ the stochastic gap, gallery
byte-identical anchor (0.000/ch), PROFILE_GPU before/after both backends, clippy -D warnings, VK
validation clean. Stack them; re-measure toward the ≥60fps-both goal after each.
