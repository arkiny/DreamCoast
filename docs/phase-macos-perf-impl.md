# macOS/M3 Perf — Implementation Plan (team execution)

> Companion to [macos-perf-plan.md](macos-perf-plan.md) (strategy). This doc holds the
> **measured M0 baseline**, the concrete per-stage work, expected wins, gates, and the team
> split. Target: **Sponza Med 60fps (≤16.6 ms) on MacBook Air M3**. Measurement-first.

## M0 baseline — MEASURED (this M3 box, 2026-07-01)

Command:
```
PROFILE_GPU=1 LEVEL=sponza RENDER_QUALITY=med WARMUP_FRAMES=90 \
  ./target/release/sandbox --backend metal --screenshot-clean out.png
```
Output: **2560×1440 native** (`render_scale=1.0`), steady-state **total ≈ 165 ms/frame**
(≈ 6 fps). **~10× over the 16.6 ms budget.** Per-pass (steady state, top-down):

| Pass                 | ms    | share | note |
|----------------------|-------|-------|------|
| **gdf_reflect**      | 77.8  | 47%   | GGX SW-RT reflection march, `reflect_max_steps=96`. Dominates. |
| **gdf_ao**           | 25.9  | 16%   | far-field GDF ambient occlusion |
| **ssao**             | 13.1  | 8%    | near-field HBAO-lite — **independent 2nd AO pass**, both on for content |
| reflect_temporal     | 10.1  | 6%    | reflection temporal resolve (full-res) |
| gdf_atrous ×2        | 10.4  | 6%    | GI à-trous denoise (2 taps) |
| ssr                  | 7.7   | 5%    | screen-mirror SSR |
| sdf_cache_light      | 5.6   | 3%    | surface-cache amortized relight (already period=40) |
| gdf_temporal         | 3.6   | 2%    | GI temporal |
| gbuffer              | 1.5   | 1%    | |
| lighting/reflect_composite/gi_upsample/tonemap/shadow | ~7 | 4% | fullscreen + misc |

**Diagnosis.** The bill is (a) **near-native QHD on an iGPU** — every SW-RT compute pass is
per-pixel over 3.7 M pixels — and (b) the **SW-RT reflection+AO stack** (`gdf_reflect` +
`gdf_ao` + `ssao` + `ssr` + reflect_temporal ≈ **135 ms, 82%** of frame). `render_scale=1.0`
is the single biggest lever; the reflection march `reflect_max_steps=96` is the single
hottest kernel. Note `reflect_half_res` is already `true` in Med, so 78 ms is *already the
half-res path* — the per-ray march cost is the real culprit, not just pixel count.

**Honest scope note.** A 10× cut to hit 60 fps at QHD-native output with the full SW-RT GI
stack is aggressive. The realistic path is **internal `render_scale ≤ 0.67` + aggressive
per-ray knobs + a reduced Apple-tier feature set** (e.g. single AO pass, shorter marches),
measured stage by stage. If 60 fps proves infeasible without unacceptable quality loss, we
report the achieved fps + the quality/fps curve and pick a defensible Apple default tier —
we do **not** silently degrade the anchor or fake the number.

## Expected-win model (to be validated per stage — never trust the model over PROFILE_GPU)

Per-pixel passes scale ≈ with internal pixel count; per-ray-march passes scale ≈ with steps.
- **render_scale 1.0 → 0.67** (~0.44× pixels): 165 → ~90 ms (gdf_reflect ~34, gdf_ao ~11,
  ssao ~6, ssr ~3.4, reflect_temporal ~4.4, atrous ~4.6; cache/temporal partly fixed-cost).
- **+ drop ssao on Apple tier** (gdf_ao already gives contact AO): −6 ms → ~84 ms.
- **+ reflect_max_steps 96 → 48, cone_k↑**: gdf_reflect ~34 → ~18 ms → ~68 ms.
- **+ gi_res_div 3 → 4, gi_max_steps↓**: GI trace + upsample cut → ~62 ms.
- **+ render_scale 0.67 → 0.5** if still short (~0.25× pixels vs native): pushes toward ~40 ms.
- **Memoryless (B1)** helps the **fullscreen bandwidth** (gbuffer/lighting/tonemap/depth) on
  TBDR; the hot passes are ALU-bound compute so expect a smaller but real slice + thermal
  headroom. Measure — do not assume.

The model says even A-axis alone likely lands ~30–40 ms (~25–30 fps). Hitting 60 fps almost
certainly needs A **and** C-axis march cuts **and** possibly `render_scale=0.5`. Stage gates
decide.

## RESULTS (measured, this M3 box, 2026-07-01)

- **M1 Apple tier — VERIFIED WIN. Sponza 165 → ~70 ms (2.3×, −57%).** Apple tier auto-selected
  (`GPU "Apple M3"`), internal ~1707×960 (render_scale 0.67) → 2560×1440 via TAAU. New per-pass:
  gdf_reflect 30.5 (was 77.8), gdf_ao 11.6 (was 25.9), **ssao 0 (dropped, was 13.1)**,
  reflect_temporal 4.6, sdf_cache_light 4.6, gdf_atrous×2 4.8, taau 3.3 (new upscale cost).
  **Gallery anchor `af70c1a5…` byte-identical + run-to-run deterministic; clippy -D clean.**
  Branch `feature/macos-perf-m1-apple-tier` (commit 859f79f). `gdf_reflect` still 44% → next target.
  NOTE: needed two gallery-force fixes beyond the agent draft — `render_scale` and
  `reflect_max_roughness` were not gallery-gated, which broke the anchor until forced legacy
  (`if gallery_scene { 1.0 / 0.5 }`), same pattern as every other tier knob.
- **M2 Memoryless — CORRECT BUT ZERO MEASURED WIN (honest negative).** Sponza 164 ≈ 165 ms;
  output byte-identical to M0 (gallery `af70c1a5…` and Sponza `1ee08a3a…`), deterministic, VK/DX
  untouched. Root cause: the eligibility criterion (a color transient **no pass reads**) is
  essentially empty on this deferred+SW-RT pipeline — nearly every transient (gbuffer, GI, reflect
  chains) is sampled by a later pass. The real TBDR payoff is the **G-buffer + depth** being
  tile-resident, but they are sampled cross-pass by the full-screen lighting/SW-RT passes, so they
  cannot be memoryless without **tile-shading / programmable blending (B3)** — a large refactor.
  The M2 plumbing (RenderTargetDesc.memoryless + graph lifetime derivation + Metal storage-mode +
  DontCare store) is correct **reusable foundation for B3** but pays nothing standalone.
  Branch `feature/macos-perf-m2-memoryless` (uncommitted; decision pending — defer to B3 vs land inert).

**60fps gap:** M1 lands ~70 ms (~14 fps). To reach 16.6 ms needs ~another 4.2× — realistically
M3 (C-axis: cut gdf_reflect 30 ms + gdf_ao 11.6 ms) **and** render_scale 0.5 **and/or** the B3
tile-shading that unlocks the real memoryless win. 60fps@QHD-native output with the full SW-RT
stack may require accepting render_scale 0.5 as the Apple default. Decision needed.

## DEEPER ANALYSIS (measured knob sweeps, Apple tier, Sponza/M3, 2026-07-01)

Serialized `PROFILE_GPU` sweeps to find what actually moves the needle:

| Config | total ms | gdf_reflect | note |
|--------|---------:|------------:|------|
| Apple baseline (rs0.67, reflect half-res) | ~69 | ~31 | |
| + render_scale 0.5 | **~42** | 17 | render_scale is THE lever |
| + render_scale 0.45 | ~37 | 14 | |
| reflect_max_steps 56→32 | ~69 | ~31 | **no effect — not step-bound** |
| reflect_max_steps 56→16 | ~65 | ~27 | barely |
| reflect_max_roughness 0.4→0.15 | ~69 | ~31 | **no effect — pass runs regardless** |
| gi_res_div 4→6 | ~43 | — | no effect (GI trace already 0.16ms) |
| gdf_ao OFF | ~57 | 31 | saves ~12ms but loses ALL AO (ssao already off on Apple) |
| **reflect_half_res=0 (full-res)** | **~158** | **120** | reflection already half-res saving ~4× |
| **P11_LEGACY_IBL (drop SW-RT ambient), rs0.67** | **~18** | — | 55fps |
| **P11_LEGACY_IBL + rs0.5** | **~12.4** | — | **80fps** |
| P11_LEGACY_IBL + rs0.5 + gdf_ao off | ~5.4 | — | |

**Conclusions (measured, not modeled):**
1. **`gdf_reflect` (~40% of frame) responds to NOTHING but resolution** — not march steps, not the
   roughness gate, not the enable flag (it's capability-forced on). It is a per-pixel SW-RT GGX
   reflection, already half-res (full-res = 120ms). The only quality-preserving lever left is a
   **quarter-res reflection trace + bilateral upsample** (a new `reflect_res_div`, mirroring the
   existing `gi_res_div` path) — the reflection is temporally accumulated + low-frequency, so it
   should tolerate it. Estimated ~half of gdf_reflect (~8ms @ rs0.5). This is the one real
   algorithmic M3-C task worth doing.
2. **The entire 60fps overage IS the SW-RT reflect+GI+AO stack.** `P11_LEGACY_IBL` (the cheap
   captured-cube ambient) hits 60fps trivially (12–18ms). So 60fps @ Sponza on M3 is a
   **quality-vs-framerate** decision, not a tuning problem.
3. **60fps with full SW-RT GI is NOT reachable by knobs alone.** Floor with everything aggressive
   (rs0.45 + quarter-res reflect + cheaper AO) is ~27–30ms ≈ **33–37 fps**. Reaching 16.6ms
   requires falling back to the cheap IBL ambient (dropping the engine's signature SW-RT GI) or a
   much lower internal resolution.

**Recommendation (aligns with the build-to-quality directive — don't drop correct GI for fps):**
Ship the Apple tier at render_scale 0.5 + a new quarter-res reflection path (M3-C) → target the
best achievable **with** SW-RT GI (~30–37 fps), and expose an optional lower "performance"
sub-tier that switches to `LEGACY_IBL` for users who need 60fps at the ambient-quality cost.
Do **not** claim 60fps+SW-RT — the measurement says it isn't there.

## M3-C RESULT — quarter-res reflection (`reflect_res_div`), VERIFIED (2026-07-01)

New `reflect_res_div` knob (mirrors `gi_res_div`): trace `gdf_reflect` at `1/div` + the existing
bilateral upsample. `div=2` reproduces the legacy half-res byte-for-byte (`cw.div_ceil(2)`); every
non-Apple tier keeps 2; **Apple tier = 4 (quarter-res)**. `P_REFLECT_RES_DIV` override.

**Perf ladder (Apple tier, Sponza/M3):**
| Stage | total ms | fps | gdf_reflect |
|-------|---------:|----:|------------:|
| M0 Med native | 165 | 6 | 77.8 |
| M1 Apple (rs0.67, half reflect) | 70 | 14 | 30.8 |
| **+ quarter-res reflect (rs0.67)** | **48.3** | **21** | **8.25** |
| **+ half-res AO (rs0.67)** | **38.8** | **26** | gdf_ao 12.1→3.56 |
| **+ SSR-stochastic + reflect 1/6 + AO 1/4** | **32.9** | **30** | gdf_reflect →~4, ssr →0.8 |
| + all + rs0.5 | ~23 | ~43 | — |

**Apple-tier further tuning (all byte-identical on the Sponza benchmark, sha `546cb91759…`):**
`ssr_stochastic=true` (half-res GGX + ratio-estimator denoise, ~1/3 the full-mirror cost),
`reflect_res_div 4→6` (gdf_reflect scales with trace res only; roughness-gated so free on rough
content — smooth mirrors read softer, `P_REFLECT_RES_DIV=4` to restore), `ao_res_div 2→4` (AO is
very low-frequency; a standard downsample). 39.9→32.9ms (26→30fps). Gallery `af70c1a5` + Med Sponza
`1ee08a3a` unchanged (Apple-only), deterministic, clippy -D clean. Moving-camera notes: the cache
relight period stays 64 (128 gives ~2s relight lag for only −1.8ms — rejected as too aggressive for
a quality default); stochastic SSR is temporally denoised under motion.

**Half-res AO (`ao_res_div`, Apple=2):** trace `gdf_ao` at 1/div + the same bilateral upsample.
Non-Apple tiers = 1 (full-res, byte-identical); the gallery never runs gdf_ao so the anchor is
unaffected. Measured: gdf_ao 12.1→3.56ms; **half-vs-full AO on Sponza is byte-identical** (AO is a
low-frequency contact term the guided upsample reconstructs within 8-bit precision) — a free win.
`P_AO_RES_DIV` override. Med Sponza `1ee08a3a` byte-identical (div=1 no-op), gallery `af70c1a5`.

**Gates PASS:** gallery anchor `af70c1a5` byte-identical + deterministic (gallery forces reflect
full-res); **Med Sponza `1ee08a3a` byte-identical → `div=2` is a proven no-op**; clippy -D clean.
**Quality:** on Sponza (rough-dominant) div=1/2/4 are byte-identical — reflection is roughness-gated
(`reflect_max_roughness=0.4`) so quarter-res is free there. On a smooth mirror (gallery copper
sphere) div=4-vs-full = mean 0.092/ch (max 124), ~2× the half-res error but temporally stable.
So quarter-res is a defensible low-power-tier default (rough content free, smooth mirrors softer),
opt-in and env-overridable. Cross-backend (DX≡VK) parity of div>2 is a Windows follow-up.

**60fps status:** best-with-SW-RT is now ~30ms @ rs0.5 (~34 fps). Reaching 16.6ms still needs
render_scale ≲0.45 + cheaper AO (gdf_ao is now the top pass) or the `LEGACY_IBL` performance
sub-tier — the quality-vs-fps decision stands.

## Stages, owners, gates

Order is by ROI and file-conflict avoidance. **All GPU measurement is serialized on this one
M3 box** (parallel runs corrupt timings) — implementation happens in parallel worktrees, the
lead runs every before/after measurement and the gate.

### M1 — Apple platform tier (axis A + the ssao redundancy). Highest ROI, shader-unchanged.
- Plumb Metal device identity (`name` / `hasUnifiedMemory` / `isLowPower`) from
  `crates/rhi-metal` up through the `rhi` facade + `rhi-types` so
  `quality::RenderQuality::platform_default()` can detect Apple GPUs (replace the honest
  `Med` stub — see quality.rs:61).
- Add an **Apple default tier** (new `platform_default` branch, not a 4th public enum unless
  needed): `render_scale ≈ 0.67`, `ssao` off (gdf_ao covers contact AO), `reflect_max_steps`
  ↓, `cone_k` ↑, `gi_res_div = 4`, `reflect_max_roughness` ↓, aggressive `relight_period`.
  Keep all as *tier defaults* — every `P11_*`/`RENDER_SCALE`/`SSAO` env override still wins
  (the seam). `RENDER_QUALITY=med` explicit must stay = today's Med (no-reg).
- Files: `apps/sandbox/src/quality.rs`, `main.rs` wiring (device→tier), `crates/rhi-metal`
  (expose name), `crates/rhi` + `crates/rhi-types` (facade passthrough).
- **Gate:** measured M3 fps ↑ (report ms) → gallery anchor **byte-identical** (`af70c1a5…`;
  the gallery is `gallery_scene` so every tier knob is already force-legacy at the call site —
  verify) → determinism (run-to-run bit-identical) → content visual sanity →
  `cargo fmt` + `clippy -D warnings` → `tools/golden-image.py` no-reg. Depends on **M0**.

### M2 — Memoryless transient targets (axis B1). Independent of M1's files.
- Add `MTLStorageModeMemoryless` support in `crates/rhi-metal` texture allocation for
  render-graph transients that are written+consumed within the tile and **never sampled
  outside their producing pass**. Wire through the render-graph transient-aliasing lifetime
  (`crates/render`) — add a "memoryless-eligible" derivation (last-read == same pass, no CPU
  readback, no cross-pass sample) rather than a hand-list.
- Apple-only behind the backend seam; **must not touch VK/D3D12 output**. Depth + any
  scratch MRT that the deferred lighting reads must **stay** non-memoryless (lighting samples
  gbuffer + position). Candidates: purely-transient scratch targets; verify each.
- Files: `crates/rhi-metal` (alloc + storage-mode plumb), `crates/render` (lifetime→flag),
  `crates/rhi-types` if a descriptor flag is needed.
- **Gate:** PROFILE_GPU bandwidth/thermal delta reported → gallery byte-identical →
  determinism → fmt/clippy → golden runner. Can start in parallel with M1 (different files);
  **its measurement is serialized after M1's**.

### M3 — Cache amortize + visibility feedback (axis C1/C2). Depends on M1 (shares quality.rs).
- Re-measure `P11_CACHE_RELIGHT_PERIOD` and `card_vis` off-screen-card feedback on M3 (the
  Windows demo shelved it for low ROI; Sponza-on-M3 has different on/off-screen card mix).
  Tune the Apple tier's `cache_relight_period` / `cache_relight_spp` from data.
- **Gate:** 60 fps hold or best-achievable + moving-camera lag acceptable + gallery
  byte-identical + determinism + golden.

### M4 (follow-up, not this wave) — tile-shading deferred (B3), SIMD-32 (B2), F4 gather (C3).

## Team protocol
- Each implementer works in its **own git worktree** off `main`, commits its stage as one
  reviewed change, and hands the branch to the lead for the GPU gate. No agent runs a perf
  measurement concurrently with another (single GPU).
- **Do not** change the gallery anchor. **Do not** ship an Apple-only path that alters VK/D3D12.
- Report every before/after in **ms from PROFILE_GPU**, not estimates. If a knob doesn't pay
  off on the M3, drop it and say so (perf-track P5 SIMD lesson: measured negatives are results).
- No trademark names in code/docs/commits (describe techniques generically).
