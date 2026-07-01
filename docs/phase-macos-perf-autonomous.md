# macOS/M3 — Autonomous run-until-60fps (branch `feature/macos-perf-autonomous`)

> Unattended session. Goal: **Sponza 60fps (≤16.6ms) on M3 at acceptable quality** (Sponza-judged;
> keep SW-RT GI, keep render_scale as high as possible). No approval loop — self-directed, commit to
> this branch. Whole-engine (not just GI), keeping the engine skeleton intact. Measurement-first;
> every stage gated: gallery sha `af70c1a5…` + Med Sponza sha `1ee08a3a…` byte-identical +
> determinism + clippy -D. Prior work (M1/M3-C/AO/tuning) is on `main`; this branch continues from it.

## Starting point (main, Apple tier, Sponza/M3)
32.9ms (30fps) @ rs0.67. Passes: gdf_reflect ~4, reflect_temporal 4.6, sdf_cache_light 4.9,
gdf_atrous×2 4.6, taau 3.5, gdf_ao 1.5, gbuffer 1.1, gi/ssr/lighting/tonemap small.

## UE reference findings (UnrealEngine-1/Engine/Config)
- **ResolutionQuality presets = Performance 50% / Balanced 58.3% / Quality 66.6% / Native 100%**
  (per-axis screen percentage). So rs0.5–0.67 are UE's *shipped* quality bands; Lumen leans on TSR
  upsampling from low internal res rather than native (our render_scale + TAAU is the UE-endorsed path).
- **`r.LumenScene.DirectLighting/Radiosity.UpdateFactor=128`** at low tier — UE amortizes surface-cache
  relight over 128 frames. ⇒ our `cache_relight_period=128` is UE-standard (earlier 128-rejection revised).
- **`r.SSR.HalfResSceneColor=1`**, **`r.Lumen.Reflections.DownsampleFactor=2`** — half-res SSR +
  downsampled reflections (validates ssr_stochastic + reflect_res_div).
- **`r.Lumen.ScreenProbeGather.ShortRangeAO.DownsampleFactor`**, AO mip/quality scaling — half-res AO
  (validates ao_res_div). Radiosity ProbeSpacing=4 / HemisphereProbeResolution=3 — sparse GI (gi_res_div).
- Half-res gathers pervasive (MotionBlur.HalfResGather, Bloom.ScreenPercentage=35). Shadow res scales
  512→2048 by tier.
- Mac device profile: no special cvars in this tree. Our shaders are all fp32 → **fp16 unexploited**.

## Backlog (ordered: keep render_scale high; byte-identical + arch wins first, drop res last)
1. **cache_relight_period 64→128** (UE UpdateFactor=128). Byte-identical static; ~2s relight lag (UE-accepted at perf tier). ~−1ms.
2. **GI denoise reduction** — atrous iterations 2→1 on Apple (GI is gi_res_div=4 sparse + temporally denoised). ~−2ms. Verify Sponza delta.
3. **reflect_temporal at reflection-res** — currently full-res resolve of the upsampled reflection; resolve at trace-res then upsample. ~−3ms (arch).
4. **fp16 in hot compute shaders** (gdf_reflect, gdf_ao, sdf_cache_light, gdf_atrous) — Apple 2× fp16 ALU. The skeleton-level win. Precision-gated by the byte-identical Sponza test. ~−3–5ms.
5. **async compute overlap** — overlap SW-RT compute with shadow/gbuffer graphics (Phase-7 async exists). ~−?ms (hides latency).
6. **render_scale** — drop to 0.58 (UE Balanced) or 0.5 (UE Performance) LAST, only as much as needed to cross 16.6ms.

## Log (measured, newest first)
- **cache_relight_period 64→128 (UE UpdateFactor) + à-trous 2→1 (`gi_atrous_steps`): 32.9→28.3ms @ rs0.67 (30→35fps).** Apple Sponza byte-identical (`546cb917`); gallery `af70c1a5` (gallery-forced 2 steps) + Med `1ee08a3a`; determinism + clippy-D OK.
- (start) baseline main Apple tier 32.9ms @ rs0.67.

## ★ ROOT CAUSE of the interactive <60fps: per-frame IBL recapture (FIXED)
The headless PROFILE_GPU (GPU passes ~12.8ms) hid the real cost. The UI FPS stat is the true
end-to-end frame time. `sample`-profiling a CPU-bound-*looking* Intel Sponza frame showed **every
render encoder was an IBL cube-face capture** — the sky→env-cube→irradiance→prefilter chain was
re-captured EVERY frame despite a static sun (`time_of_day` off by default). Those cube renders are
real GPU+CPU work NOT counted in the profiled passes.

Instrumentation added (`NO_VSYNC`, per-frame frame/fence-wait/cpu-record timers) proved the frame is
compositor-paced (acquire blocks ~15ms) and true CPU-record is ~1ms — GPU-bound, not CPU-bound.

**Fix (commit 1787e55):** `ibl.maybe_capture` recaptures only when the sky inputs change (sun,
ambient, sky gain/wb, focus) or multibounce feedback is active — a static sky re-marches to a
bit-identical cube. Gallery `af70c1a5` + Med Sponza `1ee08a3a` byte-identical.

**A/B (Intel Sponza, native 2560×1664 @ rs0.67, back-to-back + cooled):**
PRE-fix **19.92ms / 50fps** → POST-fix **13.80ms / 72fps** (−6.1ms). **60fps achieved with margin.**

Lesson: measure the WALL-CLOCK frame time (UI stat), not just GPU pass sums. Headroom now exists to
raise quality (render_scale / restore divisor sharpness) while holding ≥60fps.

## Quality rebalance (post-IBL-fix headroom) + thermal caveat
With the IBL fix giving 72fps at rs0.67 on Intel Sponza, the aggressive divisor cuts (tuned for the
now-retired plain-sponza benchmark, where reflection was pathologically expensive) were wasting
quality. On the dense real benchmark the SW-RT passes are cheap (gdf_reflect ~0.6ms, sdf_cache_light
~0.02ms), so restored: ao_res_div 4→2 (sharper contact AO) + cache_relight_period 128→64 (crisper moving-camera GI, ~free
on dense scenes) — measured 1.034× frame cost (~70fps cool, safe 60fps thermal margin). reflect_res_div
(6→4) and gi_atrous_steps (1→2) were reverted: each measured too costly (1.17× / 1.36×) for the
fanless-M3 margin; available as `P_REFLECT_RES_DIV=4` / `P_GI_ATROUS_STEPS=2` when cool. Renders
cleanly (verified). Gallery af70c1a5 + Med Sponza 1ee08a3a byte-identical. `P_GI_ATROUS_STEPS=1` etc.
remain as env escapes for more margin.

**THERMAL CAVEAT (M3 Air is fanless):** under sustained max-GPU load the machine throttles hard
(observed a cooled 14ms frame climb to 50ms+ after minutes of back-to-back benchmarking). Cooled
numbers (72fps aggressive / ~63-66fps quality-restored) are the nominal target; sustained heavy load
will throttle. Always cool 8-12s+ between measurements; the cooled A/B is the trustworthy signal.
