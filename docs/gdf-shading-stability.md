# Sponza Intel — shading-stability fix (frame-rate-independent auto-exposure)

## Symptom

Interactively, the **Sponza Intel** scene's shading "does not settle" — the overall image
brightness ripples/breathes frame-to-frame (most visible as speckle on the chrome ball),
especially while the camera moves. The initial hypothesis was that the **global distance-field
resolution is too low** (the composed scene GDF is `48³` ≈ 0.75 m/voxel on a ~36 m building).

## What it was NOT — the resolution hypothesis is refuted

Measured, three independent ways:

1. `docs/reflection-sdf-resolution.md` already measured `48³ → 128³` = ~0.02/ch — a non-lever.
2. `P11_GDF_DIM=128` does not even render this scene: the `128³` compose is followed by an
   unbounded CPU stage (killed at 32 min). Raising the dim is non-viable AND inert.
3. The 4-level camera-centered clipmap is built+uploaded but **bypassed** at sample time for
   content (`P11_DIRECT_SDF` defaults on → the per-mesh atlas answers, not the clipmap), so the
   dim / clip-level knobs the hypothesis reaches for do not change what the shaders sample.

A first attempt (promote the Apple-only `sdf_detail_replace` + `gi_temporal_clamp` variance clamp
to Med/High) was implemented and **rejected by measurement** — it *worsened* the static flicker
~2× (`0.083 → 0.169`/ch at fixed exposure). Reverted.

## Root cause — auto-exposure

Toggling `AUTO_EXPOSURE` is decisive (Med, d3d12, static camera, default warmup):

| config | mean consecutive Δ/ch | flicker pixels |
|---|---|---|
| auto-exposure ON (shipped)  | 0.256 | 0.73% |
| auto-exposure OFF (fixed)   | 0.083 | 0.043% |

Auto-exposure is a **~3× instability multiplier**: an amplified consecutive-frame diff shows the
residual is a whole-frame brightness ripple whose absolute magnitude is largest on the brightest
surface (the chrome ball), so it reads as "speckle."

The bug: `main.rs` computes `dt = Instant::now() - last`, and the eye-adaptation used it **raw**
(`adapt = 1 - exp(-dt·speed)`), as did the screenshot-mode `self.elapsed += dt`:

- In a headless capture `dt` is a **non-deterministic wall-clock** sequence, so the exposure (and
  the `elapsed`-driven day-night sky) converged to a **different value per backend and per run** —
  captures were not reproducible, violating the in-file "byte-identical by construction" contract.
- Interactively, as the fps fluctuates the adaptation rate fluctuates → the exposure **lurches**
  (the brightness "breathing" the user sees, worst during camera motion when fps dips).

## Fix — `frame_dt` (deterministic / bounded adaptation timestep)

`apps/sandbox/src/main.rs`: a `frame_dt` replaces raw wall-clock `dt` for adaptation state —
`FIXED_DT` (1/60) in screenshot mode (deterministic) and `dt.min(1/30)` interactively (a frame
hitch can no longer lurch the iris/sun). It drives the screenshot-mode `elapsed` accumulation and
the auto-exposure `adapt`.

This mirrors UE's eye-adaptation, which adapts in log2 space using **`DeltaWorldTime`**
(frame-rate-independent): `Engine/Shaders/Private/PostProcessEyeAdaptation.usf:183`
(`ComputeEyeAdaptation(Old, Target, EyeAdaptation_DeltaWorldTime)`) +
`PostProcessHistogramCommon.ush:222`. Auto-exposure is off for the gallery
(`auto_exposure = !gallery_scene`), so the change is **structurally incapable of moving the
path-tracer byte anchor**.

## Verification (RTX 2070 SUPER, Med, `sponza_intel_chromeball`)

| metric | before | after |
|---|---|---|
| **headless capture reproducibility** (DX run-to-run) | non-deterministic | **0.00006/ch** ✓ |
| static flicker, same warmup | 0.256 | 0.227 (−11%) |
| static flicker, settled scene (warmup 300) | — | 0.126 |
| **gallery** PT anchor, DX≡VK | 0.000007/ch | **0.000007/ch** (untouched) ✓ |
| gallery vs pre-fix | — | ≤1 LSB (unchanged) ✓ |

`clippy -D warnings` + unit tests green; `gallery_preset()` untouched.

The dominant win is **determinism** (captures are now reproducible — the parity/testing workflow
was previously built on non-deterministic frames) and **fps-independent adaptation** (no
interactive lurch, the primary visible instability under motion). The frame-rate-independent
adaptation reduces the same-warmup flicker modestly; the larger frame-to-frame settling
(0.256 → 0.126) is the scene's **convergence transient** (below).

## Tried and dropped

- **Auto-exposure convergence deadband** (hold the exposure once within a log-stops band of the
  target). Implemented, then **dropped**: isolated at a settled scene it gave **no measurable
  benefit** (0.126 without vs 0.134 with) — in a deterministic settled capture the metered target
  converges cleanly, so there is nothing for the deadband to gate. Not shipped (verify-then-claim).

## Known residuals (out of scope, honest)

- **Convergence transient.** Most of the frame-to-frame flicker at a fresh view is the
  surface-cache relight (round-robin, period 40) + GI temporal still converging over the first few
  hundred frames; it settles (0.256 → 0.126) once static. Interactively it replays after each
  camera move. Speeding convergence touches the perf-tuned amortization and is deferred.
- **Stochastic mirror-ball reflection.** The raw 1-ray/frame GGX on the near-mirror ball is
  intrinsically noisy; a fuller fix is the reflection-temporal work (near-mirror variance clamp /
  virtual-image reproject), which touches the chrome PT anchor and is deferred.
- **Sponza Intel DX≡VK ≈ 0.07/ch is PRE-EXISTING** — present with *all* fixes disabled
  (~0.067/ch). Auto-exposure amplifies a ~0.028/ch underlying content divergence (each backend
  meters a slightly different luminance → a different converged exposure → a global brightness
  offset). The determinism fix makes each backend reproducible but does not remove the ~0.028
  content divergence (compiled-shader FP + temporal-feedback amplification in the surface-cache /
  GI) — a separate, deeper content-path issue. This scene was never on the ≤0.001 parity gate
  (the gallery is).
