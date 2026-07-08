# 60fps margin: TAAU packed history + baked ACES tonemap LUT (+ reflect-div re-check)

Scope: the "remaining ~1-2ms to consistent 60fps" item from the macOS/M3 Apple-tier track —
the Retina-resolution output passes (`taau`, `tonemap`) — plus the deferred "Apple
`reflect_res_div = 6` too aggressive?" re-check now that the reflection-quality v2 stack is
the tier default. Measured on M3, `sponza_intel_glossyball` 2560x1440, `EV100=11
WARMUP_FRAMES=100 PROFILE_GPU=1`, RELEASE, fence-wait statistics over 99 frames.

## 1. `taau`: fp16-packed history (commit `perf(taau)`)

The TAAU history ping-pong dominated the pass at output resolution: 16B/px `hist`
(rgb+len) + 16B/px `pos` (world point + valid), read+write every frame. Two facts made it
4x smaller with zero algorithmic change:

- **The `pos` buffer's only consumed field was `.w`** (the "real prior surface" flag) —
  the world point itself became dead when the YCoCg variance clip replaced hard geometric
  disocclusion. Validity now rides in the history length (`len == 0` = sky/no-surface,
  `len >= 1` = surface, which the accumulator guarantees), and the pos buffers are gone.
- **fp16 rgb+len holds the full input precision** — the accumulated signal comes from an
  RGBA16Float HDR chain (the reference engine's TAA history is fp16 for the same reason).

Opt-in `taau_packed_history` tier knob (`P_TAAU_PACKED`), Apple ON; everything else keeps
the legacy layout (gallery golden `af70c1a5` PASS; TAAU is inactive at render-scale 1
anyway). fp16 clamp at 65504 before packing (inf would poison the EMA).

**Measured: `taau` 3.20 → 1.34 ms (−58%); fence-wait mean 25.1 → 22.6 ms; frames over
16.6ms 66 → 53 /99.** Screenshot diff vs baseline 0.127/255 avg — run-to-run noise level,
no structure.

## 2. `tonemap`: baked ACES 1.3 RRT+ODT LUT (commit `feat(tonemap)`)

Reference-engine "combine LUTs" model. A per-frame compute pass (`tonemap_lut.slang`,
N=48 strip = 110K texels, **0.04 ms**) bakes ASC-CDL grade + the full ACES 1.3 RRT +
sRGB(100 nit, dim) ODT into a 2D-strip LUT over a log2-encoded axis (2^-12..2^6, both
curve ends flat inside the table); the tonemap replaces its per-pixel curve with one strip
fetch (manual trilinear = 2 bilinear taps + lerp). Cost is neutral (0.55 ms unchanged) —
this is a **quality** upgrade: real glow/red-modifier/desaturation/surround response
instead of the Narkowicz fit, and the LUT can absorb arbitrarily heavy grading later for
free.

`aces.slang` is ported from the A.M.P.A.S. reference CTL (aces-dev v1.3), **not** from any
engine's shader port; the AMPAS license (notice + conditions + disclaimer) is retained in
`THIRD_PARTY_LICENSES.md`. The port was validated against an independent CPU
implementation of the same CTL: achromatic ramp monotonic, 0.18 → 0.1041 display-linear
(the reference mid-grey), within the known ~1-3% envelope of the published analytic fit
across 0.005..11.2 and chromatic probes.

The LUT applies to every tonemap source (main lit AND the path-trace viz) so PT-parity
captures tonemap both sides identically. Knob `tonemap_aces` (`P_TONEMAP_ACES`,
`P_TONEMAP_LUT_SIZE`), Apple ON; gallery + Low/Med/High keep the legacy curve
byte-identical pending DX≡VK.

## 3. Apple `reflect_res_div = 6` re-check: **keep 6**

With the freed margin, div 5 and 4 were swept (`P_REFLECT_RES_DIV`):

| div | `gdf_reflect` | fence mean | frames > 16.6ms | ball-crop diff vs div6 |
|-----|--------------:|-----------:|----------------:|-----------------------:|
| 6 (tier) | 5.52 ms | 22.4 ms | 53/99 | — |
| 5 | 7.58 ms | 24.6 ms | 65/99 | 1.44/255 |
| 4 | 11.03 ms | 27.8 ms | 86/99 | 2.11/255 |

The glossy ball is **visually indistinguishable** across the three — the v2 stack's
screen-hit path already serves on-screen hits from the full-res lit history, and the
prefilter split serves the rough floor from the cache MIP, so trace resolution no longer
bounds glossy quality (the old "div6 too aggressive" note predates screen-hit). Raising
the div spends 2.2–5.5 ms for no visible gain and erases the margin item 1 bought.
Decision: div stays 6; the margin goes to frame-rate consistency.

## Remaining to consistent 60fps

Frames over 16.6ms: 66 → 53 /99. The two top passes are now `gdf_reflect` (5.5 ms,
trace-res-bound — see above) and `sdf_cache_light` (5.0 ms, relight budget). Next lever
candidates: relight-period tier tuning / B2 trace compaction, not output-res work — the
Retina tail (`taau` 1.34 + `tonemap` 0.55 + `lut` 0.04) is no longer a bottleneck.
