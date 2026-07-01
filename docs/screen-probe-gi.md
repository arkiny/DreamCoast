# Screen-space radiance probe GI — implementation log

Spec / staging: `docs/screen-probe-gi-plan.md`. This file records what has landed and the
verification numbers. Reference cross-check: the screen-probe diffuse-GI method (per-tile
probes traced into an octahedral radiance atlas, gathered per pixel) — defaults adapted to
our ~37 m interior scenes. Implemented on our own SW-RT global-distance-field tracer.

## Shared bounce tracer (single source) — `gdf_bounce.slang`

The 1-bounce diffuse march/shade/shadow used by the per-pixel GI pass was extracted from
`gdf_gi.slang` into `gdf_bounce.slang`, parameterized by a `BounceScene` value the caller
fills from its own push constants. Both `gdf_gi.slang` (per-pixel) and
`screen_probe_trace.slang` now share ONE tracer — no duplicated code to drift. Pure
code-motion + parameter-passing: the gallery per-pixel ray-march output is byte-identical.

- Verify: gallery `--screenshot-clean` SHA-256 unchanged before/after the refactor
  (`dba9ff7c…`). Only `gdf_gi` recompiled.

## P1 — probe placement + trace + octahedral atlas + per-pixel gather

Opt-in `SCREEN_PROBE=1` (default OFF → the gallery anchor is byte-identical with no env).
Replaces the GI *consumption* (world-volume sample / per-pixel ray march) with:

1. **Trace** (`screen_probe_trace.slang`): one probe per `SP_DOWNSAMPLE`(=16) screen tile,
   placed on the representative G-buffer surface at the tile center (depth + world normal).
   Each probe stores an `SP_OCT_RES`(=8)² octahedral **radiance** tile in a screen-wide
   atlas; every texel decodes a full-sphere world direction and marches the shared bounce
   tracer into the scene GDF. Because the probe sits ON a visible surface, the world-grid
   placement failure (probes stranded underground / in open air) is gone structurally.
2. **Integrate** (`screen_probe_integrate.slang`): each pixel gathers its surrounding 2×2
   probes, weighted by bilinear position × surface-plane proximity (`exp(-|Δ·n|/σ)`) ×
   normal agreement (`saturate(n·nₚ)^k`), and reconstructs indirect irradiance E by
   cosine-integrating each probe's octahedral tile over the pixel hemisphere
   (uniform per-texel solid angle 4π/N → the 1/π hemisphere norm folds to 4/N). Output is
   the same raw-E image the deferred lighting multiplies by albedo; it then flows through
   the existing temporal + à-trous denoiser.

Wiring: `GiSystem::record_screen_probe` (`apps/sandbox/src/gi.rs`), push packers
`screen_probe_{trace,integrate}_push` (`push.rs`), a guarded match arm in the GI build
(`main.rs`). Atlas + output are transient graph storage images.

### Verification (Metal, this macOS box; Windows DX≡VK pending — frozen)

Path-tracer parity is measured on the **gallery** (the only path-traceable scene; the PT
miss-shader has no BLAS for the Intel-Sponza levels, so `P8_PATHTRACE` there falls back to
the raster and cannot serve as ground truth).

| Capture (gallery, vs `P8_PATHTRACE` ground truth) | avg abs diff / ch |
|---|---|
| GI baseline (per-pixel ray march) | **6.162** |
| Screen-probe GI (`SCREEN_PROBE=1`) | **6.228** |

The screen-probe gather lands within 0.07/ch of the established baseline against the path
tracer → the probe irradiance is correct in magnitude and direction (not blown out), which
is the P1 correctness bar.

- **Gallery byte-identical** anchor (no env): `dba9ff7c…` — unchanged.
- **Determinism**: two `SCREEN_PROBE=1` gallery runs byte-identical (`12b2baec…`). The
  probe trace uses fixed octahedral directions (no RNG), so it is deterministic per frame.
- `sponza_intel` (`EV100=11`): renders a plausibly lit interior with colored bleed on the
  columns near the curtains. It differs from the world-volume baseline (no ground truth on
  that scene) chiefly by a cool cast — the shipped indoor sky-vis occlusion (de-blue) is
  not yet wired into the screen-probe path (a P3 item).

## P3 — indoor skylight occlusion (sky-vis) from the probe trace

The gather now reconstructs indoor skylight occlusion directly from the probes instead of
the separate SH-volume path — more principled, since the probes already trace the exact
visibility. The shared tracer `bs_trace_bounce` reports `escaped` (1 = the ray reached the
sky, 0 = it hit geometry); the trace pass stores it in the atlas **alpha**. The integrate
pass reconstructs the cosine-weighted hemispherical sky visibility `V(n) = Σ vis·cos / Σ cos`
per probe, blends it with the same probe weights as E, and writes a full-res sky-vis image.
That image feeds the existing lighting occlusion (the shipped `P_SKYVIS_TINT` /
`P_SKYVIS_MIN_OCC` neutral-leak) exactly like the volume path's output, so open surfaces
keep full skylight and enclosed interiors de-blue to the colored scene bounce.

The per-pixel probe gather itself (bilinear × plane × normal weights, cosine octahedral
integration) is the P3 consumption; it landed with P1 as the vertical slice that made the
technique measurable, and is refined here with the sky-vis term.

### Verification (Metal; Windows DX≡VK pending — frozen)

| Capture (gallery, vs `P8_PATHTRACE` ground truth) | avg abs diff / ch |
|---|---|
| GI baseline (per-pixel ray march) | 6.162 |
| Screen-probe GI, P1 (no sky-vis) | 6.228 |
| Screen-probe GI, P3 (sky-vis active) | **5.984** |

Reconstructing the indoor occlusion from the probes brings the gallery **below the
ray-march baseline** against the path tracer — the traced sky visibility occludes the IBL
diffuse in recesses exactly where the path tracer does.

- **Gallery byte-identical** anchor (no env): `dba9ff7c…` unchanged (the `escaped` out-param
  added to `bs_trace_bounce` is behaviour-preserving on the per-pixel path).
- **Determinism**: two `SCREEN_PROBE=1` gallery runs byte-identical (`aa30b9f6…`).
- `sponza_intel` (`EV100=11`): the stone de-blues from the P1 cool cast to a warm/neutral
  indoor look with the curtains popping — matching the shipped volume-path aesthetic, now
  driven by traced per-probe visibility.

## Next

- **P2**: importance sampling (BRDF + prior radiance) + screen-space spatial/temporal probe
  filtering with octahedral borders (noise / firefly suppression).
- Gather efficiency: probe-irradiance pre-integration to drop the per-pixel tap count
  (currently oct_res² × up-to-4 probes); better disocclusion handling.
- **P4**: world radiance-cache clipmap fallback for off-screen / far-field / infinite bounce.
- **P5**: tile classification, ray budget, half/quarter-res, temporal amortization.

## P2a — spatial cross-probe filter

`screen_probe_filter.slang`: a joint-bilateral blur of the octahedral atlas ACROSS
neighboring probes (a 3×3 probe kernel by default). For each probe texel it blends the SAME
octahedral direction from surrounding probes, weighted by surface-plane proximity + normal
agreement — smoothing probe-to-probe variation on a shared surface but never blurring across
a silhouette. Filtering matching directions across probes needs no octahedral border
handling. Radiance (rgb) + traced sky-vis (alpha) filtered together. Runs between the trace
and the gather; `P_SP_FILTER=0` disables (kernel size `P_SP_FILTER=N`).

Verify (Metal): gallery path-tracer parity 5.988/ch with the filter vs 5.984 without — the
bilateral filter preserves the mean, so it is parity-neutral on the already-smooth gallery
(no bias) and reduces probe-grid blockiness on complex scenes / sparser probes. Gallery
byte-identical (no env). Deterministic (sponza filter-on `30f70511…`).

## P2b — resolution / density scalability knobs + the measured accuracy finding

The probe trace is **deterministic and noise-free** (fixed octahedral directions, no
Monte-Carlo sampling), so P2's classic role — denoising stochastic traces — does not apply.
The only accuracy lever the probes expose is **angular resolution** (octahedral texels) and
**spatial density** (probes per screen). Both are now env-tunable quality knobs (a future
`RenderQuality` tier): `P_SP_OCT` (default 8) and `P_SP_DOWNSAMPLE` (default 16).

Measured on the gallery vs the path tracer:

| knob | value → avg abs diff / ch |
|---|---|
| octahedral res `P_SP_OCT` | 8 → 5.988 · 12 → 5.997 · 16 → 5.988 |
| probe density `P_SP_DOWNSAMPLE` | 8 → 5.984 · 16 → 5.988 · 32 → 5.996 |

**Neither knob moves parity meaningfully** — the diffuse GI gather is already at its accuracy
floor; the residual is dominated by non-diffuse-GI approximations (GGX / the mirror spheres /
GDF voxelization), not the probe angular resolution or density. So the default 8² / 16 px is a
well-chosen cost/quality point, and heavier angular supersampling (temporal jitter +
accumulation) would **not** improve measured accuracy on these scenes — it is a stability /
sparse-probe feature that belongs with **P5** temporal amortization (when probes drop to
half/quarter-res and actually need the extra effective samples). Landing it now would add cost
for no parity gain, which the build-to-quality metric (path-tracer parity) argues against.

Gallery byte-identical (no env, `dba9ff7c…`); deterministic (no RNG added).

## P4 — world radiance cache fallback (opt-in) + the subsumption finding

`wrc_update.slang` + `wrc_common.slang`: a camera-following clipmap of world probes (reusing
the GDF clipmap level AABBs for placement, `WRC_GRID`=16 probes/side, `WRC_OCT`=8 octahedral
texels), each storing incoming radiance in a persistent ping-pong 2D atlas backed by a byte-
address storage buffer (escapes the 3D-volume-count limit; dense/direct-indexed — sparse
indirection is a noted refinement). Updated each frame by marching the shared bounce tracer;
escaped update rays sample the previous atlas (infinite bounce + far-field over frames). A
screen-probe ray that escapes the local trace reads the cache at the probe origin in the ray
direction (`screen_probe_trace.slang`). `P_WRC=1` opt-in.

**Measured finding — the cache's role is largely subsumed by our full-scene GDF.** With the
fallback correctly landing inside the cache (sample at the on-surface origin, not the far
point which is outside every clipmap level):

| scene | WRC on vs off |
|---|---|
| gallery (open) vs PT | 6.005 vs **5.988** — slightly *worse* (mild overshoot on an already-floor-limited scene) |
| sponza_intel (enclosed) | **0.000/ch** difference (few-LSB) — inert |

Why: unlike a screen-space-only tracer (which cannot see off-screen and *needs* a world cache),
our screen probes march the **full-scene GDF clipmap**, so their rays hit real on/off-screen
geometry instead of escaping — and the few rays that do escape are sky-gap rays the cache has no
radiance for. The reference cache's off-screen/far-field role is therefore already covered by
the GDF. The one genuine gap is **multi-bounce** (our probes are 1-bounce; the path tracer is
infinite), but the cache's escaped-ray hook is the wrong place for it — multi-bounce would come
from feeding cache irradiance at **hits**, which overlaps the existing mesh-card surface cache.

So P4 lands as **correct, deterministic, gallery-byte-identical, opt-in infrastructure** (default
OFF so it never regresses the screen-probe default), with the honest finding recorded rather than
a false quality claim. The multi-bounce-at-hits integration is the real untapped value and is
noted for a future pass (it needs a hit-side irradiance lookup + reconciliation with the surface
cache). Gallery anchor `dba9ff7c…`; screen-probe default unchanged at 5.988.
