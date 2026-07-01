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
