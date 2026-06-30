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

## Next

- **P3**: reintroduce the indoor skylight-occlusion (sky-vis) on the probe path so the
  stone de-blues to match the shipped look; refine the gather (probe-irradiance
  pre-integration to drop the per-pixel tap count; better disocclusion handling).
- **P2**: importance sampling (BRDF + prior radiance) + screen-space spatial/temporal probe
  filtering with octahedral borders.
