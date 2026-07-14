# Golden-image regression runner (F6, first increment)

`tools/golden-image.py` is a deterministic golden-image regression gate for the
verification/robustness track (F6 in the GI-fidelity phases/roadmap docs). It
complements — it does not replace — `tools/rt-compare.py`:

- `rt-compare.py` measures **quality** (rasterizer-vs-path-tracer residual).
- `golden-image.py` measures **regression** (did a fixed capture change at all?).

Path-tracer parity was only automated for the gallery; content scenes, the
surface-cache view, and temporal-shimmer regressions were checked ad hoc. This
runner makes those repeatable so every other phase can self-verify.

## What it checks

A small manifest of named, fixed-camera, deterministic headless captures, run
via the release `sandbox --backend metal --screenshot-clean`:

| config          | recipe                                                                 | what it guards |
|-----------------|------------------------------------------------------------------------|----------------|
| `gallery`       | default scene (no env)                                                  | the byte-identity **anchor** `af70c1a5…` (CLAUDE.md invariant gate) |
| `sponza_sc_viz` | `LEVEL=sponza_intel EV100=11 AUTO_EXPOSURE=0 WARMUP_FRAMES=64 CAM_EYE=-14,2,0 CAM_TARGET=14,2,0 P_SC_VIZ=1` | content surface-cache view (F1/F5 consumers) |
| `sponza_gdf_ao` | …same camera… `DEBUG_VIEW=9`                                            | content distance-field AO (F2 consumers)     |

Each render is compared two ways:

1. **SHA-256 (exact).** The renderer is deterministic run-to-run on a given
   box/backend, so a byte-identical hash is the strict pass — for the **content**
   configs too. **History (resolved 2026-07-14):** the content configs were
   byte-unstable run-to-run for a while (isolated pixels up to ~36/255 in
   `sponza_gdf_ao`, the ~0.2/ch low-bit reshuffle in `sponza_sc_viz`). The root
   cause was **not** the surface cache, GDF-AO atomics, or denoiser history (the
   AO pass is a pure function; the async relight queue is not even active in
   screenshot mode): auto-exposure defaults ON for non-gallery scenes even under
   a fixed `EV100`, and its metering EMA adapted on **wall-clock dt**
   (`adapt = 1-exp(-dt·2.5)`), so the exposure trajectory over the warmup
   differed every run. The content tier's TAAU (64-frame running-mean history at
   0.67× render scale) integrated that transient into the frame-64 capture, and
   its neighborhood clip amplified it at high-contrast edges (the far-door
   speckles). Fixed twice over: screenshot mode now adapts the AE EMA on
   `FIXED_DT` (frame-counted deterministic, which also removes the AE-trajectory
   jitter from the `pt` configs), and the content recipes pin `AUTO_EXPOSURE=0`
   (a fixed-EV regression capture should not meter at all). Measured post-fix:
   `sponza_gdf_ao` byte-exact 6/6 captures across rebuilds and GPU contention;
   `sponza_sc_viz` byte-exact 7/8 with one capture off by a SINGLE pixel × 1 LSB
   — a second-order residual (suspect: surface-cache feedback/readback timing),
   absorbed by the tolerant path below and deferred until the F4B increments
   re-baseline sc_viz anyway.
2. **Pixel mean/max diff (tolerant).** When a PNG golden is present, an exact
   miss falls back to a per-channel mean/max abs-diff check (`--mean-tol`,
   `--max-tol`). This absorbs small cross-box/cross-backend/driver
   nondeterminism so the same manifest stays useful off the authoring machine.

## Golden storage decision

**Commit hashes, not pixels.** `tools/goldens/manifest.json` (per-config SHA-256
+ the exact capture recipe) is committed and is the canonical regression. The
2560×1440 PNG blobs are **gitignored** (`tools/goldens/.gitignore`) — they are
large and fully regenerable, and are only needed for the optional tolerant
pixel-diff path. This keeps the repo lean while the strict SHA gate stays
version-controlled and reviewable in a diff.

Regenerate PNG goldens locally when you want the tolerant path:
`python tools/golden-image.py --update --save-png`.

## Usage

```bash
cargo build -p sandbox --release        # required first

python tools/golden-image.py            # check every config (default backend metal)
python tools/golden-image.py --only gallery          # one config (repeatable)
python tools/golden-image.py --update                # (re)author SHA goldens
python tools/golden-image.py --update --save-png     # also store PNG goldens
python tools/golden-image.py --mean-tol 0.5 --max-tol 12   # tune tolerance
```

Content configs auto-**SKIP** when `assets/IntelSponza/` is absent (that asset is
gitignored / runtime-fetched), so the gallery anchor always runs. Exit code is
non-zero on any FAIL or missing golden — safe to wire into CI.

## When to re-author goldens

Only after a **verified, intentional** lighting/output change (path-tracer
residual improved or neutral, DX≡VK confirmed). Re-authoring the gallery anchor
without PT verification is a roadmap "do-not" (`gi-fidelity-roadmap.md` §5). For
content configs, re-author when the scene, camera recipe, or a consumed feature
legitimately changes, and note why in the commit.

**Rebase log — gallery anchor `af70c1a5…` → `65d04ceca2c4…` (2026-07-13):** default
wrap-sampler anisotropy went `1` → `16` on all backends (branch
`fix/default-anisotropy-16`; `docs/qhd-perf.md` Stage 9). The anchor moved because
some gallery surfaces are wrap-sampled; verified intentional — rendering with
`P_ANISO=1` reproduces the old anchor `af70c1a5…` byte-for-byte, so the delta is
purely the anisotropic filter, and the gallery stays run-to-run byte-identical at
the new default. Content SHAs were **left untouched** (they are run-to-run noisy;
see the caveat above). Cross-backend DX≡VK re-verification is tracked in
`docs/windows-verify-anisotropy-default.md`.

## Content PT-residual configs (F6 Part B)

A config with `pt: True` in the CONFIGS recipe is a **budget-gated raster-vs-
path-tracer residual pair**, not a SHA golden: the runner renders the recipe
twice from the same fixed camera (raster, then `P8_PATHTRACE=1`), runs
`rt-compare.py --lit-mask=<lit_eps> --json`, and passes iff
`masked_avg <= residual_budget` (improved-or-neutral). The lit mask (PT luma >
`lit_eps`, default 8) restricts the metric to pixels the path tracer actually
reaches within its bounce budget — the PT-dim remainder (a GI-reach property
the raster's approximate GI lifts; F4 territory) is tracked as
`pt_black_frac` but never gated. `pt_black_frac >= 0.9` fails loudly (the
crushed-exposure trap). Full rationale + the capture-recipe invariants
(AUTO_EXPOSURE=1 / RENDER_SCALE=1 / WARMUP_FRAMES=192, EV100 stripped):
`docs/phase-f6b-content-pt-residual-plan.md`.

| config               | camera                          | gates |
|-----------------------|---------------------------------|-------|
| `sponza_pt_sunlit`    | atrium courtyard, upward (S2)   | `masked_avg <= residual_budget` (primary fidelity gate) |
| `sponza_pt_interior`  | interior colonnade (F1 lineage) | same, tolerant budget (high `pt_black_frac` by construction) |

Budgets are seeded by `--update` as `measured + 0.3` (`PT_BUDGET_MARGIN`,
validated against the measured two-run spread) and only move DOWN on a
verified improvement. **Do not hand-edit `manifest.json`** — `--update`
rebuilds each selected entry wholesale from the CONFIGS recipe + the fresh
measurement, so the recipe in `golden-image.py` is the single source of truth.
