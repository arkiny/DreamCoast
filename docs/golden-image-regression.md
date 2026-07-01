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
| `sponza_sc_viz` | `LEVEL=sponza_intel EV100=11 WARMUP_FRAMES=64 CAM_EYE=-14,2,0 CAM_TARGET=14,2,0 P_SC_VIZ=1` | content surface-cache view (F1/F5 consumers) |
| `sponza_gdf_ao` | …same camera… `DEBUG_VIEW=9`                                            | content distance-field AO (F2 consumers)     |

Each render is compared two ways:

1. **SHA-256 (exact).** The renderer is deterministic run-to-run on a given
   box/backend, so a byte-identical hash is the strict pass. Verified: gallery
   and both content configs are byte-identical across repeated runs on the
   Apple-Silicon box.
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

## Next increment (spec)

**Content path-tracer parity automation.** Extend the manifest so each content
config can additionally render a matched `P8_PATHTRACE=1` capture from the *same*
camera and run `rt-compare.py` between the raster and PT captures, recording the
avg/over-8/over-32 residual as a per-config **budget** in the manifest. The gate
becomes "residual ≤ recorded budget (improved or neutral)", turning today's
qualitative content-PT check into a quantitative, regressable one — the F6
"content PT residual automation" bullet. This reuses the same capture harness
(add a `pt: true` flag + a `residual_budget` field per config) and the existing
`rt-compare` math; keep PT captures deterministic (fixed sample count / seed) and
gitignore the PT PNGs the same way.
```
