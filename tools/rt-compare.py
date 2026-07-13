#!/usr/bin/env python3
"""Compare the rasterizer against the ground-truth path tracer.

The path tracer (when converged) is the reference renderer; the rasterizer uses
approximations (split-sum IBL, PCF shadows, no true GI). This tool renders or
ingests one screenshot from each, then writes a side-by-side montage
(raster | path tracer | amplified difference) and prints per-pixel difference
statistics so the rasterizer's approximation error can be tracked.

Both inputs must be captured from the *same* camera and scene (the sandbox's
headless `--screenshot-clean` uses a fixed camera, so a raster capture and a
`P8_PATHTRACE=1` capture line up pixel-for-pixel).

Usage:
    python tools/rt-compare.py RASTER.png PATHTRACER.png OUT_MONTAGE.png \
        [--amp N] [--lit-mask[=EPS]] [--json]

`--amp` scales the difference image for visibility (default 4x).

`--lit-mask[=EPS]` (F6 Part B) additionally reports the residual over the
**PT-lit pixels only** — pixels whose path-tracer luma (8-bit Rec.709) exceeds
EPS (default 8). Content interiors contain regions the path tracer legitimately
renders near-black (light paths within the bounce budget cannot reach them —
a GI-reach property, not a bug) while the rasterizer's approximate GI lifts
them; those pixels dominate the plain average without being actionable. The
masked average (`masked_avg`) restricts the metric to pixels where both images
carry signal, which is the raster-vs-reference fidelity gate golden-image.py
consumes. `pt_black_frac` (the excluded fraction) is reported for coverage
tracking, never gated. The excluded region is drawn dark blue in the montage's
diff panel. NOTE: the sub-EPS region is a noise-dominated dim continuum, not a
clean bimodal split — EPS=8 sits above the Monte-Carlo noise floor of a
converged capture (~1500 spp); report EPS alongside any number you quote.

`--json` appends one machine-readable line (`RTCOMPARE_JSON {...}`) with every
metric — the stable contract for tooling (golden-image.py). Human-readable
lines are unchanged; existing stdout consumers keep matching.
"""

import json
import sys
from PIL import Image

# Rec.709 luma weights on 8-bit sRGB values (the mask is a display-space
# criterion; its EPS is quoted with the capture's spp + auto-exposure recipe).
LUMA = (0.2126, 0.7152, 0.0722)
DEFAULT_LIT_EPS = 8.0
# Montage color for pixels excluded by the lit mask (PT below EPS).
MASKED_OUT_COLOR = (16, 16, 64)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    amp = 4.0
    lit_eps = None  # None = lit-mask off (default output byte-identical)
    want_json = False
    for a in sys.argv[1:]:
        if a.startswith("--amp"):
            amp = float(a.split("=", 1)[1]) if "=" in a else 4.0
        elif a.startswith("--lit-mask"):
            lit_eps = float(a.split("=", 1)[1]) if "=" in a else DEFAULT_LIT_EPS
        elif a == "--json":
            want_json = True
    if len(args) < 3:
        print(__doc__)
        return 2
    raster_path, pt_path, out_path = args[0], args[1], args[2]

    raster = Image.open(raster_path).convert("RGB")
    pt = Image.open(pt_path).convert("RGB")
    if raster.size != pt.size:
        print(f"size mismatch: {raster.size} vs {pt.size}")
        return 1
    w, h = raster.size
    ra, pa = raster.load(), pt.load()

    diff = Image.new("RGB", (w, h))
    da = diff.load()
    total = 0
    mx = 0
    hist = [0] * 256  # per-channel abs-diff histogram
    # Lit-mask accumulators (only touched when --lit-mask is given).
    m_total = 0
    m_hist = [0] * 256
    lit_count = 0
    lit_luma_r = 0.0
    lit_luma_p = 0.0
    for y in range(h):
        for x in range(w):
            r0 = ra[x, y]
            r1 = pa[x, y]
            d = tuple(abs(r0[c] - r1[c]) for c in range(3))
            for c in range(3):
                total += d[c]
                hist[d[c]] += 1
                if d[c] > mx:
                    mx = d[c]
            if lit_eps is None:
                da[x, y] = tuple(min(255, int(v * amp)) for v in d)
            else:
                luma_p = LUMA[0] * r1[0] + LUMA[1] * r1[1] + LUMA[2] * r1[2]
                if luma_p > lit_eps:
                    lit_count += 1
                    lit_luma_p += luma_p
                    lit_luma_r += LUMA[0] * r0[0] + LUMA[1] * r0[1] + LUMA[2] * r0[2]
                    for c in range(3):
                        m_total += d[c]
                        m_hist[d[c]] += 1
                    da[x, y] = tuple(min(255, int(v * amp)) for v in d)
                else:
                    da[x, y] = MASKED_OUT_COLOR

    n = w * h * 3
    avg = total / n
    # Fraction of channels differing by more than a just-noticeable threshold.
    over8 = sum(hist[9:]) / n * 100.0
    over32 = sum(hist[33:]) / n * 100.0

    montage = Image.new("RGB", (w * 3 + 16, h), (0, 0, 0))
    montage.paste(raster, (0, 0))
    montage.paste(pt, (w + 8, 0))
    montage.paste(diff, (w * 2 + 16, 0))
    montage.save(out_path)

    print(f"raster: {raster_path}")
    print(f"path tracer (reference): {pt_path}")
    print(f"montage -> {out_path}  (raster | path tracer | diff x{amp:g})")
    print(f"avg abs diff / channel: {avg:.3f}  (max {mx})")
    print(f"channels off by >8: {over8:.2f}%   off by >32: {over32:.2f}%")

    metrics = {
        "width": w,
        "height": h,
        "avg": round(avg, 4),
        "max": mx,
        "over8": round(over8, 3),
        "over32": round(over32, 3),
    }
    if lit_eps is not None:
        npx = w * h
        pt_black_frac = 1.0 - lit_count / npx
        if lit_count == 0:
            # Nothing lit: the masked metric is meaningless. The usual cause is a
            # crushed PT capture — a fixed EV100 exposure instead of AUTO_EXPOSURE=1
            # (the documented interior-PT trap).
            print(
                f"lit mask (PT luma > {lit_eps:g}): 0 lit pixels — masked residual "
                "undefined. Check the PT capture exposure (AUTO_EXPOSURE=1, not a "
                "fixed EV100)."
            )
            return 1
        mn = lit_count * 3
        masked_avg = m_total / mn
        masked_over8 = sum(m_hist[9:]) / mn * 100.0
        masked_over32 = sum(m_hist[33:]) / mn * 100.0
        mean_r = lit_luma_r / lit_count
        mean_p = lit_luma_p / lit_count
        print(
            f"lit mask (PT luma > {lit_eps:g}): {lit_count / npx * 100.0:.1f}% of "
            f"pixels lit   pt_black_frac: {pt_black_frac * 100.0:.1f}%"
        )
        print(f"masked avg abs diff / channel: {masked_avg:.3f}  (lit pixels only)")
        print(f"masked channels off by >8: {masked_over8:.2f}%   off by >32: {masked_over32:.2f}%")
        # Mean lit luma per side: a multiplicative exposure mismatch between the
        # captures shows up here directly (structure-independent sanity check).
        print(f"lit mean luma: raster {mean_r:.1f}   pt {mean_p:.1f}")
        metrics.update(
            {
                "lit_eps": lit_eps,
                "pt_black_frac": round(pt_black_frac, 4),
                "masked_avg": round(masked_avg, 4),
                "masked_over8": round(masked_over8, 3),
                "masked_over32": round(masked_over32, 3),
                "lit_mean_raster": round(mean_r, 2),
                "lit_mean_pt": round(mean_p, 2),
            }
        )
    if want_json:
        print("RTCOMPARE_JSON " + json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
