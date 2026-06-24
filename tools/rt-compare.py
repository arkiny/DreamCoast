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
    python tools/rt-compare.py RASTER.png PATHTRACER.png OUT_MONTAGE.png [--amp N]

`--amp` scales the difference image for visibility (default 4x).
"""

import sys
from PIL import Image


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    amp = 4.0
    for a in sys.argv[1:]:
        if a.startswith("--amp"):
            amp = float(a.split("=", 1)[1]) if "=" in a else 4.0
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
            da[x, y] = tuple(min(255, int(v * amp)) for v in d)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
