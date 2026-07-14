#!/usr/bin/env python3
"""Crop-region luma diagnostic for the GI-fidelity track (docs/phase-gi-volume-leak-plan.md).

Reports, for each image: the mean Rec.709 luma of a horizontal crop x in [x0, x1)
(full height), the crop's mean B-R (colour-cast anchor: the interior PT reference
reads ~ -0.6, the lit-crop convention ~ -1.0), and the full-image mean luma.

Diagnostic-only: NOT part of the golden-image / rt-compare contract (their output
formats are regex-parsed elsewhere and must not change). The default crop x 0..900
is the deep-occlusion left-colonnade band of the interior gate camera
(CAM_EYE=-14,2,0 CAM_TARGET=14,2,0) at 2560x1440.

Usage: python3 tools/crop-luma.py img.png [img2.png ...] [--x0 0] [--x1 900]
"""

import argparse

import numpy as np
from PIL import Image

# Rec.709 luma weights — keep in lockstep with tools/rt-compare.py.
LUMA = (0.2126, 0.7152, 0.0722)


def stats(path: str, x0: int, x1: int):
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    crop = im[:, x0:x1, :]
    luma = LUMA[0] * crop[..., 0] + LUMA[1] * crop[..., 1] + LUMA[2] * crop[..., 2]
    full = LUMA[0] * im[..., 0] + LUMA[1] * im[..., 1] + LUMA[2] * im[..., 2]
    br = float(np.mean(crop[..., 2] - crop[..., 0]))
    return float(np.mean(luma)), br, float(np.mean(full))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--x1", type=int, default=900)
    args = ap.parse_args()
    for p in args.images:
        luma, br, full = stats(p, args.x0, args.x1)
        print(f"{p}: crop[{args.x0}:{args.x1}] luma={luma:.2f}  B-R={br:+.2f}  full-luma={full:.2f}")


if __name__ == "__main__":
    main()
