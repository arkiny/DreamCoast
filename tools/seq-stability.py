#!/usr/bin/env python3
"""Temporal-stability analyzer for CAPTURE_SEQ frame dumps (PIL-only, like rt-compare.py).

The sandbox's `CAPTURE_SEQ=N` dumps frames `<base>.0000.png .. <base>.NNNN.png`
(camera static at `CAM_EYE`, or translating `CAM_EYE -> CAM_EYE_END` across the
window). This tool quantifies frame-to-frame flicker in such a sequence:

  - adjacent-pair mean |diff| per channel (avg/ch, 0..255) — flicker amplitude
  - p99 of the |diff| distribution — tail amplitude (fireflies / edge crawl)
  - the same restricted to an ROI (fractional coords, e.g. a doorway)
  - bright-pixel population instability in the ROI (firefly metric)
  - a max-over-pairs |diff| heatmap PNG (where does it flicker?)

A perfectly stable static sequence diffs to ~0. For a translating sequence the
whole-image number contains legitimate parallax; the ROI (e.g. exterior seen
through a doorway, effectively at infinity) isolates temporal noise.

Usage:
  python tools/seq-stability.py out.png
  python tools/seq-stability.py a.png b.png --roi 0.44,0.38,0.56,0.72 --heatmap hm.png
  python tools/seq-stability.py a.png b.png ... --csv table.csv

Multiple bases print one summary row each (A/B comparison table).
"""

import argparse
import csv
import glob
import os
import re
import sys

from PIL import Image, ImageChops, ImageStat


def find_frames(base: str) -> list[str]:
    """`out.png` -> sorted [`out.0000.png`, ...]."""
    root, ext = os.path.splitext(base)
    if ext.lower() != ".png":
        root = base
    pat = re.compile(re.escape(os.path.basename(root)) + r"\.(\d{4})\.png$")
    d = os.path.dirname(root) or "."
    return sorted(
        f for f in glob.glob(os.path.join(d, os.path.basename(root) + ".*.png"))
        if pat.search(os.path.basename(f)))


def roi_box(size, roi):
    w, h = size
    x0, y0, x1, y1 = roi
    return (int(x0 * w), int(y0 * h), max(int(x1 * w), int(x0 * w) + 1), max(int(y1 * h), int(y0 * h) + 1))


def diff_stats(d: Image.Image):
    """(mean avg/ch, p99 avg/ch) of an abs-diff image via per-channel histograms."""
    st = ImageStat.Stat(d)
    mean = sum(st.mean) / len(st.mean)
    hist = d.histogram()  # 256 bins per channel, concatenated
    nch = len(hist) // 256
    p99s = []
    for c in range(nch):
        h = hist[c * 256:(c + 1) * 256]
        total = sum(h)
        cutoff = total * 0.99
        acc = 0
        p = 255
        for v, n in enumerate(h):
            acc += n
            if acc >= cutoff:
                p = v
                break
        p99s.append(p)
    return mean, sum(p99s) / len(p99s)


def bright_count(img: Image.Image, thresh: int) -> int:
    """Pixels whose luma >= thresh."""
    hist = img.convert("L").histogram()
    return sum(hist[thresh:])


def analyze(base: str, roi, bright_thresh: int, heatmap: str | None):
    frames = find_frames(base)
    if len(frames) < 2:
        sys.exit(f"error: found {len(frames)} frames for {base} (need >= 2)")
    prev = Image.open(frames[0]).convert("RGB")
    box = roi_box(prev.size, roi) if roi else None
    pair_full, pair_p99, pair_roi = [], [], []
    brights = [bright_count(prev.crop(box), bright_thresh)] if box else []
    hot = None
    for f in frames[1:]:
        cur = Image.open(f).convert("RGB")
        d = ImageChops.difference(cur, prev)
        m, p99 = diff_stats(d)
        pair_full.append(m)
        pair_p99.append(p99)
        if box:
            rm, _ = diff_stats(d.crop(box))
            pair_roi.append(rm)
            brights.append(bright_count(cur.crop(box), bright_thresh))
        hot = d if hot is None else ImageChops.lighter(hot, d)
        prev = cur
    if heatmap and hot is not None:
        hot.convert("L").save(heatmap)

    n = len(brights)
    bmean = sum(brights) / n if n else float("nan")
    bstd = (sum((b - bmean) ** 2 for b in brights) / n) ** 0.5 if n else float("nan")
    return {
        "label": os.path.basename(base),
        "frames": len(frames),
        "flicker_avg": sum(pair_full) / len(pair_full),
        "flicker_max": max(pair_full),
        "p99_avg": sum(pair_p99) / len(pair_p99),
        "roi_flicker_avg": sum(pair_roi) / len(pair_roi) if pair_roi else float("nan"),
        "roi_flicker_max": max(pair_roi) if pair_roi else float("nan"),
        "roi_bright_mean": bmean,
        "roi_bright_std": bstd,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bases", nargs="+", help="capture base path(s), e.g. seq/door.png")
    ap.add_argument("--roi", help="fractional x0,y0,x1,y1 (e.g. 0.44,0.38,0.56,0.72)")
    ap.add_argument("--bright", type=int, default=240, help="ROI bright-pixel luma threshold")
    ap.add_argument("--heatmap", help="write max-|diff| heatmap PNG (first base only)")
    ap.add_argument("--csv", help="append summary rows to CSV")
    args = ap.parse_args()

    roi = tuple(float(v) for v in args.roi.split(",")) if args.roi else None
    if roi and len(roi) != 4:
        sys.exit("--roi wants x0,y0,x1,y1")

    cols = ["label", "frames", "flicker_avg", "flicker_max", "p99_avg",
            "roi_flicker_avg", "roi_flicker_max", "roi_bright_mean", "roi_bright_std"]
    rows = [analyze(b, roi, args.bright, args.heatmap if i == 0 else None)
            for i, b in enumerate(args.bases)]

    widths = {c: max(len(c), 12) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print(" | ".join(
            (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])).ljust(widths[c]) for c in cols))

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=cols)
            if new:
                wtr.writeheader()
            wtr.writerows(rows)


if __name__ == "__main__":
    main()
