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
  - `--sharpness`: per-frame gradient energy (the TAAU motion-blur metric)

A perfectly stable static sequence diffs to ~0. For a translating sequence the
whole-image number contains legitimate parallax; the ROI (e.g. exterior seen
through a doorway, effectively at infinity) isolates temporal noise.

`--sharpness` adds the motion-sharpness metric: the mean absolute first
difference of luma (|dI/dx| + |dI/dy|, averaged) over the MIDDLE HALF of the
sequence — the frames where a dolly is at steady speed and the temporal history
has settled into its moving state. Temporal blur (bilinear history resampling,
over-long history in motion) lowers it; a sharper resolve raises it. Run the
SAME camera twice — static (`CAPTURE_SEQ_STEP=0`) and dolly (`CAM_EYE_END=...`)
— and read the dolly/static ratio: 1.0 = motion costs no sharpness.

CAVEAT: gradient energy also rises with shimmer and ringing, so it is only a
win when the flicker columns hold their ratchet at the same time. Always read
`grad_*` together with `roi_flicker_avg` / `roi_bright_std`.

Usage:
  python tools/seq-stability.py out.png
  python tools/seq-stability.py a.png b.png --roi 0.44,0.38,0.56,0.72 --heatmap hm.png
  python tools/seq-stability.py a.png b.png ... --sharpness --csv table.csv

Multiple bases print one summary row each (A/B comparison table).
"""

from __future__ import annotations
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


def grad_energy(img: Image.Image) -> float:
    """Mean absolute first difference of luma — the motion-sharpness proxy.

    mean(|I(x+1,y) - I(x,y)|, |I(x,y+1) - I(x,y)|) over the image, 0..255. Any
    low-pass (temporal history resampling, upscale blur) lowers it monotonically,
    so a dolly/static ratio near 1.0 means motion costs no detail. PIL-only:
    shifted crops + ImageChops, no numpy (matches rt-compare.py's constraint).
    """
    g = img.convert("L")
    w, h = g.size
    if w < 2 or h < 2:
        return float("nan")
    dx = ImageChops.difference(g.crop((1, 0, w, h)), g.crop((0, 0, w - 1, h)))
    dy = ImageChops.difference(g.crop((0, 1, w, h)), g.crop((0, 0, w, h - 1)))
    return 0.5 * (ImageStat.Stat(dx).mean[0] + ImageStat.Stat(dy).mean[0])


def mid_window(n: int) -> range:
    """Middle half of [0, n) — a dolly is at steady speed and the temporal
    history has settled into its moving state there (the ends carry the
    start/stop transient and the first-frame history reset)."""
    if n < 4:
        return range(n)
    return range(n // 4, n - n // 4)


def analyze(base: str, roi, bright_thresh: int, heatmap: str | None, sharpness: bool):
    frames = find_frames(base)
    if len(frames) < 2:
        sys.exit(f"error: found {len(frames)} frames for {base} (need >= 2)")
    prev = Image.open(frames[0]).convert("RGB")
    box = roi_box(prev.size, roi) if roi else None
    pair_full, pair_p99, pair_roi = [], [], []
    brights = [bright_count(prev.crop(box), bright_thresh)] if box else []
    mid = set(mid_window(len(frames))) if sharpness else set()
    grads, grads_roi = [], []

    def accum_grad(i: int, img: Image.Image):
        if i not in mid:
            return
        grads.append(grad_energy(img))
        if box:
            grads_roi.append(grad_energy(img.crop(box)))

    accum_grad(0, prev)
    hot = None
    for i, f in enumerate(frames[1:], start=1):
        cur = Image.open(f).convert("RGB")
        d = ImageChops.difference(cur, prev)
        m, p99 = diff_stats(d)
        pair_full.append(m)
        pair_p99.append(p99)
        if box:
            rm, _ = diff_stats(d.crop(box))
            pair_roi.append(rm)
            brights.append(bright_count(cur.crop(box), bright_thresh))
        accum_grad(i, cur)
        hot = d if hot is None else ImageChops.lighter(hot, d)
        prev = cur
    if heatmap and hot is not None:
        hot.convert("L").save(heatmap)

    n = len(brights)
    bmean = sum(brights) / n if n else float("nan")
    bstd = (sum((b - bmean) ** 2 for b in brights) / n) ** 0.5 if n else float("nan")
    row = {
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
    if sharpness:
        row["grad_avg"] = sum(grads) / len(grads) if grads else float("nan")
        row["grad_roi_avg"] = sum(grads_roi) / len(grads_roi) if grads_roi else float("nan")
        row["grad_frames"] = len(grads)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bases", nargs="+", help="capture base path(s), e.g. seq/door.png")
    ap.add_argument("--roi", help="fractional x0,y0,x1,y1 (e.g. 0.44,0.38,0.56,0.72)")
    ap.add_argument("--bright", type=int, default=240, help="ROI bright-pixel luma threshold")
    ap.add_argument("--heatmap", help="write max-|diff| heatmap PNG (first base only)")
    ap.add_argument("--sharpness", action="store_true",
                    help="add the motion-sharpness metric (mid-window luma gradient energy)")
    ap.add_argument("--csv", help="append summary rows to CSV")
    args = ap.parse_args()

    roi = tuple(float(v) for v in args.roi.split(",")) if args.roi else None
    if roi and len(roi) != 4:
        sys.exit("--roi wants x0,y0,x1,y1")

    cols = ["label", "frames", "flicker_avg", "flicker_max", "p99_avg",
            "roi_flicker_avg", "roi_flicker_max", "roi_bright_mean", "roi_bright_std"]
    if args.sharpness:
        cols += ["grad_avg", "grad_roi_avg", "grad_frames"]
    rows = [analyze(b, roi, args.bright, args.heatmap if i == 0 else None, args.sharpness)
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
