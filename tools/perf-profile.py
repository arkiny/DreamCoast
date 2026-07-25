#!/usr/bin/env python3
"""Headless GPU perf harness (the previously session-local `measure.py`, made real).

Runs the sandbox in screenshot mode with `PROFILE_GPU=1`, parses the per-frame
"GPU profile" blocks the engine logs headlessly (per-pass GPU ms from timestamp
queries + frame / fence-wait / cpu-record wall times), and reports the mean of
the last N frames (steady state) per pass.

The standard protocol from docs/gi-fidelity-perf-measure.md: release build,
fixed CAM_EYE/CAM_TARGET, WARMUP_FRAMES sized so the measured window sits after
cache/GI convergence, 62-frame steady-state average.

Usage:
  python tools/perf-profile.py --backend d3d12 --level sponza_intel \
      --cam-eye 7,2.2,0 --cam-target 20,2.2,0 --res 1920x1080 \
      --frames 62 --settle 45 --env EV100=11 --label door_dx

`--settle` frames are discarded from the front of the profiled window; total
warmup passed to the engine is settle+frames (capture fires after it, and the
one captured PNG goes to a temp file).
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile

ROW = re.compile(r"^\s{2}(\S+)\s+([0-9.]+) ms\s*$")
TAIL = re.compile(
    r"---\s*frame ([0-9.]+) ms \| fence-wait ([0-9.]+) ms \| cpu-record ([0-9.]+) ms \| gpu-passes ([0-9.]+) ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["d3d12", "vulkan", "metal"])
    ap.add_argument("--level", default=None)
    ap.add_argument("--cam-eye", default=None)
    ap.add_argument("--cam-target", default=None)
    ap.add_argument("--res", default="1920x1080", help="WINDOW_RES WxH (output/present extent)")
    ap.add_argument("--scale", default=None, help="RENDER_SCALE (internal = output*scale; <1 activates TAAU)")
    ap.add_argument("--frames", type=int, default=62, help="steady-state frames to average")
    ap.add_argument("--settle", type=int, default=45, help="leading profiled frames to discard")
    ap.add_argument("--env", action="append", default=[], help="extra K=V (repeatable)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", default=None, help="write result row to JSON file")
    ap.add_argument("--exe", default=os.path.join("target", "release", "sandbox.exe"))
    args = ap.parse_args()

    env = os.environ.copy()
    env["PROFILE_GPU"] = "1"
    env["WARMUP_FRAMES"] = str(args.settle + args.frames)
    if args.res:
        env["WINDOW_RES"] = args.res
    if args.scale:
        env["RENDER_SCALE"] = args.scale
    if args.level:
        env["LEVEL"] = args.level
    if args.cam_eye:
        env["CAM_EYE"] = args.cam_eye
    if args.cam_target:
        env["CAM_TARGET"] = args.cam_target
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    with tempfile.TemporaryDirectory() as td:
        shot = os.path.join(td, "perf.png")
        proc = subprocess.run(
            [args.exe, "--backend", args.backend, "--screenshot-clean", shot],
            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        log = proc.stdout + proc.stderr
        if proc.returncode != 0:
            sys.stderr.write(log[-4000:])
            sys.exit(f"sandbox exited {proc.returncode}")

    # Parse per-frame blocks: pass rows accumulate until the tail line closes a frame.
    frames = []
    cur: dict[str, float] = {}
    for line in log.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)  # strip ANSI
        m = ROW.match(line)
        if m:
            cur[m.group(1)] = float(m.group(2))
            continue
        t = TAIL.search(line)
        if t:
            cur["frame"] = float(t.group(1))
            cur["fence_wait"] = float(t.group(2))
            cur["cpu_record"] = float(t.group(3))
            cur["gpu_total"] = float(t.group(4))
            frames.append(cur)
            cur = {}

    if len(frames) <= args.settle:
        sys.stderr.write(log[-4000:])
        sys.exit(f"parsed only {len(frames)} profiled frames (need > settle={args.settle})")
    window = frames[args.settle:]
    keys = sorted({k for f in window for k in f}, key=lambda k: -statistics.mean(f.get(k, 0.0) for f in window))
    result = {k: statistics.mean(f.get(k, 0.0) for f in window) for k in keys}
    result_max = {k: max(f.get(k, 0.0) for f in window) for k in keys}

    label = args.label or f"{args.level or 'gallery'}/{args.backend}"
    print(f"== {label}  ({len(window)} frames averaged, {args.res}, backend {args.backend})")
    meta = ("frame", "fence_wait", "cpu_record", "gpu_total")
    for k in keys:
        if k in meta:
            continue
        print(f"  {k:<22} {result[k]:7.3f} ms   (max {result_max[k]:7.3f})")
    print(f"  {'-- gpu_total':<22} {result.get('gpu_total', 0.0):7.3f} ms")
    print(f"  {'-- frame':<22} {result.get('frame', 0.0):7.3f} ms   "
          f"(fence-wait {result.get('fence_wait', 0.0):.3f}, cpu-record {result.get('cpu_record', 0.0):.3f})")
    fps = 1000.0 / result["gpu_total"] if result.get("gpu_total") else 0.0
    print(f"  {'-- fps(gpu)':<22} {fps:7.1f}")

    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps({"label": label, "backend": args.backend, "res": args.res,
                                "frames": len(window), "mean": result, "max": result_max}) + "\n")


if __name__ == "__main__":
    main()
