#!/usr/bin/env python3
"""Deterministic golden-image regression runner (F6, verification infra).

Path-tracer parity (``tools/rt-compare.py``) is a *quality* metric; this is a
*regression* gate. It renders a small manifest of named, fixed-camera capture
configs headlessly via the release ``sandbox`` binary and checks each render
against a stored golden two ways:

    1. **SHA-256 (exact).** The renderer is deterministic run-to-run on a given
       box/backend (see CLAUDE.md gates), so a byte-identical hash is the strict
       pass. The gallery anchor (``af70c1a5...``) is the canonical example.
    2. **Pixel mean/max diff (tolerant).** When a PNG golden exists, we also
       report per-channel mean/max abs diff. This absorbs the small
       cross-box / cross-backend / driver nondeterminism that a raw hash cannot,
       so the same manifest stays useful off the golden-authoring machine. A
       config passes tolerantly if mean <= --mean-tol and max <= --max-tol.

A config's exact-SHA match is authoritative on the golden-authoring box; the
pixel tolerance is the portable fallback and is only evaluated when a PNG golden
is present for that config.

Golden storage (committed, small):
    tools/goldens/manifest.json   -- per-config SHA-256 + capture recipe (committed)
    tools/goldens/*.png           -- OPTIONAL PNG goldens (gitignored; big 2560x1440
                                     blobs are not committed -- regenerate with
                                     `--update --save-png`). Hashes alone are the
                                     committed regression; PNGs are an opt-in
                                     convenience for the tolerant pixel-diff path.

Usage:
    # Build first: cargo build -p sandbox --release
    python tools/golden-image.py                 # check every config
    python tools/golden-image.py --only gallery  # check one (repeatable)
    python tools/golden-image.py --update        # (re)generate SHA goldens
    python tools/golden-image.py --update --save-png   # also drop PNG goldens
    python tools/golden-image.py --backend metal --mean-tol 0.5 --max-tol 12

Exit code is 0 only if every checked config passes (non-zero on any failure or
if a golden is missing and --update was not given). This is safe to wire into
CI once the DX==VK freeze lifts (add --backend vulkan / d3d12 rows).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Repo layout: this file lives in <repo>/tools/.
TOOLS_DIR = Path(__file__).resolve().parent
REPO = TOOLS_DIR.parent
GOLDEN_DIR = TOOLS_DIR / "goldens"
MANIFEST = GOLDEN_DIR / "manifest.json"

# --- Capture manifest ---------------------------------------------------------
# Each config is a fixed-camera, deterministic headless capture. `env` extends
# the process environment; `name` is both the golden key and the PNG basename.
# The gallery anchor uses the default scene (no LEVEL) -- it is the invariant
# byte-identity gate from CLAUDE.md and must match `af70c1a5...`.
#
# Content configs render the interior colonnade of the content scene from a
# fixed camera with a 64-frame warmup so caches converge to a stable image:
#   - sponza_sc_viz : the high-res surface-cache visualization (P_SC_VIZ)
#   - sponza_gdf_ao : the distance-field AO debug view (DEBUG_VIEW=9)
CONTENT_CAM = {
    "LEVEL": "sponza_intel",
    "EV100": "11",
    "WARMUP_FRAMES": "64",
    "CAM_EYE": "-14,2,0",
    "CAM_TARGET": "14,2,0",
}
CONFIGS = [
    {
        "name": "gallery",
        "desc": "gallery anchor (default scene, byte-identity invariant)",
        "env": {},
        "requires_asset": None,
    },
    {
        "name": "sponza_sc_viz",
        "desc": "content: surface-cache viz (P_SC_VIZ)",
        "env": {**CONTENT_CAM, "P_SC_VIZ": "1"},
        "requires_asset": "assets/IntelSponza",
    },
    {
        "name": "sponza_gdf_ao",
        "desc": "content: distance-field AO (DEBUG_VIEW=9)",
        "env": {**CONTENT_CAM, "DEBUG_VIEW": "9"},
        "requires_asset": "assets/IntelSponza",
    },
]


def sandbox_bin() -> Path:
    """Release sandbox path; error clearly if it is not built."""
    exe = "sandbox.exe" if os.name == "nt" else "sandbox"
    p = REPO / "target" / "release" / exe
    if not p.exists():
        sys.exit(
            f"release sandbox not found at {p}\n"
            "build it first:  cargo build -p sandbox --release"
        )
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render(cfg: dict, backend: str, out: Path) -> None:
    """Run one headless capture into `out`."""
    env = dict(os.environ)
    env.update(cfg["env"])
    cmd = [
        str(sandbox_bin()),
        "--backend",
        backend,
        "--screenshot-clean",
        str(out),
    ]
    res = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    if res.returncode != 0 or not out.exists():
        sys.stderr.write(res.stderr[-2000:])
        sys.exit(f"[{cfg['name']}] capture failed (exit {res.returncode})")


def pixel_diff(a: Path, b: Path):
    """(mean, max) per-channel abs diff, or a marker tuple/None on problems.
    Kept optional so the SHA path stays zero-dep (only this branch needs PIL)."""
    try:
        from PIL import Image  # local import: only the tolerant path needs it
    except ImportError:
        return None
    ia = Image.open(a).convert("RGB")
    ib = Image.open(b).convert("RGB")
    if ia.size != ib.size:
        return ("size", ia.size, ib.size)
    pa, pb = ia.load(), ib.load()
    w, h = ia.size
    total = 0
    mx = 0
    for y in range(h):
        for x in range(w):
            r0, r1 = pa[x, y], pb[x, y]
            for c in range(3):
                d = abs(r0[c] - r1[c])
                total += d
                if d > mx:
                    mx = d
    return (total / (w * h * 3), mx)


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"configs": {}}


def save_manifest(m: dict) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Golden-image regression runner (F6).")
    ap.add_argument("--backend", default="metal", help="metal|vulkan|d3d12 (default metal)")
    ap.add_argument("--only", action="append", help="check only this config (repeatable)")
    ap.add_argument("--update", action="store_true", help="(re)generate SHA goldens")
    ap.add_argument("--save-png", action="store_true", help="with --update, also store PNG goldens")
    ap.add_argument("--mean-tol", type=float, default=0.5, help="tolerant mean abs-diff/ch")
    ap.add_argument("--max-tol", type=int, default=16, help="tolerant max abs-diff/ch")
    ap.add_argument("--workdir", default=None, help="render scratch dir (default: system temp)")
    args = ap.parse_args()

    work = (
        Path(args.workdir)
        if args.workdir
        else Path(os.environ.get("TMPDIR", "/tmp")) / "dc-golden"
    )
    work.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    manifest.setdefault("configs", {})
    manifest["backend_note"] = (
        "SHAs authored on the maintainer's box; exact match is the strict gate, "
        "pixel tolerance is the portable fallback."
    )

    selected = [c for c in CONFIGS if not args.only or c["name"] in args.only]
    if not selected:
        sys.exit(f"no config matches --only {args.only}")

    rows = []
    all_pass = True
    for cfg in selected:
        name = cfg["name"]
        # Skip content configs whose asset is missing (gallery has none).
        req = cfg.get("requires_asset")
        if req and not (REPO / req).exists():
            rows.append((name, "SKIP", f"missing asset {req}"))
            continue

        out = work / f"{name}.png"
        render(cfg, args.backend, out)
        got = sha256(out)

        if args.update:
            entry = {"sha256": got, "desc": cfg["desc"], "env": cfg["env"]}
            manifest["configs"][name] = entry
            if args.save_png:
                GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
                (GOLDEN_DIR / f"{name}.png").write_bytes(out.read_bytes())
            rows.append((name, "UPDATED", got[:16] + "..."))
            continue

        golden = manifest["configs"].get(name)
        if not golden:
            rows.append((name, "NO-GOLDEN", "run with --update to author"))
            all_pass = False
            continue

        want = golden["sha256"]
        if got == want:
            rows.append((name, "PASS", f"sha {got[:16]}..."))
            continue

        # Exact miss -- fall back to the tolerant pixel diff if a PNG golden exists.
        golden_png = GOLDEN_DIR / f"{name}.png"
        detail = f"sha {got[:12]}... != {want[:12]}..."
        if golden_png.exists():
            d = pixel_diff(golden_png, out)
            if d is None:
                all_pass = False
                rows.append((name, "FAIL", detail + " (PIL missing for tolerant diff)"))
            elif d[0] == "size":
                all_pass = False
                rows.append((name, "FAIL", detail + f" size {d[1]} vs {d[2]}"))
            elif d[0] <= args.mean_tol and d[1] <= args.max_tol:
                rows.append((name, "PASS~", f"mean {d[0]:.3f} max {d[1]} (sha differs)"))
            else:
                all_pass = False
                rows.append((name, "FAIL", f"mean {d[0]:.3f}/max {d[1]} over tol; {detail}"))
        else:
            all_pass = False
            rows.append((name, "FAIL", detail + " (no PNG golden for tolerant diff)"))

    if args.update:
        save_manifest(manifest)

    # --- report ---
    w = max(len(r[0]) for r in rows) if rows else 8
    print(f"\ngolden-image runner  backend={args.backend}  goldens={GOLDEN_DIR}")
    print("-" * 72)
    for name, status, detail in rows:
        print(f"  {name:<{w}}  {status:<10}  {detail}")
    print("-" * 72)

    if args.update:
        print("goldens updated. Review + commit tools/goldens/manifest.json.")
        return 0
    if all_pass:
        print("ALL PASS")
        return 0
    print("REGRESSION: one or more configs failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
