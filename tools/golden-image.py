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
    3. **PT residual budget (F6 Part B, F6E gate).** A config with ``pt: True``
       renders TWICE from the same fixed camera — raster, then
       ``P8_PATHTRACE=1`` — and gates on the **lit-masked 64-px block-mean**
       raster-vs-path-tracer residual from ``rt-compare.py --lit-mask --json``
       (``block64_avg <= block64_budget``, improved-or-neutral). Block means
       measure energy allocation at the scale the GI representation can
       express; the retired per-pixel ``masked_avg`` billed sub-block
       misalignment in full (a ~28 scatter pedestal) and is kept as a shadow
       metric with the F6D bias/scatter decomposition
       (docs/phase-f6e-scatter-tolerant-gate-plan.md). The lit mask (PT luma >
       ``lit_eps``) excludes the PT-dim region that light paths within the
       bounce budget cannot reach; ``pt_black_frac`` is reported for coverage,
       never gated. There is no SHA/PNG golden for these configs — the budget
       IS the regression. Budgets are seeded from measurement (--update) and
       only re-baselined downward on a verified improvement.

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
# Cached PT references (gitignored, big). The PT capture is ~15 min per config and dominated
# the runner, but the PT image is a PURE FUNCTION of the capture recipe + the path-tracer
# sources -- a raster-side change (the common iteration) cannot alter it. So cache it keyed by
# exactly those inputs and re-render only when they change (or with --pt-refresh).
PT_CACHE_DIR = GOLDEN_DIR / "ptcache"
PT_SOURCES = [
    "crates/shader/shaders/rt_path.slang",
    "crates/shader/shaders/rt_common.slang",
    "crates/shader/shaders/rt_trace.slang",
    "crates/shader/shaders/rt_pipeline.slang",
]

# --- Capture manifest ---------------------------------------------------------
# Each config is a fixed-camera, deterministic headless capture. `env` extends
# the process environment; `name` is both the golden key and the PNG basename.
# The gallery anchor uses the default scene (no LEVEL) -- it is the invariant
# byte-identity gate from CLAUDE.md and must match `af70c1a5...`.
#
# THIS LIST IS THE SINGLE SOURCE for every non-measured manifest field: --update
# rebuilds each selected entry wholesale from the recipe here plus the fresh
# measurement, so do not hand-edit manifest.json (edits are overwritten).
#
# Content configs render the interior colonnade of the content scene from a
# fixed camera with a 64-frame warmup so caches converge to a stable image:
#   - sponza_sc_viz : the high-res surface-cache visualization (P_SC_VIZ)
#   - sponza_gdf_ao : the distance-field AO debug view (DEBUG_VIEW=9)
# AUTO_EXPOSURE=0 is load-bearing for the strict SHA gate: auto-exposure defaults
# ON for non-gallery scenes even under a fixed EV100, and its metering EMA used to
# adapt on wall-clock dt — the run-to-run trajectory difference during the
# adaptation window is what the content tier's TAAU history baked into the capture
# (the historical gdf_ao/sc_viz SHA flicker, up to ~36/255 at high-contrast edges).
# The engine now adapts on FIXED_DT in screenshot mode, but a fixed-EV regression
# capture should not meter at all; with AE off these configs are byte-stable.
CONTENT_CAM = {
    "LEVEL": "sponza_intel",
    "EV100": "11",
    "AUTO_EXPOSURE": "0",
    "WARMUP_FRAMES": "64",
    "CAM_EYE": "-14,2,0",
    "CAM_TARGET": "14,2,0",
}
# PT-residual capture recipe (F6 Part B). Differences from CONTENT_CAM are all
# load-bearing:
#   - AUTO_EXPOSURE=1, no EV100: the PT reference carries raw radiance and is
#     exposed at the tonemap from the SAME adapted-exposure buffer the raster
#     lighting bakes in (F6 Part A) -- a fixed EV100 crushes interiors to black
#     (the documented content-PT trap). EV100/EXPOSURE are also STRIPPED from
#     the inherited shell env for pt configs (they still perturb the raster's
#     firefly clamps under AE).
#   - RENDER_SCALE=1: the content tier defaults to a 0.67x internal resolution
#     with TAAU, whose per-frame sub-pixel jitter is folded into the projection
#     the PT rays consume -- native scale keeps the PT camera truly fixed and
#     the accumulation at capture resolution.
#   - WARMUP_FRAMES=192: the auto-exposure EMA adapts on wall-clock dt, so the
#     fast raster capture needs ~192 frames to converge the metered exposure to
#     the same fixed point the slow PT capture reaches almost immediately; the
#     PT side banks (warmup+1) x path_spp=8 ~= 1544 spp of accumulation, which
#     also sinks the Monte-Carlo noise floor under the lit-mask epsilon.
PT_CAM_BASE = {
    "LEVEL": "sponza_intel",
    "AUTO_EXPOSURE": "1",
    "RENDER_SCALE": "1",
    "WARMUP_FRAMES": "192",
}
# Env vars a pt config must NOT inherit from the calling shell (fixed-exposure
# trap + firefly-clamp perturbation; see PT_CAM_BASE).
PT_ENV_STRIP = ("EV100", "EXPOSURE")
# Budget seeding margin (abs, per channel). Historically this absorbed the
# content captures' run-to-run spread, whose actual root cause was the
# auto-exposure EMA adapting on wall-clock dt (fixed 2026-07-14: screenshot mode
# adapts on FIXED_DT, so same-box captures are now frame-deterministic); the
# margin is kept for cross-box/driver variance and future content drift.
# Validated against a measured two-run spread when the budgets were seeded; a
# budget only moves DOWN on a verified improvement (re-run --update and note
# the measured value in the commit).
PT_BUDGET_MARGIN = 0.3
CONFIGS = [
    {
        "name": "gallery",
        "desc": "gallery anchor (default scene, byte-identity invariant; default anisotropy 16)",
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
    {
        "name": "sponza_pt_sunlit",
        "desc": "content: raster-vs-PT lit-masked residual, sunlit atrium (primary fidelity gate)",
        # Camera seeded by measurement: courtyard center looking up the atrium --
        # lowest pt_black_frac of the 3-candidate sweep (0.19 vs 0.52/0.39; see
        # docs/phase-f6b-content-pt-residual-plan.md).
        "env": {**PT_CAM_BASE, "CAM_EYE": "0,2,0", "CAM_TARGET": "-12,9,0"},
        "requires_asset": "assets/IntelSponza",
        "pt": True,
        "lit_eps": 8,
    },
    {
        "name": "sponza_pt_interior",
        "desc": "content: raster-vs-PT lit-masked residual, interior colonnade (tolerant gate)",
        # The long-standing content measurement camera (F1 continuity); high
        # pt_black_frac by construction, so the mask carries fewer samples.
        "env": {**PT_CAM_BASE, "CAM_EYE": "-14,2,0", "CAM_TARGET": "14,2,0"},
        "requires_asset": "assets/IntelSponza",
        "pt": True,
        "lit_eps": 8,
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


def render(cfg: dict, backend: str, out: Path, extra_env=None) -> None:
    """Run one headless capture into `out`."""
    env = dict(os.environ)
    if cfg.get("pt"):
        for k in PT_ENV_STRIP:
            env.pop(k, None)
    env.update(cfg["env"])
    if extra_env:
        env.update(extra_env)
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


def rt_compare(raster: Path, pt: Path, montage: Path, lit_eps: float) -> dict:
    """Run tools/rt-compare.py with the lit mask and return its JSON metrics.

    Subprocess + the RTCOMPARE_JSON line is the stable contract -- the human-
    readable lines belong to other consumers (they regex the default output)
    and are not parsed here.
    """
    cmd = [
        sys.executable,
        str(TOOLS_DIR / "rt-compare.py"),
        str(raster),
        str(pt),
        str(montage),
        f"--lit-mask={lit_eps:g}",
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if line.startswith("RTCOMPARE_JSON "):
            return json.loads(line[len("RTCOMPARE_JSON ") :])
    # No JSON line: PIL missing, zero lit pixels, or size mismatch -- surface
    # rt-compare's own message, this is a FAIL not a skip.
    raise RuntimeError(
        f"rt-compare produced no metrics (exit {res.returncode}): "
        + (res.stdout + res.stderr).strip()[-500:]
    )


def pt_cache_key(cfg: dict, backend: str) -> str:
    """Identity of a cached PT reference: the capture recipe (env + backend) plus the
    path-tracer sources. Anything that can change the PT image is in here; anything NOT in
    here (raster shaders, lighting, TAAU, ...) provably cannot, so the cache is safe to reuse."""
    h = hashlib.sha256()
    h.update(cfg["name"].encode())
    h.update(backend.encode())
    for k in sorted(cfg["env"]):
        h.update(f"{k}={cfg['env'][k]}".encode())
    h.update(b"P8_PATHTRACE=1")
    for rel in PT_SOURCES:
        f = REPO / rel
        h.update(f.read_bytes() if f.exists() else b"<missing>")
    return h.hexdigest()


def check_pt(
    cfg: dict, backend: str, work: Path, manifest: dict, update: bool, pt_refresh: bool = False
):
    """Render the raster+PT pair for a `pt: True` config and gate the lit-masked
    residual against the recorded budget. Returns a report row."""
    name = cfg["name"]
    raster_out = work / f"{name}_raster.png"
    pt_out = work / f"{name}_pt.png"
    montage = work / f"{name}_montage.png"
    render(cfg, backend, raster_out)
    # PT reference: reuse the cache when the recipe + path-tracer sources are unchanged.
    key = pt_cache_key(cfg, backend)
    cached_png = PT_CACHE_DIR / f"{name}.png"
    cached_key = PT_CACHE_DIR / f"{name}.key"
    pt_cached = (
        not pt_refresh
        and cached_png.exists()
        and cached_key.exists()
        and cached_key.read_text().strip() == key
    )
    if pt_cached:
        pt_out.write_bytes(cached_png.read_bytes())
    else:
        render(cfg, backend, pt_out, extra_env={"P8_PATHTRACE": "1"})
        PT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_png.write_bytes(pt_out.read_bytes())
        cached_key.write_text(key)
    try:
        m = rt_compare(raster_out, pt_out, montage, float(cfg.get("lit_eps", 8)))
    except RuntimeError as e:
        return (name, "FAIL", str(e), False)

    # A near-empty lit mask means the metric has no sample to stand on; the
    # usual cause is a crushed PT exposure (the AUTO_EXPOSURE trap).
    if m["pt_black_frac"] >= 0.9:
        return (
            name,
            "FAIL",
            f"pt_black_frac {m['pt_black_frac']:.2f} -- lit mask nearly empty; "
            "check AUTO_EXPOSURE=1 in the recipe",
            False,
        )

    # F6E gate (docs/phase-f6e-scatter-tolerant-gate-plan.md §4b): the gated residual is
    # the 64-px BLOCK-MEAN lit residual — energy misallocation at the scale the GI
    # representation can express. Per-pixel masked_avg punished sub-block structural
    # misalignment in full (a ~28 scatter pedestal on both gates), which blocked
    # E-domain-correct repairs while favouring flat fills; it is now a SHADOW metric,
    # as are the F6D bias/scatter decomposition, the finer block sizes, and the
    # (AE-confounded) dark track.
    detail = (
        f"block64 {m['block64_avg']:.3f}  pt_black {m['pt_black_frac'] * 100:.1f}%  "
        f"lit_mean r/pt {m['lit_mean_raster']:.1f}/{m['lit_mean_pt']:.1f}"
    )
    detail += f"  masked_avg {m['masked_avg']:.2f}"
    detail += "  [pt cached]" if pt_cached else "  [pt rendered]"
    if "masked_bias" in m:
        detail += f"  bias {m['masked_bias']:+.2f}/scatter {m['masked_scatter']:.2f}"
    if update:
        # Down-only ratchet, enforced: a re-seed may lower a budget (verified
        # improvement) or leave it, never raise it — a config whose measured value
        # drifted up re-seeds against the OLD budget and keeps failing until the
        # regression is fixed or the trade is explicitly re-adjudicated.
        old = manifest["configs"].get(name, {})
        budget = round(m["block64_avg"] + PT_BUDGET_MARGIN, 2)
        if "block64_budget" in old:
            budget = min(budget, old["block64_budget"])
        manifest["configs"][name] = {
            "desc": cfg["desc"],
            "env": cfg["env"],
            "pt": True,
            "lit_eps": cfg.get("lit_eps", 8),
            "block64_budget": budget,
            "block64_measured": m["block64_avg"],
            "pt_black_frac": m["pt_black_frac"],
            # Shadow metrics (informational; no budget): the retired per-pixel gate,
            # the F6D decomposition, the finer blocks and the dark track.
            "masked_avg": m["masked_avg"],
            "residual_bias": m.get("masked_bias"),
            "residual_scatter": m.get("masked_scatter"),
            "block16_avg": m.get("block16_avg"),
            "block32_avg": m.get("block32_avg"),
            "block64_dark": m.get("block64_dark"),
        }
        return (name, "UPDATED", detail, True)

    golden = manifest["configs"].get(name)
    if not golden or "block64_budget" not in golden:
        return (name, "NO-GOLDEN", "run with --update to seed the budget", False)
    budget = golden["block64_budget"]
    if m["block64_avg"] <= budget:
        return (name, "PASS", f"{detail}  <= budget {budget}", True)
    return (name, "FAIL", f"{detail}  OVER budget {budget}", False)


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
    ap.add_argument(
        "--pt-refresh",
        action="store_true",
        help="force re-rendering the cached PT references (use after a path-tracer change)",
    )
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

        # PT-residual configs (F6 Part B): budget-gated raster-vs-PT pair, no SHA.
        if cfg.get("pt"):
            row_name, status, detail, ok = check_pt(
                cfg, args.backend, work, manifest, args.update, args.pt_refresh
            )
            rows.append((row_name, status, detail))
            if not ok:
                all_pass = False
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
        return 0 if all_pass else 1
    if all_pass:
        print("ALL PASS")
        return 0
    print("REGRESSION: one or more configs failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
