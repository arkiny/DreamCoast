# COLD-START PROMPT — Sponza lossless optimization to 60fps (paste into a fresh session)

You are continuing a graphics-engine performance track on **D:\Playground** (DreamCoast — a
from-scratch Rust renderer on raw Vulkan (`ash`) + raw D3D12 (`windows-rs`) behind one hand-rolled
RHI; no wgpu). Hardware: RTX 2070 SUPER (Windows). Read `CLAUDE.md` first, then these before doing
anything:

- **`docs/lossless-opt-plans.md`** — the four ready-to-implement, image-identical optimization designs
  (this is your work list). Full algorithms, file:line targets, expected ms, and per-design risk.
- Memory index `MEMORY.md`, especially **`sponza-1080p-60fps-track.md`** (the whole track history +
  gotchas), **`backend-verify-gotchas.md`**, **`unreal-engine-reference.md`** (UE5 Lumen ref paths).

## THE GOAL (never changes)

Sponza (`LEVEL=sponza`) at **1920×1080, RENDER_QUALITY=med, internal 0.6667 + TAAU**, **≥60fps
(≤16.6ms) on BOTH D3D12 and Vulkan**, at **MAXIMUM quality** — every optimization must be
**IMAGE-IDENTICAL / lossless** (pixel-identical to the max-quality baseline within the existing
stochastic gap; **NOT** a quality-reducing scalability knob like reflect_res_div / tile / ssao-off).
**DX and VK must use IDENTICAL settings → identical quality** (no backend-specific quality tuning;
perf-only plumbing like async that doesn't change the image is fine on one backend only).

Then: make **IntelSponza** (`LEVEL=sponza_intel`) hit the same fps. Also wanted: **frustum + occlusion
culling** and **distance-based mesh LOD** (the existing PR-7/PR-8 culling only touches a synthetic demo
grid — real scene draw loops have NO bounds/culling; see the cull-lod design in the track memory).

**Max-quality baseline (0.667+TAAU, no knobs): DX 25ms (40fps), VK 32ms (31fps).** Dominant cost:
`gdf_reflect` 10-12ms, `sdf_cache_light` 7-9ms, `gdf_ao` 3-4ms, `gi_volume` 2-5ms. The GDF SW-RT stack
is ~90% of the frame and thread/pixel-bound; that is WHY only lossless algorithmic wins (not knobs)
preserve quality. Combined best-case of the four designs (static/slow camera): plausibly DX→~12ms,
VK→~16-18ms at zero quality loss; fast motion falls back to full-quality baseline (never stale/wrong).

## IMPLEMENTATION ORDER (from docs/lossless-opt-plans.md — do as careful verified increments)

1. **SDF-march header hoist** (bit-identical CSE, safest). Verify the slang→DXIL/SPIRV compiler isn't
   already CSE-ing the per-step `ms_load_header` before investing — if it is, skip to #2.
2. **Coarse conservative min-distance mip** gating the dense+candidate loop to the near-surface band
   (the main per-step-cost win; image-identical; content-gated for the gallery anchor).
3. **Cache static-convergence dirty-skip** (EMA fixpoint; skip relight of converged cards; global
   sun/sky epoch). Land the carry-forward-copy-preserving MVP first.
4. **Adaptive/temporal reflect trace** (biggest single win, 10→~2ms static; **push-layout change =
   the DX≡VK parity risk** — new fields on 16-aligned rows, verify all 3 backends).
5. **Async raster-window overlap** (honest ceiling ~1.5ms VK; GDF compute is SM-saturated at max q).

Stack them and re-measure toward ≥60fps-both after each. Reference UE5 Lumen at
`D:/EpicGames/UE_5.7/Engine/Shaders/Private/Lumen/` and `D:/Repositories/UnrealEngine-1/Engine/Shaders/Private/Lumen/`.

## MEASUREMENT + VERIFICATION WORKFLOW (rebuild the harness — it lived in a per-session scratchpad)

Recreate a `measure.py` in your scratchpad: it runs `target/release/sandbox.exe --backend <b>
--screenshot-clean out.png` with the env below + `PROFILE_GPU=1 WARMUP_FRAMES=48`, captures stderr,
parses all `frame X.XXX ms` lines (median of the tail, drop first 3) and the last `GPU profile (total
…)` per-pass block. Build: `cargo build -p sandbox --release`.

- **Perf run env:** `LEVEL=sponza RENDER_QUALITY=med WINDOW_RES=1920x1080 RENDER_SCALE=0.6667 P_TAAU=1`
  (+ `ASYNC_COMPUTE=1` when testing async — headless needs it to build the compute queue, else
  `P_ASYNC_CACHE` is a SILENT no-op). Report per-pass ms + frame median, both backends.
- **On-disk `apps/sandbox/config/scalability.ron` is read at RUNTIME** → tier-knob edits take effect
  with NO rebuild (fast iteration). Shader/Rust changes need a rebuild.
- **Image-identical gate (the core check):** render both backends at a FIXED exposure to isolate the
  optimization from auto-exposure noise: `AUTO_EXPOSURE=0 EXPOSURE=8 WARMUP_FRAMES=200` + the perf env.
  Re-capture the golden once (pre-change build) as `REF_dx.png`/`REF_vk.png`. `tools/rt-compare.py
  a.png b.png diff.png` reports avg/max abs diff per channel. Each opt must diff **≤ the DX≡VK
  stochastic floor (~0.089/ch at EXPOSURE=8)** vs the pre-opt build (same backend) AND keep DX≡VK ≤
  that floor. A structural (non-stochastic) diff = the opt changed the image = not lossless → fix it.
- **Gallery byte-identical anchor:** render the default gallery scene (no LEVEL) both backends → must
  stay **0.001/ch** (the max SHA nondeterminism; VK is bit-identical). EVERY lossless opt must be gated
  so the gallery takes the legacy EXACT path (the `base = if gallery_scene { gallery_preset() } else
  { preset(tier) }` seam in main.rs; `apps/sandbox/src/quality.rs::gallery_preset()` pins the anchor).
- `cargo test` (123 passed / 3 ignored baseline). Clippy: **`RUSTFLAGS="-D warnings" cargo clippy
  --all-targets`** — do NOT use `cargo clippy -- -D warnings` (the rtk shell proxy mangles the `-- -D`
  passthrough and rustc errors "multiple input filenames"; it is NOT a code error).

## GOTCHAS (learned this track)

- **rtk proxy** rewrites `grep`/`git`/`cargo` in the Bash tool and mangles some args — prefer the
  **Grep tool** over shell grep; use the `RUSTFLAGS` clippy form above.
- **Sponza uses per-mesh direct-sample:** `clipmap.slang` `ClipMap.count == 0` → `cm_geo_*` delegate
  to a single `ms_geo(c.desc, …)`, so the SDF header is CONSTANT per thread (header-hoist is clean).
- **auto-exposure is default-ON for content** (`f1f68e4`): its adaptive histogram metering amplifies
  the content stochastic gap to ~0.14/ch DX≡VK — that's why lossless verification uses FIXED exposure.
- **Push-constant layout is the recurring DX≡VK parity bug:** a scalar packed right after a float3
  lands at +12 on HLSL/SPIRV but pads to +16 on MSL. Keep new fields on 16-byte-aligned rows and
  re-verify all three backends (see the `ground_albedo`/`cone_k` float4 comments in gdf_reflect.slang).
- **Committed goldens in `tools/goldens/` are Metal-authored** — do NOT cross-match Windows DX/VK byte
  SHAs; use the ≤0.001-tolerant diff.
- D3D12 gallery has ~1-LSB run-to-run nondeterminism (byte SHA gate flakes; the 0.001 tolerant diff is
  the real gate).

## ALREADY COMMITTED THIS TRACK (on main — do not redo)

`90b3f1b` fix(cluster) + `2201ff3` fix(atmosphere) — DX/VK parity bugs in the merged PR-1..9 work.
`c3dd369` perf(rhi-d3d12) descriptor-heap hoist (image-identical). `f1f68e4` fix(lighting) brightness
(auto-exposure default-on for content + GI sky-fill via procedural_sky — USER-APPROVED). `409b2c1`
async cache default-on for D3D12 content. `96dbb38` gi_volume frame-amortization (opt-in
`P_GI_VOLUME_PERIOD`, default 1 = off). `c5589bb` **reverted the Med scalability-knob cuts back to full
quality** (the pivot to lossless-only). `076aa9f` the lossless design doc. Start from #1 above.
