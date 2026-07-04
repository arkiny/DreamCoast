# Phase 14 — vgeo near-plane clipping (UE Nanite-referenced)

**Status: DONE (2026-07-05), verified on Windows RTX 2070 SUPER.**

## Problem

Flying the vgeo camera close to geometry dropped triangles one at a time. The vgeo **software
rasterizer** (`vgeo_scene_raster.slang::csRasterScene`) had a *crude* near guard —
`if (clip.w <= 1e-6) return;` — that **discarded the whole triangle** the instant any vertex crossed
behind the camera, instead of clipping it. Masked in normal use (the default binning path sends
large/near clusters to the HW mesh rasterizer, which clips correctly); exposed only when the SW path
handled a near-crossing triangle — `P14_VGEO_BIN=0`, or an adapter without mesh shaders.

## How UE Nanite handles it

Studied `D:/EpicGames/UE_5.7/Engine/Shaders/Private/Nanite/`. Nanite does **not** clip in its
software rasterizer. Cluster culling flags a cluster whose bounds cross the near plane
(`FFrustumCullData.bCrossesNearPlane → bNeedsClipping`, `NaniteCullingCommon.ush`) and **forces it
to the hardware rasterizer**:

```
NaniteClusterCulling.usf:869   bUseHWRaster |= Cull.bNeedsClipping;
```

The HW fixed-function rasterizer clips at the near plane for free; the SW rasterizer only ever sees
triangles guaranteed in front of the near plane.

## Solution (Phase A + B)

### Phase A — route near-crossing clusters to HW (the Nanite way)

`vgeo_scene_cut.slang`: a `crosses_near(bnd_c, bnd_r)` helper tests the cluster's world bounds sphere
against the near plane (`planes[4]` — the standard-Z `z>=0` near plane from `push::frustum_planes`).
In the binning cut (`csCutSceneBin` and the two-phase `csCutSceneP1`) the HW/SW split becomes
`useHW = (screen_diam >= bin_px) || crosses_near(...)` — mirroring `bUseHWRaster |= bNeedsClipping`.
Since a cluster's bounds sphere contains all its triangles, this routes **every** near-crossing
triangle to HW. The HW mesh path (`vgeo_scene_hwvis.slang`) already near-clips via the fixed-function
rasterizer and writes `i.pos.z`; the resolve reconstructs attributes robustly (below). No Rust change
(the cut already receives the world frustum planes). Fully fixes every mesh-shader-capable adapter.

### Phase B — real near-plane clipping in the SW rasterizer (SW-only adapters)

For adapters without mesh shaders (binning off → no HW list to route to), `csRasterScene` now
**clips** instead of dropping:

1. Fetch the 3 clip-space verts (homogeneous). Near-plane inside test = `clip.z >= 0` (standard-Z,
   matches HW). All-in-front → the original fast path (byte-identical). All-behind → drop.
2. Otherwise Sutherland-Hodgman against the single near plane, interpolating in clip space
   (`t = va.z/(va.z-vb.z)`) → a 3- or 4-vertex polygon, all `z>=0` hence `w>0`.
3. Fan-triangulate into 1–2 sub-triangles, each carrying the **original** `triId`, rasterized by a
   shared `rasterSceneTri` helper.

The resolve (`vgeo_gbuffer.slang::fsGBuffer`) is **unchanged**: it already reconstructs attributes
with a homogeneous, pre-divide perspective-correct barycentric (Schied & Dachsbacher / the method
Nanite uses) that is explicitly robust to `clip.w <= 0`. So the sub-triangles only need to bound
coverage + depth; attributes come from the original triangle. Depth stays consistent because the
sub-triangles share the original triangle's plane.

With Phase A on (binning), near-crossers go HW and Phase B's clip is dormant; it activates only in
SW-only mode (and as a defensive fallback). Clean separation.

## Verification (RTX 2070 SUPER, 1280×720 `--screenshot-clean`, `AUTO_EXPOSURE=0`)

| gate | result |
|---|---|
| No-reg, normal framing — gallery `P14_VGEO=1` (DX & VK) + sponza row (SW-only), new vs `508a403` | **0.000/ch, >8: 0.00%** (fast path byte-unchanged) |
| Near-crosser close-up (objects filling frame, well inside near range): **SW (Phase B) ≡ HW (Phase A)**, D3D12 | **0.000/ch, >8: 0.00%** — the SW clip byte-matches the HW fixed-function clip |
| Near-crosser close-up: **SW D3D12 ≡ SW Vulkan** | **0.000/ch** |
| Visual | close objects render **solid** (no missing wedges) in SW-only and binning alike |
| `cargo clippy --all-targets -D warnings` / `cargo fmt` / `cargo test` | clean / clean / all pass |

The decisive result: at an extreme close-up where objects fill the frame (so triangles certainly
cross the near plane), the SW compute rasterizer's Sutherland-Hodgman clip produces **byte-identical**
output to the HW mesh rasterizer's fixed-function clip. Had Phase B still dropped triangles, the
close objects would show large holes and SW≠HW — instead they match to 0.000.

## Scope / follow-ups

- No Rust or `build.rs` change; `vgeo_gbuffer.slang` unchanged.
- The `--vgeo-mesh` **debug viewer** keeps its own copies (`vgeo_cut`/`vgeo_swraster`/`vgeo_resolve`)
  and still has the crude drop; the same fix applies but its resolve robustness must be checked
  first. It is a debug tool, not the production path — left as an optional follow-up.
