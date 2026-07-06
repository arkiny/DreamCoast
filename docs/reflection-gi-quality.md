# Reflection quality in large scenes — GI-lit hits + trace hierarchy

Upstream: [swrt-gi-perf-track.md](swrt-gi-perf-track.md) · [per-mesh-distance-fields.md](per-mesh-distance-fields.md).
Branch: `feature/lumen-style-reflection-gi` (off `main`).

## Problem

A polished-metal object (the vertex-cache knight) in a large content scene (Intel Sponza)
rendered a **near-black silhouette** even standing in a bright sunlit strip. Two diagnostics
localised it: the captured-cube IBL specular (`P11_LEGACY_IBL=1`) reflected the object bright, but
the default SW-RT reflection left it black — even at `WARMUP_FRAMES=2`, so it was **not** temporal
accumulation. The raw `gdf_reflect` trace was returning ≈0 radiance for the object's pixels.

Root cause (three compounding failures, all specific to a large scene traced against one coarse
global distance field):

1. **GI-less hit re-light.** On a surface-cache MISS, `gdf_reflect.slang` analytically re-lit the
   reflected hit with only `sun·NdL·shadow + a crude sky fill`. A reflected **shadowed** wall
   (mid-grey from GI in the actual frame) re-lit to near-black — the reflection omitted the whole
   indirect/skylight term the real surface receives.
2. **Cache coverage.** The surface cache would supply GI-lit radiance, but (a) the hit position from
   marching a coarse 48³ field is imprecise, so it fails the cache's tight on-surface tolerance, and
   (b) the card budget (`MAX_CARDS/6 ≈ 170`) is below the scene's drawable count, so many surfaces
   have no card at all → the analytic fallback runs.
3. **One coarse field.** A single 48³ global distance field over a ~30 m scene is ~0.6 m/voxel;
   thin geometry (curtains) is not represented, and near hits are imprecise.

Reference engines solve this by (a) making reflection ray hits sample a persistent **surface cache**
holding *final lit radiance* (direct + indirect GI + emissive), never an analytic re-light; misses
fall back to a **radiance cache / skylight**, never black; and (b) a **trace hierarchy** — screen
trace → per-mesh distance field → **global-distance-field clipmaps** (fine near camera, coarse far)
→ radiance cache — so large scenes stay traceable and thin/near geometry stays sharp.

## Fixes

### Fix 1 — GI-lit reflection hits (LANDED, Metal-verified)

Feed the engine's existing world-space directional-irradiance volume (the radiance cache written by
`gi_volume.slang`, reconstructed in `gdf_gi.slang`) into the reflection hit's indirect term.
`gdf_reflect.slang` gains `sample_gi_irradiance()` (the same SH reconstruction as the GI pass); the
analytic fallback becomes `albedo · ((sun·NdL·shadow)/π + E_indirect)` when the volume is bound.

- **Seam:** the volume is bound only for content (`P_GI_VOLUME` defaults on for non-gallery); the
  gallery keeps the exact legacy `sky_fill` expression → **byte-identical anchor `af70c1a5`**.
- **Packing:** the volume base index rides in the reflect push's `flip_y` spare bits (bit 0 stays the
  Y-flip); the 240-byte push is full and the D3D12 root budget forbids growing it.
- **Result:** the knight goes from a black silhouette to polished steel that reflects its GI-lit
  surroundings (bright where it reflects the sunlit floor/curtains, dark where it reflects shadowed
  stone). Correctly darker than the sky-cube IBL test, which ignored local occlusion.

### Fix 2 — surface-cache coverage for coarse hits (LANDED, Metal-verified)

`sample_surface_cache` gains an `extra_tol` bias (world units). The primary consumers (GI gather,
cache re-light, viz) pass `0.0` → byte-identical. The reflection passes ~half a coarse voxel
(`0.006 · scene_diagonal`) for content only, so an imprecise coarse-field hit still matches the card
of the surface it grazed and reads its precise multibounce radiance instead of the analytic fallback.
Card-budget scaling for large scenes is a follow-up (memory/relight-cost tradeoff; not changed here).

### Fix 3 — trace hierarchy (clipmap) — structural

The multi-resolution **clipmap** field already exists as opt-in (`P11_GDF_CLIP_LEVELS=N`, default 1 =
the single 48³ volume); the reflection march already samples it via `reflect_clip()`. Enabling a
clipmap gives content reflections a finer near-camera field (sharper hits, better thin-geometry
capture) at bounded far cost — the reference-engine clipmap approach. Remaining structural work to
make this the default for large content scenes:

- Re-baseline the content goldens (`sponza_gdf_ao`, `sponza_sc_viz`) once the clipmap is on by
  default for those scenes (the gallery stays single-level, anchor unchanged).
- Thin geometry (cloth/curtains): capture via a screen-trace first bounce and/or per-mesh distance
  fields (`P11_DIRECT_SDF`) so it is reflected even where the coarse field drops it.
- Scale the surface-cache card budget with scene drawable count.

## Verification

- Gallery anchor `af70c1a5` byte-identical after Fix 1 and Fix 2 (Metal). Clippy clean.
- Intel Sponza knight: black → GI-lit polished steel.
- **DX≡VK parity is a Windows follow-up** — the shader and push changes are backend-uniform, but
  reflection radiance was only verified on Metal here.
