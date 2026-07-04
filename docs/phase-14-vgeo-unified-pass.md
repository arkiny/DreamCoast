# Phase 14 — Unified per-frame virtual-geometry pass

**Status:** design (2026-07-04). Follow-up 1 of the vgeo-as-default track. Prereq context:
`docs/phase-14-vgeo-integration.md`, `docs/phase-14-vgeo-perf-default.md`, and the memory
`vgeo-windows-dx-vk-verify.md`.

## Problem

`VgeoSystem::record_object` records, **per object**, the chain
`cut → clear → raster → (hwvis → hwvis_barrier) → resolve` against a shared R64 visibility
buffer, serialized. For Sponza (~103 vgeo objects) that is ~5–6 passes × 103 ≈ **~515 graph
passes/frame**. Consequences:

1. **Per-object dispatch overhead** — 103× the pipeline binds / barriers / push-constant
   uploads for what is one scene's worth of clusters.
2. **`PROFILE_GPU` is unusable under vgeo** — the per-object passes overflow the GPU
   timer-query heap, so `main.rs` skips the headless profile dump when vgeo is on
   (`if self.screenshot_mode` guard near the profile dump). We literally cannot measure vgeo
   per-pass GPU cost, which blocks every perf follow-up (HZB, etc.).

The consolidated geometry (`GlobalGeom`, 4 shared buffers) is already scene-wide; what is
still per-object is the **cut / raster / resolve dispatch** and the **per-object push
constants** (`model`, `rec_base`, material). This doc removes that.

## Reference: UE5.7 Nanite `NaniteRasterBinning.usf`

Nanite never rasters per-object. Its whole-scene bin build (`RasterBinBuild` + the
INIT/COUNT/RESERVE/SCATTER/FINALIZE passes) operates on one flat **visible-cluster list**
produced by instance+cluster culling. The load-bearing idea for us:

- A visible-cluster entry does **not** carry a transform. It stores an **`InstanceId`**; the
  rasterizer/shader fetches the transform via `GetInstanceSceneData(InstanceId)`. One raster
  pass covers every cluster of every instance, each projected by its own instance transform
  read from a scene buffer (`SceneData.ush`).
- The per-pixel VisBuffer payload stores the **`VisibleClusterIndex`** (an index into that
  flat list) + `triId`, *not* the raw cluster/instance. Shading dereferences
  `VisibleClusters[idx] → (InstanceId, PageIndex, ClusterIndex)`.
- Counts are split **SW vs HW** per bin and turned into indirect dispatch/draw args by a
  separate finalize step (with the DispatchMesh 64k-per-dim wrapping) — i.e. binning is
  scene-wide with a single indirect dispatch, not one dispatch per object.
- Atomics are **wave-scalarized** (one atomic per unique bin per wave) to kill contention.

We do **not** need the material-bin machinery (per-cluster material ranges, depth buckets,
vert-reuse batching): our page is single-material and we route by object, not by per-triangle
material. We take the two structural ideas — **instance indirection** and **payload =
list-index** — and drop the rest.

## Design

### New data (all scene-wide, rebuilt per frame from `vgeo_draws`)

1. **Instance table** — one bindless storage buffer, `VgeoInstance` per vgeo draw:
   ```
   struct VgeoInstance {           // 128 B (16-B aligned)
       float4x4 model;             //   0  object → world
       float4   base_color;        //  64
       float4   mr;                //  80  x metallic, y roughness, z <pad>, w alpha_cutoff
       uint4    tex;               //  96  base_color / metallic-rough / normal / emissive
       // 112..128 pad
   }
   ```
   Replaces the per-object `model` + material push fields. `mip_bias` stays a per-frame
   scalar in push (global, not per-instance).

2. **Work list** — one bindless storage buffer, `uint2 (instance_id, global_cluster)` per
   (instance × cluster). Length `W = Σ_instances page.total_clusters` (identical to today's
   aggregate cut/raster thread count — no new work). Built on CPU (we already iterate
   `vgeo_draws`), so no GPU expansion pass is needed.

3. **Visible list** — unchanged stride (`uint`), but each slot now stores a **work index
   `t`** (into the work list), not a raw global cluster. SW-only: one list. Binning: SW list +
   HW list, exactly as today.

### Payload

`payload = (t << 7) | triId`, where `t` is the **work index** (stable, CPU-assigned). Since
`t < W ≤ ~10⁵ < 2²⁵`, it fits the 25 payload bits above `triId` (7 bits). Using `t` (not the
visible-list slot) makes the payload unambiguous across the SW and HW visible lists and
**independent of the atomic scatter order** — the winning payload per pixel is fully
depth-determined (`atomicMax` on the full-float32 depth key), so DX≡VK parity is preserved
(same argument as today; if anything stronger, because `t` no longer depends on cross-object
serialization order).

### Passes (per frame, total — not per object)

```
csCutScene       1 dispatch, W threads   → scatter work-index t into SW/HW visible list(s)
csClear          1 dispatch, w*h threads → zero the visibility buffer
csRaster         1 dispatch, W groups    → SW-raster visible SW list (sentinel-skip, as today)
vgeo_hwvis       1 indirect mesh draw     (binning only)
vgeo_hwvis_barrier 1 tiny compute         (binning only)
vgeo_resolve     1 full-screen pass       → G-buffer MRT + SV_Depth
```

~6 passes regardless of object count (was ~6·N). PROFILE_GPU fits the timer heap again.

### Shader changes

- **`vgeo_cut.slang`** — new `csCutScene` / `csCutSceneBin` entries: thread `t` reads
  `work[t] = (inst, gc)`, fetches `instances[inst].model` for `to_world` / cone axis /
  screen-size, runs the identical `select_cluster`, and appends **`t`** (not `gc`) to the
  SW/HW list. `model`/`max_scale`/`rec_base`/`cull_backface` leave the push (now per-instance
  from the table; `max_scale` derived in-shader from the model columns, `cull_backface` from a
  per-instance flag — add a `flags` field to `VgeoInstance` or fold into `mr`/`tex` spare).
  Push gains `work_buf`, `inst_buf`; keeps planes/cam/proj_factor/tau/total(→W)/bin_px.
- **`vgeo_swraster.slang`** (`csRaster`) — `t = vis_list[gid]`; `work[t] = (inst, gc)`;
  `model = instances[inst].model`; `mvp = view_proj_noflip * model`; `rec` at `gc`;
  `payload = (t<<7)|lane`; `cull_backface` per-instance. Push swaps per-object `mvp` for
  `view_proj_noflip` + `work_buf` + `inst_buf`.
- **`vgeo_hwvis.slang`** — same indirection: HW mesh reads `t`, derives per-instance
  `mvp` (Y-flipped `view_proj`) + `cull_mvp` (flip-free), writes `payload=(t<<7)|lane`.
- **`vgeo_gbuffer.slang`** (resolve) — `payload → t → work[t]=(inst,gc)`; `model`, material
  from `instances[inst]`; geometry from `gc`. Push swaps per-object `mvp`/`model`/material for
  `view_proj_noflip` + `work_buf` + `inst_buf` + global `mip_bias`.

### Rust changes (`vgeo.rs`, `main.rs`)

- `record_object` → **`record_scene`**: takes the whole `vgeo_draws` slice, uploads the
  instance table + work list into per-FIF buffers, records the ~6 passes once.
- `prepare`: one SW (+ one HW) visible list per FIF sized to `W`, one args buffer each — reset
  args `{0,1,1}` + sentinel list once, not per object.
- `main.rs` call site: replace the `for (slot, ...) in vgeo_draws` loop with a single
  `v.record_scene(&mut graph, gbuf, fif, &vgeo_draws, visbuf_ext, cull_view_proj, view_proj,
  eye, mip_bias, extent)`.

## Verification (measured 2026-07-04, RTX 2070 SUPER, 1280×720 `--screenshot-clean`)

All with `AUTO_EXPOSURE=0` (temporal exposure would drift the compare — see memory).

- **Gallery anchor byte-identical** (`P14_VGEO=1`): before vs after **VK 0.000000 (0 px)**,
  DX 0.000000 (1-LSB nondeterminism). DX≡VK 0.000002.
- **Sponza** (`CAM_EYE=0,4,0 CAM_TARGET=8,8,6`): before vs after 0.000008/ch (DX+VK), **DX≡VK
  0.000006/ch** (≤ 0.001 ✓). The ~8e-6 vs the old path is NOT a regression: the unified path is
  deterministic run-to-run (0.000000), and the residual is the **exact-depth-tie winner** —
  `atomicMax` breaks a byte-identical depth tie by the larger payload, and the payload encoding
  changed from global-cluster to work-index, so a different (equally-valid, same-depth) triangle
  wins ~1600 tie pixels. Depth itself is bit-identical (same per-instance matrices).
- **PROFILE_GPU dumps under vgeo** (the primary goal): the 5 scene passes (`vgeo_cut` ~0.065,
  `vgeo_raster` ~0.04, `vgeo_hwvis` ~0.25, `vgeo_hwvis_barrier` ~0, `vgeo_resolve` ~0.18 ms —
  ~0.55 ms total) are now measurable. The old per-object path recorded ~515 passes/frame and
  overflowed the 32-entry timer heap (raised to 128). vgeo's G-buffer stage is sub-ms; the scene
  is GDF-SW-RT-bound (~25 ms), so this is measurability + per-object-overhead removal, not an fps
  jump. Pass count for Sponza: **~515 → 5**.
- **135 tests pass**, clippy clean (`RUSTFLAGS=-D warnings`). No cook-format change (renderer-side
  only) — no re-cook.
- The viewer (`--vgeo-mesh`) and its shaders (`vgeo_cut`/`vgeo_swraster`/`vgeo_hwvis`/
  `vgeo_resolve`) are untouched; the scene path is new shaders (`vgeo_scene_*`) + the rewritten
  production-only `vgeo_gbuffer` resolve.

## Non-goals / follow-ups

- **Indirect compute dispatch in the graph** — `csRaster` still fixed-dispatches `W` groups +
  sentinel-skips (the graph IR recorder has no indirect compute dispatch). Group count = W =
  today's aggregate, so no regression; wiring true indirect compute (dispatch exactly the
  visible count) is a separate RHI/graph enhancement.
- HZB same-frame occlusion (follow-up 2) builds on this — a single scene cut is where an HZB
  test slots in cleanly.
</content>
</invoke>
