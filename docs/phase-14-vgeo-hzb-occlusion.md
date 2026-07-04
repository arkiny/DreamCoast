# Phase 14 — vgeo two-phase same-frame HZB occlusion

**Status: DONE (2026-07-05), verified DX≡VK / OFF≡ON on Windows RTX 2070 SUPER.** The last open
follow-up of the vgeo-as-default track (`docs/phase-14-vgeo-perf-default.md` §"Remaining
follow-ups"). Builds on the unified per-frame pass (`docs/phase-14-vgeo-unified-pass.md`).

## Problem & prior attempt

The unified vgeo pass rasters every cluster the LOD cut selects, then resolves per-pixel occlusion
in the R64 visibility buffer (`atomicMax` on the depth key). That is correct but rasters clusters
that are fully hidden behind nearer geometry. HZB occlusion culls those at the cut so the
rasterizer never touches them.

A first attempt (M4b, reverted `26cbab0`) bolted a **prev-frame** HZB test onto the (then
per-object) cut. It was perfectly conservative on Vulkan (0.000/ch) but **over-culled silhouette
clusters on D3D12 (0.013/ch)** — a non-conservative sampling divergence that violates the
`DX≡VK ≤0.001` hard rule. Deferred pending (a) a single clean insertion point and (b) a
genuinely-occluded scene. The single `csCutScene` from the unified-pass work provided (a).

## The load-bearing invariant

The R64 visibility buffer resolves the final per-pixel winner by `atomicMax` on the depth key,
**independent of which clusters rastered or in what order**. So if occlusion culling only ever
drops clusters with **no visible pixel** (a *conservative* test), the resolved G-buffer is
byte-identical whether culling is on or off, and whether DX and VK cull the same set or not.
Correctness — hence DX≡VK and OFF≡ON — follows from the test being conservative, **not** from the
two backends making bit-identical cull decisions. The two-phase structure makes the test
conservative *even under camera motion* (no 1-frame popping, unlike a prev-frame single-phase test).

This is why the M4b class cannot recur: the phase-2 test (`hzbOccluded`) is a **verbatim port of
the byte-verified P7 grid test** (`cull.slang::csCullHzb`, which already passes the Sponza-atrium
conservatism gate — `docs/hzb-occlusion-culling.md` §4), factored into the shared
`crates/shader/shaders/hzb_test.slang`. M4b diverged because it rolled its own cut-side test.

## Design (Nanite two-phase, applied to the unified pass)

`status[t]` — per-work, per-FIF, device-local, carried across frames as the "was visible last
frame" seed: `0` culled, `1` drawn/visible, `2` defer.

```
csCutSceneP1   W threads   run select_cluster (LOD+frustum+cone). write NEW status[t]:
                             !select           -> 0 CULLED
                             HW/large (binning) -> 1 DRAWN + append hw_list (always phase 1)
                             SW & was-visible   -> 1 DRAWN + append sw_list (phase-1 list)
                             SW & !was-visible  -> 2 DEFER
csClearScene   w*h         zero the R64 vis buffer
csRasterScene  W groups    raster the phase-1 SW list into the vis buffer   (reused unchanged)
vgeo_hwvis     indirect    HW/large clusters (binning), before the Hi-Z build
hzb_copy_vis   L0 texels   NEW: Hi-Z level 0 from the vis buffer (unpack depth key; empty->far)
csReduce       per level   max-reduce the chain                              (reused hzb_build)
csCutSceneP2   W threads   for status[t]==2: conservative HZB test on the cluster's world bounds
                            sphere. pass -> 1 DRAWN + append p2 list; fail -> 0 CULLED (dropped)
csRasterScene  W groups    raster the phase-2 survivors into the SAME vis buffer (no clear = merge)
vgeo_resolve   fullscreen  vis buffer -> G-buffer MRT + SV_Depth             (reused unchanged)
```

Binning: HW/large clusters always draw in **phase 1** (never occlusion-tested), so they contribute
to the Hi-Z that phase-2 tests the deferred SW clusters against. Only the SW sub-list is two-phased.

The per-FIF Hi-Z pyramid (R32Float, render/2 → 1×1, storage+sampled) is produced (phase-1 raster)
and consumed (phase-2 cut) in the same frame. It is **per-FIF** because — uniquely among the
buffers here — the pyramid must NOT be raced: a half-written level would report near depths and
over-cull. `status` and the vis buffer are output-invariant to races, so they need no such care.

Gating: `P14_VGEO_HZB` — default **ON** for non-gallery scenes when the vgeo producer is active,
OFF for the gallery byte anchor; `P14_VGEO_HZB=0/1` overrides. OFF is the original single-cut
`record_scene` verbatim (byte-identical). `VGEO_HZB_STATS=1` populates a host-visible
[deferred, occlusion-culled] counter (monotonic, no FIF race; zero cost when unset).

## Files

- **`hzb_test.slang`** (new, shared include) — `hzbProject` + `hzbOccluded`, the verbatim P7 test.
- **`hzb_copy_vis.slang`** (new) — `csCopyVis`: Hi-Z level 0 from the R64 vis buffer.
- **`vgeo_scene_cut.slang`** — `csCutSceneP1` (reuses the 160-B `PushConstants` + `status_buf`/
  `bin_enabled` in its spare tail) and `csCutSceneP2` (new 112-B `CutP2Push`); shared
  `cluster_world_bounds` + `select_cluster` untouched (OFF path byte-identical).
- **`vgeo.rs`** — per-FIF pyramid + `status`/`p2_list`/`p2_args` scratch + the phase-split
  pipelines; `record_scene(occlusion)` branches to the two-phase sequence; shared `cut_push`
  packer (the OFF path passes `status=0, bin_enabled=0` → the exact original bytes); `record_hwvis`
  factored out; occlusion stats.
- **`main.rs`** — `P14_VGEO_HZB` resolve + pass `occlusion`; `[vgeo-hzb]` stats log.
- **`build.rs`** — register `hzb_copy_vis_cs` / `vgeo_scene_cut_p1_cs` / `_p2_cs`; `hzb_test.slang`
  in `SHARED_INCLUDES`.

## Verification (RTX 2070 SUPER, 1280×720 `--screenshot-clean`, `AUTO_EXPOSURE=0`)

Real Intel New Sponza assets are absent in this environment (its `.level` falls back to the
placeholder), so the occlusion cull was exercised on a placeholder scene viewed along its object
row (`LEVEL=sponza.level CAM_EYE=8,2.6,0 CAM_TARGET=-2,1.8,0`, SW-only `P14_VGEO_BIN=0` so every
cluster takes the two-phase SW path), plus the gallery vgeo anchor.

| gate | result |
|---|---|
| **occluded view, D3D12** — phase-2 stats | **306 / 396 deferred clusters occlusion-culled** over 66 frames (cull genuinely engages) |
| occluded view, D3D12, **`P14_VGEO_HZB` OFF vs ON** | **0.000/ch, max 1, >8: 0.00%** (conservative — the direct M4b-blocker gate) |
| occluded view, **Vulkan** OFF vs ON | **0.000/ch, max 0** |
| occluded view, **DX(on) ≡ VK(on)** | **0.000/ch** (≤0.001 ✓) |
| binning ON (shipping default), D3D12 OFF vs ON | **0.000/ch, max 0** (two-phase coexists with the HW path) |
| gallery vgeo (`P14_VGEO=1`) OFF vs ON, DX & VK | **0.000/ch** each; DX≡VK 0.001 = the *pre-existing* binning-HW gap (OFF baseline is also 0.001), not introduced |
| `cargo clippy --all-targets -D warnings` / `cargo fmt` / `cargo test` | clean / clean / **all pass (71 + suites)** |

The decisive result: on the view where D3D12's phase-2 **culls 306 clusters as hidden**, the image
is byte-identical to culling-off. Culling engages *and* is conservative — the M4b over-cull does
not occur.

**DX and VK cull different amounts** (D3D12 306, Vulkan 0 on one view) yet both are OFF≡ON 0.000
and DX≡VK 0.000 — the invariant in action: the phase-1/phase-2 split is seeded by prev-frame
visibility (timing-dependent, so the *efficiency* varies by backend/frame), while the *output* is
identical for any conservative subset. Cull efficiency is a perf heuristic; correctness is not.

## Honest limitations / follow-ups

- **Net fps is ~flat on current content.** The frame is GDF-SW-RT bound (~25 ms; vgeo raster is
  sub-ms), so culling a few clusters saves a fraction of a millisecond. The value delivered is the
  conservative **same-frame scaffold** (correct under motion, ready for heavy GPU-driven scenes)
  and closing the DX≡VK blocker — not a measured frame-time win on today's scenes.
- **Validated on the placeholder + gallery, not real Intel Sponza** (assets absent here). The
  correctness argument rests on the conservative invariant + the verbatim P7 test port + the
  306-clusters-culled OFF≡ON gate; re-running on Intel New Sponza when its assets are present would
  add a heavier occlusion sample.
- **Binning HW clusters are not occlusion-tested** (always phase 1). Splitting the HW path into
  phase 2 (test large clusters too) is a bounded follow-up once a scene shows HW clusters dominating
  the hidden set.
- **`cull.slang` still has its own copy** of the test body; it could adopt `hzb_test.slang` for a
  single source of truth (left untouched here to avoid disturbing the verified grid path).
