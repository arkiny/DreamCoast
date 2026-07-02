# Cached / Virtual Shadow Map — dirty-skip design (IntelSponza `shadow` 14ms)

Applies the A2/A3 **"cache-when-converged, skip-when-static"** philosophy
([lossless-opt-ledger.md](lossless-opt-ledger.md)) to the directional **shadow** pass — the
#1 GPU pass on IntelSponza (`LEVEL=sponza_intel`) at **14.1ms on both DX and VK**. The shadow
pass re-rasterizes the whole caster set from the sun's POV every frame; when the sun and geometry
(and, for camera-fit cascades, the camera) are unchanged, that re-render reproduces a
**bit-identical depth buffer** — so we skip it and re-sample last frame's map. Image-identical when
static; invalidate on change. No shader change for S1/S2.

## Investigation — what the shadow map actually depends on (cited)

### Two shadow paths exist; only the legacy one is on by default

- **Legacy single map** (default; `CSM` unset -> `CsmConfig::enabled == false`, `csm.rs:88-98`,
  `csm.rs:112-137`). Rasterized by `Deferred::record_shadow` (`deferred.rs:701-751`) into a single
  `SHADOW_SIZE = 2048` (`main.rs:136`) square depth target (`main.rs:4562-4564`). The light
  view-projection is `light_view_proj(sun_dir, shadow_center, scene_radius)` (`main.rs:4299`, fn at
  `main.rs:6814-6829`). **This matrix reads ONLY `sun_dir`, `shadow_center` and `scene_radius`.**
  `shadow_center = self.scene_center` for a level (`main.rs:4294-4298`), and `scene_center` /
  `scene_radius` are fixed scene bounds (`main.rs:3062-3063`, computed once from the level AABB).
  **=> the legacy shadow map is already 100% camera-independent — it depends only on the sun.**
- **CSM / atlas** (opt-in `CSM=<N>`, N<=`MAX_CASCADES=4`, `csm.rs:46`). Rasterized by
  `record_shadow_atlas` (`deferred.rs:858-914`) into an atlas of N tiles (grid 1->1x1, 2->2x1,
  3/4->2x2; `csm.rs:141-153`), one `set_viewport_scissor_rect` + caster loop per cascade. The
  cascade matrices come from `compute_cascades(cfg, ViewCamera{eye,target,...}, sun_dir)`
  (`main.rs:4305-4320`, fn `csm.rs:176-304`). **These DO read the camera** (`cam.eye`, `cam.target`
  at `csm.rs:181,197,207`) — the cascades are **camera-fit**.

### The draw loop (both paths)

Per shadow-casting `SceneObject`: pick static / skinned / morph shadow pipeline, push
`shadow_push(light_vp * obj.transform, base_color_tex, alpha_cutoff, skin, morph)`, bind vbuf(32)/
ibuf, `draw_indexed(index_count,0,0)` (`deferred.rs:718-747` and `:886-909`). Vertex shader
`shadow.slang:49-56` transforms by `light_mvp`; fragment `shadow.slang:122-138` is depth-only + an
alpha-test discard for masked casters. Cost on IntelSponza is **raster/geometry throughput** — many
opaque draws over the whole-scene directional frustum — which is why the ledger notes DX~=VK here
(raster-bound, not GDF-compute-bound) and why *skipping the whole pass* (not cheaper draws) is the
lever.

### The map persists in the pool across frames — skipping the writer keeps last frame's depth

The shadow depth is a **transient pooled depth** (`create_depth`, `render/lib.rs:232-241`); depth is
**never aliased** (`render/lib.rs:532-533`) and the pool caches realized targets across frames
(`ResourcePool::acquire_depth`, `render/lib.rs:1233-1249`, first-fit on `!used && extent==`).
`begin_frame` only flips `used=false` (`render/lib.rs:1206-1213`) — it does **not** free or clear.
`SHADOW_SIZE=2048` (and the `CSM_ATLAS` square) is a **unique depth extent** — every other graph
depth is a different size (`g_depth`/`cull_depth` = swapchain extent `main.rs:4555`,`cull.rs:299`;
`sv_g_depth` = env-cube size `main.rs:6334`; env capture `ibl.rs:269`). So the shadow map's pool
slot is **deterministic and stable frame-to-frame**, and its physical texture holds last frame's
rendered depth until something re-writes it.

Crucially, `record_pass` already anticipates this reuse: *"a shadow map reused across frames may be
in shader-read from the prior frame"* (`render/lib.rs:895-896`) — the pass transitions the depth from
shader-read back to depth-target on write. The sampling side reads the map via
`ctx.sampled_index(shadow_map)` in `record_lighting` (`deferred.rs:1147,1182`; barrier depth->sampled
at `render/lib.rs:842-844`).

**Consequence for caching:** if we **do not call `record_shadow`/`record_shadow_atlas`** on a frame,
but still list `shadow_map` in the lighting pass `reads`, the graph acquires the *same* pool slot
(it is the only depth of that extent), finds no writer (`depth_first_writer` has no entry,
`render/lib.rs:598-603`) so issues **no clear**, and the lighting pass barriers it depth->sampled and
samples last frame's depth **bit-identically**. This is the entire mechanism — no new persistent
resource type is required for S1/S2.

### The A2 epoch pattern to mirror

`cache_settled` at `main.rs:4770-4799`: FNV-1a hash of the invariant inputs (sun dir/intensity, sky
gain/wb, relight spp/period/alpha, cone_k — **not** the camera); if the hash equals `self.cache_epoch`
bump `cache_stable_frames`, else reset epoch and counter; "settled" once
`cache_dirty_skip && captured && !reset && stable_frames >= settle_frames`. Fields declared at
`main.rs:867-882`, initialized `main.rs:2802-2816` (`P_CACHE_DIRTY_SKIP`, `P_CACHE_SETTLE`). We mirror
this verbatim for a `shadow_epoch`.

## 1. Camera-fit vs world-stable — recommendation

- The **default (legacy) map is already camera-independent**, so a cache built on it survives *all*
  camera motion for free — no texel-snapping work needed. This is the cheapest possible cached
  shadow map and it is the default path, so it is where S1 must land.
- The **CSM path is camera-fit** (`compute_cascades` reads `cam.eye/target`), BUT it is already
  **sphere-fit** (rotation-invariant extent, `csm.rs:242-255`) and **texel-snapped**
  (`csm.rs:255-282`: `radius` rounded to 1/16, center snapped to the texel grid in light space).
  So camera **rotation** does not resize/shift a cascade (already invariant), and camera
  **translation** slides each cascade in whole-texel steps — but the sphere **center** still follows
  the camera (`center` = frustum-slice centroid, `csm.rs:245-249`); the snap only quantizes it.
- **Recommendation (cheapest path to a cache that survives camera motion — the point of a VSM):**
  do **not** chase per-cascade camera-fit caching. Instead:
  1. **S1** caches the **legacy single map**, which is *exactly* camera-independent today -> the
     cache survives arbitrary camera motion with zero geometry-fit changes. This is the 14ms pass on
     IntelSponza's default config, so S1 alone captures the target win.
  2. **S2** world-space-anchors the CSM cascades so the CSM cache also survives camera translation:
     snap each cascade's sphere **center to a world-space texel grid sized to that cascade** (quantize
     `center` before building the light view, not only the post-projection NDC offset). A cascade then
     re-renders only when the camera crosses that cascade's (coarse, far => very coarse) texel cell —
     rarely, and each such re-render is one cascade, not all. This converts "camera moved at all" into
     "camera crossed a cascade texel cell", the world-stable / virtual-page invalidation granularity.

## 2. Staged, image-identical implementation plan

Common gates (every stage): **DX==VK depth-of-image <= 0.001** on the shadowed scene (raster depth is
deterministic across backends; the sampling side already handles the clip-Y flip / `shadow.slang`
header, so a cached map samples identically), **gallery byte-anchor <= 0.001/ch** (gallery keeps the
feature OFF — mirror `!gallery_scene`), **`PROFILE_GPU` before/after** on `LEVEL=sponza_intel`, and an
**env seam** to force always-render. Verify with `scratchpad/measure.py` (gpu-passes median) +
`tools/rt-compare.py`, both backends, per the ledger's verification framework.

### S1 — cached legacy single map (host-side, mirror A2) — **DO THIS FIRST**

Host-side only, no shader change. In `main.rs`:

1. Add fields `shadow_epoch: u64`, `shadow_stable_frames: u32`, `shadow_settle_frames: u32`,
   `shadow_dirty_skip: bool`, `shadow_rendered_once: bool` (mirror `main.rs:867-882` /
   init `main.rs:2802-2816`). Env: `P_SHADOW_DIRTY_SKIP` (default on, off for gallery),
   `P_SHADOW_SETTLE` (default 1 — the map is exact the *first* frame after any change, there is no
   EMA to converge unlike A2; a settle of 1-2 only guards ordering).
2. Compute a **shadow epoch** = FNV-1a over exactly the legacy map's inputs: `sun_dir` (3),
   `shadow_center` (3), `scene_radius` (1), `shadows_on`, plus a **geometry generation counter** (see
   below). Do **not** mix the camera. Bump/reset stable-frames exactly like A2 (`main.rs:4787-4792`).
3. `let skip_shadow = self.shadow_dirty_skip && self.shadow_rendered_once && !gallery && epoch stable
   >= settle` — legacy branch only:
   ```
   if self.csm.enabled { record_shadow_atlas(...) }          // S1 leaves CSM always-on
   else if skip_shadow { /* skip: do nothing */ }            // <-- the win
   else { record_shadow(...); self.shadow_rendered_once = true; }
   ```
   Keep `shadow_map` in the lighting-pass `reads` unconditionally (already true, `deferred.rs:1147`),
   so the pool slot is acquired, unwritten, and sampled as last frame's depth.
4. **Geometry generation counter:** invalidate when the caster set changes — bump on any
   skinned/morph caster in motion, on scene (re)load, and on any `obj.transform` change for a
   `casts_shadow` object. IntelSponza is fully static (no skin/morph casters) so the counter is
   constant => the map freezes after frame 1 under a fixed sun. Conservative first cut:
   `skip_shadow &&= scene has no skinned/morph shadow casters` (`obj.skin.is_some() ||
   obj.morph.is_some()` over the caster loop) — trivially true for IntelSponza, correct-by-
   construction for dynamic scenes (they never freeze).

Why image-identical: the skipped frame re-samples the *same* depth texels the re-render would have
produced (deterministic raster of unchanged geometry with an unchanged matrix). The moment `sun_dir`
(or geometry) changes, the epoch resets and the next frame re-renders — no stale shadow, no free
lunch (a moving sun / `time_of_day` at `main.rs:4242-4247` never freezes).

Robustness to verify (mirror A2): moving-sun smoke (`time_of_day`, never-settle) both backends;
200-frame static freeze-hold; a **camera-orbit-with-static-sun** run must be **byte-identical to the
always-render baseline** (the headline property — the legacy map does not move with the camera).
Vulkan validation clean; clippy `-D warnings`; existing tests.

**Expected:** IntelSponza static/slow-sun, `shadow` **14.1ms -> ~0ms** (pass skipped) both backends
once frozen. Camera moving + static sun: still ~0ms (map is camera-independent). Moving sun: 14.1ms
every frame (no win, correct). Net win is the full shadow-pass cost whenever the sun is momentarily
still — the common case.

### S2 — world-stable cascades so the CSM cache survives camera translation

Only if CSM is enabled (opt-in). Two parts:

1. **World-anchor the cascade centers** (`csm.rs`): quantize each cascade's sphere `center` to a
   **world grid whose cell = that cascade's texel size** before building `light_view0` (`csm.rs:267`)
   — extend the existing light-space snap (`csm.rs:269-282`) to quantize the center itself, not only
   the post-proj NDC offset. Then `slot.view_proj` is a step function of the quantized camera position:
   it changes only when the camera crosses a cascade cell. Image-identical to current CSM up to the
   sub-texel snap the shimmer-free design already tolerates.
2. **Per-cascade epoch + per-tile skip** (`main.rs` + `record_shadow_atlas`): one epoch per cascade =
   hash(`slot.view_proj` bits + geometry gen); track `stable_frames[cascade]`; pass a
   `dirty: [bool; MAX_CASCADES]` mask into `record_shadow_atlas` and `continue` for clean cascades in
   the slot loop (`deferred.rs:874-910`). **Atlas-clear hazard:** the atlas is cleared once by the
   depth-first-writer rule (`render/lib.rs:593-603`); when *all* cascades are clean the pass isn't
   added (nothing clears — S1 persistence). But when *some* are dirty the pass is re-added -> becomes
   first-writer -> clears the whole atlas, wiping clean tiles. **Fix:** expose a `depth_load` flag on
   the atlas pass so a partial-dirty frame LOADs (keeps clean tiles) and only redraws dirty tiles
   (option a, smaller change, keeps the atlas layout); or give each cascade its own persistent depth
   target (option b, the stepping stone to S3).

**Expected:** static sun + moving camera: far 2-3 cascades (the bulk of whole-scene caster draws)
freeze => **~14ms -> 3-6ms** typical, ~0ms fully static. Exact split depends on per-tile caster
counts (measure). S2 only matters when CSM is enabled; **for the default IntelSponza config S1 already
delivers the full win** — S2 is the generalization for camera-fit CSM.

### S3 — true per-page virtual shadow (only if S1/S2 leave a gap) — **NOT recommended now**

A real VSM (UE5-style): a large **virtual** address space backed by a **physical page pool**, a
per-frame page-request pass (mark pages sampled by visible G-buffer pixels), and caster draws routed
only into **dirty + requested** physical pages via a page table, with per-page invalidation.

**Honest effort estimate: large, multi-week, touches every layer:**
- New RHI: sparse/tiled resource or indirection-table + physical page atlas (page-table texture,
  page-alloc compute, per-page indirect draw args). Current RHI has no sparse-texture path; the graph
  has no page-table primitive.
- New passes: page-request marking (visible depth -> mark pages), allocation/eviction, per-page caster
  culling + indirect draws (GPU-driven caster loop, tying into `cull.rs` /
  `hzb-occlusion-culling.md`), and a VSM sampling path in `pbr.slang` (virtual->physical translation +
  clipmap/mip select).
- The <=0.001 DX==VK gate is *much* harder: page-boundary filtering, mip transitions, eviction races.

**Verdict: defer.** S1 captures the whole 14ms whenever the sun is still (the common case; IntelSponza
default is a static sun), and S2 generalizes across camera motion for the CSM path at a fraction of
S3's cost/risk. S3's only extra win over S2 is bounding physical shadow memory for a huge streamed
world and skipping caster draws for *off-screen* pages — neither is IntelSponza's bottleneck. Revisit
only if a world-scale level shows the atlas resolution or the *dirty-cascade* re-render cost (not the
frozen case) is the ceiling; co-design with the GPU-driven culling track (`cull-lod-design.md`,
`hzb-occlusion-culling.md`), since per-page caster culling *is* GPU-driven culling.

## 3. Expected ms saved + interaction with A2/A3

| Scenario (IntelSponza, `shadow` = 14.1ms both backends) | S1 (legacy) | S2 (CSM, world-stable) |
|---|---|---|
| Fully static (sun + camera + geo) | 14.1 -> ~0 | 14.1 -> ~0 |
| Camera moving, sun static (**the point of a VSM**) | 14.1 -> ~0 (map is camera-independent) | ~14 -> 3-6 (near cascades re-render) |
| Sun moving (`time_of_day`) | 14.1 -> 14.1 (no freeze, correct) | same |

- **Interaction with A2/A3:** independent and additive. A2 froze the surface-cache relight (VK
  `sdf_cache_light` 8.4->0), A3 the temporal reflect; both are **view-independent** GI caches keyed on
  the sun/sky epoch. The shadow cache is keyed on **sun + geometry** (S1) or
  **sun + geometry + quantized-camera** (S2). They share the trigger "sun moved" but freeze different
  passes, so savings **stack**: on a static-sun IntelSponza frame A2/A3 keep the GDF stack frozen
  (~4.4ms reflect) *and* S1 removes the 14ms shadow => the top pass moves to gbuffer/prepass, exactly
  as the ledger's IntelSponza note predicts ("geometry passes dominate... culling + LOD is the lever").
  S1 is the **cheapest** of the three (host-side, no shader, no new resource) and removes the single
  biggest pass.
- One shared subtlety: the shadow epoch's `sun_dir` term must hash **bit-identically** to A2's so the
  two caches settle/invalidate on the same frames (avoids a shadow-frozen / cache-live mismatch on a
  sun change). Reuse the same `mix(sun_dir[i].to_bits())` FNV as `main.rs:4775-4777`.

## Critical Files for Implementation

- `D:\Playground\apps\sandbox\src\main.rs` — shadow-epoch fields + skip decision (mirror A2 at
  `:4770-4799`), the `record_shadow`/`record_shadow_atlas` call site (`:4577-4583`), and
  `light_view_proj` (`:6814`) whose camera-independence makes S1 free.
- `D:\Playground\apps\sandbox\src\deferred.rs` — `record_shadow` (`:701`), `record_shadow_atlas`
  (`:858`, add the per-cascade `dirty` mask for S2), and `record_lighting` (`:1119`) that samples the
  cached map.
- `D:\Playground\apps\sandbox\src\csm.rs` — `compute_cascades` (`:176`); S2 world-anchors the sphere
  center snap (`:255-282`).
- `D:\Playground\crates\render\src\lib.rs` — pool depth persistence (`acquire_depth` `:1233`,
  `begin_frame` `:1206`), the cross-frame reuse barrier (`:895`), and the depth first-writer clear
  rule (`:593-603`) that S2 must switch to LOAD.
