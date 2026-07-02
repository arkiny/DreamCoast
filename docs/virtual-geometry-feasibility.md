# Virtual Geometry Feasibility — IntelSponza GBuffer Cost

> Status: design decision doc, 2026-07-03. Owner: gbuffer/opaque-geometry perf workstream
> (shadow caching is a separate workstream, docs/shadow-cache-design.md). Honest benefit/effort
> judgment on whether Nanite-like cluster rendering beats incremental per-mesh cull + LOD for
> IntelSponza's opaque geometry cost.
> Relatives: docs/cull-lod-design.md (Option A detail), docs/phase-14-virtual-geometry.md (Option C
> full Nanite). This doc is the A-vs-B-vs-C verdict tying them together.
> Gates: DX≡VK ≤ 0.001 avg/ch, gallery byte-anchor held, PROFILE_GPU before/after.

## 0. TL;DR

- **Baseline**: `LEVEL=sponza_intel` 1080p/0.667 med — DX 42.8ms / VK 43.0ms (23fps). Top passes on
  BOTH backends: `gbuffer` 14.2ms + `shadow` 14.1ms; geometry dominates ~28ms of 43ms, DX≈VK because
  the bottleneck is raster throughput, not the GI compute stack (already optimized, A2/A3).
- **The draw path is the worst case for culling**: every opaque pass is a CPU `for obj in scene`
  loop issuing per-object `draw_indexed` with per-object vbuf/ibuf rebinds — no bounds, no indirect,
  no instancing (deferred.rs:980-1018 gbuffer, :787-824 prepass, :718-743 shadow).
- **Multi-draw-indirect already works on BOTH backends today**: `draw_indexed_indirect(buf, off, N)`
  → `vkCmdDrawIndexedIndirect` stride 20 (rhi-vulkan/command.rs:1207) and `ExecuteIndirect` over a
  20-byte `DRAW_INDEXED` signature (rhi-d3d12/command.rs:907, device.rs:184-199). BUT `draw_count`
  is a **CPU constant** — **no** GPU count buffer, **no `dispatch_indirect`, no mesh-shader pipeline,
  no 64-bit atomics, no BDA** anywhere in the RHI (grep: zero matches). Full Nanite needs all four.
- **AABB source already exists** (single source of truth): `fuse_scene` computes per-drawable world
  AABBs in draw-list order (fuse.rs:208-247), 1:1 with `build_scene`'s `SceneObject` order.
- **Verdict**: **Option A** (per-mesh frustum + HZB occlusion cull + discrete LOD) captures the bulk
  of the recoverable gbuffer cost at lowest effort and zero new RHI primitives (zero-index-record
  indirect trick avoids even the count buffer). **Option B** (cluster "Nanite-lite", no SW raster)
  is a bounded reach layered on A, worth it only if hero foliage still dominates after A. **Option C**
  (full Nanite + SW raster) is NOT justified for a 43→16.6ms target on Turing — multi-month, four
  missing RHI primitives, highest DX≡VK parity risk (64-bit atomics), and IntelSponza's 4.9M tris are
  mostly not subpixel. C stays a Phase-14 learning track, decoupled from the 60fps goal.

## 1. Opaque draw path (the cost center)
`build_scene` (registry.rs:173-198) walks `world.draw_list()` into `Vec<SceneObject>` each frame; each
pass borrows the slice and loops per-object `draw_indexed` with per-object vbuf/ibuf rebinds. Bindless
materials, but CPU-driven submission. `SceneObject` carries no bounding volume → nothing is cullable
without adding one. IntelSponza = all-static, so the static pipeline dominates.

## 2. Existing cull/HZB machinery — reuse vs rewrite
- **Reuse as-is**: `HzbSystem` (pyramid build/reduce, hzb.rs:225/139/180 — the expensive, validated
  part); `frustum_planes(cull_view_proj)` (push.rs:371-384); the no-flip cull matrix
  `cull_view_proj = proj_noflip * view` (main.rs:3533-3535, the root of DX≡VK cull parity); the 2-pass
  occlusion methodology + the 3 burned-down HZB bugs (extent passthrough, 4-tap corner texels,
  `uv.y = 0.5 - 0.5*ndc.y`).
- **Rewrite**: `cull.slang`/`cull_draw.slang` cull only a *synthetic parametric cube grid*
  (`instance_center(i)`, cull.slang:31-37) — must read a per-draw instance table instead.
- **AABB**: extract `fuse_scene`'s 8-corner transform into a shared `mesh::world_aabb` helper
  (CLAUDE.md #4); cache the local AABB at upload → O(1)/draw per frame.

## 3. RHI capability matrix (verified)
| Capability | Present? | Needed by |
|---|---|---|
| `draw_indexed_indirect` multi-draw (stride 20, both backends) | **Yes** | A, B |
| GPU count buffer (draw_count from GPU) | No (CPU u32) | B (full GPU-driven), C |
| `dispatch_indirect` | No | C |
| Mesh/task shader | No | C |
| 64-bit atomics | No | C (visibility buffer) |
| Buffer device address | No | C |
| Compute + storage/append + 32-bit atomics | Yes | A/B |

**Pivotal fact**: `draw_count` is a CPU constant. A/B need ZERO new RHI primitives via the
**zero-index-record trick**: issue `draw_indexed_indirect(args, 0, N_max)` with culled records zeroed to
`index_count=0` (a no-op draw) — fully GPU-driven with today's RHI, no readback latency. Count buffer /
mesh shaders / atomics / BDA are only for C.

## 4. Options (benefit / effort / risk)
- **A — per-mesh frustum + HZB occlusion + discrete LOD** (≈ docs/cull-lod-design.md). Granularity:
  whole drawable. Expected IntelSponza **43 → ~24-30ms** (frustum −6..−12, occlusion −4..−8 in this
  layered indoor scene, LOD −3..−6; does NOT alone reach 16.6 — shadow-caching + TAAU cover the rest).
  Effort ~8-14 person-days (S0 1d, S1 1-2d, S2 3-5d, S3 2-3d, S4 2-4d). RHI gaps: none for S1-S3.
  DX≡VK risk: low (no-flip cull matrix, deterministic scalar math, offline meshopt).
- **B — meshlet/cluster cull, "Nanite-lite" (NO SW raster)**. Granularity: intra-mesh (cull half a
  tree behind a wall). Delta over A: **modest, view-dependent** — real only for the few 4.9M-tri
  foliage meshes, near-zero for the 150 tight architectural nodes. Effort ~A + 10-18 days. RHI gaps:
  none required (zero-record), but a GPU count buffer becomes the first motivated primitive at 10k+
  clusters. Risk: low-moderate.
- **C — full Nanite (cluster LOD DAG + SW raster + visibility buffer)** (= phase-14-virtual-geometry.md).
  Benefit for THIS scene: **not materially better than B**, possibly worse near-term (vis-buffer
  resolve overhead); Nanite wins in the massive-overdraw subpixel regime IntelSponza barely enters.
  Effort **40-80+ person-days**, four missing RHI primitives, **highest DX≡VK risk in the engine**
  (64-bit atomics model diverges most between APIs).

## 5. Recommendation
**Do A now; treat B (no SW raster) as an optional reach layered on A; do NOT pursue C for this goal.**
1. A is the 80/20 — makes the uncullable per-object loop frustum/occlusion-cullable + LOD, reusing the
   validated HzbSystem + existing AABBs + no-flip matrix + working indirect path, zero new RHI, minimal
   parity risk, every stage byte-verifiable.
2. B's advantage is confined to a handful of huge foliage meshes — scope as an extension of A's cluster
   table only if profiling after A shows foliage still dominating.
3. C is disproportionate — a multi-month, four-primitive research track; belongs in Phase 14, decoupled
   from 60fps.
4. **Coordinate with shadow caching**: `shadow` shares the same per-object loop. Shadow-pass cull must
   use per-cascade **light** frusta (off-screen casters must survive; occlusion cull is inapplicable to
   the whole-scene shadow map — frustum only). For the DEFAULT single scene-centered shadow map,
   frustum cull helps little (map covers the whole scene) → **shadow caching, not culling, is the
   shadow lever**; culling is the gbuffer lever.

## 6. Staged plan (Option A) — each: default-OFF env seam, DX≡VK ≤0.001, gallery anchor, PROFILE_GPU
- **S0** AABB single-source extraction (no-op refactor; gallery byte-anchor unchanged).
- **S1** CPU frustum cull real draws (`SCENE_CULL=1`): filter `scene[]` by `frustum_planes(cull_view_proj)`
  into the visible subslice for gbuffer/prepass. **Gate: fixed-camera OFF≡ON byte-identical** (culled =
  off-screen = clipped anyway; false-cull count must be 0). PROFILE_GPU gbuffer drop on partial views.
- **S2** GPU cull + indirect submit (`SCENE_GPU_CULL=1`): per-draw bounds table, `scene_cull.slang`
  (AABB vs 6 no-flip planes) → indirect args, `draw_indexed_indirect(args,0,N_max)` with culled records
  zeroed (no count buffer, no readback). Gate: visible set == S1, byte-identical OFF≡ON.
- **S3** HZB occlusion cull (`SCENE_HZB_CULL=1`): reuse HzbSystem, transplant occlusion block with the
  3 bug fixes; single-pass prev-frame HZB first, 2-pass as S3.1. Gate: behind-wall view OFF≡ON
  byte-identical + occlusion count > 0.
- **S4** discrete mesh LOD (`SCENE_LOD=1`): cook-time meshopt::simplify chain (.dcasset schema bump,
  deterministic), SSE LOD select with hysteresis. **Needs meshopt dependency approval.** Gate: near
  geometry byte-identical, far residual via rt-compare.py, DX≡VK unaffected (offline decimation).
- **S5 (reach)** Option B cluster granularity — only if S1-S4 leaves hero foliage dominating.

Out of scope of this workstream: SW rasterizer, visibility buffer, 64-bit atomics, mesh shaders, BDA,
LOD DAG / continuous crack-free LOD, cluster streaming (all Option C / Phase 14).

## 7. Risks
- S1 byte-identity: conservative normalized no-flip planes + corner-expanded AABB → zero false-cull.
- Shadow: per-cascade light frustum only (not camera); occlusion cull inapplicable; default whole-scene
  map is a caching problem, not a culling one.
- Zero-index-record bandwidth: fine at 155 draws; measure before relying on it at 10k+ clusters (B).
- meshopt determinism: offline (DX≡VK-neutral); extend dcasset `cook_is_deterministic` to LOD chunks.
- TAAU jitter vs unjittered cull matrix: expand AABB 1px if a capture ever shows subpixel false-cull.
