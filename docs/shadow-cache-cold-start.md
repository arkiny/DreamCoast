# COLD-START PROMPT — implement the app-owned persistent-depth SHADOW CACHE (paste into a fresh session; Mac/Metal-ready)

You are continuing a graphics-engine performance track on **DreamCoast** (from-scratch Rust renderer on
raw Vulkan `ash` + raw D3D12 `windows-rs` + raw Metal `objc2` behind one hand-rolled RHI; bindless-first;
no wgpu). Read `CLAUDE.md` first. On **macOS the backend is Metal** (`cargo run -p sandbox -- --backend
metal`); this task deliberately lands on Mac so the RHI change is verified on Metal too (DX/VK were the
prior session's backends). Engineering rules: root-cause, scalability, single-source, **verify-then-claim**.

## THE GOAL

Implement a **cached shadow map** for the legacy directional shadow pass — the #1 GPU pass on
**IntelSponza** (`LEVEL=sponza_intel`) at **14.1ms**. The legacy map is **camera-independent**
(`light_view_proj(sun_dir, scene_center, scene_radius)` — `main.rs:~6824`; the camera never enters it),
so when the sun + geometry are unchanged the re-raster reproduces a **bit-identical depth**. Skip it and
re-sample last frame's map: **image-identical when static, survives arbitrary camera motion** (the
Virtual Shadow Map benefit). Invalidate on sun/geometry change. Expected: IntelSponza `shadow` **14→~0ms**
(measured-confirmed already), total ~37→~23ms.

## CRITICAL — DO NOT REPEAT THE FAILED APPROACH (see docs/lossless-opt-ledger.md A4)

The in-graph **transient depth pool does NOT persist depth across frames.** Two attempts both broke the
image catastrophically (shadows vanished → scene too bright): (1) skipping the shadow graph pass entirely
(Sponza 1.7/ch); (2) a `create_depth_persistent` (LOAD-not-clear) + no-draw attach pass (7.0/ch). Root
causes: the render-graph transient pool clears/reuses a depth that has no writer, AND the pool is
**per-frame-in-flight** (`self.pools[fif]`, FRAMES_IN_FLIGHT=2) so each FIF's slot is independent. **Both
reverted. Do NOT touch the render-graph transient pool for this.**

## THE CORRECT APPROACH — app-owned persistent depth, rendered OUTSIDE the graph (IBL pattern)

The RHI already has the pieces: `Device::create_depth_buffer(extent)` (app-owned depth), `depth_to_sampled`
(transition to shader-read), and `ibl.rs` renders the env cube into **app-owned** targets OUTSIDE the main
render graph via a direct recorder (`capture_depth: DepthBuffer` at `ibl.rs:66,269,495`; `smoketest.rs:320`
renders into an app-owned depth too). Mirror that:

1. **App owns the shadow depth** — a persistent `DepthBuffer` at `SHADOW_SIZE=2048` (sampleable). Because
   the shadow render is written some frames and read every frame, make it **per-FIF** (`[DepthBuffer;
   FRAMES_IN_FLIGHT]`) to avoid a write-after-read hazard across in-flight frames (when static there are no
   writes so a single one would be safe, but moving-sun writes every frame → per-FIF is the robust choice;
   this also means each FIF must be primed — render the first `FRAMES_IN_FLIGHT` stable frames before any
   frame caches, exactly the bug that bit the pool attempt).
2. **Render the shadow into it OUTSIDE the graph**, conditionally. Add a `record_shadow_direct(recorder,
   &DepthBuffer, scene, light_vp)` that begins a **depth-only render pass** on a raw `Recorder` (the shadow
   pipeline is already depth-only — no color attachment) and runs the same caster draw loop as
   `deferred.rs::record_shadow` (`:701-751`). Study `ibl.rs`'s capture recording for the render-pass
   begin/end + viewport setup on an app-owned target, and check the RHI recorder supports a **no-color +
   one-depth** begin_rendering on all three backends (add it if missing — this is the main RHI touch, and
   the reason to be on Metal: verify `MTLRenderPassDescriptor` depth-only works). Skip this call when settled.
3. **depth_to_sampled** the app-owned depth after rendering (and it stays sampled while cached).
4. **Lighting samples it by bindless index** — the deferred lighting pass (`deferred.rs:record_lighting`,
   `shadow_index` at `:1182,1200`) currently does `ctx.sampled_index(shadow_map)` on the graph resource.
   Switch the legacy path to pass the **app-owned depth's bindless sampled index** instead (like the IBL
   cubes are sampled by index). Use `graph.import_external` only if you need barrier ordering; the
   depth_to_sampled transition + same-queue ordering should suffice (verify no validation errors).
5. **Keep the CSM/atlas path on the existing graph transient** for now (S1 is legacy-map only).

## THE SKIP LOGIC (mirror A2 — main.rs cache_settled at ~4736)

- **Epoch** = FNV-1a over `sun_dir` (hash bit-identically to A2's `mix(sun_dir[i].to_bits())` so both
  caches invalidate on the same frame), `scene_center`, `scene_radius`, `shadows_on` — **NOT the camera**.
  Bump `shadow_stable_frames` when unchanged, else reset. Fields + init mirror A2
  (`main.rs` cache_epoch/stable_frames/settle_frames/dirty_skip; env `P_SHADOW_DIRTY_SKIP`,
  `P_SHADOW_SETTLE`). Gate off for the gallery (`!gallery_scene`).
- **Settle ≥ FRAMES_IN_FLIGHT** (prime every FIF's persistent depth before caching), and require
  `!has_dynamic_caster` (`scene.iter().any(|o| o.casts_shadow && (o.skin.is_some()||o.morph.is_some()))`) —
  a skinned/morph caster's depth changes each frame, so those scenes never freeze (correct by construction).
  IntelSponza is fully static ⇒ freezes after priming. `settled = dirty_skip && !dynamic && stable ≥ settle`.

## VERIFICATION (rebuild the harness; the prior session's lived in a Windows scratchpad)

- **measure.py** (recreate per docs/cold-start-lossless-opt.md): parse `PROFILE_GPU` per-pass ms + frame
  median. **Gotchas:** decode subprocess with `encoding='utf-8'` (the logs are UTF-8; a cp949/other locale
  crashes the reader); redirect captures with `>/dev/null`, **NEVER `tail -0`** (SIGPIPE kills the exe
  before it saves the PNG).
- **Perf env:** `LEVEL=sponza_intel RENDER_QUALITY=med WINDOW_RES=1920x1080 RENDER_SCALE=0.6667 P_TAAU=1
  PROFILE_GPU=1 WARMUP_FRAMES=70`. Confirm `shadow` → ~0ms on Metal. Also run `LEVEL=sponza` (Sponza) to
  confirm no regression.
- **Image-identical gate** (the load-bearing one): `AUTO_EXPOSURE=0 EXPOSURE=8 WARMUP_FRAMES=200`, cache
  **ON vs `P_SHADOW_DIRTY_SKIP=0`**, same fixed camera → **byte-identical** (a static-sun cached map ==
  the fresh raster; expect ≤ the run-to-run noise ~0.002–0.006/ch). `tools/rt-compare.py a b diff.png`.
  **Headline test:** a **camera-orbit / moving-camera** run with a **static sun** must ALSO be byte-identical
  to the always-render baseline (the whole point — the legacy map doesn't move with the camera). Script a
  moving camera (e.g. the demo orbit) or a couple `CAM_EYE`/`CAM_TARGET` captures.
- **Moving-sun robustness:** `TIME_OF_DAY=1` (epoch changes every frame → never freezes → shadow renders
  every frame, correct, no deadlock/validation error). Both must be clean.
- **Gallery byte-anchor:** default scene (no LEVEL) must stay **≤0.001/ch** (feature gallery-gated off).
- **Metal parity:** since committed goldens in `tools/goldens/` are Metal-authored, this is the session
  to also confirm Metal matches. If a Windows machine is available later, re-verify DX≡VK ≤0.001.
- `cargo test`, `RUSTFLAGS="-D warnings" cargo clippy --all-targets` (NOT `cargo clippy -- -D warnings` —
  the rtk proxy mangles the passthrough), Vulkan/Metal validation clean.

## FILES
- `apps/sandbox/src/main.rs` — shadow-epoch fields + skip decision (mirror A2 ~4736), the app-owned
  per-FIF DepthBuffer allocation, the out-of-graph `record_shadow_direct` call + `depth_to_sampled`, and
  wiring the lighting pass's `shadow_index` to the app-owned bindless index. The shadow record site is at
  `main.rs:~4633` (legacy `else` branch).
- `apps/sandbox/src/deferred.rs` — `record_shadow` (`:701`), add `record_shadow_direct` (raw recorder,
  depth-only render pass into the app-owned target); `record_lighting` (`:1119`) sampling wire-up.
- `apps/sandbox/src/ibl.rs` — reference for out-of-graph rendering into an app-owned depth (capture setup).
- `crates/rhi/` (+ `rhi-vulkan`/`rhi-d3d12`/`rhi-metal`) — a **depth-only** begin_rendering on the Recorder
  if it doesn't already exist (the main cross-backend touch; verify on Metal).
- `docs/shadow-cache-design.md` (full design, S1/S2/S3), `docs/lossless-opt-ledger.md` (append the result;
  A4 = the broken pool attempt), memory `sponza-1080p-60fps-track.md`.

## AFTER THIS: the rest of the IntelSponza→60fps track (docs/lossless-opt-ledger.md, cull-lod-design.md)
- **S2 CSM-static-mesh cache** (user-flagged long-term need): world-anchor the CSM cascade centers (snap
  to a per-cascade world texel grid, `csm.rs:255-282`) so the camera-fit CSM cache also survives camera
  translation; per-cascade dirty mask + LOAD-not-CLEAR atlas hazard fix. Generalizes S1 to the CSM path.
- **HZB occlusion cull** (S3) on top of the landed A5 frustum cull — indoor IntelSponza has lots of
  geometry occluded behind walls; reuse `hzb.rs::HzbSystem`. Then **discrete mesh LOD** (S4, needs a
  meshopt dependency approval). Then measure whether the GDF stack (~11ms) is the remaining 60fps floor.

## ALREADY LANDED ON MAIN (do not redo)
`d47eb44` A2 cache dirty-skip · `9d0814a` A3 adaptive reflect skip (**Sponza 1080p 60fps BOTH backends**) ·
`5bea332` A5 opaque-draw frustum cull (**IntelSponza gbuffer 14→8.5ms, image-identical**) · design docs
(`452c067`). A4 shadow-cache (in-graph pool) was tried and **reverted** — start straight from the app-owned
approach above.
