# Sandbox refactor — splitting `apps/sandbox/src/main.rs` by feature

`main.rs` grew to ~4.6k lines: a 60-line `const`/struct preamble, one ~2.9k-line
`run()` (device + pipeline setup followed by the per-frame render-graph loop), and
~1.6k lines of free helper functions below it. This note tracks carving it into
feature modules. **Constraint: every step is behavior-preserving** — pure code
motion, verified by a clean `build` + `clippy -D warnings` and an unchanged
VK/DX screenshot of the default scene.

## Target module layout

- `push.rs` — pure push-constant packers (`*_push`) + leaf camera/matrix math
  (`cube_face_view_proj`, `frustum_planes`, `mat4_to_3x4`). **DONE.**
- `mesh.rs` — geometry/asset helpers: `vertex_bytes` / `index_bytes`, `ground_mesh`,
  `normalize_on_ground` + `ModelBounds`, `upload_mesh`, `upload_texture`,
  `make_checker_texture`, `PtMaterial`, `build_pt_instance_table`.
- `app.rs` — process/platform plumbing: `Capture`, `screenshot_captures`,
  `interactive_screenshot_path`, `save_screenshot`, `model_path`, `log_file_path`,
  `select_backend`, `validation_enabled`, `build_render_finished`,
  `load_shader_pair`, `load_compute_shader`.
- `ibl.rs` — environment capture / IBL bake: `CubeSet`, `IblResources`,
  `record_environment_capture`, `generate_brdf_lut` (these touch RHI state, so they
  move as a unit with their helpers).
- `consts.rs` — the format/size `const` block + `Globals` / `SceneObject` +
  `globals_bytes`, `normalize3` (shared leaf used by `push.rs`).

## `run()` decomposition plan

`run()` is ~2.83k lines: **setup** (159–1497, ~1340 lines — device, swapchain,
every pipeline, persistent buffers/textures, scene) followed by the **frame loop**
(1524–~3010, ~1490 lines — events, camera, UI, render-graph build, submit,
screenshot readback). Setup and loop split into the *same* feature groups, which is
exactly the seam to cut along.

### Strategy: feature-bundle structs

Each feature becomes a struct owning its pipelines + persistent resources, with
`new(device, …) -> Result<Self>` (its slice of setup) and `record(&self, graph, …)`
/ `update(&mut self, …)` methods (its slice of the loop). The loop then calls
`feature.record(&mut graph, …)` instead of inlining the passes. Borrows are sound:
each bundle lives in the loop scope (longer than any per-frame `RenderGraph`), so
graph closures may capture `&self.field` for the graph's lifetime.

Feature groups (setup line / loop line):
- **Particles** — sim+draw pipelines, ping-pong buffers (727 / 2121, 2878). Most
  self-contained → the pattern-establishing first cut.
- **GPU cull** — args/visible buffers, reset/cull/draw pipelines (812 / 2171, 2839).
- **SW-RT / GDF (Phase 11)** — volumes, bake mesh, sdf_trace + B1–B4 pipelines +
  instance table + done-flags (376–578 / 2361–2838). Biggest single win.
- **HW RT (Phase 8)** — BLAS/TLAS, instance tables, Cornell scene, rt pipelines
  (604–715, 1124–1256 / 2253). 
- **IBL** — sky/capture/irradiance/prefilter/brdf pipelines, cube sets, capture
  depth, brdf LUT (893–1002, 1424 / graph pre-pass). `record`/`generate` already in
  `ibl.rs`; this bundles the *resources*.
- **Deferred PBR core** — gbuffer/shadow/pbr/post pipelines, globals buffer, shadow
  map (258–358, 1258 / 1900–2100, tonemap). The backbone; extract last.

### Order (each step: build + clippy + VK/DX screenshot gate, own commit)

- **R1 — `particle.rs` / `ParticleSystem`. DONE.** Owns sim+draw pipelines, the
  ping-pong buffers, and parity; `new` seeds, `record_sim` / `record_draw` add the
  graph passes (`&'a self` tied to the graph's lifetime), accessors drive the async-
  compute submit path that stays in `run()`. Parity `advance()` moved to end-of-frame
  (after the graph's `&self` borrows end — NLL releases them at `graph.execute`).
  Confirmed the pattern: graph closures capturing `&self.field` compile with the
  `&'a self` + `RenderGraph<'a>` signature, no extra lifetime gymnastics.
- **R2 — `cull.rs` / `CullSystem`.**
- **R3 — `swrt.rs` / `GdfSystem`** (Phase-11 volumes/bake/merge/trace/view).
- **R4 — `rt.rs` / `RtSystem`** (HW RT + path tracer + Cornell).
- **R5 — `ibl.rs` / `IblSystem`** (fold the IBL resources in beside the existing fns).
- **R6 — `deferred.rs` / `DeferredRenderer`** (the PBR backbone).
- **R7 — `App::new()` / `App::frame()`.** With features bundled, the residual setup
  + loop is small enough to wrap into an `App` struct; `run()` shrinks to window +
  `App::new` + `while { app.frame() }`.

### Why this order / thread-split payoff

Self-contained features first (particles, cull) de-risk the pattern before the
entangled core. Once `App` owns the bundles (R7), the eventual **main /
render-graph / RHI / worker** thread split has clean handles: `App` (or a subset of
its bundles) can move onto the render thread, the window/event pump stays on main,
and bundles that only touch RHI submission can be driven from the RHI thread —
without first untangling 2.8k lines of interleaved locals.

Risk: the frame loop's graph closures capture many locals; bundling changes captures
from `local` to `&self.field`. Expected to compile as-is (bundle outlives graph); if
the borrow checker complains, the fix is local (clone an index, or scope the borrow),
not structural. Each R-step is independently revertible.

## Order / status

1. `push.rs` — **DONE** (≈475 lines out; VK/DX default scene unchanged).
2. `mesh.rs` — **DONE** (geometry/asset helpers; `PtMaterial` / `ModelBounds` made `pub(crate)`).
3. `app.rs` — **DONE** (shader loaders, screenshot/PNG, CLI flag parsing).
4. `ibl.rs` — **DONE** (`CubeSet`, `IblResources`, `record_environment_capture`,
   `generate_brdf_lut`; struct fields + `SceneObject` made `pub(crate)`). Note: child
   modules can already name ancestor-private consts/types via `use crate::…`, so only
   items reached *upward* from `run()` (the `ibl` structs) needed `pub(crate)`.
5. `consts.rs` — preamble + shared leaves (`Globals`, `globals_bytes`, `normalize3`,
   and the small `gbuffer_push` / `pbr_push` / `mat4_bytes` / `light_view_proj` that
   stayed behind). Lower value now — defer or fold into a later pass.
6. `smoketest.rs` — **DONE** (`--clear-test` / `--triangle-test` / `--mesh-test`
   standalone loops + flag predicates). Self-contained alt entry points; `run()`
   early-returns into them.
7. `run()` decomposition — in progress (see plan below); R1 done.

main.rs: ~4.6k → ~3.0k lines after the leaf splits + smoketest + R1 (push/mesh/app/
ibl/particle/smoketest ≈ 1.8k lines moved into focused modules).
