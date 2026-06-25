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

## The hard part: `run()` itself (deferred, needs its own plan)

The ~2.9k-line `run()` holds the device, swapchain, every pipeline, and all
per-frame state as interdependent locals with overlapping borrows. Splitting it is
the real work and the real risk; it is **not** part of the mechanical-extraction
pass above. Candidate decomposition once the leaf modules are out:

- A `Renderer`/`App` struct owning device + pipelines + persistent resources, built
  by a `setup()` that returns it (pulls the first ~900 lines of `run()` out).
- Per-feature pipeline bundles (deferred PBR, IBL, particles, GPU cull, RT/path
  trace, SW-RT/SDF/GDF) as sub-structs with their own `new()` + `record()` so the
  frame loop composes passes instead of inlining them.
- Frame-loop state (camera, UI toggles) grouped so the loop body shrinks to
  orchestration.

Each of these is a separate, reviewed step — do them one at a time, screenshot-gated.

## Order / status

1. `push.rs` — **DONE** (≈475 lines out; VK/DX default scene unchanged).
2. `mesh.rs` — next (pure, low risk).
3. `app.rs` — process plumbing (low risk).
4. `ibl.rs` — RHI-touching but self-contained.
5. `consts.rs` — preamble + shared leaves.
6. `run()` decomposition — separate plan, after the leaves are out.
