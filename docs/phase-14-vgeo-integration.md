# Phase 14 — Virtual Geometry renderer integration

> Status: **in progress.** The self-contained visibility-buffer pipeline (M0–M6) is complete and
> verified in the `--vgeo-mesh` viewer. This track wires it into the REAL deferred renderer so
> virtual geometry becomes a drop-in G-buffer producer, consumed unchanged by the existing shadow /
> GDF / PBR lighting / tonemap chain. Split into gated increments; each lands as its own commit with
> the gallery anchor byte-identical when the feature is off.

## Goal & seam

The deferred frame is: **shadow → G-buffer fill → GDF (AO/GI/reflect) → PBR lighting → tonemap**
(`apps/sandbox/src/main.rs::frame`, producer at the `self.deferred.record_gbuffer(...)` call ~L4836,
consumer at `self.deferred.record_lighting(...)` ~L5702). The G-buffer is four MRTs sampled by
lighting: `albedo` (Rgba8, a=AO), `normal` (Rgba16F, world), `material` (Rgba8, metal/rough/AO),
`position` (Rgba16F, world) + `depth` — see `GBufferTargets` in `deferred.rs`.

**The integration replaces only the G-buffer producer** when opt-in `P14_VGEO=1`: the vgeo passes
(cut → SW raster → resolve) write the same four MRTs + depth, and every downstream pass is untouched.
Shadows keep using the mesh-raster `record_shadow` (the model's mesh buffers are still loaded), so the
integration surface stays minimal.

## Producer shader (LANDED)

`crates/shader/shaders/vgeo_gbuffer.slang` — full-screen `vsMain` + `fsGBuffer`. Reads the R64
visibility buffer, unpacks `(clusterId, triId)`, fetches the triangle, reconstructs
perspective-correct barycentric attributes (M6 math), and writes the four MRTs + `SV_Depth`. The
material block **mirrors `gbuffer.slang::fsMain` byte-for-byte** (base-color / metallic-roughness /
normal-map sampling with `samp_wrap` + `mip_bias`, alpha cutoff), so a resolved surface is identical
to rasterizing the same triangles. `mvp` = view_proj·model (screen projection), `model` = object→world
(world pos/normal). Compiles to metallib + SPIR-V (DXIL = Windows follow-up). Single-material scope:
one material for the whole cluster set (single-mesh/single-material object).

## Increment plan

- **I1 (LANDED):** `vgeo_gbuffer.slang` producer shader + build.rs jobs.
- **I2 + I3 (LANDED, commit `9ea0004`): `VgeoSystem` + wiring + parity.** `apps/sandbox/src/vgeo.rs`
  mirrors `CullSystem`; `P14_VGEO=1` produces the single model's G-buffer via cut → SW raster →
  resolve, `=2` is the groundless mesh reference. **Metal parity: mean diff 0.0000152/channel** vs
  the mesh fill (only ~0.02% silhouette-edge pixels differ — SW vs HW raster coverage rule). Gallery
  anchor byte-identical when off. **Deviation from the spec below:** the render-graph IR `Recorder`
  has no indirect compute dispatch, so the SW raster uses a FIXED per-cluster dispatch over a
  `0xFFFFFFFF`-sentinel visible list (the cut fills the front, `csRaster` skips sentinels) instead of
  `dispatch_indirect`; the cluster geometry is normalized in `VgeoSystem::new` (same
  `normalize_on_ground` arithmetic) so the object transform is the single model matrix.
- **Multi-object coexistence (LANDED, commit `3aaf840`): vgeo overlays the mesh raster.** The render
  graph clears a depth target only on its first writer and LOADs it afterward, so `P14_VGEO=1` now
  renders the FULL scene: the mesh fill rasters every other opaque object + ground (clears), then the
  vgeo resolve LOADs the four MRTs + depth and Less-tests the model's `SV_Depth` against them (empty
  pixels `discard` → underlying values survive). Full gallery (3 mesh objects + vgeo avocado) vs
  all-mesh = mean **0.0000153/channel** (0.019% silhouette-edge px); the avocado composites with
  correct depth/shadows and appears in the chrome sphere's reflection. **NEXT: I4** (binning-in-graph
  + M4b HZB), then multi-material/scene-cook (per-cluster material id + table for many-material
  scenes; the cook and `VgeoSystem` are still single-`MeshClusters`/single-material).
- **I2 spec (as-built above):** A subsystem mirroring `CullSystem` (`cull.rs`):
  - **Owns** persistent buffers built once at `App::new` from the loaded model's cooked clusters
    (`load_cooked_clusters`, as the viewer does): vertex pool / remap / triangles / records, plus the
    per-frame scratch — R64 visibility buffer (recreated on resize), the cut's visible list, and the
    indirect-args buffer. Pipelines: `csCut` (LOD cut, SW-only first — no binning), the SW
    `csClear`/`csRaster`, and the `vgeo_gbuffer` resolve.
  - **`import()`** exposes the scratch buffers to the graph via `graph.import_external(...)` (ordering
    handles, like `CullSystem::import`).
  - **`record_gbuffer(graph, gbuf, view_proj, model, material, ...)`** adds, into the render graph:
    1. `ComputePassInfo` reset args → `csCut` (writes the visible list + args), `storage_buffer_barrier`.
    2. `ComputePassInfo` `csClear` visbuf → `csRaster` (`dispatch_indirect` the cut), barrier.
    3. `PassInfo` resolve: colors = the four `gbuf` MRTs (with the same clears as `record_gbuffer`:
       albedo=ambient sky, others black, position α=0), depth = `gbuf.depth`, reads = the imported
       visbuf; `draw(3)` runs `fsGBuffer`. Empty pixels `discard` → the cleared "no geometry" values
       survive, exactly like the mesh fill's background.
  - **Ground:** the mesh `record_gbuffer` also draws the matte ground plane. For parity the vgeo path
    runs a follow-up ground-only fill (LOAD the G-buffer, raster just the ground) — either a small
    `deferred.rs` helper or draw the ground in the same scene list. Kept out of the vgeo passes.
  - **Wiring:** in `frame`, when `P14_VGEO` **and** the scene is a single vgeo-eligible object, call
    `self.vgeo.record_gbuffer(...)` instead of `self.deferred.record_gbuffer(...)`; else warn + fall
    back to the mesh fill. Off = the exact current call → gallery byte-identical.
  #### I2 concrete facts (turnkey)
  - **Construction plumbing:** `VgeoSystem::new` needs the model source path + cache dir + tex
    compression (as `run_vgeo_mesh` does: `load_cooked_clusters(source, cache_key, cache_dir, tex)`).
    `App::new` currently receives `&model` (the loaded `MeshData`), not the path, so build the
    `VgeoSystem` in `main()` (which has `model_path` / `model_ref` / `cache_dir` / `compress_tex`)
    **only when `P14_VGEO` is set**, and pass `Option<VgeoSystem>` into `App::new` (add a param) or a
    setter. Off → `None`, zero new work on the default path.
  - **Do NOT recenter** the cooked vertices (the viewer recentered to origin; here the object lives at
    `obj.transform`). Upload the raw model-space vertex pool / records exactly as cooked.
  - **Model-space cut (handles `obj.transform`, incl. uniform scale):** feed the cut compute frustum
    planes from `crate::push::frustum_planes(cull_view_proj * obj.transform)` (`cull_view_proj =
    proj_noflip * view`, already computed in `frame`) and `cam = obj.transform.inverse() * eye`. The
    cluster spheres stay model-space; screen-error's `err/dist` ratio is scale-invariant, so this is
    exact. `proj_factor` unchanged (`0.5*h/tan(fov/2)`).
  - **Resolve push:** `mvp = view_proj * obj.transform`, `model = obj.transform`; material from the
    single `SceneObject` (`base_color`, `metallic`, `roughness`, `tex`, `alpha_cutoff`).
  - **Ground + depth:** run `record_gbuffer` with an **empty** scene slice (clears the 4 MRT +
    draws only the ground into depth), then the vgeo resolve pass **LOADs** the MRTs and runs with
    `depth_test = Less`, `depth_write = true` against `gbuf.depth` — `fsGBuffer` emits `SV_Depth`, so
    the object occludes/were-occluded-by the ground correctly. Empty pixels `discard` → cleared
    values survive. Shadows: `record_shadow` still rasters `opaque_scene` (the mesh is loaded), so the
    object casts a mesh shadow unchanged.
  - **Eligibility:** activate only when the opaque scene is exactly one non-decal object; else log a
    warning and fall back to `record_gbuffer` (mesh fill). Skip decals/velocity/prepass on the vgeo
    path for the first step.

- **I3:** parity gate — `P14_VGEO=1` lit image vs the mesh-raster deferred render of the same
  single-model scene (path-tracer-style residual / direct diff), DX≡VK deferred to Windows.
- **I4 (later):** re-enable HW/SW **binning** in the graph (the HW mesh pass needs a color attachment
  inside the graph — resolve into a scratch/one of the MRTs as a dummy, or a depth-only mesh pass),
  then **M4b HZB** 2-pass occlusion from the real depth.

## Deferred / out of scope (see `dreamcoast-vgeo-followups`)

- **Multi-material / scene-cook:** per-cluster material id + a material table so Sponza-class scenes
  (many meshes/materials) resolve correctly. The single-material step above is single-mesh only.
- **HW/SW binning in-graph** and **M4b HZB** (I4).
- **DX≡VK Windows parity** (verification-split); the VK/DX seam already compiles (metallib + SPIR-V).

## Gates (every increment)

`tools/golden-image.py --backend metal --only gallery` = `af70c1a5…` with `P14_VGEO` off;
`cargo fmt` + `clippy -D warnings` clean; `cargo test` green; Metal-verified on this box, DX≡VK
Windows follow-up.
