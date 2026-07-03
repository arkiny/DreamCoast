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

## Multi-mesh + world-space cut (LANDED)

- **Multi-mesh (commit `6d265b2`):** a cluster page per registered mesh (from the registry's CPU
  geometry via `build_lod_dag`), routed by `Rc<GpuMesh>` identity; every eligible opaque object
  renders as virtual geometry, overlaid on the mesh remainder. Whole gallery via vgeo = 0.19% vs
  all-mesh.
- **World-space cut:** the LOD cut transforms each cluster's local bounds by the object's `model`
  matrix (a `model` mat4 + `max_scale` added to the cut push; the recentered viewer passes
  `identity`/`1` → unchanged) so the frustum/cone/error work under **non-uniform node scale** (which
  Sponza has); a local-space sphere test would skew. Backward-compatible, gallery + viewer
  unregressed.

## Known limitation — disconnected-component LOD (M1 builder)

On multi-component meshes (e.g. the Khronos Lantern's post+**base** in one primitive), a small
disconnected component can vanish under vgeo at **every** τ: the M1 `build_lod_dag` simplifier
collapses the component but under-records its LOD error (a disconnected island has no boundary edges
constraining the QEM collapse), so the cut's `parent_error` for its finest clusters stays below τ and
no LOD level is ever selected for it. The runtime integration is correct (the whole page uploads);
this is an **M1 asset-pipeline robustness gap** (the M1 gate was a single-component torus). Fix =
account for removed/collapsed geometry in the LOD error, especially per connected component, in
`crates/asset/src/{vgeo.rs,simplify.rs}`. Single-component meshes (the gallery) are unaffected.

## Scale finding — Sponza needs HW/SW binning (I4)

Full Sponza (`SCENE_GLTF=assets/Sponza/Sponza.gltf`) builds all **103 cluster pages / 25 materials**
and the small-triangle objects (plants, props) render correctly, but the large-triangle walls / floor
/ roof are **missing**: the SW rasterizer rasterizes one triangle per thread by looping over its
screen bounding box, so a wall triangle covering a large screen area loops over millions of pixels →
the compute dispatch effectively hangs / is dropped. This is precisely the case **M5b binning** exists
for (large clusters → the HW mesh path, only micro-triangles → SW). The integration is SW-only today,
so **I4 is required for Sponza-class scenes.** Two obstacles, in order of depth:
1. ~~**The render-graph IR `Recorder` (`crates/rhi/command_list.rs`) has no mesh-shader support**~~
   **RESOLVED (Track B):** `bind_mesh_pipeline` / `draw_mesh_tasks` / `push_constants_mesh` /
   `draw_mesh_tasks_indirect` are now in the `Recorder` trait, `RhiCommand`, `CommandList` translate,
   and are unit-tested. A graph-based pass can record the HW mesh path into deferred IR. (The
   self-contained viewer still drives the HW path on the raw command buffer; the graph path is ready
   for the integration below.)
2. ~~Then the HW mesh-vis pass needs a UAV storage-write declaration~~ **DONE (I4):**
   `RenderGraph::add_pass_with_storage_writes` + a scratch colour attachment; `csCutBin` +
   `vgeo_hwvis.slang` split each object HW/SW into the shared visibility buffer, wired into
   `VgeoSystem::record_object` behind `P14_VGEO_BIN=1`. See the "I4 … DONE" note below.

**Alternative to HW binning:** a tiled / threadgroup-cooperative SW rasterizer (bin triangles to screen
tiles, rasterize per-tile) stays in compute (no IR change) and also fixes large triangles. Either is a
substantial distinct effort. Small/medium-triangle meshes (gallery, Lantern) are fully correct on the
current SW-only path.

## Deferred / out of scope (see `dreamcoast-vgeo-followups`)

- **Multi-material / scene-cook:** per-cluster material id + a material table so Sponza-class scenes
  (many meshes/materials) resolve correctly. The single-material step above is single-mesh only.
- **HW/SW binning in-graph** and **M4b HZB** (I4).

## DX≡VK Windows verification (DONE — RTX 2070 SUPER)

The SW-only integration is now verified on D3D12 **and** Vulkan (branch `verify/vgeo-windows-dx-vk`).
Findings + fixes (all root-cause, cross-backend):

- **DXIL 64-bit atomics needed SM6.6.** `build.rs` compiled every non-RT DXIL stage at `sm_6_5`;
  Slang's internal auto-upgrade doesn't reach the DXC shader model, so `InterlockedMax64`
  (`vgeo_atomic` / `vgeo_swraster`'s `csRaster`) failed DXC. Fixed with a per-entry `sm_6_6` override.
- **Device features weren't enabled/reported.** rhi-vulkan now probes+enables `shaderInt64` +
  `shaderBufferInt64Atomics`; rhi-d3d12 probes SM6.6 + `AtomicInt64OnDescriptorHeapResourceSupported`
  (OPTIONS11 — the visibility buffer is bindless/descriptor-heap-indexed). `capabilities()` reports
  the real values on both. D3D12 `dispatch_indirect` implemented (DISPATCH command signature).
- **Bindless storage-buffer table was full.** The GI-heavy default scene already uses **all 64**
  slots (index 0..63), so vgeo overflowed binding 4 into the TLAS binding (VK VUID; DX silently
  overwrote). Raised `STORAGE_BUFFER_COUNT` 64→128 across bindless.slang + all three backends
  (heap offsets / root-sig ranges / registers derive from the constant). Off-path stays
  byte-identical (VK) / 1-LSB (DX nondeterminism).
- **The SW raster used the Y-flipped `view_proj`.** The raster + resolve do MANUAL `(0.5-ndc.y*0.5)`
  NDC→pixel mapping (the no-flip window convention). On Vulkan `view_proj` carries the clip-space
  Y-flip for the *hardware* pipeline, which vertically flipped the visbuf rows vs the resolve's
  `SV_Position` reads — corrupting the whole scene's G-buffer. Switched the raster/resolve `mvp` to
  the flip-free `cull_view_proj` (already used by the cut); DX unchanged, VK fixed.
- Two Metal-isms in DX host paths: the `--atomic64-test` smoke and the vgeo per-frame scratch pool
  host-wrote/read DEFAULT-heap buffers → switched to `create_storage_buffer_host` (CUSTOM-L0, both
  UAV + CPU-mappable). The vgeo eligibility gate now checks `atomic_int64`, not `mesh_shader`
  (the SW path needs no mesh shader — that was a Metal-ism, both caps ship together there).

**Measured (1280×720, `--screenshot-clean`):** gallery `P14_VGEO=1` DX≡VK **0.000004/ch** (max byte
101, silhouette-only); vgeo-vs-mesh **0.000714/ch** (DX) / **0.000715/ch** (VK) — the SW-vs-HW raster
coverage rule, matching Metal; Lantern multi-mesh DX≡VK **0.000234/ch**. `P14_VGEO` off: VK
byte-identical to pre-change main, DX ≤1 LSB. `--atomic64-test` PASSES both backends, validation-clean.

## Track B — mesh-shader HW path DX≡VK (DONE — RTX 2070 SUPER)

The hardware mesh-shader path (`--mesh-shader-test`, `--vgeo-mesh` CUT/SW/BIN, M5b binning) is now
implemented and verified on D3D12 **and** Vulkan. Findings + fixes (all root-cause, cross-backend):

- **Slang mesh→DXIL codegen.** Slang 2026.10.2 mis-emits mesh outputs: the explicit `out` on
  `indices`/`primitives` produces a malformed `out … out …` DXC rejects. Dropping the explicit `out`
  (the modifier implies it) fixes the 2-output shaders (`vgeo_meshlet`/`vgeo_cluster`). The 3-output
  `vgeo_hwvis` (vertices+indices+**primitives**) hits a deeper bug — Slang drops the `vertices`
  keyword — so its DXIL is built via a targeted `build.rs` slang→HLSL→patch→DXC path (bundled
  `slangc -pass-through dxc`). `vgeo_hwvis_fs` joined `dxil_needs_sm66` (its `InterlockedMax64`).
- **The mesh-shader pipeline seam was `unimplemented!()` on VK+DX.** VK: `VK_EXT_mesh_shader` enable +
  `taskShader`/`meshShader` probe + `create_mesh_pipeline` (task+mesh+fragment) + `cmd_draw_mesh_tasks
  [_indirect]_ext`. DX: OPTIONS7 `MeshShaderTier` probe + SM6.5 mesh PSO via
  `D3D12_PIPELINE_STATE_STREAM` + `DispatchMesh` + `ExecuteIndirect` over a DISPATCH_MESH signature.
- **VK bindless layout excluded the mesh stage.** The storage-buffer (and sampled) descriptor bindings
  had no `MESH_EXT`/`TASK_EXT` stage flags, so the cluster mesh shader read nothing and emitted no
  geometry on Vulkan (D3D12's `SHADER_VISIBILITY_ALL` covered it). Added, gated on `has_mesh_shader`.
  The mesh push-constant range/stage set is tracked per pipeline (task stage only when present).
- **NVIDIA VK `SetMeshOutputCounts` from an unbounded load = empty draw.** Feeding it a value straight
  from `ByteAddressBuffer.Load` makes the whole mesh draw emit nothing on NVIDIA/Vulkan; clamping with
  `min(count, MAX)` before the call gives the driver a static bound (and is a correctness fix — an
  out-of-range count is UB). Applied to `vgeo_cluster` + `vgeo_hwvis`; identical output on DX/Metal.
- **Viewer barriers/transitions (Metal-isms).** `run_vgeo_mesh` host-reset its indirect-args each frame
  from a DEFAULT-heap buffer (illegal DX/VK → `create_storage_buffer_host`); missing cut-args →
  INDIRECT_ARGUMENT + visbuf UAV barriers (empty draws); depth left in UNDEFINED (VUID-09588). The SW
  compute raster + M6 resolve now use the flip-free matrix (manual NDC→pixel), only the HW-rasterized
  cluster/hwvis passes use the Y-flipped `mvp`. `VGEO_ANGLE` pins the orbit for reproducible captures.

**Measured (1280×720, Avocado, `VGEO_ANGLE=0.6`):** `--vgeo-mesh` **direct / cut / sw / bin / resolve /
material all DX≡VK 0.000/ch (max 0)**. Binning verified across `VGEO_BINPX` 8/30/80 — the HW mesh and
SW compute rasterizers write bit-identical visibility (BINPX 8 vs 80 output identical → seamless HW/SW
boundary). `--mesh-shader-test` renders the RGB triangle on both backends. Mesh commands are
validation-clean (the smoke loop's swapchain-sync VUIDs are pre-existing, identical to
`--triangle-test`). Gallery anchor (features off) DX≡VK 0.001/ch — unregressed.

**I4 (integrated HW/SW binning) — DONE.** `P14_VGEO_BIN=1` (needs mesh shaders) routes each vgeo
object through the binning cut in the real deferred renderer: `csCutBin` splits the cut into a SW
sub-list (compute raster) and an HW sub-list (`vgeo_hwvis` mesh shader, `draw_mesh_tasks_indirect`),
both writing the SAME R64 visibility buffer, then the resolve → G-buffer as before. Enablers:
`RenderGraph::add_pass_with_storage_writes` (a graphics pass that also UAV-writes the external
visibility buffer, scheduled via WAW/RAW); the Vulkan `storage_buffer_barrier` src now covers the
FRAGMENT stage (the HW-vis writes the visbuf from its fragment atomicMax; D3D12/Metal UAV barriers
are stage-agnostic); `new_host` honours `INDIRECT_BUFFER` usage for the host-reset `hw_args`. The
HW mesh-vis uses the Y-flipped `view_proj` (HW rasterizer) while the SW raster/resolve use the
flip-free `cull_view_proj`, so both land the same screen pixels. **Verified (gallery, `P14_VGEO=1`):
binning output byte-identical to SW-only on DX (0.000/ch) — HW and SW write bit-identical
visibility — and DX≡VK 0.001/ch; gallery anchor (off) unchanged.** One benign SPIR-V validation
warning (`VUID-…-OpVariable-08746`: a Slang per-primitive mesh→fragment decoration mismatch in
`vgeo_hwvis`; triId is delivered correctly). NEXT (perf, not correctness): a Sponza-scale profiling
pass + M4b HZB two-pass occlusion.

## Gates (every increment)

`tools/golden-image.py --backend metal --only gallery` = `af70c1a5…` with `P14_VGEO` off;
`cargo fmt` + `clippy -D warnings` clean; `cargo test` green; Metal-verified on the macOS box,
**DX≡VK verified on Windows (RTX 2070 SUPER), ≤0.001/ch** (see the section above).
