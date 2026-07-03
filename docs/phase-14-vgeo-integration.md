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
- **I2 (NEXT): `VgeoSystem` + wiring.** A subsystem mirroring `CullSystem` (`cull.rs`):
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
