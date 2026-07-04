# Phase 14 — vgeo correctness pass (LOD soffit, pole crown, backface cull)

Status: **DONE (2026-07-04).** Three vgeo rendering fixes found while validating vgeo as a drop-in
G-buffer producer (toward making it the default renderer, with a mesh fallback). Each is verified
against the mesh fill and DX≡VK. Supersedes the "soffit is not LOD" conclusion in
`docs/phase-14-vgeo-clustering.md` (that rested on two measurement artifacts, see §1).

Landed changes:
- `crates/asset/src/simplify.rs` — LOD error span bound (§1)
- `crates/shader/shaders/vgeo_cut.slang` — normal-cone cull disabled (§2)
- glTF `doubleSided` plumbing + per-triangle backface cull (§3): `gltf_scene.rs`, `dcasset/gltf.rs`,
  `dcasset/mod.rs`, `scene/gltf_instance.rs`, `registry.rs`, `level.rs`, `main.rs`, `vgeo.rs`,
  `vgeo_swraster.slang`, `vgeo_hwvis.slang`

---

## 1. Arch-soffit slab — LOD coverage-expansion

**Symptom.** Sponza interior (`CAM_EYE=0,4,0 CAM_TARGET=8,8,6`, `P14_VGEO=1 P14_VGEO_BIN=1`): a flat
slab occludes the arch soffits in vgeo; the mesh fill is clean. Reproduces DX+VK, all-HW/all-SW,
deterministically.

**Root cause.** The LOD cut selects a coarsened (LOD1) cluster whose near-coplanar QEM collapse
**expands the wall's silhouette across the arch opening**. The QEM simplifier is planar-lossless, so
extending a flat wall edge past its true silhouette costs ≈0 error — yet it changes **coverage**,
which QEM does not measure. Evidence: `VGEO_TAU=0` (force LOD0) → 0 diff vs mesh; default `VGEO_TAU=8`
→ 9385 px slab. A per-pixel GPU probe showed the winner at a slab pixel is a LOD1 cluster
(`self_error≈0.0009`) that covers pixels its own LOD0 does not; a CPU topology scan found 103
"bridging" LOD1 triangles fanning from an interior hub with 10× the median edge length. Because the
tiny `self_error` projects to a sub-pixel screen error, the cut keeps it at essentially any `tau>0`
— lowering `tau` cannot fix it (and would be a magic number); **the error metric must be fixed.**

*(The prior "not LOD" call was wrong for two reasons: an 8-bit `saturate(self_err)` debug rounded
`0.0009` to black — miscategorising LOD1 as LOD0 — and `DEBUG_VIEW=12` distance is tonemapped, so its
absolute magnitudes were bogus.)*

**Fix.** `simplify_subset` now bounds the returned error by the **geometric span** the collapse
introduces (the longest resulting triangle edge at the survivor), not only the QEM plane distance:

```rust
let mut span = 0.0f64;
for &ti in &incident[c.v as usize] {
    if let Some(t) = tris[ti] {
        for k in 0..3 { span = span.max((pos[t[k] as usize] - pos[t[(k+1)%3] as usize]).length()); }
    }
}
max_err = max_err.max(c.cost.sqrt()).max(span);
```

So `self_error ≥ triangle span`, and the runtime cut (`vgeo_cut.slang::screen_error`) only selects a
coarse cluster once its triangles are sub-`tau` **on screen** — distance-appropriate, matching the
always-LOD0 mesh fill up close. Standard Hausdorff-style simplification bound, not a per-scene knob.

**Results.** Soffit vgeo-vs-mesh 9385 → 0 px. LOD still active: the bridging clusters' `self_error`
went 0.0 → 197 (their span), deferred to far range; a far view (`CAM_EYE=0,25,40`) is 0 diff (coarse
LODs selected AND sub-pixel-correct). No re-cook (vgeo builds the DAG at runtime). `planar_grid`
unit test updated (planar simplify now reports a bounded span error — the point of the fix).

---

## 2. Copper-sphere pole "crown" — loose normal-cone cull

**Symptom.** A jagged crown of back-face bleed at the copper sphere's top pole (gallery), present at
LOD0 and in all raster configs (so not LOD, not HW/SW binning).

**Root cause.** The normal-cone backface **cluster** cull (`vgeo_cut.slang::cone_backfacing`) dropped
clusters whose *center* cone faced away from the camera but ignored the cluster's **angular extent**,
so a cluster straddling the grazing silhouette (near edge still front-facing) was wrongly culled —
leaving coverage gaps that back faces bled through (a facing debug confirmed the crown pixels were
back-winding). The naive greedy clusterizer also produces loose, inaccurate cones.

**Fix.** (a) Corrected `cone_backfacing` to the radius-aware conservative form (`dot(axis,d) >
sin(halfAngle + asin(radius/dist))`, the meshoptimizer/niagara test), and (b) **gated the cull OFF**
(`CONE_CULL_ENABLED = false`) — with the current loose cones even the conservative test leaves a
residual apex gap. The depth test discards backfacing fragments correctly anyway; §3's per-triangle
cull is the precise replacement. Re-enable the cone cull once the LOD build produces tight cones (the
planned Nanite-style spatial clustering); the corrected `cone_backfacing` is then the form to use.

**Results.** Copper crown gone (vgeo-vs-mesh 1e-05); gallery DX≡VK 0.000002.

---

## 3. Per-triangle backface cull, gated by glTF `doubleSided`

UE Nanite does **not** do per-cluster cone culling (no cone/backface test in `NaniteCulling.ush`);
`NaniteRasterizer.ush` backface-culls **per triangle** by the screen-winding sign, gated by a
per-material two-sided flag. We do the same, driven by glTF **`doubleSided`** — which both makes the
cull safe (never cull genuinely two-sided surfaces) and recovers the raster work of back faces.

**Import → cull path.** `GltfMaterial.double_sided = mat.double_sided()` (glTF default `false`),
serialized into the `.dcasset` material chunk (format `VERSION` 5→6, so old caches re-cook). Threaded
`MaterialDesc.two_sided` → `SceneObject.two_sided` → `VgeoMaterial.two_sided` → per-object
`cull_backface = !two_sided`. Procedural/gallery materials pass `two_sided: true` (unchanged render).
The SW raster (`vgeo_swraster.slang`) and HW mesh shader (`vgeo_hwvis.slang`) skip backface triangles
for single-sided materials.

**Two DX≡VK-critical details.**
1. **Sign.** glTF is CCW-front, so backface is `area > 0` in the flip-free `cull_view_proj`
   **screen-pixel** space. (The procedural sphere winds the other way but is `two_sided` — never
   culled — so it doesn't constrain the sign.)
2. **Space.** The HW mesh shader computes the winding in the SAME screen-pixel space as the SW raster
   (not NDC). NDC's tiny ~1 area magnitude flips near-edge-on triangles between DXIL and SPIR-V at
   `area≈0` (DX≡VK drifted to 0.0011); screen-pixel units (~10⁶) fix it. The Y-flipped `mvp` used for
   `SV_Position` would invert the sign on Vulkan, hence the separate flip-free `cull_mvp` push field.

**Results.** Sponza interior vgeo-vs-mesh **0.0025** (better than the 0.003 no-cull baseline — culling
also removes residual silhouette bleed); Sponza DX≡VK **0.0007**; two-sided content (curtains,
foliage) unchanged; gallery copper (two-sided) 1e-05; gallery DX≡VK 0.000002.

**Caveat.** The sign is correct for glTF's CCW-front convention. If a *procedural* mesh is ever made
single-sided, generate it CCW-front (glTF convention) so it culls correctly.

---

## Debug tooling (used during the investigation, all reverted)

Root causes were found with a per-pixel GPU **probe dump** (the resolve wrote the winning triangle's
3 world verts + depth + cluster/self_error to a host buffer at a probe pixel; min-depth slot = the
composite winner), a **barycentric-inside** test (proved coverage was correct), a **facing** debug
(screen-winding front/back), and a CPU **topology scan** (`VGEO_TOPO=<page>`). All of it (resolve push
debug, `VGEO_RESOLVE_DEBUG`/`VGEO_PROBE`/`VGEO_TOPO`, `pbr DEBUG_VIEW 12/13`) was reverted.

Two lasting lessons: (a) `DEBUG_VIEW` output is **tonemapped** — use it for *relative* diffs only,
never absolute world units; (b) an 8-bit `saturate(x)` debug channel silently rounds tiny values
(e.g. `self_error≈0.0009`) to black — scale small quantities before encoding.
