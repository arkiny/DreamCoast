# Scalability system — robust redesign

> Goal: turn the ad-hoc tier system into a robust, general scalability subsystem. Consolidate the
> ~20 knobs (currently resolved twice — at construction and in the UI live-swap — each with a
> hand-coded gallery-legacy force + clamp) behind a **single source-of-truth resolver** with a
> **structural gallery-lock**, expose it **live in the test scene UI**, and organize the knobs into
> **UE-style scalability groups**. Byte-identical gates (`gallery af70c1a5`, `Med sponza 1ee08a3a`)
> must hold throughout — the whole point is that they become *impossible* to break by accident.

## Problems with the current system (quality.rs + main.rs)
1. **Duplicated resolution.** Each knob is resolved at construction (`main.rs` ~1935–2380) *and*
   re-derived in the UI tier live-swap (~3306+). Two sites to keep in sync; drift = bug.
2. **Scattered gallery-force.** Each knob hand-writes `if gallery_scene { <legacy> } else { qp.x }`
   (~25 sites). A new tier knob that forgets this **breaks the byte-identical anchor** — happened
   twice this cycle (`render_scale`, `reflect_max_roughness`).
3. **Clamp sprawl.** Each knob clamps inline at its site; no single validated range table.
4. **Flat 25-field preset**, no grouping — hard to reason about, extend per-platform, or expose.
5. **UI:** only the tier combo is live; individual scalability knobs aren't controllable in the
   test scene.

## Design

### 1. `ResolvedQuality` + single `resolve()` (the core — lead)
A `ResolvedQuality` struct holds the final, clamped, gallery-locked values the renderer consumes.
One function owns the whole resolution:

```
resolve(tier: RenderQuality, ctx: ResolveCtx) -> ResolvedQuality
  where ctx = { gallery: bool, <capability flags> }
```
Per knob, in ONE place: `env_override → else if gallery { GALLERY_LEGACY } → else preset[tier] → clamp(range)`.
- **`GALLERY_LEGACY`**: a single table of each knob's byte-identical gallery value (the values
  currently scattered as `if gallery_scene { .. }`). Structural — a knob without a gallery entry
  falls back to a safe default, and the resolver applies it uniformly, so the anchor can't drift.
- **Clamp ranges**: one `KnobRange` table; `resolve` validates every field.
- Both construction and the UI live-swap call `resolve()` — **no duplication**.

### 2. Scalability groups (UE-style organizing layer)
Group the knobs into named buckets, each with a level `0..=3` (`Low/Med/High/Epic`-ish):
`Resolution` (render_scale + TAAU), `GlobalIllumination` (gi_spp/res_div/atrous/temporal),
`Reflection` (reflect_res_div/max_steps/roughness/ssr), `AmbientOcclusion` (ao_res_div/ssao/gdf_ao),
`Shadow` (softness/taps), `SurfaceCache` (relight period/spp). A `RenderQuality` tier is then an
assignment of a level to each group (mirrors UE `sg.GlobalIlluminationQuality` etc.). Per-group
env override (`SG_GI=2`) sits alongside the existing fine-grained `P_*` overrides (which still win).
This keeps the fine knobs (single source) while adding a coarse, general, per-platform-friendly layer.

### 3. Live scalability UI panel (test scene)
A dedicated collapsing "Scalability" section in the ImGui panel: the tier combo, a per-group level
combo, and live sliders/toggles for the individual knobs (render_scale, res divisors, atrous steps,
cache period, reflection roughness/steps, shadow, SSR mode). Any change calls `resolve()` and
re-applies to the live `ResolvedQuality` (the graph rebuilds every frame, so it takes effect at
once). Gallery stays locked (controls that would touch the anchor are disabled/forced in the
gallery scene). This is the "robust test-scene scalability" deliverable.

### 4. Validation + tests
- `resolve()` clamps every field; a unit test asserts every tier × {gallery, content} resolves
  within range and that the gallery resolution equals the legacy anchor values for every knob.
- A test that every group level maps to a valid knob set.

## Gates (every step)
`gallery af70c1a5` + `Med sponza 1ee08a3a` byte-identical + determinism + `clippy -D` + Intel
Sponza ≥60fps (cool). The refactor is behavior-preserving by construction: `resolve()` must emit the
exact current values. VK/D3D12 unaffected (Metal-measured; the tier logic is backend-agnostic Rust).

## Team
- **Lead:** design + the core `resolve()`/`ResolvedQuality`/gallery-lock/validation refactor
  (byte-identical-critical, tightly coupled) + integration + gates + merge/push to main.
- **Agent A (UI):** the live Scalability UI panel in the test scene, on the stable core API.
- **Agent B (groups + docs + tests):** the scalability-group organizing layer, this doc's upkeep,
  and the resolver/group unit tests.
