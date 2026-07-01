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

### 1. Single-source resolver + structural gallery-lock (the core — lead) — ✅ DONE
The landed shape of the resolver: every scalability knob resolves against a single **base preset**,
then applies its env override and clamp in ONE place at the consumer site:

```
let base = if gallery { quality::gallery_preset() } else { quality::preset(tier) };
// per knob (main.rs):
let knob = env_override.unwrap_or(base.knob).clamp(lo, hi);
```
- **Gallery-lock is structural.** The gallery (the byte-identical path-tracer anchor `af70c1a5…`)
  resolves against a fixed [`gallery_preset()`](../apps/sandbox/src/quality.rs) — the single table of
  each knob's legacy value that used to be scattered as per-site `if gallery_scene { <legacy> }`. A
  newly added tier knob takes its gallery value from that one table and can no longer silently drift
  the anchor by forgetting a per-site force (the bug that hit `render_scale` and
  `reflect_max_roughness`). Fields no gallery pass reads still carry their legacy value so the table
  is a complete, self-describing snapshot.
- **Clamp ranges** live at each consumer's `.clamp(..)` (e.g. `gi_res_div.clamp(1,16)`,
  `gi_atrous_steps.clamp(1,5)`, `render_scale.clamp(0.3333,1.0)`). The `preset()`/`gallery_preset()`
  defaults are validated against these ranges by the unit tests (§4), so an out-of-range preset edit
  is caught in `cargo test` rather than silently clamped at runtime.
- **No duplication.** Both construction and the UI tier live-swap resolve against the same `base`, so
  the two paths can't drift.
- The fine-grained `P_*`/`P11_*`/`SHADOW_*`/`RENDER_SCALE` env knobs remain the precise controls and
  always win over the tier (`env_override.unwrap_or(base.x)`), and `env_bool` gives symmetric on/off
  parsing (`0`/`false`/`off` => false) so a higher tier's on-by-default can be turned off via env.

### 2. Scalability groups (organizing layer, reference-engine style) — ✅ ADDED (this change)
An ADDITIVE, self-describing layer in [`quality.rs`](../apps/sandbox/src/quality.rs) that expresses a
tier as an assignment of a level `0..=3` to each of six named GROUPS — the same shape a reference
real-time engine uses (its `sg.<Group>Quality` cvars), without any product/source names:

- **`ScalabilityGroup`** enum: `Resolution` (render_scale + TAAU), `GlobalIllumination`
  (gi_spp/res_div/atrous/half_res/max_steps), `Reflection` (reflect_res_div/max_steps/roughness/
  half_res/history_clamp/ssr), `AmbientOcclusion` (gdf_ao/ssao/ao_res_div), `Shadow`
  (softness/taps), `SurfaceCache` (surface_cache/relight period/spp).
- **`groups(tier) -> [(ScalabilityGroup, u8); 6]`** returns a level for EVERY group (exhaustive), so
  the system is self-describing and per-platform extensible: a new platform tier declares its coarse
  profile as six integers alongside its precise `QualityPreset`. `group_level(tier, group)` is the
  convenience lookup. `ScalabilityGroup::ALL` is the canonical order.
- **Per-group env override** `SG_GI` / `SG_REFLECTION` / `SG_AO` / `SG_SHADOW` / `SG_RESOLUTION` /
  `SG_SURFACE_CACHE` (`ScalabilityGroup::env_name()` / `env_level()`, parsed + clamped to `0..=3`),
  which a caller MAY consult as a coarse lever.
- **Honest boundary — descriptive, not authoritative.** `groups()` REFLECTS the levels that
  `preset()` already encodes; it does NOT feed back into `preset()` or change any resolved value, so
  the byte-identical gallery/Med anchors are untouched. Wiring group-level → a concrete knob table
  can't reproduce the ~27-field presets losslessly from six `0..=3` integers, so — per the design
  constraint that the byte-identical gate wins — it is kept a documented mapping rather than a risky
  behavioral input. The fine `P_*`/`P11_*` env knobs remain the precise controls and win over group
  levels.

### 3. Live scalability UI panel (test scene)
A dedicated collapsing "Scalability" section in the ImGui panel: the tier combo, a per-group level
combo, and live sliders/toggles for the individual knobs (render_scale, res divisors, atrous steps,
cache period, reflection roughness/steps, shadow, SSR mode). Any change calls `resolve()` and
re-applies to the live `ResolvedQuality` (the graph rebuilds every frame, so it takes effect at
once). Gallery stays locked (controls that would touch the anchor are disabled/forced in the
gallery scene). This is the "robust test-scene scalability" deliverable.

### 4. Validation + tests — ✅ DONE (`#[cfg(test)] mod tests` in `quality.rs`, `cargo test -p sandbox`)
- `preset_fields_in_range` / `gallery_preset_in_range`: every tier's `preset()` and the
  `gallery_preset()` resolve within the validated ranges their consumers clamp to (gi/reflect/ao
  res_div `1..=16`, `gi_atrous_steps 1..=5`, sample counts `1..=256`, `render_scale 0.3333..=1.0`,
  `reflect_history_clamp 0..=2`, `shadow_taps 1..=16`, `reflect_max_roughness`/`gdf_cone_k` `0..=1`,
  `cache_relight_period`/`cache_relight_spp` `>= 1`, …).
- **`gallery_preset_locks_legacy_anchor` (the guardrail):** asserts `gallery_preset()` equals the
  documented byte-identical legacy values field-by-field (`gi_spp==8`, `gi_max_steps==64`,
  `cache_relight_period==1`, `cache_relight_spp==8`, `gi_half_res==false`, `render_scale==1.0`,
  `gdf_cone_k==0.0`, `reflect_history_clamp==0`, `gi_atrous_steps==2`, `gi_temporal_clamp==1.0`,
  `gdf_ao==false`, `ssao==false`, `reflect_max_roughness==0.5`, `reflect_max_steps==96`, …). This
  LOCKS the anchor config so a future preset edit can't silently drift the gallery sha. If a value
  here must change, the gallery sha changes with it — that is the point of the lock.
- `med_matches_legacy_defaults`: `Med` still reproduces the pre-tier no-regression defaults.
- `env_bool_parses`: `0`/`false`/`off` => false, other set values => true, unset => default.
- `groups_cover_every_group` / `group_env_levels_parse_and_clamp`: `groups(tier)` returns a level for
  every `ScalabilityGroup` (exhaustive) within `0..=3`, `group_level` agrees with the table, and the
  `SG_*` env names / clamping are correct.

### 5. Data-driven tier tables — ✅ ADDED (this change)
The per-tier `QualityPreset` values and per-tier group levels are now **DATA, not code**, so a tier
can be tuned or a platform tier added **without recompiling** — the reference-engine
scalability-config-file model. The tables live in
[`apps/sandbox/config/scalability.ron`](../apps/sandbox/config/scalability.ron) (RON, the same
serde/`ron` stack the `.level` loader uses).

- **Single source with a zero-risk fallback.** The file is embedded into the binary at compile time
  via `include_str!` (`EMBEDDED_CONFIG` in `quality.rs`), so the binary always ships a complete,
  valid table and works even with no on-disk file. At startup [`load_config`](../apps/sandbox/src/quality.rs)
  resolves the on-disk file against that embedded default, searching (highest precedence first):
  `SCALABILITY_CONFIG=<path>` (explicit), then the committed
  `apps/sandbox/config/scalability.ron`, then the runtime-editable `assets/config/scalability.ron`.
  The first that **exists** is read; if it **parses** it wins; on a parse (or read) **error** a
  `tracing::warn!` is logged and the loader falls back to the embedded default. The config is parsed
  once behind a `OnceLock` (static tables), and `preset()`/`groups()` keep the same public signatures
  — they now read the loaded snapshot instead of a hard-coded `match`. A partial on-disk override that
  omits a tier falls back to that tier's embedded entry, so both functions stay total.
- **Gallery-lock stays structural AND compiled-in.** [`gallery_preset()`](../apps/sandbox/src/quality.rs)
  is deliberately **NOT** data-driven — it remains hard-coded Rust, so a stray edit to the RON file
  can never silently move the byte-identical path-tracer anchor (`af70c1a5…`). Only the tier presets
  (`preset()`) and `groups()` are data-sourced. The `gallery_preset_locks_legacy_anchor` guardrail
  test is unchanged.
- **Env precedence unchanged.** The fine `P_*`/`P11_*`/`SHADOW_*`/`RENDER_SCALE`/`SG_*` env knobs
  still win over the (now data-sourced) base preset at each consumer site
  (`env_override.unwrap_or(base.knob).clamp(..)`). No resolution-site semantics in `main.rs` changed;
  `quality.rs` keeps the same public API.
- **Tests.** `embedded_config_parses_and_covers_every_tier` (the embedded RON parses and has exactly
  one entry per tier — the invariant the fallback relies on, caught at `cargo test` before shipping);
  `data_driven_presets_match_snapshot` / `data_driven_groups_match_snapshot` (the data reproduces the
  historical hard-coded tables exactly); `on_disk_override_parses_or_falls_back` (a well-formed
  override applies its edit; a malformed document is a parse error so the loader falls back). The
  §4 range/`groups`/Med-legacy tests continue to pass unchanged against the data-sourced tables.

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
