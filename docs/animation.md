# glTF animation playback (staged)

Status: **Stage A in progress** (node TRS). Stages B/C follow.

The engine imports glTF scenes into the ECS as a `LocalTransform` hierarchy
(`instantiate_gltf`) and already drives per-frame motion through that hierarchy
(`Spin` + `advance_spin` → `propagate_transforms_parallel` → `build_scene`). glTF
animation is the **general case of that same loop**: instead of a constant angular
velocity, sample authored keyframe tracks each fixed step and write the result into
the targeted nodes' `LocalTransform`. No renderer change is needed for node
animation — the existing draw path picks up the moved transforms.

CC0 test assets (fetched by `tools/fetch-assets.sh`): **AnimatedCube**,
**InterpolationTest** (node TRS + all 3 interpolation modes), **SimpleSkin** (the
minimal vertex skin), **AnimatedMorphCube** (morph targets).

Drive it: `SCENE_GLTF=<path>` imports the glTF as the ECS scene; `GLTF_ANIM[=<i>]`
plays clip `i` (default 0). Headless determinism: the fixed-timestep / capture-seq
path advances one deterministic step per frame (exactly like `P15_SPIN`), so
`CAPTURE_SEQ=N` dumps a reproducible motion sequence. Default (no `GLTF_ANIM`) is a
no-op → byte-identical.

## Stages

### Stage A — node TRS animation (this doc) — in progress
Translate/Rotate/Scale tracks on nodes. No vertex-format change, no shader change.
- **`crates/asset` (`gltf_scene.rs`):** parse `doc.animations()` into
  `GltfScene.animations: Vec<GltfAnimation>`. Each channel = target node index +
  TRS property + sampler (interpolation + input times + typed outputs). Morph-weight
  channels are skipped here (Stage C). Pure CPU data, additive.
- **`crates/scene` (`animation.rs`):** an `AnimationClip` (channels resolved to
  target entities) + an `AnimationPlayer` component (`{ clip, time, speed }`) + a
  sampler (`Step` / `Linear` (slerp for rotation) / `CubicSpline` Hermite) +
  `advance_animation(world, dt)` — the `advance_spin` analogue: advance each player's
  clock (looping `mod duration`), sample every channel, write the target's
  `LocalTransform` T/R/S. CPU-only, no backend touch → no cross-backend risk; unit
  tested (sampler values, determinism, looping).
- **`gltf_instance.rs`:** add `instantiate_gltf_mapped` returning the node-index →
  entity map so a clip's channels can be resolved to entities.
- **`apps/sandbox`:** in the `SCENE_GLTF` branch, build clips from
  `gscene.animations` + the node map and attach an `AnimationPlayer`; call
  `advance_animation` next to `advance_spin` in the frame loop.

### Stage B — vertex skinning (next)
`MeshVertex` += `joints[u8;4]` + `weights[f32;4]`; parse `JOINTS_0/WEIGHTS_0` +
`gltf::skin` inverse-bind matrices; a joint-matrix palette (skin = per-frame joint
world × inverse-bind) fed to a skinning vertex shader. Touches the vertex format +
g-buffer/shadow vertex path → backend parity gate applies. Test: **SimpleSkin**, then
real characters (CC-BY: Fox / CesiumMan / RiggedFigure, attribution-tracked).

### Stage C — morph targets (optional)
Morph-weight channels → weighted sum of position/normal deltas. Test:
**AnimatedMorphCube**.

## Verification (per stage)
- Unit tests for the parser + sampler (deterministic, interpolation modes).
- Headless: `SCENE_GLTF=… GLTF_ANIM=0 CAPTURE_SEQ=4` → frames differ (motion) AND
  run-to-run identical (deterministic), like the `P15_SPIN` sequence.
- Default capture byte-identical (`b9778dcc`) — animation is opt-in.
- Metal verified here; VK/DX parity pending (Stage A is CPU-only → low risk; Stage B
  touches shaders → full DX≡VK gate).
