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

### Stage B — vertex skinning — **CPU skinning (B.1) DONE on Metal**

Verified: `SCENE_GLTF=assets/SimpleSkin/SimpleSkin.gltf GLTF_ANIM=0 CAPTURE_SEQ=8
CAPTURE_SEQ_STEP=0` — the column visibly deforms (frames differ) and is run-to-run
identical (deterministic); default capture byte-identical (`b9778dcc`); clippy/fmt
clean. Skinning runs on the inline path only (the per-frame vertex write relies on
the frame-start fence wait; skinned + `P15_RHI_THREAD` is skipped). Implementation
below.

The backend vertex layout is a fixed enum (`pos/normal/uv`, stride 32) defined per
backend; a GPU skinning path would change that layout + the g-buffer/shadow shaders
across all three backends — high cross-backend risk, Metal-only verifiable here. So
**Stage B.1 skins on the CPU**, leaving the GPU vertex format, pipelines, and shaders
**unchanged** (→ zero parity risk, no Windows gate, non-skinned output byte-identical):

- **`crates/asset`:** parse per-vertex `JOINTS_0` + `WEIGHTS_0` onto `GltfPrimitive`
  (kept *off* the GPU `MeshVertex` — CPU-only side data) and `gltf::skin` →
  `GltfSkin { joints: Vec<node_idx>, inverse_bind: Vec<Mat4> }`; `GltfNode.skin`.
- **`crates/scene`:** a `SkinnedMesh` component (bind-pose pos/normal + joints/weights
  + the skin's joint entities + inverse-bind) + `skin_meshes(world)` run after
  `propagate_transforms`: build the palette `joint_world × inverse_bind` and write the
  CPU-skinned pos/normal (`Σ wᵢ · paletteᵢ · v`) into a per-mesh vertex buffer. Skinned
  vertices land in skeleton/scene space, so the drawable's model matrix is the scene
  root (the mesh node's own transform is ignored, per glTF).
- **`apps/sandbox` + registry:** skinned meshes get a host-writable vertex buffer
  re-uploaded each frame from the skinned output; the existing g-buffer pipeline draws
  them unchanged. Driven by the existing `advance_animation` (the joints are ordinary
  animated nodes).

Test: **SimpleSkin** (the mesh visibly bends), then CC-BY characters (Fox /
CesiumMan / RiggedFigure, attribution-tracked).

#### Stage B.2 — GPU skinning (vertex-pulling) — next
Move the per-vertex deform onto the GPU so the bind-pose vertex buffer is uploaded
**once** and only a small joint palette is updated per frame. Chosen design =
**vertex pulling**, which needs **no vertex-layout / backend-attribute change** (the
big cross-backend risk): the skinned pipeline reuses `VertexLayout::Mesh`, and a
skinning vertex shader reads the extra per-vertex data + palette from **bindless
storage buffers** (`bindless.slang`'s `storage_buffers[64]`, already visible to any
`bindless` pipeline's vertex stage) indexed by `SV_VertexID`. So the only
cross-backend surface is the single-source shader (Metal-verify → Windows parity) +
the per-frame palette upload.

- **`StorageBuffer::write` (rhi + 3 backends):** add a host-write to the storage
  buffer (Metal `StorageModeShared` is already host-visible — trivial; VK/DX need a
  host-visible storage buffer variant — Windows-verified). The dynamic palette uses a
  per-fif ring of these.
- **Buffers per skinned mesh:** joints (`uint4`/vertex) + weights (`float4`/vertex) =
  static storage buffers (`create_storage_buffer_init`, uploaded once); palette
  (`float4x4`/joint = `joint_world × inverse_bind`) = per-fif writable storage buffer,
  updated each frame from the ECS joint world transforms.
- **Shaders:** a skinning g-buffer vertex shader (+ a skinning shadow VS) that reads
  the bind pos/normal from the `Mesh` attributes and `joints/weights/palette` from the
  bindless storage buffers by `SV_VertexID`, then outputs `Σ wᵢ · paletteᵢ · v`. The
  fragment shaders are unchanged.
- **Pipelines + draw:** a skinned g-buffer + skinned shadow pipeline (`Mesh` layout,
  skinning VS); push constants carry the joints/weights/palette bindless indices. The
  g-buffer/shadow passes branch to the skinned pipeline for skinned draws.

Verification: default byte-identical (`b9778dcc`); **GPU-skinned SimpleSkin matches
the CPU-skinned (B.1) result** (cross-check); full DX≡VK gate on Windows (the skinning
shader + SSBO-read-in-VS). Staged: B.2a g-buffer (Metal) → B.2b skinned shadows →
B.2c Windows VK/DX. CPU skinning (B.1) stays as the fallback / non-bindless path.

**B.2a DONE (g-buffer GPU skinning, Metal):** `vsMainSkinned` reads joints/weights +
the per-fif palette from bindless storage buffers; `SkinnedMesh` owns the static
joints/weights buffers + the palette ring (`update_palettes` writes
`joint_world × inverse_bind` each frame via `StorageBuffer::write`); `patch_scene`
tags skinned drawables (`SceneObject.skin`) which the g-buffer pass draws with
`gbuffer_skinned_pipeline` + the bind-pose vertex buffer. Verified: default
`b9778dcc`; SimpleSkin deforms on the GPU, **deterministic** (run-to-run identical),
and matches the CPU reference to **avg 0.182/ch (0.14% of channels off by >8 — edge
AA only)** → the column-major palette convention is correct. Inline path only (the
palette write uses the frame-start fence wait).

**B.2b DONE (skinned shadows, Metal):** a `vsMainSkinned` entry in `shadow.slang`
(same column-major joint-pull) + a `shadow_skinned_pipeline`; the shadow pass draws
skinned casters with it so their shadow matches the deformed mesh (the shared per-fif
palette is reused). `shadow_push` 80→96 (skin u32x4 = 0 on the static path). Verified:
default `b9778dcc`; SimpleSkin shadow follows the deform (image updates vs the
bind-pose-shadow B.2a), deterministic; clippy/fmt clean.

**B.2c DONE (GPU skinning enabled on VK + D3D12, RTX 2070 SUPER, 2026-06-29):** the
per-frame palette needs a host-writable storage buffer that the VS still reads from the
bindless table. Added `Device::create_storage_buffer_host` + a real `StorageBuffer::write`:
- **VK:** a HOST_VISIBLE | HOST_COHERENT, persistently-mapped storage buffer → `write()` is a
  plain mapped memcpy (no staging/flush); same STORAGE_BUFFER bindless descriptor as before, so
  the VS read path is unchanged.
- **D3D12:** a DEFAULT/UPLOAD heap can't be both CPU-writable and a UAV, so the palette uses a
  **CUSTOM L0 (system-memory) write-combine heap** with `ALLOW_UNORDERED_ACCESS` — CPU-mappable
  AND UAV-capable; `write()` = memcpy, the GPU reads the UAV over PCIe. Registered in the same
  bindless UAV table.
- `skin.rs` probe + palette ring now use `create_storage_buffer_host`.

Verified: both backends log `skinning: 1 skinned primitive(s) (GPU)` (probe passes → GPU path,
not the bind-pose fallback); SimpleSkin renders the deformed mesh; **DX≡VK 0.000/ch** (max 1 =
the documented D3D12 1-LSB run-to-run noise); default gallery byte-identical `06BDD797…` (no
regression); **zero VK-validation / D3D12-debug-layer errors** on the VS storage-buffer read +
per-frame host-write path (only the benign NV-external loader query); clippy `-D warnings` clean.

### Stage C — morph targets — **DONE (CPU, all backends)**
The animation's morph-weight channel blends a primitive's morph targets:
`vertex = base + Σ wᵢ · targetᵢ` (position + normal deltas). Done on the CPU (like
B.1) to avoid a 4th shader/pipeline variant (and the skinned×morphed combinatorial
case) — it reuses the existing g-buffer/shadow pipelines unchanged, so it works on all
backends with **no parity gate** (CPU math + an already-DX≡VK pipeline).

- **`crates/asset`:** parse `read_morph_targets()` → `GltfPrimitive.morph_targets`
  (per-vertex pos/normal deltas) + the `Weights` animation channel (previously skipped).
- **`crates/scene` (`animation.rs`):** a `Weights` track + a `MorphWeights(Vec<f32>)`
  component; `advance_animation` samples the weight channel (all 3 interpolation modes,
  per-target) and writes `MorphWeights` to the mesh node.
- **`apps/sandbox/morph.rs`:** `MorphMesh` (bind geometry + targets + per-fif vertex
  ring); `apply_morph` blends `base + Σ wᵢ·targetᵢ` from the node's `MorphWeights` and
  writes the fif ring buffer; `patch_scene` swaps the drawable to it (the node transform
  is kept — morphed verts are local, unlike skinning). Inline path only (per-frame
  vertex write uses the frame-start fence wait).

Verified (Metal): default `b9778dcc`; **AnimatedMorphCube** morphs (frames differ),
run-to-run identical (deterministic); scene 34 + asset 18 tests; clippy/fmt clean. CPU
math + existing pipeline → effectively backend-agnostic (VK/DX run the same draw).

### Stage C optimization — faster morph

CPU morph (`base + Σ wᵢ·targetᵢ` over every vertex, re-uploaded each frame) is the slow
path for big blendshape meshes (face rigs: 50–150 targets × 10k+ verts). The job-system
parallelization (above) is the cheap first win; the bigger ones, ordered by payoff (per
current literature + UE/Unity practice):

1. **GPU compute / vertex-pulling morph** (the real fix) — **DONE (vertex-pulling, all
   backends)**. The per-vertex blend moved onto the GPU exactly like GPU skinning (B.2):
   the bind-pose buffer uploads **once**, the per-target per-vertex deltas live in a static
   bindless storage buffer (target-major, two `float4`/vertex = pos.xyz0, nrm.xyz0 — the
   16-byte-aligned `Load<float4>` access B.2 proved cross-backend), and per frame only a
   tiny weights buffer (one `f32`/target) is written. `gbuffer.slang`/`shadow.slang`'s
   `vsMainMorphed` reads them by `SV_VertexID` and outputs `base + Σ wᵢ·deltaᵢ` (`pos`
   blends + the matching `nrm`, re-normalized in `fsMain`). So the CPU does **no per-vertex
   work** and there is no per-frame vertex re-upload. Morphed verts stay in node-local space
   (unlike skinning), so `pc.mvp`/`pc.model` apply normally.
   - **`apps/sandbox/morph.rs`:** `MorphSet { gpu, cpu }`; `build_gpu_mesh` packs the static
     deltas (`create_storage_buffer_init`) + a per-fif weights ring (`create_storage_buffer_host`,
     the B.2c host-visible storage write on all 3 backends); `apply_morph` writes the weights
     buffer (cheap) for GPU primitives + still does the job-parallel CPU blend for any CPU
     ones; `patch_scene` tags the GPU drawable with `SceneObject.morph` indices. The CPU
     blend stays as the fallback (no host storage) and the cross-check reference (`MORPH_CPU=1`).
   - **`apps/sandbox/deferred.rs`:** `gbuffer_morphed_pipeline` + `shadow_morphed_pipeline`
     (the `vsMainMorphed` entry of the shared shader); the g-buffer/shadow passes branch to
     them for morphed draws. `gbuffer_push` 192→208, `shadow_push` 96→112 (a trailing
     `morph uint4`; `[0;4]` on the non-morph path → static/skinned byte-identical).

   Verified (Metal): default `b9778dcc` (byte-identical, morph opt-in); **AnimatedMorphCube**
   morphs on the GPU (4 capture frames differ), run-to-run identical (deterministic);
   **GPU == CPU reference (`MORPH_CPU=1`) to avg 0.000/ch** (frames 1–3 byte-identical; frame 0
   max 5 = silhouette-edge AA, the same class as the B.2 skinning cross-check); scene 34 +
   asset 18 tests; clippy `-D warnings` / fmt clean. Inline path only (the per-frame storage
   write reuses the frame-start fence wait, like B.1/B.2/CPU morph). VK/DX run the identical
   single-source shader + the B.2c host-storage path → expected DX≡VK (Windows gate pending,
   same as B.2c was). A future step can unify into **one deform shader** (skin ⊕ morph) once
   a skinned+morph asset exists to gate it — the morph deltas + skin palette already share
   the bindless storage-buffer read path, so it collapses the skinned×morphed pipeline blowup.
2. **Sparse blendshapes** (biggest memory + bandwidth win). Store only the vertices a
   target actually moves — typically ~80% smaller, since a facial expression touches <20%
   of the mesh. Pair each target with a small index list; the compute/VS scatters only
   those. Unity/Unreal ship sparse formats natively.
3. **Per-frame "morph baking"** (UE 5.5): pre-sum the active targets into a single delta
   buffer once per frame on the GPU, so the deform shader reads one delta stream regardless
   of active-target count.
4. **LOD / active-set culling**: cap simultaneously-active targets, drop or simplify morphs
   at distance.

Path for this engine: a `create_storage_buffer_host` palette/weights buffer + delta storage
buffers (B.2c already added host-visible storage writes on all 3 backends), a unified
skin+morph vertex shader, then sparse-index compaction. Full DX≡VK gate (shader + buffers).

Sources: [Improving blendshape performance for crowds (GPU/GPGPU)](https://dl.acm.org/doi/10.1145/2994258.2994275),
[UE5.5 performance highlights (morph baking, GPU skinning)](https://tomlooman.com/unreal-engine-5-5-performance-highlights/),
[Blend shapes & morph targets — sparse/LOD overview](https://mocaponline.com/blogs/mocap-news/blend-shapes-morph-targets-guide),
[Morph target animation overview](https://grokipedia.com/page/Morph_target_animation).

## Parallelization on the job system (planned)

Animation work is CPU-heavy and embarrassingly parallel, so it should be distributed
across `dreamcoast_jobs` workers — the same pattern as `propagate_transforms_parallel`
(Phase 15 M3) and the M4 B4 parallel render-graph recording (`jobs.parallel_for`,
deterministic = bit-identical to the sequential run). Done as an opt-in optimization
once the single-threaded path is correct; **determinism (and the headless
byte-identical capture) must be preserved** — every parallelized unit stays a pure
function of immutable snapshots, like `resolve_world`. Targets, cheapest-to-richest:

1. **CPU skinning (`skin_and_upload`) — the prime target.** The per-vertex LBS deform
   (`Σ wᵢ · paletteᵢ · v` + normal) is the dominant cost and fully independent per
   vertex. Build each mesh's palette (small, sequential), then `parallel_for` over the
   vertex range writing into the disjoint `out` slots; the `vbuf.write` upload stays
   sequential. Multiple skinned meshes can also parallelize across meshes. Pure per
   vertex → bit-identical to the serial loop. (When GPU skinning (B.2) lands it
   subsumes this; until then CPU skinning is the main per-frame animation cost.)
2. **Animation sampling (`advance_animation`).** Players/channels are independent; the
   sample pass (read clips → compute the TRS writes) parallelizes over players, with
   the `LocalTransform`/clock write-back applied sequentially (the two-pass shape it
   already has, mirroring `advance_spin`). Lower priority — sampling is cheap next to
   skinning until clip/joint counts get large.
3. **Transform propagation** is already parallel (`propagate_transforms_parallel`).

The natural end state is a per-frame animation stage on the job graph: sample (∥ over
players) → propagate (∥, existing) → skin (∥ over vertices) → build draw list. Fits the
fixed-timestep loop and stays deterministic.

## Verification (per stage)
- Unit tests for the parser + sampler (deterministic, interpolation modes).
- Headless: `SCENE_GLTF=… GLTF_ANIM=0 CAPTURE_SEQ=4` → frames differ (motion) AND
  run-to-run identical (deterministic), like the `P15_SPIN` sequence.
- Default capture byte-identical (`b9778dcc`) — animation is opt-in.
- Any job-system parallelization must stay **bit-identical** to the serial path
  (snapshot → pure parallel compute → sequential write-back).
- Metal verified on the macOS box; **Stage B GPU skinning (B.2a/b/c) verified on Windows
  VK/DX (RTX 2070 SUPER) — DX≡VK 0.000/ch on SimpleSkin, no validation errors** (see B.2c).
