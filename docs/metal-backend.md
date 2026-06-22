# Metal backend (macOS)

A native Metal RHI backend (`crates/rhi-metal`) so the engine runs on macOS
alongside the Windows Vulkan / D3D12 backends, sharing the same enum-dispatch
`rhi` facade, render graph, GUI, and assets. Target parity: **through Phase 7**
(triangle → mesh → PBR deferred → compute/async/indirect). Phase 8 ray tracing on
Metal is out of scope for now (`Device::has_raytracing()` returns `false`).

## Platform layout

- **Windowing** (`crates/platform`): Win32 on Windows, hand-rolled Cocoa/AppKit on
  macOS (`window_macos.rs`) — an `NSWindow` whose content view is layer-backed by a
  `CAMetalLayer`, exposed via `Window::metal_layer()`.
- **Backends are OS-gated.** The `rhi` facade compiles `Vulkan`+`D3d12` on Windows
  and `Metal` on macOS (per-arm `#[cfg]`). `rhi-vulkan` / `rhi-d3d12` are
  `#![cfg(windows)]` (empty elsewhere) so `cargo build` works on macOS;
  `rhi-metal` is macOS-only.

## Toolchain setup (macOS)

Two tools are needed to compile shaders to Metal:

1. **Xcode + Metal Toolchain.** Full Xcode (not just Command Line Tools) plus the
   Metal toolchain component:
   ```sh
   xcodebuild -downloadComponent MetalToolchain
   xcrun metal --version   # verify
   ```
2. **Slang (`slangc`).** Download the macOS build and place it under
   `tools/slang/` (gitignored), or point `SLANGC` at it, or add it to `PATH`:
   ```sh
   curl -fsSL -o /tmp/slang.tar.gz \
     https://github.com/shader-slang/slang/releases/download/v2026.11/slang-2026.11-macos-aarch64.tar.gz
   mkdir -p tools/slang && tar -xzf /tmp/slang.tar.gz -C tools/slang
   tools/slang/bin/slangc -v   # verify
   ```

`crates/shader/build.rs` resolves `slangc` (SLANGC → `tools/slang/bin/slangc` →
PATH → `VULKAN_SDK`) and, on macOS, compiles each shader to a `.metallib` via
Slang's Metal target (which shells out to `xcrun metal`). If `slangc` or the Metal
toolchain is absent the build still succeeds — shader accessors just return `None`.

## Assets

Sample glTF models (CC0) are fetched at runtime, not committed. On macOS:
```sh
tools/fetch-assets.sh    # Avocado (default model.glb), BoomBox, Lantern
```
(Windows: `pwsh tools/fetch-assets.ps1`.)

## Running

```sh
cargo run -p sandbox -- --backend metal --clear-test          # M0 clear loop
cargo run -p sandbox -- --backend metal --clear-test --frames 60   # headless smoke test
cargo run -p sandbox -- --backend metal --triangle-test       # M2 RGB triangle
cargo run -p sandbox -- --backend metal --triangle-test --frames 60   # headless
cargo run -p sandbox -- --backend metal --triangle-test --screenshot tri.png  # capture + exit
cargo run -p sandbox -- --backend metal --mesh-test           # M3 textured bindless mesh + ImGui
cargo run -p sandbox -- --backend metal --mesh-test --screenshot mesh.png  # capture + exit
cargo run -p sandbox -- --backend metal                        # M4 full deferred-PBR scene
cargo run -p sandbox -- --backend metal --screenshot scene.png # M4 scene + ImGui, capture + exit
cargo run -p sandbox -- --backend metal --screenshot-clean scene.png  # M4 scene, 3D only
```

The flagless real renderer (M4) needs `assets/model.glb` (`tools/fetch-assets.sh`).
Phase-7 compute extras (`P7_COMPUTE_POST` / `P7_PARTICLES` / `P7_CULL`) are M5 on
Metal and stay off there.

`--triangle-test` and `--mesh-test` are cross-backend
(`--backend vulkan|d3d12|metal`); enable the Metal validation layers for a stricter
smoke test:
`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 cargo run -p sandbox -- --backend metal --mesh-test --screenshot mesh.png`.

## Milestone status

- **M0 — skeleton/clear:** done. Cocoa window + `CAMetalLayer` swapchain +
  acquire→clear→present verified.
- **M1 — Slang→metallib:** done. `build.rs` emits `*_metallib` accessors; the
  triangle and most vertex shaders compile. **Bindless blocker (resolved for
  `mesh`/`imgui` in M3):** loose unbounded `g_textures[]` / `g_cubes[]` arrays fail
  Metal compilation — *"flexible array member … is not at the end of struct"*. The
  fix is a bounded `ParameterBlock` (see M3 below). The cube/storage shaders still
  carry the loose-global form and stay `None`-on-metallib until they migrate in
  M4/M5.
- **M2 — triangle:** done. `MTLRenderPipelineState` from per-stage metallib
  blobs (`MTLLibrary` via `dispatch_data_t`, entry by name), host-visible
  `MTLBuffer` (vertex/index/uniform), vertex descriptor for every `VertexLayout`,
  viewport/scissor, draw / draw_indexed, and push constants. Screenshot readback
  (`copy_swapchain_to_buffer` blit + `MetalBuffer::read_into`, originally tagged
  M6) was pulled forward so `--triangle-test --screenshot <path>` captures the
  rendered frame to a PNG — automated pixel verification with no display. The
  layer is set non-`framebufferOnly` so its drawable can be a blit source. Verified
  by `--triangle-test` under the Metal API + GPU validation layers. **Binding
  convention:** push constants are bound at buffer index 0 (`setVertex/Fragment
  Bytes`) — Slang's `[[buffer(0)]]` when no globals/bindless precede them — and the
  vertex buffer at index `30`. The globals (M4) and bindless (M3) paths shift
  Slang's index assignment and will revisit this; see `resources.rs`.
- **M3 — bindless + textures + ImGui:** done. The `mesh.slang` / `imgui.slang`
  bindless arrays were migrated to a shared `ParameterBlock` (`bindless.slang`),
  which compiles to a Metal argument buffer; `MetalDevice` builds a tier-2 argument
  buffer (`MTLResourceID` handles, no encoder), `create_texture` /
  `create_depth_buffer` upload + register into it, depth testing got an
  `MTLDepthStencilState`, and `gui` (already RHI-agnostic) renders on Metal once its
  metallib accessors were wired. Verified by `--mesh-test --screenshot` (textured
  Avocado + ImGui overlay) under the Metal API + GPU validation layers. **Details +
  the cross-backend decision are below.**
- **M4 — render targets + PBR:** **done.** The full deferred-PBR render graph runs
  on Metal — `--backend metal --screenshot` of the real scene (shadow → G-buffer →
  IBL capture/convolve → lighting → tonemap, + ImGui) renders correctly and clean
  under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`. See "M4 plan + progress" below.
  **M5 — compute/async/indirect:** pending (compute pipelines/storage buffers are
  inert placeholders on Metal, gated off in the sandbox via `compute_supported`).

## Resume notes for M4+ (implementation pointers)

State to know when picking this up in a fresh session:

- **Toolchain is installed.** `slangc` at `tools/slang/bin/slangc` (v2026.11,
  gitignored); Metal toolchain downloaded (`xcrun metal` works). `model.glb`
  (Avocado) is in `assets/`.
- **Reference backend:** mirror `crates/rhi-vulkan` (its dynamic-rendering style is
  closest to Metal). The exact method contract every Metal type must satisfy is the
  `Metal(...)` arms in `crates/rhi/src/lib.rs`.
- **Current stubs:** `crates/rhi-metal/src/{device,command,resources}.rs` have
  `unimplemented!("…milestone Mx")` markers for everything past M3. Implemented:
  M0 (instance/device/swapchain/clear/fence/semaphore/queue submit+present),
  M2 (graphics pipelines, buffers, draw path), and M3 (bindless argument buffer,
  textures + depth, depth-stencil state, ImGui). M4 next: offscreen render targets,
  cubemaps, transient heap/aliasing, globals UBO, the MRT/render-graph passes — plus
  migrating the cube/storage shaders to `ParameterBlock` (they land at a different
  set/buffer index than `mesh`/`imgui`; see the M3 bindless section).
- **objc2 0.3 notes:** most property getters/setters are *safe* (no `unsafe`); the
  few that need `unsafe` (e.g. `NSWindow::setReleasedWhenClosed`,
  `objectAtIndexedSubscript`, the `setVertexBytes`/`drawPrimitives` family) the
  compiler will flag — let it guide you. Protocol methods need the protocol trait
  in scope (e.g. `MTLCommandEncoder` for `endEncoding`, `MTLLibrary` for
  `newFunctionWithName`). `presentDrawable` was called via `msg_send!` to avoid the
  `CAMetalDrawable`→`MTLDrawable` protocol-cast dance.

### M2 facts resolved (reuse for M3+)
- **MTLLibrary from bytes:** `device.newLibraryWithData_error(&DispatchData::from_bytes(blob))`
  (from the `dispatch2` crate — copies the blob, so `'static` shader bytes are
  fine). No temp file needed. See `pipeline.rs::load_function`.
- **Entry names:** Slang's Metal target *preserves* the entry name (`vsMain` /
  `fsMain`), so `library.newFunctionWithName("vsMain")` works directly.

### M3 bindless (done) — what shipped, and the cross-backend decision

**Shader model: a shared `ParameterBlock<Bindless>` in `bindless.slang`**, included
by `mesh.slang` + `imgui.slang` (the only bindless shaders Metal compiles in M3).
It compiles to a Metal argument buffer, a Vulkan descriptor set, and a D3D12
descriptor table from one source:

```hlsl
struct Bindless { Texture2D textures[1024]; SamplerState samp; };
[[vk::binding(0, 0)]] ParameterBlock<Bindless> g;   // usage: g.textures[i].Sample(g.samp, uv)
```

**Why only `mesh` + `imgui` migrated (not all 10 bindless shaders).** Empirically
(via `slangc` reflection + `spirv-asm`, Slang 2026.11): a `ParameterBlock` with **no
globals present** lands at **descriptor set 0** with `textures`=binding 0,
`samp`=binding 1 — *byte-identical* to the previous loose-global layout, so
**`rhi-vulkan` is unchanged** and the shared `bindless_set` still matches every
shader (migrated or not). `mesh`/`imgui` have no globals, so they qualify. The
globals-using shaders (`pbr`, `gbuffer`, `capture`, `prefilter`, `irradiance`,
`blur`, `post`, …) are **not** Metal targets until M4/M5; when they migrate, note
that a loose globals `ConstantBuffer` at set 1 **bumps** the `ParameterBlock` to set
2 — so *that* migration is the bigger cross-backend change the original spike warned
about. Defer it to the milestone that needs it.

**Pins that matter (verified, don't re-walk):**
- `[[vk::binding(0, 0)]]` on the **block** pins it to set 0 even though the push
  constant declares `register(b0, space0)` (which otherwise reserves space 0 and
  bumps the block to set 1). Pinning the *inner members* (`[[vk::binding]]` or
  `register()` on `textures`/`samp`) instead **bumps the whole block's space** —
  don't.
- The sampler must live **inside** the block (a ParameterBlock owns its whole
  descriptor set; a loose sampler can't share set 0). On Vulkan this is still set 0 /
  binding 1 (matches the old immutable sampler). The original spike's unbounded
  loose-global attempts and `DescriptorHandle<T>` both failed on the Metal target —
  the bounded ParameterBlock is the only path.

**Windows parity is NOT verified (macOS-only box).** SPIR-V is confirmed unchanged
here (so Vulkan should need no change). **D3D12: the sampler moves from a static
sampler into the bindless table** (a ParameterBlock sampler is a table entry), so
`rhi-d3d12`'s root signature for `imgui`/`mesh` needs that tweak — **verify on the
RTX 2070 SUPER**. DXIL can't be compiled on macOS (no DXC), so this could not be
checked in the M3 session.

**Metal argument buffer (tier-2, no encoder).** Apple Silicon → argument buffers
tier 2: `DeviceShared` allocates a shared `MTLBuffer` and writes 8-byte
`MTLResourceID` handles directly — texture slots `0..BINDLESS_COUNT`, the shared
sampler at slot `BINDLESS_COUNT` (Slang's id for `samp`). `create_texture` /
`create_depth_buffer` register handles; `bind_graphics_pipeline` binds the buffer at
`[[buffer(1)]]` for bindless pipelines and `useResource`s the sampled textures
(argument-buffer resources need explicit residency). See `device.rs` (`register` /
`write_handle`) and `command.rs` (`bind_graphics_pipeline`).

**Buffer-index map (M3):** push constants `[[buffer(0)]]` (`PUSH_CONSTANT_INDEX`),
bindless argument buffer `[[buffer(1)]]` (`BINDLESS_BUFFER_INDEX`), vertex buffer at
30. Globals (M4) will take the next low index and shift the bindless slot for the
globals-using shaders (their block is at set 2 / `[[buffer(2)]]`).

**Gotchas hit in M3 (reuse):**
- A depth-less pipeline (ImGui) **cannot** be bound in a render pass that has a depth
  attachment — Metal validates the pipeline's `depthAttachmentPixelFormat` against
  the pass. `--mesh-test` runs the mesh in a depth pass, then ImGui in a second
  color-load pass (mirrors the engine's geometry-then-UI structure).
- Metal validates `setBytes` length against the shader argument's *alignment-padded*
  size (ImGui's `{float2, float2, uint}` is 20 bytes of data but Metal wants 24).
  `command.rs::push_constants` rounds the upload up to 16 to always cover the pad.

## M4 plan + progress

**Goal / done-when:** the full deferred-PBR render graph runs end-to-end on Metal
— `cargo run -p sandbox -- --backend metal --screenshot scene.png` produces the same
shadowed + IBL-lit result as Vulkan, clean under `MTL_DEBUG_LAYER=1
MTL_SHADER_VALIDATION=1`. Compute/culling/particles/indirect/storage buffers stay M5.

**Ordered steps** (each verified by a cross-backend `--*-test` flag, mirroring M0–M3):

1. **Shader migration to `ParameterBlock` (the cross-backend risk — do first,
   independently). — DONE (see progress log).** The M4 shaders were loose-global so
   `metallib` failed → `None` accessors. Migrated `gbuffer`, `pbr`, `capture`,
   `irradiance`, `prefilter`, `post`, `blur` (`brdf`/`sky`/`shadow` had no bindless
   resources and already compiled). Sub-points:
   - Add `TextureCube cubes[N]` to the `Bindless` struct in `bindless.slang` (pbr
     samples `g_cubes[]` at a separate `space1` today — fold it into the block).
   - **Globals-using shaders: block stays Vulkan set 0, Metal `[[buffer(2)]]`.**
     ~~A loose globals `ConstantBuffer` at set 1 pushes the `ParameterBlock` to set
     2.~~ *Corrected empirically during the `pbr` migration (Slang 2026.11):* the
     `[[vk::binding(0,0)]]` pin holds the block at **descriptor set 0** even with the
     globals UBO at set 1, so the Vulkan layout is byte-for-byte the old loose-global
     one (textures b0, samp b1, cubes b2 on set 0; globals set 1) — **`rhi-vulkan`
     untouched**. Only the *Metal buffer index* shifts: the globals UBO takes
     `[[buffer(1)]]`, pushing the bindless argument buffer to `[[buffer(2)]]` (vs
     `[[buffer(1)]]` for non-globals shaders). `pbr.slang` has globals; the IBL gen
     shaders mostly drive cube faces via push constants — check each.
   - After every shader: `slangc -target spirv-asm` and confirm the **descriptor
     set/binding layout is unchanged** (Vulkan safe). It will NOT be byte-identical:
     the array goes unbounded→`[1024]` and `RuntimeDescriptorArray` /
     `SPV_EXT_descriptor_indexing` drop out — *exactly the M3 mesh/imgui change*, so
     same risk profile (Vulkan/D3D12 parity pending the Windows RTX 2070 SUPER box).
2. **Globals UBO path.** `MetalDevice::set_globals_buffer` (store the buffer in
   `DeviceShared`), `MetalCommandBuffer::set_globals(offset)` (stash offset in a
   `Cell`, bind globals buffer in `bind_graphics_pipeline` at a new
   `GLOBALS_BUFFER_INDEX`). Add a `uses_globals` flag to `MetalGraphicsPipeline`.
   Update the buffer-index map in `resources.rs` (push 0, globals 1, bindless block 2
   for globals shaders, vertex 30).
3. **Offscreen render targets + MRT.** `MetalRenderTarget` =
   `MTLTexture{RenderTarget|ShaderRead, Private}` + bindless slot; `create_render_target`;
   `begin_rendering_target` / `begin_rendering_targets` (G-buffer = 4 color
   attachments). Barriers (`rt_to_sampled` etc.) likely stay no-ops — Metal's
   encoder-boundary hazard tracking handles read-after-write; confirm with validation.
4. **Shadow pass.** `begin_rendering_depth_only` (depth attachment only, store
   action `Store` — M3 used `DontCare`), then sample the depth in `pbr` (register the
   M3-reserved depth slot for residency in `depth_to_sampled`).
5. **Cubemaps + IBL.** `MetalCubemap` = `MTLTextureType::Cube`, 6 layers, mipped.
   Metal is simpler than the Vulkan per-(face,mip) views: render via the color
   attachment's `setSlice(face)` + `setLevel(mip)` directly. Wire
   `begin_rendering_cube_face[_depth]`, `mip_levels`, `mip_size`.
6. **Transient heap / aliasing + post.** Use `MTLHeapType::Placement` so Vulkan's
   offset model maps 1:1: `heapTextureSizeAndAlign` → `render_target_memory`,
   `heap.newTexture(descriptor, offset:)` → `create_aliased_target`. Then post
   (tonemap/bloom) and the full deferred scene runs.

Suggested verification flags: `--gbuffer-test`, `--shadow-test`, `--ibl-test`, then
the flagless real renderer. All cross-backend (`--backend vulkan|d3d12|metal`).

### M4 progress log

- **Step 1 — `gbuffer.slang` migrated (done, verified on this macOS box).** Replaced
  the loose `g_textures[]` / `g_sampler` with `#include "bindless.slang"` + `g.textures[]`
  / `g.samp` (no globals → block stays at **set 0**, like mesh/imgui). Verified:
  `gbuffer_{vs,fs}_metallib()` now return `Some` (were `None`); `dreamcoast-shader`
  builds clean. SPIR-V: **VS byte-identical**, **FS descriptor layout identical**
  (set 0, binding 0 = textures, binding 1 = sampler) with the bounded-array /
  dropped-`SPV_EXT_descriptor_indexing` change noted above. **Vulkan/D3D12 parity
  pending the Windows box** (same as M3).

- **Step 1 — `cubes[]` folded into `Bindless` + `pbr.slang` migrated (done,
  verified on this macOS box).** Added `TextureCube cubes[64]` to the shared
  `Bindless` struct (matches `CUBE_COUNT` in `rhi-vulkan`); the cube array now lives
  inside the block instead of a separate loose `g_cubes[]` at `space1`. `pbr.slang`
  switched to `#include "bindless.slang"` — `g.textures[]` / `g.cubes[]` / `g.samp`
  — and its globals UBO renamed `g`→`globals` (the block owns the name `g` and the
  whole set). Verified:
  - `pbr_{vs,fs}_metallib()` now return `Some` (`pbr_fs` was `None`).
  - **The globals→set-2 prediction was wrong** (see step 1 above): `pbr_fs` SPIR-V is
    **identical to the loose-global baseline** — block at set 0 (textures b0, samp b1,
    cubes b2), globals at set 1 — so `rhi-vulkan` needs no change. Metal MSL confirms
    `pc`=`buffer(0)`, `globals`=`buffer(1)`, bindless block=`buffer(2)`.
  - **No regression** from the shared-struct change: `mesh`/`imgui`/`gbuffer` SPIR-V
    stays at (set 0, textures b0, samp b1) — Slang drops the unused cube binding —
    and all still compile to `metallib`.
  - **Vulkan/D3D12 parity pending the Windows RTX 2070 SUPER box** (same risk profile
    as M3: bounded array + dropped `SPV_EXT_descriptor_indexing`; D3D12 sampler-in-
    table). Next: remaining step-1 shaders (`capture`, `irradiance`, `prefilter`,
    `brdf`, `sky`, `shadow`, `post`, `blur`) — `capture`/`irradiance`/`prefilter`
    still fail `metallib` on the loose `g_cubes[]`, the same fix applies. Then step 2
    (Metal globals-UBO path: bind globals at `buffer(1)`, bindless at `buffer(2)` for
    `uses_globals` pipelines).

- **Step 1 — COMPLETE: remaining shaders migrated (done, verified on this macOS
  box).** Migrated the last loose-global shaders to `#include "bindless.slang"`:
  - `post` / `blur`: `g_textures[]`+`g_sampler` → `g.textures[]`/`g.samp` (no
    cubes/globals, exactly like `mesh`/`imgui`).
  - `capture`: `g_textures[]`+`g_sampler`+`g_cubes[]` → `g.textures[]`/`g.cubes[]`/
    `g.samp` (no globals UBO — it drives faces via push constants).
  - `irradiance` / `prefilter`: `g_sampler`+`g_cubes[]` → `g.samp`/`g.cubes[]`. These
    use **only** the sampler + cube array (not `textures`), so the unused `textures`
    member sits *before* the used ones in the block. **Verified the block still
    reserves the full descriptor set:** `samp` stays at binding 1 and `cubes` at
    binding 2 (Slang does **not** compact the unused leading binding away) — SPIR-V
    descriptor layout byte-for-byte the loose-global baseline. This was the one real
    layout risk in step 1 and it held.
  - `brdf` / `sky` / `shadow` have **no** bindless resources (push constants only) and
    already compiled to `metallib` — no change, confirmed `Some`.

  Verified on this box: `post`/`blur`/`capture`/`irradiance`/`prefilter` `_fs_metallib`
  now return `Some` (were `None`); full `cargo build` clean. SPIR-V: every migrated
  shader's set/binding layout is **identical to its pre-migration baseline** (captured
  via `slangc -target spirv-asm` before/after) — only the member name changed
  (`g_sampler`→`g_samp`), with the same bounded-array / dropped-
  `SPV_EXT_descriptor_indexing` change as M3. Metal MSL confirms these non-globals
  shaders bind push constants at `[[buffer(0)]]` and the bindless argument buffer at
  `[[buffer(1)]]` (the M3 `BINDLESS_BUFFER_INDEX`). **The only `metallib` `None`
  accessors left are the M5 compute/storage shaders** (`post_compute`, `particle_sim`,
  `particle_draw`, `cull`, `cull_draw` — loose `g_textures[]` + storage buffers,
  migrate in M5). **Vulkan/D3D12 parity pending the Windows RTX 2070 SUPER box** (same
  risk profile as M3). **Step 1 done; next is step 2 (Metal globals-UBO path).**

- **Steps 2–6 — DONE (deferred scene runs on Metal, verified on this box).**
  Implemented together in `rhi-metal` (+ a sandbox gate); all verified at once by the
  real `--backend metal --screenshot` deferred render, clean under
  `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`.
  - **Step 2 — globals UBO.** `set_globals_buffer` stores the buffer in
    `DeviceShared`; `set_globals(offset)` stashes a byte offset in a `Cell`;
    `MetalGraphicsPipeline.uses_globals` (= `desc.uniform_buffer`) makes
    `bind_graphics_pipeline` bind the globals UBO at `GLOBALS_BUFFER_INDEX = 1` and
    shift the bindless argument buffer to `buffer(2)`
    (`BINDLESS_BUFFER_INDEX_WITH_GLOBALS`) — matching `pbr.slang`'s MSL.
  - **Step 3 — render targets + MRT.** `MetalRenderTarget` = `MTLTexture`
    (`RenderTarget | ShaderRead`, `Private`) registered in the texture table;
    `begin_rendering_target` / `begin_rendering_targets` (the 4-attachment G-buffer).
    No explicit barriers — Metal's encoder-boundary hazard tracking handles
    write→sample; the graph's `rt_to_*` hooks instead toggle **bindless residency**
    (see below).
  - **Step 4 — shadow pass.** `begin_rendering_depth_only` (depth attachment only,
    `Store`); `depth_to_sampled` makes the shadow map resident so `pbr` samples it as
    `g.textures[shadow_index]`.
  - **Step 5 — cubemaps + IBL.** `MetalCubemap` = `MTLTextureType::Cube` (6 faces,
    mipped) in the cube table (`bindless.slang` cube `i` → argument-buffer slot
    `BINDLESS_COUNT + 1 + i`; the argument buffer was enlarged by `CUBE_COUNT`).
    `begin_rendering_cube_face[_depth]` selects the subresource via the color
    attachment's `setSlice(face)` + `setLevel(mip)` (no per-(face, mip) view needed,
    unlike Vulkan). Sky → env (full mip chain) → scene capture → irradiance →
    prefilter all run; reflections + ambient match.
  - **Step 6 — transient heap / aliasing.** `render_target_memory` via
    `heapTextureSizeAndAlignWithDescriptor`; `create_transient_heap` =
    `MTLHeapType::Placement` + **`Tracked`** hazard mode (so aliasing/RAW hazards are
    automatic and `aliasing_barrier` stays a no-op); `create_aliased_target` =
    `heap.newTextureWithDescriptor:offset:`. The graph's default `aliasing = true`
    path is exercised.
  - **Bindless residency model (the one non-obvious design choice).** Render targets
    / cubemaps / shadow maps are both attachments (written) and bindless sampled
    (read), but Metal forbids `useResource` on a texture that is the current render
    target. So residency is **toggled by the render-graph transition hooks**:
    `*_to_sampled` adds a resource to the resident set (made resident at the next
    bindless `bind_graphics_pipeline`), `*_to_render_target` / `cube_to_color` /
    `aliasing_barrier` drop it before it is written. Static textures
    (`create_texture`) stay resident for the app's lifetime. This mirrors the Vulkan
    layout transitions 1:1 and never makes an attachment resident in its own pass.
  - **Sandbox gate.** The Phase-7 compute features (post blur / GPU particles / GPU
    culling) are M5 on Metal. `compute_supported = backend != Metal` forces those
    flags off, gates the particle seed dispatch + the `particle_draw` / `cull_draw`
    pipelines (their metallibs are still `None`), and `load_shader_pair` /
    `load_compute_shader` now feed the Metal path the `*_metallib()` accessors. The
    compute pipelines / storage buffers create as inert placeholders (never
    dispatched on Metal).
  - **Vulkan/D3D12 parity:** the shared shaders changed in step 1 (already flagged);
    steps 2–6 are Metal-backend-only Rust + a backend-neutral sandbox gate, so they
    do not alter the Windows backends. Still **verify the step-1 shader changes on
    the Windows RTX 2070 SUPER box.**
