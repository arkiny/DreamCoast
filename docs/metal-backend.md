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
```

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
- **M4 — render targets + PBR**, **M5 — compute/async/indirect:** pending.

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
