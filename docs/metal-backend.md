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
```

`--triangle-test` is cross-backend (`--backend vulkan|d3d12|metal`); enable the
Metal validation layers for a stricter smoke test:
`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 cargo run -p sandbox -- --backend metal --triangle-test --frames 30`.

## Milestone status

- **M0 — skeleton/clear:** done. Cocoa window + `CAMetalLayer` swapchain +
  acquire→clear→present verified.
- **M1 — Slang→metallib:** done. `build.rs` emits `*_metallib` accessors; the
  triangle and most vertex shaders compile. **Known issue (deferred to M3):** the
  bindless shaders (`g_textures[]` / `g_cubes[]` unbounded arrays) fail Metal
  compilation — *"flexible array member … is not at the end of struct"*. Metal
  argument buffers need a different unbounded-array declaration / binding model
  than the SPIR-V/DXIL descriptor-indexing path; this is addressed when the Metal
  bindless tables land in M3.
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
- **M3 — bindless + textures + ImGui**, **M4 — render targets + PBR**,
  **M5 — compute/async/indirect:** pending.

## Resume notes for M3+ (implementation pointers)

State to know when picking this up in a fresh session:

- **Toolchain is installed.** `slangc` at `tools/slang/bin/slangc` (v2026.11,
  gitignored); Metal toolchain downloaded (`xcrun metal` works). `model.glb`
  (Avocado) is in `assets/`.
- **Reference backend:** mirror `crates/rhi-vulkan` (its dynamic-rendering style is
  closest to Metal). The exact method contract every Metal type must satisfy is the
  `Metal(...)` arms in `crates/rhi/src/lib.rs`.
- **Current stubs:** `crates/rhi-metal/src/{device,command,resources}.rs` have
  `unimplemented!("…milestone Mx")` markers for everything past M2. Implemented:
  M0 (instance/device/swapchain/clear/fence/semaphore/queue submit+present) and
  M2 (graphics pipelines in `pipeline.rs`, buffers, draw path).
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

### M3 next (bindless + textures + ImGui)
- **Blocker:** the bindless shaders (`g_textures[]` / `g_cubes[]`) still fail Metal
  compilation (*"flexible array member … not at end of struct"*) — `build.rs`
  emits `None` for their `*_metallib` accessors. M3 must rework the bindless
  declaration into Metal argument buffers; until then only non-bindless pipelines
  (triangle) have shader bytes on macOS.
- **Buffer-index convention to revisit:** push constants sit at buffer 0 today
  (`PUSH_CONSTANT_INDEX`), but Slang shifts indices when a globals cbuffer or
  argument buffer precedes the push block (probe showed globals→`buffer(0)`,
  push→`buffer(1)`). M3/M4 will need a deliberate index map across bindless table,
  globals, and push constants. The vertex buffer is parked at index 30 to stay
  clear of those low indices.
