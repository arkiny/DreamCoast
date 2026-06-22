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
```

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
- **M2 — triangle:** pending (pipelines, draw, push constants, vertex buffers).
- **M3 — bindless + textures + ImGui**, **M4 — render targets + PBR**,
  **M5 — compute/async/indirect:** pending.

## Resume notes for M2+ (implementation pointers)

State to know when picking this up in a fresh session:

- **Toolchain is installed.** `slangc` at `tools/slang/bin/slangc` (v2026.11,
  gitignored); Metal toolchain downloaded (`xcrun metal` works). `model.glb`
  (Avocado) is in `assets/`.
- **Reference backend:** mirror `crates/rhi-vulkan` (its dynamic-rendering style is
  closest to Metal). The exact method contract every Metal type must satisfy is the
  `Metal(...)` arms in `crates/rhi/src/lib.rs`.
- **Current stubs:** `crates/rhi-metal/src/{device,command,resources}.rs` have
  `unimplemented!("…milestone Mx")` markers for everything past M0. M0-implemented:
  instance/device/swapchain/clear/fence/semaphore/queue submit+present.
- **objc2 0.3 notes:** most property getters/setters are *safe* (no `unsafe`); the
  few that need `unsafe` (e.g. `NSWindow::setReleasedWhenClosed`,
  `objectAtIndexedSubscript`) the compiler will flag — let it guide you. Protocol
  methods need the protocol trait in scope (e.g. `MTLCommandEncoder` for
  `endEncoding`). `presentDrawable` was called via `msg_send!` to avoid the
  `CAMetalDrawable`→`MTLDrawable` protocol-cast dance.

### M2 first unknowns to solve
1. **MTLLibrary from metallib bytes.** `GraphicsPipelineDesc.vertex_bytes` /
   `fragment_bytes` are per-shader `.metallib` blobs (separate vs/fs libraries).
   Create an `MTLLibrary` from the bytes — simplest robust path: write to a temp
   file and `device.newLibraryWithURL`, or use `newLibraryWithData` (needs a
   `dispatch_data_t`). Then `library.newFunctionWithName(<entry>)`.
2. **Entry-function name inside the metallib.** Confirm whether Slang's Metal
   target preserves the entry name (`vsMain`/`fsMain`) or renames to `main`
   (`xcrun metal-objdump`/`metallib-dis`, or just probe both). The facade passes the
   entry names through `GraphicsPipelineDesc.{vertex,fragment}_entry`.
3. **Verification harness:** add a `--triangle-test` path to `apps/sandbox`
   (parallel to `run_clear_test`) that builds a pipeline from `triangle.slang`
   (non-bindless: `triangle_*_metallib` compile fine) and draws 3 vertices
   (`VertexLayout::None`, no vertex buffer). Reuse `--frames N` for headless runs.
4. Then implement `MetalBuffer` (vertex/index/uniform via `MTLBuffer`), viewport on
   the encoder, `push_constants` via `setVertexBytes`/`setFragmentBytes`.
