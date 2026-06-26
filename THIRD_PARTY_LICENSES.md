# Third-Party Licenses

DreamCoast itself is licensed under the [MIT License](LICENSE). It builds on the
third-party components below, each under its own license. This file is a summary;
the authoritative license for each crate ships in that crate's source (and is
reported by `cargo metadata` / `cargo license`).

All bundled components are permissively licensed (MIT / Apache-2.0 / BSD-family),
compatible with redistribution under MIT.

## Rust dependencies (linked into the binary)

| Crate | Version | License |
|-------|---------|---------|
| [`ash`](https://github.com/ash-rs/ash) | 0.38 | MIT OR Apache-2.0 |
| [`windows`](https://github.com/microsoft/windows-rs) / `windows-core` | 0.62 | MIT OR Apache-2.0 |
| [`objc2`](https://github.com/madsmtm/objc2) / `objc2-foundation` / `objc2-app-kit` / `objc2-metal` / `objc2-quartz-core` (macOS) | 0.6 / 0.3 | MIT OR Apache-2.0 |
| [`glam`](https://github.com/bitshifter/glam-rs) | 0.30 | MIT OR Apache-2.0 |
| [`imgui`](https://github.com/imgui-rs/imgui-rs) / `imgui-sys` | 0.12 | MIT OR Apache-2.0 |
| [`gltf`](https://github.com/gltf-rs/gltf) | 1.4 | MIT OR Apache-2.0 |
| [`image`](https://github.com/image-rs/image) | 0.25 | MIT OR Apache-2.0 |
| [`tracing`](https://github.com/tokio-rs/tracing) | 0.1 | MIT |
| [`tracing-subscriber`](https://github.com/tokio-rs/tracing) | 0.3 | MIT |
| [`anyhow`](https://github.com/dtolnay/anyhow) | 1 | MIT OR Apache-2.0 |
| [`thiserror`](https://github.com/dtolnay/thiserror) | 2 | MIT OR Apache-2.0 |

These pull in further transitive dependencies, virtually all under MIT and/or
Apache-2.0. A complete, exact list is reproducible with:

```bash
cargo metadata --format-version 1   # license field per package
# or: cargo install cargo-license && cargo license
```

## Bundled native code

- **[Dear ImGui](https://github.com/ocornut/imgui)** — MIT License, © Omar Cornut.
  Vendored and built through the `imgui-sys` crate.

## Build-time tooling (not linked into the binary)

These produce build artifacts (shader bytecode) but are not part of the shipped
binary. They live in the gitignored `tools/` directory and are obtained
separately.

- **[Slang](https://github.com/shader-slang/slang)** (`slangc`) — Apache-2.0.
  Compiles `.slang` shaders to SPIR-V and DXIL (Windows) and `metallib` (macOS).
- **DXC / DirectXShaderCompiler** (`dxcompiler`, bundled with Slang for DXIL
  output) — Apache-2.0 WITH LLVM-exception.
- **Apple Metal toolchain** (`xcrun metal`, from Xcode) — Apple SDK license.
  Invoked by Slang on macOS to produce `metallib`; not part of this repo and not
  shipped in the binary.

## Development-only (never shipped)

- **[Vulkan Validation Layers](https://github.com/KhronosGroup/Vulkan-ValidationLayers)**
  (Khronos) — Apache-2.0. Fetched into the gitignored `tools/vulkan-layers/`
  (`tools/fetch-vulkan-layers.py`) and loaded only in development builds;
  validation is compiled out of release builds (`cfg!(debug_assertions)`), so the
  layer is never part of a shipped artifact.
- **[renderdoc-mcp](https://github.com/Linkingooo/renderdoc-mcp)** — MIT (declared
  in its README). Cloned into the gitignored `tools/renderdoc-mcp/` and run as a
  separate MCP process for graphics debugging; never linked into the engine.
- **[RenderDoc](https://github.com/baldurk/renderdoc)** (Baldur Karlsson) — MIT.
  Built from source (gitignored `tools/renderdoc-src/`) only to produce a
  Python-3.12 `renderdoc.pyd` for the MCP above. RenderDoc-bundled **SWIG** (GPL)
  is invoked solely as a local build tool — its generated output is unrestricted by
  SWIG's GPL exception. No RenderDoc/SWIG sources or binaries are committed.

The RenderDoc MCP setup is documented in [`docs/renderdoc-mcp.md`](docs/renderdoc-mcp.md).

## Sample assets

CC0 / public-domain sample models may be fetched at runtime into the gitignored
`assets/` directory; none are committed to this repository.
