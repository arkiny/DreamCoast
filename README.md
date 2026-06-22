# DreamCoast

> A from-scratch Rust graphics engine (raw Vulkan + D3D12 + Metal), built as a
> human–AI pair-programming experiment.

DreamCoast is a custom renderer + engine layered directly on **raw Vulkan
(`ash`)**, **raw Direct3D 12 (`windows-rs`)**, and **raw Metal (`objc2`)** — no
`wgpu`, no engine framework. The goal is to deeply understand explicit GPU APIs
(synchronization, descriptors, bindless, the render graph, ray tracing) by
implementing them by hand, behind a single self-designed RHI.

The Windows backends (Vulkan + D3D12) are complete through Phase 7 (PBR deferred,
compute/GPGPU) with Phase 8 (ray tracing) in progress; a native **Metal backend
for macOS** is being brought up in parallel — see
[`docs/metal-backend.md`](docs/metal-backend.md).

## Built with an AI agent

This repository is also a record of a way of working: every milestone was
designed and implemented **together with [Claude Code](https://claude.com/claude-code)**
acting as a pair programmer. The human sets direction, reviews, and decides
trade-offs; the agent explores the codebase, writes the implementation, runs both
backends, and keeps the plans honest.

The process is part of the artifact:

- **[`docs/ROADMAP.md`](docs/ROADMAP.md)** — the macro plan (10 phases).
- **[`docs/phase-N-*.md`](docs/)** — a detailed, reviewed plan written and signed
  off *before* each phase is implemented.
- Each phase lands as its own commit, verified on both backends.

It's a case study in what AI-assisted systems programming actually looks like on
a project that doesn't fit in a single prompt.

## Goals (graphics)

PBR / deferred shading · compute / GPGPU · hardware ray tracing (DXR +
`VK_KHR_ray_tracing`) — with a render graph as the spine every technique hangs
off of.

## Tech stack

| Area     | Choice |
|----------|--------|
| Language | Rust (cargo workspace) |
| Vulkan   | `ash` (raw Vulkan 1.3, dynamic rendering) |
| D3D12    | `windows-rs` (raw D3D12) |
| Metal    | `objc2` / `objc2-metal` (raw Metal, macOS) |
| RHI      | hand-rolled, enum-dispatch over all backends, bindless-first |
| Shaders  | Slang → SPIR-V + DXIL + metallib (single source) |
| UI       | Dear ImGui via a custom RHI renderer |
| Math     | `glam` |
| Platform | Windows (Vulkan/D3D12) · macOS (Metal) |

## Status

Backend parity is a hard rule: every milestone must produce identical results on
**both** Vulkan and D3D12 (verified on an RTX 2070 SUPER).

- [x] **Phase 0** — Foundations (workspace, Slang build pipeline, Win32 windowing)
- [x] **Phase 1** — RHI core + Vulkan backend (hello triangle)
- [x] **Phase 2** — D3D12 backend parity (runtime backend switch)
- [x] **Phase 3** — Dear ImGui + bindless descriptors
- [x] **Phase 4** — Asset pipeline + textured mesh rendering (glTF, depth, camera)
- [x] **Phase 5** — Render graph + transient memory aliasing
- [x] **Phase 6** — PBR deferred renderer (Cook-Torrance, IBL, shadows)
- [x] **Phase 7** — Compute / GPGPU (async compute, GPU particles, GPU culling + indirect draw)
- [ ] **Phase 8** — Ray tracing (DXR + VK_KHR) — 🚧 in progress (device infrastructure + capability gating done)
- [ ] **Phase 9** — Tooling & profiling

### Metal backend (macOS)

A native Metal backend brought up in milestones, targeting parity **through
Phase 7** (Phase 8 ray tracing on Metal is out of scope for now). Details and
toolchain setup: [`docs/metal-backend.md`](docs/metal-backend.md).

- [x] **M0** — Cross-platform skeleton: Cocoa/`CAMetalLayer` window + clear loop
- [x] **M1** — Slang → `metallib` shader pipeline
- [x] **M2** — Triangle (graphics pipeline + draw)
- [x] **M3** — Bindless (argument buffers) + textures + ImGui
- [x] **M4** — Render targets + render graph + PBR deferred (shadow → G-buffer → IBL → lighting → tonemap)
- [ ] **M5** — Compute / async compute / indirect draw

## Build & run

```bash
# Windows
cargo run -p sandbox -- --backend vulkan   # or: --backend d3d12

# macOS (Metal) — defaults to Metal; the full deferred-PBR renderer runs (M4).
# Compute extras (Phase 7) are M5-pending on Metal. Toolchain setup in
# docs/metal-backend.md.
cargo run -p sandbox -- --backend metal                    # M4 full deferred-PBR scene
cargo run -p sandbox -- --backend metal --clear-test      # M0 clear loop
cargo run -p sandbox -- --backend metal --triangle-test   # M2 RGB triangle
cargo run -p sandbox -- --backend metal --mesh-test       # M3 textured bindless mesh + ImGui
```

Shaders compile via Slang's `slangc`, resolved from `tools/slang/`, the `SLANGC`
env var, or `PATH` (see [`docs/phase-0-foundations.md`](docs/phase-0-foundations.md)).
On macOS, metallib generation also needs the Xcode Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`); full setup is in
[`docs/metal-backend.md`](docs/metal-backend.md).
Vulkan validation layers are optional and **dev-only** (compiled out of release
builds) — see [`docs/vulkan-validation-setup.md`](docs/vulkan-validation-setup.md).

Sample glTF assets (CC0) are fetched at runtime, not committed:
`tools/fetch-assets.sh` (macOS/Linux) or `pwsh tools/fetch-assets.ps1` (Windows).

## Workspace layout

```
crates/
├── core/        # dreamcoast-core      — logging, errors, handle/pool, math re-export
├── platform/    # dreamcoast-platform  — Win32 (Windows) + Cocoa (macOS) windowing + input
├── shader/      # dreamcoast-shader    — Slang → SPIR-V/DXIL/metallib build pipeline
├── rhi-types/   # rhi-types            — backend-agnostic RHI descriptors/enums
├── rhi-vulkan/  # rhi-vulkan           — ash Vulkan backend (Windows)
├── rhi-d3d12/   # rhi-d3d12            — windows-rs D3D12 backend (Windows)
├── rhi-metal/   # rhi-metal            — objc2 Metal backend (macOS)
├── rhi/         # rhi                  — enum-dispatch RHI facade
├── gui/         # dreamcoast-gui       — Dear ImGui + custom RHI renderer
├── asset/       # dreamcoast-asset     — glTF / image loading
└── render/      # dreamcoast-render    — render graph + transient aliasing
apps/
└── sandbox/     # technique playground executable
```

The engine crates carry the `dreamcoast-` prefix; the `rhi-*` crates are the
Render Hardware Interface layer kept as their own sub-namespace.

## License

DreamCoast is released under the [MIT License](LICENSE). Licenses of the
third-party libraries and tools it uses are listed in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
