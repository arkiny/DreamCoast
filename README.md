# DreamCoast

> A from-scratch Rust graphics engine (raw Vulkan + D3D12), built as a human–AI
> pair-programming experiment.

DreamCoast is a custom renderer + engine layered directly on **raw Vulkan
(`ash`)** and **raw Direct3D 12 (`windows-rs`)** — no `wgpu`, no engine
framework. The goal is to deeply understand explicit GPU APIs (synchronization,
descriptors, bindless, the render graph, ray tracing) by implementing them by
hand, behind a single self-designed RHI.

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
| RHI      | hand-rolled, enum-dispatch over both backends, bindless-first |
| Shaders  | Slang → SPIR-V + DXIL (single source) |
| UI       | Dear ImGui via a custom RHI renderer |
| Math     | `glam` |
| Platform | Windows |

## Status

Backend parity is a hard rule: every milestone must produce identical results on
**both** Vulkan and D3D12 (verified on an RTX 2070 SUPER).

- [x] **Phase 0** — Foundations (workspace, Slang build pipeline, Win32 windowing)
- [x] **Phase 1** — RHI core + Vulkan backend (hello triangle)
- [x] **Phase 2** — D3D12 backend parity (runtime backend switch)
- [x] **Phase 3** — Dear ImGui + bindless descriptors
- [x] **Phase 4** — Asset pipeline + textured mesh rendering (glTF, depth, camera)
- [x] **Phase 5** — Render graph + transient memory aliasing
- [ ] **Phase 6** — PBR renderer (forward+/deferred)
- [ ] **Phase 7** — Compute / GPGPU
- [ ] **Phase 8** — Ray tracing (DXR + VK_KHR)
- [ ] **Phase 9** — Tooling & profiling

## Build & run

```bash
cargo run -p sandbox -- --backend vulkan   # or: --backend d3d12
```

Shaders compile via Slang's `slangc`, resolved from `tools/slang/`, the `SLANGC`
env var, or `PATH` (see [`docs/phase-0-foundations.md`](docs/phase-0-foundations.md)).
Vulkan validation layers are optional and **dev-only** (compiled out of release
builds) — see [`docs/vulkan-validation-setup.md`](docs/vulkan-validation-setup.md).

## Workspace layout

```
crates/
├── core/        # dreamcoast-core      — logging, errors, handle/pool, math re-export
├── platform/    # dreamcoast-platform  — Win32 windowing + input
├── shader/      # dreamcoast-shader    — Slang → SPIR-V/DXIL build pipeline
├── rhi-types/   # rhi-types            — backend-agnostic RHI descriptors/enums
├── rhi-vulkan/  # rhi-vulkan           — ash Vulkan backend
├── rhi-d3d12/   # rhi-d3d12            — windows-rs D3D12 backend
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
