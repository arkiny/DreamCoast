# DreamCoast

> A from-scratch Rust graphics engine (raw Vulkan + D3D12 + Metal), built as a
> human–AI pair-programming experiment.

DreamCoast is a custom renderer + engine layered directly on **raw Vulkan
(`ash`)**, **raw Direct3D 12 (`windows-rs`)**, and **raw Metal (`objc2`)** — no
`wgpu`, no engine framework. The goal is to deeply understand explicit GPU APIs
(synchronization, descriptors, bindless, the render graph, ray tracing) by
implementing them by hand, behind a single self-designed RHI.

The Windows backends (Vulkan + D3D12) are complete through Phase 9 (tooling &
profiling). **Phase 11 (software-RT distance-field GI) is substantially
implemented** — GDF-based AO, 1-bounce diffuse GI, and hybrid SW-RT reflections
(SSR + GDF + sky) are now the default ambient, with a mesh-card surface cache
and a `RenderQuality{low,med,high}` tier (Stage D) layered on top.
Phases 10, 12, 13 (virtual geometry, cooked-asset pipeline, scene graph) remain
experimental / planned. The native **Metal backend for macOS**
is at near-full parity — including Phase-8 ray tracing (inline `RayQuery` **and** the
DXR-style RT pipeline via Metal Shader Converter); the main gap is the Phase-9
profiling / marker tooling. See [`docs/metal-backend.md`](docs/metal-backend.md).

Beyond the graphics core, the roadmap now extends toward a **general-purpose game
engine**: Phase 14 adds skeletal animation + GPU skinning and an FBX importer, and a
planned **Phase 15+ runtime / tooling layer** (job system, physics, audio, input,
scripting, game UI, VFX, AI, networking, and a standalone editor) sits behind
RHI-style backend facades so third-party libraries can later be swapped for
from-scratch implementations. See [`docs/ROADMAP.md`](docs/ROADMAP.md) and the
[`commercial-engine gap analysis`](docs/commercial-engine-gap-analysis.md).

## Built with an AI agent

This repository is also a record of a way of working: every milestone is
designed — and, once approved, implemented — **together with
[Claude Code](https://claude.com/claude-code)** acting as a pair programmer. The
human sets direction, reviews, and decides trade-offs; the agent explores the
codebase, writes the implementation, runs both backends, and keeps the plans honest.

The process is part of the artifact:

- **[`docs/ROADMAP.md`](docs/ROADMAP.md)** — the macro plan (Phases 0–14, a planned
  Phase 15+ runtime/tooling layer, and the Metal backend).
- **[`docs/phase-N-*.md`](docs/)** (plus topic and strategy docs) — a detailed,
  reviewed plan, with the key decisions recorded, signed off *before* the work starts.
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
- [x] **Phase 8** — Ray tracing (DXR + VK_KHR) — inline ray query + full RT pipeline/SBT
- [x] **Phase 9** — Tooling & profiling (per-pass GPU timestamps, debug markers, validation toggle, sample browser)
- [ ] **Phase 10** — Virtual geometry (cluster-LOD, GPU culling/HZB, SW raster) — experimental / planned
- [~] **Phase 11** — Software ray tracing + distance-field GI — Stages A–C + D landed on Windows:
  compute SW-RT, baked global distance field, stochastic GDF GI/AO + hybrid SW-RT
  reflections (now the default ambient, replacing captured-cube IBL), a mesh-card
  surface cache, and a `RenderQuality` tier. Stage A/B run on Metal.
- [ ] **Phase 12** — Cooked-asset pipeline (`.dcasset`) + shader bytecode cook cache — planned
- [ ] **Phase 13** — Scene graph + level streaming (self-made ECS) — planned
- [ ] **Phase 14** — Skeletal animation + GPU skinning / skin cache + FBX importer (ufbx / FBX SDK) — planned
- [ ] **Phase 15+** — Commercial runtime & tooling layer: job system, physics, audio,
  input, scripting (Luau + WASM), game UI, VFX, animation graph, AI, networking, and a
  standalone editor — strategy planned, see
  [`docs/commercial-engine-gap-analysis.md`](docs/commercial-engine-gap-analysis.md)

### Metal backend (macOS)

The Metal backend (`crates/rhi-metal`) shares the same `rhi` facade, render graph,
GUI, and assets, so it tracks the phase list above rather than carrying a separate
milestone roadmap. It is at near-full parity with the Windows backends: the
deferred-PBR renderer, the Phase-7 compute / async / indirect-draw demos, Phase-8
ray tracing via **both** the inline `RayQuery` path and the DXR-style RT pipeline
(through Apple Metal Shader Converter), and the Phase-11 Stage A/B software-RT +
distance-field-volume work all run on Metal. Toolchain setup and per-milestone
bring-up notes live in [`docs/metal-backend.md`](docs/metal-backend.md).

**Not (yet) on Metal:**

- **Phase 9 tooling** — GPU timestamp profiling and debug markers / object naming are
  stubbed (no-ops); the sample browser and validation toggles work as elsewhere.
- **RT pipeline is opt-in** — the DXR-style RT pipeline needs optional build-time
  tools (Apple Metal Shader Converter + a locally built `dxc`). Without them Metal
  falls back to the inline `RayQuery` path, the default on every backend.

## Build & run

```bash
# Windows
cargo run -p sandbox -- --backend vulkan   # or: --backend d3d12

# macOS — defaults to Metal
cargo run -p sandbox -- --backend metal
```

On macOS the Metal backend runs the full deferred-PBR renderer, the Phase-7
compute demos / async compute, and Phase-8 ray tracing. The inline `RayQuery` path
is the default; the DXR-style RT pipeline additionally runs through Apple Metal
Shader Converter when its build-time tools are available. The per-milestone bring-up flags
(`--clear-test` / `--triangle-test` / `--mesh-test`), the compute-demo env toggles,
and the Metal toolchain setup are documented in
[`docs/metal-backend.md`](docs/metal-backend.md).

Shaders compile via Slang's `slangc`, resolved from `tools/slang/`, the `SLANGC`
env var, or `PATH` (see [`docs/phase-0-foundations.md`](docs/phase-0-foundations.md)).
On macOS, metallib generation also needs the Xcode Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`); full setup is in
[`docs/metal-backend.md`](docs/metal-backend.md).
Vulkan validation layers are optional and **dev-only** (compiled out of release
builds) — see [`docs/vulkan-validation-setup.md`](docs/vulkan-validation-setup.md).

Sample glTF assets (CC0) are fetched at runtime, not committed:
`tools/fetch-assets.sh` (macOS/Linux) or `pwsh tools/fetch-assets.ps1` (Windows).

Setup and usage for all developer tooling — asset/layer fetchers, the Slang and
DXC shader compilers, the rasterizer-vs-path-tracer diff, and the RenderDoc
graphics-debug MCP server — is in [`tools/README.md`](tools/README.md).

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
Render Hardware Interface layer kept as their own sub-namespace. This is the
layout as it exists today; planned crates for the upcoming phases — `anim`
(Phase 14), and the Phase 15+ `jobs` / `physics` / `audio` / `script` / `net`
facades (+ their backends) / `ui` / `vfx` / `ai` crates and an `apps/editor` —
are described in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## License

DreamCoast is released under the [MIT License](LICENSE). Licenses of the
third-party libraries and tools it uses are listed in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
