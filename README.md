# DreamCoast

> A from-scratch Rust graphics engine on raw **Vulkan + D3D12 + Metal** — built as a
> human–AI pair-programming experiment.

![Sponza rendered in DreamCoast](docs/media/sponza.png)

DreamCoast is a custom renderer layered directly on **raw Vulkan (`ash`)**, **raw
Direct3D 12 (`windows-rs`)**, and **raw Metal (`objc2`)** behind one hand-rolled,
bindless-first RHI — no `wgpu`, no engine framework. The point is to understand explicit
GPU APIs (sync, descriptors, bindless, a render graph, ray tracing) by implementing them
by hand and keeping all three backends byte-for-byte in agreement.

**Backend parity is a hard rule:** every change must produce identical output on Vulkan
and D3D12 (≤ 0.001 avg/channel, verified on an RTX 2070 SUPER); Metal is at near-full parity.

## What's in it

- **Deferred PBR** — Cook-Torrance, shadow maps, image-based lighting.
- **Physically-based lighting** — sun authored in **lux**, a physical-camera **EV100**
  exposure (+ optional auto-exposure), and an atmospheric sky.
  See [`docs/physical-lighting.md`](docs/physical-lighting.md).
- **Software-RT global illumination** — a baked global distance field drives AO,
  1-bounce + multibounce diffuse GI (a DDGI-lite world irradiance volume), and hybrid
  SW-RT reflections (SSR + GDF + sky), now the default ambient. A mesh-card surface cache
  and a `RenderQuality{low,med,high}` tier sit on top.
  See [`docs/scalable-gi.md`](docs/scalable-gi.md), [`docs/gi-radiance-cache.md`](docs/gi-radiance-cache.md).
- **Hardware ray tracing** — DXR + `VK_KHR_ray_tracing`, inline `RayQuery` and a full
  RT pipeline; a path tracer is the ground-truth parity reference.
- **Render graph** — per-frame graph with transient-resource aliasing; every technique
  hangs off it.
- **Cooked assets** — meshes, scene SDF/albedo bakes, and BCn-compressed textures cook to
  one deterministic `.dcasset`; a self-made ECS + glTF hierarchy import, RON levels, and
  camera-driven chunk streaming. Convention: **1 unit = 1 metre**.

Phases 0–12 are complete; current work is the physically-based lighting track and Phase 13
(skeletal animation + GPU skinning). The full plan — including a Phase 15+ runtime/tooling
layer and the macOS Metal backend — is in [`docs/ROADMAP.md`](docs/ROADMAP.md), with a
reviewed design doc per phase in [`docs/`](docs/).

## Built with an AI agent

Every milestone is designed and — once approved — implemented **together with
[Claude Code](https://claude.com/claude-code)** as a pair programmer: the human sets
direction, reviews, and decides trade-offs; the agent explores the codebase, writes the
implementation, runs both backends, and keeps the plans honest. The reviewed plans and
per-phase commits are part of the artifact — a case study in AI-assisted systems
programming on a project that doesn't fit in a single prompt.

## Build & run

```bash
# Windows
cargo run -p sandbox -- --backend vulkan      # or: --backend d3d12

# macOS (defaults to Metal)
cargo run -p sandbox -- --backend metal
```

Shaders compile from a single Slang source to SPIR-V + DXIL + metallib via `slangc`
(resolved from `tools/slang/`, `SLANGC`, or `PATH`). Sample glTF assets (CC0) are fetched
at runtime, not committed: `tools/fetch-assets.sh` or `pwsh tools/fetch-assets.ps1`. All
developer tooling (asset/layer fetchers, shader compilers, the raster-vs-path-tracer diff,
the RenderDoc MCP server) is documented in [`tools/README.md`](tools/README.md); macOS /
Metal setup is in [`docs/metal-backend.md`](docs/metal-backend.md).

## Tech stack

| Area | Choice |
|------|--------|
| Language | Rust (cargo workspace, edition 2024) |
| Vulkan / D3D12 / Metal | `ash` · `windows-rs` · `objc2` (all raw) |
| RHI | hand-rolled, enum-dispatch, bindless-first |
| Shaders | Slang → SPIR-V + DXIL + metallib (single source) |
| UI · Math | Dear ImGui (custom RHI renderer) · `glam` |

## Workspace layout

```
crates/  core · platform · shader · rhi-types · rhi-{vulkan,d3d12,metal} · rhi · gui · asset · render
apps/    sandbox   # technique playground executable
```

The engine crates carry the `dreamcoast-` prefix; the `rhi-*` crates are the Render
Hardware Interface layer. Planned crates for later phases (`anim`, and the Phase 15+
`jobs`/`physics`/`audio`/`script`/`net`/`ui`/`vfx`/`ai` facades + a standalone editor) are
described in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## License

[MIT](LICENSE). Third-party licenses are in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
