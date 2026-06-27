# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

DreamCoast is a from-scratch Rust graphics engine layered directly on **raw Vulkan
(`ash`)**, **raw D3D12 (`windows-rs`)**, and **raw Metal (`objc2`)** behind one
hand-rolled RHI — no `wgpu`, no engine framework. **This scene/engine is destined for a
general-purpose game**, so treat every graphics change as reusable production code, not a
one-off demo (see Engineering Rules below).

## Commands

```bash
# Build (debug). The Slang→SPIR-V/DXIL/metallib compile runs automatically via
# crates/shader/build.rs; the slangc warning line is expected, not an error.
cargo build -p sandbox

# Run (Windows defaults to d3d12, macOS to metal)
cargo run -p sandbox -- --backend vulkan      # or: d3d12 | metal

# Headless capture — fixed camera, pixel-aligned with the path tracer.
# --screenshot includes the ImGui overlay; --screenshot-clean is 3D only.
cargo run -p sandbox -- --screenshot-clean out.png

# Lint / format (CI gate is clippy clean with -D warnings)
cargo clippy --all-targets -- -D warnings
cargo fmt

# Tests (only a handful of unit tests live in rhi-types / core::pool)
cargo test                                    # all
cargo test -p rhi-types                        # one crate
cargo test -p dreamcoast-core pool             # one module's tests

# Path-tracer parity: render a raster + a P8_PATHTRACE=1 capture from the SAME
# camera, then diff. This is the canonical success metric for lighting changes.
cargo run -p sandbox -- --screenshot-clean raster.png
P8_PATHTRACE=1 cargo run -p sandbox -- --screenshot-clean pt.png
python tools/rt-compare.py raster.png pt.png montage.png

# Sample glTF assets (CC0) are fetched at runtime, not committed:
tools/fetch-assets.sh          # or: pwsh tools/fetch-assets.ps1  (Windows)
```

Toolchain: Rust stable, **edition 2024**, rustfmt `max_width = 100`. `slangc` is resolved
from `tools/slang/`, the `SLANGC` env var, or `PATH`. Vulkan validation layers are dev-only
(compiled out of release; `--no-validation` disables). Full tooling setup: `tools/README.md`.

## Architecture (big picture)

**RHI facade over three raw backends.** `rhi` (crates/rhi) is an enum-dispatch facade; the
real implementations are `rhi-vulkan` (ash), `rhi-d3d12` (windows-rs), `rhi-metal` (objc2).
`rhi-types` holds the backend-agnostic descriptors/enums both sides speak. Backend is chosen
at runtime (`--backend`), so all three must accept the same RHI calls. The design is
**bindless-first** (one big descriptor table / argument buffer, indexed by push constants).

**Backend parity is a hard rule.** Every change must produce byte-near-identical output on
Vulkan and D3D12 (verified on an RTX 2070 SUPER); the bar is **DX≡VK ≤ 0.001 avg/channel**.
A divergence is a bug, usually in cross-backend layout (e.g. push-constant alignment, Y-flip).

**Single-source shaders.** Author once in `crates/shader/shaders/*.slang`; `crates/shader/
build.rs` compiles each to SPIR-V + DXIL + metallib and generates accessor fns
`dreamcoast_shader::<name>_<stage>_<backend>() -> Option<&'static [u8]>`. Shaders pick
per-backend bytecode via `app::load_shader_pair` / `load_compute_shader`. Cross-backend
conventions live in the shaders: clip-space **Y-flip** for Vulkan, and HLSL/SPIR-V cbuffer
packing differences (see gotchas).

**Render graph.** `crates/render` provides a per-frame graph with transient-resource aliasing;
passes declare reads/writes and the graph schedules barriers/memory. The sandbox rebuilds the
graph every frame.

**The sandbox is the engine.** `apps/sandbox` is the technique playground where the renderer
actually lives (there is no separate engine binary). Key modules:
- `main.rs` — the frame loop, scene setup, `Globals` UBO assembly, camera, and the master list
  of env-var feature flags. The render loop wires every pass into the graph.
- `deferred.rs` — the deferred backbone: shadow-depth, **G-buffer fill** (4 MRT: albedo+AO,
  world normal, material[metallic/rough/AO], **world position**), the full-screen **PBR
  lighting** pass, and tonemap. `push.rs` holds the push-constant byte packers.
- `gdf.rs` / `gi.rs` / `reflect.rs` — Phase 11 software ray tracing against a baked **global
  distance field** (GDF): AO, 1-bounce diffuse GI (`gdf_gi`), GGX reflections (`gdf_reflect`),
  and a mesh-card **surface cache** (`sdf_cache_*`).
- `rt.rs` — hardware ray tracing (DXR + VK_KHR): BLAS/TLAS, the **path tracer** (ground truth),
  and the Cornell-box scene.
- `ibl.rs` — procedural sky → env cube → irradiance/prefilter cubes + the BRDF LUT.
- `cull.rs`, `particle.rs`, `mesh.rs` — Phase 7 GPU culling / particles / mesh upload.
- `app.rs` — CLI/screenshot plumbing; `smoketest.rs` — `--clear/triangle/mesh-test` bring-ups.

**Frame flow (deferred + SW-RT):** shadow map → G-buffer → compute GDF passes (AO / GI /
reflection composite, when enabled) → PBR deferred lighting (reads G-buffer + IBL + the SW-RT
reflection radiance) → tonemap. `P8_PATHTRACE=1` swaps the tonemap source for the path
tracer's accumulation (same camera) as the parity reference. **Default ambient = SW-RT
specular + GDF GI**; `P11_LEGACY_IBL` restores the captured-cube IBL.

**Feature flags are env vars.** Most milestones gate behind env vars read in `main.rs`
(`P7_*` compute, `P8_*` ray tracing, `P11_*` SW-RT/GDF, `DIAG_*` diagnostics, `DEBUG_VIEW`,
`SHADOW_SOFTNESS`/`SOFT_SHADOWS`, `PROFILE_GPU`). Grep `std::env::var` in `apps/sandbox/src`
for the full list before adding a new one.

**Process.** Each phase gets a reviewed plan in `docs/phase-N-*.md` (and topic docs like
`docs/rt-pbr-parity.md`, `docs/shadow-reflection-quality.md`) written/approved *before*
implementation, then lands as its own verified commit. `docs/ROADMAP.md` is the macro plan.

## Engineering rules (graphics work)

The scene/engine targets a future general-purpose game, so:

1. **Fix root causes, not symptoms.** No magic-number or per-scene patches. Find the cause at
   the shader / pipeline / data-flow level and generalize so it's correct for *every* object,
   angle, and future scene — not the one case in front of you.
2. **Always weigh optimization.** Game runtime cost matters: prefer fewer taps / bandwidth /
   dispatches for the same result, measure the cost of anything non-free (`PROFILE_GPU`), and
   default to the cheapest *accurate* path.
3. **Design for scalability.** Keep quality parameters (sample counts, radii, toggles) in one
   place (a shader constant block + spare `globals` slots) so they can later split into a
   `RenderQuality{low,med,high}` tier. Make heavy features opt-in (default off + an env/flag
   fallback "seam").
4. **Single source of truth.** Shared materials/constants (e.g. `GROUND_ALBEDO`) are defined
   once (Rust) and passed to every consumer (raster + SW-RT GI/reflection). Never duplicate a
   value across sites — drift is a bug.
5. **Verify, then claim.** For each change run `tools/rt-compare.py` (path-tracer residual),
   confirm DX≡VK parity and no regression, and report the numbers honestly.

## Diagnostic env tools (sandbox, all off by default)

- `DIAG_OBJ=<i>` / `DIAG_COPPER` (=scene[2]) / `DIAG_CUBE` (=scene[3]) — tight single-object
  orbit; `DIAG_ANGLE=<deg>` pins azimuth, `DIAG_PITCH=<deg>` elevation (90 = straight down).
- `DEBUG_VIEW=<n>` — pbr debug views headless (1 albedo, 2 normal, 3 metallic, 4 roughness,
  5 world-pos, 6 AO, 7 direct, 8 IBL ambient, 9 GDF AO, 10 GDF GI).
- `SHADOW_SOFTNESS=<f>` / `SOFT_SHADOWS=0` — opt-in PCSS-lite soft shadows (default hard PCF).
- `P8_PATHTRACE=1` — render the ground-truth path tracer (same camera) for parity diffs.
