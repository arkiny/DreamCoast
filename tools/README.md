# `tools/` — developer tooling

Helper scripts for setting up build/debug tooling that is **not** part of the
engine and **not** committed. Each script fetches or builds something into a
gitignored subdirectory of `tools/`; only the scripts themselves live in version
control. Run them from the repository root.

Third-party licenses for everything fetched/built here are tracked in
[`../THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).

| Script | Platform | What it does | Output (gitignored) |
|--------|----------|--------------|---------------------|
| `fetch-assets.ps1` / `fetch-assets.sh` | Win / Unix | Fetch CC0 glTF sample models | `assets/` |
| `fetch-vulkan-layers.py` | any (Win-focused) | Fetch Khronos Vulkan validation layer | `tools/vulkan-layers/` |
| `build-dxc.sh` | macOS | Build DXC with Metal Shader Converter support | `tools/dxc-src/`, `tools/dxc/` |
| `rt-compare.py` | any | Diff rasterizer vs. path-tracer screenshots | montage PNG (path you pass) |
| `setup-renderdoc-mcp.ps1` | Windows | Set up the RenderDoc graphics-debug MCP server | `tools/renderdoc-mcp/` |
| `build-renderdoc-py312.ps1` | Windows | Build `renderdoc.pyd` for Python 3.12 (CLI, no VS IDE) | `tools/renderdoc-mcp/module/` |
| `renderdoc-mcp-launch.ps1` | Windows | `.mcp.json` entrypoint for the MCP server | — |

The vendored Slang compiler lives in `tools/slang/` (also gitignored; obtain
separately — see [`../docs/phase-0-foundations.md`](../docs/phase-0-foundations.md)).

---

## Shaders & assets

### `fetch-assets.ps1` / `fetch-assets.sh`
Downloads a few **CC0 1.0** glTF sample models from Khronos into `assets/`.
Public-domain; authors are credited in `assets/CREDITS.md` as a courtesy.

```powershell
pwsh tools/fetch-assets.ps1      # Windows
```
```bash
tools/fetch-assets.sh            # macOS / Linux
```

### Slang (`tools/slang/`)
`slangc` compiles `.slang` shaders to SPIR-V / DXIL / metallib. The engine
resolves it from `tools/slang/`, the `SLANGC` env var, or `PATH`. Not committed —
obtain the SDK separately (Apache-2.0).

---

## Debugging & validation

### `fetch-vulkan-layers.py`
Fetches the Khronos `VK_LAYER_khronos_validation` layer (Apache-2.0, prebuilt MSVC
package from conda-forge) into `tools/vulkan-layers/`. The engine auto-discovers
this folder at runtime and adds it to `VK_ADD_LAYER_PATH` when validation is
requested. Dev-only — validation is compiled out of release builds.

```bash
python tools/fetch-vulkan-layers.py        # requires: pip install --user zstandard
```

### `rt-compare.py`
Renders/ingests one screenshot from the rasterizer and one from the path tracer
(same camera/scene via the sandbox's headless `--screenshot-clean`) and writes a
side-by-side montage (`raster | path tracer | amplified diff`) plus per-pixel
error stats — used to track the rasterizer's approximation error.

```bash
python tools/rt-compare.py RASTER.png PATHTRACER.png OUT_MONTAGE.png [--amp N]
```

### RenderDoc MCP — graphics debugging from `.rdc` captures
Three scripts wire [`renderdoc-mcp`](https://github.com/Linkingooo/renderdoc-mcp)
into Claude Code as a **separate-process** MCP server that analyzes RenderDoc
frame captures headlessly. Full details, the Python 3.6-vs-3.12 background, and
troubleshooting are in **[`../docs/renderdoc-mcp.md`](../docs/renderdoc-mcp.md)**.

```powershell
# 1. Build a Python-3.12 renderdoc.pyd from RenderDoc source (one-time, ~10–30 min,
#    fully command-line — no Visual Studio IDE). Needs VS 2022 + Desktop C++,
#    Python 3.12 dev libs, git.
pwsh tools/build-renderdoc-py312.ps1

# 2. Clone the MCP, make a Python 3.12 venv, install it, probe `import renderdoc`,
#    and (on success) register the server in ../.mcp.json.
pwsh tools/setup-renderdoc-mcp.ps1

# 3. Restart Claude Code / reconnect MCP and approve the 'renderdoc' server.
```

`renderdoc-mcp-launch.ps1` is the command `.mcp.json` runs; you don't invoke it
directly. If the server shows as failed, run it once by hand to see the stderr
diagnostic: `pwsh tools/renderdoc-mcp-launch.ps1`.

---

## macOS Metal toolchain

### `build-dxc.sh`
Builds the DirectX Shader Compiler from source **with** Apple Metal Shader
Converter support, for the Phase 8 / M7 RT-pipeline shader build on macOS
(`rt_pipeline.slang` → DXIL → metallib). There is no official macOS DXC binary and
the `-metal` codegen path only exists when DXC is configured against the Metal
Shader Converter at build time. DXC is permissively licensed (LLVM/NCSA + MIT).
Prerequisite (install once, build-time only): Apple **Metal Shader Converter**.
See [`../docs/metal-backend.md`](../docs/metal-backend.md).

```bash
tools/build-dxc.sh
```
