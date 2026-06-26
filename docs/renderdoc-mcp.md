# RenderDoc MCP — graphics debugging via `.rdc` captures

Wires [`Linkingooo/renderdoc-mcp`](https://github.com/Linkingooo/renderdoc-mcp)
into Claude Code as a **separate-process** MCP server. It exposes RenderDoc's
headless replay API (draw calls, pipeline state, shaders, textures, perf
counters) as MCP tools, reading `.rdc` captures of this engine's Vulkan frames.
Nothing here is linked into the Rust renderer — it is a local dev-only tool,
gitignored like the slang / DXC / Vulkan-layer tooling.

## The compatibility catch (read this first)

| Fact | Value |
|------|-------|
| Installed RenderDoc | **v1.44** (`C:\Program Files\RenderDoc`) |
| Bundled interpreter | **Python 3.6** (`python36.dll`) |
| Stock Windows `renderdoc.pyd` | built for **Python 3.6 only** |
| `renderdoc-mcp` requires | **Python 3.10+** |

A native CPython extension (`renderdoc.pyd`) can be imported **only** by the exact
`major.minor` it was built against. RenderDoc's own docs:

> On Windows by default RenderDoc builds against python 3.6 which is what it is
> distributed with. … You must use exactly the same version of python to load the
> module as was used to build it.

So the stock 3.6 module **cannot** load in a 3.12 interpreter, and the MCP server
**cannot** run on 3.6. The only way to run this on Windows without forking the MCP
is to **rebuild `renderdoc.pyd` against Python 3.12** from RenderDoc source. We
only need the headless `renderdoc` module (no `qrenderdoc`, so **no Qt**).

The build is **fully command-line — no Visual Studio IDE required.** RenderDoc's
`qrenderdoc/Code/pyrenderdoc/python.props` selects the Python version purely from
the `RENDERDOC_PYTHON_PREFIX64` env var: point it at a dir containing
`include\Python.h`, `python312.zip`, and `python312.lib` and it builds against
3.12. `build-renderdoc-py312.ps1` stages that, sets the var, and drives MSBuild on
the `pyrenderdoc_module` project (retargeting the v140 solution to v143). RenderDoc
v1.44 vendors all deps (incl. `swig.exe`) in-tree, so no submodules are needed.

## Layout

```
tools/
  setup-renderdoc-mcp.ps1     # clone MCP, make 3.12 venv, install, probe, register
  renderdoc-mcp-launch.ps1    # .mcp.json entrypoint: precondition-check + run server (stdout-clean)
  build-renderdoc-py312.ps1   # build renderdoc.pyd against Python 3.12 from source
  renderdoc-mcp/              # gitignored: repo/ + .venv/ + module/ + config.json
  renderdoc-src/              # gitignored: RenderDoc source checkout for the build
```

The MCP runs as its own process (`pwsh -> venv python -> renderdoc_mcp.server`),
launched by Claude Code from `.mcp.json`. It never touches the engine build.

## Setup

### 1. Build the Python 3.12 module (one-time, heavy — fully automated)

Prereqs (verified on this machine): **VS 2022 + Desktop C++** (MSBuild + v143
toolset), **Python 3.12 dev libs** (`python312.lib` + `Python.h`), **git**.

```powershell
pwsh tools/build-renderdoc-py312.ps1        # tag defaults to v1.44 (matches install)
```

Fully non-interactive — **no Visual Studio IDE**. The script:
1. clones RenderDoc `v1.44` (all deps incl. `swig.exe` are vendored in-tree),
2. stages a Python 3.12 override prefix (`include\`, `libs\python312.lib`,
   `python312.zip`) and sets `RENDERDOC_PYTHON_PREFIX64`,
3. verifies `python.props` selects 3.12 (`CustomPythonUsed=312`),
4. builds the breakpad deps, then `pyrenderdoc_module.vcxproj` directly via
   MSBuild (`/p:SolutionDir`, `/p:PlatformToolset=v143`, `_CL_=/WX- /wd4819` to
   defeat hardcoded `/WX` and the CP949 `C4819` locale warning),
5. copies `renderdoc.pyd` + `renderdoc.dll` into `tools/renderdoc-mcp/module/`.

Takes ~10–30 min (full RenderDoc core compile); object files cache, so re-runs are
incremental. Use `-PrepOnly` to clone + stage + verify the 3.12 selection without
the compile. Override `-PythonRoot` / `-OutDir` / `-Configuration` if needed.

### 2. Set up + register the MCP server

```powershell
pwsh tools/setup-renderdoc-mcp.ps1
```

Clones `renderdoc-mcp`, creates the 3.12 venv, `pip install -e`s it, then **probes
`import renderdoc`** under the venv. If it imports, the script registers the
server in `.mcp.json`:

```json
{
  "mcpServers": {
    "renderdoc": {
      "command": "pwsh",
      "args": ["-NoProfile", "-File", "tools/renderdoc-mcp-launch.ps1"]
    }
  }
}
```

If the probe fails (e.g. module still 3.6), it **does not** write `.mcp.json`
(avoids a perpetually-failing server) and tells you to build the 3.12 module first.

### 3. Activate in Claude Code

Restart Claude Code (or reconnect MCP) and approve the project `renderdoc` server.

## Usage

1. Capture a frame of the engine with RenderDoc → save a `.rdc`.
2. Ask Claude to inspect it; the `renderdoc` MCP tools load and analyze the
   capture headlessly (no GUI needed).

## Troubleshooting

- **`renderdoc.pyd not found` / import fails** — the 3.12 module isn't built yet
  (or is still the 3.6 one). Run `build-renderdoc-py312.ps1`, then re-run
  `setup-renderdoc-mcp.ps1`.
- **Server shows as failed in Claude Code** — run the launcher by hand to see the
  stderr diagnostic: `pwsh tools/renderdoc-mcp-launch.ps1` (it will not speak
  JSON-RPC interactively, but precondition errors print clearly).
- **`DLL load failed`** — `renderdoc.dll` must sit next to `renderdoc.pyd` in the
  module dir; the build script copies both.

## Licenses / attribution

Only the scripts, docs and `.mcp.json` in this repo are committed — they are our
own work, covered by this repository's license. **Nothing third-party is committed
or redistributed**: the cloned sources, the built `renderdoc.pyd`/`renderdoc.dll`,
the venv and its packages all live under gitignored `tools/` (same convention as
the slang / DXC / Vulkan-layer / sample-asset tooling). For reference, the things
these scripts fetch/build locally on your machine:

| Component | License | How we use it |
|-----------|---------|---------------|
| [renderdoc-mcp](https://github.com/Linkingooo/renderdoc-mcp) | MIT (declared in its README; no `LICENSE` file in the repo) | cloned + `pip install -e`, run as a separate process |
| [RenderDoc](https://github.com/baldurk/renderdoc) | MIT (© Baldur Karlsson) | built from source locally to produce the 3.12 module |
| SWIG (bundled in RenderDoc) | GPL | used only as a local build tool; SWIG's own GPL exception leaves generated output unrestricted |
| CPython 3.12 | PSF License | your existing local install; linked against, not shipped |

No third-party binaries or sources are added to version control, so there is
nothing to redistribute and no license obligations are triggered by this commit.

## Upstream caveat

The author labels renderdoc-mcp *"a personal vibe-coding project — built for fun,
not for production use."* Treat tool output as a debugging aid, not ground truth.
