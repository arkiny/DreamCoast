# Launches the renderdoc-mcp server as a SEPARATE stdio process for Claude Code.
#
# This is the `command` referenced by the project's `.mcp.json`. It is NOT part of
# the engine build — RenderDoc/Python never link into the Rust renderer. The MCP
# server only reads `.rdc` capture files via RenderDoc's headless replay API and
# exposes them as MCP tools.
#
# CONTRACT: a stdio MCP server owns stdout for the JSON-RPC stream. Therefore this
# launcher writes NOTHING to stdout — every diagnostic goes to stderr. The final
# `&` hands stdout/stdin straight to the Python server.
#
# Config is produced by tools/setup-renderdoc-mcp.ps1 (tools/renderdoc-mcp/config.json).
#
# Usage (normally invoked by Claude Code via .mcp.json, not by hand):
#   pwsh -NoProfile -File tools/renderdoc-mcp-launch.ps1

$ErrorActionPreference = 'Stop'

function Die($msg) {
    [Console]::Error.WriteLine("[renderdoc-mcp] $msg")
    exit 1
}

$base = Join-Path $PSScriptRoot 'renderdoc-mcp'
$cfgPath = Join-Path $base 'config.json'
if (-not (Test-Path $cfgPath)) {
    Die "Not set up yet. Run:  pwsh tools/setup-renderdoc-mcp.ps1"
}

$cfg = Get-Content -Raw $cfgPath | ConvertFrom-Json
$venvPython = $cfg.venvPython
$moduleDir  = $cfg.moduleDir

if (-not (Test-Path $venvPython)) {
    Die "venv Python missing ($venvPython). Re-run: pwsh tools/setup-renderdoc-mcp.ps1"
}
if (-not (Test-Path (Join-Path $moduleDir 'renderdoc.pyd'))) {
    Die @"
renderdoc.pyd not found in: $moduleDir
The official Windows RenderDoc module is Python 3.6 only and cannot be imported
by this 3.12 venv. Build a 3.12-compatible module first:
    pwsh tools/build-renderdoc-py312.ps1
See docs/renderdoc-mcp.md.
"@
}

# Fast precondition probe: confirm the module actually imports under the venv
# Python (catches a 3.6-built .pyd, a missing renderdoc.dll, etc.) BEFORE we hand
# stdout to the server, so failures surface as a clear message instead of a
# cryptic mid-handshake crash. Probe output is captured (kept off our stdout).
$probe = @'
import os, sys
p = os.environ["RENDERDOC_MODULE_PATH"]
os.add_dll_directory(p)
sys.path.insert(0, p)
import renderdoc
sys.stderr.write("ok\n")
'@
$env:RENDERDOC_MODULE_PATH = $moduleDir
$probeOut = & $venvPython -c $probe 2>&1
if ($LASTEXITCODE -ne 0) {
    Die @"
renderdoc module failed to import under the venv Python:
$probeOut

Most likely the .pyd was built for a different Python (the stock Windows module
is 3.6). Rebuild it for 3.12:  pwsh tools/build-renderdoc-py312.ps1
"@
}

# Hand off: the server now owns stdin/stdout for JSON-RPC. Use the package's
# documented module entry; fall back to the server:main entry point.
& $venvPython -c "from renderdoc_mcp.server import main; main()"
exit $LASTEXITCODE
