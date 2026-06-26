# One-shot setup for the renderdoc-mcp graphics-debugging MCP server, run as a
# SEPARATE process (never linked into the engine). Idempotent — safe to re-run.
#
# What it does:
#   1. Clones https://github.com/Linkingooo/renderdoc-mcp into tools/renderdoc-mcp/repo
#   2. Creates a Python 3.12 venv at tools/renderdoc-mcp/.venv and `pip install -e`s it
#   3. Resolves the RenderDoc module dir (where renderdoc.pyd lives)
#   4. Probes whether `import renderdoc` works under the venv Python
#   5. Writes tools/renderdoc-mcp/config.json (consumed by renderdoc-mcp-launch.ps1)
#   6. If the probe passes, registers the server in .mcp.json (otherwise tells you
#      to build a 3.12 module with build-renderdoc-py312.ps1 and re-run)
#
# Why a 3.12 module is needed: the stock Windows RenderDoc ships a Python 3.6
# module, and renderdoc-mcp requires Python 3.10+. A native .pyd only imports
# into the exact CPython it was built against — so the module must be rebuilt for
# 3.12. See docs/renderdoc-mcp.md.
#
# License: renderdoc-mcp is MIT (declared in its README). It is cloned and run
# locally as a separate process; nothing from it is committed/redistributed here
# (the clone, venv and module all live under gitignored tools/). See
# docs/renderdoc-mcp.md "Licenses / attribution".
#
# Usage:
#   pwsh tools/setup-renderdoc-mcp.ps1
#   pwsh tools/setup-renderdoc-mcp.ps1 -ModuleDir "C:\path\to\renderdoc-3.12-module"

[CmdletBinding()]
param(
    # Directory containing a Python-3.12-built renderdoc.pyd + renderdoc.dll.
    # Defaults to the output dir of build-renderdoc-py312.ps1.
    [string]$ModuleDir,
    [string]$RepoUrl = 'https://github.com/Linkingooo/renderdoc-mcp.git',
    [string]$PythonVersion = '3.12'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$base = Join-Path $PSScriptRoot 'renderdoc-mcp'
$repo = Join-Path $base 'repo'
$venv = Join-Path $base '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
if (-not $ModuleDir) { $ModuleDir = Join-Path $base 'module' }

New-Item -ItemType Directory -Force $base | Out-Null
New-Item -ItemType Directory -Force $ModuleDir | Out-Null

# --- 1. Clone / update the upstream MCP repo -------------------------------
if (Test-Path (Join-Path $repo '.git')) {
    Write-Host "==> Updating renderdoc-mcp repo"
    git -C $repo pull --ff-only
} else {
    Write-Host "==> Cloning renderdoc-mcp -> $repo"
    git clone --depth 1 $RepoUrl $repo
}

# --- 2. Python 3.12 venv + editable install --------------------------------
if (-not (Test-Path $venvPython)) {
    Write-Host "==> Creating Python $PythonVersion venv -> $venv"
    & py "-$PythonVersion" -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv. Is Python $PythonVersion installed? (py -$PythonVersion --version)" }
}
$ver = (& $venvPython -c "import sys;print('%d.%d'%sys.version_info[:2])").Trim()
if ($ver -ne $PythonVersion) {
    throw "venv Python is $ver but $PythonVersion expected. Delete $venv and re-run."
}
Write-Host "==> Installing renderdoc-mcp (editable) into venv"
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -e $repo
if ($LASTEXITCODE -ne 0) { throw "pip install -e failed." }

# --- 3 & 4. Probe the renderdoc module under the venv Python ---------------
$pyd = Join-Path $ModuleDir 'renderdoc.pyd'
$probeOk = $false
$probeMsg = ''
if (Test-Path $pyd) {
    $probe = @'
import os, sys
p = os.environ["RENDERDOC_MODULE_PATH"]
os.add_dll_directory(p)
sys.path.insert(0, p)
import renderdoc as rd
print("ok", getattr(rd, "__file__", "?"))
'@
    $env:RENDERDOC_MODULE_PATH = $ModuleDir
    $probeMsg = (& $venvPython -c $probe 2>&1) -join "`n"
    $probeOk = ($LASTEXITCODE -eq 0)
} else {
    $probeMsg = "renderdoc.pyd not present in $ModuleDir"
}

# --- 5. Persist config for the launcher ------------------------------------
$cfg = [ordered]@{
    venvPython = $venvPython
    moduleDir  = $ModuleDir
    repo       = $repo
    pythonVer  = $ver
    moduleOk   = $probeOk
}
$cfgPath = Join-Path $base 'config.json'
$cfg | ConvertTo-Json | Set-Content -Encoding UTF8 $cfgPath
Write-Host "==> Wrote $cfgPath"

# --- 6. Register in .mcp.json only when the module actually imports ---------
$launcher = Join-Path $PSScriptRoot 'renderdoc-mcp-launch.ps1'
$relLauncher = [IO.Path]::GetRelativePath($root, $launcher).Replace('\','/')

if ($probeOk) {
    Write-Host "==> renderdoc module imports OK under Python $ver"
    $mcpPath = Join-Path $root '.mcp.json'
    if (Test-Path $mcpPath) {
        $mcp = Get-Content -Raw $mcpPath | ConvertFrom-Json -AsHashtable
    } else {
        $mcp = @{ mcpServers = @{} }
    }
    if (-not $mcp.mcpServers) { $mcp.mcpServers = @{} }
    $mcp.mcpServers.renderdoc = @{
        command = 'pwsh'
        args    = @('-NoProfile', '-File', $relLauncher)
    }
    $mcp | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $mcpPath
    Write-Host "==> Registered 'renderdoc' server in .mcp.json"
    Write-Host ""
    Write-Host "DONE. Restart Claude Code (or reconnect MCP) and approve the 'renderdoc' server."
} else {
    Write-Host ""
    Write-Warning "renderdoc module is NOT usable yet:"
    Write-Host $probeMsg
    Write-Host ""
    Write-Host "The stock Windows RenderDoc module is Python 3.6; this venv is $ver."
    Write-Host "Build a 3.12-compatible module, then re-run this script to register:"
    Write-Host "    pwsh tools/build-renderdoc-py312.ps1"
    Write-Host "    pwsh tools/setup-renderdoc-mcp.ps1"
    Write-Host ""
    Write-Host "(.mcp.json was NOT written, to avoid a perpetually-failing MCP server.)"
}
