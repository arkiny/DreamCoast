# Builds a Python-3.12-compatible `renderdoc.pyd` (+ matching renderdoc.dll) from
# RenderDoc source, FULLY from the command line — no Visual Studio IDE, no manual
# property-page editing. So the renderdoc-mcp server (Python 3.10+) can import it.
#
# WHY: the official Windows RenderDoc ships only a Python 3.6 module, and a native
# CPython extension imports only into the exact major.minor it was built against.
# So a 3.12 module must be built from source.
#
# HOW (no IDE): RenderDoc's qrenderdoc/Code/pyrenderdoc/python.props selects the
# Python version purely from an override prefix it reads from the env var
# RENDERDOC_PYTHON_PREFIX64 (x64). If that dir contains include\Python.h,
# python312.zip and python312.lib (root or libs\), it builds against 3.12. We
# stage exactly that, set the env var, and drive MSBuild on the `pyrenderdoc_module`
# project (which pulls in the `renderdoc` core via project reference). The VS2015
# (v140) toolset the .sln targets is retargeted to v143 on the command line.
#
# We build ONLY the headless `renderdoc` module — no `qrenderdoc`, so no Qt.
#
# License: RenderDoc is MIT (© Baldur Karlsson); the bundled SWIG is GPL but used
# only as a local build tool (its output is unrestricted). Source, the built .pyd
# and the staged Python prefix all live under gitignored tools/ — nothing here is
# committed or redistributed. See docs/renderdoc-mcp.md "Licenses / attribution".
#
# Requirements (verified present on this machine):
#   - Visual Studio 2022 with "Desktop development with C++" (provides MSBuild + v143)
#   - Python 3.12 with dev libs (python312.lib + Include/Python.h)  [PythonRoot]
#   - git
#
# Usage:
#   pwsh tools/build-renderdoc-py312.ps1            # clone, stage, build, collect
#   pwsh tools/build-renderdoc-py312.ps1 -PrepOnly  # clone + stage + verify override only
#   pwsh tools/build-renderdoc-py312.ps1 -Tag v1.44 -PythonRoot "C:\...\Python312"

[CmdletBinding()]
param(
    [string]$Tag = 'v1.44',
    [string]$PythonRoot,
    [string]$OutDir,
    [string]$Configuration = 'Release',     # Release (fast) or Development (debuggable)
    [string]$Toolset = 'v143',              # VS2022 C++ toolset; retargets the v140 .sln
    [switch]$PrepOnly                        # stage + verify the Python override, skip the heavy compile
)

$ErrorActionPreference = 'Stop'
$srcDir = Join-Path $PSScriptRoot 'renderdoc-src'
$base   = Join-Path $PSScriptRoot 'renderdoc-mcp'
if (-not $OutDir)     { $OutDir = Join-Path $base 'module' }
if (-not $PythonRoot) { $PythonRoot = Split-Path -Parent (& py -3.12 -c "import sys;print(sys.executable)").Trim() }

# --- Resolve toolchain ------------------------------------------------------
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found; is Visual Studio installed?" }
$msbuild = & $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                      -find 'MSBuild\**\Bin\MSBuild.exe' | Select-Object -First 1
if (-not $msbuild) { throw "MSBuild with the C++ toolset not found. Install VS 'Desktop development with C++'." }
Write-Host "==> MSBuild: $msbuild"

# --- Validate Python 3.12 dev layout ---------------------------------------
$pyLib = Join-Path $PythonRoot 'libs\python312.lib'
$pyInc = Join-Path $PythonRoot 'include\Python.h'
$pyDll = Join-Path $PythonRoot 'python312.dll'
if (-not (Test-Path $pyLib)) { throw "python312.lib not found at $pyLib (need the Python 3.12 dev libs)." }
if (-not (Test-Path $pyInc)) { throw "Python.h not found at $pyInc (need the Python 3.12 headers)." }
Write-Host "==> Python 3.12 root: $PythonRoot"

# --- 1. Clone RenderDoc source (shallow; v1.44 vendors all deps in-tree) ----
if (Test-Path (Join-Path $srcDir '.git')) {
    Write-Host "==> RenderDoc source already at $srcDir (leaving as-is)"
} else {
    Write-Host "==> Cloning baldurk/renderdoc @ $Tag -> $srcDir"
    git clone --depth 1 --branch $Tag --no-tags https://github.com/baldurk/renderdoc.git $srcDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed (tag $Tag)." }
}
$sln = Join-Path $srcDir 'renderdoc.sln'
if (-not (Test-Path $sln)) { throw "renderdoc.sln not found in $srcDir." }
# swig.exe is checked in (no submodules needed for v1.44) — sanity-check it.
$swig = Join-Path $srcDir 'qrenderdoc\3rdparty\swig\swig.exe'
if (-not (Test-Path $swig)) { throw "swig.exe missing at $swig — clone may be incomplete." }

# --- 2. Stage the Python 3.12 override prefix python.props expects ----------
# Needs: include\Python.h, python312.zip (root), python312.lib (root or libs\).
$prefix = Join-Path $srcDir 'python312-prefix'
New-Item -ItemType Directory -Force (Join-Path $prefix 'libs') | Out-Null
Copy-Item $pyLib (Join-Path $prefix 'libs') -Force
if (-not (Test-Path (Join-Path $prefix 'include\Python.h'))) {
    Copy-Item (Join-Path $PythonRoot 'include') $prefix -Recurse -Force
}
if (Test-Path $pyDll) { Copy-Item $pyDll $prefix -Force }   # not consumed by the module build, staged for completeness
$stdlibZip = Join-Path $prefix 'python312.zip'
if (-not (Test-Path $stdlibZip)) {
    # python.props uses the .zip's EXISTENCE as the key that selects this version,
    # so it must be present or the build silently falls back to bundled 3.6.
    Write-Host "==> Building python312.zip stdlib snapshot (selection key)"
    Compress-Archive -Path (Join-Path $PythonRoot 'Lib\*') -DestinationPath $stdlibZip -Force
}
Write-Host "==> Python 3.12 override prefix staged: $prefix"
$env:RENDERDOC_PYTHON_PREFIX64 = $prefix

# --- 3. Verify python.props actually selects 3.12 (fast, no compile) --------
# Evaluate the property sheet in isolation and confirm CustomPythonUsed == 312.
$check = Join-Path $srcDir '_py312_check.proj'
@"
<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <Import Project="`$(SolutionDir)qrenderdoc\Code\pyrenderdoc\python.props" />
  <Target Name="Check">
    <Message Importance="high" Text="CHECK CustomPythonUsed=`$(CustomPythonUsed)" />
    <Message Importance="high" Text="CHECK PythonImportLib=`$(PythonImportLib)" />
    <Message Importance="high" Text="CHECK PythonIncludeDir=`$(PythonIncludeDir)" />
  </Target>
</Project>
"@ | Set-Content -Encoding UTF8 $check
$slnDir = (Resolve-Path $srcDir).Path
if (-not $slnDir.EndsWith('\')) { $slnDir += '\' }
$checkOut = & $msbuild $check /t:Check /nologo /v:m /p:Platform=x64 "/p:SolutionDir=$slnDir" 2>&1
$checkOut | ForEach-Object { Write-Host "    $_" }
if (-not ($checkOut -match 'CHECK CustomPythonUsed=312')) {
    Remove-Item $check -Force -ErrorAction SilentlyContinue
    throw "python.props did NOT select 3.12 (would fall back to bundled 3.6). Check the staged prefix: $prefix"
}
Remove-Item $check -Force -ErrorAction SilentlyContinue
Write-Host "==> Verified: build will target Python 3.12"

if ($PrepOnly) {
    Write-Host "==> -PrepOnly set; staged + verified, skipping the compile."
    return
}

# --- 4. Build the headless module from the command line (no IDE) ------------
# Build the project file directly (NOT `sln /t:pyrenderdoc_module` — that MSB4057s
# because the solution-target name isn't honored here). Its ProjectReference pulls
# in the `renderdoc` core, which in turn references every driver, so the whole
# dependency chain builds. $(SolutionDir) is used throughout the projects (swig
# path, include dirs, OutDir) so it MUST be passed explicitly for a direct build.
# Retarget the v140 projects to v143 and let MSBuild pick the latest Windows SDK.
# TreatWarningAsError is forced off: the source was written for v140 and a newer
# compiler may emit new warnings that would otherwise fail the build.
# Several projects HARDCODE <TreatWarningAsError>true</> (so /p:TreatWarningAsError
# can't override them), and on a non-UTF8 system locale (e.g. Korean CP949) glslang
# sources trip C4819 ("character not representable in code page"), failing as C2220.
# The MSVC `_CL_` env var is APPENDED after each project's own cl flags, so `/WX-`
# here overrides an earlier hardcoded `/WX`; `/wd4819` silences the locale warning.
$env:_CL_ = '/WX- /wd4819'

$msbFlags = @(
    '/m', '/nologo', '/v:m',
    "/p:Configuration=$Configuration", '/p:Platform=x64',
    "/p:PlatformToolset=$Toolset",
    '/p:WindowsTargetPlatformVersion=10.0',
    '/p:TreatWarningAsError=false',
    "/p:SolutionDir=$slnDir"
)

# renderdoc.dll links these breakpad static libs via AdditionalDependencies, but
# they are NOT project references, so a direct build never builds them. Build them
# first; each outputs <name>.lib into x64\$Configuration\ as the linker expects.
$bpRoot = Join-Path $srcDir 'renderdoc\3rdparty\breakpad\client\windows'
$breakpad = @(
    'common.vcxproj',
    'crash_generation\crash_generation_client.vcxproj',
    'handler\exception_handler.vcxproj'
)
foreach ($bp in $breakpad) {
    Write-Host "==> Building breakpad dependency: $bp"
    & $msbuild (Join-Path $bpRoot $bp) @msbFlags 2>&1 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw "breakpad build failed: $bp" }
}

$proj = Join-Path $srcDir 'qrenderdoc\Code\pyrenderdoc\pyrenderdoc_module.vcxproj'
Write-Host "==> Building pyrenderdoc_module ($Configuration|x64, toolset $Toolset) — this takes a while"
$buildOut = & $msbuild $proj @msbFlags 2>&1
$buildOut | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) { throw "MSBuild failed (exit $LASTEXITCODE). See output above." }
if ($buildOut -match 'Built against python from') {
    Write-Host "==> Confirmed: linked against the 3.12 override prefix"
} else {
    Write-Warning "Did not see the 'Built against python from' marker — verify the module is 3.12 before trusting it."
}

# --- 5. Collect outputs into the module dir --------------------------------
$pyd = Join-Path $srcDir "x64\$Configuration\pymodules\renderdoc.pyd"
$dll = Join-Path $srcDir "x64\$Configuration\renderdoc.dll"
if (-not (Test-Path $pyd)) { throw "Built renderdoc.pyd not found at $pyd." }
if (-not (Test-Path $dll)) { throw "Built renderdoc.dll not found at $dll." }
New-Item -ItemType Directory -Force $OutDir | Out-Null
Copy-Item $pyd $OutDir -Force; Write-Host "    copied renderdoc.pyd"
Copy-Item $dll $OutDir -Force; Write-Host "    copied renderdoc.dll"

Write-Host ""
Write-Host "==> Module ready in: $OutDir"
Write-Host "Now register the MCP server:"
Write-Host "    pwsh tools/setup-renderdoc-mcp.ps1 -ModuleDir `"$OutDir`""
