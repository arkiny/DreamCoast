# Quick launcher: run the sandbox with the Sponza level in the free-fly camera.
#
# Levels start in Fly mode automatically (see apps/sandbox/src/main.rs — non-gallery
# scenes default to CameraMode::Fly), so this just sets LEVEL=sponza and runs. It also
# fetches the Sponza asset on first use (gitignored, proprietary CryEngine license —
# see tools/fetch-sponza.ps1) if it is not present yet.
#
# Controls once running:
#   WASD          move (camera-relative, on the ground plane)
#   Q / E         down / up
#   Right-mouse   hold to look around
#   Shift         sprint (4x)
#   Mouse wheel   adjust base move speed
#   Tab           toggle Fly <-> Orbit
#
# Usage:
#   pwsh tools/run-sponza-fly.ps1                  # default backend (d3d12 on Windows)
#   pwsh tools/run-sponza-fly.ps1 -Backend vulkan  # vulkan | d3d12 | metal
#   pwsh tools/run-sponza-fly.ps1 -Release         # optimized build

param(
    [string]$Backend = '',
    [switch]$Release
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$sponza = Join-Path $root 'assets/Sponza/Sponza.gltf'

# Fetch the asset on first use (it is gitignored and never committed).
if (-not (Test-Path $sponza)) {
    Write-Host 'Sponza asset not found — fetching (local use only)...' -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot 'fetch-sponza.ps1')
}

$env:LEVEL = 'sponza'

$cargoArgs = @('run', '-p', 'sandbox')
if ($Release) { $cargoArgs += '--release' }
$cargoArgs += '--'
if ($Backend) { $cargoArgs += @('--backend', $Backend) }

Write-Host "Launching Sponza (fly camera) — LEVEL=sponza $($cargoArgs -join ' ')" -ForegroundColor Cyan
& cargo @cargoArgs
