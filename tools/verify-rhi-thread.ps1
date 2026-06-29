<#
.SYNOPSIS
    Phase 15 M4 B3 — verify the RHI submit thread (P15_RHI_THREAD) on Windows.

.DESCRIPTION
    The RHI thread split was runtime-verified only on the macOS/Metal box; this
    script runs the cross-backend (Vulkan + D3D12) parity gates on the Windows box.

    Per backend it checks the BIT-EXACT invariants (the real correctness gates):
      1. flag-off capture  == prior inline behaviour (sanity; deterministic per GPU)
      2. flag-on (P15_RHI_THREAD=1) capture  ==  flag-off capture
         -> proves the worker's acquire/translate/submit/present + the `unsafe impl
            Send` on the boundary types are correct on this backend.
      3. P15_SPIN motion sequence (4 frames): flag-on == flag-off, frame-for-frame
         -> proves the 1-frame overlap stays deterministic with moving objects.

    It also runs the existing DX==VK parity check (rt-compare) on the flag-on
    captures (informational; the hard rule is <= 0.001 avg/channel).

    Watch the cargo output for D3D12 debug-layer / Vulkan validation errors
    (VUID..., "D3D12 ERROR", live-object leaks) while the flag is on — a threading
    violation surfaces there even when pixels happen to match.

.EXAMPLE
    pwsh tools/verify-rhi-thread.ps1
    pwsh tools/verify-rhi-thread.ps1 -Backends d3d12
#>
[CmdletBinding()]
param(
    [ValidateSet('vulkan', 'd3d12')]
    [string[]]$Backends = @('vulkan', 'd3d12'),
    [string]$OutDir = 'target/rhi-verify'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$fails = [System.Collections.Generic.List[string]]::new()

function Hash($p) { (Get-FileHash -Algorithm SHA256 $p).Hash }

# Run one headless capture with the given env vars set just for this process call.
function Capture {
    param([string]$Backend, [hashtable]$EnvVars, [string]$OutPng)
    $saved = @{}
    foreach ($k in $EnvVars.Keys) {
        $saved[$k] = [Environment]::GetEnvironmentVariable($k)
        [Environment]::SetEnvironmentVariable($k, $EnvVars[$k])
    }
    try {
        & cargo run -q -p sandbox -- --backend $Backend --screenshot-clean $OutPng
        if ($LASTEXITCODE -ne 0) { throw "sandbox exited $LASTEXITCODE ($Backend)" }
    }
    finally {
        foreach ($k in $EnvVars.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
}

Write-Host "== building sandbox ==" -ForegroundColor Cyan
& cargo build -p sandbox
if ($LASTEXITCODE -ne 0) { throw "build failed" }

foreach ($b in $Backends) {
    Write-Host "`n== backend: $b ==" -ForegroundColor Cyan

    # 1+2: default capture, flag-off vs flag-on (bit-exact).
    $off = "$OutDir/${b}_off.png"
    $on = "$OutDir/${b}_on.png"
    Capture $b @{}                      $off
    Capture $b @{ P15_RHI_THREAD = '1' } $on
    if ((Hash $off) -eq (Hash $on)) {
        Write-Host "  [PASS] $b default: flag-on == flag-off  ($(Hash $off))" -ForegroundColor Green
    }
    else {
        Write-Host "  [FAIL] $b default: flag-on != flag-off" -ForegroundColor Red
        Write-Host "         off=$(Hash $off)  on=$(Hash $on)"
        $fails.Add("$b default off!=on")
    }

    # 3: P15_SPIN motion sequence, flag-off vs flag-on, frame-for-frame.
    $spinEnv = @{ P15_SPIN = '8'; CAPTURE_SEQ = '4'; CAPTURE_SEQ_STEP = '0' }
    $soff = "$OutDir/${b}_spin_off.png"
    $son = "$OutDir/${b}_spin_on.png"
    Capture $b $spinEnv                                    $soff
    Capture $b ($spinEnv + @{ P15_RHI_THREAD = '1' })      $son
    $seqOk = $true
    foreach ($i in 0..3) {
        $fi = '{0:d4}' -f $i
        $a = "$OutDir/${b}_spin_off.$fi.png"
        $c = "$OutDir/${b}_spin_on.$fi.png"
        if (-not (Test-Path $a) -or -not (Test-Path $c)) { $seqOk = $false; continue }
        if ((Hash $a) -ne (Hash $c)) { $seqOk = $false }
    }
    if ($seqOk) {
        Write-Host "  [PASS] $b P15_SPIN seq: flag-on == flag-off (4 frames)" -ForegroundColor Green
    }
    else {
        Write-Host "  [FAIL] $b P15_SPIN seq: frame mismatch off vs on" -ForegroundColor Red
        $fails.Add("$b spin seq off!=on")
    }
}

# Cross-backend DX==VK parity on the flag-on captures (informational).
$vkOn = "$OutDir/vulkan_on.png"
$dxOn = "$OutDir/d3d12_on.png"
if ((Test-Path $vkOn) -and (Test-Path $dxOn)) {
    Write-Host "`n== DX==VK parity (flag-on) — hard rule <= 0.001 avg/channel ==" -ForegroundColor Cyan
    & python tools/rt-compare.py $vkOn $dxOn "$OutDir/parity_on.png"
}

Write-Host ""
if ($fails.Count -eq 0) {
    Write-Host "ALL BIT-EXACT GATES PASSED. Confirm no VUID/D3D12-ERROR in the logs above." -ForegroundColor Green
    exit 0
}
else {
    Write-Host "FAILURES:" -ForegroundColor Red
    $fails | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
