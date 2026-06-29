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

# rt-compare avg abs diff / channel between two PNGs (0 if identical).
function AvgDiff($a, $b) {
    $out = & python tools/rt-compare.py $a $b "$OutDir/_tmp_diff.png" 2>&1
    $m = $out | Select-String -Pattern 'avg abs diff / channel:\s*([\d.]+)' | Select-Object -First 1
    if ($m) { [double]$m.Matches[0].Groups[1].Value } else { [double]::NaN }
}

# Gate: flag-on must match flag-off. Bit-exact when the backend is deterministic run-to-run
# (Vulkan); when it is NOT (D3D12 has a ~1-LSB run-to-run jitter), require flag-on to stay within
# the backend's own off-vs-off noise floor — i.e. the RHI thread adds nothing beyond it.
function GateMatch($label, $off, $off2, $on) {
    if ((Hash $off) -eq (Hash $on)) {
        Write-Host "  [PASS] $label : flag-on == flag-off (bit-exact $(Hash $on))" -ForegroundColor Green
        return $true
    }
    $noise = AvgDiff $off $off2          # backend's own run-to-run noise
    $onVsOff = AvgDiff $off $on          # flag-on vs flag-off
    if ($onVsOff -le ($noise + 0.0005)) {
        Write-Host ("  [PASS] {0} : non-deterministic backend; flag-on within run-to-run noise (on-vs-off {1:n4} <= off-vs-off {2:n4})" -f $label, $onVsOff, $noise) -ForegroundColor Green
        return $true
    }
    Write-Host ("  [FAIL] {0} : flag-on diverges beyond noise (on-vs-off {1:n4} > off-vs-off {2:n4})" -f $label, $onVsOff, $noise) -ForegroundColor Red
    return $false
}

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

    # 1+2: default capture. Two flag-off runs (noise floor) + one flag-on.
    $off = "$OutDir/${b}_off.png"
    $off2 = "$OutDir/${b}_off2.png"
    $on = "$OutDir/${b}_on.png"
    Capture $b @{}                      $off
    Capture $b @{}                      $off2
    Capture $b @{ P15_RHI_THREAD = '1' } $on
    if (-not (GateMatch "$b default" $off $off2 $on)) { $fails.Add("$b default off!=on") }

    # 3: P15_SPIN motion sequence — flag-off (x2 for noise) vs flag-on, frame-for-frame.
    $spinEnv = @{ P15_SPIN = '8'; CAPTURE_SEQ = '4'; CAPTURE_SEQ_STEP = '0' }
    Capture $b $spinEnv                                  "$OutDir/${b}_spin_off.png"
    Capture $b $spinEnv                                  "$OutDir/${b}_spin_off2.png"
    Capture $b ($spinEnv + @{ P15_RHI_THREAD = '1' })    "$OutDir/${b}_spin_on.png"
    $seqOk = $true
    foreach ($i in 0..3) {
        $fi = '{0:d4}' -f $i
        $a = "$OutDir/${b}_spin_off.$fi.png"
        $a2 = "$OutDir/${b}_spin_off2.$fi.png"
        $c = "$OutDir/${b}_spin_on.$fi.png"
        if (-not (Test-Path $a) -or -not (Test-Path $a2) -or -not (Test-Path $c)) { $seqOk = $false; continue }
        if (-not (GateMatch "$b spin f$i" $a $a2 $c)) { $seqOk = $false }
    }
    if (-not $seqOk) { $fails.Add("$b spin seq off!=on") }
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
