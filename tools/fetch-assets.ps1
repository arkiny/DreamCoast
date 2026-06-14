# Fetches sample glTF models from the Khronos glTF-Sample-Assets repository into
# `assets/` (which is gitignored). Every model selected here is licensed
# **CC0 1.0 Universal** (public domain) — no attribution is legally required;
# authors are listed in assets/CREDITS.md as a courtesy.
#
# Reference: https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/Models.md
#
# Usage:  pwsh tools/fetch-assets.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$assets = Join-Path $root 'assets'
New-Item -ItemType Directory -Force $assets | Out-Null

$base = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models'
$headers = @{ 'User-Agent' = 'engine-fetch-assets' }

# name, author, license (all CC0 1.0 Universal), is-default-model
$models = @(
    @{ Name = 'Avocado'; Author = 'Microsoft'; Default = $true },
    @{ Name = 'BoomBox'; Author = 'Microsoft'; Default = $false },
    @{ Name = 'Lantern'; Author = 'sbtron'; Default = $false }
)

$credits = @(
    '# Asset credits',
    '',
    'These models are from the Khronos glTF Sample Assets and are licensed',
    '**CC0 1.0 Universal (public domain)**. No attribution is legally required;',
    'authors are listed below as a courtesy. They are fetched at build/run time and',
    'are NOT committed to this repository (see .gitignore).',
    '',
    'Source: https://github.com/KhronosGroup/glTF-Sample-Assets',
    'License: https://creativecommons.org/publicdomain/zero/1.0/',
    ''
)

foreach ($m in $models) {
    $url = "$base/$($m.Name)/glTF-Binary/$($m.Name).glb"
    $out = Join-Path $assets "$($m.Name).glb"
    Invoke-WebRequest -Uri $url -OutFile $out -Headers $headers
    "downloaded $($m.Name).glb ($([math]::Round((Get-Item $out).Length / 1KB)) KB)"
    if ($m.Default) {
        Copy-Item $out (Join-Path $assets 'model.glb') -Force
        "  -> set as default assets/model.glb"
    }
    $credits += "- **$($m.Name)** by $($m.Author) - CC0 1.0 Universal - $base/$($m.Name)"
}

($credits -join "`n") | Set-Content -Path (Join-Path $assets 'CREDITS.md') -Encoding utf8
"wrote assets/CREDITS.md"
