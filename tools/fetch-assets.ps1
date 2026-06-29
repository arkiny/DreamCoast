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

# name, author (all CC0 1.0 Universal), is-default-model, kind:
#   binary   -> a single self-contained glTF-Binary/<name>.glb
#   separate -> glTF/<name>.gltf + its external buffers/images (into assets/<name>/)
$models = @(
    @{ Name = 'Avocado'; Author = 'Microsoft'; Default = $true; Kind = 'binary' },
    @{ Name = 'BoomBox'; Author = 'Microsoft'; Default = $false; Kind = 'binary' },
    @{ Name = 'Lantern'; Author = 'sbtron'; Default = $false; Kind = 'binary' },
    # Animation bring-up (Phase 15) — all CC0. Node TRS + interpolation modes,
    # vertex skinning (the minimal rig), and morph targets.
    @{ Name = 'InterpolationTest'; Author = 'Khronos'; Default = $false; Kind = 'binary' },
    @{ Name = 'AnimatedMorphCube'; Author = 'Microsoft'; Default = $false; Kind = 'binary' },
    @{ Name = 'AnimatedCube'; Author = 'UX3D (Norbert Nopper)'; Default = $false; Kind = 'separate' },
    @{ Name = 'SimpleSkin'; Author = 'Marco Hutter'; Default = $false; Kind = 'separate' }
)

# Download a glTF-separate model: the .gltf plus every external resource it
# references (buffers + images), resolved relative to the model's glTF/ dir, into
# assets/<name>/. (`gltf::import` in the loader resolves those relative URIs.)
function Get-SeparateModel($name) {
    $dir = Join-Path $assets $name
    New-Item -ItemType Directory -Force $dir | Out-Null
    $gltf = Join-Path $dir "$name.gltf"
    Invoke-WebRequest -Uri "$base/$name/glTF/$name.gltf" -OutFile $gltf -Headers $headers
    foreach ($u in [regex]::Matches((Get-Content -Raw $gltf), '"uri"\s*:\s*"([^"]+)"')) {
        $uri = $u.Groups[1].Value
        if ($uri -and -not $uri.StartsWith('data:')) {
            Invoke-WebRequest -Uri "$base/$name/glTF/$uri" -OutFile (Join-Path $dir $uri) -Headers $headers
        }
    }
}

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
    if ($m.Kind -eq 'separate') {
        Get-SeparateModel $m.Name
        "downloaded $($m.Name)/ (glTF + buffers; load assets/$($m.Name)/$($m.Name).gltf)"
    }
    else {
        $url = "$base/$($m.Name)/glTF-Binary/$($m.Name).glb"
        $out = Join-Path $assets "$($m.Name).glb"
        Invoke-WebRequest -Uri $url -OutFile $out -Headers $headers
        "downloaded $($m.Name).glb ($([math]::Round((Get-Item $out).Length / 1KB)) KB)"
        if ($m.Default) {
            Copy-Item $out (Join-Path $assets 'model.glb') -Force
            "  -> set as default assets/model.glb"
        }
    }
    $credits += "- **$($m.Name)** by $($m.Author) - CC0 1.0 Universal - $base/$($m.Name)"
}

($credits -join "`n") | Set-Content -Path (Join-Path $assets 'CREDITS.md') -Encoding utf8
"wrote assets/CREDITS.md"
