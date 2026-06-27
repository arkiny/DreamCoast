# Fetches the Crytek **Sponza** test scene from the Khronos glTF-Sample-Assets
# repository into `assets/Sponza/` (which is gitignored) for LOCAL USE ONLY.
#
# ============================ LICENSE WARNING ===============================
# Unlike the models fetched by fetch-assets.ps1 (all CC0 public domain), the
# Sponza **model geometry/textures** are licensed under the proprietary
#   CryEngine Limited License Agreement  (SPDX: LicenseRef-CRYENGINE-Agreement)
#   © 2016 Crytek — terms: https://www.cryengine.com/ce-terms
# Only the surrounding metadata/docs are CC-BY-4.0. This is NOT a redistributable
# open license, so Sponza is:
#   * NEVER committed to this repository (assets/ is gitignored), and
#   * fetched by THIS separate opt-in script — kept out of the CC0-only
#     fetch-assets.ps1 so that script's public-domain guarantee stays intact.
# Use it as a local rendering test asset only. Do not redistribute the files.
# ===========================================================================
#
# Why Sponza: a large, deeply-nested multi-material glTF — the canonical stress
# test for the Phase 12 scene-graph hierarchy import (Stage B) and level
# streaming (Stage D). Avocado/BoomBox/Lantern are too small to exercise those.
#
# Usage:  pwsh tools/fetch-sponza.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root 'assets/Sponza'
New-Item -ItemType Directory -Force $dest | Out-Null

$headers = @{ 'User-Agent' = 'engine-fetch-sponza' }
$api = 'https://api.github.com/repos/KhronosGroup/glTF-Sample-Assets/contents/Models/Sponza/glTF?ref=main'

Write-Host 'Sponza is under the CryEngine Limited License Agreement (proprietary).' -ForegroundColor Yellow
Write-Host 'Fetching for LOCAL USE ONLY into assets/Sponza/ (gitignored).' -ForegroundColor Yellow

# The glTF/ directory holds Sponza.gltf + Sponza.bin + ~70 textures. List it via
# the GitHub contents API and pull each file's download_url (no hashes hardcoded).
$entries = Invoke-RestMethod -Uri $api -Headers $headers
$total = 0
foreach ($e in $entries) {
    if ($e.type -ne 'file') { continue }
    $out = Join-Path $dest $e.name
    Invoke-WebRequest -Uri $e.download_url -OutFile $out -Headers $headers
    $total += (Get-Item $out).Length
}
"downloaded $($entries.Count) files ($([math]::Round($total / 1MB, 1)) MB) -> assets/Sponza/"

# Record the license alongside the files so its provenance is never ambiguous.
$notice = @(
    '# Sponza — LICENSE NOTICE (local use only)',
    '',
    'Source: https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Sponza',
    '',
    '**Model geometry & textures:** CryEngine Limited License Agreement',
    '(SPDX: LicenseRef-CRYENGINE-Agreement), (c) 2016 Crytek.',
    'Terms: https://www.cryengine.com/ce-terms',
    '',
    '**Metadata / docs only:** CC-BY-4.0.',
    '',
    'This is a PROPRIETARY license, not an open/redistributable one. These files',
    'are fetched by tools/fetch-sponza.ps1 for LOCAL rendering tests only, are NOT',
    'committed to this repository, and must NOT be redistributed.',
    ''
)
($notice -join "`n") | Set-Content -Path (Join-Path $dest 'LICENSE-NOTICE.md') -Encoding utf8
"wrote assets/Sponza/LICENSE-NOTICE.md"
"Sponza scene: assets/Sponza/Sponza.gltf"
