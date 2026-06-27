#!/usr/bin/env bash
# Fetches the Crytek **Sponza** test scene from the Khronos glTF-Sample-Assets
# repository into `assets/Sponza/` (which is gitignored) for LOCAL USE ONLY.
#
# ============================ LICENSE WARNING ===============================
# Unlike the models fetched by fetch-assets.sh (all CC0 public domain), the
# Sponza **model geometry/textures** are licensed under the proprietary
#   CryEngine Limited License Agreement  (SPDX: LicenseRef-CRYENGINE-Agreement)
#   (c) 2016 Crytek -- terms: https://www.cryengine.com/ce-terms
# Only the surrounding metadata/docs are CC-BY-4.0. This is NOT a redistributable
# open license, so Sponza is:
#   * NEVER committed to this repository (assets/ is gitignored), and
#   * fetched by THIS separate opt-in script -- kept out of the CC0-only
#     fetch-assets.sh so that script's public-domain guarantee stays intact.
# Use it as a local rendering test asset only. Do not redistribute the files.
# ===========================================================================
#
# Why Sponza: a large, deeply-nested multi-material glTF -- the canonical stress
# test for the Phase 12 scene-graph hierarchy import (Stage B) and level
# streaming (Stage D). Avocado/BoomBox/Lantern are too small to exercise those.
#
# Usage:  tools/fetch-sponza.sh   (macOS/Linux counterpart to fetch-sponza.ps1)
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$script_dir")"
dest="$root/assets/Sponza"
mkdir -p "$dest"

api='https://api.github.com/repos/KhronosGroup/glTF-Sample-Assets/contents/Models/Sponza/glTF?ref=main'

echo "Sponza is under the CryEngine Limited License Agreement (proprietary)." >&2
echo "Fetching for LOCAL USE ONLY into assets/Sponza/ (gitignored)." >&2

# The glTF/ directory holds Sponza.gltf + Sponza.bin + ~70 textures. List it via
# the GitHub contents API and pull each file's download_url (no hashes hardcoded).
urls="$(curl -fsSL -A 'engine-fetch-sponza' "$api" \
    | grep -o '"download_url"[[:space:]]*:[[:space:]]*"[^"]*"' \
    | sed 's/.*"\(https[^"]*\)"/\1/')"

count=0
for url in $urls; do
    [ -n "$url" ] || continue
    curl -fsSL -A 'engine-fetch-sponza' -o "$dest/$(basename "$url")" "$url"
    count=$((count + 1))
done
mb=$(( ($(du -sk "$dest" | cut -f1) + 512) / 1024 ))
echo "downloaded $count files (~${mb} MB) -> assets/Sponza/"

# Record the license alongside the files so its provenance is never ambiguous.
cat > "$dest/LICENSE-NOTICE.md" <<'EOF'
# Sponza — LICENSE NOTICE (local use only)

Source: https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Sponza

**Model geometry & textures:** CryEngine Limited License Agreement
(SPDX: LicenseRef-CRYENGINE-Agreement), (c) 2016 Crytek.
Terms: https://www.cryengine.com/ce-terms

**Metadata / docs only:** CC-BY-4.0.

This is a PROPRIETARY license, not an open/redistributable one. These files
are fetched by tools/fetch-sponza.sh for LOCAL rendering tests only, are NOT
committed to this repository, and must NOT be redistributed.
EOF
echo "wrote assets/Sponza/LICENSE-NOTICE.md"
echo "Sponza scene: assets/Sponza/Sponza.gltf"
