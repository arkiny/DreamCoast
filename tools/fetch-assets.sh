#!/usr/bin/env bash
# Fetches sample glTF models from the Khronos glTF-Sample-Assets repository into
# `assets/` (which is gitignored). Every model selected here is licensed
# **CC0 1.0 Universal** (public domain) — no attribution is legally required;
# authors are listed in assets/CREDITS.md as a courtesy.
#
# Reference: https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/Models.md
#
# Usage:  tools/fetch-assets.sh
#
# This is the macOS/Linux counterpart to fetch-assets.ps1.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$script_dir")"
assets="$root/assets"
mkdir -p "$assets"

base='https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models'

# name|author|is-default
models=(
    "Avocado|Microsoft|true"
    "BoomBox|Microsoft|false"
    "Lantern|sbtron|false"
)

credits="$assets/CREDITS.md"
{
    echo '# Asset credits'
    echo
    echo 'These models are from the Khronos glTF Sample Assets and are licensed'
    echo '**CC0 1.0 Universal (public domain)**. No attribution is legally required;'
    echo 'authors are listed below as a courtesy. They are fetched at build/run time and'
    echo 'are NOT committed to this repository (see .gitignore).'
    echo
    echo 'Source: https://github.com/KhronosGroup/glTF-Sample-Assets'
    echo 'License: https://creativecommons.org/publicdomain/zero/1.0/'
    echo
} > "$credits"

for entry in "${models[@]}"; do
    IFS='|' read -r name author is_default <<< "$entry"
    url="$base/$name/glTF-Binary/$name.glb"
    out="$assets/$name.glb"
    curl -fsSL -A 'engine-fetch-assets' -o "$out" "$url"
    kb=$(( ($(wc -c < "$out") + 512) / 1024 ))
    echo "downloaded $name.glb (${kb} KB)"
    if [[ "$is_default" == "true" ]]; then
        cp -f "$out" "$assets/model.glb"
        echo "  -> set as default assets/model.glb"
    fi
    echo "- **$name** by $author - CC0 1.0 Universal - $base/$name" >> "$credits"
done

echo "wrote assets/CREDITS.md"
