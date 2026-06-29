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

# name|author|is-default|kind
#   kind = binary   -> a single self-contained glTF-Binary/<name>.glb
#   kind = separate -> glTF/<name>.gltf + its external buffers/images (into assets/<name>/)
models=(
    "Avocado|Microsoft|true|binary"
    "BoomBox|Microsoft|false|binary"
    "Lantern|sbtron|false|binary"
    # Animation bring-up (Phase 15) — all CC0. Node TRS + interpolation modes,
    # vertex skinning (the minimal rig), and morph targets.
    "InterpolationTest|Khronos|false|binary"
    "AnimatedMorphCube|Microsoft|false|binary"
    "AnimatedCube|UX3D (Norbert Nopper)|false|separate"
    "SimpleSkin|Marco Hutter|false|separate"
)

# Download a glTF-separate model: the .gltf plus every external resource it
# references (buffers + images), resolved relative to the model's glTF/ dir, into
# assets/<name>/. (`gltf::import` in the loader resolves those relative URIs.)
fetch_separate() {
    local name="$1"
    local dir="$assets/$name"
    mkdir -p "$dir"
    curl -fsSL -A 'engine-fetch-assets' -o "$dir/$name.gltf" "$base/$name/glTF/$name.gltf"
    grep -oE '"uri"[[:space:]]*:[[:space:]]*"[^"]+"' "$dir/$name.gltf" \
        | sed -E 's/.*"([^"]+)"$/\1/' \
        | while IFS= read -r uri; do
            [[ -z "$uri" || "$uri" == data:* ]] && continue
            curl -fsSL -A 'engine-fetch-assets' -o "$dir/$uri" "$base/$name/glTF/$uri"
        done
}

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
    IFS='|' read -r name author is_default kind <<< "$entry"
    if [[ "$kind" == "separate" ]]; then
        fetch_separate "$name"
        kb=$(( ($(wc -c < "$assets/$name/$name.gltf") + 512) / 1024 ))
        echo "downloaded $name/ (glTF + buffers; load assets/$name/$name.gltf, ${kb} KB)"
    else
        url="$base/$name/glTF-Binary/$name.glb"
        out="$assets/$name.glb"
        curl -fsSL -A 'engine-fetch-assets' -o "$out" "$url"
        kb=$(( ($(wc -c < "$out") + 512) / 1024 ))
        echo "downloaded $name.glb (${kb} KB)"
        if [[ "$is_default" == "true" ]]; then
            cp -f "$out" "$assets/model.glb"
            echo "  -> set as default assets/model.glb"
        fi
    fi
    echo "- **$name** by $author - CC0 1.0 Universal - $base/$name" >> "$credits"
done

echo "wrote assets/CREDITS.md"
