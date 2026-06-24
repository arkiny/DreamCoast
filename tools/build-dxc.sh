#!/usr/bin/env bash
# Build the DirectX Shader Compiler (DXC) from source with Apple Metal Shader
# Converter support, for the Phase 8 / M7 RT-pipeline shader build on macOS
# (`rt_pipeline.slang` -> DXIL -> metallib via the converter). See
# `docs/metal-backend.md` ("M7 ... RT pipeline via Metal Shader Converter").
#
# Why build from source: there is no official macOS DXC binary, and DXC only gains
# the `-metal` codegen path when it is configured against the Metal Shader Converter
# headers/library at build time (CMake `find_package(MetalIRConverter)`). DXC is
# permissively licensed (LLVM/NCSA + MIT), so building/redistributing our own build
# is fine. The build tree (`tools/dxc-src/`) is gitignored; this script reproduces it.
#
# Prerequisite (install once, NOT redistributed by this repo — build-time only):
#   Apple **Metal Shader Converter** — provides
#   `/usr/local/include/metal_irconverter/metal_irconverter.h` +
#   `/usr/local/lib/libmetalirconverter.dylib` + the `metal-shaderconverter` CLI.
#   Download: https://developer.apple.com/metal/shader-converter/  (or it may already
#   be present as a Metal toolchain component). Verify: `metal-shaderconverter --version`.
#
# This script needs only `git` + Python 3 (cmake/ninja are pip-installed locally if
# absent — no Homebrew/sudo). Output: `tools/dxc-src/build/bin/dxc`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/tools/dxc-src"
DXC_TAG="${DXC_TAG:-main}"

# --- Prerequisite: Metal Shader Converter ----------------------------------------
if [ ! -f /usr/local/include/metal_irconverter/metal_irconverter.h ] ||
   [ ! -f /usr/local/lib/libmetalirconverter.dylib ]; then
  echo "error: Metal Shader Converter not found under /usr/local." >&2
  echo "       Install it from https://developer.apple.com/metal/shader-converter/" >&2
  echo "       (provides metal_irconverter.h + libmetalirconverter.dylib)." >&2
  exit 1
fi

# --- cmake + ninja (pip-local if missing; no brew/sudo) --------------------------
PYVER="$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYBIN="$HOME/Library/Python/$PYVER/bin"
export PATH="$PYBIN:$PATH"
if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null; then
  echo "==> installing cmake + ninja (pip --user)"
  python3 -m pip install --user --quiet cmake ninja
fi

# --- Source + the submodules the build needs -------------------------------------
if [ ! -d "$SRC/.git" ]; then
  echo "==> cloning DirectXShaderCompiler ($DXC_TAG)"
  git clone --depth 1 --branch "$DXC_TAG" --no-recurse-submodules \
    https://github.com/microsoft/DirectXShaderCompiler.git "$SRC"
fi
cd "$SRC"
echo "==> initializing submodules (DirectX-Headers + SPIR-V)"
git submodule update --init --depth 1 \
  external/DirectX-Headers external/SPIRV-Headers external/SPIRV-Tools

# --- Configure (Metal converter auto-detected) + build ---------------------------
echo "==> configuring (Release, -metal auto-enabled via find_package(MetalIRConverter))"
rm -rf build && mkdir build && cd build
cmake -G Ninja \
  -C ../cmake/caches/PredefinedParams.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DHLSL_INCLUDE_TESTS=OFF \
  -DSPIRV_BUILD_TESTS=OFF \
  ..
# Sanity: confirm the Metal codegen path was enabled.
if ! grep -q 'Found MetalIRConverter' CMakeCache.txt 2>/dev/null &&
   ! cmake -LA -N . 2>/dev/null | grep -qi 'MetalIRConverter_LIB'; then
  echo "warning: MetalIRConverter was not detected by CMake — '-metal' may be absent." >&2
fi
echo "==> building dxc (this takes a while)"
ninja dxc

DXC="$SRC/build/bin/dxc"
echo "==> done: $DXC"
"$DXC" --version 2>&1 | head -2 || true
echo "    verify -metal: $("$DXC" --help 2>&1 | grep -m1 -- '-metal' || echo 'NOT FOUND')"
