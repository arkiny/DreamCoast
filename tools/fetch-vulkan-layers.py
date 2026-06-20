#!/usr/bin/env python3
"""Fetch the Khronos Vulkan validation layer (standalone, no SDK install).

Downloads the prebuilt **MSVC** `VkLayer_khronos_validation` package from
conda-forge (Apache-2.0) and extracts just the layer DLL + its JSON manifest into
`tools/vulkan-layers/`, rewriting the manifest's `library_path` to point at the
co-located DLL. Nothing here is committed (see `.gitignore`); this is purely a
local developer convenience so the engine can enable Vulkan validation.

The engine auto-discovers this folder at runtime (see
`crates/rhi-vulkan/src/instance.rs`): when validation is requested it adds this
directory to `VK_ADD_LAYER_PATH`, so `cargo run -p sandbox -- --backend vulkan`
just works once this script has run.

Source package:
  https://conda.anaconda.org/conda-forge/win-64/  (vulkan-validation-layers)
License: Apache-2.0 (the validation layers). We only *use* the binary locally;
we do not redistribute it.

Usage:  python tools/fetch-vulkan-layers.py [version]
Requires: Python 3 + `zstandard` (`pip install --user zstandard`).
"""

import io
import json
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import zstandard

# Pinned default; override via argv[1]. conda-forge win-64 build string:
DEFAULT_VERSION = "1.4.341.0"
BUILD = "h49e36cd_0"
CHANNEL = "https://conda.anaconda.org/conda-forge/win-64"

LAYER_DLL = "VkLayer_khronos_validation.dll"
LAYER_JSON = "VkLayer_khronos_validation.json"


def main() -> int:
    version = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VERSION
    out_dir = Path(__file__).resolve().parent / "vulkan-layers"
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"{CHANNEL}/vulkan-validation-layers-{version}-{BUILD}.conda"
    print(f"downloading {url}")
    blob = urllib.request.urlopen(url, timeout=120).read()

    # A `.conda` file is a zip wrapping zstd-compressed tarballs. The payload we
    # want lives in `pkg-*.tar.zst`.
    dll_bytes = None
    json_text = None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        pkg_name = next(n for n in zf.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst"))
        tar_bytes = zstandard.ZstdDecompressor().decompress(
            zf.read(pkg_name), max_output_size=512 * 1024 * 1024
        )
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
            for m in tf.getmembers():
                base = Path(m.name).name
                if base == LAYER_DLL:
                    dll_bytes = tf.extractfile(m).read()
                elif base == LAYER_JSON:
                    json_text = tf.extractfile(m).read().decode("utf-8")

    if dll_bytes is None or json_text is None:
        print("error: layer DLL or manifest not found in package", file=sys.stderr)
        return 1

    # Co-locate the manifest with the DLL and point library_path at the sibling.
    manifest = json.loads(json_text)
    manifest["layer"]["library_path"] = f".\\{LAYER_DLL}"

    (out_dir / LAYER_DLL).write_bytes(dll_bytes)
    (out_dir / LAYER_JSON).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"installed validation layer {version} -> {out_dir}")
    print(f"  {LAYER_DLL} ({len(dll_bytes) // 1024} KiB)")
    print(f"  {LAYER_JSON}")
    print("imports:", ", ".join(sorted(_pe_imports(dll_bytes))))
    return 0


def _pe_imports(data: bytes) -> set[str]:
    """List the import-table DLL names of a PE file (best-effort, no deps)."""
    import struct

    try:
        e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
        assert data[e_lfanew : e_lfanew + 4] == b"PE\0\0"
        coff = e_lfanew + 4
        num_sections = struct.unpack_from("<H", data, coff + 2)[0]
        opt_size = struct.unpack_from("<H", data, coff + 16)[0]
        opt = coff + 20
        magic = struct.unpack_from("<H", data, opt)[0]
        # Import directory RVA is at offset 0x78 (PE32) / 0x80 (PE32+).
        imp_rva = struct.unpack_from("<I", data, opt + (0x80 if magic == 0x20B else 0x78))[0]
        sections = opt + opt_size

        def rva_to_off(rva: int) -> int:
            for i in range(num_sections):
                s = sections + i * 40
                va = struct.unpack_from("<I", data, s + 12)[0]
                sz = struct.unpack_from("<I", data, s + 8)[0]
                raw = struct.unpack_from("<I", data, s + 20)[0]
                if va <= rva < va + sz:
                    return raw + (rva - va)
            return -1

        names: set[str] = set()
        off = rva_to_off(imp_rva)
        while off >= 0:
            name_rva = struct.unpack_from("<I", data, off + 12)[0]
            if name_rva == 0:
                break
            no = rva_to_off(name_rva)
            end = data.index(b"\0", no)
            names.add(data[no:end].decode("ascii", "replace"))
            off += 20
        return names
    except Exception:
        return {"<unparsed>"}


if __name__ == "__main__":
    raise SystemExit(main())
