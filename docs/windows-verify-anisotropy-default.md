# WINDOWS (D3D12/Vulkan) VERIFICATION PROMPT — default anisotropy 16 (branch `fix/default-anisotropy-16`)

Paste into a fresh Claude Code session on the Windows RTX 2070 SUPER box. Goal:
confirm the wrap-sampler anisotropy default change (`P_ANISO` 1 → **16** on all
three backends) is correct on D3D12 + Vulkan, and **quantify the DX≡VK divergence**
so the CLAUDE.md hard-rule exception can be pinned to a measured number.

Authored + verified on Metal (Apple M3): grazing floor tiles that collapsed to
stripes under the isotropic default now resolve; the gallery anchor stays
run-to-run byte-identical and `P_ANISO=1` restores the pre-change anchor
byte-for-byte. Metal cannot exercise D3D12/Vulkan, so the cross-backend axis needs
this Windows pass.

## What changed (so you know what to verify)

Default `P_ANISO` went from `1` (isotropic, trilinear-only) to `16` (anisotropic)
on the **wrap** sampler only; the clamp sampler (cubes / volumes / G-buffer) stays
isotropic. Sites:
- `crates/rhi-d3d12/src/pipeline.rs` — `unwrap_or(16.0)`, `> 1.0` ⇒ `D3D12_FILTER_ANISOTROPIC`, `MaxAnisotropy` clamp `[1,16]`.
- `crates/rhi-vulkan/src/device.rs` — `unwrap_or(16.0)`, clamped to `maxSamplerAnisotropy`, needs the `samplerAnisotropy` device feature (already requested when `> 1.0`).
- `crates/rhi-metal/src/device.rs` — `unwrap_or(16)` (verified side).

Rationale + the parity trade-off: `docs/qhd-perf.md` Stage 9. `P_ANISO=1` is the
seam that restores the exact pre-change (isotropic) sampler on every backend.

## Why this needs Windows

Anisotropic filtering is **driver-dependent**: D3D12 and Vulkan on the *same*
RTX 2070 SUPER select LOD / sample counts differently, so they no longer produce
byte-identical output on grazing textured surfaces. The Stage 9 doc's prior
measurement was **DX≡VK 0.427 avg/ch (0.46 % of channels > 8)** with anisotropy on.
That measurement predates making it the default — reconfirm it and record the
current number as the pinned hard-rule exception.

## Setup

```
git fetch origin && git checkout fix/default-anisotropy-16
cargo build -p sandbox --release          # Windows default backend picks d3d12; also test vulkan
cargo fmt --check                          # NOTE: pre-existing violations in main.rs (freeze_horizon)
                                           #  + shader/build.rs (perprim) are from the earlier pull,
                                           #  NOT this branch — this branch's 6 files are fmt-clean.
cargo clippy --all-targets -- -D warnings
```

## Checks

1. **Visual — grazing floor.** Capture the same scene the Metal pass used and
   confirm the floor reads as tiles, not stripes, on **both** backends by default
   (no `P_ANISO` env):
   ```
   EV100=11 LEVEL=sponza_intel_chromeball cargo run -p sandbox --release -- --backend d3d12 --screenshot-clean dx_aniso.png
   EV100=11 LEVEL=sponza_intel_chromeball cargo run -p sandbox --release -- --backend vulkan --screenshot-clean vk_aniso.png
   ```

2. **DX≡VK divergence (the number to pin).** Diff the two default captures:
   ```
   python tools/rt-compare.py dx_aniso.png vk_aniso.png dxvk_aniso.png
   ```
   Expect a **non-zero** avg/ch on textured grazing surfaces (prior: ~0.43). Record
   avg / >8 / >32. This is the value the CLAUDE.md exception should cite.

3. **Isotropic seam restores byte-identity.** With `P_ANISO=1`, DX≡VK must return
   to the normal ≤ 0.001 bar (proves the divergence is *only* the anisotropic path,
   not a new cross-backend layout bug):
   ```
   P_ANISO=1 ...--backend d3d12 --screenshot-clean dx_iso.png
   P_ANISO=1 ...--backend vulkan --screenshot-clean vk_iso.png
   python tools/rt-compare.py dx_iso.png vk_iso.png    # expect ≤ 0.001
   ```

4. **Gallery anchor.** The gallery golden was rebased to the anisotropic baseline on
   Metal (`tools/goldens/manifest.json`, SHA `65d04ceca2c4…`). On Windows the SHA
   will differ (cross-box/driver), so use the tolerant path or just confirm DX≡VK on
   the gallery matches the same anisotropic divergence profile — not a regression.

## Pass criteria

- Floor renders as tiles (not stripes) by default on D3D12 **and** Vulkan.
- `P_ANISO=1` returns DX≡VK to ≤ 0.001 (isotropic seam is clean).
- The default-on DX≡VK divergence is bounded and localized to grazing textured
  surfaces (~0.4/ch, matching the prior 0.427). Update the CLAUDE.md exception and
  `docs/qhd-perf.md` Stage 9 with the reconfirmed number.

## Out of scope

- Content goldens `sponza_sc_viz` / `sponza_gdf_ao` were run-to-run non-deterministic
  under the strict-SHA gate when this plan was written (root-caused 2026-07-14 to the
  wall-clock-dt auto-exposure EMA, independent of anisotropy, and fixed — the recipes
  now pin `AUTO_EXPOSURE=0` and screenshot mode adapts on `FIXED_DT`). Their SHAs
  were reseeded with the fix; a mismatch on the Windows box is expected cross-box
  variance, not an aniso regression. See `docs/golden-image-regression.md`.
