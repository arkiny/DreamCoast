# MAC (Metal) VERIFICATION PROMPT — A2/A3 measured-convergence cache fix (commit 0b56606)

> **SUPERSEDED (2026-07-13, commit `2039342`).** The *measured*-convergence trigger
> below (`cache_conv_k` below-eps frames on the GPU `InterlockedAdd` probe) was
> replaced by a **backend-deterministic freeze horizon**: the relight now freezes
> once the lighting epoch has held for `cache_freeze_passes × cache_relight_period`
> frames (`P_CACHE_FREEZE_PASSES`, default 3). Reason: the measured EMA step is a
> stochastic-gather AR(1) signal whose first below-eps frame differs per backend
> (DX froze at 48, VK at 8 ⇒ ~2.7/ch DX≡VK gap). The horizon's terms are all
> backend-independent, so DX/VK/Metal freeze at the **same** frame. The
> `InterlockedAdd` probe is kept as a diagnostic only. **Metal-verified** (`9b0dacd`
> pull, Apple M3): freeze arms at exactly `frame = passes × period`
> (gallery 3, tier-default 192), step bit-identical run-to-run, gallery byte-anchor
> 0.000, Sponza frozen 0.035 avg/ch. The measured-probe verification steps below are
> retained for historical context.

Paste into a fresh Claude Code session on the Mac. Goal: verify commit `0b56606`
("fix(cache): freeze A2/A3 on MEASURED convergence") is correct + lossless on Metal.
It was authored + verified on Windows (DX/VK); Metal needs its own pass because the
fix adds a **GPU atomic (`InterlockedAdd`) into a host-visible buffer + a CPU
read-back** — behaviour that can differ on Metal (Shared storage / write-combine /
coherency) from D3D12/VK.

## What the fix does (so you know what to verify)
The surface-cache dirty-skip (A2) + adaptive reflect-skip (A3) used to freeze the
relight after a fixed frame count (`P_CACHE_SETTLE`). That froze IntelSponza's cache
mid-convergence and changed the image (coloured drapes). The fix measures real
convergence: `sdf_cache_light.slang` `InterlockedAdd`s the **mean** per-channel EMA
step (sum + count fixed-point) into a host-visible probe buffer
(`GdfSystem::cache_conv`, 2-slot ping-pong by `cache_frame % 2`, `conv_buf` = the old
pad0 push slot at byte offset 132). `App::cache_conv_probe` reads it (2-frame
latency, fence-complete) and arms a freeze **latch** (`cache_frozen`) once the mean
step stays below `cache_conv_eps` (0.02) for `cache_conv_k` (6) frames; the latch
holds until the sun/sky/geometry epoch changes. Env overrides: `P_CACHE_CONV_EPS`,
`P_CACHE_CONV_K`, `P_CACHE_DIRTY_SKIP=0`, `P_REFLECT_SKIP=0`.

## Setup
```
git fetch origin && git checkout main && git pull   # HEAD should be 0b56606 or later
cargo build -p sandbox                              # macOS default backend = metal
cargo test                                          # expect 123 passed, 3 ignored
cargo fmt --check && cargo clippy --all-targets -- -D warnings   # CI gate
```

## Verification (Metal). Rebuild a small headless perf harness or reuse tools.
The convergence LATCH needs ~120+ warmup frames to arm in headless, so use
`WARMUP_FRAMES=200`. Metal screenshots via `--screenshot-clean`. Compare with
`python tools/rt-compare.py a.png b.png diff.png` (avg abs diff/channel).

**Config:** `LEVEL=sponza RENDER_QUALITY=med WINDOW_RES=1920x1080 RENDER_SCALE=0.6667 P_TAAU=1`.
For the lossless image checks use a FIXED exposure to isolate from auto-exposure:
`AUTO_EXPOSURE=0 EXPOSURE=8 WARMUP_FRAMES=230`.

1. **The Metal-specific risk — does the atomic+readback work at all?**
   Run `LEVEL=sponza ... PROFILE_GPU=1 WARMUP_FRAMES=200 cargo run -p sandbox --release -- --backend metal --screenshot-clean /tmp/s.png` and check the per-pass `sdf_cache_light` line drops toward ~0 in the converged tail (the freeze latched). If it never drops, the convergence probe isn't reading back on Metal — inspect `create_storage_buffer_host` (Metal = Shared) + `read_into` for the Metal path, and whether the Metal `InterlockedAdd` on a Shared buffer is visible to the CPU read. (Temporary debug: re-add the `DEBUG_CONV` print near `cache_conv_probe` in main.rs to log the mean-delta — Sponza should read ~0.0003-0.003, IntelSponza ~1.4-18.)

2. **Sponza still 60fps + lossless on Metal.** Perf: converged gpu-passes should be well under 16.6ms (Windows: DX 12.8 / VK 13.8). Lossless: render opts-ON (default) vs opts-OFF (`P_CACHE_DIRTY_SKIP=0 P_REFLECT_SKIP=0`), same backend, fixed exposure — diff must be within the Metal stochastic floor (Windows saw 0.055; establish the Metal floor with two OFF-vs-OFF runs).

3. **IntelSponza is lossless (the fix's whole point).** `LEVEL=sponza_intel`, opts ON vs OFF (also set `P_SHADOW_DIRTY_SKIP=0 SCENE_CULL=0` to isolate A2/A3), fixed exposure. **Must be ≈ the noise floor (Windows: 0.008; was 2.278 before the fix).** If it's high on Metal, the freeze latched when it shouldn't — check that IntelSponza's measured mean-delta stays >> `cache_conv_eps` (never converges) on Metal too.

4. **Gallery byte-identical anchor.** `cargo run -p sandbox --release -- --backend metal --screenshot-clean /tmp/gal.png` and compare to the committed Metal golden (`tools/goldens/` — the gallery sha, Metal-authored). Must stay byte-identical: the gallery is `gallery_scene`, which force-disables the dirty-skip (`cache_dirty_skip = !gallery_scene && ...`), so the probe is never armed and the relight runs every frame. Confirm the gallery sha is unchanged. Also run `python tools/golden-image.py --backend metal` (Metal is the golden-authoring box, so exact SHA should PASS).

5. **DX≡VK/Metal parity is NOT the goal here** (you can't run DX/VK on Mac), but note: on Windows the freeze latch widened the Sponza content DX≡VK gap slightly (~0.09→0.114, each backend latches at a marginally different frame). That is expected and within the pre-existing content stochastic gap; the byte-identical GALLERY anchor is unaffected (step 4).

## Report
State plainly: does the atomic+readback convergence probe work on Metal (step 1)?
Is IntelSponza lossless (step 3, the fix)? Is Sponza still 60fps + lossless (step 2)?
Is the gallery golden byte-identical (step 4)? If any Metal-specific issue (atomic
visibility, readback coherency, latch never arming), report the exact symptom +
which file:line (`crates/rhi-metal/src/resources.rs` create_storage_buffer_host /
`buffer read_into`, `apps/sandbox/src/gdf.rs::cache_conv_probe`,
`crates/shader/shaders/sdf_cache_light.slang` InterlockedAdd).
```
```
