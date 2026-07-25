# Phase — High tier at 60 fps, 1080p (`sponza_intel`, RTX 2070 SUPER)

Status: **LANDED and verified.** This document is the record of what was measured, what shipped, and
what is explicitly still open. Every number below is from `tools/perf-profile.py` on a release build
(62-frame steady state after a 45-frame settle, 1920×1080) or from a captured PNG — nothing is
projected. Design reference: **UE5 Lumen + the UE Scalability system**, read from the local checkout
`D:/Repositories/UnrealEngine-1`.

Target: `RENDER_QUALITY=high`, `LEVEL=sponza_intel`, 1920×1080, `gpu_total ≤ 16.67 ms` on **both**
backends at **both** cameras, conceding as little image quality as the arithmetic allows.

**Result: met, and image fidelity improved.** Worst case 15.94 ms (VK, vault) = 62.7 fps; the
low-frequency lighting residual against the path tracer *fell* 12.4–13.2 % at the vault and 2.5–3.2 %
at the door, and DX≡VK content divergence *fell* 27 %. The one thing this wave does **not** fix is
the transient after a lighting change (§6) — that is stated and measured rather than hidden.

Cameras (both must pass; the vault is the binding case):
- **door** `--cam-eye 7,2.2,0 --cam-target 20,2.2,0`
- **vault** `--cam-eye=-6.31,3.11,1.17 --cam-target=-7.22,3.9,1.17`

---

## 1. Baseline — measured

```
High, door, D3D12    gpu_total 153.762 ms = 6.5 fps    fence-wait 153.587   cpu-record 10.006
  gi_volume  64.845   gdf_reflect  60.760   gdf_gi  12.908   gdf_ao  6.082   ssao 1.892

High, vault, D3D12   gpu_total 197.744 ms = 5.1 fps
  gdf_reflect 101.590  gi_volume  57.777   gdf_gi  15.199   gdf_ao  8.975   ssao 2.190

High, door, Vulkan   119.252      High, vault, Vulkan   159.736
Med,  door, D3D12     21.487 = 46.5 fps  (for scale: Med did not reach 60 fps either)
```

**The frame is 100 % GPU-bound**: `fence-wait 153.587` of `frame 155.675` is 99.9 %. The CPU's own
10.0 ms of record work is pipelined against the previous frame. This kills an entire class of
proposals up front — CPU culling, draw batching, a depth pre-pass, RHI threading — none can move a
number that is already fully GPU. Only GPU work removed counts.

### 1a. Why High was 8× Med — provenance

`docs/lighting-ao-shadow-closure.md:149-171` already diagnosed this on 2026-07-25 (commit `ff1411e`)
and named it: High is **not "the tier for strong GPUs", it is the tier that never received an
optimization wave.** The 2026-07-25 Med promotion stack was written into the `Med` RON entry only,
and High kept settings inherited from the **gallery anchor**, whose purpose is byte-identity, not
interactivity. Three of them were load-bearing, and none buys image quality at this scene's scale:

| Anchor setting kept at High | What it did |
|---|---|
| `reflect_half_res: false` | made High's own `reflect_res_div: 2` **dead code** — the divisor is only consumed when the half-res flag is on (`main.rs:3714`), so the GGX trace ran at full internal res |
| `gi_half_res: false` | same for `gi_res_div: 2` (`main.rs:3378`) |
| `cache_relight_period: 1` | the anchor's "every card, every frame" path (`sdf_cache_light.slang:166`) |

---

## 2. The finding that reframed the phase — the relight was hiding behind a freeze

`sdf_cache_light` **does not appear at all** in the High baseline profile, yet it costs 2.918 ms at
Med. That is not a naming difference. The surface-cache relight is skipped once the cache is declared
converged (`main.rs:6876`, `if !self.async_cache_on && !cache_settled`), and the latch arms on a
**deterministic horizon**:

```
freeze_horizon = cache_freeze_passes × cache_relight_period      (main.rs:6835-6837)
               = 3 × 1 = 3 frames                                 at High's period 1
```

So at High the relight froze at **frame 3** — long before the 45-frame settle — and every profile
ever taken of this tier, including the 153.762 ms baseline, measured a frame with the relight
switched off. Measuring with the latch disarmed (`P_CACHE_DIRTY_SKIP=0`) exposes the real cost:

| config | gpu_total | `sdf_cache_light` | rest |
|---|---:|---:|---:|
| baseline High, vault, relight live | **909.4 ms** | **719.8** | 189.6 |
| optimized GI/reflect, vault, relight live | **446.6 ms** | **431.3** | 15.3 |
| optimized GI/reflect, door, relight live | **290.2 ms** | **276.8** | 13.4 |

The GI/reflection optimization was real (the "rest" column matches the frozen measurement exactly),
but the dominant cost of a *dynamic* High frame was never `gi_volume` or `gdf_reflect` — it was a
relight running **all cards, every frame, at 8 rays each**.

The amortization to fix it already existed and High simply was not using it:
`selected = (card % period) == (frame % period)` (`sdf_cache_light.slang:195`) — a card round-robin,
so the cost is **flat per frame**, not a burst. Tier values before this wave:

| tier | `cache_relight_period` | `cache_relight_spp` |
|---|---:|---:|
| Low | 48 | 2 |
| Med | 40 | 1 |
| **High** | **1** | **8** |
| Apple | 64 | 1 |

---

## 3. What shipped

One RON block (`apps/sandbox/config/scalability.ron`, `tier: High`) plus one code fix (§4).

| field | before | after | why |
|---|---|---|---|
| `cache_relight_period` | 1 | **8** | fixed-budget amortization, 1/8 of cards per frame. `cache_freeze_passes` counts full amortization **passes**, so total relight work before convergence is *identical* — only its distribution changes, from 3 catastrophic frames to 24 cheap ones. Converged quality bit-unchanged. |
| `cache_relight_spp` | 8 | **8** (kept) | free: in steady state the relight is skipped, so the ray count costs nothing and buys a cleaner transient. Measured p8/spp8 = 14.805 ms vs p8/spp4 = 14.8 — identical. |
| `reflect_half_res` | false | **true** | un-deadens `reflect_res_div` |
| `reflect_res_div` | 2 | **4** | `gdf_reflect` 101.6 → 3.1 ms at the vault |
| `gi_half_res` | false | **true** | un-deadens `gi_res_div` |
| `gi_res_div` | 2 | **3** | `gdf_gi` 15.2 → 1.5 ms |
| `render_scale` | 1.0 | **0.90** | UE's ScreenPercentage lever. Also gives High the temporal AA it did not have — at 1.0 no `taau` pass ran at all. |
| `ao_res_div` | 1 | **2** | `gdf_ao` 9.0 → 2.1 ms |
| `ssao` | true | **false** | the screen-space near-field AO multiplies into the **same** diffuse ambient the GDF far-field AO already occludes (`pbr.slang:708`). Med/Low/Apple already ship it off; High was the only tier paying twice. 2.19 ms. |
| `gdf_cone_k` | 0.0 | **0.06** | cone-LOD march slope (Med's value) |
| `gi_volume_period` | 1 | **2** | probe-grid slab amortization |
| `cache_grid` | (absent) | **true** | card-grid gather |
| `gi_volume_spp` | (absent → 4) | **1** | paired with `gi_dir_sets: 16` — the same 16 directions per rotation period as Med's 4×4, integrated by the EMA over more frames |
| `gi_dir_sets` | (absent → 4) | **16** | keeps High's finer volume-direction rotation |
| `taau_*` (4 fields) | (absent) | Med's values | High now upscales, so it owns the "TAA is blurry when you move" artifact and needs the landed Catmull-Rom stack (`docs/taau-motion-sharpness.md`). Without them the serde defaults give the pre-Catmull-Rom bilinear history. |

---

## 4. One code fix — edge-aware sky-vis upscale follows the image, not the producer

Turning `gi_half_res` on makes the **sky-visibility image** (and the bent normal it carries) come out
at the GI trace extent = 1/3 res. The deferred lighting chose its upscale by asking *which producer
made the image*:

```rust
skyvis_pp_out.is_some(),   // "edge-aware only for the per-pixel producer"
```

That encoded "the volume producer is always full-res", which was true only while the gallery and the
full-res tiers were the only consumers. Every tier that traces GI at 1/N — Med, Low, Apple, and now
High — was plain-bilinear-upsampling a 1/N sky-vis, bleeding open-sky V across silhouettes onto
occluded stone. Fixed at the root by testing the property that actually matters:

```rust
skyvis_pp_out.is_some() || (gi_skyvis_out.is_some() && self.gi_half_res),
```

`gi_half_res` is off for the gallery, so the byte-identical anchor is unchanged. This composes with
the F6P bent-normal contract (`crates/shader/shaders/skyvis_sh.slang`) landed immediately before it.

---

## 5. Verification — measured on the shipped RON, no env overrides

### 5a. Performance (gate: ≤ 16.67 ms)

| | vault DX | door DX | vault VK | door VK |
|---|---:|---:|---:|---:|
| before | 197.744 | 153.762 | 159.736 | 119.252 |
| **after** | **15.462** (rep 15.536) | **12.991** | **15.941** | **13.636** |
| fps | 64.7 | 77.0 | **62.7** | 73.3 |
| cut | **12.8×** | 11.8× | 10.0× | 8.7× |

Binding margin: **0.73 ms (VK, vault)**. D3D12 vault repeatability 15.462 / 15.536.

### 5b. Fidelity vs the path tracer — *improved*

Low-frequency lighting residual against a 1080p `P8_PATHTRACE=1` reference at the same camera
(box-downsampled to strip albedo detail, per-channel gain-matched; lower = closer to truth):

| camera | 1/8 | 1/16 | 1/32 | B−R colour-cast error |
|---|---|---|---|---|
| vault | 35.62 → **30.91** (−13.2 %) | 34.19 → **29.84** (−12.7 %) | 32.98 → **28.87** (−12.4 %) | 10.57 → **6.54** |
| door | 12.82 → **12.42** (−3.2 %) | 12.71 → **12.34** (−3.0 %) | 12.52 → **12.20** (−2.5 %) | 0.69 → **0.16** |

The old High carried a strong blue skylight cast; that cast was a **deviation from ground truth**, and
removing the double-counted `ssao` layer plus the cone-LOD/cache changes moved the tier toward the
path tracer, not away from it.

### 5c. Backend parity — *improved*

Like-for-like, same build, same camera, DX vs VK:

| | old preset | new preset |
|---|---:|---:|
| vault | 3.670 avg/ch | **2.670** (−27 %) |
| door | 2.340 avg/ch | **1.676** (−28 %) |

(This content-scene divergence is the pre-existing open item — see the PT-divergence phase — not
something this wave introduces. The gate is that it must not grow; it shrank.)

### 5d. Gallery anchor — held

`RENDER_QUALITY=high` vs `med` on the gallery: **0.0001 avg/ch, max 4** = D3D12's known 1-LSB
run-to-run nondeterminism. The gallery resolves against `quality::gallery_preset()`, not the active
tier, so the tier edit cannot reach it. Gallery DX vs VK is **0.0063 before and after** — unchanged.

---

## 6. OPEN — the transient after a lighting change

This is the honest gap and it is not closed by this wave.

`cache_relight_period: 8` makes the *steady state* free, but when the lighting epoch changes the latch
disarms and the relight runs for the ~24 frames it takes to re-converge. Measured at the vault with
the latch held off:

```
optimized config, relight live, vault:  50.396 ms   (sdf_cache_light 35.68)
```

So a lighting change costs a ~50 ms frame (≈20 fps) for roughly 24 frames — a visible hitch, not a
stall. Three observations for whoever picks this up:

1. The per-card-texel relight is simply expensive: 1/8 of the cards at 4 rays = 35.7 ms implies
   ~285 ms for a full pass. Raising the period further trades hitch depth for convergence latency
   linearly and cannot alone reach 16.67 ms during the transient (period 40 → ~7 ms, but total ~21 ms
   because of item 2).
2. **`cache_converged` is overloaded.** It gates the relight skip *and* the A3 reflect-skip reuse
   (`main.rs:7764`, `let stable = cache_converged;`). Measured: with the latch not armed, vault
   `gdf_reflect` goes 3.06 → ~15 ms. Two unrelated optimizations ride one flag, so any change to the
   relight schedule silently moves reflection cost by 12 ms. Decoupling them is the first thing to do.
3. The standing directive prohibits **convergence-freeze latches** (the `gi_volume` freeze was
   reverted, `ea22a93`), and this wave adds none — `cache_relight_period` is genuine fixed-budget
   amortization. But the *pre-existing* `cache_frozen` / `cache_converged` latch remains, and it is
   what makes the steady-state numbers above possible. It is flagged for the same review.

---

## 7. Rejected / not done, with reasons

- **CPU-side work** (culling, batching, depth pre-pass, RHI threading): the frame is 99.9 % GPU-bound
  (§1). Zero available.
- **Async compute**: the infrastructure exists (`docs/async-compute.md`) but the frame is ~94 % GDF
  sphere-march compute; async redistributes work across queues, it does not remove FLOPs. Also the
  async relight path cannot read the sky-visibility SH volumes, which is why `cache_sky_occlude`
  forces the sync path.
- **`render_scale` alone**: `gi_volume` is a *world-space* probe update and is inert to render scale
  (measured flat 2.70–2.84 ms across every scale). Partitioning the baseline door frame gives
  `gpu(s) = 66.8 + 87.0·s²`, an asymptote of **67.5 ms (14.8 fps) as s → 0**. UE's ScreenPercentage
  is a structurally weaker lever here than in a conventional deferred renderer.
- **Demoting High to Med's settings**: Med measured 21.487 ms = 46.5 fps. Not a solution.
- **`cache_relight_period: 40`** (proposed during design): never measured, and it pushes
  `freeze_horizon` to 120 frames — past the 107-frame profile window — so it would have disarmed both
  the relight skip and the A3 reflect reuse and silently added ~15 ms. Rejected.

## 8. Follow-ups

1. Decouple A3 reflect-skip reuse from `cache_converged` (§6-2) — worth ~12 ms of transient headroom.
2. **Tier inversion**: High now measures 15.5 ms where Med measures 21.8 ms at the same camera
   (Med's `gi_volume` is 10.53 ms, spp 4 / dir_sets 4). Med needs the same `gi_volume` re-layout, or
   the tier ordering is wrong.
3. Re-confirm the numbers on a **dolly** (`CAPTURE_SEQ` + `tools/seq-stability.py`): a static camera
   lets the temporal caches idle, and TAAU/reflection temporal quality is a motion property.
4. Metal / Apple tier unverified — this wave was measured on D3D12 and Vulkan only.
