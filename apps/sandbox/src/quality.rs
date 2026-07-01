//! RenderQuality tiers (Phase 11 Stage D) — the single source of truth that maps a
//! quality tier to the quality knobs that were previously scattered across `main.rs`
//! as ad-hoc `env.unwrap_or(<hardcoded>)` defaults.
//!
//! Selection: `RENDER_QUALITY=low|med|high` (unset / unknown => `Med`). `Med` reproduces
//! the historical defaults byte-for-byte (the no-regression target), so an unset env is
//! identical to before this module existed.
//!
//! Seam preserved: this module only supplies *defaults*. Every consumer still reads its
//! individual env var first and falls back to the preset (`env.unwrap_or(preset.x)`), so an
//! explicit `P11_*` / `SHADOW_*` override always wins over the tier. The tier is a thin
//! selection layer — no rendering logic lives here, and capability gates stay at the call site.
//!
//! Design rules (CLAUDE.md): default tier = cheapest *accurate* path; heavy features opt in at
//! higher tiers; one place owns the tier→knob table; measurement-rejected knobs are excluded
//! (`P11_GDF_DIM` resolution, `CARD_TILE` — see `docs/reflection-sdf-resolution.md`).

/// QHD/UHD track (Stage 8): TAA-aware texture LOD bias. When sub-pixel jitter is active the
/// temporal accumulation super-samples the image, so we can bias the G-buffer texture fetches
/// toward *sharper* mips and let TAA resolve the extra aliasing — this is the PRIMARY lever for
/// distant-texture sharpness (the 레퍼런스 엔진/DLSS/FSR2 approach), not anisotropy. It is added on top of
/// the resolution term `log2(internal/output)` and applies even at native resolution under forced
/// TAA (`P_TAAU_FORCE`). Driver-independent (a plain LOD offset on the existing trilinear sampler),
/// so it carries no DX≡VK risk. `-1.0` ≈ one mip sharper; tuning range -0.5..-1.5 (too negative ->
/// motion shimmer the temporal pass can't hide). Overridable via the `TAA_MIP_BIAS` env for sweeps.
/// Single source of truth — read once in `main.rs`. Gallery (TAA off => no jitter) never applies it,
/// so the byte-identical anchor is preserved.
pub const TAA_MIP_BIAS: f32 = -1.0;

/// Render quality tier. `Med` is the explicit default (`RENDER_QUALITY=med`) and matches the legacy
/// behavior byte-for-byte. `Apple` is a platform-default tier auto-selected on Apple GPUs (never via
/// an explicit `RENDER_QUALITY` value) — see [`RenderQuality::from_env_for_device`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderQuality {
    /// Low-end fallback: heavy reflection/GI features off, fewer samples, cheaper SSR.
    Low,
    /// Default — byte-identical to the pre-tier behavior (no-regression baseline).
    Med,
    /// Quality: opt-in surface cache / GDF AO, doubled GI samples, aesthetic soft shadows.
    High,
    /// Apple-platform default (macOS perf, axis A): a Med-derived tier tuned for the weak unified
    /// iGPU + TBDR of Apple Silicon. Auto-selected only when `RENDER_QUALITY` is UNSET and an Apple
    /// GPU is detected; an explicit `RENDER_QUALITY=med|low|high` always wins over it. Drops the
    /// internal render resolution, turns off the redundant second AO pass, and shortens the SW-RT
    /// reflection/GI marches — all as *tier defaults*, so every `RENDER_SCALE`/`SSAO`/`P11_*`
    /// override still wins at the consumer site. Never affects the gallery anchor (forced legacy at
    /// the call site) or the VK/D3D12 backends (they never report an Apple GPU).
    Apple,
}

impl RenderQuality {
    /// Resolve the active tier, consulting the GPU identity for the platform default. An explicit
    /// `RENDER_QUALITY=low|med|high` still wins (returned verbatim); only the UNSET path consults
    /// the device, and only an Apple GPU changes the result (to the aggressive [`Apple`] tier).
    /// Non-Apple / unknown falls back to the honest `Med`. This is the sole entry point that can
    /// return [`RenderQuality::Apple`] — the tier can never be forced via an env string.
    ///
    /// [`Apple`]: RenderQuality::Apple
    pub fn from_env_for_device(info: &rhi::DeviceInfo) -> Self {
        Self::from_explicit_env().unwrap_or_else(|| Self::device_default(info))
    }

    /// The explicit `RENDER_QUALITY` selection, or `None` when unset / unrecognized (=> a
    /// platform-default path should decide). `Apple` is intentionally NOT reachable here.
    fn from_explicit_env() -> Option<Self> {
        match std::env::var("RENDER_QUALITY")
            .ok()
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("low") => Some(RenderQuality::Low),
            Some("med") | Some("medium") => Some(RenderQuality::Med),
            Some("high") => Some(RenderQuality::High),
            _ => None,
        }
    }

    /// The default tier for a given GPU when `RENDER_QUALITY` is unset. Apple GPUs (detected by the
    /// vendor name substring, `hasUnifiedMemory` as a secondary hint) map to the aggressive [`Apple`]
    /// tier; every other / unknown GPU keeps the honest `Med` fallback (matches VK/D3D12, which never
    /// report an Apple adapter — so this is a no-op for those backends).
    ///
    /// [`Apple`]: RenderQuality::Apple
    fn device_default(info: &rhi::DeviceInfo) -> RenderQuality {
        // Primary signal: the adapter name contains "Apple". Secondary: a unified-memory low-power
        // GPU is an integrated part that also wants the aggressive tier (defends against a driver
        // formatting change that drops "Apple" from the name).
        if info.is_apple_gpu() || (info.unified_memory && info.low_power) {
            RenderQuality::Apple
        } else {
            RenderQuality::Med
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            RenderQuality::Low => "low",
            RenderQuality::Med => "med",
            RenderQuality::High => "high",
            RenderQuality::Apple => "apple",
        }
    }
}

/// Per-tier default values for the quality knobs. The ONE place the tier→knob mapping lives;
/// every field is overridable by its individual env var at the consumer site.
pub struct QualityPreset {
    /// C3 hemisphere rays per pixel (`P11_GI_SPP`).
    pub gi_spp: u32,
    /// C4 spatio-temporal GI denoise (`P11_GI_DENOISE`).
    pub gi_denoise: bool,
    /// C8g reflection hit cache (`P11_REFLECT_CACHE`).
    pub reflect_cache: bool,
    /// C8b3 GI surface-cache multibounce lookup (`P11_SURFACE_CACHE`) — heavy, High only.
    pub surface_cache: bool,
    /// SSR mode: stochastic half-res glossy path vs full-res mirror (`P11_SSR_STOCHASTIC`).
    pub ssr_stochastic: bool,
    /// C8d roughness above which screen-mirror SSR fades to the GDF fallback (`P11_REFLECT_MAX_ROUGHNESS`).
    pub reflect_max_roughness: f32,
    /// C2 GDF ambient occlusion (`P11_GDF_AO`).
    pub gdf_ao: bool,
    /// Near-field screen-space AO (HBAO-lite), a SECOND independent AO pass composed with `gdf_ao`
    /// (`SSAO`). On for content on most tiers; the Apple platform tier turns it OFF (gdf_ao already
    /// supplies contact AO, and this reclaims a full ~13 ms pass on the M3). Gallery forces it via
    /// `!gallery_scene` at the call site regardless of the tier, so the anchor is unaffected.
    pub ssao: bool,
    /// Firefly clamp on the reflection/GI radiance (`P11_FIREFLY_CLAMP`).
    pub firefly_clamp: bool,
    /// PCSS-lite penumbra scale; 0 = hard 3x3 PCF (`SHADOW_SOFTNESS`).
    pub shadow_softness: f32,
    /// Soft-shadow blocker/PCF tap count, written to `globals.shadow.w` (`SHADOW_TAPS`).
    /// Only consumed by the soft path (softness > 0); clamped to [1, 16] (POISSON16) in the shader.
    pub shadow_taps: u32,
    /// Stage D2 (Sponza 60fps): surface-cache amortized-relight period — relight `1/period` of the
    /// cards per frame (round-robin), the rest persist their radiance (`P11_CACHE_RELIGHT_PERIOD`).
    /// `1` = the legacy every-frame relight (byte-identical; forced for the gallery anchor at the
    /// call site). Higher = cheaper `sdf_cache_light`, slower convergence. 레퍼런스 엔진 SW GI surface-cache
    /// update budget; see `sdf_cache_light.slang` and `docs/sponza-perf.md`.
    pub cache_relight_period: u32,
    /// Stage D1 (Sponza 60fps): trace the C3 GI at half resolution + joint-bilateral upsample
    /// (1/4 the rays) (`P11_GI_HALF_RES`). Forced off for the gallery anchor (full-res =
    /// byte-identical) at the call site. 레퍼런스 엔진 SW GI screen-probe / 다른 레퍼런스 엔진 half-res GI; see
    /// `gdf_gi_upsample.slang`.
    pub gi_half_res: bool,
    /// Stage D3 (Sponza 60fps): surface-cache relight indirect-gather rays per texel
    /// (`P11_CACHE_RELIGHT_SPP`). Forced to the legacy 8 for the gallery anchor (byte-identical).
    /// The gather dominates `sdf_cache_light`; halving it ~halves the pass. Denoised by the
    /// cache's temporal EMA, so fewer rays converge to the same static result.
    pub cache_relight_spp: u32,
    /// Stage D3 (Sponza 60fps): C3 GI bounce-ray march step cap (`P11_GI_MAX_STEPS`). Forced to
    /// the legacy 64 for the gallery anchor (byte-identical). Fewer steps = cheaper march; the
    /// indirect bounce is low-frequency + denoised, so a shorter march holds up for content.
    pub gi_max_steps: u32,
    /// Stage D3 (Sponza 60fps): GGX reflection-ray march step cap (`P11_REFLECT_MAX_STEPS`).
    /// Forced to the legacy 96 for the gallery anchor (byte-identical). The reflection is
    /// temporally accumulated, so a shorter march holds up for content. NOTE: capping the march
    /// too low makes distant reflection rays leak sky — half-res is the cheaper lever (full march,
    /// 1/4 the pixels), so keep this near the legacy 96 except on Low.
    pub reflect_max_steps: u32,
    /// Stage D3 (Sponza 60fps): trace the GGX reflection at half resolution + joint-bilateral
    /// upsample (1/4 the rays) (`P11_REFLECT_HALF_RES`). Forced off for the gallery anchor
    /// (full-res = byte-identical). Reuses `gdf_gi_upsample.slang`.
    pub reflect_half_res: bool,
    /// QHD/UHD track: internal render-resolution scale (fraction of the display extent the scene
    /// renders at; tonemap upscales to the display) (`RENDER_SCALE`). `1.0` = native (byte-
    /// identical). The seam for a future dynamic-resolution controller; default 1.0 at every tier
    /// (the scale that hits a given fps depends on the display resolution + target, not the tier).
    pub render_scale: f32,
    /// P3 (SW-RT GI 레퍼런스급 SW-RT): cone-trace LOD march slope (`P_CONE_K`). The SW-RT march loops
    /// (GI bounce / reflection / surface-cache gather + their soft-shadow marches) widen the step
    /// with distance: floor `max(d, cone_k·t)` and shadow ceiling `max(0.2, cone_k·t)`. Fewer steps
    /// at distance (grazing rays stop crawling). `0.0` = legacy linear march (byte-identical; forced
    /// for the gallery anchor at the call site). Higher = cheaper march, softer distant GI/reflection.
    /// Denoised/EMA signals tolerate it; see `docs/swrt-gi-perf-track.md`.
    pub gdf_cone_k: f32,
    /// P1 (SW-RT GI 레퍼런스급 SW-RT): GI trace-resolution divisor when `gi_half_res` is on — trace the
    /// C3 GI at `1/div` of the render extent per axis, then joint-bilateral upsample (the spatial
    /// half of 레퍼런스 엔진 SW GI's ScreenProbeGather: sparser trace origins, guided interpolation).
    /// `2` = the legacy half-res (Stage D1). `3` = third-res — measured sweet spot: gdf_gi ~-48% vs
    /// half with DX=VK back at the baseline ~0.006/ch. `4` = quarter-res is faster still (~-65%) but
    /// over-reaches: each divergent coarse stochastic GI ray spreads over a larger upsample footprint,
    /// raising the DX=VK gap to ~0.117/ch (a broad parity divergence, not just fireflies) — rejected.
    /// Only active where `gi_half_res` is (content; the gallery traces full-res = byte-identical).
    /// `P_GI_RES_DIV` override. See `docs/swrt-gi-perf-track.md`.
    pub gi_res_div: u32,
    /// macOS/M3 perf (M3-C): GGX reflection trace-resolution divisor when `reflect_half_res` is on —
    /// trace `gdf_reflect` at `1/div` of the render extent per axis, then the same joint-bilateral
    /// upsample the GI uses. `gdf_reflect.slang` samples the G-buffer by normalized UV, so any div
    /// traces correctly with no shader change. `2` = the legacy half-res (byte-identical with the old
    /// `hcw/hch` = `div_ceil(2)`; every non-Apple tier keeps it). `4` = quarter-res — the ONE
    /// quality-preserving lever on `gdf_reflect`, which the M3 sweep proved responds to trace
    /// resolution and nothing else (steps/roughness had ~0 effect; full-res = 120ms, half = 31ms). The
    /// reflection is temporally accumulated + low-frequency, so the coarser trace holds up for content.
    /// Only active where `reflect_half_res` is (content; the gallery traces full-res = byte-identical).
    /// `P_REFLECT_RES_DIV` override. See `docs/phase-macos-perf-impl.md`. Metal-measured here; the
    /// cross-backend parity of div>2 is a Windows follow-up (cf. the `gi_res_div=4` DX=VK note above).
    pub reflect_res_div: u32,
    /// macOS/M3 perf: GDF ambient-occlusion trace-resolution divisor — trace `gdf_ao` at `1/div` of
    /// the render extent, then the same joint-bilateral upsample (guided by depth+normal, which keeps
    /// the contact band crisp across depth edges — the standard half-res-AO reconstruction). `1` =
    /// full-res (byte-identical; every non-Apple tier keeps it, so Med/High content is unchanged). The
    /// Apple tier uses `2` (half-res): after the quarter-res reflection, `gdf_ao` is the top SW-RT pass
    /// (~12ms @ rs0.67), and AO is a low-frequency contact term that survives a half-res trace + guided
    /// upsample. The gallery never runs `gdf_ao` (`!gallery_scene`), so the anchor is unaffected
    /// regardless. `P_AO_RES_DIV` override. See `docs/phase-macos-perf-impl.md`.
    pub ao_res_div: u32,
    /// Reflection temporal-resolve history neighbourhood clamp (`reflect_temporal.slang`): removes the
    /// view-dependent specular smear when the camera rotates (stale reprojected history dragged across
    /// chrome). A scalability permutation: `0` = off (byte-identical legacy resolve; forced for the
    /// gallery anchor), `1` = hard AABB clamp (cheapest), `2` = variance clamp (mean +- gamma*sigma,
    /// gentler on sharp mirrors). `P_REFL_CLAMP` override. See `docs/swrt-gi-perf-track.md`.
    pub reflect_history_clamp: u32,
    /// Variance-clamp tightness for `reflect_history_clamp == 2` (`P_REFL_CLAMP_GAMMA`). Lower = tighter
    /// (more lag removed, more risk of clipping valid history); ~1.0-1.5 typical. Ignored for modes 0/1.
    pub reflect_clamp_gamma: f32,
    /// GI temporal denoiser history-clamp mode (`gdf_temporal.slang` params.w), `P_GI_TEMPORAL_CLAMP`:
    /// `0` = off — the measured fix for the static GI shimmer (the legacy hard 3x3 clamp is built from
    /// the noisy spp1 GI, so it drags the converged history back to per-frame noise = flicker; off lets
    /// the EMA converge, with the per-sample firefly clamp + a-trous still catching fireflies). `1` =
    /// hard 3x3 min/max (legacy; forced for the gallery byte-identical anchor). `> 1.5` = variance clamp
    /// with gamma = this value (a wide outlier reject that still converges). Content defaults off.
    pub gi_temporal_clamp: f32,
}

/// The tier→knob table. Med must equal the legacy hardcoded defaults (no-regression).
pub fn preset(q: RenderQuality) -> QualityPreset {
    match q {
        // Low-end fallback: reflection hit cache off, cheap stochastic half-res SSR, half the
        // GI samples, lower reflection roughness cutoff (GDF takes over sooner). Hard shadows.
        RenderQuality::Low => QualityPreset {
            gi_spp: 2,
            gi_max_steps: 24,
            reflect_max_steps: 64,
            gi_denoise: true,
            reflect_cache: false,
            surface_cache: false,
            ssr_stochastic: true,
            reflect_max_roughness: 0.3,
            gdf_ao: false,
            ssao: true, // content default (gallery forces via !gallery_scene at the call site)
            firefly_clamp: true,
            shadow_softness: 0.0,
            shadow_taps: 8,
            cache_relight_period: 48,
            gi_half_res: true,
            cache_relight_spp: 2,
            reflect_half_res: true,
            gdf_cone_k: 0.05,
            gi_res_div: 3,
            reflect_res_div: 2,       // legacy half-res reflection
            ao_res_div: 1,            // full-res AO
            reflect_history_clamp: 1, // hard (cheapest) — kills rotation smear
            reflect_clamp_gamma: 1.25,
            gi_temporal_clamp: 0.0,
            // Low-end / high-res performance mode: render at 2/3 of the output extent and let the
            // TAAU jitter reconstruction (B-track) upscale it. 2/3 (not 1/2) keeps detailed scenes
            // legible — at 1/2 the internal resolution undersamples texture/geometry detail enough
            // that even the temporal reconstruction reads as soft (poor visibility). Measured
            // (Sponza, output 2052x1133): internal 0.6667 = d3d12 14.5ms / vk 18.5ms (async vk
            // ~9.9ms); the reconstruction needs the jitter, on by default in this path.
            render_scale: 0.6667,
        },
        // Default — identical to the pre-tier behavior. Do not change without re-baselining no-reg.
        RenderQuality::Med => QualityPreset {
            gi_spp: 1,
            gi_max_steps: 24,
            reflect_max_steps: 96,
            gi_denoise: true,
            reflect_cache: true,
            surface_cache: false,
            ssr_stochastic: false,
            reflect_max_roughness: 0.5,
            gdf_ao: true, // PBR contact AO (fixed contact-scale reach) — depth for content
            ssao: true,   // content default (= the legacy !gallery_scene default; no-reg)
            firefly_clamp: true,
            shadow_softness: 0.0,
            shadow_taps: 16,
            // Stage D2b/D3: visibility feedback (off-screen cards relit 8x less) + period-aware EMA
            // alpha let the period reach 레퍼런스 엔진's 32 range; gather spp 2 (denoised) + half-res GI/reflect
            // bring the GDF SW-RT stack into the 60fps frame budget on both backends.
            cache_relight_period: 40,
            gi_half_res: true,
            cache_relight_spp: 1,
            reflect_half_res: true,
            render_scale: 1.0,
            gdf_cone_k: 0.02,
            gi_res_div: 3,
            reflect_res_div: 2,       // legacy half-res reflection (no-reg)
            ao_res_div: 1,            // full-res AO (no-reg)
            reflect_history_clamp: 1, // hard (matches the WIP default) — kills rotation smear
            reflect_clamp_gamma: 1.25,
            gi_temporal_clamp: 0.0,
        },
        // Quality: opt-in multibounce surface cache + GDF AO, 2x GI samples, higher reflection
        // roughness cutoff, aesthetic soft shadows (diverges slightly from PT — see docs).
        RenderQuality::High => QualityPreset {
            gi_spp: 16,
            gi_max_steps: 64,
            reflect_max_steps: 96,
            gi_denoise: true,
            reflect_cache: true,
            surface_cache: true,
            ssr_stochastic: false,
            reflect_max_roughness: 0.6,
            gdf_ao: true,
            ssao: true, // content default (gallery forces via !gallery_scene at the call site)
            firefly_clamp: true,
            shadow_softness: 0.03,
            shadow_taps: 16,
            cache_relight_period: 1,
            gi_half_res: false,
            cache_relight_spp: 8,
            reflect_half_res: false,
            render_scale: 1.0,
            gdf_cone_k: 0.0,
            gi_res_div: 2,
            reflect_res_div: 2, // legacy half-res reflection (quality tier keeps it sharp)
            ao_res_div: 1,      // full-res AO (quality tier)
            reflect_history_clamp: 2, // variance (gentler on sharp mirrors) — quality tier
            reflect_clamp_gamma: 1.25,
            gi_temporal_clamp: 0.0,
        },
        // Apple-platform default (macOS perf, axis A). Derived from `Med` — same feature SET and the
        // same denoise/clamp behaviour, so the look matches Med — with the cost knobs pushed for the
        // weak unified iGPU + TBDR of Apple Silicon. Auto-selected only on an Apple GPU with
        // `RENDER_QUALITY` unset; every value here is a tier DEFAULT that its own env var overrides.
        //
        // The M0 baseline (Sponza Med, native 1440p on M3) is ~165 ms; the SW-RT reflection+AO stack
        // is ~82% of it, and `render_scale=1.0` (native QHD on an iGPU) is the single biggest lever.
        // Starting points (the lead measures the real ms; these are picked, not measured here):
        //   * render_scale 0.67 — ~0.44x internal pixels vs native; every per-pixel SW-RT pass
        //     (gdf_reflect/gdf_ao/ssao/ssr/temporal) scales ~with it. The 2/3 (not 1/2) point keeps
        //     detailed geometry legible under TAAU (see the Low-tier note). This is THE big win.
        //   * ssao OFF — gdf_ao already supplies contact AO; the near-field HBAO-lite pass is a
        //     second, independent ~13 ms AO pass whose contribution largely overlaps gdf_ao's on
        //     content. Dropping it reclaims that pass outright. (gdf_ao stays on — the depth cue.)
        //   * reflect_max_steps 96 -> 56 — gdf_reflect dominates the frame; the reflection is
        //     temporally accumulated + half-res, so a shorter GGX march holds up. Mid of the 48-64
        //     band: low enough to pay off, high enough that distant rays don't obviously leak sky.
        //   * gdf_cone_k 0.06 — widen the cone-trace step with distance across GI/reflection/cache
        //     marches (grazing rays stop crawling); denoised/EMA signals tolerate it. Above Med's
        //     0.02, near Low's 0.05.
        //   * gi_res_div 4 — quarter-res GI trace (vs Med's third-res). The div-4 parity risk noted
        //     for Med is a DX=VK concern; macOS is Metal-only, so we can take the cheaper trace here.
        //   * reflect_max_roughness 0.4 — fade screen-mirror SSR to the cheap GDF fallback sooner
        //     (rougher surfaces don't need the sharp SSR march). Below Med's 0.5, above Low's 0.3.
        //   * cache_relight_period 64 — relight fewer surface-cache cards per frame (vs Med's 40).
        //     The cache persists radiance + EMA-denoises, so a longer period trades moving-camera
        //     convergence lag for a cheaper `sdf_cache_light`. Aggressive but reversible.
        // Everything else tracks Med (feature set, GI half-res, spp, shadows, temporal clamps).
        RenderQuality::Apple => QualityPreset {
            gi_spp: 1,
            gi_max_steps: 24,
            reflect_max_steps: 56,
            gi_denoise: true,
            reflect_cache: true,
            surface_cache: false,
            ssr_stochastic: false,
            reflect_max_roughness: 0.4,
            gdf_ao: true, // contact AO retained — it is the AO source once ssao is off
            ssao: false,  // OFF on Apple: gdf_ao covers contact AO; reclaims the ~13 ms 2nd AO pass
            firefly_clamp: true,
            shadow_softness: 0.0,
            shadow_taps: 16,
            cache_relight_period: 64,
            gi_half_res: true,
            cache_relight_spp: 1,
            reflect_half_res: true,
            render_scale: 0.67,
            gdf_cone_k: 0.06,
            gi_res_div: 4,
            reflect_res_div: 4, // quarter-res reflection (M3-C): the one lever that cuts gdf_reflect
            ao_res_div: 2,      // half-res AO: gdf_ao is the top pass after quarter-res reflection
            reflect_history_clamp: 1, // hard (matches Med) — kills rotation smear
            reflect_clamp_gamma: 1.25,
            gi_temporal_clamp: 0.0,
        },
    }
}

/// Resolve a boolean knob: explicit env (`0`/`false`/`off` => false, any other value => true)
/// overrides the tier default; unset => `tier_default`. Replaces the old presence-only
/// (`var_os(..).is_some()`) toggles so a higher tier's on-by-default can still be turned off
/// via env (e.g. `P11_GDF_AO=0` on High), keeping the override seam symmetric.
pub fn env_bool(name: &str, tier_default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => tier_default,
    }
}
