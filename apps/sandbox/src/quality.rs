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
/// distant-texture sharpness (the UE/DLSS/FSR2 approach), not anisotropy. It is added on top of
/// the resolution term `log2(internal/output)` and applies even at native resolution under forced
/// TAA (`P_TAAU_FORCE`). Driver-independent (a plain LOD offset on the existing trilinear sampler),
/// so it carries no DX≡VK risk. `-1.0` ≈ one mip sharper; tuning range -0.5..-1.5 (too negative ->
/// motion shimmer the temporal pass can't hide). Overridable via the `TAA_MIP_BIAS` env for sweeps.
/// Single source of truth — read once in `main.rs`. Gallery (TAA off => no jitter) never applies it,
/// so the byte-identical anchor is preserved.
pub const TAA_MIP_BIAS: f32 = -1.0;

/// Render quality tier. `Med` is the default (unset env) and matches the legacy behavior.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderQuality {
    /// Low-end fallback: heavy reflection/GI features off, fewer samples, cheaper SSR.
    Low,
    /// Default — byte-identical to the pre-tier behavior (no-regression baseline).
    Med,
    /// Quality: opt-in surface cache / GDF AO, doubled GI samples, aesthetic soft shadows.
    High,
}

impl RenderQuality {
    /// Resolve the active tier from `RENDER_QUALITY` (unset / unrecognized => platform default).
    pub fn from_env() -> Self {
        match std::env::var("RENDER_QUALITY")
            .ok()
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("low") => RenderQuality::Low,
            Some("med") | Some("medium") => RenderQuality::Med,
            Some("high") => RenderQuality::High,
            _ => Self::platform_default(),
        }
    }

    /// The default tier when `RENDER_QUALITY` is unset. `Med` everywhere today (= the legacy
    /// no-reg baseline). This is the seam where future per-platform / per-GPU selection plugs in:
    /// a low-end mobile/iGPU would map to `Low`, a high-end desktop GPU to `High`. Kept honest —
    /// without a GPU perf-tier lookup we don't fake detection, so it returns `Med` for now.
    pub fn platform_default() -> RenderQuality {
        RenderQuality::Med
    }

    pub fn label(self) -> &'static str {
        match self {
            RenderQuality::Low => "low",
            RenderQuality::Med => "med",
            RenderQuality::High => "high",
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
    /// call site). Higher = cheaper `sdf_cache_light`, slower convergence. UE Lumen surface-cache
    /// update budget; see `sdf_cache_light.slang` and `docs/sponza-perf.md`.
    pub cache_relight_period: u32,
    /// Stage D1 (Sponza 60fps): trace the C3 GI at half resolution + joint-bilateral upsample
    /// (1/4 the rays) (`P11_GI_HALF_RES`). Forced off for the gallery anchor (full-res =
    /// byte-identical) at the call site. UE5 Lumen screen-probe / Frostbite half-res GI; see
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
    /// P3 (Lumen-parity SW-RT): cone-trace LOD march slope (`P_CONE_K`). The SW-RT march loops
    /// (GI bounce / reflection / surface-cache gather + their soft-shadow marches) widen the step
    /// with distance: floor `max(d, cone_k·t)` and shadow ceiling `max(0.2, cone_k·t)`. Fewer steps
    /// at distance (grazing rays stop crawling). `0.0` = legacy linear march (byte-identical; forced
    /// for the gallery anchor at the call site). Higher = cheaper march, softer distant GI/reflection.
    /// Denoised/EMA signals tolerate it; see `docs/lumen-parity-swrt.md`.
    pub gdf_cone_k: f32,
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
            firefly_clamp: true,
            shadow_softness: 0.0,
            shadow_taps: 8,
            cache_relight_period: 48,
            gi_half_res: true,
            cache_relight_spp: 2,
            reflect_half_res: true,
            gdf_cone_k: 0.05,
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
            gdf_ao: false,
            firefly_clamp: true,
            shadow_softness: 0.0,
            shadow_taps: 16,
            // Stage D2b/D3: visibility feedback (off-screen cards relit 8x less) + period-aware EMA
            // alpha let the period reach UE's 32 range; gather spp 2 (denoised) + half-res GI/reflect
            // bring the GDF SW-RT stack into the 60fps frame budget on both backends.
            cache_relight_period: 40,
            gi_half_res: true,
            cache_relight_spp: 1,
            reflect_half_res: true,
            render_scale: 1.0,
            gdf_cone_k: 0.02,
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
            firefly_clamp: true,
            shadow_softness: 0.03,
            shadow_taps: 16,
            cache_relight_period: 1,
            gi_half_res: false,
            cache_relight_spp: 8,
            reflect_half_res: false,
            render_scale: 1.0,
            gdf_cone_k: 0.0,
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
