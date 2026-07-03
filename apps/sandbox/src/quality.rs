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
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
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

/// Per-tier default values for the quality knobs. The tier→knob table now lives in the
/// data-driven config (`apps/sandbox/config/scalability.ron`, embedded + on-disk-overridable);
/// this struct is the deserialized shape and every field is overridable by its individual env
/// var at the consumer site. (The gallery anchor stays hard-coded in [`gallery_preset`].)
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
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
    /// supplies contact AO, and this reclaims a full ~13 ms pass on the M3). The gallery anchor
    /// resolves its `ssao` against [`gallery_preset`] (which pins it OFF), not the active tier, so
    /// the byte-identical gallery never runs this pass — the lock is now structural (a single
    /// preset table), not a per-call-site `!gallery_scene` force.
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
    /// macOS/M3 perf: number of edge-aware à-trous spatial GI-denoise iterations after the temporal
    /// pass. `2` (legacy: steps 1,2) everywhere except the Apple tier, which uses `1` — the GI is
    /// traced sparse (`gi_res_div=4`) + temporally EMA-denoised + upsampled, so one wide à-trous
    /// cleans the low-frequency residual (UE reduces/disables Lumen spatial filters at its low tier).
    /// Each iteration is a full-res compute pass (~2.2ms @ rs0.67), so dropping one is a direct save.
    /// `P_GI_ATROUS_STEPS` override. Non-Apple stays 2 (byte-identical). See `docs/phase-macos-perf-autonomous.md`.
    pub gi_atrous_steps: u32,
    /// Reflection temporal-resolve history neighbourhood clamp (`reflect_temporal.slang`): removes the
    /// view-dependent specular smear when the camera rotates (stale reprojected history dragged across
    /// chrome). A scalability permutation: `0` = off (byte-identical legacy resolve; forced for the
    /// gallery anchor), `1` = hard AABB clamp (cheapest), `2` = variance clamp (mean +- gamma*sigma,
    /// gentler on sharp mirrors). `P_REFL_CLAMP` override. See `docs/swrt-gi-perf-track.md`.
    pub reflect_history_clamp: u32,
    /// Variance-clamp tightness for `reflect_history_clamp == 2` (`P_REFL_CLAMP_GAMMA`). Lower = tighter
    /// (more lag removed, more risk of clipping valid history); ~1.0-1.5 typical. Ignored for modes 0/1.
    pub reflect_clamp_gamma: f32,
    /// SSR-resolve history neighbourhood clamp (`ssr_resolve.slang`), `P_SSR_HISTORY_CLAMP`. The SSR
    /// mirror path samples the previous-frame lit-history, so it forms a lighting feedback loop; a plain
    /// EMA only low-passes the resulting period-2 limit cycle (columns/thin geo shimmer), it does not
    /// kill it. `1` = variance clamp (mean +- `ssr_clamp_gamma`*sigma) of the reprojected history into
    /// the current spatial neighbourhood — the step that breaks the oscillation. `0` = off (byte-
    /// identical legacy resolve; forced for the gallery anchor). Default 0 pending DX=VK verification.
    pub ssr_history_clamp: u32,
    /// Variance-clamp tightness for `ssr_history_clamp == 1` (`P_SSR_CLAMP_GAMMA`). ~1.0-1.5 typical.
    pub ssr_clamp_gamma: f32,
    /// GI temporal denoiser history-clamp mode (`gdf_temporal.slang` params.w), `P_GI_TEMPORAL_CLAMP`:
    /// `0` = off — the measured fix for the static GI shimmer (the legacy hard 3x3 clamp is built from
    /// the noisy spp1 GI, so it drags the converged history back to per-frame noise = flicker; off lets
    /// the EMA converge, with the per-sample firefly clamp + a-trous still catching fireflies). `1` =
    /// hard 3x3 min/max (legacy; forced for the gallery byte-identical anchor). `> 1.5` = variance clamp
    /// with gamma = this value (a wide outlier reject that still converges). Content defaults off.
    pub gi_temporal_clamp: f32,
}

// ---------------------------------------------------------------------------
// Data-driven tier tables (single source: apps/sandbox/config/scalability.ron)
// ---------------------------------------------------------------------------
//
// The per-tier `QualityPreset` values and per-tier group levels are DATA, not code: they live
// in a RON config so a tier can be tuned or added without recompiling (the reference-engine
// scalability-config-file model). The file is embedded at compile time via `include_str!` as
// the built-in default, so the binary always works even with no on-disk file; an on-disk copy
// overrides it at startup (parse error => a warning + the embedded fallback). The gallery
// anchor ([`gallery_preset`]) is deliberately NOT data-driven — it stays compiled in, so a stray
// file edit can never silently move the byte-identical path-tracer anchor.

/// The embedded default config — the built-in copy of the tier tables. The binary always has
/// this, so a missing/corrupt on-disk file degrades to the shipped defaults, never to nothing.
const EMBEDDED_CONFIG: &str = include_str!("../config/scalability.ron");

/// One tier's data-driven entry: its full [`QualityPreset`] plus its six scalability-group
/// levels. `tier` keys the entry to a [`RenderQuality`]; the file lists one entry per tier.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
struct TierConfig {
    tier: RenderQuality,
    preset: QualityPreset,
    groups: GroupLevels,
}

/// The six scalability-group levels for a tier (0..=3), a coarse VIEW of the preset (see the
/// group layer below). Named fields keep the RON self-describing and order-independent.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
struct GroupLevels {
    resolution: u8,
    global_illumination: u8,
    reflection: u8,
    ambient_occlusion: u8,
    shadow: u8,
    surface_cache: u8,
}

impl GroupLevels {
    /// The `(group, level)` pairs in [`ScalabilityGroup::ALL`] order — the shape [`groups`] returns.
    fn as_pairs(&self) -> [(ScalabilityGroup, u8); 6] {
        use ScalabilityGroup::*;
        [
            (Resolution, self.resolution),
            (GlobalIllumination, self.global_illumination),
            (Reflection, self.reflection),
            (AmbientOcclusion, self.ambient_occlusion),
            (Shadow, self.shadow),
            (SurfaceCache, self.surface_cache),
        ]
    }
}

/// The parsed scalability config: the list of per-tier entries. One entry per [`RenderQuality`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ScalabilityConfig {
    tiers: Vec<TierConfig>,
}

impl ScalabilityConfig {
    /// Parse a RON document into a config.
    fn parse(text: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(text)
    }

    /// The entry for a tier, or `None` when the config omits it (a partial on-disk override).
    fn tier(&self, q: RenderQuality) -> Option<&TierConfig> {
        self.tiers.iter().find(|t| t.tier == q)
    }
}

/// Load the active config, resolving the on-disk override against the embedded default.
///
/// Search order for the on-disk file: `SCALABILITY_CONFIG=<path>` (explicit), else the committed
/// `apps/sandbox/config/scalability.ron`, else the runtime-editable `assets/config/scalability.ron`.
/// The first that EXISTS is read; if it parses, it wins; if it fails to parse, a warning is logged
/// and we fall back to the embedded default. When no on-disk file exists, the embedded default is
/// used silently. The embedded default is `include_str!`d, so it is guaranteed to parse (a bad
/// commit would be caught by the `embedded_config_parses` unit test).
fn load_config() -> ScalabilityConfig {
    let embedded = ScalabilityConfig::parse(EMBEDDED_CONFIG)
        .expect("embedded scalability.ron must parse (locked by unit test)");
    // Candidate on-disk paths, highest precedence first. Only the first that exists is consulted.
    let candidates: [Option<std::path::PathBuf>; 3] = [
        std::env::var_os("SCALABILITY_CONFIG").map(std::path::PathBuf::from),
        Some(std::path::PathBuf::from(
            "apps/sandbox/config/scalability.ron",
        )),
        Some(std::path::PathBuf::from("assets/config/scalability.ron")),
    ];
    for path in candidates.into_iter().flatten() {
        if !path.exists() {
            continue;
        }
        match std::fs::read_to_string(&path) {
            Ok(text) => match ScalabilityConfig::parse(&text) {
                Ok(cfg) => {
                    tracing::info!("scalability config: loaded {}", path.display());
                    return cfg;
                }
                Err(e) => {
                    tracing::warn!(
                        "scalability config: {} failed to parse ({e}); using embedded default",
                        path.display()
                    );
                    return embedded;
                }
            },
            Err(e) => {
                tracing::warn!(
                    "scalability config: {} unreadable ({e}); using embedded default",
                    path.display()
                );
                return embedded;
            }
        }
    }
    embedded
}

/// Process-wide cache of the loaded config (parsed once at first use). The tier tables are static
/// data, so a single load is correct; the resolver + UI both read this one snapshot.
fn config() -> &'static ScalabilityConfig {
    static CONFIG: std::sync::OnceLock<ScalabilityConfig> = std::sync::OnceLock::new();
    CONFIG.get_or_init(load_config)
}

/// The tier→knob table, sourced from the data-driven config. Med must equal the legacy hardcoded
/// defaults (no-regression), which the RON reproduces exactly. A tier missing from an on-disk
/// override falls back to the embedded default for that tier, so `preset()` is always total.
pub fn preset(q: RenderQuality) -> QualityPreset {
    let cfg = config();
    if let Some(t) = cfg.tier(q) {
        return t.preset;
    }
    // The active on-disk config omits this tier: fall back to the embedded default's entry (which,
    // being `include_str!`d, always contains every tier). This keeps a partial override safe.
    ScalabilityConfig::parse(EMBEDDED_CONFIG)
        .ok()
        .and_then(|e| e.tier(q).map(|t| t.preset))
        .expect("embedded scalability.ron covers every tier (locked by unit test)")
}

/// The gallery's byte-identical legacy configuration, as a preset. The gallery is the path-tracer
/// parity + regression anchor (`af70c1a5…`), so every scalability knob that would otherwise shift
/// its pixels is pinned here to the value it had before the tier system existed: full-res trace
/// (no half-res / divisors), no amortization (relight period 1, gather spp 8), the high
/// path-trace-parity sample counts (gi_spp 8, gi_max_steps 64), and the neutral march / clamp /
/// AO settings. Consumers resolve a knob against `base = if gallery { gallery_preset() } else
/// { preset(tier) }` (see `main.rs`), so the gallery-lock is STRUCTURAL: a newly added tier knob
/// takes its gallery value from this one table and can no longer silently break the anchor by
/// forgetting a per-call-site `if gallery_scene { .. }` (the bug that hit `render_scale` and
/// `reflect_max_roughness`). Fields that no gallery pass reads (e.g. the res divisors, unused while
/// `gi_half_res`/`reflect_half_res` are false, and `ssao`/`ao_res_div`, unused while AO is off)
/// still carry their legacy value so the table is a complete, self-describing snapshot.
pub fn gallery_preset() -> QualityPreset {
    QualityPreset {
        gi_spp: 8,        // path-trace-parity sample count (gallery is the PT reference)
        gi_max_steps: 64, // full bounce march
        reflect_max_steps: 96,
        gi_denoise: true,
        reflect_cache: true,
        surface_cache: false,
        ssr_stochastic: false, // full-res mirror SSR (does not affect the gallery image)
        reflect_max_roughness: 0.5,
        gdf_ao: false, // gallery runs no GDF AO
        ssao: false,   // gallery runs no screen-space AO
        firefly_clamp: true,
        shadow_softness: 0.0,
        shadow_taps: 16,
        cache_relight_period: 1, // every-frame relight (no amortization)
        gi_half_res: false,      // full-res GI trace
        cache_relight_spp: 8,
        reflect_half_res: false,  // full-res reflection trace
        render_scale: 1.0,        // native (no upscale)
        gdf_cone_k: 0.0,          // linear march (no cone LOD)
        gi_res_div: 2,            // unused (gi_half_res=false); legacy value for completeness
        reflect_res_div: 2,       // unused (reflect_half_res=false)
        ao_res_div: 1,            // unused (gdf_ao=false)
        gi_atrous_steps: 2,       // two à-trous iterations (legacy denoise)
        reflect_history_clamp: 0, // off (legacy resolve, no neighbourhood clamp)
        reflect_clamp_gamma: 1.25,
        ssr_history_clamp: 0, // off (byte-identical anchor; SSR feedback clamp is opt-in)
        ssr_clamp_gamma: 1.25,
        gi_temporal_clamp: 1.0, // hard 3x3 GI temporal clamp (legacy byte-identical anchor)
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

// ---------------------------------------------------------------------------
// Scalability groups (organizing layer)
// ---------------------------------------------------------------------------
//
// A reference real-time engine expresses scalability as a small set of quality *groups* — global
// illumination, reflection, shadow, etc. — each carried at a level 0..=3, and a quality "tier" is
// just an assignment of a level to every group (its ini `sg.<Group>Quality` cvars). Our knob set is
// finer-grained (~27 fields in [`QualityPreset`]), so this module keeps that precise table as the
// single source of truth and layers the group concept ON TOP as an ADDITIVE, self-describing map.
//
// IMPORTANT — this layer is DESCRIPTIVE, not authoritative. [`groups`] REFLECTS the levels that
// [`preset`] already encodes; it does NOT feed back into [`preset`] or change any resolved value
// (the byte-identical gallery/Med anchors must hold). A group level is a coarse label ("this tier
// runs GI at level 2"), useful for UI, per-platform reasoning, and a per-group env override a
// caller MAY consult — but the fine-grained `P_*`/`P11_*`/`SHADOW_*`/`RENDER_SCALE` env knobs remain
// the precise controls and WIN over any group level. Wiring group-level -> a concrete knob table
// would change resolved values (it can't reproduce the 27-field presets losslessly from six 0..=3
// integers), so we deliberately keep it a documented mapping rather than a behavioral input.

/// A reference-engine-style scalability GROUP: a named bucket of related quality knobs carried at a
/// level `0..=3`. A [`RenderQuality`] tier is an assignment of a level to every group (see
/// [`groups`]). This is an organizing/UI layer over the precise [`QualityPreset`] table — the fine
/// `P_*`/`P11_*` env knobs remain the exact controls and win over group levels.
///
// `dead_code`: this is an additive organizing/API layer for the scalability system. Its consumers
// (the `main.rs` resolver + the test-scene Scalability UI panel) are owned by other agents and land
// separately, so nothing in the binary references it yet — the unit tests below exercise the whole
// surface. The allow keeps `clippy -D warnings` green without a placeholder call site; remove it
// once `main.rs`/the UI panel consult the groups layer.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScalabilityGroup {
    /// Internal render resolution + TAAU upscale (`render_scale`). Env override: `SG_RESOLUTION`.
    Resolution,
    /// Diffuse global illumination: samples, trace-resolution divisor, à-trous, half-res
    /// (`gi_spp`/`gi_res_div`/`gi_atrous_steps`/`gi_half_res`/`gi_max_steps`). Env: `SG_GI`.
    GlobalIllumination,
    /// Screen-mirror + GGX reflection: trace-resolution divisor, march cap, roughness cutoff,
    /// half-res, history clamp (`reflect_*`, `ssr_stochastic`). Env: `SG_REFLECTION`.
    Reflection,
    /// Ambient occlusion: GDF contact AO, screen-space HBAO-lite, AO trace divisor
    /// (`gdf_ao`/`ssao`/`ao_res_div`). Env: `SG_AO`.
    AmbientOcclusion,
    /// Shadow filtering: PCSS-lite softness + tap count (`shadow_softness`/`shadow_taps`).
    /// Env: `SG_SHADOW`.
    Shadow,
    /// GI surface (mesh-card) cache: multibounce lookup + amortized relight period/spp
    /// (`surface_cache`/`cache_relight_period`/`cache_relight_spp`). Env: `SG_SURFACE_CACHE`.
    SurfaceCache,
}

#[allow(dead_code)] // additive API layer; consumed by tests + the (separately-landing) resolver/UI.
impl ScalabilityGroup {
    /// Every group, in a fixed order. The length of [`groups`]'s return value.
    pub const ALL: [ScalabilityGroup; 6] = [
        ScalabilityGroup::Resolution,
        ScalabilityGroup::GlobalIllumination,
        ScalabilityGroup::Reflection,
        ScalabilityGroup::AmbientOcclusion,
        ScalabilityGroup::Shadow,
        ScalabilityGroup::SurfaceCache,
    ];

    /// The per-group env-override name a caller MAY consult (`SG_GI`, `SG_REFLECTION`, `SG_AO`,
    /// `SG_SHADOW`, `SG_RESOLUTION`, `SG_SURFACE_CACHE`). This names the coarse group lever; the
    /// fine-grained `P_*`/`P11_*`/`SHADOW_*` env knobs remain the precise controls and win over it.
    pub fn env_name(self) -> &'static str {
        match self {
            ScalabilityGroup::Resolution => "SG_RESOLUTION",
            ScalabilityGroup::GlobalIllumination => "SG_GI",
            ScalabilityGroup::Reflection => "SG_REFLECTION",
            ScalabilityGroup::AmbientOcclusion => "SG_AO",
            ScalabilityGroup::Shadow => "SG_SHADOW",
            ScalabilityGroup::SurfaceCache => "SG_SURFACE_CACHE",
        }
    }

    /// A stable short label for logs/UI (`resolution`, `gi`, `reflection`, ...).
    pub fn label(self) -> &'static str {
        match self {
            ScalabilityGroup::Resolution => "resolution",
            ScalabilityGroup::GlobalIllumination => "gi",
            ScalabilityGroup::Reflection => "reflection",
            ScalabilityGroup::AmbientOcclusion => "ao",
            ScalabilityGroup::Shadow => "shadow",
            ScalabilityGroup::SurfaceCache => "surface_cache",
        }
    }

    /// The per-group env override, parsed + clamped to `0..=3`, or `None` when unset / unparseable.
    /// This is an OPTIONAL coarse lever a caller may consult; unset (the default) means "use the
    /// tier's [`groups`] level". Returning the level does NOT reach into [`preset`] — a caller that
    /// honors it would map the level to knobs itself; the fine `P_*` env knobs still win. Kept here
    /// so the group env names live in one place (single source of truth).
    pub fn env_level(self) -> Option<u8> {
        std::env::var(self.env_name())
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .map(|lvl| lvl.min(3))
    }
}

/// The tier -> per-group level table (levels `0..=3`), the group-layer VIEW of [`preset`].
///
/// Returns a level for EVERY [`ScalabilityGroup`] (exhaustive, order = [`ScalabilityGroup::ALL`]),
/// so the system is self-describing and per-platform extensible: a new platform tier declares its
/// coarse profile as six integers here alongside its precise [`QualityPreset`]. The levels are a
/// DESCRIPTIVE summary of what [`preset(tier)`](preset) already encodes — changing them does not
/// change any resolved render value (the fine knobs are the source of truth). Level scale (looser
/// = more expensive): `0` cheapest / `3` reference ceiling.
///
/// How each tier's levels reflect its preset:
/// - **Low**: 2/3 internal res (Resolution 0), spp2 + third-res GI (GI 1), half-res reflection with
///   a short 64-step march (Reflection 0), GDF AO off / SSAO on (AO 1), hard shadows / 8 taps
///   (Shadow 0), no surface cache / long relight period (SurfaceCache 0).
/// - **Med** (the byte-identical no-reg baseline): native res (Resolution 2), spp1 third-res GI
///   (GI 1), half-res 96-step reflection (Reflection 1), GDF AO + SSAO both on (AO 2), hard shadows
///   / 16 taps (Shadow 1), no surface cache / period-40 relight (SurfaceCache 1).
/// - **High**: native res (Resolution 2), spp16 + full-res GI (GI 3), full-res 96-step reflection
///   with a variance history clamp (Reflection 3), GDF AO + SSAO both on (AO 2), aesthetic soft
///   shadows / 16 taps (Shadow 3), multibounce surface cache with every-frame relight
///   (SurfaceCache 3).
/// - **Apple** (platform tier, Med-derived, cost knobs pushed for the fanless iGPU): 0.67 internal
///   res (Resolution 1), spp1 quarter-res single-à-trous GI (GI 0), 1/6-res 56-step reflection
///   (Reflection 0), GDF AO on / SSAO off / half-res AO (AO 1), hard shadows / 16 taps (Shadow 1),
///   no surface cache / period-64 relight (SurfaceCache 0).
pub fn groups(q: RenderQuality) -> [(ScalabilityGroup, u8); 6] {
    let cfg = config();
    if let Some(t) = cfg.tier(q) {
        return t.groups.as_pairs();
    }
    // Partial on-disk override that omits this tier: fall back to the embedded default's levels
    // (which always cover every tier), matching `preset`'s same-tier fallback.
    ScalabilityConfig::parse(EMBEDDED_CONFIG)
        .ok()
        .and_then(|e| e.tier(q).map(|t| t.groups.as_pairs()))
        .expect("embedded scalability.ron covers every tier (locked by unit test)")
}

/// The group level a tier carries for one group (convenience lookup over [`groups`]). Panics never:
/// [`groups`] is exhaustive over [`ScalabilityGroup`], so the group is always present.
#[allow(dead_code)] // additive API layer; consumed by tests + the (separately-landing) resolver/UI.
pub fn group_level(q: RenderQuality, group: ScalabilityGroup) -> u8 {
    groups(q)
        .into_iter()
        .find_map(|(g, lvl)| (g == group).then_some(lvl))
        .expect("groups() is exhaustive over ScalabilityGroup")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every [`RenderQuality`] tier, for exhaustive per-tier assertions. Kept in one place so a new
    /// tier variant forces an obvious edit here (and the compiler flags the missing match arm in
    /// `preset`/`groups`).
    const TIERS: [RenderQuality; 4] = [
        RenderQuality::Low,
        RenderQuality::Med,
        RenderQuality::High,
        RenderQuality::Apple,
    ];

    /// Assert that a preset's every field sits inside the range its consumer clamps it to in
    /// `main.rs` (the `env_override.unwrap_or(base.x).clamp(..)` sites). This is the validated-range
    /// table from the design doc, checked structurally so a future preset edit that lands an
    /// out-of-range default is caught here rather than silently clamped at runtime.
    fn assert_preset_in_range(label: &str, p: &QualityPreset) {
        // Trace-resolution divisors: clamp(1, 16) at the res-div consumer sites.
        assert!(
            (1..=16).contains(&p.gi_res_div),
            "{label}: gi_res_div {} out of 1..=16",
            p.gi_res_div
        );
        assert!(
            (1..=16).contains(&p.reflect_res_div),
            "{label}: reflect_res_div {} out of 1..=16",
            p.reflect_res_div
        );
        assert!(
            (1..=16).contains(&p.ao_res_div),
            "{label}: ao_res_div {} out of 1..=16",
            p.ao_res_div
        );
        // à-trous iterations: clamp(1, 5).
        assert!(
            (1..=5).contains(&p.gi_atrous_steps),
            "{label}: gi_atrous_steps {} out of 1..=5",
            p.gi_atrous_steps
        );
        // Surface-cache relight period: .max(1) — must be >= 1 (0 = divide-by-zero round-robin).
        assert!(
            p.cache_relight_period >= 1,
            "{label}: cache_relight_period {} must be >= 1",
            p.cache_relight_period
        );
        // Sample counts: clamp(1, 256).
        assert!(
            (1..=256).contains(&p.gi_spp),
            "{label}: gi_spp {} out of 1..=256",
            p.gi_spp
        );
        assert!(
            (1..=256).contains(&p.gi_max_steps),
            "{label}: gi_max_steps {} out of 1..=256",
            p.gi_max_steps
        );
        assert!(
            (1..=256).contains(&p.reflect_max_steps),
            "{label}: reflect_max_steps {} out of 1..=256",
            p.reflect_max_steps
        );
        // Cache relight gather spp: a positive per-texel ray count.
        assert!(
            p.cache_relight_spp >= 1,
            "{label}: cache_relight_spp {} must be >= 1",
            p.cache_relight_spp
        );
        // Internal render scale: clamp(0.3333, 1.0) at the render-scale consumer site.
        assert!(
            (0.3333..=1.0).contains(&p.render_scale),
            "{label}: render_scale {} out of 0.3333..=1.0",
            p.render_scale
        );
        // Reflection history-clamp MODE: .min(2) => 0/1/2.
        assert!(
            p.reflect_history_clamp <= 2,
            "{label}: reflect_history_clamp {} out of 0..=2",
            p.reflect_history_clamp
        );
        // SSR-resolve history-clamp MODE: .min(1) => 0/1.
        assert!(
            p.ssr_history_clamp <= 1,
            "{label}: ssr_history_clamp {} out of 0..=1",
            p.ssr_history_clamp
        );
        assert!(
            (0.0..=8.0).contains(&p.ssr_clamp_gamma),
            "{label}: ssr_clamp_gamma {} out of 0..=8",
            p.ssr_clamp_gamma
        );
        // Soft-PCF tap count: clamp(1, 16) (POISSON16 array bound in the shader).
        assert!(
            (1..=16).contains(&p.shadow_taps),
            "{label}: shadow_taps {} out of 1..=16",
            p.shadow_taps
        );
        // Reflection roughness cutoff: a 0..=1 fraction.
        assert!(
            (0.0..=1.0).contains(&p.reflect_max_roughness),
            "{label}: reflect_max_roughness {} out of 0..=1",
            p.reflect_max_roughness
        );
        // Cone-trace LOD slope: clamp(0.0, 1.0).
        assert!(
            (0.0..=1.0).contains(&p.gdf_cone_k),
            "{label}: gdf_cone_k {} out of 0..=1",
            p.gdf_cone_k
        );
        // Shadow softness (penumbra scale): non-negative; clamped at the consumer only for < 0.
        assert!(
            p.shadow_softness >= 0.0,
            "{label}: shadow_softness {} must be >= 0",
            p.shadow_softness
        );
        // Reflection variance-clamp tightness: clamp(0.0, 8.0).
        assert!(
            (0.0..=8.0).contains(&p.reflect_clamp_gamma),
            "{label}: reflect_clamp_gamma {} out of 0..=8",
            p.reflect_clamp_gamma
        );
        // GI temporal-clamp mode/gamma: clamp(0.0, 16.0).
        assert!(
            (0.0..=16.0).contains(&p.gi_temporal_clamp),
            "{label}: gi_temporal_clamp {} out of 0..=16",
            p.gi_temporal_clamp
        );
    }

    /// Every tier's `preset()` resolves within the validated ranges its consumers clamp to.
    #[test]
    fn preset_fields_in_range() {
        for tier in TIERS {
            assert_preset_in_range(tier.label(), &preset(tier));
        }
    }

    /// The gallery preset — which the gallery scene resolves against instead of the active tier —
    /// is also a valid, in-range configuration.
    #[test]
    fn gallery_preset_in_range() {
        assert_preset_in_range("gallery", &gallery_preset());
    }

    /// GUARDRAIL: [`gallery_preset`] equals the byte-identical legacy anchor values, field by field.
    /// The gallery is the path-tracer parity + regression anchor (`af70c1a5…`); this test locks its
    /// resolved config so a future `preset`/tier edit can't silently drift the anchor. If a value
    /// here must change, the gallery sha changes too — that is the whole point of the lock.
    #[test]
    fn gallery_preset_locks_legacy_anchor() {
        let g = gallery_preset();
        assert_eq!(g.gi_spp, 8, "gallery gi_spp (PT-parity sample count)");
        assert_eq!(
            g.gi_max_steps, 64,
            "gallery gi_max_steps (full bounce march)"
        );
        assert_eq!(g.reflect_max_steps, 96, "gallery reflect_max_steps");
        assert!(g.gi_denoise, "gallery gi_denoise");
        assert!(g.reflect_cache, "gallery reflect_cache");
        assert!(!g.surface_cache, "gallery surface_cache");
        assert!(
            !g.ssr_stochastic,
            "gallery ssr_stochastic (full-res mirror)"
        );
        assert_eq!(
            g.reflect_max_roughness, 0.5,
            "gallery reflect_max_roughness"
        );
        assert!(!g.gdf_ao, "gallery gdf_ao (no GDF AO)");
        assert!(!g.ssao, "gallery ssao (no screen-space AO)");
        assert!(g.firefly_clamp, "gallery firefly_clamp");
        assert_eq!(g.shadow_softness, 0.0, "gallery shadow_softness (hard PCF)");
        assert_eq!(g.shadow_taps, 16, "gallery shadow_taps");
        assert_eq!(
            g.cache_relight_period, 1,
            "gallery cache_relight_period (every-frame relight)"
        );
        assert!(!g.gi_half_res, "gallery gi_half_res (full-res trace)");
        assert_eq!(g.cache_relight_spp, 8, "gallery cache_relight_spp");
        assert!(!g.reflect_half_res, "gallery reflect_half_res (full-res)");
        assert_eq!(g.render_scale, 1.0, "gallery render_scale (native)");
        assert_eq!(g.gdf_cone_k, 0.0, "gallery gdf_cone_k (linear march)");
        assert_eq!(g.gi_res_div, 2, "gallery gi_res_div (legacy value)");
        assert_eq!(
            g.reflect_res_div, 2,
            "gallery reflect_res_div (legacy value)"
        );
        assert_eq!(g.ao_res_div, 1, "gallery ao_res_div (legacy value)");
        assert_eq!(g.gi_atrous_steps, 2, "gallery gi_atrous_steps");
        assert_eq!(
            g.reflect_history_clamp, 0,
            "gallery reflect_history_clamp (off)"
        );
        assert_eq!(g.reflect_clamp_gamma, 1.25, "gallery reflect_clamp_gamma");
        assert_eq!(
            g.ssr_history_clamp, 0,
            "gallery ssr_history_clamp (off = byte-identical anchor)"
        );
        assert_eq!(g.ssr_clamp_gamma, 1.25, "gallery ssr_clamp_gamma");
        assert_eq!(
            g.gi_temporal_clamp, 1.0,
            "gallery gi_temporal_clamp (hard 3x3)"
        );
    }

    /// `Med` reproduces the pre-tier hardcoded defaults (the no-regression baseline) for the fields
    /// the design doc calls out. A drift here is a byte-identical-anchor risk for content Med.
    #[test]
    fn med_matches_legacy_defaults() {
        let m = preset(RenderQuality::Med);
        assert_eq!(m.gi_spp, 1, "Med gi_spp");
        assert_eq!(m.render_scale, 1.0, "Med render_scale (native)");
        assert_eq!(m.reflect_max_roughness, 0.5, "Med reflect_max_roughness");
        assert_eq!(m.reflect_max_steps, 96, "Med reflect_max_steps");
        assert!(m.gdf_ao, "Med gdf_ao");
        assert!(m.ssao, "Med ssao (legacy content default)");
    }

    /// The embedded (`include_str!`) config parses and covers every tier. This is the invariant the
    /// data-driven fallback relies on: the binary always has a complete, valid table, so a missing
    /// or corrupt on-disk file degrades to the shipped defaults, never to nothing. A malformed edit
    /// to `config/scalability.ron` is caught here at `cargo test`, before it can ship.
    #[test]
    fn embedded_config_parses_and_covers_every_tier() {
        let cfg =
            ScalabilityConfig::parse(EMBEDDED_CONFIG).expect("embedded scalability.ron must parse");
        for tier in TIERS {
            assert!(
                cfg.tier(tier).is_some(),
                "embedded config missing tier {}",
                tier.label()
            );
        }
        // One entry per tier (no duplicates / strays that would shadow a lookup).
        assert_eq!(
            cfg.tiers.len(),
            TIERS.len(),
            "embedded config must have exactly one entry per tier"
        );
    }

    /// The data-driven presets reproduce the historical hard-coded tables EXACTLY. Spot-checks the
    /// fields most likely to drift on a hand edit of the RON, per tier — the range/`groups`/Med tests
    /// cover the rest structurally. If the RON and this snapshot disagree, one of them is wrong.
    #[test]
    fn data_driven_presets_match_snapshot() {
        let low = preset(RenderQuality::Low);
        assert_eq!(low.gi_spp, 2, "Low gi_spp");
        assert_eq!(low.render_scale, 0.6667, "Low render_scale");
        assert!(!low.reflect_cache, "Low reflect_cache off");
        assert!(low.ssr_stochastic, "Low ssr_stochastic");
        assert_eq!(low.gi_res_div, 3, "Low gi_res_div");

        let high = preset(RenderQuality::High);
        assert_eq!(high.gi_spp, 16, "High gi_spp");
        assert!(high.surface_cache, "High surface_cache on");
        assert_eq!(high.shadow_softness, 0.03, "High shadow_softness");
        assert_eq!(high.reflect_history_clamp, 2, "High reflect_history_clamp");
        assert_eq!(high.gi_res_div, 2, "High gi_res_div");

        let apple = preset(RenderQuality::Apple);
        assert_eq!(apple.render_scale, 0.67, "Apple render_scale");
        assert!(!apple.ssao, "Apple ssao off");
        assert_eq!(apple.reflect_res_div, 6, "Apple reflect_res_div");
        assert_eq!(apple.ao_res_div, 2, "Apple ao_res_div");
        assert_eq!(apple.gi_atrous_steps, 1, "Apple gi_atrous_steps");
        assert_eq!(apple.gi_res_div, 4, "Apple gi_res_div");
    }

    /// The data-driven group levels reproduce the historical hard-coded `groups()` table exactly.
    #[test]
    fn data_driven_groups_match_snapshot() {
        use ScalabilityGroup::*;
        let expect = |q, want: [(ScalabilityGroup, u8); 6]| {
            assert_eq!(groups(q), want, "{} group levels", q.label());
        };
        expect(
            RenderQuality::Low,
            [
                (Resolution, 0),
                (GlobalIllumination, 1),
                (Reflection, 0),
                (AmbientOcclusion, 1),
                (Shadow, 0),
                (SurfaceCache, 0),
            ],
        );
        expect(
            RenderQuality::Med,
            [
                (Resolution, 2),
                (GlobalIllumination, 1),
                (Reflection, 1),
                (AmbientOcclusion, 2),
                (Shadow, 1),
                (SurfaceCache, 1),
            ],
        );
        expect(
            RenderQuality::High,
            [
                (Resolution, 2),
                (GlobalIllumination, 3),
                (Reflection, 3),
                (AmbientOcclusion, 2),
                (Shadow, 3),
                (SurfaceCache, 3),
            ],
        );
        expect(
            RenderQuality::Apple,
            [
                (Resolution, 1),
                (GlobalIllumination, 0),
                (Reflection, 0),
                (AmbientOcclusion, 1),
                (Shadow, 1),
                (SurfaceCache, 0),
            ],
        );
    }

    /// An on-disk override that parses is applied (a valid tweak wins), and one that FAILS to parse
    /// falls back to the embedded default without panicking. Exercises the loader's parse/fallback
    /// contract directly (`load_config`'s file path is env/CWD-dependent, so this drives the
    /// parse+select core via `ScalabilityConfig::parse` on temp-file contents).
    #[test]
    fn on_disk_override_parses_or_falls_back() {
        // A well-formed single-tier override changes only that tier when merged over the embedded
        // base — here we assert the parsed override carries the edited value.
        let good = r#"(tiers: [(
            tier: Med,
            preset: (
                gi_spp: 4, gi_denoise: true, reflect_cache: true, surface_cache: false,
                ssr_stochastic: false, reflect_max_roughness: 0.5, gdf_ao: true, ssao: true,
                firefly_clamp: true, shadow_softness: 0.0, shadow_taps: 16, cache_relight_period: 40,
                gi_half_res: true, cache_relight_spp: 1, gi_max_steps: 24, reflect_max_steps: 96,
                reflect_half_res: true, render_scale: 1.0, gdf_cone_k: 0.02, gi_res_div: 3,
                reflect_res_div: 2, ao_res_div: 1, gi_atrous_steps: 2, reflect_history_clamp: 1,
                reflect_clamp_gamma: 1.25, ssr_history_clamp: 0, ssr_clamp_gamma: 1.25,
                gi_temporal_clamp: 0.0,
            ),
            groups: (resolution: 2, global_illumination: 1, reflection: 1, ambient_occlusion: 2,
                     shadow: 1, surface_cache: 1),
        )])"#;
        let parsed = ScalabilityConfig::parse(good).expect("good override parses");
        assert_eq!(
            parsed.tier(RenderQuality::Med).map(|t| t.preset.gi_spp),
            Some(4),
            "override applies its edited gi_spp"
        );

        // A malformed document must be a parse error (the loader logs + falls back to embedded).
        assert!(
            ScalabilityConfig::parse("(tiers: [ this is not ron").is_err(),
            "malformed RON must fail to parse (so the loader falls back to embedded)"
        );
        // The fallback target — the embedded default — is always valid and total.
        assert!(
            ScalabilityConfig::parse(EMBEDDED_CONFIG).is_ok(),
            "embedded default (the fallback) must always parse"
        );
    }

    /// `env_bool` parsing: `0`/`false`/`off` (case/space-insensitive) => false; any other set value
    /// => true; unset => the tier default (either polarity).
    #[test]
    fn env_bool_parses() {
        // A test-local env name so this can't collide with a real feature flag.
        let name = "DC_TEST_ENV_BOOL_PARSE";
        // Unset => default is returned verbatim (both polarities).
        // SAFETY: single-threaded test on a test-local env name; set/remove are balanced here.
        unsafe { std::env::remove_var(name) };
        assert!(env_bool(name, true), "unset => default true");
        assert!(!env_bool(name, false), "unset => default false");

        for falsy in ["0", "false", "off", "OFF", "  false  ", "False"] {
            // SAFETY: single-threaded test, restored via remove_var before the next case.
            unsafe { std::env::set_var(name, falsy) };
            assert!(
                !env_bool(name, true),
                "{falsy:?} => false (overrides default)"
            );
        }
        for truthy in ["1", "true", "on", "yes", "anything"] {
            unsafe { std::env::set_var(name, truthy) };
            assert!(
                env_bool(name, false),
                "{truthy:?} => true (overrides default)"
            );
        }
        unsafe { std::env::remove_var(name) };
    }

    /// `groups(tier)` returns a level for EVERY [`ScalabilityGroup`] (exhaustive), each within the
    /// `0..=3` range, for every tier. Guards the self-describing invariant that a tier's group
    /// profile covers the whole group set.
    #[test]
    fn groups_cover_every_group() {
        for tier in TIERS {
            let g = groups(tier);
            assert_eq!(
                g.len(),
                ScalabilityGroup::ALL.len(),
                "{}: groups() must have one entry per ScalabilityGroup",
                tier.label()
            );
            for group in ScalabilityGroup::ALL {
                let present = g.iter().any(|(gg, _)| *gg == group);
                assert!(
                    present,
                    "{}: groups() missing {}",
                    tier.label(),
                    group.label()
                );
            }
            for (group, level) in g {
                assert!(
                    level <= 3,
                    "{}: {} level {} out of 0..=3",
                    tier.label(),
                    group.label(),
                    level
                );
                // The convenience lookup agrees with the table.
                assert_eq!(
                    group_level(tier, group),
                    level,
                    "{}: group_level disagrees with groups() for {}",
                    tier.label(),
                    group.label()
                );
            }
        }
    }

    /// Each group's env-override name is the documented `SG_*` string, and `env_level` clamps a set
    /// value to `0..=3` / returns `None` when unset. Confirms the coarse group lever a caller may
    /// consult is wired to the right env names.
    #[test]
    fn group_env_levels_parse_and_clamp() {
        assert_eq!(ScalabilityGroup::Resolution.env_name(), "SG_RESOLUTION");
        assert_eq!(ScalabilityGroup::GlobalIllumination.env_name(), "SG_GI");
        assert_eq!(ScalabilityGroup::Reflection.env_name(), "SG_REFLECTION");
        assert_eq!(ScalabilityGroup::AmbientOcclusion.env_name(), "SG_AO");
        assert_eq!(ScalabilityGroup::Shadow.env_name(), "SG_SHADOW");
        assert_eq!(
            ScalabilityGroup::SurfaceCache.env_name(),
            "SG_SURFACE_CACHE"
        );

        // SAFETY: single-threaded test; each set is followed by a read then a balanced remove.
        let g = ScalabilityGroup::GlobalIllumination;
        unsafe { std::env::remove_var(g.env_name()) };
        assert_eq!(g.env_level(), None, "unset SG_GI => None");
        unsafe { std::env::set_var(g.env_name(), "2") };
        assert_eq!(g.env_level(), Some(2), "SG_GI=2 => Some(2)");
        unsafe { std::env::set_var(g.env_name(), "9") };
        assert_eq!(g.env_level(), Some(3), "SG_GI=9 => clamped to Some(3)");
        unsafe { std::env::set_var(g.env_name(), "not-a-number") };
        assert_eq!(g.env_level(), None, "SG_GI=garbage => None");
        unsafe { std::env::remove_var(g.env_name()) };
    }
}
