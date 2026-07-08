// Phase 11 Stage C8b2 — surface-cache lighting (continuous, multibounce).
//
// One thread per captured cache texel. Re-lights the texel's surface point (captured in
// C8b1) every frame: direct sun (GDF soft-shadow) + sky fill + an indirect gather that
// casts cosine-hemisphere rays into the scene GDF and, at each object hit, looks up the
// *previous* frame's surface cache radiance. Reading last frame's cache makes bounces
// accumulate (multibounce) over frames; an EMA temporal blend damps the gather noise.
// The result (outgoing radiance) is written to the ping-pong write buffer, which next
// frame's gather — and the C8b3 GI / reflection consumers — read.

#include "bindless.slang"
#include "clipmap.slang"
#include "surface_cache.slang"
#include "sky_common.slang"

static const float PI = 3.14159265359;

struct CacheLightPush
{
    uint cards_index;
    uint cache_pos_index;
    uint cache_alb_index;
    uint cache_rad_read;  // previous frame's radiance (gather + temporal source)
    uint cache_rad_write; // this frame's radiance out
    uint gdf_sampled; // REPURPOSED (never read as a volume): Track C card grid for the gather —
                      // (cells+1) | (pool+1)<<8; 0 = no grid (legacy full card scan).
    uint num_cards;
    uint tile; // bits 0..7 tile edge; bits 8..15 C2a layout index + 1 (0 = uniform legacy)
    uint num_texels;
    uint spp;             // indirect gather rays / texel
    uint frame;           // RNG decorrelation
    uint reset;           // 1 = ignore the read buffer (first frame / re-alloc)
    float4 sun;           // xyz direction toward the sun, w = intensity
    float4 aabb_min;      // xyz GDF world min, w = ground plane Y
    float4 aabb_max;      // xyz GDF world max, w = sample distance clamp
    float4 params;        // x = skylight floor (min sky-visibility; 0 = off = byte-identical), y temporal
                          // alpha, z surface bias, w ray max
    uint4 clip;           // Stage B clipmap: x = descriptor index, y = level count
    float cone_k;         // P3: cone-trace LOD slope (0 = legacy linear march = byte-identical)
    uint conv_buf;        // A2-fix: relight InterlockedAdds the per-channel EMA step into this host-
                          // visible buffer ([0]=sum_fp, [4]=count); the host divides to the MEAN step and
                          // freezes the relight ONLY on MEASURED convergence (a general engine budgets by
                          // priority rather than a frame-count freeze). 0xFFFFFFFF = disabled.
    uint irradiance_index; // IBL diffuse-irradiance cube (skylight-floor fill; 0xFFFFFFFF = absent = off)
    float gather_firefly;  // per-sample firefly clamp for the indirect gather (max radiance, physical
                           // units; 0 = off = byte-identical). Reference-engine LumenRadiosity clamps
                           // each hemisphere sample to MaxRayIntensity·OneOverPreExposure at the source.
    float4 sky;           // x = sky gain, yzw = white balance (matches the path tracer's sky)
};

[[vk::push_constant]]
ConstantBuffer<CacheLightPush> pc : register(b0, space0);

ClipMap light_clip()
{
    ClipMap c;
    c.desc = pc.clip.x;
    c.count = pc.clip.y;
    return c;
}

uint pcg(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}
float rnd(inout uint s) { return float(pcg(s) & 0xFFFFFFu) / float(0x1000000); }

// Reference-engine (LumenRadiosity) per-sample firefly clamp: cap the max channel to `maxr`
// (physical radiance), scaling the whole sample so the hue is preserved. A single bright gather
// sample can otherwise dominate the low-spp mean and — since the ray set varies per frame — pop as a
// moving sparkle that keeps the relight's EMA step above the convergence-freeze epsilon. `maxr <= 0`
// disables it (byte-identical). Applied at the SOURCE, before the mean + temporal EMA.
float3 clamp_firefly(float3 v, float maxr)
{
    if (maxr <= 0.0)
        return v;
    float m = max(v.x, max(v.y, v.z));
    return m > maxr ? v * (maxr / m) : v;
}
void onb(float3 n, out float3 t, out float3 b)
{
    t = normalize(abs(n.x) > 0.9 ? cross(n, float3(0, 1, 0)) : cross(n, float3(1, 0, 0)));
    b = cross(n, t);
}
float3 cosine_hemisphere(float3 n, inout uint rng)
{
    float u1 = rnd(rng), u2 = rnd(rng);
    float r = sqrt(u1), phi = 2.0 * PI * u2;
    float3 t, b;
    onb(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}

// Object-only geometry field (no ground union — cards are object surfaces). Single level
// ⇒ identical to the legacy single-volume sample.
float geo_inside(float3 p)
{
    return cm_geo_inside(light_clip(), p, pc.aabb_max.w);
}
float geo_march(float3 p)
{
    return cm_geo_march(light_clip(), p, pc.aabb_max.w);
}
float3 geo_normal(float3 p)
{
    const float e = 0.012;
    float2 k = float2(1.0, -1.0);
    float3 grad = k.xyy * geo_inside(p + k.xyy * e) + k.yyx * geo_inside(p + k.yyx * e)
                  + k.yxy * geo_inside(p + k.yxy * e) + k.xxx * geo_inside(p + k.xxx * e);
    return safe_normalize(grad, float3(0.0, 1.0, 0.0));
}

// Soft penumbra shadow toward the sun (object occluders only).
float sun_shadow(float3 origin, float3 dir)
{
    float res = 1.0, t = 0.02, maxt = pc.params.w;
    for (int i = 0; i < 48; i++)
    {
        float h = geo_inside(origin + dir * t);
        if (h < 0.002)
            return 0.0;
        res = min(res, 16.0 * h / t);
        // P3 cone-trace LOD: widen the max step with distance (cone_k=0 => clamp(.,0.01,0.2) legacy).
        t += clamp(h, 0.01, max(0.2, pc.cone_k * t));
        if (t > maxt)
            break;
    }
    return saturate(res);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lightMain(uint3 tid: SV_DispatchThreadID)
{
    uint t = tid.x;
    if (t >= pc.num_texels)
        return;

    // Stage D2 (Sponza 60fps) — amortized relight budget. Re-lighting all `num_cards*tile^2`
    // texels every frame (the legacy path) is the dominant Sponza cost. Instead refresh only a
    // round-robin slice of cards each frame and persist the rest, since the cache is a temporally
    // accumulated, mostly-static signal (EMA below). This is 레퍼런스 엔진 SW GI's "surface cache update
    // budget": a fixed fraction of cards relit per frame, the others keep last frame's radiance.
    //   Ref: D. Wright et al. (레퍼런스 엔진 팀), "레퍼런스 SW-RT GI: Real-time Global Illumination in 레퍼런스 엔진",
    //        SIGGRAPH 2022 Advances in Real-Time Rendering.
    // `relight_period == 1` ⇒ every card every frame ⇒ byte-identical to the legacy path (the
    // gallery regression anchor passes period 1). `reset` always relights all cards (first frame).
    uint relight_period = max(pc.clip.z, 1u);
    // C2a: variable-tile texel→card mapping (sentinel = the exact legacy `t / tile²`).
    uint lay8 = (pc.tile >> 8) & 0xFFu;
    uint sc_layout = lay8 == 0u ? 0xFFFFFFFFu : lay8 - 1u;
    uint card = layout_find_card(sc_layout, t, max(pc.tile & 0xFFu, 1u), pc.num_cards);
    // Stage D2b: camera-visibility priority. On-screen cards relight at the base budget period;
    // off-screen cards relight HIDDEN_MULT× less often (most of a dense scene's cards are out of
    // any one view, and they don't affect the current image — 레퍼런스 엔진 SW GI's visibility/staleness
    // priority). `clip.w == 0xFFFFFFFF` ⇒ no feedback ⇒ uniform period (Stage D2 behavior).
    const uint HIDDEN_MULT = 8u;
    uint period = relight_period;
    if (pc.clip.w != 0xFFFFFFFFu)
    {
        uint vis = g.storage_buffers[pc.clip.w].Load(card * 4u);
        period = (vis != 0u) ? relight_period : relight_period * HIDDEN_MULT;
    }
    bool selected = (pc.reset != 0u) || ((card % period) == (pc.frame % period));
    if (!selected)
    {
        // Persist: carry last frame's radiance into the write (ping-pong) buffer so the GI /
        // reflection consumers always read a fully-populated atlas; only the heavy march is skipped.
        g.storage_buffers[pc.cache_rad_write].Store4(
            t * 16u, g.storage_buffers[pc.cache_rad_read].Load4(t * 16u));
        return;
    }

    float4 pos = asfloat(g.storage_buffers[pc.cache_pos_index].Load4(t * 16u));
    if (pos.w < 0.5)
    {
        g.storage_buffers[pc.cache_rad_write].Store4(t * 16u, asuint(float4(0, 0, 0, 0)));
        return;
    }
    float3 p = pos.xyz;
    float3 albedo = asfloat(g.storage_buffers[pc.cache_alb_index].Load3(t * 16u));
    float3 n = geo_normal(p);

    float3 sun_dir = normalize(pc.sun.xyz);
    float sun_i = pc.sun.w;
    float ndl = max(dot(n, sun_dir), 0.0);
    float sh = ndl > 0.0 ? sun_shadow(p + n * 0.004, sun_dir) : 1.0;
    // Direct sun only (the skylight now comes from the indirect gather's sky-on-miss below, matching
    // the reference: a per-ray OCCLUDED skylight instead of the old unoccluded `sky_fill` ambient).
    float3 direct_irr = float3(sun_i * ndl * sh, sun_i * ndl * sh, sun_i * ndl * sh);

    // Indirect gather: cosine rays into the GDF; at an object hit read last frame's cache, on a MISS
    // add the sky (same procedural model as the path tracer, sun disk omitted) so the cache seeds
    // the skylight from the first frame — it converges from a lit state, not from black, and the
    // occluded skylight (only rays that escape see the sky) matches the ground truth.
    float3 indirect = float3(0, 0, 0);
    uint spp = max(pc.spp, 1u);
    uint miss_count = 0u; // gather rays that escaped to sky = this texel's sky visibility estimate
    // CONVERGE mode (params.y < 0 ⇒ K = -params.y): the legacy frame-varying seed injects FRESH white
    // noise into every relight, and a fixed-alpha EMA of white noise is an AR(1) process with a
    // permanent variance floor α/(2-α)·σ² — the cache NEVER settles. The reference engine keeps
    // accumulating CONTINUOUSLY (radiosity never stops) but converges statistically: its per-update
    // jitter cycles over a DETERMINISTIC period equal to the accumulation window
    // (Hammersley(TemporalIndex % MaxFramesAccumulated)) and it blends with a running mean
    // α = 1/(1+N) saturating at that window (NumFramesAccumulated atlas). Here: the jitter index is
    // this texel's RELIGHT ordinal (frame/period — one step per relight, regardless of amortization
    // or the off-screen HIDDEN_MULT) modulo K, so K distinct stratified ray sets cycle forever; the
    // running mean below averages the whole cycle exactly, after which the input is PERIODIC and the
    // equilibrium is a bounded ripple ≈ σ/(1+K) — continuous accumulation, no freeze required, and
    // lighting changes keep being tracked (a large change re-arms via the host epoch reset).
    // params.y > 0 (gallery / legacy) keeps the frame-varying seed = byte-identical.
    bool converge = pc.params.y < 0.0;
    uint conv_k = converge ? max(uint(-pc.params.y + 0.5), 1u) : 1u;
    uint jitter = converge ? ((pc.frame / period) % conv_k) : pc.frame;
    uint seed = (t * 9277u + jitter * 26699u) | 1u;
    float bias = pc.params.z;
    float maxt = pc.params.w;
    for (uint s = 0; s < spp; s++)
    {
        float3 dir = cosine_hemisphere(n, seed);
        float3 o = p + n * bias;
        float tt = bias;
        bool hit = false;
        for (int i = 0; i < 48; i++)
        {
            float3 q = o + dir * tt;
            float d = geo_march(q);
            if (d < 0.0015)
            {
                hit = true;
                bool cache_hit;
                // Track C: the gather's card lookup goes through the acceleration grid when the
                // host built one (identical results — superset + ascending order; see the helper).
                uint gc8 = pc.gdf_sampled & 0xFFu;
                uint gp8 = (pc.gdf_sampled >> 8) & 0xFFu;
                float3 s = sample_surface_cache_grid(
                    q, geo_normal(q), pc.cards_index, pc.cache_pos_index, pc.cache_rad_read,
                    pc.num_cards, pc.tile, 0.0, gc8 == 0u ? 0xFFFFFFFFu : gc8 - 1u,
                    gp8 == 0u ? 0xFFFFFFFFu : gp8 - 1u, cache_hit);
                indirect += clamp_firefly(s, pc.gather_firefly);
                break;
            }
            // P3 cone-trace LOD: raise the step floor with distance (cone_k=0 => max(d,0.01) legacy).
            tt += max(d, max(0.01, pc.cone_k * tt));
            if (tt > maxt)
                break;
        }
        if (!hit)
        {
            float3 s = procedural_sky(dir, sun_dir, sun_i, pc.sky.x, false) * pc.sky.yzw;
            indirect += clamp_firefly(s, pc.gather_firefly);
            miss_count++;
        }
    }
    indirect /= float(spp);

    // Skylight-visibility FLOOR (matches the deferred pass's `occlude_sky_diffuse` MinOcclusion):
    // an enclosed interior face whose gather rays rarely escape still receives a floor of the sky
    // irradiance, so a REFLECTED (or cached) shadowed surface reads its true skylit grey instead of
    // crushing to black. The gather already contributes ~E·(escaped/spp); top it up to E·floor when
    // the estimated sky visibility falls below the floor. floor = 0 (gallery) → exact byte-identical.
    float sky_floor = pc.params.x;
    if (sky_floor > 0.0 && pc.irradiance_index != 0xFFFFFFFFu)
    {
        float esc = float(miss_count) / float(spp);
        float3 E = g.cubes[pc.irradiance_index].SampleLevel(g.samp, n, 0.0).rgb;
        indirect += E * max(sky_floor - esc, 0.0);
    }

    // Lambertian: outgoing radiance = albedo/PI * irradiance. The direct term is an
    // irradiance, so it needs the 1/PI (matching the deferred direct + the GI fix); the
    // indirect gather is already cosine-weighted mean radiance (its PI cancels), so it
    // stays as-is. Omitting the direct 1/PI made cached (reflected) surfaces ~PI too bright.
    float3 radiance = albedo * (direct_irr * (1.0 / PI) + indirect);

    // Temporal blend (skip on reset). Legacy: fixed-alpha EMA (params.y). CONVERGE mode: a running
    // mean α = 1/(1+N) with the per-texel relight counter N carried in the radiance `.w` (unread by
    // every consumer — mipgen validity comes from `src_pos.w` and sampling uses Load3). N saturates at
    // K, matching the K-cycle jitter above: the first K relights average the K stratified ray sets
    // EXACTLY (running mean), after which the input repeats periodically and the blend is a bounded
    // ripple the freeze latch measures below its epsilon (reference NumFramesAccumulated atlas).
    float n_count = 1.0;
    if (pc.reset == 0u)
    {
        float4 prev4 = asfloat(g.storage_buffers[pc.cache_rad_read].Load4(t * 16u));
        float3 prev = make_finite(prev4.rgb);
        float alpha = pc.params.y;
        if (converge)
        {
            float n_prev = clamp(prev4.w, 0.0, float(conv_k));
            alpha = 1.0 / (1.0 + n_prev);
            n_count = min(n_prev + 1.0, float(conv_k));
        }
        radiance = lerp(prev, radiance, alpha);
        // A2-fix convergence probe: the MEAN per-channel EMA step over the relit texels (buffer[0] =
        // sum in 1e4 fixed-point, buffer[1] = count). Mean, not max — a single stochastic-gather
        // outlier keeps a max pinned high forever, so max never reports convergence; the mean tracks
        // the whole cache settling. InterlockedAdd is order-independent (=> DX≡VK). The host divides
        // sum/count to get the mean radiance step and freezes only once it falls below its epsilon.
        if (pc.conv_buf != 0xFFFFFFFFu)
        {
            float3 dv = abs(radiance - prev);
            float d = max(dv.x, max(dv.y, dv.z));
            // CONVERGE mode: report the step RELATIVE to the texel's radiance. The absolute step is
            // in physical units (a 100k-lux sun ⇒ cache values in the hundreds), so the host epsilon
            // (0.02) was unreachable by construction — the freeze latch never armed on any physically
            // lit content scene. Relative, eps 0.02 = "2% change", scale-invariant. Legacy keeps the
            // absolute step (its eps was tuned for the gallery's unit-scale lighting).
            if (converge)
                d /= max(max(prev.x, max(prev.y, prev.z)), 1e-3);
            g.storage_buffers[pc.conv_buf].InterlockedAdd(0u, uint(min(d, 1e5) * 1e4));
            g.storage_buffers[pc.conv_buf].InterlockedAdd(4u, 1u);
        }
    }
    // Sanitize before storing: a NaN here would spread through the cache's gather + EMA
    // every frame (the "black creeps in" failure on complex scenes). `.w` = the CONVERGE-mode
    // relight counter N (legacy stores its constant 1.0 = byte-identical).
    g.storage_buffers[pc.cache_rad_write].Store4(
        t * 16u, asuint(float4(make_finite(radiance), converge ? n_count : 1.0)));
}
