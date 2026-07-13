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
static const float SH_C0 = 0.2820947918; // sqrt(1/(4*pi))
static const float SH_C1 = 0.4886025119; // sqrt(3/(4*pi))

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
    uint skyvis_index;    // sampled base of the 4 sky-visibility SH volumes (deferred-parity skylight;
                          // 0xFFFFFFFF = off = the legacy sky-on-miss + skylight-floor path)
    float skyvis_tint;    // neutral leak fraction of the occluded skylight (same value as the deferred)
    float skyvis_min_occ; // min sky-visibility floor (lerp V toward 1; same value as the deferred)
    uint flags;           // bit0: TLAS gather (CACHE_HWRT_SHADOW permutation only) — the indirect
                          // rays trace exact triangles instead of the leak-prone GDF march
    float ao_reach;       // GDF-AO march reach (the SAME GiSystem::ao_params the screen AO uses —
                          // the parity skylight is occluded by the identical formula)
    float ao_strength;    // exponential occlusion strength
    float ao_bias;        // surface offset before the march
    float ao_floor;       // remap floor: deep contacts keep >= this fraction of the ambient
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
        // 2 = mirror-read (reflection feedback): the mirror shows this card DIRECTLY, so it
        // relights on the base (on-screen) budget instead of idling on the HIDDEN_MULT one —
        // an off-screen card a mirror reflects otherwise waits period×8 (~17 s at the Apple
        // tier's period 64 / 30 fps) and shades from the skylight-heavy fallback meanwhile
        // (the "two colour families" a moving mirror exposes). A chrome ball reads a broad
        // slice of the pool, so the boost stops at the BASE period: an accelerated /2 tier
        // measured +1.9 ms relight on the chrome scene for ~1 s faster convergence — not
        // worth it against the 60 fps budget (env-tunable later if a scene demands it).
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
#ifdef CACHE_HWRT_SHADOW
    // Exact triangle sun visibility (the permutation the host selects when the content TLAS
    // exists): the coarse GDF march can close small openings — a courtyard roof aperture reads
    // as solid — which shadows entire sunlit card regions. The mirror then shows a flat grey
    // floor where the deferred (and the path tracer) show a crisp sun shaft: the reflected
    // shadow boundary splits between the two sources. One opaque any-hit ray fixes the
    // visibility at the source; hard edges also match the deferred's default hard PCF closer
    // than the wide 16·h/t GDF penumbra. Same pattern as the hit-lighting shadow ray.
    float sh = 1.0;
    if (ndl > 0.0)
    {
        RayDesc sray;
        sray.Origin = p + n * 0.01;
        sray.Direction = sun_dir;
        sray.TMin = 0.001;
        sray.TMax = pc.params.w;
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
        sq.TraceRayInline(g.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, sray);
        sq.Proceed();
        if (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
            sh = 0.0;
    }
#else
    float sh = ndl > 0.0 ? sun_shadow(p + n * 0.004, sun_dir) : 1.0;
#endif
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
        bool hit = false;
        float3 qhit = float3(0.0, 0.0, 0.0);
#ifdef CACHE_HWRT_SHADOW
        if ((pc.flags & 1u) != 0u)
        {
            // TLAS gather (flags bit0): the indirect ray hits exact triangles. The GDF march
            // leaks through thin geometry (a shadowed floor texel's ray tunnels below the slab
            // and reads the SUNLIT top-side cards) and its budget-exhausted rays vanish — both
            // inflate the gathered bounce; the shadowed-floor cache measured ~50 of bounce where
            // the deferred's volume term reads ~28 and the path tracer ~14. The small fixed
            // origin bias replaces the coarse-field bias (exact triangles need no voxel margin).
            RayDesc gray;
            gray.Origin = p + n * 0.01;
            gray.Direction = dir;
            gray.TMin = 0.001;
            gray.TMax = maxt;
            RayQuery<RAY_FLAG_NONE> gq;
            gq.TraceRayInline(g.tlas, RAY_FLAG_NONE, 0xFF, gray);
            gq.Proceed();
            if (gq.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
            {
                hit = true;
                qhit = gray.Origin + dir * gq.CommittedRayT();
            }
        }
        else
#endif
        {
            float3 o = p + n * bias;
            float tt = bias;
            for (int i = 0; i < 48; i++)
            {
                float3 q = o + dir * tt;
                float d = geo_march(q);
                if (d < 0.0015)
                {
                    hit = true;
                    qhit = q;
                    break;
                }
                // P3 cone-trace LOD: raise the step floor with distance (cone_k=0 => max(d,0.01)
                // legacy).
                tt += max(d, max(0.01, pc.cone_k * tt));
                if (tt > maxt)
                    break;
            }
        }
        if (hit)
        {
            float3 nq = geo_normal(qhit);
            bool cache_hit;
            // Track C: the gather's card lookup goes through the acceleration grid when the
            // host built one (identical results — superset + ascending order; see the helper).
            uint gc8 = pc.gdf_sampled & 0xFFu;
            uint gp8 = (pc.gdf_sampled >> 8) & 0xFFu;
            float3 s = sample_surface_cache_grid(
                qhit, nq, pc.cards_index, pc.cache_pos_index, pc.cache_rad_read,
                pc.num_cards, pc.tile, 0.0, gc8 == 0u ? 0xFFFFFFFFu : gc8 - 1u,
                gp8 == 0u ? 0xFFFFFFFFu : gp8 - 1u, cache_hit);
            // F1 Stage 0 — surface-cache virtualization fallback. When the gather ray hits a surface
            // that carries NO card (a coarse-fallback drawable over the card budget, or an
            // LRU-evicted page once streaming lands), the cache lookup returns float3(0). Adding that
            // zero drops all multibounce contribution from non-resident geometry to BLACK — the real
            // interior lighting hole. Instead approximate the hit's outgoing radiance with the SAME
            // dense-field direct-sun re-light this shader gives the texel's own point
            // (albedo/PI · sun_i·ndl·shadow); the occluded-SH term supplies skylight, exactly as in
            // the texel's own `direct_irr` decomposition, so there is no double count. Content-only
            // (flags bit1 = `gather_fallback`, host-gated off for the gallery ⇒ legacy zero-add ⇒
            // byte-identical anchor). This is also the correct degrade for an evicted page.
            if (!cache_hit && (pc.flags & 2u) != 0u)
            {
                float ndlq = max(dot(nq, sun_dir), 0.0);
                float shq = ndlq > 0.0 ? sun_shadow(qhit + nq * 0.004, sun_dir) : 1.0;
                s = cm_albedo(light_clip(), qhit, 0.7) * (sun_i * ndlq * shq * (1.0 / PI));
            }
            indirect += clamp_firefly(s, pc.gather_firefly);
        }
        if (!hit)
        {
            // Deferred-parity mode: an escaped ray contributes 0 (the skylight is supplied by the
            // occluded SH term below, exactly like gdf_gi's decomposition — no double counting).
            // A ray that merely exhausts its step/distance budget INSIDE the scene also lands here;
            // in legacy mode it wrongly reads as sky, which over-injects the blue skylight on
            // interior cards with long sightlines — the cache-vs-deferred tone split.
            if (pc.skyvis_index == 0xFFFFFFFFu)
            {
                float3 s = procedural_sky(dir, sun_dir, sun_i, pc.sky.x, false) * pc.sky.yzw;
                indirect += clamp_firefly(s, pc.gather_firefly);
            }
            miss_count++;
        }
    }
    indirect /= float(spp);

    if (pc.skyvis_index != 0xFFFFFFFFu && pc.irradiance_index != 0xFFFFFFFFu)
    {
        // Deferred-parity skylight: the SAME SH sky-visibility volumes + bent normal + min-occlusion
        // floor + neutral OcclusionTint leak the lighting pass applies (occlude_sky_diffuse_bent) —
        // one skylight definition for both pipelines, so a cached (reflected) surface reads the sky
        // term the deferred gives the same surface directly. Replaces the per-ray sky-on-miss (whose
        // escape estimate leaks/exhausts to blue indoors) AND the unoccluded skylight floor.
        float3 uvw = (p - pc.aabb_min.xyz) / (pc.aabb_max.xyz - pc.aabb_min.xyz);
        float v = 1.0; // outside the volume = open sky (matches gdf_gi)
        float3 bent = float3(0.0, 0.0, 0.0);
        if (all(uvw >= 0.0) && all(uvw <= 1.0))
        {
            float sc[4];
            [unroll]
            for (int k = 0; k < 4; k++)
                sc[k] = g.volumes[pc.skyvis_index + uint(k)].SampleLevel(g.samp, uvw, 0.0).x;
            v = saturate(sc[0] * SH_C0
                         + (2.0 / 3.0)
                               * (SH_C1 * n.y * sc[1] + SH_C1 * n.z * sc[2] + SH_C1 * n.x * sc[3]));
            // The band-1 SH vector is the bent normal (visibility-weighted average unoccluded
            // direction); basis maps sc1→y, sc2→z, sc3→x (same recovery as gdf_gi).
            float3 bv = float3(sc[3], sc[1], sc[2]);
            if (dot(bv, bv) > 1e-8)
                bent = normalize(bv);
        }
        float3 skyN = n;
        float dot_factor = 1.0;
        if (dot(bent, bent) > 1e-8)
        {
            float w = v; // open -> surface normal, corner -> bent (reference weighting)
            skyN = normalize(lerp(bent, n, w));
            dot_factor = max(lerp(dot(bent, n), 1.0, w), 0.0);
        }
        float3 E = g.cubes[pc.irradiance_index].SampleLevel(g.samp, skyN, 0.0).rgb;
        float vis = lerp(v, 1.0, pc.skyvis_min_occ);
        float lum = dot(E, float3(0.2126, 0.7152, 0.0722));
        // Deferred-parity contact AO: `ambient_ibl` multiplies its diffuse skylight by the marched
        // GDF AO (Stage C2), so the card's skylight must carry the same factor or a cached
        // (reflected) shadowed surface reads ~1/AO brighter than the deferred shades it directly
        // (measured 1.8x on the chromeball's shadowed floor). The exact gdf_ao.slang formula —
        // 5 taps along the normal, exponential falloff, [floor, 1] remap — with the same host
        // params (GiSystem::ao_params). The per-texel material AO map is not captured on cards,
        // so that (mild) factor is the one term intentionally left out.
        float ao = 1.0;
        if (pc.ao_reach > 0.0)
        {
            float3 q0 = p + n * pc.ao_bias;
            float occ = 0.0;
            float sca = 1.0;
            [unroll]
            for (int k = 0; k < 5; k++)
            {
                float h = 0.001 + pc.ao_reach * (float(k) + 1.0) / 5.0;
                float3 q = q0 + n * h;
                float d = min(geo_inside(q), q.y - pc.aabb_min.w);
                occ += (h - d) * sca;
                sca *= 0.92;
            }
            ao = exp(-pc.ao_strength * max(occ, 0.0));
            ao = pc.ao_floor + (1.0 - pc.ao_floor) * ao;
        }
        indirect += (E * (vis * dot_factor) + (pc.skyvis_tint * lum) * (1.0 - vis)) * ao;
    }
    // Legacy skylight-visibility FLOOR (matches the old deferred `occlude_sky_diffuse` MinOcclusion):
    // an enclosed interior face whose gather rays rarely escape still receives a floor of the sky
    // irradiance, so a REFLECTED (or cached) shadowed surface reads its true skylit grey instead of
    // crushing to black. The gather already contributes ~E·(escaped/spp); top it up to E·floor when
    // the estimated sky visibility falls below the floor. floor = 0 (gallery) → exact byte-identical.
    float sky_floor = pc.params.x;
    if (pc.skyvis_index == 0xFFFFFFFFu && sky_floor > 0.0 && pc.irradiance_index != 0xFFFFFFFFu)
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
