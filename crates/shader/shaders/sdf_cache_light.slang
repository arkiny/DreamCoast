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
#include "surface_cache.slang"

static const float PI = 3.14159265359;

struct CacheLightPush
{
    uint cards_index;
    uint cache_pos_index;
    uint cache_alb_index;
    uint cache_rad_read;  // previous frame's radiance (gather + temporal source)
    uint cache_rad_write; // this frame's radiance out
    uint gdf_sampled;
    uint num_cards;
    uint tile;
    uint num_texels;
    uint spp;             // indirect gather rays / texel
    uint frame;           // RNG decorrelation
    uint reset;           // 1 = ignore the read buffer (first frame / re-alloc)
    float4 sun;           // xyz direction toward the sun, w = intensity
    float4 aabb_min;      // xyz GDF world min, w = ground plane Y
    float4 aabb_max;      // xyz GDF world max, w = sample distance clamp
    float4 params;        // x sky_fill, y temporal alpha, z surface bias, w ray max dist
};

[[vk::push_constant]]
ConstantBuffer<CacheLightPush> pc : register(b0, space0);

#define GDF_MIN (pc.aabb_min.xyz)
#define GDF_MAX (pc.aabb_max.xyz)

uint pcg(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}
float rnd(inout uint s) { return float(pcg(s) & 0xFFFFFFu) / float(0x1000000); }
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

// Object-only geometry field (no ground union — cards are object surfaces).
float geo_inside(float3 p)
{
    float3 uvw = (p - GDF_MIN) / (GDF_MAX - GDF_MIN);
    if (any(uvw < 0.0) || any(uvw > 1.0))
        return 1e3;
    return min(g.volumes[pc.gdf_sampled].SampleLevel(g.samp, uvw, 0.0).x, pc.aabb_max.w);
}
float geo_march(float3 p)
{
    float3 uvw = (p - GDF_MIN) / (GDF_MAX - GDF_MIN);
    if (any(uvw < 0.0) || any(uvw > 1.0))
    {
        float3 q = abs(p - 0.5 * (GDF_MIN + GDF_MAX)) - 0.5 * (GDF_MAX - GDF_MIN);
        float box = length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
        return max(box, 0.003);
    }
    return min(g.volumes[pc.gdf_sampled].SampleLevel(g.samp, uvw, 0.0).x, pc.aabb_max.w);
}
float3 geo_normal(float3 p)
{
    const float e = 0.012;
    float2 k = float2(1.0, -1.0);
    return normalize(k.xyy * geo_inside(p + k.xyy * e) + k.yyx * geo_inside(p + k.yyx * e)
                     + k.yxy * geo_inside(p + k.yxy * e) + k.xxx * geo_inside(p + k.xxx * e));
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
        t += clamp(h, 0.01, 0.2);
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
    float sky_fill = pc.params.x * (0.5 + 0.5 * n.y);
    float3 direct_irr = float3(sun_i * ndl * sh + sky_fill, sun_i * ndl * sh + sky_fill,
                               sun_i * ndl * sh + sky_fill);

    // Indirect gather: cosine rays into the GDF; at an object hit read last frame's cache.
    float3 indirect = float3(0, 0, 0);
    uint spp = max(pc.spp, 1u);
    uint seed = (t * 9277u + pc.frame * 26699u) | 1u;
    float bias = pc.params.z;
    float maxt = pc.params.w;
    for (uint s = 0; s < spp; s++)
    {
        float3 dir = cosine_hemisphere(n, seed);
        float3 o = p + n * bias;
        float tt = bias;
        for (int i = 0; i < 48; i++)
        {
            float3 q = o + dir * tt;
            float d = geo_march(q);
            if (d < 0.0015)
            {
                bool cache_hit;
                indirect += sample_surface_cache(q, geo_normal(q), pc.cards_index,
                                                 pc.cache_pos_index, pc.cache_rad_read,
                                                 pc.num_cards, pc.tile, cache_hit);
                break;
            }
            tt += max(d, 0.01);
            if (tt > maxt)
                break;
        }
    }
    indirect /= float(spp);

    // Lambertian: outgoing radiance = albedo/PI * irradiance. The direct term is an
    // irradiance, so it needs the 1/PI (matching the deferred direct + the GI fix); the
    // indirect gather is already cosine-weighted mean radiance (its PI cancels), so it
    // stays as-is. Omitting the direct 1/PI made cached (reflected) surfaces ~PI too bright.
    float3 radiance = albedo * (direct_irr * (1.0 / PI) + indirect);

    // Temporal EMA (skip on reset) — static cache converges; damps gather noise.
    if (pc.reset == 0u)
    {
        float3 prev = asfloat(g.storage_buffers[pc.cache_rad_read].Load3(t * 16u));
        radiance = lerp(prev, radiance, pc.params.y);
    }
    g.storage_buffers[pc.cache_rad_write].Store4(t * 16u, asuint(float4(radiance, 1.0)));
}
