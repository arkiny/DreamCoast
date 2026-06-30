// Shared 1-bounce diffuse tracer against the global distance field (clipmap).
//
// One definition of "march a bounce ray into the scene GDF and return the incoming
// radiance along it" for every consumer: the full-screen per-pixel GI pass
// (`gdf_gi.slang`) and the screen-space radiance probes (`screen_probe_trace.slang`).
// The tracer is parameterized by a `BounceScene` value (built by each caller from its
// own push constants) instead of reading a fixed global `pc`, so the two passes can have
// different push layouts while sharing ONE tracer — no duplicated march/shade/shadow
// code to drift. Behaviour is identical to the prior in-file `trace_bounce` (the gallery
// per-pixel path is byte-identical: a pure code-motion + parameter-passing refactor).
//
// Include order requirement (the includer must have already pulled these in, matching the
// existing gdf_gi.slang include block): bindless.slang, clipmap.slang, surface_cache.slang.

static const float BOUNCE_PI = 3.14159265359;

// Everything the bounce tracer needs about the scene, gathered from the caller's push.
struct BounceScene
{
    uint clip_desc;     // clipmap descriptor storage-buffer index
    uint clip_count;    // clipmap level count (1 = single volume = legacy)
    float sample_clamp; // GDF sample distance clamp (was aabb_max.w)
    float ground_y;     // analytic ground plane height (was aabb_min.w)
    float3 sun_dir;     // direction toward the sun (need not be normalized)
    float sun_i;        // sun intensity
    float ray_max;      // bounce-ray max march distance
    float bias;         // surface start bias along the ray
    float sky_term;     // small hemispheric sky fill at the bounce hit
    float albedo_fallback; // constant albedo when no per-voxel colour volume is bound
    uint4 cache;        // surface-cache lookup: cards, cache_pos, cache_rad, num_cards
    uint cache_tile;    // surface-cache atlas tile edge
    float3 ground_albedo; // analytic ground material (floor bounce hits)
    float cone_k;       // cone-trace LOD slope (0 = legacy linear march)
    uint max_steps;     // bounce-ray march step cap (64 = legacy / gallery)
};

// --- RNG + cosine-hemisphere sampling (mirrors rt_common.slang so the noise is identical
// across backends and consistent with the path tracer). ---
uint bounce_pcg(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}
float bounce_rnd(inout uint s)
{
    return float(bounce_pcg(s) & 0xFFFFFFu) / float(0x1000000);
}
void bounce_onb(float3 n, out float3 t, out float3 b)
{
    t = normalize(abs(n.x) > 0.9 ? cross(n, float3(0, 1, 0)) : cross(n, float3(1, 0, 0)));
    b = cross(n, t);
}
float3 bounce_cosine_hemisphere(float3 n, inout uint rng)
{
    float u1 = bounce_rnd(rng);
    float u2 = bounce_rnd(rng);
    float r = sqrt(u1);
    float phi = 2.0 * BOUNCE_PI * u2;
    float3 t, b;
    bounce_onb(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}

// --- GDF field via the clipmap (single level => identical to the legacy single volume +
// ground union). ---
ClipMap bs_clip(BounceScene bs)
{
    ClipMap c;
    c.desc = bs.clip_desc;
    c.count = bs.clip_count;
    return c;
}

float bs_geo_inside(BounceScene bs, float3 p)
{
    return cm_geo_inside(bs_clip(bs), p, bs.sample_clamp);
}

float bs_geo_march(BounceScene bs, float3 p)
{
    return cm_geo_march(bs_clip(bs), p, bs.sample_clamp);
}

float bs_scene_march(BounceScene bs, float3 p)
{
    return min(bs_geo_march(bs, p), p.y - bs.ground_y);
}
float bs_scene_occ(BounceScene bs, float3 p)
{
    return min(bs_geo_inside(bs, p), p.y - bs.ground_y);
}

// Nearest-triangle albedo at a GDF hit (trilinear from the colour volumes), or the
// constant fallback when no albedo volume is bound.
float3 bs_albedo_at(BounceScene bs, float3 p)
{
    return cm_albedo(bs_clip(bs), p, bs.albedo_fallback);
}

float3 bs_scene_normal(BounceScene bs, float3 p)
{
    const float e = 0.012;
    float2 k = float2(1.0, -1.0);
    float3 grad = k.xyy * bs_scene_occ(bs, p + k.xyy * e) + k.yyx * bs_scene_occ(bs, p + k.yyx * e)
                  + k.yxy * bs_scene_occ(bs, p + k.yxy * e) + k.xxx * bs_scene_occ(bs, p + k.xxx * e);
    return safe_normalize(grad, float3(0.0, 1.0, 0.0));
}

// Cheap penumbra shadow toward the sun for the bounce hit (fewer steps than the primary
// trace's — indirect shadowing is low-frequency and gets denoised later).
float bs_gi_shadow(BounceScene bs, float3 origin, float3 dir)
{
    float res = 1.0;
    float t = 0.02;
    float maxt = bs.ray_max;
    for (int i = 0; i < 48; i++)
    {
        float h = bs_scene_occ(bs, origin + dir * t);
        if (h < 0.002)
            return 0.0;
        res = min(res, 16.0 * h / t);
        // cone-trace LOD: widen the max step with distance (cone_k=0 => clamp(.,0.01,0.2) legacy).
        t += clamp(h, 0.01, max(0.2, bs.cone_k * t));
        if (t > maxt)
            break;
    }
    return saturate(res);
}

// Sphere-trace a bounce ray; returns the incoming radiance along it (0 if it escapes).
float3 bs_trace_bounce(BounceScene bs, float3 origin, float3 dir)
{
    int max_steps = int(max(bs.max_steps, 1u));
    const float HIT_EPS = 0.001;
    float t = bs.bias; // start past the surface bias
    float maxt = bs.ray_max;
    for (int i = 0; i < max_steps; i++)
    {
        float3 p = origin + dir * t;
        float d = bs_scene_march(bs, p);
        if (d < HIT_EPS)
        {
            float3 hn = bs_scene_normal(bs, p);
            // When the surface cache is bound, the incoming radiance along this ray is the hit
            // surface's cached (multibounce) radiance — a direct lookup, no per-ray re-light. On a
            // cache MISS (e.g. the analytic ground) fall back to the per-ray colored re-light.
            if (bs.cache.x != 0xFFFFFFFFu)
            {
                bool cached = false;
                float3 r = sample_surface_cache(p, hn, bs.cache.x, bs.cache.y, bs.cache.z,
                                                bs.cache.w, bs.cache_tile, cached);
                if (cached)
                    return r;
            }
            float3 sun_dir = normalize(bs.sun_dir);
            float ndl = max(dot(hn, sun_dir), 0.0);
            float sh = ndl > 0.0 ? bs_gi_shadow(bs, p + hn * 0.004, sun_dir) : 1.0;
            // Colored re-light: the hit's real surface albedo × sun (soft-shadowed) + a small
            // hemispheric sky fill so shadowed surfaces still bounce a little. The analytic ground
            // is not in the albedo volume, so use the ground's own albedo for a ground hit.
            bool is_ground = (p.y - bs.ground_y) < bs_geo_inside(bs, p);
            float3 hit_albedo = is_ground ? bs.ground_albedo : bs_albedo_at(bs, p);
            // Lambertian: the bounce surface re-emits albedo/PI * irradiance (same 1/PI as the
            // deferred direct term).
            float sky_fill = bs.sky_term * (0.5 + 0.5 * hn.y);
            return hit_albedo * (bs.sun_i * ndl * sh + sky_fill) * (1.0 / BOUNCE_PI);
        }
        // cone-trace LOD: raise the step floor with distance (cone_k=0 => max(d,0.01) legacy).
        t += max(d, max(0.01, bs.cone_k * t));
        if (t > maxt)
            break;
    }
    return float3(0.0, 0.0, 0.0); // escaped to sky -> IBL already supplies it
}
