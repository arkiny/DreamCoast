// Phase 11 Stage C3 — stochastic 1-bounce diffuse GI against the global distance field.
//
// A full-screen compute pass: for every shaded G-buffer pixel it reconstructs the
// world surface point from the depth buffer (object-space position MRT can't be used,
// see gdf_ao.slang) and the world normal, then casts `spp` cosine-weighted hemisphere
// rays into the world-space scene GDF (C1). Each ray that hits geometry/ground is
// shaded at the hit (design fork A: gradient normal + sun w/ soft shadow + a small sky
// term, a single constant albedo — the GDF carries no material; colored bleed needs a
// surface cache, a later refinement). The cosine-weighted Monte-Carlo estimator makes
// the per-sample contribution simply the incoming radiance, so the output is the mean
// incoming *radiance* (indirect irradiance E); the deferred lighting pass multiplies
// it by the surface albedo (× 1-metallic) and adds it to the ambient term.
//
// Rays that escape to the sky return 0, NOT the sky color: the existing IBL diffuse
// already supplies the (unoccluded) sky, the C2 GDF-AO removes the sky that geometry
// occludes, and this pass adds back the light those occluders bounce. So the ambient
// decomposition is IBL·AO (sky) + albedo·E_gi (scene bounce) with no double counting.
//
// Output is raw radiance (pre-exposure), matching the deferred lighting's direct term;
// the lighting pass applies the camera exposure to the whole ambient + direct sum.

#include "bindless.slang"
#include "clipmap.slang"
#include "surface_cache.slang"

static const float PI = 3.14159265359;

struct GiPush
{
    float4x4 inv_view_proj; // clip -> world (surface reconstruction)
    float4 sun;             // xyz direction toward the sun, w = intensity
    uint depth_index;       // sampled G-buffer depth
    uint normal_index;      // sampled G-buffer world normal
    uint gdf_sampled;       // volumes[] (SRV) index of the scene GDF
    uint out_index;         // storage-image index (indirect irradiance output)
    uint width;
    uint height;
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
    uint spp;               // hemisphere rays per pixel
    uint frame;             // frame counter (RNG decorrelation, for C4 temporal)
    uint albedo_r;          // C8a: volumes[] index of the red albedo channel (0xFFFFFFFF = constant)
    uint albedo_g;          // green
    uint albedo_b;          // blue
    float4 aabb_min;        // xyz GDF world min, w = ground plane Y
    float4 aabb_max;        // xyz GDF world max, w = sample distance clamp
    float4 params;          // x = ray max distance, y = surface bias, z = sky term, w = constant hit albedo (fallback)
    uint4 cache;            // C8b3: cards, cache_pos, cache_rad, num_cards (cards=0xFFFFFFFF = off)
    uint cache_tile;        // surface-cache atlas tile edge
    float clamp_max;        // firefly clamp: max bounce-sample radiance component (1e30 = off)
    uint clip_desc;         // Stage B: clipmap descriptor storage-buffer index
    uint clip_count;        // Stage B: clipmap level count (1 = single volume = legacy)
    // xyz = analytic ground material (not in the albedo volume); w = cone_k (P3 cone-trace LOD
    // slope; 0 = legacy linear march). float4, NOT float3 + a trailing scalar: HLSL/SPIR-V pack
    // a scalar after a float3 at +12 but MSL pads the float3 to +16, so the old layout mis-read
    // the following scalars on Metal (max_steps picked up cone_k's bytes -> wrong GI). A float4's
    // .w sits at +12 on all three backends; max_steps then gets its own clean 16-byte row.
    float4 ground_albedo;
    uint max_steps;         // Stage D3: bounce-ray march step cap (64 = legacy / gallery)
    uint vol_r;             // GI irradiance-volume R sampled index (0xFFFFFFFF = off -> trace rays)
    uint vol_g;             // green
    uint vol_b;             // blue (4 contiguous uints w/ max_steps -> Metal-safe row, no float3 pad)
};

[[vk::push_constant]]
ConstantBuffer<GiPush> pc : register(b0, space0);

#define GDF_MIN (pc.aabb_min.xyz)
#define GDF_MAX (pc.aabb_max.xyz)
#define GROUND_Y (pc.aabb_min.w)

// --- RNG + sampling (mirrors rt_common.slang so the noise is identical across
// backends and consistent with the path tracer). ---
uint pcg(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}
float rnd(inout uint s)
{
    return float(pcg(s) & 0xFFFFFFu) / float(0x1000000);
}
void onb(float3 n, out float3 t, out float3 b)
{
    t = normalize(abs(n.x) > 0.9 ? cross(n, float3(0, 1, 0)) : cross(n, float3(1, 0, 0)));
    b = cross(n, t);
}
float3 cosine_hemisphere(float3 n, inout uint rng)
{
    float u1 = rnd(rng);
    float u2 = rnd(rng);
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    float3 t, b;
    onb(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}

// --- GDF field via the clipmap (single level ⇒ identical to the legacy single volume +
// ground union of gdf_trace.slang). ---
ClipMap gi_clip()
{
    ClipMap c;
    c.desc = pc.clip_desc;
    c.count = pc.clip_count;
    return c;
}

float geo_inside(float3 p)
{
    return cm_geo_inside(gi_clip(), p, pc.aabb_max.w);
}

// Geometry distance for marching: outside the box, return the (clamped) box distance so
// the empty bounding box never reads as a surface while the ray steps sensibly.
float geo_march(float3 p)
{
    return cm_geo_march(gi_clip(), p, pc.aabb_max.w);
}

float scene_march(float3 p)
{
    return min(geo_march(p), p.y - GROUND_Y);
}
float scene_occ(float3 p)
{
    return min(geo_inside(p), p.y - GROUND_Y);
}

// C8a: the nearest-triangle albedo at a GDF hit (trilinear from the 3 R32F color volumes
// baked in `sdf_albedo_bake.slang`), or the constant fallback when no albedo volume is
// bound (albedo_r == 0xFFFFFFFF). The albedo grid shares the distance field's voxel grid.
float3 albedo_at(float3 p)
{
    return cm_albedo(gi_clip(), p, pc.params.w);
}

float3 scene_normal(float3 p)
{
    const float e = 0.012;
    float2 k = float2(1.0, -1.0);
    float3 grad = k.xyy * scene_occ(p + k.xyy * e) + k.yyx * scene_occ(p + k.yyx * e)
                  + k.yxy * scene_occ(p + k.yxy * e) + k.xxx * scene_occ(p + k.xxx * e);
    return safe_normalize(grad, float3(0.0, 1.0, 0.0));
}

// Cheap penumbra shadow toward the sun for the bounce hit (fewer steps than the primary
// trace's — indirect shadowing is low-frequency and gets denoised in C4).
float gi_shadow(float3 origin, float3 dir)
{
    float res = 1.0;
    float t = 0.02;
    float maxt = pc.params.x;
    for (int i = 0; i < 48; i++)
    {
        float h = scene_occ(origin + dir * t);
        if (h < 0.002)
            return 0.0;
        res = min(res, 16.0 * h / t);
        // P3 cone-trace LOD: widen the max step with distance (cone_k=0 => clamp(.,0.01,0.2) legacy).
        t += clamp(h, 0.01, max(0.2, pc.ground_albedo.w * t));
        if (t > maxt)
            break;
    }
    return saturate(res);
}

// Sphere-trace a bounce ray; returns the incoming radiance along it (0 if it escapes).
float3 trace_bounce(float3 origin, float3 dir)
{
    // Stage D3: content lowers the step cap (cheaper march); gallery passes 64 = legacy.
    int max_steps = int(max(pc.max_steps, 1u));
    const float HIT_EPS = 0.001;
    float t = pc.params.y; // start past the surface bias
    float maxt = pc.params.x;
    for (int i = 0; i < max_steps; i++)
    {
        float3 p = origin + dir * t;
        float d = scene_march(p);
        if (d < HIT_EPS)
        {
            float3 hn = scene_normal(p);
            // C8b3/C8g: when the surface cache is bound, the incoming radiance along this ray is
            // the hit surface's cached (multibounce) radiance — a direct lookup, no per-ray
            // re-light / shadow ray. On a cache MISS (e.g. the analytic ground, which has no
            // cards) fall back to the C8a per-ray colored re-light instead of black.
            if (pc.cache.x != 0xFFFFFFFFu)
            {
                bool cached = false;
                float3 r = sample_surface_cache(p, hn, pc.cache.x, pc.cache.y, pc.cache.z,
                                                pc.cache.w, pc.cache_tile, cached);
                if (cached)
                    return r;
            }
            float3 sun_dir = normalize(pc.sun.xyz);
            float ndl = max(dot(hn, sun_dir), 0.0);
            float sh = ndl > 0.0 ? gi_shadow(p + hn * 0.004, sun_dir) : 1.0;
            // C8a colored re-light: the hit's real surface albedo (per-voxel color volume)
            // × sun (soft-shadowed) + a small hemispheric sky fill so surfaces in full
            // shadow still bounce a little. Colored albedo => colored indirect bleed.
            // The analytic ground is not in the albedo volume, so albedo_at() returns the
            // nearest object's colour for a ground hit — use the ground's own albedo
            // ([0.8], matching deferred.rs) so floor bounce isn't tinted by nearby objects.
            bool is_ground = (p.y - GROUND_Y) < geo_inside(p);
            float3 hit_albedo = is_ground ? pc.ground_albedo.xyz : albedo_at(p);
            // Lambertian: the bounce surface re-emits albedo/PI * irradiance — the same
            // 1/PI the deferred direct term uses. Omitting it over-brightened indirect
            // diffuse by ~PI (the GDF-GI over-lighting vs the path tracer).
            float sky_fill = pc.params.z * (0.5 + 0.5 * hn.y);
            return hit_albedo * (pc.sun.w * ndl * sh + sky_fill) * (1.0 / PI);
        }
        // P3 cone-trace LOD: raise the step floor with distance (cone_k=0 => max(d,0.01) legacy).
        t += max(d, max(0.01, pc.ground_albedo.w * t));
        if (t > maxt)
            break;
    }
    return float3(0.0, 0.0, 0.0); // escaped to sky -> IBL already supplies it
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;

    // Far plane (no geometry): no indirect bounce.
    if (depth >= 1.0)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 hpos = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    float3 world_pos = hpos.xyz / hpos.w;

    // GI irradiance-volume mode (DDGI-lite radiance cache): instead of marching per-pixel rays,
    // sample the world-space irradiance volume (updated in gi_volume.slang with multibounce). This
    // is the indirect E the lighting multiplies by albedo; it fills enclosed interiors a single
    // bounce can't reach. Feeds the existing temporal+a-trous denoiser unchanged.
    if (pc.vol_r != 0xFFFFFFFFu)
    {
        float3 uvw = (world_pos - GDF_MIN) / (GDF_MAX - GDF_MIN);
        float3 e = float3(0.0, 0.0, 0.0);
        if (all(uvw >= 0.0) && all(uvw <= 1.0))
        {
            uvw = saturate(uvw);
            e = float3(g.volumes[pc.vol_r].SampleLevel(g.samp, uvw, 0.0).x,
                       g.volumes[pc.vol_g].SampleLevel(g.samp, uvw, 0.0).x,
                       g.volumes[pc.vol_b].SampleLevel(g.samp, uvw, 0.0).x);
        }
        g.storage_images[pc.out_index][tid.xy] = float4(e, 1.0);
        return;
    }

    float3 n = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);

    uint spp = max(pc.spp, 1u);
    uint seed = (tid.x * 1973u + tid.y * 9277u + pc.frame * 26699u) | 1u;
    float3 origin = world_pos + n * pc.params.y;

    float3 sum = float3(0.0, 0.0, 0.0);
    for (uint s = 0; s < spp; s++)
    {
        float3 dir = cosine_hemisphere(n, seed);
        // Firefly clamp per sample (before averaging): one ray hitting a bright spot can't
        // blow out the mean. Hue-preserving; `clamp_max = 1e30` disables it.
        float3 b = trace_bounce(origin, dir);
        float m = max(b.r, max(b.g, b.b));
        if (m > pc.clamp_max)
            b *= pc.clamp_max / m;
        sum += b;
    }
    // Cosine-weighted estimator: mean incoming radiance = indirect irradiance E. The
    // lighting pass multiplies by the surface albedo (× 1-metallic) to get L_o.
    float3 e = sum / float(spp);

    g.storage_images[pc.out_index][tid.xy] = float4(e, 1.0);
}
