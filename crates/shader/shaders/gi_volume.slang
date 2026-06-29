// World-space irradiance volume (DDGI-lite radiance cache). One thread per volume probe
// (voxel). Each probe casts `spp` uniform-sphere rays into the scene GDF; at a hit it shades
// the surface with direct sun (soft-shadowed) + a sky fill + the PREVIOUS frame's volume
// sampled at the hit (= multibounce), on a miss it adds the sky. The mean incoming radiance is
// EMA-accumulated into the write volume. Because each probe samples the volume itself, bright
// regions (a sunlit atrium) propagate into dark ones (an enclosed nave) over frames — the
// infinite-bounce fill a single-bounce trace can't reach (UE Lumen radiance cache / DDGI).
//
// RGB is three R32F volumes (reusing the albedo-volume pattern), ping-pong read/write.

#include "bindless.slang"
#include "clipmap.slang"

static const float GV_PI = 3.14159265359;

struct GiVolumePush
{
    float4 aabb_min;     // xyz volume/world min, w = ground plane Y
    float4 aabb_max;     // xyz volume/world max, w = sample distance clamp
    float4 sun;          // xyz direction toward the sun, w = intensity
    uint4 dims;          // x,y,z = probe grid resolution, w = frame (RNG)
    uint4 read_rgb;      // sampled volumes[] indices of the previous-frame irradiance (R,G,B); w = reset
    uint4 write_rgb;     // storage_volumes[] indices to write this frame (R,G,B); w unused
    uint4 albedo_rgb;    // sampled volumes[] indices of the per-voxel albedo (R,G,B); x=0xFFFFFFFF = constant
    uint4 clip;          // x = clipmap descriptor index, y = level count
    float4 params;       // x = spp, y = ray max dist, z = sky fill, w = EMA alpha
    float4 ground;       // xyz analytic ground albedo, w = surface bias
};

[[vk::push_constant]]
ConstantBuffer<GiVolumePush> pc : register(b0, space0);

#define GROUND_Y (pc.aabb_min.w)

uint pcg(inout uint s) { s = s * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28) + 4u)) ^ s) * 277803737u; return (w >> 22) ^ w; }
float rnd(inout uint s) { return float(pcg(s) & 0xFFFFFFu) / float(0x1000000); }

ClipMap gv_clip() { ClipMap c; c.desc = pc.clip.x; c.count = pc.clip.y; return c; }
float geo_inside(float3 p) { return cm_geo_inside(gv_clip(), p, pc.aabb_max.w); }
float geo_march(float3 p) { return cm_geo_march(gv_clip(), p, pc.aabb_max.w); }
float scene_march(float3 p) { return min(geo_march(p), p.y - GROUND_Y); }
float scene_occ(float3 p) { return min(geo_inside(p), p.y - GROUND_Y); }
float3 albedo_at(float3 p) { return cm_albedo(gv_clip(), p, 0.7); }

float3 scene_normal(float3 p)
{
    const float e = 0.05;
    float2 k = float2(1.0, -1.0);
    float3 g = k.xyy * scene_occ(p + k.xyy * e) + k.yyx * scene_occ(p + k.yyx * e)
               + k.yxy * scene_occ(p + k.yxy * e) + k.xxx * scene_occ(p + k.xxx * e);
    return safe_normalize(g, float3(0.0, 1.0, 0.0));
}

float gv_shadow(float3 o, float3 d)
{
    float res = 1.0, t = 0.05, maxt = pc.params.y;
    for (int i = 0; i < 32; i++)
    {
        float h = scene_occ(o + d * t);
        if (h < 0.003)
            return 0.0;
        res = min(res, 16.0 * h / t);
        t += clamp(h, 0.02, 0.4);
        if (t > maxt)
            break;
    }
    return saturate(res);
}

// Simple gradient sky (the volume's miss radiance; the env cube isn't sampleable here).
float3 gv_sky(float3 d)
{
    float up = saturate(d.y * 0.5 + 0.5);
    return lerp(float3(0.45, 0.5, 0.6), float3(0.18, 0.34, 0.7), up) * pc.params.z;
}

// Trilinear sample of the ping-pong RGB read volume at world position p.
float3 read_volume(float3 p)
{
    float3 uvw = (p - pc.aabb_min.xyz) / (pc.aabb_max.xyz - pc.aabb_min.xyz);
    if (any(uvw < 0.0) || any(uvw > 1.0))
        return float3(0.0, 0.0, 0.0);
    uvw = saturate(uvw);
    return float3(g.volumes[pc.read_rgb.x].SampleLevel(g.samp, uvw, 0.0).x,
                  g.volumes[pc.read_rgb.y].SampleLevel(g.samp, uvw, 0.0).x,
                  g.volumes[pc.read_rgb.z].SampleLevel(g.samp, uvw, 0.0).x);
}

// Uniform sphere direction.
float3 uniform_sphere(inout uint rng)
{
    float z = rnd(rng) * 2.0 - 1.0;
    float a = rnd(rng) * 2.0 * GV_PI;
    float r = sqrt(max(0.0, 1.0 - z * z));
    return float3(r * cos(a), z, r * sin(a));
}

[shader("compute")]
[numthreads(4, 4, 4)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dims.x || tid.y >= pc.dims.y || tid.z >= pc.dims.z)
        return;

    // Probe world position = voxel centre mapped through the volume AABB.
    float3 uvw = (float3(tid) + 0.5) / float3(pc.dims.xyz);
    float3 probe = lerp(pc.aabb_min.xyz, pc.aabb_max.xyz, uvw);

    uint spp = uint(max(pc.params.x, 1.0));
    uint seed = (tid.x * 1973u + tid.y * 9277u + tid.z * 26699u + pc.dims.w * 51059u) | 1u;
    float3 sun_dir = normalize(pc.sun.xyz);
    float bias = pc.ground.w;
    float maxt = pc.params.y;

    float3 sum = float3(0.0, 0.0, 0.0);
    for (uint s = 0; s < spp; s++)
    {
        float3 dir = uniform_sphere(seed);
        float t = bias;
        bool hit = false;
        float3 hp = probe;
        for (int i = 0; i < 48; i++)
        {
            float3 p = probe + dir * t;
            float d = scene_march(p);
            if (d < 0.003)
            {
                hp = p;
                hit = true;
                break;
            }
            t += max(d, 0.02);
            if (t > maxt)
                break;
        }
        if (hit)
        {
            float3 hn = scene_normal(hp);
            float ndl = max(dot(hn, sun_dir), 0.0);
            float sh = ndl > 0.0 ? gv_shadow(hp + hn * 0.02, sun_dir) : 1.0;
            bool is_ground = (hp.y - GROUND_Y) < geo_inside(hp);
            float3 alb = is_ground ? pc.ground.xyz : albedo_at(hp);
            float sky_fill = pc.params.z * (0.5 + 0.5 * hn.y);
            // Outgoing radiance = albedo/PI * (direct sun + sky fill + previous-frame indirect at
            // the hit). The last term is the multibounce that propagates light into enclosed areas.
            float3 indirect = read_volume(hp);
            sum += alb * (pc.sun.w * ndl * sh + sky_fill + indirect) * (1.0 / GV_PI);
        }
        else
        {
            sum += gv_sky(dir);
        }
    }
    float3 e = sum / float(spp);

    // EMA with the previous frame (reset on the first frame / re-alloc).
    if (pc.read_rgb.w == 0u)
    {
        float3 prev = read_volume(probe);
        e = lerp(prev, e, pc.params.w);
    }
    e = make_finite(e);
    g.storage_volumes[pc.write_rgb.x][tid] = e.x;
    g.storage_volumes[pc.write_rgb.y][tid] = e.y;
    g.storage_volumes[pc.write_rgb.z][tid] = e.z;
}
