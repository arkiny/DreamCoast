// World-space directional irradiance volume (radiance-cache probes). One thread per volume
// probe (voxel). Each probe casts `spp` uniform-sphere rays into the scene GDF; at a hit it
// shades the surface with direct sun (soft-shadowed) + a sky fill + the PREVIOUS frame's
// volume sampled at the hit (= multibounce), on a miss it adds the sky. Instead of one scalar
// average per probe, the incoming radiance L(dir) is projected onto a band-0+1 spherical-
// harmonic basis (4 coefficients per colour channel) and EMA-accumulated per coefficient.
// Because each probe samples the volume itself, bright regions (a sunlit atrium) propagate
// into dark ones (an enclosed nave) over frames — the infinite-bounce fill a single-bounce
// trace can't reach.
//
// Storing direction (not just the omnidirectional mean) is what lets a coloured surface bleed
// onto its neighbours: a receiver reconstructs the cosine-weighted hemispherical average in
// ITS normal direction, so a floor next to a red curtain weights the up/curtain hemisphere and
// keeps the colour the full-sphere average washed out to ~0.
//
// Layout: 12 R32F volumes per ping-pong slot — slot index = channel*4 + coeff (R coeffs 0..3,
// G coeffs 4..7, B coeffs 8..11). The 12 are allocated contiguously, so only the base bindless
// index is passed (read_rgb.x = sampled base of the previous slot, write_rgb.x = storage base
// of this slot); read_rgb.w = reset.
//
// Sky-visibility (레퍼런스식 indoor skylight occlusion): the probe also projects a SCALAR directional
// sky-visibility onto band-0/1 SH (4 R32F volumes/slot) — 1 on a sky-escape, 0 on a geometry hit. A
// receiver reconstructs the cosine-weighted hemispherical sky visibility V(n); the lighting occludes
// the IBL DIFFUSE skylight by it (with a tint/min-occlusion leak), so an enclosed interior isn't
// flooded by the full (blue) diffuse skylight. bases: read_rgb.y = sampled, write_rgb.y = storage.

#include "bindless.slang"
#include "clipmap.slang"

static const float GV_PI = 3.14159265359;

// Real SH band-0/1 basis normalisation: Y0 = sqrt(1/4pi), Y1 = sqrt(3/4pi)*{y,z,x}.
static const float SH_C0 = 0.2820947918; // sqrt(1/(4*pi))
static const float SH_C1 = 0.4886025119; // sqrt(3/(4*pi))

struct GiVolumePush
{
    float4 aabb_min;     // xyz volume/world min, w = ground plane Y
    float4 aabb_max;     // xyz volume/world max, w = sample distance clamp
    float4 sun;          // xyz direction toward the sun, w = intensity
    uint4 dims;          // x,y,z = probe grid resolution, w = frame (RNG)
    uint4 read_rgb;      // x = sampled base of the previous radiance SH volumes; y = sampled base of
                         // the previous sky-vis SH volumes (0xFFFFFFFF = sky-vis off); w = reset
    uint4 write_rgb;     // x = storage base of the radiance SH volumes; y = storage base of the
                         // sky-vis SH volumes written this frame
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

// Trilinear sample of the 12 SH coefficient volumes (base index `base`) at world position p,
// filling c[coeff].rgb. Out of the volume AABB -> all zero. coeff `k` channel `ch` lives in
// volume `base + ch*4 + k`.
void read_coeffs(uint base, float3 p, out float3 c[4])
{
    [unroll]
    for (int k = 0; k < 4; k++)
        c[k] = float3(0.0, 0.0, 0.0);
    float3 uvw = (p - pc.aabb_min.xyz) / (pc.aabb_max.xyz - pc.aabb_min.xyz);
    if (any(uvw < 0.0) || any(uvw > 1.0))
        return;
    uvw = saturate(uvw);
    [unroll]
    for (int ch = 0; ch < 3; ch++)
    {
        [unroll]
        for (int k = 0; k < 4; k++)
            c[k][ch] = g.volumes[base + uint(ch * 4 + k)].SampleLevel(g.samp, uvw, 0.0).x;
    }
}

// Trilinear sample of the 4 SCALAR sky-visibility SH coefficient volumes (base index `base`).
void read_coeffs_scalar(uint base, float3 p, out float c[4])
{
    [unroll]
    for (int k = 0; k < 4; k++)
        c[k] = 0.0;
    float3 uvw = (p - pc.aabb_min.xyz) / (pc.aabb_max.xyz - pc.aabb_min.xyz);
    if (any(uvw < 0.0) || any(uvw > 1.0))
        return;
    uvw = saturate(uvw);
    [unroll]
    for (int k = 0; k < 4; k++)
        c[k] = g.volumes[base + uint(k)].SampleLevel(g.samp, uvw, 0.0).x;
}

// Cosine-weighted hemispherical average radiance in direction n, reconstructed from the SH
// radiance coefficients (the clamped-cosine convolution: A0/PI = 1, A1/PI = 2/3). For an
// isotropic field (only the band-0 coeff) this returns exactly the omnidirectional mean, so it
// is a strict directional generalisation of the old scalar probe. Clamped >= 0 (band-1 SH can
// ring negative).
float3 eval_irradiance(float3 c[4], float3 n)
{
    float3 e = c[0] * SH_C0
               + (2.0 / 3.0) * (SH_C1 * n.y * c[1] + SH_C1 * n.z * c[2] + SH_C1 * n.x * c[3]);
    return max(e, float3(0.0, 0.0, 0.0));
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
    uint read_base = pc.read_rgb.x;

    // SH band-0/1 accumulators (4 coeffs x RGB) + scalar sky-visibility (4 coeffs).
    float3 sh0 = float3(0.0, 0.0, 0.0);
    float3 sh1 = float3(0.0, 0.0, 0.0);
    float3 sh2 = float3(0.0, 0.0, 0.0);
    float3 sh3 = float3(0.0, 0.0, 0.0);
    float sv0 = 0.0, sv1 = 0.0, sv2 = 0.0, sv3 = 0.0;
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
        // Incoming radiance L(dir) arriving at the probe along this ray.
        float3 radiance;
        if (hit)
        {
            float3 hn = scene_normal(hp);
            float ndl = max(dot(hn, sun_dir), 0.0);
            float sh = ndl > 0.0 ? gv_shadow(hp + hn * 0.02, sun_dir) : 1.0;
            bool is_ground = (hp.y - GROUND_Y) < geo_inside(hp);
            float3 alb = is_ground ? pc.ground.xyz : albedo_at(hp);
            float sky_fill = pc.params.z * (0.5 + 0.5 * hn.y);
            // Multibounce: the hit surface re-emits its own indirect irradiance, read
            // DIRECTIONALLY from the previous-frame SH volume in the hit's normal direction.
            // The DIRECT sun + sky fill are irradiance, so they get the albedo/PI BRDF; the
            // stored irradiance (E) re-enters as albedo*indirect (the ×PI to irradiance cancels
            // the BRDF 1/PI), matching the old scalar probe's energy.
            float3 cprev[4];
            read_coeffs(read_base, hp, cprev);
            float3 indirect = eval_irradiance(cprev, hn);
            radiance = alb * (pc.sun.w * ndl * sh + sky_fill) * (1.0 / GV_PI) + alb * indirect;
        }
        else
        {
            // Sky-escape contributes 0 to the stored indirect, matching the per-pixel march
            // (`gdf_gi.slang trace_bounce`): the deferred IBL diffuse already supplies the
            // (AO-occluded) sky, so adding the sky here too would double-count it AND flood the
            // directional irradiance with blue — drowning the coloured scene bounce (e.g. a red
            // curtain) the volume exists to propagate. The volume now carries ONLY scene radiance
            // (single-source sky = IBL). The hemispheric `sky_fill` at HITS stays: that is the sky
            // bouncing off a surface, a legitimate indirect term the march also keeps.
            radiance = float3(0.0, 0.0, 0.0);
        }
        // Project L(dir) onto the band-0/1 SH basis.
        sh0 += radiance * SH_C0;
        sh1 += radiance * (SH_C1 * dir.y);
        sh2 += radiance * (SH_C1 * dir.z);
        sh3 += radiance * (SH_C1 * dir.x);
        // Sky-visibility: 1 if this ray escaped to sky, 0 if it hit geometry — projected the same
        // way so a receiver reconstructs the cosine-weighted hemispherical sky visibility V(n).
        float vis = hit ? 0.0 : 1.0;
        sv0 += vis * SH_C0;
        sv1 += vis * (SH_C1 * dir.y);
        sv2 += vis * (SH_C1 * dir.z);
        sv3 += vis * (SH_C1 * dir.x);
    }
    // Monte-Carlo estimator for uniform-sphere sampling (pdf = 1/4pi): coeff = (4pi/N) * sum.
    float kw = (4.0 * GV_PI) / float(spp);
    float3 c[4] = { sh0 * kw, sh1 * kw, sh2 * kw, sh3 * kw };
    float sv[4] = { sv0 * kw, sv1 * kw, sv2 * kw, sv3 * kw };

    // EMA with the previous frame (per coefficient; reset on the first frame / re-alloc).
    if (pc.read_rgb.w == 0u)
    {
        float3 prev[4];
        read_coeffs(read_base, probe, prev);
        [unroll]
        for (int i = 0; i < 4; i++)
            c[i] = lerp(prev[i], c[i], pc.params.w);
        if (pc.read_rgb.y != 0xFFFFFFFFu)
        {
            float svprev[4];
            read_coeffs_scalar(pc.read_rgb.y, probe, svprev);
            [unroll]
            for (int i = 0; i < 4; i++)
                sv[i] = lerp(svprev[i], sv[i], pc.params.w);
        }
    }
    [unroll]
    for (int i = 0; i < 4; i++)
        c[i] = make_finite(c[i]);

    uint write_base = pc.write_rgb.x;
    [unroll]
    for (int ch = 0; ch < 3; ch++)
    {
        [unroll]
        for (int k = 0; k < 4; k++)
            g.storage_volumes[write_base + uint(ch * 4 + k)][tid] = c[k][ch];
    }

    // Sky-visibility SH (4 scalar volumes), when bound.
    if (pc.write_rgb.y != 0xFFFFFFFFu)
    {
        uint sv_base = pc.write_rgb.y;
        [unroll]
        for (int k = 0; k < 4; k++)
            g.storage_volumes[sv_base + uint(k)][tid] = make_finite(float3(sv[k], 0.0, 0.0)).x;
    }
}
