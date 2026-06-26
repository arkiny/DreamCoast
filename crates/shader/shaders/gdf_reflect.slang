// Phase 11 Stage C6 — GDF reflections (off-screen fallback for SSR misses).
//
// A full-screen compute pass: for each G-buffer pixel it reflects the view ray about
// the world normal and sphere-traces that reflection ray through the world scene GDF
// (the same baked field + helpers as the C3 GI / B4 trace). On a hit it re-lights the
// hit point (gradient normal + soft-shadowed sun + a small sky fill, constant albedo —
// the GDF carries no material, like C3); on escape it returns the procedural sky.
//
// This is the off-screen counterpart to C5's screen-space reflections: SSR is accurate
// where the reflected surface is on screen but misses off-screen / disoccluded / screen-
// edge rays, which this fills with plausible GDF reflections. C7 composites the two
// (SSR where confident, GDF otherwise, sky on the GDF miss) over the IBL specular.
//
// KEY vs C3 diffuse GI: a GDF ray that escapes returns the SKY, not 0. Specular misses
// must show the sky (a mirror reflects the sky); C3's diffuse misses return 0 because
// the IBL diffuse already supplies the sky there. Output is raw radiance (the tonemap
// applies exposure), like the C1 scene-GDF trace.

#include "bindless.slang"

struct ReflectPush
{
    float4x4 inv_view_proj; // clip -> world (surface reconstruction)
    float4 cam_pos;         // xyz camera world position
    float4 sun;             // xyz direction toward the sun, w = intensity
    uint depth_index;       // sampled G-buffer depth
    uint normal_index;      // sampled G-buffer world normal
    uint gdf_sampled;       // volumes[] (SRV) index of the scene GDF
    uint out_index;         // storage image (reflection color)
    uint width;
    uint height;
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
    uint pad0;
    float4 aabb_min;        // xyz GDF world min, w = ground plane Y
    float4 aabb_max;        // xyz GDF world max, w = sample distance clamp
    float4 params;          // x = ray max distance, y = constant hit albedo (fallback), z = sky fill, w = surface bias
    uint albedo_r;          // C8a: volumes[] index of the red albedo channel (0xFFFFFFFF = constant)
    uint albedo_g;          // green
    uint albedo_b;          // blue
    uint pad1;
};

[[vk::push_constant]]
ConstantBuffer<ReflectPush> pc : register(b0, space0);

#define GDF_MIN (pc.aabb_min.xyz)
#define GDF_MAX (pc.aabb_max.xyz)
#define GROUND_Y (pc.aabb_min.w)

// Distance to the baked geometry inside the GDF box (trilinear sample, clamped). Outside
// the box: no geometry. (Same as the C3 GI / B4 trace `geo_inside`.)
float geo_inside(float3 p)
{
    float3 uvw = (p - GDF_MIN) / (GDF_MAX - GDF_MIN);
    if (any(uvw < 0.0) || any(uvw > 1.0))
        return 1e3;
    return min(g.volumes[pc.gdf_sampled].SampleLevel(g.samp, uvw, 0.0).x, pc.aabb_max.w);
}

// Marching distance: outside the box, return the (clamped) box distance so the empty
// bounding box never reads as a surface while the ray steps sensibly.
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

float scene_march(float3 p)
{
    return min(geo_march(p), p.y - GROUND_Y);
}
float scene_occ(float3 p)
{
    return min(geo_inside(p), p.y - GROUND_Y);
}

// C8a: the nearest-triangle albedo at a GDF reflection hit (trilinear from the 3 R32F
// color volumes), or the constant fallback when no albedo volume is bound. Shares the
// distance field's voxel grid; gives colored reflections instead of grey.
float3 albedo_at(float3 p)
{
    if (pc.albedo_r == 0xFFFFFFFFu)
        return float3(pc.params.y, pc.params.y, pc.params.y);
    float3 uvw = saturate((p - GDF_MIN) / (GDF_MAX - GDF_MIN));
    return float3(g.volumes[pc.albedo_r].SampleLevel(g.samp, uvw, 0.0).x,
                  g.volumes[pc.albedo_g].SampleLevel(g.samp, uvw, 0.0).x,
                  g.volumes[pc.albedo_b].SampleLevel(g.samp, uvw, 0.0).x);
}

float3 scene_normal(float3 p)
{
    const float e = 0.012;
    float2 k = float2(1.0, -1.0);
    return normalize(k.xyy * scene_occ(p + k.xyy * e) + k.yyx * scene_occ(p + k.yyx * e)
                     + k.yxy * scene_occ(p + k.yxy * e) + k.xxx * scene_occ(p + k.xxx * e));
}

// Gradient sky + HDR sun disk (same model as the Stage-A SW-RT / path-tracer sky).
float3 sky(float3 dir, float3 sun_dir, float sun_i)
{
    float up = dir.y;
    float3 zenith = float3(0.04, 0.16, 0.55);
    float3 horizon = float3(0.50, 0.66, 0.92);
    float3 ground = float3(0.16, 0.15, 0.14);
    float3 col = up >= 0.0 ? lerp(horizon, zenith, pow(saturate(up), 0.45))
                           : lerp(horizon, ground, saturate(-up * 6.0));
    float c = saturate(dot(dir, sun_dir));
    col += sun_i * (pow(c, 4000.0) * 6.0 + pow(c, 32.0) * 0.4);
    return col;
}

// Short penumbra shadow toward the sun for the reflected hit (cheap; reflections are
// secondary so a coarse shadow is fine).
float refl_shadow(float3 origin, float3 dir, float maxt)
{
    float res = 1.0;
    float t = 0.02;
    for (int i = 0; i < 48; i++)
    {
        float h = scene_occ(origin + dir * t);
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
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;
    float3 sun_dir = normalize(pc.sun.xyz);
    float sun_i = pc.sun.w;

    // Background: reflect nothing — show the sky along the view ray (so the viz reads as
    // the scene; C7 only consumes this for shaded pixels anyway).
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    if (depth >= 1.0)
    {
        float4 far = mul(pc.inv_view_proj, float4(ndc, 1.0, 1.0));
        float3 vdir = normalize(far.xyz / far.w - pc.cam_pos.xyz);
        g.storage_images[pc.out_index][tid.xy] = float4(sky(vdir, sun_dir, sun_i), 1.0);
        return;
    }

    float4 hpos = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    float3 p = hpos.xyz / hpos.w;
    float3 n = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);
    float3 v = normalize(pc.cam_pos.xyz - p);
    float3 dir = reflect(-v, n);

    float max_dist = pc.params.x;
    float sky_fill = pc.params.z;
    float bias = pc.params.w;

    // Sphere-trace the reflection ray.
    const int MAX_STEPS = 96;
    const float HIT_EPS = 0.001;
    float t = bias;
    bool hit = false;
    float3 hp = p;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        float3 rp = p + dir * t;
        float d = scene_march(rp);
        if (d < HIT_EPS)
        {
            hp = rp;
            hit = true;
            break;
        }
        t += max(d, 0.01);
        if (t > max_dist)
            break;
    }

    float3 color;
    if (hit)
    {
        float3 hn = scene_normal(hp);
        float ndl = max(dot(hn, sun_dir), 0.0);
        float sh = ndl > 0.0 ? refl_shadow(hp + hn * 0.004, sun_dir, max_dist) : 1.0;
        float fill = sky_fill * (0.5 + 0.5 * hn.y);
        color = albedo_at(hp) * (sun_i * ndl * sh + fill);
    }
    else
    {
        color = sky(dir, sun_dir, sun_i); // specular miss -> the sky (NOT 0)
    }

    g.storage_images[pc.out_index][tid.xy] = float4(color, 1.0);
}
