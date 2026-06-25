// Phase 11 Stage A (A1+A2) — compute software ray tracing via SDF sphere tracing.
//
// A compute shader that traces one primary camera ray per pixel against an
// *analytic* signed-distance-field scene (spheres / box / ground plane) by sphere
// tracing — no hardware ray tracing, no TLAS. This stands up the software-RT
// machinery (camera-ray generation, the march loop, gradient normals, storage-image
// output) that Stage B will reuse against a *baked* global distance field instead of
// these hardcoded primitives.
//
// The analytic scene mirrors the rasterizer's sample-scene layout (model_radius is
// normalized to 1.0, so the coordinates are concrete): three spheres + a box on a
// ground plane, framed by the same camera/sun as the raster and hardware-RT paths so
// the result is directly comparable. The avocado mesh is proxied by a sphere here;
// Stage B's SDF bake replaces every primitive with the real geometry.
//
// Output is raw scene radiance into a storage image; the tonemap pass applies the
// camera exposure + filmic curve, exactly like the path tracer's output.

#include "bindless.slang"

static const float PI = 3.14159265358979;

// Push constants — a self-contained block (no shared rt_common push, since the SW-RT
// path deliberately needs none of the TLAS / material-table plumbing). Kept within
// 128 bytes to match the other compute push blocks.
struct SdfPush
{
    float4x4 inv_view_proj; // clip -> world (primary ray reconstruction)
    float4 cam_pos;         // xyz camera world position
    float4 sun;             // xyz direction toward the sun, w = intensity
    uint out_index;         // bindless storage-image index (output)
    uint width;
    uint height;
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
};

[[vk::push_constant]]
ConstantBuffer<SdfPush> pc : register(b0, space0);

// ---- Analytic SDF scene -----------------------------------------------------

// Primitive SDFs (all return signed distance; negative inside).
float sd_sphere(float3 p, float3 c, float r)
{
    return length(p - c) - r;
}
float sd_box(float3 p, float3 c, float3 half_extent)
{
    float3 q = abs(p - c) - half_extent;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}
float sd_ground(float3 p) // plane at y = 0
{
    return p.y;
}

// Scene distance + nearest-primitive material id (0 ground, 1 chrome sphere,
// 2 copper sphere, 3 avocado proxy, 4 red box). The coordinates match the
// sample scene in apps/sandbox (model_radius = 1.0).
float scene_sdf(float3 p, out int mat)
{
    float d = sd_ground(p);
    mat = 0;
    float ds;
    ds = sd_sphere(p, float3(-1.7, 0.75, 0.5), 0.75);
    if (ds < d) { d = ds; mat = 1; }
    ds = sd_sphere(p, float3(1.9, 0.5, -0.4), 0.5);
    if (ds < d) { d = ds; mat = 2; }
    ds = sd_sphere(p, float3(0.0, 0.65, 0.0), 0.65); // avocado proxy
    if (ds < d) { d = ds; mat = 3; }
    ds = sd_box(p, float3(0.0, 0.45, -2.0), float3(0.45, 0.45, 0.45));
    if (ds < d) { d = ds; mat = 4; }
    return d;
}

// Distance only (for gradient / shadow marches that don't need the material).
float scene_dist(float3 p)
{
    int m;
    return scene_sdf(p, m);
}

// Central-difference gradient of the distance field = surface normal at a hit.
float3 scene_normal(float3 p)
{
    const float e = 0.0015;
    float2 k = float2(1.0, -1.0);
    return normalize(k.xyy * scene_dist(p + k.xyy * e) + k.yyx * scene_dist(p + k.yyx * e)
                     + k.yxy * scene_dist(p + k.yxy * e) + k.xxx * scene_dist(p + k.xxx * e));
}

// Per-material base color (linear). Matches the sample-scene materials closely
// enough for visual comparison; Stage B/C take real materials from a surface cache.
float3 material_albedo(int mat)
{
    switch (mat)
    {
    case 1:
        return float3(0.55, 0.58, 0.62); // chrome sphere (grey)
    case 2:
        return float3(0.72, 0.45, 0.20); // copper sphere
    case 3:
        return float3(0.28, 0.55, 0.12); // avocado proxy (green)
    case 4:
        return float3(0.80, 0.20, 0.18); // red box
    default:
        return float3(0.82, 0.82, 0.85); // ground (near-white)
    }
}

// Gradient sky + HDR sun disk (same model as the path tracer's sky, so the
// background and sun match across the raster / HW-RT / SW-RT paths).
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

// Penumbra soft shadow by sphere tracing toward the light (Inigo Quilez): march a
// shadow ray and track the closest the ray passes to any surface, scaled by 1/t, to
// approximate how much of the (finite) sun disk is occluded. `k` sets the penumbra
// hardness (larger = sharper); tuned to roughly match the path tracer's ~1.1° disk
// sun. Returns a [0,1] visibility factor with soft edges (A2 — replaces A1's none).
float soft_shadow(float3 origin, float3 dir, float k)
{
    float res = 1.0;
    float t = 0.02; // start off the surface to avoid self-shadowing
    const float MAX_T = 40.0;
    for (int i = 0; i < 96; i++)
    {
        float h = scene_dist(origin + dir * t);
        if (h < 0.001)
            return 0.0; // fully occluded
        res = min(res, k * h / t);
        t += clamp(h, 0.01, 0.4);
        if (t > MAX_T)
            break;
    }
    return saturate(res);
}

// Marched ambient occlusion (Inigo Quilez 5-tap): step short distances along the
// surface normal and compare the expected free distance to the actual field value;
// where the field falls short, geometry is nearby and the point is occluded. Scales
// the sky ambient so creases/contacts darken (A2).
float calc_ao(float3 p, float3 n)
{
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++)
    {
        float h = 0.01 + 0.12 * float(i) / 4.0;
        float d = scene_dist(p + n * h);
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return saturate(1.0 - 3.0 * occ);
}

// Sphere-trace a ray; returns true on a surface hit and writes the hit point.
bool sphere_trace(float3 origin, float3 dir, out float3 hit_p)
{
    const int MAX_STEPS = 192;
    const float MAX_DIST = 60.0;
    const float HIT_EPS = 0.001;
    float t = 0.01;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        float3 p = origin + dir * t;
        float d = scene_dist(p);
        if (d < HIT_EPS)
        {
            hit_p = p;
            return true;
        }
        t += d;
        if (t > MAX_DIST)
            break;
    }
    hit_p = origin + dir * t;
    return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float3 sun_dir = normalize(pc.sun.xyz);
    float sun_i = pc.sun.w;

    // Primary camera ray (identical reconstruction to the path tracer's
    // primary_ray_dir: clip -> world through inv_view_proj at z = 1).
    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 world = mul(pc.inv_view_proj, float4(ndc, 1.0, 1.0));
    float3 dir = normalize(world.xyz / world.w - pc.cam_pos.xyz);

    float3 hit_p;
    float3 color;
    if (sphere_trace(pc.cam_pos.xyz, dir, hit_p))
    {
        int mat;
        scene_sdf(hit_p, mat);
        float3 n = scene_normal(hit_p);
        float3 albedo = material_albedo(mat);
        // A2 = soft shadows + AO on top of A1's Lambert sun + hemisphere sky ambient.
        // The sun term is gated by a sphere-traced penumbra shadow; the sky ambient
        // is modulated by marched ambient occlusion. Raw radiance is exposed +
        // tonemapped downstream.
        float ndl = max(dot(n, sun_dir), 0.0);
        float shadow = ndl > 0.0 ? soft_shadow(hit_p + n * 0.002, sun_dir, 24.0) : 1.0;
        float ao = calc_ao(hit_p, n);
        float3 direct = albedo * sun_i * ndl * shadow;
        float3 ambient = albedo * lerp(0.15, 0.35, saturate(n.y * 0.5 + 0.5)) * ao;
        color = direct + ambient;
    }
    else
    {
        color = sky(dir, sun_dir, sun_i);
    }

    g.storage_images[pc.out_index][tid.xy] = float4(color, 1.0);
}
