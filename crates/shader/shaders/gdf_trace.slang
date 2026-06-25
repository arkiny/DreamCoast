// Phase 11 Stage B (B4) — software ray tracing against the global distance field.
//
// The Stage-A SW-RT machinery (camera-ray generation, sphere-trace march, gradient
// normals, penumbra soft shadows, marched AO — all unchanged) now walks a *baked*
// distance field instead of analytic primitives: `scene_dist` trilinear-samples the
// merged GDF volume (B3) and mins it with a ground plane. So the surface the ray hits
// is real, baked mesh geometry (B2 bake → B3 merge → here), not a hardcoded sphere.
//
// A `mode` bit swaps the GDF sample for the analytic field the GDF was baked from
// (a centred sphere) under the *same* camera/march — the B4 correctness check: the
// two images may differ only by trilinear/voxelisation error, never in silhouette or
// shading model. Verification drives the full-coverage single-instance GDF
// (P11_GDF_INSTANCES=1) so every voxel carries a valid distance and the march never
// overshoots; robust tracing of sparse multi-box clipmaps is a later refinement.
//
// Output is raw scene radiance into a storage image; the tonemap pass applies the
// camera exposure + filmic curve, exactly like the Stage-A SW-RT and path-tracer paths.

#include "bindless.slang"

// The GDF world extent: the unit cube, matching B3's merge AABB. When the clipmap
// becomes multi-level/configurable these move into the push block.
static const float3 GDF_MIN = float3(0.0, 0.0, 0.0);
static const float3 GDF_MAX = float3(1.0, 1.0, 1.0);
// Ground plane height; the verification sphere (centre 0.5, radius 0.3) rests on it.
static const float GROUND_Y = 0.2;
// The analytic field the bake came from (B2): a sphere at the unit-cube centre.
static const float3 REF_CENTRE = float3(0.5, 0.5, 0.5);
static const float REF_RADIUS = 0.3;

struct GdfPush
{
    float4x4 inv_view_proj; // clip -> world (primary ray reconstruction)
    float4 cam_pos;         // xyz camera world position
    float4 sun;             // xyz direction toward the sun, w = intensity
    uint out_index;         // bindless storage-image index (output)
    uint width;
    uint height;
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
    uint gdf_sampled;       // volumes[] (SRV) index of the merged GDF
    uint mode;              // bit0: 1 = analytic reference field instead of the GDF
    uint pad0;
    uint pad1;
};

[[vk::push_constant]]
ConstantBuffer<GdfPush> pc : register(b0, space0);

// Distance to the baked geometry inside the GDF box — the trilinear sample, or the
// analytic field it was baked from when mode bit0 is set. Outside the box there is no
// geometry, so this returns a large distance: occlusion/shadow queries must not see
// the empty bounding box as a solid (which would tile the ground with the box's square
// isocontours). The primary march adds a box proxy separately (`geo_march`).
float geo_inside(float3 p)
{
    if ((pc.mode & 1u) != 0u)
        return length(p - REF_CENTRE) - REF_RADIUS;

    float3 uvw = (p - GDF_MIN) / (GDF_MAX - GDF_MIN);
    if (any(uvw < 0.0) || any(uvw > 1.0))
        return 1e3; // outside the box: no geometry here
    // Clamp the sample so a sparse GDF's empty-voxel sentinel can't make the march
    // overshoot; 0.6 exceeds the true max distance of the full-coverage field, so the
    // verification sphere is unaffected.
    return min(g.volumes[pc.gdf_sampled].SampleLevel(g.samp, uvw, 0.0).x, 0.6);
}

// Geometry distance for the *primary* march: like `geo_inside`, but outside the box it
// returns the (clamped) box distance so the camera ray advances toward the GDF without
// the box boundary ever registering as a surface. Wrap-addressing is dodged because the
// sampler is only touched inside [0,1].
float geo_march(float3 p)
{
    if ((pc.mode & 1u) != 0u)
        return length(p - REF_CENTRE) - REF_RADIUS;

    float3 uvw = (p - GDF_MIN) / (GDF_MAX - GDF_MIN);
    if (any(uvw < 0.0) || any(uvw > 1.0))
    {
        float3 q = abs(p - 0.5 * (GDF_MIN + GDF_MAX)) - 0.5 * (GDF_MAX - GDF_MIN);
        float box = length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
        return max(box, 0.003); // never below the hit epsilon -> box is not a surface
    }
    return min(g.volumes[pc.gdf_sampled].SampleLevel(g.samp, uvw, 0.0).x, 0.6);
}

// Scene distance for the primary march (geometry-with-box-proxy unioned with ground).
float scene_march(float3 p)
{
    return min(geo_march(p), p.y - GROUND_Y);
}

// Scene distance for occlusion (AO / shadows / normals): real geometry only, no box.
float scene_occ(float3 p)
{
    return min(geo_inside(p), p.y - GROUND_Y);
}

// Central-difference gradient of the (occlusion) distance field = surface normal. The
// step matches the GDF voxel size (~1/64): a sub-voxel epsilon would read a single
// trilinear cell and produce faceted normals, so it is widened to average across cells.
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

// Penumbra soft shadow by sphere tracing toward the light (Inigo Quilez); identical
// to the Stage-A march, now over the GDF-backed `scene_dist`.
float soft_shadow(float3 origin, float3 dir, float k)
{
    float res = 1.0;
    float t = 0.02;
    const float MAX_T = 8.0;
    for (int i = 0; i < 96; i++)
    {
        float h = scene_occ(origin + dir * t);
        if (h < 0.001)
            return 0.0;
        res = min(res, k * h / t);
        t += clamp(h, 0.004, 0.1);
        if (t > MAX_T)
            break;
    }
    return saturate(res);
}

// Marched ambient occlusion (Inigo Quilez 5-tap), scaled to the unit-cube scene.
float calc_ao(float3 p, float3 n)
{
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++)
    {
        float h = 0.004 + 0.04 * float(i) / 4.0;
        float d = scene_occ(p + n * h);
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return saturate(1.0 - 3.0 * occ);
}

// Sphere-trace a ray; returns true on a surface hit and writes the hit point.
bool sphere_trace(float3 origin, float3 dir, out float3 hit_p)
{
    const int MAX_STEPS = 256;
    const float MAX_DIST = 12.0;
    const float HIT_EPS = 0.0008;
    float t = 0.001;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        float3 p = origin + dir * t;
        float d = scene_march(p);
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

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 world = mul(pc.inv_view_proj, float4(ndc, 1.0, 1.0));
    float3 dir = normalize(world.xyz / world.w - pc.cam_pos.xyz);

    float3 hit_p;
    float3 color;
    if (sphere_trace(pc.cam_pos.xyz, dir, hit_p))
    {
        float3 n = scene_normal(hit_p);
        // Ground vs baked geometry: whichever term defines the surface here picks the
        // albedo (no material in the GDF — Stage C adds a surface cache).
        bool on_ground = (hit_p.y - GROUND_Y) < geo_inside(hit_p);
        float3 albedo = on_ground ? float3(0.82, 0.82, 0.85) : float3(0.28, 0.55, 0.12);
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
