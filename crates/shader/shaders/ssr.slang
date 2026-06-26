// Phase 11 Stage C5 — screen-space reflections.
//
// A full-screen compute pass: for each G-buffer pixel it reflects the view ray about
// the world normal and marches that reflection ray through the depth buffer in world
// space (projecting each sample to screen to compare against the stored depth). On a
// hit it samples the shaded HDR color at the hit pixel, so reflective surfaces show the
// *actual neighbouring geometry* — the avocado/cube reflected in the chrome sphere —
// which the captured-cube IBL can't (single probe, parallax + neighbour-reflection
// error). Rays that leave the screen or find no surface miss (confidence 0); Stage C6
// (GDF reflection) fills those, and C7 composites the two over the IBL specular.
//
// Color source: this pass runs AFTER the lighting pass, so it samples the CURRENT
// frame's HDR result (exposure already baked, like the raster path). When SSR later
// feeds back INTO lighting (C7) it must switch to a reprojected previous-frame history
// to avoid the read-before-write cycle; for the standalone C5 viz the current frame is
// simplest and exact.
//
// Output: float4(reflected color, confidence) into a storage image. The deferred
// raster's exposure is already in the sampled HDR, so the tonemap pass applies exposure
// 1.0 to the SSR viz (same as the raster HDR path).

#include "bindless.slang"

struct SsrPush
{
    float4x4 view_proj;     // world -> clip (project ray samples to screen)
    float4x4 inv_view_proj; // clip -> world (surface + scene reconstruction)
    float4 cam_pos;         // xyz camera world position
    uint depth_index;       // sampled G-buffer depth
    uint normal_index;      // sampled G-buffer world normal
    uint material_index;    // sampled G-buffer material (g = roughness)
    uint color_index;       // sampled shaded HDR color (this frame's lighting result)
    uint out_index;         // storage image (reflected color + confidence)
    uint width;
    uint height;
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
    float4 params;          // x = max ray distance, y = thickness, z = step count, w = edge fade
};

[[vk::push_constant]]
ConstantBuffer<SsrPush> pc : register(b0, space0);

float3 reconstruct(float2 uv, float depth)
{
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 h = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    return h.xyz / h.w;
}

// Project a world point to screen UV + NDC depth. Returns false behind the camera.
bool project(float3 p, out float2 uv, out float ndc_z)
{
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float4 clip = mul(pc.view_proj, float4(p, 1.0));
    if (clip.w <= 0.0)
        return false;
    float3 ndc = clip.xyz / clip.w;
    uv = float2(ndc.x * 0.5 + 0.5, ndc.y * sy * 0.5 + 0.5);
    ndc_z = ndc.z;
    return true;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;

    // Background: no surface to reflect from.
    if (depth >= 1.0)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float3 p = reconstruct(uv, depth);
    float3 n = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);
    float3 v = normalize(pc.cam_pos.xyz - p);
    float3 dir = reflect(-v, n); // reflection of the view ray about the normal

    float max_dist = pc.params.x;
    float thickness = pc.params.y;
    int steps = int(pc.params.z);

    bool hit = false;
    float2 hit_uv = uv;
    float t_prev = 0.0;
    float t_hit = 0.0;
    // Linear world-space march, projecting each step to screen to test the depth buffer.
    for (int i = 1; i <= steps; i++)
    {
        float t = max_dist * float(i) / float(steps);
        float3 rp = p + dir * t;
        float2 ruv;
        float rz;
        if (!project(rp, ruv, rz))
            break;
        if (any(ruv < 0.0) || any(ruv > 1.0))
            break; // left the screen
        float sd = g.textures[pc.depth_index].SampleLevel(g.samp, ruv, 0.0).r;
        if (rz > sd)
        {
            // Ray passed behind the stored surface. Accept only if the surface is within
            // `thickness` of the ray point (else it's a far background gap, not a hit).
            float3 scene_p = reconstruct(ruv, sd);
            if (distance(rp, scene_p) < thickness)
            {
                hit = true;
                t_hit = t;
                break;
            }
        }
        t_prev = t;
    }

    if (!hit)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Binary-refine the crossing between the last miss (t_prev) and the hit (t_hit) for
    // a sharper contact, then sample the shaded color there.
    float lo = t_prev;
    float hi = t_hit;
    for (int b = 0; b < 6; b++)
    {
        float mid = 0.5 * (lo + hi);
        float3 rp = p + dir * mid;
        float2 ruv;
        float rz;
        if (project(rp, ruv, rz) && all(ruv >= 0.0) && all(ruv <= 1.0))
        {
            float sd = g.textures[pc.depth_index].SampleLevel(g.samp, ruv, 0.0).r;
            if (rz > sd)
            {
                hi = mid;
                hit_uv = ruv;
            }
            else
            {
                lo = mid;
            }
        }
        else
        {
            hi = mid;
        }
    }

    float3 color = g.textures[pc.color_index].SampleLevel(g.samp, hit_uv, 0.0).rgb;
    // Confidence: fade toward the screen edges (the march cuts off there) so C7 can
    // blend in the GDF / sky fallback without a hard seam.
    float2 e = abs(hit_uv * 2.0 - 1.0);
    float edge = saturate((1.0 - max(e.x, e.y)) / max(pc.params.w, 1e-3));
    g.storage_images[pc.out_index][tid.xy] = float4(color, edge);
}
