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
// Color source — two modes (flip_y bit1):
//   * Current-HDR (bit1=0, the standalone C5 viz): runs after lighting and samples this
//     frame's HDR at the hit (exposure already baked, like the raster path); the tonemap
//     applies exposure 1.0.
//   * Reprojected history (bit1=1, C7b, feeds lighting): samples the PREVIOUS frame's
//     raw-radiance lit-color history buffer at the reprojected world hit point, so SSR can
//     feed back into the lighting specular (C7c) without a read-before-write cycle. The
//     reprojection uses `globals.prev_view_proj`; the result is raw radiance.
//
// Output: float4(reflected color, confidence) into a storage image.

#include "bindless.slang"

struct SsrPush
{
    float4x4 view_proj;     // world -> clip (project ray samples to screen)
    float4x4 inv_view_proj; // clip -> world (surface + scene reconstruction)
    float4 cam_pos;         // xyz camera world position
    uint depth_index;       // sampled G-buffer depth
    uint normal_index;      // sampled G-buffer world normal
    uint hist_index;        // history mode: storage buffer (raw-radiance prev-frame color)
    uint color_index;       // current-HDR mode: sampled shaded HDR color (this frame)
    uint out_index;         // storage image (reflected color + confidence)
    uint width;
    uint height;
    uint flip_y;            // bit0: 1 = Vulkan (Y down); bit1: 1 = reprojected history mode
    float4 params;          // x = max ray distance, y = thickness, z = step count, w = edge fade
};

[[vk::push_constant]]
ConstantBuffer<SsrPush> pc : register(b0, space0);

// Per-frame globals (set 1 / b1) — the same UBO the deferred PBR pass binds, opted into
// via the compute pipeline's `uniform_buffer`. Only `prev_view_proj` (appended at the end)
// is read here, for the history reprojection; the layout must match the host `Globals`.
struct Globals
{
    float4 camera_pos;
    float4 sun_direction;
    float4 sun_color;
    float4 ambient;
    int4 counts;
    float4 point_pos[4];
    float4 point_color[4];
    float4x4 light_view_proj;
    float4 shadow;
    float4x4 inv_view_proj;
    int4 ibl;
    float4 probe;
    float4 probe_box_min;
    float4 probe_box_max;
    float4x4 prev_view_proj; // world -> previous clip
};

[[vk::binding(0, 1)]]
ConstantBuffer<Globals> globals : register(b1, space0);

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

    float3 color;
    if ((pc.flip_y & 2u) != 0u)
    {
        // History mode: reproject the world hit point into the previous frame and read the
        // raw-radiance lit-color history (nearest). `hi` is the binary-refined hit distance.
        float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
        float3 hp = p + dir * hi;
        float4 pclip = mul(globals.prev_view_proj, float4(hp, 1.0));
        if (pclip.w <= 0.0)
        {
            g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
            return;
        }
        float3 pndc = pclip.xyz / pclip.w;
        float2 puv = float2(pndc.x * 0.5 + 0.5, pndc.y * sy * 0.5 + 0.5);
        if (any(puv < 0.0) || any(puv > 1.0))
        {
            // Off-screen last frame: no history there -> miss, let the GDF fallback fill it.
            g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
            return;
        }
        int2 ppix = clamp(int2(floor(puv * float2(pc.width, pc.height))), int2(0, 0),
                          int2(int(pc.width) - 1, int(pc.height) - 1));
        uint pidx = uint(ppix.y) * pc.width + uint(ppix.x);
        color = asfloat(g.storage_buffers[pc.hist_index].Load4(pidx * 16u)).rgb;

        // Neighborhood clamp (flip_y bit2): bound the reprojected history sample to the
        // min/max of its 8 neighbours, pulling an isolated reprojected firefly back into the
        // local reflection range. Canonical for temporal reprojection.
        if ((pc.flip_y & 4u) != 0u)
        {
            float3 mn = float3(1e30, 1e30, 1e30);
            float3 mx = float3(-1e30, -1e30, -1e30);
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0)
                        continue;
                    int2 q = clamp(ppix + int2(dx, dy), int2(0, 0),
                                   int2(int(pc.width) - 1, int(pc.height) - 1));
                    float3 c = asfloat(g.storage_buffers[pc.hist_index].Load4(
                                           (uint(q.y) * pc.width + uint(q.x)) * 16u))
                                   .rgb;
                    mn = min(mn, c);
                    mx = max(mx, c);
                }
            }
            color = clamp(color, mn, mx);
        }
    }
    else
    {
        color = g.textures[pc.color_index].SampleLevel(g.samp, hit_uv, 0.0).rgb;
    }
    // Confidence: fade toward the screen edges (the march cuts off there) so C7 can
    // blend in the GDF / sky fallback without a hard seam.
    float2 e = abs(hit_uv * 2.0 - 1.0);
    float edge = saturate((1.0 - max(e.x, e.y)) / max(pc.params.w, 1e-3));
    g.storage_images[pc.out_index][tid.xy] = float4(color, edge);
}
