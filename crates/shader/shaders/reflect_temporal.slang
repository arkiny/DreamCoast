// Phase 11 Stage C8j — stochastic GDF reflection temporal resolve.
//
// `gdf_reflect.slang` now casts one GGX-importance-sampled ray per pixel per frame (UE Lumen's
// approach to rough reflections — sample the real lobe, then denoise — instead of an image-space
// prefilter that can't average in the off-screen sky a rough lobe sees). This pass accumulates
// those noisy single-ray samples over frames: reproject the surface into the previous frame, read
// the accumulated reflection, validate it (disocclusion reject), and EMA-blend the new sample in.
// On the static screenshot the warm-up converges to the lobe average (a muted glossy reflection
// like the path tracer); on a moving camera the reprojection keeps it stable (SSR-history style).
//
// Radiance is accumulated in a tonemap-weighted space (Karis / UE `RGBToDenoiserSpace`) so a
// bright sample can't dominate the average — the principled version of a per-tap clamp.

#include "bindless.slang"

struct ReflTemporalPush
{
    float4x4 inv_view_proj;  // current clip -> world (surface recon)
    float4x4 prev_view_proj; // world -> previous clip (reprojection)
    float4 cam_pos;          // xyz camera world position
    uint refl_index;         // sampled: this frame's stochastic GDF reflection (rgb)
    uint depth_index;        // sampled: full-res G-buffer depth
    uint out_index;          // storage image: resolved reflection
    uint accum_read;         // storage buffer: previous accumulation (rgb + frame count)
    uint accum_write;
    uint pos_read;           // storage buffer: previous surface world pos (xyz + valid)
    uint pos_write;
    uint width;
    uint height;
    uint flip_y;             // bit0 = Vulkan Y
    uint reset;              // 1 = ignore history (first frame / view change)
    uint material_index;     // sampled G-buffer material (g = roughness) for the spatial radius
    float4 params;           // x = reject dist, y = max history len, z = firefly clamp, w = tonemap range
};

[[vk::push_constant]]
ConstantBuffer<ReflTemporalPush> pc : register(b0, space0);

float3 reconstruct(float2 uv, float depth)
{
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 h = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    return h.xyz / h.w;
}
// Tonemap-weighted space for stable HDR averaging (UE RGBToDenoiserSpace).
float3 to_tm(float3 v) { return v / (1.0 + dot(v, float3(0.2126, 0.7152, 0.0722)) * pc.params.w); }
float3 from_tm(float3 v)
{
    float l = dot(v, float3(0.2126, 0.7152, 0.0722));
    return v / max(1.0 - l * pc.params.w, 1e-3);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    uint idx = tid.y * pc.width + tid.x;
    float2 res = float2(pc.width, pc.height);
    float2 uv = (float2(tid.xy) + 0.5) / res;
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;

    if (depth >= 1.0)
    {
        // Background: copy the reflection straight through (the sky along the view ray).
        float3 bg = g.textures[pc.refl_index].SampleLevel(g.samp, uv, 0.0).rgb;
        g.storage_images[pc.out_index][tid.xy] = float4(bg, 1.0);
        g.storage_buffers[pc.accum_write].Store4(idx * 16u, asuint(float4(to_tm(bg), 1.0)));
        g.storage_buffers[pc.pos_write].Store4(idx * 16u, asuint(float4(0, 0, 0, 0)));
        return;
    }

    float3 world_pos = reconstruct(uv, depth);
    // Spatial reconstruction: average the current single-ray GGX samples over a roughness-scaled,
    // depth-weighted neighbourhood (each neighbour pixel traced a DIFFERENT GGX ray) to cut the
    // per-pixel variance before temporal accumulation — UE resolves spatially + temporally. Sharp
    // surfaces (chrome) use radius 0 (no blur).
    float roughness = g.textures[pc.material_index].SampleLevel(g.samp, uv, 0.0).g;
    int sr = clamp(int(roughness * 8.0), 0, 3);
    float2 texel = 1.0 / res;
    float3 ssum = g.textures[pc.refl_index].SampleLevel(g.samp, uv, 0.0).rgb;
    float swsum = 1.0;
    for (int dy = -sr; dy <= sr; dy++)
    {
        for (int dx = -sr; dx <= sr; dx++)
        {
            if (dx == 0 && dy == 0)
                continue;
            float2 nuv = uv + float2(dx, dy) * texel;
            float dz = g.textures[pc.depth_index].SampleLevel(g.samp, nuv, 0.0).r;
            if (dz >= 1.0 || abs(dz - depth) > 0.003)
                continue; // different surface — would bleed across the silhouette
            ssum += g.textures[pc.refl_index].SampleLevel(g.samp, nuv, 0.0).rgb;
            swsum += 1.0;
        }
    }
    float3 sample_tm = to_tm(ssum / swsum);

    float3 accum = sample_tm;
    float len = 1.0;
    if (pc.reset == 0u)
    {
        float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
        float4 pclip = mul(pc.prev_view_proj, float4(world_pos, 1.0));
        if (pclip.w > 0.0)
        {
            float3 pndc = pclip.xyz / pclip.w;
            float2 puv = float2(pndc.x * 0.5 + 0.5, pndc.y * sy * 0.5 + 0.5);
            if (all(puv >= 0.0) && all(puv <= 1.0))
            {
                int2 ppix = clamp(int2(floor(puv * res)), int2(0, 0),
                                  int2(int(pc.width) - 1, int(pc.height) - 1));
                uint pidx = uint(ppix.y) * pc.width + uint(ppix.x);
                float4 ph = asfloat(g.storage_buffers[pc.accum_read].Load4(pidx * 16u));
                float4 pp = asfloat(g.storage_buffers[pc.pos_read].Load4(pidx * 16u));
                if (pp.w > 0.5 && distance(world_pos, pp.xyz) < pc.params.x)
                {
                    len = min(ph.a + 1.0, pc.params.y);
                    accum = lerp(ph.rgb, sample_tm, 1.0 / len);
                }
            }
        }
    }

    // Sanitize before storing: a NaN would persist in the accumulator and spread via EMA.
    accum = make_finite(accum);
    g.storage_buffers[pc.accum_write].Store4(idx * 16u, asuint(float4(accum, len)));
    g.storage_buffers[pc.pos_write].Store4(idx * 16u, asuint(float4(world_pos, 1.0)));

    float3 resolved = from_tm(accum);
    // Hue-preserving firefly clamp on the resolved radiance.
    float m = max(resolved.r, max(resolved.g, resolved.b));
    if (m > pc.params.z)
        resolved *= pc.params.z / m;
    g.storage_images[pc.out_index][tid.xy] = float4(resolved, 1.0);
}
