// Phase 11 Stage C4 — edge-aware à-trous spatial filter for the GDF GI.
//
// Runs after the temporal pass (gdf_temporal.slang): an à-trous wavelet pass with a
// 5x5 B3-spline kernel, dilated by `step` (run it with step 1, 2, 4… for a wide blur
// at low cost). Each tap is weighted by an edge-stopping function on the G-buffer so
// the low-frequency indirect irradiance is smoothed across surfaces without bleeding
// across geometry boundaries: a world-position term (reconstructed from depth) rejects
// taps on different surfaces, and a normal term rejects taps facing elsewhere. This
// cleans the residual noise the temporal pass leaves on freshly disoccluded / short-
// history pixels; on converged regions the edge weights keep it near-identity.

#include "bindless.slang"

struct AtrousPush
{
    float4x4 inv_view_proj; // clip -> world (edge-stopping reconstruction)
    uint in_index;          // sampled: input irradiance (temporal result, or prior à-trous)
    uint depth_index;       // sampled: G-buffer depth
    uint normal_index;      // sampled: G-buffer world normal
    uint out_index;         // storage image: filtered output
    uint width;
    uint height;
    uint step;              // à-trous tap spacing in texels
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
    float4 params;          // x = position sigma, y = normal power, z/w unused
};

[[vk::push_constant]]
ConstantBuffer<AtrousPush> pc : register(b0, space0);

float3 reconstruct(float2 uv, float depth)
{
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 h = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    return h.xyz / h.w;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 texel = 1.0 / float2(pc.width, pc.height);
    float2 uv = (float2(tid.xy) + 0.5) * texel;
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;

    // Background: pass the value through unfiltered.
    if (depth >= 1.0)
    {
        float3 c = g.textures[pc.in_index].SampleLevel(g.samp, uv, 0.0).rgb;
        g.storage_images[pc.out_index][tid.xy] = float4(c, 1.0);
        return;
    }

    float3 cp = reconstruct(uv, depth);
    float3 cn = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);

    // B3-spline à-trous weights {1,4,6,4,1}/16.
    const float h[5] = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };

    float sigma_pos = pc.params.x;
    float npow = pc.params.y;

    float3 sum = float3(0.0, 0.0, 0.0);
    float wsum = 0.0;
    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            int2 q = int2(tid.xy) + int2(dx, dy) * int(pc.step);
            if (q.x < 0 || q.y < 0 || q.x >= int(pc.width) || q.y >= int(pc.height))
                continue;
            float2 quv = (float2(q) + 0.5) * texel;
            float qd = g.textures[pc.depth_index].SampleLevel(g.samp, quv, 0.0).r;
            if (qd >= 1.0)
                continue; // skip background neighbours

            float3 qcol = g.textures[pc.in_index].SampleLevel(g.samp, quv, 0.0).rgb;
            float3 qp = reconstruct(quv, qd);
            float3 qn = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, quv, 0.0).xyz);

            float w_pos = exp(-distance(cp, qp) / max(sigma_pos, 1e-4));
            float w_nrm = pow(max(dot(cn, qn), 0.0), npow);
            float w = h[dx + 2] * h[dy + 2] * w_pos * w_nrm;

            sum += qcol * w;
            wsum += w;
        }
    }

    float3 outc = wsum > 1e-5 ? sum / wsum : g.textures[pc.in_index].SampleLevel(g.samp, uv, 0.0).rgb;
    g.storage_images[pc.out_index][tid.xy] = float4(outc, 1.0);
}
