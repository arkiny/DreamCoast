// Phase 11 Stage D1 (Sponza 60fps) — joint bilateral upsample of half-resolution GDF GI.
//
// The C3 GI trace (gdf_gi.slang) is the dominant Sponza cost, so it is run at half
// resolution (1/4 the rays). This pass reconstructs the full-resolution indirect
// irradiance from the half-res result before the full-res C4 denoiser + lighting consume
// it. For each full-res pixel it gathers the 2x2 half-res texels under it and weights them
// by a joint bilateral edge-stopping function on the full-res G-buffer (world position +
// normal, identical to gdf_atrous.slang), so the upsample keeps GI from bleeding across
// silhouettes / depth discontinuities — the classic guided upsample used for half-res
// effects in commercial engines.
//   Ref: J. Kopf, M. Cohen, D. Lischinski, M. Uyttendaele, "Joint Bilateral Upsampling",
//        ACM SIGGRAPH 2007. (레퍼런스 엔진 SW GI screen-probe / 다른 레퍼런스 엔진 half-res GI integrate the
//        same depth/normal-aware upscale.)
//
// The half-res GI was traced from the SAME full-res depth at each half texel's center, so a
// candidate half texel's guide surface is the full-res depth/normal at that texel's UV.

#include "bindless.slang"

struct GiUpsamplePush
{
    float4x4 inv_view_proj; // clip -> world (edge-stopping reconstruction)
    uint gi_half_index;     // sampled: half-res indirect irradiance (C3 output)
    uint depth_index;       // sampled: full-res G-buffer depth
    uint normal_index;      // sampled: full-res G-buffer world normal
    uint out_index;         // storage image: full-res upsampled irradiance
    uint width;             // full-res target width
    uint height;            // full-res target height
    uint half_width;        // half-res source width
    uint half_height;       // half-res source height
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
    float4 params;          // x = position sigma, y = normal power, z/w unused
};

[[vk::push_constant]]
ConstantBuffer<GiUpsamplePush> pc : register(b0, space0);

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

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;

    // Background: no indirect; matches gdf_gi.slang's far-plane write.
    if (depth >= 1.0)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float3 cp = reconstruct(uv, depth);
    float3 cn = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);

    // 2x2 half-res footprint under this full-res pixel (bilinear support). `hf` is the
    // continuous half-res coordinate; `h0` its lower-left texel.
    float2 half_dim = float2(pc.half_width, pc.half_height);
    float2 hf = uv * half_dim - 0.5;
    int2 h0 = int2(floor(hf));
    float2 frac = hf - float2(h0);

    float sigma_pos = pc.params.x;
    float npow = pc.params.y;

    float3 sum = float3(0.0, 0.0, 0.0);
    float asum = 0.0;
    float wsum = 0.0;
    for (int dy = 0; dy <= 1; dy++)
    {
        for (int dx = 0; dx <= 1; dx++)
        {
            int2 hq = h0 + int2(dx, dy);
            if (hq.x < 0 || hq.y < 0 || hq.x >= int(pc.half_width) || hq.y >= int(pc.half_height))
                continue;
            // The half texel's guide surface = the full-res depth/normal at its center UV
            // (the same UV the half-res trace reconstructed its world point from).
            float2 quv = (float2(hq) + 0.5) / half_dim;
            float qd = g.textures[pc.depth_index].SampleLevel(g.samp, quv, 0.0).r;
            if (qd >= 1.0)
                continue;

            float3 qp = reconstruct(quv, qd);
            float3 qn = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, quv, 0.0).xyz);
            float4 qc4 = g.textures[pc.gi_half_index].SampleLevel(g.samp, quv, 0.0);
            float3 qcol = qc4.rgb;

            // Bilinear spatial weight x edge-stopping (position + normal), as in à-trous.
            float w_bil = (dx == 0 ? 1.0 - frac.x : frac.x) * (dy == 0 ? 1.0 - frac.y : frac.y);
            float w_pos = exp(-distance(cp, qp) / max(sigma_pos, 1e-4));
            float w_nrm = pow(max(dot(cn, qn), 0.0), npow);
            float w = w_bil * w_pos * w_nrm;

            sum += qcol * w;
            asum += qc4.a * w; // carry the input alpha (reflection HIT DISTANCE; AO/GI = 1 → unchanged)
            wsum += w;
        }
    }

    // All taps rejected (thin geometry / silhouette): fall back to a plain bilinear fetch so
    // the pixel still gets indirect light rather than going black.
    float4 fb = g.textures[pc.gi_half_index].SampleLevel(g.samp, uv, 0.0);
    float3 outc = wsum > 1e-5 ? sum / wsum : fb.rgb;
    float outa = wsum > 1e-5 ? asum / wsum : fb.a;
    g.storage_images[pc.out_index][tid.xy] = float4(outc, outa);
}
