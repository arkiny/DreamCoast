// Track A1 — stochastic GDF-reflection spatial resolve (ratio estimator), at TRACE resolution.
//
// `gdf_reflect.slang` casts ONE GGX-importance-sampled ray per pixel per frame. The legacy denoise
// was a LINEAR box average of that neighbourhood inside `reflect_temporal.slang` — which blurs the
// glossy reflection to the cache-resolution look. This pass instead reconstructs each neighbour's
// exact ray direction (from the same deterministic per-pixel jitter — `reflect_ggx.slang`) and
// reweights the borrowed sample by the RATIO `pdf_p(dir)/pdf_q`: how likely the CENTRE pixel would
// itself have drawn that direction under its own GGX lobe. This is Stachowiak's unbiased ratio
// estimator (identical scheme to `ssr_resolve.slang` for screen traces): many effective samples per
// pixel from one ray each, roughness-adaptive for free (a sharp lobe weights only near-mirror
// neighbours → stays crisp; a wide lobe pulls in the kernel → denoises) — so it sharpens the glossy
// reflection instead of blurring it.
//
// Runs at the trace resolution BEFORE the bilateral upsample (the ray directions are only coherent
// at trace res). Near-mirror pixels (roughness <= mirror_thresh, which the trace draws with a
// DETERMINISTIC mirror ray, not GGX) pass straight through — they carry high-frequency detail a
// gather would destroy. Stateless: no history buffers (the temporal EMA stays in reflect_temporal).
//
// Content-gated (`P_REFLECT_RESOLVE`); the gallery never runs this pass, so its anchor is untouched.

#include "bindless.slang"
#include "reflect_ggx.slang"

struct ResolvePush
{
    float4x4 inv_view_proj; // current clip -> world (surface recon), full-res camera
    float4 cam_pos;         // xyz camera world position
    uint refl_index;        // sampled: this frame's stochastic GDF reflection (rgb + hit_dist in .a)
    uint depth_index;       // sampled: full-res G-buffer depth
    uint normal_index;      // sampled: full-res G-buffer world normal
    uint material_index;    // sampled: full-res G-buffer material (g = roughness)
    uint out_index;         // storage image: resolved reflection (trace res)
    uint width;             // trace res
    uint height;
    uint flip_y;            // bit0 = Vulkan Y
    uint frame;             // the jitter frame the trace used (must match gdf_reflect's `frame`)
    float mirror_thresh;    // roughness <= this => deterministic mirror ray (passthrough, no gather)
    float kernel_radius;    // max spatial gather radius in texels (roughness-scaled below)
    uint stochastic;        // A5: 0 = white jitter, 1 = blue-noise — must match gdf_reflect's sampler
};

[[vk::push_constant]]
ConstantBuffer<ResolvePush> pc : register(b0, space0);

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

    float2 res = float2(pc.width, pc.height);
    float2 uv = (float2(tid.xy) + 0.5) / res;
    float4 center = g.textures[pc.refl_index].SampleLevel(g.samp, uv, 0.0);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;

    // Background (sky along the view ray) or near-mirror surface: pass the traced value through
    // unchanged. A mirror ray is deterministic + sharp; a ratio gather would blur it.
    float roughness = g.textures[pc.material_index].SampleLevel(g.samp, uv, 0.0).g;
    if (depth >= 1.0 || roughness <= pc.mirror_thresh)
    {
        g.storage_images[pc.out_index][tid.xy] = center;
        return;
    }

    float3 world_pos = reconstruct(uv, depth);
    float3 n = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);
    float3 v = normalize(pc.cam_pos.xyz - world_pos);

    // Roughness-adaptive kernel: a sharp lobe wants only its immediate neighbours (a wide kernel
    // mixes in off-lobe neighbours whose pdf_p ~ 0, adding noise); a wide lobe pulls the whole
    // neighbourhood to denoise. Matches ssr_resolve's radius law.
    int rmax = int(pc.kernel_radius);
    int r = clamp(int(ceil(roughness * float(rmax) * 3.0)), 1, rmax);
    float3 num = float3(0, 0, 0);
    float den = 0.0;
    for (int dy = -r; dy <= r; dy++)
    {
        for (int dx = -r; dx <= r; dx++)
        {
            int2 np = int2(tid.xy) + int2(dx, dy);
            if (np.x < 0 || np.y < 0 || np.x >= int(pc.width) || np.y >= int(pc.height))
                continue;
            float2 nuv = (float2(np) + 0.5) / res;
            float ndepth = g.textures[pc.depth_index].SampleLevel(g.samp, nuv, 0.0).r;
            if (ndepth >= 1.0 || abs(ndepth - depth) > 0.003)
                continue; // different surface / background — would bleed across the silhouette
            float nrough = g.textures[pc.material_index].SampleLevel(g.samp, nuv, 0.0).g;
            if (nrough <= pc.mirror_thresh)
                continue; // neighbour cast a deterministic mirror ray (delta) — not reusable
            float3 nq = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, nuv, 0.0).xyz);
            float3 nworld = reconstruct(nuv, ndepth);
            float3 vq = normalize(pc.cam_pos.xyz - nworld);
            // Reconstruct the exact ray this neighbour cast (same deterministic jitter as the trace).
            float3 dir_q = refl_ggx_dir(nq, vq, nrough, uint(np.x), uint(np.y), pc.frame, pc.stochastic);
            if (dot(n, dir_q) <= 0.0)
                continue; // not in the centre pixel's upper hemisphere
            float pdf_q = refl_ggx_pdf(nq, vq, dir_q, nrough);
            float pdf_p = refl_ggx_pdf(n, v, dir_q, roughness);
            float w = pdf_p / max(pdf_q, 1e-9); // ratio: how likely p would draw this borrowed ray
            float3 ncol = g.textures[pc.refl_index].SampleLevel(g.samp, nuv, 0.0).rgb;
            num += ncol * w;
            den += w;
        }
    }

    // Keep the centre pixel's hit distance in .a (the temporal pass's virtual-image reproject needs
    // it). If no neighbour contributed (every kernel ray off-lobe), fall back to the centre sample.
    float3 resolved = den > 1e-6 ? num / den : center.rgb;
    g.storage_images[pc.out_index][tid.xy] = float4(resolved, center.a);
}
