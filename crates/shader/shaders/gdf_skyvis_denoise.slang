// F6O — sky-visibility spatial denoise (runs at the 1/div TRACE resolution).
//
// The per-pixel sky-vis trace (gdf_skyvis.slang) uses few cosine rays, so its scalar V is
// noisy (Monte-Carlo, std ~0.18 at 8 rays). This pass edge-aware-blurs V over a 5x5
// low-res-texel neighbourhood BEFORE the deferred upscales it, so the wide kernel runs at
// 1/div^2 the pixels (~10x cheaper than the same kernel inline in the full-res lighting pass,
// which cost ~8 ms at Retina). The world-position edge-stop keeps the blur within one surface
// (no cross-silhouette bleed); the deferred then does the final cheap 2x2 edge-aware upscale.
// Output stays float4(V, 0, 0, 0) — scalar V in .r, bent 0 (increment-1 contract).

#include "bindless.slang"

struct SkyvisDenoisePush
{
    uint src_index;      // sampled: the raw low-res sky-vis V (in .r)
    uint position_index; // sampled: full-res world-position G-buffer (read at low-res UVs)
    uint out_index;      // storage image: denoised low-res V
    uint width;          // low-res (trace) width
    uint height;         // low-res height
    uint vol_index;      // sampled: the smooth probe-VOLUME sky-vis V (0xFFFFFFFF = none, no blend)
    uint pad0;
    uint pad1;
    float4 cam_pos;      // xyz camera world position (for the distance blend to the volume V)
};

[[vk::push_constant]]
ConstantBuffer<SkyvisDenoisePush> pc : register(b0, space0);

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 res = float2(pc.width, pc.height);
    float2 uv = (float2(tid.xy) + 0.5) / res;
    float3 p_c = g.textures[pc.position_index].SampleLevel(g.samp, uv, 0.0).xyz;
    float v_c = g.textures[pc.src_index].SampleLevel(g.samp, uv, 0.0).r;

    float vsum = 0.0;
    float v2sum = 0.0;
    float wsum = 0.0;
    [unroll]
    for (int dy = -2; dy <= 2; dy++)
    {
        [unroll]
        for (int dx = -2; dx <= 2; dx++)
        {
            float2 tuv = (float2(tid.xy) + float2(dx, dy) + 0.5) / res;
            float sw = exp(-float(dx * dx + dy * dy) * 0.35); // spatial Gaussian
            float tv = g.textures[pc.src_index].SampleLevel(g.samp, tuv, 0.0).r;
            float3 tp = g.textures[pc.position_index].SampleLevel(g.samp, tuv, 0.0).xyz;
            float3 dp = tp - p_c;
            float ew = exp(-dot(dp, dp) * 4.0); // world-pos edge-stop (~0.5 m falloff)
            float ww = sw * ew;
            vsum += tv * ww;
            v2sum += tv * tv * ww;
            wsum += ww;
        }
    }
    float v = wsum > 1e-4 ? vsum / wsum : v_c;
    // Local V variance over the edge-stopped neighbourhood: HIGH where the per-pixel trace is
    // aliasing a thin bright aperture (door slats + sky gaps → extreme texel-to-texel V) or is
    // un-converged during motion; LOW on a converged smooth surface.
    float mean = wsum > 1e-4 ? vsum / wsum : 0.0;
    float variance = wsum > 1e-4 ? max(v2sum / wsum - mean * mean, 0.0) : 0.0;

    // Distance blend to the smooth probe-VOLUME V. The per-pixel trace fixes the NEAR "검은 Block"
    // (probe-grid quantization visible up close), but at FAR distance a thin backlit feature (a
    // door's slats + sky gaps) is undersampled at 1/div res → aliased speckle even after temporal
    // convergence. The volume V is smooth there (its grid projects tiny), so fade per-pixel → volume
    // as view distance grows: near = sharp per-pixel (no blocks), far = smooth volume (no speckle).
    if (pc.vol_index != 0xFFFFFFFFu)
    {
        float vol = g.textures[pc.vol_index].SampleLevel(g.samp, uv, 0.0).r;
        // VARIANCE-guided blend to the smooth volume V: where the per-pixel V has high local
        // variance (thin-aperture aliasing OR motion noise) fall back to the smooth volume; where
        // it is a clean converged surface (low variance) keep the sharp per-pixel V (so the near
        // "검은 Block" the volume has stays fixed). Distance-independent — the block-prone and the
        // aliasing-prone surfaces can share a distance, but not a variance signature.
        float t_var = saturate(variance / 0.010);
        // Plus a gentle far-distance fade (very distant thin geometry undersamples even at low
        // variance once it is sub-texel).
        float d = distance(p_c, pc.cam_pos.xyz);
        float t_dist = saturate((d - 14.0) / 12.0);
        v = lerp(v, vol, max(t_var, t_dist));
    }
    g.storage_images[pc.out_index][tid.xy] = float4(v, 0.0, 0.0, 0.0);
}
