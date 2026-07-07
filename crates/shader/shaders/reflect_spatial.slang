// Track A4b — variance-guided bilateral reflection denoiser (post-temporal, full resolution).
//
// The reference engine's reflection denoiser ends with ONE variance-guided bilateral pass (not a
// multi-pass à-trous): after temporal accumulation, a single edge-aware blur whose LUMINANCE edge
// stop is sized by the per-pixel temporally-accumulated StdDev (A4a) — so it blurs only where the
// signal is still noisy and preserves detail where it has converged. Roughness scales the kernel
// (mirror → radius 0 → passthrough). Edge stops: plane-distance depth, normal-lobe angle, and the
// variance-driven luminance term. Runs entirely in the Reinhard tonemap ("denoiser") space that
// A4a's StdDev is measured in, then converts back to linear for the composite.
//
// Content-gated (`P_REFLECT_SPATIAL`); the gallery never runs it, so the anchor is untouched.

#include "bindless.slang"

struct SpatialPush
{
    float4x4 inv_view_proj; // clip -> world (surface recon)
    float4 cam_pos;         // xyz camera world position
    uint refl_index;        // sampled: temporal output (rgb linear + StdDev in .a, tonemap-luma space)
    uint depth_index;       // sampled: full-res G-buffer depth
    uint normal_index;      // sampled: full-res G-buffer world normal
    uint material_index;    // sampled: full-res G-buffer material (g = roughness)
    uint out_index;         // storage image: denoised reflection (linear, for the composite)
    uint width;
    uint height;
    uint flip_y;            // bit0 = Vulkan Y
    float kernel_radius;    // max bilateral radius in pixels (scaled by saturate(roughness*8))
    float tonemap_range;    // Reinhard denoiser-space range (matches reflect_temporal params.w)
    uint pad0;
    uint pad1;
};

[[vk::push_constant]]
ConstantBuffer<SpatialPush> pc : register(b0, space0);

static const float SP_PI = 3.14159265359;

float3 reconstruct(float2 uv, float depth)
{
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 h = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    return h.xyz / h.w;
}
float sp_luma(float3 v) { return dot(v, float3(0.2126, 0.7152, 0.0722)); }
float3 sp_to_tm(float3 v) { return v / (1.0 + sp_luma(v) * pc.tonemap_range); }
float3 sp_from_tm(float3 v) { return v / max(1.0 - sp_luma(v) * pc.tonemap_range, 1e-3); }
// Reference-engine specular lobe half-angle from roughness (StochasticLightingCommon).
float sp_lobe_half_angle(float r) { return atan(0.75 * r * r / 0.25); }

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
    float roughness = g.textures[pc.material_index].SampleLevel(g.samp, uv, 0.0).g;

    float std_dev = center.a;                     // A4a: per-pixel StdDev (tonemap-luma space)
    float3 center_tm = sp_to_tm(center.rgb);
    float luma_c = sp_luma(center_tm);
    float kernel = pc.kernel_radius * saturate(roughness * 8.0);

    // Passthrough: background, near-mirror (kernel→0), or a converged pixel (relative variance low).
    // Matches the reference gate `StdDev/max(Luma,0.1) > 0.5` + the mirror radius guard.
    if (depth >= 1.0 || kernel < 1.0 || std_dev / max(luma_c, 0.1) <= 0.5)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(center.rgb, 1.0);
        return;
    }

    float3 world_pos = reconstruct(uv, depth);
    float3 n = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);
    float4 plane = float4(n, dot(world_pos, n));
    float lobe_half = sp_lobe_half_angle(roughness) + 0.01;

    float3 sum = center_tm; // accumulate in tonemap space (firefly-stable), convert back at the end
    float wsum = 1.0;
    // 8-sample golden-angle disk — deterministic (cross-backend safe), even coverage.
    const int N = 8;
    [unroll]
    for (int i = 0; i < N; i++)
    {
        float fi = (float(i) + 0.5) / float(N);
        float rr = sqrt(fi);
        float theta = float(i) * 2.3999632; // golden angle
        float2 off = float2(cos(theta), sin(theta)) * rr * kernel;
        float2 nuv = uv + off / res;
        if (any(nuv < 0.0) || any(nuv > 1.0))
            continue;
        float ndepth = g.textures[pc.depth_index].SampleLevel(g.samp, nuv, 0.0).r;
        if (ndepth >= 1.0)
            continue;
        float3 nworld = reconstruct(nuv, ndepth);
        // Depth: plane-distance (reference `exp2(-10000*(planeDist/depth)^2)`).
        float plane_dist = abs(dot(float4(nworld, -1.0), plane));
        float rel = plane_dist / max(depth, 1e-4);
        float depth_w = exp2(-10000.0 * rel * rel);
        // Normal: lobe-angle falloff.
        float3 nn = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, nuv, 0.0).xyz);
        float angle = acos(saturate(dot(n, nn)));
        float normal_w = 1.0 - saturate(angle / lobe_half);
        // Luminance: the variance-guided edge stop — tight where StdDev is small (converged detail),
        // loose where it is large (residual noise).
        float3 ntm = sp_to_tm(g.textures[pc.refl_index].SampleLevel(g.samp, nuv, 0.0).rgb);
        float dl = abs(luma_c - sp_luma(ntm));
        float lum_w = exp2(-dl / max(std_dev, 1e-3));
        float w = depth_w * normal_w * lum_w;
        sum += ntm * w;
        wsum += w;
    }

    float3 outv = sp_from_tm(sum / wsum);
    g.storage_images[pc.out_index][tid.xy] = float4(outv, 1.0);
}
