// Phase 11 Stage C8g — GDF-reflection roughness mip pyramid (downsample pass).
//
// An energy-conserving 4x box downsample (each pass quarters each dimension). Chaining a few
// passes builds a mip pyramid of the GDF reflection image; the composite then samples it at a
// roughness-derived LOD (trilinear across two levels) instead of a wide screen-space gather —
// the way UE prefilters its reflection captures / skylight into roughness mips. Averaging is
// energy-conserving, so a bright pixel integrates into a soft highlight rather than being
// replicated as a fixed-pattern speckle (the gather's failure mode).
//
// Each destination texel covers a 4x4 source block; four bilinear taps (one per 2x2 quadrant)
// average the 16 source texels.

#include "bindless.slang"

struct DownsamplePush
{
    uint src_index;   // sampled source image (the previous, higher-res level)
    uint out_index;   // storage image (this level, 1/4 the area)
    uint dst_width;   // this level's dims
    uint dst_height;
};

[[vk::push_constant]]
ConstantBuffer<DownsamplePush> pc : register(b0, space0);

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dst_width || tid.y >= pc.dst_height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.dst_width, pc.dst_height);
    // Source texel size in UV (the source is 4x this level per dimension).
    float2 stex = 1.0 / (float2(pc.dst_width, pc.dst_height) * 4.0);

    float3 c = g.textures[pc.src_index].SampleLevel(g.samp, uv + stex * float2(-1.0, -1.0), 0.0).rgb;
    c += g.textures[pc.src_index].SampleLevel(g.samp, uv + stex * float2(1.0, -1.0), 0.0).rgb;
    c += g.textures[pc.src_index].SampleLevel(g.samp, uv + stex * float2(-1.0, 1.0), 0.0).rgb;
    c += g.textures[pc.src_index].SampleLevel(g.samp, uv + stex * float2(1.0, 1.0), 0.0).rgb;
    g.storage_images[pc.out_index][tid.xy] = float4(c * 0.25, 1.0);
}
