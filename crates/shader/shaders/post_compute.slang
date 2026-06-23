// Compute post-processing pass (Phase 7): reads the HDR lighting result as a
// sampled texture, applies a 3x3 box blur, and writes the result into a storage
// image (UAV) that the tonemap pass then samples. Demonstrates a compute effect
// wired into the render graph between the lighting and tonemap passes.
//
// Bindless layout (shared `Bindless` block): sampled textures (binding 0) +
// sampler (binding 1) for the input, plus the storage-image array (binding 3) for
// the output, indexed by push constants (b0). Dispatched over 8x8 thread groups.

#include "bindless.slang"

struct PushConstants
{
    uint hdr_index; // bindless sampled index of the HDR lighting target (input)
    uint out_index; // bindless storage-image index of the output target
    uint width;
    uint height;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 texel = 1.0 / float2(pc.width, pc.height);
    float2 uv = (float2(tid.xy) + 0.5) * texel;

    float3 sum = float3(0.0, 0.0, 0.0);
    [unroll]
    for (int dy = -1; dy <= 1; dy++)
    {
        [unroll]
        for (int dx = -1; dx <= 1; dx++)
        {
            float2 o = float2(float(dx), float(dy)) * texel;
            sum += g.textures[pc.hdr_index].SampleLevel(g.samp, uv + o, 0).rgb;
        }
    }
    sum /= 9.0;

    g.storage_images[pc.out_index][tid.xy] = float4(sum, 1.0);
}
