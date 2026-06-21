// Compute post-processing pass (Phase 7): reads the HDR lighting result as a
// sampled texture, applies a 3x3 box blur, and writes the result into a storage
// image (UAV) that the tonemap pass then samples. Demonstrates a compute effect
// wired into the render graph between the lighting and tonemap passes.
//
// Bindless layout extends post.slang: sampled textures (t0/binding 0) + sampler
// (s0/binding 1) for the input, plus a storage-image array (u0/binding 3) for the
// output, indexed by push constants (b0). Dispatched over 8x8 thread groups.

struct PushConstants
{
    uint hdr_index; // bindless sampled index of the HDR lighting target (input)
    uint out_index; // bindless storage-image index of the output target
    uint width;
    uint height;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(1, 0)]]
SamplerState g_sampler : register(s0, space0);

[[vk::binding(3, 0)]]
RWTexture2D<float4> g_storage_images[] : register(u0, space0);

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
            sum += g_textures[pc.hdr_index].SampleLevel(g_sampler, uv + o, 0).rgb;
        }
    }
    sum /= 9.0;

    g_storage_images[pc.out_index][tid.xy] = float4(sum, 1.0);
}
