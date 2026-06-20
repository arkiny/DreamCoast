// Separable Gaussian blur used to build the bloom chain. Reused for each link:
// the host passes the source bindless index and a blur direction (in uv/texel
// units), so chaining horizontal + vertical + horizontal passes produces a
// progressively blurred image. The first link extracts bright areas via
// `threshold` (later links pass 0).
//
// Bindless layout + the full-screen-triangle vertex shader match post.slang.

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    uint src_index;   // bindless index of the source texture
    uint flip_y;      // 0 = D3D12 (Y up), 1 = Vulkan (Y down)
    float2 direction; // blur step in uv units (e.g. (1/w, 0) or (0, 1/h))
    float threshold;  // brightness knee for the first link (0 = passthrough)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(1, 0)]]
SamplerState g_sampler : register(s0, space0);

[shader("vertex")]
VSOutput vsMain(uint id: SV_VertexID)
{
    float2 uv = float2((id << 1) & 2, id & 2);
    float s = pc.flip_y != 0 ? 1.0 : -1.0;

    VSOutput o;
    o.uv = uv;
    o.pos = float4(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * s, 0.0, 1.0);
    return o;
}

float3 tap(float2 uv, float threshold)
{
    float3 c = g_textures[pc.src_index].Sample(g_sampler, uv).rgb;
    if (threshold > 0.0)
    {
        // Soft-knee bright-pass extraction.
        float luma = dot(c, float3(0.2126, 0.7152, 0.0722));
        c *= saturate(luma - threshold) / max(luma, 1e-4);
    }
    return c;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    // 5-tap Gaussian using linear-sampling offsets/weights.
    float2 uv = input.uv;
    float2 d = pc.direction;
    float3 c = tap(uv, pc.threshold) * 0.2270270270;
    c += tap(uv + d * 1.3846153846, pc.threshold) * 0.3162162162;
    c += tap(uv - d * 1.3846153846, pc.threshold) * 0.3162162162;
    c += tap(uv + d * 3.2307692308, pc.threshold) * 0.0702702703;
    c += tap(uv - d * 3.2307692308, pc.threshold) * 0.0702702703;
    return float4(c, 1.0);
}
