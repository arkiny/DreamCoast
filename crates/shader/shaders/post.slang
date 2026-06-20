// Composite pass: combines the offscreen scene color with the blurred bloom
// chain result and applies a selectable tonemap/effect, writing to the
// backbuffer. The final node of the render graph's color chain.
//
// Bindless layout mirrors imgui.slang: a sampled-texture array (t0/binding 0) +
// a static sampler (s0/binding 1), indexed by root/push constants (b0). The
// vertex shader synthesizes a full-screen triangle from SV_VertexID; `flip_y`
// selects clip-space Y orientation per backend (D3D12 = 0, Vulkan = 1).

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    uint scene_index; // bindless index of the scene color target
    uint bloom_index; // bindless index of the blurred bloom target
    uint mode;        // 0 = none, 1 = grayscale, 2 = vignette
    uint flip_y;      // 0 = D3D12 (Y up), 1 = Vulkan (Y down)
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

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float3 scene = g_textures[pc.scene_index].Sample(g_sampler, input.uv).rgb;
    float3 bloom = g_textures[pc.bloom_index].Sample(g_sampler, input.uv).rgb;
    float3 color = scene + bloom * 0.6; // additive bloom

    if (pc.mode == 1)
    {
        float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
        color = float3(luma, luma, luma);
    }
    else if (pc.mode == 2)
    {
        float2 d = input.uv - 0.5;
        float vignette = saturate(1.0 - dot(d, d) * 1.8);
        color *= vignette;
    }

    return float4(color, 1.0);
}
