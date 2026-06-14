// ImGui / textured-2D shader. Also drives the Phase 3 bindless smoke test.
//
// Bindless: a sampled-texture array indexed by a push/root constant. On SPIR-V
// the [[vk::binding]] attributes place the array at set 0 binding 0 and the
// sampler at binding 1; on DXIL Slang assigns t0 (table), s0 (static sampler),
// and b0 (root constants).

struct VSInput
{
    float2 pos : POSITION;
    float2 uv : TEXCOORD;
    float4 col : COLOR;
};

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
    float4 col : COLOR0;
};

struct PushConstants
{
    float2 scale;
    float2 translate;
    uint tex_index;
};

// Registers are pinned for D3D12 (must match the root signature: SRV table t0,
// static sampler s0, root constants b0); vk::binding/vk::push_constant drive SPIR-V.
[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(1, 0)]]
SamplerState g_sampler : register(s0, space0);

[shader("vertex")]
VSOutput vsMain(VSInput input)
{
    VSOutput o;
    o.pos = float4(input.pos * pc.scale + pc.translate, 0.0, 1.0);
    o.uv = input.uv;
    o.col = input.col;
    return o;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    return input.col * g_textures[pc.tex_index].Sample(g_sampler, input.uv);
}
