// ImGui / textured-2D shader. Also drives the Phase 3 bindless smoke test.
//
// Bindless: a sampled-texture array indexed by a push/root constant, supplied by
// the shared `ParameterBlock` in `bindless.slang` — set 0 / binding 0 (textures)
// + binding 1 (sampler) on SPIR-V, a Metal argument buffer at [[buffer(1)]], and
// a D3D12 descriptor table from one source.

#include "bindless.slang"

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

// Root constants stay at b0; vk::push_constant drives SPIR-V.
[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

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
    return input.col * g.textures[pc.tex_index].Sample(g.samp, input.uv);
}
