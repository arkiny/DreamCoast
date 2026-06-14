// Mesh shader: diffuse (lambert) lighting + bindless base-color texture.
//
// The model matrix is identity (the sandbox orbits the camera, not the model),
// so input normals are already world-space.

struct VSInput
{
    float3 pos : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
};

struct VSOutput
{
    float4 pos : SV_Position;
    float3 normal : NORMAL0;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    float4x4 mvp;
    uint tex_index;
};

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
    o.pos = mul(pc.mvp, float4(input.pos, 1.0));
    o.normal = input.normal;
    o.uv = input.uv;
    return o;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float3 n = normalize(input.normal);
    float3 l = normalize(float3(0.4, 0.8, 0.4));
    float diffuse = max(dot(n, l), 0.0) * 0.85 + 0.15;
    float4 base = g_textures[pc.tex_index].Sample(g_sampler, input.uv);
    return float4(base.rgb * diffuse, base.a);
}
