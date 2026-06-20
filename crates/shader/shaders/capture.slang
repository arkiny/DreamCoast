// Forward capture shader: renders scene geometry into the environment cubemap
// faces from the capture point (the camera) with simple direct lighting only — no
// IBL, so the captured environment can feed IBL without recursion. Used so
// reflective surfaces reflect the live scene (camera-based real-time capture).

// Position + normal only (uv unused); matches the position+normal vertex layout
// so the input signature aligns on both backends.
struct VSInput
{
    float3 pos : POSITION;
    float3 normal : NORMAL;
};

struct VSOutput
{
    float4 clip : SV_Position;
    float3 normal : NORMAL0;
};

struct PushConstants
{
    float4x4 mvp;       // face view-projection (model is identity)
    float4 base_color;  // rgb albedo
    float4 sun;         // xyz = direction TO the sun, w = intensity
    float4 misc;        // x = ambient
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[shader("vertex")]
VSOutput vsMain(VSInput input)
{
    VSOutput o;
    o.clip = mul(pc.mvp, float4(input.pos, 1.0));
    o.normal = input.normal;
    return o;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float3 n = normalize(input.normal);
    float n_dot_l = max(dot(n, normalize(pc.sun.xyz)), 0.0);
    float3 color = pc.base_color.rgb * (pc.misc.x + n_dot_l * pc.sun.w);
    return float4(color, 1.0);
}
