// Diffuse irradiance convolution: for each output cube direction (a surface
// normal), integrate the cosine-weighted hemisphere of the environment cubemap.
// The result, sampled by a normal in pbr.slang, gives the diffuse IBL term.
// Rendered one fullscreen draw per cube face.

static const float PI = 3.14159265359;

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    uint face;      // cube face 0..5
    uint flip_y;    // 0 = D3D12, 1 = Vulkan
    uint env_index; // bindless cube index of the source environment map
    uint pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[[vk::binding(1, 0)]]
SamplerState g_sampler : register(s0, space0);

[[vk::binding(2, 0)]]
TextureCube g_cubes[] : register(t0, space1);

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

float3 cube_direction(uint face, float2 uv)
{
    float u = uv.x * 2.0 - 1.0;
    float v = uv.y * 2.0 - 1.0;
    float3 dir;
    switch (face)
    {
    case 0: dir = float3(1.0, -v, -u); break;
    case 1: dir = float3(-1.0, -v, u); break;
    case 2: dir = float3(u, 1.0, v); break;
    case 3: dir = float3(u, -1.0, -v); break;
    case 4: dir = float3(u, -v, 1.0); break;
    default: dir = float3(-u, -v, -1.0); break;
    }
    return normalize(dir);
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float3 n = cube_direction(pc.face, input.uv);
    float3 up = abs(n.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0);
    float3 right = normalize(cross(up, n));
    up = normalize(cross(n, right));

    float3 irradiance = float3(0.0);
    float num_samples = 0.0;
    // Sample a pre-blurred env mip so a coarser hemisphere step stays low-noise
    // (keeps the per-frame real-time cost down).
    const float delta = 0.1;
    const float src_mip = 2.0;
    for (float phi = 0.0; phi < 2.0 * PI; phi += delta)
    {
        for (float theta = 0.0; theta < 0.5 * PI; theta += delta)
        {
            // Tangent-space sample -> world.
            float3 tangent = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            float3 sample_dir = tangent.x * right + tangent.y * up + tangent.z * n;
            float3 c = g_cubes[pc.env_index].SampleLevel(g_sampler, sample_dir, src_mip).rgb;
            irradiance += c * cos(theta) * sin(theta);
            num_samples += 1.0;
        }
    }
    irradiance = PI * irradiance / max(num_samples, 1.0);
    return float4(irradiance, 1.0);
}
