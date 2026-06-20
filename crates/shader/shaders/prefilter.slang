// Specular prefilter: for each output cube direction (= reflection vector, with
// the split-sum approximation N = V = R), GGX-importance-sample the environment
// at a given roughness. Each prefilter mip holds a successively rougher
// convolution; pbr.slang samples it at a roughness-derived LOD. Rendered one
// fullscreen draw per (face, mip).

static const float PI = 3.14159265359;

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    uint face;
    uint flip_y;
    uint env_index;
    float roughness;
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

// Van der Corput radical inverse + Hammersley point set.
float radical_inverse_vdc(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float2 hammersley(uint i, uint n)
{
    return float2(float(i) / float(n), radical_inverse_vdc(i));
}

// GGX importance-sampled half-vector around normal `n` for `roughness`.
float3 importance_sample_ggx(float2 xi, float3 n, float roughness)
{
    float a = roughness * roughness;
    float phi = 2.0 * PI * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    float3 h = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

    float3 up = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, n));
    float3 bitangent = cross(n, tangent);
    return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float3 n = cube_direction(pc.face, input.uv);
    float3 v = n;

    const uint SAMPLE_COUNT = 256u;
    float3 prefiltered = float3(0.0);
    float total_weight = 0.0;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        float2 xi = hammersley(i, SAMPLE_COUNT);
        float3 h = importance_sample_ggx(xi, n, pc.roughness);
        float3 l = normalize(2.0 * dot(v, h) * h - v);
        float n_dot_l = dot(n, l);
        if (n_dot_l > 0.0)
        {
            prefiltered += g_cubes[pc.env_index].SampleLevel(g_sampler, l, 0.0).rgb * n_dot_l;
            total_weight += n_dot_l;
        }
    }
    prefiltered /= max(total_weight, 1e-4);
    return float4(prefiltered, 1.0);
}
