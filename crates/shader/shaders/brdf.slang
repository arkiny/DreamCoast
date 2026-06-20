// BRDF integration LUT: the environment-BRDF split-sum scale/bias terms as a
// function of (N·V, roughness), stored in an Rg16Float 2D texture. pbr.slang
// looks it up to combine with the prefiltered specular: F * lut.x + lut.y.
// Rendered as a single fullscreen pass.

static const float PI = 3.14159265359;

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0; // x = N·V, y = roughness
};

struct PushConstants
{
    // Scalar pads (a uint3 would align to 16 and overrun the 16-byte range).
    uint flip_y;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

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

// Smith geometry with the IBL k = a^2 / 2.
float geometry_schlick_ggx(float n_dot_v, float roughness)
{
    float a = roughness * roughness;
    float k = a * a / 2.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

[shader("fragment")]
float2 fsMain(VSOutput input) : SV_Target
{
    float n_dot_v = max(input.uv.x, 1e-3);
    float roughness = input.uv.y;

    float3 v = float3(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    float3 n = float3(0.0, 0.0, 1.0);

    float a = 0.0;
    float b = 0.0;
    const uint SAMPLE_COUNT = 1024u;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        float2 xi = hammersley(i, SAMPLE_COUNT);
        float3 h = importance_sample_ggx(xi, n, roughness);
        float3 l = normalize(2.0 * dot(v, h) * h - v);

        float n_dot_l = max(l.z, 0.0);
        float n_dot_h = max(h.z, 0.0);
        float v_dot_h = max(dot(v, h), 0.0);
        if (n_dot_l > 0.0)
        {
            float g = geometry_schlick_ggx(n_dot_v, roughness)
                    * geometry_schlick_ggx(n_dot_l, roughness);
            float g_vis = (g * v_dot_h) / max(n_dot_h * n_dot_v, 1e-4);
            float fc = pow(1.0 - v_dot_h, 5.0);
            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }
    return float2(a, b) / float(SAMPLE_COUNT);
}
