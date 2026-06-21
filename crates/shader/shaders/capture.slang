// Forward capture shader: renders scene geometry into the environment cubemap
// faces from the capture point (the camera). Surfaces are shaded with direct sun
// light plus an optional image-based ambient term sampled from the *previous*
// frame's convolved cubes (irradiance + prefilter + BRDF LUT). Feeding the prior
// frame's environment back into the capture lets reflective surfaces reflect
// other reflective surfaces (multi-bounce) without recursion or a stall: the
// double-buffered probe converges over a few frames. When no previous cubes are
// bound (irradiance index < 0) it falls back to flat ambient (single-bounce).
//
// Output is linear HDR radiance (no exposure — the main lighting pass applies
// exposure once when it samples this environment).

static const float PI = 3.14159265359;

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
    float3 world_pos : TEXCOORD0;
};

struct PushConstants
{
    float4x4 mvp;       // face view-projection * model
    float4x4 model;     // model -> world (for the world position)
    float4 base_color;  // rgb albedo
    float4 sun;         // xyz = direction TO the sun, w = intensity
    float4 misc;        // x = ambient, y = roughness, z = metallic, w = prefilter max LOD
    float4 eye;         // xyz = capture/camera position
    int4 ibl;           // x irradiance, y prefilter, z BRDF LUT (-1 = no previous env)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(1, 0)]]
SamplerState g_sampler : register(s0, space0);

[[vk::binding(2, 0)]]
TextureCube g_cubes[] : register(t0, space1);

[shader("vertex")]
VSOutput vsMain(VSInput input)
{
    VSOutput o;
    o.clip = mul(pc.mvp, float4(input.pos, 1.0));
    o.world_pos = mul(pc.model, float4(input.pos, 1.0)).xyz;
    o.normal = mul((float3x3)pc.model, input.normal);
    return o;
}

float distribution_ggx(float n_dot_h, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

float geometry_schlick_ggx(float n_dot_v, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

float geometry_smith(float n_dot_v, float n_dot_l, float roughness)
{
    return geometry_schlick_ggx(n_dot_v, roughness)
         * geometry_schlick_ggx(n_dot_l, roughness);
}

float3 fresnel_schlick(float cos_theta, float3 f0)
{
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

float3 fresnel_schlick_roughness(float cos_theta, float3 f0, float roughness)
{
    float3 fr = max(float3(1.0 - roughness), f0);
    return f0 + (fr - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

// One directional light's Cook-Torrance contribution.
float3 shade_sun(float3 n, float3 v, float3 l, float3 radiance, float3 albedo,
                 float metallic, float roughness, float3 f0)
{
    float3 h = normalize(v + l);
    float n_dot_v = max(dot(n, v), 1e-4);
    float n_dot_l = max(dot(n, l), 0.0);
    float n_dot_h = max(dot(n, h), 0.0);

    float ndf = distribution_ggx(n_dot_h, roughness);
    float geo = geometry_smith(n_dot_v, n_dot_l, roughness);
    float3 fr = fresnel_schlick(max(dot(h, v), 0.0), f0);

    float3 specular = (ndf * geo * fr) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
    float3 kd = (1.0 - fr) * (1.0 - metallic);
    return (kd * albedo / PI + specular) * radiance * n_dot_l;
}

// Image-based ambient from the previous frame's convolved cubes (split-sum).
// Falls back to flat ambient when no previous environment is bound.
float3 ambient_ibl(float3 n, float3 v, float3 albedo, float metallic,
                   float roughness, float3 f0)
{
    if (pc.ibl.x < 0)
        return pc.misc.x * albedo;

    float n_dot_v = max(dot(n, v), 1e-4);
    float3 r = reflect(-v, n);
    float3 f = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    float3 kd = (1.0 - f) * (1.0 - metallic);

    float3 irradiance = g_cubes[pc.ibl.x].SampleLevel(g_sampler, n, 0.0).rgb;
    float3 diffuse = irradiance * albedo;

    float max_lod = pc.misc.w;
    float3 prefiltered = g_cubes[pc.ibl.y].SampleLevel(g_sampler, r, roughness * max_lod).rgb;
    float2 brdf = g_textures[pc.ibl.z].SampleLevel(g_sampler, float2(n_dot_v, roughness), 0.0).rg;
    float3 specular = prefiltered * (f * brdf.x + brdf.y);

    return kd * diffuse + specular;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float3 albedo = pc.base_color.rgb;
    float roughness = max(pc.misc.y, 0.04);
    float metallic = pc.misc.z;
    float3 n = normalize(input.normal);
    float3 v = normalize(pc.eye.xyz - input.world_pos);
    float3 f0 = lerp(float3(0.04), albedo, metallic);

    float3 l = normalize(pc.sun.xyz);
    float3 radiance = float3(pc.sun.w);
    float3 lo = shade_sun(n, v, l, radiance, albedo, metallic, roughness, f0);

    float3 ambient = ambient_ibl(n, v, albedo, metallic, roughness, f0);
    return float4(ambient + lo, 1.0);
}
