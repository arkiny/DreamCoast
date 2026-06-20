// Deferred PBR lighting: a full-screen pass that reads the G-buffer (albedo,
// world normal, material, world position) and shades it with a Cook-Torrance
// metallic-roughness BRDF for one directional ("sun") light plus a few point
// lights. Output is linear HDR; tonemapping happens later (post.slang).
//
// G-buffer textures are sampled bindlessly (t0/binding 0 array, indexed by push
// constants b0). Per-frame camera + light data comes from the globals uniform
// buffer (set 1 / b1).

static const float PI = 3.14159265359;

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    uint albedo_index;
    uint normal_index;
    uint material_index;
    uint position_index;
    uint flip_y; // 0 = D3D12 (Y up), 1 = Vulkan (Y down)
    uint shadow_index; // bindless index of the directional shadow map
};

// Must match the Rust `Globals` layout (all float4 / int4 for std140/cbuffer
// agreement between Vulkan and D3D12).
struct Globals
{
    float4 camera_pos;    // xyz
    float4 sun_direction; // xyz = direction TO the light (normalized)
    float4 sun_color;     // rgb color, a = intensity
    float4 ambient;       // rgb ambient, a = exposure
    int4 counts;          // x = point count, y = debug view, w = shadows enabled
    float4 point_pos[4];  // xyz position, a = radius
    float4 point_color[4]; // rgb color, a = intensity
    float4x4 light_view_proj; // world -> light clip (for shadow lookup)
    float4 shadow;        // x = depth bias, y = texel size (1 / shadow map size)
    float4x4 inv_view_proj; // clip -> world (skybox ray reconstruction)
    int4 ibl;             // x env, y irradiance, z prefilter, w BRDF LUT (-1 = none)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[[vk::binding(0, 1)]]
ConstantBuffer<Globals> g : register(b1, space0);

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

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

// Fresnel with a roughness term, for the IBL ambient.
float3 fresnel_schlick_roughness(float cos_theta, float3 f0, float roughness)
{
    float3 fr = max(float3(1.0 - roughness), f0);
    return f0 + (fr - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

// Image-based ambient: diffuse from the irradiance cube + specular from the
// prefilter cube and BRDF LUT (split-sum). Falls back to flat ambient when no
// IBL is bound.
float3 ambient_ibl(float3 n, float3 v, float3 albedo, float metallic,
                   float roughness, float ao, float3 f0)
{
    if (g.ibl.y < 0)
        return g.ambient.rgb * albedo * ao;

    float n_dot_v = max(dot(n, v), 1e-4);
    float3 r = reflect(-v, n);
    float3 f = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    float3 kd = (1.0 - f) * (1.0 - metallic);

    float3 irradiance = g_cubes[g.ibl.y].SampleLevel(g_sampler, n, 0.0).rgb;
    float3 diffuse = irradiance * albedo;

    float max_lod = float(g.counts.z);
    float3 prefiltered = g_cubes[g.ibl.z].SampleLevel(g_sampler, r, roughness * max_lod).rgb;
    float2 brdf = g_textures[g.ibl.w].SampleLevel(g_sampler, float2(n_dot_v, roughness), 0.0).rg;
    float3 specular = prefiltered * (f * brdf.x + brdf.y);

    return (kd * diffuse + specular) * ao;
}

// Fraction of the directional sun reaching `world_pos` (1 = fully lit, 0 = fully
// shadowed), via 3x3 PCF over the directional shadow map. Returns 1 when shadows
// are disabled or the point falls outside the light frustum.
float sun_shadow(float3 world_pos)
{
    if (g.counts.w == 0)
        return 1.0;

    float4 lc = mul(g.light_view_proj, float4(world_pos, 1.0));
    float3 proj = lc.xyz / lc.w; // ortho: w = 1
    float2 uv;
    uv.x = proj.x * 0.5 + 0.5;
    // Map clip Y to texture V. Vulkan rasterizes the shadow map with +Y down,
    // D3D12 with +Y up, so the sign of the V flip depends on the backend.
    uv.y = pc.flip_y != 0 ? (proj.y * 0.5 + 0.5) : (proj.y * -0.5 + 0.5);

    // Outside the light frustum (or beyond its far plane): treat as lit.
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0)
        return 1.0;

    float bias = g.shadow.x;
    float texel = g.shadow.y;
    float current = proj.z - bias;

    float sum = 0.0;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float2 o = float2(x, y) * texel;
            float closest = g_textures[pc.shadow_index].Sample(g_sampler, uv + o).r;
            sum += current <= closest ? 1.0 : 0.0;
        }
    }
    return sum / 9.0;
}

// One light's contribution. `radiance` already folds color * intensity * atten.
float3 shade(float3 n, float3 v, float3 l, float3 radiance, float3 albedo,
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

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float4 albedo_ao = g_textures[pc.albedo_index].Sample(g_sampler, input.uv);
    float3 normal = g_textures[pc.normal_index].Sample(g_sampler, input.uv).xyz;
    float4 mat = g_textures[pc.material_index].Sample(g_sampler, input.uv);
    float4 pos = g_textures[pc.position_index].Sample(g_sampler, input.uv);

    // Background (no geometry written): position.w stays 0. Sample the environment
    // cubemap along the view ray (skybox), or fall back to flat ambient.
    if (pos.w < 0.5)
    {
        if (g.ibl.x >= 0)
        {
            float sy = pc.flip_y != 0 ? 1.0 : -1.0;
            float2 ndc = float2(input.uv.x * 2.0 - 1.0, (input.uv.y * 2.0 - 1.0) * sy);
            float4 far = mul(g.inv_view_proj, float4(ndc, 1.0, 1.0));
            float3 world = far.xyz / far.w;
            float3 dir = normalize(world - g.camera_pos.xyz);
            float3 sky = g_cubes[g.ibl.x].SampleLevel(g_sampler, dir, 0.0).rgb;
            return float4(sky * g.ambient.a, 1.0); // a = exposure
        }
        return float4(g.ambient.rgb, 1.0);
    }

    float3 albedo = albedo_ao.rgb;
    float metallic = mat.r;
    float roughness = max(mat.g, 0.04);
    float ao = mat.b;
    float3 world_pos = pos.xyz;
    float3 n = normalize(normal);
    float3 v = normalize(g.camera_pos.xyz - world_pos);
    float3 f0 = lerp(float3(0.04), albedo, metallic);

    float3 lo = float3(0.0);

    // Directional sun (attenuated by the shadow map).
    {
        float3 l = normalize(g.sun_direction.xyz);
        float3 radiance = g.sun_color.rgb * g.sun_color.a;
        lo += shade(n, v, l, radiance, albedo, metallic, roughness, f0)
            * sun_shadow(world_pos);
    }

    // Point lights.
    int count = min(g.counts.x, 4);
    for (int i = 0; i < count; ++i)
    {
        float3 to_light = g.point_pos[i].xyz - world_pos;
        float dist = length(to_light);
        float3 l = to_light / max(dist, 1e-4);
        float atten = 1.0 / max(dist * dist, 1e-4);
        float3 radiance = g.point_color[i].rgb * g.point_color[i].a * atten;
        lo += shade(n, v, l, radiance, albedo, metallic, roughness, f0);
    }

    float3 ambient = ambient_ibl(n, v, albedo, metallic, roughness, ao, f0);
    float3 color = (ambient + lo) * g.ambient.a; // a = exposure

    // Debug views (counts.y): 1 albedo, 2 normal, 3 metallic, 4 roughness,
    // 5 world pos, 6 AO. 0 = lit.
    int dbg = g.counts.y;
    if (dbg == 1) color = albedo;
    else if (dbg == 2) color = n * 0.5 + 0.5;
    else if (dbg == 3) color = float3(metallic);
    else if (dbg == 4) color = float3(roughness);
    else if (dbg == 5) color = frac(world_pos);
    else if (dbg == 6) color = float3(ao);

    return float4(color, 1.0);
}
