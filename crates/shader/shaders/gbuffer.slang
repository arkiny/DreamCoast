// Deferred G-buffer fill: rasterizes the mesh and writes albedo, world-space
// normal, material (metallic/roughness/AO), and world position into four MRT
// color attachments. A later full-screen pass (pbr.slang) reads these to do the
// lighting. The model matrix is identity (the sandbox orbits the camera), so
// input positions/normals are already world-space.
//
// Bindless layout mirrors mesh.slang: a sampled-texture array (t0/binding 0) +
// a static sampler (s0/binding 1), indexed by push constants (b0). A texture
// index of 0xFFFFFFFF means "absent — use the scalar factor".

static const uint NO_TEXTURE = 0xFFFFFFFF;

struct VSInput
{
    float3 pos : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
};

struct VSOutput
{
    float4 clip : SV_Position;
    float3 world_pos : POSITION0;
    float3 world_normal : NORMAL0;
    float2 uv : TEXCOORD0;
};

struct GBufferOut
{
    float4 albedo : SV_Target0;   // rgb base color (linear), a = AO
    float4 normal : SV_Target1;   // xyz world-space normal
    float4 material : SV_Target2; // r metallic, g roughness, b AO
    float4 position : SV_Target3; // xyz world position
};

struct PushConstants
{
    float4x4 mvp;
    float4 base_color_factor;
    float4 mr_factor;     // x metallic, y roughness
    uint4 tex;            // x base_color, y metallic_roughness, z normal, w emissive
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
    o.clip = mul(pc.mvp, float4(input.pos, 1.0));
    o.world_pos = input.pos;
    o.world_normal = input.normal;
    o.uv = input.uv;
    return o;
}

// Build a tangent frame from screen-space derivatives (no per-vertex tangents),
// then transform a tangent-space normal-map sample into world space.
float3 apply_normal_map(float3 n, float3 world_pos, float2 uv, float3 tangent_normal)
{
    float3 dp1 = ddx(world_pos);
    float3 dp2 = ddy(world_pos);
    float2 duv1 = ddx(uv);
    float2 duv2 = ddy(uv);

    float3 dp2perp = cross(dp2, n);
    float3 dp1perp = cross(n, dp1);
    float3 t = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 b = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = rsqrt(max(dot(t, t), dot(b, b)));
    float3x3 tbn = float3x3(t * invmax, b * invmax, n);
    return normalize(mul(tangent_normal, tbn));
}

[shader("fragment")]
GBufferOut fsMain(VSOutput input)
{
    float3 albedo = pc.base_color_factor.rgb;
    if (pc.tex.x != NO_TEXTURE)
        albedo *= g_textures[pc.tex.x].Sample(g_sampler, input.uv).rgb;

    float metallic = pc.mr_factor.x;
    float roughness = pc.mr_factor.y;
    if (pc.tex.y != NO_TEXTURE)
    {
        // glTF metallic-roughness: G = roughness, B = metallic.
        float3 mr = g_textures[pc.tex.y].Sample(g_sampler, input.uv).rgb;
        roughness *= mr.g;
        metallic *= mr.b;
    }

    float3 n = normalize(input.world_normal);
    if (pc.tex.z != NO_TEXTURE)
    {
        float3 tn = g_textures[pc.tex.z].Sample(g_sampler, input.uv).rgb * 2.0 - 1.0;
        n = apply_normal_map(n, input.world_pos, input.uv, tn);
    }

    GBufferOut o;
    o.albedo = float4(albedo, 1.0);
    o.normal = float4(n, 0.0);
    o.material = float4(metallic, saturate(roughness), 1.0, 0.0);
    o.position = float4(input.world_pos, 1.0);
    return o;
}
