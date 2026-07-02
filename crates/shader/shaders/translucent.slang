// Forward translucency pass (PR-3, render-pipeline re-baseline track,
// `docs/render-pipeline-reference.md` §1.7 #12 / §3 PR-3).
//
// Deferred shading has no native slot for semi-transparent surfaces (a G-buffer texel
// stores one opaque surface), so translucent geometry is drawn in a SEPARATE forward
// pass AFTER the opaque scene color is fully composited (deferred lighting + reflections
// + fog) and BEFORE post (TAAU/tonemap). This is the canonical hybrid deferred+forward
// approach: the caller sorts translucent objects back-to-front on the CPU and issues one
// draw each; the pass runs with depth-test ON (so opaque geometry occludes translucency)
// and depth-write OFF (so overlapping translucent layers all blend, none is Z-rejected),
// alpha-blending onto the HDR scene color. No G-buffer is written.
//
// Lighting is simple forward PBR reusing the SAME Cook-Torrance BRDF as the deferred
// pass (`pbr_brdf.slang`, single source): one directional sun (with a hard-PCF shadow
// lookup from the shared shadow map) + image-based ambient (irradiance + prefilter cube
// from `globals.ibl`). OIT / refraction / a dedicated translucency lighting volume are
// out of scope — Phase 20 extends this slot (see `docs/translucency-pass.md`).

#include "bindless.slang"

static const float PI = 3.14159265359;
#include "pbr_brdf.slang"

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

struct PushConstants
{
    float4x4 mvp;            // world->clip for this object (folds view_proj * model)
    float4x4 model;          // object->world (world-space position + normal outputs)
    float4 base_color;       // rgb tint (linear), a = coverage/opacity for the alpha blend
    float4 material;         // x metallic, y roughness, z base_color tex (uint bits), w flip_y (uint bits)
    uint4 misc;              // x shadow_index (0xFFFFFFFF = no shadow), yzw reserved
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

// Must match the Rust `Globals` layout (all float4 / int4 for std140/cbuffer agreement
// between Vulkan and D3D12) — the same declaration `pbr.slang` / `ssr.slang` carry. The
// translucency pass reuses the per-frame sun + IBL + exposure the deferred pass uses, so
// there is no duplicated lighting constant (single source, driven from `main.rs`).
struct Globals
{
    float4 camera_pos;
    float4 sun_direction;
    float4 sun_color;
    float4 ambient;       // rgb ambient, a = exposure
    int4 counts;          // x = point count, y = debug view, w = shadows enabled
    float4 point_pos[4];
    float4 point_color[4];
    float4x4 light_view_proj;
    float4 shadow;        // x = depth bias, y = texel size
    float4x4 inv_view_proj;
    int4 ibl;             // x env, y irradiance, z prefilter, w BRDF LUT (-1 = none)
    float4 probe;
    float4 probe_box_min;
    float4 probe_box_max;
    float4x4 prev_view_proj;
    float4 cluster_view_z_row;
    float4 cluster_params;
};

[[vk::binding(0, 1)]]
ConstantBuffer<Globals> globals : register(b1, space0);

[shader("vertex")]
VSOutput vsMain(VSInput input)
{
    VSOutput o;
    o.clip = mul(pc.mvp, float4(input.pos, 1.0));
    o.world_pos = mul(pc.model, float4(input.pos, 1.0)).xyz;
    o.world_normal = mul((float3x3)pc.model, input.normal);
    o.uv = input.uv;
    return o;
}

// Hard 3x3 PCF against the shared directional shadow map (the cheap path the deferred
// pass falls back to when soft shadows are off). Translucency is a thin, soft surface,
// so the extra PCSS penumbra machinery is unnecessary here.
float sun_shadow(float3 world_pos, uint shadow_index)
{
    if (globals.counts.w == 0 || shadow_index == NO_TEXTURE)
        return 1.0;

    float4 lc = mul(globals.light_view_proj, float4(world_pos, 1.0));
    float3 proj = lc.xyz / lc.w;
    float2 uv;
    uv.x = proj.x * 0.5 + 0.5;
    // Y-flip: Vulkan rasterizes the shadow map +Y down, D3D12 +Y up.
    uint flip_y = asuint(pc.material.w);
    uv.y = flip_y != 0 ? (proj.y * 0.5 + 0.5) : (proj.y * -0.5 + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0)
        return 1.0;

    float bias = globals.shadow.x;
    float texel = globals.shadow.y;
    float current = proj.z - bias;
    float sum = 0.0;
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
        {
            float closest =
                g.textures[shadow_index].Sample(g.samp, uv + float2(x, y) * texel).r;
            sum += current <= closest ? 1.0 : 0.0;
        }
    return sum / 9.0;
}

// Image-based ambient (diffuse irradiance + GGX prefilter specular). A lean version of
// `pbr.slang::ambient_ibl` without the indoor sky-visibility occlusion / box-parallax
// (those need per-pixel screen-space inputs the forward pass doesn't have); falls back to
// flat ambient when no IBL is bound.
float3 ambient_ibl(float3 n, float3 v, float3 albedo, float metallic, float roughness,
                   float3 f0)
{
    if (globals.ibl.y < 0)
        return globals.ambient.rgb * albedo;

    float n_dot_v = max(dot(n, v), 1e-4);
    float3 r = reflect(-v, n);
    float3 f = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    float3 kd = (1.0 - f) * (1.0 - metallic);

    float3 irradiance = g.cubes[globals.ibl.y].SampleLevel(g.samp, n, 0.0).rgb;
    float3 diffuse = irradiance * albedo;

    float max_lod = float(globals.counts.z >= 0 ? globals.counts.z : 0);
    float3 refl = g.cubes[globals.ibl.z].SampleLevel(g.samp, r, roughness * max_lod).rgb;
    float2 brdf =
        g.textures[globals.ibl.w].SampleLevel(g.samp, float2(n_dot_v, roughness), 0.0).rg;
    float3 specular = refl * (f * brdf.x + brdf.y);
    return kd * diffuse + specular;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target0
{
    float3 albedo = pc.base_color.rgb;
    float alpha = saturate(pc.base_color.a);
    uint base_tex = asuint(pc.material.z);
    if (base_tex != NO_TEXTURE)
    {
        float4 t = g.textures[base_tex].Sample(g.samp_wrap, input.uv);
        albedo *= t.rgb;
        alpha *= t.a;
    }
    float metallic = pc.material.x;
    float roughness = clamp(pc.material.y, 0.045, 1.0);

    // Two-sided: orient the normal toward the viewer so a plane lit from either face shades.
    float3 v = normalize(globals.camera_pos.xyz - input.world_pos);
    float3 n = normalize(input.world_normal);
    if (dot(n, v) < 0.0)
        n = -n;

    float3 f0 = lerp(float3(0.04), albedo, metallic);

    // Direct sun (shared BRDF) with a hard-PCF shadow.
    float3 l = normalize(globals.sun_direction.xyz);
    float3 radiance = globals.sun_color.rgb * globals.sun_color.a;
    float shadow = sun_shadow(input.world_pos, pc.misc.x);
    float3 direct = shade(n, v, l, radiance, albedo, metallic, roughness, f0) * shadow;

    // Image-based ambient.
    float3 ambient = ambient_ibl(n, v, albedo, metallic, roughness, f0);

    // Camera exposure (the deferred pass bakes exposure into the lit HDR; match it so the
    // translucent surface sits in the same radiance range as the opaque scene behind it).
    float exposure = globals.ambient.a;
    float3 color = (direct + ambient) * exposure;

    // Straight (non-premultiplied) alpha: the pipeline blends the color with
    // (SRC_ALPHA, ONE_MINUS_SRC_ALPHA) and the alpha with (ONE, ONE_MINUS_SRC_ALPHA), so
    // the shader returns the surface's own radiance + its coverage. Back-to-front CPU
    // sorting makes the over-operator sequence produce the correct layered result.
    return float4(color, alpha);
}
