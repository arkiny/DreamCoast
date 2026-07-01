// Deferred PBR lighting: a full-screen pass that reads the G-buffer (albedo,
// world normal, material, world position) and shades it with a Cook-Torrance
// metallic-roughness BRDF for one directional ("sun") light plus a few point
// lights. Output is linear HDR; tonemapping happens later (post.slang).
//
// G-buffer textures and the IBL cubemaps are sampled bindlessly through the
// shared `ParameterBlock` in `bindless.slang` (`g.textures[]` / `g.cubes[]` /
// `g.samp`), indexed by push constants (b0). Per-frame camera + light data comes
// from a loose globals uniform buffer (set 1 / b1) named `globals` — the block
// owns set 0, so the globals UBO cannot share it.
//
// Layout: the block pins to set 0 even with the globals UBO at set 1
// (`[[vk::binding(0,0)]]`), so the SPIR-V descriptor layout is identical to the
// previous loose-global form (textures b0, samp b1, cubes b2 on set 0; globals on
// set 1) — Vulkan untouched. On Metal the globals UBO takes `[[buffer(1)]]` and
// the bindless argument buffer lands at `[[buffer(2)]]`. The bounded-array /
// dropped-`SPV_EXT_descriptor_indexing` change is the same as the M3 mesh/imgui
// migration (Vulkan/D3D12 parity pending the Windows box).

#include "bindless.slang"

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
    uint gdf_ao_index; // Stage-C2 GDF AO image (0xFFFFFFFF = absent)
    uint ssao_index; // screen-space near-field AO image (0xFFFFFFFF = absent → 1.0, no-op)
    uint gdf_gi_index; // Stage-C3 GDF indirect-irradiance image (0xFFFFFFFF = absent)
    uint reflect_index; // Stage-C7c hybrid SW-RT reflection image (0xFFFFFFFF = legacy IBL prefilter cube)
    uint two_sided; // 1 = orient the G-buffer normal toward the viewer (two-sided surfaces)
    uint exposure_buf_index; // auto-exposure: storage buffer holding the adapted exposure at offset 0
                             // (0xFFFFFFFF = use the static globals.ambient.a)
    uint skyvis_index; // directional sky-visibility image (레퍼런스식 indoor skylight occlusion);
                       // occludes the IBL DIFFUSE skylight. 0xFFFFFFFF = absent -> 1.0 (gallery anchor).
    float skyvis_tint; // neutral leak as a fraction of the occluded skylight luminance.
    float skyvis_min_occ; // min sky-visibility floor (lerp V toward 1); higher = weaker occlusion.
};

static const uint NO_TEXTURE = 0xFFFFFFFF;

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
    float4 probe;         // xyz = reflection-probe capture centre, w = parallax on (1) / off (0)
    float4 probe_box_min; // xyz = reflection proxy AABB min corner (w unused)
    float4 probe_box_max; // xyz = reflection proxy AABB max corner (w unused)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[[vk::binding(0, 1)]]
ConstantBuffer<Globals> globals : register(b1, space0);

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
// `has_swrt` / `swrt_refl`: Stage-C7c hybrid SW-RT reflection radiance, composited in a
// prior compute pass (SSR -> GDF -> sky). When present it replaces the prefilter-cube
// specular lookup; the env-BRDF term (`f * brdf.x + brdf.y`) and the diffuse irradiance
// are unchanged. The composite already sees the real neighbouring geometry, so the
// box-proxy parallax correction is skipped on that path.
// 레퍼런스식 indoor skylight occlusion of the diffuse skylight (irradiance). `sky_vis` (V in [0,1]) is
// the cosine-weighted hemispherical sky visibility at the receiver normal. Occluded directions are
// replaced by a neutral OcclusionTint = a desaturated fraction of the LOCAL skylight luminance (so
// the indoor floor keeps brightness but loses the blue cast — the coloured GI bounce then shows),
// with a MinOcclusion floor so it never crushes to black. sky_vis = 1 -> exact no-op. The tint
// fraction and min-occlusion floor come from the push (`P_SKYVIS_TINT` / `P_SKYVIS_MIN_OCC`).
float3 occlude_sky_diffuse(float3 irradiance, float sky_vis)
{
    float vis = lerp(saturate(sky_vis), 1.0, pc.skyvis_min_occ);
    float lum = dot(irradiance, float3(0.2126, 0.7152, 0.0722));
    return irradiance * vis + (pc.skyvis_tint * lum) * (1.0 - vis);
}

float3 ambient_ibl(float3 n, float3 v, float3 world_pos, float3 albedo, float metallic,
                   float roughness, float ao, float3 f0, float3 swrt_refl, bool has_swrt,
                   float sky_vis)
{
    if (globals.ibl.y < 0)
        return globals.ambient.rgb * albedo * ao;

    float n_dot_v = max(dot(n, v), 1e-4);
    float3 r = reflect(-v, n);

    float3 f = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    float3 kd = (1.0 - f) * (1.0 - metallic);

    float3 irradiance = g.cubes[globals.ibl.y].SampleLevel(g.samp, n, 0.0).rgb;
    // 레퍼런스식 indoor skylight occlusion: occlude the diffuse skylight (irradiance) by the scene-
    // scale sky visibility before the albedo, so an enclosed interior isn't flooded by the full
    // (blue) diffuse skylight (sky_vis = 1 -> exact no-op, the gallery anchor).
    float3 diffuse = occlude_sky_diffuse(irradiance, sky_vis) * albedo;

    float3 refl;
    if (has_swrt)
    {
        // Stage C7c: raw-radiance hybrid reflection (drop-in for the prefiltered cube;
        // the final `* ambient.a` below applies exposure once, like the cube path).
        refl = swrt_refl;
    }
    else
    {
        // Box-proxy parallax correction (legacy IBL): the reflection probe is a single
        // cubemap captured at `probe.xyz`, but this surface is elsewhere, so the raw
        // reflection vector samples the wrong part of the cube (the reflected
        // ground/horizon slides around as the surface moves). Model the captured scene as
        // an axis-aligned box (floor at the ground plane, walls/ceiling at the captured
        // extent), intersect the reflection ray with that box, and re-anchor the lookup to
        // the exit point relative to the capture centre. A box fits the flat ground far
        // better than a sphere proxy, which mapped floor reflections up toward the horizon
        // and darkened them.
        if (globals.probe.w > 0.0)
        {
            float3 inv_r = 1.0 / r;
            float3 t1 = (globals.probe_box_min.xyz - world_pos) * inv_r;
            float3 t2 = (globals.probe_box_max.xyz - world_pos) * inv_r;
            float3 t_far = max(t1, t2);
            float t = min(min(t_far.x, t_far.y), t_far.z); // box exit distance
            if (t > 0.0)
            {
                float3 hit = world_pos + r * t;
                r = normalize(hit - globals.probe.xyz);
            }
        }
        float max_lod = float(globals.counts.z);
        refl = g.cubes[globals.ibl.z].SampleLevel(g.samp, r, roughness * max_lod).rgb;
    }

    float2 brdf = g.textures[globals.ibl.w].SampleLevel(g.samp, float2(n_dot_v, roughness), 0.0).rg;
    float3 specular = refl * (f * brdf.x + brdf.y);

    return (kd * diffuse + specular) * ao;
}

// --- Soft-shadow quality tier (the scalability seam): one place to swap counts.
// RenderQuality{low,med,high} drives the per-frame tap count via `globals.shadow.w`
// (0/out-of-range => SHADOW_MAX_SAMPLES, the legacy default). `light_size`
// (globals.shadow.z, 0 = hard) is the penumbra scale / off switch. POISSON16 caps the
// tap count at 16, so SHADOW_MAX_SAMPLES is the array bound, not a tier knob.
static const int SHADOW_MAX_SAMPLES = 16;
static const float SHADOW_SEARCH_UV = 0.012; // blocker-search disk radius (shadow-map UV)
static const float SHADOW_MAX_PENUMBRA = 0.015; // clamp so a far caster can't over-blur

static const float2 POISSON16[16] = {
    float2(-0.94201624, -0.39906216), float2(0.94558609, -0.76890725),
    float2(-0.09418410, -0.92938870), float2(0.34495938, 0.29387760),
    float2(-0.91588581, 0.45771432), float2(-0.81544232, -0.87912464),
    float2(-0.38277543, 0.27676845), float2(0.97484398, 0.75648379),
    float2(0.44323325, -0.97511554), float2(0.53742981, -0.47373420),
    float2(-0.26496911, -0.41893023), float2(0.79197514, 0.19090188),
    float2(-0.24188840, 0.99706507), float2(-0.81409955, 0.91437590),
    float2(0.19984126, 0.78641367), float2(0.14383161, -0.14100790),
};

// One PCF tap: 1 if `current` (biased receiver depth) is in front of the stored blocker.
float shadow_tap(float2 uv, float current)
{
    float closest = g.textures[pc.shadow_index].Sample(g.samp, uv).r;
    return current <= closest ? 1.0 : 0.0;
}

// Fraction of the directional sun reaching `world_pos` (1 = fully lit, 0 = fully
// shadowed). PCSS-lite: a blocker search sizes a distance-proportional penumbra,
// then a rotated Poisson PCF filters it (soft contact-hard, wide far from the caster).
// `light_size` (globals.shadow.z) <= 0 falls back to the cheap 3x3 PCF (hard shadows).
// `pix` = screen pixel coord, for the per-pixel rotation that hides banding.
float sun_shadow(float3 world_pos, float2 pix)
{
    if (globals.counts.w == 0)
        return 1.0;

    float4 lc = mul(globals.light_view_proj, float4(world_pos, 1.0));
    float3 proj = lc.xyz / lc.w; // ortho: w = 1
    float2 uv;
    uv.x = proj.x * 0.5 + 0.5;
    // Map clip Y to texture V. Vulkan rasterizes the shadow map with +Y down,
    // D3D12 with +Y up, so the sign of the V flip depends on the backend.
    uv.y = pc.flip_y != 0 ? (proj.y * 0.5 + 0.5) : (proj.y * -0.5 + 0.5);

    // Outside the light frustum (or beyond its far plane): treat as lit.
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0)
        return 1.0;

    float bias = globals.shadow.x;
    float texel = globals.shadow.y;
    float current = proj.z - bias;
    // Penumbra factor: shadow-map UV of penumbra per unit normalized depth gap. Folds
    // the sun's angular size + the ortho depth-range/width on the CPU so it matches the
    // path tracer's sun disk. 0 = hard 3x3 PCF.
    float penumbra_factor = globals.shadow.z;

    // Per-pixel rotation of the Poisson disk (interleaved gradient noise).
    float ign = frac(52.9829189 * frac(dot(pix, float2(0.06711056, 0.00583715))));
    float ca = cos(ign * 6.2831853);
    float sa = sin(ign * 6.2831853);
    float2x2 rot = float2x2(ca, -sa, sa, ca);

    // Hard fallback (soft shadows disabled): cheap 3x3 PCF.
    if (penumbra_factor <= 0.0)
    {
        float sum = 0.0;
        for (int x = -1; x <= 1; ++x)
            for (int y = -1; y <= 1; ++y)
                sum += shadow_tap(uv + float2(x, y) * texel, current);
        return sum / 9.0;
    }

    // Soft-shadow tap count from the RenderQuality tier (globals.shadow.w). 0 / out-of-range
    // falls back to the array bound, so an unset slot reproduces the legacy 16-tap behavior.
    int taps = int(globals.shadow.w);
    if (taps < 1 || taps > SHADOW_MAX_SAMPLES)
        taps = SHADOW_MAX_SAMPLES;

    // 1) Blocker search over a fixed disk; average the occluders' depth.
    float blocker_sum = 0.0;
    int blocker_n = 0;
    for (int i = 0; i < taps; ++i)
    {
        float2 o = mul(rot, POISSON16[i]) * SHADOW_SEARCH_UV;
        float d = g.textures[pc.shadow_index].Sample(g.samp, uv + o).r;
        if (d < current)
        {
            blocker_sum += d;
            blocker_n += 1;
        }
    }
    if (blocker_n == 0)
        return 1.0; // no occluder -> fully lit (the common, cheap case)
    float avg_blocker = blocker_sum / float(blocker_n);

    // 2) Penumbra radius = depth gap x calibrated factor (directional light, linear gap).
    float radius = clamp((current - avg_blocker) * penumbra_factor, texel, SHADOW_MAX_PENUMBRA);

    // 3) PCF over the penumbra radius.
    float sum = 0.0;
    for (int i = 0; i < taps; ++i)
    {
        float2 o = mul(rot, POISSON16[i]) * radius;
        sum += shadow_tap(uv + o, current);
    }
    return sum / float(taps);
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

// Camera exposure applied to the whole (ambient + direct) sum. Auto-exposure stores the adapted
// value in a storage buffer (offset 0); otherwise the static EV100-derived globals.ambient.a is
// used. A sentinel index keeps the non-auto path bit-identical to the pre-auto-exposure lighting.
float scene_exposure()
{
    if (pc.exposure_buf_index != 0xFFFFFFFFu)
        return asfloat(g.storage_buffers[pc.exposure_buf_index].Load(0));
    return globals.ambient.a;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float exposure = scene_exposure();
    float4 albedo_ao = g.textures[pc.albedo_index].Sample(g.samp, input.uv);
    float3 normal = g.textures[pc.normal_index].Sample(g.samp, input.uv).xyz;
    float4 mat = g.textures[pc.material_index].Sample(g.samp, input.uv);
    float4 pos = g.textures[pc.position_index].Sample(g.samp, input.uv);

    // Background (no geometry written): position.w stays 0. Sample the environment
    // cubemap along the view ray (skybox), or fall back to flat ambient.
    if (pos.w < 0.5)
    {
        if (globals.ibl.x >= 0)
        {
            float sy = pc.flip_y != 0 ? 1.0 : -1.0;
            float2 ndc = float2(input.uv.x * 2.0 - 1.0, (input.uv.y * 2.0 - 1.0) * sy);
            float4 far = mul(globals.inv_view_proj, float4(ndc, 1.0, 1.0));
            float3 world = far.xyz / far.w;
            float3 dir = normalize(world - globals.camera_pos.xyz);
            // Mip 1+ holds sky only (scene geometry is captured into mip 0), so the
            // skybox stays clean sky; the gradient is low-frequency so half-res is
            // imperceptible.
            float3 sky = g.cubes[globals.ibl.x].SampleLevel(g.samp, dir, 1.0).rgb;
            return float4(sky * exposure, 1.0);
        }
        return float4(globals.ambient.rgb, 1.0);
    }

    float3 albedo = albedo_ao.rgb;
    float metallic = mat.r;
    float roughness = max(mat.g, 0.04);
    float ao = mat.b;
    float3 world_pos = pos.xyz;
    float3 n = normalize(normal);
    float3 v = normalize(globals.camera_pos.xyz - world_pos);
    // Two-sided surfaces (e.g. an imported building's single-sided walls seen from the
    // inside): the engine rasterizes with no back-face culling, so a back-face arrives
    // with its geometric normal pointing away from the camera. Orient it toward the
    // viewer so it shades correctly. View-dependent (not SV_IsFrontFace) to stay
    // winding/Y-flip agnostic across backends.
    if (pc.two_sided != 0 && dot(n, v) < 0.0)
        n = -n;
    float3 f0 = lerp(float3(0.04), albedo, metallic);

    float3 lo = float3(0.0);

    // Directional sun (attenuated by the shadow map).
    {
        float3 l = normalize(globals.sun_direction.xyz);
        float3 radiance = globals.sun_color.rgb * globals.sun_color.a;
        lo += shade(n, v, l, radiance, albedo, metallic, roughness, f0)
            * sun_shadow(world_pos, input.pos.xy);
    }

    // Point lights. point_color.a is the luminous intensity in CANDELA (lm/sr) — the same
    // photometric unit as the lux sun, so the single end-of-pass exposure maps both. The inverse-
    // square `atten` converts candela → illuminance (lux) at the surface: E = I / d². (glTF
    // KHR_lights_punctual authors point/spot intensity in candela already; a light specified in
    // lumens Φ converts as I = Φ / 4π.)
    int count = min(globals.counts.x, 4);
    for (int i = 0; i < count; ++i)
    {
        float3 to_light = globals.point_pos[i].xyz - world_pos;
        float dist = length(to_light);
        float3 l = to_light / max(dist, 1e-4);
        float atten = 1.0 / max(dist * dist, 1e-4);
        float3 radiance = globals.point_color[i].rgb * globals.point_color[i].a * atten;
        lo += shade(n, v, l, radiance, albedo, metallic, roughness, f0);
    }

    // Stage C2: GDF ambient occlusion, marched against the world-space scene GDF in a
    // prior compute pass. Multiplies into the ambient/IBL term only (direct light keeps
    // its own shadow-mapped visibility). Absent (index 0xFFFFFFFF) -> unoccluded.
    float gdf_ao = 1.0;
    if (pc.gdf_ao_index != NO_TEXTURE)
        gdf_ao = g.textures[pc.gdf_ao_index].Sample(g.samp, input.uv).r;
    // Screen-space near-field AO composes with the GDF far-field AO (the way 레퍼런스 엔진 layers
    // GTAO × DFAO). Absent (0xFFFFFFFF) -> 1.0, an exact no-op (byte-identical baseline).
    if (pc.ssao_index != NO_TEXTURE)
        gdf_ao *= g.textures[pc.ssao_index].Sample(g.samp, input.uv).r;

    // Stage C7c: the hybrid SW-RT reflection (composited in a prior compute pass) replaces
    // the IBL prefilter-cube specular when bound; absent (0xFFFFFFFF) -> legacy IBL path.
    bool has_swrt = pc.reflect_index != NO_TEXTURE;
    float3 swrt_refl = float3(0.0);
    if (has_swrt)
        swrt_refl = g.textures[pc.reflect_index].Sample(g.samp, input.uv).rgb;

    // 레퍼런스식 indoor skylight occlusion: directional sky-visibility occludes the IBL diffuse
    // skylight. Absent (0xFFFFFFFF) -> 1.0 (exact no-op, gallery anchor).
    float sky_vis = 1.0;
    if (pc.skyvis_index != NO_TEXTURE)
        sky_vis = saturate(g.textures[pc.skyvis_index].Sample(g.samp, input.uv).r);

    float3 ambient = ambient_ibl(n, v, world_pos, albedo, metallic, roughness, ao, f0,
                                 swrt_refl, has_swrt, sky_vis) * gdf_ao;

    // Stage C3: 1-bounce diffuse GI marched against the scene GDF in a prior compute
    // pass. The GI image holds the mean incoming radiance (indirect irradiance) from
    // scene geometry; weight it by the diffuse albedo (× 1-metallic) and add it to the
    // ambient — the scene-bounce term IBL·AO can't capture. Absent -> 0.
    float3 gi = float3(0.0);
    if (pc.gdf_gi_index != NO_TEXTURE)
    {
        float3 e = g.textures[pc.gdf_gi_index].Sample(g.samp, input.uv).rgb;
        gi = (1.0 - metallic) * albedo * e;
        // Multi-bounce energy compensation (레퍼런스 엔진 DistantIlluminationRescale / Jimenez): the GDF GI is
        // a SINGLE bounce, so it misses the infinite-bounce energy that fills interiors. In an
        // environment of ~the surface albedo, the geometric series 1/(1 - albedo) restores it.
        // globals.probe_box_max.w = per-scene strength (0 = off = the gallery anchor; ~0.6 for
        // content); the max() clamps the boost to <=4x so a near-white albedo can't blow up.
        gi /= max(1.0 - albedo * globals.probe_box_max.w, 0.25);
    }
    ambient += gi;

    float3 color = (ambient + lo) * exposure;

    // Debug views (counts.y): 1 albedo, 2 normal, 3 metallic, 4 roughness,
    // 5 world pos, 6 AO, 7 direct light, 8 IBL ambient, 9 GDF AO, 10 GDF GI. 0 = lit.
    int dbg = globals.counts.y;
    if (dbg == 1) color = albedo;
    else if (dbg == 2) color = n * 0.5 + 0.5;
    else if (dbg == 3) color = float3(metallic);
    else if (dbg == 4) color = float3(roughness);
    else if (dbg == 5) color = frac(world_pos);
    else if (dbg == 6) color = float3(ao);
    else if (dbg == 7) color = lo * exposure;      // direct lighting only
    else if (dbg == 8) color = ambient * exposure; // IBL ambient only
    else if (dbg == 9) color = float3(gdf_ao);     // GDF AO factor
    else if (dbg == 10) color = gi * exposure;     // GDF indirect diffuse

    return float4(color, 1.0);
}
