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
#include "light_cluster_common.slang"

static const float PI = 3.14159265359;

// Cook-Torrance BRDF helpers (distribution/geometry/fresnel/shade) live in a shared
// header so the forward translucency pass (`translucent.slang`) shades with the exact
// same math — single source, no drift. Textual include after `PI` is defined, so the
// generated code is unchanged from the previous inline definitions (byte-identical anchor).
#include "pbr_brdf.slang"

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
    // Clustered light culling (PR-6). When cluster_grid_buf != 0xFFFFFFFF the point-light
    // loop reads only this pixel's froxel light list (storage buffers below) instead of the
    // globals.point_pos[] brute-force array. All three 0xFFFFFFFF = brute-force (gallery anchor).
    uint cluster_grid_buf;  // per-cluster light COUNT u32[CLUSTER_COUNT]
    uint cluster_index_buf; // flat light-index list u32[CLUSTER_COUNT * MAX_LIGHTS_PER_CLUSTER]
    uint cluster_light_buf; // packed Light[] (position/radius/color/intensity)
    uint cluster_light_count; // number of lights in cluster_light_buf (dbg heatmap normalisation)
    uint ao_multibounce; // 1 = albedo-tinted multi-bounce AO on the diffuse ambient (opt-in); 0 = off
    uint spec_occlusion; // 1 = bent-normal cone-cone occlusion on the prefilter-cube specular (opt-in)
    // Interior-bias root phase: aperture shaping of the neutral OcclusionTint leak. The flat
    // (1-V)-proportional leak puts its energy where V ~= 0 — exactly the fully-enclosed regions
    // the path-traced reference keeps dark (the EV11 deep crop measured 2.24x the reference),
    // while the raster's REAL deficit sits in partially-occluded lit surfaces. Shaping the leak
    // by saturate(V / v0) zeroes it in enclosed cavities (a real skylight leak scales with the
    // aperture) and keeps it where V is middling. 0 = the flat legacy leak, byte-identical.
    float skyvis_tint_v0;
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
    float4x4 prev_view_proj; // world -> previous clip (SSR history reprojection; mirrors Rust)
    float4 cluster_view_z_row; // row 2 of world->view: view_z = dot(row, float4(p,1)) (PR-6)
    float4 cluster_params;     // x = z_near, y = z_far (clustered froxel Z slicing)
    // --- PR-7 shadow atlas / CSM. All zero on the legacy single-map path (csm_params.x == 0),
    // where sun_shadow() takes the `light_view_proj` branch → byte-identical anchor.
    int4 csm_params;         // x = cascade count (0 = off), y = debug viz, z = tile texels, w = atlas texels
    float4 csm_split;        // per-cascade view-space far distance (up to 4)
    float4 csm_opts;         // x = cross-cascade blend fraction (of the cascade depth range)
    float4x4 csm_view_proj[4]; // per-cascade world -> atlas-tile light clip
    float4 csm_atlas_uv[4];    // per-cascade atlas UV sub-rect: xy offset, zw scale
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
// Multi-bounce ambient occlusion (reference engine AOMultiBounce): a scalar AO over-darkens because
// light bounces multiple times inside a cavity, and those inter-reflections are tinted by the
// surface's own albedo. This cubic per-channel polynomial restores that energy — it lifts the AO
// back toward 1 in the surface's albedo colour (a red cavity stays red instead of going grey-dark),
// while a black surface keeps the raw AO. `max(AO, ·)` guarantees it never darkens below the input.
// Ref: DeferredShadingCommon.ush AOMultiBounce. Reduces to the scalar AO when disabled (opt-in).
float3 ao_multi_bounce(float3 base_color, float ao)
{
    float3 a = 2.0404 * base_color - 0.3324;
    float3 b = -4.7951 * base_color + 0.6417;
    float3 c = 2.7552 * base_color + 0.6903;
    return max(float3(ao), ((ao * a + b) * ao + c) * ao);
}

// Bent-normal specular occlusion (reference engine GetDistanceFieldAOSpecularOcclusion). The
// diffuse skylight is occluded by the sky-visibility scalar; the ambient SPECULAR should be
// occluded by how much of the glossy REFLECTION cone is blocked by scene geometry. Model both the
// reflection lobe (roughness -> cone angle) and the unoccluded sky (bent normal -> a cone of
// half-angle V·PI around the bent direction) as cones and take their overlap. This applies ONLY to
// the prefilter-cube specular: a traced SW-RT reflection already resolves occlusion along its ray
// (per the reflection track), so occluding it again would double-count.
static const float SPEC_PI = 3.14159265359;

// Overlap fraction of two cones separated by `angle_between` (reference ApproximateConeConeIntersection).
float approx_cone_cone_intersection(float arc0, float arc1, float angle_between)
{
    float diff = abs(arc0 - arc1);
    return smoothstep(0.0, 1.0, 1.0 - saturate((angle_between - diff) / (arc0 + arc1 - diff)));
}

// Specular occlusion in [0,1] from the (unit) bent normal, its sky visibility V, the mirror
// reflection direction, and roughness. bent length < eps (gallery / no producer) -> 1 (no-op).
float spec_occlusion_bent(float3 bent_unit, float sky_vis, float3 mirror, float roughness)
{
    if (length(bent_unit) < 1e-4)
        return 1.0; // no bent normal => no directional occlusion (byte-identical anchor)
    float unoccluded_angle = saturate(sky_vis) * SPEC_PI; // bent-normal cone half-angle (= length·PI)
    float refl_cone_angle = max(roughness, 0.1) * SPEC_PI; // glossy reflection lobe half-angle
    float angle_between = acos(clamp(dot(bent_unit, normalize(mirror)), -1.0, 1.0));
    float occ = approx_cone_cone_intersection(refl_cone_angle, unoccluded_angle, angle_between);
    // Near-fully-occluded: the bent direction is unreliable, so fade to shadowed (reference).
    occ = lerp(0.0, occ, saturate((unoccluded_angle - 0.1) / 0.2));
    // MinOcclusion floor (shared with the diffuse path): =1 disables it -> occ 1, no-op.
    return lerp(occ, 1.0, pc.skyvis_min_occ);
}

// Directional (bent-normal) skylight occlusion — the reference engine's DFAO -> skylight coupling.
// Instead of a scalar V multiply of the surface-normal irradiance, sample the diffuse skylight
// ALONG the bent normal (the unoccluded average direction) so an interior receives skylight biased
// toward where the sky actually is, then attenuate by V and the projection factor. Grounded in the
// reference SkyLightingDiffuseShared: SkyLightingNormal = lerp(bent, n, V) ("more bent normal in
// corners"), DotProductFactor = lerp(dot(bent,n), 1, V), Lookup = irradiance(SkyLightingNormal),
// Mul = V*DotProductFactor. When `bent` is zero (no producer, gallery, screen-probe path) this
// reduces EXACTLY to the scalar path (skyN = n, dotFactor = 1: irradiance(n)*V + tint*lum*(1-V),
// the previous occlude_sky_diffuse) — byte-identical fallback.
float3 occlude_sky_diffuse_bent(float3 n, float sky_vis, float3 bent, uint irr_cube)
{
    float3 skyN = n;
    float dot_factor = 1.0;
    float bl = length(bent);
    if (bl > 1e-4)
    {
        float3 nb = bent / bl;
        float w = saturate(sky_vis); // BentNormalWeightFactor: open -> surface normal, corner -> bent
        skyN = normalize(lerp(nb, n, w));
        dot_factor = max(lerp(dot(nb, n), 1.0, w), 0.0); // never subtract light (full-sphere bent guard)
    }
    float3 irradiance = g.cubes[irr_cube].SampleLevel(g.samp, skyN, 0.0).rgb;
    float vis = lerp(saturate(sky_vis), 1.0, pc.skyvis_min_occ);
    float lum = dot(irradiance, float3(0.2126, 0.7152, 0.0722));
    // Occluded fraction -> neutral OcclusionTint leak (as in the scalar path); the direction bias
    // is carried by `dot_factor` on the visible fraction. bent = 0 => dot_factor = 1 => identical.
    // Aperture-shaped leak (skyvis_tint_v0 > 0): zero in fully-enclosed regions, full above v0.
    float leak = 1.0 - vis;
    if (pc.skyvis_tint_v0 > 0.0)
        leak *= saturate(saturate(sky_vis) / pc.skyvis_tint_v0);
    return irradiance * (vis * dot_factor) + (pc.skyvis_tint * lum) * leak;
}

float3 ambient_ibl(float3 n, float3 v, float3 world_pos, float3 albedo, float metallic,
                   float roughness, float ao, float3 f0, float3 swrt_refl, bool has_swrt,
                   float sky_vis, float3 sky_bent, float gdf_ao)
{
    if (globals.ibl.y < 0)
        return globals.ambient.rgb * albedo * ao;

    float n_dot_v = max(dot(n, v), 1e-4);
    float3 r = reflect(-v, n);

    float3 f = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    float3 kd = (1.0 - f) * (1.0 - metallic);

    // 레퍼런스식 indoor skylight occlusion: sample the diffuse skylight along the bent normal (the
    // unoccluded average direction) and occlude by the scene-scale sky visibility, so an enclosed
    // interior isn't flooded by the full (blue) diffuse skylight and instead picks up light biased
    // toward the actual sky opening. sky_vis = 1 with a zero bent normal -> exact no-op (gallery).
    float3 diffuse = occlude_sky_diffuse_bent(n, sky_vis, sky_bent, globals.ibl.y) * albedo;

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

    // Bent-normal specular occlusion (opt-in): occlude the prefilter-CUBE specular by how much of
    // the reflection cone the scene blocks (cone-cone vs the bent-normal visibility cone). Only the
    // cube path — the SW-RT reflection (`has_swrt`) already resolves occlusion along its ray, so it
    // is left untouched (the reflection-track rule). bent == 0 / spec_occlusion off => no-op.
    if (pc.spec_occlusion != 0u && !has_swrt)
    {
        float so = spec_occlusion_bent(sky_bent, sky_vis, reflect(-v, n), roughness);
        float lum = dot(specular, float3(0.2126, 0.7152, 0.0722));
        // Same aperture shaping as the diffuse leak (lockstep; dead on content — has_swrt).
        float leak_s = 1.0 - so;
        if (pc.skyvis_tint_v0 > 0.0)
            leak_s *= saturate(saturate(sky_vis) / pc.skyvis_tint_v0);
        specular = specular * so + (pc.skyvis_tint * lum) * leak_s; // neutral leak (reference ExtraIndirectSpecular)
    }

    // AO (material + GDF far-field) attenuates only the DIFFUSE ambient. The traced SW-RT
    // reflection already accounts for occlusion along its own ray, so multiplying the mirror
    // specular by AO double-counts it — and for a reflective object baked INTO the distance field
    // (which self-occludes -> gdf_ao≈0) it would kill the reflection entirely (a black chrome
    // ball). So AO applies to diffuse only; the specular reflection carries its own occlusion.
    // (Gallery anchor stays byte-identical: its specular pixels have ao·gdf_ao == 1.)
    // Multi-bounce AO (opt-in): recolour the scalar diffuse AO by the albedo so cavities keep the
    // surface's own bounced colour instead of going flat-dark. Off (default) => scalar ao·gdf_ao,
    // byte-identical anchor. `ao·gdf_ao == 1` => ao_multi_bounce returns 1 => unchanged even when on.
    float3 diff_ao = (pc.ao_multibounce != 0u)
                         ? ao_multi_bounce(albedo, ao * gdf_ao)
                         : float3(ao * gdf_ao);
    return kd * diffuse * diff_ao + specular;
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
// PCF / PCSS-lite filter over a shadow-atlas UV. `uv` is the receiver's position in the
// (sub-)tile it must stay within; `uv_min`/`uv_max` clamp the tap disk to that tile so a
// blocker search can't bleed across atlas slots (identity [0,1] for the single map). `texel`
// is one shadow-map texel in atlas UV. Shared by the single-map and CSM paths so both keep the
// same quality (this is the extracted body of the legacy sun_shadow filter — behavior
// preserved for the single map: uv_min=0, uv_max=1, texel=globals.shadow.y).
float shadow_filter(float2 uv, float current, float2 pix, float texel,
                    float2 uv_min, float2 uv_max)
{
    float penumbra_factor = globals.shadow.z;

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
                sum += shadow_tap(clamp(uv + float2(x, y) * texel, uv_min, uv_max), current);
        return sum / 9.0;
    }

    int taps = int(globals.shadow.w);
    if (taps < 1 || taps > SHADOW_MAX_SAMPLES)
        taps = SHADOW_MAX_SAMPLES;

    // 1) Blocker search over a fixed disk; average the occluders' depth.
    float blocker_sum = 0.0;
    int blocker_n = 0;
    for (int i = 0; i < taps; ++i)
    {
        float2 o = mul(rot, POISSON16[i]) * SHADOW_SEARCH_UV;
        float d = g.textures[pc.shadow_index].Sample(g.samp, clamp(uv + o, uv_min, uv_max)).r;
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
        sum += shadow_tap(clamp(uv + o, uv_min, uv_max), current);
    }
    return sum / float(taps);
}

// `shade()` (one light's Cook-Torrance contribution) now lives in the shared
// `pbr_brdf.slang` header included above — same math, shared with the translucency pass.

// Project `world_pos` by a cascade / light view-projection into shadow-atlas UV + depth.
// Returns false when outside the light frustum (or past its far plane) so the caller treats
// the pixel as lit. `atlas_uv` = xy offset, zw scale of the tile within the atlas.
bool project_to_atlas(float4x4 vp, float4 atlas_uv, float3 world_pos, out float2 uv, out float depth)
{
    float4 lc = mul(vp, float4(world_pos, 1.0));
    float3 proj = lc.xyz / lc.w; // ortho: w = 1
    float2 tuv;
    tuv.x = proj.x * 0.5 + 0.5;
    // Backend V flip (Vulkan rasterizes +Y down, D3D12 +Y up) — same convention as the map.
    tuv.y = pc.flip_y != 0 ? (proj.y * 0.5 + 0.5) : (proj.y * -0.5 + 0.5);
    depth = proj.z;
    // Map the [0,1] tile UV into the atlas sub-rect.
    uv = atlas_uv.xy + tuv * atlas_uv.zw;
    return !(tuv.x < 0.0 || tuv.x > 1.0 || tuv.y < 0.0 || tuv.y > 1.0 || proj.z > 1.0);
}

// CSM cascade select: the tightest cascade whose split-far covers the view-space depth. A small
// blend band near each boundary cross-fades to the next cascade so the resolution step isn't a
// hard seam. `view_depth` = distance from the camera along the view direction. Returns the
// sun-visibility [0,1]; also writes the chosen cascade index (for the debug overlay).
float csm_shadow(float3 world_pos, float2 pix, float view_depth, out int cascade_out)
{
    int n = globals.csm_params.x;
    float atlas = float(max(globals.csm_params.w, 1));
    float texel = 1.0 / atlas; // one atlas texel in atlas UV

    // Cascade select by CONTAINMENT (not just the depth split): iterate tightest → widest and
    // pick the first cascade whose frustum actually covers this world point. The split ordering
    // makes near cascades the tightest, so this yields the highest-resolution cascade available
    // for the pixel. Radial view-depth alone can route a point to a cascade whose sphere doesn't
    // contain it (perspective vs radial + the sphere overshoots the slice) → the depth-only
    // select leaves it wrongly lit; containment fixes that.
    int ci = -1;
    float2 uv = float2(0.0);
    float depth = 0.0;
    for (int i = 0; i < n; ++i)
    {
        float2 tuv;
        float td;
        if (project_to_atlas(globals.csm_view_proj[i], globals.csm_atlas_uv[i], world_pos, tuv, td))
        {
            ci = i;
            uv = tuv;
            depth = td;
            break;
        }
    }
    cascade_out = ci;
    if (ci < 0)
        return 1.0; // outside every cascade (beyond the far cascade) -> lit
    float4 rect = globals.csm_atlas_uv[ci];
    float2 uv_min = rect.xy + texel;               // inset one texel so clamp never samples a neighbor
    float2 uv_max = rect.xy + rect.zw - texel;
    float current = depth - globals.shadow.x;
    float vis = shadow_filter(uv, current, pix, texel, uv_min, uv_max);

    // Cross-cascade blend: within the last `csm_opts.x` fraction of this cascade's view-depth
    // coverage, fade to the next (wider) cascade so the resolution step at the split boundary
    // isn't a hard seam. Blend fraction is CPU-owned (CsmConfig::blend_frac, single source).
    if (ci + 1 < n)
    {
        float near_split = (ci == 0) ? 0.0 : globals.csm_split[ci - 1];
        float far_split = globals.csm_split[ci];
        float band = (far_split - near_split) * globals.csm_opts.x;
        float t = saturate((view_depth - (far_split - band)) / max(band, 1e-4));
        if (t > 0.0)
        {
            float2 uv2;
            float depth2;
            if (project_to_atlas(globals.csm_view_proj[ci + 1], globals.csm_atlas_uv[ci + 1],
                                 world_pos, uv2, depth2))
            {
                float4 rect2 = globals.csm_atlas_uv[ci + 1];
                float2 uv2_min = rect2.xy + texel;
                float2 uv2_max = rect2.xy + rect2.zw - texel;
                float cur2 = depth2 - globals.shadow.x;
                float vis2 = shadow_filter(uv2, cur2, pix, texel, uv2_min, uv2_max);
                vis = lerp(vis, vis2, t);
            }
        }
    }
    return vis;
}

// Fraction of the directional sun reaching `world_pos` (1 = fully lit, 0 = fully shadowed).
// `view_depth` = camera-space distance (for the CSM cascade select). The CSM path is taken when
// the seam is on (csm_params.x > 0); otherwise the legacy single-map path runs unchanged.
float sun_shadow(float3 world_pos, float2 pix, float view_depth, out int cascade_out)
{
    cascade_out = -1;
    if (globals.counts.w == 0)
        return 1.0;

    if (globals.csm_params.x > 0)
        return csm_shadow(world_pos, pix, view_depth, cascade_out);

    // Legacy single directional map (byte-identical anchor): identity atlas rect, texel =
    // globals.shadow.y, clamp bounds [0,1] (no-op) → the extracted filter reproduces the
    // original 3x3 / PCSS-lite behavior exactly.
    float2 uv;
    float depth;
    float4 full = float4(0.0, 0.0, 1.0, 1.0);
    if (!project_to_atlas(globals.light_view_proj, full, world_pos, uv, depth))
        return 1.0;
    float current = depth - globals.shadow.x;
    return shadow_filter(uv, current, pix, globals.shadow.y, float2(0.0), float2(1.0));
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
    // DEBUG_VIEW=20: force the ENTIRE scene to a polished near-mirror (metallic 1, low roughness,
    // bright neutral F0) so every surface shows its reflection — used to watch the SW-RT reflection
    // fill in / converge over frames. Not a color-replace debug view: the normal lit result stands.
    if (globals.counts.y == 20)
    {
        metallic = 1.0;
        roughness = 0.05;
        albedo = float3(0.95, 0.95, 0.95);
    }
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

    // View-space distance for the CSM cascade select. Radial distance from the eye — a stable,
    // view-direction-agnostic proxy for the perspective depth the CPU split scheme tiled (the
    // single-map path ignores it). `cascade_idx` is the chosen cascade for the debug overlay.
    float view_depth = length(globals.camera_pos.xyz - world_pos);
    int cascade_idx = -1;

    // Directional sun (attenuated by the shadow map / cascade atlas).
    {
        float3 l = normalize(globals.sun_direction.xyz);
        float3 radiance = globals.sun_color.rgb * globals.sun_color.a;
        lo += shade(n, v, l, radiance, albedo, metallic, roughness, f0)
            * sun_shadow(world_pos, input.pos.xy, view_depth, cascade_idx);
    }

    // Point lights. point_color.a is the luminous intensity in CANDELA (lm/sr) — the same
    // photometric unit as the lux sun, so the single end-of-pass exposure maps both. The inverse-
    // square `atten` converts candela → illuminance (lux) at the surface: E = I / d². (glTF
    // KHR_lights_punctual authors point/spot intensity in candela already; a light specified in
    // lumens Φ converts as I = Φ / 4π.)
    //
    // Clustered path (PR-6): when the cluster buffers are bound, iterate only this pixel's
    // froxel light list. The list was built in ascending global-light-index order, and we
    // read it in that same order, so with few lights (all in-cluster) the accumulation is
    // bit-identical to the brute-force loop below. `lights_touched` feeds the debug heatmap.
    uint lights_touched = 0;
    if (pc.cluster_grid_buf != NO_TEXTURE)
    {
        RWByteAddressBuffer lbuf = g.storage_buffers[pc.cluster_light_buf];
        if (pc.cluster_index_buf == NO_TEXTURE)
        {
            // Brute-force-over-storage A/B baseline (CLUSTERED_BRUTE=1): loop ALL lights in the
            // buffer, no froxel list. Same lights + same shader as the clustered path — only the
            // culling differs, so the PROFILE_GPU delta isolates the culling's payoff at scale.
            for (uint li = 0; li < pc.cluster_light_count; ++li)
            {
                Light L = load_light(lbuf, li);
                float3 to_light = L.position - world_pos;
                float dist = length(to_light);
                float3 l = to_light / max(dist, 1e-4);
                float atten = 1.0 / max(dist * dist, 1e-4);
                float3 radiance = L.color * L.intensity * atten;
                lo += shade(n, v, l, radiance, albedo, metallic, roughness, f0);
            }
            lights_touched = pc.cluster_light_count;
        }
        else
        {
            // Positive linear view depth of this surface -> its froxel Z slice.
            float view_z = -dot(globals.cluster_view_z_row, float4(world_pos, 1.0));
            // Screen UV from the fullscreen fragment (input.uv is the [0,1] triangle uv;
            // matches the cluster build's D3D-style uv->ndc — no Y flip, cancels in world).
            float2 cuv = input.uv;
            uint ci = cluster_index_for(cuv, view_z, globals.cluster_params.x,
                                        globals.cluster_params.y);
            RWByteAddressBuffer grid = g.storage_buffers[pc.cluster_grid_buf];
            RWByteAddressBuffer index_list = g.storage_buffers[pc.cluster_index_buf];
            uint lcount = grid.Load(ci * 4);
            uint lbase = ci * MAX_LIGHTS_PER_CLUSTER;
            for (uint k = 0; k < lcount; ++k)
            {
                uint li = index_list.Load((lbase + k) * 4);
                Light L = load_light(lbuf, li);
                float3 to_light = L.position - world_pos;
                float dist = length(to_light);
                float3 l = to_light / max(dist, 1e-4);
                float atten = 1.0 / max(dist * dist, 1e-4);
                float3 radiance = L.color * L.intensity * atten;
                lo += shade(n, v, l, radiance, albedo, metallic, roughness, f0);
            }
            lights_touched = lcount;
        }
    }
    else
    {
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
        lights_touched = uint(count);
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
    // skylight. Absent (0xFFFFFFFF) -> 1.0 (exact no-op, gallery anchor). `.gba` carries the bent
    // normal (the unoccluded average direction) when the producer wrote one; a zero vector = scalar
    // fallback (sample the skylight along the surface normal = the legacy path, byte-identical).
    float sky_vis = 1.0;
    float3 sky_bent = float3(0.0, 0.0, 0.0);
    if (pc.skyvis_index != NO_TEXTURE)
    {
        float4 sv = g.textures[pc.skyvis_index].Sample(g.samp, input.uv);
        sky_vis = saturate(sv.r);
        sky_bent = sv.gba;
    }

    // `ambient_ibl` already applied the GDF far-field AO to the DIFFUSE ambient and left the traced
    // specular reflection un-occluded (it carries its own ray occlusion), so there is no separate
    // `ambient *= gdf_ao` here — that would double-count AO on diffuse and wrongly darken specular.
    float3 ambient = ambient_ibl(n, v, world_pos, albedo, metallic, roughness, ao, f0,
                                 swrt_refl, has_swrt, sky_vis, sky_bent, gdf_ao);

    // Stage C3: 1-bounce diffuse GI marched against the scene GDF in a prior compute
    // pass. The GI image holds the mean incoming radiance (indirect irradiance) from
    // scene geometry; weight it by the diffuse albedo (× 1-metallic) and add it to the
    // ambient — the scene-bounce term IBL·AO can't capture. Absent -> 0.
    float3 gi = float3(0.0);
    float3 gi_e = float3(0.0); // E(n) before receiver albedo / multi-bounce boost (debug view 15)
    if (pc.gdf_gi_index != NO_TEXTURE)
    {
        float3 e = g.textures[pc.gdf_gi_index].Sample(g.samp, input.uv).rgb;
        gi_e = e;
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
    // 5 world pos, 6 AO, 7 direct light, 8 IBL ambient, 9 GDF AO, 10 GDF GI,
    // 11 light-count heatmap, 12 reflection buffer (the raw SW-RT reflected radiance sampled at
    // this pixel — what a mirror here would show; 0 where no reflection is bound),
    // 13 sky visibility, 14 bent normal, 15 raw GI E(n) (oracle domain). 0 = lit.
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
    else if (dbg == 11)                            // clustered light-count heatmap
    {
        // Per-pixel lights-touched as a blue->green->red ramp (0 = black). Normalised to a
        // fixed 16-light budget so hot clusters (many overlapping lights) glow red.
        float t = saturate(float(lights_touched) / 16.0);
        color = float3(saturate(t * 2.0 - 1.0), saturate(1.0 - abs(t - 0.5) * 2.0), saturate(1.0 - t * 2.0));
        if (lights_touched == 0)
            color = float3(0.0);
    }
    // 12: the reflection buffer — the raw SW-RT reflected radiance this pixel receives. MAGENTA
    // flags a pathological value (NaN / Inf / negative) so corruption is obvious; otherwise the raw
    // reflected radiance (no exposure) so magnitudes read directly. 0 when no reflection is bound.
    else if (dbg == 12)
    {
        bool bad = any(isnan(swrt_refl)) || any(isinf(swrt_refl)) || any(swrt_refl < 0.0);
        color = bad ? float3(1.0, 0.0, 1.0) : (has_swrt ? swrt_refl : float3(0.0));
    }
    else if (dbg == 13) color = float3(sky_vis);           // directional sky visibility V(n)
    else if (dbg == 14)                                    // bent normal (unoccluded avg dir); gray = none
        color = length(sky_bent) > 1e-4 ? (normalize(sky_bent) * 0.5 + 0.5) : float3(0.5);
    // 15: raw GDF GI E(n) — the GI image before the receiver albedo and multi-bounce boost.
    // Same physical quantity as the PT E-view (P8_PT_EVIEW), so the two are directly
    // comparable; view 10 is albedo-weighted and NOT oracle-comparable.
    else if (dbg == 15) color = gi_e * exposure;

    // PR-7 CSM cascade-index overlay (csm_params.y): tint the lit color by which cascade
    // shades each pixel so the split boundaries are inspectable. Only active with CSM on.
    if (globals.csm_params.y != 0 && globals.csm_params.x > 0 && cascade_idx >= 0)
    {
        // 4 distinct hues for cascades 0..3 (near→far): red, green, blue, yellow.
        float3 tints[4] = {
            float3(1.0, 0.3, 0.3), float3(0.3, 1.0, 0.3),
            float3(0.3, 0.5, 1.0), float3(1.0, 1.0, 0.3),
        };
        float3 tint = tints[clamp(cascade_idx, 0, 3)];
        color = lerp(color, color * tint * 1.5 + tint * 0.15, 0.5);
    }

    return float4(color, 1.0);
}
