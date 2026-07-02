// Shared Cook-Torrance metallic-roughness BRDF helpers — the single source for the
// direct-light shading math used by BOTH the deferred lighting pass (`pbr.slang`) and
// the forward translucency pass (`translucent.slang`, PR-3). Extracting these into a
// header keeps the two passes bit-for-bit consistent (no drift between the opaque
// deferred BRDF and the translucent forward BRDF) without duplicating the math.
//
// These are pure functions of their arguments (no resource / globals access), so this
// header is safe to `#include` from any shader. `PI` must be defined by the includer
// before this header (both consumers already declare `static const float PI` for their
// other math), matching the pre-extraction inline definitions exactly — so the
// generated code for `pbr.slang` is unchanged (byte-identical golden anchor).
#ifndef PBR_BRDF_SLANG
#define PBR_BRDF_SLANG

// GGX / Trowbridge-Reitz normal distribution.
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

#endif // PBR_BRDF_SLANG
