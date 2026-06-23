// Shared path-tracer code for the inline (rt_path) and pipeline (rt_pipeline)
// backends (Phase 8 + ground-truth PBR track). Keeping the push constants,
// material record, RNG, sky model, sampling, and geometry/material fetch in one
// place guarantees the two paths stay bit-identical (the cross-backend and
// inline-vs-pipeline parity tests depend on it).

#include "bindless.slang"

static const float PI = 3.14159265358979;
static const float INV_PI = 0.31830988618379;
static const uint NO_TEXTURE = 0xFFFFFFFFu;

// Push constants shared by both path-tracer entry points.
struct PushConstants
{
    float4x4 inv_view_proj; // clip -> world
    float4 cam_pos;         // xyz camera world position
    float4 sun;             // xyz direction toward the sun, w = intensity
    uint out_index;         // bindless storage-image index (averaged output)
    uint accum_index;       // bindless storage-buffer index (float4 sum per pixel)
    uint inst_index;        // bindless storage-buffer index (instance table)
    uint frame;             // accumulation frame (0 = first, clears the sum)
    uint width;
    uint height;
    uint flip_y;            // bit0: 1 = Vulkan (Y down); bit1: 1 = Cornell box mode
    uint spp;               // samples per dispatch
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

// 64-byte per-instance record (matches the host packing in build_pt_instance_table).
// Carries the full glTF metallic-roughness material so the path tracer can shade
// with the same model as the rasterizer (ground-truth PBR track).
struct Instance
{
    uint vtx;          // bindless storage index of the 32-byte/vertex buffer
    uint idx;          // bindless storage index of the u32 index buffer
    uint tex_base;     // base-color texture (NO_TEXTURE if absent)
    uint tex_mr;       // metallic-roughness texture (g = roughness, b = metallic)
    float4 base_color; // rgb base color, a = emissive scale
    float4 params;     // x metallic, y roughness, z ao, w unused
    uint tex_normal;   // tangent-space normal map
    uint tex_emissive; // emissive texture
    uint pad0;
    uint pad1;
};

// PCG hash RNG (deterministic per seed -> identical across backends).
uint pcg(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}
float rnd(inout uint s)
{
    return float(pcg(s) & 0xFFFFFFu) / float(0x1000000);
}

// Gradient sky + HDR sun disk. `with_sun` adds the bright sun disk; it is only
// enabled for primary (camera) rays. Diffuse/indirect rays use `with_sun = false`
// because the sun's contribution is added analytically as a direct light (the
// shadow-ray / NEE term) — sampling the disk again would double-count the sun.
float3 sky(float3 dir, float3 sun_dir, float sun_i, bool with_sun)
{
    float up = dir.y;
    float3 zenith = float3(0.04, 0.16, 0.55);
    float3 horizon = float3(0.50, 0.66, 0.92);
    float3 ground = float3(0.16, 0.15, 0.14);
    float3 col = up >= 0.0 ? lerp(horizon, zenith, pow(saturate(up), 0.45))
                           : lerp(horizon, ground, saturate(-up * 6.0));
    if (with_sun)
    {
        float c = saturate(dot(dir, sun_dir));
        col += sun_i * (pow(c, 4000.0) * 6.0 + pow(c, 32.0) * 0.4);
    }
    return col;
}

// Orthonormal basis around a unit normal.
void onb(float3 n, out float3 t, out float3 b)
{
    t = normalize(abs(n.x) > 0.9 ? cross(n, float3(0, 1, 0)) : cross(n, float3(1, 0, 0)));
    b = cross(n, t);
}

// Cosine-weighted hemisphere sample about `n`.
float3 cosine_hemisphere(float3 n, inout uint rng)
{
    float u1 = rnd(rng);
    float u2 = rnd(rng);
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    float3 t, b;
    onb(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}

float luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// ---- Microfacet BSDF (Cook-Torrance, GGX) -------------------------------------
// Physically-based metallic-roughness BSDF. Unlike the rasterizer (which uses the
// UE direct-light G approximation k=(r+1)^2/8), the path tracer uses the exact
// Smith-GGX masking term so it is an unbiased reference the rasterizer converges
// toward. `alpha = roughness^2`.

// GGX normal distribution.
float d_ggx(float n_dot_h, float alpha)
{
    float a2 = alpha * alpha;
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Exact Smith-GGX masking/shadowing for one direction (separable Smith: the full
// G is g1(v) * g1(l)).
float g1_smith(float n_dot_s, float alpha)
{
    float a2 = alpha * alpha;
    float s2 = n_dot_s * n_dot_s;
    return 2.0 * n_dot_s / max(n_dot_s + sqrt(a2 + (1.0 - a2) * s2), 1e-7);
}

float3 f_schlick(float cos_theta, float3 f0)
{
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

// Surface BSDF parameters derived from a hit's material.
struct Surf
{
    float3 diffuse; // Lambert albedo (energy not in the specular lobe)
    float3 f0;      // specular reflectance at normal incidence
    float alpha;    // GGX roughness^2
};

Surf make_surf(float3 base_color, float metallic, float roughness)
{
    Surf s;
    s.diffuse = base_color * (1.0 - metallic);
    s.f0 = lerp(float3(0.04, 0.04, 0.04), base_color, metallic);
    float r = clamp(roughness, 0.02, 1.0); // clamp to avoid a singular specular lobe
    s.alpha = r * r;
    return s;
}

// Evaluate the full BSDF (diffuse + specular) for a light direction `l` seen from
// `v` about normal `n`. Used for direct-light next-event estimation.
float3 eval_bsdf(float3 n, float3 v, float3 l, Surf s)
{
    float ndl = dot(n, l);
    float ndv = dot(n, v);
    if (ndl <= 0.0 || ndv <= 0.0)
        return float3(0, 0, 0);
    float3 h = normalize(v + l);
    float ndh = saturate(dot(n, h));
    float vdh = saturate(dot(v, h));
    float d = d_ggx(ndh, s.alpha);
    float gg = g1_smith(ndv, s.alpha) * g1_smith(ndl, s.alpha);
    float3 fr = f_schlick(vdh, s.f0);
    float3 spec = d * gg * fr / max(4.0 * ndv * ndl, 1e-7);
    float3 diff = s.diffuse * INV_PI * (1.0 - fr); // energy split (kd = 1 - F)
    return diff + spec;
}

// Sample the visible normal distribution (Heitz 2018) in the local frame where the
// surface normal is +Z. Returns the microfacet normal.
float3 sample_ggx_vndf(float3 ve, float alpha, float u1, float u2)
{
    float3 vh = normalize(float3(alpha * ve.x, alpha * ve.y, ve.z));
    float lensq = vh.x * vh.x + vh.y * vh.y;
    float3 t1 = lensq > 0.0 ? float3(-vh.y, vh.x, 0.0) * rsqrt(lensq) : float3(1, 0, 0);
    float3 t2 = cross(vh, t1);
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    float tx = r * cos(phi);
    float ty = r * sin(phi);
    float sc = 0.5 * (1.0 + vh.z);
    ty = (1.0 - sc) * sqrt(saturate(1.0 - tx * tx)) + sc * ty;
    float3 nh = tx * t1 + ty * t2 + sqrt(saturate(1.0 - tx * tx - ty * ty)) * vh;
    return normalize(float3(alpha * nh.x, alpha * nh.y, max(1e-6, nh.z)));
}

// A sampled bounce direction + its throughput weight (BRDF*cos/pdf, already
// divided by the lobe-selection probability so the estimate is unbiased).
struct BsdfSample
{
    float3 dir;
    float3 weight;
    bool valid;
};

// Importance-sample the BSDF: pick the diffuse or specular lobe (Fresnel-weighted),
// sample it, and return the throughput weight. Specular uses VNDF sampling, whose
// weight reduces to F * G1(l); diffuse uses cosine sampling, weight = diffuse*(1-F).
// Each branch consumes the same number of RNG draws (1 select + 2 sample) so the
// inline and pipeline paths stay in lockstep.
BsdfSample sample_bsdf(float3 n, float3 v, Surf s, inout uint rng)
{
    BsdfSample r;
    r.dir = float3(0, 0, 0);
    r.weight = float3(0, 0, 0);
    r.valid = false;
    float ndv = dot(n, v);
    if (ndv <= 0.0)
        return r;

    // Lobe-selection probability from the Fresnel-weighted specular vs diffuse
    // reflectance (clamped so neither lobe is ever sampled with zero probability).
    float3 f_ndv = f_schlick(ndv, s.f0);
    float pd = luminance(s.diffuse);
    float psp = luminance(f_ndv);
    float p_spec = clamp(psp / max(psp + pd, 1e-4), 0.1, 0.9);

    float3 t, b;
    onb(n, t, b);
    float sel = rnd(rng);
    if (sel < p_spec)
    {
        // Specular: sample a microfacet normal, reflect the view about it.
        float3 ve = float3(dot(v, t), dot(v, b), dot(v, n));
        float3 hh = sample_ggx_vndf(ve, s.alpha, rnd(rng), rnd(rng));
        float3 ll = reflect(-ve, hh); // 2*dot(ve,hh)*hh - ve
        if (ll.z <= 0.0)
            return r;
        float vdh = saturate(dot(ve, hh));
        float3 fr = f_schlick(vdh, s.f0);
        r.weight = fr * g1_smith(ll.z, s.alpha) / p_spec;
        r.dir = normalize(ll.x * t + ll.y * b + ll.z * n);
        r.valid = true;
    }
    else
    {
        // Diffuse: cosine-weighted hemisphere.
        float3 l = cosine_hemisphere(n, rng);
        float ndl = dot(n, l);
        if (ndl <= 0.0)
            return r;
        float3 h = normalize(v + l);
        float vdh = saturate(dot(v, h));
        float3 fr = f_schlick(vdh, s.f0);
        r.weight = s.diffuse * (1.0 - fr) / (1.0 - p_spec);
        r.dir = l;
        r.valid = true;
    }
    return r;
}

// Read an instance record from the table.
Instance load_instance(uint id)
{
    uint base = id * 64u;
    Instance inst;
    inst.vtx = g.storage_buffers[pc.inst_index].Load(base + 0u);
    inst.idx = g.storage_buffers[pc.inst_index].Load(base + 4u);
    inst.tex_base = g.storage_buffers[pc.inst_index].Load(base + 8u);
    inst.tex_mr = g.storage_buffers[pc.inst_index].Load(base + 12u);
    inst.base_color = asfloat(g.storage_buffers[pc.inst_index].Load4(base + 16u));
    inst.params = asfloat(g.storage_buffers[pc.inst_index].Load4(base + 32u));
    inst.tex_normal = g.storage_buffers[pc.inst_index].Load(base + 48u);
    inst.tex_emissive = g.storage_buffers[pc.inst_index].Load(base + 52u);
    inst.pad0 = 0u;
    inst.pad1 = 0u;
    return inst;
}

// Barycentric-interpolated, normalized surface normal for a hit triangle.
// Vertex layout: position @0 (float3), normal @12 (float3), uv @24 (float2), 32B.
float3 hit_normal(Instance inst, uint prim, float2 bary)
{
    uint i0 = g.storage_buffers[inst.idx].Load(prim * 12u + 0u);
    uint i1 = g.storage_buffers[inst.idx].Load(prim * 12u + 4u);
    uint i2 = g.storage_buffers[inst.idx].Load(prim * 12u + 8u);
    float3 n0 = asfloat(g.storage_buffers[inst.vtx].Load3(i0 * 32u + 12u));
    float3 n1 = asfloat(g.storage_buffers[inst.vtx].Load3(i1 * 32u + 12u));
    float3 n2 = asfloat(g.storage_buffers[inst.vtx].Load3(i2 * 32u + 12u));
    float b0 = 1.0 - bary.x - bary.y;
    return normalize(n0 * b0 + n1 * bary.x + n2 * bary.y);
}
