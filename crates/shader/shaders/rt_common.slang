// Shared path-tracer code for the inline (rt_path) and pipeline (rt_pipeline)
// backends (Phase 8 + ground-truth PBR track). Keeping the push constants,
// material record, RNG, sky model, sampling, and geometry/material fetch in one
// place guarantees the two paths stay bit-identical (the cross-backend and
// inline-vs-pipeline parity tests depend on it).

#include "bindless.slang"
#include "sky_common.slang"

static const float PI = 3.14159265358979;
static const float INV_PI = 0.31830988618379;
static const uint NO_TEXTURE = 0xFFFFFFFFu;

// Path-tracer transport controls (shared so the inline and pipeline paths match).
// Russian roulette keeps the estimator unbiased while allowing deep paths: after
// RR_START_BOUNCE bounces a path survives with probability ~ its throughput.
static const uint MAX_BOUNCES = 8u;
static const uint RR_START_BOUNCE = 3u;
// The sun is a finite disk light (not a delta direction) so shadows have a soft
// penumbra. `SUN_COS_MAX` is the cosine of its angular radius (~1.1 degrees).
static const float SUN_COS_MAX = 0.9998;

// Push constants shared by both path-tracer entry points.
struct PushConstants
{
    float4x4 inv_view_proj; // clip -> world
    float4 cam_pos;         // xyz camera world position; w = instance count (bounds)
    float4 sun;             // xyz direction toward the sun, w = intensity
    uint out_index;         // bindless storage-image index (averaged output)
    uint accum_index;       // bindless storage-buffer index (float4 sum per pixel)
    uint inst_index;        // bindless storage-buffer index (instance table)
    uint frame;             // accumulation frame (0 = first, clears the sum)
    uint width;
    uint height;
    uint flip_y;            // bit0: 1 = Vulkan (Y down); bit1: 1 = Cornell box mode; bit2: E-oracle
                            // view (P8_PT_EVIEW — cosine-hemisphere irradiance at the primary hit,
                            // scene-bounce only: a bounce-1 sky escape contributes 0, matching the
                            // GI volume's stored-E convention; docs/phase-e-oracle-plan.md)
    uint spp;               // samples per dispatch
    float4 sky;             // x = sky_gain (sun:sky ratio), yzw = sky white balance (per-channel)
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
    uint prim_count;   // triangle count (for closest-hit PrimitiveIndex bounds check)
    uint idx_base;     // content mode: this drawable's offset into the shared index
                       // buffer (u32 units); 0 for the per-instance gallery tables
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

// Procedural sky (miss shader). Evaluates the SAME physically-based atmosphere as the
// rasterizer's env-cube capture (`sky_common.slang`), so the parity residual measures
// shading/GI error rather than a sky-model mismatch. `with_sun` adds the bright sun disk;
// it is only enabled for primary (camera) rays. Diffuse/indirect rays use `with_sun = false`
// because the sun's contribution is added analytically as a direct light (the shadow-ray /
// NEE term) — sampling the disk again would double-count the sun. `sky_gain` + white balance
// come from the push (threaded from the host, matching the env capture's values).
float3 sky(float3 dir, float3 sun_dir, float sun_i, bool with_sun)
{
    return procedural_sky(dir, sun_dir, sun_i, pc.sky.x, with_sun) * pc.sky.yzw;
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

// Uniformly sample a direction within a cone of half-angle acos(cos_max) about
// `dir` (for sampling the sun disk in next-event estimation). Consumes 2 RNG draws.
float3 sample_cone(float3 dir, float cos_max, inout uint rng)
{
    float u1 = rnd(rng);
    float u2 = rnd(rng);
    float ct = 1.0 - u1 * (1.0 - cos_max);
    float st = sqrt(saturate(1.0 - ct * ct));
    float phi = 2.0 * PI * u2;
    float3 t, b;
    onb(dir, t, b);
    return normalize(t * (st * cos(phi)) + b * (st * sin(phi)) + dir * ct);
}

// ---- Microfacet BSDF (Cook-Torrance, GGX) -------------------------------------
// Physically-based metallic-roughness BSDF. Unlike the rasterizer (which uses the
// 레퍼런스 엔진 direct-light G approximation k=(r+1)^2/8), the path tracer uses the exact
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

// Content-level instance source (the PT oracle on a level scene): `pc.inst_index`
// packs the Phase 16 E consolidated hit table — drawable-record table in the low
// byte, shared vertex / index storage indices (+1, 0 = off) in bits 8..15 / 16..23.
// The gallery / Cornell tables pass a bare bindless index (< 64, upper bits zero),
// so this predicate is what selects the record format — the legacy path is
// byte-identical.
bool content_instances()
{
    return (pc.inst_index >> 8) != 0u;
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
    inst.prim_count = g.storage_buffers[pc.inst_index].Load(base + 56u);
    inst.idx_base = 0u;
    return inst;
}

// Read a drawable record from the CONSOLIDATED content table (48 B, see
// build_content_hit_table): all meshes share one vertex + one index buffer, so the
// per-drawable record carries an index-buffer offset instead of buffer indices.
Instance load_content_instance(uint id)
{
    uint tbl = pc.inst_index & 0xFFu;
    uint base = id * 48u;
    Instance inst;
    inst.vtx = ((pc.inst_index >> 8) & 0xFFu) - 1u;
    inst.idx = ((pc.inst_index >> 16) & 0xFFu) - 1u;
    inst.idx_base = g.storage_buffers[tbl].Load(base + 0u);
    inst.prim_count = g.storage_buffers[tbl].Load(base + 4u);
    inst.tex_base = g.storage_buffers[tbl].Load(base + 8u);
    inst.tex_mr = g.storage_buffers[tbl].Load(base + 12u);
    inst.base_color = asfloat(g.storage_buffers[tbl].Load4(base + 16u));
    // Content materials are not emitters — the record's .a is opacity, but the path
    // tracer reads base_color.a as the emissive scale (the Cornell light), so zero it.
    inst.base_color.a = 0.0;
    float2 mr = asfloat(g.storage_buffers[tbl].Load2(base + 32u));
    // params.w carries the decal opacity (build_content_hit_table offset 44): >0 marks a
    // coplanar alpha-blended tint mesh the path tracer composites stochastically (see
    // rt_path); the value is the material base-color alpha for the blend. 0 = normal surface.
    float decal = asfloat(g.storage_buffers[tbl].Load(base + 44u));
    inst.params = float4(mr.x, mr.y, 1.0, decal);
    // The consolidated record has no normal / emissive map slots (the HWRT hit
    // lighting doesn't need them); the oracle shades with the geometric normal.
    inst.tex_normal = NO_TEXTURE;
    inst.tex_emissive = NO_TEXTURE;
    return inst;
}

// Bounds-checked hit fetch. The M7 RT-pipeline path (Metal Shader Converter) can
// hand the closest-hit shader an out-of-range InstanceID/PrimitiveIndex for a few
// rays (floor-biased, non-deterministic). Validating BEFORE any
// storage_buffers[].Load avoids the out-of-bounds access entirely — both the
// instance-table read and the index/vertex reads keyed off it — which is the root
// fix (the hit_normal isfinite guard only masked the resulting NaN). Returns false
// for an out-of-range hit; callers treat it as "no valid surface" (terminate path).
bool load_valid_instance(uint id, uint prim, out Instance inst)
{
    inst = (Instance)0;
    if (id >= (uint)pc.cam_pos.w)
        return false;
    inst = content_instances() ? load_content_instance(id) : load_instance(id);
    return prim < inst.prim_count;
}

// Barycentric-interpolated, normalized surface normal for a hit triangle.
// Vertex layout: position @0 (float3), normal @12 (float3), uv @24 (float2), 32B.
float3 hit_normal(Instance inst, uint prim, float2 bary)
{
    uint tri = (inst.idx_base + prim * 3u) * 4u;
    uint i0 = g.storage_buffers[inst.idx].Load(tri + 0u);
    uint i1 = g.storage_buffers[inst.idx].Load(tri + 4u);
    uint i2 = g.storage_buffers[inst.idx].Load(tri + 8u);
    float3 n0 = asfloat(g.storage_buffers[inst.vtx].Load3(i0 * 32u + 12u));
    float3 n1 = asfloat(g.storage_buffers[inst.vtx].Load3(i1 * 32u + 12u));
    float3 n2 = asfloat(g.storage_buffers[inst.vtx].Load3(i2 * 32u + 12u));
    float b0 = 1.0 - bary.x - bary.y;
    float3 n = n0 * b0 + n1 * bary.x + n2 * bary.y;
    // Defense-in-depth for degenerate geometry: a zero-area triangle (coincident
    // vertices) gives a zero/non-finite interpolated normal, whose normalize() is
    // NaN — and a NaN normal yields NaN BSDF radiance that the persistent
    // accumulator bakes into a permanent black speckle. (The out-of-range
    // InstanceID/PrimitiveIndex from the M7 converter path is now rejected upstream
    // by load_valid_instance before any fetch, so this only guards real geometry.)
    // Valid hits keep the exact normalize() result, so inline == pipeline is intact.
    float len2 = dot(n, n);
    return (len2 > 1e-12 && len2 < 1e12) ? normalize(n) : float3(0, 1, 0);
}

// Barycentric-interpolated UV for material texture lookup.
float2 hit_uv(Instance inst, uint prim, float2 bary)
{
    uint tri = (inst.idx_base + prim * 3u) * 4u;
    uint i0 = g.storage_buffers[inst.idx].Load(tri + 0u);
    uint i1 = g.storage_buffers[inst.idx].Load(tri + 4u);
    uint i2 = g.storage_buffers[inst.idx].Load(tri + 8u);
    float2 uv0 = asfloat(g.storage_buffers[inst.vtx].Load2(i0 * 32u + 24u));
    float2 uv1 = asfloat(g.storage_buffers[inst.vtx].Load2(i1 * 32u + 24u));
    float2 uv2 = asfloat(g.storage_buffers[inst.vtx].Load2(i2 * 32u + 24u));
    float b0 = 1.0 - bary.x - bary.y;
    return uv0 * b0 + uv1 * bary.x + uv2 * bary.y;
}

// World-space direction of a primary camera ray through clip-space `ndc` (z=1).
// Shared so the inline and pipeline paths build identical rays + ray cones.
float3 primary_ray_dir(float2 ndc)
{
    float4 world = mul(pc.inv_view_proj, float4(ndc, 1.0, 1.0));
    return normalize(world.xyz / world.w - pc.cam_pos.xyz);
}

// Per-pixel primary ray-cone spread (full angle), from the divergence of two
// adjacent pixel rays. Drives texture LOD since ray shaders lack derivatives.
float primary_cone_spread(float2 ndc)
{
    float2 ndc_dx = ndc + float2(2.0 / float(pc.width), 0.0);
    return length(primary_ray_dir(ndc_dx) - primary_ray_dir(ndc));
}

// Texture-independent ray-cone LOD term for a hit (Akenine-Möller, planar approx).
// `cone_width` = cone footprint width at the hit (Σ t*spread); `cos_theta` =
// |n·rayDir|. Per-texture `0.5*log2(w*h)` is added in sample_texture_lod.
float hit_lod_base(Instance inst, uint prim, float cone_width, float cos_theta)
{
    uint tri = (inst.idx_base + prim * 3u) * 4u;
    uint i0 = g.storage_buffers[inst.idx].Load(tri + 0u);
    uint i1 = g.storage_buffers[inst.idx].Load(tri + 4u);
    uint i2 = g.storage_buffers[inst.idx].Load(tri + 8u);
    float3 p0 = asfloat(g.storage_buffers[inst.vtx].Load3(i0 * 32u + 0u));
    float3 p1 = asfloat(g.storage_buffers[inst.vtx].Load3(i1 * 32u + 0u));
    float3 p2 = asfloat(g.storage_buffers[inst.vtx].Load3(i2 * 32u + 0u));
    float2 uv0 = asfloat(g.storage_buffers[inst.vtx].Load2(i0 * 32u + 24u));
    float2 uv1 = asfloat(g.storage_buffers[inst.vtx].Load2(i1 * 32u + 24u));
    float2 uv2 = asfloat(g.storage_buffers[inst.vtx].Load2(i2 * 32u + 24u));
    float a_uv = abs((uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv2.x - uv0.x) * (uv1.y - uv0.y));
    float a_world = length(cross(p1 - p0, p2 - p0));
    float ratio = a_world > 1e-12 ? a_uv / a_world : 0.0;
    return 0.5 * log2(max(ratio, 1e-20)) + log2(max(cone_width, 1e-8) / max(cos_theta, 1e-3));
}

// The material index varies per ray (per hit instance); a sampled-image array
// accessed with a non-uniform index needs the NonUniform decoration or Vulkan
// reads the wrong descriptor in the ray-tracing stages (compute happens to be
// uniform per workgroup, so it works there without it; D3D12 is lenient).
// Metal indexes argument-buffer arrays non-uniformly without any decoration, and
// Slang's Metal target rejects the intrinsic (E36107), so build.rs defines
// RT_METAL_TARGET for the direct slangc->metallib compile to drop it there.
#ifdef RT_METAL_TARGET
#define RT_NONUNIFORM(i) (i)
#else
#define RT_NONUNIFORM(i) NonUniformResourceIndex(i)
#endif

// HW-sampled material fetch at an explicit ray-cone LOD. The shared trilinear/repeat
// sampler `g.samp` is in the engine bindless block (inline path) and a static sampler
// in the M7 converter root signature (pipeline path), so both paths mip-filter
// identically. `lod_base` is the texture-independent term from hit_lod_base.
float4 sample_texture_lod(uint tex, float2 uv, float lod_base)
{
    uint width;
    uint height;
    g.textures[RT_NONUNIFORM(tex)].GetDimensions(width, height);
    if (width == 0u || height == 0u)
        return float4(1, 1, 1, 1);
    float lod = max(lod_base + 0.5 * log2(float(max(width * height, 1u))), 0.0);
    return g.textures[RT_NONUNIFORM(tex)].SampleLevel(g.samp, uv, lod);
}

float3 hit_shading_normal(Instance inst, uint prim, float2 bary, float3 geom_n, float2 uv,
                          float lod_base)
{
    if (inst.tex_normal == NO_TEXTURE)
        return geom_n;

    float3 tangent_normal = sample_texture_lod(inst.tex_normal, uv, lod_base).xyz * 2.0 - 1.0;
    if (dot(tangent_normal, tangent_normal) < 1e-6)
        return geom_n;

    uint tri = (inst.idx_base + prim * 3u) * 4u;
    uint i0 = g.storage_buffers[inst.idx].Load(tri + 0u);
    uint i1 = g.storage_buffers[inst.idx].Load(tri + 4u);
    uint i2 = g.storage_buffers[inst.idx].Load(tri + 8u);
    float3 p0 = asfloat(g.storage_buffers[inst.vtx].Load3(i0 * 32u + 0u));
    float3 p1 = asfloat(g.storage_buffers[inst.vtx].Load3(i1 * 32u + 0u));
    float3 p2 = asfloat(g.storage_buffers[inst.vtx].Load3(i2 * 32u + 0u));
    float2 uv0 = asfloat(g.storage_buffers[inst.vtx].Load2(i0 * 32u + 24u));
    float2 uv1 = asfloat(g.storage_buffers[inst.vtx].Load2(i1 * 32u + 24u));
    float2 uv2 = asfloat(g.storage_buffers[inst.vtx].Load2(i2 * 32u + 24u));

    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;
    float2 duv1 = uv1 - uv0;
    float2 duv2 = uv2 - uv0;
    float det = duv1.x * duv2.y - duv1.y * duv2.x;

    float3 t;
    float3 b;
    if (abs(det) > 1e-8)
    {
        float inv_det = 1.0 / det;
        t = (e1 * duv2.y - e2 * duv1.y) * inv_det;
        b = (e2 * duv1.x - e1 * duv2.x) * inv_det;
        if (dot(t, t) < 1e-8 || dot(b, b) < 1e-8)
        {
            onb(geom_n, t, b);
        }
        else
        {
            t = normalize(t - geom_n * dot(geom_n, t));
            b = normalize(b - geom_n * dot(geom_n, b));
            if (dot(cross(t, b), geom_n) < 0.0)
                b = -b;
        }
    }
    else
    {
        onb(geom_n, t, b);
    }

    float3 mapped = normalize(tangent_normal.x * t + tangent_normal.y * b + tangent_normal.z * geom_n);
    // Fall back to the (already sanitized) geometric normal if the tangent-space
    // basis degenerated to a non-finite mapped normal (same bad-attribute origin).
    if (!all(isfinite(mapped)))
        return geom_n;
    return dot(mapped, geom_n) < 0.0 ? -mapped : mapped;
}

struct RtMaterialSample
{
    float3 base_color;
    float metallic;
    float roughness;
    float ao;
    float3 emissive;
};

RtMaterialSample sample_material(Instance inst, float2 uv, float lod_base)
{
    RtMaterialSample m;
    m.base_color = inst.base_color.rgb;
    if (inst.tex_base != NO_TEXTURE)
        m.base_color *= sample_texture_lod(inst.tex_base, uv, lod_base).rgb;

    m.metallic = inst.params.x;
    m.roughness = inst.params.y;
    if (inst.tex_mr != NO_TEXTURE)
    {
        // glTF metallic-roughness: G = roughness, B = metallic.
        float3 mr = sample_texture_lod(inst.tex_mr, uv, lod_base).rgb;
        m.roughness *= mr.g;
        m.metallic *= mr.b;
    }

    m.ao = inst.params.z;
    m.emissive = m.base_color * inst.base_color.a;
    if (inst.tex_emissive != NO_TEXTURE)
        m.emissive += sample_texture_lod(inst.tex_emissive, uv, lod_base).rgb * inst.base_color.a;

    m.metallic = saturate(m.metallic);
    m.roughness = clamp(m.roughness, 0.02, 1.0);
    return m;
}

// --- Anisotropic texture filtering for primary hits (SampleGrad) -----------------
// The ray-cone LOD above is isotropic: at grazing angles it inflates the LOD by
// 1/cos_theta and over-blurs the whole footprint, collapsing floor tiles into
// view-aligned smears (the path tracer's counterpart of the raster's no-anisotropy
// stripes). For the PRIMARY hit we reconstruct the anisotropic footprint instead —
// the ray-cone disk projected onto the surface is an ellipse whose major axis is
// stretched 1/cos_theta along the in-plane view direction — map both axes to UV via
// the triangle's dP/dUV Jacobian, and feed SampleGrad so the hardware anisotropic
// wrap sampler (`samp_wrap`, maxAnisotropy 16) keeps the across-view tile joints.
// Content scenes only; the isotropic path stays byte-for-byte for gallery/Cornell.
float4 sample_texture_grad(uint tex, float2 uv, float2 ddx, float2 ddy)
{
    uint width;
    uint height;
    g.textures[RT_NONUNIFORM(tex)].GetDimensions(width, height);
    if (width == 0u || height == 0u)
        return float4(1, 1, 1, 1);
    return g.textures[RT_NONUNIFORM(tex)].SampleGrad(g.samp_wrap, uv, ddx, ddy);
}

// UV-space gradients of the primary-hit footprint. `o2w` rotates the object-space
// triangle Jacobian into world space so the world footprint maps to UV correctly
// under instance rotation/scale. `cone_width` = footprint width at the hit.
void hit_uv_grads(Instance inst, uint prim, float3x3 o2w, float cone_width, float3 n,
                  float3 dir, out float2 ddx_uv, out float2 ddy_uv)
{
    ddx_uv = float2(0, 0);
    ddy_uv = float2(0, 0);
    uint tri = (inst.idx_base + prim * 3u) * 4u;
    uint i0 = g.storage_buffers[inst.idx].Load(tri + 0u);
    uint i1 = g.storage_buffers[inst.idx].Load(tri + 4u);
    uint i2 = g.storage_buffers[inst.idx].Load(tri + 8u);
    float3 p0 = asfloat(g.storage_buffers[inst.vtx].Load3(i0 * 32u + 0u));
    float3 p1 = asfloat(g.storage_buffers[inst.vtx].Load3(i1 * 32u + 0u));
    float3 p2 = asfloat(g.storage_buffers[inst.vtx].Load3(i2 * 32u + 0u));
    float2 uv0 = asfloat(g.storage_buffers[inst.vtx].Load2(i0 * 32u + 24u));
    float2 uv1 = asfloat(g.storage_buffers[inst.vtx].Load2(i1 * 32u + 24u));
    float2 uv2 = asfloat(g.storage_buffers[inst.vtx].Load2(i2 * 32u + 24u));

    float3 e1 = mul(o2w, p1 - p0);
    float3 e2 = mul(o2w, p2 - p0);
    float2 duv1 = uv1 - uv0;
    float2 duv2 = uv2 - uv0;
    float det = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(det) < 1e-12)
        return;
    float inv_det = 1.0 / det;
    float3 dpdu = (e1 * duv2.y - e2 * duv1.y) * inv_det; // world dP/du
    float3 dpdv = (e2 * duv1.x - e1 * duv2.x) * inv_det; // world dP/dv

    // Ray-cone disk (radius r ⟂ ray) projected onto the surface plane: major axis
    // along the in-plane view direction stretched 1/cos_theta, minor axis = r.
    float r = max(cone_width, 0.0) * 0.5;
    float cos_theta = max(abs(dot(n, dir)), 1e-3);
    float3 d_proj = dir - n * dot(dir, n);
    float dlen = length(d_proj);
    float3 axis_major;
    float3 axis_minor;
    if (dlen < 1e-5)
    {
        onb(n, axis_major, axis_minor); // head-on: isotropic disk
        axis_major *= r;
        axis_minor *= r;
    }
    else
    {
        float3 t_along = d_proj / dlen;
        axis_major = t_along * (r / cos_theta);
        axis_minor = normalize(cross(n, t_along)) * r;
    }

    // Solve each world footprint axis for its (du,dv) via the Gram matrix of
    // (dpdu,dpdv); both axes lie in the surface plane, so the solve is exact.
    float aa = dot(dpdu, dpdu);
    float ab = dot(dpdu, dpdv);
    float bb = dot(dpdv, dpdv);
    float gdet = aa * bb - ab * ab;
    if (gdet < 1e-20)
        return;
    float ginv = 1.0 / gdet;
    float wa = dot(axis_major, dpdu);
    float wb = dot(axis_major, dpdv);
    ddx_uv = float2(wa * bb - wb * ab, aa * wb - ab * wa) * ginv;
    wa = dot(axis_minor, dpdu);
    wb = dot(axis_minor, dpdv);
    ddy_uv = float2(wa * bb - wb * ab, aa * wb - ab * wa) * ginv;
}

// Material fetch at explicit UV gradients (anisotropic). Mirrors sample_material but
// draws no RNG, so swapping it in on the primary hit changes only the filtered texel
// values — the path/lighting and the random stream stay identical.
RtMaterialSample sample_material_grad(Instance inst, float2 uv, float2 ddx, float2 ddy)
{
    RtMaterialSample m;
    m.base_color = inst.base_color.rgb;
    if (inst.tex_base != NO_TEXTURE)
        m.base_color *= sample_texture_grad(inst.tex_base, uv, ddx, ddy).rgb;

    m.metallic = inst.params.x;
    m.roughness = inst.params.y;
    if (inst.tex_mr != NO_TEXTURE)
    {
        float3 mr = sample_texture_grad(inst.tex_mr, uv, ddx, ddy).rgb;
        m.roughness *= mr.g;
        m.metallic *= mr.b;
    }

    m.ao = inst.params.z;
    m.emissive = m.base_color * inst.base_color.a;
    if (inst.tex_emissive != NO_TEXTURE)
        m.emissive += sample_texture_grad(inst.tex_emissive, uv, ddx, ddy).rgb * inst.base_color.a;

    m.metallic = saturate(m.metallic);
    m.roughness = clamp(m.roughness, 0.02, 1.0);
    return m;
}
