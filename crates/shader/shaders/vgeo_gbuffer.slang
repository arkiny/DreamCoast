// Phase 14 renderer integration — visibility buffer → Phase-6 G-buffer MRT.
//
// This is the M6 material resolve wired for the REAL deferred renderer: instead of shading the
// resolved attributes itself (the self-contained viewer's `vgeo_resolve.slang`), it writes the four
// deferred G-buffer render targets — albedo / world-normal / material / world-position — plus
// SV_Depth, in the exact format the existing `gbuffer.slang` fill produces. The downstream deferred
// pipeline (shadow / GDF / PBR lighting / tonemap) then consumes it UNCHANGED, so virtual geometry
// becomes a drop-in G-buffer producer rather than a debug view.
//
// Single-material scope (first integration step): the whole cluster set uses one material (the
// loaded model's), matching a single-mesh/single-material object. Per-cluster materials (Sponza,
// many meshes) are a scene-cook follow-up (material id per cluster + a material table). The material
// logic below mirrors `gbuffer.slang`'s `fsMain` byte-for-byte so a resolved surface shades
// identically to rasterizing the same triangles directly.
#include "bindless.slang"

static const uint NO_TEXTURE = 0xFFFFFFFF;

struct PushConstants
{
    float4x4 mvp;               //   0: view_proj * model (project verts to screen for barycentric)
    float4x4 model;             //  64: object → world (world position + normal)
    uint vtx_buf;               // 128: vertex pool (32 B/vertex: pos[3], normal[3], uv[2])
    uint remap_buf;             // 132: cluster_vertices (u32)
    uint tri_buf;               // 136: cluster_triangles (u32, 3/tri)
    uint rec_buf;               // 140: GpuCluster records (96 B)
    uint vis_buf;               // 144: R64 visibility buffer (u64/pixel)
    uint width;                 // 148
    uint height;                // 152
    uint pad0;                  // 156
    float4 base_color_factor;   // 160
    float4 mr_factor;           // 176: x metallic, y roughness, z mip bias, w alpha cutoff
    uint4 tex;                   // 192: x base_color, y metallic_roughness, z normal, w emissive
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

struct GBufferOut
{
    float4 albedo : SV_Target0;   // rgb base color (linear), a = AO
    float4 normal : SV_Target1;   // xyz world-space normal
    float4 material : SV_Target2; // r metallic, g roughness, b AO
    float4 position : SV_Target3; // xyz world position
    float depth : SV_Depth;       // window-space depth (matches the raster the resolve reads)
};

struct VSOut
{
    float4 pos : SV_Position;
};

// Full-screen triangle from the vertex id — no vertex buffer.
[shader("vertex")]
VSOut vsMain(uint vid: SV_VertexID)
{
    VSOut o;
    float2 uv = float2((vid << 1) & 2, vid & 2);
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

float edge(float2 a, float2 b, float2 p)
{
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// Screen-space TBN from world-position + uv derivatives, identical to `gbuffer.slang`.
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
GBufferOut fsGBuffer(VSOut i)
{
    uint x = (uint)i.pos.x;
    uint y = (uint)i.pos.y;
    GBufferOut o;
    if (x >= pc.width || y >= pc.height)
        discard;
    uint px = y * pc.width + x;
    uint64_t packed = g.storage_buffers[pc.vis_buf].Load<uint64_t>(px * 8);
    if (packed == 0)
        discard; // no geometry → leave the cleared G-buffer (position.a == 0)

    uint payload = (uint)(packed & 0xFFFFFFFFu);
    uint cluster = payload >> 7;
    uint tri = payload & 0x7Fu;
    uint depth_key = (uint)(packed >> 32);

    uint4 rec = g.storage_buffers[pc.rec_buf].Load4(cluster * 96);
    uint voff = rec.x, toff = rec.z;

    uint3 li = g.storage_buffers[pc.tri_buf].Load3((toff + tri * 3) * 4);
    float2 sp[3];
    float invw[3];
    float3 nrm[3];
    float2 uvs[3];
    float3 wpos[3];
    for (int k = 0; k < 3; ++k)
    {
        uint remap = g.storage_buffers[pc.remap_buf].Load((voff + li[k]) * 4);
        uint b = remap * 32;
        float3 p = asfloat(g.storage_buffers[pc.vtx_buf].Load3(b));
        float3 nl = asfloat(g.storage_buffers[pc.vtx_buf].Load3(b + 12));
        uvs[k] = asfloat(g.storage_buffers[pc.vtx_buf].Load2(b + 24));
        wpos[k] = mul(pc.model, float4(p, 1.0)).xyz;
        nrm[k] = mul((float3x3)pc.model, nl);
        float4 clip = mul(pc.mvp, float4(p, 1.0));
        float w = max(clip.w, 1e-6);
        invw[k] = 1.0 / w;
        float3 ndc = clip.xyz / w;
        sp[k] = float2((ndc.x * 0.5 + 0.5) * pc.width, (0.5 - ndc.y * 0.5) * pc.height);
    }

    float2 p = float2(x + 0.5, y + 0.5);
    float area = edge(sp[0], sp[1], sp[2]);
    float inv = (abs(area) < 1e-12) ? 0.0 : 1.0 / area;
    float b0 = edge(sp[1], sp[2], p) * inv;
    float b1 = edge(sp[2], sp[0], p) * inv;
    float b2 = edge(sp[0], sp[1], p) * inv;
    // Perspective-correct weights (÷ clip w, renormalize) — matches HW fixed-function interpolation.
    float g0 = b0 * invw[0], g1 = b1 * invw[1], g2 = b2 * invw[2];
    float gs = g0 + g1 + g2;
    float rcp = (abs(gs) < 1e-12) ? 0.0 : 1.0 / gs;
    g0 *= rcp;
    g1 *= rcp;
    g2 *= rcp;

    float3 world_pos = g0 * wpos[0] + g1 * wpos[1] + g2 * wpos[2];
    float3 world_normal = g0 * nrm[0] + g1 * nrm[1] + g2 * nrm[2];
    float2 uv = g0 * uvs[0] + g1 * uvs[1] + g2 * uvs[2];

    // ── Material resolve — mirrors gbuffer.slang fsMain ──────────────────────────────────────────
    float mip_bias = pc.mr_factor.z;
    float cutoff = pc.mr_factor.w;
    float3 albedo = pc.base_color_factor.rgb;
    if (pc.tex.x != NO_TEXTURE)
    {
        float4 bc = g.textures[pc.tex.x].SampleBias(g.samp_wrap, uv, mip_bias);
        albedo *= bc.rgb;
        float a = bc.a * pc.base_color_factor.a;
        if (cutoff > 0.0 && a < cutoff)
            discard;
    }
    float metallic = pc.mr_factor.x;
    float roughness = pc.mr_factor.y;
    if (pc.tex.y != NO_TEXTURE)
    {
        float3 mr = g.textures[pc.tex.y].SampleBias(g.samp_wrap, uv, mip_bias).rgb;
        roughness *= mr.g;
        metallic *= mr.b;
    }
    float3 n = normalize(world_normal);
    if (pc.tex.z != NO_TEXTURE)
    {
        float3 tn = g.textures[pc.tex.z].SampleBias(g.samp_wrap, uv, mip_bias).rgb * 2.0 - 1.0;
        n = apply_normal_map(n, world_pos, uv, tn);
    }

    o.albedo = float4(albedo, 1.0);
    o.normal = float4(n, 0.0);
    o.material = float4(metallic, saturate(roughness), 1.0, 0.0);
    o.position = float4(world_pos, 1.0);
    // Reconstruct window-space depth from the visibility buffer's packed depth key (nearest→max).
    o.depth = 1.0 - float(depth_key) / 4294967040.0;
    return o;
}
