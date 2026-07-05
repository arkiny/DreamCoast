// Phase 14 M6 — material resolve: visibility buffer → attributes → shaded G-buffer surface.
//
// The SW rasterizer (M5a) and HW mesh path (M5b) leave only `(clusterId, triId)` + depth in the
// R64 visibility buffer — no shading. This full-screen pass resolves each covered pixel: unpack the
// cluster + triangle, fetch its three vertices, reconstruct the pixel's **perspective-correct
// barycentric** weights analytically (project the verts with the same MVP, solve edge functions,
// then perspective-divide by clip w), and interpolate normal / uv / world position. The result is
// the deferred G-buffer surface — here shaded with the SAME lambert term as the M2 direct cluster
// render (`vgeo_cluster.slang`) so the visibility→resolve path is provably identical to drawing the
// clusters directly. The full Phase-6 G-buffer MRT + deferred PBR wiring is the renderer-integration
// follow-up (same bucket as M4b HZB); this proves the resolve math in the self-contained viewer.
#include "bindless.slang"

struct PushConstants
{
    float4x4 mvp;   // 64
    uint vtx_buf;   // vertex pool (32 B/vertex: pos[3], normal[3], uv[2])
    uint remap_buf; // cluster_vertices (u32)
    uint tri_buf;   // cluster_triangles (u32, 3/tri)
    uint rec_buf;   // GpuCluster records (96 B)
    uint vis_buf;   // R64 visibility buffer (u64/pixel)
    uint width;
    uint height;
    uint mode;      // 0 = per-cluster colour, 1 = base-colour texture
    uint tex_index; // base-colour bindless slot (material mode)
    uint3 pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

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

// Matches `vgeo_cluster.slang` / `vgeo_visbuffer.slang` so the resolve reads the same as M2/M5.
float3 cluster_hue(uint id)
{
    float h = frac(float(id) * 0.61803398875);
    float3 k = frac(h + float3(0.0, 2.0 / 3.0, 1.0 / 3.0));
    float3 c = saturate(abs(k * 6.0 - 3.0) - 1.0);
    return lerp(float3(1.0, 1.0, 1.0), c, 0.72);
}

[shader("fragment")]
float4 fragMain(VSOut i): SV_Target
{
    uint x = (uint)i.pos.x;
    uint y = (uint)i.pos.y;
    float3 bg = float3(0.08, 0.09, 0.12);
    if (x >= pc.width || y >= pc.height)
        return float4(bg, 1.0);
    uint px = y * pc.width + x;
    uint64_t packed = g.storage_buffers[pc.vis_buf].Load<uint64_t>(px * 8);
    if (packed == 0)
        return float4(bg, 1.0); // background (no triangle)

    uint payload = (uint)(packed & 0xFFFFFFFFu);
    uint cluster = payload >> 7;
    uint tri = payload & 0x7Fu;

    // Cluster record: first 16 B = {vertex_offset, vertex_count, tri_offset, tri_count}.
    uint4 rec = g.storage_buffers[pc.rec_buf].Load4(cluster * 96);
    uint voff = rec.x, toff = rec.z;

    // Fetch the triangle's three vertices; keep clip HOMOGENEOUS (a near-crossing vertex has w<=0
    // and would project to garbage under a per-vertex xy/w) + attributes.
    uint3 li = g.storage_buffers[pc.tri_buf].Load3((toff + tri * 3) * 4);
    float4 clip[3];
    float3 nrm[3];
    float2 uvs[3];
    for (int k = 0; k < 3; ++k)
    {
        uint remap = g.storage_buffers[pc.remap_buf].Load((voff + li[k]) * 4);
        uint b = remap * 32;
        float3 pw = asfloat(g.storage_buffers[pc.vtx_buf].Load3(b));
        nrm[k] = asfloat(g.storage_buffers[pc.vtx_buf].Load3(b + 12));
        uvs[k] = asfloat(g.storage_buffers[pc.vtx_buf].Load2(b + 24));
        clip[k] = mul(pc.mvp, float4(pw, 1.0));
    }

    // Perspective-correct barycentric via a HOMOGENEOUS (pre-divide) solve — robust to w<=0
    // (near-crossing) verts, exactly like the HW rasterizer / the production `vgeo_gbuffer` resolve
    // (Schied & Dachsbacher). Solve `lam` (sum 1) so Σ lam·clip projects to this pixel's NDC (u,v);
    // no division by a per-vertex w. Identical to the old screen-space weights for in-front verts.
    float u = 2.0 * (float(x) + 0.5) / float(pc.width) - 1.0;
    float v = 1.0 - 2.0 * (float(y) + 0.5) / float(pc.height);
    float3 W = float3(clip[0].w, clip[1].w, clip[2].w);
    float3 A = float3(clip[0].x, clip[1].x, clip[2].x) - u * W;
    float3 B = float3(clip[0].y, clip[1].y, clip[2].y) - v * W;
    float3 L = cross(A, B);
    float s = L.x + L.y + L.z;
    float rs = (abs(s) < 1e-20) ? 0.0 : 1.0 / s;
    float3 lam = L * rs;

    float3 n = normalize(lam.x * nrm[0] + lam.y * nrm[1] + lam.z * nrm[2]);
    float2 uv = lam.x * uvs[0] + lam.y * uvs[1] + lam.z * uvs[2];

    // Shade identically to the M2 direct cluster render so the resolve is visually verifiable.
    float ndl = saturate(dot(n, normalize(float3(0.4, 0.8, 0.5)))) * 0.75 + 0.25;
    float3 base = (pc.mode == 1) ? g.textures[pc.tex_index].Sample(g.samp, uv).rgb : cluster_hue(cluster);
    return float4(base * ndl, 1.0);
}
