// Phase 14 M2 — resident cluster render via a mesh shader (HW path, no culling yet).
//
// One mesh threadgroup per cluster (`SV_GroupID` = cluster within the drawn LOD): it reads the
// cluster's record + geometry from bindless storage buffers and emits its ~128 triangles. This
// is the GPU counterpart of the CPU meshlet debug view — same per-cluster colouring — and the
// first consumer of the M0 mesh-shader pipeline + M2a bindless mesh-stage binding.
//
// Buffers (all in the bindless `g.storage_buffers[]` table, indexed by push constants):
//  - vtx_buf   : the vertex pool, 32-byte `MeshVertex` {pos[3], normal[3], uv[2]} per vertex.
//  - remap_buf : `cluster_vertices` — u32 indices into the vertex pool.
//  - tri_buf   : `cluster_triangles` expanded to u32 (3 local indices per triangle).
//  - rec_buf   : per-cluster records, `uint4 {vertex_offset, vertex_count, tri_offset, tri_count}`
//                (the first four fields of the cooked `Cluster`).
//
// Digit-less interpolant semantics (`NORMAL`/`TEXCOORD`/`COLOR`): Slang's Metal mesh-stage codegen
// keeps a trailing semantic digit while the fragment stage strips it, so a digit would fail
// mesh↔fragment linking (the M0 lesson).
#include "bindless.slang"

// Cluster caps (mirror `crates/asset/src/vgeo.rs`). The threadgroup is sized to cover the larger
// of the two so one lane handles vertex[lane] and triangle[lane].
static const uint MAX_VERTS = 64;
static const uint MAX_TRIS = 124;

struct PushConstants
{
    float4x4 mvp;      // 64 B
    uint vtx_buf;      // vertex pool slot
    uint remap_buf;    // cluster_vertices slot
    uint tri_buf;      // cluster_triangles (u32) slot
    uint rec_buf;      // GpuCluster records slot (64 B/record, ALL clusters)
    uint base_cluster; // direct path: global cluster index of SV_GroupID 0
    uint mode;         // 0 = per-cluster colour, 1 = base-colour texture
    uint tex_index;    // base-colour bindless slot (material mode)
    uint vis_buf;      // M3 cut: visible-cluster list slot (0xFFFFFFFF = direct path)
    uint4 pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

struct VertexOut
{
    float4 pos : SV_Position;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
    float cluster : COLOR; // global cluster id (for stable debug colours)
};

// Integer id → a well-separated hue (golden-ratio spacing), lightly desaturated. Matches the
// CPU `hue_palette` intent so the GPU and CPU meshlet views read the same.
float3 cluster_hue(uint id)
{
    float h = frac(float(id) * 0.61803398875);
    float3 k = frac(h + float3(0.0, 2.0 / 3.0, 1.0 / 3.0));
    float3 c = saturate(abs(k * 6.0 - 3.0) - 1.0);
    return lerp(float3(1.0, 1.0, 1.0), c, 0.72);
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("triangle")]
void meshMain(
    uint gid: SV_GroupID,
    uint lane: SV_GroupThreadID,
    out indices uint3 tris[MAX_TRIS],
    out vertices VertexOut verts[MAX_VERTS])
{
    // Direct path (M2): cluster = base + threadgroup. Cut path (M3): the LOD-cut compute wrote
    // the selected cluster indices into `vis_buf`, so look ours up by threadgroup.
    uint cluster = (pc.vis_buf == 0xFFFFFFFFu)
        ? (pc.base_cluster + gid)
        : g.storage_buffers[pc.vis_buf].Load(gid * 4);
    // GpuCluster record (96 B): first 16 bytes are {vertex_offset, vertex_count, tri_offset,
    // tri_count}.
    uint4 rec = g.storage_buffers[pc.rec_buf].Load4(cluster * 96);
    uint voff = rec.x;
    uint vcount = rec.y;
    uint toff = rec.z;
    uint tcount = rec.w;
    SetMeshOutputCounts(vcount, tcount);

    if (lane < vcount)
    {
        uint remap = g.storage_buffers[pc.remap_buf].Load((voff + lane) * 4);
        uint b = remap * 32;
        float3 p = asfloat(g.storage_buffers[pc.vtx_buf].Load3(b));
        float3 n = asfloat(g.storage_buffers[pc.vtx_buf].Load3(b + 12));
        float2 uv = asfloat(g.storage_buffers[pc.vtx_buf].Load2(b + 24));
        VertexOut v;
        v.pos = mul(pc.mvp, float4(p, 1.0));
        v.normal = n;
        v.uv = uv;
        v.cluster = float(cluster);
        verts[lane] = v;
    }
    if (lane < tcount)
    {
        tris[lane] = g.storage_buffers[pc.tri_buf].Load3((toff + lane * 3) * 4);
    }
}

[shader("fragment")]
float4 fragMain(VertexOut i): SV_Target
{
    float3 n = normalize(i.normal);
    float ndl = saturate(dot(n, normalize(float3(0.4, 0.8, 0.5)))) * 0.75 + 0.25;
    float3 base;
    if (pc.mode == 1)
    {
        base = g.textures[pc.tex_index].Sample(g.samp, i.uv).rgb;
    }
    else
    {
        base = cluster_hue((uint)(i.cluster + 0.5));
    }
    return float4(base * ndl, 1.0);
}
