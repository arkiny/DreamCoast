// Phase 14 unified pass — SCENE HW (mesh-shader) path writing the SAME R64 visibility buffer as the
// scene SW rasterizer. One indirect mesh draw covers the whole scene's HW cut. Each threadgroup
// reads the HW visible list slot `t = hw_list[gid]` (a WORK INDEX), recovers `work[t] =
// (instance, cluster)`, projects with that instance's Y-flipped `mvp_hw` for `SV_Position`, and
// records `payload = (t << 7) | triId`. Backface cull uses the flip-free `mvp` (DX≡VK-consistent
// screen winding). Mirrors the viewer's `vgeo_hwvis.slang`; only per-instance sourcing differs.
#include "vgeo_scene_common.slang"

static const uint MAX_VERTS = 64;
static const uint MAX_TRIS = 124;

struct PushConstants
{
    uint vtx_buf;   //  0: vertex pool (32 B/vertex)
    uint remap_buf; //  4: cluster_vertices (u32)
    uint tri_buf;   //  8: cluster_triangles (u32, 3/tri)
    uint rec_buf;   // 12: GpuCluster records (96 B)
    uint hw_list;   // 16: HW visible list (work indices, from the binning cut)
    uint vis_buf;   // 20: R64 visibility buffer (u64/pixel)
    uint work_buf;  // 24: work list (uint2: instance, global cluster)
    uint inst_buf;  // 28: instance table (SceneInstance, 256 B)
    uint width;     // 32
    uint height;    // 36
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

struct VertexOut
{
    float4 pos : SV_Position;
};

struct PrimOut
{
    uint payload : COLOR; // workIndex<<7 | triId
};

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("triangle")]
void meshMain(
    uint3 gid: SV_GroupID,
    uint lane: SV_GroupThreadID,
    indices uint3 tris[MAX_TRIS],
    primitives PrimOut prims[MAX_TRIS],
    vertices VertexOut verts[MAX_VERTS])
{
    // 2-D grid flattening (see VGEO_HW_GRID_X): the indirect args fold the HW count into an
    // (x <= GRID_X, y, 1) grid because mesh dispatch caps at 65,535 threadgroups PER DIMENSION
    // — the old 1-D `(count, 1, 1)` grid dropped the whole draw past that (native-res sponza).
    // y == 1 reproduces the legacy `gid.x` indexing exactly. The last row can overshoot the
    // count by < GRID_X; those slots read the sentinel-cleared tail of hw_list (the Rust side
    // allocates GRID_X slots of slack) and emit nothing.
    uint flat = gid.y * VGEO_HW_GRID_X + gid.x;
    uint t = g.storage_buffers[pc.hw_list].Load(flat * 4);
    if (t == 0xFFFFFFFFu)
    {
        SetMeshOutputCounts(0, 0);
        return;
    }
    uint2 w = loadWork(pc.work_buf, t);
    InstanceData inst = loadInstance(pc.inst_buf, w.x);
    uint cull_backface = (inst.mr.z != 0.0) ? 1u : 0u;

    uint4 rec = g.storage_buffers[pc.rec_buf].Load4(w.y * 96);
    // Clamp counts to the declared output maxima before SetMeshOutputCounts (NVIDIA VK mesh-shader
    // needs a static bound or the draw emits nothing — see vgeo_hwvis.slang / vgeo_cluster.slang).
    uint voff = rec.x, vcount = min(rec.y, MAX_VERTS), toff = rec.z, tcount = min(rec.w, MAX_TRIS);
    SetMeshOutputCounts(vcount, tcount);

    if (lane < vcount)
    {
        uint remap = g.storage_buffers[pc.remap_buf].Load((voff + lane) * 4);
        float3 p = asfloat(g.storage_buffers[pc.vtx_buf].Load3(remap * 32));
        VertexOut v;
        v.pos = mul(inst.mvp_hw, float4(p, 1.0));
        verts[lane] = v;
    }
    if (lane < tcount)
    {
        uint3 tl = g.storage_buffers[pc.tri_buf].Load3((toff + lane * 3) * 4);
        // Per-triangle backface cull (single-sided): winding in the flip-free `mvp` SCREEN-PIXEL
        // space so HW matches the SW raster sign DX≡VK (see vgeo_hwvis.slang). Backface → degenerate.
        if (cull_backface != 0)
        {
            uint3 rm = uint3(
                g.storage_buffers[pc.remap_buf].Load((voff + tl.x) * 4),
                g.storage_buffers[pc.remap_buf].Load((voff + tl.y) * 4),
                g.storage_buffers[pc.remap_buf].Load((voff + tl.z) * 4));
            float4 c0 = mul(inst.mvp, float4(asfloat(g.storage_buffers[pc.vtx_buf].Load3(rm.x * 32)), 1.0));
            float4 c1 = mul(inst.mvp, float4(asfloat(g.storage_buffers[pc.vtx_buf].Load3(rm.y * 32)), 1.0));
            float4 c2 = mul(inst.mvp, float4(asfloat(g.storage_buffers[pc.vtx_buf].Load3(rm.z * 32)), 1.0));
            bool near_ok = c0.w > 1e-6 && c1.w > 1e-6 && c2.w > 1e-6;
            float wd = float(pc.width), ht = float(pc.height);
            float2 s0 = float2((c0.x / c0.w * 0.5 + 0.5) * wd, (0.5 - c0.y / c0.w * 0.5) * ht);
            float2 s1 = float2((c1.x / c1.w * 0.5 + 0.5) * wd, (0.5 - c1.y / c1.w * 0.5) * ht);
            float2 s2 = float2((c2.x / c2.w * 0.5 + 0.5) * wd, (0.5 - c2.y / c2.w * 0.5) * ht);
            float area = (s1.x - s0.x) * (s2.y - s0.y) - (s1.y - s0.y) * (s2.x - s0.x);
            if (near_ok && area > 0.0)
                tl = uint3(0, 0, 0);
        }
        tris[lane] = tl;
        PrimOut pr;
        pr.payload = (t << 7) | lane;
        prims[lane] = pr;
    }
}

[shader("fragment")]
float4 fragMain(VertexOut i, PrimOut prim): SV_Target
{
    uint x = (uint)i.pos.x;
    uint y = (uint)i.pos.y;
    if (x >= pc.width || y >= pc.height)
        return float4(0.0, 0.0, 0.0, 1.0);
    uint px = y * pc.width + x;
    float d = saturate(i.pos.z);
    uint depth_key = 0xFFFFFFFFu - asuint(d);
    uint64_t packed = ((uint64_t)depth_key << 32) | (uint64_t)prim.payload;
    atomicMaxU64(g.storage_buffers[pc.vis_buf], px, packed);
    return float4(0.0, 0.0, 0.0, 1.0);
}
