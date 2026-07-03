// Phase 14 M5b — HW (mesh-shader) path writing into the SAME R64 visibility buffer as the SW
// rasterizer. Binning (in the LOD cut) routes large-triangle clusters here and micro-triangle
// clusters to the compute SW rasterizer (`vgeo_swraster.slang`); both record the identical packed
// `(depthKey << 32) | (clusterId << 7 | triId)` with a single 64-bit `atomicMax`, so the HW/SW
// boundary has no seam — the same per-pixel nearest-wins rule decides every pixel.
//
// The mesh stage emits `triId` as a PER-PRIMITIVE output (Slang `out primitives`), so the fragment
// shader gets the exact triangle id without an interpolated approximation. The fragment writes only
// the visibility buffer (its colour output is unused — the M5 visualization / M6 resolve reads the
// buffer), matching the SW path bit-for-bit in intent.
#include "bindless.slang"

static const uint MAX_VERTS = 64;
static const uint MAX_TRIS = 124;

// ── Cross-backend 64-bit atomicMax (the M0 seam, shared with vgeo_swraster.slang) ────────────────
#ifdef RT_METAL_TARGET
void atomicMaxU64(RWByteAddressBuffer buf, uint elem, uint64_t val)
{
    __intrinsic_asm "atomic_max_explicit((device atomic_ulong*)$0 + $1, $2, memory_order_relaxed)";
}
#else
void atomicMaxU64(RWByteAddressBuffer buf, uint elem, uint64_t val)
{
    uint64_t prev;
    buf.InterlockedMax64(elem * 8, val, prev);
}
#endif

struct PushConstants
{
    float4x4 mvp;   // 64
    uint vtx_buf;   // vertex pool (32 B/vertex)
    uint remap_buf; // cluster_vertices (u32)
    uint tri_buf;   // cluster_triangles (u32, 3/tri)
    uint rec_buf;   // GpuCluster records (96 B)
    uint hw_list;   // HW visible-cluster list (from the binning cut)
    uint vis_buf;   // R64 visibility buffer (u64/pixel)
    uint width;
    uint height;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

struct VertexOut
{
    float4 pos : SV_Position;
};

// Per-primitive payload: clusterId<<7 | triId (triId < 128). Flat, no interpolation.
struct PrimOut
{
    uint payload : COLOR;
};

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("triangle")]
void meshMain(
    uint gid: SV_GroupID,
    uint lane: SV_GroupThreadID,
    out indices uint3 tris[MAX_TRIS],
    out primitives PrimOut prims[MAX_TRIS],
    out vertices VertexOut verts[MAX_VERTS])
{
    uint cluster = g.storage_buffers[pc.hw_list].Load(gid * 4);
    uint4 rec = g.storage_buffers[pc.rec_buf].Load4(cluster * 96);
    uint voff = rec.x, vcount = rec.y, toff = rec.z, tcount = rec.w;
    SetMeshOutputCounts(vcount, tcount);

    if (lane < vcount)
    {
        uint remap = g.storage_buffers[pc.remap_buf].Load((voff + lane) * 4);
        float3 p = asfloat(g.storage_buffers[pc.vtx_buf].Load3(remap * 32));
        VertexOut v;
        v.pos = mul(pc.mvp, float4(p, 1.0));
        verts[lane] = v;
    }
    if (lane < tcount)
    {
        tris[lane] = g.storage_buffers[pc.tri_buf].Load3((toff + lane * 3) * 4);
        PrimOut pr;
        pr.payload = (cluster << 7) | lane;
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
    // Window-space depth (perspective-correct z, [0,1]) — the same NDC depth the SW raster
    // interpolates, so nearest-wins is identical across HW/SW.
    float d = saturate(i.pos.z);
    uint depth_key = (uint)((1.0 - d) * 4294967040.0);
    uint64_t packed = ((uint64_t)depth_key << 32) | (uint64_t)prim.payload;
    atomicMaxU64(g.storage_buffers[pc.vis_buf], px, packed);
    return float4(0.0, 0.0, 0.0, 1.0);
}
