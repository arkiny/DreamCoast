// Phase 14 M5 — software rasterizer into an R64 visibility buffer.
//
// Sub-pixel / small triangles are inefficient on the HW rasterizer, so a compute shader rasters
// them directly: one threadgroup per visible cluster, one thread per triangle. Each covered pixel
// records `packed = (depthKey << 32) | (clusterId << 7 | triId)` into the visibility buffer with a
// single 64-bit `atomicMax` — the highest depthKey (nearest, after inversion) wins per pixel with
// no lock. This is the first real consumer of the M0 64-bit-atomicMax pivot. A later resolve pass
// (M6) reads `(clusterId, triId)` back to shade; M5 visualizes cluster id to prove the raster.
#include "bindless.slang"

// ── Cross-backend 64-bit atomicMax (the M0 seam) ─────────────────────────────────────────────
// Slang's `InterlockedMax(uint64_t)` lowers to Metal's `atomic_fetch_max_explicit`, which has no
// 64-bit overload; Metal spells it `atomic_max_explicit`. Emit that directly under the Metal gate;
// SPIR-V/D3D12 use the portable `InterlockedMax64` (both → `OpAtomicUMax` / SM6.6 atomics).
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
    uint vis_buf;   // R64 visibility buffer (u64/pixel)
    uint vis_list;  // visible-cluster list (from the cut)
    uint width;     // render target size
    uint height;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

// 2D edge function (twice the signed area of triangle a,b,p).
float edge(float2 a, float2 b, float2 p)
{
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// Clear the visibility buffer to 0 (empty). One thread per pixel.
[shader("compute")]
[numthreads(64, 1, 1)]
void csClear(uint3 tid: SV_DispatchThreadID)
{
    uint px = tid.x;
    if (px >= pc.width * pc.height)
        return;
    g.storage_buffers[pc.vis_buf].Store<uint64_t>(px * 8, 0);
}

// One threadgroup per visible cluster (`SV_GroupID` indexes the cut's visible list); one thread
// per triangle rasterizes it into the visibility buffer.
[shader("compute")]
[numthreads(128, 1, 1)]
void csRaster(uint gid: SV_GroupID, uint lane: SV_GroupThreadID)
{
    uint cluster = g.storage_buffers[pc.vis_list].Load(gid * 4);
    // Sentinel (0xFFFFFFFF) = an unused visible-list slot. The graph integration dispatches a
    // FIXED group count (total clusters) rather than the cut's exact count (the render-graph IR
    // recorder has no indirect compute dispatch), pre-clearing the list to the sentinel; skipping
    // it makes the fixed dispatch equivalent to the indirect one. The viewer's indirect path only
    // dispatches the exact cut count, so every slot is valid there and this never triggers.
    if (cluster == 0xFFFFFFFFu)
        return;
    uint4 rec = g.storage_buffers[pc.rec_buf].Load4(cluster * 96);
    uint voff = rec.x, toff = rec.z, tcount = rec.w;
    if (lane >= tcount)
        return;

    // Fetch the triangle's three vertices, project to screen space (pixels) + NDC depth.
    uint3 li = g.storage_buffers[pc.tri_buf].Load3((toff + lane * 3) * 4);
    float3 sp[3];
    for (int k = 0; k < 3; ++k)
    {
        uint remap = g.storage_buffers[pc.remap_buf].Load((voff + li[k]) * 4);
        float3 p = asfloat(g.storage_buffers[pc.vtx_buf].Load3(remap * 32));
        float4 clip = mul(pc.mvp, float4(p, 1.0));
        if (clip.w <= 1e-6)
            return; // crosses behind the camera → skip (crude near clip)
        float3 ndc = clip.xyz / clip.w;
        sp[k] = float3((ndc.x * 0.5 + 0.5) * pc.width, (0.5 - ndc.y * 0.5) * pc.height, ndc.z);
    }

    float2 a = sp[0].xy, b = sp[1].xy, c = sp[2].xy;
    float area = edge(a, b, c);
    if (abs(area) < 1e-9)
        return;
    float inv = 1.0 / area;

    int minx = max(0, (int)floor(min(a.x, min(b.x, c.x))));
    int maxx = min((int)pc.width - 1, (int)ceil(max(a.x, max(b.x, c.x))));
    int miny = max(0, (int)floor(min(a.y, min(b.y, c.y))));
    int maxy = min((int)pc.height - 1, (int)ceil(max(a.y, max(b.y, c.y))));

    uint payload = (cluster << 7) | lane; // clusterId<<7 | triId (triId < 128)

    for (int y = miny; y <= maxy; ++y)
    {
        for (int x = minx; x <= maxx; ++x)
        {
            float2 p = float2(x + 0.5, y + 0.5);
            // Barycentric weights (vertex a↔edge bc, etc.), scaled by 1/area so signs match `area`.
            float w0 = edge(b, c, p) * inv;
            float w1 = edge(c, a, p) * inv;
            float w2 = edge(a, b, p) * inv;
            if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0)
                continue;
            float d = saturate(w0 * sp[0].z + w1 * sp[1].z + w2 * sp[2].z);
            // Nearest (small depth) → largest key, so a single atomicMax keeps the front-most.
            uint depth_key = (uint)((1.0 - d) * 4294967040.0);
            uint64_t packed = ((uint64_t)depth_key << 32) | (uint64_t)payload;
            atomicMaxU64(g.storage_buffers[pc.vis_buf], (uint)y * pc.width + (uint)x, packed);
        }
    }
}
