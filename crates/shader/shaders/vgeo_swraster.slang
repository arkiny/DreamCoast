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
    uint cull_backface; // 1 = backface-cull (single-sided material); 0 = two-sided (no cull)
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

// Clip-space (homogeneous) → screen (pixel xy + NDC z). Valid only for `clip.z >= 0` (in front of
// the standard-Z near plane, so `clip.w > 0`); the caller guarantees that (fast path or post-clip).
float3 project(float4 clip)
{
    float3 ndc = clip.xyz / clip.w;
    return float3((ndc.x * 0.5 + 0.5) * pc.width, (0.5 - ndc.y * 0.5) * pc.height, ndc.z);
}

// Rasterize ONE screen-space triangle (verts = pixel xy + NDC z) into the R64 visibility buffer.
// Shared by the fast path and each near-clipped sub-triangle (see `csRaster`).
void rasterTri(float3 s0, float3 s1, float3 s2, uint cull_backface, uint payload)
{
    float2 a = s0.xy, b = s1.xy, c = s2.xy;
    float area = edge(a, b, c);
    if (abs(area) < 1e-9)
        return;
    if (cull_backface != 0 && area > 0.0)
        return;
    float inv = 1.0 / area;
    int minx = max(0, (int)floor(min(a.x, min(b.x, c.x))));
    int maxx = min((int)pc.width - 1, (int)ceil(max(a.x, max(b.x, c.x))));
    int miny = max(0, (int)floor(min(a.y, min(b.y, c.y))));
    int maxy = min((int)pc.height - 1, (int)ceil(max(a.y, max(b.y, c.y))));
    for (int y = miny; y <= maxy; ++y)
    {
        for (int x = minx; x <= maxx; ++x)
        {
            float2 p = float2(x + 0.5, y + 0.5);
            float w0 = edge(b, c, p) * inv;
            float w1 = edge(c, a, p) * inv;
            float w2 = edge(a, b, p) * inv;
            if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0)
                continue;
            float d = saturate(w0 * s0.z + w1 * s1.z + w2 * s2.z);
            uint depth_key = 0xFFFFFFFFu - asuint(d);
            uint64_t packed = ((uint64_t)depth_key << 32) | (uint64_t)payload;
            atomicMaxU64(g.storage_buffers[pc.vis_buf], (uint)y * pc.width + (uint)x, packed);
        }
    }
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

    uint3 li = g.storage_buffers[pc.tri_buf].Load3((toff + lane * 3) * 4);
    uint payload = (cluster << 7) | lane; // clusterId<<7 | triId (triId < 128)

    // Fetch the 3 clip-space verts (HOMOGENEOUS). `pc.mvp` is the FLIP-FREE `cull_view_proj`, so the
    // screen winding sign is DX≡VK-consistent (front = `area > 0`, screen y-down).
    float4 clip[3];
    [unroll]
    for (int k = 0; k < 3; ++k)
    {
        uint remap = g.storage_buffers[pc.remap_buf].Load((voff + li[k]) * 4);
        float3 p = asfloat(g.storage_buffers[pc.vtx_buf].Load3(remap * 32));
        clip[k] = mul(pc.mvp, float4(p, 1.0));
    }

    // Near-plane coverage (standard-Z: in front ⇔ clip.z >= 0). Fast path = fully in front. Fully
    // behind = drop. Otherwise Sutherland-Hodgman clip against z>=0 into 1–2 sub-triangles carrying
    // the SAME payload — the resolve reconstructs attributes from the original triangle via its
    // w-robust homogeneous barycentric (mirrors the production vgeo_scene_raster near-clip).
    bool in0 = clip[0].z >= 0.0, in1 = clip[1].z >= 0.0, in2 = clip[2].z >= 0.0;
    if (in0 && in1 && in2)
    {
        rasterTri(project(clip[0]), project(clip[1]), project(clip[2]), pc.cull_backface, payload);
        return;
    }
    if (!in0 && !in1 && !in2)
        return;

    float3 poly[4];
    uint n = 0;
    [unroll]
    for (int e = 0; e < 3; ++e)
    {
        float4 va = clip[e];
        float4 vb = clip[(e + 1) % 3];
        bool ain = va.z >= 0.0;
        bool bin = vb.z >= 0.0;
        if (ain)
            poly[n++] = project(va);
        if (ain != bin)
        {
            float tpar = va.z / (va.z - vb.z); // z = 0 crossing; opposite signs ⇒ denom != 0
            poly[n++] = project(lerp(va, vb, tpar));
        }
    }
    if (n >= 3)
        rasterTri(poly[0], poly[1], poly[2], pc.cull_backface, payload);
    if (n == 4)
        rasterTri(poly[0], poly[2], poly[3], pc.cull_backface, payload);
}
