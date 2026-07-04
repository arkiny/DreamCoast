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
    float4x4 mvp;   // 0: Y-flipped view_proj*model for HW-raster SV_Position
    uint vtx_buf;   // vertex pool (32 B/vertex)
    uint remap_buf; // cluster_vertices (u32)
    uint tri_buf;   // cluster_triangles (u32, 3/tri)
    uint rec_buf;   // GpuCluster records (96 B)
    uint hw_list;   // HW visible-cluster list (from the binning cut)
    uint vis_buf;   // R64 visibility buffer (u64/pixel)
    uint width;
    uint height;         // 8 uints @64..96
    float4x4 cull_mvp;   // 96: FLIP-FREE cull_view_proj*model for the DX≡VK-consistent backface test
    uint cull_backface;  // 160: 1 = backface-cull (single-sided material); 0 = two-sided (no cull)
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
    // `indices` / `primitives` / `vertices` imply `out`; the explicit `out` triggers a Slang
    // 2026.10.2 → DXC 1.9 malformed-codegen bug (`out indices out uint3`). Dropping it warns
    // E54001 but emits valid DXIL; SPIR-V / Metal unaffected. (See vgeo_meshlet.slang.)
    indices uint3 tris[MAX_TRIS],
    primitives PrimOut prims[MAX_TRIS],
    vertices VertexOut verts[MAX_VERTS])
{
    uint cluster = g.storage_buffers[pc.hw_list].Load(gid * 4);
    uint4 rec = g.storage_buffers[pc.rec_buf].Load4(cluster * 96);
    // Clamp counts to the declared output maxima before `SetMeshOutputCounts` — required for
    // correctness and the fix for an NVIDIA VK_EXT_mesh_shader issue where an unbounded
    // storage-loaded count makes the mesh draw emit nothing (see vgeo_cluster.slang).
    uint voff = rec.x, vcount = min(rec.y, MAX_VERTS), toff = rec.z, tcount = min(rec.w, MAX_TRIS);
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
        uint3 tl = g.storage_buffers[pc.tri_buf].Load3((toff + lane * 3) * 4);
        // Per-triangle backface cull for single-sided materials (Nanite NaniteRasterizer.ush). The
        // winding uses the FLIP-FREE `cull_mvp` mapped to the SAME SCREEN-PIXEL space as the SW raster
        // (`vgeo_swraster.slang`), so HW and SW cull the SAME faces with the SAME predicate (`area >
        // 0` = backface). Screen-pixel units (×width/height, ~10⁶ scale) instead of NDC (~1) keep the
        // sign robust for near-edge-on triangles — the DXIL-vs-SPIR-V rounding at `area ≈ 0` no longer
        // flips a triangle's cull between backends (the tiny-NDC-area form drifted DX≡VK past 0.001).
        // A vertex crossing the near plane (w ≤ 0) makes the winding unreliable → keep the triangle.
        // Backface → DEGENERATE the triangle so the HW rasterizer emits no fragments.
        if (pc.cull_backface != 0)
        {
            uint3 rm = uint3(
                g.storage_buffers[pc.remap_buf].Load((voff + tl.x) * 4),
                g.storage_buffers[pc.remap_buf].Load((voff + tl.y) * 4),
                g.storage_buffers[pc.remap_buf].Load((voff + tl.z) * 4));
            float4 c0 = mul(pc.cull_mvp, float4(asfloat(g.storage_buffers[pc.vtx_buf].Load3(rm.x * 32)), 1.0));
            float4 c1 = mul(pc.cull_mvp, float4(asfloat(g.storage_buffers[pc.vtx_buf].Load3(rm.y * 32)), 1.0));
            float4 c2 = mul(pc.cull_mvp, float4(asfloat(g.storage_buffers[pc.vtx_buf].Load3(rm.z * 32)), 1.0));
            bool near_ok = c0.w > 1e-6 && c1.w > 1e-6 && c2.w > 1e-6;
            float w = float(pc.width), h = float(pc.height);
            float2 s0 = float2((c0.x / c0.w * 0.5 + 0.5) * w, (0.5 - c0.y / c0.w * 0.5) * h);
            float2 s1 = float2((c1.x / c1.w * 0.5 + 0.5) * w, (0.5 - c1.y / c1.w * 0.5) * h);
            float2 s2 = float2((c2.x / c2.w * 0.5 + 0.5) * w, (0.5 - c2.y / c2.w * 0.5) * h);
            float area = (s1.x - s0.x) * (s2.y - s0.y) - (s1.y - s0.y) * (s2.x - s0.x);
            if (near_ok && area > 0.0)
                tl = uint3(0, 0, 0); // backface → degenerate → no fragments
        }
        tris[lane] = tl;
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
    // interpolates, so nearest-wins is identical across HW/SW. Bit-reinterpret (monotonic for
    // d >= 0), NOT a float 2^32 scale, so the key keeps full float32 precision and matches the SW
    // path exactly — else near-coplanar surfaces tie and dither (see vgeo_swraster.slang).
    float d = saturate(i.pos.z);
    uint depth_key = 0xFFFFFFFFu - asuint(d);
    uint64_t packed = ((uint64_t)depth_key << 32) | (uint64_t)prim.payload;
    atomicMaxU64(g.storage_buffers[pc.vis_buf], px, packed);
    return float4(0.0, 0.0, 0.0, 1.0);
}
