// Phase 14 unified per-frame vgeo pass — shared helpers for the SCENE path shaders
// (`vgeo_scene_cut` / `vgeo_scene_raster` / `vgeo_scene_hwvis` / `vgeo_gbuffer`).
//
// The scene path renders EVERY vgeo object in one cut/clear/raster/resolve set (not per object,
// see docs/phase-14-vgeo-unified-pass.md). The Nanite-style indirection (UE `NaniteRasterBinning`):
// a per-pixel payload stores a WORK INDEX `t` (not a raw cluster); `work[t] = (instance, cluster)`
// ties the shared cluster geometry to a per-instance transform + material fetched from the instance
// table. This mirrors Nanite's `VisibleCluster.InstanceId → GetInstanceSceneData` — one raster pass
// covers all instances, each projected by its own transform read from a scene buffer.
#include "bindless.slang"

// ── Per-instance record in the instance table (256 B, 16-B aligned). Matches the byte layout the
// app writes in `vgeo.rs` (`VgeoSystem::record_scene`). The three matrices are CPU-computed exactly
// as the old per-object push path did, so there is no in-shader matrix-multiply layout risk:
//   model   = object → world              (world pos/normal; cut bounds/cone to world)
//   mvp     = cull_view_proj (flip-FREE) * model   (SW raster + resolve barycentric + hwvis cull)
//   mvp_hw  = view_proj (Y-flipped)      * model    (hwvis SV_Position — HW rasterizer)
struct SceneInstance
{
    float4x4 model;      //   0
    float4x4 mvp;        //  64
    float4x4 mvp_hw;     // 128
    float4 base_color;   // 192
    float4 mr;           // 208: x metallic, y roughness, z cull_backface (0/1), w alpha_cutoff
    uint4 tex;           // 224: base_color / metallic-rough / normal / emissive bindless indices
    float max_scale;     // 240: max column scale of `model` (conservative sphere-radius scale)
    float3 pad;          // 244
};

// Load a column-major (glam `to_cols_array`) float4x4 from a byte-address buffer into a Slang
// float4x4 such that `mul(M, v)` reproduces `glam_mat * v` — identical to how a push-constant
// float4x4 reads the same bytes. Load4 fetches columns c0..c3; `transpose(float4x4(c0,c1,c2,c3))`
// (which builds them as rows) yields M with `M[i][j] = c_j[i]`, i.e. `mul(M,v)_i = Σ_j c_j[i] v[j]`
// = the glam column-major product. (The app stores matrices column-major, matching the push path.)
float4x4 loadMat4(RWByteAddressBuffer b, uint off)
{
    float4 c0 = asfloat(b.Load4(off));
    float4 c1 = asfloat(b.Load4(off + 16));
    float4 c2 = asfloat(b.Load4(off + 32));
    float4 c3 = asfloat(b.Load4(off + 48));
    return transpose(float4x4(c0, c1, c2, c3));
}

// Fetch instance `i`'s three matrices + scalars from the instance table. Matrices via `loadMat4`;
// float4/uint4 via direct Load4.
struct InstanceData
{
    float4x4 model;
    float4x4 mvp;
    float4x4 mvp_hw;
    float4 base_color;
    float4 mr;
    uint4 tex;
    float max_scale;
};
InstanceData loadInstance(uint inst_buf, uint i)
{
    RWByteAddressBuffer ib = g.storage_buffers[inst_buf];
    uint base = i * 256;
    InstanceData d;
    d.model = loadMat4(ib, base);
    d.mvp = loadMat4(ib, base + 64);
    d.mvp_hw = loadMat4(ib, base + 128);
    d.base_color = asfloat(ib.Load4(base + 192));
    d.mr = asfloat(ib.Load4(base + 208));
    d.tex = ib.Load4(base + 224);
    d.max_scale = asfloat(ib.Load(base + 240));
    return d;
}

// A work-list entry: (instance index, GLOBAL cluster index into the shared record buffer).
uint2 loadWork(uint work_buf, uint t)
{
    return g.storage_buffers[work_buf].Load2(t * 8);
}

// ── Cross-backend 64-bit atomicMax (the M0 seam, shared with the viewer's vgeo_swraster/hwvis) ──
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

// 2D edge function (twice the signed area of triangle a,b,p).
float sceneEdge(float2 a, float2 b, float2 p)
{
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}
