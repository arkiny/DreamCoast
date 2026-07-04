// Phase 14 unified pass — SCENE software rasterizer into the R64 visibility buffer. One dispatch of
// W threadgroups (one per work-list slot) rasters the whole scene's SW cut, versus one dispatch per
// object. Each group reads the visible list slot `t = vis_list[gid]` (a WORK INDEX), recovers
// `work[t] = (instance, cluster)`, projects with that instance's flip-free `mvp`, and records
// `payload = (t << 7) | triId` (the work index, so the resolve can recover the instance). Matches
// the viewer's `vgeo_swraster.slang` bit-for-bit in rule; only the per-instance sourcing differs.
#include "vgeo_scene_common.slang"

struct PushConstants
{
    uint vtx_buf;    //  0: vertex pool (32 B/vertex)
    uint remap_buf;  //  4: cluster_vertices (u32)
    uint tri_buf;    //  8: cluster_triangles (u32, 3/tri)
    uint rec_buf;    // 12: GpuCluster records (96 B)
    uint vis_buf;    // 16: R64 visibility buffer (u64/pixel)
    uint vis_list;   // 20: visible list (work indices, from the cut)
    uint work_buf;   // 24: work list (uint2: instance, global cluster)
    uint inst_buf;   // 28: instance table (SceneInstance, 256 B)
    uint width;      // 32
    uint height;     // 36
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

// Clear the visibility buffer to 0 (empty). One thread per pixel.
[shader("compute")]
[numthreads(64, 1, 1)]
void csClearScene(uint3 tid: SV_DispatchThreadID)
{
    uint px = tid.x;
    if (px >= pc.width * pc.height)
        return;
    g.storage_buffers[pc.vis_buf].Store<uint64_t>(px * 8, 0);
}

// Clip-space (homogeneous) → screen: pixel xy + NDC z. Valid only for `clip.z >= 0` (in front of
// the standard-Z near plane), i.e. `clip.w > 0`; the caller guarantees that (fast path or post-clip).
float3 sceneProject(float4 clip)
{
    float3 ndc = clip.xyz / clip.w;
    return float3((ndc.x * 0.5 + 0.5) * pc.width, (0.5 - ndc.y * 0.5) * pc.height, ndc.z);
}

// Rasterize ONE screen-space triangle (verts = pixel xy + NDC z) into the R64 visibility buffer.
// Shared by the fast path and each near-clipped sub-triangle. Backface cull for single-sided
// materials uses the flip-free screen winding (front = `area > 0`, screen y-down; see
// vgeo_swraster.slang). Depth is the barycentric-interpolated NDC z; the atomicMax keeps the nearest.
void rasterSceneTri(float3 s0, float3 s1, float3 s2, uint cull_backface, uint payload)
{
    float2 a = s0.xy, b = s1.xy, c = s2.xy;
    float area = sceneEdge(a, b, c);
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
            float w0 = sceneEdge(b, c, p) * inv;
            float w1 = sceneEdge(c, a, p) * inv;
            float w2 = sceneEdge(a, b, p) * inv;
            if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0)
                continue;
            float d = saturate(w0 * s0.z + w1 * s1.z + w2 * s2.z);
            uint depth_key = 0xFFFFFFFFu - asuint(d);
            uint64_t packed = ((uint64_t)depth_key << 32) | (uint64_t)payload;
            atomicMaxU64(g.storage_buffers[pc.vis_buf], (uint)y * pc.width + (uint)x, packed);
        }
    }
}

// One threadgroup per visible-list slot; one thread per triangle rasterizes it.
[shader("compute")]
[numthreads(128, 1, 1)]
void csRasterScene(uint gid: SV_GroupID, uint lane: SV_GroupThreadID)
{
    uint t = g.storage_buffers[pc.vis_list].Load(gid * 4);
    // Sentinel (0xFFFFFFFF) = an unused visible-list slot (the fixed dispatch covers W groups; the
    // cut fills only the selected ones). Skipping it makes the fixed dispatch equal to the indirect.
    if (t == 0xFFFFFFFFu)
        return;
    uint2 w = loadWork(pc.work_buf, t);
    InstanceData inst = loadInstance(pc.inst_buf, w.x);
    uint cull_backface = (inst.mr.z != 0.0) ? 1u : 0u;

    uint4 rec = g.storage_buffers[pc.rec_buf].Load4(w.y * 96);
    uint voff = rec.x, toff = rec.z, tcount = rec.w;
    if (lane >= tcount)
        return;

    uint3 li = g.storage_buffers[pc.tri_buf].Load3((toff + lane * 3) * 4);
    uint payload = (t << 7) | lane; // workIndex<<7 | triId (triId < 128)

    // Fetch the 3 clip-space verts (HOMOGENEOUS — do not divide by w yet; a near-crossing vertex
    // has w <= 0 and would project to garbage).
    float4 clip[3];
    [unroll]
    for (int k = 0; k < 3; ++k)
    {
        uint remap = g.storage_buffers[pc.remap_buf].Load((voff + li[k]) * 4);
        float3 p = asfloat(g.storage_buffers[pc.vtx_buf].Load3(remap * 32));
        clip[k] = mul(inst.mvp, float4(p, 1.0));
    }

    // Near-plane coverage (standard-Z: in front ⇔ clip.z >= 0). Fast path = fully in front, byte-
    // identical to the pre-clip rasterizer (same projection + edge functions). Fully behind = drop.
    bool in0 = clip[0].z >= 0.0, in1 = clip[1].z >= 0.0, in2 = clip[2].z >= 0.0;
    if (in0 && in1 && in2)
    {
        rasterSceneTri(sceneProject(clip[0]), sceneProject(clip[1]), sceneProject(clip[2]),
                       cull_backface, payload);
        return;
    }
    if (!in0 && !in1 && !in2)
        return;

    // Near-clip: Sutherland-Hodgman against the single near plane (z >= 0), interpolating in clip
    // space, then project the 3–4 output verts. Fan-triangulate into 1–2 sub-triangles carrying the
    // SAME triId — the resolve reconstructs attributes from the ORIGINAL triangle via its w-robust
    // homogeneous barycentric, so the split only bounds coverage + depth (docs/phase-14-vgeo-nearclip).
    // (Mirrors what the HW mesh rasterizer does for free; UE Nanite routes such clusters to HW.)
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
            poly[n++] = sceneProject(va);
        if (ain != bin)
        {
            float tpar = va.z / (va.z - vb.z); // z = 0 crossing; opposite signs ⇒ denom != 0
            poly[n++] = sceneProject(lerp(va, vb, tpar));
        }
    }
    // Clipping a triangle by one plane yields a 3- or 4-gon (winding preserved).
    if (n >= 3)
        rasterSceneTri(poly[0], poly[1], poly[2], cull_backface, payload);
    if (n == 4)
        rasterSceneTri(poly[0], poly[2], poly[3], cull_backface, payload);
}
