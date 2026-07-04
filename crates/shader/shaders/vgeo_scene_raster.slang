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
    float3 sp[3];
    for (int k = 0; k < 3; ++k)
    {
        uint remap = g.storage_buffers[pc.remap_buf].Load((voff + li[k]) * 4);
        float3 p = asfloat(g.storage_buffers[pc.vtx_buf].Load3(remap * 32));
        float4 clip = mul(inst.mvp, float4(p, 1.0));
        if (clip.w <= 1e-6)
            return; // crosses behind the camera → skip (crude near clip)
        float3 ndc = clip.xyz / clip.w;
        sp[k] = float3((ndc.x * 0.5 + 0.5) * pc.width, (0.5 - ndc.y * 0.5) * pc.height, ndc.z);
    }

    float2 a = sp[0].xy, b = sp[1].xy, c = sp[2].xy;
    float area = sceneEdge(a, b, c);
    if (abs(area) < 1e-9)
        return;
    // Per-triangle backface cull for single-sided materials — flip-free `mvp` screen space, front is
    // `area > 0` (screen y-down); see vgeo_swraster.slang. Two-sided keeps both faces (CULL_NONE).
    if (cull_backface != 0 && area > 0.0)
        return;
    float inv = 1.0 / area;

    int minx = max(0, (int)floor(min(a.x, min(b.x, c.x))));
    int maxx = min((int)pc.width - 1, (int)ceil(max(a.x, max(b.x, c.x))));
    int miny = max(0, (int)floor(min(a.y, min(b.y, c.y))));
    int maxy = min((int)pc.height - 1, (int)ceil(max(a.y, max(b.y, c.y))));

    uint payload = (t << 7) | lane; // workIndex<<7 | triId (triId < 128)

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
            float d = saturate(w0 * sp[0].z + w1 * sp[1].z + w2 * sp[2].z);
            uint depth_key = 0xFFFFFFFFu - asuint(d);
            uint64_t packed = ((uint64_t)depth_key << 32) | (uint64_t)payload;
            atomicMaxU64(g.storage_buffers[pc.vis_buf], (uint)y * pc.width + (uint)x, packed);
        }
    }
}
