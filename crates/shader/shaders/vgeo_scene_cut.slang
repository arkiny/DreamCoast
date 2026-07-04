// Phase 14 unified pass — SCENE LOD-DAG cut (compute). One dispatch over the whole scene's work
// list (W = Σ instance clusters), not one dispatch per object. Thread `t` reads `work[t] =
// (instance, global_cluster)`, fetches that instance's transform from the instance table, runs the
// identical view-dependent LOD + frustum + normal-cone selection as the viewer's `vgeo_cut.slang`
// (per-instance `model` in place of the per-object push `model`), and — on selection — appends the
// WORK INDEX `t` (not the cluster) to the visible list so the raster/resolve can recover the
// instance via `work[t]`. `csCutSceneBin` additionally splits by projected screen size into the
// HW (mesh-shader) and SW (compute-raster) sub-lists. See docs/phase-14-vgeo-unified-pass.md.
//
// The selection math mirrors `vgeo_cut.slang` (the self-contained viewer's per-object cut); it is
// duplicated here rather than shared because the two differ in how they source `model` (per-instance
// table vs per-object push) — the same reason `atomicMaxU64` is duplicated across the raster paths.
#include "vgeo_scene_common.slang"

struct PushConstants
{
    float4 planes[6];  //   0: WORLD frustum planes (xyz=inward normal, w=d), normalized (96 B)
    float cam_x;       //  96: WORLD camera position (scalars, no float3 alignment)
    float cam_y;
    float cam_z;
    float proj_factor; // 108: 0.5 * screen_height / tan(fov/2): world error → px at unit distance
    float tau;         // 112: pixel-error threshold (the LOD "tightness" knob)
    uint total;        // 116: W — total work-list length (Σ instance clusters, all LODs)
    uint rec_buf;      // 120: GpuCluster records slot (96 B/record, shared)
    uint vis_buf;      // 124: output list — `csCutScene`: the whole cut; `csCutSceneBin`: HW list
    uint args_buf;     // 128: output args {countX,1,1} — the cut / HW args
    uint sw_list;      // 132: `csCutSceneBin` SW-path visible list slot
    uint sw_args;      // 136: `csCutSceneBin` SW-path indirect args {countX,1,1}
    float bin_px;      // 140: SW/HW split threshold — projected bounds diameter (px) below → SW
    uint work_buf;     // 144: work list (uint2: instance, global cluster)
    uint inst_buf;     // 148: instance table (SceneInstance, 256 B)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

float3 cam_pos()
{
    return float3(pc.cam_x, pc.cam_y, pc.cam_z);
}

float3 to_world(float4x4 model, float3 p)
{
    return mul(model, float4(p, 1.0)).xyz;
}

// Screen-space (pixel) error of a WORLD error on a WORLD sphere (project by near distance).
float screen_error(float err_w, float3 center_w, float radius_w)
{
    float d = max(distance(cam_pos(), center_w) - radius_w, 1e-3);
    return err_w * pc.proj_factor / d;
}

// Frustum cull: a bounding sphere is outside iff fully behind any inward-facing plane.
bool outside_frustum(float3 center, float radius)
{
    [unroll]
    for (int p = 0; p < 6; ++p)
        if (dot(pc.planes[p].xyz, center) + pc.planes[p].w < -radius)
            return true;
    return false;
}

// Radius-aware normal-cone backface cull (meshoptimizer/niagara), mirroring `vgeo_cut.slang`.
bool cone_backfacing(float3 axis, float cutoff, float3 center, float radius)
{
    if (cutoff <= 0.0)
        return false;
    float3 cw = center - cam_pos();
    float dist = length(cw);
    if (dist <= radius)
        return false;
    float3 d = cw / dist;
    float sin_half = sqrt(max(1.0 - cutoff * cutoff, 0.0));
    float sr = saturate(radius / dist);
    float cr = sqrt(max(1.0 - sr * sr, 0.0));
    return dot(axis, d) > sin_half * cr + cutoff * sr;
}

// Cone cull is enabled for SINGLE-SIDED materials only (per-instance `cull_backface`), with the
// Morton spatial clusterizer's tight cones — identical policy to `vgeo_cut.slang`.
static const bool CONE_CULL_ENABLED = true;

// Shared cut selection: LOD cut + frustum + (gated) normal-cone, returning the cluster's WORLD
// bounds sphere so `csCutSceneBin` can estimate screen size. `gc` is the global cluster index.
bool select_cluster(uint gc, float4x4 model, float max_scale, uint cull_backface,
                    out float3 bnd_c, out float bnd_r)
{
    RWByteAddressBuffer rec = g.storage_buffers[pc.rec_buf];
    uint base = gc * 96;
    float self_err = asfloat(rec.Load(base + 16));
    float3 self_c = asfloat(rec.Load3(base + 20));
    float self_r = asfloat(rec.Load(base + 32));
    float par_err = asfloat(rec.Load(base + 36));
    float3 par_c = asfloat(rec.Load3(base + 40));
    float par_r = asfloat(rec.Load(base + 52));
    float3 cone_axis = asfloat(rec.Load3(base + 72));
    float cone_cutoff = asfloat(rec.Load(base + 84));

    float ms = max_scale;
    bnd_c = to_world(model, asfloat(rec.Load3(base + 56)));
    bnd_r = asfloat(rec.Load(base + 68)) * ms;

    float s_self = screen_error(self_err * ms, to_world(model, self_c), self_r * ms);
    float s_par =
        (par_err >= 3.0e38) ? 1.0e30 : screen_error(par_err * ms, to_world(model, par_c), par_r * ms);
    if (!(s_self <= pc.tau && s_par > pc.tau))
        return false;
    if (outside_frustum(bnd_c, bnd_r))
        return false;
    if (CONE_CULL_ENABLED && cull_backface != 0
        && cone_backfacing(normalize(mul((float3x3)model, cone_axis)), cone_cutoff, bnd_c, bnd_r))
        return false;
    return true;
}

// SW-only cut: append the work index `t` of every selected cluster to the single visible list.
[shader("compute")]
[numthreads(64, 1, 1)]
void csCutScene(uint3 tid: SV_DispatchThreadID)
{
    uint t = tid.x;
    if (t >= pc.total)
        return;
    uint2 w = loadWork(pc.work_buf, t);
    InstanceData inst = loadInstance(pc.inst_buf, w.x);
    uint cull_backface = (inst.mr.z != 0.0) ? 1u : 0u;
    float3 bnd_c;
    float bnd_r;
    if (!select_cluster(w.y, inst.model, inst.max_scale, cull_backface, bnd_c, bnd_r))
        return;
    uint slot;
    g.storage_buffers[pc.args_buf].InterlockedAdd(0, 1, slot);
    g.storage_buffers[pc.vis_buf].Store(slot * 4, t);
}

// HW/SW binning cut: classify each selected cluster by projected screen diameter into the SW
// (compute-raster) or HW (mesh-shader) visible list; both store the WORK INDEX `t`.
[shader("compute")]
[numthreads(64, 1, 1)]
void csCutSceneBin(uint3 tid: SV_DispatchThreadID)
{
    uint t = tid.x;
    if (t >= pc.total)
        return;
    uint2 w = loadWork(pc.work_buf, t);
    InstanceData inst = loadInstance(pc.inst_buf, w.x);
    uint cull_backface = (inst.mr.z != 0.0) ? 1u : 0u;
    float3 bnd_c;
    float bnd_r;
    if (!select_cluster(w.y, inst.model, inst.max_scale, cull_backface, bnd_c, bnd_r))
        return;

    float dist = max(distance(cam_pos(), bnd_c) - bnd_r, 1e-3);
    float screen_diam = 2.0 * bnd_r * pc.proj_factor / dist;

    uint list = (screen_diam < pc.bin_px) ? pc.sw_list : pc.vis_buf;
    uint args = (screen_diam < pc.bin_px) ? pc.sw_args : pc.args_buf;
    uint slot;
    g.storage_buffers[args].InterlockedAdd(0, 1, slot);
    g.storage_buffers[list].Store(slot * 4, t);
}
