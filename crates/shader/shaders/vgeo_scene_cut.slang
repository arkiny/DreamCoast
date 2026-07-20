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
#include "hzb_test.slang" // conservative HZB occlusion test (P2), shared with the P7 grid cull

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
    uint status_buf;   // 152: per-work status (0 culled / 1 drawn / 2 defer) — csCutSceneP1 only
    uint bin_enabled;  // 156: 1 = HW/SW split like csCutSceneBin — csCutSceneP1 only
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

// True iff the cluster's WORLD bounds sphere straddles the near plane (`planes[4]` = the standard-Z
// `z>=0` near plane from `push::frustum_planes`). Near-crossing clusters must rasterize on the HW
// mesh path, whose fixed-function rasterizer clips at the near plane; the SW compute raster would
// otherwise drop whole near-crossing triangles (see docs/phase-14-vgeo-nearclip.md). This mirrors
// Nanite's `bUseHWRaster |= Cull.bNeedsClipping` (NaniteClusterCulling.usf). The cluster already
// passed the frustum cull, so it is not fully behind the near plane — this only flags a straddle.
bool crosses_near(float3 bnd_c, float bnd_r)
{
    return dot(pc.planes[4].xyz, bnd_c) + pc.planes[4].w < bnd_r;
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

    // HW for large clusters OR any near-plane straddler (the HW rasterizer clips near; the SW raster
    // cannot). SW otherwise.
    bool use_hw = (screen_diam >= pc.bin_px) || crosses_near(bnd_c, bnd_r);
    uint list = use_hw ? pc.vis_buf : pc.sw_list;
    uint args = use_hw ? pc.args_buf : pc.sw_args;
    uint slot;
    g.storage_buffers[args].InterlockedAdd(0, 1, slot);
    g.storage_buffers[list].Store(slot * 4, t);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Two-phase same-frame HZB occlusion (docs/phase-14-vgeo-hzb-occlusion.md). Phase 1 draws the
// clusters that were visible last frame (+ all HW/large clusters); the app builds a Hi-Z from
// that partial depth; phase 2 re-tests the DEFERRED clusters against it and draws the ones that
// survive. Correctness (hence DX≡VK and OFF≡ON) follows from the phase-2 test being *conservative*
// (`hzbOccluded`, the byte-verified P7 test): the R64 vis buffer resolves the final per-pixel
// winner by depth regardless of which clusters rastered, so dropping a genuinely-occluded cluster
// leaves the resolved G-buffer identical.
//
// `status[t]` (per-work, per-FIF, carried across frames): 0 = culled, 1 = drawn/visible, 2 = defer.
// csCutSceneP1 reuses `pc` (PushConstants) — the SW visible list is always `sw_list`/`sw_args`,
// the HW list is `vis_buf`/`args_buf` (binning), exactly matching csCutSceneBin's slots.

static const uint STATUS_CULLED = 0u;
static const uint STATUS_DRAWN = 1u;
static const uint STATUS_DEFER = 2u;

// Phase 1: run the identical cut selection, then route. HW/large clusters always draw in phase 1
// (never occlusion-tested). SW clusters draw in phase 1 only if they were visible last frame,
// else they are deferred to phase 2's HZB re-test. Every processed `t` gets a fresh status.
[shader("compute")]
[numthreads(64, 1, 1)]
void csCutSceneP1(uint3 tid: SV_DispatchThreadID)
{
    uint t = tid.x;
    if (t >= pc.total)
        return;
    uint2 w = loadWork(pc.work_buf, t);
    InstanceData inst = loadInstance(pc.inst_buf, w.x);
    uint cull_backface = (inst.mr.z != 0.0) ? 1u : 0u;
    RWByteAddressBuffer status = g.storage_buffers[pc.status_buf];

    float3 bnd_c;
    float bnd_r;
    if (!select_cluster(w.y, inst.model, inst.max_scale, cull_backface, bnd_c, bnd_r))
    {
        status.Store(t * 4, STATUS_CULLED);
        return;
    }

    // HW/SW split (binning). Large clusters — and any near-plane straddler (HW clips near, SW can't)
    // — go to the HW mesh list and always draw in phase 1.
    if (pc.bin_enabled != 0u)
    {
        float dist = max(distance(cam_pos(), bnd_c) - bnd_r, 1e-3);
        float screen_diam = 2.0 * bnd_r * pc.proj_factor / dist;
        if (screen_diam >= pc.bin_px || crosses_near(bnd_c, bnd_r))
        {
            uint hslot;
            g.storage_buffers[pc.args_buf].InterlockedAdd(0, 1, hslot);
            g.storage_buffers[pc.vis_buf].Store(hslot * 4, t);
            status.Store(t * 4, STATUS_DRAWN);
            return;
        }
    }

    // SW cluster: two-phase gated on last-frame visibility (status carried over per FIF).
    uint was = status.Load(t * 4);
    if (was == STATUS_DRAWN)
    {
        uint slot;
        g.storage_buffers[pc.sw_args].InterlockedAdd(0, 1, slot);
        g.storage_buffers[pc.sw_list].Store(slot * 4, t);
        status.Store(t * 4, STATUS_DRAWN);
    }
    else
    {
        status.Store(t * 4, STATUS_DEFER);
    }
}

// Phase 2 push — the deferred re-test needs only the cluster's world bounds + the Hi-Z. No
// frustum/LOD/cone (already passed in phase 1). `vp` = flip-free view_proj (columns), matching
// the instance-table `mvp` and the P7 cull convention.
struct CutP2Push
{
    float4 vp[4];    //   0: unjittered, no-Y-flip view_proj (columns)
    uint total;      //  64: W — total work-list length
    uint rec_buf;    //  68: GpuCluster records (96 B)
    uint work_buf;   //  72: work list (uint2)
    uint inst_buf;   //  76: instance table (SceneInstance, 256 B)
    uint status_buf; //  80: per-work status
    uint p2_list;    //  84: phase-2 SW visible list (work indices)
    uint p2_args;    //  88: phase-2 indirect args {countX,1,1}
    uint hzb_base;   //  92: bindless sampled index of vgeo HZB level 0
    uint hzb_levels; //  96: HZB mip count
    uint hzb_w;      // 100: HZB level-0 width  (texels)
    uint hzb_h;      // 104: HZB level-0 height (texels)
    uint stats_buf;  // 108: host-visible [deferred, occluded] counters; 0xffffffff = disabled
};

[[vk::push_constant]]
ConstantBuffer<CutP2Push> pc2 : register(b0, space0);

// The cluster's WORLD bounds sphere (center to world via `model`, radius scaled by `max_scale`) —
// the same two values `select_cluster` computes; factored out so phase 2 can recover them without
// re-running the full LOD/frustum/cone selection.
void cluster_world_bounds(uint rec_buf, uint gc, float4x4 model, float max_scale,
                          out float3 bnd_c, out float bnd_r)
{
    RWByteAddressBuffer rec = g.storage_buffers[rec_buf];
    uint base = gc * 96;
    bnd_c = to_world(model, asfloat(rec.Load3(base + 56)));
    bnd_r = asfloat(rec.Load(base + 68)) * max_scale;
}

// Phase 2: for each DEFERRED cluster, run the conservative HZB occlusion test on its world bounds
// sphere against the phase-1 Hi-Z. Survivors draw in phase 2 (append to the p2 SW list) and are
// marked visible; occluded clusters are dropped for the frame (status → culled).
[shader("compute")]
[numthreads(64, 1, 1)]
void csCutSceneP2(uint3 tid: SV_DispatchThreadID)
{
    uint t = tid.x;
    if (t >= pc2.total)
        return;
    RWByteAddressBuffer status = g.storage_buffers[pc2.status_buf];
    if (status.Load(t * 4) != STATUS_DEFER)
        return;
    if (pc2.stats_buf != 0xffffffffu)
    {
        uint dummy;
        g.storage_buffers[pc2.stats_buf].InterlockedAdd(0, 1, dummy); // deferred (HZB-tested)
    }

    uint2 w = loadWork(pc2.work_buf, t);
    InstanceData inst = loadInstance(pc2.inst_buf, w.x);
    float3 bnd_c;
    float bnd_r;
    cluster_world_bounds(pc2.rec_buf, w.y, inst.model, inst.max_scale, bnd_c, bnd_r);

    if (hzbOccluded(pc2.vp, pc2.hzb_base, pc2.hzb_levels, pc2.hzb_w, pc2.hzb_h, bnd_c, bnd_r))
    {
        if (pc2.stats_buf != 0xffffffffu)
        {
            uint dummy;
            g.storage_buffers[pc2.stats_buf].InterlockedAdd(4, 1, dummy); // occlusion-culled
        }
        status.Store(t * 4, STATUS_CULLED); // provably hidden → drop (resolve unchanged)
        return;
    }

    uint slot;
    g.storage_buffers[pc2.p2_args].InterlockedAdd(0, 1, slot);
    g.storage_buffers[pc2.p2_list].Store(slot * 4, t);
    status.Store(t * 4, STATUS_DRAWN);
}

// ── HW indirect-args fixup (single thread, after the cut, before the HW mesh draw). ──
// Mesh dispatch caps at 65,535 threadgroups PER DIMENSION (Metal drawMeshThreadgroups and
// D3D12 DispatchMesh alike); the cut's flat `InterlockedAdd` count exceeded that on real
// content (native-res sponza_hero: 68,520 HW clusters) and the ENTIRE draw was dropped.
// Fold the count into an (x <= VGEO_HW_GRID_X, y, 1) grid; `vgeo_scene_hwvis` flattens
// `gid.y * VGEO_HW_GRID_X + gid.x` back. count <= GRID_X keeps y = 1 and x = count — the
// exact legacy grid. The raw count is preserved at byte offset 12 (diagnostics: the
// VGEO_BIN_STATS readback), which the 16-B args buffer carries after the 12-B dispatch args.

struct FixupPush
{
    uint args_buf; // 0: the HW indirect-args buffer {x, y, 1, raw_count}
};

[[vk::push_constant]]
ConstantBuffer<FixupPush> pcf : register(b0, space0);

[shader("compute")]
[numthreads(1, 1, 1)]
void csBinArgsFixup(uint3 tid: SV_DispatchThreadID)
{
    RWByteAddressBuffer args = g.storage_buffers[pcf.args_buf];
    uint count = args.Load(0);
    uint x = count;
    uint y = 1;
    if (count > VGEO_HW_GRID_X)
    {
        x = VGEO_HW_GRID_X;
        y = (count + VGEO_HW_GRID_X - 1) / VGEO_HW_GRID_X;
    }
    args.Store(0, x);
    args.Store(4, y);
    args.Store(12, count);
}
