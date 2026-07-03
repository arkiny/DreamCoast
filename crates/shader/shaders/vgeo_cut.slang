// Phase 14 M3 — view-dependent LOD DAG cut selection (compute).
//
// One thread per cluster walks the cooked DAG and picks the **cut**: the set of clusters whose
// own LOD is fine enough on screen but whose parent (one level coarser) is not. Because the
// `self`/`parent` error+sphere are group-uniform (M1d) and monotone, exactly one cluster per
// surface lineage passes, giving a complete, crack-free, continuous LOD. Selected cluster indices
// are appended to a visible list, and the count is written as the mesh-draw threadgroup grid so
// `draw_mesh_tasks_indirect` renders exactly the cut.
//
// GpuCluster record (64 B, ALL clusters), matching the CPU upload:
//   0  : vertex_offset (u32)   16 : self_error (f32)     36 : parent_error (f32)
//   4  : vertex_count  (u32)   20 : self_center.xyz      40 : parent_center.xyz
//   8  : tri_offset    (u32)   32 : self_radius (f32)    52 : parent_radius (f32)
//   12 : tri_count     (u32)                             56..64 : pad
#include "bindless.slang"

// NOTE: `float4` aligns to 16 B (so `planes[6]` packs tight), but a bare `float3` also aligns to
// 16 B in a constant buffer and would silently shift every scalar after it — so the camera
// position is three SCALAR floats, matching the byte layout the app writes.
struct PushConstants
{
    float4 planes[6];  //   0: WORLD frustum planes (xyz=inward normal, w=d), normalized (96 B)
    float cam_x;       //  96: WORLD camera position (scalars, no float3 alignment)
    float cam_y;
    float cam_z;
    float proj_factor; // 108: 0.5 * screen_height / tan(fov/2): world error → px at unit distance
    float tau;         // 112: pixel-error threshold (the LOD "tightness" knob)
    uint total;        // 116: total cluster count (all LODs)
    uint rec_buf;      // 120: GpuCluster records slot (96 B/record)
    uint vis_buf;      // 124: output list slot — `csCut`: the whole cut; `csCutBin`: the HW list
    uint args_buf;     // 128: output args {countX,1,1} — `csCut`: the cut; `csCutBin`: HW args
    uint sw_list;      // 132: `csCutBin` SW-path visible-cluster list slot
    uint sw_args;      // 136: `csCutBin` SW-path indirect args {countX,1,1}
    float bin_px;      // 140: SW/HW split threshold — projected bounds diameter (px) below → SW
    // Cluster bounds are stored in the mesh's LOCAL space; `model` transforms them to WORLD so the
    // cull/error work for ANY object transform (incl. non-uniform node scale, where a local-space
    // sphere test would be skewed). The self-contained viewer recenters its mesh and passes
    // `model = identity`, `max_scale = 1` → a no-op (its old local == world behavior).
    float4x4 model;    // 144: object → world (192 B total after the mat4)
    float max_scale;   // 208: max column scale of `model` (conservatively scales sphere radii/errors)
    uint rec_base;     // 212: this page's first GLOBAL cluster index in the shared record buffer.
                       //      Local cluster c → global gc = rec_base + c; the visible list stores gc
                       //      so the raster/mesh/resolve address the consolidated buffers directly.
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

float3 cam_pos()
{
    return float3(pc.cam_x, pc.cam_y, pc.cam_z);
}

// A local-space point → world.
float3 to_world(float3 p)
{
    return mul(pc.model, float4(p, 1.0)).xyz;
}

// Screen-space (pixel) error of a WORLD error `err_w` on a WORLD sphere (`center_w`, `radius_w`):
// project by the near distance to the sphere. A cluster right at the camera → huge error.
float screen_error(float err_w, float3 center_w, float radius_w)
{
    float d = max(distance(cam_pos(), center_w) - radius_w, 1e-3);
    return err_w * pc.proj_factor / d;
}

// Frustum cull: a bounding sphere is outside iff it is fully behind any inward-facing plane.
bool outside_frustum(float3 center, float radius)
{
    [unroll]
    for (int p = 0; p < 6; ++p)
        if (dot(pc.planes[p].xyz, center) + pc.planes[p].w < -radius)
            return true;
    return false;
}

// Normal-cone backface cull: every face normal lies within a cone of half-angle acos(cutoff) of
// `axis`. The cluster is entirely backfacing iff the direction to the camera is more than 90°
// past that spread from the axis: dot(axis, dirToCam) < -sin(halfAngle) = -sqrt(1-cutoff²).
// `cutoff <= 0` means the spread is too wide to cull (stored -1 at build). Conservative → never
// culls a visible cluster.
bool cone_backfacing(float3 axis, float cutoff, float3 center)
{
    if (cutoff <= 0.0)
        return false;
    float3 to_cam = normalize(cam_pos() - center);
    float sin_half = sqrt(max(1.0 - cutoff * cutoff, 0.0));
    return dot(axis, to_cam) < -sin_half;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csCut(uint3 tid: SV_DispatchThreadID)
{
    uint c = tid.x;
    if (c >= pc.total)
        return;
    uint gc = pc.rec_base + c; // global cluster index into the shared record buffer

    RWByteAddressBuffer rec = g.storage_buffers[pc.rec_buf];
    uint base = gc * 96;
    float self_err = asfloat(rec.Load(base + 16));
    float3 self_c = asfloat(rec.Load3(base + 20));
    float self_r = asfloat(rec.Load(base + 32));
    float par_err = asfloat(rec.Load(base + 36));
    float3 par_c = asfloat(rec.Load3(base + 40));
    float par_r = asfloat(rec.Load(base + 52));
    float3 bnd_c = asfloat(rec.Load3(base + 56));
    float bnd_r = asfloat(rec.Load(base + 68));
    float3 cone_axis = asfloat(rec.Load3(base + 72));
    float cone_cutoff = asfloat(rec.Load(base + 84));

    float ms = pc.max_scale;
    // LOD cut: draw iff this cluster's LOD is fine enough (self <= tau) but the parent is too
    // coarse (parent > tau). Group-uniform errors make every cluster in a group agree → no cracks.
    float s_self = screen_error(self_err * ms, to_world(self_c), self_r * ms);
    float s_par =
        (par_err >= 3.0e38) ? 1.0e30 : screen_error(par_err * ms, to_world(par_c), par_r * ms);
    if (!(s_self <= pc.tau && s_par > pc.tau))
        return;

    // Visibility cull (M4a): drop clusters that are off-screen (frustum) or entirely backfacing
    // (normal cone), in WORLD space. Both use the cluster's OWN bounds sphere — conservative
    // (radius scaled by the max axis scale), never drops a visible cluster.
    float3 bnd_w = to_world(bnd_c);
    if (outside_frustum(bnd_w, bnd_r * ms))
        return;
    if (cone_backfacing(normalize(mul((float3x3)pc.model, cone_axis)), cone_cutoff, bnd_w))
        return;

    {
        uint slot;
        // Bump the indirect threadgroup count (args.x) and append our GLOBAL cluster index.
        g.storage_buffers[pc.args_buf].InterlockedAdd(0, 1, slot);
        g.storage_buffers[pc.vis_buf].Store(slot * 4, gc);
    }
}

// Shared selection: the LOD cut + frustum + normal-cone tests (identical to `csCut`), returning
// the cluster's own bounds sphere so the caller can also estimate its screen size for binning.
bool select_cluster(uint c, out float3 bnd_c, out float bnd_r)
{
    RWByteAddressBuffer rec = g.storage_buffers[pc.rec_buf];
    uint base = c * 96;
    float self_err = asfloat(rec.Load(base + 16));
    float3 self_c = asfloat(rec.Load3(base + 20));
    float self_r = asfloat(rec.Load(base + 32));
    float par_err = asfloat(rec.Load(base + 36));
    float3 par_c = asfloat(rec.Load3(base + 40));
    float par_r = asfloat(rec.Load(base + 52));
    float3 cone_axis = asfloat(rec.Load3(base + 72));
    float cone_cutoff = asfloat(rec.Load(base + 84));

    float ms = pc.max_scale;
    // Bounds returned in WORLD space (the binning caller projects them to a screen diameter).
    bnd_c = to_world(asfloat(rec.Load3(base + 56)));
    bnd_r = asfloat(rec.Load(base + 68)) * ms;

    float s_self = screen_error(self_err * ms, to_world(self_c), self_r * ms);
    float s_par =
        (par_err >= 3.0e38) ? 1.0e30 : screen_error(par_err * ms, to_world(par_c), par_r * ms);
    if (!(s_self <= pc.tau && s_par > pc.tau))
        return false;
    if (outside_frustum(bnd_c, bnd_r))
        return false;
    if (cone_backfacing(normalize(mul((float3x3)pc.model, cone_axis)), cone_cutoff, bnd_c))
        return false;
    return true;
}

// M5b — HW/SW binning cut. Same cut selection as `csCut`, then classify each selected cluster by
// its projected screen diameter: clusters smaller than `bin_px` (their ~124 triangles are
// sub-pixel) go to the SW compute rasterizer, larger clusters to the HW mesh shader. Both paths
// write the SAME R64 visibility buffer with the same packed rule, so the boundary has no seam.
[shader("compute")]
[numthreads(64, 1, 1)]
void csCutBin(uint3 tid: SV_DispatchThreadID)
{
    uint c = tid.x;
    if (c >= pc.total)
        return;
    uint gc = pc.rec_base + c; // global cluster index into the shared record buffer
    float3 bnd_c;
    float bnd_r;
    if (!select_cluster(gc, bnd_c, bnd_r))
        return;

    // Projected bounds diameter in pixels: 2r * proj_factor / (nearest distance to the sphere).
    float dist = max(distance(cam_pos(), bnd_c) - bnd_r, 1e-3);
    float screen_diam = 2.0 * bnd_r * pc.proj_factor / dist;

    uint list = (screen_diam < pc.bin_px) ? pc.sw_list : pc.vis_buf;
    uint args = (screen_diam < pc.bin_px) ? pc.sw_args : pc.args_buf;
    uint slot;
    g.storage_buffers[args].InterlockedAdd(0, 1, slot);
    g.storage_buffers[list].Store(slot * 4, gc);
}
