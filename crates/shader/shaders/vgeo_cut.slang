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

// NOTE: the camera position is three SCALAR floats, NOT a `float3` — Metal aligns `float3` to 16
// bytes in a constant buffer, which would shift every field after it and break the CPU-side push
// packing. Scalars pack tight (4 B each), matching the byte layout the app writes.
struct PushConstants
{
    float cam_x; // world-space camera position (scalar to avoid float3 alignment)
    float cam_y;
    float cam_z;
    float proj_factor; // 0.5 * screen_height / tan(fov/2): world error → pixels at unit distance
    float tau;         // pixel-error threshold (the LOD "tightness" knob)
    uint total;        // total cluster count (all LODs)
    uint rec_buf;      // GpuCluster records slot
    uint vis_buf;      // output: visible-cluster list slot
    uint args_buf;     // output: MTLDrawMeshThreadgroupsIndirectArguments {countX, 1, 1}
    uint2 pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

// Screen-space (pixel) error of a world-space error `err` on a sphere (`center`, `radius`):
// project by the near distance to the sphere. A cluster right at the camera → huge error.
float screen_error(float err, float3 center, float radius)
{
    float3 cam = float3(pc.cam_x, pc.cam_y, pc.cam_z);
    float d = max(distance(cam, center) - radius, 1e-3);
    return err * pc.proj_factor / d;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csCut(uint3 tid: SV_DispatchThreadID)
{
    uint c = tid.x;
    if (c >= pc.total)
        return;

    RWByteAddressBuffer rec = g.storage_buffers[pc.rec_buf];
    uint base = c * 64;
    float self_err = asfloat(rec.Load(base + 16));
    float3 self_c = asfloat(rec.Load3(base + 20));
    float self_r = asfloat(rec.Load(base + 32));
    float par_err = asfloat(rec.Load(base + 36));
    float3 par_c = asfloat(rec.Load3(base + 40));
    float par_r = asfloat(rec.Load(base + 52));

    float s_self = screen_error(self_err, self_c, self_r);
    // A root cluster stores parent_error = FLT_MAX ("never coarsen past here") → always > tau.
    float s_par = (par_err >= 3.0e38) ? 1.0e30 : screen_error(par_err, par_c, par_r);

    // Draw this cluster iff its LOD is fine enough (self <= tau) but the parent is too coarse
    // (parent > tau). Leaves have self_err 0 (always fine); the root has parent = inf (always
    // the stop). Group-uniform errors make every cluster in a group agree → no cracks.
    if (s_self <= pc.tau && s_par > pc.tau)
    {
        uint slot;
        // Bump the indirect threadgroup count (args.x) and append our cluster index.
        g.storage_buffers[pc.args_buf].InterlockedAdd(0, 1, slot);
        g.storage_buffers[pc.vis_buf].Store(slot * 4, c);
    }
}
