// GPU frustum culling (Phase 7): a grid of cube instances is culled on the GPU.
// `csReset` clears the indirect draw-args header; `csCull` tests each instance's
// bounding sphere against the six frustum planes and, for visible ones, appends
// its index to a list and atomically bumps the draw's InstanceCount. The draw
// pass then issues a single `draw_indexed_indirect` over the visible instances.
//
// Buffers are raw byte-address UAVs in the bindless storage-buffer table:
//  - args:    5x u32 VkDrawIndexedIndirectCommand / D3D12_DRAW_INDEXED_ARGUMENTS
//             [index_count, instance_count, first_index, vertex_offset, first_inst]
//  - visible: u32[count] list of visible instance indices

struct PushConstants
{
    float4 planes[6];   // frustum planes (xyz=normal, w=d), inward-facing, normalized
    uint args_index;    // bindless index of the indirect args buffer
    uint visible_index; // bindless index of the visible-list buffer
    uint count;         // total instances
    uint grid_dim;      // instances per row
    float spacing;      // grid cell spacing (world units)
    float cube_radius;  // instance bounding-sphere radius (world units)
    float y_height;     // grid plane height
    uint index_count;   // cube index count (written into args)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

#include "bindless.slang"

// World-space center of grid instance `i`.
float3 instance_center(uint i)
{
    uint gx = i % pc.grid_dim;
    uint gz = i / pc.grid_dim;
    float half = (float(pc.grid_dim) - 1.0) * 0.5;
    return float3((float(gx) - half) * pc.spacing, pc.y_height, (float(gz) - half) * pc.spacing);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csReset(uint3 tid: SV_DispatchThreadID)
{
    // Header: index_count, instance_count=0, first_index=0, vertex_offset=0,
    // first_instance=0.
    RWByteAddressBuffer args = g.storage_buffers[pc.args_index];
    args.Store4(0, uint4(pc.index_count, 0, 0, 0));
    args.Store(16, 0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csCull(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.count)
        return;

    float3 center = instance_center(i);
    bool visible = true;
    [unroll]
    for (int p = 0; p < 6; p++)
    {
        if (dot(pc.planes[p].xyz, center) + pc.planes[p].w < -pc.cube_radius)
            visible = false;
    }
    if (!visible)
        return;

    // Append this instance and bump the draw's InstanceCount (offset 4).
    uint slot;
    g.storage_buffers[pc.args_index].InterlockedAdd(4, 1, slot);
    g.storage_buffers[pc.visible_index].Store(slot * 4, i);
}

// ---------------------------------------------------------------------------
// HZB occlusion culling (PR-8). `csCullHzb` is a SEPARATE 256-byte-push entry so
// the default `csCull` path above stays byte-identical when HZB is disabled. Its
// push is a superset of `PushConstants`: bytes 0..128 are the identical frustum
// block, bytes 128..256 add the occlusion block (view_proj + HZB metadata).
//
// The HZB stores the FARTHEST occluder depth per texel region (max-reduced, see
// hzb_build.slang; standard-Z near=0 far=1). An instance is occluded iff its
// NEAREST screen-space depth is still greater (farther) than the HZB depth over
// its whole screen AABB. We read the coarsest mip whose texel covers the AABB, so
// ONE tap conservatively bounds the region (the canonical single-tap Hi-Z test).
// Conservative by construction: the HZB is the PREVIOUS frame's depth, and any
// ambiguity (behind near plane, off-screen edge, co-planar) keeps it visible.
// ---------------------------------------------------------------------------
struct CullHzbPush
{
    float4 planes[6];    //   0: frustum planes (identical to PushConstants)
    uint args_index;     //  96
    uint visible_index;  // 100
    uint count;          // 104
    uint grid_dim;       // 108
    float spacing;       // 112
    float cube_radius;   // 116
    float y_height;      // 120
    uint index_count;    // 124
    float4 vp[4];        // 128: unjittered, no-Y-flip view_proj (matches frustum)
    uint hzb_base;       // 192: bindless sampled index of HZB level 0
    uint hzb_levels;     // 196: HZB mip count
    uint hzb_w;          // 200: HZB level-0 width  (texels)
    uint hzb_h;          // 204: HZB level-0 height (texels)
    uint enabled;        // 208: 1 = run the occlusion test (0 = frustum only)
    uint stats_index;    // 212: bindless index of the host-visible stats buffer
                         //      ([0]=survived, [1]=occlusion-culled); 0xffffffff = none
    uint _pad1;          // 216
    uint _pad2;          // 220
};

[[vk::push_constant]]
ConstantBuffer<CullHzbPush> hc : register(b0, space0);

// Shared conservative HZB occlusion test (single source of truth, also used by the Phase-14 vgeo
// two-phase cut). Requires `bindless.slang` (included above).
#include "hzb_test.slang"

// Zero the (survived, occlusion-culled) stats counters. Runs as a 1-thread dispatch
// ordered (via a UAV barrier) BEFORE the cull, so the per-frame count is correct even
// with frames in flight (a host-side zero would race the in-flight GPU writes).
[shader("compute")]
[numthreads(1, 1, 1)]
void csClearStats(uint3 tid: SV_DispatchThreadID)
{
    if (hc.stats_index != 0xffffffffu)
        g.storage_buffers[hc.stats_index].Store2(0, uint2(0, 0));
}

float3 hzb_instance_center(uint i)
{
    uint gx = i % hc.grid_dim;
    uint gz = i / hc.grid_dim;
    float half = (float(hc.grid_dim) - 1.0) * 0.5;
    return float3((float(gx) - half) * hc.spacing, hc.y_height, (float(gz) - half) * hc.spacing);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csCullHzb(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= hc.count)
        return;

    float3 center = hzb_instance_center(i);
    // Frustum test (identical logic to csCull).
    [unroll]
    for (int p = 0; p < 6; p++)
    {
        if (dot(hc.planes[p].xyz, center) + hc.planes[p].w < -hc.cube_radius)
            return;
    }

    // Conservative HZB occlusion test (shared `hzbOccluded`, the single source of truth also used
    // by the Phase-14 vgeo two-phase cut), against last frame's pyramid.
    if (hc.enabled != 0
        && hzbOccluded(hc.vp, hc.hzb_base, hc.hzb_levels, hc.hzb_w, hc.hzb_h, center, hc.cube_radius))
    {
        if (hc.stats_index != 0xffffffffu)
        {
            uint dummy;
            g.storage_buffers[hc.stats_index].InterlockedAdd(4, 1, dummy); // occlusion-culled
        }
        return;
    }

    if (hc.stats_index != 0xffffffffu)
    {
        uint dummy;
        g.storage_buffers[hc.stats_index].InterlockedAdd(0, 1, dummy); // survived
    }
    uint slot;
    g.storage_buffers[hc.args_index].InterlockedAdd(4, 1, slot);
    g.storage_buffers[hc.visible_index].Store(slot * 4, i);
}
