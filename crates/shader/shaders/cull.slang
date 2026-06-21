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

[[vk::binding(4, 0)]]
RWByteAddressBuffer g_storage_buffers[] : register(u0, space1);

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
    RWByteAddressBuffer args = g_storage_buffers[pc.args_index];
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
    g_storage_buffers[pc.args_index].InterlockedAdd(4, 1, slot);
    g_storage_buffers[pc.visible_index].Store(slot * 4, i);
}
