// Global distance field — per-clipmap instance cull (gdf-scale-follow-plan.md U1).
//
// One thread per instance: test the instance's world AABB against each clipmap level's
// box (grown by the level's own voxel so a surface just outside still influences border
// voxels) and append its index to that level's candidate list. The composite kernel
// then iterates only the survivors — the reference engine's CullObjectsToClipmapCS
// stage at our scale (its per-page object grid is the U3 refinement if counts demand).
//
// Instance records are the SAME 112-byte layout `mesh_sdf_sample.slang` reads (rows of
// world→local + world AABB + tile map), re-uploaded every frame from the draw list —
// the dynamic-game seam: a mover's transform reaches this cull the frame it moves.

#include "bindless.slang"

struct GdfCullPush
{
    // x = instance buffer (112 B records), y = instance count, z = culled list buffer,
    // w = level count.
    uint4 a;
    // Level boxes, xyz min / xyz max packed as two float4 rows per level follow in the
    // CONSTS buffer at `b.x`; b.y = per-level list capacity. b.zw spare.
    uint4 b;
};

[[vk::push_constant]]
ConstantBuffer<GdfCullPush> pc : register(b0, space0);

// Culled list layout: per level, [0] = count (atomic), then capacity indices.
uint list_base(uint level, uint capacity)
{
    return level * (capacity + 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.a.y)
        return;
    RWByteAddressBuffer inst = g.storage_buffers[pc.a.x];
    uint base = i * 112u;
    float3 wmin = asfloat(inst.Load4(base + 48u)).xyz;
    float3 wmax = asfloat(inst.Load4(base + 64u)).xyz;

    RWByteAddressBuffer consts = g.storage_buffers[pc.b.x];
    RWByteAddressBuffer list = g.storage_buffers[pc.a.z];
    for (uint l = 0u; l < pc.a.w; l++)
    {
        float4 lmin = asfloat(consts.Load4(l * 64u + 0u));
        float4 lmax = asfloat(consts.Load4(l * 64u + 16u));
        // Grow by one voxel (lmin.w carries the level's voxel size) so borderline
        // surfaces still write their border voxels.
        float grow = lmin.w;
        if (any(wmax < lmin.xyz - grow) || any(wmin > lmax.xyz + grow))
            continue;
        uint slot;
        list.InterlockedAdd(list_base(l, pc.b.y) * 4u, 1u, slot);
        if (slot < pc.b.y)
            list.Store((list_base(l, pc.b.y) + 1u + slot) * 4u, i);
    }
}
