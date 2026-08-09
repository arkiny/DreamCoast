// Global distance field — per-clipmap instance cull (gdf-scale-follow-plan.md U1/U2).
//
// One thread per instance of the pushed RANGE: test the instance's WORLD AABB against
// each clipmap level's box (grown by the level's own voxel so a surface just outside
// still influences border voxels) and append its index to that level's candidate list.
// The composite kernel then iterates only the survivors — the reference engine's
// clipmap-cull stage at our scale (its per-page object grid is the U3 refinement if
// counts demand).
//
// U2 splits the instance table into a static block and a movable block ([static..]
// [movable..], re-partitioned on membership change): the static range culls only on
// batch frames into the static list section, the movable range culls into its own
// section whenever movers have work. World AABBs live in a parallel 32-byte side
// table — the 112-byte record's rows 3/4 are the tile's LOCAL bounds (the composite's
// sampling frame), which are NOT a world-space cull box for a transformed instance.

#include "bindless.slang"

struct GdfCullPush
{
    // x = instance world-AABB table (32 B rows: min.xyz pad, max.xyz pad), y = range
    // instance count, z = culled list buffer, w = level count.
    uint4 a;
    // x = consts buffer (level boxes at 64 B/level), y = per-level list capacity,
    // z = range base instance index, w = culled list section offset (u32 words).
    uint4 b;
};

[[vk::push_constant]]
ConstantBuffer<GdfCullPush> pc : register(b0, space0);

// Culled list layout: per level, [0] = count (atomic), then capacity indices.
uint list_base(uint section, uint level, uint capacity)
{
    return section + level * (capacity + 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.a.y)
        return;
    uint i = pc.b.z + tid.x;
    RWByteAddressBuffer aabbs = g.storage_buffers[pc.a.x];
    float3 wmin = asfloat(aabbs.Load4(i * 32u + 0u)).xyz;
    float3 wmax = asfloat(aabbs.Load4(i * 32u + 16u)).xyz;
    // A retired record (a despawned mover) keeps its slot with an inverted AABB — it
    // fails every overlap test below and never reaches a list.
    if (any(wmin > wmax))
        return;

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
        list.InterlockedAdd(list_base(pc.b.w, l, pc.b.y) * 4u, 1u, slot);
        if (slot < pc.b.y)
            list.Store((list_base(pc.b.w, l, pc.b.y) + 1u + slot) * 4u, i);
    }
}
