// Phase 11 Stage B (B3) — global distance field (GDF) merge.
//
// Composites N instances of per-mesh baked SDFs (B2) into one world-space global
// distance volume via a min-combine. One thread per GDF voxel: its world-space
// centre (mapped from the GDF's world AABB) is transformed into each instance's
// local bake box, the instance's baked SDF volume is trilinear-sampled there, the
// sampled (bake-space) distance is scaled back to world units, and the minimum over
// all instances is written. This is the union of all instanced surfaces — the field
// the SW ray marcher (B4) walks.
//
// An instance contributes only where the GDF voxel falls inside its bake box (uvw in
// [0,1]); outside, the trilinear sample would clamp to the border and underestimate
// the true distance, so it is skipped (treated as +inf). The valid GDF region is
// therefore the union of the instance boxes — exactly how a real clipmap tiles the
// camera neighbourhood. A single instance whose box equals the GDF AABB reproduces
// its baked volume byte-for-byte (the B3 regression anchor against B2).
//
// First cut: a single global level, static instances merged once. A multi-resolution
// clipmap and per-frame dynamic-instance updates are later refinements.

#include "bindless.slang"

struct MergePush
{
    uint gdf_storage; // storage_volumes[] index to write (the GDF)
    uint dim;         // GDF edge length in voxels (cube)
    uint inst_table;  // storage-buffer index of the instance record array
    uint inst_count;  // number of instances
    float4 aabb_min;  // xyz world-space min of the GDF extent
    float4 aabb_max;  // xyz world-space max of the GDF extent
};

[[vk::push_constant]]
ConstantBuffer<MergePush> pc : register(b0, space0);

// One 32-byte instance record (see `gdf_instance_record` on the Rust side):
//   off 0  float3 origin      world position of the instance's bake-box min corner
//   off 12 float  dist_scale  multiply a sampled bake distance to get world units
//   off 16 float3 inv_extent  1 / (world-space size of the bake box) — world delta -> uvw
//   off 28 uint   sampled     volumes[] (SRV) index of this instance's baked SDF
struct Instance
{
    float3 origin;
    float dist_scale;
    float3 inv_extent;
    uint sampled;
};

Instance load_instance(uint i)
{
    uint base = i * 32u;
    RWByteAddressBuffer t = g.storage_buffers[pc.inst_table];
    Instance inst;
    inst.origin = asfloat(t.Load3(base + 0u));
    inst.dist_scale = asfloat(t.Load(base + 12u));
    inst.inv_extent = asfloat(t.Load3(base + 16u));
    inst.sampled = t.Load(base + 28u);
    return inst;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void mergeMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dim || tid.y >= pc.dim || tid.z >= pc.dim)
        return;

    float3 t = (float3(tid) + 0.5) / float(pc.dim);
    float3 p = lerp(pc.aabb_min.xyz, pc.aabb_max.xyz, t); // world-space voxel centre

    float best = 1e30;
    for (uint i = 0; i < pc.inst_count; i++)
    {
        Instance inst = load_instance(i);
        float3 uvw = (p - inst.origin) * inst.inv_extent; // [0,1] inside the bake box
        // Outside the box the trilinear clamp would lie about the distance; skip it so
        // this instance contributes only where its baked field is trustworthy.
        if (any(uvw < 0.0) || any(uvw > 1.0))
            continue;
        float d = g.volumes[inst.sampled].SampleLevel(g.samp, uvw, 0.0).x * inst.dist_scale;
        best = min(best, d);
    }

    g.storage_volumes[pc.gdf_storage][tid] = best;
}
