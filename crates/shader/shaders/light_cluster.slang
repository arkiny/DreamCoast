// Clustered light culling (PR-6 · pipeline re-baseline §2 #8).
//
// Splits the view frustum into a fixed 3D grid of froxels (clusters) and, per
// cluster, builds a compact list of the point/spot lights whose sphere of
// influence overlaps that cluster's AABB. The deferred lighting pass then reads
// only its pixel's cluster list instead of looping every light in the scene, so
// the shading cost scales with lights-per-cluster rather than lights-in-scene.
//
// Design (see docs/clustered-lighting.md): a full 3D froxel grid with per-cluster
// index lists, lights visited in ASCENDING GLOBAL LIGHT INDEX order. Modern engines
// (DOOM 2016 / Detroit / Granite) decouple XY tiles from a 1D Z-bin range
// (O(X*Y + Z) memory) and sort lights by depth, which is superior at thousands of
// lights but reorders per-pixel light accumulation. This engine's hard parity gate
// wants byte-identical output vs the brute-force loop, which needs the SAME lights
// in the SAME order — froxel lists preserve global-index order, Z-binning does not.
// The XY*Z froxel count here is tiny (see CLUSTER_* below), so the memory cost is a
// non-issue at the current scale; the Z-bin evolution is the documented scale path.
//
// Two passes:
//   csBuildClusters — one thread per cluster: build the cluster's view-space AABB
//                     from its screen tile + exponential Z slice, cull every light
//                     against it, and append the survivors to that cluster's slot in
//                     the global index list (fixed MAX_LIGHTS_PER_CLUSTER capacity).
//   (the light grid is implicit: cluster c owns index-list slots
//    [c*MAX_LIGHTS_PER_CLUSTER .. c*MAX_LIGHTS_PER_CLUSTER + count], count in a
//    parallel per-cluster u32 array — a flat, alloc-free layout, so no atomics.)

#include "bindless.slang"
#include "light_cluster_common.slang"

struct PushConstants
{
    uint light_buf;    // bindless storage index: packed Light[] (see Light in common)
    uint grid_buf;     // bindless storage index: per-cluster light COUNT u32[num_clusters]
    uint index_buf;    // bindless storage index: flat light-index list u32[num_clusters * MAX]
    uint light_count;  // number of point lights in light_buf

    float4 view_z_row; // row 2 of the world->view matrix: view_z = dot(view_z_row, float4(p,1))

    float z_near;
    float z_far;
    uint num_x;
    uint num_y;

    uint num_z;
    uint screen_w;      // full-res width (cluster tiles map to it)
    uint screen_h;      // full-res height
    uint pad0;

    float4x4 inv_view_proj; // clip -> world, to unproject tile corners to world-space rays
    float4 camera_pos;      // xyz eye (world) — cluster AABB apex
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

// Exponential (log) Z slice boundary in positive linear view depth (DOOM 2016 scheme):
//   Z(slice) = near * (far/near) ^ (slice / num_z)
// Self-similar slices counteract perspective non-linearity so near clusters (where
// most lights matter) are thin and far ones coarse.
float z_slice_depth(uint slice)
{
    return pc.z_near * pow(pc.z_far / pc.z_near, float(slice) / float(pc.num_z));
}

// Unproject a screen-space point (pixel coords) at clip-depth 1 to a world-space
// point on the far plane, then form the eye->point ray. We intersect that ray with
// the cluster's near/far view-Z planes to get the cluster's world extent.
float3 screen_to_world_dir(float2 pix)
{
    float2 uv = pix / float2(float(pc.screen_w), float(pc.screen_h));
    // NDC. The cluster AABB is built in WORLD space, matched at shade time by the same
    // world position, so the Vulkan/D3D12 clip-Y flip cancels: use the D3D-style
    // (y-down-uv -> y-up-ndc) here and the identical mapping in the shade path.
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 far = mul(pc.inv_view_proj, float4(ndc, 1.0, 1.0));
    float3 world = far.xyz / far.w;
    return world - pc.camera_pos.xyz;
}

// World-space point at linear view-depth `vz` along the eye ray through `pix`.
float3 point_at_depth(float2 pix, float vz)
{
    float3 dir = screen_to_world_dir(pix);
    // dir spans eye->far-plane; scale so its view-Z magnitude equals vz.
    // The far plane sits at view-Z = -z_far (RH), so |view_z(dir)| = z_far.
    float ray_vz = -(pc.view_z_row.x * dir.x + pc.view_z_row.y * dir.y + pc.view_z_row.z * dir.z);
    float t = vz / max(ray_vz, 1e-6);
    return pc.camera_pos.xyz + dir * t;
}

[shader("compute")]
[numthreads(CLUSTER_BUILD_GROUP, 1, 1)]
void csBuildClusters(uint3 tid: SV_DispatchThreadID)
{
    uint num_clusters = pc.num_x * pc.num_y * pc.num_z;
    uint cluster = tid.x;
    if (cluster >= num_clusters)
        return;

    // Decode the cluster's (cx, cy, cz) from its flat index (x fastest, then y, then z).
    uint cx = cluster % pc.num_x;
    uint cy = (cluster / pc.num_x) % pc.num_y;
    uint cz = cluster / (pc.num_x * pc.num_y);

    float tile_w = float(pc.screen_w) / float(pc.num_x);
    float tile_h = float(pc.screen_h) / float(pc.num_y);
    float2 pmin = float2(float(cx) * tile_w, float(cy) * tile_h);
    float2 pmax = float2(float(cx + 1) * tile_w, float(cy + 1) * tile_h);

    float near_z = z_slice_depth(cz);
    float far_z = z_slice_depth(cz + 1);

    // Cluster world-space AABB: the 8 corners are the tile's 4 screen corners placed at
    // the near and far view-depth planes. min/max over them bounds the froxel.
    float3 corners[8];
    corners[0] = point_at_depth(float2(pmin.x, pmin.y), near_z);
    corners[1] = point_at_depth(float2(pmax.x, pmin.y), near_z);
    corners[2] = point_at_depth(float2(pmin.x, pmax.y), near_z);
    corners[3] = point_at_depth(float2(pmax.x, pmax.y), near_z);
    corners[4] = point_at_depth(float2(pmin.x, pmin.y), far_z);
    corners[5] = point_at_depth(float2(pmax.x, pmin.y), far_z);
    corners[6] = point_at_depth(float2(pmin.x, pmax.y), far_z);
    corners[7] = point_at_depth(float2(pmax.x, pmax.y), far_z);

    float3 aabb_min = corners[0];
    float3 aabb_max = corners[0];
    [unroll]
    for (int i = 1; i < 8; ++i)
    {
        aabb_min = min(aabb_min, corners[i]);
        aabb_max = max(aabb_max, corners[i]);
    }

    RWByteAddressBuffer lights = g.storage_buffers[pc.light_buf];
    RWByteAddressBuffer grid = g.storage_buffers[pc.grid_buf];
    RWByteAddressBuffer index_list = g.storage_buffers[pc.index_buf];

    uint base = cluster * MAX_LIGHTS_PER_CLUSTER;
    uint count = 0;
    // Visit lights in ascending global index; append survivors. The lighting pass then
    // iterates this list in the SAME order, so per-pixel accumulation matches the
    // brute-force loop exactly (byte-identical parity when few lights).
    for (uint li = 0; li < pc.light_count; ++li)
    {
        Light L = load_light(lights, li);
        // Sphere-vs-AABB: closest point on the AABB to the light centre within radius?
        float3 closest = clamp(L.position, aabb_min, aabb_max);
        float3 d = closest - L.position;
        if (dot(d, d) <= L.radius * L.radius)
        {
            if (count < MAX_LIGHTS_PER_CLUSTER)
            {
                index_list.Store((base + count) * 4, li);
                count += 1;
            }
        }
    }
    grid.Store(cluster * 4, count);
}
