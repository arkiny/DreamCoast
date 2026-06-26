// Phase 11 Stage C8a — per-voxel albedo bake (compute).
//
// The companion to `sdf_bake.slang`: one thread per voxel of the scene's distance-field
// grid, but instead of the signed distance it writes the *albedo* of the nearest triangle
// into three R32Float storage volumes (one per RGB channel). The distance bake is left
// untouched (a separate pass) so the C1/C2/C3 distance field stays bit-identical; this
// only adds color.
//
// Each fused triangle carries its source object's representative linear albedo in a
// parallel storage buffer (`albedo_index`, 12 bytes / triangle = float3). The nearest-
// triangle search reuses the same closest-point-on-triangle test as `sdf_bake.slang`; the
// winner's albedo is stored. The C3 GI and C6 GDF-reflection passes then trilinear-sample
// these volumes at a ray hit to re-light with the real surface color (colored bleed +
// colored reflections) instead of a single constant albedo.
//
// Brute force O(voxels * triangles), one-time bake (same budget as the distance bake).

#include "bindless.slang"

struct AlbedoBakePush
{
    uint vol_storage_r; // storage_volumes[] index for the red channel
    uint vol_storage_g; // green
    uint vol_storage_b; // blue
    uint dim;           // volume edge length in voxels (cube)
    uint tri_count;     // number of triangles (indices / 3)
    uint vtx_index;     // bindless storage-buffer index of the 32B/vertex buffer
    uint idx_index;     // bindless storage-buffer index of the u32 index buffer
    uint albedo_index;  // bindless storage-buffer index of the 12B/triangle albedo buffer
    float4 aabb_min;    // xyz world-space min of the volume's extent
    float4 aabb_max;    // xyz world-space max
};

[[vk::push_constant]]
ConstantBuffer<AlbedoBakePush> pc : register(b0, space0);

float3 load_position(uint vertex)
{
    return asfloat(g.storage_buffers[pc.vtx_index].Load3(vertex * 32u));
}
uint load_index(uint i)
{
    return g.storage_buffers[pc.idx_index].Load(i * 4u);
}
// The triangle's source-object albedo (linear RGB), 12 bytes per triangle.
float3 load_tri_albedo(uint tri)
{
    return asfloat(g.storage_buffers[pc.albedo_index].Load3(tri * 12u));
}

// Closest point on triangle abc to p (Ericson, Real-Time Collision Detection) — identical
// to `sdf_bake.slang` so the nearest-triangle winner matches the distance field's.
float3 closest_on_triangle(float3 p, float3 a, float3 b, float3 c)
{
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0)
        return a;
    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3)
        return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        float v = d1 / (d1 - d3);
        return a + v * ab;
    }
    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6)
        return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        float w = d2 / (d2 - d6);
        return a + w * ac;
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }
    float denom = 1.0 / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void albedoBakeMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dim || tid.y >= pc.dim || tid.z >= pc.dim)
        return;

    float3 t = (float3(tid) + 0.5) / float(pc.dim);
    float3 p = lerp(pc.aabb_min.xyz, pc.aabb_max.xyz, t);

    float best = 1e30;
    uint best_tri = 0;
    for (uint tri = 0; tri < pc.tri_count; tri++)
    {
        float3 a = load_position(load_index(tri * 3u + 0u));
        float3 b = load_position(load_index(tri * 3u + 1u));
        float3 c = load_position(load_index(tri * 3u + 2u));
        float3 q = closest_on_triangle(p, a, b, c);
        float d = length(p - q);
        if (d < best)
        {
            best = d;
            best_tri = tri;
        }
    }

    float3 albedo = pc.tri_count > 0u ? load_tri_albedo(best_tri) : float3(0.7, 0.7, 0.7);
    g.storage_volumes[pc.vol_storage_r][tid] = albedo.r;
    g.storage_volumes[pc.vol_storage_g][tid] = albedo.g;
    g.storage_volumes[pc.vol_storage_b][tid] = albedo.b;
}
