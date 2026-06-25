// Phase 11 Stage B (B2) — per-mesh signed-distance-field bake (compute).
//
// One thread per voxel of a 3D storage volume. For each voxel centre (mapped from
// the volume's world-space AABB) it brute-forces the minimum distance to every
// triangle of a mesh (closest-point-on-triangle, Ericson), and signs it by the
// closest triangle's stored vertex normals (negative inside). The mesh vertices/
// indices are read from bindless storage buffers — the same 32-byte vertex stride
// (position at byte 0, normal at byte 12) the rasterizer and the HW path tracer use,
// so one geometry upload feeds all paths. The baked volume is then trilinear-sampled
// by the SW ray marcher (B4).
//
// Brute force is O(voxels * triangles): correct and simple, intended as a one-time
// bake (jump flooding is a later optimization). The stored-normal sign is exact for
// convex meshes (the B2 verification sphere) and winding-independent; a robust
// angle-weighted pseudonormal / winding number for concave meshes is a refinement.

#include "bindless.slang"

struct BakePush
{
    uint vol_storage; // storage_volumes[] index to write
    uint dim;         // volume edge length in voxels (cube)
    uint tri_count;   // number of triangles (indices / 3)
    uint vtx_index;   // bindless storage-buffer index of the 32B/vertex buffer
    uint idx_index;   // bindless storage-buffer index of the u32 index buffer
    uint pad0;
    float4 aabb_min;  // xyz world-space min of the volume's extent
    float4 aabb_max;  // xyz world-space max
};

[[vk::push_constant]]
ConstantBuffer<BakePush> pc : register(b0, space0);

// Vertex position from the 32-byte vertex record (position is the first float3).
float3 load_position(uint vertex)
{
    return asfloat(g.storage_buffers[pc.vtx_index].Load3(vertex * 32u));
}
// Vertex normal from the 32-byte record (normal follows position at byte 12).
float3 load_normal(uint vertex)
{
    return asfloat(g.storage_buffers[pc.vtx_index].Load3(vertex * 32u + 12u));
}
uint load_index(uint i)
{
    return g.storage_buffers[pc.idx_index].Load(i * 4u);
}

// Closest point on triangle abc to p (Ericson, Real-Time Collision Detection).
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
void bakeMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dim || tid.y >= pc.dim || tid.z >= pc.dim)
        return;

    float3 t = (float3(tid) + 0.5) / float(pc.dim);
    float3 p = lerp(pc.aabb_min.xyz, pc.aabb_max.xyz, t);

    float best = 1e30;
    float sign_d = 1.0;
    for (uint tri = 0; tri < pc.tri_count; tri++)
    {
        uint i0 = load_index(tri * 3u + 0u);
        uint i1 = load_index(tri * 3u + 1u);
        uint i2 = load_index(tri * 3u + 2u);
        float3 a = load_position(i0);
        float3 b = load_position(i1);
        float3 c = load_position(i2);
        float3 q = closest_on_triangle(p, a, b, c);
        float d = length(p - q);
        if (d < best)
        {
            best = d;
            // Sign by the closest triangle's geometric normal. Use the mesh's stored
            // (outward) vertex normals rather than cross(b-a, c-a): the latter depends
            // on triangle winding (inward for this uv-sphere) and is undefined-direction
            // when the closest point lies on a shared edge/vertex. The face-average
            // vertex normal is reliably outward, so dot(p-q, n) < 0 ⇒ inside (negative).
            float3 n = load_normal(i0) + load_normal(i1) + load_normal(i2);
            sign_d = dot(p - q, n) < 0.0 ? -1.0 : 1.0;
        }
    }

    g.storage_volumes[pc.vol_storage][tid] = best * sign_d;
}
