// Shared clustered-lighting constants + light record layout (PR-6).
// Included by both the culling compute (light_cluster.slang) and the deferred
// lighting fragment (pbr.slang) so the froxel grid dimensions, the per-cluster list
// capacity, and the packed Light byte layout are a SINGLE SOURCE OF TRUTH (drift
// between producer and consumer would silently light the wrong lights).
//
// The grid dimensions double as the scalability seam: bump them (and the matching
// Rust CLUSTER_* consts + the buffer sizing) for a higher RenderQuality tier. 16x9
// tiles matches a 16:9 aspect; 24 exponential Z slices is the DOOM-2016-era default.

#ifndef LIGHT_CLUSTER_COMMON_INCLUDED
#define LIGHT_CLUSTER_COMMON_INCLUDED

// Froxel grid. Keep in lockstep with `CLUSTER_X/Y/Z` in main.rs.
#define CLUSTER_X 16u
#define CLUSTER_Y 9u
#define CLUSTER_Z 24u
#define CLUSTER_COUNT (CLUSTER_X * CLUSTER_Y * CLUSTER_Z)

// Max lights binned into one cluster. Overflow lights past this are dropped for that
// cluster (a bounded, alloc-free flat index list — no per-cluster atomic reservation).
#define MAX_LIGHTS_PER_CLUSTER 128u

// csBuildClusters thread-group size (one thread per cluster).
#define CLUSTER_BUILD_GROUP 64

// A point light, packed as two float4 (32 bytes) in the light storage buffer:
//   [0..16)  position.xyz, radius (influence cutoff, world units; <= 0 = no cutoff)
//   [16..32) color.rgb,    intensity (candela — matches pbr's photometric point path)
struct Light
{
    float3 position;
    float radius;
    float3 color;
    float intensity;
};

// Point-light distance attenuation — the SINGLE SOURCE for every point-light consumer
// (the brute-force `globals.point_pos[]` UBO loop, the clustered froxel loop, and the
// clustered brute-force A/B). A light moved between those paths must shade identically,
// which is only guaranteed if they all call this.
//
//  - `radius <= 0` => pure inverse-square `1/d²`, bit-for-bit the pre-range expression.
//    That is the default every authored light carries, so the golden anchors are exactly
//    the historical image.
//  - `radius > 0`  => the same inverse-square, windowed by `(1 - (d/r)^4)^2` so the
//    contribution reaches EXACTLY zero at `d = r` with a smooth (C1) approach — no visible
//    cutoff edge. This is what makes clustered culling sound: a froxel that fails the
//    sphere-vs-AABB test contains only pixels with `d > r`, whose contribution is exactly
//    +0.0, so culling the light out of that froxel is not an approximation.
float point_attenuation(float dist, float radius)
{
    float atten = 1.0 / max(dist * dist, 1e-4);
    if (radius > 0.0)
    {
        float t2 = saturate(dist / radius);
        t2 = t2 * t2;
        float w = saturate(1.0 - t2 * t2);
        atten *= w * w;
    }
    return atten;
}

// Does this light's sphere of influence overlap a cluster's world-space AABB? The froxel
// binning's only culling test — kept here beside `point_attenuation` because the two are a
// contract: this may only reject a cluster when `point_attenuation` is exactly 0 for every
// point in it. A light with no authored cutoff (`radius <= 0`) influences every pixel, so
// it is binned into every cluster (correct, but no culling payoff — author a range).
bool light_overlaps_aabb(Light L, float3 aabb_min, float3 aabb_max)
{
    if (L.radius <= 0.0)
        return true;
    float3 closest = clamp(L.position, aabb_min, aabb_max);
    float3 d = closest - L.position;
    return dot(d, d) <= L.radius * L.radius;
}

Light load_light(RWByteAddressBuffer buf, uint i)
{
    uint4 a = buf.Load4(i * 32 + 0);
    uint4 b = buf.Load4(i * 32 + 16);
    Light L;
    L.position = float3(asfloat(a.x), asfloat(a.y), asfloat(a.z));
    L.radius = asfloat(a.w);
    L.color = float3(asfloat(b.x), asfloat(b.y), asfloat(b.z));
    L.intensity = asfloat(b.w);
    return L;
}

// The cluster index (cx, cy, cz -> flat) for a screen pixel + positive linear view
// depth. `view_z` is the distance in front of the camera (>0). Matches the encode in
// csBuildClusters (x fastest, then y, then z).
uint cluster_index_for(float2 uv, float view_z, float z_near, float z_far)
{
    uint cx = min(uint(uv.x * float(CLUSTER_X)), CLUSTER_X - 1u);
    uint cy = min(uint(uv.y * float(CLUSTER_Y)), CLUSTER_Y - 1u);
    // Inverse of Z(slice) = near * (far/near)^(slice/num_z):
    //   slice = log(z/near) / log(far/near) * num_z
    float lz = log(max(view_z, z_near) / z_near) / log(z_far / z_near);
    uint cz = min(uint(lz * float(CLUSTER_Z)), CLUSTER_Z - 1u);
    return cx + cy * CLUSTER_X + cz * CLUSTER_X * CLUSTER_Y;
}

#endif // LIGHT_CLUSTER_COMMON_INCLUDED
