// Phase 11 Stage B (B1) — 3D volume texture RHI smoke test.
//
// Verifies the new bindless 3D-volume tables end to end: `fillMain` writes a signed
// radial distance field into a `storage_volumes[]` (RWTexture3D UAV) entry; `viewMain`
// trilinear-samples the same resource through `volumes[]` (sampled Texture3D SRV) at a
// chosen Z slice and writes it to the screen storage image. A fill→sample round trip
// over a real 3D texture, identical on both backends. Stage B2/B4 replace the radial
// fill with a real per-mesh SDF bake and the slice view with a volume ray march.

#include "bindless.slang"

struct VolPush
{
    uint vol_storage; // storage_volumes[] index (fill target, UAV)
    uint vol_sampled; // volumes[] index (sample source, SRV) — same resource
    uint dim;         // volume edge length in voxels (cube)
    uint out_index;   // screen storage-image index (view pass)
    uint width;       // screen width (view pass)
    uint height;      // screen height (view pass)
    float slice;      // Z slice to visualize, [0,1]
    uint pad;
};

[[vk::push_constant]]
ConstantBuffer<VolPush> pc : register(b0, space0);

// Fill: signed distance to a sphere centred in the unit-cube volume (radius 0.3),
// one thread per voxel. Distances are in volume-local units (cube spans [0,1]).
[shader("compute")]
[numthreads(4, 4, 4)]
void fillMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dim || tid.y >= pc.dim || tid.z >= pc.dim)
        return;
    float3 p = (float3(tid) + 0.5) / float(pc.dim); // voxel centre in [0,1]
    float d = length(p - float3(0.5, 0.5, 0.5)) - 0.3;
    g.storage_volumes[pc.vol_storage][tid] = d;
}

// View: trilinear-sample the volume at a fixed Z slice and map the signed distance to
// a simple grayscale (inside = bright, the zero isocontour = mid grey) so the round
// trip is visually obvious and pixel-comparable across backends.
[shader("compute")]
[numthreads(8, 8, 1)]
void viewMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;
    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float d = g.volumes[pc.vol_sampled].SampleLevel(g.samp, float3(uv, pc.slice), 0.0);
    // Signed distance -> grayscale: remap [-0.3, 0.3] to [1, 0] and tint the surface.
    float shade = saturate(0.5 - d / 0.6);
    float3 col = float3(shade, shade, shade);
    if (abs(d) < 0.01)
        col = float3(0.2, 0.9, 0.4); // highlight the zero isosurface (green ring)
    g.storage_images[pc.out_index][tid.xy] = float4(col, 1.0);
}
