// Inline ray-query TLAS-trace validation (Phase 8 M3).
//
// A compute shader that generates a primary camera ray per pixel, traces it
// against the scene TLAS with an inline `RayQuery` (DXR 1.1 / VK_KHR_ray_query),
// and writes a validation image into a bindless storage image:
//   * miss        -> sky color
//   * primary hit -> per-instance color (from the instance custom index),
//                    modulated by a hardware-traced shadow ray toward the sun.
//
// This isolates the trace path (camera ray gen + TLAS traversal + hit data +
// a second shadow ray) before the full path tracer (M4). No G-buffer dependency:
// rays are generated from the inverse view-projection, so hit positions are in
// true world space (matching the TLAS instance transforms).

#include "bindless.slang"

struct PushConstants
{
    float4x4 inv_view_proj; // clip -> world (unproject pixel to a world ray)
    float4 cam_pos;         // xyz camera world position
    float4 sun_dir;         // xyz normalized direction toward the sun
    uint out_index;         // bindless storage-image index of the output
    uint width;
    uint height;
    uint flip_y;            // 0 = D3D12 (Y up), 1 = Vulkan (Y down)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

// Stable pseudo-random color from an integer id (instance visualization).
float3 id_color(uint id)
{
    uint h = id * 2654435761u;
    float r = ((h >> 0) & 0xFFu) / 255.0;
    float g_ = ((h >> 8) & 0xFFu) / 255.0;
    float b = ((h >> 16) & 0xFFu) / 255.0;
    return float3(r, g_, b) * 0.6 + 0.2;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    // Pixel center -> NDC, using the engine's clip-space Y convention (matches the
    // deferred sky reconstruction in pbr.slang): Vulkan's projection is Y-flipped,
    // so the sign depends on the backend.
    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float sy = pc.flip_y != 0 ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 world = mul(pc.inv_view_proj, float4(ndc, 1.0, 1.0));
    float3 dir = normalize(world.xyz / world.w - pc.cam_pos.xyz);

    RayDesc ray;
    ray.Origin = pc.cam_pos.xyz;
    ray.Direction = dir;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(g.tlas, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    float3 color;
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        uint id = q.CommittedInstanceID();
        float t = q.CommittedRayT();
        float3 hit = ray.Origin + ray.Direction * t;

        // Shadow ray toward the sun; TMin lifts the origin off the surface to
        // avoid self-intersection acne.
        RayDesc sray;
        sray.Origin = hit;
        sray.Direction = normalize(pc.sun_dir.xyz);
        sray.TMin = 0.05;
        sray.TMax = 10000.0;
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
        sq.TraceRayInline(g.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, sray);
        sq.Proceed();
        float shadow = (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.3 : 1.0;

        color = id_color(id) * shadow;
    }
    else
    {
        color = float3(0.35, 0.5, 0.7); // sky
    }

    g.storage_images[pc.out_index][tid.xy] = float4(color, 1.0);
}
