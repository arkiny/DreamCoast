// Phase 11 Stage C7b — lit-color history capture.
//
// Copies this frame's deferred lighting result (the HDR target) into a per-pixel storage
// buffer so the NEXT frame's screen-space reflections can sample the previous frame's lit
// color (reprojected) instead of the current HDR — breaking the read-before-write cycle
// when SSR feeds back into the lighting specular (C7c).
//
// The lighting pass bakes the camera exposure into the HDR (`color = (ambient+lo)*exposure`),
// but the reflection composite that replaces the IBL prefilter-cube specular works in RAW
// radiance (the prefilter cube was raw too, and `pbr.slang` re-applies exposure once). So
// this divides the HDR back out by exposure (`inv_exposure`) and stores raw radiance.
//
// History lives in a byte-address storage buffer (float4 per pixel, rgb = raw radiance,
// a = 1), ping-ponged by the host so the SSR read and this write target different buffers.

#include "bindless.slang"

struct LitHistoryPush
{
    uint hdr_index;     // sampled: this frame's lit HDR (exposure baked in)
    uint out_buffer;    // storage buffer: raw-radiance history (this frame's write)
    uint width;
    uint height;
    float inv_exposure; // 1 / camera exposure (recovers raw radiance)
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]]
ConstantBuffer<LitHistoryPush> pc : register(b0, space0);

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    uint idx = tid.y * pc.width + tid.x;
    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float3 rgb = g.textures[pc.hdr_index].SampleLevel(g.samp, uv, 0.0).rgb * pc.inv_exposure;
    g.storage_buffers[pc.out_buffer].Store4(idx * 16u, asuint(float4(rgb, 1.0)));
}
