// Phase 14 M5 — visibility-buffer visualization.
//
// A full-screen pass that reads the R64 visibility buffer the SW rasterizer wrote and colours
// each pixel by the cluster id it recorded (the same golden-ratio palette as the CPU/HW debug
// views), so the SW raster + `atomicMax` path can be verified visually before the M6 material
// resolve exists. Empty pixels (packed == 0) fall through to the background.
#include "bindless.slang"

struct PushConstants
{
    uint vis_buf; // R64 visibility buffer slot
    uint width;
    uint height;
    uint pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

struct VSOut
{
    float4 pos : SV_Position;
};

// Full-screen triangle from the vertex id — no vertex buffer.
[shader("vertex")]
VSOut vsMain(uint vid: SV_VertexID)
{
    VSOut o;
    float2 uv = float2((vid << 1) & 2, vid & 2);
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

float3 cluster_hue(uint id)
{
    float h = frac(float(id) * 0.61803398875);
    float3 k = frac(h + float3(0.0, 2.0 / 3.0, 1.0 / 3.0));
    float3 c = saturate(abs(k * 6.0 - 3.0) - 1.0);
    return lerp(float3(1.0, 1.0, 1.0), c, 0.72);
}

[shader("fragment")]
float4 fragMain(VSOut i): SV_Target
{
    uint x = (uint)i.pos.x;
    uint y = (uint)i.pos.y;
    if (x >= pc.width || y >= pc.height)
        return float4(0.08, 0.09, 0.12, 1.0);
    uint px = y * pc.width + x;
    uint64_t packed = g.storage_buffers[pc.vis_buf].Load<uint64_t>(px * 8);
    if (packed == 0)
        return float4(0.08, 0.09, 0.12, 1.0); // background (no triangle)
    uint payload = (uint)(packed & 0xFFFFFFFFu);
    uint cluster = payload >> 7;
    // Flat depth cue so the form reads: brighter for nearer (higher depthKey).
    uint depth_key = (uint)(packed >> 32);
    float shade = 0.35 + 0.65 * saturate(float(depth_key) / 4294967040.0);
    return float4(cluster_hue(cluster) * shade, 1.0);
}
