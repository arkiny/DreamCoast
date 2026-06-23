// GPU particle draw (Phase 7): instanced billboard quads, one instance per
// particle. The vertex stage pulls each particle's position/life from the same
// storage buffer the sim compute pass wrote (vertex-stage UAV read), builds a
// camera-facing quad, and the fragment stage draws a soft alpha-blended sprite
// tinted by remaining life. Drawn after tonemap so particles composite on top.

struct PushConstants
{
    float4x4 view_proj; // 64
    float4 cam_right;   // world-space camera right (xyz)
    float4 cam_up;      // world-space camera up (xyz)
    uint buffer_index;  // bindless storage-buffer index of the particle array
    uint count;
    float size;         // billboard half-extent (world units)
    uint pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

#include "bindless.slang"

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
    float life : TEXCOORD1;
};

static const float2 CORNERS[6] = {
    float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0),
    float2(-1.0, 1.0), float2(1.0, -1.0), float2(1.0, 1.0)
};

[shader("vertex")]
VSOutput vsMain(uint vid: SV_VertexID, uint iid: SV_InstanceID)
{
    uint base = iid * 32;
    float3 ppos = g.storage_buffers[pc.buffer_index].Load<float3>(base + 0);
    float life = g.storage_buffers[pc.buffer_index].Load<float>(base + 12);

    float2 c = CORNERS[vid];
    // Dead particles collapse to a point (invisible).
    float s = life > 0.0 ? pc.size : 0.0;
    float3 world = ppos + (c.x * pc.cam_right.xyz + c.y * pc.cam_up.xyz) * s;

    VSOutput o;
    o.pos = mul(pc.view_proj, float4(world, 1.0));
    o.uv = c;
    o.life = life;
    return o;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float d = length(input.uv);
    if (d > 1.0)
        discard;
    float falloff = saturate(1.0 - d);
    float a = falloff * falloff * saturate(input.life);
    // Hot core (yellow) fading to ember (orange-red) as life runs out.
    float3 color = lerp(float3(1.0, 0.35, 0.1), float3(1.0, 0.9, 0.5), saturate(input.life));
    return float4(color, a);
}
