// GPU culling draw (Phase 7): the indirect, instanced render of the visible cube
// grid. The vertex stage reads its instance index from the compute-written
// visible-list buffer (SV_InstanceID -> visible[id] -> grid cell), offsets the
// cube vertices, and the fragment stage applies simple directional shading. The
// instance count comes from the indirect args buffer, so only visible cubes draw.

struct PushConstants
{
    float4x4 view_proj; // 64
    float4 sun_dir;     // xyz = direction to the sun
    uint visible_index; // bindless index of the visible-list buffer
    uint grid_dim;
    float spacing;
    float cube_scale;
    float y_height;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

#include "bindless.slang"

struct VSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
};

struct VSOutput
{
    float4 pos : SV_Position;
    float3 normal : TEXCOORD0;
    float3 tint : TEXCOORD1;
};

[shader("vertex")]
VSOutput vsMain(VSInput input, uint iid: SV_InstanceID)
{
    uint inst = g.storage_buffers[pc.visible_index].Load<uint>(iid * 4);

    uint gx = inst % pc.grid_dim;
    uint gz = inst / pc.grid_dim;
    float half = (float(pc.grid_dim) - 1.0) * 0.5;
    float3 center = float3((float(gx) - half) * pc.spacing, pc.y_height,
                           (float(gz) - half) * pc.spacing);

    float3 world = center + input.position * pc.cube_scale;

    VSOutput o;
    o.pos = mul(pc.view_proj, float4(world, 1.0));
    o.normal = input.normal;
    // Checker tint by grid parity so individual instances are distinguishable.
    float c = ((gx + gz) & 1) ? 1.0 : 0.55;
    o.tint = float3(0.35, 0.55, 0.95) * c;
    return o;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float3 n = normalize(input.normal);
    float ndl = saturate(dot(n, normalize(pc.sun_dir.xyz)));
    float3 color = input.tint * (0.25 + 0.75 * ndl);
    // UNORM backbuffer — encode sRGB (see linear_to_srgb in bindless.slang).
    return float4(linear_to_srgb(color), 1.0);
}
