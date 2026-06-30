// Procedural sky rendered into the six faces of the environment cubemap (one
// fullscreen draw per face). For each face texel we reconstruct the world-space
// view direction and evaluate a simple gradient sky + sun disk. Output is linear
// HDR so the IBL convolution (irradiance/prefilter) picks up the sun's energy.
//
// The atmosphere model itself lives in `sky_common.slang`, shared with the path
// tracer's miss shader so both renderers evaluate the SAME procedural sky.
#include "sky_common.slang"

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    float4 sun;     // xyz = direction TO the sun (normalized), w = intensity
    uint face;      // cube face index 0..5 (+X,-X,+Y,-Y,+Z,-Z)
    uint flip_y;    // 0 = D3D12 (Y up), 1 = Vulkan (Y down)
    float sky_gain; // inscatter→radiance gain (sun:sky ratio knob; legacy 6.0)
    uint pad;
    // Sky white balance: a per-channel gain on the sky radiance (xyz; w unused). The captured
    // env cube feeds the visible sky + the irradiance/prefilter cubes (so the IBL diffuse/specular
    // ambient and, via IBL, the SW-RT GI ambient), so this neutralises/warms the whole sky-sourced
    // ambient in one place while leaving the DIRECT sun untinted. `(1,1,1)` = neutral, a no-op.
    float4 wb;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[shader("vertex")]
VSOutput vsMain(uint id: SV_VertexID)
{
    float2 uv = float2((id << 1) & 2, id & 2);
    float s = pc.flip_y != 0 ? 1.0 : -1.0;

    VSOutput o;
    o.uv = uv;
    o.pos = float4(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * s, 0.0, 1.0);
    return o;
}

// World direction for a cube face texel, matching the D3D/Vulkan `TextureCube`
// sampling convention (face texel coords: u right, v down). `uv` has v=0 at top.
float3 cube_direction(uint face, float2 uv)
{
    float u = uv.x * 2.0 - 1.0;
    float v = uv.y * 2.0 - 1.0;
    float3 dir;
    switch (face)
    {
    case 0: dir = float3(1.0, -v, -u); break;  // +X
    case 1: dir = float3(-1.0, -v, u); break;  // -X
    case 2: dir = float3(u, 1.0, v); break;    // +Y
    case 3: dir = float3(u, -1.0, -v); break;  // -Y
    case 4: dir = float3(u, -v, 1.0); break;   // +Z
    default: dir = float3(-u, -v, -1.0); break; // -Z
    }
    return normalize(dir);
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float3 dir = cube_direction(pc.face, input.uv);
    // `with_sun = true`: the env capture bakes the sun disk into the cube (it feeds the
    // reflection/IBL specular highlight). The shared model is byte-identical to the prior
    // in-file `procedural_sky` for this call, so the capture is unchanged.
    float3 col = procedural_sky(dir, normalize(pc.sun.xyz), pc.sun.w, pc.sky_gain, true);
    // White balance the sky radiance (neutral (1,1,1) = exact no-op → byte-identical capture).
    col *= pc.wb.rgb;
    return float4(col, 1.0);
}
