// Procedural sky rendered into the six faces of the environment cubemap (one
// fullscreen draw per face). For each face texel we reconstruct the world-space
// view direction and evaluate a simple gradient sky + sun disk. Output is linear
// HDR so the IBL convolution (irradiance/prefilter) picks up the sun's energy.
//
// This is the swappable piece: replacing `procedural_sky` with a physically based
// atmospheric model later keeps the cubemap-capture + IBL pipeline unchanged.

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    float4 sun;   // xyz = direction TO the sun (normalized), w = intensity
    uint face;    // cube face index 0..5 (+X,-X,+Y,-Y,+Z,-Z)
    uint flip_y;  // 0 = D3D12 (Y up), 1 = Vulkan (Y down)
    uint2 pad;
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

// Gradient sky (zenith -> horizon -> ground) plus an HDR sun disk. Returns linear
// radiance.
float3 procedural_sky(float3 dir, float3 sun_dir, float sun_intensity)
{
    float up = dir.y;
    float3 zenith = float3(0.04, 0.16, 0.55);
    float3 horizon = float3(0.50, 0.66, 0.92);
    float3 ground = float3(0.16, 0.15, 0.14);

    float3 col;
    if (up >= 0.0)
        col = lerp(horizon, zenith, pow(saturate(up), 0.45));
    else
        col = lerp(horizon, ground, saturate(-up * 6.0));

    // Sun: a sharp HDR disk plus a soft glow around it.
    float c = saturate(dot(dir, sun_dir));
    float disk = pow(c, 4000.0) * 6.0;
    float glow = pow(c, 32.0) * 0.4;
    col += sun_intensity * (disk + glow);
    return col;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float3 dir = cube_direction(pc.face, input.uv);
    float3 col = procedural_sky(dir, normalize(pc.sun.xyz), pc.sun.w);
    return float4(col, 1.0);
}
