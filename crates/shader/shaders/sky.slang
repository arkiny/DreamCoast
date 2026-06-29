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

// Physically-based daylight sky — Preetham et al., "A Practical Analytic Model for
// Daylight" (SIGGRAPH 1999). The Perez luminance/chromaticity distribution fitted to
// measured CIE clear-sky data, parameterised by turbidity. This replaces the ad-hoc
// blue gradient: the integrated sky irradiance becomes a balanced daylight white (warm
// near the horizon, blue at the zenith) instead of an over-saturated blue, so a sun-
// shadowed interior is no longer flooded with unphysical blue. Output is linear HDR.
static const float SKY_PI = 3.14159265359;

// Perez F(theta, gamma) = (1 + A e^{B/cos theta})(1 + C e^{D gamma} + E cos^2 gamma).
float perez(float cos_theta, float gamma, float A, float B, float C, float D, float E)
{
    float ct = max(cos_theta, 0.01); // clamp at the horizon (1/cos blows up)
    return (1.0 + A * exp(B / ct)) * (1.0 + C * exp(D * gamma) + E * cos(gamma) * cos(gamma));
}

// xyY -> linear sRGB.
float3 xyY_to_rgb(float x, float y, float Y)
{
    float yy = max(y, 1e-4);
    float X = x / yy * Y;
    float Z = (1.0 - x - y) / yy * Y;
    float3 rgb = float3(3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z,
                        -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z,
                        0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z);
    return max(rgb, float3(0.0, 0.0, 0.0));
}

float3 procedural_sky(float3 dir, float3 sun_dir, float sun_intensity)
{
    const float T = 2.5; // turbidity: ~2-3 = clear sky
    float theta_s = acos(clamp(sun_dir.y, 0.0, 1.0)); // sun zenith angle
    float cos_theta = dir.y;                          // view: cos(zenith)
    float cos_gamma = clamp(dot(dir, sun_dir), -1.0, 1.0);
    float gamma = acos(cos_gamma); // angle between view and sun

    // Perez distribution coefficients (linear in turbidity), for Y / x / y.
    float AY = 0.1787 * T - 1.4630, BY = -0.3554 * T + 0.4275, CY = -0.0227 * T + 5.3251,
          DY = 0.1206 * T - 2.5771, EY = -0.0670 * T + 0.3703;
    float Ax = -0.0193 * T - 0.2592, Bx = -0.0665 * T + 0.0008, Cx = -0.0004 * T + 0.2125,
          Dx = -0.0641 * T - 0.8989, Ex = -0.0033 * T + 0.0452;
    float Ay = -0.0167 * T - 0.2608, By = -0.0950 * T + 0.0092, Cy = -0.0079 * T + 0.2102,
          Dy = -0.0441 * T - 1.6537, Ey = -0.0109 * T + 0.0529;

    // Zenith chromaticity (polynomial in turbidity + sun zenith angle).
    float ts = theta_s, ts2 = ts * ts, ts3 = ts2 * ts, T2 = T * T;
    float xz = T2 * (0.00166 * ts3 - 0.00375 * ts2 + 0.00209 * ts)
               + T * (-0.02903 * ts3 + 0.06377 * ts2 - 0.03202 * ts + 0.00394)
               + (0.11693 * ts3 - 0.21196 * ts2 + 0.06052 * ts + 0.25886);
    float yz = T2 * (0.00275 * ts3 - 0.00610 * ts2 + 0.00317 * ts)
               + T * (-0.04214 * ts3 + 0.08970 * ts2 - 0.04153 * ts + 0.00516)
               + (0.15346 * ts3 - 0.26756 * ts2 + 0.06670 * ts + 0.26688);
    // Zenith luminance (kcd/m^2); used relatively, scaled to the engine range below.
    float chi = (4.0 / 9.0 - T / 120.0) * (SKY_PI - 2.0 * theta_s);
    float Yz = (4.0453 * T - 4.9710) * tan(chi) - 0.2155 * T + 2.4192;

    // Sky = zenith value * F(view) / F(zenith,sun). Below the horizon fades to a dim
    // neutral ground so the env cube's lower hemisphere isn't black/negative.
    float denomY = perez(cos(theta_s), 0.0, AY, BY, CY, DY, EY);
    float fx = perez(cos_theta, gamma, Ax, Bx, Cx, Dx, Ex)
               / perez(cos(theta_s), 0.0, Ax, Bx, Cx, Dx, Ex);
    float fy = perez(cos_theta, gamma, Ay, By, Cy, Dy, Ey)
               / perez(cos(theta_s), 0.0, Ay, By, Cy, Dy, Ey);
    float fY = perez(cos_theta, gamma, AY, BY, CY, DY, EY) / max(denomY, 1e-3);
    float x = xz * fx;
    float y = yz * fy;
    float Y = max(Yz, 0.0) * fY;

    // Map kcd/m^2 luminance into the engine's linear-radiance range (calibrated so the
    // overall scene brightness stays in the same ballpark as before).
    const float SKY_SCALE = 0.05;
    float3 col = xyY_to_rgb(x, y, Y) * SKY_SCALE;
    if (cos_theta < 0.0)
        col = lerp(col, float3(0.16, 0.15, 0.14) * 0.3, saturate(-cos_theta * 4.0));

    // Sun: a sharp HDR disk plus a soft glow (the analytic sky already brightens near the
    // sun via the Perez term; this adds the bright disk the IBL convolution needs).
    float c = saturate(cos_gamma);
    col += sun_intensity * (pow(c, 4000.0) * 6.0 + pow(c, 32.0) * 0.4);
    return max(col, float3(0.0, 0.0, 0.0));
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float3 dir = cube_direction(pc.face, input.uv);
    float3 col = procedural_sky(dir, normalize(pc.sun.xyz), pc.sun.w);
    return float4(col, 1.0);
}
