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

// Physically-based atmosphere — single-scattering ray-march (Rayleigh + Mie + ozone) in
// the spirit of Bruneton 2008 / Hillaire 2020 (UE5 Sky Atmosphere). Earth parameters; the
// view ray is integrated through the atmosphere with a per-sample sun-transmittance march,
// so the sky is physically derived from sun elevation: blue zenith, brighter/whiter horizon,
// reddening as the sun nears the horizon. The env cube is generated infrequently (startup /
// sun change), so the nested march is affordable without the full per-frame LUT pipeline.
static const float ATM_PI = 3.14159265359;
static const float PLANET_R = 6360.0; // km
static const float ATMOS_R = 6460.0;  // km (100 km atmosphere)
static const float3 RAY_BETA = float3(5.802, 13.558, 33.1) * 1e-3; // Rayleigh scattering /km (RGB)
static const float MIE_BETA = 3.996e-3;   // Mie scattering /km
static const float MIE_EXT = 4.44e-3;     // Mie extinction /km (scatter + absorb)
static const float3 OZONE_BETA = float3(0.650, 1.881, 0.085) * 1e-3; // ozone absorption /km
static const float H_RAY = 8.0; // Rayleigh scale height km
static const float H_MIE = 1.2; // Mie scale height km
static const float MIE_G = 0.8; // Mie phase asymmetry

// (rayleigh, mie, ozone) density at altitude h (km). Ozone is a tent peaking ~25 km.
float3 atm_density(float h)
{
    return float3(exp(-h / H_RAY), exp(-h / H_MIE), max(0.0, 1.0 - abs(h - 25.0) / 15.0));
}

// Ray-sphere: returns (near, far) t; (-1,-1) on miss. Origin may be inside (near < 0).
float2 atm_ray_sphere(float3 ro, float3 rd, float radius)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0)
        return float2(-1.0, -1.0);
    float s = sqrt(disc);
    return float2(-b - s, -b + s);
}

// Transmittance from p toward `dir` to the atmosphere edge (8-tap optical-depth march).
float3 atm_transmittance(float3 p, float3 dir)
{
    float t_top = atm_ray_sphere(p, dir, ATMOS_R).y;
    if (t_top <= 0.0)
        return float3(1.0, 1.0, 1.0);
    const int N = 8;
    float dt = t_top / float(N);
    float3 od = float3(0.0, 0.0, 0.0); // (rayleigh, mie, ozone) optical-depth accumulators
    for (int i = 0; i < N; i++)
    {
        float h = length(p + dir * (float(i) + 0.5) * dt) - PLANET_R;
        od += atm_density(h) * dt;
    }
    float3 tau = RAY_BETA * od.x + MIE_EXT * od.y + OZONE_BETA * od.z;
    return exp(-tau);
}

float3 procedural_sky(float3 dir, float3 sun_dir, float sun_intensity, float sky_gain)
{
    float3 ro = float3(0.0, PLANET_R + 0.5, 0.0); // viewer ~500 m above the surface
    float t_top = atm_ray_sphere(ro, dir, ATMOS_R).y;
    float2 gnd = atm_ray_sphere(ro, dir, PLANET_R);
    float t_max = (gnd.x > 0.0) ? gnd.x : t_top; // stop at the ground if the ray hits it

    float cos_t = clamp(dot(dir, sun_dir), -1.0, 1.0);
    float phaseR = 3.0 / (16.0 * ATM_PI) * (1.0 + cos_t * cos_t);
    float g = MIE_G;
    float phaseM = (1.0 - g * g)
                   / (4.0 * ATM_PI * pow(1.0 + g * g - 2.0 * g * cos_t, 1.5));

    const int N = 24;
    float dt = max(t_max, 0.0) / float(N);
    float3 inscatter = float3(0.0, 0.0, 0.0);
    float3 view_od = float3(0.0, 0.0, 0.0); // optical depth camera -> sample
    for (int i = 0; i < N; i++)
    {
        float3 sp = ro + dir * (float(i) + 0.5) * dt;
        float h = length(sp) - PLANET_R;
        float3 d = atm_density(h) * dt;
        view_od += d;
        float3 view_tr = exp(-(RAY_BETA * view_od.x + MIE_EXT * view_od.y + OZONE_BETA * view_od.z));
        float3 sun_tr = atm_transmittance(sp, sun_dir);
        inscatter += (RAY_BETA * d.x * phaseR + MIE_BETA * d.y * phaseM) * sun_tr * view_tr;
    }
    // Sun illuminance scale: the engine's sun_intensity drives the whole atmosphere; `sky_gain`
    // maps the per-km scattering integral into radiance and sets the sun:sky illuminance ratio
    // (legacy 6.0 = the gallery anchor; content lowers it so the direct sun dominates the sky as
    // it does physically, with interiors filled by multibounce GI rather than a flat sky ambient).
    float3 col = inscatter * sun_intensity * sky_gain;

    // Sharp HDR sun disk (the IBL convolution turns it into the directional highlight); only
    // when the view ray actually reaches the sky (not blocked by the ground).
    if (gnd.x <= 0.0)
    {
        float c = saturate(cos_t);
        col += atm_transmittance(ro, sun_dir) * sun_intensity * pow(c, 4000.0) * 8.0;
    }
    // The env cube is Rgba16Float (max ~65504). With the sun authored in lux (~1e5), the baked
    // sun disk would overflow to +inf and poison the IBL convolution. Clamp to a safe HDR ceiling:
    // the env disk only feeds reflections/irradiance, while the analytic directional light carries
    // the real sun energy in the lighting pass. Gallery intensities (≤~24 here) are far below this,
    // so its env is unchanged (byte-identical anchor).
    return clamp(col, float3(0.0, 0.0, 0.0), float3(60000.0, 60000.0, 60000.0));
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float3 dir = cube_direction(pc.face, input.uv);
    float3 col = procedural_sky(dir, normalize(pc.sun.xyz), pc.sun.w, pc.sky_gain);
    // White balance the sky radiance (neutral (1,1,1) = exact no-op → byte-identical capture).
    col *= pc.wb.rgb;
    return float4(col, 1.0);
}
