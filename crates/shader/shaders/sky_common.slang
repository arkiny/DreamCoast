// Shared procedural sky — a single-source atmosphere model used by BOTH the
// rasterizer's env-cube capture (`sky.slang`) and the path tracer's miss shader
// (`rt_common.slang`). Authoring the sky once here keeps the two renderers on the
// SAME physically-based model, so the path-tracer parity residual measures the
// rasterizer's shading/GI/reflection approximation — not a sky-model mismatch.
//
// Physically-based atmosphere — single-scattering ray-march (Rayleigh + Mie + ozone)
// in the spirit of Bruneton 2008 / Hillaire 2020 (UE5 Sky Atmosphere). Earth
// parameters; the view ray is integrated through the atmosphere with a per-sample
// sun-transmittance march, so the sky is physically derived from sun elevation: blue
// zenith, brighter/whiter horizon, reddening as the sun nears the horizon.

#ifndef SKY_COMMON_INCLUDED
#define SKY_COMMON_INCLUDED

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

// Evaluate the procedural sky radiance along `dir`. `with_sun` adds the sharp HDR sun
// disk — enabled for the env capture and for path-tracer PRIMARY (camera) rays, but
// disabled for indirect/diffuse rays (the sun is added analytically via NEE there, so
// re-sampling the disk would double-count it).
float3 procedural_sky(float3 dir, float3 sun_dir, float sun_intensity, float sky_gain,
                      bool with_sun)
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
    // when the view ray actually reaches the sky (not blocked by the ground), and only when the
    // caller wants the disk (primary/camera rays, not indirect bounces).
    if (with_sun && gnd.x <= 0.0)
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

#endif // SKY_COMMON_INCLUDED
