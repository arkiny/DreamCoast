// PR-4 (render-pipeline re-baseline track) — sky/atmosphere composite slot.
//
// A full-screen pass sitting AFTER opaque lighting + reflections are fully composited
// (the `hdr` scene color is final) and BEFORE the (future) transparency pass / post
// chain — see docs/render-pipeline-reference.md §1.6 (#10 Sky/Atmosphere, #11
// Height Fog) and §3 PR-4. The slot exists unconditionally in the frame graph
// wiring; today it hosts one opt-in feature (analytic exponential height fog,
// `P_HEIGHT_FOG=1`) and is the reserved landing spot for aerial perspective / light
// shafts / volumetric fog in a future phase — those extend `PushConstants` +
// `fsMain`, not the call site.
//
// Height fog: an exponential-falloff participating medium whose density decays with
// world-space height, integrated ANALYTICALLY along the camera ray (closed form, not
// ray-marched) — the canonical real-time technique (I. Quilez, "fog", 2010:
// https://iquilezles.org/articles/fog/; the same closed form used by the reference
// engine's exponential height fog). Applied in LINEAR HDR, before tonemapping (see
// §1.6 in the reference doc: fog composites onto the finished opaque HDR scene color,
// ahead of translucency and the tonemap/post chain).
//
// Derivation: density at world height y is d(y) = a·exp(-b·y) (a = density at y=0,
// b = 1/height-falloff). For a ray o + t·rd, the optical depth (integral of d along
// the ray from 0 to T) has a closed form:
//   b·rd.y != 0:  depth(T) = (a/b) * exp(-b·o.y) * (1 - exp(-b·rd.y·T)) / rd.y
//   b·rd.y  = 0:  depth(T) = a * exp(-b·o.y) * T                    (horizontal ray;
//                 the removable singularity's limit as rd.y -> 0, verified by direct
//                 integration: d(y) is then CONSTANT along the ray, so depth = d·T).
// Transmittance = exp(-depth); fog factor = 1 - transmittance (how much of the
// inscatter color replaces the surface color, Beer-Lambert). The inscatter color is
// NOT a new authored constant — it is the SAME single-source procedural sky
// (`sky_common.slang`'s `procedural_sky`) the env-cube capture and path tracer use,
// but sampled at a FIXED zenith direction (single source of truth for the fog's
// "which way is the sky" question), not toward each pixel's short indoor ray: a
// reference-engine height fog's inscatter is a stable ambient/directional color (see
// e.g. a commercial engine's "Fog Inscattering Color" + a separate sun-facing term),
// not the raw atmosphere integral re-evaluated per pixel — `procedural_sky` varies
// sharply near the horizon (the physically-correct bright horizon band an outdoor
// view ray would show), which looks like a banding artifact when re-sampled along
// short, steeply-angled indoor rays that never actually reach the sky.
//
// Background (sky) pixels are left untouched: they already show the full procedural
// sky (via the G-buffer position.w miss path in pbr.slang's fsMain), so compositing
// height fog on top of them would double the atmosphere. Height fog only affects
// pixels with real scene geometry (position.w >= 0.5).

#include "bindless.slang"
#include "sky_common.slang"

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

// Every row is an explicit float4 (xyz payload + a neighbour scalar in `.w`), never
// a bare `float3` followed by a `float` — a scalar packed right after a float3
// lands at +12 on HLSL/SPIR-V but pads to +16 on MSL (documented Metal push-constant
// gotcha; see push.rs's `gdf_gi_push` / `ground_albedo`), so this mirrors that fix.
struct PushConstants
{
    uint hdr_index;      // sampled index: the finished opaque HDR scene color (post-lighting/reflect)
    uint position_index; // sampled index: G-buffer world-position MRT (xyz = world pos, w = geometry flag)
    uint out_index;      // unused by the graphics entry (kept for layout symmetry with compute passes)
    uint flip_y;         // 0 = D3D12 (Y up), 1 = Vulkan (Y down) — unused (fsMain has no ray reconstruction)
    float4 camera_pos_density; // xyz = camera world position (ray origin); w = `a` density at height 0
    float4 sun_dir_intensity;  // xyz = normalized direction TOWARD the sun; w = sun intensity (procedural_sky)
    float4 sky_wb_gain;        // xyz = sky white-balance gain (matches sky.slang's `wb`); w = inscatter gain
    // x = height_falloff `b` (1/world-unit; 0 => uniform density); y = camera exposure (the SAME
    // scalar `record_lighting` bakes into `hdr` — `procedural_sky` returns RAW unexposed radiance,
    // like the sky-background miss path in pbr.slang, so it must be exposed before blending with
    // the already-exposed `hdr` or a physically-scaled sun (e.g. 100000 lux) blows out to white).
    float4 falloff_exposure_pad;
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

// Analytic optical depth of an exponential-falloff height fog along a ray of length
// `t_len`, starting at world height `ro_y` with direction `rd_y` (the ray's
// normalized y-component). See the file header for the derivation.
float height_fog_optical_depth(float ro_y, float rd_y, float t_len, float a, float b)
{
    float by = b * rd_y;
    // The horizontal-ray branch also covers b == 0 (uniform density: no height
    // falloff), where the general form's 1/by blows up but the density is constant
    // along the WHOLE ray regardless of rd_y.
    if (abs(by) < 1e-5)
    {
        return a * exp(-b * ro_y) * t_len;
    }
    return (a / b) * exp(-b * ro_y) * (1.0 - exp(-by * t_len)) / rd_y;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float4 hdr = g.textures[pc.hdr_index].Sample(g.samp, input.uv);
    float4 pos = g.textures[pc.position_index].Sample(g.samp, input.uv);

    // Background (sky) pixel: the procedural sky already carries the full
    // atmosphere (pbr.slang's miss path), so leave it untouched — no double fog.
    if (pos.w < 0.5)
    {
        return hdr;
    }

    float3 camera_pos = pc.camera_pos_density.xyz;
    float density = pc.camera_pos_density.w;
    float3 sun_dir = pc.sun_dir_intensity.xyz;
    float sun_intensity = pc.sun_dir_intensity.w;
    float3 sky_wb = pc.sky_wb_gain.xyz;
    float inscatter_gain = pc.sky_wb_gain.w;
    float height_falloff = pc.falloff_exposure_pad.x;
    float exposure = pc.falloff_exposure_pad.y;

    float3 world_pos = pos.xyz;
    float3 to_pixel = world_pos - camera_pos;
    float t_len = length(to_pixel);
    float3 rd = t_len > 1e-5 ? (to_pixel / t_len) : float3(0.0, 1.0, 0.0);

    float depth = height_fog_optical_depth(camera_pos.y, rd.y, t_len, density, height_falloff);
    float fog_factor = saturate(1.0 - exp(-max(depth, 0.0)));

    // Inscatter color: the same single-source procedural sky the env-cube capture and
    // path tracer use, sampled at a FIXED zenith direction (not `rd`, and not
    // `with_sun` — the fog is a low-frequency ambient scatter term, not a sharp
    // sun-disk source; matches the indirect-ray convention `sky_common.slang`
    // documents). A fixed direction gives a single, stable fog color per frame — the
    // same "fixed inscattering color" a reference engine's height fog uses — instead
    // of the raw atmosphere integral's sharp per-direction variation (bright horizon
    // band) leaking into interior geometry via short, steep indoor rays.
    float3 inscatter = procedural_sky(float3(0.0, 1.0, 0.0), sun_dir, sun_intensity, inscatter_gain, false);
    inscatter *= sky_wb;
    // `procedural_sky` returns RAW scene radiance (unexposed) — the same convention the sky-
    // background miss path in pbr.slang follows (`sky * exposure`). `hdr` here is already
    // exposed (baked in by `record_lighting`), so the inscatter needs the same camera exposure
    // applied before blending, or a physically-scaled sun (tens of thousands of lux) blows the
    // composite out to white.
    inscatter *= exposure;

    float3 color = lerp(hdr.rgb, inscatter, fog_factor);
    return float4(color, hdr.a);
}
