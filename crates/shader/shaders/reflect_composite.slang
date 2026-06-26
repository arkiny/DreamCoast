// Phase 11 Stage C7 — hybrid reflection composite.
//
// A full-screen compute pass that fuses the two reflection sources into the single
// reflection radiance the lighting pass will use in place of the prefilter-cube IBL
// specular lookup:
//   * C5 screen-space reflections (`ssr.slang`) — accurate where the reflected surface
//     is on screen; rgb = reflected color, a = confidence (0 on a miss / screen edge).
//   * C6 GDF reflections (`gdf_reflect.slang`) — the off-screen fallback; rgb = the
//     GDF-traced reflection with the procedural sky already baked in on a ray escape.
//
// Composite = lerp(gdf, ssr, ssr.confidence): SSR where it is confident, the GDF / sky
// fallback elsewhere, blended smoothly by the SSR edge-fade so there is no hard seam.
// Roughness / Fresnel weighting is NOT applied here — that stays in `pbr.slang`'s env
// BRDF term (the composite is a drop-in replacement for the prefiltered cube sample).
//
// Exposure (C7a viz only): the SSR image samples the post-exposure lit HDR while the GDF
// image is raw scene radiance, so `gdf_scale` multiplies the GDF source up into the same
// post-exposure space for a consistent standalone viz (tonemapped at exposure 1.0). C7b
// switches SSR to a raw-radiance history, after which both sources are raw and the
// composite feeds the lighting pass directly.

#include "bindless.slang"

struct CompositePush
{
    uint ssr_index;      // sampled C5 SSR image (rgb reflected, a confidence)
    uint gdf_index;      // sampled C6 GDF reflection image (rgb, sky on escape)
    uint out_index;      // storage image (composited reflection radiance)
    uint width;
    uint height;
    float gdf_scale;     // exposure applied to the raw GDF radiance (post-exposure viz space)
    float clamp_max;     // firefly clamp: max composited reflection-radiance component (1e30 = off)
    uint material_index; // sampled full-res G-buffer material (g = roughness) for the blend
};

[[vk::push_constant]]
ConstantBuffer<CompositePush> pc : register(b0, space0);

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float4 ssr = g.textures[pc.ssr_index].SampleLevel(g.samp, uv, 0.0);
    float3 gdf = g.textures[pc.gdf_index].SampleLevel(g.samp, uv, 0.0).rgb * pc.gdf_scale;
    float roughness = g.textures[pc.material_index].SampleLevel(g.samp, uv, 0.0).g;

    // Roughness-aware blend (C8c). The two sources have opposite reliability profiles here:
    //   * GDF reflect = a single sphere-trace = a CRISP, correct mirror (sky / neighbours /
    //     colour via C8a) — excellent for SHARP metals, but a single ray so it can't blur.
    //   * Stochastic half-res SSR = a GGX-jittered ray + ratio-estimator resolve = reliable for
    //     GLOSSY lobes (the kernel denoises a wide cone), but for a near-mirror its jittered ray
    //     misses off-screen → the resolve goes dark, dragging the metal below the GDF truth
    //     (measured: SSR composite 5.90/ch vs GDF-only 3.42/ch vs PT).
    // So fade SSR in with roughness: GDF owns the sharp end, SSR adds on-screen glossy detail as
    // the lobe widens. `ssr.a` (confidence / edge-fade) still gates SSR where its trace left the
    // screen, so off-screen reflections fall through to the GDF / sky fallback.
    float ssr_trust = smoothstep(0.2, 0.6, roughness);
    float3 refl = lerp(gdf, ssr.rgb, saturate(ssr.a) * ssr_trust);
    // Firefly clamp at the reflection chokepoint feeding the lighting specular (C7c): cap an
    // isolated bright reflected pixel. Hue-preserving; `clamp_max = 1e30` disables it.
    float m = max(refl.r, max(refl.g, refl.b));
    if (m > pc.clamp_max)
        refl *= pc.clamp_max / m;
    g.storage_images[pc.out_index][tid.xy] = float4(refl, 1.0);
}
