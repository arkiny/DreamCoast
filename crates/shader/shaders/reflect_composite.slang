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
    uint ssr_index; // sampled C5 SSR image (rgb reflected, a confidence)
    uint gdf_index; // sampled C6 GDF reflection image (rgb, sky on escape)
    uint out_index; // storage image (composited reflection radiance)
    uint width;
    uint height;
    float gdf_scale; // exposure applied to the raw GDF radiance (post-exposure viz space)
    float clamp_max; // firefly clamp: max composited reflection-radiance component (1e30 = off)
    uint pad1;
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

    float3 refl = lerp(gdf, ssr.rgb, saturate(ssr.a));
    // Firefly clamp at the reflection chokepoint feeding the lighting specular (C7c): cap an
    // isolated bright reflected pixel. Hue-preserving; `clamp_max = 1e30` disables it.
    float m = max(refl.r, max(refl.g, refl.b));
    if (m > pc.clamp_max)
        refl *= pc.clamp_max / m;
    g.storage_images[pc.out_index][tid.xy] = float4(refl, 1.0);
}
