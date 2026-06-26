// Phase 11 Stage C7 — hybrid reflection composite (+ C8g roughness mip prefilter).
//
// A full-screen compute pass that fuses the two reflection sources into the single
// reflection radiance the lighting pass uses in place of the prefilter-cube IBL specular:
//   * C5 screen-space reflections (`ssr.slang`) — accurate where the reflected surface is on
//     screen; rgb = reflected color, a = confidence. (Largely sky/off-screen here -> GDF fills.)
//   * C6 GDF reflections (`gdf_reflect.slang`) — the reliable fallback: a single sphere-trace
//     = a CRISP mirror (sky on escape, neighbour colour via C8a / the surface cache).
//
// Two roughness behaviours, kept separate ("which source" vs "how blurry"):
//   * Blend (C8d): `ssr_trust` selects the source — SSR owns the sharp end, the GDF prefilter
//     takes over past `max_roughness`. `ssr.a` + a luminance gate route off-screen / bad SSR
//     hits to the bright GDF (which also keeps the full-res mirror VK≡DX-stable).
//   * Prefilter (C8g): the GDF mirror is roughness-prefiltered by sampling a MIP PYRAMID of the
//     GDF reflection (built by `reflect_downsample.slang`) at a roughness-derived LOD — the way
//     UE prefilters its reflection captures / skylight into roughness mips. Energy-conserving
//     box downsampling integrates a bright pixel into a soft highlight (vs the old screen-space
//     gather, which replicated point sources as fixed-pattern speckles). Levels: 0 = full,
//     1 = 1/4, 2 = 1/16, 3 = 1/64 (each downsample quarters each dimension).
//
// Exposure: `gdf_scale` lifts the raw GDF radiance into the SSR's space (1.0 once both are raw).

#include "bindless.slang"

struct CompositePush
{
    uint ssr_index;      // sampled C5 SSR image (rgb reflected, a confidence)
    uint gdf_index;      // sampled GDF reflection, full-res (mip level 0)
    uint mip1_index;     // 1/4-res prefilter level
    uint mip2_index;     // 1/16-res
    uint mip3_index;     // 1/64-res
    uint out_index;      // storage image (composited reflection radiance)
    uint width;
    uint height;
    float gdf_scale;     // exposure applied to the raw GDF radiance (post-exposure viz space)
    float clamp_max;     // firefly clamp: max composited reflection-radiance component (1e30 = off)
    uint material_index; // sampled full-res G-buffer material (g = roughness)
    float max_roughness; // SSR (screen mirror) fades out by this roughness; above it = GDF only
};

[[vk::push_constant]]
ConstantBuffer<CompositePush> pc : register(b0, space0);

// Sample GDF-reflection mip `level` (0 = full, 1 = 1/4, 2 = 1/16, 3 = 1/64) at `uv`.
float3 gdf_level(uint level, float2 uv)
{
    uint idx = level == 0u ? pc.gdf_index
                           : (level == 1u ? pc.mip1_index : (level == 2u ? pc.mip2_index : pc.mip3_index));
    return g.textures[idx].SampleLevel(g.samp, uv, 0.0).rgb * pc.gdf_scale;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float4 ssr = g.textures[pc.ssr_index].SampleLevel(g.samp, uv, 0.0);
    float roughness = g.textures[pc.material_index].SampleLevel(g.samp, uv, 0.0).g;

    // --- Roughness mip prefilter of the GDF mirror. Map roughness to a LOD across the pyramid
    // (each level is a 4x-coarser energy-conserving average = a wider lobe) and trilinearly blend
    // the two bracketing levels. Near-mirror (chrome) stays at LOD 0 = crisp; glossy (copper)
    // pulls a coarser level = a soft highlight, like UE's prefiltered reflection mips. ---
    float lod = saturate((roughness - 0.04) / 0.56) * 3.0; // roughness ~0.6 -> coarsest level
    uint l0 = uint(floor(lod));
    uint l1 = min(l0 + 1u, 3u);
    float3 gdf = lerp(gdf_level(l0, uv), gdf_level(l1, uv), frac(lod));

    // --- Blend (C8d): SSR owns the sharp end; the GDF prefilter takes over past `max_roughness`.
    //   * `ssr.a` (edge-fade / hit confidence) routes off-screen misses to the GDF / sky fallback.
    //   * luminance validity — a near-black SSR hit (off-screen / grazing) falls back to the bright
    //     GDF rather than darkening the metal (also routes backend-divergent march pixels to the
    //     deterministic GDF, keeping the full-res mirror VK≡DX-stable). ---
    float ssr_trust = 1.0 - smoothstep(0.1, max(pc.max_roughness, 0.12), roughness);
    float ssr_lum = dot(ssr.rgb, float3(0.299, 0.587, 0.114));
    float gdf_lum = dot(gdf, float3(0.299, 0.587, 0.114));
    float valid = ssr_lum > 0.25 * gdf_lum ? 1.0 : 0.0;
    float3 refl = lerp(gdf, ssr.rgb, saturate(ssr.a) * ssr_trust * valid);
    // Firefly clamp at the reflection chokepoint feeding the lighting specular (C7c): cap an
    // isolated bright reflected pixel. Hue-preserving; `clamp_max = 1e30` disables it.
    float m = max(refl.r, max(refl.g, refl.b));
    if (m > pc.clamp_max)
        refl *= pc.clamp_max / m;
    g.storage_images[pc.out_index][tid.xy] = float4(refl, 1.0);
}
