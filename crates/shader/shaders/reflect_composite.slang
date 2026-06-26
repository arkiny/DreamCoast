// Phase 11 Stage C7 — hybrid reflection composite.
//
// A full-screen compute pass that fuses the two reflection sources into the single reflection
// radiance the lighting pass uses in place of the prefilter-cube IBL specular:
//   * C5 screen-space reflections (`ssr.slang`) — accurate where the reflected surface is on
//     screen; rgb = reflected color, a = confidence. (Largely sky/off-screen here -> GDF fills.)
//   * C6/C8j GDF reflections (`gdf_reflect.slang` + `reflect_temporal.slang`) — the reliable
//     source: a GGX-importance-sampled, sphere-traced, temporally-resolved reflection (UE Lumen
//     style — the rough lobe is sampled by real rays, so it is already correctly blurred; no
//     image-space prefilter needed). Mirror for near-zero roughness (the chrome).
//
// Blend (C8d): `ssr_trust` selects the source — SSR owns the sharp end, the GDF takes over past
// `max_roughness`. `ssr.a` + a luminance gate route off-screen / bad SSR hits to the bright GDF
// (which also keeps the full-res mirror VK≡DX-stable).
//
// Exposure: `gdf_scale` lifts the raw GDF radiance into the SSR's space (1.0 once both are raw).

#include "bindless.slang"

struct CompositePush
{
    uint ssr_index;      // sampled C5 SSR image (rgb reflected, a confidence)
    uint gdf_index;      // sampled GDF reflection (GGX-resolved, already roughness-blurred)
    uint out_index;      // storage image (composited reflection radiance)
    uint width;
    uint height;
    float gdf_scale;     // exposure applied to the raw GDF radiance (post-exposure viz space)
    float clamp_max;     // firefly clamp: max composited reflection-radiance component (1e30 = off)
    uint material_index; // sampled full-res G-buffer material (g = roughness)
    float max_roughness; // SSR (screen mirror) fades out by this roughness; above it = GDF only
    uint pad0;
    uint pad1;
    uint pad2;
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

    // --- Blend (C8d): SSR owns the sharp end; the GDF takes over past `max_roughness`.
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
