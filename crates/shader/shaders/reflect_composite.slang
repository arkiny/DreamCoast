// Phase 11 Stage C7 — hybrid reflection composite.
//
// A full-screen compute pass that fuses the two reflection sources into the single reflection
// radiance the lighting pass uses in place of the prefilter-cube IBL specular:
//   * C5 screen-space reflections (`ssr.slang`) — accurate where the reflected surface is on
//     screen; rgb = reflected color, a = confidence. (Largely sky/off-screen here -> GDF fills.)
//   * C6/C8j GDF reflections (`gdf_reflect.slang` + `reflect_temporal.slang`) — the reliable
//     source: a GGX-importance-sampled, sphere-traced, temporally-resolved reflection (레퍼런스 엔진 SW GI
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
    float rough_blur;    // max roughness-blur radius in texels (0 = off, gallery byte-identical)
    uint ssr_cut;        // B1-lite hard handoff: 1 = zero the SSR blend (GDF/screen-hit path only —
                         // the SCREEN_HIT trace already carries validated on-screen colours)
};

[[vk::push_constant]]
ConstantBuffer<CompositePush> pc : register(b0, space0);

// Roughness-scaled blur of the low-res GDF reflection. The rough lobe is traced at reduced
// resolution (perf), so the reflection of bright geometry arrives as coarse blocks the bilateral
// upsample leaves as soft "sparkle" patches on the floor. A rough reflection is low-frequency, so
// blurring it is physically fine AND keeps the traced reflection's correct LOCAL colour (unlike a
// prefilter-cube fallback, which is sky-blue). Taps are weighted by roughness similarity so a rough
// surface never pulls in a neighbouring mirror's sharp reflection. rough_blur == 0 => plain fetch
// (gallery anchor). Deterministic (fixed tap ring) => cross-backend safe.
float3 blur_gdf(float2 uv, float roughness, float2 res)
{
    float3 c = g.textures[pc.gdf_index].SampleLevel(g.samp, uv, 0.0).rgb;
    float radius = pc.rough_blur * smoothstep(0.4, 0.9, roughness);
    if (radius < 0.5)
        return c;
    float2 texel = radius / res;
    float3 sum = c;
    float wsum = 1.0;
    // 8-tap ring (unit circle), deterministic — one blurred sample per rough pixel, cheap full-res.
    const float2 dirs[8] = {
        float2(1, 0), float2(0.707, 0.707), float2(0, 1), float2(-0.707, 0.707),
        float2(-1, 0), float2(-0.707, -0.707), float2(0, -1), float2(0.707, -0.707),
    };
    [unroll]
    for (int i = 0; i < 8; i++)
    {
        float2 suv = uv + dirs[i] * texel;
        float3 s = g.textures[pc.gdf_index].SampleLevel(g.samp, suv, 0.0).rgb;
        float rn = g.textures[pc.material_index].SampleLevel(g.samp, suv, 0.0).g;
        float w = saturate(1.0 - abs(rn - roughness) * 4.0); // roughness-similarity edge stop
        sum += s * w;
        wsum += w;
    }
    return sum / wsum;
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
    float3 gdf = blur_gdf(uv, roughness, float2(pc.width, pc.height)) * pc.gdf_scale;

    // --- Blend (C8d): SSR owns the sharp end; the GDF takes over past `max_roughness`.
    //   * `ssr.a` (edge-fade / hit confidence) routes off-screen misses to the GDF / sky fallback.
    //   * luminance validity — a near-black SSR hit (off-screen / grazing) falls back to the bright
    //     GDF rather than darkening the metal (also routes backend-divergent march pixels to the
    //     deterministic GDF, keeping the full-res mirror VK≡DX-stable). ---
    float ssr_trust = 1.0 - smoothstep(0.1, max(pc.max_roughness, 0.12), roughness);
    // Content (pad0): on a NEAR-MIRROR surface, use the GDF/surface-cache reflection ONLY — do not
    // blend SSR. Our SSR has no hit-validation (uncertain / depth-thickness / history-depth like the
    // reference engine's screen-trace), so on a convex SW-RT mirror (the chrome sphere) it finds
    // unreliable glancing hits that a partial blend smears in as a messy silhouette. Glossy surfaces
    // keep SSR (its error is hidden by the broad lobe). The reference engine keeps SSR only as a
    // validated hard-switch upgrade; until that exists, the HQ GDF/cache is the cleaner mirror source.
    if (pc.pad0 == 1u && roughness < 0.1)
        ssr_trust = 0.0;
    // B1-lite hard handoff: the GDF image already holds validated on-screen colours (SCREEN_HIT
    // trace) — a partial SSR blend would double-count them and re-add its feedback wiggle.
    if (pc.ssr_cut == 1u)
        ssr_trust = 0.0;
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
