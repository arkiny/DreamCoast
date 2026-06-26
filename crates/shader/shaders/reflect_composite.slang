// Phase 11 Stage C7 — hybrid reflection composite (+ C8c roughness prefilter).
//
// A full-screen compute pass that fuses the two reflection sources into the single
// reflection radiance the lighting pass uses in place of the prefilter-cube IBL specular:
//   * C5 screen-space reflections (`ssr.slang` + `ssr_resolve.slang`) — accurate where the
//     reflected surface is on screen; rgb = reflected color, a = confidence. The stochastic
//     resolve already prefilters it to the surface roughness (the ratio estimator).
//   * C6 GDF reflections (`gdf_reflect.slang`) — the reliable fallback: a single sphere-trace
//     = a CRISP mirror (sky on escape, neighbour colour via C8a) but with NO roughness blur.
//
// Two roughness behaviours, kept separate ("which source" vs "how blurry"):
//   * Blend (C8c): `ssr_trust = smoothstep(roughness)` selects the source — GDF owns sharp
//     metals (where the stochastic SSR ray misses off-screen and goes dark), SSR fades in for
//     glossy lobes. `ssr.a` still routes off-screen misses to the GDF / sky fallback.
//   * Prefilter (C8c): the GDF mirror is roughness-prefiltered here by a roughness-adaptive
//     bilateral gather (a screen-space GGX-lobe approximation), so a glossy metal (copper) gets
//     a blurred reflection like the path tracer instead of a sharp mirror. Sharp metals keep a
//     ~0 radius => crisp. The gather is depth-weighted to avoid bleeding across silhouettes.
//
// Exposure: `gdf_scale` lifts the raw GDF radiance into the SSR's space (1.0 once both are raw
// radiance, C7b); the composite then feeds the lighting pass directly.

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
    uint material_index; // sampled full-res G-buffer material (g = roughness)
    uint depth_index;    // sampled full-res G-buffer depth (silhouette guard for the prefilter)
    float blur_scale;    // GDF prefilter radius (px) per unit roughness; 0 = prefilter off
    float depth_reject;  // max |Δ NDC depth| for a prefilter tap to count as the same surface
    float max_roughness; // SSR (screen mirror) fades out by this roughness; above it = GDF only
};

[[vk::push_constant]]
ConstantBuffer<CompositePush> pc : register(b0, space0);

// Poisson disk (unit radius) for the roughness prefilter gather — a fixed set keeps the pass
// deterministic (VK≡DX) while spreading taps more evenly than a box kernel.
static const float2 POISSON12[12] = {
    float2(-0.326, -0.406), float2(-0.840, -0.074), float2(-0.696, 0.457),
    float2(-0.203, 0.621),  float2(0.962, -0.195),  float2(0.473, -0.480),
    float2(0.519, 0.767),   float2(0.185, -0.893),  float2(0.507, 0.064),
    float2(0.896, 0.412),   float2(-0.322, -0.933), float2(-0.792, -0.598)
};

// Hue-preserving radiance clamp (scale by the max component).
float3 clamp_hue(float3 c, float m)
{
    float mx = max(c.r, max(c.g, c.b));
    return mx > m ? c * (m / mx) : c;
}

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

    // --- Roughness prefilter of the GDF mirror: a depth-weighted Poisson gather whose radius
    // grows with roughness. This approximates integrating the GGX lobe over screen directions
    // (neighbouring surface points reflect slightly different directions on a curved surface),
    // turning the single sphere-trace mirror into a glossy reflection. (Radius scales with
    // roughness only — not reflected-hit distance — a documented first-order approximation.) ---
    float center_z = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;
    // Near-mirror surfaces (chrome) stay crisp — and skip the gather entirely, so the cost is
    // paid only on genuinely glossy pixels. Cap the radius so a very rough surface (or the large
    // ground area) doesn't gather an enormous, expensive disk (a 30 px blur already reads soft).
    float radius = min(roughness * pc.blur_scale, 30.0); // px
    if (roughness > 0.12 && center_z < 1.0)
    {
        float2 texel = 1.0 / float2(pc.width, pc.height);
        // Firefly suppression in the prefilter: clamp each tap before averaging so a single very
        // bright pixel (the sharp reflected sun disk) can't dominate the gather. A fixed-pattern
        // gather of an unclamped point source replicates it as the Poisson speckles seen on the
        // copper highlight, and the spread blows out to flat white; per-tap clamping integrates
        // it into a soft glossy highlight instead (what an energy-conserving mip prefilter does).
        float prefilter_cap = min(pc.clamp_max, 2.5);
        float3 acc = clamp_hue(gdf, prefilter_cap); // centre tap
        float wsum = 1.0;
        // Three concentric rings (36 taps) so a wide glossy radius stays smooth instead of
        // banding; the inner rings fill the disk the outer ring would leave sparse. (A per-pixel
        // tap rotation would dither banding with fewer taps but scatters the gather across the
        // texture cache — measured ~2.5x slower — so fixed rings win here.)
        static const float RING_SCALE[3] = { 1.0, 0.66, 0.33 };
        [unroll]
        for (int ring = 0; ring < 3; ring++)
        {
            float rr = radius * RING_SCALE[ring];
            [unroll]
            for (int i = 0; i < 12; i++)
            {
                float2 nuv = uv + POISSON12[i] * rr * texel;
                float dz = g.textures[pc.depth_index].SampleLevel(g.samp, nuv, 0.0).r;
                if (dz >= 1.0 || abs(dz - center_z) > pc.depth_reject)
                    continue; // background / different surface — would bleed across the silhouette
                float3 tap = g.textures[pc.gdf_index].SampleLevel(g.samp, nuv, 0.0).rgb * pc.gdf_scale;
                acc += clamp_hue(tap, prefilter_cap);
                wsum += 1.0;
            }
        }
        gdf = acc / wsum;
    }

    // --- Blend (C8d): SSR is the accurate source (a full-res screen mirror = the real
    // neighbour pixels: correct colour + geometry the GDF sphere-trace can't match), so it owns
    // the SHARP end; the GDF prefilter takes over as the lobe widens past `max_roughness` (the
    // screen mirror can't blur, and a stochastic trace goes dark on sharp metals). Two gates keep
    // SSR honest:
    //   * `ssr.a` (edge-fade / hit confidence) routes off-screen misses to the GDF / sky fallback.
    //   * luminance validity — an off-screen or grazing SSR ray that found a bad/no-history hit
    //     reads near-black; rather than darkening the metal, fall back to the bright GDF there.
    //     (This also routes the few backend-divergent march pixels to the deterministic GDF, so
    //     the full-res mirror stays VK≡DX-stable.)
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
