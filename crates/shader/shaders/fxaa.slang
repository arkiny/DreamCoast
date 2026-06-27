// QHD/UHD track — HDR-aware FXAA (the Decima-style FXAA→TAA pre-pass).
//
// Guerrilla's Decima runs a spatial FXAA on the current frame BEFORE the temporal pass: smoothing
// the per-frame edge aliasing first means the sub-pixel jitter no longer flips edge pixels frame
// to frame, so the temporal accumulation stays stable (the visible "jitter/crawl" goes away) while
// TAA's history rectification handles ghosting. This is the compact directional FXAA (NVIDIA /
// Lottes) made HDR-correct: edge detection + the blend decision use a PERCEPTUAL luma (a reversible
// Reinhard tonemap of luminance, L/(1+L)) so bright HDR pixels don't read as a constant white edge,
// and the directional blur averages the linear HDR colors directly (the taps are a 1px neighborhood,
// so no firefly bleed in practice).

#include "bindless.slang"

struct FxaaPush
{
    uint in_index;  // sampled: input HDR (internal-res lit frame)
    uint out_index; // storage image: edge-smoothed HDR (same extent)
    uint width;
    uint height;
};

[[vk::push_constant]]
ConstantBuffer<FxaaPush> pc : register(b0, space0);

// Perceptual luma: Reinhard tonemap of luminance so edge detection works across HDR range.
float plum(float3 c)
{
    float l = dot(c, float3(0.299, 0.587, 0.114));
    return l / (1.0 + l);
}

static const float EDGE_THRESHOLD = 0.125;     // relative contrast to treat as an edge
static const float EDGE_THRESHOLD_MIN = 0.0625; // absolute floor (skip near-flat regions)
static const float REDUCE_MUL = 1.0 / 8.0;
static const float REDUCE_MIN = 1.0 / 128.0;
static const float SPAN_MAX = 8.0;

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 texel = 1.0 / float2(pc.width, pc.height);
    float2 uv = (float2(tid.xy) + 0.5) * texel;

    float3 cM = g.textures[pc.in_index].SampleLevel(g.samp, uv, 0.0).rgb;
    float3 cNW = g.textures[pc.in_index].SampleLevel(g.samp, uv + float2(-1, -1) * texel, 0.0).rgb;
    float3 cNE = g.textures[pc.in_index].SampleLevel(g.samp, uv + float2(1, -1) * texel, 0.0).rgb;
    float3 cSW = g.textures[pc.in_index].SampleLevel(g.samp, uv + float2(-1, 1) * texel, 0.0).rgb;
    float3 cSE = g.textures[pc.in_index].SampleLevel(g.samp, uv + float2(1, 1) * texel, 0.0).rgb;

    float lM = plum(cM);
    float lNW = plum(cNW);
    float lNE = plum(cNE);
    float lSW = plum(cSW);
    float lSE = plum(cSE);
    float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));

    // No edge here: pass the center through unchanged.
    if (lMax - lMin < max(EDGE_THRESHOLD_MIN, lMax * EDGE_THRESHOLD))
    {
        g.storage_images[pc.out_index][tid.xy] = float4(cM, 1.0);
        return;
    }

    // Edge-perpendicular blur direction from the diagonal luma gradients.
    float2 dir;
    dir.x = -((lNW + lNE) - (lSW + lSE));
    dir.y = ((lNW + lSW) - (lNE + lSE));
    float dirReduce = max((lNW + lNE + lSW + lSE) * 0.25 * REDUCE_MUL, REDUCE_MIN);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, -SPAN_MAX, SPAN_MAX) * texel;

    // Two-tap (rgbA) and four-tap (rgbB) directional averages; rgbB is sharper but can overshoot,
    // so fall back to rgbA when its luma leaves the local range.
    float3 rgbA = 0.5
                  * (g.textures[pc.in_index].SampleLevel(g.samp, uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb
                     + g.textures[pc.in_index].SampleLevel(g.samp, uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb);
    float3 rgbB = rgbA * 0.5
                  + 0.25
                        * (g.textures[pc.in_index].SampleLevel(g.samp, uv + dir * -0.5, 0.0).rgb
                           + g.textures[pc.in_index].SampleLevel(g.samp, uv + dir * 0.5, 0.0).rgb);
    float lB = plum(rgbB);
    float3 outc = (lB < lMin || lB > lMax) ? rgbA : rgbB;
    g.storage_images[pc.out_index][tid.xy] = float4(make_finite(outc), 1.0);
}
