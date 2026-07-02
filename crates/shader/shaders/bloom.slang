// Bloom — pipeline re-baseline PR-5 post node #16.
//
// A progressive dual-filter bloom (the canonical real-time approach popularized by
// Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare",
// SIGGRAPH 2014): a bright-pass prefilter, a chain of 13-tap box downsamples that
// build a mip pyramid, then 3x3 tent-filter upsamples that additively accumulate
// coarse mips back up the chain. This is stable under motion (wide, smooth kernels)
// and cheap (each link works at half resolution of the previous).
//
// Why these kernels (vs. a separable Gaussian): the 13-tap "downsample with 4 inner
// + 9 outer taps" and the 9-tap tent upsample are the reference filters from the CoD
// talk — they give a rounder, artefact-free bloom in far fewer passes than a
// separable Gaussian chain, and the prefilter's *partial Karis average* (weight the
// bright-pass by 1/(1+luma)) tames single-pixel fireflies before they smear.
//
// The pass runs entirely in linear HDR *before* tonemap (post node ordering: motion
// blur -> TAA/upscale -> exposure -> BLOOM -> DoF -> tonemap). The host threads it as
// "read old, write new": the prefilter reads the finished HDR scene color and writes
// mip0 of a private pyramid; the tonemap pass later reads HDR + the bloom mip0 and
// adds `intensity * bloom` — the scene HDR target itself is never mutated, so bloom
// off = the byte-identical anchor.
//
// Bindless layout + the full-screen-triangle vertex shader match post.slang /
// blur.slang: the shared `ParameterBlock` (`g.textures[]` / `g.samp`) at set 0.

#include "bindless.slang"

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

// One push block shared by every bloom entry. Unused fields per entry are simply
// ignored (all scalars/uints, so the HLSL/SPIR-V/MSL packing is identical — no
// float3-then-scalar hazard). `src_texel` is 1/size of the *source* texture so the
// tap offsets land on source texel centers regardless of the dest size.
struct PushConstants
{
    uint src_index;    // bloom source (HDR scene for prefilter; prev mip otherwise)
    uint add_index;    // upsample: the same-size mip to add the tent result onto (u32 max = none)
    uint flip_y;       // 0 = D3D12 (Y up), 1 = Vulkan (Y down)
    uint pad0;
    float2 src_texel;  // 1 / source size
    float threshold;   // prefilter bright-pass knee (0 = passthrough for the plain downsample)
    float intensity;   // final composite: bloom scale added onto the HDR (unused mid-chain)
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

float luma(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float3 fetch(float2 uv)
{
    return g.textures[pc.src_index].SampleLevel(g.samp, uv, 0).rgb;
}

// --- Prefilter + downsample (13-tap box, Jimenez CoD:AW) ------------------------
// 4 inner taps (each at +-0.5 texel: a 2x2 box) + 9 outer taps (a 3x3 box at +-1
// texel), combined with the reference weights (0.5 total on the inner box, 0.125
// each on the four outer boxes). On the *first* downsample the inner/outer groups are
// Karis-averaged (weight 1/(1+luma)) to suppress fireflies; later downsamples pass
// `threshold < 0` sentinel? No -- we split into two entries so mid-chain stays a plain
// box average (cheaper, and the pyramid is already firefly-clamped by mip0).
float3 downsample13(float2 uv, bool karis, float threshold)
{
    float2 t = pc.src_texel;

    float3 a = fetch(uv + t * float2(-2.0, 2.0));
    float3 b = fetch(uv + t * float2(0.0, 2.0));
    float3 c = fetch(uv + t * float2(2.0, 2.0));

    float3 d = fetch(uv + t * float2(-2.0, 0.0));
    float3 e = fetch(uv + t * float2(0.0, 0.0));
    float3 f = fetch(uv + t * float2(2.0, 0.0));

    float3 g0 = fetch(uv + t * float2(-2.0, -2.0));
    float3 h = fetch(uv + t * float2(0.0, -2.0));
    float3 i = fetch(uv + t * float2(2.0, -2.0));

    float3 j = fetch(uv + t * float2(-1.0, 1.0));
    float3 k = fetch(uv + t * float2(1.0, 1.0));
    float3 l = fetch(uv + t * float2(-1.0, -1.0));
    float3 m = fetch(uv + t * float2(1.0, -1.0));

    // Bright-pass knee applied per-tap on the prefilter link (threshold > 0). Soft
    // knee: keep only luma above the threshold, scaled to preserve hue.
    if (threshold > 0.0)
    {
        float3 taps[13] = { a, b, c, d, e, f, g0, h, i, j, k, l, m };
        [unroll]
        for (int n = 0; n < 13; n++)
        {
            float lum = luma(taps[n]);
            taps[n] *= saturate(lum - threshold) / max(lum, 1e-4);
        }
        a = taps[0]; b = taps[1]; c = taps[2]; d = taps[3]; e = taps[4]; f = taps[5];
        g0 = taps[6]; h = taps[7]; i = taps[8]; j = taps[9]; k = taps[10];
        l = taps[11]; m = taps[12];
    }

    // Five 2x2 box groups (the CoD partitioning).
    float3 grp0 = (j + k + l + m) * 0.25;       // inner box
    float3 grp1 = (a + b + d + e) * 0.25;        // top-left box
    float3 grp2 = (b + c + e + f) * 0.25;        // top-right box
    float3 grp3 = (d + e + g0 + h) * 0.25;       // bottom-left box
    float3 grp4 = (e + f + h + i) * 0.25;        // bottom-right box

    if (karis)
    {
        // Partial Karis average: weight each box by 1/(1+luma) so a single fireball
        // pixel cannot dominate the downsample (the classic bloom firefly fix).
        float w0 = 1.0 / (1.0 + luma(grp0));
        float w1 = 1.0 / (1.0 + luma(grp1));
        float w2 = 1.0 / (1.0 + luma(grp2));
        float w3 = 1.0 / (1.0 + luma(grp3));
        float w4 = 1.0 / (1.0 + luma(grp4));
        float wsum = w0 + w1 + w2 + w3 + w4;
        return (grp0 * w0 + grp1 * w1 + grp2 * w2 + grp3 * w3 + grp4 * w4) / max(wsum, 1e-4);
    }
    // Reference downsample weights: 0.5 on the inner box, 0.125 on each outer box.
    return grp0 * 0.5 + (grp1 + grp2 + grp3 + grp4) * 0.125;
}

// Prefilter (mip0 build): bright-pass + Karis-averaged 13-tap downsample.
[shader("fragment")]
float4 fsPrefilter(VSOutput input) : SV_Target
{
    return float4(downsample13(input.uv, true, pc.threshold), 1.0);
}

// Mid-chain downsample: plain 13-tap box (no threshold, no Karis).
[shader("fragment")]
float4 fsDownsample(VSOutput input) : SV_Target
{
    return float4(downsample13(input.uv, false, 0.0), 1.0);
}

// --- Upsample (3x3 tent, additive) ----------------------------------------------
// Reads the coarser mip (`src_index`), tent-filters it up, and adds the same-size
// finer mip (`add_index`). Writes to a NEW target (read old, write new).
[shader("fragment")]
float4 fsUpsample(VSOutput input) : SV_Target
{
    float2 uv = input.uv;
    float2 t = pc.src_texel; // 1 / source (coarse) size -> tent radius scales with the coarse mip

    // 3x3 tent: center *4, edges *2, corners *1, /16.
    float3 c = float3(0.0, 0.0, 0.0);
    c += fetch(uv + t * float2(-1.0, 1.0)) * 1.0;
    c += fetch(uv + t * float2(0.0, 1.0)) * 2.0;
    c += fetch(uv + t * float2(1.0, 1.0)) * 1.0;
    c += fetch(uv + t * float2(-1.0, 0.0)) * 2.0;
    c += fetch(uv + t * float2(0.0, 0.0)) * 4.0;
    c += fetch(uv + t * float2(1.0, 0.0)) * 2.0;
    c += fetch(uv + t * float2(-1.0, -1.0)) * 1.0;
    c += fetch(uv + t * float2(0.0, -1.0)) * 2.0;
    c += fetch(uv + t * float2(1.0, -1.0)) * 1.0;
    c *= (1.0 / 16.0);

    float3 add = float3(0.0, 0.0, 0.0);
    if (pc.add_index != 0xFFFFFFFFu)
        add = g.textures[pc.add_index].SampleLevel(g.samp, uv, 0).rgb;
    return float4(c + add, 1.0);
}
