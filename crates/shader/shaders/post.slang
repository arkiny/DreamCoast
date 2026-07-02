// Tonemap pass: reads the linear-HDR lighting result and maps it to the
// backbuffer with an ACES filmic curve (exposure is already applied in the
// lighting pass). Optional debug effects (grayscale/vignette) are retained from
// the Phase 5 composite. The backbuffer is a UNORM target (see `linear_to_srgb`
// in bindless.slang for why), so this pass applies the linear-to-sRGB encode
// itself. This is the final node of the render graph's color chain.
//
// Bindless layout mirrors imgui.slang: the shared `ParameterBlock` in
// `bindless.slang` (`g.textures[]` / `g.samp`), indexed by push constants (b0).
// No globals/cubes here, so the block stays at set 0 (textures b0, samp b1) — the
// SPIR-V layout is identical to the previous loose-global form (Vulkan untouched),
// and on Metal it binds at `[[buffer(1)]]`. The vertex shader synthesizes a
// full-screen triangle from SV_VertexID; `flip_y` selects clip-space Y orientation
// per backend (D3D12 = 0, Vulkan = 1).

#include "bindless.slang"

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    uint hdr_index; // bindless index of the HDR lighting target
    uint mode;      // 0 = none, 1 = grayscale, 2 = vignette
    uint flip_y;    // 0 = D3D12 (Y up), 1 = Vulkan (Y down)
    float exposure; // camera exposure applied before the filmic curve (1 = none)
    float sharpen;  // QHD/UHD: unsharp amount to recover crispness after upscale (0 = off)
    float inv_w;    // 1 / output width  (sharpen tap spacing)
    float inv_h;    // 1 / output height
    // PR-5 bloom composite: the bloom pyramid mip0 (u32 max = no bloom) added onto the
    // HDR at `bloom_intensity` before the filmic curve. Off = byte-identical anchor.
    uint bloom_index;
    float bloom_intensity;
    // PR-5 color grading (ASC CDL: out = (slope*in + offset)^power) applied in linear
    // HDR after exposure/bloom, before the tonemap curve. Neutral = slope 1, offset 0,
    // power 1 -> exact identity. `grade_on` gates it so the neutral case is not even
    // evaluated (a pure no-op, no pow() rounding) -> byte-identical anchor.
    uint grade_on;    // 0 = grading disabled (identity), 1 = apply CDL
    uint pad0;
    uint pad1;
    // Each CDL vector is a full float4 row (never float3 + trailing scalar) to dodge the
    // HLSL/SPIR-V vs. MSL push-constant packing divergence (a scalar after a float3
    // packs at +12 on HLSL/SPIR-V but pads to +16 on MSL).
    float4 cdl_slope;  // gain  (per-channel multiplier; neutral 1,1,1)
    float4 cdl_offset; // lift  (per-channel additive;   neutral 0,0,0)
    float4 cdl_power;  // gamma (per-channel exponent;   neutral 1,1,1)
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

// Narkowicz ACES filmic approximation.
float3 aces(float3 x)
{
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float3 hdr = g.textures[pc.hdr_index].Sample(g.samp, input.uv).rgb;
    // QHD/UHD: contrast-adaptive unsharp (FSR-RCAS-style) to recover crispness lost in the
    // temporal upscale. A 4-tap cross at the output texel spacing; `sharpen == 0` is a no-op.
    if (pc.sharpen > 0.0)
    {
        float3 n = g.textures[pc.hdr_index].Sample(g.samp, input.uv + float2(0.0, -pc.inv_h)).rgb;
        float3 s = g.textures[pc.hdr_index].Sample(g.samp, input.uv + float2(0.0, pc.inv_h)).rgb;
        float3 e = g.textures[pc.hdr_index].Sample(g.samp, input.uv + float2(pc.inv_w, 0.0)).rgb;
        float3 w = g.textures[pc.hdr_index].Sample(g.samp, input.uv + float2(-pc.inv_w, 0.0)).rgb;
        float3 blur = (n + s + e + w) * 0.25;
        hdr = max(hdr + pc.sharpen * (hdr - blur), 0.0);
    }
    // PR-5 bloom composite: add the bloom pyramid's mip0 (already the accumulated,
    // tent-upsampled bright energy in linear HDR) scaled by `bloom_intensity`. Done in
    // linear HDR before exposure/curve. `bloom_index == 0xFFFFFFFF` -> skipped (anchor).
    if (pc.bloom_index != 0xFFFFFFFFu)
    {
        float3 bloom = g.textures[pc.bloom_index].Sample(g.samp, input.uv).rgb;
        hdr += bloom * pc.bloom_intensity;
    }
    // Camera exposure before the filmic curve. The rasterized HDR already bakes
    // exposure into the lighting pass (so it passes 1.0 here); the path-traced
    // output carries raw scene radiance and is exposed at this stage instead.
    float3 exposed = hdr * pc.exposure;
    // PR-5 color grading (ASC CDL) in linear HDR, before the tonemap curve. Neutral
    // params or `grade_on == 0` leave `exposed` untouched (byte-identical).
    if (pc.grade_on != 0u)
    {
        float3 graded = pc.cdl_slope.rgb * exposed + pc.cdl_offset.rgb;
        exposed = pow(max(graded, 0.0), pc.cdl_power.rgb);
    }
    float3 color = aces(exposed);

    if (pc.mode == 1)
    {
        float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
        color = float3(luma, luma, luma);
    }
    else if (pc.mode == 2)
    {
        float2 d = input.uv - 0.5;
        float vignette = saturate(1.0 - dot(d, d) * 1.8);
        color *= vignette;
    }

    // UNORM backbuffer — encode sRGB here (the hardware no longer does it).
    return float4(linear_to_srgb(color), 1.0);
}
