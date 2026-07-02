// Motion blur — pipeline re-baseline PR-5 post node #13.
//
// A per-pixel velocity-along blur: for each pixel we read its screen-space motion
// vector (PR-2 velocity target, NDC units) and average N taps of the HDR scene color
// spread along that vector, centered on the pixel. This is the simplest correct
// camera+object motion blur — it consumes the same velocity buffer the TAA
// reprojection uses, so a fast-moving object (spin / camera pan) streaks along its
// on-screen travel direction. It runs in linear HDR *before* TAA/upscale and tonemap.
//
// Scope: this is the single-velocity "reconstruction lite" variant. It does NOT do
// the tile-max / neighbor-max dominant-velocity dilation (McGuire 2012) that lets a
// fast foreground object bleed its blur over a slow background — that is a documented
// follow-up (see docs/post-process-chain.md). For camera motion and moderate object
// motion the per-pixel gather already reads plausibly.
//
// Bindless layout + full-screen triangle match post.slang: the shared
// `ParameterBlock` (`g.textures[]` / `g.samp`), indexed by push constants (b0).
// "Read old, write new": reads the HDR scene + velocity, writes a NEW HDR target the
// host threads forward; off = the pass is never recorded (byte-identical anchor).

#include "bindless.slang"

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    uint hdr_index;      // HDR scene color (input)
    uint velocity_index; // RG16F NDC motion vector (PR-2 velocity target)
    uint flip_y;         // 0 = D3D12 (Y up), 1 = Vulkan (Y down)
    uint sample_count;   // taps along the velocity vector (odd; clamped [1..MAX])
    float intensity;     // scales the sampled motion (1 = one frame of travel)
    float max_uv;        // clamp on the per-pixel blur length in UV (avoids whole-screen smear)
    float pad0;
    float pad1;
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

static const uint MAX_SAMPLES = 16u;

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    float2 uv = input.uv;

    // Velocity is stored as cur_ndc - prev_ndc (x in [-2,2] over the screen width, y
    // likewise). Convert NDC delta to a UV delta: NDC spans 2 across the screen, UV
    // spans 1, so uv_delta = ndc_delta * 0.5. The velocity target's Y sign already
    // matches the sampling convention on both backends (written unflipped in PR-2),
    // but the sampled UV origin differs per API, so mirror Y with flip_y.
    float2 vel = g.textures[pc.velocity_index].SampleLevel(g.samp, uv, 0).xy;
    float2 uv_motion = vel * 0.5 * pc.intensity;
    if (pc.flip_y == 0)
        uv_motion.y = -uv_motion.y;

    // Clamp the blur length so a huge motion vector cannot smear across the frame.
    float len = length(uv_motion);
    if (len > pc.max_uv)
        uv_motion *= pc.max_uv / max(len, 1e-6);

    // No motion -> passthrough (keeps still pixels crisp; near-anchor when off).
    float3 center = g.textures[pc.hdr_index].SampleLevel(g.samp, uv, 0).rgb;
    if (len < 1e-5)
        return float4(center, 1.0);

    uint n = clamp(pc.sample_count, 1u, MAX_SAMPLES);
    float3 sum = float3(0.0, 0.0, 0.0);
    // Symmetric taps from -0.5..+0.5 of the motion vector around the pixel center.
    for (uint i = 0u; i < n; i++)
    {
        float t = (float(i) + 0.5) / float(n) - 0.5; // [-0.5, 0.5)
        float2 s = uv + uv_motion * t;
        sum += g.textures[pc.hdr_index].SampleLevel(g.samp, s, 0).rgb;
    }
    return float4(sum / float(n), 1.0);
}
