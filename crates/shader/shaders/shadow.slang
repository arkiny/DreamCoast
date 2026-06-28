// Directional shadow map: a depth-only pass that rasterizes the scene from the
// light's point of view. The vertex shader transforms world-space positions by
// the light's view-projection (`light_mvp`); the fragment stage writes nothing
// (depth only) except, for masked materials, an alpha-test discard so cutout
// geometry (foliage, chains) punches holes in the shadow too. A later full-screen
// pass (pbr.slang) samples the stored depth to shadow the directional sun. The
// model matrix is identity (the sandbox orbits the camera), so input positions are
// already world-space.

#include "bindless.slang"

static const uint NO_TEXTURE = 0xFFFFFFFF;

// POSITION + TEXCOORD are consumed (NORMAL is declared to match the shared mesh
// vertex layout but unused). The UV feeds the masked alpha test.
struct VSInput
{
    float3 pos : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
};

struct VSOutput
{
    float4 clip : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    float4x4 light_mvp;  // light view-projection (model is identity)
    uint base_color_tex; // base-color bindless index (NO_TEXTURE if absent)
    float alpha_cutoff;  // >0 = masked (alpha-test); 0 = opaque (no test)
    float2 _pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[shader("vertex")]
VSOutput vsMain(VSInput input)
{
    VSOutput o;
    o.clip = mul(pc.light_mvp, float4(input.pos, 1.0));
    o.uv = input.uv;
    return o;
}

[shader("fragment")]
void fsMain(VSOutput input)
{
    // Opaque (cutoff 0) never touches the texture -> depth-only, byte-identical to the
    // pre-mask pass. Masked materials discard sub-cutoff texels so the shadow has the
    // same holes as the lit geometry (uses the wrap sampler to match the G-buffer).
    if (pc.alpha_cutoff > 0.0 && pc.base_color_tex != NO_TEXTURE)
    {
        float a = g.textures[pc.base_color_tex].Sample(g.samp_wrap, input.uv).a;
        if (a < pc.alpha_cutoff)
            discard;
    }
}
