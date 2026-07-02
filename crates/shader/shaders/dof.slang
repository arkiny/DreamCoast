// Depth of field — pipeline re-baseline PR-5 post node #17 (STUB).
//
// Placeholder node that reserves the DoF slot in the ordered post sequence (motion
// blur -> TAA/upscale -> exposure -> bloom -> DOF -> tonemap). Today it is an exact
// passthrough (copies HDR in -> HDR out), so wiring it changes nothing; it exists so
// the graph ordering and the host seam are already in place for the real
// implementation.
//
// Phase 20 expansion point: a circle-of-confusion (CoC) pass computed from the depth
// G-buffer + camera focus distance/aperture, a near/far separable or
// scatter-as-gather bokeh blur, and a composite. Those slot in here by replacing this
// fragment with the CoC + gather chain (same read-old/write-new seam; the host would
// additionally thread the depth target in). See docs/post-process-chain.md.
//
// Bindless layout + full-screen triangle match post.slang.

#include "bindless.slang"

struct VSOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    uint hdr_index; // HDR scene color (input)
    uint flip_y;    // 0 = D3D12 (Y up), 1 = Vulkan (Y down)
    uint pad0;
    uint pad1;
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

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    // STUB: exact passthrough (identity). Replace with the CoC + bokeh chain.
    return float4(g.textures[pc.hdr_index].SampleLevel(g.samp, input.uv, 0).rgb, 1.0);
}
