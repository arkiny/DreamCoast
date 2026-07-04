// Startup loading screen — a self-contained progress bar drawn PROCEDURALLY in the fragment shader
// (no vertex buffer, no bindless table, no device access per frame). A dedicated loading thread owns
// this pipeline + the swapchain and presents it every frame while the main thread runs the cold cook
// (docs/loading-screen-thread.md). One push constant carries the progress fraction + a time value
// for a subtle shimmer, plus the target size.

struct PushConstants
{
    float frac;   // 0..1 progress
    float time;   // seconds (animates the shimmer so the bar never looks frozen)
    uint width;   // render-target size (px)
    uint height;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

struct VSOut
{
    float4 pos : SV_Position;
};

// Full-screen triangle from the vertex id — no vertex buffer.
[shader("vertex")]
VSOut vsMain(uint vid: SV_VertexID)
{
    VSOut o;
    float2 uv = float2((vid << 1) & 2, vid & 2);
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 fsMain(VSOut i): SV_Target
{
    float2 p = i.pos.xy; // pixel coords
    float w = float(pc.width), h = float(pc.height);

    float3 col = float3(0.055, 0.065, 0.085); // background

    // A centred progress bar: 55% of the width, 16 px tall.
    float bar_w = w * 0.55;
    float bar_h = 16.0;
    float bar_x = (w - bar_w) * 0.5;
    float bar_y = h * 0.5 - bar_h * 0.5;

    if (p.x >= bar_x && p.x <= bar_x + bar_w && p.y >= bar_y && p.y <= bar_y + bar_h)
    {
        float t = (p.x - bar_x) / bar_w; // 0..1 across the bar
        if (t <= pc.frac)
        {
            // Filled: teal with a travelling shimmer so it reads as "working" even when `frac` is
            // momentarily static (a GPU upload between cook phases).
            float shimmer = 0.12 * sin(t * 24.0 - pc.time * 5.0);
            col = float3(0.18, 0.55, 0.78) + shimmer;
        }
        else
        {
            col = float3(0.13, 0.15, 0.19); // empty track
        }
    }

    return float4(saturate(col), 1.0);
}
