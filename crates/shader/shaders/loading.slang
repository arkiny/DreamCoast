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

// 3×5 bitmap font for the percentage text: glyphs 0–9 then '%' (index 10). Each glyph packs 15 bits
// (5 rows × 3 cols), bit `row*3 + col` set = lit pixel, row 0 at the top.
static const uint GLYPH[11] = {
    31599u, 29850u, 29671u, 31207u, 18925u, 31183u, 31695u, 9383u, 31727u, 31215u, 21157u
};

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

    // ── Percentage text ABOVE the bar, drawn from a compact 3×5 bitmap font (no font atlas). ──
    int pct = clamp(int(pc.frac * 100.0 + 0.5), 0, 100);
    int digits[3];
    int ndig;
    if (pct >= 100) { digits[0] = 1; digits[1] = 0; digits[2] = 0; ndig = 3; }
    else if (pct >= 10) { digits[0] = pct / 10; digits[1] = pct % 10; ndig = 2; }
    else { digits[0] = pct; ndig = 1; }
    int nglyph = ndig + 1; // + the '%' glyph (index 10)

    float cell = 4.0;                  // px per font cell
    float adv = 3.0 * cell + cell;     // glyph width (3 cells) + 1-cell gap
    float glyph_h = 5.0 * cell;
    float block_w = float(nglyph) * adv;
    float tx0 = (w - block_w) * 0.5;
    float ty0 = bar_y - glyph_h - 14.0; // sit above the bar

    if (p.y >= ty0 && p.y < ty0 + glyph_h && p.x >= tx0 && p.x < tx0 + block_w)
    {
        int slot = int((p.x - tx0) / adv);
        float lx = (p.x - tx0) - float(slot) * adv;
        int cx = int(lx / cell);
        int cy = int((p.y - ty0) / cell);
        if (cx < 3) // inside the 3-wide glyph (not the trailing gap)
        {
            int gi = (slot < ndig) ? digits[slot] : 10;
            if (((GLYPH[gi] >> (cy * 3 + cx)) & 1u) != 0u)
                col = float3(0.75, 0.82, 0.90); // text colour
        }
    }

    return float4(saturate(col), 1.0);
}
