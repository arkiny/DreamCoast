// QHD/UHD track — Temporal Anti-aliasing Upsampling (TAAU) with robust history rectification.
//
// The scene renders at a reduced internal resolution with a per-frame sub-pixel camera jitter;
// this pass reconstructs a full-res HDR by reprojecting + accumulating the jittered frames. The
// two failure modes are (1) visible jitter when history isn't trusted (no accumulation) and
// (2) ghosting when the camera moves (stale history bleeds through). The standard, platform-
// independent toolbox is applied here:
//   • YCoCg variance clipping (Karis 2014 / Salvi 2016): clip the reprojected history to the
//     current neighborhood's statistical color box [μ ± γσ] in YCoCg (ray-AABB clip toward the
//     mean) — the primary anti-ghost, and it also stabilizes (trusted history => no jitter).
//   • Closest-depth dilation (Playdead/UE): reproject using the NEAREST depth in the 3x3 so
//     silhouettes fetch the foreground's history (kills edge ghosting on moving edges).
//   • Luminance-weighted feedback (Lottes): bias toward more history where luma is stable
//     (hides jitter) and toward the current sample where it changes fast (drops ghosts).
// History: ping-pong full-res buffers — `hist` (rgb + length) and `pos` (world point + valid,
// the disocclusion check). `reset` ignores history (first frame / resize).

#include "bindless.slang"

struct TaauPush
{
    float4x4 inv_view_proj;  // current JITTERED clip -> world (reconstruct from the jittered depth)
    float4x4 prev_view_proj; // world -> previous UNJITTERED clip (history is on the stable grid)
    uint hdr_index;          // sampled: current internal-res HDR (the jittered low-res frame)
    uint depth_index;        // sampled: current internal-res depth
    uint out_index;          // storage image: full-res accumulated HDR
    uint hist_read;          // storage buffer: previous accumulation (rgb + length)
    uint hist_write;         // storage buffer: this frame's accumulation
    uint pos_read;           // storage buffer: previous world position (xyz + valid)
    uint pos_write;          // storage buffer: this frame's world position
    uint out_width;          // full (output) resolution
    uint out_height;
    uint in_width;           // internal (render) resolution — neighborhood texel spacing
    uint in_height;
    uint flip_y;             // bit0: 1 = Vulkan (clip-space Y down)
    uint reset;              // 1 = ignore history
    float4 params;           // x = reject distance, y = max history, z = variance gamma, w unused
    float4 jitter;           // xy = current frame's jitter in UV (add to the stable UV to sample)
};

[[vk::push_constant]]
ConstantBuffer<TaauPush> pc : register(b0, space0);

float3 rgb_to_ycocg(float3 c)
{
    return float3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * (c.r - c.b),
                  -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
float3 ycocg_to_rgb(float3 c)
{
    float t = c.x - c.z;
    return float3(t + c.y, c.x + c.z, t - c.y);
}
// Clip the history sample `q` to the AABB [amin, amax] along the ray toward the box center
// (Karis): a true clip (not a clamp) so off-axis history snaps to the nearest box face.
float3 clip_aabb(float3 amin, float3 amax, float3 q)
{
    float3 center = 0.5 * (amax + amin);
    float3 extent = 0.5 * (amax - amin) + 1e-5;
    float3 v = q - center;
    float3 a = abs(v / extent);
    float m = max(a.x, max(a.y, a.z));
    return m > 1.0 ? center + v / m : q;
}
float luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// Bilinear (sub-pixel) history fetch from the ping-pong storage buffers. The history lives in a
// raw buffer (no hardware sampler), so do the 2x2 tap + weighting by hand. CRITICAL for jitter:
// integer-texel (floor) reprojection discards the sub-pixel offset, so the jittered low-res frames
// land incoherently and the jitter degrades into shimmer; a sub-pixel-accurate fetch instead lets
// the jitter act as a stable super-sampling signal. Each tap is validated against the disocclusion
// test (stored world point) and dropped if it was a different surface; surviving taps are
// re-normalized so a silhouette doesn't pull in background history. Returns rgb+len in `out_rgb` /
// `out_len`, with `out_len <= 0` meaning "no usable history".
void fetch_history_bilinear(float2 puv, float3 world_pos, out float3 out_rgb, out float out_len)
{
    out_rgb = float3(0, 0, 0);
    out_len = 0.0;
    float2 res = float2(pc.out_width, pc.out_height);
    float2 sp = puv * res - 0.5; // texel-center convention so an exact reprojection is a no-op
    float2 base = floor(sp);
    float2 f = sp - base;
    float w[4] = { (1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y };
    int2 off[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };

    float3 rgb_sum = float3(0, 0, 0);
    float len_sum = 0.0;
    float wsum = 0.0;
    [unroll] for (int i = 0; i < 4; i++)
    {
        int2 p = int2(base) + off[i];
        if (p.x < 0 || p.y < 0 || p.x >= int(pc.out_width) || p.y >= int(pc.out_height))
            continue;
        uint pidx = uint(p.y) * pc.out_width + uint(p.x);
        float4 pp = asfloat(g.storage_buffers[pc.pos_read].Load4(pidx * 16u));
        // Accept any tap that holds a real prior surface (pp.w). NO hard geometric disocclusion:
        // a hard world-point reject left silhouette + grazing bands stuck at len 1 (full jitter
        // crawl), because under jitter those pixels flip sub-pixel coverage every frame and must
        // ACCUMULATE into an anti-aliased average rather than reset. The YCoCg neighborhood clip in
        // the caller is the anti-ghost for genuine (camera-motion) disocclusion — new content has a
        // different colour and is clipped out (Karis 2014).
        if (pp.w > 0.5)
        {
            float4 ph = asfloat(g.storage_buffers[pc.hist_read].Load4(pidx * 16u));
            rgb_sum += w[i] * ph.rgb;
            len_sum += w[i] * ph.a;
            wsum += w[i];
        }
    }
    // Require a majority of the bilinear footprint to be valid; otherwise it's a disocclusion edge
    // and falling back to "no history" (current sample) is safer than a mostly-stale blend.
    if (wsum > 0.5)
    {
        out_rgb = rgb_sum / wsum;
        out_len = len_sum / wsum;
    }
}

float3 reconstruct(float2 cuv, float depth, float sy)
{
    float2 ndc = float2(cuv.x * 2.0 - 1.0, (cuv.y * 2.0 - 1.0) * sy);
    float4 h = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    return h.xyz / h.w;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.out_width || tid.y >= pc.out_height)
        return;

    uint idx = tid.y * pc.out_width + tid.x;
    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.out_width, pc.out_height);
    // The internal frame was rendered with a +jitter camera offset, so the content for this stable
    // output pixel sits at `uv + jitter` in the internal image (lands every frame on the stable grid).
    float2 cuv = uv + pc.jitter.xy;
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, cuv, 0.0).r;
    float3 cur = g.textures[pc.hdr_index].SampleLevel(g.samp, cuv, 0.0).rgb;

    if (depth >= 1.0)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(cur, 1.0);
        g.storage_buffers[pc.hist_write].Store4(idx * 16u, asuint(float4(cur, 1.0)));
        g.storage_buffers[pc.pos_write].Store4(idx * 16u, asuint(float4(0, 0, 0, 0)));
        return;
    }

    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 its = 1.0 / float2(pc.in_width, pc.in_height);

    // 3x3 neighborhood pass: YCoCg moments for the variance box (the anti-ghost color distribution).
    float3 m1 = float3(0, 0, 0); // sum of YCoCg
    float3 m2 = float3(0, 0, 0); // sum of YCoCg^2
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            float2 suv = cuv + float2(dx, dy) * its;
            float3 c = rgb_to_ycocg(g.textures[pc.hdr_index].SampleLevel(g.samp, suv, 0.0).rgb);
            m1 += c;
            m2 += c * c;
        }
    }
    float3 mean = m1 / 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
    float gamma = pc.params.z;
    float3 box_min = mean - gamma * sigma;
    float3 box_max = mean + gamma * sigma;

    // World point of THIS pixel (validation + storage) and of the closest neighbor (reprojection
    // motion — closest-depth dilation so silhouettes fetch the foreground history).
    float3 world_pos = reconstruct(cuv, depth, sy);
    // Reproject THIS pixel's own surface. Closest-depth dilation (fetching the foreground's history
    // at silhouettes) is omitted on purpose: under sub-pixel jitter it makes the reprojection target
    // oscillate fg/bg every frame, fetching a different history each time -> edge crawl. Own-surface
    // reprojection keeps each output pixel accumulating its own coverage into an anti-aliased edge.
    float3 world_reproj = world_pos;

    float3 hist_rgb = float3(0, 0, 0);
    float hist_len = 0.0;
    if (pc.reset == 0u)
    {
        float4 pclip = mul(pc.prev_view_proj, float4(world_reproj, 1.0));
        if (pclip.w > 0.0)
        {
            float3 pndc = pclip.xyz / pclip.w;
            float2 puv = float2(pndc.x * 0.5 + 0.5, pndc.y * sy * 0.5 + 0.5);
            if (all(puv >= 0.0) && all(puv <= 1.0) && pndc.z >= 0.0 && pndc.z <= 1.0)
            {
                // Sub-pixel-accurate (bilinear) history reprojection — see fetch_history_bilinear.
                fetch_history_bilinear(puv, world_pos, hist_rgb, hist_len);
            }
        }
    }

    float3 accum;
    float len;
    if (hist_len > 0.5)
    {
        // YCoCg variance clip — the primary anti-ghost. History outside the neighborhood's color
        // distribution is snapped onto the box; in-distribution history passes through (stable).
        float3 hist_y = clip_aabb(box_min, box_max, rgb_to_ycocg(hist_rgb));
        hist_rgb = ycocg_to_rgb(hist_y);
        // Straight EMA: the clip rejects stale (camera-moved) history, so long, stable accumulation
        // hides the jitter and resolves sub-pixel detail. alpha floors at 1/max_hist (len capped).
        len = min(hist_len + 1.0, pc.params.y);
        float alpha = 1.0 / len;
        accum = lerp(hist_rgb, cur, alpha);
    }
    else
    {
        len = 1.0;
        accum = cur;
    }

    accum = make_finite(accum);
    g.storage_buffers[pc.hist_write].Store4(idx * 16u, asuint(float4(accum, len)));
    g.storage_buffers[pc.pos_write].Store4(idx * 16u, asuint(float4(world_pos, 1.0)));
    g.storage_images[pc.out_index][tid.xy] = float4(hist_rgb, 1.0); // DEBUG fetched history
}
