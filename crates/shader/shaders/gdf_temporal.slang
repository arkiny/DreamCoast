// Phase 11 Stage C4 — temporal reprojection + accumulation for the noisy GDF GI.
//
// The C3 GI pass produces a noisy indirect-irradiance image (spp ~8). This pass
// accumulates it over frames: each pixel reconstructs its world surface point from
// the current depth, projects that point with the PREVIOUS frame's view-projection to
// find where it landed last frame, and blends the reprojected history with the current
// noisy sample (exponential moving average, alpha = 1/history_length clamped to a floor
// so it stays responsive). Because the sample scene's geometry is static, the world
// point is frame-invariant, so reprojection is exact for a moving camera; a stored
// per-pixel world position validates the history (disocclusion: reject when the prior
// front surface at the reprojected texel was a different point).
//
// History lives in two ping-pong byte-address storage buffers (float4 per pixel):
// `hist` (rgb = accumulated irradiance, a = history length) and `pos` (xyz = the world
// point, w = valid). `reset` (first frame / lighting or resolution change) ignores the
// history. The output is the temporally-denoised irradiance the spatial à-trous pass
// (gdf_atrous.slang) then cleans further before lighting consumes it.

#include "bindless.slang"

struct TemporalPush
{
    float4x4 inv_view_proj;  // current clip -> world (surface reconstruction)
    float4x4 prev_view_proj; // world -> previous clip (reprojection)
    uint gi_raw_index;       // sampled: current noisy GI irradiance
    uint depth_index;        // sampled: current G-buffer depth
    uint normal_index;       // sampled: current G-buffer world normal (unused here; kept for parity)
    uint out_index;          // storage image: temporally accumulated output
    uint hist_read;          // storage buffer: previous accumulation (rgb + length)
    uint hist_write;         // storage buffer: this frame's accumulation
    uint pos_read;           // storage buffer: previous world position (xyz + valid)
    uint pos_write;          // storage buffer: this frame's world position
    uint width;
    uint height;
    uint flip_y;             // bit0: 1 = Vulkan (clip-space Y down)
    uint reset;              // 1 = ignore history (first frame / view change)
    float4 params;           // x = reject distance, y = max history, z = min alpha, w unused
};

[[vk::push_constant]]
ConstantBuffer<TemporalPush> pc : register(b0, space0);

// Fetch reprojected GI history (accumulated irradiance + length). bit1 of flip_y selects a
// sub-pixel bilinear fetch for the TAAU camera-jitter path (so the GI accumulation tracks the
// jittered reprojection instead of snapping to integer texels, which blurs/destabilizes it);
// default keeps the integer-floor fetch so the non-jittered path is byte-identical. Per-tap
// disocclusion via the stored world point.
void fetch_gi_history(float2 puv, float3 world_pos, out float3 out_rgb, out float out_len)
{
    out_rgb = float3(0, 0, 0);
    out_len = 0.0;
    float2 res = float2(pc.width, pc.height);
    if ((pc.flip_y & 2u) == 0u)
    {
        int2 ppix = clamp(int2(floor(puv * res)), int2(0, 0),
                          int2(int(pc.width) - 1, int(pc.height) - 1));
        uint pidx = uint(ppix.y) * pc.width + uint(ppix.x);
        float4 pp = asfloat(g.storage_buffers[pc.pos_read].Load4(pidx * 16u));
        if (pp.w > 0.5 && distance(world_pos, pp.xyz) < pc.params.x)
        {
            float4 ph = asfloat(g.storage_buffers[pc.hist_read].Load4(pidx * 16u));
            out_rgb = ph.rgb;
            out_len = ph.a;
        }
        return;
    }
    float2 sp = puv * res - 0.5;
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
        if (p.x < 0 || p.y < 0 || p.x >= int(pc.width) || p.y >= int(pc.height))
            continue;
        uint pidx = uint(p.y) * pc.width + uint(p.x);
        float4 pp = asfloat(g.storage_buffers[pc.pos_read].Load4(pidx * 16u));
        if (pp.w > 0.5 && distance(world_pos, pp.xyz) < pc.params.x)
        {
            float4 ph = asfloat(g.storage_buffers[pc.hist_read].Load4(pidx * 16u));
            rgb_sum += w[i] * ph.rgb;
            len_sum += w[i] * ph.a;
            wsum += w[i];
        }
    }
    if (wsum > 0.5)
    {
        out_rgb = rgb_sum / wsum;
        out_len = len_sum / wsum;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    uint idx = tid.y * pc.width + tid.x;
    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;

    // Background: nothing to accumulate; clear this pixel's history so a later
    // disocclusion onto it can't pick up stale radiance.
    if (depth >= 1.0)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 1.0);
        g.storage_buffers[pc.hist_write].Store4(idx * 16u, asuint(float4(0, 0, 0, 0)));
        g.storage_buffers[pc.pos_write].Store4(idx * 16u, asuint(float4(0, 0, 0, 0)));
        return;
    }

    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 hpos = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    float3 world_pos = hpos.xyz / hpos.w;

    float3 gi_cur = g.textures[pc.gi_raw_index].SampleLevel(g.samp, uv, 0.0).rgb;

    // Reproject into the previous frame and validate against the stored world point.
    float3 hist_rgb = float3(0.0, 0.0, 0.0);
    float hist_len = 0.0;
    if (pc.reset == 0u)
    {
        float4 pclip = mul(pc.prev_view_proj, float4(world_pos, 1.0));
        if (pclip.w > 0.0)
        {
            float3 pndc = pclip.xyz / pclip.w;
            float2 puv = float2(pndc.x * 0.5 + 0.5, pndc.y * sy * 0.5 + 0.5);
            if (all(puv >= 0.0) && all(puv <= 1.0) && pndc.z >= 0.0 && pndc.z <= 1.0)
            {
                fetch_gi_history(puv, world_pos, hist_rgb, hist_len);
            }
        }
    }

    float3 accum;
    float len;
    if (hist_len > 0.5)
    {
        // Temporal history clamp permutation (pc.params.w):
        //   <= 0.5 : OFF — let the EMA converge. The hard min/max box below is built from the
        //            NOISY spp1 GI, so it sits on the per-frame noise and drags the converged
        //            history back to noise every frame -> static shimmer (the rotating-camera-free
        //            flicker). Off lets the accumulation average out; the per-sample firefly clamp
        //            (gdf_gi clamp_max) + the spatial a-trous still suppress fireflies, and the
        //            world-position disocclusion test handles ghosting. (Content default.)
        //   ~1.0   : HARD 3x3 min/max box (legacy; the gallery byte-identical anchor).
        //   > 1.5  : VARIANCE — clamp to mean +- gamma*sigma (gamma = params.w), a wide box that
        //            rejects only true outliers while allowing convergence.
        if (pc.params.w > 0.5)
        {
            float2 ts = 1.0 / float2(pc.width, pc.height);
            float3 mn = gi_cur, mx = gi_cur;
            float3 m1 = gi_cur, m2 = gi_cur * gi_cur;
            float cnt = 1.0;
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0)
                        continue;
                    float3 c = g.textures[pc.gi_raw_index]
                                   .SampleLevel(g.samp, uv + float2(dx, dy) * ts, 0.0)
                                   .rgb;
                    mn = min(mn, c);
                    mx = max(mx, c);
                    m1 += c;
                    m2 += c * c;
                    cnt += 1.0;
                }
            }
            if (pc.params.w > 1.5)
            {
                float3 mean = m1 / cnt;
                float3 sig = sqrt(max(m2 / cnt - mean * mean, float3(0.0, 0.0, 0.0)));
                float gma = pc.params.w;
                hist_rgb = clamp(hist_rgb, mean - gma * sig, mean + gma * sig);
            }
            else
            {
                hist_rgb = clamp(hist_rgb, mn, mx); // hard (legacy)
            }
        }
        len = min(hist_len + 1.0, pc.params.y);
        float alpha = max(1.0 / len, pc.params.z); // EMA weight on the new sample
        accum = lerp(hist_rgb, gi_cur, alpha);
    }
    else
    {
        len = 1.0;
        accum = gi_cur;
    }

    // Sanitize before storing: a NaN would persist in the history and spread via the EMA.
    accum = make_finite(accum);
    g.storage_buffers[pc.hist_write].Store4(idx * 16u, asuint(float4(accum, len)));
    g.storage_buffers[pc.pos_write].Store4(idx * 16u, asuint(float4(world_pos, 1.0)));
    g.storage_images[pc.out_index][tid.xy] = float4(accum, 1.0);
}
