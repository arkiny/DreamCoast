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
                int2 ppix = int2(floor(puv * float2(pc.width, pc.height)));
                ppix = clamp(ppix, int2(0, 0), int2(pc.width - 1, pc.height - 1));
                uint pidx = uint(ppix.y) * pc.width + uint(ppix.x);
                float4 ph = asfloat(g.storage_buffers[pc.hist_read].Load4(pidx * 16u));
                float4 pp = asfloat(g.storage_buffers[pc.pos_read].Load4(pidx * 16u));
                if (pp.w > 0.5 && distance(world_pos, pp.xyz) < pc.params.x)
                {
                    hist_rgb = ph.rgb;
                    hist_len = ph.a;
                }
            }
        }
    }

    float3 accum;
    float len;
    if (hist_len > 0.5)
    {
        // Neighborhood clamp (params.w > 0.5): clip the reprojected history to the current
        // GI's 3x3 min/max box before accumulating — removes history fireflies / ghosting.
        if (pc.params.w > 0.5)
        {
            float2 ts = 1.0 / float2(pc.width, pc.height);
            float3 mn = gi_cur, mx = gi_cur;
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    float3 c = g.textures[pc.gi_raw_index]
                                   .SampleLevel(g.samp, uv + float2(dx, dy) * ts, 0.0)
                                   .rgb;
                    mn = min(mn, c);
                    mx = max(mx, c);
                }
            }
            hist_rgb = clamp(hist_rgb, mn, mx);
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
