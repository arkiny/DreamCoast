// Auto-exposure — a physical-camera adaptation that meters the lit HDR with a LUMINANCE
// HISTOGRAM and drives the exposure each frame (vs a fixed EV100). Histogram metering is robust
// to a high-dynamic-range scene (a bright atrium next to a dark nave): a percentile window
// ignores the darkest/brightest pixels instead of letting them drag a plain average.
//
// Two passes, GPU-only (no CPU readback, no stall; the adapted value lives on the GPU and is
// carried frame to frame, ordered by in-order queue submission):
//   csHistogram — every pixel bins its log2-luminance into a 256-bin histogram (a per-group
//                 groupshared histogram merged into the global buffer with few atomics).
//   csResolve   — reduce the histogram to a trimmed-percentile geometric-mean luminance, then
//                 (the HDR is PRE-exposed by last frame, so de-expose ÷prev) target = key/L_scene,
//                 EMA-adapt, write the exposure buffer + clear the histogram for next frame.

#include "bindless.slang"

static const float AE_MIN_LOG = -10.0; // histogram covers 2^-10 .. 2^4 cd-ish (14 stops)
static const float AE_MAX_LOG = 4.0;
static const uint AE_BINS = 256;

struct AeHistPush
{
    uint hdr_index; // sampled lit HDR (pre-tonemap, last frame's exposure baked in)
    uint hist_buf;  // RWByteAddressBuffer: 256 uint bins
    uint width;
    uint height;
};

struct AeResolvePush
{
    uint hist_buf;   // 256 uint bins (read + cleared here)
    uint expo_buf;   // [0]=exposure, [4]=scene luminance
    uint num_pixels; // width*height (histogram total)
    float key;       // target middle-grey luminance (display-linear)
    float adapt;     // this frame's EMA factor = 1-exp(-dt*speed)
    float min_exp;
    float max_exp;
    float lo_pct;    // trim fraction at the dark end (e.g. 0.5) ...
    float hi_pct;    // ... and keep up to here (e.g. 0.95)
};

float bin_to_luma(float bin)
{
    float t = bin / float(AE_BINS - 1);
    return exp2(lerp(AE_MIN_LOG, AE_MAX_LOG, t));
}

// ---- Pass 1: build the histogram ----------------------------------------------------------

[[vk::push_constant]]
ConstantBuffer<AeHistPush> hp : register(b0, space0);

groupshared uint s_hist[AE_BINS];

[shader("compute")]
[numthreads(16, 16, 1)]
void csHistogram(uint3 dtid: SV_DispatchThreadID, uint gi: SV_GroupIndex)
{
    s_hist[gi] = 0; // 256 threads zero the 256 groupshared bins
    GroupMemoryBarrierWithGroupSync();

    if (dtid.x < hp.width && dtid.y < hp.height)
    {
        float3 c = g.textures[hp.hdr_index].Load(int3(dtid.xy, 0)).rgb;
        float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
        // log2 bin; the darkest pixels (lum→0) fall in bin 0 and are trimmed in the resolve.
        float ln = (log2(max(lum, 1e-8)) - AE_MIN_LOG) / (AE_MAX_LOG - AE_MIN_LOG);
        uint bin = uint(clamp(ln * float(AE_BINS - 1), 0.0, float(AE_BINS - 1)));
        InterlockedAdd(s_hist[bin], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    // Merge this group's local histogram into the global buffer (256 atomics/group, not 1/pixel).
    g.storage_buffers[hp.hist_buf].InterlockedAdd(gi * 4u, s_hist[gi]);
}

// ---- Pass 2: resolve the exposure ---------------------------------------------------------

[[vk::push_constant]]
ConstantBuffer<AeResolvePush> rp : register(b0, space0);

groupshared float s_lum[AE_BINS];   // per-bin (count·luma) within the kept window
groupshared float s_cnt[AE_BINS];   // per-bin kept count

[shader("compute")]
[numthreads(256, 1, 1)]
void csResolve(uint gi: SV_GroupIndex)
{
    uint count = g.storage_buffers[rp.hist_buf].Load(gi * 4u);

    // Prefix-style trim: keep only the bins whose cumulative population sits in [lo_pct, hi_pct].
    // Cheap exact version — each thread counts how many pixels are in lower bins (serial scan in
    // groupshared would be faster, but 256 is tiny and this runs once per frame).
    s_cnt[gi] = float(count);
    GroupMemoryBarrierWithGroupSync();

    float below = 0.0;
    for (uint b = 0; b < gi; b++)
        below += s_cnt[b];
    float total = 0.0;
    for (uint b2 = 0; b2 < AE_BINS; b2++)
        total += s_cnt[b2];

    float lo = total * rp.lo_pct;
    float hi = total * rp.hi_pct;
    // fraction of this bin that lies inside the [lo,hi] window
    float a = clamp(below, lo, hi);
    float bnd = clamp(below + float(count), lo, hi);
    float kept = max(bnd - a, 0.0);
    // weight by log-luma midpoint of the bin so the mean is geometric (in log space)
    s_lum[gi] = kept * (log2(bin_to_luma(float(gi))));
    s_cnt[gi] = kept;
    GroupMemoryBarrierWithGroupSync();

    // Reduce (thread 0; 256 is small).
    if (gi == 0)
    {
        float sum_log = 0.0, sum_w = 0.0;
        for (uint b = 0; b < AE_BINS; b++)
        {
            sum_log += s_lum[b];
            sum_w += s_cnt[b];
        }
        float lum_exposed = (sum_w > 0.0) ? exp2(sum_log / sum_w) : rp.key;
        float prev_exp = asfloat(g.storage_buffers[rp.expo_buf].Load(0));
        prev_exp = (prev_exp > 0.0) ? prev_exp : 1.0;
        float lum_scene = lum_exposed / max(prev_exp, 1e-8); // de-expose → true scene luminance
        float target = clamp(rp.key / max(lum_scene, 1e-6), rp.min_exp, rp.max_exp);
        float next = lerp(prev_exp, target, saturate(rp.adapt));
        g.storage_buffers[rp.expo_buf].Store(0, asuint(next));
        g.storage_buffers[rp.expo_buf].Store(4, asuint(lum_scene));
    }
    GroupMemoryBarrierWithGroupSync();

    // Clear this bin for next frame's accumulation.
    g.storage_buffers[rp.hist_buf].Store(gi * 4u, 0u);
}
