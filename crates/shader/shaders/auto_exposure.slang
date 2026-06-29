// Auto-exposure: a physical-camera adaptation that meters the lit HDR and drives the camera
// exposure each frame (instead of a fixed EV100). A single 16×16 thread group samples the HDR on a
// stratified grid, reduces the geometric-mean (log-average) luminance in groupshared memory, and
// thread 0 writes the time-adapted exposure into a 1-element storage buffer that next frame's
// lighting reads (pbr.slang scene_exposure). GPU-only: no CPU readback, no stall; the adapted value
// lives on the GPU and is carried frame to frame, ordered by in-order queue submission.
//
// The HDR is already PRE-EXPOSED by the previous frame's exposure (lighting bakes `* exposure`), so
// the metered luminance is de-exposed (÷ prev) to recover the scene luminance before computing the
// new target = key / L_scene. Adaptation is an exponential approach (eye/iris speed).

#include "bindless.slang"

struct AePush
{
    uint hdr_index;   // sampled lit HDR (pre-tonemap, pre-exposure baked from last frame)
    uint expo_buf;    // RWByteAddressBuffer index: [0]=exposure, [4]=scene luminance (debug)
    float key;        // target middle-grey luminance in display-linear (e.g. 0.18)
    float adapt;      // this frame's EMA factor = 1 - exp(-dt * speed)
    float min_exp;    // exposure clamp (avoids runaway in all-black / all-white frames)
    float max_exp;
    float2 pad;
};

[[vk::push_constant]]
ConstantBuffer<AePush> pc : register(b0, space0);

static const uint AE_THREADS = 256; // 16×16
groupshared float s_sum[AE_THREADS];

[shader("compute")]
[numthreads(16, 16, 1)]
void csMain(uint3 gt: SV_GroupThreadID, uint gi: SV_GroupIndex)
{
    // Stratified sample: one tap per thread over a 16×16 grid covering the frame. The linear
    // sampler box-averages, so 256 taps give a stable low-frequency luminance estimate.
    float2 uv = (float2(gt.xy) + 0.5) / 16.0;
    float3 c = g.textures[pc.hdr_index].SampleLevel(g.samp, uv, 0.0).rgb;
    float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
    s_sum[gi] = log(max(lum, 1e-4));
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction to the log-average (→ geometric mean = robust to outliers).
    [unroll]
    for (uint s = AE_THREADS / 2; s > 0; s >>= 1)
    {
        if (gi < s)
            s_sum[gi] += s_sum[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0)
    {
        float lum_exposed = exp(s_sum[0] / float(AE_THREADS)); // geometric mean of the PRE-EXPOSED hdr
        float prev_exp = asfloat(g.storage_buffers[pc.expo_buf].Load(0));
        prev_exp = (prev_exp > 0.0) ? prev_exp : 1.0;
        float lum_scene = lum_exposed / max(prev_exp, 1e-8); // de-expose → true scene luminance
        float target = pc.key / max(lum_scene, 1e-6);
        target = clamp(target, pc.min_exp, pc.max_exp);
        float next = lerp(prev_exp, target, saturate(pc.adapt));
        g.storage_buffers[pc.expo_buf].Store(0, asuint(next));
        g.storage_buffers[pc.expo_buf].Store(4, asuint(lum_scene));
    }
}
