// Phase 11 — stochastic SSR resolve: spatial ratio estimator + temporal (half-res).
//
// 레퍼런스식 (Stachowiak, SIGGRAPH 2015): the trace casts ONE GGX-jittered ray per
// half-res pixel, storing the reflected colour (+ confidence) and the ray direction (+ the
// GGX pdf it was sampled with). This pass resolves the noise *spatially* by reusing the
// rays from a neighbourhood — each borrowed ray `q` is reweighted by the *ratio*
// `pdf_p(dir_q) / pdf_q`, i.e. how likely the centre pixel `p` would itself have sampled
// that direction under its own GGX lobe. This is an unbiased ratio estimator: it gives
// many effective samples per pixel from a single ray each, and adapts to roughness for
// free (a sharp lobe weights only near-mirror neighbours -> stays crisp; a wide lobe pulls
// in the whole kernel -> blurs). A light temporal accumulation + firefly clamp stabilise
// the rest, so we no longer lean on a long temporal feedback (which diverged across
// backends). The resolved half-res reflection feeds the C7 composite (bilinear upsample).

#include "bindless.slang"

static const float PI = 3.14159265359;

struct ResolvePush
{
    float4x4 inv_view_proj;  // current half-res clip -> world (surface recon)
    float4x4 prev_view_proj; // world -> previous clip (reprojection)
    float4 cam_pos;          // xyz camera world position
    uint ssr_a_index;        // sampled: trace colour + confidence (rgb + a)
    uint ssr_b_index;        // sampled: trace ray dir + GGX pdf (xyz + w; w <= 0 = miss)
    uint depth_index;        // sampled: full-res G-buffer depth
    uint normal_index;       // sampled: G-buffer world normal
    uint material_index;     // sampled: G-buffer material (g = roughness)
    uint out_index;          // storage image: resolved reflection (rgb + confidence)
    uint accum_read;         // storage buffer: previous accumulation (rgb + confidence)
    uint accum_write;
    uint pos_read;           // storage buffer: previous surface world pos (xyz + valid)
    uint pos_write;
    uint width;              // half-res
    uint height;
    uint flip_y;             // bit0 = Vulkan Y
    uint reset;              // 1 = ignore history (first frame / view change)
    float4 params;           // x = reject dist, y = EMA alpha, z = firefly clamp max, w = kernel radius
    // History neighbourhood clamp (breaks the mirror lit-history FEEDBACK loop that a plain EMA only
    // low-passes, not kills — a period-2 limit cycle survives an EMA whose loop gain >= 1). Clamps the
    // reprojected history into the current spatial neighbourhood's radiance range before the blend, so
    // a feedback overshoot lands outside the range and is pulled back. Reference SW-GI / reflection
    // denoisers clamp history the same way (mean +- gamma*sigma). 0 = off (byte-identical legacy resolve).
    uint clamp_mode;         // 0 = off, 1 = variance clamp (mean +- clamp_gamma*sigma)
    float clamp_gamma;       // variance-clamp tightness (mode 1); ignored when off
};

[[vk::push_constant]]
ConstantBuffer<ResolvePush> pc : register(b0, space0);

float3 reconstruct(float2 uv, float depth)
{
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 h = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    return h.xyz / h.w;
}
float ggx_d(float ndh, float alpha)
{
    float a2 = alpha * alpha;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-9);
}
// pdf of a reflected direction under GGX NDF importance sampling at this surface.
float ggx_reflect_pdf(float3 n, float3 v, float3 l, float roughness)
{
    float3 h = normalize(v + l);
    float ndh = max(dot(n, h), 1e-4);
    float vdh = max(dot(v, h), 1e-4);
    float alpha = max(roughness, 0.04);
    alpha *= alpha;
    return ggx_d(ndh, alpha) * ndh / (4.0 * vdh);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    uint idx = tid.y * pc.width + tid.x;
    float2 res = float2(pc.width, pc.height);
    float2 uv = (float2(tid.xy) + 0.5) / res;
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;

    if (depth >= 1.0)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
        g.storage_buffers[pc.accum_write].Store4(idx * 16u, asuint(float4(0, 0, 0, 0)));
        g.storage_buffers[pc.pos_write].Store4(idx * 16u, asuint(float4(0, 0, 0, 0)));
        return;
    }

    float3 world_pos = reconstruct(uv, depth);
    float3 n = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);
    float3 v = normalize(pc.cam_pos.xyz - world_pos);
    float roughness = g.textures[pc.material_index].SampleLevel(g.samp, uv, 0.0).g;

    // --- Spatial ratio estimator: borrow neighbour rays, reweight by pdf_p(dir)/pdf_q. ---
    // Roughness-adaptive kernel: a sharp lobe (near-mirror) wants only its own / immediate
    // neighbours (a wide kernel mixes in off-lobe neighbours whose pdf_p is ~0, adding noise
    // and dark misses); a wide lobe pulls the whole neighbourhood to denoise. params.w = max r.
    int rmax = int(pc.params.w);
    int r = clamp(int(ceil(roughness * float(rmax) * 3.0)), 1, rmax);
    float3 num = float3(0, 0, 0);
    float conf_num = 0.0;
    float den = 0.0;
    // Unweighted neighbourhood moments (radiance) for the history clamp box below. Accumulated over the
    // same valid-ray set; only consumed when `clamp_mode != 0`, so OFF stays byte-identical.
    float3 m1 = float3(0, 0, 0);
    float3 m2 = float3(0, 0, 0);
    float mcnt = 0.0;
    for (int dy = -r; dy <= r; dy++)
    {
        for (int dx = -r; dx <= r; dx++)
        {
            float2 nuv = uv + float2(dx, dy) / res;
            float4 b = g.textures[pc.ssr_b_index].SampleLevel(g.samp, nuv, 0.0);
            if (b.w <= 0.0)
                continue; // neighbour ray missed
            float3 l = b.xyz; // its reflected direction
            if (dot(n, l) <= 0.0)
                continue;
            float pdf_p = ggx_reflect_pdf(n, v, l, roughness);
            float w = pdf_p / b.w; // ratio: how likely p would draw this borrowed ray
            float4 a = g.textures[pc.ssr_a_index].SampleLevel(g.samp, nuv, 0.0);
            num += a.rgb * w;
            conf_num += a.a * w;
            den += w;
            m1 += a.rgb;
            m2 += a.rgb * a.rgb;
            mcnt += 1.0;
        }
    }
    bool have_sample = den > 1e-6;
    float3 spatial = have_sample ? num / den : float3(0, 0, 0);
    float conf = have_sample ? saturate(conf_num / den) : 0.0;

    // --- Light temporal: reproject the surface, EMA-blend (disocclusion reject). ---
    float3 accum_rgb = spatial;
    float accum_conf = conf;
    if (pc.reset == 0u)
    {
        float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
        float4 pclip = mul(pc.prev_view_proj, float4(world_pos, 1.0));
        if (pclip.w > 0.0)
        {
            float3 pndc = pclip.xyz / pclip.w;
            float2 puv = float2(pndc.x * 0.5 + 0.5, pndc.y * sy * 0.5 + 0.5);
            if (all(puv >= 0.0) && all(puv <= 1.0))
            {
                int2 ppix = clamp(int2(floor(puv * res)), int2(0, 0),
                                  int2(int(pc.width) - 1, int(pc.height) - 1));
                uint pidx = uint(ppix.y) * pc.width + uint(ppix.x);
                float4 ph = asfloat(g.storage_buffers[pc.accum_read].Load4(pidx * 16u));
                float4 pp = asfloat(g.storage_buffers[pc.pos_read].Load4(pidx * 16u));
                if (pp.w > 0.5 && distance(world_pos, pp.xyz) < pc.params.x)
                {
                    if (have_sample)
                    {
                        // Clamp the (view-dependent, possibly fed-back) reprojected history into this
                        // frame's local reflection range before the EMA — the step that actually kills
                        // the mirror lit-history feedback oscillation the EMA alone only low-passes.
                        // Needs >= 2 samples for a meaningful sigma; else fall back to the raw EMA.
                        float3 hist_rgb = ph.rgb;
                        if (pc.clamp_mode != 0u && mcnt > 1.5)
                        {
                            float3 mean = m1 / mcnt;
                            float3 sigma = sqrt(max(m2 / mcnt - mean * mean, float3(0, 0, 0)));
                            float3 ext = pc.clamp_gamma * sigma;
                            hist_rgb = clamp(hist_rgb, mean - ext, mean + ext);
                        }
                        accum_rgb = lerp(hist_rgb, spatial, pc.params.y);
                        accum_conf = lerp(ph.a, conf, pc.params.y);
                    }
                    else
                    {
                        // No valid ray this frame (every kernel ray missed / off-lobe): a sharp
                        // mirror reflecting off-screen sky hits this every frame. EMA-ing the 0 in
                        // decays the accumulation to black (the dark-metal blotches); instead keep
                        // the history radiance and only bleed confidence down so a persistently
                        // empty surface yields smoothly to the GDF / sky fallback in the composite.
                        accum_rgb = ph.rgb;
                        accum_conf = ph.a * 0.9;
                    }
                }
            }
        }
    }

    // Firefly clamp on the resolved radiance (hue-preserving).
    float m = max(accum_rgb.r, max(accum_rgb.g, accum_rgb.b));
    if (m > pc.params.z)
        accum_rgb *= pc.params.z / m;

    g.storage_buffers[pc.accum_write].Store4(idx * 16u, asuint(float4(accum_rgb, accum_conf)));
    g.storage_buffers[pc.pos_write].Store4(idx * 16u, asuint(float4(world_pos, 1.0)));
    g.storage_images[pc.out_index][tid.xy] = float4(accum_rgb, accum_conf);
}
