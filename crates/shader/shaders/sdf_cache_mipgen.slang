// Reflection cone-LOD — surface-cache radiance MIP generation.
//
// A demagnified far reflection maps MANY reflection pixels onto FEW mip0 surface-cache texels, so a
// NEAREST (or even bilinear) mip0 read reads them as hard stair-stepped blocks — the blocky chrome
// centre in a large scene. The reference-engine surface-cache samples a ray-CONE footprint: it
// selects a coarse MIP whose 2×2-averaged texel is the mean of the surface over the footprint, and
// the far reflection reads smooth.
//
// This pass builds that MIP pyramid from the freshly-lit mip0 radiance atlas. One dispatch per
// level (1..=mip_levels): each output texel averages its 2×2 parent, weighting OUT uncaptured
// texels and renormalising by the valid weight — exactly the mip0 bilinear rule in
// `surface_cache.slang`, propagated up the pyramid via a per-texel COVERAGE stored in `.w` so a
// coarse texel formed from partly-empty cards stays energy-correct and the sampler can keep
// skipping the still-empty ones. Level 1's parent is mip0 (radiance in `src_rad`, validity in
// `src_pos.w`); higher levels read the pyramid's previous level (`float4(radiance, coverage)`).

#include "bindless.slang"

struct MipGenPush
{
    uint src_rad;       // level 1: mip0 radiance slot; level >1: the pyramid buffer (== dst)
    uint src_pos;       // mip0 captured-geometry buffer (validity `.w`); used at level 1 only
    uint dst;           // the MIP pyramid buffer (write)
    uint num_cards;     // cards in the atlas
    uint level;         // current MIP level, 1-based (1 = first level above mip0)
    uint res_cur;       // this level's card tile edge (tile >> level)
    uint res_prev;      // parent's tile edge (tile >> (level-1)) == 2*res_cur
    uint parent_stride; // per-card texel stride of the parent buffer (tile² at level 1, else dst_stride)
    uint off_prev;      // per-card texel offset of the parent level within its buffer (0 at level 1)
    uint dst_stride;    // per-card texel stride of the pyramid buffer (Σ level texels)
    uint off_cur;       // per-card texel offset of this level within the pyramid buffer
    uint pad;
};

[[vk::push_constant]]
ConstantBuffer<MipGenPush> pc : register(b0, space0);

[shader("compute")]
[numthreads(64, 1, 1)]
void mipMain(uint3 tid: SV_DispatchThreadID)
{
    uint t = tid.x;
    uint per_card = pc.res_cur * pc.res_cur;
    uint total = pc.num_cards * per_card;
    if (t >= total)
        return;

    uint card = t / per_card;
    uint local = t - card * per_card;
    uint tx = local % pc.res_cur;
    uint ty = local / pc.res_cur;

    // 2×2 parent footprint (parent edge is exactly twice this level's).
    uint px0 = tx * 2u, py0 = ty * 2u;
    uint parent_base = card * pc.parent_stride + pc.off_prev;

    float3 sum = float3(0.0, 0.0, 0.0);
    float wsum = 0.0;
    [unroll]
    for (uint j = 0u; j < 4u; j++)
    {
        uint px = px0 + (j & 1u);
        uint py = py0 + (j >> 1u);
        if (px >= pc.res_prev || py >= pc.res_prev)
            continue;
        uint pt = parent_base + py * pc.res_prev + px;
        float w;
        float3 rad;
        if (pc.level == 1u)
        {
            // mip0 parent: radiance from the lit atlas, validity from the captured-geometry flag.
            float validw = asfloat(g.storage_buffers[pc.src_pos].Load4(pt * 16u)).w;
            w = validw > 0.5 ? 1.0 : 0.0;
            rad = asfloat(g.storage_buffers[pc.src_rad].Load3(pt * 16u));
        }
        else
        {
            // pyramid parent: float4(radiance, coverage). Coverage is the weight (uncaptured → 0).
            float4 c = asfloat(g.storage_buffers[pc.src_rad].Load4(pt * 16u));
            w = c.w;
            rad = c.rgb;
        }
        sum += w * rad;
        wsum += w;
    }

    float3 out_rad = wsum > 1e-4 ? sum / wsum : float3(0.0, 0.0, 0.0);
    float coverage = wsum * 0.25; // captured fraction of the 2×2 footprint
    uint dt = card * pc.dst_stride + pc.off_cur + ty * pc.res_cur + tx;
    g.storage_buffers[pc.dst].Store4(dt * 16u, asuint(float4(out_rad, coverage)));
}
