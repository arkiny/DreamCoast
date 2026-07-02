// Hi-Z (HZB) pyramid build (PR-8, occlusion culling).
//
// A max-reduction depth pyramid. Standard-Z is in use (near = 0, far = 1; depth
// clears to 1.0, `LessEqual` test), so an occluder writes the SMALLER depth and the
// background stays at far (1.0). To test "is an instance fully behind every occluder
// over its screen footprint", the pyramid stores the FARTHEST occluder depth in each
// texel region (`max`) — an instance whose NEAREST point is still farther than that
// max is provably occluded. Empty/background texels stay at 1.0 (far), so they never
// occlude anything (conservative).
//
//  - csCopy   : level 0 <- scene depth. `max` over the source footprint (source may be
//               larger than dest when the HZB base is half-res), clamped to the source
//               extent so out-of-range taps do not pull in a stale value.
//  - csReduce : level n+1 <- level n. `max` over the 2x2 parent footprint, plus the
//               odd-size guard (one extra row/column when the parent dimension is odd
//               so no source texel is dropped — the reduction stays a true max over the
//               whole parent region, which is what keeps the cull conservative).
//
// Single source of truth for the reduction operator: both the build here and the
// occlusion test in cull.slang assume `max`-of-depth + `instance_min > hzb_max`.

struct PushConstants
{
    uint src_index;  // bindless sampled index of the source (scene depth or level n)
    uint dst_index;  // bindless storage index of the destination level
    uint dst_w;      // destination level width  (texels)
    uint dst_h;      // destination level height (texels)
    uint src_w;      // source width  (texels) — for the odd-size / clamp guard
    uint src_h;      // source height (texels)
    uint tap_x;      // taps to reduce along X (footprint width, >= 1)
    uint tap_y;      // taps to reduce along Y (footprint height, >= 1)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

#include "bindless.slang"

// Max-reduce a `tap_x x tap_y` source footprint whose top-left is `base` (source
// texels), clamped to `[0, src_w) x [0, src_h)`. `tap_*` is at most 3 (2x2 + odd guard).
float reduce_max(uint2 base)
{
    float m = 0.0; // near; any real occluder tap raises it
    [unroll]
    for (uint dy = 0; dy < 3; dy++)
    {
        if (dy >= pc.tap_y)
            break;
        [unroll]
        for (uint dx = 0; dx < 3; dx++)
        {
            if (dx >= pc.tap_x)
                break;
            uint sx = min(base.x + dx, pc.src_w - 1);
            uint sy = min(base.y + dy, pc.src_h - 1);
            m = max(m, g.textures[pc.src_index].Load(int3(int(sx), int(sy), 0)).r);
        }
    }
    return m;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csCopy(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dst_w || tid.y >= pc.dst_h)
        return;
    // Map destination texel to its source footprint (fx = src/dst >= 1). Host passes
    // tap_x = fx clamped to <=2 plus the base handles the remainder via clamp.
    uint fx = pc.src_w / max(pc.dst_w, 1u);
    uint fy = pc.src_h / max(pc.dst_h, 1u);
    g.storage_images[pc.dst_index][int2(tid.xy)] =
        float4(reduce_max(uint2(tid.x * fx, tid.y * fy)), 0, 0, 0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csReduce(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dst_w || tid.y >= pc.dst_h)
        return;
    g.storage_images[pc.dst_index][int2(tid.xy)] =
        float4(reduce_max(uint2(tid.x * 2, tid.y * 2)), 0, 0, 0);
}
