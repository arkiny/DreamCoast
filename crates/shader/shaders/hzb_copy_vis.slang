// Phase 14 vgeo two-phase occlusion — HZB level-0 build from the R64 visibility buffer.
//
// The vgeo scene pass rasters into a packed R64 visibility buffer (u64/pixel) whose HIGH 32
// bits are the depth key `0xFFFFFFFF - asuint(d)` (so `atomicMax` keeps the NEAREST fragment);
// an untouched pixel is 0. To build a max-reduced Hi-Z from the mid-frame phase-1 depth we
// unpack that key back to `d` and `max`-reduce the source footprint, mapping an empty pixel to
// 1.0 (far) so a region with any hole never occludes (conservative). Level 1+ reuse the
// standard `hzb_build.slang::csReduce` (which reads the R32F level as a sampled texture).
//
// Mirrors `hzb_build.slang::csCopy`: `dst` (HZB level 0) is typically half the render extent,
// so each dst texel `max`-reduces an `fx x fy` (≈ 2x2) footprint of the vis buffer.

struct PushConstants
{
    uint vis_buf;    //  0: R64 visibility buffer (u64/pixel, row-major render extent)
    uint dst_index;  //  4: bindless storage-image index of HZB level 0
    uint dst_w;      //  8: level-0 width  (texels)
    uint dst_h;      // 12: level-0 height (texels)
    uint src_w;      // 16: render width   (vis-buffer stride, texels)
    uint src_h;      // 20: render height  (texels)
    uint fx;         // 24: source footprint width  per dst texel (>= 1)
    uint fy;         // 28: source footprint height per dst texel (>= 1)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

#include "bindless.slang"

// Unpack the vis buffer's depth at render pixel (x,y): the HIGH word of the u64 is
// `0xFFFFFFFF - asuint(d)`; an empty pixel (0) maps to 1.0 (far, occludes nothing).
float visDepth(uint x, uint y)
{
    uint idx = y * pc.src_w + x;
    uint key = g.storage_buffers[pc.vis_buf].Load2(idx * 8).y; // .y = high 32 bits (depth key)
    if (key == 0u)
        return 1.0; // empty pixel -> far
    return asfloat(0xFFFFFFFFu - key);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csCopyVis(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dst_w || tid.y >= pc.dst_h)
        return;
    uint bx = tid.x * pc.fx;
    uint by = tid.y * pc.fy;
    float m = 0.0; // near; any real occluder / empty(far) tap raises it
    [unroll]
    for (uint dy = 0; dy < 2; dy++)
    {
        if (dy >= pc.fy)
            break;
        [unroll]
        for (uint dx = 0; dx < 2; dx++)
        {
            if (dx >= pc.fx)
                break;
            uint sx = min(bx + dx, pc.src_w - 1);
            uint sy = min(by + dy, pc.src_h - 1);
            m = max(m, visDepth(sx, sy));
        }
    }
    g.storage_images[pc.dst_index][int2(tid.xy)] = float4(m, 0, 0, 0);
}
