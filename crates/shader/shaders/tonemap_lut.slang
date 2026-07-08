// Tonemap-LUT bake (reference-engine "combine LUTs" model): bake the whole color pipeline tail
// — ASC-CDL grade + the ACES 1.3 RRT + sRGB ODT (see `aces.slang`) — into one small 3D lookup
// table, unwrapped as a 2D strip of N slices (width N*N, height N; blue = slice). The tonemap
// pass then applies the entire chain as ONE strip fetch per pixel (two bilinear taps + a lerp)
// instead of evaluating curve + grade ALU per Retina pixel, and the LUT entries can carry an
// arbitrarily faithful transform (the full segmented-spline ACES port, not a cheap fit).
//
// Domain: LUT axes are log2-encoded linear scene color (`tonemap_lut_encode`, 2^-12..2^6 —
// both curve ends flat inside the table). Entries store DISPLAY-LINEAR Rec.709 in [0,1]
// (RGBA16Float); the final sRGB encode stays in `post.slang`, whose debug modes keep working
// on display-linear values. Baked per frame: N=48 -> 110K texels, far below one full-screen
// pass, and live grading edits need no invalidation logic.

#include "bindless.slang"
#include "aces.slang"

struct TonemapLutPush
{
    uint out_index; // storage image: the (size*size) x size RGBA16F strip
    uint size;      // LUT resolution N per axis
    uint grade_on;  // 1 = bake the ASC-CDL grade (slope/offset/power) ahead of the curve
    uint pad0;
    // Full float4 rows (MSL vs HLSL/SPIR-V packing; see post.slang's PushConstants note).
    float4 cdl_slope;
    float4 cdl_offset;
    float4 cdl_power;
};

[[vk::push_constant]]
ConstantBuffer<TonemapLutPush> pc : register(b0, space0);

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    uint n = pc.size;
    if (tid.x >= n * n || tid.y >= n)
        return;

    // Strip layout: x = slice * N + red texel, y = green texel, slice = blue texel.
    uint slice = tid.x / n;
    float3 t = float3(float(tid.x - slice * n), float(tid.y), float(slice)) / float(n - 1);

    // LUT axis -> linear scene color (the tonemap input domain, exposure already applied).
    float3 lin = tonemap_lut_decode(t);

    // ASC CDL (out = (slope*in + offset)^power) in linear HDR, exactly as the legacy
    // per-pixel path in post.slang applies it before the curve.
    if (pc.grade_on != 0u)
    {
        float3 graded = pc.cdl_slope.rgb * lin + pc.cdl_offset.rgb;
        lin = pow(max(graded, 0.0), pc.cdl_power.rgb);
    }

    // ACES 1.3 RRT + sRGB(100 nits, dim) ODT -> display-linear [0,1].
    float3 display = aces_rrt_odt(lin);

    g.storage_images[pc.out_index][tid.xy] = float4(display, 1.0);
}
