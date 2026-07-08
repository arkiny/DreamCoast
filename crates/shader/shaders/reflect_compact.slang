// B2 mirror trace compaction — classify + indirect-args passes.
//
// The reflection is traced at a reduced resolution (`reflect_res_div`, Apple 1/6) and
// bilaterally upsampled — correct for the temporally-denoised glossy lobe, but a NEAR-MIRROR
// surface (the chrome ball) shows the trace texels directly: the upsample cannot reconstruct a
// mirror image, so it arrives as blocks no denoiser can sharpen (the information was never
// traced). Tracing everything denser is ~res² cost for pixels that don't need it.
//
// Compaction gives ONLY the near-mirror pixels dense rays (the reference-engine reflection
// pipeline's compacted-trace model): a classify pass appends every near-mirror pixel of the
// refine grid to a list (one atomic append per pixel), an args pass turns the count into a
// `DispatchIndirect` argument, and the `REFLECT_COMPACT` permutation of `gdf_reflect.slang`
// re-traces exactly those pixels at the refine resolution (deterministic mirror rays — the
// same trace/shade path, screen-hit + cache cone included). The composite then prefers the
// refined texels for near-mirror pixels. Cost scales with the MIRROR AREA on screen, not the
// frame: a ball is a few % of pixels; a scene with no mirrors pays two tiny passes and an
// empty dispatch.
//
// Buffer layout (both app-owned, imported as graph externals):
//   list — header u32 (count, mirrored here by csArgs so the compact trace never reads the
//          args buffer while it sits in INDIRECT state) + one u32 per appended pixel:
//          packed coord (x | y << 16), csClassify appends at offset 4 + slot*4.
//   args — 16 B: [group_x, group_y, group_z, count]. csReset zeroes {group_x, count} and
//          seeds {1, 1} for y/z; csClassify atomically bumps `count` (offset 12); csArgs
//          writes `group_x = ceil(count / 64)` (the compact csMain is [numthreads(64,1,1)])
//          and mirrors `count` into the list header, then the host flips the buffer to
//          indirect-args state for `dispatch_indirect`.

#include "bindless.slang"

struct CompactPush
{
    uint depth_index;    // sampled G-buffer depth
    uint material_index; // sampled G-buffer material (g = roughness)
    uint list_index;     // storage buffer: packed near-mirror pixel coords
    uint args_index;     // storage buffer: dispatch args + append counter (see header)
    uint out_index;      // storage image: the refine target (zeroed here; the trace overwrites)
    uint width;          // refine grid width
    uint height;         // refine grid height
    float mirror_thresh; // roughness below this = near-mirror (lockstep with gdf_reflect.slang)
};

[[vk::push_constant]]
ConstantBuffer<CompactPush> pc : register(b0, space0);

[shader("compute")]
[numthreads(1, 1, 1)]
void csReset(uint3 tid: SV_DispatchThreadID)
{
    // group_x = 0 (csArgs writes the real value), group_y/z = 1, count = 0.
    g.storage_buffers[pc.args_index].Store4(0, uint4(0u, 1u, 1u, 0u));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csClassify(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;
    // Zero the refine texel first: composite's validity-weighted fetch keys on `.a > 0`
    // (a traced texel stores the hit distance there), so an un-traced texel must never
    // hold stale memory. The appended texels are overwritten by the compact trace.
    g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 0.0);

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;
    if (depth >= 1.0)
        return;
    float roughness = g.textures[pc.material_index].SampleLevel(g.samp, uv, 0.0).g;
    if (roughness >= pc.mirror_thresh)
        return;
    uint slot;
    g.storage_buffers[pc.args_index].InterlockedAdd(12u, 1u, slot);
    // Capacity is one entry per refine texel (+ header), so the append can never overflow.
    g.storage_buffers[pc.list_index].Store(4u + slot * 4u, tid.x | (tid.y << 16));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csArgs(uint3 tid: SV_DispatchThreadID)
{
    uint count = g.storage_buffers[pc.args_index].Load(12u);
    g.storage_buffers[pc.args_index].Store(0u, (count + 63u) / 64u);
    // Mirror the count into the list header: the compact trace bounds-checks against it there,
    // because by dispatch time the args buffer is in INDIRECT state (not shader-readable on
    // every backend).
    g.storage_buffers[pc.list_index].Store(0u, count);
}
