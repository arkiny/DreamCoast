// GI fine-level content shift (the F4B recenter, made toroidal-style).
//
// When the camera-anchored fine box recenters, the OLD park-and-reconverge window
// (consumers dropped to the coarse level for two super-cycles) read as a one-frame GI
// blink at a fixed world position — the dead-zone boundary (playtest report). The box
// snap is voxel-lattice aligned by construction, so recentering is exactly an INTEGER
// texel shift of the fine half: physically moving the content keeps box and data
// coherent in the same frame — no park, no EMA reset, no reconvergence, and the EMA
// history the next update reads is already in new-box coordinates.
//
// In-place shift is not expressible in one parallel dispatch (threads would read
// neighbours mid-overwrite), so the host records two passes through a scratch set:
// slot -> scratch applies the shift, scratch -> slot copies back. Out-of-range sources
// clamp to the border texel (GI is smooth; a stretched border beats a dark reseed, and
// those cells sit under the edge-fade margin anyway).

#include "bindless.slang"

struct GiShiftPush
{
    // x = source volume storage base, y = destination storage base, z = volume count
    // (12 radiance SH / 4 sky-vis SH), w = the SOURCE side's fine-row offset
    // (GI_VOL_DIM inside a double-height slot, 0 inside the fine-sized scratch).
    uint4 a;
    // xyz = source texel offset in fine-level-local texels: dst[i] = src[i + shift]
    // (shift = (new_min - old_min) / voxel, integral by the recenter snap).
    // w = the DESTINATION side's fine-row offset.
    int4 shift;
    uint4 dims; // xyz = fine level probe grid resolution
};

[[vk::push_constant]]
ConstantBuffer<GiShiftPush> pc : register(b0, space0);

[shader("compute")]
[numthreads(4, 4, 4)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid >= pc.dims.xyz))
        return;
    int3 src = clamp(int3(tid) + pc.shift.xyz, int3(0, 0, 0), int3(pc.dims.xyz) - 1);
    uint3 s = uint3(src) + uint3(0u, pc.a.w, 0u);
    uint3 d = tid + uint3(0u, uint(pc.shift.w), 0u);
    for (uint v = 0u; v < pc.a.z; v++)
        g.storage_volumes[pc.a.y + v][d] = g.storage_volumes[pc.a.x + v][s];
}
