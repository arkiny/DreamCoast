// GPU frustum culling (Phase 7): a grid of cube instances is culled on the GPU.
// `csReset` clears the indirect draw-args header; `csCull` tests each instance's
// bounding sphere against the six frustum planes and, for visible ones, appends
// its index to a list and atomically bumps the draw's InstanceCount. The draw
// pass then issues a single `draw_indexed_indirect` over the visible instances.
//
// Buffers are raw byte-address UAVs in the bindless storage-buffer table:
//  - args:    5x u32 VkDrawIndexedIndirectCommand / D3D12_DRAW_INDEXED_ARGUMENTS
//             [index_count, instance_count, first_index, vertex_offset, first_inst]
//  - visible: u32[count] list of visible instance indices

struct PushConstants
{
    float4 planes[6];   // frustum planes (xyz=normal, w=d), inward-facing, normalized
    uint args_index;    // bindless index of the indirect args buffer
    uint visible_index; // bindless index of the visible-list buffer
    uint count;         // total instances
    uint grid_dim;      // instances per row
    float spacing;      // grid cell spacing (world units)
    float cube_radius;  // instance bounding-sphere radius (world units)
    float y_height;     // grid plane height
    uint index_count;   // cube index count (written into args)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

#include "bindless.slang"

// World-space center of grid instance `i`.
float3 instance_center(uint i)
{
    uint gx = i % pc.grid_dim;
    uint gz = i / pc.grid_dim;
    float half = (float(pc.grid_dim) - 1.0) * 0.5;
    return float3((float(gx) - half) * pc.spacing, pc.y_height, (float(gz) - half) * pc.spacing);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csReset(uint3 tid: SV_DispatchThreadID)
{
    // Header: index_count, instance_count=0, first_index=0, vertex_offset=0,
    // first_instance=0.
    RWByteAddressBuffer args = g.storage_buffers[pc.args_index];
    args.Store4(0, uint4(pc.index_count, 0, 0, 0));
    args.Store(16, 0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csCull(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.count)
        return;

    float3 center = instance_center(i);
    bool visible = true;
    [unroll]
    for (int p = 0; p < 6; p++)
    {
        if (dot(pc.planes[p].xyz, center) + pc.planes[p].w < -pc.cube_radius)
            visible = false;
    }
    if (!visible)
        return;

    // Append this instance and bump the draw's InstanceCount (offset 4).
    uint slot;
    g.storage_buffers[pc.args_index].InterlockedAdd(4, 1, slot);
    g.storage_buffers[pc.visible_index].Store(slot * 4, i);
}

// ---------------------------------------------------------------------------
// HZB occlusion culling (PR-8). `csCullHzb` is a SEPARATE 256-byte-push entry so
// the default `csCull` path above stays byte-identical when HZB is disabled. Its
// push is a superset of `PushConstants`: bytes 0..128 are the identical frustum
// block, bytes 128..256 add the occlusion block (view_proj + HZB metadata).
//
// The HZB stores the FARTHEST occluder depth per texel region (max-reduced, see
// hzb_build.slang; standard-Z near=0 far=1). An instance is occluded iff its
// NEAREST screen-space depth is still greater (farther) than the HZB depth over
// its whole screen AABB. We read the coarsest mip whose texel covers the AABB, so
// ONE tap conservatively bounds the region (the canonical single-tap Hi-Z test).
// Conservative by construction: the HZB is the PREVIOUS frame's depth, and any
// ambiguity (behind near plane, off-screen edge, co-planar) keeps it visible.
// ---------------------------------------------------------------------------
struct CullHzbPush
{
    float4 planes[6];    //   0: frustum planes (identical to PushConstants)
    uint args_index;     //  96
    uint visible_index;  // 100
    uint count;          // 104
    uint grid_dim;       // 108
    float spacing;       // 112
    float cube_radius;   // 116
    float y_height;      // 120
    uint index_count;    // 124
    float4 vp[4];        // 128: unjittered, no-Y-flip view_proj (matches frustum)
    uint hzb_base;       // 192: bindless sampled index of HZB level 0
    uint hzb_levels;     // 196: HZB mip count
    uint hzb_w;          // 200: HZB level-0 width  (texels)
    uint hzb_h;          // 204: HZB level-0 height (texels)
    uint enabled;        // 208: 1 = run the occlusion test (0 = frustum only)
    uint stats_index;    // 212: bindless index of the host-visible stats buffer
                         //      ([0]=survived, [1]=occlusion-culled); 0xffffffff = none
    uint _pad1;          // 216
    uint _pad2;          // 220
};

[[vk::push_constant]]
ConstantBuffer<CullHzbPush> hc : register(b0, space0);

// Zero the (survived, occlusion-culled) stats counters. Runs as a 1-thread dispatch
// ordered (via a UAV barrier) BEFORE the cull, so the per-frame count is correct even
// with frames in flight (a host-side zero would race the in-flight GPU writes).
[shader("compute")]
[numthreads(1, 1, 1)]
void csClearStats(uint3 tid: SV_DispatchThreadID)
{
    if (hc.stats_index != 0xffffffffu)
        g.storage_buffers[hc.stats_index].Store2(0, uint2(0, 0));
}

float3 hzb_instance_center(uint i)
{
    uint gx = i % hc.grid_dim;
    uint gz = i / hc.grid_dim;
    float half = (float(hc.grid_dim) - 1.0) * 0.5;
    return float3((float(gx) - half) * hc.spacing, hc.y_height, (float(gz) - half) * hc.spacing);
}

// Project a world point to screen UV (0..1) + NDC depth (0..1, standard-Z).
// Returns false when the point is behind the camera (w <= 0).
bool hzb_project(float3 wp, out float2 uv, out float ndc_z)
{
    float4 c = hc.vp[0] * wp.x + hc.vp[1] * wp.y + hc.vp[2] * wp.z + hc.vp[3];
    uv = float2(0, 0);
    ndc_z = 0;
    if (c.w <= 1e-6)
        return false;
    float3 ndc = c.xyz / c.w;
    // NDC -> texture UV. NDC +Y is UP (screen top) while depth-texture row 0 is ALSO
    // the top, so V flips: uv.y = 0.5 - 0.5*ndc.y. Backend-uniform — the cull matrix
    // is the no-flip view_proj on every backend (Vulkan's clip-space flip is baked
    // into the scene render, so its depth image has the same top-down orientation).
    // (The unflipped V sampled the vertical mirror: a top-of-screen instance read the
    // floor's near depth and false-culled — caught by the conservatism gate.)
    uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    ndc_z = ndc.z; // proj_noflip -> z already in [0,1]
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csCullHzb(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= hc.count)
        return;

    float3 center = hzb_instance_center(i);
    // Frustum test (identical logic to csCull).
    [unroll]
    for (int p = 0; p < 6; p++)
    {
        if (dot(hc.planes[p].xyz, center) + hc.planes[p].w < -hc.cube_radius)
            return;
    }

    // HZB occlusion test. Screen AABB from the 8 world corners (per-axis half-extent =
    // the bounding-sphere radius: a conservative over-estimate that never shrinks the
    // footprint). Track the AABB in UV and the NEAREST NDC depth over the corners.
    if (hc.enabled != 0)
    {
        float r = hc.cube_radius;
        float2 uv_min = float2(1e9, 1e9);
        float2 uv_max = float2(-1e9, -1e9);
        float z_near = 1e9;
        bool any_front = false;
        [unroll]
        for (int cx = -1; cx <= 1; cx += 2)
            [unroll] for (int cy = -1; cy <= 1; cy += 2)
                [unroll] for (int cz = -1; cz <= 1; cz += 2)
                {
                    float3 wp = center + float3(cx, cy, cz) * r;
                    float2 uv;
                    float ndc_z;
                    if (hzb_project(wp, uv, ndc_z))
                    {
                        any_front = true;
                        uv_min = min(uv_min, uv);
                        uv_max = max(uv_max, uv);
                        z_near = min(z_near, ndc_z);
                    }
                }
        // Only cull on a fully-projected, on-screen, in-front AABB (never on a partial
        // projection — that is the conservative fallback that guarantees no false cull).
        bool safe = any_front && uv_min.x >= 0.0 && uv_min.y >= 0.0 && uv_max.x <= 1.0 &&
                    uv_max.y <= 1.0 && z_near >= 0.0;
        if (safe)
        {
            // Coarsest mip whose texel size covers the AABB extent, then the 4-tap
            // (2x2) test on the AABB's CORNER texels: a texel-sized AABB can still
            // straddle a texel boundary, so a single center tap misses the neighbour
            // texel's depths and can false-cull at a depth discontinuity (roof/sky
            // edge — caught by the conservatism gate). If the corner texels span
            // more than 2 per axis (boundary-aligned edge case), go one mip coarser.
            float2 ext_px = (uv_max - uv_min) * float2(hc.hzb_w, hc.hzb_h);
            float max_px = max(ext_px.x, ext_px.y);
            int mip = int(ceil(log2(max(max_px, 1.0))));
            mip = clamp(mip, 0, int(hc.hzb_levels) - 1);
            uint mw = max(hc.hzb_w >> uint(mip), 1u);
            uint mh = max(hc.hzb_h >> uint(mip), 1u);
            int2 t0 = int2(clamp(int(uv_min.x * float(mw)), 0, int(mw) - 1),
                           clamp(int(uv_min.y * float(mh)), 0, int(mh) - 1));
            int2 t1 = int2(clamp(int(uv_max.x * float(mw)), 0, int(mw) - 1),
                           clamp(int(uv_max.y * float(mh)), 0, int(mh) - 1));
            if ((t1.x - t0.x > 1 || t1.y - t0.y > 1) && mip + 1 < int(hc.hzb_levels))
            {
                mip += 1;
                mw = max(hc.hzb_w >> uint(mip), 1u);
                mh = max(hc.hzb_h >> uint(mip), 1u);
                t0 = int2(clamp(int(uv_min.x * float(mw)), 0, int(mw) - 1),
                          clamp(int(uv_min.y * float(mh)), 0, int(mh) - 1));
                t1 = int2(clamp(int(uv_max.x * float(mw)), 0, int(mw) - 1),
                          clamp(int(uv_max.y * float(mh)), 0, int(mh) - 1));
            }
            // HZB mips live in consecutive bindless slots (base + level).
            float hzb_max = g.textures[hc.hzb_base + uint(mip)].Load(int3(t0, 0)).r;
            hzb_max = max(hzb_max,
                          g.textures[hc.hzb_base + uint(mip)].Load(int3(t1.x, t0.y, 0)).r);
            hzb_max = max(hzb_max,
                          g.textures[hc.hzb_base + uint(mip)].Load(int3(t0.x, t1.y, 0)).r);
            hzb_max = max(hzb_max, g.textures[hc.hzb_base + uint(mip)].Load(int3(t1, 0)).r);
            // Occluded iff the nearest point is farther than every occluder in the
            // region. Epsilon keeps a co-planar instance visible (no z-fight false cull).
            if (z_near > hzb_max + 1e-5)
            {
                if (hc.stats_index != 0xffffffffu)
                {
                    uint dummy;
                    g.storage_buffers[hc.stats_index].InterlockedAdd(4, 1, dummy); // occlusion-culled
                }
                return;
            }
        }
    }

    if (hc.stats_index != 0xffffffffu)
    {
        uint dummy;
        g.storage_buffers[hc.stats_index].InterlockedAdd(0, 1, dummy); // survived
    }
    uint slot;
    g.storage_buffers[hc.args_index].InterlockedAdd(4, 1, slot);
    g.storage_buffers[hc.visible_index].Store(slot * 4, i);
}
