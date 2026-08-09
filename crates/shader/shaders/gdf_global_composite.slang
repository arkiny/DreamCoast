// Global distance field — page composite (gdf-scale-follow-plan.md U1).
//
// One thread per WORLD voxel of the refresh set (a full level at bring-up/invalidate,
// the scrolled-in page rows on recenter): union the culled instances' atlas distances
// by nearest-surface-wins — the same union `compose.rs` bakes on the CPU and `ms_geo`
// evaluates per ray step, evaluated once per voxel into the level volume instead. The
// write lands at the TOROIDAL memory position (world voxel + scroll, mod res), so the
// wrap-sampling consumer sees box and content move together.
//
// Albedo composes in the same pass from the per-mesh albedo atlas at the nearest
// instance (constant where uncovered / atlas absent) — one culled list walk feeds all
// four channel writes.
//
// U2 dual layer: with a SEED source (`seed.x` valid), the union starts from the static
// cache's value at the same toroidal memory voxel instead of `empty`, and the culled
// list holds only the MOVABLE block — a mover's dirty pages recomposite by seeding the
// cached static result and unioning the movers on top, never re-walking static content
// (the reference's mostly-static / movable page split). Without a seed the pass IS the
// static layer composite (or the U1 single-layer full composite).

#include "bindless.slang"

struct GdfCompositePush
{
    // x = instance buffer, y = culled list buffer, z = consts buffer, w = level index.
    uint4 a;
    // x = sdf dst volume (storage), y/z/w = albedo r/g/b dst volumes (0xFFFFFFFF = no
    // albedo compose).
    uint4 dst;
    // x = atlas volume (sampled), y = albedo atlas r, z = g, w = b (0xFFFFFFFF = none).
    uint4 atlas;
    // xyz = refresh box origin (world-voxel coords on the level grid), w = list capacity.
    uint4 box0;
    // xyz = refresh box extents (voxels), w = level res.
    uint4 ext;
    // xyz = level content scroll in VOXELS (already reduced mod res), w = culled list
    // section offset (u32 words; 0 = the static section).
    uint4 scroll;
    // x = empty distance (the level's open-space value), y = constant albedo fallback,
    // zw spare.
    float4 params;
    // x = seed sdf volume (storage; 0xFFFFFFFF = no seed → start from `empty`),
    // y/z/w = seed albedo r/g/b volumes (valid iff x is).
    uint4 seed;
};

[[vk::push_constant]]
ConstantBuffer<GdfCompositePush> pc : register(b0, space0);

[shader("compute")]
[numthreads(4, 4, 4)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid >= pc.ext.xyz))
        return;
    uint res = pc.ext.w;
    uint3 world = pc.box0.xyz + tid;

    RWByteAddressBuffer consts = g.storage_buffers[pc.a.z];
    uint l = pc.a.w;
    float3 lmin = asfloat(consts.Load4(l * 64u + 0u)).xyz;
    float3 lmax = asfloat(consts.Load4(l * 64u + 16u)).xyz;
    float3 p = lmin + (lmax - lmin) * ((float3(world) + 0.5) / float(res));

    // Culled list for this level: [0] count, then indices (see gdf_global_cull.slang).
    RWByteAddressBuffer list = g.storage_buffers[pc.a.y];
    uint lbase = pc.scroll.w + l * (pc.box0.w + 1u);
    uint count = min(list.Load(lbase * 4u), pc.box0.w);

    uint3 mem = (world + pc.scroll.xyz) % res;
    RWByteAddressBuffer inst = g.storage_buffers[pc.a.x];
    float best = pc.seed.x != 0xFFFFFFFFu ? g.storage_volumes[pc.seed.x][mem] : pc.params.x;
    float best_abs = abs(best);
    uint best_i = 0xFFFFFFFFu;
    float3 best_uvw = float3(0.0, 0.0, 0.0);
    for (uint k = 0u; k < count; k++)
    {
        uint ii = list.Load((lbase + 1u + k) * 4u);
        uint base = ii * 112u;
        float4 r0 = asfloat(inst.Load4(base + 0u));
        float4 r1 = asfloat(inst.Load4(base + 16u));
        float4 r2 = asfloat(inst.Load4(base + 32u));
        uint4 a48 = inst.Load4(base + 48u);
        float3 aabb_min = asfloat(a48.xyz);
        float dist_scale = asfloat(a48.w);
        float3 aabb_max = asfloat(inst.Load4(base + 64u)).xyz;
        float3 uvw_bias = asfloat(inst.Load4(base + 80u)).xyz;
        float3 uvw_scale = asfloat(inst.Load4(base + 96u)).xyz;

        float4 pw = float4(p, 1.0);
        float3 lp = float3(dot(r0, pw), dot(r1, pw), dot(r2, pw));
        float3 t = (lp - aabb_min) / max(aabb_max - aabb_min, float3(1e-8));
        if (any(t < 0.0) || any(t > 1.0))
            continue;
        float3 uvw = uvw_bias + saturate(t) * uvw_scale;
        float d = g.volumes[pc.atlas.x].SampleLevel(g.samp, uvw, 0.0).x * dist_scale;
        // Nearest surface wins — the sign comes from the closest well-formed field
        // (the compose.rs union's robustness argument, verbatim).
        if (abs(d) < best_abs)
        {
            best_abs = abs(d);
            best = d;
            best_i = ii;
            best_uvw = uvw;
        }
    }

    g.storage_volumes[pc.dst.x][mem] = best;

    if (pc.dst.y != 0xFFFFFFFFu)
    {
        // Albedo precedence mirrors the distance: a winning instance samples its atlas
        // tile; otherwise a seeded pass carries the static cache's albedo through, and
        // an unseeded pass falls back to the constant.
        float3 alb = float3(pc.params.y, pc.params.y, pc.params.y);
        if (best_i == 0xFFFFFFFFu && pc.seed.x != 0xFFFFFFFFu)
        {
            alb = float3(g.storage_volumes[pc.seed.y][mem],
                         g.storage_volumes[pc.seed.z][mem],
                         g.storage_volumes[pc.seed.w][mem]);
        }
        if (best_i != 0xFFFFFFFFu && pc.atlas.y != 0xFFFFFFFFu)
        {
            alb = float3(g.volumes[pc.atlas.y].SampleLevel(g.samp, best_uvw, 0.0).x,
                         g.volumes[pc.atlas.z].SampleLevel(g.samp, best_uvw, 0.0).x,
                         g.volumes[pc.atlas.w].SampleLevel(g.samp, best_uvw, 0.0).x);
        }
        g.storage_volumes[pc.dst.y][mem] = alb.x;
        g.storage_volumes[pc.dst.z][mem] = alb.y;
        g.storage_volumes[pc.dst.w][mem] = alb.z;
    }
}
