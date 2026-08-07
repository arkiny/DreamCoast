// Per-mesh SDF **direct sampling** (per-mesh-sdf-direct-sample-plan.md, P2).
//
// Instead of resampling every per-mesh distance field into one dense whole-scene grid (which
// throws away the per-mesh resolution — thin geometry vanishes, the surface cache mis-registers),
// keep each unique mesh's field in the shared atlas volume and sample only the instances near the
// query point. A camera/scene uniform cell grid turns a world point into a short candidate list.
//
// **Hybrid, not a replacement.** Per-mesh fields only know distance near their own surface, so a
// sphere-marcher still needs a global lower bound in open space. We keep the cheap coarse dense
// field for that and take `min(dense, atlas)`: in open space `dense` drives safe long steps; near
// a surface the exact atlas distance pulls the isosurface onto the real mesh, catching thin
// features the coarse field blurs away. The coarse field is never raised in resolution — all the
// precision comes from the atlas.
//
// Wiring: `gdf.rs::install_mesh_sdf` reports the clipmap descriptor as `count == 0` (the
// direct-sample sentinel) with `desc` pointing at the header below; `clipmap.slang`'s `cm_geo_*`
// delegate here on that `0`, so every SW-RT consumer inherits it unchanged.
//
// Requires "bindless.slang" (the `g` ParameterBlock) included first.

// Header storage buffer — byte layout mirrors `gdf.rs::install_mesh_sdf` (read via `Load4`):
//    0: float3 grid_min, uint atlas_sampled_idx
//   16: float3 grid_max, uint res
//   32: uint4  instance_buf, cell_buf, index_buf, instance_count
//   48: uint4  albedo_r, albedo_g, albedo_b, dense_sdf_idx    (dense fallback albedo)
//   64: float3 scene_aabb_min, _
//   80: float3 scene_aabb_max, _
//   96: uint4  alb_atlas_r, alb_atlas_g, alb_atlas_b, alb_atlas_active   (F5: per-mesh albedo)
//  112: uint4  flags, global_latch, _, _   (global field: latch buffer idx, 0xFFFFFFFF = off)
struct MeshSdfHeader
{
    float3 grid_min;
    uint atlas_idx;
    float3 grid_max;
    uint res;
    uint instance_buf;
    uint cell_buf;
    uint index_buf;
    uint instance_count;
    uint alb_r;
    uint alb_g;
    uint alb_b;
    uint dense_idx;
    float3 scene_min;
    float3 scene_max;
    // F5: per-mesh albedo atlas — the R/G/B channel volumes packed with the SDF atlas's tile
    // geometry, so the hit instance's colour is read at the SAME UVW as its distance.
    // `alb_atlas_active == 0` ⇒ no per-mesh albedo atlas, fall back to the dense grid above.
    uint alb_atlas_r;
    uint alb_atlas_g;
    uint alb_atlas_b;
    uint alb_atlas_active;
    // bit0 = detail-replace: where any instance's atlas covers `p`, the atlas union IS the
    // field (the dense term only answers uncovered space). See ms_geo.
    uint flags;
    // Global distance field latch (gdf-scale-follow-plan.md U1): a storage buffer whose
    // uint4 is {live consts idx, active, res, levels}. 0xFFFFFFFF = the legacy dense
    // scene volume below. Indirected so a recenter flip is ONE host word store.
    uint global_latch;
};

// NOTE (lighting-closure P2, measured negative): a per-invocation `static` header cache
// (load once, register compare after) cut D3D12 gi_volume ~3 ms but REGRESSED Vulkan by ~10 ms
// (gi_volume 19->29 ms on the sponza_intel door profile) — the SPIR-V Private-storage struct
// copy did not promote to registers on the NV driver and every cached access paid spill
// latency. Net negative across backends; reverted. The 8 redundant Load4s per march step stay
// until a cross-backend-safe hoist exists (e.g. threading the header by value through ms_geo).
MeshSdfHeader ms_load_header(uint hidx)
{
    RWByteAddressBuffer b = g.storage_buffers[hidx];
    uint4 w0 = b.Load4(0u);
    uint4 w1 = b.Load4(16u);
    uint4 w2 = b.Load4(32u);
    uint4 w3 = b.Load4(48u);
    uint4 w4 = b.Load4(64u);
    uint4 w5 = b.Load4(80u);
    uint4 w6 = b.Load4(96u);
    uint4 w7 = b.Load4(112u);
    MeshSdfHeader h;
    h.grid_min = asfloat(w0.xyz);
    h.atlas_idx = w0.w;
    h.grid_max = asfloat(w1.xyz);
    h.res = w1.w;
    h.instance_buf = w2.x;
    h.cell_buf = w2.y;
    h.index_buf = w2.z;
    h.instance_count = w2.w;
    h.alb_r = w3.x;
    h.alb_g = w3.y;
    h.alb_b = w3.z;
    h.dense_idx = w3.w;
    h.scene_min = asfloat(w4.xyz);
    h.scene_max = asfloat(w5.xyz);
    h.alb_atlas_r = w6.x;
    h.alb_atlas_g = w6.y;
    h.alb_atlas_b = w6.z;
    h.alb_atlas_active = w6.w;
    h.flags = w7.x;
    h.global_latch = w7.y;
    return h;
}

// The grid cell (linear index) containing world point `p` (clamped to the grid).
uint ms_cell_index(MeshSdfHeader h, float3 p)
{
    float3 t = (p - h.grid_min) / max(h.grid_max - h.grid_min, float3(1e-8));
    int3 c = int3(floor(t * float(h.res)));
    c = clamp(c, int3(0, 0, 0), int3(int(h.res) - 1, int(h.res) - 1, int(h.res) - 1));
    return uint(c.x) + h.res * (uint(c.y) + h.res * uint(c.z));
}

// The coarse empty-space field at `p`.
//
// Global-field path (header.global_latch valid): the camera-centered clipmap — pick the
// finest level containing `p` (border-shrunk by half a texel so REPEAT filtering never
// blends the opposite WORLD edge), add the level's toroidal scroll and sample with the
// WRAP sampler; outside every level, the analytic distance to the coarsest box keeps
// the marcher stepping sensibly (the reference contract: contents stay put, addresses
// scroll). Legacy path: the static dense scene volume, bit-for-bit as before.
float ms_dense(MeshSdfHeader h, float3 p)
{
    if (h.global_latch != 0xFFFFFFFFu)
    {
        uint4 latch = g.storage_buffers[h.global_latch].Load4(0u);
        RWByteAddressBuffer c = g.storage_buffers[latch.x];
        float res = float(latch.z);
        for (uint l = 0u; l < latch.w; l++)
        {
            float4 r0 = asfloat(c.Load4(l * 64u + 0u)); // min + voxel
            float4 r1 = asfloat(c.Load4(l * 64u + 16u)); // max + border
            float3 uvw = (p - r0.xyz) / (r1.xyz - r0.xyz);
            float b = r1.w;
            if (any(uvw < b) || any(uvw > 1.0 - b))
                continue;
            uint4 vols = c.Load4(l * 64u + 32u);
            uint4 scr = c.Load4(l * 64u + 48u);
            float3 suvw = uvw + float3(scr.xyz) / res;
            return g.volumes[vols.x].SampleLevel(g.samp_wrap, suvw, 0.0).x;
        }
        uint last = latch.w - 1u;
        float4 r0 = asfloat(c.Load4(last * 64u + 0u));
        float4 r1 = asfloat(c.Load4(last * 64u + 16u));
        float3 q = abs(p - 0.5 * (r0.xyz + r1.xyz)) - 0.5 * (r1.xyz - r0.xyz);
        return max(length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0), 0.01);
    }
    float3 u = saturate((p - h.scene_min) / max(h.scene_max - h.scene_min, float3(1e-8)));
    return g.volumes[h.dense_idx].SampleLevel(g.samp, u, 0.0).x;
}

// Signed distance at world `p`: the coarse dense field, refined by `min` with the exact atlas
// distance of every candidate instance whose local AABB contains `p` (nearest-surface-wins — the
// same union `compose.rs` bakes). `clamp_d` bounds the returned magnitude (matches `cm_geo_*`).
float ms_geo(uint hidx, float3 p, float clamp_d)
{
    MeshSdfHeader h = ms_load_header(hidx);
    float best = min(ms_dense(h, p), clamp_d);
    // Detail-replace (flags bit0): the atlas union over the covering instances, WITHOUT the
    // dense seed. The ~0.75 m dense voxels read d≈0 throughout a contact gap (under the chrome
    // ball) and min() drags the accurate atlas down to them — the exact place mirror rays need
    // the atlas's answer. Where any instance covers `p` the atlas IS the field (the reference
    // engine's detail traces replace its global field the same way); the dense term keeps
    // answering uncovered space (empty-space stepping + the few instances culled from the
    // atlas set).
    float atlas_best = clamp_d;
    bool covered = false;

    uint c = ms_cell_index(h, p);
    uint2 range = g.storage_buffers[h.cell_buf].Load2(c * 8u); // (offset, count)
    for (uint k = 0u; k < range.y; k++)
    {
        uint ii = g.storage_buffers[h.index_buf].Load((range.x + k) * 4u);
        uint base = ii * 112u;
        RWByteAddressBuffer ib = g.storage_buffers[h.instance_buf];
        float4 r0 = asfloat(ib.Load4(base + 0u));
        float4 r1 = asfloat(ib.Load4(base + 16u));
        float4 r2 = asfloat(ib.Load4(base + 32u));
        uint4 a48 = ib.Load4(base + 48u);
        float3 aabb_min = asfloat(a48.xyz);
        float dist_scale = asfloat(a48.w);
        float3 aabb_max = asfloat(ib.Load4(base + 64u)).xyz;
        float3 uvw_bias = asfloat(ib.Load4(base + 80u)).xyz;
        float3 uvw_scale = asfloat(ib.Load4(base + 96u)).xyz;

        float4 pw = float4(p, 1.0);
        float3 lp = float3(dot(r0, pw), dot(r1, pw), dot(r2, pw));
        float3 t = (lp - aabb_min) / max(aabb_max - aabb_min, float3(1e-8));
        if (any(t < 0.0) || any(t > 1.0))
            continue;
        covered = true;
        float3 uvw = uvw_bias + saturate(t) * uvw_scale;
        float d = g.volumes[h.atlas_idx].SampleLevel(g.samp, uvw, 0.0).x * dist_scale;
        // Nearest surface wins (smallest |distance|), matching the CPU composite's union.
        if (abs(d) < abs(best))
            best = d;
        if (abs(d) < abs(atlas_best))
            atlas_best = d;
    }
    if ((h.flags & 1u) != 0u && covered)
        return atlas_best;
    return best;
}

// The coarse dense scene-albedo grid at `p`, or the constant fallback when there are none — the
// F5 fallback path (kept for open-space hits with no candidate instance, and when the per-mesh
// albedo atlas is off).
float3 ms_albedo_dense(MeshSdfHeader h, float3 p, float fallback)
{
    if (h.global_latch != 0xFFFFFFFFu)
    {
        uint4 latch = g.storage_buffers[h.global_latch].Load4(0u);
        RWByteAddressBuffer c = g.storage_buffers[latch.x];
        float res = float(latch.z);
        for (uint l = 0u; l < latch.w; l++)
        {
            float4 r0 = asfloat(c.Load4(l * 64u + 0u));
            float4 r1 = asfloat(c.Load4(l * 64u + 16u));
            float3 uvw = (p - r0.xyz) / (r1.xyz - r0.xyz);
            float b = r1.w;
            if (any(uvw < b) || any(uvw > 1.0 - b))
                continue;
            uint4 vols = c.Load4(l * 64u + 32u);
            uint4 scr = c.Load4(l * 64u + 48u);
            float3 suvw = uvw + float3(scr.xyz) / res;
            return float3(g.volumes[vols.y].SampleLevel(g.samp_wrap, suvw, 0.0).x,
                          g.volumes[vols.z].SampleLevel(g.samp_wrap, suvw, 0.0).x,
                          g.volumes[vols.w].SampleLevel(g.samp_wrap, suvw, 0.0).x);
        }
        return float3(fallback, fallback, fallback);
    }
    if (h.alb_r == 0xFFFFFFFFu)
        return float3(fallback, fallback, fallback);
    float3 u = saturate((p - h.scene_min) / max(h.scene_max - h.scene_min, float3(1e-8)));
    return float3(g.volumes[h.alb_r].SampleLevel(g.samp, u, 0.0).x,
                  g.volumes[h.alb_g].SampleLevel(g.samp, u, 0.0).x,
                  g.volumes[h.alb_b].SampleLevel(g.samp, u, 0.0).x);
}

// Nearest-surface albedo at `p` (F5). When the per-mesh albedo atlas is active, find the same
// nearest-surface candidate instance `ms_geo` uses and read its colour from the atlas at the
// SAME tile UVW — so the hit colour has the per-mesh precision the distance already has (no
// blur across neighbouring meshes). Falls back to the coarse dense grid when the atlas is off,
// or when no candidate instance contains `p` (open space / a mesh culled from the instance set).
float3 ms_albedo(uint hidx, float3 p, float fallback)
{
    MeshSdfHeader h = ms_load_header(hidx);
    if (h.alb_atlas_active == 0u)
        return ms_albedo_dense(h, p, fallback);

    // Track the nearest-surface candidate's atlas UVW (mirrors ms_geo's candidate scan).
    float best_abs = 1e30;
    float3 best_uvw = float3(0.0, 0.0, 0.0);
    bool hit = false;

    uint c = ms_cell_index(h, p);
    uint2 range = g.storage_buffers[h.cell_buf].Load2(c * 8u); // (offset, count)
    for (uint k = 0u; k < range.y; k++)
    {
        uint ii = g.storage_buffers[h.index_buf].Load((range.x + k) * 4u);
        uint base = ii * 112u;
        RWByteAddressBuffer ib = g.storage_buffers[h.instance_buf];
        float4 r0 = asfloat(ib.Load4(base + 0u));
        float4 r1 = asfloat(ib.Load4(base + 16u));
        float4 r2 = asfloat(ib.Load4(base + 32u));
        uint4 a48 = ib.Load4(base + 48u);
        float3 aabb_min = asfloat(a48.xyz);
        float dist_scale = asfloat(a48.w);
        float3 aabb_max = asfloat(ib.Load4(base + 64u)).xyz;
        float3 uvw_bias = asfloat(ib.Load4(base + 80u)).xyz;
        float3 uvw_scale = asfloat(ib.Load4(base + 96u)).xyz;

        float4 pw = float4(p, 1.0);
        float3 lp = float3(dot(r0, pw), dot(r1, pw), dot(r2, pw));
        float3 t = (lp - aabb_min) / max(aabb_max - aabb_min, float3(1e-8));
        if (any(t < 0.0) || any(t > 1.0))
            continue;
        float3 uvw = uvw_bias + saturate(t) * uvw_scale;
        // Nearest surface wins — the SAME criterion (and the same atlas tap) ms_geo uses, so the
        // colour is read from the instance whose distance the marcher hit.
        float d = g.volumes[h.atlas_idx].SampleLevel(g.samp, uvw, 0.0).x * dist_scale;
        if (abs(d) < best_abs)
        {
            best_abs = abs(d);
            best_uvw = uvw;
            hit = true;
        }
    }

    if (!hit)
        return ms_albedo_dense(h, p, fallback);
    return float3(g.volumes[h.alb_atlas_r].SampleLevel(g.samp, best_uvw, 0.0).x,
                  g.volumes[h.alb_atlas_g].SampleLevel(g.samp, best_uvw, 0.0).x,
                  g.volumes[h.alb_atlas_b].SampleLevel(g.samp, best_uvw, 0.0).x);
}

