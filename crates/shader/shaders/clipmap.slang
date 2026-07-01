// Camera-centered GDF clipmap sampling (Scalable-GI Stage B).
//
// One shared definition of "sample the scene distance field / albedo at world `p`" for
// every SW-RT consumer (gdf_gi / gdf_reflect / gdf_trace / gdf_ao / surface cache): pick
// the **finest** level whose AABB contains `p`, else fall through to the **coarsest**
// (which always covers the whole scene). For a single-level clipmap this reduces
// *exactly* to the legacy single-volume sampling — so the gallery stays byte-identical.
//
// Descriptor: a storage buffer, 48 bytes / level, ordered finest → coarsest:
//   offset  0: float4  aabb_min (.xyz; .w unused)
//   offset 16: float4  aabb_max (.xyz; .w unused)
//   offset 32: uint4   (sdf_volume_idx, albedo_r, albedo_g, albedo_b)
// `albedo_r == 0xFFFFFFFF` ⇒ no albedo volume on this clipmap (constant fallback).
//
// Requires "bindless.slang" (the `g` ParameterBlock) included first.
//
// Direct-sample override (per-mesh-sdf-direct-sample-plan.md): when `count == 0`, `desc` is a
// per-mesh-SDF header, not a clipmap level array — the `cm_geo_*` / `cm_albedo` below delegate to
// `mesh_sdf_sample.slang` (per-mesh atlas + cell grid). One switch point ⇒ every SW-RT consumer
// inherits it. `count >= 1` is the legacy dense clipmap path (unchanged; gallery byte-identical).
#include "mesh_sdf_sample.slang"

struct ClipMap
{
    uint desc;  // storage_buffers[] index of the descriptor (above)
    uint count; // level count (1 = single volume = legacy; 0 = direct per-mesh sampling)
};

float3 cm_min(ClipMap c, uint i)
{
    return asfloat(g.storage_buffers[c.desc].Load4(i * 48u + 0u)).xyz;
}
float3 cm_max(ClipMap c, uint i)
{
    return asfloat(g.storage_buffers[c.desc].Load4(i * 48u + 16u)).xyz;
}
uint4 cm_idx(ClipMap c, uint i)
{
    return g.storage_buffers[c.desc].Load4(i * 48u + 32u);
}

// Geometry distance for marching: the finest containing level's trilinear SDF (clamped),
// else the coarsest level's analytic box distance (so the empty bounding box steps
// sensibly). Single level ⇒ identical to the legacy `geo_march`.
float cm_geo_march(ClipMap c, float3 p, float clamp_d)
{
    if (c.count == 0u)
        return ms_geo(c.desc, p, clamp_d);
    for (uint i = 0u; i + 1u < c.count; i++)
    {
        float3 mn = cm_min(c, i);
        float3 mx = cm_max(c, i);
        float3 uvw = (p - mn) / (mx - mn);
        if (all(uvw >= 0.0) && all(uvw <= 1.0))
            return min(g.volumes[cm_idx(c, i).x].SampleLevel(g.samp, uvw, 0.0).x, clamp_d);
    }
    uint last = c.count - 1u;
    float3 mn = cm_min(c, last);
    float3 mx = cm_max(c, last);
    float3 uvw = (p - mn) / (mx - mn);
    if (any(uvw < 0.0) || any(uvw > 1.0))
    {
        float3 q = abs(p - 0.5 * (mn + mx)) - 0.5 * (mx - mn);
        float box = length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
        return max(box, 0.003);
    }
    return min(g.volumes[cm_idx(c, last).x].SampleLevel(g.samp, uvw, 0.0).x, clamp_d);
}

// Occupancy distance (no box fallback): finest containing level's SDF, else 1e3 outside
// every level. Single level ⇒ identical to the legacy `geo_inside`.
float cm_geo_inside(ClipMap c, float3 p, float clamp_d)
{
    if (c.count == 0u)
        return ms_geo(c.desc, p, clamp_d);
    for (uint i = 0u; i + 1u < c.count; i++)
    {
        float3 mn = cm_min(c, i);
        float3 mx = cm_max(c, i);
        float3 uvw = (p - mn) / (mx - mn);
        if (all(uvw >= 0.0) && all(uvw <= 1.0))
            return min(g.volumes[cm_idx(c, i).x].SampleLevel(g.samp, uvw, 0.0).x, clamp_d);
    }
    uint last = c.count - 1u;
    float3 mn = cm_min(c, last);
    float3 mx = cm_max(c, last);
    float3 uvw = (p - mn) / (mx - mn);
    if (any(uvw < 0.0) || any(uvw > 1.0))
        return 1e3;
    return min(g.volumes[cm_idx(c, last).x].SampleLevel(g.samp, uvw, 0.0).x, clamp_d);
}

// Nearest-triangle albedo at `p`: the finest containing level's trilinear color (or the
// constant fallback when that level has no albedo volume). The coarsest level samples
// regardless of range (saturated UVW), matching the legacy `albedo_at`. Single level ⇒
// identical to legacy.
float3 cm_albedo(ClipMap c, float3 p, float constant_fallback)
{
    if (c.count == 0u)
        return ms_albedo(c.desc, p, constant_fallback);
    for (uint i = 0u; i + 1u < c.count; i++)
    {
        float3 mn = cm_min(c, i);
        float3 mx = cm_max(c, i);
        float3 uvw = (p - mn) / (mx - mn);
        if (all(uvw >= 0.0) && all(uvw <= 1.0))
        {
            uint4 idx = cm_idx(c, i);
            if (idx.y == 0xFFFFFFFFu)
                return float3(constant_fallback, constant_fallback, constant_fallback);
            float3 u = saturate(uvw);
            return float3(g.volumes[idx.y].SampleLevel(g.samp, u, 0.0).x,
                          g.volumes[idx.z].SampleLevel(g.samp, u, 0.0).x,
                          g.volumes[idx.w].SampleLevel(g.samp, u, 0.0).x);
        }
    }
    uint last = c.count - 1u;
    uint4 idx = cm_idx(c, last);
    if (idx.y == 0xFFFFFFFFu)
        return float3(constant_fallback, constant_fallback, constant_fallback);
    float3 mn = cm_min(c, last);
    float3 mx = cm_max(c, last);
    float3 u = saturate((p - mn) / (mx - mn));
    return float3(g.volumes[idx.y].SampleLevel(g.samp, u, 0.0).x,
                  g.volumes[idx.z].SampleLevel(g.samp, u, 0.0).x,
                  g.volumes[idx.w].SampleLevel(g.samp, u, 0.0).x);
}

