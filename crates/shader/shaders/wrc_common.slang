// World radiance cache — shared addressing + sampling (P4).
//
// A camera-following clipmap of world probes, each storing incoming radiance as an
// octahedral tile, used as the OFF-SCREEN / FAR-FIELD / infinite-bounce fallback that a
// screen-probe (or cache-update) ray samples when it escapes the traced region. Probe
// placement reuses the existing GDF clipmap descriptor (clipmap.slang: per-level AABBs),
// so the cache inherits the GDF's camera-centered levels for free.
//
// STORAGE — a dense 2D atlas laid out in a persistent byte-address STORAGE BUFFER (a
// ping-pong pair, so the update reads last frame and writes this frame; graph storage images
// are transient, so a device-owned buffer is used for the cross-frame history, matching the
// engine's other temporal histories). No indirection yet (a sparse page table is a later
// refinement noted in docs/world-radiance-cache.md). The atlas escapes the hard limit on 3D
// volume textures: all `levels * grid^3` probe tiles live in ONE buffer. A probe (level L,
// cell x,y,z) with `grid` cells/side and `oct` octahedral texels/side maps to a 2D pixel:
//   px.x = x*oct + octx  (row width = grid*oct)
//   px.y = ((L*grid + z)*grid + y)*oct + octy
// stored at the linear float4 texel index `px.y*(grid*oct) + px.x` (16 bytes/texel).
//
// Requires bindless.slang, clipmap.slang, octahedral.slang included first.

struct Wrc
{
    uint atlas;     // storage_buffers[] index (0xFFFFFFFF = unbound => no cache fallback)
    uint clip_desc; // clipmap descriptor (reuse the GDF clip AABBs for probe placement)
    uint levels;    // clipmap level count
    uint grid;      // probes per side per level
    uint oct;       // octahedral texels per side
};

// Atlas base pixel (top-left) of probe (level, cell)'s octahedral tile.
uint2 wrc_tile_base(uint level, uint3 cell, uint grid, uint oct)
{
    uint ty = (level * grid + cell.z) * grid + cell.y;
    return uint2(cell.x * oct, ty * oct);
}

// Byte offset of the float4 texel at atlas pixel `px` (row width = grid*oct).
uint wrc_texel_offset(uint2 px, uint grid, uint oct)
{
    return (px.y * (grid * oct) + px.x) * 16u;
}

// Read a float4 texel (rgb = radiance, a = per-direction sky visibility) from the atlas buffer.
float4 wrc_load4(uint atlas, uint2 px, uint grid, uint oct)
{
    return asfloat(g.storage_buffers[atlas].Load4(wrc_texel_offset(px, grid, oct)));
}
float3 wrc_load(uint atlas, uint2 px, uint grid, uint oct)
{
    return wrc_load4(atlas, px, grid, oct).rgb;
}

// World-space center of probe (level, cell), from the clipmap level AABB.
float3 wrc_probe_center(Wrc w, uint level, uint3 cell)
{
    ClipMap c;
    c.desc = w.clip_desc;
    c.count = w.levels;
    float3 mn = cm_min(c, level);
    float3 mx = cm_max(c, level);
    return mn + (float3(cell) + 0.5) / float(w.grid) * (mx - mn);
}

// Bilinearly sample probe (level, cell)'s octahedral tile in direction `dir`. Clamps at the
// tile border (no octahedral wrap yet — the low-frequency cache tolerates it).
float3 wrc_sample_tile(Wrc w, uint level, uint3 cell, float3 dir)
{
    uint2 base = wrc_tile_base(level, cell, w.grid, w.oct);
    // Direction -> continuous texel coord in [0, oct]; clamp to tile-center range for bilinear.
    float2 t = oct_dir_to_texel(dir, w.oct) - 0.5;
    t = clamp(t, float2(0.0, 0.0), float2(float(w.oct) - 1.0, float(w.oct) - 1.0));
    int2 t0 = int2(floor(t));
    float2 f = t - float2(t0);
    float3 s = float3(0.0, 0.0, 0.0);
    [unroll]
    for (int j = 0; j < 2; j++)
    {
        [unroll]
        for (int i = 0; i < 2; i++)
        {
            int2 tc = clamp(t0 + int2(i, j), int2(0, 0),
                            int2(int(w.oct) - 1, int(w.oct) - 1));
            float wgt = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
            uint2 px = base + uint2(tc);
            s += wrc_load(w.atlas, px, w.grid, w.oct) * wgt;
        }
    }
    return s;
}

// Sample the cache: the finest clipmap level containing `p`, its nearest probe cell, in the
// direction `dir`. Returns 0 if the cache is unbound or `p` is outside every level.
float3 wrc_sample(Wrc w, float3 p, float3 dir)
{
    if (w.atlas == 0xFFFFFFFFu)
        return float3(0.0, 0.0, 0.0);
    ClipMap c;
    c.desc = w.clip_desc;
    c.count = w.levels;
    for (uint i = 0u; i < w.levels; i++)
    {
        float3 mn = cm_min(c, i);
        float3 mx = cm_max(c, i);
        float3 uvw = (p - mn) / (mx - mn);
        if (all(uvw >= 0.0) && all(uvw <= 1.0))
        {
            uint3 cell = uint3(clamp(int3(floor(uvw * float(w.grid))), int3(0, 0, 0),
                                     int3(int(w.grid) - 1, int(w.grid) - 1, int(w.grid) - 1)));
            return wrc_sample_tile(w, i, cell, dir);
        }
    }
    return float3(0.0, 0.0, 0.0);
}

// Reconstruct the indirect IRRADIANCE the cache holds at world point `p` for a surface with
// normal `n`: the finest clipmap level's nearest probe, its octahedral radiance tile integrated
// over the cosine hemisphere about `n` (same (1/pi)*sum(L*cos*dOmega), dOmega = 4*pi/oct^2
// estimator the screen-probe gather uses). Used by the GI-on-distance-field visualization.
// Returns 0 if the cache is unbound or `p` is outside every level.
float3 wrc_irradiance(Wrc w, float3 p, float3 n)
{
    if (w.atlas == 0xFFFFFFFFu)
        return float3(0.0, 0.0, 0.0);
    ClipMap c;
    c.desc = w.clip_desc;
    c.count = w.levels;
    for (uint i = 0u; i < w.levels; i++)
    {
        float3 mn = cm_min(c, i);
        float3 mx = cm_max(c, i);
        float3 uvw = (p - mn) / (mx - mn);
        if (all(uvw >= 0.0) && all(uvw <= 1.0))
        {
            uint3 cell = uint3(clamp(int3(floor(uvw * float(w.grid))), int3(0, 0, 0),
                                     int3(int(w.grid) - 1, int(w.grid) - 1, int(w.grid) - 1)));
            uint2 base = wrc_tile_base(i, cell, w.grid, w.oct);
            float3 e = float3(0.0, 0.0, 0.0);
            for (uint oy = 0u; oy < w.oct; oy++)
            {
                for (uint ox = 0u; ox < w.oct; ox++)
                {
                    float3 dir = oct_to_dir(oct_texel_to_square(uint2(ox, oy), w.oct));
                    float cw = max(dot(dir, n), 0.0);
                    if (cw <= 0.0)
                        continue;
                    e += wrc_load(w.atlas, base + uint2(ox, oy), w.grid, w.oct) * cw;
                }
            }
            return e * (4.0 / float(w.oct * w.oct));
        }
    }
    return float3(0.0, 0.0, 0.0);
}

// Gather the cache at world point `p` for surface normal `n`: the bounce indirect irradiance
// (`bounce_E`, rgb) AND the cosine-weighted hemispherical SKY VISIBILITY (`skyvis`, in [0,1],
// from the per-direction sky visibility stored in the tile alpha). The reference applies the
// skylight OCCLUDED by how much the surface sees the sky, rather than a flat fill — so a
// visualization / lighting consumer can add `sky * skyvis`. `skyvis` defaults to 1 (fully open)
// when the cache is unbound or `p` is outside every level.
void wrc_gather(Wrc w, float3 p, float3 n, out float3 bounce_E, out float skyvis)
{
    bounce_E = float3(0.0, 0.0, 0.0);
    skyvis = 1.0;
    if (w.atlas == 0xFFFFFFFFu)
        return;
    ClipMap c;
    c.desc = w.clip_desc;
    c.count = w.levels;
    for (uint i = 0u; i < w.levels; i++)
    {
        float3 mn = cm_min(c, i);
        float3 mx = cm_max(c, i);
        float3 uvw = (p - mn) / (mx - mn);
        if (all(uvw >= 0.0) && all(uvw <= 1.0))
        {
            uint3 cell = uint3(clamp(int3(floor(uvw * float(w.grid))), int3(0, 0, 0),
                                     int3(int(w.grid) - 1, int(w.grid) - 1, int(w.grid) - 1)));
            uint2 base = wrc_tile_base(i, cell, w.grid, w.oct);
            float3 e = float3(0.0, 0.0, 0.0);
            float vis_num = 0.0;
            float cos_sum = 0.0;
            for (uint oy = 0u; oy < w.oct; oy++)
            {
                for (uint ox = 0u; ox < w.oct; ox++)
                {
                    float3 dir = oct_to_dir(oct_texel_to_square(uint2(ox, oy), w.oct));
                    float cw = max(dot(dir, n), 0.0);
                    if (cw <= 0.0)
                        continue;
                    float4 s = wrc_load4(w.atlas, base + uint2(ox, oy), w.grid, w.oct);
                    e += s.rgb * cw;
                    vis_num += s.a * cw; // a = per-direction sky visibility
                    cos_sum += cw;
                }
            }
            bounce_E = e * (4.0 / float(w.oct * w.oct));
            skyvis = cos_sum > 0.0 ? vis_num / cos_sum : 1.0;
            return;
        }
    }
}
