// World radiance cache — UPDATE pass (P4).
//
// Refreshes the camera-following clipmap of world probes each frame. Every probe re-traces
// its full-sphere octahedral directions into the scene global distance field (GDF) via the
// shared bounce tracer (gdf_bounce.slang) and EMA-accumulates the result into a 2D atlas
// (addressing shared with wrc_common.slang). A ray that ESCAPES the traced region samples the
// PREVIOUS frame's cache at its far point — this couples the cache to itself across frames,
// giving both infinite-bounce build-up and far-field propagation for free (a ray leaving one
// probe picks up radiance another probe already integrated last frame).
//
// Dispatched over the whole atlas: one thread per octahedral texel of every probe. The atlas
// is (grid*oct) wide and (levels*grid*grid*oct) tall; a texel at (tid.x, tid.y) decodes to a
// probe (level, cell) + oct texel using the exact wrc_common.slang layout.

#include "bindless.slang"
#include "clipmap.slang"
#include "surface_cache.slang"
#include "gdf_bounce.slang"
#include "octahedral.slang"
#include "wrc_common.slang"

// 128 bytes. Vectors (float4/uint4) first, then scalars, for stable cross-backend alignment.
struct WrcUpdatePush
{
    float4 sun;           // 0   xyz dir toward sun, w = intensity
    float4 params;        // 16  x=ray_max, y=bias, z=sky_term, w=albedo_fallback
    float4 ground_albedo; // 32  xyz ground material, w=cone_k
    uint4 cache;          // 48  surface cache: cards, cache_pos, cache_rad, num_cards
    uint clip_desc;       // 64  clipmap descriptor storage-buffer index
    uint clip_count;      // 68  clipmap level count (= cache levels)
    uint grid;            // 72  probes per side per level
    uint oct;             // 76  octahedral texels per side
    uint atlas_write;     // 80  storage_buffers[] index (new cache atlas)
    uint atlas_prev;      // 84  storage_buffers[] index (prev cache; 0xFFFFFFFF = none this frame)
    uint cache_tile;      // 88  surface-cache atlas tile edge
    uint max_steps;       // 92  bounce-ray march step cap
    uint frame;           // 96
    uint reset;           // 100 1 = ignore history (first frame)
    float alpha;          // 104 EMA blend weight for the new sample
    float sample_clamp;   // 108 GDF sample distance clamp
    float ground_y;       // 112 analytic ground plane Y
    uint pad0;            // 116
    uint pad1;            // 120
    uint pad2;            // 124
};

[[vk::push_constant]]
ConstantBuffer<WrcUpdatePush> pc : register(b0, space0);

// Build the shared bounce scene from this pass's push constants (mirrors probe_bounce_scene).
BounceScene wrc_bounce_scene()
{
    BounceScene bs;
    bs.clip_desc = pc.clip_desc;
    bs.clip_count = pc.clip_count;
    bs.sample_clamp = pc.sample_clamp;
    bs.ground_y = pc.ground_y;
    bs.sun_dir = pc.sun.xyz;
    bs.sun_i = pc.sun.w;
    bs.ray_max = pc.params.x;
    bs.bias = pc.params.y;
    bs.sky_term = pc.params.z;
    bs.albedo_fallback = pc.params.w;
    bs.cache = pc.cache;
    bs.cache_tile = pc.cache_tile;
    bs.ground_albedo = pc.ground_albedo.xyz;
    bs.cone_k = pc.ground_albedo.w;
    bs.max_steps = pc.max_steps;
    return bs;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    uint atlas_w = pc.grid * pc.oct;
    uint atlas_h = pc.clip_count * pc.grid * pc.grid * pc.oct;
    if (tid.x >= atlas_w || tid.y >= atlas_h)
        return;

    // Decode the tile + oct texel from the atlas pixel using the wrc_common layout:
    //   px.x = cellx*oct + octx
    //   px.y = ((level*grid + cellz)*grid + celly)*oct + octy
    uint octx = tid.x % pc.oct;
    uint cellx = tid.x / pc.oct;
    uint octy = tid.y % pc.oct;
    uint row = tid.y / pc.oct;
    uint cy = row % pc.grid;
    uint tmp = row / pc.grid;
    uint cz = tmp % pc.grid;
    uint level = tmp / pc.grid;
    uint3 cell = uint3(cellx, cy, cz);

    // The cache we sample (prev atlas) + place probes from (shared clip descriptor).
    Wrc w;
    w.atlas = pc.atlas_prev;
    w.clip_desc = pc.clip_desc;
    w.levels = pc.clip_count;
    w.grid = pc.grid;
    w.oct = pc.oct;

    float3 probe_center = wrc_probe_center(w, level, cell);
    float3 dir = oct_to_dir(oct_texel_to_square(uint2(octx, octy), pc.oct));

    // March this probe's direction into the scene GDF.
    BounceScene bs = wrc_bounce_scene();
    float escaped;
    float3 radiance = bs_trace_bounce(bs, probe_center, dir, escaped);

    // Far-field / infinite-bounce: an escaped ray samples the PREVIOUS cache at its far point
    // (unbound prev => leave the escaped 0 so it just reads as dark sky this frame).
    if (escaped > 0.5 && pc.atlas_prev != 0xFFFFFFFFu)
        radiance = wrc_sample(w, probe_center + dir * pc.params.x, dir);

    // Per-direction SKY VISIBILITY: `escaped` = 1 when the ray reached the sky (the cache stores
    // bounce-only radiance, so a consumer adds the skylight OCCLUDED by this). Accumulated in the
    // tile alpha alongside the radiance so the world cache carries indoor sky occlusion, matching
    // the reference (which applies the skylight weighted by how much the surface sees the sky).
    float sky_vis = escaped;

    // EMA temporal accumulation. On reset (first frame) or with no prev atlas bound there is no
    // history to blend against, so the new sample is written straight through.
    float3 outv;
    float outa;
    if (pc.reset != 0u || pc.atlas_prev == 0xFFFFFFFFu)
    {
        outv = radiance;
        outa = sky_vis;
    }
    else
    {
        float4 prev = wrc_load4(pc.atlas_prev, tid.xy, pc.grid, pc.oct);
        outv = lerp(prev.rgb, radiance, pc.alpha);
        outa = lerp(prev.a, sky_vis, pc.alpha);
    }

    uint off = wrc_texel_offset(tid.xy, pc.grid, pc.oct);
    g.storage_buffers[pc.atlas_write].Store4(off, asuint(float4(outv, outa)));
}
