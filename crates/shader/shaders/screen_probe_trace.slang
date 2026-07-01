// Screen-space radiance probes — P1: placement + trace + octahedral atlas.
//
// The screen is divided into `downsample`-sized tiles; one probe per tile is placed on
// the representative G-buffer surface at the tile center (depth + world normal). Because
// the probe sits ON a visible surface, the placement problems of a world-space grid
// (probes stranded underground or in the open) vanish structurally. Each probe stores its
// incoming radiance as an `oct_res` x `oct_res` octahedral tile in a screen-wide atlas:
// for every texel we decode a world direction and march the SAME shared bounce tracer
// (`gdf_bounce.slang`, also used by the per-pixel GI pass) into the scene distance field.
// A later pass (the per-pixel integrate) gathers these tiles into indirect irradiance.
//
// Dispatched over the atlas: one thread per octahedral texel. Sky / off-surface tiles
// write zero radiance; the integrate pass recomputes probe validity from the same
// depth sample, so no separate validity atlas is needed yet.

#include "bindless.slang"
#include "clipmap.slang"
#include "surface_cache.slang"
#include "gdf_bounce.slang"
#include "octahedral.slang"
#include "wrc_common.slang"

struct ProbeTracePush
{
    float4x4 inv_view_proj; // clip -> world (probe surface reconstruction)
    float4 sun;             // xyz direction toward the sun, w = intensity
    float4 aabb_min;        // xyz GDF world min, w = ground plane Y
    float4 aabb_max;        // xyz GDF world max, w = sample distance clamp
    float4 params;          // x = ray max, y = surface bias, z = sky term, w = albedo fallback
    float4 ground_albedo;   // xyz analytic ground material, w = cone_k
    uint4 cache;            // surface cache: cards, cache_pos, cache_rad, num_cards (cards=~0 => off)
    uint depth_index;       // sampled G-buffer depth
    uint normal_index;      // sampled G-buffer world normal
    uint atlas_index;       // storage-image index (radiance octahedral atlas)
    uint cache_tile;        // surface-cache atlas tile edge
    uint screen_w;          // render extent
    uint screen_h;
    uint probes_x;          // probe grid dims (atlas = probes * oct_res)
    uint probes_y;
    uint downsample;        // screen pixels per probe tile
    uint oct_res;           // octahedral tile resolution (texels per side)
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
    uint frame;             // frame counter (RNG decorrelation)
    uint max_steps;         // bounce-ray march step cap
    uint clip_desc;         // clipmap descriptor storage-buffer index
    uint clip_count;        // clipmap level count
    float clamp_max;        // firefly clamp: max bounce-sample radiance component (1e30 = off)
    uint wrc_atlas;         // world radiance cache atlas (sampled); 0xFFFFFFFF = unbound
    uint wrc_grid;          // world cache probes per side per level
    uint wrc_oct;           // world cache octahedral texels per side
    uint pad0;
};

[[vk::push_constant]]
ConstantBuffer<ProbeTracePush> pc : register(b0, space0);

BounceScene probe_bounce_scene()
{
    BounceScene bs;
    bs.clip_desc = pc.clip_desc;
    bs.clip_count = pc.clip_count;
    bs.sample_clamp = pc.aabb_max.w;
    bs.ground_y = pc.aabb_min.w;
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

// World position + normal of the probe whose tile center maps to `probe` grid coord.
// Returns false for an invalid probe (tile center sees the sky / far plane).
bool probe_surface(uint2 probe, out float3 world_pos, out float3 normal)
{
    // Tile center pixel, clamped to the render extent.
    uint2 px = probe * pc.downsample + pc.downsample / 2u;
    px = min(px, uint2(pc.screen_w - 1u, pc.screen_h - 1u));
    float2 uv = (float2(px) + 0.5) / float2(pc.screen_w, pc.screen_h);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;
    if (depth >= 1.0)
        return false;
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 hpos = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    world_pos = hpos.xyz / hpos.w;
    normal = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);
    return true;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    uint atlas_w = pc.probes_x * pc.oct_res;
    uint atlas_h = pc.probes_y * pc.oct_res;
    if (tid.x >= atlas_w || tid.y >= atlas_h)
        return;

    uint2 probe = tid.xy / pc.oct_res;
    uint2 oct_texel = tid.xy % pc.oct_res;

    float3 world_pos, normal;
    if (!probe_surface(probe, world_pos, normal))
    {
        g.storage_images[pc.atlas_index][tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Decode this texel's world direction (full sphere) and march the shared bounce tracer.
    float2 sq = oct_texel_to_square(oct_texel, pc.oct_res);
    float3 dir = oct_to_dir(sq);

    BounceScene bs = probe_bounce_scene();
    float3 origin = world_pos + normal * pc.params.y;
    float escaped;
    float3 radiance = bs_trace_bounce(bs, origin, dir, escaped);

    // Off-screen / far-field / infinite-bounce fallback: a ray that escaped the local trace
    // (would otherwise contribute 0) reads the world radiance cache at the PROBE ORIGIN in the
    // ray direction. The cache probe nearest this on-surface origin already integrated that
    // direction over the whole scene with multi-bounce (its EMA self-sampling), so it supplies
    // the far-field / multi-bounce radiance the single escaped ray missed. Sampling at the origin
    // (inside the scene) — not the far point (outside every clipmap level) — is what makes the
    // fallback land inside the cache. Unbound (0xFFFFFFFF) => no cache => unchanged.
    if (escaped > 0.5 && pc.wrc_atlas != 0xFFFFFFFFu)
    {
        Wrc w;
        w.atlas = pc.wrc_atlas;
        w.clip_desc = pc.clip_desc;
        w.levels = pc.clip_count;
        w.grid = pc.wrc_grid;
        w.oct = pc.wrc_oct;
        radiance = wrc_sample(w, origin, dir);
    }

    // Firefly clamp (hue-preserving): one bright sample can't dominate the gather.
    float m = max(radiance.r, max(radiance.g, radiance.b));
    if (m > pc.clamp_max)
        radiance *= pc.clamp_max / m;

    // Alpha = per-ray sky visibility (1 = reached sky, 0 = hit geometry). The integrate pass
    // builds the per-pixel hemispherical sky-vis V(n) from these for indoor skylight occlusion.
    g.storage_images[pc.atlas_index][tid.xy] = float4(radiance, escaped);
}
