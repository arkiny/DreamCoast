// Phase 11 Stage C8b1 — mesh-card surface-cache capture (compute).
//
// Lumen-style surface cache: each scene object is parameterized by 6 axis-aligned
// "cards" (oriented quads on its world AABB faces). This pass captures, once, the
// surface geometry seen by each card texel — for texel (u,v) of a card it starts on the
// card plane and sphere-traces the scene GDF *inward* (along -card.normal) to the first
// object surface, recording the hit's world position (+ a valid flag) and its albedo
// (from the C8a per-voxel color volumes). C8b2 re-lights these texels every frame into a
// radiance atlas; C8b3 looks the radiance up at GI / reflection hits.
//
// The cache lives in flat storage buffers (one float4 / texel), addressed as
// `card * (tile*tile) + ty*tile + tx`, like the GI-denoiser / lit-color history.

#include "bindless.slang"
#include "clipmap.slang"

struct CacheCapturePush
{
    uint cards_index;     // storage buffer of card records (64 B each)
    uint cache_pos_index; // storage buffer out: float4(hit pos, valid)
    uint cache_alb_index; // storage buffer out: float4(albedo rgb, 0)
    uint gdf_sampled;     // volumes[] (SRV) index of the scene GDF (legacy; clipmap below)
    uint num_cards;
    uint tile;            // card atlas tile edge (texels per side)
    uint num_texels;      // num_cards * tile * tile
    uint clip_desc;       // Stage B: clipmap descriptor storage-buffer index
    uint albedo_r;        // C8a albedo channel volumes (0xFFFFFFFF = constant 0.7)
    uint albedo_g;
    uint albedo_b;
    uint clip_count;      // Stage B: clipmap level count (1 = single volume = legacy)
    float4 aabb_min;      // xyz GDF world min, w unused
    float4 aabb_max;      // xyz GDF world max, w = sample distance clamp
};

[[vk::push_constant]]
ConstantBuffer<CacheCapturePush> pc : register(b0, space0);

ClipMap cap_clip()
{
    ClipMap c;
    c.desc = pc.clip_desc;
    c.count = pc.clip_count;
    return c;
}

// Object-only geometry distance (no ground union — cards capture object surfaces). Single
// level ⇒ identical to the legacy single-volume sample.
float geo_march(float3 p)
{
    return cm_geo_march(cap_clip(), p, pc.aabb_max.w);
}

float3 albedo_at(float3 p)
{
    return cm_albedo(cap_clip(), p, 0.7);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cacheMain(uint3 tid: SV_DispatchThreadID)
{
    uint t = tid.x;
    if (t >= pc.num_texels)
        return;

    uint per_card = pc.tile * pc.tile;
    uint card = t / per_card;
    uint local = t % per_card;
    uint tx = local % pc.tile;
    uint ty = local / pc.tile;
    float u = ((float(tx) + 0.5) / float(pc.tile)) * 2.0 - 1.0;
    float v = ((float(ty) + 0.5) / float(pc.tile)) * 2.0 - 1.0;

    // Card record: center.xyz/trace_depth, normal.xyz, u_axis.xyz (half-extent),
    // v_axis.xyz (half-extent). 64 bytes.
    uint base = card * 64u;
    float4 c0 = asfloat(g.storage_buffers[pc.cards_index].Load4(base + 0u));
    float3 normal = asfloat(g.storage_buffers[pc.cards_index].Load3(base + 16u));
    float3 u_axis = asfloat(g.storage_buffers[pc.cards_index].Load3(base + 32u));
    float3 v_axis = asfloat(g.storage_buffers[pc.cards_index].Load3(base + 48u));
    float3 center = c0.xyz;
    float trace_depth = c0.w;

    // Point on the card plane (the AABB face), then sphere-trace inward.
    float3 plane_p = center + u * u_axis + v * v_axis;
    float3 dir = -normal;
    float3 origin = plane_p + normal * (0.01 * trace_depth);
    float max_t = trace_depth * 1.05 + 0.02 * trace_depth;

    float3 hit_pos = float3(0.0, 0.0, 0.0);
    float valid = 0.0;
    float tt = 0.0;
    const int STEPS = 96;
    const float HIT_EPS = 0.0015;
    for (int i = 0; i < STEPS; i++)
    {
        float3 p = origin + dir * tt;
        float d = geo_march(p);
        if (d < HIT_EPS)
        {
            hit_pos = p;
            valid = 1.0;
            break;
        }
        tt += max(d, 0.005);
        if (tt > max_t)
            break;
    }

    float3 albedo = valid > 0.5 ? albedo_at(hit_pos) : float3(0.0, 0.0, 0.0);
    g.storage_buffers[pc.cache_pos_index].Store4(t * 16u, asuint(float4(hit_pos, valid)));
    g.storage_buffers[pc.cache_alb_index].Store4(t * 16u, asuint(float4(albedo, 0.0)));
}
