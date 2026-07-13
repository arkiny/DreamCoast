// Phase 11 Stage C8b1 — mesh-card surface-cache capture (compute).
//
// 레퍼런스식 surface cache: each scene object is parameterized by 6 axis-aligned
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
#include "surface_cache.slang" // C2a: card_layout / layout_find_card helpers

struct CacheCapturePush
{
    uint cards_index;     // storage buffer of card records (64 B each)
    uint cache_pos_index; // storage buffer out: float4(hit pos, valid)
    uint cache_alb_index; // storage buffer out: float4(albedo rgb, 0)
    uint gdf_sampled;     // volumes[] (SRV) index of the scene GDF (legacy; clipmap below)
    uint num_cards;
    uint tile;            // bits 0..7 card atlas tile edge; bits 8..15 C2a layout index + 1 (0 = uniform)
    uint num_texels;      // num_cards * tile * tile
    uint clip_desc;       // Stage B: clipmap descriptor storage-buffer index
    uint albedo_r;        // C8a albedo channel volumes (0xFFFFFFFF = constant 0.7)
    uint albedo_g;
    uint albedo_b;
    uint clip_count;      // Stage B: clipmap level count (1 = single volume = legacy)
    float4 aabb_min;      // xyz GDF world min, w = per-card albedo buffer (C; 0xFFFFFFFF = legacy)
    float4 aabb_max;      // xyz GDF world max, w = sample distance clamp
    // C1 mesh-triangle capture (all 0xFFFFFFFF = off): the consolidated content geometry the
    // HWRT hit-lighting table also reads (object-space vertices 32 B pos/normal/uv, absolute u32
    // indices, 48 B per-drawable material records) + the per-resident-drawable card-instance map
    // (64 B: table row u32, pad3, world->object 3x4 rows). A captured texel then reads its TRUE
    // triangle albedo (closest triangle -> interpolated UV -> base-color texture) + opacity into
    // the atlas .w, instead of one flat stamped colour per drawable.
    uint mesh_vtx;
    uint mesh_idx;
    uint mesh_table;
    uint mesh_card_inst;
    // F1 Stage 3 — page-pool streaming re-capture. When bound (`slot_dirty != 0xFFFFFFFF`) only
    // cards flagged 1 in this buffer are re-traced (a slot the live camera just re-owned); the rest
    // early-out, keeping their captured atlas. The initial full capture passes the sentinel.
    uint slot_dirty;
    uint pad_s0;
    uint pad_s1;
    uint pad_s2;
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

// C1: closest point on triangle abc to p, with barycentrics (Ericson, Real-Time Collision
// Detection 5.1.5) — the capture projects each texel's GDF hit onto the drawable's real
// triangles to read the true interpolated-UV albedo.
float3 closest_tri_point(float3 p, float3 a, float3 b, float3 c, out float3 bary)
{
    float3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { bary = float3(1, 0, 0); return a; }
    float3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { bary = float3(0, 1, 0); return b; }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        float t = d1 / (d1 - d3);
        bary = float3(1.0 - t, t, 0.0);
        return a + ab * t;
    }
    float3 cp = p - c;
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { bary = float3(0, 0, 1); return c; }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        float t = d2 / (d2 - d6);
        bary = float3(1.0 - t, 0.0, t);
        return a + ac * t;
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        float t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        bary = float3(0.0, 1.0 - t, t);
        return b + (c - b) * t;
    }
    float denom = 1.0 / (va + vb + vc);
    float v = vb * denom, w = vc * denom;
    bary = float3(1.0 - v - w, v, w);
    return a + ab * v + ac * w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cacheMain(uint3 tid: SV_DispatchThreadID)
{
    uint t = tid.x;
    if (t >= pc.num_texels)
        return;

    // C2a: variable-tile mapping through the layout buffer (sentinel = uniform tile — the
    // helpers collapse to the exact legacy `t / tile²` arithmetic, byte-identical).
    uint lay8 = (pc.tile >> 8) & 0xFFu;
    uint sc_layout = lay8 == 0u ? 0xFFFFFFFFu : lay8 - 1u;
    uint tile0 = pc.tile & 0xFFu;
    // F1 Stage 1: page-pool reverse-table index (bits 17..28 +1; 0 = no pool; bit 16 is occlusion).
    uint sm12 = (pc.tile >> 17) & 0xFFFu;
    uint sc_slotmap = sm12 == 0u ? 0xFFFFFFFFu : sm12 - 1u;
    uint card = layout_find_card(sc_layout, t, tile0, pc.num_cards, sc_slotmap);
    // F1 Stage 3: streaming re-capture only re-traces the cards the live camera just re-owned; the
    // rest keep their existing atlas texels (the initial capture passes the sentinel = trace all).
    if (pc.slot_dirty != 0xFFFFFFFFu && g.storage_buffers[pc.slot_dirty].Load(card * 4u) == 0u)
        return;
    uint2 lay = card_layout(sc_layout, card, tile0);
    uint res = lay.y;
    uint local = t - lay.x;
    uint tx = local % res;
    uint ty = local / res;
    float u = ((float(tx) + 0.5) / float(res)) * 2.0 - 1.0;
    float v = ((float(ty) + 0.5) / float(res)) * 2.0 - 1.0;

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

    // C: stamp the card's drawable albedo (fine, true surface color) when bound; else fall
    // back to the per-voxel albedo volume (legacy / gallery). The index is packed into the
    // unused `aabb_min.w` slot; 0xFFFFFFFF = legacy. One float3 (12 B) per card.
    float3 albedo = float3(0.0, 0.0, 0.0);
    float opacity = 0.0;
    if (valid > 0.5)
    {
        uint card_albedo_index = asuint(pc.aabb_min.w);
        if (card_albedo_index != 0xFFFFFFFFu)
            albedo = asfloat(g.storage_buffers[card_albedo_index].Load3(card * 12u));
        else
            albedo = albedo_at(hit_pos);

        // C1 mesh-triangle capture: project the GDF hit onto the drawable's REAL triangles and
        // read the interpolated-UV base-color texture — per-texel texture detail (curtain
        // patterns, trim) instead of one flat colour per drawable. One-time pass ⇒ the brute
        // closest-triangle scan is affordable; the flat stamp above stays as the fallback when
        // no triangle lands within a texel-scale radius (GDF hit imprecision).
        if (pc.mesh_table != 0xFFFFFFFFu && pc.mesh_card_inst != 0xFFFFFFFFu)
        {
            uint grp = card / 6u;
            uint ibase = grp * 64u;
            uint row = g.storage_buffers[pc.mesh_card_inst].Load(ibase);
            float4 w2o0 = asfloat(g.storage_buffers[pc.mesh_card_inst].Load4(ibase + 16u));
            float4 w2o1 = asfloat(g.storage_buffers[pc.mesh_card_inst].Load4(ibase + 32u));
            float4 w2o2 = asfloat(g.storage_buffers[pc.mesh_card_inst].Load4(ibase + 48u));
            float4 hw = float4(hit_pos, 1.0);
            float3 po = float3(dot(w2o0, hw), dot(w2o1, hw), dot(w2o2, hw));
            uint rbase = row * 48u;
            uint idx_base = g.storage_buffers[pc.mesh_table].Load(rbase + 0u);
            uint prim_count = g.storage_buffers[pc.mesh_table].Load(rbase + 4u);
            uint tex_albedo = g.storage_buffers[pc.mesh_table].Load(rbase + 8u);
            float4 base_color = asfloat(g.storage_buffers[pc.mesh_table].Load4(rbase + 16u));
            float best_d2 = 1e30;
            float2 best_uv = float2(0.0, 0.0);
            float3 best_q = po; // closest point on the drawable's own triangles (occlusion test)
            float3 best_a = po, best_b = po, best_c = po;
            float2 uv_a = float2(0, 0), uv_b = float2(0, 0), uv_c = float2(0, 0);
            [loop]
            for (uint pr = 0u; pr < prim_count; pr++)
            {
                uint tri = (idx_base + pr * 3u) * 4u;
                uint i0 = g.storage_buffers[pc.mesh_idx].Load(tri);
                uint i1 = g.storage_buffers[pc.mesh_idx].Load(tri + 4u);
                uint i2 = g.storage_buffers[pc.mesh_idx].Load(tri + 8u);
                float3 a = asfloat(g.storage_buffers[pc.mesh_vtx].Load3(i0 * 32u));
                float3 b = asfloat(g.storage_buffers[pc.mesh_vtx].Load3(i1 * 32u));
                float3 c = asfloat(g.storage_buffers[pc.mesh_vtx].Load3(i2 * 32u));
                float3 bary;
                float3 q = closest_tri_point(po, a, b, c, bary);
                float d2 = dot(po - q, po - q);
                if (d2 < best_d2)
                {
                    best_d2 = d2;
                    best_q = q;
                    float2 u0 = asfloat(g.storage_buffers[pc.mesh_vtx].Load2(i0 * 32u + 24u));
                    float2 u1 = asfloat(g.storage_buffers[pc.mesh_vtx].Load2(i1 * 32u + 24u));
                    float2 u2 = asfloat(g.storage_buffers[pc.mesh_vtx].Load2(i2 * 32u + 24u));
                    best_uv = u0 * bary.x + u1 * bary.y + u2 * bary.z;
                    best_a = a; best_b = b; best_c = c;
                    uv_a = u0; uv_b = u1; uv_c = u2;
                }
            }
            // Occlusion invalidation (tile bit16): the sphere-trace runs on the UNION field, so
            // a texel whose own surface is occluded captures the OCCLUDER instead — floor
            // texels under the chrome ball hold the BALL's surface, and reflections near the
            // contact gap answered ~4x over the path tracer. Genuine occlusion needs BOTH:
            // the hit is farther from the card's OWN drawable's triangles than the field blur
            // explains (absolute world tolerance — the per-texel acceptance below is
            // card-relative and lets a whole occluder through on a large card), AND the own
            // surface lies FARTHER ALONG the trace than the hit (something in front of it).
            // The along-trace condition spares texels whose own surface is merely laterally
            // offset by union imprecision (far walls) — invalidating those exposed the bright
            // analytic fallback across whole reflected regions.
            if ((pc.tile & 0x10000u) != 0u && prim_count > 0u)
            {
                float occl_tol = 0.003 * length(pc.aabb_max.xyz - pc.aabb_min.xyz);
                float3 dir_o = normalize(float3(dot(w2o0.xyz, dir), dot(w2o1.xyz, dir),
                                                dot(w2o2.xyz, dir)));
                bool own_behind = dot(best_q - po, dir_o) > occl_tol;
                // valid = -1 (not 0): an OCCLUDER WITNESS — the texel still carries the
                // occluder's hit position, and the reflection sampler's invalid-routing uses
                // that to report "this query point is enclosed by something the cache cannot
                // answer for" (dark fallback, no sky top-up) instead of letting the bilinear
                // renormalise onto the bright neighbouring texels outside the occluder's
                // footprint. Every existing consumer tests `w > 0.5` / `w < 0.5`, so -1 is
                // indistinguishable from the never-captured 0 for them.
                if (best_d2 > occl_tol * occl_tol && own_behind)
                    valid = -1.0;
            }
            // Accept when the nearest triangle is within a few card texels of the GDF hit
            // (object space ≈ world for content's mostly-uniform transforms).
            float texel_w = 2.0 * max(length(u_axis), length(v_axis)) / float(res);
            if (prim_count > 0u && best_d2 < texel_w * texel_w * 16.0)
            {
                float3 col = base_color.rgb;
                float a_out = base_color.a;
                if (tex_albedo != 0xFFFFFFFFu)
                {
                    // LOD: match the texture footprint to a card texel — texels/world-unit from
                    // the winning triangle's UV-vs-object areas.
                    float3 e1 = best_b - best_a, e2 = best_c - best_a;
                    float area_w = max(length(cross(e1, e2)) * 0.5, 1e-8);
                    float2 t1 = uv_b - uv_a, t2 = uv_c - uv_a;
                    float area_uv = max(abs(t1.x * t2.y - t1.y * t2.x) * 0.5, 1e-12);
                    uint tw, th;
                    g.textures[tex_albedo].GetDimensions(tw, th);
                    float texels_per_m = sqrt(area_uv / area_w) * float(max(tw, th));
                    float lod = clamp(log2(max(texel_w * texels_per_m, 1.0)), 0.0, 12.0);
                    float4 s = g.textures[tex_albedo].SampleLevel(g.samp_wrap, best_uv, lod);
                    col *= s.rgb;
                    a_out *= s.a;
                }
                albedo = col;
                opacity = a_out; // .w stays 0.0 outside mesh mode (legacy atlas bit-identical)
            }
        }
    }
    g.storage_buffers[pc.cache_pos_index].Store4(t * 16u, asuint(float4(hit_pos, valid)));
    g.storage_buffers[pc.cache_alb_index].Store4(t * 16u, asuint(float4(albedo, opacity)));
}
