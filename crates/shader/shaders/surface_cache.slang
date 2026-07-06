// Phase 11 Stage C8b — shared mesh-card surface-cache lookup.
//
// Given a world point + surface normal, find the best-matching card and return its cached
// radiance (0 on a miss). Used by the cache lighting pass (C8b2, multibounce gather) and
// the GI / reflection consumers (C8b3) — the bindless buffer indices are passed in so each
// caller keeps its own push layout. Card record (64 B): center.xyz/trace_depth, normal.xyz,
// u_axis.xyz (half-extent), v_axis.xyz (half-extent).

// `found` reports whether a card covered the query point; on a miss it is false and the
// caller can fall back (e.g. the analytic ground, which has no cards, would otherwise read
// black). Returns 0 on a miss.
// `extra_tol` widens the on-surface acceptance distance (world units) — 0 for the primary
// consumers (GI gather / cache re-light / viz), so their coverage is byte-identical. A reflection
// ray whose HIT position is imprecise (coarse distance-field march) passes a small positive bias
// so it can still match the card of the surface it grazed instead of falling through to the darker
// analytic fallback (reference-engine surface-cache reflection: a hit reads cached lit radiance).
float3 sample_surface_cache(float3 p, float3 n, uint cards_index, uint cache_pos_index,
                            uint cache_rad_index, uint num_cards, uint tile, float extra_tol,
                            out bool found)
{
    found = false;
    int best = -1;
    uint best_texel = 0u;
    float best_score = 0.0;
    float best_uu = 0.0, best_vv = 0.0;
    for (uint c = 0u; c < num_cards; c++)
    {
        uint b = c * 64u;
        float4 c0 = asfloat(g.storage_buffers[cards_index].Load4(b + 0u));
        float3 cn = asfloat(g.storage_buffers[cards_index].Load3(b + 16u));
        float3 ua = asfloat(g.storage_buffers[cards_index].Load3(b + 32u));
        float3 va = asfloat(g.storage_buffers[cards_index].Load3(b + 48u));

        float align = dot(cn, n);
        if (align < 0.2)
            continue; // card must face roughly the same way as the surface

        float3 rel = p - c0.xyz;
        float uu = dot(rel, ua) / max(dot(ua, ua), 1e-8);
        float vv = dot(rel, va) / max(dot(va, va), 1e-8);
        if (abs(uu) > 1.0 || abs(vv) > 1.0)
            continue; // outside the card footprint

        uint tx = min(uint((uu * 0.5 + 0.5) * float(tile)), tile - 1u);
        uint ty = min(uint((vv * 0.5 + 0.5) * float(tile)), tile - 1u);
        uint texel = c * (tile * tile) + ty * tile + tx;

        float4 capt = asfloat(g.storage_buffers[cache_pos_index].Load4(texel * 16u));
        if (capt.w < 0.5)
            continue; // that texel captured no geometry
        // The query must lie ON the captured surface, within a few texels. Using the card's
        // trace_depth (c0.w) as the tolerance was far too loose: a card on an object's AABB
        // face reaches a full object-depth inward, so a ground point under the cube matched
        // the cube's up-facing top card (dist ~= trace_depth) and the floor reflected the
        // cube's cached colour. A texel-sized tolerance keeps legitimate on-surface hits
        // (even on deep concavities, since this measures distance to the captured point).
        float dist = length(capt.xyz - p);
        float tol = max((length(ua) + length(va)) / float(tile) * 4.0, 0.02) + extra_tol;
        if (dist > tol)
            continue; // captured a different (farther) surface than the query point

        float score = align / (1.0 + dist * 4.0);
        if (score > best_score)
        {
            best_score = score;
            best = int(c);
            best_texel = texel;
            best_uu = uu;
            best_vv = vv;
        }
    }
    if (best < 0)
        return float3(0.0, 0.0, 0.0);
    found = true;

    // NEAREST texel (primary consumers / gallery, extra_tol == 0) → byte-identical.
    if (extra_tol <= 0.0)
        return asfloat(g.storage_buffers[cache_rad_index].Load3(best_texel * 16u));

    // Reference-engine (Lumen) BILINEAR card-atlas fetch (content, extra_tol > 0): a demagnified far
    // reflection maps many pixels onto few cache texels — a NEAREST read then reads them as hard
    // blocks (the blocky chrome centre). Interpolate the best card's 2x2 texel quad, skipping texels
    // that captured no geometry and renormalising by the valid weight, so the reflection reads a
    // smooth cached radiance instead of stair-stepped blocks.
    float fu = (best_uu * 0.5 + 0.5) * float(tile) - 0.5;
    float fv = (best_vv * 0.5 + 0.5) * float(tile) - 0.5;
    float ff = floor(fu), fg = floor(fv);
    float wu = saturate(fu - ff), wv = saturate(fv - fg);
    int tx0 = clamp(int(ff), 0, int(tile) - 1);
    int ty0 = clamp(int(fg), 0, int(tile) - 1);
    int tx1 = min(tx0 + 1, int(tile) - 1);
    int ty1 = min(ty0 + 1, int(tile) - 1);
    int bt2[4] = { ty0 * int(tile) + tx0, ty0 * int(tile) + tx1, ty1 * int(tile) + tx0,
                   ty1 * int(tile) + tx1 };
    float w4[4] = { (1.0 - wu) * (1.0 - wv), wu * (1.0 - wv), (1.0 - wu) * wv, wu * wv };
    uint base = uint(best) * (tile * tile);
    float3 sum = float3(0.0, 0.0, 0.0);
    float wsum = 0.0;
    [unroll] for (int i = 0; i < 4; i++)
    {
        uint tex = base + uint(bt2[i]);
        float capw = asfloat(g.storage_buffers[cache_pos_index].Load4(tex * 16u)).w;
        if (capw > 0.5)
        {
            sum += w4[i] * asfloat(g.storage_buffers[cache_rad_index].Load3(tex * 16u));
            wsum += w4[i];
        }
    }
    if (wsum > 1e-4)
        return sum / wsum;
    return asfloat(g.storage_buffers[cache_rad_index].Load3(best_texel * 16u));
}
