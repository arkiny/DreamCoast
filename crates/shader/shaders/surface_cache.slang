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
float3 sample_surface_cache(float3 p, float3 n, uint cards_index, uint cache_pos_index,
                            uint cache_rad_index, uint num_cards, uint tile, out bool found)
{
    found = false;
    int best = -1;
    uint best_texel = 0u;
    float best_score = 0.0;
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
        float tol = max((length(ua) + length(va)) / float(tile) * 4.0, 0.02);
        if (dist > tol)
            continue; // captured a different (farther) surface than the query point

        float score = align / (1.0 + dist * 4.0);
        if (score > best_score)
        {
            best_score = score;
            best = int(c);
            best_texel = texel;
        }
    }
    if (best < 0)
        return float3(0.0, 0.0, 0.0);
    found = true;
    return asfloat(g.storage_buffers[cache_rad_index].Load3(best_texel * 16u));
}
