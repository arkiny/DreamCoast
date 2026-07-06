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
// Per-card texel offset of MIP level `L` (1-based) within the pyramid buffer, and its total
// per-card stride — Σ (tile>>l)² for l = 1..maxmip. Mirrors the CPU layout in gdf.rs
// (`card_mip_texels` / `record_cache_mipgen`). tile ≤ 64 ⇒ ≤ 6 iterations.
uint mip_off(uint tile, uint level)
{
    uint off = 0u;
    for (uint l = 1u; l < level; l++)
        off += (tile >> l) * (tile >> l);
    return off;
}
uint mip_stride(uint tile)
{
    uint s = 0u;
    for (uint l = 1u; (tile >> l) >= 1u; l++)
    {
        s += (tile >> l) * (tile >> l);
        if ((tile >> l) == 1u)
            break;
    }
    return s;
}

// One card's cached radiance at (uu,vv) card-space, sampled at the ray-cone MIP level. `half_ext`
// is the card's max half-extent (world) → mip0 texel size; the cone footprint `sample_radius`
// selects the level (reference-engine ComputeSurfaceCacheSample: log2(CardExtent / SampleRadius)),
// bilinearly filtered with uncaptured texels weighted out + renormalised. Coarse levels come from
// the pyramid (`float4(radiance, coverage)`); mip0 from the base atlas + capture-flag validity.
float3 card_mip_sample(uint c, float uu, float vv, float half_ext, float sample_radius, uint tile,
                       uint mip_index, uint max_mip, uint cache_rad_index, uint cache_pos_index)
{
    int mip = 0;
    if (mip_index != 0xFFFFFFFFu && half_ext > 1e-5)
    {
        float texel0_world = (2.0 * half_ext) / float(tile); // mip0 texel size in world units
        float footprint = sample_radius / max(texel0_world, 1e-6);
        float mipf = clamp(log2(max(footprint, 1e-4)), 0.0, float(max_mip));
        mip = int(mipf + 0.5); // nearest level
    }
    if (mip >= 1)
    {
        uint res = max(tile >> uint(mip), 1u);
        uint base = c * mip_stride(tile) + mip_off(tile, uint(mip));
        float fu = (uu * 0.5 + 0.5) * float(res) - 0.5;
        float fv = (vv * 0.5 + 0.5) * float(res) - 0.5;
        float ff = floor(fu), fg = floor(fv);
        float wu = saturate(fu - ff), wv = saturate(fv - fg);
        int mx0 = clamp(int(ff), 0, int(res) - 1);
        int my0 = clamp(int(fg), 0, int(res) - 1);
        int mx1 = min(mx0 + 1, int(res) - 1);
        int my1 = min(my0 + 1, int(res) - 1);
        int mt[4] = { my0 * int(res) + mx0, my0 * int(res) + mx1, my1 * int(res) + mx0,
                      my1 * int(res) + mx1 };
        float mw[4] = { (1.0 - wu) * (1.0 - wv), wu * (1.0 - wv), (1.0 - wu) * wv, wu * wv };
        float3 msum = float3(0.0, 0.0, 0.0);
        float mwsum = 0.0;
        [unroll] for (int i = 0; i < 4; i++)
        {
            float4 mc = asfloat(g.storage_buffers[mip_index].Load4((base + uint(mt[i])) * 16u));
            if (mc.w > 1e-4)
            {
                msum += mw[i] * mc.rgb;
                mwsum += mw[i];
            }
        }
        if (mwsum > 1e-4)
            return msum / mwsum;
        // Coarse level empty here (shouldn't happen for a covered card) → fall to mip0 below.
    }
    // mip0 bilinear (the f1fa851 fetch): 2×2 base texels, skip uncaptured + renormalise.
    uint b0 = c * (tile * tile);
    float fu = (uu * 0.5 + 0.5) * float(tile) - 0.5;
    float fv = (vv * 0.5 + 0.5) * float(tile) - 0.5;
    float ff = floor(fu), fg = floor(fv);
    float wu = saturate(fu - ff), wv = saturate(fv - fg);
    int tx0 = clamp(int(ff), 0, int(tile) - 1);
    int ty0 = clamp(int(fg), 0, int(tile) - 1);
    int tx1 = min(tx0 + 1, int(tile) - 1);
    int ty1 = min(ty0 + 1, int(tile) - 1);
    int bt[4] = { ty0 * int(tile) + tx0, ty0 * int(tile) + tx1, ty1 * int(tile) + tx0,
                  ty1 * int(tile) + tx1 };
    float w4[4] = { (1.0 - wu) * (1.0 - wv), wu * (1.0 - wv), (1.0 - wu) * wv, wu * wv };
    float3 sum = float3(0.0, 0.0, 0.0);
    float wsum = 0.0;
    [unroll] for (int i = 0; i < 4; i++)
    {
        uint tex = b0 + uint(bt[i]);
        float capw = asfloat(g.storage_buffers[cache_pos_index].Load4(tex * 16u)).w;
        if (capw > 0.5)
        {
            sum += w4[i] * asfloat(g.storage_buffers[cache_rad_index].Load3(tex * 16u));
            wsum += w4[i];
        }
    }
    if (wsum > 1e-4)
        return sum / wsum;
    uint tx = min(uint((uu * 0.5 + 0.5) * float(tile)), tile - 1u);
    uint ty = min(uint((vv * 0.5 + 0.5) * float(tile)), tile - 1u);
    return asfloat(g.storage_buffers[cache_rad_index].Load3((b0 + ty * tile + tx) * 16u));
}

// Reference-engine cone-LOD surface-cache lookup. `sample_radius` is the ray-cone footprint at the
// hit (world units, grows with hit distance); `mip_index`/`max_mip` bind the radiance MIP pyramid
// (`mip_index == 0xFFFFFFFF` ⇒ no pyramid). Two blocky-chrome causes are addressed together for a
// demagnified far reflection: (1) WITHIN-card high frequency → a coarse MIP averages the footprint;
// (2) CROSS-card discontinuity (adjacent reflection pixels grazing DIFFERENT cards, each a different
// colour) → accumulate EVERY card covering the imprecise hit, weighted by fit, each at its cone MIP
// (reference-engine multi-card cone accumulation). Near/face-on hits collapse to a single sharp card
// at mip0. Active only for content (`extra_tol > 0`); the primary consumers keep the legacy path.
float3 sample_surface_cache_cone(float3 p, float3 n, uint cards_index, uint cache_pos_index,
                                 uint cache_rad_index, uint num_cards, uint tile, float extra_tol,
                                 float sample_radius, uint mip_index, uint max_mip, out bool found)
{
    found = false;
    int best = -1;
    uint best_texel = 0u;
    float best_score = 0.0;
    float best_uu = 0.0, best_vv = 0.0;
    // Cone mode = content reflection with a pyramid bound. Accumulate multi-card cone samples.
    bool cone_mode = (extra_tol > 0.0) && (mip_index != 0xFFFFFFFFu);
    float3 acc = float3(0.0, 0.0, 0.0);
    float acc_w = 0.0;
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
        if (cone_mode)
        {
            // Accumulate this card's cone-MIP radiance, weighted by fit — averages across the
            // different cards adjacent reflection pixels graze, so the cross-card colour jumps
            // (the residual blocky speckle) smooth out instead of snapping card-to-card.
            float half_ext = max(length(ua), length(va));
            acc += score * card_mip_sample(c, uu, vv, half_ext, sample_radius, tile, mip_index,
                                           max_mip, cache_rad_index, cache_pos_index);
            acc_w += score;
        }
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

    // Content cone mode: the fit-weighted multi-card cone average.
    if (cone_mode && acc_w > 1e-6)
        return acc / acc_w;

    // Content without a pyramid (P11_REFLECT_MIP=0): the single-best-card mip0 bilinear (f1fa851).
    return card_mip_sample(uint(best), best_uu, best_vv, 0.0, 0.0, tile, 0xFFFFFFFFu, 0u,
                           cache_rad_index, cache_pos_index);
}

// Legacy entry point (no cone LOD) — the primary consumers (GI gather / cache re-light /
// viz / world radiance cache) call this so their coverage + sampling stay byte-identical.
// Forwards sentinel MIP args, so `sample_surface_cache_cone` runs the exact pre-MIP path.
float3 sample_surface_cache(float3 p, float3 n, uint cards_index, uint cache_pos_index,
                            uint cache_rad_index, uint num_cards, uint tile, float extra_tol,
                            out bool found)
{
    return sample_surface_cache_cone(p, n, cards_index, cache_pos_index, cache_rad_index, num_cards,
                                     tile, extra_tol, 0.0, 0xFFFFFFFFu, 0u, found);
}
