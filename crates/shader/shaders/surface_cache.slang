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

// ---- C2a adaptive card resolution ------------------------------------------------------------
// The atlas is variable-tile when a LAYOUT buffer is bound (16 B per card: mip0 atlas base texel,
// this card's tile edge `res` (pow2), the card's MIP-pyramid base texel, pad). `layout ==
// 0xFFFFFFFF` = the legacy uniform-tile atlas — every helper collapses to the exact legacy
// arithmetic, so consumers stay byte-identical with the feature off.

// (base, res) of card `c` — legacy: (c·tile², tile).
uint2 card_layout(uint layout, uint c, uint tile)
{
    if (layout == 0xFFFFFFFFu)
        return uint2(c * tile * tile, tile);
    return g.storage_buffers[layout].Load2(c * 16u);
}
// The card's MIP-pyramid base texel (level-1 texel 0). Legacy: c · mip_stride(tile).
uint card_mip_base(uint layout, uint c, uint tile)
{
    if (layout == 0xFFFFFFFFu)
        return c * mip_stride(tile);
    return g.storage_buffers[layout].Load(c * 16u + 8u);
}
// texel index -> card, via binary search on the layout's ascending `base` column (the per-texel
// passes dispatch one thread per REAL atlas texel). Legacy: t / tile².
uint layout_find_card(uint layout, uint t, uint tile, uint num_cards)
{
    if (layout == 0xFFFFFFFFu)
        return t / max(tile * tile, 1u);
    uint lo = 0u;
    uint hi = num_cards - 1u;
    [loop]
    while (lo < hi)
    {
        uint mid = (lo + hi + 1u) >> 1u;
        if (g.storage_buffers[layout].Load(mid * 16u) <= t)
            lo = mid;
        else
            hi = mid - 1u;
    }
    return lo;
}

// One card's cached radiance at (uu,vv) card-space, sampled at the ray-cone MIP level. `half_ext`
// is the card's max half-extent (world) → mip0 texel size; the cone footprint `sample_radius`
// selects the level (reference-engine ComputeSurfaceCacheSample: log2(CardExtent / SampleRadius)),
// bilinearly filtered with uncaptured texels weighted out + renormalised. Coarse levels come from
// the pyramid (`float4(radiance, coverage)`); mip0 from the base atlas + capture-flag validity.
// C2a: the card is addressed by its layout triple — `c_base` (mip0 atlas base texel), `c_res`
// (this card's tile edge), `c_mipb` (its pyramid base texel). The legacy uniform atlas passes
// (c·tile², tile, c·mip_stride(tile)) — bit-identical arithmetic.
float3 card_mip_sample(uint c_base, uint c_res, uint c_mipb, float uu, float vv, float half_ext,
                       float sample_radius, uint mip_index, uint max_mip, uint cache_rad_index,
                       uint cache_pos_index)
{
    int mip = 0;
    if (mip_index != 0xFFFFFFFFu && half_ext > 1e-5)
    {
        float texel0_world = (2.0 * half_ext) / float(c_res); // mip0 texel size in world units
        float footprint = sample_radius / max(texel0_world, 1e-6);
        // Clamp to the card's OWN pyramid depth (an adaptive small card has fewer levels).
        float lmax = min(float(max_mip), float(firstbithigh(max(c_res, 1u))));
        float mipf = clamp(log2(max(footprint, 1e-4)), 0.0, lmax);
        mip = int(mipf + 0.5); // nearest level
    }
    if (mip >= 1)
    {
        uint res = max(c_res >> uint(mip), 1u);
        uint base = c_mipb + mip_off(c_res, uint(mip));
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
    uint b0 = c_base;
    float fu = (uu * 0.5 + 0.5) * float(c_res) - 0.5;
    float fv = (vv * 0.5 + 0.5) * float(c_res) - 0.5;
    float ff = floor(fu), fg = floor(fv);
    float wu = saturate(fu - ff), wv = saturate(fv - fg);
    int tx0 = clamp(int(ff), 0, int(c_res) - 1);
    int ty0 = clamp(int(fg), 0, int(c_res) - 1);
    int tx1 = min(tx0 + 1, int(c_res) - 1);
    int ty1 = min(ty0 + 1, int(c_res) - 1);
    int bt[4] = { ty0 * int(c_res) + tx0, ty0 * int(c_res) + tx1, ty1 * int(c_res) + tx0,
                  ty1 * int(c_res) + tx1 };
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
    uint tx = min(uint((uu * 0.5 + 0.5) * float(c_res)), c_res - 1u);
    uint ty = min(uint((vv * 0.5 + 0.5) * float(c_res)), c_res - 1u);
    return asfloat(g.storage_buffers[cache_rad_index].Load3((b0 + ty * c_res + tx) * 16u));
}

// Reference-engine cone-LOD surface-cache lookup. `sample_radius` is the ray-cone footprint at the
// hit (world units, grows with hit distance); `mip_index`/`max_mip` bind the radiance MIP pyramid
// (`mip_index == 0xFFFFFFFF` ⇒ no pyramid). Two blocky-chrome causes are addressed together for a
// demagnified far reflection: (1) WITHIN-card high frequency → a coarse MIP averages the footprint;
// (2) CROSS-card discontinuity (adjacent reflection pixels grazing DIFFERENT cards, each a different
// colour) → accumulate EVERY card covering the imprecise hit, weighted by fit, each at its cone MIP
// (reference-engine multi-card cone accumulation). Near/face-on hits collapse to a single sharp card
// at mip0. Active only for content (`extra_tol > 0`); the primary consumers keep the legacy path.
// `grid_cells`/`grid_pool` (0xFFFFFFFF = off): the Track C card-lookup acceleration grid — a
// uniform world grid the HOST built over the cards, each cell listing every card whose influence
// volume (dilated by the maximum acceptance tolerance this function can apply) overlaps it. The
// query point's cell therefore holds a SUPERSET of the cards the full O(num_cards) scan would
// accept, in the same ascending order → identical results, O(cell) cost. Content reflection only;
// every other consumer passes the sentinel and keeps the exact legacy scan.
// `layout` (0xFFFFFFFF = uniform tile): the C2a adaptive-resolution layout buffer — per-card
// (base, res, mip_base). All addressing below goes through it.
// `feedback` (0xFFFFFFFF = off): the C2b per-card desired-resolution feedback buffer — every
// cone-mode sample InterlockedMax-es the pow2 resolution its footprint WANTED into the card's
// slot, so the host can re-layout the atlas around what reflections actually read (the static
// reference camera's distance prior under-resolves exactly those far cards).
// `mark` (0xFFFFFFFF = off): the mirror-feedback buffer — every card this REFLECTION query
// reads gets its flag set, and the visibility pass folds the flags into the relight priority
// (a mirror mostly reflects OFF-SCREEN cards, which the frustum test relegates to the slow
// HIDDEN_MULT budget; until they relight they shade from the skylight-heavy fallback and the
// mirror shows two colour families during camera motion). Reflection-only (the GI/relight
// consumers pass the sentinel — marking everything would erase the priority).
// `alb_index` (0xFFFFFFFF = off) + `out_alb`: also return the CAPTURED-ALBEDO average over the
// exact texels/weights the radiance sample used. The relight stores
// `radiance = albedo * (direct/PI + indirect)`, so `radiance / out_alb` recovers the card's
// converged multibounce lighting — the parity-grade ambient the HWRT hit lighting modulates
// with its own sharp texture albedo (cache lighting x true material detail).
// `corr` (0xFFFFFFFF = off): the lit-calibration correction — per-card lit/cache luminance
// ratio the visibility pass EMAs (sdf_cache_visibility.slang). Applied to the REFLECTION
// paths only (cone + best-card content); the GI/relight consumers sample uncorrected so the
// radiosity fixed point stays pure and the feedback loop cannot self-amplify.
float3 sample_surface_cache_cone(float3 p, float3 n, uint cards_index, uint cache_pos_index,
                                 uint cache_rad_index, uint num_cards, uint tile, float extra_tol,
                                 float sample_radius, uint mip_index, uint max_mip,
                                 uint grid_cells, uint grid_pool, uint layout, uint feedback,
                                 uint mark, uint alb_index, uint corr, out float3 out_alb,
                                 out bool found)
{
    found = false;
    out_alb = float3(0.0, 0.0, 0.0);
    float3 acc_alb = float3(0.0, 0.0, 0.0);
    int best = -1;
    uint best_texel = 0u;
    float best_score = 0.0;
    float best_uu = 0.0, best_vv = 0.0;
    // Cone mode = content reflection with a pyramid bound. Accumulate multi-card cone samples.
    bool cone_mode = (extra_tol > 0.0) && (mip_index != 0xFFFFFFFFu);
    float3 acc = float3(0.0, 0.0, 0.0);
    float acc_w = 0.0;
    // Card iteration domain: the query cell's card list (grid) or all cards (legacy scan).
    bool use_grid = grid_cells != 0xFFFFFFFFu;
    uint iter_n = num_cards;
    uint pool_off = 0u;
    if (use_grid)
    {
        // Header: gmin.xyz f32, cell_size f32, res.xyz u32, pool len — then (offset,count)/cell.
        float3 gmin = asfloat(g.storage_buffers[grid_cells].Load3(0u));
        float csize = asfloat(g.storage_buffers[grid_cells].Load(12u));
        uint3 res = g.storage_buffers[grid_cells].Load3(16u);
        int3 ci = clamp(int3(floor((p - gmin) / csize)), int3(0, 0, 0),
                        int3(res) - int3(1, 1, 1));
        uint cell = (uint(ci.z) * res.y + uint(ci.y)) * res.x + uint(ci.x);
        uint2 oc = g.storage_buffers[grid_cells].Load2(32u + cell * 8u);
        pool_off = oc.x;
        iter_n = oc.y;
    }
    for (uint k = 0u; k < iter_n; k++)
    {
        uint c = use_grid ? g.storage_buffers[grid_pool].Load((pool_off + k) * 4u) : k;
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

        uint2 lay = card_layout(layout, c, tile); // (mip0 base texel, this card's tile edge)
        uint c_res = lay.y;
        uint tx = min(uint((uu * 0.5 + 0.5) * float(c_res)), c_res - 1u);
        uint ty = min(uint((vv * 0.5 + 0.5) * float(c_res)), c_res - 1u);
        uint texel = lay.x + ty * c_res + tx;

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
        float tol = max((length(ua) + length(va)) / float(c_res) * 4.0, 0.02) + extra_tol;
        if (dist > tol)
            continue; // captured a different (farther) surface than the query point

        float score = align / (1.0 + dist * 4.0);
        if (cone_mode)
        {
            // Accumulate this card's cone-MIP radiance, weighted by fit — averages across the
            // different cards adjacent reflection pixels graze, so the cross-card colour jumps
            // (the residual blocky speckle) smooth out instead of snapping card-to-card.
            float half_ext = max(length(ua), length(va));
            // C2b: record the resolution this footprint WANTED (texel size ≈ sample_radius),
            // rounded up to pow2 in [8, 64]. InterlockedMax keeps the card's peak demand.
            if (feedback != 0xFFFFFFFFu)
            {
                float texels = (2.0 * half_ext) / max(sample_radius, 1e-4);
                uint want = 8u;
                while (want < 64u && float(want) < texels)
                    want <<= 1u;
                g.storage_buffers[feedback].InterlockedMax(c * 4u, want);
            }
            // Mirror feedback: this card feeds a reflection — boost its relight priority.
            if (mark != 0xFFFFFFFFu)
                g.storage_buffers[mark].InterlockedOr(c * 4u, 1u);
            // Captured-albedo average over the same cone weights (mip0 texel; the albedo is a
            // per-card capture with no pyramid, and it is low-frequency at card scale).
            if (alb_index != 0xFFFFFFFFu)
                acc_alb += score * asfloat(g.storage_buffers[alb_index].Load3(texel * 16u));
            float cw = corr != 0xFFFFFFFFu
                           ? clamp(asfloat(g.storage_buffers[corr].Load(texel * 4u)), 0.5, 2.0)
                           : 1.0;
            acc += (score * cw) * card_mip_sample(lay.x, c_res, card_mip_base(layout, c, tile),
                                                  uu, vv, half_ext, sample_radius, mip_index,
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
    // Mirror feedback for the non-cone (single-best-card) content path.
    if (mark != 0xFFFFFFFFu)
        g.storage_buffers[mark].InterlockedOr(uint(best) * 4u, 1u);
    if (alb_index != 0xFFFFFFFFu)
        out_alb = asfloat(g.storage_buffers[alb_index].Load3(best_texel * 16u));

    // NEAREST texel (primary consumers / gallery, extra_tol == 0) → byte-identical.
    if (extra_tol <= 0.0)
        return asfloat(g.storage_buffers[cache_rad_index].Load3(best_texel * 16u));

    // Content cone mode: the fit-weighted multi-card cone average.
    if (cone_mode && acc_w > 1e-6)
    {
        if (alb_index != 0xFFFFFFFFu)
            out_alb = acc_alb / acc_w;
        return acc / acc_w;
    }

    // Content without a pyramid (P11_REFLECT_MIP=0): the single-best-card mip0 bilinear (f1fa851).
    uint2 blay = card_layout(layout, uint(best), tile);
    float bcw = corr != 0xFFFFFFFFu
                    ? clamp(asfloat(g.storage_buffers[corr].Load(best_texel * 4u)), 0.5, 2.0)
                    : 1.0;
    return bcw * card_mip_sample(blay.x, blay.y, 0u, best_uu, best_vv, 0.0, 0.0, 0xFFFFFFFFu, 0u,
                                 cache_rad_index, cache_pos_index);
}

// Legacy entry point (no cone LOD) — the primary consumers (GI gather / cache re-light /
// viz / world radiance cache) call this so their coverage + sampling stay byte-identical.
// Forwards sentinel MIP args, so `sample_surface_cache_cone` runs the exact pre-MIP path.
// `tile` doubles as the C2a carrier: bits 0..7 = the uniform tile edge, bits 8..15 = the
// adaptive layout buffer index + 1 (0 = uniform/legacy) — every legacy caller passes a bare
// tile (≤ 64), so the packed form is backward-compatible without new push fields.
float3 sample_surface_cache(float3 p, float3 n, uint cards_index, uint cache_pos_index,
                            uint cache_rad_index, uint num_cards, uint tile, float extra_tol,
                            out bool found)
{
    uint lay8 = (tile >> 8) & 0xFFu;
    uint layout = lay8 == 0u ? 0xFFFFFFFFu : lay8 - 1u;
    float3 alb_unused;
    return sample_surface_cache_cone(p, n, cards_index, cache_pos_index, cache_rad_index, num_cards,
                                     tile & 0xFFu, extra_tol, 0.0, 0xFFFFFFFFu, 0u, 0xFFFFFFFFu,
                                     0xFFFFFFFFu, layout, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                                     0xFFFFFFFFu, alb_unused, found);
}

// Grid-accelerated variant of the legacy lookup (the relight gather's hot path — its hemisphere
// rays hit geometry and scanned ALL cards per hit). The grid cell holds a SUPERSET of the cards
// the acceptance can pass, in the same ascending order, and the best-pick is order-independent
// for distinct scores (ties keep the first = lowest index in both) → identical results, O(cell).
float3 sample_surface_cache_grid(float3 p, float3 n, uint cards_index, uint cache_pos_index,
                                 uint cache_rad_index, uint num_cards, uint tile, float extra_tol,
                                 uint grid_cells, uint grid_pool, out bool found)
{
    uint lay8 = (tile >> 8) & 0xFFu;
    uint layout = lay8 == 0u ? 0xFFFFFFFFu : lay8 - 1u;
    float3 alb_unused2;
    return sample_surface_cache_cone(p, n, cards_index, cache_pos_index, cache_rad_index, num_cards,
                                     tile & 0xFFu, extra_tol, 0.0, 0xFFFFFFFFu, 0u, grid_cells,
                                     grid_pool, layout, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                                     0xFFFFFFFFu, alb_unused2, found);
}
