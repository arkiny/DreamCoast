// Stage D2b (Sponza 60fps) — surface-cache card visibility for the amortized relight budget.
//
// One thread per card: test the card's bounding sphere against the camera frustum and write a
// per-card visibility flag. `sdf_cache_light` reads it to relight on-screen cards at the base
// budget period and off-screen cards far less often (the bulk of a dense scene's cards are
// outside any one view). This is the camera-visibility half of 레퍼런스 엔진 SW GI's surface-cache update
// priority — 레퍼런스 SW-RT GI drives card updates from feedback (cards actually sampled) + staleness; here
// the frustum test is the deterministic, backend-identical proxy for "is this card on screen",
// and the relight period supplies the staleness term.
//   Ref: D. Wright et al., "Real-time GI", SIGGRAPH 2022 (surface-cache update budget +
//        visibility/priority — the reference real-time SW-RT GI system).
//
// The frustum planes are extracted from a Y-flip-free view-projection on the CPU, so DX and VK
// compute the identical visible set (matching the existing GPU-cull convention). The bounding
// sphere is padded by the card's inward trace depth so a card whose captured geometry is visible
// is never culled even when its plane center sits behind the frustum edge (conservative).

#include "bindless.slang"

struct CacheVisPush
{
    float4 planes[6];  // frustum planes (xyz = inward normal, w = d), Y-flip-free
    uint cards_index;  // card records (64 B each)
    uint out_index;    // storage buffer: 1 uint / card (2 = mirror-read, 1 = visible, 0 = off)
    uint num_cards;
    uint marks_index;  // mirror-feedback flags (1 uint / card; 0xFFFFFFFF = off) — merged + cleared
    // Lit-calibration probe (CACHE_VIS_CALIB permutation; corr_index = 0xFFFFFFFF disables):
    // last frame's camera + lit history, the cache atlas, and the per-card correction buffer.
    float4x4 prev_view_proj; // world -> prev-frame clip (the lit history's camera)
    float4 prev_eye;         // xyz prev camera position (TLAS occlusion ray origin),
                             // w = probe stratification seed (frame counter, asuint)
    uint corr_index;         // per-TEXEL correction (1 float32 bits / texel, EMA state)
    uint lit_hist;           // prev-frame full-res lit radiance (float4 / pixel)
    uint cache_pos_index;    // captured texel positions (float4: xyz + valid)
    uint cache_rad_index;    // cache radiance to compare against (the consumers' READ buffer)
    uint tile;               // uniform card tile edge (the calib knob requires the uniform layout)
    uint width;              // lit-history pixel dims
    uint height;
    uint flip_y;             // bit0: Vulkan Y convention (matches gdf_reflect's history fetch)
    // F1 Stage 2 — page-pool LRU clock. When bound (`touched_index != 0xFFFFFFFF`) every on-screen
    // card records the current `frame` here; Stage 3 evicts by `frame - touched[c]` (frames since a
    // card was last on screen / mirror-reflected), so a card a mirror is still reading survives.
    uint touched_index;      // 1 uint / card (0xFFFFFFFF = off = no LRU clock, legacy)
    uint frame;              // current frame counter (the touched timestamp)
};

[[vk::push_constant]]
ConstantBuffer<CacheVisPush> pc : register(b0, space0);

#ifdef CACHE_VIS_CALIB
// Lit-calibration probe (reference-engine sampled-feedback style): an ON-SCREEN texel yields
// BOTH tones for the same surface — the deferred lit history at its projected pixel, and its
// own cached radiance. Their luminance ratio, EMA'd PER TEXEL, is the correction the
// reflection sampler applies — pinning the cache family to the lit family's tone regardless
// of which estimator (skylight/bounce/AO) still disagrees. Per texel, not per card: a large
// card (the floor slab) spans sunlit and shadowed regions whose ratios differ. Scalar
// luminance only: the parity skylight already fixed the hue, and a scalar cannot oscillate
// the colour balance. 64 stratified probes per card per frame (a Weyl stride decorrelated
// per card): every texel of a 32^2 card is revisited every ~16 frames, so the 0.3 EMA
// settles over a couple of seconds and follows lighting/camera changes at the same cadence
// as the relight itself.
void calib_probe(uint c, uint k)
{
    if (pc.corr_index == 0xFFFFFFFFu)
        return;
    uint tt2 = pc.tile * pc.tile;
    uint t0 = c * tt2; // uniform layout — the knob is gated off under the adaptive atlas
    uint seed = asuint(pc.prev_eye.w);
    uint idx = ((seed * 64u + k) * 2654435761u + c * 40503u) % tt2;
    uint texel = t0 + idx;
    float4 capt = asfloat(g.storage_buffers[pc.cache_pos_index].Load4(texel * 16u));
    if (capt.w < 0.5)
        return; // captured no geometry
    float4 clip = mul(pc.prev_view_proj, float4(capt.xyz, 1.0));
    if (clip.w <= 0.0)
        return; // behind the lit history's camera
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float3 ndc = clip.xyz / clip.w;
    float2 uv = float2(ndc.x * 0.5 + 0.5, ndc.y * sy * 0.5 + 0.5);
    // Interior margin: a border pixel's lit value mixes other surfaces at the clamp.
    if (any(uv < 0.02) || any(uv > 0.98))
        return;
    // Occlusion: the probe point must be what the camera actually SAW — one any-hit ray
    // from the history camera; a closer hit means the lit pixel shows an occluder.
    float3 to_p = capt.xyz - pc.prev_eye.xyz;
    float dist = length(to_p);
    RayDesc oray;
    oray.Origin = pc.prev_eye.xyz;
    oray.Direction = to_p / max(dist, 1e-4);
    oray.TMin = 0.001;
    oray.TMax = dist - 0.02;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> oq;
    oq.TraceRayInline(g.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, oray);
    oq.Proceed();
    if (oq.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        return;
    uint2 pix = uint2(clamp(uv * float2(pc.width, pc.height), float2(0.0, 0.0),
                            float2(float(pc.width) - 1.0, float(pc.height) - 1.0)));
    float3 lit =
        asfloat(g.storage_buffers[pc.lit_hist].Load4((pix.y * pc.width + pix.x) * 16u)).rgb;
    float3 rad = asfloat(g.storage_buffers[pc.cache_rad_index].Load3(texel * 16u));
    float3 lw = float3(0.2126, 0.7152, 0.0722);
    float cache_lum = dot(rad, lw);
    float lit_lum = dot(lit, lw);
    if (cache_lum < 1e-4 || !isfinite(cache_lum) || !isfinite(lit_lum))
        return; // unlit/cold texel or a non-finite outlier — no meaningful ratio
    // Clamp the instantaneous ratio hard (an anti-outlier bound: specular highlights in the
    // lit pixel, a mid-relight texel) and EMA; the correction the sampler applies is the
    // converged mean, bounded to [0.5, 2] by construction of the clamp.
    float ratio = clamp(lit_lum / cache_lum, 0.5, 2.0);
    float prev = asfloat(g.storage_buffers[pc.corr_index].Load(texel * 4u));
    float corr = lerp(prev, ratio, 0.3);
    g.storage_buffers[pc.corr_index].Store(texel * 4u, asuint(corr));
}
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
#ifdef CACHE_VIS_CALIB
    // Calib permutation: one thread per PROBE (64/card — the host dispatches num_cards * 64
    // threads), so the occlusion ray queries run in parallel instead of 64-deep serial loops
    // per thread (measured 4.2 ms serial vs sub-ms parallel). Lane 0 of each card also runs
    // the frustum/marks logic below, exactly once per card.
    uint c = tid.x / 64u;
    uint probe = tid.x % 64u;
    if (c >= pc.num_cards)
        return;
    if (probe != 0u)
    {
        calib_probe(c, probe);
        return;
    }
#else
    uint c = tid.x;
    if (c >= pc.num_cards)
        return;
#endif

    uint b = c * 64u;
    float4 c0 = asfloat(g.storage_buffers[pc.cards_index].Load4(b + 0u)); // center.xyz, trace_depth
    float3 ua = asfloat(g.storage_buffers[pc.cards_index].Load3(b + 32u));
    float3 va = asfloat(g.storage_buffers[pc.cards_index].Load3(b + 48u));

    float3 center = c0.xyz;
    // Conservative bounding-sphere radius: the card's in-plane half-extents plus the inward
    // trace depth (captured geometry reaches `trace_depth` in front of the card plane).
    float radius = length(ua) + length(va) + abs(c0.w);

    uint visible = 1u;
    for (int i = 0; i < 6; i++)
    {
        float4 pl = pc.planes[i];
        if (dot(pl.xyz, center) + pl.w < -radius)
        {
            visible = 0u; // fully outside this plane -> off-screen
            break;
        }
    }
    // Mirror feedback: a card the reflection read LAST frame outranks plain frustum visibility
    // (2 -> the relight runs it at an accelerated period; see sdf_cache_light.slang). A mirror
    // mostly reflects off-screen cards — without this they idle on the HIDDEN_MULT budget and
    // shade from the skylight-heavy fallback for seconds after a camera move. Cleared here so
    // the priority follows what the mirror is CURRENTLY reading.
    if (pc.marks_index != 0xFFFFFFFFu)
    {
        if (g.storage_buffers[pc.marks_index].Load(c * 4u) != 0u)
            visible = 2u;
        g.storage_buffers[pc.marks_index].Store(c * 4u, 0u);
    }
    g.storage_buffers[pc.out_index].Store(c * 4u, visible);

    // F1 Stage 2 — page-pool LRU clock: stamp the current frame on every on-screen card (mirror
    // reads count via `visible == 2`). One thread per card, so a plain store (no atomic). A card
    // that goes off-screen keeps its last-seen frame, so Stage 3 evicts by staleness.
    if (pc.touched_index != 0xFFFFFFFFu && visible != 0u)
        g.storage_buffers[pc.touched_index].Store(c * 4u, pc.frame);

#ifdef CACHE_VIS_CALIB
    calib_probe(c, 0u); // lane 0 contributes its probe too (lanes 1..63 returned early above)
#endif
}
