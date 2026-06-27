// Stage D2b (Sponza 60fps) — surface-cache card visibility for the amortized relight budget.
//
// One thread per card: test the card's bounding sphere against the camera frustum and write a
// per-card visibility flag. `sdf_cache_light` reads it to relight on-screen cards at the base
// budget period and off-screen cards far less often (the bulk of a dense scene's cards are
// outside any one view). This is the camera-visibility half of UE5 Lumen's surface-cache update
// priority — Lumen drives card updates from feedback (cards actually sampled) + staleness; here
// the frustum test is the deterministic, backend-identical proxy for "is this card on screen",
// and the relight period supplies the staleness term.
//   Ref: D. Wright et al. (Epic Games), "Lumen: Real-time GI in UE5", SIGGRAPH 2022
//        (surface-cache update budget + visibility/priority).
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
    uint out_index;    // storage buffer: 1 uint / card (1 = visible, 0 = off-screen)
    uint num_cards;
    uint pad;
};

[[vk::push_constant]]
ConstantBuffer<CacheVisPush> pc : register(b0, space0);

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    uint c = tid.x;
    if (c >= pc.num_cards)
        return;

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
    g.storage_buffers[pc.out_index].Store(c * 4u, visible);
}
