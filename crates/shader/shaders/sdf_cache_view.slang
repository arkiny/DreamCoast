// Phase 11 Stage C8b1 — surface-cache atlas viz (compute).
//
// A validation pass: tiles the captured cards across the screen (each card a vertical
// band, its tile texels mapped to the band) and shows the captured albedo where the
// card texel hit geometry, a dim magenta where it missed. Confirms the mesh cards
// actually capture the scene surfaces + colors before the cache lighting (C8b2) is
// built. Replaces the tonemap source as a standalone viz (`P11_CACHE_VIZ`).

#include "bindless.slang"
#include "surface_cache.slang" // C2a: card_layout helper

struct CacheViewPush
{
    uint cache_pos_index; // storage buffer: float4(hit pos, valid)
    uint cache_src_index; // storage buffer shown: albedo (C8b1) or radiance (C8b2)
    uint out_index;       // storage image
    uint num_cards;
    uint tile;            // bits 0..7 tile edge; bits 8..15 C2a layout index + 1 (0 = uniform)
    uint width;
    uint height;
    uint pad0;
};

[[vk::push_constant]]
ConstantBuffer<CacheViewPush> pc : register(b0, space0);

[shader("compute")]
[numthreads(8, 8, 1)]
void viewMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float cf = uv.x * float(pc.num_cards);
    uint card = min(uint(cf), pc.num_cards - 1u);
    float fx = cf - float(card);
    uint lay8 = (pc.tile >> 8) & 0xFFu;
    uint sc_layout = lay8 == 0u ? 0xFFFFFFFFu : lay8 - 1u;
    uint2 lay = card_layout(sc_layout, card, pc.tile & 0xFFu);
    uint res = lay.y;
    uint tx = min(uint(fx * float(res)), res - 1u);
    uint ty = min(uint(uv.y * float(res)), res - 1u);
    uint t = lay.x + ty * res + tx;

    float4 pos = asfloat(g.storage_buffers[pc.cache_pos_index].Load4(t * 16u));
    float3 src = asfloat(g.storage_buffers[pc.cache_src_index].Load3(t * 16u));
    float3 color = pos.w > 0.5 ? src : float3(0.06, 0.0, 0.06);

    // Thin black gridline between cards for readability.
    if (fx < 0.01)
        color = float3(0.0, 0.0, 0.0);

    g.storage_images[pc.out_index][tid.xy] = float4(color, 1.0);
}
