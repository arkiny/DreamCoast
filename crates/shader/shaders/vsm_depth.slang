// VSM caster rasterization (plan V1): attachmentless raster over a level's 8k² virtual
// viewport; the fragment stage translates virtual -> physical through the page table and
// InterlockedMin's ortho depth into the pool buffer (asuint(z), nearest-to-light wins —
// commutative, so raster order cannot break DX≡VK determinism). Pixels on unmapped pages
// discard: the raster cost of a caster is its covered virtual area, the WRITE cost only
// its mapped-page area (page caching, V2+, is what shrinks the former).
//
// Orientation contract (vsm_common.slang): `mvp` is FLIP-FREE on every backend; the
// vertex shader applies the Vulkan clip-Y flip itself, so SV_Position lands on the same
// virtual texel row everywhere and the PS/page math is backend-uniform.
//
// The vertex variants mirror shadow.slang (static / skinned / morphed) so every caster
// type the legacy map handles can enter the VSM from day one.

#include "bindless.slang"
#include "vsm_common.slang"

static const uint NO_TEXTURE = 0xFFFFFFFF;

struct VSInput
{
    float3 pos : POSITION;
    float2 uv : TEXCOORD;
};

struct VSOutput
{
    float4 clip : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    // Rows 0-2: the flip-free level world->clip x model AFFINE transform (the level
    // projections are orthographic, so the true w row is (0,0,0,1) by construction and
    // the VS reconstructs w = 1). Row 3 REPURPOSES that dead space as the caster's
    // whole-draw cull key: {sphere NDC x, sphere NDC y, sphere NDC radius, unused}.
    float4x4 mvp;
    uint base_color_tex; // masked casters: base-color bindless index (NO_TEXTURE if absent)
    float alpha_cutoff;  // |cutoff| alpha-test threshold (0 = opaque), sign ignored here
    uint flip_y;         // 1 = Vulkan (VS negates clip.y)
    uint level;          // clipmap level being rasterized
    uint table_buf; // page table storage index
    uint pool_buf;  // physical pool storage index
    uint counter_buf; // stats + per-level RENDER bounds (VSM_BOUNDS_BASE rows)
    uint _pad;
    uint4 skin;  // vsMainSkinned only: joints / weights / palette / count
    uint4 morph; // vsMainMorphed only: deltas / weights / target_count / vertex_count
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

// Whole-caster cull against this frame's RENDER-page bounds (written by the pages
// pass earlier this frame — no stale-readback shadow holes by construction). A caster
// whose bounding sphere overlaps no RENDER page can only produce PS discards; the VS
// collapses it instead, which removes the raster fill that made vsm_raster the top
// pass. The decision is UNIFORM per draw (sphere and bounds are per-draw constants),
// so triangles are collapsed whole, never partially.
bool vsm_caster_culled()
{
    uint b = VSM_BOUNDS_BASE + pc.level * 16u;
    RWByteAddressBuffer counter = g.storage_buffers[pc.counter_buf];
    uint minx = counter.Load(b + 0u);
    uint maxx = counter.Load(b + 8u);
    if (minx > maxx)
        return true; // no RENDER pages on this level this frame
    uint miny = counter.Load(b + 4u);
    uint maxy = counter.Load(b + 12u);
    // Bounds pages -> NDC box (page p spans [p, p+1) / VSM_TABLE_DIM of the grid).
    // Page X grows WITH ndc x, but page Y is INVERTED (vsm_ndc_to_texel: ndc.y = +1
    // is row 0) — mirror the Y interval or the test culls against the wrong half,
    // which truncated character self-shadows in play (their RENDER pages sit at the
    // true y, the un-mirrored sphere landed at the mirrored one).
    float inv = 2.0 / float(VSM_TABLE_DIM);
    float lo_x = float(minx) * inv - 1.0;
    float hi_x = float(maxx + 1u) * inv - 1.0;
    float lo_y = 1.0 - float(maxy + 1u) * inv;
    float hi_y = 1.0 - float(miny) * inv;
    float sx = pc.mvp[3].x;
    float sy = pc.mvp[3].y;
    float sr = pc.mvp[3].z;
    return sx + sr < lo_x || sx - sr > hi_x || sy + sr < lo_y || sy - sr > hi_y;
}

float4 vsm_clip(float4 world)
{
    if (vsm_caster_culled())
        return float4(4.0, 4.0, 4.0, 1.0); // uniform: every vertex lands here -> clipped
    // Affine rows (see PushConstants): w reconstructs to 1.
    float4 clip = float4(dot(pc.mvp[0], world), dot(pc.mvp[1], world), dot(pc.mvp[2], world), 1.0);
    if (pc.flip_y != 0u)
        clip.y = -clip.y;
    return clip;
}

[shader("vertex")]
VSOutput vsMain(VSInput input)
{
    VSOutput o;
    o.clip = vsm_clip(float4(input.pos, 1.0));
    o.uv = input.uv;
    return o;
}

// Column-major M*v against a joint palette entry (matches shadow.slang / gbuffer.slang).
float4 vsm_joint_mul(uint palette_index, uint j, float4 v)
{
    RWByteAddressBuffer palette = g.storage_buffers[palette_index];
    uint base = j * 64;
    float4 c0 = palette.Load<float4>(base + 0);
    float4 c1 = palette.Load<float4>(base + 16);
    float4 c2 = palette.Load<float4>(base + 32);
    float4 c3 = palette.Load<float4>(base + 48);
    return c0 * v.x + c1 * v.y + c2 * v.z + c3 * v.w;
}

[shader("vertex")]
VSOutput vsMainSkinned(VSInput input, uint vid: SV_VertexID)
{
    RWByteAddressBuffer joints_buf = g.storage_buffers[pc.skin.x];
    RWByteAddressBuffer weights_buf = g.storage_buffers[pc.skin.y];
    uint4 j = joints_buf.Load<uint4>(vid * 16);
    float4 w = weights_buf.Load<float4>(vid * 16);
    uint p = pc.skin.z;

    float4 bind_pos = float4(input.pos, 1.0);
    float4 pos = w.x * vsm_joint_mul(p, j.x, bind_pos) + w.y * vsm_joint_mul(p, j.y, bind_pos)
               + w.z * vsm_joint_mul(p, j.z, bind_pos) + w.w * vsm_joint_mul(p, j.w, bind_pos);

    VSOutput o;
    o.clip = vsm_clip(pos);
    o.uv = input.uv;
    return o;
}

[shader("vertex")]
VSOutput vsMainMorphed(VSInput input, uint vid: SV_VertexID)
{
    RWByteAddressBuffer deltas = g.storage_buffers[pc.morph.x];
    RWByteAddressBuffer weights = g.storage_buffers[pc.morph.y];
    uint target_count = pc.morph.z;
    uint vertex_count = pc.morph.w;

    float3 pos = input.pos;
    for (uint t = 0; t < target_count; ++t)
    {
        float w = weights.Load<float>(t * 4);
        if (w != 0.0)
        {
            uint base = (t * vertex_count + vid) * 32;
            pos += w * deltas.Load<float4>(base + 0).xyz;
        }
    }

    VSOutput o;
    o.clip = vsm_clip(float4(pos, 1.0));
    o.uv = input.uv;
    return o;
}

[shader("fragment")]
void fsMain(VSOutput input)
{
    // Masked casters punch the same holes as the lit geometry (hard |cutoff| test — the
    // map isn't TAA-resolved; mirrors shadow.slang).
    float cutoff = abs(pc.alpha_cutoff);
    if (cutoff > 0.0 && pc.base_color_tex != NO_TEXTURE)
    {
        float a = g.textures[pc.base_color_tex].Sample(g.samp_wrap, input.uv).a;
        if (a < cutoff)
            discard;
    }

    uint2 texel = uint2(input.clip.xy); // viewport = the full virtual grid
    uint2 page = texel / VSM_PAGE_SIZE;
    uint pte = g.storage_buffers[pc.table_buf].Load(vsm_pte_offset(pc.level, page));
    if ((pte & VSM_PTE_RENDER) == 0u)
        discard;
    uint phys = pte & VSM_PTE_PAGE_MASK;
    uint _prev;
    g.storage_buffers[pc.pool_buf].InterlockedMin(
        vsm_pool_offset(phys, texel % VSM_PAGE_SIZE), asuint(input.clip.z), _prev);
}
