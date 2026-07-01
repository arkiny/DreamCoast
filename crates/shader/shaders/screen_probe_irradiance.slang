// Screen-space radiance probes — P5: per-probe irradiance pre-integration.
//
// Converts each probe's octahedral RADIANCE tile into an octahedral IRRADIANCE tile ONCE,
// so the per-pixel gather becomes a cheap directional lookup (bilinear in the pixel normal
// direction) instead of a full hemisphere integral per pixel. For each output direction
// `d`, the tile stores the cosine-weighted hemispherical integral of the probe's radiance
// about `d` — i.e. the indirect irradiance a surface with normal `d` would receive — in rgb,
// plus the cosine-weighted sky visibility about `d` in alpha. This is the dominant cost cut:
// the ~oct^2-tap-per-pixel gather collapses to a 4-probe bilinear lookup.
//
// Dispatched over the atlas (one thread per output texel of every probe); each thread reads
// its own probe's oct^2 radiance texels via groupshared (the 8x8 threadgroup = one probe
// tile when oct = 8), so each source texel is loaded once per probe, not once per output.

#include "bindless.slang"
#include "octahedral.slang"

struct ProbeIrradiancePush
{
    uint atlas_in;  // sampled radiance atlas (probes * oct)
    uint atlas_out; // storage irradiance atlas (same dims)
    uint probes_x;
    uint probes_y;
    uint oct;       // octahedral texels per side (<= 16; groupshared sized for 16*16)
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]]
ConstantBuffer<ProbeIrradiancePush> pc : register(b0, space0);

// Groupshared cache of one probe tile's radiance+visibility texels (max 16x16).
groupshared float4 g_tile[256];

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID, uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    uint atlas_w = pc.probes_x * pc.oct;
    uint atlas_h = pc.probes_y * pc.oct;

    // The probe this threadgroup covers (8x8 group == one tile when oct == 8; for oct != 8 the
    // group still maps into a single tile because the atlas is tiled by oct).
    uint2 probe = tid.xy / pc.oct;
    uint2 out_texel = tid.xy % pc.oct;
    uint2 base = probe * pc.oct;

    // Cooperatively load this probe's oct^2 source texels into groupshared. The 8x8 group has 64
    // threads; a tile has oct^2 texels — each thread loads a strided share.
    uint tile_n = pc.oct * pc.oct;
    uint tgi = lid.y * 8u + lid.x;
    for (uint i = tgi; i < tile_n; i += 64u)
    {
        uint sx = i % pc.oct;
        uint sy = i / pc.oct;
        uint2 spx = base + uint2(sx, sy);
        float4 v = (spx.x < atlas_w && spx.y < atlas_h)
                       ? g.textures[pc.atlas_in].Load(int3(int(spx.x), int(spx.y), 0))
                       : float4(0.0, 0.0, 0.0, 0.0);
        g_tile[i] = v;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid.x >= atlas_w || tid.y >= atlas_h)
        return;

    float3 dir_out = oct_to_dir(oct_texel_to_square(out_texel, pc.oct));

    float3 e = float3(0.0, 0.0, 0.0);
    float vis_num = 0.0;
    float cos_sum = 0.0;
    for (uint s = 0u; s < tile_n; s++)
    {
        uint sx = s % pc.oct;
        uint sy = s / pc.oct;
        float3 dir_s = oct_to_dir(oct_texel_to_square(uint2(sx, sy), pc.oct));
        float c = max(dot(dir_s, dir_out), 0.0);
        if (c <= 0.0)
            continue;
        float4 L = g_tile[s];
        e += L.rgb * c;
        vis_num += L.a * c;
        cos_sum += c;
    }
    // Same estimator as the per-pixel integrate: (1/pi) * sum(L*cos*dOmega), dOmega = 4*pi/N.
    e *= 4.0 / float(tile_n);
    float vis = cos_sum > 0.0 ? vis_num / cos_sum : 1.0;

    g.storage_images[pc.atlas_out][tid.xy] = float4(e, vis);
}
