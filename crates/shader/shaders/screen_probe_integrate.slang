// Screen-space radiance probes — per-pixel integrate (P3 core).
//
// Each shaded pixel gathers the 2x2 neighborhood of screen probes whose tile centers
// surround it, weights them by bilinear position + surface-plane proximity + normal
// agreement (so a probe on a different surface than the pixel doesn't leak across the
// silhouette), and reconstructs the indirect irradiance E by integrating each probe's
// octahedral radiance tile over the pixel's cosine hemisphere. Output matches the
// per-pixel GI pass (`gdf_gi.slang`): raw indirect irradiance the deferred lighting
// multiplies by surface albedo (x 1-metallic) and adds to the ambient term.
//
// The cosine-weighted octahedral integral uses a uniform per-texel solid angle
// (4*pi / oct_res^2); the (1/pi) hemisphere normalization folds in as 4/oct_res^2. This
// is the correctness-first gather (oct_res^2 taps x up to 4 probes); a probe-irradiance
// pre-integration is a later optimization.

#include "bindless.slang"
#include "octahedral.slang"

struct ProbeIntegratePush
{
    float4x4 inv_view_proj; // clip -> world (surface reconstruction)
    uint depth_index;       // sampled G-buffer depth
    uint normal_index;      // sampled G-buffer world normal
    uint atlas_index;       // sampled radiance octahedral atlas (probes * oct_res)
    uint out_index;         // storage-image index (indirect irradiance output)
    uint screen_w;          // render extent
    uint screen_h;
    uint probes_x;          // probe grid dims
    uint probes_y;
    uint downsample;        // screen pixels per probe tile
    uint oct_res;           // octahedral tile resolution
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
    uint skyvis_index;      // storage-image index for indoor sky-visibility (0xFFFFFFFF = off)
    float pos_sigma;        // surface-plane proximity weight scale (world units)
    float normal_power;     // normal-agreement weight exponent
    float pad1;
    float pad2;
};

[[vk::push_constant]]
ConstantBuffer<ProbeIntegratePush> pc : register(b0, space0);

// Reconstruct world position for a pixel from the G-buffer depth (returns false on sky).
bool reconstruct(uint2 px, out float3 world_pos, out float3 normal)
{
    float2 uv = (float2(px) + 0.5) / float2(pc.screen_w, pc.screen_h);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;
    if (depth >= 1.0)
        return false;
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 hpos = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    world_pos = hpos.xyz / hpos.w;
    normal = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);
    return true;
}

// World surface of the probe at grid coord `probe` (its tile center pixel).
bool probe_surface(uint2 probe, out float3 world_pos, out float3 normal)
{
    uint2 px = probe * pc.downsample + pc.downsample / 2u;
    px = min(px, uint2(pc.screen_w - 1u, pc.screen_h - 1u));
    return reconstruct(px, world_pos, normal);
}

// Integrate one probe's octahedral tile over the cosine hemisphere about `n`. Returns the
// indirect irradiance E and (via `vis`) the cosine-weighted hemispherical sky visibility
// V(n) in [0,1] (1 = fully sky-exposed, 0 = enclosed) reconstructed from the atlas alpha.
float3 integrate_probe(uint2 probe, float3 n, out float vis)
{
    uint2 base = probe * pc.oct_res;
    float3 e = float3(0.0, 0.0, 0.0);
    float vis_num = 0.0;
    float cos_sum = 0.0;
    for (uint oy = 0u; oy < pc.oct_res; oy++)
    {
        for (uint ox = 0u; ox < pc.oct_res; ox++)
        {
            float2 sq = oct_texel_to_square(uint2(ox, oy), pc.oct_res);
            float3 dir = oct_to_dir(sq);
            float c = max(dot(dir, n), 0.0);
            if (c <= 0.0)
                continue;
            float4 s = g.textures[pc.atlas_index].Load(int3(int(base.x + ox), int(base.y + oy), 0));
            e += s.rgb * c;
            vis_num += s.a * c; // alpha = per-ray sky visibility
            cos_sum += c;
        }
    }
    vis = cos_sum > 0.0 ? vis_num / cos_sum : 1.0;
    // (1/pi) * sum(L * cos * dOmega), dOmega = 4*pi / oct_res^2  ->  4/oct_res^2.
    return e * (4.0 / float(pc.oct_res * pc.oct_res));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.screen_w || tid.y >= pc.screen_h)
        return;

    float3 world_pos, normal;
    if (!reconstruct(tid.xy, world_pos, normal))
    {
        g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 1.0);
        // Sky / far plane: fully sky-visible (defined output so lighting never reads stale).
        if (pc.skyvis_index != 0xFFFFFFFFu)
            g.storage_images[pc.skyvis_index][tid.xy] = float4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    // Fractional probe coordinate (integer = a probe tile center).
    float2 pf = (float2(tid.xy) - float(pc.downsample) * 0.5) / float(pc.downsample);
    float2 base_f = floor(pf);
    float2 fr = pf - base_f;
    int2 base = int2(base_f);

    float3 accum = float3(0.0, 0.0, 0.0);
    float vis_accum = 0.0;
    float wsum = 0.0;
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < 2; i++)
        {
            int2 pc_i = base + int2(i, j);
            if (pc_i.x < 0 || pc_i.y < 0 || pc_i.x >= int(pc.probes_x) || pc_i.y >= int(pc.probes_y))
                continue;
            uint2 probe = uint2(pc_i);

            float3 ppos, pnorm;
            if (!probe_surface(probe, ppos, pnorm))
                continue;

            float wx = (i == 0) ? (1.0 - fr.x) : fr.x;
            float wy = (j == 0) ? (1.0 - fr.y) : fr.y;
            float w_bi = wx * wy;
            // Surface-plane proximity: a probe off the pixel's tangent plane (a different surface
            // at the same screen distance) is rejected so indirect light doesn't leak across edges.
            float plane_d = abs(dot(ppos - world_pos, normal));
            float w_d = exp(-plane_d / max(pc.pos_sigma, 1e-4));
            float w_n = pow(saturate(dot(normal, pnorm)), pc.normal_power);
            float w = w_bi * w_d * w_n;
            if (w < 1e-5)
                continue;

            float v_probe;
            accum += w * integrate_probe(probe, normal, v_probe);
            vis_accum += w * v_probe;
            wsum += w;
        }
    }

    float inv_w = wsum > 0.0 ? 1.0 / wsum : 0.0;
    float3 e = accum * inv_w;
    g.storage_images[pc.out_index][tid.xy] = float4(e, 1.0);

    // Indoor skylight occlusion: cosine-weighted hemispherical sky visibility V(n). Open
    // surfaces -> ~1 (full skylight); enclosed interiors -> ~0 (the diffuse skylight is
    // occluded, so the colored scene bounce shows instead of flat blue sky). No probe coverage
    // (wsum == 0) -> fully sky-visible (matches the volume path's graceful fallback).
    if (pc.skyvis_index != 0xFFFFFFFFu)
    {
        float v = wsum > 0.0 ? saturate(vis_accum * inv_w) : 1.0;
        g.storage_images[pc.skyvis_index][tid.xy] = float4(v, 0.0, 0.0, 1.0);
    }
}
