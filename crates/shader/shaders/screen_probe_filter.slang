// Screen-space radiance probes — P2: spatial cross-probe filter.
//
// A joint-bilateral blur of the octahedral radiance atlas ACROSS neighboring probes: for
// each probe texel, blend the SAME octahedral direction from the surrounding probes,
// weighted by surface-plane proximity + normal agreement (so the filter smooths probe-to-
// probe variation on a shared surface but never blurs across a wall corner / silhouette).
// Filtering matching directions across probes needs no octahedral border handling — each
// probe tile stays independent. Reduces the probe-grid blockiness the per-pixel gather
// would otherwise show, and lowers angular noise, without biasing the mean (bilateral).
//
// Radiance (rgb) and per-ray sky visibility (alpha, from the trace) are filtered together.

#include "bindless.slang"
#include "octahedral.slang"

struct ProbeFilterPush
{
    float4x4 inv_view_proj; // clip -> world (probe surface reconstruction)
    uint depth_index;       // sampled G-buffer depth
    uint normal_index;      // sampled G-buffer world normal
    uint atlas_in_index;    // sampled input radiance atlas
    uint atlas_out_index;   // storage output filtered atlas
    uint screen_w;          // render extent
    uint screen_h;
    uint probes_x;          // probe grid dims
    uint probes_y;
    uint downsample;        // screen pixels per probe tile
    uint oct_res;           // octahedral tile resolution
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
    uint half_kernel;       // probe-neighborhood half size (1 = 3x3)
    float pos_sigma;        // surface-plane proximity weight scale (world units)
    float normal_power;     // normal-agreement weight exponent
    float pad0;
    float pad1;
};

[[vk::push_constant]]
ConstantBuffer<ProbeFilterPush> pc : register(b0, space0);

// World surface of the probe at grid coord `probe` (its tile center pixel). false on sky.
bool probe_surface(uint2 probe, out float3 world_pos, out float3 normal)
{
    uint2 px = probe * pc.downsample + pc.downsample / 2u;
    px = min(px, uint2(pc.screen_w - 1u, pc.screen_h - 1u));
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

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    uint atlas_w = pc.probes_x * pc.oct_res;
    uint atlas_h = pc.probes_y * pc.oct_res;
    if (tid.x >= atlas_w || tid.y >= atlas_h)
        return;

    uint2 probe = tid.xy / pc.oct_res;
    uint2 oct_texel = tid.xy % pc.oct_res;

    // Invalid (sky) probe: pass the texel through unchanged.
    float3 pos, nrm;
    if (!probe_surface(probe, pos, nrm))
    {
        g.storage_images[pc.atlas_out_index][tid.xy] =
            g.textures[pc.atlas_in_index].Load(int3(int(tid.x), int(tid.y), 0));
        return;
    }

    int hk = int(pc.half_kernel);
    float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    float wsum = 0.0;
    for (int dy = -hk; dy <= hk; dy++)
    {
        for (int dx = -hk; dx <= hk; dx++)
        {
            int2 np = int2(probe) + int2(dx, dy);
            if (np.x < 0 || np.y < 0 || np.x >= int(pc.probes_x) || np.y >= int(pc.probes_y))
                continue;
            uint2 nprobe = uint2(np);

            float w;
            if (dx == 0 && dy == 0)
            {
                w = 1.0;
            }
            else
            {
                float3 npos, nnrm;
                if (!probe_surface(nprobe, npos, nnrm))
                    continue;
                float plane_d = abs(dot(npos - pos, nrm));
                float w_d = exp(-plane_d / max(pc.pos_sigma, 1e-4));
                float w_n = pow(saturate(dot(nrm, nnrm)), pc.normal_power);
                w = w_d * w_n;
                if (w < 1e-4)
                    continue;
            }

            uint2 src = nprobe * pc.oct_res + oct_texel;
            float4 s = g.textures[pc.atlas_in_index].Load(int3(int(src.x), int(src.y), 0));
            sum += s * w;
            wsum += w;
        }
    }

    float4 outv = wsum > 0.0 ? sum / wsum : g.textures[pc.atlas_in_index].Load(int3(int(tid.x), int(tid.y), 0));
    g.storage_images[pc.atlas_out_index][tid.xy] = outv;
}
