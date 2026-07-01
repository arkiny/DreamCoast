// Screen-space ambient occlusion (HBAO-lite obscurance) — the NEAR-field AO the coarse
// GDF AO (gdf_ao.slang) misses. It reconstructs each pixel's world position + normal from
// the G-buffer depth, samples neighbours in a depth-scaled screen-space disk, and accumulates
// horizon obscurance. The result composes with the GDF AO in the lighting pass (near-field ×
// far-field, the way 레퍼런스 엔진 layers GTAO × DFAO), restoring the crevice/contact definition that
// makes surfaces read as solid rather than flat.
//
// Determinism (the DX≡VK gate): the per-pixel sampling rotation is a PURE INTEGER hash of the
// pixel coordinate (no screen-space matrix term, no frame counter), so SPIR-V and DXIL produce
// the identical pattern; world reconstruction matches gdf_ao.slang. A separable depth-aware blur
// (`csBlur`) cleans the dither without bleeding across depth edges.

#include "bindless.slang"

static const float GTAO_PI = 3.14159265359;

struct GtaoPush
{
    float4x4 inv_view_proj; // clip -> world (surface reconstruction, matches gdf_ao)
    float4 camera_pos;      // xyz world camera (for view-depth radius scaling)
    uint depth_index;       // sampled G-buffer depth
    uint normal_index;      // sampled G-buffer world normal
    uint out_index;         // storage-image index (AO output)
    uint width;
    uint height;
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
    uint dir_index;         // csBlur: blur direction (0 = X, 1 = Y); reads `in_index`
    uint in_index;          // csBlur: sampled input AO image
    // x = world radius (m), y = intensity, z = bias (m), w = proj_scale (0.5/tan(fovY/2))
    float4 params;
    float4 params2;         // x = aspect (w/h), y = power, z = depth-edge sigma (blur), w unused
};

[[vk::push_constant]]
ConstantBuffer<GtaoPush> pc : register(b0, space0);

// PCG integer hash (same family as the foliage hash) — backend-identical.
uint pcg(uint v)
{
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

float depth_to_view_z(float2 uv, float depth, out float3 world_pos)
{
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 h = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    world_pos = h.xyz / h.w;
    return length(world_pos - pc.camera_pos.xyz); // view distance (radius scaling)
}

float3 world_at(float2 uv)
{
    float d = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 h = mul(pc.inv_view_proj, float4(ndc, d, 1.0));
    return h.xyz / h.w;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;
    float2 res = float2(pc.width, pc.height);
    float2 uv = (float2(tid.xy) + 0.5) / res;
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;
    if (depth >= 1.0)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    float3 P;
    float view_z = depth_to_view_z(uv, depth, P);
    float3 N = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);

    float radius = pc.params.x;
    float bias = pc.params.z;
    // Depth-scaled screen radius: a fixed world radius projects to fewer UVs the farther the
    // surface, so the AO footprint stays world-consistent (no near-surface over-occlusion).
    float r_uv = radius * pc.params.w / max(view_z, 1e-3);
    r_uv = min(r_uv, 0.1); // cap the screen footprint (perf + halo guard)
    float aspect = max(pc.params2.x, 1e-3);

    const int DIRS = 6;
    const int STEPS = 4;
    // Per-pixel rotation (integer hash → deterministic) decorrelates the dither between pixels.
    float rot = float(pcg(tid.x * 1973u + tid.y * 9277u) & 0xFFFFu) / 65536.0 * GTAO_PI;

    float occ = 0.0;
    for (int d = 0; d < DIRS; d++)
    {
        float ang = rot + float(d) * (GTAO_PI / float(DIRS)); // 0..PI (each dir samples both sides)
        float2 dir = float2(cos(ang), sin(ang));
        dir.x /= aspect; // keep the disk circular in world/screen
        for (int s = 1; s <= STEPS; s++)
        {
            float t = (float(s) - 0.5) / float(STEPS);
            float2 off = dir * r_uv * t;
            // Both sides of the direction (horizon obscurance is symmetric).
            for (int sign = -1; sign <= 1; sign += 2)
            {
                float2 suv = uv + off * float(sign);
                if (any(suv < 0.0) || any(suv > 1.0))
                    continue;
                float3 Sp = world_at(suv);
                float3 v = Sp - P;
                float dist = length(v);
                if (dist < 1e-4)
                    continue;
                float ndotv = max(dot(N, v / dist) - bias / max(dist, 1e-3), 0.0);
                // Range falloff: distant samples occlude less (and stop haloing across big gaps).
                float falloff = saturate(1.0 - dist / radius);
                occ += ndotv * falloff;
            }
        }
    }
    float norm = 1.0 / float(DIRS * STEPS * 2);
    float ao = saturate(1.0 - pc.params.y * occ * norm);
    ao = pow(ao, max(pc.params2.y, 1e-3)); // contrast/power
    g.storage_images[pc.out_index][tid.xy] = float4(ao, ao, ao, 1.0);
}

// Separable depth-aware (bilateral) blur to denoise the dithered AO without crossing depth edges.
[shader("compute")]
[numthreads(8, 8, 1)]
void csBlur(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;
    float2 res = float2(pc.width, pc.height);
    float2 uv = (float2(tid.xy) + 0.5) / res;
    float2 step = (pc.dir_index == 0u) ? float2(1.0 / res.x, 0.0) : float2(0.0, 1.0 / res.y);

    float center_d = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;
    float sigma = max(pc.params2.z, 1e-5);
    float sum = 0.0;
    float wsum = 0.0;
    for (int i = -3; i <= 3; i++)
    {
        float2 suv = uv + step * float(i);
        float a = g.textures[pc.in_index].SampleLevel(g.samp, suv, 0.0).r;
        float sd = g.textures[pc.depth_index].SampleLevel(g.samp, suv, 0.0).r;
        float wd = exp(-abs(sd - center_d) / sigma); // depth-edge stop
        float wg = exp(-float(i * i) / 8.0);         // gaussian
        float w = wd * wg;
        sum += a * w;
        wsum += w;
    }
    float ao = (wsum > 0.0) ? sum / wsum : 1.0;
    g.storage_images[pc.out_index][tid.xy] = float4(ao, ao, ao, 1.0);
}
