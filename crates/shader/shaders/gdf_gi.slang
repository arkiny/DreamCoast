// Phase 11 Stage C3 — stochastic 1-bounce diffuse GI against the global distance field.
//
// A full-screen compute pass: for every shaded G-buffer pixel it reconstructs the
// world surface point from the depth buffer (object-space position MRT can't be used,
// see gdf_ao.slang) and the world normal, then casts `spp` cosine-weighted hemisphere
// rays into the world-space scene GDF (C1). Each ray that hits geometry/ground is
// shaded at the hit (design fork A: gradient normal + sun w/ soft shadow + a small sky
// term, a single constant albedo — the GDF carries no material; colored bleed needs a
// surface cache, a later refinement). The cosine-weighted Monte-Carlo estimator makes
// the per-sample contribution simply the incoming radiance, so the output is the mean
// incoming *radiance* (indirect irradiance E); the deferred lighting pass multiplies
// it by the surface albedo (× 1-metallic) and adds it to the ambient term.
//
// Rays that escape to the sky return 0, NOT the sky color: the existing IBL diffuse
// already supplies the (unoccluded) sky, the C2 GDF-AO removes the sky that geometry
// occludes, and this pass adds back the light those occluders bounce. So the ambient
// decomposition is IBL·AO (sky) + albedo·E_gi (scene bounce) with no double counting.
//
// Output is raw radiance (pre-exposure), matching the deferred lighting's direct term;
// the lighting pass applies the camera exposure to the whole ambient + direct sum.

#include "bindless.slang"
#include "clipmap.slang"
#include "surface_cache.slang"
#include "gdf_bounce.slang"

struct GiPush
{
    float4x4 inv_view_proj; // clip -> world (surface reconstruction)
    float4 sun;             // xyz direction toward the sun, w = intensity
    uint depth_index;       // sampled G-buffer depth
    uint normal_index;      // sampled G-buffer world normal
    uint gdf_sampled;       // volumes[] (SRV) index of the scene GDF
    uint out_index;         // storage-image index (indirect irradiance output)
    uint width;
    uint height;
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
    uint spp;               // hemisphere rays per pixel
    uint frame;             // frame counter (RNG decorrelation, for C4 temporal)
    uint albedo_r;          // C8a: volumes[] index of the red albedo channel (0xFFFFFFFF = constant)
    uint albedo_g;          // green
    uint albedo_b;          // blue
    float4 aabb_min;        // xyz GDF world min, w = ground plane Y
    float4 aabb_max;        // xyz GDF world max, w = sample distance clamp
    float4 params;          // x = ray max distance, y = surface bias, z = sky term, w = constant hit albedo (fallback)
    uint4 cache;            // C8b3: cards, cache_pos, cache_rad, num_cards (cards=0xFFFFFFFF = off)
    uint cache_tile;        // surface-cache atlas tile edge
    float clamp_max;        // firefly clamp: max bounce-sample radiance component (1e30 = off)
    uint clip_desc;         // Stage B: clipmap descriptor storage-buffer index
    uint clip_count;        // Stage B: clipmap level count (1 = single volume = legacy)
    // xyz = analytic ground material (not in the albedo volume); w = cone_k (P3 cone-trace LOD
    // slope; 0 = legacy linear march). float4, NOT float3 + a trailing scalar: HLSL/SPIR-V pack
    // a scalar after a float3 at +12 but MSL pads the float3 to +16, so the old layout mis-read
    // the following scalars on Metal (max_steps picked up cone_k's bytes -> wrong GI). A float4's
    // .w sits at +12 on all three backends; max_steps then gets its own clean 16-byte row.
    float4 ground_albedo;
    uint max_steps;         // Stage D3: bounce-ray march step cap (64 = legacy / gallery)
    uint vol_r;             // directional-irradiance volume: sampled BASE index of the 12 SH
                            // coefficient volumes (0xFFFFFFFF = off -> trace rays)
    uint vol_g;             // sky-visibility: sampled BASE index of the 4 scalar SH volumes
                            // (0xFFFFFFFF = sky-vis off). Reuses vol_r's freed slot.
    uint vol_b;             // sky-visibility OUTPUT storage-image index (0xFFFFFFFF = don't write).
};

// Real SH band-0/1 basis normalisation (matches gi_volume.slang).
static const float SH_C0 = 0.2820947918; // sqrt(1/(4*pi))
static const float SH_C1 = 0.4886025119; // sqrt(3/(4*pi))

[[vk::push_constant]]
ConstantBuffer<GiPush> pc : register(b0, space0);

#define GDF_MIN (pc.aabb_min.xyz)
#define GDF_MAX (pc.aabb_max.xyz)
#define GROUND_Y (pc.aabb_min.w)

// Build the shared bounce-tracer scene from this pass's push constants. The march/shade/
// shadow code lives once in `gdf_bounce.slang` (also used by the screen-space probes);
// this is a pure mapping of `pc` fields, so the per-pixel ray-march output is unchanged.
BounceScene gi_bounce_scene()
{
    BounceScene bs;
    bs.clip_desc = pc.clip_desc;
    bs.clip_count = pc.clip_count;
    bs.sample_clamp = pc.aabb_max.w;
    bs.ground_y = GROUND_Y;
    bs.sun_dir = pc.sun.xyz;
    bs.sun_i = pc.sun.w;
    bs.ray_max = pc.params.x;
    bs.bias = pc.params.y;
    bs.sky_term = pc.params.z;
    bs.albedo_fallback = pc.params.w;
    bs.cache = pc.cache;
    bs.cache_tile = pc.cache_tile;
    bs.ground_albedo = pc.ground_albedo.xyz;
    bs.cone_k = pc.ground_albedo.w;
    bs.max_steps = pc.max_steps;
    return bs;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;

    // Far plane (no geometry): no indirect bounce. Sky-vis = 1 (the skybox isn't IBL-lit, but keep
    // the image defined so the lighting never reads stale storage).
    if (depth >= 1.0)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 1.0);
        if (pc.vol_b != 0xFFFFFFFFu)
            g.storage_images[pc.vol_b][tid.xy] = float4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 hpos = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    float3 world_pos = hpos.xyz / hpos.w;

    float3 n = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);

    // Directional-irradiance volume mode (radiance cache): instead of marching per-pixel rays,
    // sample the world-space SH volume (updated in gi_volume.slang with multibounce) and
    // reconstruct the cosine-weighted hemispherical average radiance in THIS pixel's normal
    // direction. That directionality is what lets a coloured surface bleed onto a neighbour the
    // old omnidirectional average washed out. The result is the indirect E the lighting
    // multiplies by albedo; it fills enclosed interiors a single bounce can't reach. Feeds the
    // existing temporal + a-trous denoiser unchanged.
    if (pc.vol_r != 0xFFFFFFFFu)
    {
        float3 uvw = (world_pos - GDF_MIN) / (GDF_MAX - GDF_MIN);
        float3 e = float3(0.0, 0.0, 0.0);
        if (all(uvw >= 0.0) && all(uvw <= 1.0))
        {
            uvw = saturate(uvw);
            // The 12 SH coefficient volumes are contiguous from vol_r: coeff k, channel ch lives
            // in volume vol_r + ch*4 + k. Reconstruct E(n) with the clamped-cosine convolution
            // (A0/PI = 1, A1/PI = 2/3); clamp >= 0 (band-1 SH can ring negative).
            float3 c[4];
            [unroll]
            for (int k = 0; k < 4; k++)
                c[k] = float3(0.0, 0.0, 0.0);
            [unroll]
            for (int ch = 0; ch < 3; ch++)
            {
                [unroll]
                for (int k = 0; k < 4; k++)
                    c[k][ch] =
                        g.volumes[pc.vol_r + uint(ch * 4 + k)].SampleLevel(g.samp, uvw, 0.0).x;
            }
            e = c[0] * SH_C0
                + (2.0 / 3.0) * (SH_C1 * n.y * c[1] + SH_C1 * n.z * c[2] + SH_C1 * n.x * c[3]);
            e = max(e, float3(0.0, 0.0, 0.0));
        }
        g.storage_images[pc.out_index][tid.xy] = float4(e, 1.0);

        // Sky-visibility (레퍼런스식 indoor skylight occlusion): reconstruct the cosine-weighted
        // hemispherical sky visibility V(n) from the 4 scalar SH volumes (contiguous from vol_g) and
        // write it for the lighting to occlude the IBL diffuse skylight. Open surfaces -> ~1 (full
        // skylight); enclosed interiors -> ~0 (the diffuse skylight is occluded, so the coloured
        // scene bounce shows instead of flat blue sky).
        if (pc.vol_g != 0xFFFFFFFFu && pc.vol_b != 0xFFFFFFFFu)
        {
            float v = 1.0;
            if (all(uvw >= 0.0) && all(uvw <= 1.0))
            {
                float sc[4];
                [unroll]
                for (int k = 0; k < 4; k++)
                    sc[k] = g.volumes[pc.vol_g + uint(k)].SampleLevel(g.samp, uvw, 0.0).x;
                v = sc[0] * SH_C0
                    + (2.0 / 3.0) * (SH_C1 * n.y * sc[1] + SH_C1 * n.z * sc[2] + SH_C1 * n.x * sc[3]);
                v = saturate(v);
            }
            g.storage_images[pc.vol_b][tid.xy] = float4(v, 0.0, 0.0, 1.0);
        }
        return;
    }

    uint spp = max(pc.spp, 1u);
    uint seed = (tid.x * 1973u + tid.y * 9277u + pc.frame * 26699u) | 1u;
    float3 origin = world_pos + n * pc.params.y;
    BounceScene bs = gi_bounce_scene();

    float3 sum = float3(0.0, 0.0, 0.0);
    for (uint s = 0; s < spp; s++)
    {
        float3 dir = bounce_cosine_hemisphere(n, seed);
        // Firefly clamp per sample (before averaging): one ray hitting a bright spot can't
        // blow out the mean. Hue-preserving; `clamp_max = 1e30` disables it.
        float esc;
        float3 b = bs_trace_bounce(bs, origin, dir, esc);
        float m = max(b.r, max(b.g, b.b));
        if (m > pc.clamp_max)
            b *= pc.clamp_max / m;
        sum += b;
    }
    // Cosine-weighted estimator: mean incoming radiance = indirect irradiance E. The
    // lighting pass multiplies by the surface albedo (× 1-metallic) to get L_o.
    float3 e = sum / float(spp);

    g.storage_images[pc.out_index][tid.xy] = float4(e, 1.0);
}
