// F6O — per-pixel sky-visibility (V), first increment (scalar).
//
// Root cause (docs/phase-f6o-per-pixel-skyvis-plan.md): the indoor skylight occlusion V that
// pbr.slang's `occlude_sky_diffuse_bent` multiplies onto the IBL diffuse was sampled from a
// PER-PROBE SH volume (gi_volume.slang), whose grid resolution quantizes V into hard
// axis-aligned rectangular blocks on interior walls — and, because the fine volume is
// camera-anchored, the blocks shift as the camera moves ("검은 Block", DEBUG_VIEW=8).
//
// This pass instead computes V PER-PIXEL: from every shaded G-buffer pixel it casts `spp`
// COSINE-weighted hemisphere rays into the scene GDF (the same march gdf_gi/gdf_bounce use)
// and returns the fraction that escape to the sky. Cosine weighting makes the mean of the
// binary escape term the cosine-weighted hemispherical visibility — exactly the V that
// pbr.slang:139-141 documents and the SH volume reconstruct approximated, so the deferred
// consumer needs no change. Per-pixel evaluation removes the probe-grid quantization by
// construction (the gallery gdf_gi ray-march path shows the same march is screen-smooth).
//
// Increment 1 = SCALAR V only: output `float4(V, 0, 0, 1)`. The zero bent vector makes
// `occlude_sky_diffuse_bent` take its scalar fallback (irradiance(n)·V + tint leak), the
// byte-identical `bent-free` contract (pbr.slang:210). The directional bent normal
// (direction × VALIDITY, NOT × V — F6L contract) is increment 1b. The sky-vis SH VOLUME
// keeps being produced for the world-space consumers (surface cache, reflections); this pass
// only supplies the deferred pbr `skyvis_index` when `P_SKYVIS_PP=1`.

#include "bindless.slang"
#include "clipmap.slang"
#include "surface_cache.slang" // gdf_bounce.slang's bs_shade_hit references it (dead here, but the
                               // module must resolve — mirrors gdf_gi.slang's include order)
#include "gdf_bounce.slang"

struct SkyvisPush
{
    float4x4 inv_view_proj; // clip -> world (surface reconstruction)
    float4 aabb_min;        // xyz GDF world min, w = ground plane Y
    float4 aabb_max;        // xyz GDF world max, w = GDF sample distance clamp
    uint depth_index;       // sampled G-buffer depth
    uint normal_index;      // sampled G-buffer world normal
    uint out_index;         // storage-image output (V in .r, bent 0 in .gba)
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down); bit1: two-sided facing flip
    uint width;
    uint height;
    uint spp;               // cosine-hemisphere visibility rays per pixel
    uint frame;             // RNG decorrelation (for the increment-1b temporal accumulation)
    float ray_max;          // visibility-ray max march distance
    float march_bias;       // ray start offset past the surface (along the ray)
    float normal_offset;    // surface-point push along the normal (self-occlusion guard)
    float cone_k;           // cone-trace LOD slope (0 = legacy linear march)
    uint max_steps;         // visibility-ray march step cap
    uint clip_desc;         // clipmap descriptor storage-buffer index
    uint clip_count;        // clipmap level count (1 = single volume)
    uint pad0;
};

[[vk::push_constant]]
ConstantBuffer<SkyvisPush> pc : register(b0, space0);

// A BounceScene carrying only the fields the visibility march reads (clip / clamp / ground /
// ray_max / bias / cone_k / max_steps). Lighting fields are inert: this is a pure occlusion query.
BounceScene skyvis_scene()
{
    BounceScene bs;
    bs.clip_desc = pc.clip_desc;
    bs.clip_count = pc.clip_count;
    bs.sample_clamp = pc.aabb_max.w;
    bs.ground_y = pc.aabb_min.w;
    bs.sun_dir = float3(0.0, 1.0, 0.0);
    bs.sun_i = 0.0;
    bs.ray_max = pc.ray_max;
    bs.bias = pc.march_bias;
    bs.sky_term = 0.0;
    bs.albedo_fallback = 0.0;
    bs.cache = uint4(0xFFFFFFFFu, 0u, 0u, 0u);
    bs.cache_tile = 0u;
    bs.ground_albedo = float3(0.0, 0.0, 0.0);
    bs.cone_k = pc.cone_k;
    bs.max_steps = pc.max_steps;
    return bs;
}

// Sphere-march one hemisphere ray for sky visibility only (no shading). Mirrors the escape
// logic of gdf_bounce.slang `bs_trace_bounce` (same HIT_EPS / step floor / ground union) but
// skips the surface-cache/sun shading a lighting bounce does. Returns 1.0 if the ray reaches
// the sky, 0.0 if it hits geometry or the ground plane.
float skyvis_ray(BounceScene bs, float3 origin, float3 dir)
{
    int steps = int(max(bs.max_steps, 1u));
    const float HIT_EPS = 0.001;
    float t = bs.bias;
    float maxt = bs.ray_max;
    for (int i = 0; i < steps; i++)
    {
        float3 p = origin + dir * t;
        float d = bs_scene_march(bs, p);
        if (d < HIT_EPS)
            return 0.0; // hit geometry/ground -> occluded
        t += max(d, max(0.01, bs.cone_k * t));
        if (t > maxt)
            break;
    }
    return 1.0; // reached the sky
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;

    // Far plane (no geometry): sky fully visible. Replicate the existing producers' sky-pixel
    // contract `float4(1, 0, 0, 1)` (gdf_gi.slang:313-314) so the consumer reads V=1, bent=0.
    if (depth >= 1.0)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 hpos = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    float3 world_pos = hpos.xyz / hpos.w;

    float3 n = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);
    // Two-sided facing contract (F6M): orient toward the viewer exactly as the lighting pass
    // and gdf_ao do, or a back-face pixel of thin double-sided geometry marches its hemisphere
    // INTO the sheet. Same-pixel near-plane unprojection = a point toward the camera.
    if ((pc.flip_y & 2u) != 0u)
    {
        float4 hnear = mul(pc.inv_view_proj, float4(ndc, 0.0, 1.0));
        if (dot(n, hnear.xyz / hnear.w - world_pos) < 0.0)
            n = -n;
    }

    BounceScene bs = skyvis_scene();
    float3 origin = world_pos + n * pc.normal_offset;

    // Cosine-weighted hemisphere visibility: with cosine sampling the mean of the binary escape
    // term estimates ∫V(ω)cosθ dω / ∫cosθ dω = the cosine-weighted V the deferred skylight wants.
    // FRAME-VARYING RNG seed: each frame casts different rays so the temporal accumulation pass
    // (gdf_temporal, reprojected EMA) averages independent samples toward the true V — that is what
    // removes both the per-frame sparkle AND the spatial speckle (a fixed seed would accumulate the
    // SAME noisy value every frame and never converge).
    uint rng = (tid.y * pc.width + tid.x) * 9781u + pc.frame * 6271u + 1u;
    uint spp = max(pc.spp, 1u);
    float vis = 0.0;
    for (uint s = 0; s < spp; s++)
    {
        float3 dir = bounce_cosine_hemisphere(n, rng);
        vis += skyvis_ray(bs, origin, dir);
    }
    float v = vis / float(spp);

    // Increment 1: scalar V in .r, bent = 0 in .gba (consumer scalar fallback — pbr reads
    // sv.gba as the bent vector, so .a MUST be 0, not 1, or |bent|=1 forces the directional
    // path with a bogus +z direction). Directional bent (direction × validity) is increment 1b.
    g.storage_images[pc.out_index][tid.xy] = float4(v, 0.0, 0.0, 0.0);
}
