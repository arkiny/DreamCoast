// Inline path tracer (Phase 8 M4 + ground-truth PBR track): diffuse global
// illumination via an inline `RayQuery` bounce loop in a compute shader.
//
// Shared push constants / material record / RNG / sky / sampling / geometry fetch
// live in `rt_common.slang` so this inline path and the pipeline path
// (`rt_pipeline.slang`) stay bit-identical. This file holds only the inline trace
// loop and the compute entry point.
//
// Per pixel, per sample: a primary camera ray is traced against the scene TLAS; at
// each hit the surface is shaded from the per-instance material table; direct sun
// light (with a hardware shadow ray) plus a cosine-weighted diffuse bounce
// accumulate radiance until a miss (sky) or the bounce limit. Results accumulate
// progressively in a float storage buffer; the running mean is the output image
// (exposed + tonemapped by the post pass). The host resets the accumulator on
// camera/scene changes.

#include "rt_common.slang"

// MAX_BOUNCES / RR_START_BOUNCE / SUN_COS_MAX live in rt_common.slang.

// Trace a single path and return its radiance. `cornell` switches to an enclosed
// box lit only by emissive surfaces (black background, no sun).
float3 trace(float3 origin, float3 dir, float cone_spread, float3 sun_dir, bool cornell,
             inout uint rng)
{
    float3 radiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);
    // Ray-cone footprint: width 0 at the camera, grows by t*spread each segment;
    // the spread widens on glossy/diffuse bounces so indirect hits use coarse mips.
    float cone_width = 0.0;

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    for (uint bounce = 0; bounce < MAX_BOUNCES; bounce++)
    {
        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(g.tlas, RAY_FLAG_NONE, 0xFF, ray);
        q.Proceed();
        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        {
            // Sun disk only on the primary ray; bounces get the sun analytically.
            if (!cornell)
                radiance += throughput * sky(ray.Direction, sun_dir, pc.sun.w, bounce == 0u);
            break;
        }

        uint prim = q.CommittedPrimitiveIndex();
        Instance inst;
        if (!load_valid_instance(q.CommittedInstanceID(), prim, inst))
            break; // out-of-range hit (never a valid surface) — terminate the path
        float2 bary = q.CommittedTriangleBarycentrics();
        float2 material_uv = hit_uv(inst, prim, bary);
        float3 n = hit_normal(inst, prim, bary);
        // Content instances sit in the TLAS under real world transforms (rotations
        // included), so rotate the object-space normal into world space; normalize
        // eats the uniform scale. The gallery/Cornell tables keep the raw normal
        // (their transforms are direction-preserving) — byte-identical.
        if (content_instances())
            n = normalize(mul((float3x3)q.CommittedObjectToWorld3x4(), n));
        if (dot(n, ray.Direction) > 0.0)
            n = -n; // face the incoming ray (two-sided)
        // Ray-cone texture LOD: advance the cone to this hit, then derive the LOD
        // from the cone footprint vs the triangle's world/UV areas.
        float cos_theta = max(abs(dot(n, ray.Direction)), 1e-3);
        cone_width += q.CommittedRayT() * cone_spread;
        float lod_base = hit_lod_base(inst, prim, cone_width, cos_theta);

        // Decal (MaterialKind::Decal, params.w = base-color opacity > 0): a coplanar
        // alpha-blended tint mesh the raster composites over the surface with
        // `alpha = base_color.a × texture.a` (DecalAlbedo blend). The path tracer does the
        // same "over" blend stochastically — with probability (1-alpha) the ray passes
        // through to the surface behind, otherwise the decal is shaded as an opaque diffuse
        // patch; averaged over samples this is the alpha composite. Content only
        // (gallery/Cornell have no decals and keep params.w = 0).
        if (content_instances() && inst.params.w > 0.0)
        {
            float dalpha = inst.params.w
                           * (inst.tex_base != NO_TEXTURE
                                  ? sample_texture_lod(inst.tex_base, material_uv, lod_base).a
                                  : 1.0);
            if (rnd(rng) >= dalpha)
            {
                // Pass through: restart the ray just past the decal. Consumes a bounce
                // (a safety bound on stacked decals); throughput/radiance unchanged.
                float advance = q.CommittedRayT() + 1e-3;
                ray.Origin = ray.Origin + ray.Direction * advance;
                ray.TMax = max(ray.TMax - advance, 0.0);
                continue;
            }
        }

        RtMaterialSample mat;
        if (bounce == 0u && content_instances())
        {
            // Primary hit: anisotropic texture filtering so grazing floor tiles keep
            // their across-view joints instead of the isotropic ray-cone LOD's smear
            // (see hit_uv_grads). Content only; bounces stay on the cheap cone LOD
            // (diffuse/rough — anisotropy invisible there). RNG stream unchanged.
            float3x3 o2w = (float3x3)q.CommittedObjectToWorld3x4();
            float2 ddx_uv;
            float2 ddy_uv;
            hit_uv_grads(inst, prim, o2w, cone_width, n, ray.Direction, ddx_uv, ddy_uv);
            mat = sample_material_grad(inst, material_uv, ddx_uv, ddy_uv);
        }
        else
        {
            mat = sample_material(inst, material_uv, lod_base);
        }
        n = hit_shading_normal(inst, prim, bary, n, material_uv, lod_base);
        float3 hit = ray.Origin + ray.Direction * q.CommittedRayT();
        float3 v = -ray.Direction;
        Surf surf = make_surf(mat.base_color, mat.metallic, mat.roughness);

        // Emissive surfaces add light directly.
        radiance += throughput * mat.emissive;

        // Direct sun light (next-event estimation: full BSDF toward a direction
        // sampled on the sun disk, so shadows are soft) with a hardware shadow ray
        // (outdoor mode only; the Cornell box is lit purely by its emissive ceiling).
        // The cone sample is taken for every outdoor hit (even back-facing) so the
        // RNG stream stays in lockstep with the pipeline path. Disk irradiance is
        // pc.sun.w (L*Ω), so the contribution is just eval_bsdf * sun.w * ndl.
        if (!cornell)
        {
            float3 wl = sample_cone(sun_dir, SUN_COS_MAX, rng);
            float ndl = dot(n, wl);
            if (ndl > 0.0)
            {
                RayDesc sray;
                sray.Origin = hit + n * 0.002;
                sray.Direction = wl;
                sray.TMin = 0.001;
                sray.TMax = 10000.0;
                RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
                sq.TraceRayInline(g.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, sray);
                sq.Proceed();
                if (sq.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
                    radiance += throughput * eval_bsdf(n, v, wl, surf) * pc.sun.w * ndl;
            }
        }

        // Importance-sample the BSDF for the next bounce (diffuse or specular lobe).
        BsdfSample bs = sample_bsdf(n, v, surf, rng);
        if (!bs.valid)
            break;
        throughput *= bs.weight;
        // Widen the cone for the next bounce by surface roughness (cheap proxy for
        // the BSDF lobe spread): mirrors stay tight, rough surfaces blur toward
        // coarse mips. Specular (roughness ~0) leaves the cone unchanged.
        cone_spread += mat.roughness * 0.5;

        // Russian roulette: unbiased termination once paths get deep, so MAX_BOUNCES
        // can be large without the cost of always tracing every bounce.
        if (bounce >= RR_START_BOUNCE)
        {
            float surv = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 1.0);
            if (rnd(rng) >= surv)
                break;
            throughput /= surv;
        }

        ray.Origin = hit + n * 0.002;
        ray.Direction = bs.dir;
        ray.TMin = 0.001;
        ray.TMax = 10000.0;
    }
    return radiance;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    bool cornell = (pc.flip_y & 2u) != 0u;
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    uint pixel = tid.y * pc.width + tid.x;

    float3 sun_dir = normalize(pc.sun.xyz);
    float3 sum = float3(0, 0, 0);
    for (uint s = 0; s < pc.spp; s++)
    {
        // Deterministic seed per pixel + global sample index (same on both APIs).
        uint rng = pixel * 9781u + (pc.frame * pc.spp + s) * 26699u + 1u;
        rng = pcg(rng);
        // Sub-pixel jitter for anti-aliasing.
        float2 jitter = float2(rnd(rng), rnd(rng)) - 0.5;
        float2 uv = (float2(tid.xy) + 0.5 + jitter) / float2(pc.width, pc.height);
        float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
        float3 dir = primary_ray_dir(ndc);
        float cone_spread = primary_cone_spread(ndc);
        sum += trace(pc.cam_pos.xyz, dir, cone_spread, sun_dir, cornell, rng);
    }
    sum /= float(pc.spp);

    // Progressive accumulation: frame 0 clears, later frames add this frame's mean.
    float4 prev = pc.frame == 0u
                      ? float4(0, 0, 0, 0)
                      : asfloat(g.storage_buffers[pc.accum_index].Load4(pixel * 16u));
    float4 acc = prev + float4(sum, 1.0);
    g.storage_buffers[pc.accum_index].Store4(pixel * 16u, asuint(acc));

    g.storage_images[pc.out_index][tid.xy] = float4(acc.rgb / acc.w, 1.0);
}
