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
float3 trace(float3 origin, float3 dir, float3 sun_dir, bool cornell, inout uint rng)
{
    float3 radiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);

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
        RtMaterialSample mat = sample_material(inst, material_uv);
        float3 n = hit_normal(inst, prim, bary);
        if (dot(n, ray.Direction) > 0.0)
            n = -n; // face the incoming ray (two-sided)
        n = hit_shading_normal(inst, prim, bary, n, material_uv);
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
        float4 world = mul(pc.inv_view_proj, float4(ndc, 1.0, 1.0));
        float3 dir = normalize(world.xyz / world.w - pc.cam_pos.xyz);
        sum += trace(pc.cam_pos.xyz, dir, sun_dir, cornell, rng);
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
