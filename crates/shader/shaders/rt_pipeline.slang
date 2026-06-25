// Path tracer via the hardware ray-tracing *pipeline* (Phase 8 M5 + ground-truth
// PBR track): raygen / miss / closest-hit + shader binding table.
//
// Shared push constants / material record / RNG / sky / sampling / geometry fetch
// live in `rt_common.slang` so this path stays bit-identical to the inline path
// (`rt_path.slang`). This file holds only the payload, the raygen bounce loop, and
// the miss / closest-hit shaders.
//
// The bounce loop lives in raygen: each iteration issues one `TraceRay`; the
// closest-hit shader fills the payload with the surface's emitted + direct light,
// the diffuse attenuation, and the next bounce ray; the miss shader fills it with
// the sky and terminates. Shadow rays use an inline `RayQuery` inside closest-hit,
// so the binding table stays one raygen / one miss / one hit group and
// `MaxTraceRecursionDepth` stays at 1.

#include "rt_common.slang"

// MAX_BOUNCES / RR_START_BOUNCE / SUN_COS_MAX live in rt_common.slang.

// Bounce-loop state carried between the raygen loop and the hit/miss shaders.
struct Payload
{
    float3 radiance;    // light emitted + directly lit at this hit (or sky on miss)
    float3 atten;       // multiply throughput by this for the next bounce (albedo)
    float3 next_origin; // spawn point of the next bounce ray
    float3 next_dir;    // cosine-weighted diffuse bounce direction
    uint rng;           // PCG state, advanced by the hit shader's bounce sampling
    uint done;          // 1 = terminate the path (miss or absorbed)
    uint primary;       // 1 = camera ray (miss shows the sun disk), 0 = bounce
};

// ---- Miss: sky (or black for the enclosed Cornell box) ----
[shader("miss")]
void msMain(inout Payload p)
{
    bool cornell = (pc.flip_y & 2u) != 0u;
    float3 sun_dir = normalize(pc.sun.xyz);
    // Sun disk only on the primary ray; bounces get the sun analytically.
    p.radiance = cornell ? float3(0, 0, 0)
                         : sky(WorldRayDirection(), sun_dir, pc.sun.w, p.primary != 0u);
    p.atten = float3(0, 0, 0);
    p.done = 1u;
}

// ---- Closest hit: emitted + direct sun (inline shadow ray) + next bounce ----
[shader("closesthit")]
void chMain(inout Payload p, BuiltInTriangleIntersectionAttributes attr)
{
    bool cornell = (pc.flip_y & 2u) != 0u;
    float3 sun_dir = normalize(pc.sun.xyz);

    uint prim = PrimitiveIndex();
    Instance inst;
    if (!load_valid_instance(InstanceID(), prim, inst))
    {
        // Out-of-range hit (never a valid surface) — terminate the path with no
        // contribution, matching the inline path's `break`.
        p.radiance = float3(0, 0, 0);
        p.atten = float3(0, 0, 0);
        p.done = 1u;
        return;
    }
    float2 material_uv = hit_uv(inst, prim, attr.barycentrics);
    RtMaterialSample mat = sample_material(inst, material_uv);
    float3 n = hit_normal(inst, prim, attr.barycentrics);
    float3 wdir = WorldRayDirection();
    if (dot(n, wdir) > 0.0)
        n = -n; // face the incoming ray (two-sided)
    n = hit_shading_normal(inst, prim, attr.barycentrics, n, material_uv);
    float3 hit = WorldRayOrigin() + wdir * RayTCurrent();
    float3 v = -wdir;
    Surf surf = make_surf(mat.base_color, mat.metallic, mat.roughness);

    // Emissive surfaces add light directly.
    float3 radiance = mat.emissive;

    // Direct sun light (next-event estimation: full BSDF toward a direction sampled
    // on the sun disk, so shadows are soft) with a hardware shadow ray (outdoor mode
    // only; the Cornell box is lit purely by its emissive ceiling). The cone sample
    // is taken for every outdoor hit so the RNG stream matches the inline path.
    if (!cornell)
    {
        float3 wl = sample_cone(sun_dir, SUN_COS_MAX, p.rng);
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
                radiance += eval_bsdf(n, v, wl, surf) * pc.sun.w * ndl;
        }
    }

    p.radiance = radiance;
    // Importance-sample the BSDF for the next bounce (diffuse or specular lobe).
    BsdfSample bs = sample_bsdf(n, v, surf, p.rng);
    p.atten = bs.weight;
    p.next_origin = hit + n * 0.002;
    p.next_dir = bs.dir;
    p.done = bs.valid ? 0u : 1u;
}

// ---- Ray generation: per-pixel sampling + bounce loop + accumulation ----
[shader("raygeneration")]
void rgMain()
{
    uint2 tid = DispatchRaysIndex().xy;
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
        float2 uv = (float2(tid) + 0.5 + jitter) / float2(pc.width, pc.height);
        float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
        float4 world = mul(pc.inv_view_proj, float4(ndc, 1.0, 1.0));
        float3 dir = normalize(world.xyz / world.w - pc.cam_pos.xyz);

        // Bounce loop: one TraceRay per bounce, state carried in the payload.
        float3 radiance = float3(0, 0, 0);
        float3 throughput = float3(1, 1, 1);
        RayDesc ray;
        ray.Origin = pc.cam_pos.xyz;
        ray.Direction = dir;
        ray.TMin = 0.001;
        ray.TMax = 10000.0;
        for (uint bounce = 0; bounce < MAX_BOUNCES; bounce++)
        {
            Payload p;
            p.radiance = float3(0, 0, 0);
            p.atten = float3(0, 0, 0);
            p.next_origin = float3(0, 0, 0);
            p.next_dir = float3(0, 0, 0);
            p.rng = rng;
            p.done = 0u;
            p.primary = bounce == 0u ? 1u : 0u;
            TraceRay(g.tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, p);
            rng = p.rng;
            radiance += throughput * p.radiance;
            if (p.done != 0u)
                break;
            throughput *= p.atten;

            // Russian roulette: unbiased deep-path termination (matches the inline
            // path's RNG order — the hit shader advanced p.rng for NEE + BSDF, this
            // RR draw comes next).
            if (bounce >= RR_START_BOUNCE)
            {
                float surv = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 1.0);
                if (rnd(rng) >= surv)
                    break;
                throughput /= surv;
            }

            ray.Origin = p.next_origin;
            ray.Direction = p.next_dir;
            ray.TMin = 0.001;
            ray.TMax = 10000.0;
        }
        sum += radiance;
    }
    sum /= float(pc.spp);

    // Progressive accumulation: frame 0 clears, later frames add this frame's mean.
    float4 prev = pc.frame == 0u
                      ? float4(0, 0, 0, 0)
                      : asfloat(g.storage_buffers[pc.accum_index].Load4(pixel * 16u));
    float4 acc = prev + float4(sum, 1.0);
    g.storage_buffers[pc.accum_index].Store4(pixel * 16u, asuint(acc));

    g.storage_images[pc.out_index][tid] = float4(acc.rgb / acc.w, 1.0);
}
