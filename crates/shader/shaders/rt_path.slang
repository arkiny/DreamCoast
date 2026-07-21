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

// F6M: cap on cutout-texel pass-throughs per ray/shadow query. A dense canopy crossing
// can graze dozens of `alphaMode: MASK` leaf cards; each pass-through is a full
// re-traversal, so the cap bounds the cost (budget exhausted => treat as opaque —
// biases dark, never leaks light).
static const uint MASK_MAX_SKIPS = 32u;

// F6M: sun-shadow visibility with the raster's MASK cutout contract — a transparent
// leaf texel must not block the sun (with opaque cards the whole cypress self-shadowed
// into a black blob in the reference). The TLAS is opaque, so instead of candidate
// iteration each committed cutout hit restarts the query just beyond the texel.
// Consumes no RNG: non-MASK scenes keep bit-identical sampling streams.
bool sun_blocked(RayDesc sray)
{
    for (uint i = 0; i <= MASK_MAX_SKIPS; i++)
    {
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
        sq.TraceRayInline(g.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFE, sray);
        sq.Proceed();
        if (sq.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
            return false;
        uint prim = sq.CommittedPrimitiveIndex();
        Instance inst;
        if (!load_valid_instance(sq.CommittedInstanceID(), prim, inst))
            return true; // malformed hit — treat as an occluder
        if (!(content_instances() && inst.params.z > 0.0 && inst.tex_base != NO_TEXTURE))
            return true; // opaque material
        float2 suv = hit_uv(inst, prim, sq.CommittedTriangleBarycentrics());
        // Mip 0: the cutoff needs the authored mask, not a mip-averaged alpha.
        if (sample_mask_alpha(inst.tex_base, suv) >= inst.params.z)
            return true; // opaque texel of a MASK material
        float advance = sq.CommittedRayT() + 1e-3;
        sray.Origin = sray.Origin + sray.Direction * advance;
        sray.TMax = max(sray.TMax - advance, 0.0);
    }
    return true; // skip budget exhausted — conservatively opaque
}

// Trace a single path and return its radiance. `cornell` switches to an enclosed
// box lit only by emissive surfaces (black background, no sun).
float3 trace(float3 origin, float3 dir, float cone_spread, float3 sun_dir, bool cornell,
             bool e_view, bool v_view, inout uint rng)
{
    float3 radiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);
    // Ray-cone footprint: width 0 at the camera, grows by t*spread each segment;
    // the spread widens on glossy/diffuse bounces so indirect hits use coarse mips.
    float cone_width = 0.0;
    // F6M: per-path budget of MASK cutout pass-throughs (see MASK_MAX_SKIPS).
    uint mask_skips = 0u;

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
            // E-oracle: a bounce-1 sky escape carries no SCENE radiance — the GI volume's
            // miss = 0 convention (the IBL supplies direct sky separately). Deeper bounces
            // keep the sky: a hit surface's own shading includes its skylight, exactly as
            // the volume's hit-point sky_fill term does.
            // V-oracle (F6H): the bounce-1 sky escape IS the tally — 1, exactly the volume's
            // sky-visibility projection (vis = hit ? 0 : 1); the progressive mean is the
            // cosine-weighted V(n) the deferred skylight needs at this surface.
            if (v_view && bounce == 1u)
            {
                radiance = float3(1.0, 1.0, 1.0);
                break;
            }
            if (!cornell && !(e_view && bounce == 1u))
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

        // alphaMode MASK (params.z = cutoff > 0): the raster's G-buffer/shadow passes
        // discard cutout texels, so the reference must pass through them too — with
        // opaque cards the cypress canopy rendered as a black blob (no light between
        // leaves, F6M). Restart just past the texel WITHOUT consuming a bounce (a
        // cutout texel is not a surface; bounce-keyed logic like the V/E-oracle
        // bounce-1 tally must see the real geometry). Deterministic test, no RNG:
        // sampling streams stay bit-identical for every non-MASK path. Mip-0 sample
        // (sample_mask_alpha): the cutoff needs the authored mask — a mip-averaged
        // alpha sits below the cutoff and dissolves the canopy wholesale.
        if (content_instances() && inst.params.z > 0.0 && inst.tex_base != NO_TEXTURE
            && mask_skips < MASK_MAX_SKIPS
            && sample_mask_alpha(inst.tex_base, material_uv) < inst.params.z)
        {
            mask_skips++;
            float advance = q.CommittedRayT() + 1e-3;
            ray.Origin = ray.Origin + ray.Direction * advance;
            ray.TMax = max(ray.TMax - advance, 0.0);
            bounce--; // uint wrap at 0 is re-incremented by the loop — same bounce resumes
            continue;
        }

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

        // E-oracle primary hit: skip this surface's emissive/direct/BSDF entirely — draw ONE
        // cosine-weighted direction about the geometric normal and continue. From bounce 1 on
        // the path is a standard PT estimate of the incoming scene radiance L_in, and the
        // cosine draw makes the progressive mean exactly E(n) at the primary surface (the
        // same Monte-Carlo estimator gdf_gi's march uses), in the volume's pre-albedo,
        // pre-exposure units. Debug-mode only; the flag is never set on the gate/anchor paths.
        if ((e_view || v_view) && bounce == 0u)
        {
            ray.Origin = ray.Origin + ray.Direction * q.CommittedRayT() + n * 0.002;
            ray.Direction = cosine_hemisphere(n, rng);
            ray.TMin = 0.001;
            ray.TMax = 10000.0;
            continue;
        }
        // V-oracle: a bounce-1 GEOMETRY hit is occlusion — tally 0 and terminate (no shading).
        if (v_view && bounce == 1u)
            break;

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
                // Mask 0xFE excludes decal instances (mask bit 0x01, see build_content_accel):
                // decals are surface tints, not occluders, so they must not cast a shadow (the
                // raster's deferred decals don't). Non-decals (mask 0xFF) are still shadow-tested,
                // with the MASK cutout contract applied per texel (F6M, see sun_blocked).
                if (!sun_blocked(sray))
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
    bool e_view = (pc.flip_y & 4u) != 0u;
    bool v_view = (pc.flip_y & 8u) != 0u;
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
        sum += trace(pc.cam_pos.xyz, dir, cone_spread, sun_dir, cornell, e_view, v_view, rng);
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
