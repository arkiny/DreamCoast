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
    // F3 (HW-RT high-fidelity path, first increment): 0 = off (SW sphere-march, the scalable
    // default — bytecode-inert here so the OFF path is byte-identical to the pre-F3 shader);
    // 1 = cast the same cosine-hemisphere GI rays against the scene TLAS with an inline RayQuery
    // and return a hardware-traced VISIBILITY term instead of the SW march. Opt-in High tier
    // (`P_HWRT_GI=1`); requires the BLAS/TLAS built by rt.rs (gallery scene). This first increment
    // returns occlusion only (hit LIGHTING from the surface cache is the next increment).
    uint hwrt;
    // F4 (importance-sampled final gather, first increment): fraction of the `spp` gather rays
    // drawn from the sun-steered irradiance lobe instead of the plain cosine hemisphere (a
    // balance-heuristic MIS mixture, evaluated in gdf_bounce.slang -> `bounce_importance_dir`).
    // The estimator stays unbiased (each sample is divided by its mixture pdf), so this only
    // lowers variance at a fixed spp — it converges to the SAME mean as cosine. `0.0` (default)
    // keeps the shader on the SEPARATE legacy loop below (the gallery byte-identical anchor).
    // Shares F3's 16-byte row (hwrt at +240, this at +244 — mirrors the host packer).
    float gi_importance;
    uint gi_fine_buf;       // F4 — storage-buffer index of the fine-level AABB (2×float4:
                            // min.xyz+0, max.xyz+0); 0xFFFFFFFF = single level (legacy path
                            // untouched).
    uint vol_occ;           // bit0: GI-volume occupancy-weighted consumption (`P_GI_VOL_OCC`);
                            // bit1 (F6L, `P_GI_TRAPPED_FILL`): all-corners-rejected cells fall
                            // back to the ungated trilinear instead of hard black.
                            // Legacy meaning of the whole field:
                            // 0 = hardware trilinear (legacy, byte-identical anchor); 1 = manual
                            // 2x2x2 trilinear excluding probes whose centre sits inside geometry
                            // (the reflection fallback's sample_gi_irradiance_valid pattern).
                            // Rides the former pad slot, so 0 keeps the block layout unchanged.
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


// F4B edge-fade helpers: ONE level's volume consumption, parameterized by the level box and
// its physical row placement in the double-height volume (fine_half). Two variants mirroring
// the two consumption paths (hardware trilinear / occupancy-weighted manual trilinear) —
// kept in lockstep with the single-level blocks in csMain. `v` is meaningful only when
// pc.vol_g is bound (callers guard the write); `bent` is unit-length or zero (the scalar
// fallback), gated on the P_BENT_NORMAL flip bit like the legacy paths.
void gv_level_hw(float3 wp, float3 n, float3 lmin, float3 lmax, bool fine_half, uint fdh,
                 out float3 e, out float v, out float3 bent)
{
    e = float3(0.0, 0.0, 0.0);
    v = 1.0;
    bent = float3(0.0, 0.0, 0.0);
    float ldim = float(fdh / 2u);
    float3 t = saturate((wp - lmin) / (lmax - lmin));
    float y = (0.5 + t.y * (ldim - 1.0)) / float(fdh);
    float3 uvw = float3(t.x, fine_half ? y + 0.5 : y, t.z);
    float3 c[4];
    [unroll]
    for (int k = 0; k < 4; k++)
        c[k] = float3(0.0, 0.0, 0.0);
    [unroll]
    for (int ch = 0; ch < 3; ch++)
    {
        [unroll]
        for (int k = 0; k < 4; k++)
            c[k][ch] = g.volumes[pc.vol_r + uint(ch * 4 + k)].SampleLevel(g.samp, uvw, 0.0).x;
    }
    e = max(c[0] * SH_C0
                + (2.0 / 3.0) * (SH_C1 * n.y * c[1] + SH_C1 * n.z * c[2] + SH_C1 * n.x * c[3]),
            float3(0.0, 0.0, 0.0));
    if (pc.vol_g != 0xFFFFFFFFu)
    {
        float sc[4];
        [unroll]
        for (int k = 0; k < 4; k++)
            sc[k] = g.volumes[pc.vol_g + uint(k)].SampleLevel(g.samp, uvw, 0.0).x;
        v = saturate(sc[0] * SH_C0
                     + (2.0 / 3.0)
                           * (SH_C1 * n.y * sc[1] + SH_C1 * n.z * sc[2] + SH_C1 * n.x * sc[3]));
        if ((pc.flip_y & 2u) != 0u)
        {
            float3 b = float3(sc[3], sc[1], sc[2]);
            float bl = length(b);
            if (bl > 1e-5)
                bent = (b / bl) * saturate(1.7320508 * sc[0] / bl);
        }
    }
}

// Occupancy-weighted variant (lockstep with the coarse-only block in csMain and the
// reflection helper sample_gi_irradiance_valid).
void gv_level_occ(float3 wp, float3 n, float3 lmin, float3 lmax, bool fine_half, uint fdw,
                  uint fdh, uint fdd, out float3 e, out float v, out float3 bent)
{
    e = float3(0.0, 0.0, 0.0);
    v = 1.0;
    bent = float3(0.0, 0.0, 0.0);
    ClipMap occ_clip;
    occ_clip.desc = pc.clip_desc;
    occ_clip.count = pc.clip_count;
    float3 dims = float3(float(fdw), float(fdh / 2u), float(fdd));
    float3 lext = lmax - lmin;
    float3 f = saturate((wp - lmin) / lext) * dims - 0.5;
    float3 fl = floor(f);
    float3 fr = f - fl;
    int3 i0 = int3(fl);
    int3 imax = int3(int(fdw) - 1, int(fdh / 2u) - 1, int(fdd) - 1);
    int yoff = fine_half ? int(fdh / 2u) : 0;
    float3 c[4];
    float sc[4];
    [unroll]
    for (int k = 0; k < 4; k++)
    {
        c[k] = float3(0.0, 0.0, 0.0);
        sc[k] = 0.0;
    }
    float wsum = 0.0;
    // F6L (vol_occ bit1): parallel UNGATED accumulation. When every corner is rejected the
    // gated result had no witnesses and fell to hard black — on a VISIBLE surface that verdict
    // is a field-resolution artifact (a dense canopy bakes as a solid blob and swallows whole
    // cells), not a sealed sliver. The fallback is the plain trilinear of the STORED field,
    // whose trapped cells the injector fill (gi_volume.slang flags bit3) refills from
    // occlusion-safe neighbours — a truly sealed sliver is rejected by BOTH layers and stays
    // dark, preserving the wall-sliver semantics this gate was built for.
    float3 cu[4];
    float scu[4];
    [unroll]
    for (int k = 0; k < 4; k++)
    {
        cu[k] = float3(0.0, 0.0, 0.0);
        scu[k] = 0.0;
    }
    float wu = 0.0;
    [loop]
    for (int ci = 0; ci < 8; ci++)
    {
        int3 off = int3(ci & 1, (ci >> 1) & 1, (ci >> 2) & 1);
        float3 fw = lerp(float3(1.0, 1.0, 1.0) - fr, fr, float3(off));
        float w = fw.x * fw.y * fw.z;
        if (w <= 1e-4)
            continue;
        int3 ip = clamp(i0 + off, int3(0, 0, 0), imax);
        float3 probe = lmin + ((float3(ip) + 0.5) / dims) * lext;
        bool witness =
            min(cm_geo_inside(occ_clip, probe, pc.aabb_max.w), probe.y - GROUND_Y) > 0.0;
        int4 lp = int4(ip.x, ip.y + yoff, ip.z, 0);
        [unroll]
        for (int ch = 0; ch < 3; ch++)
        {
            [unroll]
            for (int k = 0; k < 4; k++)
            {
                float cv = g.volumes[pc.vol_r + uint(ch * 4 + k)].Load(lp).x;
                cu[k][ch] += w * cv;
                if (witness)
                    c[k][ch] += w * cv;
            }
        }
        if (pc.vol_g != 0xFFFFFFFFu)
        {
            [unroll]
            for (int k = 0; k < 4; k++)
            {
                float sv = g.volumes[pc.vol_g + uint(k)].Load(lp).x;
                scu[k] += w * sv;
                if (witness)
                    sc[k] += w * sv;
            }
        }
        wu += w;
        if (witness)
            wsum += w;
    }
    if (wsum <= 1e-4 && (pc.vol_occ & 2u) != 0u && wu > 1e-4)
    {
        [unroll]
        for (int k = 0; k < 4; k++)
        {
            c[k] = cu[k];
            sc[k] = scu[k];
        }
        wsum = wu;
    }
    if (wsum > 1e-4)
    {
        float rw = 1.0 / wsum;
        e = max((c[0] * SH_C0
                 + (2.0 / 3.0)
                       * (SH_C1 * n.y * c[1] + SH_C1 * n.z * c[2] + SH_C1 * n.x * c[3]))
                    * rw,
                float3(0.0, 0.0, 0.0));
        if (pc.vol_g != 0xFFFFFFFFu)
        {
            v = saturate((sc[0] * SH_C0
                          + (2.0 / 3.0)
                                * (SH_C1 * n.y * sc[1] + SH_C1 * n.z * sc[2]
                                   + SH_C1 * n.x * sc[3]))
                         * rw);
            if ((pc.flip_y & 2u) != 0u)
            {
                float3 b = float3(sc[3], sc[1], sc[2]);
                float bl = length(b);
                if (bl > 1e-5)
                    bent = (b / bl) * saturate(1.7320508 * sc[0] / bl);
            }
        }
    }
    else
        v = 0.0; // fully enclosed — dark, and no sky reaches it either
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
        // F4 (hierarchical radiance cache, first increment): camera-anchored FINE level, packed
        // into the SAME volumes by doubling their height (coarse half y∈[0,h/2), fine half
        // y∈[h/2,h)). A SEPARATE leading block — not a parameterized rewrite of the body below —
        // so the single-level path keeps its exact legacy instruction stream (the gi_importance
        // separate-loop precedent). Finest-first containment: the fine box answers when it holds
        // the pixel, else the scene box (else e = 0, as legacy); the y remap keeps every tap
        // >= half a texel inside its half so trilinear never crosses the level seam (mirrors
        // gv_level_uvw in gi_volume.slang).
        if (pc.gi_fine_buf != 0xFFFFFFFFu)
        {
            float3 fmin = asfloat(g.storage_buffers[pc.gi_fine_buf].Load4(0)).xyz;
            float3 fmax = asfloat(g.storage_buffers[pc.gi_fine_buf].Load4(16)).xyz;
            // F4B edge-fade margin (world metres, third buffer row; 0 = hard containment,
            // the F4 first-increment behaviour).
            float fade_m = asfloat(g.storage_buffers[pc.gi_fine_buf].Load(32));
            uint fdw, fdh, fdd;
            g.volumes[pc.vol_r].GetDimensions(fdw, fdh, fdd);
            bool fine = all(world_pos >= fmin) && all(world_pos <= fmax);
            float3 cuvw = (world_pos - GDF_MIN) / (GDF_MAX - GDF_MIN);
            bool cin = all(cuvw >= 0.0) && all(cuvw <= 1.0);
            // F4B edge-fade: the fine weight ramps 0->1 over the boundary band (smoothstep of
            // the world distance to the nearest box face) so the level hand-off has no hard
            // seam; outside the band the single-level cost is unchanged.
            float wf = 0.0;
            if (fine)
            {
                float3 dm = min(world_pos - fmin, fmax - world_pos);
                float d_edge = min(dm.x, min(dm.y, dm.z));
                wf = fade_m > 0.0 ? smoothstep(0.0, fade_m, d_edge) : 1.0;
            }
            float3 e = float3(0.0, 0.0, 0.0);
            float v = 1.0;
            float3 bent = float3(0.0, 0.0, 0.0);
            if (fine || cin)
            {
                bool want_f = wf > 0.0;
                bool want_c = wf < 1.0 && cin;
                float3 ef = float3(0.0, 0.0, 0.0), ec = float3(0.0, 0.0, 0.0);
                float vf = 1.0, vc = 1.0;
                float3 bf = float3(0.0, 0.0, 0.0), bc = float3(0.0, 0.0, 0.0);
                if (pc.vol_occ != 0u)
                {
                    if (want_f)
                        gv_level_occ(world_pos, n, fmin, fmax, true, fdw, fdh, fdd, ef, vf, bf);
                    if (want_c)
                        gv_level_occ(world_pos, n, GDF_MIN, GDF_MAX, false, fdw, fdh, fdd, ec,
                                     vc, bc);
                }
                else
                {
                    if (want_f)
                        gv_level_hw(world_pos, n, fmin, fmax, true, fdh, ef, vf, bf);
                    if (want_c)
                        gv_level_hw(world_pos, n, GDF_MIN, GDF_MAX, false, fdh, ec, vc, bc);
                }
                // Blend: fine-only where the coarse box cannot answer, else the fade weight.
                float t = !want_f ? 0.0 : (want_c ? wf : 1.0);
                e = lerp(ec, ef, t);
                v = lerp(vc, vf, t);
                // E/V are linear in the SH coefficients (reconstruct-then-lerp is exact). The
                // bent is direction × validity (F6L) — a PLAIN lerp is now the right blend: the
                // magnitude carries confidence, agreeing levels keep it, opposing bents shrink
                // toward zero and land on the consumers' bent-free contract. Renormalizing here
                // would resurrect a full-strength direction from two low-confidence inputs.
                bent = lerp(bc, bf, t);
            }
            g.storage_images[pc.out_index][tid.xy] = float4(e, 1.0);
            if (pc.vol_g != 0xFFFFFFFFu && pc.vol_b != 0xFFFFFFFFu)
                g.storage_images[pc.vol_b][tid.xy] = float4(v, bent);
            return;
        }

        // GI-volume-leak increment A (`P_GI_VOL_OCC`, opt-in): occupancy-weighted manual
        // trilinear consumption. The hardware trilinear below mixes in probes whose voxel centre
        // sits INSIDE geometry — their SH is ~0 and not a witness of this receiver's radiance, so
        // an interior receiver near a wall interpolates toward black and the field under-delivers
        // the real bounce energy (the measured F3-vs-PT deep-crop deficit,
        // docs/phase-gi-volume-leak-plan.md §1b). Port of the reflection fallback's
        // sample_gi_irradiance_valid (gdf_reflect.slang): re-do the trilinear by hand over the
        // 2x2x2 probe corners, weight each by its occupancy (probe centre in open space) and
        // renormalise. The SAME weights filter the radiance SH, the sky-vis SH and the bent
        // normal — one grid, one defect, one fix. Every corner enclosed -> E 0 and sky-vis 0
        // (a fully enclosed sliver is genuinely dark, the reflection precedent's accepted
        // falloff). A SEPARATE block (the F4 fine-level precedent above) so the OFF path below
        // keeps its exact legacy instruction stream. Coarse level only: this branch is behind
        // the fine block, and a double-height (F4) volume without an installed fine box remaps
        // to the coarse half like sample_gi_irradiance_valid.
        if (pc.vol_occ != 0u)
        {
            float3 vext = GDF_MAX - GDF_MIN;
            float3 vuvw = (world_pos - GDF_MIN) / vext;
            float3 e = float3(0.0, 0.0, 0.0);
            float v = 1.0; // outside the volume: no occlusion info (legacy semantics)
            float3 bent = float3(0.0, 0.0, 0.0);
            if (all(vuvw >= 0.0) && all(vuvw <= 1.0))
            {
                ClipMap occ_clip;
                occ_clip.desc = pc.clip_desc;
                occ_clip.count = pc.clip_count;
                uint dw, dh, dd;
                g.volumes[pc.vol_r].GetDimensions(dw, dh, dd);
                uint ldh = (dh == dw * 2u) ? dw : dh;
                float3 dims = float3(float(dw), float(ldh), float(dd));
                float3 f = saturate(vuvw) * dims - 0.5;
                float3 fl = floor(f);
                float3 fr = f - fl;
                int3 i0 = int3(fl);
                int3 imax = int3(int(dw) - 1, int(ldh) - 1, int(dd) - 1);
                float3 c[4];
                float sc[4];
                [unroll]
                for (int k = 0; k < 4; k++)
                {
                    c[k] = float3(0.0, 0.0, 0.0);
                    sc[k] = 0.0;
                }
                float wsum = 0.0;
                // F6L (vol_occ bit1): all-corners-rejected -> ungated-trilinear fallback,
                // lockstep with gv_level_occ above (see that comment for the rationale).
                float3 cu[4];
                float scu[4];
                [unroll]
                for (int k = 0; k < 4; k++)
                {
                    cu[k] = float3(0.0, 0.0, 0.0);
                    scu[k] = 0.0;
                }
                float wu = 0.0;
                [loop]
                for (int ci = 0; ci < 8; ci++)
                {
                    int3 off = int3(ci & 1, (ci >> 1) & 1, (ci >> 2) & 1);
                    float3 fw = lerp(float3(1.0, 1.0, 1.0) - fr, fr, float3(off));
                    float w = fw.x * fw.y * fw.z;
                    if (w <= 1e-4)
                        continue;
                    int3 ip = clamp(i0 + off, int3(0, 0, 0), imax);
                    float3 probe = GDF_MIN + ((float3(ip) + 0.5) / dims) * vext;
                    bool witness =
                        min(cm_geo_inside(occ_clip, probe, pc.aabb_max.w), probe.y - GROUND_Y)
                        > 0.0;
                    [unroll]
                    for (int ch = 0; ch < 3; ch++)
                    {
                        [unroll]
                        for (int k = 0; k < 4; k++)
                        {
                            float cv = g.volumes[pc.vol_r + uint(ch * 4 + k)].Load(int4(ip, 0)).x;
                            cu[k][ch] += w * cv;
                            if (witness)
                                c[k][ch] += w * cv;
                        }
                    }
                    if (pc.vol_g != 0xFFFFFFFFu)
                    {
                        [unroll]
                        for (int k = 0; k < 4; k++)
                        {
                            float sv = g.volumes[pc.vol_g + uint(k)].Load(int4(ip, 0)).x;
                            scu[k] += w * sv;
                            if (witness)
                                sc[k] += w * sv;
                        }
                    }
                    wu += w;
                    if (witness)
                        wsum += w;
                }
                if (wsum <= 1e-4 && (pc.vol_occ & 2u) != 0u && wu > 1e-4)
                {
                    [unroll]
                    for (int k = 0; k < 4; k++)
                    {
                        c[k] = cu[k];
                        sc[k] = scu[k];
                    }
                    wsum = wu;
                }
                if (wsum > 1e-4)
                {
                    float rw = 1.0 / wsum;
                    e = (c[0] * SH_C0
                         + (2.0 / 3.0)
                               * (SH_C1 * n.y * c[1] + SH_C1 * n.z * c[2] + SH_C1 * n.x * c[3]))
                        * rw;
                    e = max(e, float3(0.0, 0.0, 0.0));
                    if (pc.vol_g != 0xFFFFFFFFu)
                    {
                        v = saturate((sc[0] * SH_C0
                                      + (2.0 / 3.0)
                                            * (SH_C1 * n.y * sc[1] + SH_C1 * n.z * sc[2]
                                               + SH_C1 * n.x * sc[3]))
                                     * rw);
                        if ((pc.flip_y & 2u) != 0u)
                        {
                            float3 b = float3(sc[3], sc[1], sc[2]);
                            float bl = length(b);
                            if (bl > 1e-5)
                                bent = (b / bl) * saturate(1.7320508 * sc[0] / bl);
                        }
                    }
                }
                else
                    v = 0.0; // fully enclosed — dark, and no sky reaches it either
            }
            g.storage_images[pc.out_index][tid.xy] = float4(e, 1.0);
            if (pc.vol_g != 0xFFFFFFFFu && pc.vol_b != 0xFFFFFFFFu)
                g.storage_images[pc.vol_b][tid.xy] = float4(v, bent);
            return;
        }

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
            // Bent normal (unoccluded average direction). A zero vector = scalar fallback: the
            // lighting then samples the diffuse skylight along the surface normal (the legacy path,
            // byte-identical). A real bent normal is written only when P_BENT_NORMAL (flip bit1) is
            // set and the probe grid covers this pixel.
            float3 bent = float3(0.0, 0.0, 0.0);
            if (all(uvw >= 0.0) && all(uvw <= 1.0))
            {
                float sc[4];
                [unroll]
                for (int k = 0; k < 4; k++)
                    sc[k] = g.volumes[pc.vol_g + uint(k)].SampleLevel(g.samp, uvw, 0.0).x;
                v = sc[0] * SH_C0
                    + (2.0 / 3.0) * (SH_C1 * n.y * sc[1] + SH_C1 * n.z * sc[2] + SH_C1 * n.x * sc[3]);
                v = saturate(v);
                // The band-1 SH vector IS the bent normal: sc[k] = Σ V(ω)·(SH_C1·ω_axis), so the
                // vector (sc3, sc1, sc2) = ∫ V(ω)·ω dω = the visibility-weighted average unoccluded
                // direction (the reference DFAO Σ ConeVisibility·ConeDir, recovered for free from
                // coefficients we already sample). SH basis maps sc1→y, sc2→z, sc3→x. The scalar V
                // above stays the occlusion strength (as the reference keeps SkyVisibility separate
                // from the bent direction).
                // F6L: written DIRECTION × VALIDITY, no longer plain unit-length. Plain
                // normalization destroys the magnitude, and with it the only signal separating a
                // genuine direction from estimator noise: a real non-negative V(ω) satisfies
                // |band-1| <= sqrt(3)·band-0 (equality at a point aperture), while a converged-dark
                // cell has band-0 ~= 0 under pure-noise band-1 — normalizing that noise into a
                // full-strength random bent painted voxel-sharp dark boxes on enclosed receivers
                // (the sponza_hero cypress/curtain). validity = saturate(sqrt(3)·c0/|b|) is 1 for
                // every physically-possible direction (genuine small-V cells keep their EXACT
                // legacy unit bent — the PT gates measured real signal there) and -> 0 only where
                // the bound is violated; consumers treat magnitude as confidence and fade to their
                // bent-free contract. Scale-free (any common factor on sc cancels), no tuned knob.
                if ((pc.flip_y & 2u) != 0u)
                {
                    float3 b = float3(sc[3], sc[1], sc[2]);
                    float bl = length(b);
                    if (bl > 1e-5)
                        bent = (b / bl) * saturate(1.7320508 * sc[0] / bl);
                }
            }
            g.storage_images[pc.vol_b][tid.xy] = float4(v, bent);
        }
        return;
    }

    uint spp = max(pc.spp, 1u);
    uint seed = (tid.x * 1973u + tid.y * 9277u + pc.frame * 26699u) | 1u;
    float3 origin = world_pos + n * pc.params.y;

    // F3 — HARDWARE-RAYTRACED gather (High tier). Compiled in ONLY by the `gdf_gi_hwrt` permutation
    // (`HWRT_GI` define, see build.rs `defines_for`); the default `gdf_gi` variant strips this whole
    // block so it references no acceleration structure — Slang then omits the `g.tlas` binding,
    // keeping the scalable SW-default GI independent of RT capability and free of the RT path's
    // register pressure. The host loads this variant only when HW-RT GI is opted in (`P_HWRT_GI=1`)
    // AND the device is RT-capable AND a scene TLAS exists. Cast the same cosine-hemisphere GI rays
    // against the scene TLAS with an inline RayQuery (the ground-truth path tracer's trace
    // primitive) and accumulate a hardware-traced VISIBILITY term (1 = ray escaped / sky seen). It
    // is bound as `gi_pipeline` only when `pc.hwrt != 0`, so the runtime guard is redundant with the
    // compile-time gate but kept as a belt-and-braces no-op. PARTIAL: occlusion only; reading the
    // hit's surface-cache lighting is the next increment.
#ifdef HWRT_GI
    if (pc.hwrt != 0u)
    {
        // Phase 16 D: hardware-ray-traced GI. Cast the same cosine-hemisphere bounce rays against the
        // real triangle BVH (closest hit) and shade the hit from the surface cache — the actual
        // bounced radiance, identical to the SW march's `bs_shade_hit`, not the visibility-only first
        // increment. Accurate hits (no distance-field leaking) with the same GI look. Requires the
        // content TLAS (P_HWRT) and the ray-march GI path (P_GI_VOLUME=0; the volume path returns
        // above). The escape returns 0 — the sky is supplied by the IBL diffuse, as the SW march.
        BounceScene bs = gi_bounce_scene();
        float3 sum = float3(0.0, 0.0, 0.0);
        for (uint s = 0; s < spp; s++)
        {
            float3 dir = bounce_cosine_hemisphere(n, seed);
            RayDesc ray;
            ray.Origin = origin;
            ray.Direction = dir;
            ray.TMin = 0.0;
            ray.TMax = pc.params.x; // bounce reach = scene diagonal (same as the SW ray_max)
            RayQuery<RAY_FLAG_NONE> q;
            q.TraceRayInline(g.tlas, RAY_FLAG_NONE, 0xFF, ray);
            q.Proceed();
            float3 b = float3(0.0, 0.0, 0.0);
            if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
            {
                float3 hp = origin + dir * q.CommittedRayT();
                b = bs_shade_hit(bs, hp, bs_scene_normal(bs, hp), q.CommittedRayT());
            }
            // Firefly clamp per sample (before averaging), matching the SW path.
            float m = max(b.r, max(b.g, b.b));
            if (m > pc.clamp_max)
                b *= pc.clamp_max / m;
            sum += b;
        }
        float3 e = sum / float(spp);
        g.storage_images[pc.out_index][tid.xy] = float4(e, 1.0);
        return;
    }
#endif

    BounceScene bs = gi_bounce_scene();

    float3 sum = float3(0.0, 0.0, 0.0);
    // F4 importance-sampled final gather (`gi_importance` > 0, content opt-in): draw the gather
    // rays from a two-lobe MIS mixture (cosine BRDF lobe + a broad sun-steered power-cosine lobe,
    // exponent 4 — a diffuse gather integrates a wide hemisphere) and divide each sample by its
    // mixture pdf: unbiased (same mean as cosine), lower variance at the same spp. Kept as a
    // SEPARATE loop — not a per-sample branch — so the 0.0 path below compiles to the exact
    // legacy instruction stream (a shared loop with a `* weight` on the trace result shifted the
    // last-ulp rounding and broke the gallery's byte-identical anchor).
    if (pc.gi_importance > 0.0)
    {
        const float GI_SUN_LOBE_K = 4.0;
        for (uint s = 0; s < spp; s++)
        {
            float w;
            float3 dir =
                bounce_importance_dir(n, bs.sun_dir, GI_SUN_LOBE_K, pc.gi_importance, seed, w);
            // Firefly clamp per sample (before averaging), on the WEIGHTED sample — the MIS
            // weight is part of the sample value the clamp must bound.
            float esc;
            float3 b = bs_trace_bounce(bs, origin, dir, esc) * w;
            float m = max(b.r, max(b.g, b.b));
            if (m > pc.clamp_max)
                b *= pc.clamp_max / m;
            sum += b;
        }
    }
    else
    {
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
    }
    // Cosine-weighted (or MIS f/pdf-weighted) estimator: mean incoming radiance = indirect
    // irradiance E. The lighting pass multiplies by the surface albedo (× 1-metallic) to get L_o.
    float3 e = sum / float(spp);

    g.storage_images[pc.out_index][tid.xy] = float4(e, 1.0);
}
