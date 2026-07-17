// World-space directional irradiance volume (radiance-cache probes). One thread per volume
// probe (voxel). Each probe casts `spp` uniform-sphere rays into the scene GDF; at a hit it
// shades the surface with direct sun (soft-shadowed) + a sky fill + the PREVIOUS frame's
// volume sampled at the hit (= multibounce), on a miss it adds the sky. Instead of one scalar
// average per probe, the incoming radiance L(dir) is projected onto a band-0+1 spherical-
// harmonic basis (4 coefficients per colour channel) and EMA-accumulated per coefficient.
// Because each probe samples the volume itself, bright regions (a sunlit atrium) propagate
// into dark ones (an enclosed nave) over frames — the infinite-bounce fill a single-bounce
// trace can't reach.
//
// Storing direction (not just the omnidirectional mean) is what lets a coloured surface bleed
// onto its neighbours: a receiver reconstructs the cosine-weighted hemispherical average in
// ITS normal direction, so a floor next to a red curtain weights the up/curtain hemisphere and
// keeps the colour the full-sphere average washed out to ~0.
//
// Layout: 12 R32F volumes per ping-pong slot — slot index = channel*4 + coeff (R coeffs 0..3,
// G coeffs 4..7, B coeffs 8..11). The 12 are allocated contiguously, so only the base bindless
// index is passed (read_rgb.x = sampled base of the previous slot, write_rgb.x = storage base
// of this slot); read_rgb.w = reset.
//
// Sky-visibility (레퍼런스식 indoor skylight occlusion): the probe also projects a SCALAR directional
// sky-visibility onto band-0/1 SH (4 R32F volumes/slot) — 1 on a sky-escape, 0 on a geometry hit. A
// receiver reconstructs the cosine-weighted hemispherical sky visibility V(n); the lighting occludes
// the IBL DIFFUSE skylight by it (with a tint/min-occlusion leak), so an enclosed interior isn't
// flooded by the full (blue) diffuse skylight. bases: read_rgb.y = sampled, write_rgb.y = storage.

#include "bindless.slang"
#include "clipmap.slang"
#include "sky_common.slang"

static const float GV_PI = 3.14159265359;
// F6I: average opacity of ONE unresolved surface band for the partial-occlusion march
// (open-mesh seam only). Swept in the E/lit/V oracle domain; see the march comment.
static const float GV_TAU = 0.4;

// Real SH band-0/1 basis normalisation: Y0 = sqrt(1/4pi), Y1 = sqrt(3/4pi)*{y,z,x}.
static const float SH_C0 = 0.2820947918; // sqrt(1/(4*pi))
static const float SH_C1 = 0.4886025119; // sqrt(3/(4*pi))

struct GiVolumePush
{
    float4 aabb_min;     // xyz volume/world min, w = ground plane Y
    float4 aabb_max;     // xyz volume/world max, w = sample distance clamp
    float4 sun;          // xyz direction toward the sun, w = intensity
    uint4 dims;          // x,y,z = probe grid resolution, w = frame (RNG)
    uint4 read_rgb;      // x = sampled base of the previous radiance SH volumes; y = sampled base of
                         // the previous sky-vis SH volumes (0xFFFFFFFF = sky-vis off); z = z-slice
                         // offset of this dispatch's slab (cost amortization: the host updates
                         // dims.z/period slices per frame so every texel refreshes once per period
                         // at a FLAT per-frame cost — a whole-grid burst every Nth frame perturbed
                         // the wall-clock auto-exposure into visible luma pumping); w = reset
    uint4 write_rgb;     // x = storage base of the radiance SH volumes; y = storage base of the
                         // sky-vis SH volumes written this frame; z = tid.y offset of this
                         // dispatch's level (F4B level-interleaved slabs: 0 = coarse rows,
                         // dims.y = the fine half; 0 with a full-height dispatch = legacy)
    uint4 albedo_rgb;    // sampled volumes[] indices of the per-voxel albedo (R,G,B); x=0xFFFFFFFF = constant
    uint4 clip;          // x = clipmap descriptor index, y = level count
    float4 params;       // x = spp, y = ray max dist, z = sky fill, w = EMA alpha
    float4 ground;       // xyz analytic ground albedo, w = surface bias
    float4 fine_min;     // F4: xyz fine-level world min, w = fine_active (0 = single level)
    float4 fine_max;     // F4: xyz fine-level world max (fine half packed at y ∈ [dims.y, 2*dims.y))
    uint flags;          // E-oracle repair seam: bit0 = feedback reads offset one voxel toward
                         // the ray side (no cross-wall import), bit1 = intersection-only sun
                         // visibility (no near-miss penumbra); bit2 = unsigned-safe hit epsilon
                         // (F6I: an |d| sheet band can be stepped over by the 0.02 m min step —
                         // eps 0.012 makes a perpendicular crossing mathematically un-skippable,
                         // 0.012 + 0.012 > 0.02). 0 = the exact legacy estimator.
    float sun_k;         // F6F: gv_shadow penumbra cone slope — 0 = the reference-derived
                         // SUN_PENUMBRA_K (single source with the path tracer's sun disk),
                         // >0 = explicit override (`P_GI_SUN_K`; 16 = the legacy slope).
};

[[vk::push_constant]]
ConstantBuffer<GiVolumePush> pc : register(b0, space0);

#define GROUND_Y (pc.aabb_min.w)

uint pcg(inout uint s) { s = s * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28) + 4u)) ^ s) * 277803737u; return (w >> 22) ^ w; }
float rnd(inout uint s) { return float(pcg(s) & 0xFFFFFFu) / float(0x1000000); }

ClipMap gv_clip() { ClipMap c; c.desc = pc.clip.x; c.count = pc.clip.y; return c; }
float geo_inside(float3 p) { return cm_geo_inside(gv_clip(), p, pc.aabb_max.w); }
float geo_march(float3 p) { return cm_geo_march(gv_clip(), p, pc.aabb_max.w); }
float scene_march(float3 p) { return min(geo_march(p), p.y - GROUND_Y); }
float scene_occ(float3 p) { return min(geo_inside(p), p.y - GROUND_Y); }
float3 albedo_at(float3 p) { return cm_albedo(gv_clip(), p, 0.7); }

float3 scene_normal(float3 p)
{
    const float e = 0.05;
    float2 k = float2(1.0, -1.0);
    float3 g = k.xyy * scene_occ(p + k.xyy * e) + k.yyx * scene_occ(p + k.yyx * e)
               + k.yxy * scene_occ(p + k.yxy * e) + k.xxx * scene_occ(p + k.xxx * e);
    return safe_normalize(g, float3(0.0, 1.0, 0.0));
}

float gv_shadow(float3 o, float3 d)
{
    // F6F: penumbra cone slope. The legacy 16 emulates a ~3.6° cone — ×3.1 the REFERENCE
    // sun's angular radius (SUN_COS_MAX, ~1.15°), so a ray passing 10 cm from geometry 5 m
    // out read 68% shadowed. sun_k = 0 selects the reference-matched SUN_PENUMBRA_K (~50,
    // single source with the path tracer's sampled sun disk); `P_GI_SUN_HARDVIS` (flags
    // bit1) is the k→∞ end point (intersection kill only — kept for diagnostics; it
    // over-injects on the sunlit gate, F6E §4c).
    float k = pc.sun_k > 0.0 ? pc.sun_k : SUN_PENUMBRA_K;
    float res = 1.0, t = 0.05, maxt = pc.params.y;
    for (int i = 0; i < 32; i++)
    {
        float h = scene_occ(o + d * t);
        // F6I: unresolved-band sun occlusion (bit2 = the open-mesh seam). The content
        // field blurs sub-voxel structure, so eroded sheet bands have HOLES a 12 mm
        // kill slips through — the sun lit nearly the whole interior floor (footprint
        // audit, plan 2g). Passing within 0.2 m of the band IS passing through
        // unresolved geometry at this resolution: killing there lands the E total at
        // x1.10 of the reference (26.0 vs 23.71; band 0.1 -> 44, 0.3 -> 24) with the
        // sky-visibility march untouched. The residual blue skew is the sky_fill
        // share and belongs to the next spectral unit, not this band.
        if (h < ((pc.flags & 4u) != 0u ? 0.2 : 0.003))
            return 0.0;
        if ((pc.flags & 2u) == 0u)
            res = min(res, k * h / t);
        t += clamp(h, 0.02, 0.4);
        if (t > maxt)
            break;
    }
    return saturate(res);
}

// Trilinear sample of the 12 SH coefficient volumes (base index `base`) at world position p,
// filling c[coeff].rgb. Out of the volume AABB -> all zero. coeff `k` channel `ch` lives in
// volume `base + ch*4 + k`.
void read_coeffs(uint base, float3 p, out float3 c[4])
{
    [unroll]
    for (int k = 0; k < 4; k++)
        c[k] = float3(0.0, 0.0, 0.0);
    float3 uvw = (p - pc.aabb_min.xyz) / (pc.aabb_max.xyz - pc.aabb_min.xyz);
    if (any(uvw < 0.0) || any(uvw > 1.0))
        return;
    uvw = saturate(uvw);
    [unroll]
    for (int ch = 0; ch < 3; ch++)
    {
        [unroll]
        for (int k = 0; k < 4; k++)
            c[k][ch] = g.volumes[base + uint(ch * 4 + k)].SampleLevel(g.samp, uvw, 0.0).x;
    }
}

// Trilinear sample of the 4 SCALAR sky-visibility SH coefficient volumes (base index `base`).
void read_coeffs_scalar(uint base, float3 p, out float c[4])
{
    [unroll]
    for (int k = 0; k < 4; k++)
        c[k] = 0.0;
    float3 uvw = (p - pc.aabb_min.xyz) / (pc.aabb_max.xyz - pc.aabb_min.xyz);
    if (any(uvw < 0.0) || any(uvw > 1.0))
        return;
    uvw = saturate(uvw);
    [unroll]
    for (int k = 0; k < 4; k++)
        c[k] = g.volumes[base + uint(k)].SampleLevel(g.samp, uvw, 0.0).x;
}

// F4 fine mode: both levels live in one volume of height 2*dims.y — map a world position to the
// finest containing level's half (fine box first, the clipmap containment convention), keeping
// every tap >= half a texel inside that half so trilinear never crosses the level seam (the same
// voxel-center clamp SdfVolume::sample uses). `inside` is false only when p is outside BOTH boxes
// (the caller returns zeros, matching the legacy out-of-box behaviour). Called ONLY when
// fine_active (the legacy functions above stay the untouched single-level instruction stream).
float3 gv_level_uvw(float3 p, out bool inside)
{
    float dh2 = 2.0 * float(pc.dims.y);
    // F4B recentering (fine_max.w): while the box reconverges, the previous slot's fine half
    // holds OLD-box data — reading it at new-box coordinates would pollute the multibounce /
    // B2 sky-vis hit terms, so the containment falls through to the coarse level for the
    // window. 0 = the legacy finest-first behaviour, bit for bit.
    if (pc.fine_max.w < 0.5 && all(p >= pc.fine_min.xyz) && all(p <= pc.fine_max.xyz))
    {
        inside = true;
        float3 t = saturate((p - pc.fine_min.xyz) / (pc.fine_max.xyz - pc.fine_min.xyz));
        return float3(t.x, (0.5 + t.y * (float(pc.dims.y) - 1.0)) / dh2 + 0.5, t.z);
    }
    float3 t = (p - pc.aabb_min.xyz) / (pc.aabb_max.xyz - pc.aabb_min.xyz);
    if (any(t < 0.0) || any(t > 1.0))
    {
        inside = false;
        return float3(0.0, 0.0, 0.0);
    }
    inside = true;
    t = saturate(t);
    return float3(t.x, (0.5 + t.y * (float(pc.dims.y) - 1.0)) / dh2, t.z);
}

// F4: trilinear read of the 12 SH volumes at an ALREADY-REMAPPED fine-mode uvw (gv_level_uvw, or
// a probe's own texel centre for the EMA history). `inside == false` returns zeros.
void read_coeffs_fine(uint base, float3 uvw, bool inside, out float3 c[4])
{
    [unroll]
    for (int k = 0; k < 4; k++)
        c[k] = float3(0.0, 0.0, 0.0);
    if (!inside)
        return;
    [unroll]
    for (int ch = 0; ch < 3; ch++)
    {
        [unroll]
        for (int k = 0; k < 4; k++)
            c[k][ch] = g.volumes[base + uint(ch * 4 + k)].SampleLevel(g.samp, uvw, 0.0).x;
    }
}

// F4: scalar (sky-visibility) variant of read_coeffs_fine.
void read_coeffs_scalar_fine(uint base, float3 uvw, bool inside, out float c[4])
{
    [unroll]
    for (int k = 0; k < 4; k++)
        c[k] = 0.0;
    if (!inside)
        return;
    [unroll]
    for (int k = 0; k < 4; k++)
        c[k] = g.volumes[base + uint(k)].SampleLevel(g.samp, uvw, 0.0).x;
}

// Cosine-weighted hemispherical average radiance in direction n, reconstructed from the SH
// radiance coefficients (the clamped-cosine convolution: A0/PI = 1, A1/PI = 2/3). For an
// isotropic field (only the band-0 coeff) this returns exactly the omnidirectional mean, so it
// is a strict directional generalisation of the old scalar probe. Clamped >= 0 (band-1 SH can
// ring negative).
float3 eval_irradiance(float3 c[4], float3 n)
{
    float3 e = c[0] * SH_C0
               + (2.0 / 3.0) * (SH_C1 * n.y * c[1] + SH_C1 * n.z * c[2] + SH_C1 * n.x * c[3]);
    return max(e, float3(0.0, 0.0, 0.0));
}

// Uniform sphere direction.
float3 uniform_sphere(inout uint rng)
{
    float z = rnd(rng) * 2.0 - 1.0;
    float a = rnd(rng) * 2.0 * GV_PI;
    float r = sqrt(max(0.0, 1.0 - z * z));
    return float3(r * cos(a), z, r * sin(a));
}

// Octant-stratified sphere direction: sample `s` is folded into octant `s % 8` (uniform within
// the octant). With spp a multiple of 8 every octant gets the same sample count, so the plain
// `4pi/spp` Monte-Carlo weight below stays EXACTLY correct (stratification changes variance,
// not the estimator) while killing the octant-level luck that made the EMA field breathe —
// the per-mesh shimmer the live full-grid field exposed. Callers gate on `spp % 8 == 0`.
float3 stratified_sphere(uint s, inout uint rng)
{
    float3 u = uniform_sphere(rng);
    float3 a = float3(abs(u.x), abs(u.y), abs(u.z));
    return float3((s & 1u) != 0u ? -a.x : a.x, (s & 2u) != 0u ? -a.y : a.y,
                  (s & 4u) != 0u ? -a.z : a.z);
}

[shader("compute")]

// Shade a march band/hit point: direct sun (soft-shadowed) + the V-gated sky fill +
// the previous frame's stored irradiance (multibounce), returning the outgoing
// radiance toward the probe. Shared by the legacy binary march and the F6I
// partial-occlusion march (both pass the same quantities; pc supplies the rest).
float3 gv_shade_hit(float3 hp, float3 sun_dir, uint read_base, bool fine_on)
{
    float3 hn = scene_normal(hp);
    float ndl = max(dot(hn, sun_dir), 0.0);
    float sh = ndl > 0.0 ? gv_shadow(hp + hn * 0.02, sun_dir) : 1.0;
    bool is_ground = (hp.y - GROUND_Y) < geo_inside(hp);
    float3 alb = is_ground ? pc.ground.xyz : albedo_at(hp);
    // Sky fill at the bounce hit: the ACTUAL procedural-sky radiance in the hit's
    // upper hemisphere (single source of truth with the IBL capture / path tracer /
    // sdf_cache_light), NOT a flat pre-photometric constant. `pc.params.z` now carries
    // the sky gain (was the old flat fill value); sun intensity is `pc.sun.w`. This is
    // the fix for interiors reading near-black: the old constant was ~5 orders of
    // magnitude below the lux-scaled sun, so occluded/bounce-lit geometry lost all fill.
    float3 sky_fill = procedural_sky(hn, sun_dir, pc.sun.w, pc.params.z, false)
                      * (0.5 + 0.5 * hn.y);
    // Multibounce: the hit surface re-emits its own indirect irradiance, read
    // DIRECTIONALLY from the previous-frame SH volume in the hit's normal direction.
    // The DIRECT sun + sky fill are irradiance, so they get the albedo/PI BRDF; the
    // stored irradiance (E) re-enters as albedo*indirect (the ×PI to irradiance cancels
    // the BRDF 1/PI), matching the old scalar probe's energy.
    float3 cprev[4];
    // `P_GI_READ_OFFSET` (flags bit0): the hit sits ON geometry, so a plain trilinear
    // feedback read straddles the wall and imports the far side's radiance — a transport
    // path the reference doesn't have (measured feedback gain ×2.2 vs the ~1.25 a
    // closed room's albedo supports, most of it sky-blue). Reading one voxel toward the
    // ray side keeps the footprint on the hemisphere the ray actually came from.
    float3 hp_read = hp;
    if ((pc.flags & 1u) != 0u)
        hp_read = hp + hn * ((pc.aabb_max.xyz - pc.aabb_min.xyz) * (1.0 / 32.0));
    // F4 fine mode: read the finest level containing the hit (fine-first fall-through);
    // the OFF path keeps the exact legacy call (untouched instruction stream).
    if (fine_on)
    {
        bool cin;
        float3 cuvw = gv_level_uvw(hp_read, cin);
        read_coeffs_fine(read_base, cuvw, cin, cprev);
    }
    else
        read_coeffs(read_base, hp_read, cprev);
    float3 indirect = eval_irradiance(cprev, hn);
    // Occlude the sky fill by the PREVIOUS frame's sky-visibility V(hn) at the hit
    // (single source with the deferred skylight occlusion — no new state): a hit deep
    // inside an enclosed interior must not receive the open-sky fill. Before the
    // full-grid dispatch fix this term looked minor (3/4 of the probes never ran);
    // on the working field it flooded every interior with sky-blue E (crop B-R +12
    // vs the reference's -0.6). The reset frame (no history) and a missing sky-vis
    // binding fall back to V = 1 — the unoccluded legacy term, so the EMA bootstrap
    // starts bright and converges down instead of crushing to black.
    if (pc.read_rgb.y != 0xFFFFFFFFu && pc.read_rgb.w == 0u)
    {
        float svprev[4];
        // The sky-vis feedback read shares the wall-straddling geometry — same offset.
        if (fine_on)
        {
            bool sin_;
            float3 suvw = gv_level_uvw(hp_read, sin_);
            read_coeffs_scalar_fine(pc.read_rgb.y, suvw, sin_, svprev);
        }
        else
            read_coeffs_scalar(pc.read_rgb.y, hp_read, svprev);
        float hv = svprev[0] * SH_C0
                   + (2.0 / 3.0)
                         * (SH_C1 * hn.y * svprev[1] + SH_C1 * hn.z * svprev[2]
                            + SH_C1 * hn.x * svprev[3]);
        sky_fill *= saturate(hv);
    }
    return alb * (pc.sun.w * ndl * sh + sky_fill) * (1.0 / GV_PI) + alb * indirect;
}


[numthreads(4, 4, 4)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    // Slab amortization: this dispatch covers z slices [read_rgb.z, read_rgb.z + gz*4); offset
    // the thread id so every downstream tid use (probe position, EMA texel, writes) is exact.
    // A zero offset with the full dispatch is the legacy whole-grid update, bit for bit.
    tid.z += pc.read_rgb.z;
    // F4B level-interleaved slabs: a fine-mode dispatch covers ONE level per frame (write_rgb.z
    // = 0 selects the coarse rows, GI_VOL_DIM the fine half), so the per-frame cost stays the
    // single-level slab instead of doubling. Applied BEFORE the guard/seed/probe/EMA reads so
    // the offset tid.y reproduces the exact physical texel (and thus the exact deterministic
    // direction set) the old double-height dispatch gave that probe. Zero = legacy, bit for bit.
    tid.y += pc.write_rgb.z;
    // F4: fine mode processes a second (camera-anchored) level packed in the volume's upper
    // half. When fine_active == 0 these expressions reduce to the legacy values exactly
    // (levels == 1, level == 0, lid == tid, lvl_* == aabb_* — arithmetic-identical float math,
    // and the tid.y bound `dims.y * 1` equals the legacy bound).
    bool fine_on = pc.fine_min.w > 0.5;
    uint levels = fine_on ? 2u : 1u;
    if (tid.x >= pc.dims.x || tid.y >= pc.dims.y * levels || tid.z >= pc.dims.z)
        return;
    uint level = tid.y / pc.dims.y; // 0 = coarse (scene box), 1 = fine (camera box)
    uint3 lid = uint3(tid.x, tid.y % pc.dims.y, tid.z);
    float3 lvl_min = level == 0u ? pc.aabb_min.xyz : pc.fine_min.xyz;
    float3 lvl_max = level == 0u ? pc.aabb_max.xyz : pc.fine_max.xyz;

    // Probe world position = voxel centre mapped through the level's AABB.
    float3 uvw = (float3(lid) + 0.5) / float3(pc.dims.xyz);
    float3 probe = lerp(lvl_min, lvl_max, uvw);

    uint spp = uint(max(pc.params.x, 1.0));
    uint seed = (tid.x * 1973u + tid.y * 9277u + tid.z * 26699u + pc.dims.w * 51059u) | 1u;
    float3 sun_dir = normalize(pc.sun.xyz);
    float bias = pc.ground.w;
    float maxt = pc.params.y;
    uint read_base = pc.read_rgb.x;

    // SH band-0/1 accumulators (4 coeffs x RGB) + scalar sky-visibility (4 coeffs).
    float3 sh0 = float3(0.0, 0.0, 0.0);
    float3 sh1 = float3(0.0, 0.0, 0.0);
    float3 sh2 = float3(0.0, 0.0, 0.0);
    float3 sh3 = float3(0.0, 0.0, 0.0);
    float sv0 = 0.0, sv1 = 0.0, sv2 = 0.0, sv3 = 0.0;
    // Octant-stratified when spp is a multiple of 8 (equal per-octant counts keep the 4pi/spp
    // estimator exact); odd spp keeps the legacy plain-uniform draw.
    bool strat = (spp & 7u) == 0u;
    for (uint s = 0; s < spp; s++)
    {
        float3 dir = strat ? stratified_sphere(s, seed) : uniform_sphere(seed);
        float3 radiance = float3(0.0, 0.0, 0.0);
        float vis;
        if ((pc.flags & 4u) != 0u)
        {
            // F6I partial-occlusion march (the open-mesh seam): the content field
            // cannot resolve sub-voxel structure, so a surface band is a PARTIAL
            // occluder — a binary march must either leak through band holes (the sun
            // and sky floods, plan 2h/2i) or over-block the upward hemisphere (the V
            // collapse). Instead: each outside->inside band transition sheds
            // (1 - GV_TAU) of the remaining transmittance into that surface's shading
            // and carries GV_TAU onward; the sky visibility is the transmittance that
            // survives to escape. GV_TAU is the honest average opacity of one
            // unresolved band (swept in the E/lit/V domain).
            float T = 1.0;
            float t = bias;
            bool was_in = false;
            bool shaded = false;
            for (int i = 0; i < 48; i++)
            {
                float3 p = probe + dir * t;
                float d = scene_march(p);
                // Radiance keeps the LEGACY first-hit semantics (12 mm, tunnel-proof with
                // the 0.02 in-band crawl): a thick wall's "partial" band transparency is a
                // resolution artifact, and shading anything beyond the first surface
                // re-imports the sun/sky floods (measured: tau-weighted accumulation AND
                // first-encounter-at-band both land E ~82, plan 2j).
                if (!shaded && d < 0.012)
                {
                    radiance = gv_shade_hit(p, sun_dir, read_base, fine_on);
                    shaded = true;
                }
                // Sky VISIBILITY marches on as transmittance: for the upward hemisphere the
                // unresolved cloth/rail layer really is a partial occluder — each
                // outside->inside band transition keeps GV_TAU of the remaining
                // transmittance (V 62.6 -> 82.6 at tau 0.4).
                bool in_band = d < 0.2;
                if (in_band && !was_in)
                    T *= GV_TAU;
                if (shaded && T < 0.05)
                {
                    T = 0.0;
                    break;
                }
                was_in = in_band;
                t += in_band ? 0.02 : max(d, 0.02);
                if (t > maxt)
                    break;
            }
            vis = T < 0.05 ? 0.0 : T;
        }
        else
        {
            float t = bias;
            bool hit = false;
            float3 hp = probe;
            for (int i = 0; i < 48; i++)
            {
                float3 p = probe + dir * t;
                float d = scene_march(p);
                if (d < 0.003)
                {
                    hp = p;
                    hit = true;
                    break;
                }
                t += max(d, 0.02);
                if (t > maxt)
                    break;
            }
            if (hit)
                radiance = gv_shade_hit(hp, sun_dir, read_base, fine_on);
            vis = hit ? 0.0 : 1.0;
        }
        // Project L(dir) onto the band-0/1 SH basis.
        sh0 += radiance * SH_C0;
        sh1 += radiance * (SH_C1 * dir.y);
        sh2 += radiance * (SH_C1 * dir.z);
        sh3 += radiance * (SH_C1 * dir.x);
        // Sky-visibility: the surviving transmittance (legacy: 1 on escape, 0 on hit) —
        // projected the same way so a receiver reconstructs the cosine-weighted V(n).
        sv0 += vis * SH_C0;
        sv1 += vis * (SH_C1 * dir.y);
        sv2 += vis * (SH_C1 * dir.z);
        sv3 += vis * (SH_C1 * dir.x);
    }
    // Monte-Carlo estimator for uniform-sphere sampling (pdf = 1/4pi): coeff = (4pi/N) * sum.
    float kw = (4.0 * GV_PI) / float(spp);
    float3 c[4] = { sh0 * kw, sh1 * kw, sh2 * kw, sh3 * kw };
    float sv[4] = { sv0 * kw, sv1 * kw, sv2 * kw, sv3 * kw };

    // EMA with the previous frame (per coefficient; reset on the first frame / re-alloc).
    if (pc.read_rgb.w == 0u)
    {
        // F4 fine mode: the EMA history must read the probe's OWN level — a coarse probe whose
        // world position falls inside the fine box must NOT blend the fine half into its history
        // (finest-first containment is for consumers, not the self-EMA). The probe's own texel
        // centre in the packed volume is exact from tid, preserving the legacy semantic (the
        // legacy read at `probe` lands on the probe's own texel centre too).
        float3 puvw = float3((float(tid.x) + 0.5) / float(pc.dims.x),
                             (float(tid.y) + 0.5) / (float(pc.dims.y) * float(levels)),
                             (float(tid.z) + 0.5) / float(pc.dims.z));
        // F4B recentering: a fine-level probe's own history is OLD-box data while the box
        // reconverges (fine_max.w) — skip BOTH blends (radiance AND sky-vis: the B2 gating and
        // the deferred skylight both consume the sky-vis half) and write this cycle's estimate
        // fresh, exactly like the global reset. Coarse probes blend as normal.
        bool fine_ema_reset = level == 1u && pc.fine_max.w > 0.5;
        if (!fine_ema_reset)
        {
            float3 prev[4];
            if (fine_on)
                read_coeffs_fine(read_base, puvw, true, prev);
            else
                read_coeffs(read_base, probe, prev);
            [unroll]
            for (int i = 0; i < 4; i++)
                c[i] = lerp(prev[i], c[i], pc.params.w);
            if (pc.read_rgb.y != 0xFFFFFFFFu)
            {
                float svprev[4];
                if (fine_on)
                    read_coeffs_scalar_fine(pc.read_rgb.y, puvw, true, svprev);
                else
                    read_coeffs_scalar(pc.read_rgb.y, probe, svprev);
                [unroll]
                for (int i = 0; i < 4; i++)
                    sv[i] = lerp(svprev[i], sv[i], pc.params.w);
            }
        }
    }
    [unroll]
    for (int i = 0; i < 4; i++)
        c[i] = make_finite(c[i]);

    uint write_base = pc.write_rgb.x;
    [unroll]
    for (int ch = 0; ch < 3; ch++)
    {
        [unroll]
        for (int k = 0; k < 4; k++)
            g.storage_volumes[write_base + uint(ch * 4 + k)][tid] = c[k][ch];
    }

    // Sky-visibility SH (4 scalar volumes), when bound.
    if (pc.write_rgb.y != 0xFFFFFFFFu)
    {
        uint sv_base = pc.write_rgb.y;
        [unroll]
        for (int k = 0; k < 4; k++)
            g.storage_volumes[sv_base + uint(k)][tid] = make_finite(float3(sv[k], 0.0, 0.0)).x;
    }
}
