// Phase 11 Stage C6 — GDF reflections (off-screen fallback for SSR misses).
//
// A full-screen compute pass: for each G-buffer pixel it reflects the view ray about
// the world normal and sphere-traces that reflection ray through the world scene GDF
// (the same baked field + helpers as the C3 GI / B4 trace). On a hit it re-lights the
// hit point (gradient normal + soft-shadowed sun + a small sky fill, constant albedo —
// the GDF carries no material, like C3); on escape it returns the procedural sky.
//
// This is the off-screen counterpart to C5's screen-space reflections: SSR is accurate
// where the reflected surface is on screen but misses off-screen / disoccluded / screen-
// edge rays, which this fills with plausible GDF reflections. C7 composites the two
// (SSR where confident, GDF otherwise, sky on the GDF miss) over the IBL specular.
//
// KEY vs C3 diffuse GI: a GDF ray that escapes returns the SKY, not 0. Specular misses
// must show the sky (a mirror reflects the sky); C3's diffuse misses return 0 because
// the IBL diffuse already supplies the sky there. Output is raw radiance (the tonemap
// applies exposure), like the C1 scene-GDF trace.

#include "bindless.slang"
#include "clipmap.slang"
#include "surface_cache.slang"
#include "reflect_ggx.slang" // A5: shared blue-noise GGX sampler (content path; kept in lockstep)

struct ReflectPush
{
    float4x4 inv_view_proj; // clip -> world (surface reconstruction)
    float4 cam_pos;         // xyz camera world position
    float4 sun;             // xyz direction toward the sun, w = intensity
    uint depth_index;       // sampled G-buffer depth
    uint normal_index;      // sampled G-buffer world normal
    uint skip_packed;       // A3 adaptive skip, byte-packed: read | write<<8 | K<<16 | frame<<24
                            //   (was gdf_sampled — unused; clip_desc drives sampling). 0xFF = disabled.
    uint out_index;         // storage image (reflection color)
    uint width;
    uint height;
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
    uint material_index;    // C8j: G-buffer material (g = roughness) for the GGX-sampled ray
    float4 aabb_min;        // xyz GDF world min, w = ground plane Y
    float4 aabb_max;        // xyz GDF world max, w = sample distance clamp
    float4 params;          // x = ray max distance, y = constant hit albedo (fallback), z = sky fill, w = surface bias
    uint albedo_r;          // C8a: volumes[] index of the red albedo channel (0xFFFFFFFF = constant)
    uint albedo_g;          // green
    uint albedo_b;          // blue
    uint frame;             // C8j: RNG decorrelation for the per-frame GGX jitter
    uint4 cache;            // C8b3: cards, cache_pos, cache_rad, num_cards (cards=0xFFFFFFFF = off)
    uint cache_tile;        // surface-cache atlas tile edge
    uint clip_desc;         // Stage B: clipmap descriptor storage-buffer index
    uint clip_count;        // Stage B: clipmap level count (1 = single volume = legacy)
    uint max_steps;         // Stage D3: reflection-ray march step cap (96 = legacy / gallery)
    // xyz = analytic ground material (not in the albedo volume); w = cone_k (P3 cone-trace LOD
    // slope; 0 = legacy linear march). Packed as ONE float4, NOT float3 + a trailing scalar:
    // HLSL/SPIR-V pack a scalar after a float3 at +12, but MSL pads the float3 to +16, so the
    // old layout mis-read cone_k and bloated the Metal push struct to 256 bytes (validation
    // failure + wrong reflections). A float4's .w sits at +12 on all three backends.
    float4 ground_albedo;
};

// A3 adaptive temporal skip: reuse the previous frame's traced reflection radiance for a pixel whose
// surface point is unchanged (world-pos gate) instead of re-marching. The ping-pong storage-buffer
// indices + stagger period + frame are byte-packed into `skip_packed` (keeps the push at 240 B to
// avoid a D3D12 root-CBV spill). read/write == 0xFF ⇒ disabled (gallery / viz / lighting-in-flux).
struct SkipCfg { uint read; uint write; uint k; uint frame; };
SkipCfg unpack_skip()
{
    SkipCfg s;
    s.read = pc.skip_packed & 0xFFu;
    s.write = (pc.skip_packed >> 8) & 0xFFu;
    s.k = (pc.skip_packed >> 16) & 0xFFu;
    s.frame = (pc.skip_packed >> 24) & 0xFFu;
    return s;
}

[[vk::push_constant]]
ConstantBuffer<ReflectPush> pc : register(b0, space0);

#ifdef HWRT_REFLECT
// Per-frame globals (set 1 / b1) — the same UBO the deferred PBR / SSR passes bind (opted into via
// the compute pipeline's `uniform_buffer`). Only `prev_view_proj` is read, for the B.2 screen-color-
// at-hit reprojection; the layout must match the host `Globals` (mirrors ssr.slang's subset).
struct Globals
{
    float4 camera_pos;
    float4 sun_direction;
    float4 sun_color;
    float4 ambient;
    int4 counts;
    float4 point_pos[4];
    float4 point_color[4];
    float4x4 light_view_proj;
    float4 shadow;
    float4x4 inv_view_proj;
    int4 ibl;
    float4 probe;
    float4 probe_box_min;
    float4 probe_box_max;
    float4x4 prev_view_proj; // world -> previous clip
};

[[vk::binding(0, 1)]]
ConstantBuffer<Globals> globals : register(b1, space0);
#endif

#define GDF_MIN (pc.aabb_min.xyz)
#define GDF_MAX (pc.aabb_max.xyz)
#define GROUND_Y (pc.aabb_min.w)

// C8j — GGX importance sampling for a stochastic reflection ray (레퍼런스 엔진 SW GI casts ONE GGX ray /
// pixel into the real lobe directions, then denoises; an image-space prefilter of a single mirror
// ray can't average in the off-screen sky a rough lobe sees, so a rough metal's reflection of the
// bright ground reads too bright). One jittered ray/frame; the temporal resolve accumulates them.
static const float GGX_PI = 3.14159265359;
uint pcg_hash(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}
float rnd01(inout uint s) { return float(pcg_hash(s) & 0xFFFFFFu) / float(0x1000000); }
void onb(float3 n, out float3 t, out float3 b)
{
    t = normalize(abs(n.x) > 0.9 ? cross(n, float3(0, 1, 0)) : cross(n, float3(1, 0, 0)));
    b = cross(n, t);
}
// GGX NDF importance sample: a microfacet half-vector around the surface normal.
float3 sample_ggx(float3 n, float roughness, inout uint rng)
{
    float a = max(roughness * roughness, 1e-3);
    float u1 = rnd01(rng), u2 = rnd01(rng);
    float phi = 2.0 * GGX_PI * u1;
    float ct = sqrt((1.0 - u2) / (1.0 + (a * a - 1.0) * u2));
    float st = sqrt(saturate(1.0 - ct * ct));
    float3 t, bvec;
    onb(n, t, bvec);
    return normalize(t * (st * cos(phi)) + bvec * (st * sin(phi)) + n * ct);
}

ClipMap reflect_clip()
{
    ClipMap c;
    c.desc = pc.clip_desc;
    c.count = pc.clip_count;
    return c;
}

// Distance to the baked geometry via the clipmap (single level ⇒ identical to the legacy
// single-volume sample). Outside every level: no geometry.
float geo_inside(float3 p)
{
    return cm_geo_inside(reflect_clip(), p, pc.aabb_max.w);
}

// Marching distance: outside the box, return the (clamped) box distance so the empty
// bounding box never reads as a surface while the ray steps sensibly.
float geo_march(float3 p)
{
    return cm_geo_march(reflect_clip(), p, pc.aabb_max.w);
}

float scene_march(float3 p)
{
    return min(geo_march(p), p.y - GROUND_Y);
}
float scene_occ(float3 p)
{
    return min(geo_inside(p), p.y - GROUND_Y);
}

// C8a: the nearest-triangle albedo at a GDF reflection hit (trilinear from the 3 R32F
// color volumes), or the constant fallback when no albedo volume is bound. Shares the
// distance field's voxel grid; gives colored reflections instead of grey.
float3 albedo_at(float3 p)
{
    return cm_albedo(reflect_clip(), p, pc.params.y);
}

// Cone-blurred per-voxel albedo for the UNcovered (analytic) reflection path. The surface cache is
// sampled with a ray-cone MIP so cached far reflections are smooth; the analytic fallback used the
// raw per-voxel `albedo_at` at a single (imprecise) point, so uncarded far hits read blocky, colour-
// bleeding voxels — the residual speckle in the demagnified centre. Average the albedo over the cone
// footprint (a few taps in the hit's tangent plane, radius = the same cone `sample_radius`) so an
// uncovered far surface reads a smooth averaged colour matching its cached neighbours. `radius == 0`
// (near hits / cone off) collapses to the single tap = the legacy value.
float3 albedo_cone(float3 p, float3 n, float radius)
{
    if (radius <= 1e-4)
        return albedo_at(p);
    float3 t, bt;
    onb(n, t, bt);
    float3 s = albedo_at(p);
    s += albedo_at(p + t * radius) + albedo_at(p - t * radius);
    s += albedo_at(p + bt * radius) + albedo_at(p - bt * radius);
    return s * (1.0 / 5.0);
}

float3 scene_normal(float3 p)
{
    const float e = 0.012;
    float2 k = float2(1.0, -1.0);
    float3 grad = k.xyy * scene_occ(p + k.xyy * e) + k.yyx * scene_occ(p + k.yyx * e)
                  + k.yxy * scene_occ(p + k.yxy * e) + k.xxx * scene_occ(p + k.xxx * e);
    return safe_normalize(grad, float3(0.0, 1.0, 0.0));
}

// Gradient sky (+ optional HDR sun disk). `with_sun` adds the bright sun disk — enabled only
// for the background viz. The REFLECTION escape passes `with_sun = false`: like a path tracer's
// indirect rays, the reflected sky carries no sun (which would double the directional light's
// analytic specular highlight — two sun glints on a metal — and the sharp reflected disk also
// blows out when the roughness prefilter spreads it). The sun appears once, via direct lighting.
float3 sky(float3 dir, float3 sun_dir, float sun_i, bool with_sun)
{
    float up = dir.y;
    float3 zenith = float3(0.04, 0.16, 0.55);
    float3 horizon = float3(0.50, 0.66, 0.92);
    float3 ground = float3(0.16, 0.15, 0.14);
    float3 col = up >= 0.0 ? lerp(horizon, zenith, pow(saturate(up), 0.45))
                           : lerp(horizon, ground, saturate(-up * 6.0));
    if (with_sun)
    {
        float c = saturate(dot(dir, sun_dir));
        col += sun_i * (pow(c, 4000.0) * 6.0 + pow(c, 32.0) * 0.4);
    }
    return col;
}

// Short penumbra shadow toward the sun for the reflected hit (cheap; reflections are
// secondary so a coarse shadow is fine).
float refl_shadow(float3 origin, float3 dir, float maxt)
{
    float res = 1.0;
    float t = 0.02;
    for (int i = 0; i < 48; i++)
    {
        float h = scene_occ(origin + dir * t);
        if (h < 0.002)
            return 0.0;
        res = min(res, 16.0 * h / t);
        // P3 cone-trace LOD: widen the max step with distance (cone_k=0 => clamp(.,0.01,0.2) legacy).
        t += clamp(h, 0.01, max(0.2, pc.ground_albedo.w * t));
        if (t > maxt)
            break;
    }
    return saturate(res);
}

// Real SH band-0/1 basis normalisation (matches gi_volume.slang / gdf_gi.slang — keep in lockstep).
static const float SH_C0 = 0.2820947918; // sqrt(1/(4*pi))
static const float SH_C1 = 0.4886025119; // sqrt(3/(4*pi))

// Reference-engine reflection technique: sample the world-space directional-irradiance volume
// (radiance cache, the 12 SH coeff volumes contiguous from `base`, updated in gi_volume.slang with
// multibounce) at a reflection HIT and reconstruct the cosine-weighted hemispherical irradiance
// E(n). This is the indirect term that keeps a reflected shadowed surface at its GI-lit colour
// instead of the analytic sun-only re-light going near-black. Returns 0 outside the volume bounds.
// Mirrors the reconstruction in gdf_gi.slang's volume path (A0/PI = 1, A1/PI = 2/3).
float3 sample_gi_irradiance(uint base, float3 p, float3 n)
{
    float3 uvw = (p - GDF_MIN) / (GDF_MAX - GDF_MIN);
    if (!(all(uvw >= 0.0) && all(uvw <= 1.0)))
        return float3(0.0, 0.0, 0.0);
    uvw = saturate(uvw);
    float3 c[4];
    [unroll]
    for (int k = 0; k < 4; k++)
        c[k] = float3(0.0, 0.0, 0.0);
    [unroll]
    for (int ch = 0; ch < 3; ch++)
    {
        [unroll]
        for (int k = 0; k < 4; k++)
            c[k][ch] = g.volumes[base + uint(ch * 4 + k)].SampleLevel(g.samp, uvw, 0.0).x;
    }
    float3 e = c[0] * SH_C0
               + (2.0 / 3.0) * (SH_C1 * n.y * c[1] + SH_C1 * n.z * c[2] + SH_C1 * n.x * c[3]);
    return max(e, float3(0.0, 0.0, 0.0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;
    float3 sun_dir = normalize(pc.sun.xyz);
    float sun_i = pc.sun.w;
    uint lin = tid.y * pc.width + tid.x;
    SkipCfg sk = unpack_skip();
    bool skip_write = (sk.write != 0xFFu); // A3: persist this pixel's (pos, radiance)

    // Background: reflect nothing — show the sky along the view ray (so the viz reads as
    // the scene; C7 only consumes this for shaded pixels anyway).
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    // GI irradiance volume base index packed into flip_y's spare bits (bit0 = Y-flip; the 240-byte
    // push is full, D3D12 root budget forbids growing it). Encoded as (base+1)<<1 so 0 = off.
    uint vol_pack = pc.flip_y >> 1;
    bool vol_on = vol_pack != 0u;
    uint vol_base = vol_on ? (vol_pack - 1u) : 0xFFFFFFFFu;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    if (depth >= 1.0)
    {
        float4 far = mul(pc.inv_view_proj, float4(ndc, 1.0, 1.0));
        float3 vdir = normalize(far.xyz / far.w - pc.cam_pos.xyz);
        // Background: no surface — invalidate this pixel's skip slot so it isn't reused next frame.
        if (skip_write)
            g.storage_buffers[sk.write].Store4(lin * 32u, asuint(float4(0.0, 0.0, 0.0, 0.0)));
        g.storage_images[pc.out_index][tid.xy] = float4(sky(vdir, sun_dir, sun_i, true), 1.0);
        return;
    }

    float4 hpos = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    float3 p = hpos.xyz / hpos.w;

    // A3 adaptive temporal skip: if this pixel's surface point is unchanged from last frame (the
    // world-pos gate) and we aren't on the staggered re-trace floor, reuse last frame's traced
    // reflection radiance instead of re-marching. The host only enables the READ (skip.x != sentinel)
    // once the lighting + surface cache are stable, so a reused value equals the value a fresh trace
    // would produce (image-identical); camera motion moves the surface point > reject → re-trace;
    // the 1/K staggered floor re-traces a rotating subset so nothing goes stale under un-modeled change.
    if (sk.read != 0xFFu)
    {
        // Staggered re-trace floor — WAVE-COHERENT: force whole 8x8 dispatch tiles (not scattered
        // pixels) so a re-trace frame doesn't put one marching thread in every wave (which would make
        // the reused threads pay the full march via SIMD divergence). 1/K tiles re-march per frame.
        uint tiles_x = (pc.width + 7u) / 8u;
        uint tile_id = (tid.y / 8u) * tiles_x + (tid.x / 8u);
        bool force = (sk.k != 0u) && (((tile_id + sk.frame) % sk.k) == 0u);
        if (!force)
        {
            float reject = 0.02 * length(GDF_MAX - GDF_MIN);
            float4 ppos = asfloat(g.storage_buffers[sk.read].Load4(lin * 32u));
            if (ppos.w > 0.5 && distance(p, ppos.xyz) < reject)
            {
                // The persisted radiance slot's .w carries the reflection HIT DISTANCE (world) so the
                // temporal resolve can reproject the virtual image (reference-engine mirror denoise).
                float4 pradf = asfloat(g.storage_buffers[sk.read].Load4(lin * 32u + 16u));
                float3 prad = pradf.rgb;
                if (skip_write)
                {
                    g.storage_buffers[sk.write].Store4(lin * 32u, asuint(float4(p, 1.0)));
                    g.storage_buffers[sk.write].Store4(lin * 32u + 16u, asuint(float4(prad, pradf.w)));
                }
                g.storage_images[pc.out_index][tid.xy] = float4(prad, pradf.w);
                return;
            }
        }
    }
    float3 n = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);
    float3 v = normalize(pc.cam_pos.xyz - p);
    // C8j: GGX-importance-sample the reflection direction by roughness (one jittered ray/frame;
    // the temporal resolve accumulates them into the glossy lobe). Near-mirror stays sharp.
    float roughness = g.textures[pc.material_index].SampleLevel(g.samp, uv, 0.0).g;
    float3 dir;
    // Mirror threshold: a polished metal (roughness up to ~0.125) traces the DETERMINISTIC reflection
    // ray (no GGX jitter), so it settles to a stable sharp reflection every frame instead of a
    // stochastic 1-ray/frame sample that sparkles (the reflect_temporal near-mirror boundary is the
    // same 0.125 = sr==0, so these carry no neighbourhood clamp and rely on being deterministic). Only
    // genuinely GLOSSY surfaces (> threshold) need the importance-sampled lobe + temporal reconstruction.
    // Content raises the threshold to 0.125; the gallery keeps 0.04 (its chrome sphere is 0.08 = GGX,
    // byte-identical anchor). The content flag rides `max_steps` bit 31 (the step cap is < 2^31), since
    // `vol_on` is true for the gallery too and can't distinguish it.
    bool content_refl = (pc.max_steps & 0x80000000u) != 0u;
    float mirror_thresh = content_refl ? 0.125 : 0.04;
    if (roughness > mirror_thresh)
    {
        if (content_refl)
        {
            // A5: content routes through the shared sampler so the A1 resolve reconstructs the SAME
            // ray. `stochastic` (max_steps bit30) selects blue-noise (frame-varying, low-discrepancy)
            // vs the legacy white jitter. Blue-noise + the A1/A4 denoiser lets the temporal pass
            // converge the lobe instead of the fixed-jitter dead end.
            uint stochastic = (pc.max_steps & 0x40000000u) != 0u ? 1u : 0u;
            dir = refl_ggx_dir(n, v, roughness, tid.x, tid.y, pc.frame, stochastic);
        }
        else
        {
            // Gallery: inline white jitter (byte-identical anchor). `pc.frame` decorrelates per frame.
            uint rng = (tid.x * 1973u + tid.y * 9277u + pc.frame * 26699u) | 1u;
            float3 h = sample_ggx(n, roughness, rng);
            dir = reflect(-v, h);
            if (dot(dir, n) <= 0.0)
                dir = reflect(-v, n); // keep the ray in the upper hemisphere
        }
    }
    else
    {
        dir = reflect(-v, n);
    }

    float max_dist = pc.params.x;
    float sky_fill = pc.params.z;
    float bias = pc.params.w;
    // Content (vol_on) overloads the sky_fill slot with the IBL diffuse-irradiance cube index — the
    // physical-units skylight the deferred pass samples. Used to fill an uncovered/shadowed reflected
    // surface (analytic path) and an escaped (sky) ray with real skylight instead of near-black.
    uint irr_idx = vol_on ? asuint(pc.params.z) : 0xFFFFFFFFu;
    bool has_irr = irr_idx != 0xFFFFFFFFu;

    float t = bias;
    bool hit = false;
    float3 hp = p;
#ifdef HWRT_REFLECT
    // Phase 16 hybrid: trace the reflection ray against the REAL triangle BVH (the scene TLAS) with
    // an inline RayQuery instead of the GDF sphere-march — exact hit points, no distance-field
    // leaking or low-res card blockiness (the floor sparkle / blurry chrome). The hit is shaded from
    // the surface cache below EXACTLY as the SW path (SurfaceCache mode), so only the trace changes.
    // Closest-hit pattern mirrors rt_trace.slang (RAY_FLAG_NONE + single Proceed + CommittedRayT).
    // E: the hit's instance/primitive/barycentrics/object-to-world are captured for Hit Lighting.
    uint hl_inst = 0u;
    uint hl_prim = 0u;
    float2 hl_bary = float2(0.0, 0.0);
    float3x4 hl_o2w = float3x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    {
        RayDesc ray;
        ray.Origin = p;
        ray.Direction = dir;
        ray.TMin = bias; // avoid self-intersecting the origin surface
        ray.TMax = max_dist;
        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(g.tlas, RAY_FLAG_NONE, 0xFF, ray);
        q.Proceed();
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            t = q.CommittedRayT();
            hp = p + dir * t;
            hit = true;
            hl_inst = q.CommittedInstanceID();
            hl_prim = q.CommittedPrimitiveIndex();
            hl_bary = q.CommittedTriangleBarycentrics();
            hl_o2w = q.CommittedObjectToWorld3x4();
        }
    }
#else
    // Sphere-trace the reflection ray. Stage D3: content lowers the step cap; gallery = 96.
    // Mask off the content flag (bit31) AND the A5 stochastic flag (bit30) to get the real step cap
    // (content caps are < 2^30). Gallery cap (96) is unaffected → byte-identical.
    int max_steps = int(max(pc.max_steps & 0x3FFFFFFFu, 1u));
    const float HIT_EPS = 0.001;
    for (int i = 0; i < max_steps; i++)
    {
        float3 rp = p + dir * t;
        float d = scene_march(rp);
        if (d < HIT_EPS)
        {
            hp = rp;
            hit = true;
            break;
        }
        // P3 cone-trace LOD: raise the step floor with distance (cone_k=0 => max(d,0.01) legacy).
        t += max(d, max(0.01, pc.ground_albedo.w * t));
        if (t > max_dist)
            break;
    }
#endif

    float3 color;
    if (hit)
    {
        float3 hn = scene_normal(hp);
        bool cached = false;
#ifdef HWRT_REFLECT
        // Phase 16 B.2 — screen-color-at-hit (reference-engine `SampleSceneColorAtHit`): the hardware
        // ray hit REAL geometry, so if that surface is visible on screen its FULL-RES lit colour is
        // already in the previous frame's lit-radiance history (`globals.prev_view_proj` + the
        // `lit_hist` buffer). Sample it directly for a SHARP reflection — the low-res surface cache
        // is only the off-screen / occluded fallback. This is the sharp-reflection win the SW trace
        // couldn't give, and the validated HW hit is exactly what our screen-space SSR lacked.
        // The lit-history index rides the (march-cap) field — unused in the HW variant (no march).
        {
            uint lit_hist = pc.max_steps & 0x7FFFFFFFu;
            float4 pclip = mul(globals.prev_view_proj, float4(hp, 1.0));
            if (lit_hist != 0x7FFFFFFFu && pclip.w > 0.0)
            {
                float3 pndc = pclip.xyz / pclip.w;
                float2 puv = float2(pndc.x * 0.5 + 0.5, pndc.y * sy * 0.5 + 0.5);
                if (all(puv >= 0.0) && all(puv <= 1.0))
                {
                    float sd = g.textures[pc.depth_index].SampleLevel(g.samp, puv, 0.0).r;
                    if (sd < 1.0)
                    {
                        // Visibility validation: the depth-buffer surface at `puv` must BE the hit,
                        // not an occluder in front of it — reconstruct its world point and compare.
                        float2 vndc = float2(puv.x * 2.0 - 1.0, (puv.y * 2.0 - 1.0) * sy);
                        float4 vw = mul(pc.inv_view_proj, float4(vndc, sd, 1.0));
                        float3 sworld = vw.xyz / vw.w;
                        float vtol = 0.01 * length(GDF_MAX - GDF_MIN) + t * 0.02;
                        if (distance(sworld, hp) < vtol)
                        {
                            // lit_hist is FULL-res; the trace may be half-res, so index it with the
                            // depth texture's own (full-res) dimensions — no extra push field needed.
                            uint fw, fh;
                            g.textures[pc.depth_index].GetDimensions(fw, fh);
                            int2 pp = clamp(int2(puv * float2(fw, fh)), int2(0, 0),
                                            int2(int(fw) - 1, int(fh) - 1));
                            uint pidx = uint(pp.y) * fw + uint(pp.x);
                            color = asfloat(g.storage_buffers[lit_hist].Load4(pidx * 16u)).rgb;
                            cached = true; // sharp on-screen hit — skip the low-res cache + analytic
                        }
                    }
                }
            }
        }
        // Phase 16 E — Hit Lighting (OFF-SCREEN): the on-screen path failed (behind camera / occluded),
        // so shade this HW hit with the REAL material instead of the low-res surface cache. `frame`
        // bit31 = enabled; the consolidated (vertex, index, drawable-table) bindless indices ride the
        // coarse-albedo push slots (unused here — Hit Lighting replaces the analytic fallback). Fetch
        // the hit triangle's interpolated normal + UV, sample the albedo texture, and re-light with the
        // sun (HW shadow ray) + GI radiance cache + IBL sky — the same energy convention as the analytic
        // path below, but with the accurate material/normal/shadow. Reference: Lumen HitLighting mode.
        if (!cached && (pc.frame & 0x80000000u) != 0u)
        {
            uint tbl = pc.albedo_b;
            uint rbase = hl_inst * 48u;
            uint idx_base = g.storage_buffers[tbl].Load(rbase + 0u);
            uint prim_count = g.storage_buffers[tbl].Load(rbase + 4u);
            if (hl_prim < prim_count)
            {
                uint tex_albedo = g.storage_buffers[tbl].Load(rbase + 8u);
                float4 base_color = asfloat(g.storage_buffers[tbl].Load4(rbase + 16u));
                uint tri = (idx_base + hl_prim * 3u) * 4u;
                uint i0 = g.storage_buffers[pc.albedo_g].Load(tri);
                uint i1 = g.storage_buffers[pc.albedo_g].Load(tri + 4u);
                uint i2 = g.storage_buffers[pc.albedo_g].Load(tri + 8u);
                float3 n0 = asfloat(g.storage_buffers[pc.albedo_r].Load3(i0 * 32u + 12u));
                float3 n1 = asfloat(g.storage_buffers[pc.albedo_r].Load3(i1 * 32u + 12u));
                float3 n2 = asfloat(g.storage_buffers[pc.albedo_r].Load3(i2 * 32u + 12u));
                float2 uv0 = asfloat(g.storage_buffers[pc.albedo_r].Load2(i0 * 32u + 24u));
                float2 uv1 = asfloat(g.storage_buffers[pc.albedo_r].Load2(i1 * 32u + 24u));
                float2 uv2 = asfloat(g.storage_buffers[pc.albedo_r].Load2(i2 * 32u + 24u));
                float b0 = 1.0 - hl_bary.x - hl_bary.y;
                float3 n_obj = n0 * b0 + n1 * hl_bary.x + n2 * hl_bary.y;
                float2 uv = uv0 * b0 + uv1 * hl_bary.x + uv2 * hl_bary.y;
                float3 nw = normalize(mul((float3x3)hl_o2w, n_obj));
                if (dot(nw, dir) > 0.0)
                    nw = -nw; // face the incoming ray (two-sided)
                float3 alb = base_color.rgb;
                if (tex_albedo != 0xFFFFFFFFu)
                    alb *= g.textures[tex_albedo].SampleLevel(g.samp_wrap, uv, 0.0).rgb;
                // Sun with a HARDWARE shadow ray (accurate vs the GDF shadow march).
                float ndl = max(dot(nw, sun_dir), 0.0);
                float sh = 1.0;
                if (ndl > 0.0)
                {
                    RayDesc sray;
                    sray.Origin = hp + nw * 0.01;
                    sray.Direction = sun_dir;
                    sray.TMin = 0.001;
                    sray.TMax = max_dist;
                    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
                    sq.TraceRayInline(g.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, sray);
                    sq.Proceed();
                    if (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
                        sh = 0.0;
                }
                // Same energy convention as the analytic content path: direct/PI + GI cache + IBL sky.
                float3 e = vol_on ? sample_gi_irradiance(vol_base, hp, nw) : float3(0.0, 0.0, 0.0);
                color = alb * ((sun_i * ndl * sh) * (1.0 / GGX_PI) + e);
                if (has_irr)
                    color += g.cubes[irr_idx].SampleLevel(g.samp, nw, 0.0).rgb * alb;
                cached = true; // shaded with the real material — skip the surface cache + analytic
            }
        }
#endif
        // C8b3/C8g: the reflected surface's cached (multibounce, accurately-lit) radiance — a
        // direct lookup when the cache is bound. The cache only covers the meshed objects (no
        // cards for the analytic ground), so on a cache MISS fall back to the C8a per-ray
        // re-light instead of black (which previously left the ground reflection dark).
        if (!cached && pc.cache.x != 0xFFFFFFFFu)
        {
            // Reference-engine (Lumen) cone-based card acceptance: the coarse-field reflection HIT is
            // imprecise, and the imprecision GROWS with the ray distance `t` (a far hit lands farther
            // from the true surface). A FIXED tolerance (the old 0.006·diag) matched near cards but
            // FAILED far ones → those far hits fell to the coarse per-voxel-albedo analytic path = the
            // blocky, colour-bleeding chrome centre. Grow the acceptance with distance (like Lumen's
            // ray-cone SampleRadius) so far reflected surfaces still read their smooth cached lit
            // radiance instead of albedo. 0 for the gallery (vol_on false) -> byte-identical.
            float cache_bias = vol_on ? (0.006 * length(GDF_MAX - GDF_MIN) + t * 0.03) : 0.0;
            // The reflection packs the MIP pyramid into cache_tile's spare bits: tile | max_mip<<8 |
            // mip_index<<16 (mip_index 0xFFFF = no pyramid). The same distance-growing cone_bias is
            // the ray-cone footprint (world units) that selects the MIP — near hits stay at mip0,
            // far demagnified hits read a coarse averaged level → smooth chrome instead of blocks.
            uint c_tile = pc.cache_tile & 0xFFu;
            uint c_maxmip = (pc.cache_tile >> 8) & 0xFu;
            uint c_mip16 = (pc.cache_tile >> 16) & 0xFFFFu;
            uint mip_index = (c_mip16 == 0xFFFFu) ? 0xFFFFFFFFu : c_mip16;
            // Ray-cone footprint that selects the MIP (world units) — DECOUPLED from `cache_bias`
            // (which is the card-acceptance tolerance; widening THAT scatters cross-card). This grows
            // with hit distance `t` so a far demagnified reflection reads a coarse averaged level
            // while a near/face-on hit stays near mip0 (sharp). Slope ≈ a per-pixel cone half-angle.
            float mip_radius = 0.05 + t * 0.08;
            color = sample_surface_cache_cone(hp, hn, pc.cache.x, pc.cache.y, pc.cache.z, pc.cache.w,
                                              c_tile, cache_bias, mip_radius, mip_index, c_maxmip,
                                              cached);
        }
        if (!cached)
        {
            float ndl = max(dot(hn, sun_dir), 0.0);
            float sh = ndl > 0.0 ? refl_shadow(hp + hn * 0.004, sun_dir, max_dist) : 1.0;
            // The ground plane is analytic — it's NOT in the per-voxel albedo volume, so
            // albedo_at() would return the nearest *object's* colour for a ground hit (the
            // red cube / green avocado bleeding onto the reflected floor). Use the ground's
            // own albedo (matches deferred.rs's [0.8] ground material) for ground hits.
            bool is_ground = (hp.y - GROUND_Y) < geo_inside(hp);
            // Cone-blurred albedo (content, vol_on) so an uncovered far reflection is as smooth as
            // its cached neighbours instead of blocky per-voxel colour. Near hits (radius→0) and the
            // gallery (vol_on false) keep the single-tap legacy albedo.
            float ablur = vol_on ? (0.05 + t * 0.08) : 0.0;
            float3 hit_albedo = is_ground ? pc.ground_albedo.xyz : albedo_cone(hp, hn, ablur);
            // Lambertian reflected surface: albedo/PI * irradiance, so the reflection of a
            // diffuse surface matches its directly-shaded appearance. The cache path (meshed
            // hits) gets the same 1/PI in sdf_cache_light.slang.
            if (vol_on)
            {
                // Reference-engine approach: add the GI-lit INDIRECT irradiance (radiance-cache
                // volume) so a reflected SHADOWED surface reads its real GI colour, not black. The
                // indirect E matches gdf_gi.slang's convention (outgoing = albedo * E, no extra
                // /PI), so it is added OUTSIDE the direct term's 1/PI to be energy-consistent with
                // how the deferred lighting consumes the same volume.
                float3 e = sample_gi_irradiance(vol_base, hp, hn);
                color = hit_albedo * ((sun_i * ndl * sh) * (1.0 / GGX_PI) + e);
                // Physical IBL-diffuse skylight fill: the radiance-cache volume `e` is a LOCAL bounce
                // term and is ~0 where the cache is cold, so an uncovered reflected surface would go
                // near-black. Add the same sky irradiance the deferred pass gives every surface
                // (`irradiance_cube(hn) * albedo`) so a reflected shadowed/uncovered face reads its
                // true skylit grey — the reflection then matches the directly-shaded scene.
                if (has_irr)
                    color += g.cubes[irr_idx].SampleLevel(g.samp, hn, 0.0).rgb * hit_albedo;
            }
            else
            {
                // Volume absent (e.g. the gallery) -> the legacy crude sky fill. Byte-identical.
                float fill = sky_fill * (0.5 + 0.5 * hn.y);
                color = hit_albedo * (sun_i * ndl * sh + fill) * (1.0 / GGX_PI);
            }
        }
    }
    else
    {
        // Specular miss -> the sky the mirror ray sees. Content samples the physical IBL cube (the
        // deferred pass's captured sky, in scene lux) so an escaped ray reads a bright sky instead of
        // the tiny unscaled procedural gradient (which metered to ~black in a lux-lit scene, leaving
        // black speckle holes). The irradiance cube is cosine-convolved (≈π·radiance), so /π brings it
        // back to a directional sky radiance. Gallery keeps the legacy gradient (byte-identical).
        if (has_irr)
            color = g.cubes[irr_idx].SampleLevel(g.samp, dir, 0.0).rgb * (1.0 / GGX_PI);
        else
            color = sky(dir, sun_dir, sun_i, false);
    }


    // Reflection HIT DISTANCE (world units) in the output/persist `.w` — the temporal resolve uses it
    // to reproject the VIRTUAL IMAGE (the reflected point at `surface_dist + t` along the view ray) so a
    // mirror reflection stays stable under camera motion instead of swimming (reference-engine
    // LumenReflectionDenoiserTemporal virtual-image reproject). `t` is the ray parameter at the hit
    // (≈ ray max on a miss = the sky, effectively far). Alpha is unused by the composite/lighting, so
    // this is inert for every non-temporal-resolve consumer (gallery byte-identical).
    float hit_dist = t;
    // A3: persist this freshly-traced (surface point, radiance) for next frame's reuse gate.
    if (skip_write)
    {
        g.storage_buffers[sk.write].Store4(lin * 32u, asuint(float4(p, 1.0)));
        g.storage_buffers[sk.write].Store4(lin * 32u + 16u, asuint(float4(color, hit_dist)));
    }
    g.storage_images[pc.out_index][tid.xy] = float4(color, hit_dist);
}
