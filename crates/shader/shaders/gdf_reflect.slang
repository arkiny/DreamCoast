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

#if defined(HWRT_REFLECT) || defined(SCREEN_HIT)
// Per-frame globals (set 1 / b1) — the same UBO the deferred PBR / SSR passes bind (opted into via
// the compute pipeline's `uniform_buffer`). Only `prev_view_proj` is read, for the B.2 screen-color-
// at-hit reprojection (HWRT) / the B2' screen-hit early-out march (SW SCREEN_HIT permutation); the
// layout must match the host `Globals` (mirrors ssr.slang's subset).
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
#ifdef SCREEN_HIT
    // In this permutation `params.y` carries the lit-history buffer index (content always binds
    // the albedo volumes, so the constant-albedo fallback slot is dead); keep a sane constant.
    return cm_albedo(reflect_clip(), p, 0.7);
#else
    return cm_albedo(reflect_clip(), p, pc.params.y);
#endif
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

// Validity-weighted probe interpolation (reference radiance-cache style; cache_tile bit13,
// opt-in): the hardware trilinear above mixes in probes whose voxel centre sits INSIDE
// geometry — their SH is ~0 and not a witness of the query's radiance, so a contact gap
// under a mirror interpolates toward black (measured as a sharp black disc, perceptually
// worse than the bright bowl it replaced). Re-do the trilinear by hand over the 2x2x2 probe
// corners, weighting each by its OCCUPANCY (probe centre in open space, `scene_occ > 0`) and
// renormalising: the nearest unoccluded probes answer, and the result ramps to 0 only where
// EVERY neighbouring probe is enclosed — the deepest sliver of the gap, which the path
// tracer also shows near-black, so the falloff is smooth instead of a disc edge. Fallback
// path only (a cache miss), so the extra taps stay off the hot lanes.
float3 sample_gi_irradiance_valid(uint base, float3 p, float3 n)
{
    float3 ext = GDF_MAX - GDF_MIN;
    float3 uvw = (p - GDF_MIN) / ext;
    if (!(all(uvw >= 0.0) && all(uvw <= 1.0)))
        return float3(0.0, 0.0, 0.0);
    uint dw, dh, dd;
    g.volumes[base].GetDimensions(dw, dh, dd);
    float3 dims = float3(float(dw), float(dh), float(dd));
    float3 f = saturate(uvw) * dims - 0.5;
    float3 f0 = floor(f);
    float3 fr = f - f0;
    int3 i0 = int3(f0);
    int3 imax = int3(int(dw) - 1, int(dh) - 1, int(dd) - 1);
    float3 c[4];
    [unroll]
    for (int k = 0; k < 4; k++)
        c[k] = float3(0.0, 0.0, 0.0);
    float wsum = 0.0;
    [loop]
    for (int ci = 0; ci < 8; ci++)
    {
        int3 off = int3(ci & 1, (ci >> 1) & 1, (ci >> 2) & 1);
        float3 fw = lerp(float3(1.0, 1.0, 1.0) - fr, fr, float3(off));
        float w = fw.x * fw.y * fw.z;
        if (w <= 1e-4)
            continue;
        int3 ip = clamp(i0 + off, int3(0, 0, 0), imax);
        float3 probe = GDF_MIN + ((float3(ip) + 0.5) / dims) * ext;
        if (scene_occ(probe) <= 0.0)
            continue; // occupied probe — exclude it from the interpolation
        [unroll]
        for (int ch = 0; ch < 3; ch++)
        {
            [unroll]
            for (int k = 0; k < 4; k++)
                c[k][ch] += w * g.volumes[base + uint(ch * 4 + k)].Load(int4(ip, 0)).x;
        }
        wsum += w;
    }
    if (wsum <= 1e-4)
        return float3(0.0, 0.0, 0.0); // fully enclosed — genuinely dark
    float rw = 1.0 / wsum;
    float3 e = (c[0] * SH_C0
                + (2.0 / 3.0) * (SH_C1 * n.y * c[1] + SH_C1 * n.z * c[2] + SH_C1 * n.x * c[3]))
               * rw;
    return max(e, float3(0.0, 0.0, 0.0));
}

#ifdef SCREEN_HIT
// B2' screen-hit early-out — the SW screen-color-at-hit. March the reflection ray a short distance
// and test each sample against the depth buffer; where the ray provably hits an on-screen surface,
// take that surface's FULL-RES lit radiance from the previous frame's history instead of paying the
// GDF sphere-march + surface-cache shade. That is both the trace budget that pays for the B2' K
// rays (most indoor reflected surfaces ARE on screen) and a strictly better colour (exact full-res
// lighting vs 32² card texels — the same reason the HWRT screen-color-at-hit is the sharpness win).
// Projection uses `globals.prev_view_proj` and validates against the CURRENT depth buffer with the
// same tolerance law as the Phase 16 B.2 HWRT screen path, so camera motion fails the validation
// (falls back to the GDF march) instead of smearing. Returns true + colour + hit ray-parameter.
// `spread` (radians of reflected-ray divergence per output texel) + `pa_full` (full-res pixel
// angle) drive the B2-compaction anti-aliasing: at the validated hit, the ray's footprint in
// lit-history pixels is `spread·t / (hit_dist·pa_full)`. A point fetch of the history is only
// correct while that footprint is ~a pixel; a minified mirror (a distant curved reflector)
// skips tens of pixels of reflected content per ray and point fetches turn into colour speckle
// no later pass can remove. Moderate footprints take a footprint-spaced 3x3 box of the history;
// past ~24 px the fetch REJECTS so the ray falls through to the cache cone, which is
// prefiltered (the cone-traced mip chain is the correct minified answer). `spread = 0`
// (every non-compact caller) keeps the legacy point fetch bit-exactly.
bool screen_trace(float3 p, float3 dir, float sy, float bias, float max_dist, uint lit_hist,
                  out float3 col, out float thit, float spread, float pa_full)
{
    col = float3(0.0, 0.0, 0.0);
    thit = 0.0;
    if (lit_hist >= 0x7FFFFFFFu)
        return false; // 0x7FFFFFFF = no history bound (same sentinel as the HWRT path)
    float diag = length(GDF_MAX - GDF_MIN);
    float range = min(max_dist, 0.5 * diag);
    // Exponential march: fine near the surface (contact reflections), coarse far.
    const int STEPS = 24;
    float t = max(bias * 2.0, range * 0.004);
    float grow = pow(range / t, 1.0 / float(STEPS)); // t · grow^STEPS == range
    [loop]
    for (int i = 0; i < STEPS; i++)
    {
        t *= grow;
        float3 rp = p + dir * t;
        float4 pclip = mul(globals.prev_view_proj, float4(rp, 1.0));
        if (pclip.w <= 0.0)
            return false; // marched behind the camera — not an on-screen ray
        float3 pndc = pclip.xyz / pclip.w;
        float2 puv = float2(pndc.x * 0.5 + 0.5, pndc.y * sy * 0.5 + 0.5);
        if (!(all(puv >= 0.0) && all(puv <= 1.0)))
            return false; // left the screen — the GDF owns the rest of the ray
        float sd = g.textures[pc.depth_index].SampleLevel(g.samp, puv, 0.0).r;
        if (sd >= 1.0)
            continue; // background pixel — nothing to hit here
        float2 vndc = float2(puv.x * 2.0 - 1.0, (puv.y * 2.0 - 1.0) * sy);
        float4 vw = mul(pc.inv_view_proj, float4(vndc, sd, 1.0));
        float3 sworld = vw.xyz / vw.w;
        // Acceptance: the depth-buffer surface at `puv` must BE the ray sample (tolerance grows
        // with t — a far sample tolerates more parallax). Mirrors the HWRT B.2 validation.
        float vtol = 0.01 * diag + t * 0.02;
        if (distance(sworld, rp) < vtol)
        {
            // Self-acceptance guard: at small t the sample still hugs the REFLECTOR, projects
            // onto its own pixels, and the parallax tolerance (sized for far hits) swallows the
            // gap — the mirror then feeds back its own lit colour (a bright featureless band
            // along a convex mirror's rim). A depth surface this close to the ray ORIGIN cannot
            // be reflected content; skip it and keep marching.
            if (distance(sworld, p) < vtol)
                continue;
            // lit history is FULL-res; index it with the depth texture's own dimensions.
            uint fw, fh;
            g.textures[pc.depth_index].GetDimensions(fw, fh);
            int2 pp = clamp(int2(puv * float2(fw, fh)), int2(0, 0),
                            int2(int(fw) - 1, int(fh) - 1));
            // B2 compaction footprint (see the function docs). 0 = legacy point fetch.
            float fp_px = 0.0;
            if (spread > 0.0)
            {
                float hit_dist = distance(pc.cam_pos.xyz, sworld);
                fp_px = spread * t / max(hit_dist * pa_full, 1e-6);
                if (fp_px > 24.0)
                    return false; // too minified — the prefiltered cache cone owns this ray
            }
            if (fp_px > 3.0)
            {
                // Footprint-spaced 3x3 box of the history: a cheap low-pass matched to the
                // minification, so adjacent rays integrate overlapping content instead of
                // skipping between speckles.
                int r = int(clamp(fp_px * (1.0 / 3.0), 1.0, 8.0));
                float3 sum = float3(0.0, 0.0, 0.0);
                [unroll]
                for (int dy = -1; dy <= 1; dy++)
                {
                    [unroll]
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int2 sp2 = clamp(pp + int2(dx, dy) * r, int2(0, 0),
                                         int2(int(fw) - 1, int(fh) - 1));
                        uint si = uint(sp2.y) * fw + uint(sp2.x);
                        sum += asfloat(g.storage_buffers[lit_hist].Load4(si * 16u)).rgb;
                    }
                }
                col = sum * (1.0 / 9.0);
            }
            else
            {
                uint pidx = uint(pp.y) * fw + uint(pp.x);
                col = asfloat(g.storage_buffers[lit_hist].Load4(pidx * 16u)).rgb;
            }
            thit = t;
            return true;
        }
    }
    return false;
}
#endif

[shader("compute")]
#ifdef REFLECT_COMPACT
// B2 compaction: one thread per LISTED near-mirror pixel (see reflect_compact.slang) instead of
// one per grid texel. The classify pass appended the pixel coords; this permutation re-traces
// exactly those pixels at the (denser) refine resolution and writes them into the refine target.
// The A3 skip slot is overloaded to carry the list/args storage-buffer indices (16 bits each —
// bindless buffer indices are < 2048), so the 240 B push does not grow.
[numthreads(64, 1, 1)]
void csMain(uint3 tid_in: SV_DispatchThreadID)
{
    uint list_index = pc.skip_packed & 0xFFFFu;
    // Count lives in the LIST header (mirrored by csArgs) — the args buffer is in INDIRECT
    // state at dispatch time and must not be shader-read here.
    uint count = g.storage_buffers[list_index].Load(0u);
    if (tid_in.x >= count)
        return;
    uint packed = g.storage_buffers[list_index].Load(4u + tid_in.x * 4u);
    uint3 tid = uint3(packed & 0xFFFFu, packed >> 16, 0u);
#else
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
#endif
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;
    float3 sun_dir = normalize(pc.sun.xyz);
    float sun_i = pc.sun.w;
    uint lin = tid.y * pc.width + tid.x;
#ifdef REFLECT_COMPACT
    // skip_packed carries the list/args indices here — the A3 reuse/persist is off by
    // construction (the refine texels have no 32 B skip slots at this resolution).
    SkipCfg sk;
    sk.read = 0xFFu;
    sk.write = 0xFFu;
    sk.k = 0u;
    sk.frame = 0u;
#else
    SkipCfg sk = unpack_skip();
#endif
    bool skip_write = (sk.write != 0xFFu); // A3: persist this pixel's (pos, radiance)

    // Background: reflect nothing — show the sky along the view ray (so the viz reads as
    // the scene; C7 only consumes this for shaded pixels anyway).
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    // GI irradiance volume base index packed into flip_y's spare bits (bit0 = Y-flip; the 240-byte
    // push is full, D3D12 root budget forbids growing it). Encoded as (base+1)<<1 so 0 = off.
    uint vol_pack = pc.flip_y >> 1;
    bool vol_on = vol_pack != 0u;
    uint vol_base = vol_on ? (vol_pack - 1u) : 0xFFFFFFFFu;
    // Contact-gap routing flags (cache_tile bits 12..13 — the max_mip nibble is 8..11 and the
    // mip index starts at 16, so these ride free; the gallery packs a bare tile = both off).
    // bit12 (`P_CACHE_OCCL_ROUTE`): a cache miss whose query saw an occluder-witness texel
    // routes to the DARK analytic fallback (no sky top-up) — the point is enclosed.
    // bit13 (`P_REFLECT_PROBE_VALID`): the fallback's GI-volume read uses the
    // validity-weighted probe interpolation instead of the plain trilinear.
    bool occl_route = (pc.cache_tile & 0x1000u) != 0u;
    bool probe_valid = (pc.cache_tile & 0x2000u) != 0u;
    // bit14 (`P_REFLECT_GRAZE_EPS`): cone-proportional march hit threshold — see the march.
    bool graze_eps = (pc.cache_tile & 0x4000u) != 0u;
    // bit15 (`P_REFLECT_HIT_FETCH`): SW compact screen-color-at-hit — the marched hit gets the
    // same on-screen lit-history fetch the HWRT hybrid shades from (see the fetch block).
    bool hit_fetch = (pc.cache_tile & 0x8000u) != 0u;
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
#ifdef REFLECT_COMPACT
    // B2 anti-aliasing inputs (see screen_trace): per-texel reflected-ray divergence = the view
    // cone of one refine texel + twice the surface-normal change across it (a curved mirror
    // magnifies; the reflected direction turns 2x the normal). pa_full = full-res pixel angle.
    float geo_spread;
    float pa_full;
    {
        // One-texel view angle from the projection itself (two unprojected near-plane rays).
        float4 c0 = mul(pc.inv_view_proj, float4(0.0, 0.0, 0.5, 1.0));
        float4 c1 = mul(pc.inv_view_proj, float4(2.0 / float(pc.width), 0.0, 0.5, 1.0));
        float3 r0 = normalize(c0.xyz / c0.w - pc.cam_pos.xyz);
        float3 r1 = normalize(c1.xyz / c1.w - pc.cam_pos.xyz);
        float texel_angle = acos(clamp(dot(r0, r1), -1.0, 1.0));
        uint fw, fh;
        g.textures[pc.depth_index].GetDimensions(fw, fh);
        pa_full = texel_angle * float(pc.width) / max(float(fw), 1.0);
        float3 n1 = normalize(
            g.textures[pc.normal_index].SampleLevel(g.samp, uv + float2(1.0 / pc.width, 0.0), 0.0)
                .xyz);
        float dn = acos(clamp(dot(n, n1), -1.0, 1.0));
        geo_spread = texel_angle + 2.0 * dn;
    }
#else
    const float geo_spread = 0.0; // legacy: point-fetch screen hits (bit-identical)
    const float pa_full = 1.0;
#endif
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
    // Rough-prefilter split (SW content opt-in): above this roughness the stochastic 1-ray/frame
    // GGX estimator is the WRONG tool — the lobe is so wide that any reachable sample count leaves
    // residual speckle (the ~50x undersampling diagnosis), while the surface-cache cone MIP already
    // holds a prefiltered version of exactly the radiance the wide lobe integrates. So trace ONE
    // deterministic mirror ray and sample the cache with a roughness-driven cone footprint instead:
    // structural noise ZERO, and the visual difference from the true lobe integral is hidden by the
    // lobe's own blur (reference engine: roughness above RadianceCache.MaxRoughness skips tracing
    // and reads the prefiltered probe). Threshold rides `max_steps` bits 16..23 (roughness*255,
    // 0 = off; the SW step cap fits in bits 0..15, tiers cap it at 256). The HWRT path never sets
    // these bits (its max_steps field carries the lit-history index).
    uint prefilter8 = (pc.max_steps >> 16) & 0xFFu;
    bool prefiltered =
        content_refl && prefilter8 != 0u && roughness >= float(prefilter8) * (1.0 / 255.0);
    // A5 sampler select (max_steps bit30): blue-noise (frame-varying, low-discrepancy) vs the
    // legacy white jitter. Hoisted here because the B2' sample loop below redraws with it too.
    uint stoch_flag = (pc.max_steps & 0x40000000u) != 0u ? 1u : 0u;
    if (roughness > mirror_thresh && !prefiltered)
    {
        if (content_refl)
        {
            // A5: content routes through the shared sampler so the A1 resolve reconstructs the SAME
            // ray. Blue-noise + the A1/A4 denoiser lets the temporal pass converge the lobe instead
            // of the fixed-jitter dead end.
            dir = refl_ggx_dir(n, v, roughness, tid.x, tid.y, pc.frame, stoch_flag);
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

    // B2' glossy sample density (`P_REFLECT_GLOSSY_SPP`, max_steps bits 24..29, SW content
    // only): K GGX rays per pixel per frame for the stochastic glossy band — the direct fix for
    // the ~50x undersampling that leaves residual speckle no denoiser can remove (it cannot
    // average in radiance the rays never sampled). Mirror / prefiltered / gallery pixels keep the
    // single deterministic ray (K applies only where the ray is a GGX draw).
    uint spp = 1u;
#ifndef HWRT_REFLECT
    if (content_refl && !prefiltered && roughness > mirror_thresh)
        spp = ((pc.max_steps >> 24) & 0x3Fu) + 1u; // packed as K-1 (0 = legacy single ray)
#endif
    // Accumulators: K samples average in the Reinhard denoiser space (V / (1 + luma*0.1)) like the
    // A1 resolve / temporal — a single bright HDR hit (sky through the lobe) must not dominate the
    // mean. spp == 1 bypasses the tonemap round-trip entirely (bit-exact legacy single-sample
    // path — the gallery anchor).
    float3 acc_tm = float3(0.0, 0.0, 0.0);
    float acc_t = 0.0;
    float3 out_color = float3(0.0, 0.0, 0.0);
    float out_t = 0.0;
    [loop]
    for (uint samp = 0u; samp < spp; samp++)
    {
        // Sample s of K redraws the GGX ray farther along the SAME low-discrepancy sequence
        // (seq = frame*K + s): K samples/frame advance the R2 rotation exactly as K frames would,
        // and the A1 resolve reconstructs every neighbour ray with the same law.
        if (spp > 1u)
            dir = refl_ggx_dir(n, v, roughness, tid.x, tid.y, pc.frame * spp + samp, stoch_flag);
        float t = bias;
        bool hit = false;
        // A ray that never saw open space (a contact gap under a mirror) — its point is
        // topologically ENCLOSED, so the analytic fallback must not top it up with the
        // unoccluded sky irradiance (that read as a bright blue pool under the ball).
        bool trapped = false;
        float3 hp = p;
        // B2' screen-hit early-out: a validated on-screen hit takes the previous frame's full-res
        // lit radiance and skips the GDF march + cache shade for this sample. Prefiltered pixels
        // skip it — their colour must be the cache CONE average, not one sharp full-res sample.
        bool screen_done = false;
        float3 screen_col = float3(0.0, 0.0, 0.0);
#ifdef SCREEN_HIT
        if (content_refl && !prefiltered)
        {
            float t_screen;
            screen_done = screen_trace(p, dir, sy, bias, max_dist, asuint(pc.params.y),
                                       screen_col, t_screen, geo_spread, pa_full);
            if (screen_done)
                t = t_screen; // a miss must NOT touch t (the GDF march starts at `bias`)
        }
#endif
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
        // The step cap lives in bits 0..15 (tiers cap it at 256); the upper bits carry flags — bit31
        // content, bit30 A5 stochastic, bits 16..23 the prefilter threshold. Gallery passes a bare
        // cap (96, upper bits zero) → byte-identical.
        int max_steps = int(max(pc.max_steps & 0xFFFFu, 1u));
        const float HIT_EPS = 0.001;
        // Content self-hit filter: a grazing ray off a convex mirror's rim hugs — or wraps
        // around — its own coarse blurred field (the direction bias cannot clear it; a normal
        // test cannot identify it either, since the wrap lands on the far side of the curve),
        // "hits" its own surface, and cache-samples the MIRROR'S OWN cards: the chrome ball's
        // bottom band read as a featureless bright blob of its own card radiance. A hit only
        // counts once the ray has ESCAPED open space (field value past ~0.4% of the scene
        // diagonal). Rays TRAPPED to the end (a contact gap, e.g. under the ball, never sees
        // open space) resolve as a hit at their final position below — the local cache/analytic
        // floor tone — NOT as a sky miss (that read as a blue pool under the mirror). Gallery
        // starts pre-escaped — the accept-first-hit legacy, byte-identical even for rays whose
        // biased origin reads a NEGATIVE field value.
        // The escape threshold keys to the field's actual blur band: with detail-replace
        // (direct per-mesh sampling + flags bit0) the atlas voxel is ~an order finer than the
        // dense grid, so a quarter threshold lets contact-gap rays that genuinely see accurate
        // open space (the ball-floor gap clears ~3x this) escape and hit the real floor instead
        // of resolving as trapped.
        bool ms_detail =
            pc.clip_count == 0u && (ms_load_header(pc.clip_desc).flags & 1u) != 0u;
        float d_escape = (ms_detail ? 0.001 : 0.004) * length(GDF_MAX - GDF_MIN);
        bool escaped = !vol_on;
        // Closest approach along a trapped ray (the march can end up crawling INSIDE geometry
        // once its step floor bottoms out — that point answers nothing; the min-|d| point sits
        // ON a real surface and its cards can).
        float d_min = 1e9;
        float t_min = t;
        for (int i = 0; i < max_steps && !screen_done; i++)
        {
            float3 rp = p + dir * t;
            float d = scene_march(rp);
            escaped = escaped || d >= d_escape;
            if (abs(d) < d_min)
            {
                d_min = abs(d);
                t_min = t;
            }
            // Graze ramp (bit14, content opt-in): the fixed 1 mm threshold under-detects a
            // GRAZING surface — a near-tangent ray off the mirror's bottom rim escapes the
            // contact blur, then skims metres above the floor it geometrically re-enters
            // (d stays at the clearance, never < 1 mm) and finally dips somewhere on the
            // SUNLIT mid-corridor floor: the contact-gap band read that bright far card
            // (~96) instead of the shadowed near floor the exact trace hits (hybrid t was
            // 0.2..0.5 m where the SW march ran 2..8 m). Growing the threshold with the
            // marched distance — the reference SW trace's expand-surface ramp ("avoid
            // self-occlusion near the start, detect thin surfaces far") — resolves the ray
            // at its first true graze; the slope is a quarter of the cone-LOD step floor so
            // the acceptance can never outrun the marcher's own step, and the positional
            // slack it introduces (≤ the cone footprint at that distance) is exactly what
            // the cache cone MIP / acceptance tolerance already absorb. cone_k = 0 (gallery)
            // keeps the ramp at the legacy constant even if the bit were set.
            float hit_eps = graze_eps ? HIT_EPS + 0.25 * pc.ground_albedo.w * t : HIT_EPS;
            if (d < hit_eps && escaped)
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
        // Trapped ray (see above): shade its closest approach instead of falling to the sky miss.
        if (!hit && !escaped && !screen_done)
        {
            t = min(t_min, max_dist);
            hp = p + dir * t;
            hit = true;
            trapped = true;
        }
#endif

        float3 color;
        // Ray-cone slope (world radius per unit hit distance). The stochastic path uses a small
        // per-pixel half-angle (~0.08, the trace-res pixel cone) — the LOBE's spread is realised by
        // the distribution of rays. The prefiltered path traces ONE mirror ray, so the lobe's spread
        // must come from the cone itself: the median GGX half-vector tilt is tan(θ_h) ≈ α (= r²), and
        // the REFLECTED direction deviates by 2·θ_h, so slope = tan(2θ_h) ≈ 2α (0.32 at the 0.4
        // threshold, ~63° cone at roughness 1). The cache cone MIP / albedo cone then average the
        // footprint the wide lobe would have integrated.
        float alpha_pf = roughness * roughness;
#ifdef REFLECT_COMPACT
        // The compacted refine runs at div2 — its per-texel cone is ~3x tighter than the 1/6-res
        // trace the 0.08 slope was tuned for, so the inherited slope over-blurs the cache MIP for
        // the mirror (washy floor shaft / soft far reflections). The dense ray grid already
        // handles the blockiness the wide cone used to hide.
        const float cone_slope = 0.03;
#else
        float cone_slope = prefiltered ? clamp(2.0 * alpha_pf, 0.08, 2.0) : 0.08;
#endif
        if (screen_done)
        {
            // B2' screen hit: the previous frame's full-res lit radiance at the validated on-screen
            // hit — no march, no cache shade (exact colour, ~free).
            color = screen_col;
        }
        else if (hit)
        {
            float3 hn = scene_normal(hp);
            bool cached = false;
            // Occluder-witness routing (bit12): set by the cache lookup below when the query's
            // texel proves the point sits UNDER an occluder — route to the dark fallback.
            bool occluded = false;
#ifdef REFLECT_COMPACT
            // Gate-band hand-off state: in the footprint band the screen colour does NOT finalize
            // the pixel — the cache/hybrid colour still resolves below and the two are blended at
            // the end of the hit branch, so the source transition is a gradient, not a cut.
            float3 screen_mix = float3(0.0, 0.0, 0.0);
            float screen_w = 0.0;
#endif
#if defined(HWRT_REFLECT) || (defined(SCREEN_HIT) && defined(REFLECT_COMPACT))
            // Phase 16 B.2 — screen-color-at-hit (reference-engine `SampleSceneColorAtHit`): the
            // ray hit REAL geometry, so if that surface is visible on screen its FULL-RES lit colour is
            // already in the previous frame's lit-radiance history (`globals.prev_view_proj` + the
            // `lit_hist` buffer). Sample it directly for a SHARP reflection — the low-res surface cache
            // is only the off-screen / occluded fallback. This is the sharp-reflection win the SW trace
            // couldn't give, and the validated HW hit is exactly what our screen-space SSR lacked.
            // The lit-history index rides the (march-cap) field — unused in the HW variant (no march).
            //
            // SW compact (`SCREEN_HIT`+`REFLECT_COMPACT`, bit15 opt-in): the same fetch serves the
            // MARCHED hit. The pre-march screen_trace above cannot serve a contact band — its
            // samples cross the mirror-occluded zone (validation correctly fails against the
            // mirror's own depth) and then leave the frame below it — but the fetch AT the hit
            // validates by proximity: near a resting mirror's contact the depth surface (the
            // mirror's on-screen bottom) is spatially within the parallax tolerance of the true
            // reflected floor, so the band reads the on-screen contact region's lit colour — the
            // grazing-aware witness (screen-space reflections darken a glossy floor at grazing)
            // the diffuse-only card can never provide. This is exactly how the HWRT hybrid's
            // contact band gets its tone; without it the SW band read the card's diffuse
            // radiance, ~1.5x the hybrid and ~4x the path tracer.
            {
#ifdef HWRT_REFLECT
                // bit31 = content flag; bit30 = compact screen-fetch enable (REFLECT_COMPACT
                // only — see below); the low bits carry the lit-history bindless index.
                uint lit_hist = pc.max_steps & 0x3FFFFFFFu;
                bool fetch_on = (pc.max_steps & 0x40000000u) != 0u;
#else
                // SW: the lit-history index rides params.y (the SCREEN_HIT overload); max_steps
                // bit30 is the A5 stochastic flag here, so the gate is the bit15 knob instead.
                uint lit_hist = asuint(pc.params.y);
                bool fetch_on = hit_fetch;
#endif
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
#ifdef REFLECT_COMPACT
                                // Compacted mirror screen fetch (max_steps bit30, opt-in): serve
                                // an on-screen hit whose reflected footprint is near a pixel from
                                // the full-res lit history (sharp), box-filtered as the footprint
                                // grows; wider footprints fall through to the hybrid cache cone.
                                // This distance-driven gate used to split the mirror into two
                                // tone families (the cache family ran skylight-blue vs the lit
                                // history), so it was dropped for a single source; the relight's
                                // deferred-parity skylight re-unified the tones, which is what
                                // makes the mix viable again. Gate OFF = single-source hybrid.
                                if (fetch_on)
                                {
                                    float fp_px = geo_spread * t /
                                                  max(distance(pc.cam_pos.xyz, hp) * pa_full, 1e-6);
                                    if (fp_px <= 48.0)
                                    {
                                        float3 sc;
                                        if (fp_px > 3.0)
                                        {
                                            int rr = int(clamp(fp_px * (1.0 / 3.0), 1.0, 16.0));
                                            float3 bsum = float3(0.0, 0.0, 0.0);
                                            [unroll]
                                            for (int by = -1; by <= 1; by++)
                                            {
                                                [unroll]
                                                for (int bx = -1; bx <= 1; bx++)
                                                {
                                                    int2 sp2 =
                                                        clamp(pp + int2(bx, by) * rr, int2(0, 0),
                                                              int2(int(fw) - 1, int(fh) - 1));
                                                    uint si = uint(sp2.y) * fw + uint(sp2.x);
                                                    bsum += asfloat(g.storage_buffers[lit_hist]
                                                                        .Load4(si * 16u))
                                                                .rgb;
                                                }
                                            }
                                            sc = bsum * (1.0 / 9.0);
                                        }
                                        else
                                        {
                                            uint pidx = uint(pp.y) * fw + uint(pp.x);
                                            sc = asfloat(g.storage_buffers[lit_hist].Load4(
                                                             pidx * 16u))
                                                     .rgb;
                                        }
                                        // Hand-off band: pure screen colour while the footprint is
                                        // comfortably sub-filter (≤ 6 px), fading to the cache
                                        // hybrid toward 48 px. A curved mirror sweeps the whole
                                        // footprint range across FEW mirror pixels, so the band
                                        // must be wide in footprint terms to read as a gradient
                                        // on the mirror at all; past ~24 px the sparse box filter
                                        // undersamples, but its weight is ≤ ~0.35 there and
                                        // falling — the cache dominates before the sparseness
                                        // could read as speckle.
                                        screen_w = 1.0 - smoothstep(6.0, 48.0, fp_px);
                                        // The fetch's OTHER gates are binary too and their
                                        // pass/fail contours are where the source cut actually
                                        // shows (the footprint varies smoothly; the validation
                                        // doesn't). A curved mirror compresses reflected-space
                                        // margins into few mirror pixels, so the fades lean on
                                        // quantities that stay smooth ACROSS the mirror: the
                                        // reflected point's view depth (pclip.w → 0 = the
                                        // behind-camera boundary, faded over ~5% of the scene
                                        // diagonal), a wide screen-border margin, and the
                                        // occlusion tolerance shell (about to lose the
                                        // depth-buffer witness to a nearer surface).
                                        float2 eb = min(puv, float2(1.0, 1.0) - puv);
                                        screen_w *= smoothstep(0.0, 0.12, min(eb.x, eb.y));
                                        screen_w *= smoothstep(
                                            0.0, 0.05 * length(GDF_MAX - GDF_MIN), pclip.w);
                                        screen_w *=
                                            1.0 - smoothstep(0.5 * vtol, vtol,
                                                             distance(sworld, hp));
                                        if (screen_w >= 1.0)
                                        {
                                            // Pure screen: finalize here and DISARM the end-of-hit
                                            // blend (screen_mix stays zero — blending against it
                                            // would black the pixel out).
                                            color = sc;
                                            cached = true;
                                            screen_w = 0.0;
                                        }
                                        else
                                        {
                                            screen_mix = sc;
                                        }
                                    }
                                }
#else
                                uint pidx = uint(pp.y) * fw + uint(pp.x);
                                color = asfloat(g.storage_buffers[lit_hist].Load4(pidx * 16u)).rgb;
                                cached = true; // sharp on-screen hit — skip the low-res cache + analytic
#endif
                            }
                        }
                    }
                }
            }
#endif
#ifdef HWRT_REFLECT
            // Phase 16 E — Hit Lighting (OFF-SCREEN): the on-screen path failed (behind camera / occluded),
            // so shade this HW hit with the REAL material instead of the low-res surface cache. `frame`
            // bit31 = enabled; the consolidated (vertex, index, drawable-table) bindless indices ride the
            // coarse-albedo push slots (unused here — Hit Lighting replaces the analytic fallback). Fetch
            // the hit triangle's interpolated normal + UV, sample the albedo texture, and re-light with the
            // sun (HW shadow ray) + GI radiance cache + IBL sky — the same energy convention as the analytic
            // path below, but with the accurate material/normal/shadow. Reference: Lumen HitLighting mode.
            // Hybrid hand-off (reference-engine surface-cache decomposition — FinalLighting =
            // (Direct+Indirect)·Albedo/π): hit lighting no longer finalizes the colour. It
            // captures the TRUE material albedo at the exact hit (`hl_alb`) and its analytic
            // re-light as a fallback; the surface-cache cone below recovers the card's CONVERGED
            // multibounce lighting as radiance/albedo, and the combine re-multiplies it by
            // `hl_alb` — parity-grade tone (the cache path measures 1.2/255 vs the path tracer)
            // carrying real texture detail. The pure analytic hit-light alone measured 27.5/255
            // off the ground truth (its GI-volume ambient underestimates interior bounce).
            float3 hl_alb = float3(0.0, 0.0, 0.0);
            float3 hl_fallback = float3(0.0, 0.0, 0.0);
            // The IBL-sky part of the fallback, kept separate so the occluder-witness routing
            // can drop it (an enclosed hit sees no sky) without losing the direct + GI terms.
            float3 hl_sky = float3(0.0, 0.0, 0.0);
            bool hl_have = false;
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
                    {
                        // Ray-cone texture LOD (the path tracer's hit_lod_base): a mip-0 point
                        // sample of a MINIFIED reflection turns the albedo into per-ray speckle
                        // no cone/resolve can average out. Derive the LOD from the cone footprint
                        // at the hit vs the triangle's world/UV areas (object-space areas — a
                        // uniform instance scale shifts the LOD slightly, same caveat as the PT).
                        float3 p0 = asfloat(g.storage_buffers[pc.albedo_r].Load3(i0 * 32u));
                        float3 p1 = asfloat(g.storage_buffers[pc.albedo_r].Load3(i1 * 32u));
                        float3 p2 = asfloat(g.storage_buffers[pc.albedo_r].Load3(i2 * 32u));
                        float a_uv = abs((uv1.x - uv0.x) * (uv2.y - uv0.y)
                                         - (uv2.x - uv0.x) * (uv1.y - uv0.y));
                        float a_world = length(cross(p1 - p0, p2 - p0));
                        float ratio = a_world > 1e-12 ? a_uv / a_world : 0.0;
                        float cosw = max(abs(dot(nw, dir)), 1e-3);
                        // Cone width at the hit: the same footprint the cache MIP selection uses.
                        float cwid = 0.05 + t * cone_slope;
                        uint tw, th;
                        g.textures[tex_albedo].GetDimensions(tw, th);
                        float lod = max(0.5 * log2(max(ratio, 1e-20))
                                            + log2(max(cwid, 1e-8) / cosw)
                                            + 0.5 * log2(float(max(tw * th, 1u))),
                                        0.0);
                        alb *= g.textures[tex_albedo].SampleLevel(g.samp_wrap, uv, lod).rgb;
                    }
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
                    float3 e = vol_on ? (probe_valid ? sample_gi_irradiance_valid(vol_base, hp, nw)
                                                     : sample_gi_irradiance(vol_base, hp, nw))
                                      : float3(0.0, 0.0, 0.0);
                    hl_alb = alb;
                    hl_fallback = alb * ((sun_i * ndl * sh) * (1.0 / GGX_PI) + e);
                    if (has_irr)
                        hl_sky = g.cubes[irr_idx].SampleLevel(g.samp, nw, 0.0).rgb * alb;
                    hl_have = true; // combine after the cache cone below (do NOT finalize here)
                }
            }
#endif
            float3 cone_alb = float3(0.0, 0.0, 0.0);
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
                // Track C card grid: unpack early — grid mode also CAPS the distance-growing
                // acceptance term (at 1.2% of the diag, ≈ t = 0.4·diag of slack): the unbounded
                // growth is what forced the host to dilate every card by the worst case (fat
                // cells); beyond the cap a far hit accepts cards >1% of the scene away, which
                // score ≈ 0 anyway (align/(1+4·dist)) and only pollute the cone average.
                uint gc8 = (pc.cache.x >> 8) & 0xFFu;
                uint gp8 = (pc.cache.x >> 16) & 0xFFu;
                float diag_sc = length(GDF_MAX - GDF_MIN);
                float bias_t = gc8 != 0u ? min(t * 0.03, 0.012 * diag_sc) : t * 0.03;
                float cache_bias = vol_on ? (0.006 * diag_sc + bias_t) : 0.0;
                // A TRAPPED endpoint (see the march) is not a converged surface hit — its
                // positional uncertainty is the whole contact gap — so give the card acceptance
                // the matching radius: the floor card ~a gap-width away holds the converged
                // (tone-calibrated) radiance this enclosed region actually reflects. Without it
                // the lookup missed and the analytic (GI-volume ~0 in a gap, no sky) went black.
                if (trapped)
                    cache_bias *= 2.0;
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
                float mip_radius = 0.05 + t * cone_slope;
                // Track C card grid + C2a layout: `cache.x` packs cards | (grid_cells+1)<<8 |
                // (grid_pool+1)<<16 | (layout+1)<<24 (0 = off — gallery/legacy pass a bare index,
                // bits 8.. zero). Bindless storage indices are < 64, so the low byte always holds
                // the real cards index.
                uint cards_idx = pc.cache.x & 0xFFu;
                uint la8 = (pc.cache.x >> 24) & 0xFFu;
                uint grid_cells = gc8 == 0u ? 0xFFFFFFFFu : gc8 - 1u;
                uint grid_pool = gp8 == 0u ? 0xFFFFFFFFu : gp8 - 1u;
                uint sc_layout = la8 == 0u ? 0xFFFFFFFFu : la8 - 1u;
                // C2b: `cache.y` packs cache_pos | (feedback+1)<<8 — the desired-res feedback the
                // host re-layouts the adaptive atlas around (0 = off, legacy bare index).
                uint cpos_idx = pc.cache.y & 0xFFu;
                uint fb8 = (pc.cache.y >> 8) & 0xFFu;
                uint sc_feedback = fb8 == 0u ? 0xFFFFFFFFu : fb8 - 1u;
                // Mirror-feedback mark buffer rides cache.y bits 16..23 (0 = off): cards a
                // NEAR-MIRROR ray reads get relight priority (see surface_cache.slang).
                // COMPACT-ONLY: the main trace's glossy/prefiltered pixels cover most of the
                // frame and read most of the pool — marking there accelerated ~half the cards
                // and more than doubled the relight cost (measured 5.0 -> 13.2 ms). The mirror
                // band is the one place the un-relit fallback is DIRECTLY visible.
#ifdef REFLECT_COMPACT
                uint mk8 = (pc.cache.y >> 16) & 0xFFu;
                uint sc_mark = mk8 == 0u ? 0xFFFFFFFFu : mk8 - 1u;
#else
                const uint sc_mark = 0xFFFFFFFFu;
#endif
                // Captured-albedo atlas rides cache.y bits 24..31 (0 = off) — only fetched when
                // the hit-lighting hybrid needs the radiance/albedo split (HWRT builds only).
#ifdef HWRT_REFLECT
                uint ab8 = (pc.cache.y >> 24) & 0xFFu;
                uint sc_alb = (ab8 == 0u || !hl_have) ? 0xFFFFFFFFu : ab8 - 1u;
#else
                const uint sc_alb = 0xFFFFFFFFu;
#endif
                // Lit-calibration correction rides cache.w bits 16..23 (+1; 0 = off — the card
                // count needs only the low 16 bits, and legacy/gallery pass it bare).
                uint nc16 = pc.cache.w & 0xFFFFu;
                uint cr8 = (pc.cache.w >> 16) & 0xFFu;
                uint sc_corr = cr8 == 0u ? 0xFFFFFFFFu : cr8 - 1u;
                color = sample_surface_cache_cone(hp, hn, cards_idx, cpos_idx, pc.cache.z, nc16,
                                                  c_tile, cache_bias, mip_radius, mip_index, c_maxmip,
                                                  grid_cells, grid_pool, sc_layout, sc_feedback,
                                                  sc_mark, sc_alb, sc_corr, occl_route ? 1u : 0u,
                                                  cone_alb, cached, occluded);
            }
#ifdef HWRT_REFLECT
            // Hit-lighting hybrid combine: cache lighting × true material albedo. The clamp
            // bounds the ratio where the 32² card albedo underestimates a dark texel.
            if (hl_have)
            {
                if (cached && dot(cone_alb, float3(1.0, 1.0, 1.0)) > 1e-3)
                {
                    float3 ratio = clamp(hl_alb / max(cone_alb, 0.02), 0.0, 4.0);
                    color *= ratio;
                }
                else if (!cached)
                {
                    // No card coverage: keep the analytic hit-light. An occluder-witness miss
                    // (bit12) drops the sky term — the enclosed point sees no sky.
                    color = hl_fallback + (occluded ? float3(0.0, 0.0, 0.0) : hl_sky);
                    cached = true;
                }
            }
#endif
            if (!cached)
            {
                float ndl = max(dot(hn, sun_dir), 0.0);
                float sh = ndl > 0.0 ? refl_shadow(hp + hn * 0.004, sun_dir, max_dist) : 1.0;
                // The ground plane is analytic — it's NOT in the per-voxel albedo volume, so
                // albedo_at() would return the nearest *object's* colour for a ground hit (the
                // red cube / green avocado bleeding onto the reflected floor). Use the ground's
                // own albedo (matches deferred.rs's [0.8] ground material) for ground hits.
                // Content scenes (vol_on) bake their real floor MESH into the albedo volume, so
                // the volume answers correctly at a ground-plane hit too — and the constant grey
                // is visibly wrong there (the SW near-mirror band: short rays reject the card
                // tolerance, fall here, and the mirror's bottom turns into a featureless blob of
                // pc.ground_albedo). The constant stays for the gallery, whose analytic ground is
                // NOT in the volume (nearest-object colour would bleed onto it) — byte-identical.
                bool is_ground = (hp.y - GROUND_Y) < geo_inside(hp) && !vol_on;
                // Cone-blurred albedo (content, vol_on) so an uncovered far reflection is as smooth as
                // its cached neighbours instead of blocky per-voxel colour. Near hits (radius→0) and the
                // gallery (vol_on false) keep the single-tap legacy albedo.
                float ablur = vol_on ? (0.05 + t * cone_slope) : 0.0;
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
                    float3 e = probe_valid ? sample_gi_irradiance_valid(vol_base, hp, hn)
                                           : sample_gi_irradiance(vol_base, hp, hn);
                    color = hit_albedo * ((sun_i * ndl * sh) * (1.0 / GGX_PI) + e);
                    // Physical IBL-diffuse skylight fill: the radiance-cache volume `e` is a LOCAL bounce
                    // term and is ~0 where the cache is cold, so an uncovered reflected surface would go
                    // near-black. Add the same sky irradiance the deferred pass gives every surface
                    // (`irradiance_cube(hn) * albedo`) so a reflected shadowed/uncovered face reads its
                    // true skylit grey — the reflection then matches the directly-shaded scene. A
                    // TRAPPED endpoint (see the march) is enclosed by construction — no sky there;
                    // same for an occluder-witness cache miss (bit12: the capture PROVED the point
                    // sits under an occluder, so the unoccluded top-up is exactly the bright bowl).
                    if (has_irr && !trapped && !occluded)
                        color += g.cubes[irr_idx].SampleLevel(g.samp, hn, 0.0).rgb * hit_albedo;
                }
                else
                {
                    // Volume absent (e.g. the gallery) -> the legacy crude sky fill. Byte-identical.
                    float fill = sky_fill * (0.5 + 0.5 * hn.y);
                    color = hit_albedo * (sun_i * ndl * sh + fill) * (1.0 / GGX_PI);
                }
            }
#ifdef REFLECT_COMPACT
            // Gate-band hand-off: blend the (box-filtered) screen colour over the resolved
            // cache/hybrid colour across the footprint band (see the fetch above).
            if (screen_w > 0.0)
                color = lerp(color, screen_mix, screen_w);
#endif
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

        out_color = color;
        out_t = t;
        if (spp > 1u)
        {
            acc_tm += color / (1.0 + dot(color, float3(0.2126, 0.7152, 0.0722)) * 0.1);
            acc_t += t;
        }
    }
    float3 color;
    float t;
    if (spp == 1u)
    {
        color = out_color; // bit-exact legacy single-sample path
        t = out_t;
    }
    else
    {
        float3 tm = acc_tm / float(spp);
        color = tm / max(1.0 - dot(tm, float3(0.2126, 0.7152, 0.0722)) * 0.1, 1e-3);
        t = acc_t / float(spp);
    }
    // Reflection HIT DISTANCE (world units) in the output/persist `.w` — the temporal resolve uses it
    // to reproject the VIRTUAL IMAGE (the reflected point at `surface_dist + t` along the view ray) so a
    // mirror reflection stays stable under camera motion instead of swimming (reference-engine
    // LumenReflectionDenoiserTemporal virtual-image reproject). `t` is the ray parameter at the hit
    // (≈ ray max on a miss = the sky, effectively far). Alpha is unused by the composite/lighting, so
    // this is inert for every non-temporal-resolve consumer (gallery byte-identical).
    float hit_dist = t;
#ifdef REFLECT_COMPACT
    // The composite's validity-weighted fetch keys on `.a > 0` (the classify pass zeroed every
    // un-traced refine texel), so a traced texel must never store an exact-zero hit distance.
    hit_dist = max(hit_dist, 1e-4);
#endif
    // A3: persist this freshly-traced (surface point, radiance) for next frame's reuse gate.
    if (skip_write)
    {
        g.storage_buffers[sk.write].Store4(lin * 32u, asuint(float4(p, 1.0)));
        g.storage_buffers[sk.write].Store4(lin * 32u + 16u, asuint(float4(color, hit_dist)));
    }
    g.storage_images[pc.out_index][tid.xy] = float4(color, hit_dist);
}
