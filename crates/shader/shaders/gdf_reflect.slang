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
                float3 prad = asfloat(g.storage_buffers[sk.read].Load4(lin * 32u + 16u)).rgb;
                if (skip_write)
                {
                    g.storage_buffers[sk.write].Store4(lin * 32u, asuint(float4(p, 1.0)));
                    g.storage_buffers[sk.write].Store4(lin * 32u + 16u, asuint(float4(prad, 0.0)));
                }
                g.storage_images[pc.out_index][tid.xy] = float4(prad, 1.0);
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
    if (roughness > 0.04)
    {
        // GGX importance-sampled reflection ray. `pc.frame` decorrelates the jitter per frame; the
        // host passes a FIXED frame (0) for content so the per-pixel jitter is temporally STABLE —
        // the temporal pass can't average a 1-ray/frame stochastic lobe to convergence (auto-exposure
        // shifts reject the history too), so a frame-varying ray leaves sparkle, while a fixed jitter
        // + the resolve's spatial gather reconstructs the lobe smoothly. The gallery passes the real
        // frame to stay byte-identical to its legacy anchor.
        uint rng = (tid.x * 1973u + tid.y * 9277u + pc.frame * 26699u) | 1u;
        float3 h = sample_ggx(n, roughness, rng);
        dir = reflect(-v, h);
        if (dot(dir, n) <= 0.0)
            dir = reflect(-v, n); // keep the ray in the upper hemisphere
    }
    else
    {
        dir = reflect(-v, n);
    }

    float max_dist = pc.params.x;
    float sky_fill = pc.params.z;
    float bias = pc.params.w;

    // Sphere-trace the reflection ray. Stage D3: content lowers the step cap; gallery = 96.
    int max_steps = int(max(pc.max_steps, 1u));
    const float HIT_EPS = 0.001;
    float t = bias;
    bool hit = false;
    float3 hp = p;
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

    float3 color;
    if (hit)
    {
        float3 hn = scene_normal(hp);
        // C8b3/C8g: the reflected surface's cached (multibounce, accurately-lit) radiance — a
        // direct lookup when the cache is bound. The cache only covers the meshed objects (no
        // cards for the analytic ground), so on a cache MISS fall back to the C8a per-ray
        // re-light instead of black (which previously left the ground reflection dark).
        bool cached = false;
        if (pc.cache.x != 0xFFFFFFFFu)
        {
            color = sample_surface_cache(hp, hn, pc.cache.x, pc.cache.y, pc.cache.z,
                                         pc.cache.w, pc.cache_tile, cached);
        }
        if (!cached)
        {
            float ndl = max(dot(hn, sun_dir), 0.0);
            float sh = ndl > 0.0 ? refl_shadow(hp + hn * 0.004, sun_dir, max_dist) : 1.0;
            float fill = sky_fill * (0.5 + 0.5 * hn.y);
            // The ground plane is analytic — it's NOT in the per-voxel albedo volume, so
            // albedo_at() would return the nearest *object's* colour for a ground hit (the
            // red cube / green avocado bleeding onto the reflected floor). Use the ground's
            // own albedo (matches deferred.rs's [0.8] ground material) for ground hits.
            bool is_ground = (hp.y - GROUND_Y) < geo_inside(hp);
            float3 hit_albedo = is_ground ? pc.ground_albedo.xyz : albedo_at(hp);
            // Lambertian reflected surface: albedo/PI * irradiance, so the reflection of a
            // diffuse surface matches its directly-shaded appearance. The cache path (meshed
            // hits) gets the same 1/PI in sdf_cache_light.slang.
            color = hit_albedo * (sun_i * ndl * sh + fill) * (1.0 / GGX_PI);
        }
    }
    else
    {
        color = sky(dir, sun_dir, sun_i, false); // specular miss -> sky w/o sun (direct light owns the glint)
    }

    // A3: persist this freshly-traced (surface point, radiance) for next frame's reuse gate.
    if (skip_write)
    {
        g.storage_buffers[sk.write].Store4(lin * 32u, asuint(float4(p, 1.0)));
        g.storage_buffers[sk.write].Store4(lin * 32u + 16u, asuint(float4(color, 0.0)));
    }
    g.storage_images[pc.out_index][tid.xy] = float4(color, 1.0);
}
