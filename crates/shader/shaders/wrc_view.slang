// World radiance cache debug view — GI state painted onto the distance field.
//
// A full-screen debug visualization: march the camera ray into the scene's baked global
// distance field (the same clipmap the SW ray tracer uses), and shade each hit surface with
// the diffuse indirect irradiance held in the world radiance cache. The result is a flyable
// "surface / radiance cache" view — it shows what indirect light the cache currently stores,
// resolved onto the real (baked) distance-field geometry rather than any G-buffer.
//
// The primary march + hit normal REUSE the shared bounce tracer's field functions
// (`bs_scene_march` / `bs_scene_normal`) so the surface the ray finds is identical to the one
// the GI passes see. The shade REUSES the cache's `wrc_irradiance` reconstruction. This pass
// adds no new distance-field or cache math of its own.
//
// Output is raw radiance into a storage image; the host tonemap pass applies exposure + the
// filmic curve, exactly like the SW-RT primary trace (gdf_trace.slang).
//
// Include order requirement (dependencies before dependents): bindless.slang, clipmap.slang,
// surface_cache.slang, gdf_bounce.slang, octahedral.slang, wrc_common.slang.

#include "bindless.slang"
#include "clipmap.slang"
#include "surface_cache.slang"
#include "gdf_bounce.slang"
#include "octahedral.slang"
#include "wrc_common.slang"

struct WrcViewPush
{
    float4x4 inv_view_proj; // 0   clip -> world (primary ray reconstruction)
    float4 cam_pos;         // 64  xyz camera world position
    float4 aabb_min;        // 80  xyz GDF world min, w = ground plane Y
    float4 aabb_max;        // 96  xyz GDF world max, w = GDF sample distance clamp
    float4 clay;            // 112 xyz neutral clay albedo, w = display gain
    uint out_index;         // 128 storage-image output
    uint width;             // 132
    uint height;            // 136
    uint flip_y;            // 140 bit0: 1 = Vulkan (clip-space Y down)
    uint clip_desc;         // 144 clipmap descriptor storage-buffer index
    uint clip_count;        // 148 clipmap level count
    uint wrc_atlas;         // 152 world radiance cache atlas (storage_buffers[] index)
    uint wrc_grid;          // 156 cache probes per side per level
    uint wrc_oct;           // 160 cache octahedral texels per side
    uint mode;              // 164 0 = irradiance (grayscale), 1 = irradiance x clay albedo
    uint source;            // 168 0 = world radiance cache (coarse 3D probes); 1 = surface cache
                            //     (high-res 2D mesh cards, final lit radiance) — the higher-res view
    uint sc_cards;          // 172 surface-cache card metadata buffer index (0xFFFFFFFF = unbound)
    uint sc_pos;            // 176 surface-cache card world-position buffer index
    uint sc_rad;            // 180 surface-cache lit radiance atlas index
    uint sc_num;            // 184 number of cards
    uint sc_tile;           // 188 card atlas tile edge
};

[[vk::push_constant]]
ConstantBuffer<WrcViewPush> pc : register(b0, space0);

// Skylight at hits (the cache stores bounce-only radiance, so this adds back the sky term the
// real pipeline supplies separately). The reference applies the skylight OCCLUDED by how much the
// surface sees the sky, plus a small "skylight leaking" floor so fully enclosed surfaces are never
// pure black. `SKY_STRENGTH` scales the sky term; `SKY_LEAK` is the minimum sky visibility floor.
static const float SKY_STRENGTH = 1.0;
static const float SKY_LEAK = 0.15;

// A dim neutral sky gradient (horizon -> zenith by the ray's up component). Painted on ray
// misses and used as the hemispheric skylight fill at hits, so the view never reads pure black.
float3 viz_sky(float3 dir)
{
    float up = saturate(dir.y * 0.5 + 0.5);
    float3 horizon = float3(0.34, 0.42, 0.55);
    float3 zenith = float3(0.12, 0.20, 0.42);
    return lerp(horizon, zenith, up);
}

// A minimal BounceScene carrying only the fields `bs_scene_march` / `bs_scene_normal` read
// (clipmap descriptor + count, sample clamp, ground plane). Every other field is unused by
// those two functions but is initialized to a neutral zero so the value is fully defined.
BounceScene view_scene()
{
    BounceScene bs;
    bs.clip_desc = pc.clip_desc;
    bs.clip_count = pc.clip_count;
    bs.sample_clamp = pc.aabb_max.w;
    bs.ground_y = pc.aabb_min.w;
    bs.sun_dir = float3(0.0, 0.0, 0.0);
    bs.sun_i = 0.0;
    bs.ray_max = 0.0;
    bs.bias = 0.0;
    bs.sky_term = 0.0;
    bs.albedo_fallback = 0.0;
    bs.cache = uint4(0u, 0u, 0u, 0u);
    bs.cache_tile = 0u;
    bs.ground_albedo = float3(0.0, 0.0, 0.0);
    bs.cone_k = 0.0;
    bs.max_steps = 0u;
    return bs;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    // Primary camera ray (identical reconstruction to gdf_trace.slang).
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 world = mul(pc.inv_view_proj, float4(ndc, 1.0, 1.0));
    float3 dir = normalize(world.xyz / world.w - pc.cam_pos.xyz);

    BounceScene bs = view_scene();

    // Sphere-trace the primary ray into the distance field.
    const int MAX_STEPS = 256;
    const float HIT_EPS = 0.001;
    const float MAX_T = 1e4;
    float t = 0.02;
    bool hit = false;
    float3 p = pc.cam_pos.xyz;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        p = pc.cam_pos.xyz + dir * t;
        float d = bs_scene_march(bs, p);
        if (d < HIT_EPS)
        {
            hit = true;
            break;
        }
        t += d;
        if (t > MAX_T)
            break;
    }

    if (!hit)
    {
        // Miss: paint the sky, not black — the reference visualization falls back to the sky for
        // rays that leave the scene (through windows / openings), so the background never reads as
        // a dead black hole.
        g.storage_images[pc.out_index][tid.xy] = float4(viz_sky(dir), 1.0);
        return;
    }

    float3 n = bs_scene_normal(bs, p);

    // --- Surface-cache source (mode 1): the higher-resolution view, matching the reference "scene"
    // visualization. The mesh-card surface cache stores FINAL lit radiance (direct + indirect +
    // skylight) at per-texel resolution (32^2 per card, following the geometry), far finer than the
    // coarse world radiance-cache probe grid. Sample the lit card at the hit; where no card covers
    // the point (e.g. the analytic ground), fall through to the world-cache path below. ---
    if (pc.source == 1u && pc.sc_cards != 0xFFFFFFFFu)
    {
        bool found = false;
        float3 sc = sample_surface_cache(p, n, pc.sc_cards, pc.sc_pos, pc.sc_rad, pc.sc_num,
                                         pc.sc_tile, 0.0, found);
        if (found)
        {
            // Card radiance is already final lit (incl. skylight) at high res — display directly.
            float3 c = sc * pc.clay.w;
            float lum = max(dot(c, float3(0.2126, 0.7152, 0.0722)), 1e-5);
            c *= (lum / (1.0 + lum)) / lum;
            float form = 0.7 + 0.3 * (n.y * 0.5 + 0.5);
            g.storage_images[pc.out_index][tid.xy] = float4(c * form, 1.0);
            return;
        }
    }

    Wrc w;
    w.atlas = pc.wrc_atlas;
    w.clip_desc = pc.clip_desc;
    w.levels = pc.clip_count;
    w.grid = pc.wrc_grid;
    w.oct = pc.wrc_oct;

    // Gather the cache's bounce irradiance AND the per-surface sky visibility. The world cache
    // stores BOUNCE-only radiance (sky-escaping rays contribute 0, since the real pipeline adds
    // sky/IBL separately), so a sky-lit surface would read pure black. Matching the reference, add
    // the skylight OCCLUDED by how much the surface actually sees the sky (`skyvis`), plus a small
    // leak floor (`SKY_LEAK`) so fully enclosed surfaces are dim, not dead black. Enclosed interior
    // stays dark (low skyvis); open surfaces get the sky — the honest, occluded skylight.
    float3 bounce_E;
    float skyvis;
    wrc_gather(w, p, n, bounce_E, skyvis);
    float sky_amount = max(skyvis, SKY_LEAK) * SKY_STRENGTH;
    float3 E = bounce_E + viz_sky(n) * sky_amount;
    float3 color = (pc.mode == 1u) ? (pc.clay.xyz * E) : E;
    color *= pc.clay.w; // display gain (sets the mid-point of the compression below)

    // The cache holds high-dynamic-range lighting (sun-lit surfaces are orders of magnitude
    // brighter than shadowed ones), so a single exposure crushes shadows to black and blows
    // highlights to pure white — the GI structure is lost. Compress the LUMINANCE with a
    // Reinhard curve (which asymptotes to 1, so nothing clips to pure white) while PRESERVING
    // chroma, so a bright neutral wall reads light-grey (not white) and the coloured indirect
    // bounces stay visible everywhere. The host applies its filmic curve with exposure 1 on top.
    float lum = max(dot(color, float3(0.2126, 0.7152, 0.0722)), 1e-5);
    float lt = lum / (1.0 + lum);
    color *= lt / lum; // scale to the compressed luminance, keep the colour ratio

    // A gentle form cue so the blobby distance-field surfaces read as 3D shapes, not flat
    // patches: a hemispheric wrap of the surface normal, kept subtle so it never hides the GI.
    float form = 0.7 + 0.3 * (n.y * 0.5 + 0.5);
    color *= form;

    g.storage_images[pc.out_index][tid.xy] = float4(color, 1.0);
}
