// Phase 11 Stage C5 — screen-space reflections.
//
// A full-screen compute pass: for each G-buffer pixel it reflects the view ray about
// the world normal and marches that reflection ray through the depth buffer in world
// space (projecting each sample to screen to compare against the stored depth). On a
// hit it samples the shaded HDR color at the hit pixel, so reflective surfaces show the
// *actual neighbouring geometry* — the avocado/cube reflected in the chrome sphere —
// which the captured-cube IBL can't (single probe, parallax + neighbour-reflection
// error). Rays that leave the screen or find no surface miss (confidence 0); Stage C6
// (GDF reflection) fills those, and C7 composites the two over the IBL specular.
//
// Color source — two modes (flip_y bit1):
//   * Current-HDR (bit1=0, the standalone C5 viz): runs after lighting and samples this
//     frame's HDR at the hit (exposure already baked, like the raster path); the tonemap
//     applies exposure 1.0.
//   * Reprojected history (bit1=1, C7b, feeds lighting): samples the PREVIOUS frame's
//     raw-radiance lit-color history buffer at the reprojected world hit point, so SSR can
//     feed back into the lighting specular (C7c) without a read-before-write cycle. The
//     reprojection uses `globals.prev_view_proj`; the result is raw radiance.
//
// Output: float4(reflected color, confidence) into a storage image.

#include "bindless.slang"

struct SsrPush
{
    float4x4 view_proj;     // world -> clip (project ray samples to screen)
    float4x4 inv_view_proj; // clip -> world (surface + scene reconstruction)
    float4 cam_pos;         // xyz camera world position
    uint depth_index;       // sampled G-buffer depth (full-res)
    uint normal_index;      // sampled G-buffer world normal
    uint material_index;    // sampled G-buffer material (g = roughness) for the GGX jitter
    uint hist_index;        // history mode: storage buffer (raw-radiance prev-frame color)
    uint color_index;       // current-HDR mode: sampled shaded HDR color (this frame)
    uint out_index;         // storage image (reflected color + confidence)
    uint width;             // OUTPUT dims (half-res for the stochastic trace)
    uint height;
    uint flip_y;            // bit0 = Vulkan Y; bit1 = history; bit2 = nbhd-clamp; bit3 = stochastic
    uint frame;             // RNG decorrelation for the stochastic jitter
    uint full_width;        // full-res dims (the DDA marches in full-res screen pixels)
    uint full_height;
    float4 params;          // x = max ray distance, y = thickness, z = step cap, w = edge fade
    uint out_b_index;       // 2nd output: reflection ray dir.xyz + GGX pdf.w (for the resolve)
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]]
ConstantBuffer<SsrPush> pc : register(b0, space0);

// Per-frame globals (set 1 / b1) — the same UBO the deferred PBR pass binds, opted into
// via the compute pipeline's `uniform_buffer`. Only `prev_view_proj` (appended at the end)
// is read here, for the history reprojection; the layout must match the host `Globals`.
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

float3 reconstruct(float2 uv, float depth)
{
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 h = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    return h.xyz / h.w;
}

// --- RNG + GGX importance sampling for the stochastic reflection ray. One jittered ray
// per (half-res) pixel + frame; the temporal resolve pass accumulates them into a glossy
// reflection. roughness -> 0 reduces to the mirror ray. ---
static const float PI = 3.14159265359;
uint pcg(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}
float rnd(inout uint s) { return float(pcg(s) & 0xFFFFFFu) / float(0x1000000); }
void onb(float3 n, out float3 t, out float3 b)
{
    t = normalize(abs(n.x) > 0.9 ? cross(n, float3(0, 1, 0)) : cross(n, float3(1, 0, 0)));
    b = cross(n, t);
}
// GGX NDF importance sample: a microfacet half-vector around the surface normal.
float3 sample_ggx(float3 n, float roughness, inout uint rng)
{
    float a = max(roughness * roughness, 1e-3);
    float u1 = rnd(rng), u2 = rnd(rng);
    float phi = 2.0 * PI * u1;
    float ct = sqrt((1.0 - u2) / (1.0 + (a * a - 1.0) * u2));
    float st = sqrt(saturate(1.0 - ct * ct));
    float3 t, bvec;
    onb(n, t, bvec);
    return normalize(t * (st * cos(phi)) + bvec * (st * sin(phi)) + n * ct);
}
// GGX NDF (alpha = roughness²) — used to weigh borrowed neighbour samples in the resolve.
float ggx_d(float ndh, float alpha)
{
    float a2 = alpha * alpha;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-9);
}
// pdf of the reflected direction under GGX NDF importance sampling.
float ggx_reflect_pdf(float3 n, float3 v, float3 l, float roughness)
{
    float3 h = normalize(v + l);
    float ndh = max(dot(n, h), 1e-4);
    float vdh = max(dot(v, h), 1e-4);
    float alpha = max(roughness, 0.04);
    alpha *= alpha;
    return ggx_d(ndh, alpha) * ndh / (4.0 * vdh);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;

    // Background: no surface to reflect from.
    if (depth >= 1.0)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
        g.storage_images[pc.out_b_index][tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float3 p = reconstruct(uv, depth);
    float3 n = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);
    float3 v = normalize(pc.cam_pos.xyz - p);

    // Stochastic GGX jitter (bit3): importance-sample a microfacet half-vector by the
    // surface roughness and reflect about it, so glossy surfaces get blurred reflections
    // (one ray/frame; the temporal resolve accumulates them). Mirror ray when off / smooth.
    float3 dir;
    float pdf = 1.0;
    if ((pc.flip_y & 8u) != 0u)
    {
        float roughness = max(g.textures[pc.material_index].SampleLevel(g.samp, uv, 0.0).g, 0.04);
        uint rng = (tid.x * 1973u + tid.y * 9277u + pc.frame * 26699u) | 1u;
        float3 h = sample_ggx(n, roughness, rng);
        dir = reflect(-v, h);
        if (dot(dir, n) <= 0.0)
            dir = reflect(-v, n); // keep the ray in the upper hemisphere
        // pdf of the sampled reflected direction (stored so the ratio-estimator resolve can
        // reweight this ray when neighbouring pixels borrow it).
        pdf = max(ggx_reflect_pdf(n, v, dir, roughness), 1e-4);
    }
    else
    {
        dir = reflect(-v, n);
    }

    float max_dist = pc.params.x;
    float thickness = pc.params.y;
    int max_steps = int(pc.params.z);
    float2 res = float2(pc.full_width, pc.full_height);
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;

    // Screen-space DDA march: step ~1 pixel at a time along the projected reflection ray,
    // comparing the ray's NDC depth against the depth buffer. World-uniform steps gave
    // non-uniform screen coverage, so the ray could skip over a thin silhouette (hit) where
    // a neighbour didn't (miss) -> isolated speckles. Stepping in screen space tests every
    // pixel the ray crosses, so silhouettes stay consistent. NDC z is linear in screen space
    // (like the rasterizer's interpolated depth), so `lerp(Z0, Z1, s)` is exact.
    float3 p0 = p + dir * (max_dist * 0.002);
    float3 p1 = p + dir * max_dist;
    float4 H0 = mul(pc.view_proj, float4(p0, 1.0));
    float4 H1 = mul(pc.view_proj, float4(p1, 1.0));
    // Clip the endpoint to in front of the near plane (clip.w is linear in the world param).
    const float W_EPS = 1e-4;
    if (H1.w < W_EPS)
    {
        float tw = (W_EPS - H0.w) / (H1.w - H0.w);
        p1 = lerp(p0, p1, tw);
        H1 = mul(pc.view_proj, float4(p1, 1.0));
    }
    float2 S0 = float2(H0.x / H0.w * 0.5 + 0.5, (H0.y / H0.w) * sy * 0.5 + 0.5);
    float2 S1 = float2(H1.x / H1.w * 0.5 + 0.5, (H1.y / H1.w) * sy * 0.5 + 0.5);
    float Z0 = H0.z / H0.w;
    float Z1 = H1.z / H1.w;

    float2 dpx = (S1 - S0) * res;
    int nsteps = int(clamp(max(abs(dpx.x), abs(dpx.y)), 1.0, float(max_steps)));
    float inv_n = 1.0 / float(nsteps);

    bool hit = false;
    float2 hit_uv = uv;
    float3 world_hit = p;
    float s_prev = 0.0;
    float s_hit = 0.0;
    for (int i = 1; i <= nsteps; i++)
    {
        float s = float(i) * inv_n;
        float2 ruv = lerp(S0, S1, s);
        if (any(ruv < 0.0) || any(ruv > 1.0))
            break; // left the screen
        float ray_z = lerp(Z0, Z1, s);
        float scene_z = g.textures[pc.depth_index].SampleLevel(g.samp, ruv, 0.0).r;
        if (scene_z < 1.0 && ray_z > scene_z)
        {
            // Ray passed behind the stored surface. Accept only if the surface is within
            // `thickness` of the ray point (else it's a far background gap, not a hit).
            float3 ray_p = reconstruct(ruv, ray_z);
            float3 scene_p = reconstruct(ruv, scene_z);
            if (distance(ray_p, scene_p) < thickness)
            {
                hit = true;
                s_hit = s;
                break;
            }
        }
        s_prev = s;
    }

    if (!hit)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
        g.storage_images[pc.out_b_index][tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Binary-refine the crossing between the last miss and the hit (in screen-s).
    float lo = s_prev;
    float hi = s_hit;
    for (int b = 0; b < 6; b++)
    {
        float mid = 0.5 * (lo + hi);
        float2 ruv = lerp(S0, S1, mid);
        float ray_z = lerp(Z0, Z1, mid);
        float scene_z = g.textures[pc.depth_index].SampleLevel(g.samp, ruv, 0.0).r;
        if (scene_z < 1.0 && ray_z > scene_z)
        {
            hi = mid;
            hit_uv = ruv;
        }
        else
        {
            lo = mid;
        }
    }
    world_hit = reconstruct(hit_uv, lerp(Z0, Z1, hi));

    float3 color;
    if ((pc.flip_y & 2u) != 0u)
    {
        // History mode: reproject the world hit into the previous frame and read the FULL-res
        // raw-radiance lit-color history (the trace runs at half-res, so index it with the
        // full-res dims). The ratio-estimator resolve does the firefly / neighbourhood clamp.
        float4 pclip = mul(globals.prev_view_proj, float4(world_hit, 1.0));
        float3 pndc = pclip.xyz / max(pclip.w, 1e-6);
        float2 puv = float2(pndc.x * 0.5 + 0.5, pndc.y * sy * 0.5 + 0.5);
        if (pclip.w <= 0.0 || any(puv < 0.0) || any(puv > 1.0))
        {
            // Off-screen last frame: no history -> miss (pdf 0), let the GDF fallback fill it.
            g.storage_images[pc.out_index][tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
            g.storage_images[pc.out_b_index][tid.xy] = float4(0.0, 0.0, 0.0, 0.0);
            return;
        }
        int2 ppix = clamp(int2(floor(puv * float2(pc.full_width, pc.full_height))), int2(0, 0),
                          int2(int(pc.full_width) - 1, int(pc.full_height) - 1));
        uint pidx = uint(ppix.y) * pc.full_width + uint(ppix.x);
        color = asfloat(g.storage_buffers[pc.hist_index].Load4(pidx * 16u)).rgb;
    }
    else
    {
        color = g.textures[pc.color_index].SampleLevel(g.samp, hit_uv, 0.0).rgb;
    }
    // Confidence: fade toward the screen edges (the march cuts off there) so C7 can
    // blend in the GDF / sky fallback without a hard seam.
    float2 e = abs(hit_uv * 2.0 - 1.0);
    float edge = saturate((1.0 - max(e.x, e.y)) / max(pc.params.w, 1e-3));
    g.storage_images[pc.out_index][tid.xy] = float4(color, edge);
    g.storage_images[pc.out_b_index][tid.xy] = float4(dir, pdf); // dir + GGX pdf (>0 = hit)
}
