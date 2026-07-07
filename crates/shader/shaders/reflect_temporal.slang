// Phase 11 Stage C8j — stochastic GDF reflection temporal resolve.
//
// `gdf_reflect.slang` now casts one GGX-importance-sampled ray per pixel per frame (레퍼런스 엔진 SW GI's
// approach to rough reflections — sample the real lobe, then denoise — instead of an image-space
// prefilter that can't average in the off-screen sky a rough lobe sees). This pass accumulates
// those noisy single-ray samples over frames: reproject the surface into the previous frame, read
// the accumulated reflection, validate it (disocclusion reject), and EMA-blend the new sample in.
// On the static screenshot the warm-up converges to the lobe average (a muted glossy reflection
// like the path tracer); on a moving camera the reprojection keeps it stable (SSR-history style).
//
// Radiance is accumulated in a tonemap-weighted space (Karis / 레퍼런스 엔진 `RGBToDenoiserSpace`) so a
// bright sample can't dominate the average — the principled version of a per-tap clamp.

#include "bindless.slang"

struct ReflTemporalPush
{
    float4x4 inv_view_proj;  // current clip -> world (surface recon)
    float4x4 prev_view_proj; // world -> previous clip (reprojection)
    float4 cam_pos;          // xyz camera world position
    uint refl_index;         // sampled: this frame's stochastic GDF reflection (rgb)
    uint depth_index;        // sampled: full-res G-buffer depth
    uint out_index;          // storage image: resolved reflection
    uint accum_read;         // storage buffer: previous accumulation (rgb + frame count)
    uint accum_write;
    uint pos_read;           // storage buffer: previous surface world pos (xyz + valid)
    uint pos_write;
    uint width;
    uint height;
    uint flip_y;             // bit0 = Vulkan Y
    uint reset;              // 1 = ignore history (first frame / view change)
    uint material_index;     // sampled G-buffer material (g = roughness) for the spatial radius
    float4 params;           // x = reject dist, y = max history len, z = firefly clamp, w = tonemap range
    uint clamp_mode;         // history neighbourhood clamp: 0 = off, 1 = hard AABB, 2 = variance (mean +- g*sigma)
    float clamp_gamma;       // variance-clamp tightness (mode 2); ignored otherwise
    uint spatial_off;        // A1: 1 = the ratio-estimator resolve already ran → skip the box average
    // A4a: per-pixel 2nd-moment (luminance²) accumulation → variance for the A4b spatial denoiser.
    // `denoise == 0` (gallery / legacy) skips all M2 work and leaves `out.a = 1.0` = byte-identical.
    uint moment_read;        // storage buffer: previous accumulated M2 (x = luma², reprojected)
    uint moment_write;
    uint denoise;            // 1 = accumulate M2 + output StdDev in out.a for the spatial pass
};

[[vk::push_constant]]
ConstantBuffer<ReflTemporalPush> pc : register(b0, space0);

float3 reconstruct(float2 uv, float depth)
{
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 h = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    return h.xyz / h.w;
}

// Fetch reprojected history (accumulated tonemap-space rgb + length). bit1 of flip_y selects a
// sub-pixel-accurate bilinear fetch — required when the camera is sub-pixel jittered (TAAU), so the
// reflection accumulation tracks the jittered reprojection instead of snapping to integer texels
// (which blurs the resolved reflection). Default (bit1 clear) keeps the integer-floor fetch so the
// non-jittered path stays byte-identical. Per-tap disocclusion via the stored world point.
void fetch_refl_history(float2 puv, float3 world_pos, out float3 out_rgb, out float out_len)
{
    out_rgb = float3(0, 0, 0);
    out_len = 0.0;
    float2 res = float2(pc.width, pc.height);
    if ((pc.flip_y & 2u) == 0u)
    {
        int2 ppix = clamp(int2(floor(puv * res)), int2(0, 0),
                          int2(int(pc.width) - 1, int(pc.height) - 1));
        uint pidx = uint(ppix.y) * pc.width + uint(ppix.x);
        float4 pp = asfloat(g.storage_buffers[pc.pos_read].Load4(pidx * 16u));
        if (pp.w > 0.5 && distance(world_pos, pp.xyz) < pc.params.x)
        {
            float4 ph = asfloat(g.storage_buffers[pc.accum_read].Load4(pidx * 16u));
            out_rgb = ph.rgb;
            out_len = ph.a;
        }
        return;
    }
    float2 sp = puv * res - 0.5;
    float2 base = floor(sp);
    float2 f = sp - base;
    float w[4] = { (1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y };
    int2 off[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
    float3 rgb_sum = float3(0, 0, 0);
    float len_sum = 0.0;
    float wsum = 0.0;
    [unroll] for (int i = 0; i < 4; i++)
    {
        int2 p = int2(base) + off[i];
        if (p.x < 0 || p.y < 0 || p.x >= int(pc.width) || p.y >= int(pc.height))
            continue;
        uint pidx = uint(p.y) * pc.width + uint(p.x);
        float4 pp = asfloat(g.storage_buffers[pc.pos_read].Load4(pidx * 16u));
        if (pp.w > 0.5 && distance(world_pos, pp.xyz) < pc.params.x)
        {
            float4 ph = asfloat(g.storage_buffers[pc.accum_read].Load4(pidx * 16u));
            rgb_sum += w[i] * ph.rgb;
            len_sum += w[i] * ph.a;
            wsum += w[i];
        }
    }
    if (wsum > 0.5)
    {
        out_rgb = rgb_sum / wsum;
        out_len = len_sum / wsum;
    }
}
// Tonemap-weighted space for stable HDR averaging (레퍼런스 엔진 RGBToDenoiserSpace).
float3 to_tm(float3 v) { return v / (1.0 + dot(v, float3(0.2126, 0.7152, 0.0722)) * pc.params.w); }
float3 from_tm(float3 v)
{
    float l = dot(v, float3(0.2126, 0.7152, 0.0722));
    return v / max(1.0 - l * pc.params.w, 1e-3);
}
float luma(float3 v) { return dot(v, float3(0.2126, 0.7152, 0.0722)); }

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    uint idx = tid.y * pc.width + tid.x;
    float2 res = float2(pc.width, pc.height);
    float2 uv = (float2(tid.xy) + 0.5) / res;
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;

    if (depth >= 1.0)
    {
        // Background: copy the reflection straight through (the sky along the view ray).
        float3 bg = g.textures[pc.refl_index].SampleLevel(g.samp, uv, 0.0).rgb;
        g.storage_images[pc.out_index][tid.xy] = float4(bg, 1.0);
        g.storage_buffers[pc.accum_write].Store4(idx * 16u, asuint(float4(to_tm(bg), 1.0)));
        g.storage_buffers[pc.pos_write].Store4(idx * 16u, asuint(float4(0, 0, 0, 0)));
        // A4a: reset the M2 accumulator on background so a later disocclusion doesn't read stale variance.
        if (pc.denoise != 0u)
            g.storage_buffers[pc.moment_write].Store4(idx * 16u, asuint(float4(0, 0, 0, 0)));
        return;
    }

    float3 world_pos = reconstruct(uv, depth);
    // Spatial reconstruction: average the current single-ray GGX samples over a roughness-scaled,
    // depth-weighted neighbourhood (each neighbour pixel traced a DIFFERENT GGX ray) to cut the
    // per-pixel variance before temporal accumulation — 레퍼런스 엔진 resolves spatially + temporally. Sharp
    // surfaces (chrome) use radius 0 (no blur).
    //
    // The SAME neighbourhood also builds the statistics for a TAA-style neighbourhood clamp of the
    // reprojected history (below). A specular reflection is view-dependent, so on a moving/rotating
    // camera the surface-reprojected history is STALE — without a clamp it drags the previous
    // direction's radiance (dark off-screen sky / shadowed geometry) across the chrome as a black
    // smear (the "no history" trails). Clamping the history into the current frame's local reflection
    // range kills that lag while keeping the static accumulation (history already in range -> a
    // no-op, so the converged screenshot/parity baseline is preserved). 레퍼런스 엔진 SW GI / 다른 레퍼런스 엔진
    // reflections clamp the same way.
    //
    // Two permutations (`pc.clamp_mode`), so the strength scales with the RenderQuality tier:
    //   1 = HARD: clamp to the tonemap-space colour AABB [cmin, cmax]. Cheapest, but on a sharp
    //       mirror (high spatial frequency) a 3x3 box can be too tight -> can clip valid history.
    //   2 = VARIANCE: clamp to mean +- gamma*sigma (intersected with the AABB) — Salvi/Karis variance
    //       clipping. Gentler on high-frequency mirrors; `clamp_gamma` tunes the tightness.
    //   0 = OFF: no clamp (byte-identical to the pre-clamp resolve; the spatial average below is
    //       computed over the same `sr` set regardless, so OFF reproduces the legacy output exactly).
    // The neighbourhood is always >= 3x3 (`cr`) so even a sharp mirror (spatial radius 0) has stats.
    float roughness = g.textures[pc.material_index].SampleLevel(g.samp, uv, 0.0).g;
    int sr = clamp(int(roughness * 8.0), 0, 3);
    // A1: when the spatial ratio-estimator resolve already denoised the trace, skip this box average
    // (it would re-blur the sharpened glossy lobe) — but keep the >= 3x3 stats radius `cr` for the
    // history neighbourhood clamp. `spatial_off == 0` (gallery / legacy) keeps the box = byte-identical.
    if (pc.spatial_off != 0u)
        sr = 0;
    int cr = max(sr, 1);
    float2 texel = 1.0 / res;
    float3 center = g.textures[pc.refl_index].SampleLevel(g.samp, uv, 0.0).rgb;
    float3 ssum = center; // linear average (radius sr) — to_tm'd at the end, unchanged from before
    float swsum = 1.0;
    float3 center_tm = to_tm(center);
    float3 cmin = center_tm; // tonemap-space colour AABB (radius cr) for the hard clamp
    float3 cmax = center_tm;
    float3 m1 = center_tm;             // 1st/2nd moments (tonemap space) for the variance clamp
    float3 m2 = center_tm * center_tm;
    float mcnt = 1.0;
    for (int dy = -cr; dy <= cr; dy++)
    {
        for (int dx = -cr; dx <= cr; dx++)
        {
            if (dx == 0 && dy == 0)
                continue;
            float2 nuv = uv + float2(dx, dy) * texel;
            float dz = g.textures[pc.depth_index].SampleLevel(g.samp, nuv, 0.0).r;
            if (dz >= 1.0 || abs(dz - depth) > 0.003)
                continue; // different surface — would bleed across the silhouette
            float3 nrefl = g.textures[pc.refl_index].SampleLevel(g.samp, nuv, 0.0).rgb;
            float3 ntm = to_tm(nrefl);
            cmin = min(cmin, ntm);
            cmax = max(cmax, ntm);
            m1 += ntm;
            m2 += ntm * ntm;
            mcnt += 1.0;
            if (abs(dx) <= sr && abs(dy) <= sr)
            {
                ssum += nrefl; // average only over the (narrower) spatial radius — value unchanged
                swsum += 1.0;
            }
        }
    }
    float3 sample_tm = to_tm(ssum / swsum);

    // Resolve the clamp box once (per the permutation). OFF leaves it as the full AABB but the clamp
    // is skipped entirely below, so OFF stays byte-identical to the legacy resolve.
    float3 clamp_lo = cmin;
    float3 clamp_hi = cmax;
    if (pc.clamp_mode == 2u)
    {
        float3 mean = m1 / mcnt;
        float3 sigma = sqrt(max(m2 / mcnt - mean * mean, float3(0.0, 0.0, 0.0)));
        // mean +- gamma*sigma, intersected with the true neighbourhood AABB so it never exceeds it.
        clamp_lo = max(cmin, mean - pc.clamp_gamma * sigma);
        clamp_hi = min(cmax, mean + pc.clamp_gamma * sigma);
    }

    // VIRTUAL-IMAGE reprojection (content, flip_y bit2 — reference-engine LumenReflectionDenoiser-
    // Temporal): a mirror's reflected content appears at the HIT distance BEHIND the surface, so
    // reprojecting the SURFACE point into the previous frame (the legacy path) fetches the wrong
    // history under camera motion and the reflection SWIMS / never converges. Instead reproject the
    // reflected point itself — the point `surface_dist + hit_dist` along the view ray (the virtual
    // image). `hit_dist` is the reflection ray length, carried in the reflection's `.w` (gdf_reflect).
    // Gallery / legacy (bit2 clear) reprojects the surface = byte-identical anchor.
    float hit_dist = g.textures[pc.refl_index].SampleLevel(g.samp, uv, 0.0).a;
    float3 to_surf = world_pos - pc.cam_pos.xyz;
    float surf_dist = length(to_surf);
    float3 virt_pos =
        pc.cam_pos.xyz + (to_surf / max(surf_dist, 1e-4)) * (surf_dist + max(hit_dist, 0.0));
    // Only near-mirrors (sr == 0) use the virtual image; a glossy lobe's hit distance is diffuse,
    // so those keep the surface reproject (unchanged). Content flag = flip_y bit2.
    float3 reproj_pos = (((pc.flip_y & 4u) != 0u) && sr == 0) ? virt_pos : world_pos;

    float3 accum = sample_tm;
    float len = 1.0;
    // A4a: 2nd moment of the tonemap-space luminance (matches `accum`'s space) — EMA'd with the same
    // alpha as radiance so `Var = M2 - Luma(accum)²` is the per-pixel temporal variance the A4b spatial
    // denoiser uses to weight its luminance edge stop (reference-engine SpecularAndSecondMoment).
    float cur_m2 = luma(sample_tm) * luma(sample_tm);
    float moment2 = cur_m2;
    if (pc.reset == 0u)
    {
        float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
        float4 pclip = mul(pc.prev_view_proj, float4(reproj_pos, 1.0));
        if (pclip.w > 0.0)
        {
            float3 pndc = pclip.xyz / pclip.w;
            float2 puv = float2(pndc.x * 0.5 + 0.5, pndc.y * sy * 0.5 + 0.5);
            if (all(puv >= 0.0) && all(puv <= 1.0))
            {
                float3 hist_rgb;
                float hist_len;
                fetch_refl_history(puv, reproj_pos, hist_rgb, hist_len);
                if (hist_len > 0.5)
                {
                    // Neighbourhood-clamp the (view-dependent) stale history into this frame's local
                    // reflection range before blending — removes the dark drag on a rotating camera.
                    // OFF (mode 0) skips it -> byte-identical legacy resolve.
                    //
                    // ROUGHNESS GATE: only clamp glossy/rough reflections (sr > 0, i.e. roughness >=
                    // ~0.125), where a 3x3 neighbourhood is a valid radiance bound. A near-mirror
                    // (sr == 0) carries high-frequency detail that the tight box would clip into a
                    // hard dark band on the silhouette (measured on the gallery chrome sphere); mirrors
                    // are kept stable by the per-frame re-trace + EMA instead.
                    if (pc.clamp_mode != 0u && sr > 0)
                        hist_rgb = clamp(hist_rgb, clamp_lo, clamp_hi);
                    len = min(hist_len + 1.0, pc.params.y);
                    accum = lerp(hist_rgb, sample_tm, 1.0 / len);
                    // A4a: EMA the 2nd moment with the same alpha (integer-floor tap on the reprojected
                    // history — the accumulated M2 is low-frequency, so a nearest fetch is sufficient).
                    if (pc.denoise != 0u)
                    {
                        int2 pp = clamp(int2(floor(puv * res)), int2(0, 0),
                                        int2(int(pc.width) - 1, int(pc.height) - 1));
                        uint pidx = uint(pp.y) * pc.width + uint(pp.x);
                        float hist_m2 = asfloat(g.storage_buffers[pc.moment_read].Load4(pidx * 16u)).x;
                        moment2 = lerp(hist_m2, cur_m2, 1.0 / len);
                    }
                }
            }
        }
    }

    // Sanitize before storing: a NaN would persist in the accumulator and spread via EMA.
    accum = make_finite(accum);
    g.storage_buffers[pc.accum_write].Store4(idx * 16u, asuint(float4(accum, len)));
    g.storage_buffers[pc.pos_write].Store4(idx * 16u, asuint(float4(reproj_pos, 1.0)));

    float3 resolved = from_tm(accum);
    // Hue-preserving firefly clamp on the resolved radiance.
    float m = max(resolved.r, max(resolved.g, resolved.b));
    if (m > pc.params.z)
        resolved *= pc.params.z / m;
    // A4a: emit per-pixel StdDev (tonemap-space luminance) in `.a` for the A4b spatial denoiser; the
    // variance is M2 - mean². `denoise == 0` keeps `.a = 1.0` (byte-identical legacy output).
    float out_a = 1.0;
    if (pc.denoise != 0u)
    {
        moment2 = make_finite(float3(moment2, 0, 0)).x;
        g.storage_buffers[pc.moment_write].Store4(idx * 16u, asuint(float4(moment2, 0, 0, 0)));
        float mean = luma(accum);
        out_a = sqrt(max(moment2 - mean * mean, 0.0));
    }
    g.storage_images[pc.out_index][tid.xy] = float4(resolved, out_a);
}
