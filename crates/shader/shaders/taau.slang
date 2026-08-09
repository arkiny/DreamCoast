// QHD/UHD track — Temporal Anti-aliasing Upsampling (TAAU) with robust history rectification.
//
// The scene renders at a reduced internal resolution with a per-frame sub-pixel camera jitter;
// this pass reconstructs a full-res HDR by reprojecting + accumulating the jittered frames. The
// two failure modes are (1) visible jitter when history isn't trusted (no accumulation) and
// (2) ghosting when the camera moves (stale history bleeds through). The standard, platform-
// independent toolbox is applied here:
//   • YCoCg variance clipping (Karis 2014 / Salvi 2016): clip the reprojected history to the
//     current neighborhood's statistical color box [μ ± γσ] in YCoCg (ray-AABB clip toward the
//     mean) — the primary anti-ghost, and it also stabilizes (trusted history => no jitter).
//   • Closest-depth dilation (Playdead/레퍼런스 엔진): reproject using the NEAREST depth in the 3x3 so
//     silhouettes fetch the foreground's history (kills edge ghosting on moving edges).
//   • Luminance-weighted feedback (Lottes): bias toward more history where luma is stable
//     (hides jitter) and toward the current sample where it changes fast (drops ghosts).
// History: ping-pong full-res buffers — `hist` (rgb + length) and `pos` (dynamic flag + valid;
// see the dynamic anti-ghost below). `reset` ignores history (first frame / resize).
//   • Dynamic anti-ghost (reference engine TemporalAA.usf AA_DYNAMIC_ANTIGHOST): each history
//     texel remembers whether it was accumulated on a DYNAMIC object; history that was, fetched
//     by a pixel that no dynamic object covers anymore, is the trail a mover left behind —
//     rejected outright (provenance, not color, so the wide near-silhouette variance box can't
//     save it). Active only when the velocity target is bound.

#include "bindless.slang"

struct TaauPush
{
    float4x4 inv_view_proj;  // current JITTERED clip -> world (reconstruct from the jittered depth)
    float4x4 prev_view_proj; // world -> previous UNJITTERED clip (history is on the stable grid)
    uint hdr_index;          // sampled: current internal-res HDR (the jittered low-res frame)
    uint depth_index;        // sampled: current internal-res depth
    uint out_index;          // storage image: full-res accumulated HDR
    uint hist_read;          // storage buffer: previous accumulation (rgb + length)
    uint hist_write;         // storage buffer: this frame's accumulation
    uint pos_read;           // storage buffer: previous world position (xyz + valid)
    uint pos_write;          // storage buffer: this frame's world position
    uint out_width;          // full (output) resolution
    uint out_height;
    uint in_width;           // internal (render) resolution — neighborhood texel spacing
    uint in_height;
    uint flip_y;             // bit0: 1 = Vulkan (clip-space Y down)
    uint reset;              // 1 = ignore history
    float4 params;           // x = reject distance, y = max history, z = variance gamma, w unused
    float4 jitter;           // xy = current frame's jitter in UV (add to the stable UV to sample)
                             // z = motion ramp in output px (0 = motion policy off), w = motion history cap
    uint velocity_index;     // sampled: RG16Float per-pixel motion vector, or 0xFFFFFFFF = absent
    uint packed_hist;        // 1 = fp16-packed history, no pos buffer (see the packed helpers)
    uint taau_flags;         // bit0: 1 = Catmull-Rom history resample (else bilinear)
    uint pad2;               // 16-byte align the tail
};

// Motion-sharpness policy bit (see `fetch_history_catmull_rom` / the motion gate in csMain).
static const uint TAAU_FLAG_CATMULL_ROM = 1u;

[[vk::push_constant]]
ConstantBuffer<TaauPush> pc : register(b0, space0);

float3 rgb_to_ycocg(float3 c)
{
    return float3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * (c.r - c.b),
                  -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
float3 ycocg_to_rgb(float3 c)
{
    float t = c.x - c.z;
    return float3(t + c.y, c.x + c.z, t - c.y);
}
// Clip the history sample `q` to the AABB [amin, amax] along the ray toward the box center
// (Karis): a true clip (not a clamp) so off-axis history snaps to the nearest box face.
float3 clip_aabb(float3 amin, float3 amax, float3 q)
{
    float3 center = 0.5 * (amax + amin);
    float3 extent = 0.5 * (amax - amin) + 1e-5;
    float3 v = q - center;
    float3 a = abs(v / extent);
    float m = max(a.x, max(a.y, a.z));
    return m > 1.0 ? center + v / m : q;
}
float luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// ---- history dynamic flag --------------------------------------------------------------------
// Alongside rgb+len, every history texel carries one bit: "this accumulation currently sits on
// a DYNAMIC object" (the reference engine stores the same bit in its TAA history alpha —
// TemporalAA.usf `OutputPayload.Color.a = Dynamic4 ? 1 : 0`). The unpacked layout keeps it in
// the pos buffer's otherwise-dead `.x` (only `.w` was ever consumed); the packed layout folds
// it into the length's fraction (+0.25 — lengths are integers 1..64, exactly representable in
// fp16 with the fraction intact, and validity checks (`> 0.5`) are unaffected). Strip helpers:
float packed_len_flag(float stored, out float dyn)
{
    float fr = frac(stored);
    dyn = fr >= 0.125 ? 1.0 : 0.0;
    return stored - 0.25 * dyn;
}
float packed_len_store(float len, float dyn)
{
    return len + (dyn > 0.05 ? 0.25 : 0.0);
}

// ---- fp16-packed history (packed_hist == 1) -------------------------------------------------
// The history I/O dominates the pass at Retina-class output (hist 16B + pos 16B, ping-pong =
// read+write ≈ 64B/px/frame). Two observations shrink it 4x with no algorithmic change:
//   • The pos buffer's consumed fields are `.w` (the "real prior surface" flag) and `.x` (the
//     dynamic anti-ghost flag) — the world point it originally held is never read (the YCoCg
//     clip replaced hard geometric disocclusion, see fetch_history_bilinear). A separate
//     16B/px buffer for two flags is pure waste: fold both into the history length (len == 0 =
//     sky/no-surface, len >= 1 = valid, fraction +0.25 = dynamic — the accumulator only ever
//     writes integer lengths, exactly representable in fp16 up to the 64 cap).
//   • The accumulated rgb comes from an RGBA16Float HDR chain, so an fp16 history holds the
//     full input precision (the reference engine's TAA history is fp16 for the same reason).
// Layout: uint2 per pixel = f16(r)|f16(g)<<16, f16(b)|f16(len)<<16. The unpacked path (and its
// buffers) is untouched — the packed layout is a strict opt-in (quality-tier knob).
float4 hist_load_packed(uint buf, uint idx)
{
    uint2 d = g.storage_buffers[buf].Load2(idx * 8u);
    return float4(f16tof32(d.x & 0xFFFFu), f16tof32(d.x >> 16), f16tof32(d.y & 0xFFFFu),
                  f16tof32(d.y >> 16));
}
void hist_store_packed(uint buf, uint idx, float4 v)
{
    // Clamp to the fp16 finite range: f32tof16 of a larger float lands on +inf, which would
    // poison every later EMA blend through this texel.
    v = min(v, 65504.0);
    uint2 d = uint2(f32tof16(v.x) | (f32tof16(v.y) << 16), f32tof16(v.z) | (f32tof16(v.w) << 16));
    g.storage_buffers[buf].Store2(idx * 8u, d);
}

// Bilinear (sub-pixel) history fetch from the ping-pong storage buffers. The history lives in a
// raw buffer (no hardware sampler), so do the 2x2 tap + weighting by hand. CRITICAL for jitter:
// integer-texel (floor) reprojection discards the sub-pixel offset, so the jittered low-res frames
// land incoherently and the jitter degrades into shimmer; a sub-pixel-accurate fetch instead lets
// the jitter act as a stable super-sampling signal. Each tap is validated against the disocclusion
// test (stored world point) and dropped if it was a different surface; surviving taps are
// re-normalized so a silhouette doesn't pull in background history. Returns rgb+len in `out_rgb` /
// `out_len`, with `out_len <= 0` meaning "no usable history".
void fetch_history_bilinear(float2 puv, float3 world_pos, out float3 out_rgb, out float out_len,
                            out float out_dyn)
{
    out_rgb = float3(0, 0, 0);
    out_len = 0.0;
    out_dyn = 0.0;
    float2 res = float2(pc.out_width, pc.out_height);
    float2 sp = puv * res - 0.5; // texel-center convention so an exact reprojection is a no-op
    float2 base = floor(sp);
    float2 f = sp - base;
    float w[4] = { (1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y };
    int2 off[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };

    float3 rgb_sum = float3(0, 0, 0);
    float len_sum = 0.0;
    float dyn_sum = 0.0;
    float wsum = 0.0;
    [unroll] for (int i = 0; i < 4; i++)
    {
        int2 p = int2(base) + off[i];
        if (p.x < 0 || p.y < 0 || p.x >= int(pc.out_width) || p.y >= int(pc.out_height))
            continue;
        uint pidx = uint(p.y) * pc.out_width + uint(p.x);
        // Accept any tap that holds a real prior surface. NO hard geometric disocclusion:
        // a hard world-point reject left silhouette + grazing bands stuck at len 1 (full jitter
        // crawl), because under jitter those pixels flip sub-pixel coverage every frame and must
        // ACCUMULATE into an anti-aliased average rather than reset. The YCoCg neighborhood clip in
        // the caller is the anti-ghost for genuine (camera-motion) disocclusion — new content has a
        // different colour and is clipped out (Karis 2014); the DYNAMIC-object trail it cannot catch
        // is handled by the dynamic anti-ghost in the caller (via `out_dyn`).
        if (pc.packed_hist != 0u)
        {
            // Packed layout: validity is folded into the length (len >= 1 = real surface, 0 = sky),
            // the dynamic flag into the length's fraction.
            float4 ph = hist_load_packed(pc.hist_read, pidx);
            if (ph.a > 0.5)
            {
                float dyn;
                float plen = packed_len_flag(ph.a, dyn);
                rgb_sum += w[i] * ph.rgb;
                len_sum += w[i] * plen;
                dyn_sum += w[i] * dyn;
                wsum += w[i];
            }
        }
        else
        {
            float4 pp = asfloat(g.storage_buffers[pc.pos_read].Load4(pidx * 16u));
            if (pp.w > 0.5)
            {
                float4 ph = asfloat(g.storage_buffers[pc.hist_read].Load4(pidx * 16u));
                rgb_sum += w[i] * ph.rgb;
                len_sum += w[i] * ph.a;
                dyn_sum += w[i] * pp.x; // pos.x = the history dynamic flag
                wsum += w[i];
            }
        }
    }
    // Accept whenever ANY of the footprint is a real prior surface (re-normalized by the valid
    // weight). A higher threshold left foreground silhouette pixels whose 2x2 footprint straddles
    // the background (sky stores w=0) stuck at "no history" -> those 1px edges crawled every frame;
    // re-normalizing over just the valid taps lets them accumulate into a stable anti-aliased edge.
    if (wsum > 1e-3)
    {
        out_rgb = rgb_sum / wsum;
        out_len = len_sum / wsum;
        out_dyn = dyn_sum / wsum;
    }
}

// ---- Catmull-Rom history resample (TAAU_FLAG_CATMULL_ROM) -----------------------------------
// The motion-blur root cause: with a bilinear history fetch, EVERY frame of camera motion
// re-samples the accumulated image at a new sub-pixel phase, and a bilinear kernel is a low-pass.
// Applied once it is invisible; applied once per frame into a feedback loop with blend weight
// alpha, the steady-state response is the low-pass raised to ~1/alpha — which is why a 64-frame
// history goes soft the instant the camera moves while a converged still frame stays sharp
// (measured: door dolly gradient energy 3.98 static -> 1.92 moving, and NEITHER the anti-flicker
// nor the clamp-expansion knob moved it). Catmull-Rom is the industry-standard fix: a bicubic
// interpolating kernel whose negative outer lobes restore the high frequencies bilinear drops.
//
// The 5-bilinear-tap trick every engine uses needs a hardware sampler; this history lives in a
// raw storage buffer, so the taps are manual point loads: the separable 4x4 support minus its 4
// corners (weight products of the two outer lobes — the standard corner-cut), i.e. 12 taps,
// renormalized by the retained weight.
//
// Robustness: CR is only taken when ALL 12 taps hold a real prior surface. A partially-valid
// footprint (silhouette, sky edge) would have to renormalize over a subset of a kernel with
// NEGATIVE weights — unstable, and exactly where CR would ring anyway — so those pixels fall
// back to the (validity-renormalizing) bilinear fetch verbatim. The result is clamped to the
// component-wise min/max of the taps: no new extrema means no ringing halo, while the kernel's
// sharpening (which comes from the weight distribution, not from overshoot) is preserved.
// `out_len` stays bilinear — the history length is an accumulation counter, not an image.
float4 catmull_rom_weights(float f)
{
    // Basis for the offsets {-1, 0, 1, 2}; sums to 1 for every f, w = (0,1,0,0) at f = 0.
    float f2 = f * f;
    float f3 = f2 * f;
    return 0.5 * float4(-f3 + 2.0 * f2 - f, 3.0 * f3 - 5.0 * f2 + 2.0,
                        -3.0 * f3 + 4.0 * f2 + f, f3 - f2);
}

// Load one history texel; returns false when the texel is out of bounds or holds no real prior
// surface. Packed layout folds validity into the length; the unpacked layout keeps it in `pos.w`
// (loaded as the single dword it is — the other three are never read).
bool hist_tap(int2 p, out float3 rgb, out float len, out float dyn)
{
    rgb = float3(0, 0, 0);
    len = 0.0;
    dyn = 0.0;
    if (p.x < 0 || p.y < 0 || p.x >= int(pc.out_width) || p.y >= int(pc.out_height))
        return false;
    uint pidx = uint(p.y) * pc.out_width + uint(p.x);
    if (pc.packed_hist != 0u)
    {
        float4 ph = hist_load_packed(pc.hist_read, pidx);
        rgb = make_finite(ph.rgb); // heal a poisoned texel instead of spreading it
        len = packed_len_flag(ph.a, dyn);
        return ph.a > 0.5;
    }
    float4 pp = asfloat(g.storage_buffers[pc.pos_read].Load4(pidx * 16u));
    if (pp.w <= 0.5)
        return false;
    dyn = pp.x; // pos.x = the history dynamic flag
    float4 ph = asfloat(g.storage_buffers[pc.hist_read].Load4(pidx * 16u));
    rgb = make_finite(ph.rgb);
    len = ph.a;
    return true;
}

void fetch_history_catmull_rom(float2 puv, out float3 out_rgb, out float out_len,
                               out float out_dyn)
{
    out_rgb = float3(0, 0, 0);
    out_len = 0.0;
    out_dyn = 0.0;
    float2 res = float2(pc.out_width, pc.out_height);
    float2 sp = puv * res - 0.5; // same texel-center convention as the bilinear fetch
    float2 base = floor(sp);
    float2 f = sp - base;
    float4 wx = catmull_rom_weights(f.x);
    float4 wy = catmull_rom_weights(f.y);

    float3 rgb_sum = float3(0, 0, 0);
    float3 lo = float3(1e30, 1e30, 1e30);
    float3 hi = float3(-1e30, -1e30, -1e30);
    // The inner 2x2 (offsets 0,1 on both axes) doubles as the bilinear fallback + the length.
    float3 bil_rgb = float3(0, 0, 0);
    float len_sum = 0.0;
    float dyn_sum = 0.0;
    float wsum = 0.0;
    float bilsum = 0.0;
    bool all_valid = true;
    [unroll] for (int j = 0; j < 4; j++)
    {
        [unroll] for (int i = 0; i < 4; i++)
        {
            // Corner-cut: skip the four taps whose weight is the product of both outer lobes.
            if ((i == 0 || i == 3) && (j == 0 || j == 3))
                continue;
            float3 rgb;
            float len;
            float dyn;
            if (!hist_tap(int2(base) + int2(i - 1, j - 1), rgb, len, dyn))
            {
                all_valid = false;
                continue;
            }
            float w = wx[i] * wy[j];
            rgb_sum += w * rgb;
            wsum += w;
            lo = min(lo, rgb);
            hi = max(hi, rgb);
            if (i >= 1 && i <= 2 && j >= 1 && j <= 2)
            {
                // Bilinear weights over the inner 2x2 (the same footprint the bilinear fetch uses).
                float bw = (i == 1 ? 1.0 - f.x : f.x) * (j == 1 ? 1.0 - f.y : f.y);
                bil_rgb += bw * rgb;
                len_sum += bw * len;
                dyn_sum += bw * dyn;
                bilsum += bw;
            }
        }
    }
    if (bilsum <= 1e-3)
        return; // no usable history in the footprint at all
    out_len = len_sum / bilsum;
    out_dyn = dyn_sum / bilsum;
    out_rgb = all_valid && abs(wsum) > 1e-3 ? clamp(rgb_sum / wsum, lo, hi) : bil_rgb / bilsum;
}

float3 reconstruct(float2 cuv, float depth, float sy)
{
    float2 ndc = float2(cuv.x * 2.0 - 1.0, (cuv.y * 2.0 - 1.0) * sy);
    float4 h = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    return h.xyz / h.w;
}

// Velocity-aware reprojection (pipeline re-baseline PR-2). When a per-pixel motion-vector
// target is bound, reproject each pixel by ITS OWN screen motion instead of the camera-only
// world-space reprojection — this is what removes ghosting on moving objects (spin / skinning
// / morph), whose surface points move on screen independently of the camera.
//   • The stored value is `cur_ndc - prev_ndc` (raw NDC delta, jitter-free, Y-flip cancels in
//     the difference). Previous UV = current UV − motion, converting the NDC delta to UV:
//     Δuv = (Δndc.x, Δndc.y·sy) · 0.5 (the same sy the reconstruct/reproject use).
//   • 3×3 DILATION (canonical, Playdead/FSR2): fetch the LONGEST velocity in the 3×3
//     neighborhood, so silhouette pixels adopt the foreground's motion and their history is
//     pulled from the moving object, not the static background behind the edge (kills the
//     smear at the trailing edge of a moving object).
// `uv` is the STABLE output-grid coordinate (the history grid); `cuv` = uv + jitter is where the
// current geometry sits in the jittered internal frame (where the velocity was rasterized). The
// velocity target is UNJITTERED, so we sample the motion at `cuv` (the surface point) but apply it
// on the stable grid: prev_uv = uv − motion. Mixing the two (reprojecting the jittered `cuv`) leaves
// the sub-pixel jitter in the reprojection and destabilizes the history — sampling the motion at the
// geometry while reprojecting the stable grid keeps both the motion accurate and the history aligned.
// `dyn_cur` / `dyn_center` report the velocity target's DYNAMIC-OBJECT flag (B channel):
// `dyn_cur` = max over the 3×3 (the reference engine samples a 5-tap cross for the same
// purpose — TemporalAA.usf AA_DYNAMIC_ANTIGHOST; the 3×3 is the loop this function already
// runs, one texel more conservative), `dyn_center` = the center tap, which is what gets
// STORED into this pixel's history flag (the reference engine's `Dynamic4`). Both are
// written even when the function reprojects nothing (static pixel / early-out), because the
// anti-ghost needs them on the camera-reprojection path too.
bool velocity_reproject(float2 uv, float2 cuv, float sy, out float2 prev_uv,
                        out float dyn_cur, out float dyn_center)
{
    prev_uv = uv;
    dyn_cur = 0.0;
    dyn_center = 0.0;
    if (pc.velocity_index == 0xFFFFFFFFu)
        return false;
    float2 its = 1.0 / float2(pc.in_width, pc.in_height);
    float2 best = float2(0.0, 0.0);
    float best_len = -1.0;
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            float2 suv = cuv + float2(dx, dy) * its;
            float3 s = g.textures[pc.velocity_index].SampleLevel(g.samp, suv, 0.0).xyz;
            float2 v = s.xy;
            dyn_cur = max(dyn_cur, s.z);
            if (dx == 0 && dy == 0)
                dyn_center = s.z;
            float l = dot(v, v);
            if (l > best_len)
            {
                best_len = l;
                best = v;
            }
        }
    }
    // Static-pixel fall-through: an exactly-zero motion (RG16 stores 0 exactly for static geometry)
    // hands back to the camera-only world-space reprojection — that path is the byte-identical anchor
    // for a static scene, so a genuinely static frame stays bit-stable rather than diverging by a
    // sub-pixel numerical path. A tiny epsilon (well below one texel) keeps every actually-moving
    // pixel — even slow interior motion — on the velocity path so the anti-ghosting is preserved.
    if (best_len <= 1e-12)
        return false;
    // NDC delta -> UV delta (y carries the Y-flip sign so a Vulkan-flipped clip still maps to
    // the same on-screen direction as D3D12 — DX≡VK).
    float2 duv = float2(best.x, best.y * sy) * 0.5;
    prev_uv = uv - duv;
    return true;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.out_width || tid.y >= pc.out_height)
        return;

    uint idx = tid.y * pc.out_width + tid.x;
    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.out_width, pc.out_height);
    // The internal frame was rendered with a +jitter camera offset, so the content for this stable
    // output pixel sits at `uv + jitter` in the internal image (lands every frame on the stable grid).
    float2 cuv = uv + pc.jitter.xy;
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, cuv, 0.0).r;
    // NaN/Inf guard (make_finite, bindless.slang): TAAU is a temporal ACCUMULATOR —
    // one non-finite lit pixel entering the history spreads through the variance
    // boxes and repaints the whole frame black within seconds (observed in play).
    // Finite values pass through unchanged, so the anchors stay byte-identical.
    float3 cur = make_finite(g.textures[pc.hdr_index].SampleLevel(g.samp, cuv, 0.0).rgb);

    if (depth >= 1.0)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(cur, 1.0);
        if (pc.packed_hist != 0u)
        {
            // len 0 = "no prior surface" (the packed replacement for pos.w = 0); the rgb is
            // never consumed for an invalid texel (matching the unpacked path, whose sky rgb
            // is likewise dead — taps gate on validity before touching the history).
            hist_store_packed(pc.hist_write, idx, float4(cur, 0.0));
        }
        else
        {
            g.storage_buffers[pc.hist_write].Store4(idx * 16u, asuint(float4(cur, 1.0)));
            g.storage_buffers[pc.pos_write].Store4(idx * 16u, asuint(float4(0, 0, 0, 0)));
        }
        return;
    }

    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 its = 1.0 / float2(pc.in_width, pc.in_height);

    // 3x3 neighborhood pass: YCoCg moments for the variance box (the anti-ghost color distribution).
    float3 m1 = float3(0, 0, 0); // sum of YCoCg
    float3 m2 = float3(0, 0, 0); // sum of YCoCg^2
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            float2 suv = cuv + float2(dx, dy) * its;
            float3 c =
                rgb_to_ycocg(make_finite(g.textures[pc.hdr_index].SampleLevel(g.samp, suv, 0.0).rgb));
            m1 += c;
            m2 += c * c;
        }
    }
    float3 mean = m1 / 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
    float gamma = pc.params.z;
    float3 box_min = mean - gamma * sigma;
    float3 box_max = mean + gamma * sigma;
    // Firefly cap on the INPUT sample (content only, params.w > 0): a pixel far brighter than its own
    // 3x3 σ is an outlier (a sub-pixel specular spark the jitter lands on this texel) that pops as a
    // firefly when it blends into the history. Clip it toward the neighbourhood box so it enters as a
    // highlight, not a spark. The gallery (params.w == 0) keeps the raw input → byte-identical anchor.
    if (pc.params.w > 0.0)
        cur = ycocg_to_rgb(clip_aabb(box_min, box_max, rgb_to_ycocg(cur)));
    // World point of THIS pixel (validation + storage) and of the closest neighbor (reprojection
    // motion — closest-depth dilation so silhouettes fetch the foreground history).
    float3 world_pos = reconstruct(cuv, depth, sy);
    // Reproject THIS pixel's own surface. Closest-depth dilation (fetching the foreground's history
    // at silhouettes) is omitted on purpose: under sub-pixel jitter it makes the reprojection target
    // oscillate fg/bg every frame, fetching a different history each time -> edge crawl. Own-surface
    // reprojection keeps each output pixel accumulating its own coverage into an anti-aliased edge.
    float3 world_reproj = world_pos;

    float3 hist_rgb = float3(0, 0, 0);
    float hist_len = 0.0;
    float hist_dyn = 0.0; // history's dynamic-object flag at the reprojected location
    // Where this pixel's surface sat last frame, on the stable output grid. `prev_uv == uv` when the
    // reprojection produced nothing usable (reset / off-screen), which reads as zero motion — inert,
    // because there is then no history for the motion policy to weigh anyway.
    float2 prev_uv = uv;
    bool cr = (pc.taau_flags & TAAU_FLAG_CATMULL_ROM) != 0u;
    // Sampled unconditionally (cheap early-out when the velocity target is absent): the dynamic
    // flags are needed by the anti-ghost below and by the history store even when the pixel takes
    // the camera-reprojection path (its own motion may be zero while a mover sits nearby).
    float2 vuv;
    float dyn_cur, dyn_center;
    bool vel_ok = velocity_reproject(uv, cuv, sy, vuv, dyn_cur, dyn_center);
    if (pc.reset == 0u)
    {
        // When the PR-2 motion-vector target is bound, prefer per-pixel velocity reprojection
        // (removes moving-object ghosting). When it is ABSENT (the default), run the original
        // camera-only world-space reprojection block VERBATIM so its codegen — and therefore the
        // accumulated history — is bit-identical to the pre-PR-2 pipeline (content-scene byte
        // parity; the velocity path is a strict opt-in addition, never a rewrite of the default).
        if (pc.velocity_index != 0xFFFFFFFFu && vel_ok)
        {
            if (all(vuv >= 0.0) && all(vuv <= 1.0))
            {
                prev_uv = vuv;
                if (cr)
                    fetch_history_catmull_rom(vuv, hist_rgb, hist_len, hist_dyn);
                else
                    fetch_history_bilinear(vuv, world_pos, hist_rgb, hist_len, hist_dyn);
            }
        }
        else
        {
            float4 pclip = mul(pc.prev_view_proj, float4(world_reproj, 1.0));
            if (pclip.w > 0.0)
            {
                float3 pndc = pclip.xyz / pclip.w;
                float2 puv = float2(pndc.x * 0.5 + 0.5, pndc.y * sy * 0.5 + 0.5);
                if (all(puv >= 0.0) && all(puv <= 1.0) && pndc.z >= 0.0 && pndc.z <= 1.0)
                {
                    // Sub-pixel-accurate history reprojection — see fetch_history_bilinear /
                    // fetch_history_catmull_rom.
                    prev_uv = puv;
                    if (cr)
                        fetch_history_catmull_rom(puv, hist_rgb, hist_len, hist_dyn);
                    else
                        fetch_history_bilinear(puv, world_pos, hist_rgb, hist_len, hist_dyn);
                }
            }
        }
    }

    // DYNAMIC ANTI-GHOST (the reference engine's TemporalAA.usf AA_DYNAMIC_ANTIGHOST, verbatim
    // logic): if NO dynamic object covers this pixel's neighborhood NOW, but the reprojected
    // history was accumulated ON a dynamic object, that history is the trail a mover left on the
    // background it just uncovered — `IgnoreHistory`. This is the ghost the YCoCg clip cannot
    // catch: next to a high-contrast mover silhouette the variance box is wide (and the content
    // tiers widen it further via `clamp_expand`), so the mover-colored history passes the clip
    // and smears for many frames. Rejecting on provenance instead of color is exact: the trail
    // resets to the current frame in ONE frame and re-accumulates. The mover itself keeps its
    // history (`dyn_cur` suppresses the reject on and around it — the reference engine's
    // `!Dynamic` term), and a mover that stops decays its flags via the CPU-side hysteresis
    // before its own history reads "static over dynamic history" (no pop). Velocity-absent
    // (the default path) and static scenes never set a flag — byte-identical.
    if (pc.velocity_index != 0xFFFFFFFFu && dyn_cur < 0.05 && hist_dyn > 0.05)
        hist_len = 0.0;

    // Velocity gate (`jitter.z` = ramp in output px; 0 = OFF, which is the shipping default —
    // see the measured negative result on the history cap below). This pixel's screen motion in
    // OUTPUT pixels per frame, jitter-free on both reprojection paths (`uv` is the stable grid,
    // `prev_view_proj` / the motion vector are both unjittered) and per-pixel, so parallax is
    // respected: a near pillar sweeping past reads as moving while the distant nave behind it does not.
    float motion_px = length((uv - prev_uv) * float2(pc.out_width, pc.out_height));
    float motion_w = pc.jitter.z > 0.0 ? saturate(motion_px / pc.jitter.z) : 0.0;

    // Reference-engine (TSR) anti-flicker clamp-error: a tight μ±γσ box RE-CLIPS the converged history
    // every frame on a static high-contrast edge — the sub-pixel jitter shifts the low-res neighborhood
    // statistics, so the shifting box snaps the (stable) history to a different face each frame = the
    // dominant upsampling shimmer. Widen the box by a fraction of its OWN size (∝ local contrast, so
    // flat areas — where the box is tiny — barely move and don't ghost), letting static detail
    // CONVERGE. `params.w` = 0 (gallery) is exact byte-identical; content passes the expansion factor.
    // NOT motion-gated: retiring the expansion while moving was measured to make the door ROI flicker
    // WORSE (2.32 -> 2.66 avg/ch), because the shimmer this box fights is itself a moving-camera
    // artifact — the sub-pixel aperture flips coverage as the view slides, not while it sits still.
    float3 clamp_err = (box_max - box_min) * pc.params.w;
    box_min -= clamp_err;
    box_max += clamp_err;

    float3 accum;
    float len;
    if (hist_len > 0.5)
    {
        // YCoCg variance clip — the primary anti-ghost. History outside the neighborhood's color
        // distribution is snapped onto the box; in-distribution history passes through (stable).
        float3 hist_y = clip_aabb(box_min, box_max, rgb_to_ycocg(hist_rgb));
        hist_rgb = ycocg_to_rgb(hist_y);
        // Straight EMA: the clip rejects stale (camera-moved) history, so long, stable accumulation
        // hides the jitter and resolves sub-pixel detail. alpha floors at 1/max_hist (len capped).
        // Motion-gated history cap (`jitter.w`) — the classic "responsive TAA" weight: shorten the
        // history in proportion to this pixel's motion so a moving pixel is dominated by the current,
        // un-resampled sample. **Measured NEGATIVE here and therefore OFF by default** (ramp 0):
        // this pass is TAA-U, and in an upsampler the history is not merely anti-aliasing, it IS the
        // reconstruction — the full-res detail exists only as the accumulation of jittered low-res
        // frames. Cutting the history to 8 frames throws that reconstruction away, and both metrics
        // agreed: motion gradient energy 2.88 -> 2.74 (LESS sharp, the opposite of the intent) and
        // the lag residual against a converged still at the same pose 16.1 -> 20.9 (worse). The knob
        // is kept because it is the right lever for a NON-upsampling temporal pass (`P_TAAU_FORCE=1`
        // at native res, where the history carries no resolution), not for the upscale path.
        // `motion_w == 0` yields `min(hist_len + 1, params.y)` bit-exactly.
        len = min(hist_len + 1.0, lerp(pc.params.y, pc.jitter.w, motion_w));
        float alpha = 1.0 / len;
        // Luminance anti-flicker (Lottes/Karis) — OPT-IN via flip_y bit2 (`P_TAAU_ANTIFLICKER`),
        // default OFF so the sealed content baseline (sc_viz / gdf_ao goldens, which capture at the
        // TAAU tier res) stays byte-identical; the PT gate runs RENDER_SCALE=1 (TAAU off) so it
        // cannot measure this, hence no quality gate exists to justify a default flip yet.
        // What it fixes: a sub-pixel BRIGHT APERTURE — a door's sky-gaps, a window grille — flips
        // the current sample bright↔dark each jittered frame; because it's a bright CLUSTER the
        // YCoCg variance box can't clamp it, so it shimmers as dense speckle. When the current luma
        // diverges from the (stable) clipped history, damp the new sample's weight so the flip
        // averages toward the converged value (a stable anti-aliased edge) instead of popping.
        // Measured: door HF 5.19 → 3.67 (below the jitter-off 3.92 baseline).
        if ((pc.flip_y & 4u) != 0u)
        {
            float lc = luma(cur);
            float lh = luma(hist_rgb);
            float d = abs(lc - lh) / max(max(lc, lh), 0.2);
            float flick_w = saturate(1.0 - d);
            // NOT motion-gated. Releasing the damping while the camera moves looks like the obvious
            // "it only costs responsiveness on a moving pixel" win, but the artifact it was measured
            // against is itself a DOLLY artifact (the closure batch's door-ROI numbers are all from a
            // moving camera — a still frame was already at 0.026 avg/ch before this existed), so
            // releasing it under motion simply undoes the fix: ROI flicker 2.21 -> 2.66 avg/ch and
            // bright-pixel std 47 -> 72 when it was tried. Left unconditional.
            alpha *= flick_w * flick_w; // low weight to a divergent (flickering) sample; 1 if stable
        }
        accum = lerp(hist_rgb, cur, alpha);
    }
    else
    {
        len = 1.0;
        accum = cur;
    }

    accum = make_finite(accum);
    // Store this pixel's dynamic flag with the accumulation (the reference engine's
    // `OutputPayload.Color.a = Dynamic4` — the CENTER velocity tap, not the dilated max, so
    // the flag marks actual mover coverage, not its neighborhood). `dyn_center` is 0 whenever
    // the velocity target is absent, so the default path stores exactly what it always did.
    if (pc.packed_hist != 0u)
    {
        // len >= 1 for every real surface, so the validity flag rides in the length (and the
        // dynamic flag in its fraction — see packed_len_store).
        hist_store_packed(pc.hist_write, idx, float4(accum, packed_len_store(len, dyn_center)));
    }
    else
    {
        g.storage_buffers[pc.hist_write].Store4(idx * 16u, asuint(float4(accum, len)));
        // pos = (dynamic flag, 0, 0, valid). The world point this buffer once held was never
        // consumed (only .w — see fetch_history_bilinear), so .x now carries the flag.
        g.storage_buffers[pc.pos_write].Store4(idx * 16u,
                                               asuint(float4(dyn_center, 0.0, 0.0, 1.0)));
    }
    // Output the accumulated result: for disoccluded / no-history pixels this is `cur` (the current
    // frame), so newly-revealed regions show the live frame instead of empty (black) history.
    g.storage_images[pc.out_index][tid.xy] = float4(accum, 1.0);
}
