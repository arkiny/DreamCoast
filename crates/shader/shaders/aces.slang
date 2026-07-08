// ACES 1.3 RRT + sRGB ODT (100 nits, dim surround) — a Slang port of the Academy Color
// Encoding System reference transforms, for the baked tonemap LUT (`tonemap_lut.slang`).
//
// Ported from the A.M.P.A.S. reference implementation (aces-dev v1.3, CTL):
//   transforms/ctl/rrt/RRT.ctl
//   transforms/ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl
//   transforms/ctl/lib/ACESlib.{RRT_Common,ODT_Common,Tonescales,Transform_Common,Utilities_Color}.ctl
//
// Copyright © 2015 Academy of Motion Picture Arts and Sciences (A.M.P.A.S.).
// Portions contributed by others as indicated. All rights reserved.
// This is a derivative work; the full license terms (conditions + Disclaimer of Warranty)
// are retained in THIRD_PARTY_LICENSES.md ("Academy Color Encoding System components").
//
// Port notes (structure kept 1:1 with the CTL for auditability):
//   • CTL's `mult_f3_f33(v, M)` is a ROW-vector product; the equivalent constants here are
//     stored transposed where needed so every product is the HLSL column-vector `mul(M, v)`.
//   • The chain is specialized to this engine's use: input is LINEAR Rec.709/sRGB-primaries
//     scene color (the lighting pipeline's working space), converted to ACES2065-1 via the
//     Bradford-adapted matrix below (the standard sRGB→AP0 "IDT"); output is DISPLAY-LINEAR
//     Rec.709 in [0,1] — the final IEC 61966-2-1 (sRGB) encode stays in `post.slang`
//     (`linear_to_srgb` == the ODT's `moncurve_r(2.4, 0.055)`).
//   • The RRT→OCES→ODT AP0 round-trip is kept verbatim (bake-time cost is 1 LUT, not per-pixel).
//   • Port validated against an independent CPU implementation of the same CTL: achromatic ramp
//     monotonic, mid-grey 0.18 → 0.1041 display-linear, and within the published ~1-3% envelope
//     of the widely-used analytic fit of RRT+ODT across 0.005..11.2 + chromatic probes.

#pragma once

// ---------------------------------------------------------------------------------------------
// Color-space constants (derived from the AP0/AP1/Rec.709 chromaticities + Bradford CAT, the
// exact values the CTL computes at load; row-major for column-vector mul).
// ---------------------------------------------------------------------------------------------

// Linear Rec.709 (D65) -> ACES2065-1 AP0 (D60), Bradford-adapted.
static const float3x3 ACES_SRGB_2_AP0 = float3x3(
    0.4396329819, 0.3829886982, 0.1773783199,
    0.0897764430, 0.8134394287, 0.0967841283,
    0.0175411704, 0.1115465533, 0.8709122763);

static const float3x3 ACES_AP0_2_AP1 = float3x3(
    1.4514393161, -0.2365107469, -0.2149285693,
    -0.0765537734, 1.1762296998, -0.0996759264,
    0.0083161484, -0.0060324498, 0.9977163014);

static const float3x3 ACES_AP1_2_AP0 = float3x3(
    0.6954522414, 0.1406786965, 0.1638690622,
    0.0447945634, 0.8596711185, 0.0955343182,
    -0.0055258826, 0.0040252103, 1.0015006723);

static const float3x3 ACES_AP1_2_XYZ = float3x3(
    0.6624541811, 0.1340042065, 0.1561876870,
    0.2722287168, 0.6740817658, 0.0536895174,
    -0.0055746495, 0.0040607335, 1.0103391003);

static const float3x3 ACES_XYZ_2_AP1 = float3x3(
    1.6410233797, -0.3248032942, -0.2364246952,
    -0.6636628587, 1.6153315917, 0.0167563477,
    0.0117218943, -0.0082844420, 0.9883948585);

// Bradford CAT, ACES white (D60-like x 0.32168 y 0.33767) -> D65.
static const float3x3 ACES_D60_2_D65 = float3x3(
    0.9872240087, -0.0061132286, 0.0159532883,
    -0.0075983718, 1.0018614847, 0.0053300358,
    0.0030725771, -0.0050959615, 1.0816806031);

// CIE XYZ -> linear Rec.709 primaries (D65).
static const float3x3 ACES_XYZ_2_REC709 = float3x3(
    3.2409699419, -1.5373831776, -0.4986107603,
    -0.9692436363, 1.8759675015, 0.0415550574,
    0.0556300797, -0.2039769589, 1.0569715142);

// AP1 luminance weights (the Y row of AP1_2_XYZ).
static const float3 ACES_AP1_RGB2Y = float3(0.2722287168, 0.6740817658, 0.0536895174);

// calc_sat_adjust_matrix(RRT_SAT_FACTOR = 0.96, AP1_RGB2Y): out = (1-sat)*luma + sat*in.
static const float3x3 ACES_RRT_SAT_MAT = float3x3(
    0.9708891487, 0.0269632706, 0.0021475807,
    0.0108891487, 0.9869632706, 0.0021475807,
    0.0108891487, 0.0269632706, 0.9621475807);

// calc_sat_adjust_matrix(ODT_SAT_FACTOR = 0.93, AP1_RGB2Y).
static const float3x3 ACES_ODT_SAT_MAT = float3x3(
    0.9490560102, 0.0471857236, 0.0037582662,
    0.0190560102, 0.9771857236, 0.0037582662,
    0.0190560102, 0.0471857236, 0.9337582662);

// RRT "sweetener" constants (ACESlib.RRT_Common).
static const float ACES_RRT_GLOW_GAIN = 0.05;
static const float ACES_RRT_GLOW_MID = 0.08;
static const float ACES_RRT_RED_SCALE = 0.82;
static const float ACES_RRT_RED_PIVOT = 0.03;
static const float ACES_RRT_RED_HUE = 0.0;
static const float ACES_RRT_RED_WIDTH = 135.0;

// ODT constants (ACESlib.ODT_Common).
static const float ACES_CINEMA_WHITE = 48.0;
static const float ACES_CINEMA_BLACK = 0.02; // CINEMA_WHITE / 2400
static const float ACES_DIM_SURROUND_GAMMA = 0.9811;

static const float ACES_HALF_MIN = 5.96046448e-08;
static const float ACES_HALF_MAX = 65504.0;

// ---------------------------------------------------------------------------------------------
// Utility functions (ACESlib.Utilities_Color / Transform_Common)
// ---------------------------------------------------------------------------------------------

float aces_rgb_2_saturation(float3 rgb)
{
    const float TINY = 1e-10;
    float mx = max(rgb.r, max(rgb.g, rgb.b));
    float mn = min(rgb.r, min(rgb.g, rgb.b));
    return (max(mx, TINY) - max(mn, TINY)) / max(mx, 1e-2);
}

// Luminance proxy YC ~ Y + K * chroma (normalized: RGB 1,1,1 -> 1).
float aces_rgb_2_yc(float3 rgb)
{
    const float yc_radius_weight = 1.75;
    float chroma = sqrt(max(rgb.b * (rgb.b - rgb.g) + rgb.g * (rgb.g - rgb.r) +
                                rgb.r * (rgb.r - rgb.b),
                            0.0));
    return (rgb.r + rgb.g + rgb.b + yc_radius_weight * chroma) / 3.0;
}

// Geometric hue angle in degrees [0, 360). The CTL returns NaN for exact neutrals; every
// consumer multiplies the result's weight by saturation (0 for neutrals), so returning 0
// here is equivalent and NaN-free.
float aces_rgb_2_hue(float3 rgb)
{
    if (rgb.r == rgb.g && rgb.g == rgb.b)
        return 0.0;
    float hue = (180.0 / 3.14159265358979) *
                atan2(sqrt(3.0) * (rgb.g - rgb.b), 2.0 * rgb.r - rgb.g - rgb.b);
    return hue < 0.0 ? hue + 360.0 : hue;
}

float aces_center_hue(float hue, float center)
{
    float c = hue - center;
    if (c < -180.0)
        c += 360.0;
    else if (c > 180.0)
        c -= 360.0;
    return c;
}

// ---------------------------------------------------------------------------------------------
// RRT sweeteners (ACESlib.RRT_Common)
// ---------------------------------------------------------------------------------------------

float aces_sigmoid_shaper(float x)
{
    // Sigmoid in 0..1 spanning -2..+2.
    float t = max(1.0 - abs(x / 2.0), 0.0);
    float y = 1.0 + sign(x) * (1.0 - t * t);
    return y / 2.0;
}

float aces_glow_fwd(float yc_in, float glow_gain, float glow_mid)
{
    if (yc_in <= 2.0 / 3.0 * glow_mid)
        return glow_gain;
    if (yc_in >= 2.0 * glow_mid)
        return 0.0;
    return glow_gain * (glow_mid / yc_in - 0.5);
}

// Cubic B-spline basis bump over full base width `w` degrees, peak 1 at x = 0.
float aces_cubic_basis_shaper(float x, float w)
{
    // Textbook 4x4 monomial->basis matrix (rows: t^3, t^2, t, 1 coefficients per segment).
    const float4x4 M = float4x4(-1.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0, 1.0 / 6.0, //
                                3.0 / 6.0, -6.0 / 6.0, 3.0 / 6.0, 0.0,        //
                                -3.0 / 6.0, 0.0, 3.0 / 6.0, 0.0,              //
                                1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0, 0.0);
    if (x <= -w / 2.0 || x >= w / 2.0)
        return 0.0;
    float knot_coord = (x + w / 2.0) * 4.0 / w;
    int j = int(knot_coord);
    float t = knot_coord - float(j);
    float4 monomials = float4(t * t * t, t * t, t, 1.0);
    // CTL selects basis column (3 - j); guard j to the valid 0..3 (x strictly inside the span).
    float y = 0.0;
    if (j == 3)
        y = dot(monomials, float4(M[0][0], M[1][0], M[2][0], M[3][0]));
    else if (j == 2)
        y = dot(monomials, float4(M[0][1], M[1][1], M[2][1], M[3][1]));
    else if (j == 1)
        y = dot(monomials, float4(M[0][2], M[1][2], M[2][2], M[3][2]));
    else if (j == 0)
        y = dot(monomials, float4(M[0][3], M[1][3], M[2][3], M[3][3]));
    return y * 3.0 / 2.0;
}

// ---------------------------------------------------------------------------------------------
// Segmented-spline tonescales (ACESlib.Tonescales). The splines are quadratic B-splines in
// log10-log10 space; `knots - 1` scales the knot coordinate (c5: 4 knots, c9: 8 knots).
// ---------------------------------------------------------------------------------------------

// Monomial->basis matrix M (3x3) applied as the CTL's row-vector product cf * M.
float3 aces_spline_basis(float3 cf)
{
    // cf * M with M = {{0.5,-1,0.5},{-1,1,0.5},{0.5,0,0}} (row-major).
    return float3(0.5 * cf.x - cf.y + 0.5 * cf.z, -cf.x + cf.y, 0.5 * (cf.x + cf.y));
}

// RRT tonescale: segmented_spline_c5_fwd with RRT_PARAMS.
float aces_segmented_spline_c5_fwd(float x)
{
    const float coefs_low[6] = { -4.0000000000, -4.0000000000, -3.1573765773,
                                 -0.4852499958, 1.8477324706,  1.8477324706 };
    const float coefs_high[6] = { -0.7185482425, 2.0810307172, 3.6681241237,
                                  4.0000000000,  4.0000000000, 4.0000000000 };
    // log10 of {0.18*2^-15, 0.18, 0.18*2^18}; y = {0.0001, 4.8, 10000}, end slopes 0.
    const float log_min_x = -5.2601774298564115; // log10(0.18 * 2^-15)
    const float log_mid_x = -0.7447274948966940; // log10(0.18)
    const float log_max_x = 4.6738124270549670;  // log10(0.18 * 2^18)
    const float log_min_y = -4.0;                // log10(0.0001)
    const float log_max_y = 4.0;                 // log10(10000)

    float logx = log10(max(x, ACES_HALF_MIN));
    float logy;
    if (logx <= log_min_x)
    {
        logy = log_min_y; // slopeLow = 0
    }
    else if (logx < log_mid_x)
    {
        float knot_coord = 3.0 * (logx - log_min_x) / (log_mid_x - log_min_x);
        int j = int(knot_coord);
        float t = knot_coord - float(j);
        float3 b = aces_spline_basis(float3(coefs_low[j], coefs_low[j + 1], coefs_low[j + 2]));
        logy = dot(float3(t * t, t, 1.0), b);
    }
    else if (logx < log_max_x)
    {
        float knot_coord = 3.0 * (logx - log_mid_x) / (log_max_x - log_mid_x);
        int j = int(knot_coord);
        float t = knot_coord - float(j);
        float3 b = aces_spline_basis(float3(coefs_high[j], coefs_high[j + 1], coefs_high[j + 2]));
        logy = dot(float3(t * t, t, 1.0), b);
    }
    else
    {
        logy = log_max_y; // slopeHigh = 0
    }
    return pow(10.0, logy);
}

// ODT tonescale: segmented_spline_c9_fwd with ODT_48nits.
float aces_segmented_spline_c9_fwd(float x)
{
    const float coefs_low[10] = { -1.6989700043, -1.6989700043, -1.4779000000, -1.2291000000,
                                  -0.8648000000, -0.4480000000, 0.0051800000,  0.4511080334,
                                  0.9113744414,  0.9113744414 };
    const float coefs_high[10] = { 0.5154386965, 0.8470437783, 1.1358000000, 1.3802000000,
                                   1.5197000000, 1.5985000000, 1.6467000000, 1.6746091357,
                                   1.6878733390, 1.6878733390 };
    // Endpoints are segmented_spline_c5_fwd(0.18 * 2^{-6.5, 0, 6.5}) (precomputed);
    // y = {0.02, 4.8, 48}, end slopes 0.
    const float log_min_x = -2.5406236188539997; // log10(c5_fwd(0.18*2^-6.5)) = log10(0.0028798932)
    const float log_mid_x = 0.6812412373500000;  // log10(c5_fwd(0.18)) = log10(4.8)
    const float log_max_x = 3.0024768101229165;  // log10(c5_fwd(0.18*2^6.5)) = log10(1005.71936)
    const float log_min_y = -1.6989700043360187; // log10(0.02)
    const float log_max_y = 1.6812412373755872;  // log10(48)

    float logx = log10(max(x, ACES_HALF_MIN));
    float logy;
    if (logx <= log_min_x)
    {
        logy = log_min_y;
    }
    else if (logx < log_mid_x)
    {
        float knot_coord = 7.0 * (logx - log_min_x) / (log_mid_x - log_min_x);
        int j = int(knot_coord);
        float t = knot_coord - float(j);
        float3 b = aces_spline_basis(float3(coefs_low[j], coefs_low[j + 1], coefs_low[j + 2]));
        logy = dot(float3(t * t, t, 1.0), b);
    }
    else if (logx < log_max_x)
    {
        float knot_coord = 7.0 * (logx - log_mid_x) / (log_max_x - log_mid_x);
        int j = int(knot_coord);
        float t = knot_coord - float(j);
        float3 b = aces_spline_basis(float3(coefs_high[j], coefs_high[j + 1], coefs_high[j + 2]));
        logy = dot(float3(t * t, t, 1.0), b);
    }
    else
    {
        logy = log_max_y;
    }
    return pow(10.0, logy);
}

// ---------------------------------------------------------------------------------------------
// RRT (RRT.ctl): ACES2065-1 (AP0) -> OCES
// ---------------------------------------------------------------------------------------------

float3 aces_rrt(float3 aces)
{
    // --- Glow module ---
    float saturation = aces_rgb_2_saturation(aces);
    float yc_in = aces_rgb_2_yc(aces);
    float s = aces_sigmoid_shaper((saturation - 0.4) / 0.2);
    float added_glow = 1.0 + aces_glow_fwd(yc_in, ACES_RRT_GLOW_GAIN * s, ACES_RRT_GLOW_MID);
    aces *= added_glow;

    // --- Red modifier ---
    float hue = aces_rgb_2_hue(aces);
    float centered_hue = aces_center_hue(hue, ACES_RRT_RED_HUE);
    float hue_weight = aces_cubic_basis_shaper(centered_hue, ACES_RRT_RED_WIDTH);
    aces.r += hue_weight * saturation * (ACES_RRT_RED_PIVOT - aces.r) * (1.0 - ACES_RRT_RED_SCALE);

    // --- ACES to RGB rendering space ---
    aces = max(aces, 0.0); // avoid saturated negatives turning positive in the matrix
    float3 rgb_pre = clamp(mul(ACES_AP0_2_AP1, aces), 0.0, ACES_HALF_MAX);

    // --- Global desaturation ---
    rgb_pre = mul(ACES_RRT_SAT_MAT, rgb_pre);

    // --- Tonescale, independently per rendering-space channel ---
    float3 rgb_post = float3(aces_segmented_spline_c5_fwd(rgb_pre.r),
                             aces_segmented_spline_c5_fwd(rgb_pre.g),
                             aces_segmented_spline_c5_fwd(rgb_pre.b));

    // --- RGB rendering space to OCES ---
    return mul(ACES_AP1_2_AP0, rgb_post);
}

// ---------------------------------------------------------------------------------------------
// ODT (ODT.Academy.sRGB_100nits_dim.ctl): OCES -> DISPLAY-LINEAR Rec.709 in [0,1].
// The final IEC 61966-2-1 encode (moncurve_r 2.4/0.055 == linear_to_srgb) is applied by the
// consumer (`post.slang`), which keeps its existing tail unchanged.
// ---------------------------------------------------------------------------------------------

float3 aces_odt_srgb_display_linear(float3 oces)
{
    // OCES to RGB rendering space + tonescale.
    float3 rgb_pre = mul(ACES_AP0_2_AP1, oces);
    float3 rgb_post = float3(aces_segmented_spline_c9_fwd(rgb_pre.r),
                             aces_segmented_spline_c9_fwd(rgb_pre.g),
                             aces_segmented_spline_c9_fwd(rgb_pre.b));

    // Scale luminance to linear code value: Y_2_linCV.
    float3 lin_cv = (rgb_post - ACES_CINEMA_BLACK) / (ACES_CINEMA_WHITE - ACES_CINEMA_BLACK);

    // Dim-surround compensation (darkSurround_to_dimSurround): gamma on Y in xyY.
    float3 xyz = mul(ACES_AP1_2_XYZ, lin_cv);
    float divisor = max(xyz.x + xyz.y + xyz.z, 1e-10);
    float2 xy = float2(xyz.x, xyz.y) / divisor;
    float y_out = pow(clamp(xyz.y, 0.0, ACES_HALF_MAX), ACES_DIM_SURROUND_GAMMA);
    xyz = float3(xy.x * y_out / max(xy.y, 1e-10), y_out,
                 (1.0 - xy.x - xy.y) * y_out / max(xy.y, 1e-10));
    lin_cv = mul(ACES_XYZ_2_AP1, xyz);

    // Desaturation to compensate for the luminance difference.
    lin_cv = mul(ACES_ODT_SAT_MAT, lin_cv);

    // Rendering space -> XYZ -> D65 -> display primaries; clip out-of-gamut.
    xyz = mul(ACES_AP1_2_XYZ, lin_cv);
    xyz = mul(ACES_D60_2_D65, xyz);
    return saturate(mul(ACES_XYZ_2_REC709, xyz));
}

// Full chain for this engine: linear Rec.709 scene color -> display-linear Rec.709 [0,1].
float3 aces_rrt_odt(float3 linear_srgb)
{
    float3 aces = mul(ACES_SRGB_2_AP0, linear_srgb);
    return aces_odt_srgb_display_linear(aces_rrt(aces));
}

// ---------------------------------------------------------------------------------------------
// Tonemap-LUT shaper: the LUT axis is log2-encoded linear scene color. Shared by the bake
// (`tonemap_lut.slang`) and the sample (`post.slang`) so the two can never drift.
// The range spans 2^-12 (deep shadow; the ODT maps it to exactly 0 output) to 2^6 (well past
// the ODT's ~2^4 white saturation), so both curve ends are flat inside the LUT.
// ---------------------------------------------------------------------------------------------

static const float TONEMAP_LUT_LOG_MIN = -12.0;
static const float TONEMAP_LUT_LOG_MAX = 6.0;

float3 tonemap_lut_encode(float3 linear_rgb)
{
    float3 l2 = log2(max(linear_rgb, 6.1e-5)); // 6.1e-5 < 2^-13: below LUT min -> t = 0
    return saturate((l2 - TONEMAP_LUT_LOG_MIN) / (TONEMAP_LUT_LOG_MAX - TONEMAP_LUT_LOG_MIN));
}

float3 tonemap_lut_decode(float3 t)
{
    return exp2(t * (TONEMAP_LUT_LOG_MAX - TONEMAP_LUT_LOG_MIN) + TONEMAP_LUT_LOG_MIN);
}
