// F6P — sky-visibility SH (band-0/1) -> bent normal. ONE definition, shared by every consumer
// (gdf_gi's four reconstruction paths + sdf_cache_light's deferred-parity skylight), replacing
// five hand-copied instances that had already drifted in local variable naming.
//
// The band-1 vector b = INT V(w)·SH_C1·w dw IS the DFAO bent normal (the visibility-weighted
// average unoccluded direction); band-0 sc0 = INT V(w)·SH_C0 dw measures the aperture. Consumers
// read |bent| as CONFIDENCE and fade to their bent-free contract (skyN = n, dot_factor = 1) as it
// falls, so whatever goes into the magnitude decides whether the receiver trusts the direction.
// TWO independent factors belong there. The pre-F6P expression `saturate(sqrt(3)·sc0/|b|)`
// carried NEITHER:
//
//  1) DIRECTIONALITY = |b| / (sqrt(3)·sc0), in [0,1]: 1 for a point aperture, 1/2 for an open
//     hemisphere, -> 0 where V(w) is near-isotropic and band-1 is pure cancellation residue, i.e.
//     "is there a direction here at all". The pre-F6P code computed the RECIPROCAL of this and
//     saturate()d it, which is provably == 1 in every non-degenerate cell: the producer's
//     estimator is a sum of NON-NEGATIVE visibilities (gi_volume.slang, vis >= 0), so
//     |b| <= sqrt(3)·sc0 is a per-SAMPLE triangle inequality — not a convergence property — and
//     every downstream mixer (trilinear, witness renormalise, EMA, fine/coarse blend) is a
//     non-negative-weighted combination that preserves it. The intended fade to the bent-free
//     contract was dead code; the true discriminator is the ratio the other way up.
//
//  2) SUPPORT, in [0,1]: how much of the reconstruction footprint actually voted for this
//     direction. The pre-F6P factor was deliberately SCALE-FREE, so when a cell's only nonzero
//     contributor is a single probe P, its trilinear weight w_P cancelled in BOTH the direction
//     (b/|b|) and the magnitude — the reconstruction was then bit-exactly constant over P's whole
//     2x2x2 support and stepped to zero across a probe-cell face. That is a piecewise-constant
//     bent field whose discontinuities are world-axis planes: the hard-edged bright polygons the
//     skylight painted across curved interior masonry (a plane cuts a barrel vault in a straight
//     ruling, so the edges are straight ON the curve, not merely in projection).
//
//     `skyvis_sh_support` measures it as the participation ratio of the band-0 weights:
//         support = (SUM w·s)^2 / ((SUM w) · SUM w·s^2)
//     = 1 for ANY uniform neighbourhood at ANY absolute openness (so a genuinely dark but
//     uniformly-supported interior keeps its exact legacy confidence — the F6L PT gates measured
//     real directional signal in those cells), and = w_P/SUM w for a lone contributor, ramping
//     continuously to 0 at the far face of its support. Numerator and denominator are both
//     trilinear interpolants of per-corner quantities, so the ratio is C0 ACROSS cell faces too —
//     unlike a mean/peak form, whose peak is piecewise constant and would re-introduce a (smaller)
//     step at every cell boundary. Cauchy-Schwarz bounds it to [0,1] with no tuned constant: the
//     normalisation is the footprint's own second moment.
//
//     NOTE it must be a within-footprint concentration measure, NOT sc0 against a full-sphere
//     reference: the latter measures sky OPENNESS, which indoors is ~0.05 even at a perfectly
//     supported probe, and would fade the bent normal to nothing everywhere — that is deleting
//     the feature, not fixing it.
//
// Callers without per-corner access (the hardware-trilinear paths, the cache's SampleLevel) pass
// support = 1.0 and get the directionality fix only; the manual 2x2x2 paths — which are the
// shipping consumption on content (`P_GI_VOL_OCC`) — compute the real support for free from
// loads they already issue.

// Real SH band-0/1 basis normalisation (same values as the producer's SH_C0/SH_C1).
static const float SKYVIS_SH_C0 = 0.2820947918; // sqrt(1/(4*pi))
static const float SKYVIS_SH_C1 = 0.4886025119; // sqrt(3/(4*pi))
static const float SKYVIS_SH_RATIO = 1.7320508; // SKYVIS_SH_C1 / SKYVIS_SH_C0 = sqrt(3)

// Participation ratio of the band-0 corner weights over one trilinear footprint.
// `s1` = SUM w·s, `s2` = SUM w·s^2, `wsum` = SUM w — all over the SAME (witness-gated) corner set
// that built the coefficients, so the measure describes the population that actually voted.
float skyvis_sh_support(float s1, float s2, float wsum)
{
    return saturate(s1 * s1 / max(s2 * wsum, 1e-12));
}

// `sc0` / `b` may be pre- or post-renormalisation: any common factor cancels in the directionality
// term, so callers pass the raw accumulators and supply `support` (which is NOT scale-free)
// separately. `legacy` restores the exact pre-F6P bytes for the A/B seam (`P_SKYVIS_BENT_FIX=0`).
float3 skyvis_bent_from_sh(float sc0, float3 b, float support, bool legacy)
{
    float bl = length(b);
    if (bl <= 1e-5)
        return float3(0.0, 0.0, 0.0);
    if (legacy)
        return (b / bl) * saturate(SKYVIS_SH_RATIO * sc0 / bl);
    // sc0 > 0 whenever bl > 0 by the same triangle inequality, so max() only guards a denormal.
    float dirness = saturate(bl / (SKYVIS_SH_RATIO * max(sc0, 1e-6)));
    return (b / bl) * (dirness * saturate(support));
}
