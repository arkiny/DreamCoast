# Screen-Probe GI Optimization — Design Doc

Techniques extracted from a mature reference renderer's screen-space radiance-probe
diffuse-GI implementation, described generically. DreamCoast today: 1 probe / 16px
tile, 8x8 octahedral trace per probe, cross-probe spatial filter, then every full-res
pixel gathers 2x2 neighbor probes cosine-integrating each 8x8 tile (~256 taps/pixel).
Notably the reference uses the SAME probe density (16px tile) and SAME trace resolution
(8x8) as DreamCoast, so its wins are structural, not from coarser probes.

## 1. Screen tile classification + indirect dispatch
The reference tags each integrate tile with a bitmask of the shading complexity it
contains (simple diffuse / needs-BRDF-importance-sample / full), computed from the
GBuffer material of the pixels under the tile. A build-lists pass reads the per-tile
bitmask, uses wave intrinsics (WaveActiveCountBits / prefix-count) to compact tiles into
one contiguous list PER class, and writes indirect-dispatch args. Each class is then
dispatched with its own (cheaper or costlier) shader permutation; fully-empty/unlit
tiles go to a separate "clear" list and are skipped entirely. So the expensive
importance-sampled BRDF integration only runs on the minority of tiles that need it,
while the bulk run a cheap preintegrated path. Classification granularity is an 8x8
Z-order tile; the fast path is chosen unless a material flag (anisotropy, subsurface,
per-pixel bent-normal occlusion) forces the BRDF path.

## 2. Adaptive ray budget / importance sampling + adaptive probes
Two independent mechanisms:
- **Structured importance sampling of trace rays.** Before tracing, per probe it builds
  a PDF over the octahedral trace directions from (a) a 3-band SH fit of the BRDF and
  (b) reprojected previous-frame probe radiance (a lighting PDF). Rays whose PDF is below
  a floor (MinPDFToTrace = 0.1) are culled and their budget is redistributed: a parallel
  in-group sort merges 3 low-PDF rays to subdivide 1 high-PDF ray into a finer octahedral
  mip (adaptive octahedral resolution). Ray *count* stays fixed (still 8x8 = 64) but rays
  concentrate where BRDF*lighting is bright. Defaults: importance sampling ON, NumLevels=1
  (one subdivision level in shipping; reference mode uses 3), BRDF PDF at an 8x8 oct map.
- **Adaptive probe placement.** Uniform probes sit on the 16px grid. A mark pass samples
  N points per uniform tile and, where the bilinear interpolation weight from the 4
  surrounding uniform probes falls below a threshold (i.e. depth/plane/normal
  discontinuities — edges, thin geometry), it spawns an extra "adaptive" probe, appended
  to the atlas. Budget is capped (adaptive allocation fraction = 0.5 of uniform count).
  This adds probes only at geometrically-complex tiles instead of globally raising density.

## 3. Downsampled integration: preintegrate radiance -> irradiance ONCE (the big win)
This is the core cost-cut and it is the DEFAULT path (diffuse integral method = 0).
Two-stage:
1. **Trace** at 8x8 octahedral radiance per probe (resampled to a gather resolution,
   scale 1.0 => also ~8x8), spatially filtered.
2. **Preintegrate per probe, once:** a filtering compute pass projects each probe's
   octahedral radiance into a directional irradiance representation shared by all pixels
   that later read the probe. Two interchangeable formats:
   - **SH3** (3-band spherical harmonics, 9 RGB coeffs) — the default-quality, cheapest
     per-pixel lookup; per-pixel diffuse = one SH-dot with the cosine transfer of the
     pixel normal.
   - **Octahedral irradiance map** (format = 1, the shipping default): a tiny **6x6**
     per-probe irradiance oct (8x8 with a 1-texel border for bilinear wrap), produced by
     convolving the radiance SH with the diffuse transfer kernel per output texel.
3. **Per-pixel gather becomes a cheap directional lookup:** each pixel maps its normal to
   an oct UV and does a bilinear tap of the 6x6 irradiance oct (or a 9-coeff SH eval),
   times the 2x2 probe interpolation weights. This replaces the ~256-tap hemisphere
   integral with 4 probes x (1 bilinear tap or 1 SH-dot). The heavy cosine integral is
   paid once per PROBE (thousands) instead of once per PIXEL (millions).

## 4. Temporal amortization
- **Rotating trace directions.** The per-probe octahedral texel-center offset is jittered
  every frame by a blue-noise / Hammersley(frameIndex%8) sequence, so the fixed 8x8 rays
  sweep different directions across frames; temporal accumulation reconstructs a
  higher effective angular resolution for free.
- **Screen-tile jitter.** Probe screen positions are jittered within the 16px tile per
  frame (Hammersley over 8), so probe placement rotates too.
- **History accumulation.** A temporal reprojection pass blends into history with
  alpha = 1/(1+NumFramesAccumulated), NumFramesAccumulated capped at 10. A "fast update"
  disocclusion/moving-lighting term shortens the history (alpha floor) where lighting
  changes, and history is rejected on plane-distance / relative-depth mismatch. The
  importance-sampling lighting PDF (see #2) also reprojects last-frame probe radiance.

## 5. Downsampled integrate + bilateral upsample
The integrate pass itself runs at a downsample factor (IntegrateDownsampleFactor,
clamp 1-2; shipping default 1, "2 makes this pass faster, blurs fine normal-map detail").
At factor 2 it integrates at quarter-res (half per axis) with a per-tile jitter, then a
following pass bilaterally upsamples to full res using depth/normal weights. Combined with
#3 this makes the per-full-res-pixel work approach a single bilinear fetch.

## DreamCoast mapping / priority
Ranked by expected cost reduction for our exact pipeline (16px tile, 8x8 trace, ~256-tap
gather at full res):

1. **#3 Per-probe irradiance preintegration (SH3 or 6x6 oct) — BIGGEST WIN.** Confirmed:
   our ~256-tap per-pixel gather is exactly the cost this eliminates. Add a filtering pass
   that converts each probe's 8x8 octahedral radiance into either 9 SH3 coeffs or a 6x6
   irradiance oct; the INTEGRATE pass then becomes 4-probe x 1-tap. This is the single
   largest reduction and does not touch trace cost or quality (it's the reference default).
   Start with SH3 (smallest storage, no border/wrap logic); move to 6x6 oct if directional
   sharpness on flat walls needs it.
2. **#5 Downsampled integrate (factor 2) + bilateral upsample.** Cheap, stacks on top of
   #1: quarter the integrate invocations, depth/normal upsample. Second-biggest, low risk.
3. **#1 Tile classification / indirect dispatch.** Skips unlit/simple tiles; big win when
   scenes have sky/empty regions. Needs indirect-dispatch + wave compaction plumbing, so
   more engineering per unit gain than #1/#2. Worth it after the above.
4. **#4 Temporal rotation of the 8x8 rays.** Lets us DROP trace resolution (e.g. 8x8 -> 6x6
   or 4x4) while holding quality via accumulation — attacks the dominant TRACE cost, but
   requires solid reprojection + history validity first. High ceiling, higher risk.
5. **#2 Structured importance sampling / adaptive probes.** Most complex (per-probe PDF,
   in-group ray sort, adaptive atlas). Best reserved for later; adaptive probes are the
   quality lever for edges rather than a raw cost cut.

Concrete defaults found (verified in source): trace oct = 8x8; probe downsample = 16px;
gather oct scale = 1.0; irradiance format default = octahedral (6x6, +1 border);
diffuse integral method default = 0 (preintegrated); integrate downsample = 1 (2 avail);
temporal max frames = 10; importance sampling ON, NumLevels = 1, MinPDFToTrace = 0.1,
BRDF PDF oct = 8x8; adaptive probe allocation fraction = 0.5.
