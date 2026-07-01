# World Radiance Cache (camera-following probe clipmap)

Off-screen / far-field / infinite-bounce fallback for screen-space radiance probe GI.
When a screen-probe ray escapes the on-screen region (reaches max trace distance
without a hit), it samples this world cache instead of returning sky. The cache is a
camera-centered clipmap of world probes, each probe storing directional radiance in an
octahedral tile. Techniques below are distilled generically from a reference renderer's
world radiance cache.

## Addressing

- **Clipmap levels.** N nested levels centered on the camera. Level 0 (finest) has the
  smallest world extent; each successive level's extent is `base^i` larger. World
  positions near the camera resolve to a fine level; far positions fall through to a
  coarse level. Level selection with edge fading avoids a hard seam: near a level's
  outer shell, an edge-fade factor is compared against a per-pixel dither so the
  transition to the next level is stochastic rather than a visible boundary.
- **Cell / probe coordinate.** Each level is a uniform grid of `R^3` cells. A probe
  sits at each cell center. Mapping a world position to a probe coord is just
  `coord = floor((P - clipmapCorner) / cellSize)`; the fractional part drives trilinear
  interpolation of the 8 surrounding probes. `cellSize = levelExtent / R`.
- **Grid follows the camera.** Each frame the clipmap corner (min AABB) is recomputed
  from the camera position, so the grid re-centers on the viewer. Probe world positions
  are anchored in world space, not to grid slots.
- **Toroidal / history-preserving update.** The grid is NOT rebuilt from scratch when the
  camera moves. Each frame, every probe from the *previous* frame is re-mapped by its
  stored world position into the *current* frame's clipmap coordinate. If that coord is
  still in bounds and the probe was used recently, the probe (and its atlas slot) is
  reused in place; otherwise it is freed. Net effect: moving the camera only frees the
  cells that scrolled off one side and allocates/re-traces the thin shell of new cells
  that scrolled in — the interior keeps its converged radiance. This is the wrap-around
  ("toroidal") behavior without needing literal modulo addressing, and it keeps temporal
  history valid across camera motion.

## Atlas + indirection storage

Rather than a dense 3D radiance volume per level (which would hit the platform limit on
the number of 3D volume textures and waste memory on empty space), storage is split:

- **Radiance atlas (2D).** A single 2D texture packed as a grid of `A x A` probe tiles
  (default `A = 128`, so up to 16384 live probes). Each tile is an octahedral projection
  of that probe's directional radiance at `probeRes^2` texels (default `32^2`). A probe's
  atlas pixel origin is derived from its linear slot via `(slot & mask, slot >> shift)`.
- **Indirection / page table (3D).** A small 3D texture, `R^3` per level (levels packed
  along X: coord `(x + level*R, y, z)`), stores for each grid cell the atlas slot index of
  its probe, or an INVALID sentinel. Sampling a probe = read the page table for the cell,
  get the slot, address into the atlas. This is the level of indirection that lets only
  *marked* cells (those near geometry / actually queried this frame) own an atlas slot.
- **Marking + allocation.** A mark pass flags cells whose probes are needed this frame
  (cells touched by on-screen geometry / query positions). Newly-marked cells pull a slot
  from a free-list allocator; probes unused for more than `NumFramesToKeepCachedProbes`
  (default 8) frames are freed back. A per-frame **trace budget** (default ~100 probes)
  caps how many probes are (re)traced, so cost is bounded regardless of how many new cells
  appear; distant probes cost less (downsampled) and near probes more (supersampled).

## Chebyshev (variance) occlusion

To stop light leaking through walls when trilinearly blending 8 probes, each probe also
stores, per octahedral direction, two distance moments to the nearest geometry: the mean
hit distance and the mean of squared hit distance. At interpolation time, for a shading
point at distance `d` from the probe along the probe→point direction:

    mean   = moments.x                       // E[dist]
    meanSq = moments.y                       // E[dist^2]
    if (d > mean):                           // point is farther than the probe "sees"
        variance   = abs(mean*mean - meanSq) // Var = E[dist^2] - E[dist]^2
        visibility = variance / (variance + (d - mean)^2)   // Chebyshev one-tail bound
        chebyshev  = max(visibility^3, 0)    // cubed to sharpen the falloff
    else:
        chebyshev  = 1                        // point in front of surface: fully visible
    weight *= max(chebyshev, 0.05)            // small floor so nothing is fully culled

This is Chebyshev's inequality: `P(dist >= d) <= variance / (variance + (d - mean)^2)`,
used as a soft visibility weight. A **min-variance clamp** (a small floor added to
variance, or the `abs(...)` guard above) prevents divide-by-near-zero on flat/degenerate
regions. Additionally a **view/normal bias of ~0.8 * cellSize** pushes the sample position
toward the camera and along the normal before interpolation, so self-occlusion of the
shaded surface does not darken it; the bias is scaled down when the camera is closer than
one cell so nearby surfaces stay crisp. A weight-crush below ~0.2 further suppresses
weak, likely-occluded contributions before normalization.

## Default parameters

| Parameter                        | Default        |
|----------------------------------|----------------|
| Clipmap levels                   | 4 (max 6)      |
| Level scale factor (`base`)      | 2.0            |
| First-level world extent         | ~2500 units    |
| Grid resolution per level `R`    | 48 (`R^3` cells)|
| Probe octahedral resolution      | 32 (`32^2` rays/probe) |
| Atlas resolution (probes/dim)    | 128 (16384 slots) |
| Probes re-traced per frame       | ~100 (budgeted, scaled by update speed) |
| Frames to keep unused probes     | 8              |

## Update & infinite bounce

Each frame: (1) reproject/reuse last-frame probes into the new clipmap (toroidal step);
(2) mark needed cells and allocate slots for newly-marked ones from the free list;
(3) select up to the trace budget of probes to (re)trace, prioritizing stale/near probes;
(4) trace each selected probe's octahedral rays through the scene and shade the hits;
(5) filter/mip and write the radiance + distance-moment atlases.

**Infinite bounce** falls out for free: when tracing a probe's rays, the *hit* shading
samples the *previous frame's* radiance cache for that hit's incoming light (the same
`SampleRadianceCacheInterpolated` path the screen probes use). So frame N's cache
incorporates frame N-1's cached radiance at every ray hit — each frame adds one more
light bounce, converging to multi-bounce GI at a static camera and amortizing over time
under motion. The cache is single-buffered logically but reads last-frame lighting, so
there is no unbounded feedback: contributions are re-derived each frame from geometry.

## DreamCoast mapping

- **Reuse the existing clipmap descriptor.** Our GDF already carries a camera-centered
  clipmap descriptor in `crates/shader/shaders/clipmap.slang` (`ClipMap` with per-level
  AABBs via `cm_min`/`cm_max`, finest→coarsest, `count` levels). Probe placement can reuse
  those same per-level AABBs and `cellSize = (cm_max - cm_min) / R` — no new spatial
  structure needed; the world→level→cell math above maps directly onto `cm_min`/`cm_max`.
- **Start dense, no indirection.** First cut: a *dense* probe atlas with **direct
  indexing** (linear slot = flattened `(level, x, y, z)`, no page table). This is simple
  and correct while grid resolution is modest; skip marking/allocation/free-list initially.
- **Static-camera EMA first.** Accumulate each probe's radiance with an exponential moving
  average across frames. This is valid and converges at a static camera (our current
  parity target) and gives free infinite bounce by sampling last frame's atlas at ray hits.
- **Noted refinements (later).** (1) Toroidal reproject/reuse so history survives camera
  motion; (2) sparse indirection (page table + free-list + marking) to drop the dense-grid
  memory cost and the 3D-volume-count limit; (3) Chebyshev distance-moment occlusion to
  kill wall leaks; (4) octahedral mip chain + view/normal bias for cone-angle sampling.
