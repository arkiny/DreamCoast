//! Circle-vs-tile-grid movement: sliding, depenetration, overlap.

use glam::Vec2;

use super::map::{SolidMap, tile_min, tile_range, world_to_tile};

/// Clearance kept between a resolved circle and the surface it rests on.
///
/// Resolution parks the circle at `radius + CONTACT_SKIN` from the contact
/// feature, so [`circle_overlaps`] is `false` afterwards with room to spare
/// instead of by one float ulp. It is the reason "never ends inside solid" is a
/// hard guarantee rather than a rounding coin flip.
pub const CONTACT_SKIN: f32 = 1e-4;

/// Passes of the per-tile resolver inside one substep.
///
/// One pass fixes a single flat wall; two fix a corner or dead end. The extra
/// budget only matters for pathological starts (spawned deep inside geometry),
/// and the final [`nearest_free`] fallback catches whatever is left.
const MAX_RESOLVE_PASSES: usize = 4;

/// Upper bound on substeps for a single [`move_circle`] call.
///
/// Together with [`substep_len`] this caps how far one call can travel — see
/// [`max_travel`]. A movement call is not a teleport.
const MAX_SUBSTEPS: usize = 256;

/// Displacement below which the achieved/requested comparison stops reporting a
/// blocked axis (see [`MoveResult`]).
const HIT_EPS: f32 = 1e-4;

/// Outcome of one [`move_circle`] call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveResult {
    /// The resolved position. Guaranteed not to overlap any solid tile (unless
    /// the map offers no free space at all within the depenetration search).
    pub pos: Vec2,
    /// The solver removed motion along world **X**: the achieved displacement
    /// differs from the requested one on that axis by more than a hair.
    pub hit_x: bool,
    /// The solver removed motion along world **Z** (`Vec2::y`).
    pub hit_z: bool,
}

impl MoveResult {
    /// Whether either axis was blocked.
    #[inline]
    pub fn hit_any(&self) -> bool {
        self.hit_x || self.hit_z
    }
}

/// Length of one integration substep for the given tile size and radius.
///
/// Two constraints, both about never missing a contact:
/// 1. `<= tile_size / 2` — a step can never carry the center across a
///    one-tile-thick wall (that needs a full `tile_size`), so the endpoint of
///    every substep still overlaps the wall it hit and gets resolved.
/// 2. `<= radius` — a grazing contact is never skipped over by more than the
///    circle's own radius.
///
/// The `tile_size * 0.05` floor keeps the substep count finite for a
/// degenerate (near-zero) radius.
#[inline]
fn substep_len(tile_size: f32, radius: f32) -> f32 {
    ((radius.max(0.0) * 2.0).min(tile_size)).max(tile_size * 0.05) * 0.5
}

/// Farthest one [`move_circle`] call will travel; longer deltas are clamped to
/// this length (direction preserved).
///
/// With the shipping numbers — `tile_size = 2.0`, `radius = 0.4` — this is
/// 102.4 m, i.e. 780x a sprint step at 8 m/s and 1/60 s. Clamping keeps the
/// no-tunneling guarantee unconditional: it does not depend on the caller
/// passing a sane delta.
#[inline]
pub fn max_travel(tile_size: f32, radius: f32) -> f32 {
    substep_len(tile_size, radius) * MAX_SUBSTEPS as f32
}

/// Whether a circle intersects any solid tile.
///
/// Pure geometry against the union of solid tiles — touching exactly (distance
/// == radius) does **not** count as overlap, which is what makes the
/// `radius + CONTACT_SKIN` resting distance safe.
pub fn circle_overlaps<M: SolidMap + ?Sized>(
    map: &M,
    tile_size: f32,
    pos: Vec2,
    radius: f32,
) -> bool {
    debug_assert!(tile_size > 0.0, "tile_size must be positive");
    let radius = radius.max(0.0);
    let Some((x0, z0, x1, z1)) = tile_range(pos, radius, tile_size) else {
        // A non-finite position is not a free position.
        return true;
    };
    let r2 = radius * radius;
    for tz in z0..=z1 {
        for tx in x0..=x1 {
            if !map.is_solid(tx, tz) {
                continue;
            }
            let min = tile_min(tx, tz, tile_size);
            let max = min + Vec2::splat(tile_size);
            let closest = pos.clamp(min, max);
            if (pos - closest).length_squared() < r2 {
                return true;
            }
        }
    }
    false
}

/// Resolve the circle out of one solid tile, returning the corrected **absolute**
/// position (`None` when this tile does not push).
///
/// Closest-point resolution against the tile AABB, with *internal-feature
/// rejection*: a face or corner that is covered by another solid tile is not a
/// feature of the solid **union**, so it must not push. That rejection is what
/// kills the classic "edge catch" — without it, a circle sliding along a flat
/// wall gets a diagonal push from the *corner* of the next tile in the row and
/// loses tangential speed at every seam.
///
/// The returned position is computed from the contact feature, not accumulated
/// from the current one, so resting against a flat wall is bit-stable frame over
/// frame (no jitter).
fn resolve_tile<M: SolidMap + ?Sized>(
    map: &M,
    tile_size: f32,
    pos: Vec2,
    radius: f32,
    tx: i32,
    tz: i32,
) -> Option<Vec2> {
    let min = tile_min(tx, tz, tile_size);
    let max = min + Vec2::splat(tile_size);
    let out = radius + CONTACT_SKIN;

    // Region code: -1 below the tile on this axis, +1 above, 0 inside its span.
    let rx = if pos.x < min.x {
        -1
    } else if pos.x > max.x {
        1
    } else {
        0
    };
    let rz = if pos.y < min.y {
        -1
    } else if pos.y > max.y {
        1
    } else {
        0
    };

    match (rx, rz) {
        // Center inside the tile: escape through the nearest face that is not
        // buried behind another solid tile.
        (0, 0) => {
            let candidates = [
                (
                    map.is_solid(tx - 1, tz),
                    pos.x - min.x,
                    Vec2::new(min.x - out, pos.y),
                ),
                (
                    map.is_solid(tx + 1, tz),
                    max.x - pos.x,
                    Vec2::new(max.x + out, pos.y),
                ),
                (
                    map.is_solid(tx, tz - 1),
                    pos.y - min.y,
                    Vec2::new(pos.x, min.y - out),
                ),
                (
                    map.is_solid(tx, tz + 1),
                    max.y - pos.y,
                    Vec2::new(pos.x, max.y + out),
                ),
            ];
            let mut best: Option<(f32, Vec2)> = None;
            for (buried, depth, target) in candidates {
                if buried {
                    continue;
                }
                if best.is_none_or(|(b, _)| depth < b) {
                    best = Some((depth, target));
                }
            }
            best.map(|(_, target)| target)
        }
        // Face contact along Z (the circle is above or below the tile's span).
        (0, _) => {
            if map.is_solid(tx, tz + rz) {
                return None; // internal face
            }
            let face = if rz < 0 { min.y } else { max.y };
            if (pos.y - face).abs() >= radius {
                return None;
            }
            Some(Vec2::new(pos.x, face + rz as f32 * out))
        }
        // Face contact along X.
        (_, 0) => {
            if map.is_solid(tx + rx, tz) {
                return None; // internal face
            }
            let face = if rx < 0 { min.x } else { max.x };
            if (pos.x - face).abs() >= radius {
                return None;
            }
            Some(Vec2::new(face + rx as f32 * out, pos.y))
        }
        // Corner contact. Only a *convex* corner of the union pushes: if either
        // edge-adjacent neighbor is solid the wall continues past this corner and
        // that neighbor owns the contact (as a face).
        _ => {
            if map.is_solid(tx + rx, tz) || map.is_solid(tx, tz + rz) {
                return None; // internal corner
            }
            let corner = Vec2::new(
                if rx < 0 { min.x } else { max.x },
                if rz < 0 { min.y } else { max.y },
            );
            let d = pos - corner;
            let len = d.length();
            if len >= radius {
                return None;
            }
            let n = if len > 1e-12 {
                d / len
            } else {
                // Exactly on the corner: push along the diagonal of the quadrant
                // the circle came from.
                Vec2::new(rx as f32, rz as f32).normalize()
            };
            Some(corner + n * out)
        }
    }
}

/// Push the circle out of every solid tile it currently overlaps.
fn resolve<M: SolidMap + ?Sized>(map: &M, tile_size: f32, pos: Vec2, radius: f32) -> Vec2 {
    let mut p = pos;
    for _ in 0..MAX_RESOLVE_PASSES {
        let Some((x0, z0, x1, z1)) = tile_range(p, radius, tile_size) else {
            return p;
        };
        let mut changed = false;
        for tz in z0..=z1 {
            for tx in x0..=x1 {
                if !map.is_solid(tx, tz) {
                    continue;
                }
                if let Some(fixed) = resolve_tile(map, tile_size, p, radius, tx, tz)
                    && fixed != p
                {
                    p = fixed;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    p
}

/// Move a circle through the grid with wall sliding.
///
/// `pos`/`delta` are on the XZ plane: `.x` is world X, `.y` is world Z.
///
/// The step is integrated in substeps short enough that a contact can never be
/// skipped (see [`substep_len`]); each substep moves the full 2D delta and then
/// pushes the circle back out along the contact normal of the closest solid
/// feature. Correcting only along the normal is what produces sliding: the
/// tangential component of the motion is left untouched, so running into a wall
/// at a glancing angle costs nothing along the wall.
///
/// Guarantees:
/// * **Never ends inside solid.** A start position that already overlaps is
///   depenetrated first (see [`nearest_free`]), and the result is re-checked.
/// * **Never tunnels.** Independent of `delta`, which is clamped to
///   [`max_travel`].
/// * **Deterministic.** Pure f32 arithmetic, no allocation, no global state.
pub fn move_circle<M: SolidMap + ?Sized>(
    map: &M,
    tile_size: f32,
    pos: Vec2,
    radius: f32,
    delta: Vec2,
) -> MoveResult {
    debug_assert!(tile_size > 0.0, "tile_size must be positive");
    let radius = radius.max(0.0);
    if !pos.is_finite() || !delta.is_finite() {
        return MoveResult {
            pos,
            hit_x: false,
            hit_z: false,
        };
    }

    // Start of frame: get out of anything we are already inside of.
    let mut p = resolve(map, tile_size, pos, radius);
    if circle_overlaps(map, tile_size, p, radius)
        && let Some(free) = nearest_free(map, tile_size, p, radius)
    {
        p = free;
    }

    let step_len = substep_len(tile_size, radius);
    let len = delta.length();
    let delta = if len > step_len * MAX_SUBSTEPS as f32 {
        delta * (step_len * MAX_SUBSTEPS as f32 / len)
    } else {
        delta
    };
    let steps = if len > step_len {
        ((len / step_len).ceil() as usize).clamp(1, MAX_SUBSTEPS)
    } else {
        1
    };
    let step = delta / steps as f32;

    for _ in 0..steps {
        p += step;
        p = resolve(map, tile_size, p, radius);
    }

    // Belt and braces: a circle wedged somewhere the per-tile solver cannot fix
    // (a gap narrower than its diameter) still must not end up inside geometry.
    if circle_overlaps(map, tile_size, p, radius)
        && let Some(free) = nearest_free(map, tile_size, p, radius)
    {
        p = free;
    }

    let achieved = p - pos;
    MoveResult {
        pos: p,
        hit_x: (achieved.x - delta.x).abs() > HIT_EPS,
        hit_z: (achieved.y - delta.y).abs() > HIT_EPS,
    }
}

/// Rings of tiles scanned by [`nearest_free`] before giving up.
const MAX_FREE_RING: i32 = 8;

/// Nearest position at which the circle does not overlap any solid tile.
///
/// Returns `pos` itself when it is already free. Otherwise it scans tiles in
/// growing rings (Chebyshev distance) around the circle and, for each non-solid
/// tile, tries two candidates: the point of the tile's inset rectangle closest
/// to `pos`, and the tile center. The closest validated candidate wins.
///
/// This is the **safe-spawn** helper: place an entity from the generator's tile
/// coordinates, then snap it out of whatever it clipped. Bounded to
/// `MAX_FREE_RING` rings, so a fully sealed region returns `None` instead of
/// searching the level.
///
/// Approximation, stated honestly: two candidates per tile is not the exact
/// closest free point of the free-space region — for a circle jammed into a
/// concave pocket the answer can be a few centimeters off the true minimum. It
/// is always *a* free point, and it is deterministic.
pub fn nearest_free<M: SolidMap + ?Sized>(
    map: &M,
    tile_size: f32,
    pos: Vec2,
    radius: f32,
) -> Option<Vec2> {
    debug_assert!(tile_size > 0.0, "tile_size must be positive");
    let radius = radius.max(0.0);
    if !pos.is_finite() {
        return None;
    }
    if !circle_overlaps(map, tile_size, pos, radius) {
        return Some(pos);
    }

    let (ctx, ctz) = world_to_tile(pos, tile_size);
    let inset = radius + CONTACT_SKIN;
    let mut best: Option<(f32, Vec2)> = None;
    let mut found_ring: Option<i32> = None;

    for ring in 0..=MAX_FREE_RING {
        // One ring beyond the first success, because a diagonal tile of the
        // successful ring can be farther away than an orthogonal tile of the next.
        if found_ring.is_some_and(|r| ring > r + 1) {
            break;
        }
        for dz in -ring..=ring {
            for dx in -ring..=ring {
                if dx.abs().max(dz.abs()) != ring {
                    continue;
                }
                let (tx, tz) = (ctx + dx, ctz + dz);
                if map.is_solid(tx, tz) {
                    continue;
                }
                let min = tile_min(tx, tz, tile_size);
                let max = min + Vec2::splat(tile_size);
                let clamped = Vec2::new(
                    axis_candidate(pos.x, min.x, max.x, inset),
                    axis_candidate(pos.y, min.y, max.y, inset),
                );
                let center = (min + max) * 0.5;
                for cand in [clamped, center] {
                    if circle_overlaps(map, tile_size, cand, radius) {
                        continue;
                    }
                    let d2 = (cand - pos).length_squared();
                    if best.is_none_or(|(b, _)| d2 < b) {
                        best = Some((d2, cand));
                    }
                    found_ring.get_or_insert(ring);
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

/// Clamp one axis into `[lo + inset, hi - inset]`, falling back to the midpoint
/// when the tile is narrower than the circle.
#[inline]
fn axis_candidate(v: f32, lo: f32, hi: f32, inset: f32) -> f32 {
    let (a, b) = (lo + inset, hi - inset);
    if a > b {
        (lo + hi) * 0.5
    } else {
        v.clamp(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::fixture::StringMap;

    const TS: f32 = 2.0;
    const R: f32 = 0.4;
    /// Sprint (8 m/s) at the engine's fixed timestep (1/60 s).
    const SPRINT_STEP: f32 = 8.0 / 60.0;

    /// Room bounded by walls; interior tiles (1..=4, 1..=4) are free.
    fn room() -> StringMap {
        StringMap::new(&["######", "#....#", "#....#", "#....#", "#....#", "######"])
    }

    #[test]
    fn free_move_is_exact() {
        let m = room();
        let r = move_circle(&m, TS, Vec2::new(5.0, 5.0), R, Vec2::new(0.1, -0.05));
        assert_eq!(r.pos, Vec2::new(5.1, 4.95));
        assert!(!r.hit_x && !r.hit_z);
    }

    #[test]
    fn rests_at_radius_plus_skin_from_a_wall() {
        let m = room();
        // Wall row tz=0 occupies z in [0, 2]; free space starts at z = 2.
        let r = move_circle(&m, TS, Vec2::new(5.0, 2.5), R, Vec2::new(0.0, -1.0));
        assert!((r.pos.y - (2.0 + R + CONTACT_SKIN)).abs() < 1e-6, "{:?}", r);
        assert!(r.hit_z && !r.hit_x);
        assert!(!circle_overlaps(&m, TS, r.pos, R));
    }

    /// The headline anti-"edge catch" test: slide along a long flat wall, at full
    /// sprint speed, across many tile seams. Tangential speed must be *exactly*
    /// preserved and the resting line must be bit-identical every frame.
    #[test]
    fn flat_wall_slide_is_smooth_across_seams() {
        // A 24-tile-long corridor: wall row on top, open below.
        let m = StringMap::new(&[
            "########################",
            "........................",
            "........................",
        ]);
        let rest = 2.0 + R + CONTACT_SKIN; // wall face at z = 2
        let mut p = Vec2::new(0.5, rest);
        let mut prev_x = p.x;
        for i in 0..300 {
            // Push into the wall every frame while running along it.
            let r = move_circle(&m, TS, p, R, Vec2::new(SPRINT_STEP, -0.05));
            assert_eq!(r.pos.y, rest, "step {i}: rest line drifted: {:?}", r.pos);
            // Exact: the solver corrects along the contact normal only, so the
            // tangential axis is left bit-for-bit alone at every seam.
            assert_eq!(
                r.pos.x,
                prev_x + SPRINT_STEP,
                "step {i}: tangential speed lost at a seam"
            );
            assert!(r.hit_z && !r.hit_x, "step {i}: spurious axis flag {r:?}");
            prev_x = r.pos.x;
            p = r.pos;
        }
        // 300 frames x 0.1333 m = 40 m, i.e. 20 tile seams crossed.
        assert!(p.x > 40.0, "{p:?}");
    }

    /// Same, but sliding along a vertical wall (X faces) and moving backwards.
    #[test]
    fn vertical_wall_slide_is_smooth() {
        let m = StringMap::new(&[
            "#..", "#..", "#..", "#..", "#..", "#..", "#..", "#..", "#..", "#..",
        ]);
        let rest = 2.0 + R + CONTACT_SKIN; // wall column tx=0 ends at x = 2
        let mut p = Vec2::new(rest, 18.0);
        for _ in 0..100 {
            let r = move_circle(&m, TS, p, R, Vec2::new(-0.05, -SPRINT_STEP));
            assert_eq!(r.pos.x, rest);
            assert!(r.hit_x && !r.hit_z);
            p = r.pos;
        }
    }

    #[test]
    fn diagonal_into_wall_preserves_tangential_component() {
        let m = room();
        let start = Vec2::new(5.0, 2.0 + R + CONTACT_SKIN);
        let d = Vec2::new(SPRINT_STEP, -SPRINT_STEP);
        let r = move_circle(&m, TS, start, R, d);
        assert_eq!(r.pos.x, start.x + d.x, "tangential motion was clipped");
        assert_eq!(r.pos.y, start.y);
        assert!(r.hit_z && !r.hit_x);
    }

    #[test]
    fn dead_end_blocks_both_axes() {
        let m = room();
        // Interior corner at (2, 2): walls on -X and -Z.
        let start = Vec2::new(2.0 + R + CONTACT_SKIN, 2.0 + R + CONTACT_SKIN);
        let r = move_circle(&m, TS, start, R, Vec2::new(-0.3, -0.3));
        assert!(r.hit_x && r.hit_z, "{r:?}");
        assert_eq!(r.pos, start);
        assert!(!circle_overlaps(&m, TS, r.pos, R));
    }

    /// A lone pillar: grazing its convex corner must resolve along the *corner
    /// normal* (both components move), not snap to an axis.
    #[test]
    fn convex_corner_resolves_along_the_corner_normal() {
        let m = StringMap::new(&["....", ".#..", "....", "...."]);
        // Pillar tile (1,1) covers x,z in [2,4].
        let corner = Vec2::new(2.0, 2.0);
        // Sit inside the corner's circle, diagonally out from it.
        let start = corner + Vec2::new(-0.1, -0.1);
        let r = move_circle(&m, TS, start, R, Vec2::ZERO);
        let d = r.pos - corner;
        assert!(
            (d.length() - (R + CONTACT_SKIN)).abs() < 1e-5,
            "not pushed to the corner normal distance: {d:?}"
        );
        // Axis-snapping would leave one component at -0.1.
        assert!(d.x < -0.2 && d.y < -0.2, "axis-snapped instead: {d:?}");
        assert!(!circle_overlaps(&m, TS, r.pos, R));
    }

    /// Grazing past a pillar corner must not permanently catch: the circle keeps
    /// making forward progress and slips past.
    #[test]
    fn corner_graze_does_not_catch() {
        let m = StringMap::new(&["......", ".#....", "......", "......"]);
        let mut p = Vec2::new(0.5, 2.0 - 0.15); // just below the pillar's -Z face
        let mut stalls = 0;
        for _ in 0..60 {
            let r = move_circle(&m, TS, p, R, Vec2::new(SPRINT_STEP, 0.0));
            assert!(!circle_overlaps(&m, TS, r.pos, R));
            // The corner push may cost some forward motion in the frames where
            // the circle is actually wedged against it, but never all of it for
            // long: count the frames that made less than a tenth of the step.
            if r.pos.x - p.x < SPRINT_STEP * 0.1 {
                stalls += 1;
            }
            p = r.pos;
        }
        assert!(stalls <= 2, "caught on the corner for {stalls} frames");
        assert!(p.x > 5.0, "no forward progress past the pillar: {p:?}");
    }

    #[test]
    fn no_tunneling_at_absurd_delta() {
        // Wall row at tz = 1; the circle starts below it and charges through.
        let m = StringMap::new(&["......", "######", "......", "......"]);
        for delta in [3.0f32, 30.0, 300.0, max_travel(TS, R) * 4.0] {
            let start = Vec2::new(5.0, 5.0);
            let r = move_circle(&m, TS, start, R, Vec2::new(0.0, -delta));
            assert!(!circle_overlaps(&m, TS, r.pos, R), "delta {delta}: {r:?}");
            assert!(
                r.pos.y >= 4.0 + R,
                "delta {delta}: tunneled through the wall: {:?}",
                r.pos
            );
        }
    }

    #[test]
    fn no_tunneling_from_any_angle() {
        let m = StringMap::new(&["#####", "#...#", "#...#", "#...#", "#####"]);
        let center = Vec2::new(5.0, 5.0);
        for i in 0..360 {
            let a = (i as f32).to_radians();
            let dir = Vec2::new(a.cos(), a.sin());
            let r = move_circle(&m, TS, center, R, dir * 3.0);
            assert!(!circle_overlaps(&m, TS, r.pos, R), "angle {i}: {r:?}");
            // The room's free interior is x,z in [2, 8].
            assert!(
                r.pos.x > 2.0 && r.pos.x < 8.0 && r.pos.y > 2.0 && r.pos.y < 8.0,
                "angle {i}: escaped the room: {:?}",
                r.pos
            );
        }
    }

    #[test]
    fn depenetrates_from_inside_a_wall() {
        let m = room();
        // Dead center of a wall tile.
        let inside = Vec2::new(1.0, 1.0);
        assert!(circle_overlaps(&m, TS, inside, R));
        let free = nearest_free(&m, TS, inside, R).expect("free space exists");
        assert!(!circle_overlaps(&m, TS, free, R));

        let r = move_circle(&m, TS, inside, R, Vec2::ZERO);
        assert!(!circle_overlaps(&m, TS, r.pos, R), "{r:?}");
    }

    #[test]
    fn depenetration_prefers_the_near_side() {
        let m = room();
        // Barely clipped into the top wall (face at z = 2) from below.
        let start = Vec2::new(5.0, 2.0 + 0.1);
        let r = move_circle(&m, TS, start, R, Vec2::ZERO);
        assert_eq!(r.pos.x, start.x);
        assert!((r.pos.y - (2.0 + R + CONTACT_SKIN)).abs() < 1e-6, "{r:?}");
    }

    #[test]
    fn nearest_free_returns_the_input_when_already_free() {
        let m = room();
        let p = Vec2::new(5.0, 5.0);
        assert_eq!(nearest_free(&m, TS, p, R), Some(p));
    }

    #[test]
    fn nearest_free_gives_up_in_a_sealed_map() {
        struct Solid;
        impl SolidMap for Solid {
            fn is_solid(&self, _tx: i32, _tz: i32) -> bool {
                true
            }
        }
        assert_eq!(nearest_free(&Solid, TS, Vec2::new(1.0, 1.0), R), None);
        // ...and moving inside it is a no-op rather than a panic or a leak.
        let r = move_circle(&Solid, TS, Vec2::new(1.0, 1.0), R, Vec2::new(1.0, 1.0));
        assert!(r.pos.is_finite());
    }

    #[test]
    fn nearest_free_handles_a_radius_wider_than_a_tile() {
        // Tile size 1, radius 0.9: only the 2x2 free pocket can hold the circle
        // loosely, and the helper must still return a genuinely free point.
        let m = StringMap::new(&["#####", "#...#", "#...#", "#...#", "#####"]);
        let free = nearest_free(&m, 1.0, Vec2::new(0.5, 0.5), 0.45).expect("pocket exists");
        assert!(!circle_overlaps(&m, 1.0, free, 0.45));
    }

    #[test]
    fn narrow_gap_between_two_tiles_never_ends_inside() {
        // A one-tile gap in a wall, with a circle wider than half the tile.
        let m = StringMap::new(&["......", "##.###", "......", "......"]);
        let mut p = Vec2::new(5.0, 5.0);
        for _ in 0..40 {
            let r = move_circle(&m, 2.0, p, 0.9, Vec2::new(0.11, -0.3));
            assert!(!circle_overlaps(&m, 2.0, r.pos, 0.9), "{r:?}");
            p = r.pos;
        }
    }

    #[test]
    fn zero_and_negative_radius_are_tolerated() {
        let m = room();
        let r = move_circle(&m, TS, Vec2::new(5.0, 5.0), 0.0, Vec2::new(0.5, 0.0));
        assert!(!circle_overlaps(&m, TS, r.pos, 0.0));
        let r = move_circle(&m, TS, Vec2::new(5.0, 5.0), -1.0, Vec2::new(0.5, 0.0));
        assert!(r.pos.is_finite());
    }

    #[test]
    fn non_finite_inputs_are_inert() {
        let m = room();
        let p = Vec2::new(5.0, 5.0);
        let r = move_circle(&m, TS, p, R, Vec2::new(f32::NAN, 0.0));
        assert_eq!(r.pos, p);
        let r = move_circle(&m, TS, Vec2::new(f32::INFINITY, 0.0), R, Vec2::ZERO);
        assert!(!r.pos.is_finite()); // returned untouched, not silently teleported
    }

    /// Deterministic xorshift — a dependency-free stand-in for `rand` in tests.
    struct Rng(u32);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }
        /// Uniform in `[-1, 1)`.
        fn signed(&mut self) -> f32 {
            (self.next_u32() >> 8) as f32 / 8_388_608.0 - 1.0
        }
    }

    #[test]
    fn fuzz_10k_moves_never_end_inside_solid() {
        let m = StringMap::new(&[
            "##########",
            "#....#...#",
            "#.##.#.#.#",
            "#.#..#.#.#",
            "#.#.##.#.#",
            "#.#....#.#",
            "#.######.#",
            "#........#",
            "#.####.#.#",
            "##########",
        ]);
        let ts = 1.0;
        let radius = 0.3;
        let mut rng = Rng(0x1234_5678);
        let mut p = Vec2::new(1.5, 1.5);
        assert!(!circle_overlaps(&m, ts, p, radius));
        for i in 0..10_000 {
            // Mostly game-speed steps, with occasional absurd ones mixed in.
            let scale = if i % 97 == 0 { 3.0 } else { 0.15 };
            let d = Vec2::new(rng.signed(), rng.signed()) * scale;
            let r = move_circle(&m, ts, p, radius, d);
            assert!(
                !circle_overlaps(&m, ts, r.pos, radius),
                "iteration {i}: ended inside solid at {:?} (from {p:?}, delta {d:?})",
                r.pos
            );
            // The maze is sealed: the circle can never leave the 10x10 grid.
            assert!(
                r.pos.x > 1.0 - 1e-3
                    && r.pos.x < 9.0 + 1e-3
                    && r.pos.y > 1.0 - 1e-3
                    && r.pos.y < 9.0 + 1e-3,
                "iteration {i}: escaped the maze at {:?}",
                r.pos
            );
            p = r.pos;
        }
    }

    #[test]
    fn is_deterministic() {
        let m = room();
        let run = || {
            let mut p = Vec2::new(3.0, 3.0);
            for i in 0..500 {
                let a = i as f32 * 0.37;
                p = move_circle(&m, TS, p, R, Vec2::new(a.cos(), a.sin()) * 0.3).pos;
            }
            p
        };
        assert_eq!(run(), run());
    }
}
