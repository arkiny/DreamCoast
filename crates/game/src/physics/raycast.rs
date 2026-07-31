//! Grid raycasting by incremental voxel traversal (Amanatides & Woo, 1987).

use glam::Vec2;

use super::map::{SolidMap, world_to_tile};

/// What a ray ran into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// Distance along the ray, in world units, measured with a **normalized**
    /// direction regardless of the length of the `dir` passed in.
    pub dist: f32,
    /// The solid tile that was entered.
    pub tile: (i32, i32),
    /// Unit face normal of the crossed face, pointing back toward the ray origin.
    /// Always axis aligned: `(±1, 0)` or `(0, ±1)`.
    pub normal: Vec2,
}

/// Cast a ray against the solid tiles of the grid.
///
/// `from`/`dir` are on the XZ plane: `.x` is world X, `.y` is world Z. `dir`
/// does not need to be normalized; `dist` is reported in world units either way.
///
/// Exact grid traversal — no marching, no step size, no missed thin walls. The
/// ray visits tiles strictly in order of entry, so the first solid tile reported
/// is the true nearest one, and boundary crossings are computed from the exact
/// parametric distance to each grid line rather than from a sampled position.
///
/// Behavior at the edges:
/// * **Starting inside solid** returns `dist = 0`, the containing tile, and
///   `-dir` as the normal (there is no crossed face to report).
/// * **Axis-aligned rays** never divide by zero: the perpendicular axis simply
///   never produces a crossing.
/// * **A perfect corner crossing** (both axes crossing at the same distance,
///   e.g. an exactly 45-degree ray through a lattice point) takes the **X**
///   crossing first and the Z crossing immediately after, both at the same
///   `dist`. So a diagonal ray never leaks *through* a corner: if the tile on
///   the X side is solid it is reported with an X normal, otherwise the ray
///   enters the diagonal tile via the Z crossing and reports a Z normal.
/// * A hit exactly at `max_dist` counts; anything beyond does not.
///
/// Returns `None` for a zero-length/non-finite direction, a negative `max_dist`,
/// or when nothing solid is met within range.
pub fn raycast<M: SolidMap + ?Sized>(
    map: &M,
    tile_size: f32,
    from: Vec2,
    dir: Vec2,
    max_dist: f32,
) -> Option<RayHit> {
    debug_assert!(tile_size > 0.0, "tile_size must be positive");
    if !from.is_finite() || !dir.is_finite() || max_dist.is_nan() || max_dist < 0.0 {
        return None;
    }
    let len = dir.length();
    if len <= 0.0 {
        return None;
    }
    let d = dir / len;

    let (mut tx, mut tz) = world_to_tile(from, tile_size);
    if map.is_solid(tx, tz) {
        return Some(RayHit {
            dist: 0.0,
            tile: (tx, tz),
            normal: -d,
        });
    }

    let step_x = if d.x > 0.0 {
        1
    } else if d.x < 0.0 {
        -1
    } else {
        0
    };
    let step_z = if d.y > 0.0 {
        1
    } else if d.y < 0.0 {
        -1
    } else {
        0
    };

    // Distance along the ray to the first grid line on each axis, and the
    // distance between consecutive lines.
    let (mut t_max_x, t_delta_x) = axis_setup(from.x, tx, d.x, step_x, tile_size);
    let (mut t_max_z, t_delta_z) = axis_setup(from.y, tz, d.y, step_z, tile_size);

    // Bound: each tile crossing advances at least one grid line on one axis.
    let max_iter = ((max_dist / tile_size) * 2.0 + 4.0).min(1.0e6) as usize;
    for _ in 0..max_iter {
        let (t, normal) = if t_max_x <= t_max_z {
            let t = t_max_x;
            tx += step_x;
            t_max_x += t_delta_x;
            (t, Vec2::new(-step_x as f32, 0.0))
        } else {
            let t = t_max_z;
            tz += step_z;
            t_max_z += t_delta_z;
            (t, Vec2::new(0.0, -step_z as f32))
        };
        if t > max_dist {
            return None;
        }
        if map.is_solid(tx, tz) {
            return Some(RayHit {
                dist: t,
                tile: (tx, tz),
                normal,
            });
        }
    }
    None
}

/// Per-axis DDA setup: distance to the first grid line and between lines.
#[inline]
fn axis_setup(origin: f32, tile: i32, d: f32, step: i32, tile_size: f32) -> (f32, f32) {
    if step == 0 {
        return (f32::INFINITY, f32::INFINITY);
    }
    let inv = 1.0 / d.abs();
    let boundary = if step > 0 {
        (tile + 1) as f32 * tile_size - origin
    } else {
        origin - tile as f32 * tile_size
    };
    (boundary * inv, tile_size * inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::fixture::StringMap;

    const TS: f32 = 2.0;

    /// 6x6 room; interior tiles (1..=4, 1..=4) are free.
    fn room() -> StringMap {
        StringMap::new(&["######", "#....#", "#....#", "#....#", "#....#", "######"])
    }

    #[test]
    fn axis_aligned_distances_and_normals() {
        let m = room();
        let from = Vec2::new(5.0, 5.0);

        let h = raycast(&m, TS, from, Vec2::X, 100.0).unwrap();
        assert_eq!(h.dist, 5.0); // wall column tx=5 starts at x = 10
        assert_eq!(h.tile, (5, 2));
        assert_eq!(h.normal, Vec2::new(-1.0, 0.0));

        let h = raycast(&m, TS, from, Vec2::NEG_X, 100.0).unwrap();
        assert_eq!(h.dist, 3.0); // wall column tx=0 ends at x = 2
        assert_eq!(h.tile, (0, 2));
        assert_eq!(h.normal, Vec2::new(1.0, 0.0));

        let h = raycast(&m, TS, from, Vec2::Y, 100.0).unwrap();
        assert_eq!(h.dist, 5.0);
        assert_eq!(h.tile, (2, 5));
        assert_eq!(h.normal, Vec2::new(0.0, -1.0));

        let h = raycast(&m, TS, from, Vec2::NEG_Y, 100.0).unwrap();
        assert_eq!(h.dist, 3.0);
        assert_eq!(h.tile, (2, 0));
        assert_eq!(h.normal, Vec2::new(0.0, 1.0));
    }

    #[test]
    fn direction_need_not_be_normalized() {
        let m = room();
        let a = raycast(&m, TS, Vec2::new(5.0, 5.0), Vec2::X, 100.0).unwrap();
        let b = raycast(&m, TS, Vec2::new(5.0, 5.0), Vec2::X * 37.0, 100.0).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn exact_diagonal_hits_the_corner_and_resolves_on_x() {
        let m = room();
        // From the center of tile (1,1) toward the room's far corner: an exactly
        // 45-degree ray that threads lattice points.
        let from = Vec2::new(3.0, 3.0);
        let dir = Vec2::new(1.0, 1.0).normalize();
        let h = raycast(&m, TS, from, dir, 100.0).unwrap();
        // Free interior ends at (10, 10); the ray travels 7 units on each axis.
        let expected = (7.0f32 * 7.0 + 7.0 * 7.0).sqrt();
        assert!((h.dist - expected).abs() < 1e-4, "{h:?}");
        assert_eq!(h.tile, (5, 4), "{h:?}");
        assert_eq!(h.normal, Vec2::new(-1.0, 0.0), "tie must resolve on X");
    }

    /// A diagonal ray aimed exactly at a lone pillar's corner must still hit it —
    /// it must not slip diagonally between the two free neighbors.
    #[test]
    fn diagonal_into_a_corner_pillar() {
        // Single solid tile at (2,2) => x,z in [4,6] with tile size 2.
        let m = StringMap::new(&[".....", ".....", "..#..", ".....", "....."]);
        let from = Vec2::new(1.0, 1.0);
        let h = raycast(&m, TS, from, Vec2::new(1.0, 1.0), 100.0).unwrap();
        let expected = (3.0f32 * 3.0 + 3.0 * 3.0).sqrt(); // corner (4,4)
        assert!((h.dist - expected).abs() < 1e-4, "{h:?}");
        assert_eq!(h.tile, (2, 2));
        // The X-side neighbor (2,1) is free, so the pillar is entered through the
        // Z crossing of the tie.
        assert_eq!(h.normal, Vec2::new(0.0, -1.0));
    }

    #[test]
    fn starting_inside_solid_returns_zero() {
        let m = room();
        let dir = Vec2::new(0.6, 0.8);
        let h = raycast(&m, TS, Vec2::new(1.0, 1.0), dir, 100.0).unwrap();
        assert_eq!(h.dist, 0.0);
        assert_eq!(h.tile, (0, 0));
        assert_eq!(h.normal, -dir.normalize());
    }

    #[test]
    fn starting_out_of_bounds_is_solid() {
        let m = room();
        let h = raycast(&m, TS, Vec2::new(-100.0, -100.0), Vec2::X, 10.0).unwrap();
        assert_eq!(h.dist, 0.0);
    }

    #[test]
    fn max_dist_cutoff_is_exact_on_both_sides() {
        let m = room();
        let from = Vec2::new(5.0, 5.0); // wall at x = 10 => dist 5
        assert!(raycast(&m, TS, from, Vec2::X, 4.999).is_none());
        assert!(raycast(&m, TS, from, Vec2::X, 5.0).is_some());
        assert!(raycast(&m, TS, from, Vec2::X, 5.001).is_some());
    }

    #[test]
    fn starting_exactly_on_a_tile_boundary() {
        let m = room();
        // x = 2.0 is the boundary between wall tile 0 and free tile 1.
        let h = raycast(&m, TS, Vec2::new(2.0, 5.0), Vec2::NEG_X, 10.0).unwrap();
        assert_eq!(h.dist, 0.0, "the wall is touching the origin");
        assert_eq!(h.tile, (0, 2));
        assert_eq!(h.normal, Vec2::new(1.0, 0.0));

        // The same origin pointing away starts in the free tile.
        let h = raycast(&m, TS, Vec2::new(2.0, 5.0), Vec2::X, 100.0).unwrap();
        assert_eq!(h.dist, 8.0);
        assert_eq!(h.tile, (5, 2));
    }

    #[test]
    fn passes_through_a_gap_and_hits_the_far_wall() {
        // Row tz=2 is a wall with a hole at tx=2.
        let m = StringMap::new(&["#####", "#...#", "##.##", "#...#", "#####"]);
        let from = Vec2::new(5.0, 3.0); // center of the gap column
        let h = raycast(&m, TS, from, Vec2::Y, 100.0).unwrap();
        assert_eq!(h.tile, (2, 4), "the ray must pass through the gap");
        assert_eq!(h.dist, 5.0);
    }

    #[test]
    fn degenerate_inputs_return_none() {
        let m = room();
        let from = Vec2::new(5.0, 5.0);
        assert!(raycast(&m, TS, from, Vec2::ZERO, 10.0).is_none());
        assert!(raycast(&m, TS, from, Vec2::X, -1.0).is_none());
        assert!(raycast(&m, TS, from, Vec2::new(f32::NAN, 0.0), 10.0).is_none());
        assert!(raycast(&m, TS, Vec2::new(f32::NAN, 0.0), Vec2::X, 10.0).is_none());
        assert!(raycast(&m, TS, from, Vec2::X, f32::NAN).is_none());
    }

    #[test]
    fn nothing_within_range_returns_none() {
        // An open (non-sealed) map to prove the range test, not the walls.
        struct Open;
        impl SolidMap for Open {
            fn is_solid(&self, _tx: i32, _tz: i32) -> bool {
                false
            }
        }
        assert!(raycast(&Open, TS, Vec2::ZERO, Vec2::X, 1000.0).is_none());
    }

    #[test]
    fn agrees_with_the_circle_sweep_on_open_ground() {
        // A ray fired along a corridor must report the same wall the mover ends
        // against, minus the radius. Cross-check of the two traversals.
        let m = room();
        let from = Vec2::new(5.0, 5.0);
        let h = raycast(&m, TS, from, Vec2::X, 100.0).unwrap();
        let r = crate::physics::move_circle(&m, TS, from, 0.4, Vec2::new(100.0, 0.0));
        assert!(
            (r.pos.x - (from.x + h.dist - 0.4)).abs() < 1e-3,
            "{h:?} {r:?}"
        );
    }
}
