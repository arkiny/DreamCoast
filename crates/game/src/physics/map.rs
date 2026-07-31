//! The tile-grid interface and the world <-> tile mapping every query shares.

use glam::Vec2;

use super::circle::{MoveResult, move_circle, nearest_free};
use super::raycast::{RayHit, raycast};

/// A read-only view of a tile grid — the **only** thing the collision layer needs
/// to know about a level.
///
/// The dungeon generator, a hand-authored level, or a unit test each implement
/// this over whatever storage they already have; nothing here allocates, copies,
/// or takes ownership of the grid.
///
/// # Contract
///
/// `is_solid` **must return `true` for out-of-bounds tiles.** The collision
/// routines rely on it: a sealed world means the circle can never leave the map
/// and the raycaster always terminates. An implementation that returns `false`
/// outside its bounds leaks the player into the void.
///
/// It must also be **pure**: the same `(tx, tz)` gives the same answer for the
/// whole duration of a query. Everything in this module is deterministic and
/// side-effect free on top of that assumption.
pub trait SolidMap {
    /// Whether tile `(tx, tz)` blocks movement. Out-of-bounds tiles are solid.
    fn is_solid(&self, tx: i32, tz: i32) -> bool;
}

impl<T: SolidMap + ?Sized> SolidMap for &T {
    #[inline]
    fn is_solid(&self, tx: i32, tz: i32) -> bool {
        (**self).is_solid(tx, tz)
    }
}

/// The tile containing a world position.
///
/// `pos.x` is world **X**, `pos.y` is world **Z** (see the module docs). Tile
/// `(tx, tz)` covers `[tx * tile_size, (tx + 1) * tile_size)` on each axis, so
/// the mapping is a plain `floor` — negative coordinates included.
#[inline]
pub fn world_to_tile(pos: Vec2, tile_size: f32) -> (i32, i32) {
    (
        (pos.x / tile_size).floor() as i32,
        (pos.y / tile_size).floor() as i32,
    )
}

/// The minimum (`-X`, `-Z`) corner of a tile in world space.
#[inline]
pub fn tile_min(tx: i32, tz: i32, tile_size: f32) -> Vec2 {
    Vec2::new(tx as f32 * tile_size, tz as f32 * tile_size)
}

/// The center of a tile in world space.
#[inline]
pub fn tile_center(tx: i32, tz: i32, tile_size: f32) -> Vec2 {
    Vec2::new((tx as f32 + 0.5) * tile_size, (tz as f32 + 0.5) * tile_size)
}

/// Widest tile span (per axis) any single query will scan.
///
/// A guard against absurd inputs (a radius of `1e30`, a `NaN`-adjacent position):
/// the loops stay bounded instead of walking the whole `i32` range.
const MAX_TILE_SPAN: i32 = 64;

/// Inclusive tile range covering `pos` expanded by `extent`, or `None` if the
/// inputs are not finite.
#[inline]
pub(super) fn tile_range(pos: Vec2, extent: f32, tile_size: f32) -> Option<(i32, i32, i32, i32)> {
    if !pos.is_finite() || !extent.is_finite() {
        return None;
    }
    let e = Vec2::splat(extent.max(0.0));
    let (x0, z0) = world_to_tile(pos - e, tile_size);
    let (x1, z1) = world_to_tile(pos + e, tile_size);
    Some((
        x0,
        z0,
        x1.min(x0.saturating_add(MAX_TILE_SPAN)),
        z1.min(z0.saturating_add(MAX_TILE_SPAN)),
    ))
}

/// A grid plus its tile size, so callers stop threading `tile_size` by hand.
///
/// Purely a convenience wrapper: every method forwards to the free function of
/// the same name with `self.tile_size`. Borrowed, `Copy`, zero-cost.
///
/// ```
/// use dreamcoast_game::physics::{GridCollision, SolidMap};
/// use glam::Vec2;
///
/// struct Pillar;
/// impl SolidMap for Pillar {
///     fn is_solid(&self, tx: i32, tz: i32) -> bool {
///         !(0..8).contains(&tx) || !(0..8).contains(&tz) || (tx, tz) == (4, 4)
///     }
/// }
///
/// let grid = GridCollision::new(&Pillar, 2.0);
/// assert!(grid.circle_overlaps(Vec2::new(9.0, 9.0), 0.4)); // inside the pillar
/// assert!(grid.nearest_free(Vec2::new(9.0, 9.0), 0.4).is_some());
/// ```
pub struct GridCollision<'a, M: ?Sized> {
    /// The grid being queried.
    pub map: &'a M,
    /// World-space edge length of one tile (meters).
    pub tile_size: f32,
}

impl<M: ?Sized> Clone for GridCollision<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<M: ?Sized> Copy for GridCollision<'_, M> {}

impl<'a, M: SolidMap + ?Sized> GridCollision<'a, M> {
    /// Bind a map to a tile size.
    #[inline]
    pub fn new(map: &'a M, tile_size: f32) -> Self {
        Self { map, tile_size }
    }

    /// See [`move_circle`](super::move_circle).
    #[inline]
    pub fn move_circle(&self, pos: Vec2, radius: f32, delta: Vec2) -> MoveResult {
        move_circle(self.map, self.tile_size, pos, radius, delta)
    }

    /// See [`raycast`](super::raycast).
    #[inline]
    pub fn raycast(&self, from: Vec2, dir: Vec2, max_dist: f32) -> Option<RayHit> {
        raycast(self.map, self.tile_size, from, dir, max_dist)
    }

    /// See [`circle_overlaps`](super::circle_overlaps).
    #[inline]
    pub fn circle_overlaps(&self, pos: Vec2, radius: f32) -> bool {
        super::circle::circle_overlaps(self.map, self.tile_size, pos, radius)
    }

    /// See [`nearest_free`](super::nearest_free).
    #[inline]
    pub fn nearest_free(&self, pos: Vec2, radius: f32) -> Option<Vec2> {
        nearest_free(self.map, self.tile_size, pos, radius)
    }

    /// Whether the tile containing `pos` is solid (a point test, no radius).
    #[inline]
    pub fn is_solid_at(&self, pos: Vec2) -> bool {
        let (tx, tz) = world_to_tile(pos, self.tile_size);
        self.map.is_solid(tx, tz)
    }

    /// See [`world_to_tile`].
    #[inline]
    pub fn world_to_tile(&self, pos: Vec2) -> (i32, i32) {
        world_to_tile(pos, self.tile_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_to_tile_floors_including_negatives() {
        assert_eq!(world_to_tile(Vec2::new(0.0, 0.0), 2.0), (0, 0));
        assert_eq!(world_to_tile(Vec2::new(1.9, 0.1), 2.0), (0, 0));
        assert_eq!(world_to_tile(Vec2::new(2.0, 2.0), 2.0), (1, 1));
        assert_eq!(world_to_tile(Vec2::new(-0.001, -2.0), 2.0), (-1, -1));
        assert_eq!(world_to_tile(Vec2::new(-2.001, -4.0), 2.0), (-2, -2));
    }

    #[test]
    fn tile_helpers_round_trip() {
        for (tx, tz) in [(0, 0), (3, -7), (-2, 5)] {
            let c = tile_center(tx, tz, 2.0);
            assert_eq!(world_to_tile(c, 2.0), (tx, tz));
            let m = tile_min(tx, tz, 2.0);
            assert_eq!(world_to_tile(m, 2.0), (tx, tz));
        }
    }

    #[test]
    fn tile_range_is_bounded_for_absurd_inputs() {
        let r = tile_range(Vec2::ZERO, 1e30, 1.0).unwrap();
        assert!(r.2 - r.0 <= MAX_TILE_SPAN && r.3 - r.1 <= MAX_TILE_SPAN);
        assert!(tile_range(Vec2::new(f32::NAN, 0.0), 1.0, 1.0).is_none());
        assert!(tile_range(Vec2::ZERO, f32::INFINITY, 1.0).is_none());
    }
}
