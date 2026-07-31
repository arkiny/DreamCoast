//! The generated [`TileGrid`] as the game's collision world
//! (`docs/game-framework-plan.md` §3.4).
//!
//! `dreamcoast_game::physics` knows nothing about this game: it asks a [`SolidMap`]
//! whether tile `(tx, tz)` blocks, and it defines tile `(tx, tz)` as covering
//! `[tx * TILE_SIZE, (tx + 1) * TILE_SIZE)` on each axis — i.e. **tile (0, 0) starts at
//! the coordinate-system origin**.
//!
//! # The alignment story
//!
//! The dungeon's world mapping is *not* that: [`TileGrid`] is **centred on the world
//! origin**, so its tile `(0, 0)` starts at `(world_min_x, world_min_z)` — for the
//! default 40x40 dungeon, `(-40, -40)` m. Wiring the mover to raw world coordinates
//! would therefore read the tile 20 columns and 20 rows away from the one the geometry
//! was built on: walls would feel shifted by half a dungeon.
//!
//! Two ways to reconcile that; only one of them is correct in general:
//!
//! * Offset the *tile indices* inside the [`SolidMap`] impl (`tx + width / 2`). That is
//!   exact only when the grid's dimensions are **even** — an odd width puts the world
//!   origin in the middle of a tile column, and the adapter silently lands half a tile
//!   off. The default is 40x40, so the bug would hide until the day someone generates a
//!   41-wide floor.
//! * Offset the *positions*: run collision in **grid-local space**, whose origin is the
//!   grid's own `(0, 0)` corner. Then the physics tile index and the grid tile index are
//!   the same integer by construction, for every grid size, and the [`SolidMap`] impl is
//!   a straight forward of [`TileGrid::is_solid`] with no arithmetic to get wrong.
//!
//! This module takes the second road. [`to_collision`] / [`to_world`] are the only
//! places the two spaces meet, and [`tests`](self#tests) pin the alignment against the
//! *geometry*: a wall tile blocks exactly across the world span the mesher walled off,
//! on both an even- and an odd-sized grid.
//!
//! Y is ignored throughout (the floors are flat); the 2D convention is the physics
//! module's — `Vec2::x` is world X, `Vec2::y` is world Z.

use dreamcoast_game::physics::{self, GridCollision, SolidMap};
use glam::{Vec2, Vec3};

use crate::procgen::{TILE_SIZE, TileGrid};

/// The player's collision radius, metres.
///
/// 0.4 m against a 2 m tile leaves 1.2 m of clearance in a one-tile corridor — enough
/// that a diagonal run does not scrape both walls, tight enough that a doorway still
/// reads as a doorway. The placeholder sphere is authored at this radius too (see
/// [`crate::level`]), so what you see is what collides.
pub const PLAYER_RADIUS: f32 = 0.4;

/// World height a character's *origin* sits at, metres.
///
/// Zero, because [`crate::rigs`] authors both rigs with their soles on `y = 0` and
/// `rig_geometry_is_grounded_and_outward_facing` holds them there. So "place a character"
/// is "place its origin on the floor plane", with no per-rig offset to keep in sync — the
/// M1 sphere placeholder needed one (it was placed at its own radius so the ball rested on
/// the floor) and the rigged characters that replaced it do not.
pub const CHARACTER_Y: f32 = 0.0;

/// The grid answers the collision layer's only question directly.
///
/// Out-of-bounds is solid — [`TileGrid::get`] already defines it that way, which is
/// exactly the [`SolidMap`] contract (a sealed world: the mover cannot leave the map and
/// the raycaster always terminates).
impl SolidMap for TileGrid {
    #[inline]
    fn is_solid(&self, tx: i32, tz: i32) -> bool {
        // `self.get(..)` rather than the inherent `is_solid` of the same name: identical
        // result, but no reader has to wonder which one this resolves to.
        self.get(tx, tz).is_solid()
    }
}

/// World-space position of the grid's `(0, 0)` corner — the origin of collision space.
#[inline]
pub fn grid_origin(grid: &TileGrid) -> Vec2 {
    Vec2::new(grid.world_min_x(), grid.world_min_z())
}

/// World position (Y dropped) → collision space.
#[inline]
pub fn to_collision(grid: &TileGrid, world: Vec3) -> Vec2 {
    Vec2::new(world.x, world.z) - grid_origin(grid)
}

/// Collision space → world position at height `y`.
#[inline]
pub fn to_world(grid: &TileGrid, local: Vec2, y: f32) -> Vec3 {
    let origin = grid_origin(grid);
    Vec3::new(local.x + origin.x, y, local.y + origin.y)
}

/// The grid bound to the dungeon's tile size — the handle every collision query goes
/// through, so [`TILE_SIZE`] is threaded from one place.
#[inline]
pub fn collision(grid: &TileGrid) -> GridCollision<'_, TileGrid> {
    GridCollision::new(grid, TILE_SIZE)
}

/// Tile containing a collision-space point. Because collision space and grid space share
/// an origin, this **is** the grid tile index (it may be out of bounds; the grid reads
/// out-of-bounds as solid).
#[inline]
pub fn tile_of(local: Vec2) -> (i32, i32) {
    physics::world_to_tile(local, TILE_SIZE)
}

/// Where the player starts, in **collision space**: the entry tile's centre, pushed out
/// of any geometry it happens to touch.
///
/// The push matters even though the entry is a room centre: the level file and the
/// simulation must agree on one spawn, and "centre, then `nearest_free`" is a rule both
/// can evaluate from the grid alone. [`crate::level`] places the warrior here and
/// [`crate::game`] starts the simulation here.
pub fn player_spawn_local(grid: &TileGrid) -> Vec2 {
    let local = to_collision(grid, grid.entry_world());
    collision(grid)
        .nearest_free(local, PLAYER_RADIUS)
        .unwrap_or(local)
}

/// [`player_spawn_local`] in world space, on the floor ([`CHARACTER_Y`]).
pub fn player_spawn(grid: &TileGrid) -> Vec3 {
    to_world(grid, player_spawn_local(grid), CHARACTER_Y)
}

/// Does a circle in collision space overlap the square of tile `(tx, tz)`?
///
/// Closest-point test against the tile's AABB — the same geometry the mover resolves
/// against, so "touching the exit" means exactly what "resting on a wall" means.
pub fn circle_overlaps_tile(local: Vec2, radius: f32, (tx, tz): (i32, i32)) -> bool {
    let min = physics::tile_min(tx, tz, TILE_SIZE);
    let max = min + Vec2::splat(TILE_SIZE);
    local.clamp(min, max).distance_squared(local) < radius * radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procgen::{DungeonParams, Tile, generate};

    /// A 5x4 fixture with an odd width, so the world origin falls *inside* tile column 2
    /// rather than on a tile boundary. Every alignment claim below is checked on this
    /// grid as well as on a generated one — an index-offset adapter passes the even case
    /// and fails here.
    const ODD_ROWS: [&str; 4] = ["#####", "#E.X#", "#...#", "#####"];

    /// The claim: the tile the collision layer blocks is the tile the mesher walled off,
    /// over exactly the same world span.
    ///
    /// Checked against the *geometry's* own numbers ([`TileGrid::tile_edge_x`] — what
    /// `meshing.rs` places wall quads on), not against the collision layer's, so the two
    /// cannot agree by sharing a mistake.
    #[test]
    fn a_wall_tile_blocks_exactly_where_its_geometry_stands() {
        for grid in [
            TileGrid::from_rows(&ODD_ROWS),
            generate(7, &DungeonParams::default()),
        ] {
            let c = collision(&grid);
            for z in 0..grid.height() {
                for x in 0..grid.width() {
                    let centre = grid.tile_center(x, z);
                    let local = to_collision(&grid, centre);
                    assert_eq!(tile_of(local), (x, z), "tile centre maps back to its tile");
                    assert_eq!(
                        c.is_solid_at(local),
                        grid.is_solid(x, z),
                        "tile ({x}, {z}) at world {centre:?}"
                    );
                }
            }

            // The boundary itself: a hair either side of a wall's world-space edge must
            // land on opposite sides of the collision answer. A half-tile (1 m) shift —
            // the failure mode an index offset produces on an odd grid — moves this.
            let eps = 1e-3;
            for z in 0..grid.height() {
                for x in 0..grid.width() {
                    if !grid.is_solid(x, z) || !grid.is_walkable(x - 1, z) {
                        continue;
                    }
                    let edge = grid.tile_edge_x(x); // world X where the wall quad stands
                    let zc = grid.tile_center(x, z).z;
                    let inside = to_collision(&grid, Vec3::new(edge + eps, 0.0, zc));
                    let outside = to_collision(&grid, Vec3::new(edge - eps, 0.0, zc));
                    assert!(c.is_solid_at(inside), "just inside the wall at x={edge}");
                    assert!(!c.is_solid_at(outside), "just outside the wall at x={edge}");
                }
            }
        }
    }

    /// Collision space and world space round-trip (within a micron — the conversion is
    /// one f32 add each way at dungeon scale).
    #[test]
    fn world_and_collision_space_round_trip() {
        let grid = generate(11, &DungeonParams::default());
        for &p in &[
            grid.entry_world(),
            grid.exit_world(),
            Vec3::new(-39.5, 0.0, 12.25),
            Vec3::ZERO,
        ] {
            let back = to_world(&grid, to_collision(&grid, p), 0.0);
            assert!((back - p).length() < 1e-5, "{p:?} -> {back:?}");
        }
    }

    /// The spawn is inside the dungeon, on the entry tile, and free of geometry.
    #[test]
    fn the_spawn_is_free_space_on_the_entry_tile() {
        for seed in 0..20u64 {
            let grid = generate(seed, &DungeonParams::default());
            let spawn = player_spawn(&grid);
            let local = to_collision(&grid, spawn);
            assert!(
                !collision(&grid).circle_overlaps(local, PLAYER_RADIUS),
                "seed {seed}: spawn overlaps geometry"
            );
            assert_eq!(tile_of(local), grid.entry(), "seed {seed}: spawn tile");
            assert_eq!(grid.get(grid.entry().0, grid.entry().1), Tile::Entry);
            assert_eq!(spawn.y, CHARACTER_Y, "the character stands on the floor");
            assert_eq!(to_collision(&grid, spawn), player_spawn_local(&grid));
        }
    }

    /// The exit overlap test is the tile's square, not a point sample: standing next to
    /// the exit's edge counts, standing a tile away does not.
    #[test]
    fn circle_overlaps_tile_is_the_tiles_square() {
        let grid = TileGrid::from_rows(&ODD_ROWS);
        let exit = grid.exit();
        let centre = to_collision(&grid, grid.exit_world());
        assert!(circle_overlaps_tile(centre, PLAYER_RADIUS, exit));

        // Just outside the tile's -X edge, by less than the radius: still overlapping.
        let near = centre - Vec2::new(TILE_SIZE * 0.5 + PLAYER_RADIUS * 0.5, 0.0);
        assert!(circle_overlaps_tile(near, PLAYER_RADIUS, exit));
        // A full tile away: clear.
        let far = centre - Vec2::new(TILE_SIZE, 0.0);
        assert!(!circle_overlaps_tile(far, PLAYER_RADIUS, exit));
    }
}
