//! Physics: tile-grid collision, raycasting, and the combat overlap shapes
//! (`docs/game-framework-plan.md` §3.4).
//!
//! Collision v1 is **the grid itself**. A top-down action game whose level is
//! generated as a tile grid already has a perfect broadphase, a perfect
//! narrowphase, and a perfect navigation structure sitting in one array; a
//! general rigid-body solver would add a dependency, a tuning surface, and a
//! source of frame-to-frame nondeterminism to answer a question the grid answers
//! exactly. So: circles slide against tiles, rays walk the grid, and nothing
//! here allocates, keeps state, or reads a clock.
//!
//! # Coordinates
//!
//! Everything is 2D on the **XZ plane**, with Y (height) ignored — the game is
//! top-down and its floors are flat. The 2D vector convention is used
//! *everywhere* in this module, so read it once:
//!
//! | [`glam::Vec2`] component | world axis |
//! |---|---|
//! | `.x` | world **X** |
//! | `.y` | world **Z** |
//!
//! A 3D world position `Vec3::new(x, y, z)` becomes `Vec2::new(x, z)`, and the
//! `hit_x` / `hit_z` flags on [`MoveResult`] name the *world* axes accordingly.
//!
//! Tile `(tx, tz)` covers `[tx * tile_size, (tx + 1) * tile_size)` on X and the
//! same on Z, so [`world_to_tile`] is a plain floor and negative tile
//! coordinates are perfectly legal. The tile size is **passed explicitly** to
//! every free function: the collision layer has no configuration and no globals,
//! and a caller that dislikes threading it can bind it once with
//! [`GridCollision`].
//!
//! # The map
//!
//! The only thing this module knows about a level is [`SolidMap`] — one method,
//! `is_solid(tx, tz)`, with out-of-bounds defined as solid. The dungeon
//! generator implements it over its own grid type; nothing is copied, and this
//! crate never depends on the generator.
//!
//! # What is here
//!
//! * [`move_circle`] — the character mover: substepped, sliding, never tunnels,
//!   never ends inside geometry.
//! * [`raycast`] — exact grid traversal (line of sight, aim assist, projectiles).
//! * [`circle_overlaps`] / [`nearest_free`] — overlap test and the safe-spawn
//!   depenetration helper.
//! * [`circle_hit`] / [`sector_hit`] — grid-free combat shapes (M2).
//!
//! # Guarantees
//!
//! Deterministic (pure `f32` arithmetic, no globals, no randomness, no
//! iteration-order dependence on a hash map), allocation-free in the hot paths,
//! and dependency-free beyond `glam`.
//!
//! ```
//! use dreamcoast_game::physics::{SolidMap, move_circle, raycast, sector_hit};
//! use glam::Vec2;
//!
//! /// A 10x10 room with a wall of solid tiles around it.
//! struct Room;
//! impl SolidMap for Room {
//!     fn is_solid(&self, tx: i32, tz: i32) -> bool {
//!         !(1..9).contains(&tx) || !(1..9).contains(&tz)
//!     }
//! }
//!
//! const TILE: f32 = 2.0;
//! let radius = 0.4;
//!
//! // Run diagonally into the north wall: the blocked axis stops, the other slides.
//! let start = Vec2::new(6.0, 2.5);
//! let moved = move_circle(&Room, TILE, start, radius, Vec2::new(0.13, -0.13));
//! assert!(moved.hit_z && !moved.hit_x);
//! assert_eq!(moved.pos.x, start.x + 0.13); // tangential speed fully preserved
//!
//! // Line of sight to that wall.
//! let hit = raycast(&Room, TILE, Vec2::new(6.0, 6.0), Vec2::NEG_Y, 100.0).unwrap();
//! assert_eq!(hit.dist, 4.0);
//! assert_eq!(hit.normal, Vec2::new(0.0, 1.0));
//!
//! // A 90-degree swing in front of the player.
//! assert!(sector_hit(
//!     Vec2::ZERO,
//!     Vec2::X,
//!     45f32.to_radians(),
//!     2.0,
//!     Vec2::new(1.5, 1.0),
//!     0.5,
//! ));
//! ```

mod circle;
#[cfg(test)]
mod fixture;
mod map;
mod raycast;
mod shapes;

pub use circle::{
    CONTACT_SKIN, MoveResult, circle_overlaps, max_travel, move_circle, nearest_free,
};
pub use map::{GridCollision, SolidMap, tile_center, tile_min, world_to_tile};
pub use raycast::{RayHit, raycast};
pub use shapes::{circle_hit, sector_hit};
