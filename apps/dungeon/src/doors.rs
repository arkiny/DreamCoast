//! Doorway doors (docs/door-props-plan.md M-D1) — the simulation half of the door
//! prop: a per-door swing state machine, proximity auto-opening, and the door-aware
//! collision map that makes a closed door solid.
//!
//! The door list is derived from the grid ([`crate::level::door_spots`] — a pure
//! function), so the level's `door_<i>` entities and this module's `doors[i]` agree by
//! construction. Auto-open serves EVERY character (player and monsters alike): a door
//! is a flow device, not a puzzle, and a monster that could not open one would wedge
//! its own pathfinding — A* stays door-blind and plans through doorways, physics
//! blocks for the ~0.35 s a swing takes.
//!
//! This is also the global distance field's first rigid content mover (U2): the game
//! writes the swing onto the panel entity's `LocalTransform`, and the field's per-frame
//! entity sync promotes the panel to the movable layer the first frame it turns.

use dreamcoast_game::physics::SolidMap;
use glam::Vec2;

use crate::level::DoorSpot;
use crate::procgen::{TILE_SIZE, TileGrid};

/// Full swing, radians (105° — past a right angle so the open panel hugs the wall
/// side of the corridor rather than standing square in it).
pub const OPEN_ANGLE: f32 = 105.0 * std::f32::consts::PI / 180.0;
/// Swing duration, fixed steps (0.35 s at 60 Hz).
const SWING_STEPS: u32 = 21;
/// Swing rate per fixed step.
const SWING_RATE: f32 = OPEN_ANGLE / SWING_STEPS as f32;
/// A character's circle centre within this range of the door tile's centre asks the
/// door to open (and keeps it open).
const TRIGGER_RANGE: f32 = 1.6;
/// Steps the doorway must stay clear before the door swings shut (1 s at 60 Hz).
const CLEAR_STEPS: u32 = 60;

/// What one tick did — the game plays a creak/thud per event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DoorEvent {
    /// Door `i` started opening. Carries the door's world-plane position for panning.
    Opening(usize),
    /// Door `i` latched shut.
    Closed(usize),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    Closed,
    Opening,
    Open { clear: u32 },
    Closing,
}

struct Door {
    tile: (i32, i32),
    /// Tile centre, collision space — the proximity test's anchor.
    centre: Vec2,
    state: State,
    /// Current swing angle `[0, OPEN_ANGLE]`, advanced [`SWING_RATE`] per step.
    angle: f32,
}

/// Every door on the floor, in [`crate::level::door_spots`] order.
#[derive(Default)]
pub struct DoorWorld {
    doors: Vec<Door>,
}

impl DoorWorld {
    pub fn new(grid: &TileGrid) -> Self {
        Self::from_spots(&crate::level::door_spots(grid))
    }

    pub fn from_spots(spots: &[DoorSpot]) -> Self {
        DoorWorld {
            doors: spots
                .iter()
                .map(|s| Door {
                    tile: s.tile,
                    centre: Vec2::new(
                        (s.tile.0 as f32 + 0.5) * TILE_SIZE,
                        (s.tile.1 as f32 + 0.5) * TILE_SIZE,
                    ),
                    state: State::Closed,
                    angle: 0.0,
                })
                .collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.doors.len()
    }

    #[allow(dead_code)] // len()'s conventional pair
    pub fn is_empty(&self) -> bool {
        self.doors.is_empty()
    }

    /// The door at `i`'s swing angle, radians — what the game writes onto the panel
    /// entity's rotation.
    pub fn angle(&self, i: usize) -> f32 {
        self.doors.get(i).map_or(0.0, |d| d.angle)
    }

    /// Whether the door at `i` is fully shut (the HUD/debug readout; physics asks
    /// [`Self::blocks`]).
    #[allow(dead_code)] // debug-panel readout seam (physics asks `blocks`)
    pub fn is_closed(&self, i: usize) -> bool {
        self.doors
            .get(i)
            .is_some_and(|d| matches!(d.state, State::Closed))
    }

    /// One fixed step: proximity opens, a clear doorway closes, swings advance.
    /// `occupants` are every character circle centre (collision space) this step.
    pub fn tick(&mut self, occupants: &[Vec2], events: &mut Vec<DoorEvent>) {
        for (i, door) in self.doors.iter_mut().enumerate() {
            let wanted = occupants
                .iter()
                .any(|p| p.distance_squared(door.centre) <= TRIGGER_RANGE * TRIGGER_RANGE);
            match door.state {
                State::Closed => {
                    if wanted {
                        door.state = State::Opening;
                        events.push(DoorEvent::Opening(i));
                    }
                }
                State::Opening => {
                    door.angle = (door.angle + SWING_RATE).min(OPEN_ANGLE);
                    // Epsilon: SWING_STEPS f32 additions of OPEN_ANGLE/SWING_STEPS can
                    // land a hair under the target; the last step must still latch.
                    if door.angle + 1.0e-4 >= OPEN_ANGLE {
                        door.angle = OPEN_ANGLE;
                        door.state = State::Open { clear: 0 };
                    }
                }
                State::Open { clear } => {
                    let clear = if wanted { 0 } else { clear + 1 };
                    if clear >= CLEAR_STEPS {
                        door.state = State::Closing;
                    } else {
                        door.state = State::Open { clear };
                    }
                }
                State::Closing => {
                    if wanted {
                        // Someone stepped back in mid-swing: reopen from here.
                        door.state = State::Opening;
                    } else {
                        door.angle = (door.angle - SWING_RATE).max(0.0);
                        if door.angle <= 1.0e-4 {
                            door.angle = 0.0;
                            door.state = State::Closed;
                            events.push(DoorEvent::Closed(i));
                        }
                    }
                }
            }
        }
    }

    /// Whether tile `(tx, tz)` is currently blocked by a door. Blocked unless FULLY
    /// open — a swinging panel is solid (no mid-swing clipping; the swing is 0.35 s).
    pub fn blocks(&self, tx: i32, tz: i32) -> bool {
        self.doors
            .iter()
            .any(|d| d.tile == (tx, tz) && !matches!(d.state, State::Open { .. }))
    }

    /// The door tile's centre, collision space — the SFX pan anchor.
    pub fn centre(&self, i: usize) -> Vec2 {
        self.doors.get(i).map_or(Vec2::ZERO, |d| d.centre)
    }
}

/// The grid with its doors overlaid: solid rock PLUS whichever doorways are shut this
/// step. Every mover (warrior and grunts) collides against this; A* and line-of-sight
/// keep the plain grid (a door is an eventuality, not a wall).
pub struct DoorMap<'a> {
    pub grid: &'a TileGrid,
    pub doors: &'a DoorWorld,
}

impl SolidMap for DoorMap<'_> {
    #[inline]
    fn is_solid(&self, tx: i32, tz: i32) -> bool {
        self.grid.get(tx, tz).is_solid() || self.doors.blocks(tx, tz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level::door_spots;
    use crate::procgen::Tile;

    fn fixture() -> (TileGrid, DoorWorld) {
        // One room (ids assigned by from_rows? — the crossing test needs room ids, so
        // use a generated-style grid instead: a hand grid with a door tile marked).
        let grid = TileGrid::from_rows(&[
            "#####", //
            "#...#", //
            "##+##", //
            "#...#", //
            "#####",
        ]);
        assert_eq!(grid.get(2, 2), Tile::Door);
        let doors = DoorWorld::new(&grid);
        (grid, doors)
    }

    #[test]
    fn a_closed_door_blocks_and_an_open_one_does_not() {
        let (grid, mut doors) = fixture();
        assert_eq!(doors.len(), 1, "the fixture's one doorway gets one door");
        let map = DoorMap {
            grid: &grid,
            doors: &doors,
        };
        assert!(map.is_solid(2, 2), "closed door tile is solid");
        assert!(!map.is_solid(2, 1), "room floor stays open");

        // Walk a character to the doorway: it opens over SWING_STEPS and unblocks.
        let at_door = Vec2::new(5.0, 5.0); // tile (2,2) centre
        let mut events = Vec::new();
        for _ in 0..=SWING_STEPS {
            doors.tick(&[at_door], &mut events);
        }
        assert_eq!(events, vec![DoorEvent::Opening(0)]);
        assert!(!doors.blocks(2, 2), "a fully open door does not block");
        assert!((doors.angle(0) - OPEN_ANGLE).abs() < 1.0e-5);

        // Leave: after the clear delay + the swing, it latches shut and blocks again.
        events.clear();
        let far = Vec2::new(1.0, 1.0);
        for _ in 0..(CLEAR_STEPS + SWING_STEPS + 2) {
            doors.tick(&[far], &mut events);
        }
        assert_eq!(events, vec![DoorEvent::Closed(0)]);
        assert!(doors.blocks(2, 2));
        assert_eq!(doors.angle(0), 0.0);
    }

    #[test]
    fn stepping_back_mid_close_reopens_without_a_second_creak() {
        let (_, mut doors) = fixture();
        let at_door = Vec2::new(5.0, 5.0);
        let far = Vec2::new(1.0, 1.0);
        let mut events = Vec::new();
        for _ in 0..=SWING_STEPS {
            doors.tick(&[at_door], &mut events);
        }
        // Clear long enough to start closing, then step back mid-swing.
        for _ in 0..(CLEAR_STEPS + SWING_STEPS / 2) {
            doors.tick(&[far], &mut events);
        }
        let mid = doors.angle(0);
        assert!(
            mid > 0.0 && mid < OPEN_ANGLE,
            "mid-swing when re-approached"
        );
        for _ in 0..=SWING_STEPS {
            doors.tick(&[at_door], &mut events);
        }
        assert!(!doors.blocks(2, 2), "reopened from mid-swing");
        assert_eq!(
            events,
            vec![DoorEvent::Opening(0)],
            "a mid-close reopen is a continuation, not a new opening event"
        );
    }

    #[test]
    fn spots_and_doors_agree_by_construction() {
        let (grid, doors) = fixture();
        assert_eq!(door_spots(&grid).len(), doors.len());
    }
}
