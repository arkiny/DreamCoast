//! Health potions: where they lie, what picking one up does, and what carrying them
//! means (`docs/game-framework-plan.md` §5).
//!
//! Three things live here and nothing else:
//!
//! 1. **Placement** — [`potion_spawn_points`], a deterministic function of the floor and
//!    a seed, built to the same rules [`crate::ai::spawn_points`] uses for monsters.
//! 2. **The runtime** — [`ItemWorld`], the list of placed flasks and the overlap test
//!    that turns "the player walked over one" into a [`PickupEvent`].
//! 3. **The carry** — [`Inventory`], a count with a cap and one verb each way
//!    ([`Inventory::try_pickup`], [`Inventory::drink`]).
//!
//! # Spaces
//!
//! Every [`Vec2`] here is **collision space**, the grid-local space
//! [`crate::collision`] defines (`.x` = world X, `.y` = world Z) and the one the player
//! and the monsters are simulated in. World space appears at exactly one seam,
//! [`potion_level_entities`], which is where a placement becomes a transform in a
//! `.level` file — the same shape [`crate::level`] uses for the characters.
//!
//! # What this module deliberately does *not* do
//!
//! It never touches the ECS, the warrior, or the level loader. A picked-up potion is
//! reported, not erased: [`ItemWorld`] marks it taken and hands back its scene-graph name
//! ([`potion_name`]), and the integrator is what removes the visual and applies the heal
//! to [`WarriorController::heal`](crate::warrior::WarriorController::heal). That split is
//! the reason the whole item loop — placement, pickup, a full inventory, a drink —
//! runs in a unit test with no device, no window and no world.
//!
//! # A floor, end to end
//!
//! ```text
//! potion_spawn_points(grid, potions_for_floor(n), potion_seed(seed), &grunt_spawns, MIN_POTION_SPACING)
//!   -> potion_level_entities(grid, &points, &potion_asset_key())   // spliced into the .level
//!   -> ItemWorld::new(&points)                                     // the runtime state
//!        .tick(player_pos, &mut inventory) -> [PickupEvent]        // per fixed step
//!   -> (integrator) hide the named entity, HUD shows inventory.potions
//!   -> Q: items.drink(&mut inventory) -> Some(heal) -> warrior.heal(heal)
//! ```

// As `crate::ai` and `crate::rigs`: this module is authored complete and wired in by the
// next integration wave, and it exposes more than that one caller reads (the tuning
// constants, `Potion::name`, `ItemWorld::potions`) as the surface the tests and a future
// HUD/overlay use. This silences "never used", not "never checked".
#![allow(dead_code)]

use dreamcoast_asset::level::Entity as LevelEntity;
use dreamcoast_game::physics::tile_center;
use glam::{Mat4, Vec2};

use crate::collision::{self, PLAYER_RADIUS, to_world};
use crate::pathing::{DEFAULT_MAX_EXPANSIONS, astar};
use crate::procgen::{Rng, TILE_SIZE, TileGrid};
use crate::rigs;

// -------------------------------------------------------------------------------------
// Tuning
// -------------------------------------------------------------------------------------

/// Hit points one potion restores.
///
/// 40 against the warrior's 100 and the grunt's 8-damage claw: five claws kill you, and a
/// potion buys back exactly five. Deliberately *not* a full heal — a potion that erases a
/// bad fight makes the fight optional, and one that heals a fifth is loot the player
/// ignores. Big enough to change what you dare, small enough that you still have to not
/// get hit.
pub const POTION_HEAL: f32 = 40.0;

/// How many potions the player may carry at once.
///
/// The cap is the whole reason picking one up can *fail*, and that failure is the design:
/// a floor's potions are a decision about when to drink, not a resource to hoover up. A
/// third one left on the ground is a healing station you know where to find.
pub const POTION_MAX_CARRY: u32 = 3;

/// Centre-to-centre distance at which the player picks a potion up, metres.
///
/// Comfortably over the player's own 0.4 m body radius
/// ([`PLAYER_RADIUS`](crate::collision::PLAYER_RADIUS)) so that *touching* the flask is
/// enough — the player never has to stand exactly on it — and well under a 2 m tile, so a
/// potion is never collected from the far side of a doorway.
pub const PICKUP_RADIUS: f32 = 0.6;

/// The trigger has to be at least as big as the thing it is drawn under.
///
/// A flask wider than its own pickup circle would have a visible rim the player can touch
/// without collecting it — the classic "I walked over it and nothing happened" bug. The
/// geometry is authored in [`crate::rigs`] and the radius here, so this is a
/// **compile-time** guard rather than a test somebody might not run.
const _: () = assert!(
    rigs::POTION_HALF_WIDTH < PICKUP_RADIUS,
    "the potion prop is wider than the circle that collects it"
);

/// Potions on the first floor.
pub const POTIONS_PER_FLOOR: u32 = 3;

/// Ceiling on [`potions_for_floor`]'s scaling.
pub const MAX_POTIONS_PER_FLOOR: u32 = 6;

/// Minimum distance between a potion and any other potion, or any excluded point,
/// metres.
///
/// One tile plus the pickup radius, rounded up: two flasks closer than this read as one
/// pickup from the camera, and a flask this close to a monster's spawn is loot the player
/// cannot take without taking the fight. It is a *default*, not a rule —
/// [`potion_spawn_points`] takes the spacing as an argument so a caller placing one potion
/// on a tiny floor can relax it rather than get an empty list.
pub const MIN_POTION_SPACING: f32 = 4.0;

/// The potion as data: what it heals for and how many fit in a pocket.
///
/// A struct rather than two bare constants because the shipping values will eventually
/// come out of a `.ron` next to the class definition ([`crate::warrior`] already loads its
/// class that way). Until then [`PotionDef::health_potion`] is the one instance, and
/// **it is the single source**: [`Inventory`]'s verbs take a `PotionDef` rather than
/// keeping their own copy, so the cap the pickup enforces and the cap the HUD prints
/// cannot drift apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PotionDef {
    /// Hit points restored by one drink.
    pub heal: f32,
    /// How many may be carried at once.
    pub max_carry: u32,
    /// Centre-to-centre pickup distance, metres.
    pub pickup_radius: f32,
}

impl Default for PotionDef {
    fn default() -> Self {
        Self::health_potion()
    }
}

impl PotionDef {
    /// The shipping health potion.
    pub const fn health_potion() -> Self {
        Self {
            heal: POTION_HEAL,
            max_carry: POTION_MAX_CARRY,
            pickup_radius: PICKUP_RADIUS,
        }
    }
}

/// How many potions floor `floor` gets.
///
/// The gentle scaling hook the progression loop asks for: one extra flask every third
/// floor descended, capped at [`MAX_POTIONS_PER_FLOOR`]. The monsters scale *per floor*
/// ([`grunts_for_floor`](crate::game::grunts_for_floor)) and the potions every third, so
/// the curve is a cushion that falls behind the pressure rather than a supply line that
/// keeps up — which is the intended direction.
///
/// Floors are **1-based**, the run's own numbering
/// ([`FIRST_FLOOR`](crate::game::FIRST_FLOOR)), read from there rather than re-declared,
/// exactly as `grunts_for_floor` does — an off-by-one between the two curves would put
/// floor 1's potion count on floor 2's monsters. The count is an *argument* to
/// [`potion_spawn_points`], so a caller that wants a different curve writes a different
/// function and nothing here has to know.
pub fn potions_for_floor(floor: u32) -> u32 {
    let descended = floor.saturating_sub(crate::game::FIRST_FLOOR);
    (POTIONS_PER_FLOOR + descended / 3).min(MAX_POTIONS_PER_FLOOR)
}

/// The placement seed for a floor, derived from the dungeon's own seed.
///
/// Decorrelated from the seed [`crate::ai::spawn_points`] runs on (which is the raw
/// `grid.seed()`), because the two placers share an algorithm: fed the same seed they
/// shuffle the same rooms in the same order and every potion lands in the lap of a
/// monster. The constant is the golden-ratio odd word [`Rng`]'s own mixer uses, applied
/// once — enough to make the two streams independent, and a pure function of the seed so
/// a replayed seed still replays exactly.
pub fn potion_seed(dungeon_seed: u64) -> u64 {
    dungeon_seed ^ 0x9E37_79B9_7F4A_7C15
}

// -------------------------------------------------------------------------------------
// Placement
// -------------------------------------------------------------------------------------

/// Deterministic positions for up to `count` potions, in **collision space**.
///
/// The rules, in the order they are applied — deliberately the same shape as
/// [`crate::ai::spawn_points`], because "where does a thing go on a floor" is one question
/// and two answers to it would drift:
///
/// 1. **Rooms only.** A flask in a one-tile corridor is furniture the player walks
///    through without seeing; a flask in a room is a reason to enter the room.
/// 2. **Never the entry room.** The player starts at full health, so a potion within
///    three steps of the spawn is either wasted or hoarded from the first second.
/// 3. **Distinct rooms first.** The eligible rooms are shuffled and then filled
///    round-robin, so three potions across five rooms are 1/1/1, never 3/0/0.
/// 4. **At least `min_spacing` metres from every point in `exclude` and from every
///    potion already placed.** `exclude` is the caller's list of things a potion should
///    not share a spot with — the monster spawns and the player's own start — passed in
///    rather than recomputed here, for the same reason the grid is: the game owns those
///    lists, and a second call to a deterministic function is still a second source of
///    truth.
/// 5. **Free space the player can reach.** The tile centre is snapped with
///    [`nearest_free`](dreamcoast_game::physics::nearest_free) for **[`PLAYER_RADIUS`]**,
///    not for some smaller prop radius: the point of the snap is that the player's body
///    can occupy the flask's spot, which is what makes a 0.6 m pickup radius always
///    satisfiable. A point that still overlaps geometry is rejected.
/// 6. **Reachable by the shipping pathfinder.** Confirmed with an [`astar`] from the
///    entry under [`DEFAULT_MAX_EXPANSIONS`] — the same budget the monsters run under.
///    Loot the player cannot walk to is not loot.
///
/// Deterministic in `(grid, count, rng_seed, exclude, min_spacing)` and nothing else.
/// Returns **fewer than `count`** points when the floor has nowhere left to put them; a
/// caller that needs the exact count should check the length rather than trust it.
pub fn potion_spawn_points(
    grid: &TileGrid,
    count: u32,
    rng_seed: u64,
    exclude: &[Vec2],
    min_spacing: f32,
) -> Vec<Vec2> {
    let mut out: Vec<Vec2> = Vec::new();
    let count = count as usize;
    if count == 0 {
        return out;
    }
    let entry = grid.entry();
    let map = collision::collision(grid);
    let entry_room = grid.room_id_at(entry.0, entry.1);
    // A non-finite or negative spacing is a caller bug, not a licence to reject every
    // candidate: treat it as "no spacing rule" rather than propagating a NaN comparison
    // that silently returns an empty list.
    let spacing = if min_spacing.is_finite() {
        min_spacing.max(0.0)
    } else {
        0.0
    };
    let mut rng = Rng::new(rng_seed);

    // Candidate tiles, grouped by room. Rooms are visited in id order first so the
    // grouping itself does not depend on iteration order, then shuffled as a whole.
    let mut rooms: Vec<Vec<(i32, i32)>> = Vec::new();
    for room in grid.rooms() {
        if room.id == entry_room {
            continue;
        }
        let mut tiles: Vec<(i32, i32)> = Vec::new();
        for z in room.z..room.z + room.h {
            for x in room.x..room.x + room.w {
                if grid.is_walkable(x, z) {
                    tiles.push((x, z));
                }
            }
        }
        if !tiles.is_empty() {
            rng.shuffle(&mut tiles);
            rooms.push(tiles);
        }
    }
    if rooms.is_empty() {
        return out;
    }
    rng.shuffle(&mut rooms);

    // Round-robin over the rooms until the quota is met or every room is exhausted.
    let mut cursor = vec![0usize; rooms.len()];
    let mut exhausted = 0;
    while out.len() < count && exhausted < rooms.len() {
        exhausted = 0;
        for (room, next) in rooms.iter().zip(cursor.iter_mut()) {
            if out.len() >= count {
                break;
            }
            let mut placed = false;
            while *next < room.len() {
                let (x, z) = room[*next];
                *next += 1;
                let Some(free) = map.nearest_free(tile_center(x, z, TILE_SIZE), PLAYER_RADIUS)
                else {
                    continue;
                };
                if map.circle_overlaps(free, PLAYER_RADIUS)
                    || crowds(free, exclude, spacing)
                    || crowds(free, &out, spacing)
                    || astar(grid, entry, (x, z), DEFAULT_MAX_EXPANSIONS).is_none()
                {
                    continue;
                }
                out.push(free);
                placed = true;
                break;
            }
            if !placed && *next >= room.len() {
                exhausted += 1;
            }
        }
    }
    out
}

/// Whether `point` is within `spacing` metres of anything in `others`.
fn crowds(point: Vec2, others: &[Vec2], spacing: f32) -> bool {
    let spacing_sq = spacing * spacing;
    others
        .iter()
        .any(|&other| (other - point).length_squared() < spacing_sq)
}

// -------------------------------------------------------------------------------------
// The runtime
// -------------------------------------------------------------------------------------

/// Scene-graph name of the `i`-th potion (`potion_0`, `potion_1`, …).
///
/// Positional, exactly as [`crate::level::grunt_name`] is and for the same reason:
/// [`potion_spawn_points`] produces one ordered list, [`potion_level_entities`] writes
/// point `i` under this name, and [`ItemWorld`] reports pickups by it. One list, three
/// readers, no matching heuristic — a picked-up potion cannot hide a different flask.
pub fn potion_name(index: usize) -> String {
    format!("potion_{index}")
}

/// One placed flask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Potion {
    /// Index in the floor's list — also the `_N` in [`potion_name`].
    pub id: u32,
    /// Whether it has been collected. A taken potion is inert: it never fires again, and
    /// its visual is the integrator's to remove.
    pub taken: bool,
}

/// A potion the player just collected.
///
/// Carries what the integrator needs to finish the job and nothing else: which flask (so
/// its visual can be found by name) and how much it heals for (so the heal is applied
/// from the same [`PotionDef`] the pickup was judged against, not from a constant read
/// twice).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickupEvent {
    /// The potion's index — see [`Potion::id`].
    pub id: u32,
    /// Where it was, collision space.
    pub pos: Vec2,
    /// The carried count *after* this pickup, for a HUD line that cannot lag a step.
    pub carried: u32,
}

impl PickupEvent {
    /// The scene-graph name of the entity this pickup emptied.
    pub fn name(&self) -> String {
        potion_name(self.id as usize)
    }
}

/// Every potion on the floor, and the overlap test that collects them.
///
/// Built from [`potion_spawn_points`]' output, once, at the same moment the level that
/// draws them is written — so the flask the player sees and the flask [`Self::tick`]
/// hands out are the same object by construction.
#[derive(Clone, Debug)]
pub struct ItemWorld {
    def: PotionDef,
    potions: Vec<Potion>,
    positions: Vec<Vec2>,
}

impl ItemWorld {
    /// The floor's potions, in placement order, none taken.
    pub fn new(points: &[Vec2]) -> Self {
        Self::with_def(points, PotionDef::health_potion())
    }

    /// As [`Self::new`], with a non-shipping definition (tests, a future difficulty
    /// setting).
    pub fn with_def(points: &[Vec2], def: PotionDef) -> Self {
        Self {
            def,
            potions: (0..points.len())
                .map(|id| Potion {
                    id: id as u32,
                    taken: false,
                })
                .collect(),
            positions: points.to_vec(),
        }
    }

    /// The definition every potion on this floor shares — **the single source** for the
    /// heal, the cap and the pickup radius.
    #[inline]
    pub fn def(&self) -> PotionDef {
        self.def
    }

    /// Every placed potion, in order.
    #[inline]
    pub fn potions(&self) -> &[Potion] {
        &self.potions
    }

    /// Where potion `id` lies, collision space.
    pub fn position(&self, id: u32) -> Option<Vec2> {
        self.positions.get(id as usize).copied()
    }

    /// How many are still on the ground.
    pub fn remaining(&self) -> usize {
        self.potions.iter().filter(|p| !p.taken).count()
    }

    /// Whether potion `id` has been collected. Unknown ids read as taken — there is
    /// nothing there to pick up.
    pub fn is_taken(&self, id: u32) -> bool {
        self.potions.get(id as usize).is_none_or(|p| p.taken)
    }

    /// Collect every untaken potion the player is standing on, in id order.
    ///
    /// # Why the inventory is an argument
    ///
    /// Because a full inventory has to leave the flask **on the ground**, and that is a
    /// decision only something holding both halves can make. A `tick(player_pos)` that
    /// marked potions taken on its own would either consume a potion the player cannot
    /// carry or report a pickup that did not happen, and the integrator would then own a
    /// rule this module is supposed to own. So the overlap test and the capacity test
    /// resolve together, here, and a returned [`PickupEvent`] means exactly one thing: a
    /// potion left the floor and entered the pocket.
    ///
    /// Order is id order and the cap is honoured as it fills, so a player who steps onto
    /// two flasks with one slot free takes the lower-numbered one and leaves the other
    /// standing — deterministic, and the same on every machine.
    ///
    /// Call it once per fixed step, after the player has moved. A dead player should not
    /// be ticked at all (the integrator skips it): drinking is refused after death by
    /// [`WarriorController::heal`](crate::warrior::WarriorController::heal), but a corpse
    /// sliding onto a flask should not pocket it either.
    pub fn tick(&mut self, player_pos: Vec2, inventory: &mut Inventory) -> Vec<PickupEvent> {
        let mut events = Vec::new();
        if !player_pos.is_finite() {
            return events;
        }
        let radius_sq = self.def.pickup_radius * self.def.pickup_radius;
        for (potion, &pos) in self.potions.iter_mut().zip(&self.positions) {
            if potion.taken || (pos - player_pos).length_squared() > radius_sq {
                continue;
            }
            if !inventory.try_pickup(self.def) {
                // Full pocket: the flask stays exactly where it is, and stepping off and
                // back on after a drink collects it.
                continue;
            }
            potion.taken = true;
            events.push(PickupEvent {
                id: potion.id,
                pos,
                carried: inventory.potions,
            });
        }
        events
    }

    /// Drink one, using this floor's definition. `None` when the pocket is empty.
    ///
    /// The convenience the integrator should call, rather than
    /// [`Inventory::drink`] with a definition of its own: it is what keeps the heal that
    /// is applied and the potion that was picked up the same number.
    pub fn drink(&self, inventory: &mut Inventory) -> Option<f32> {
        inventory.drink(self.def)
    }

    /// Scene-graph names of every potion, in order — what a level splice writes and what
    /// a pickup later hides.
    pub fn names(&self) -> Vec<String> {
        (0..self.potions.len()).map(potion_name).collect()
    }
}

/// What the player is carrying.
///
/// A count, not a bag: v1 has exactly one item type, and an inventory that models slots,
/// stacks and item ids before there is a second item is a data structure written against
/// an imagined game. The two verbs below are the whole interface, and both are total —
/// neither can panic, and neither can leave the count above the cap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Inventory {
    /// Potions carried.
    pub potions: u32,
}

impl Inventory {
    /// An empty pocket.
    pub const fn new() -> Self {
        Self { potions: 0 }
    }

    /// Whether another potion would exceed the cap.
    #[inline]
    pub fn is_full(&self, def: PotionDef) -> bool {
        self.potions >= def.max_carry
    }

    /// Take one if there is room. `false` means the potion was **not** consumed — the
    /// caller must leave it where it is.
    pub fn try_pickup(&mut self, def: PotionDef) -> bool {
        if self.is_full(def) {
            return false;
        }
        self.potions += 1;
        true
    }

    /// Drink one, returning the hit points it restores; `None` when empty.
    ///
    /// The count drops here and the healing happens at the call site (the controller owns
    /// the player's health — see [`crate::warrior`]), which is also why this returns the
    /// amount instead of taking a `&mut WarriorController`: an item module that reaches
    /// into the character controller is an item module that cannot be tested without one.
    /// A refused heal (a dead warrior) still costs the potion — see the integration note
    /// in the module docs: the integrator checks `is_dead` before offering the drink.
    pub fn drink(&mut self, def: PotionDef) -> Option<f32> {
        if self.potions == 0 {
            return None;
        }
        self.potions -= 1;
        Some(def.heal)
    }
}

// -------------------------------------------------------------------------------------
// The level seam
// -------------------------------------------------------------------------------------

/// The cwd-relative asset key of the potion `.glb`, normalised to forward slashes so the
/// cook key is identical on every platform.
///
/// The same string [`rigs::ensure_rigs`] writes, derived from the same
/// [`rigs::rig_asset_path`] — not a literal repeated here.
pub fn potion_asset_key() -> String {
    rigs::rig_asset_path(rigs::POTION_PROP)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Height a potion's origin sits at, metres. The prop's own base is authored at y = 0
/// ([`rigs::potion`]), so this places it standing on the floor.
pub const POTION_Y: f32 = 0.0;

/// The floor's potions as ready-made `.level` entities — one splice, no arithmetic at the
/// call site.
///
/// `points` is [`potion_spawn_points`]' output in **collision space**; `grid` is what
/// turns it into world space, and it is required rather than optional because that
/// conversion is the one place a potion could silently end up in a different room than
/// the simulation thinks (the grid's origin is what relates the two). `asset` is the
/// `.glb` key — pass [`potion_asset_key`].
///
/// Every entity is named [`potion_name`]`(i)` at index `i`, at the prop's authored metre
/// scale, with no material override (the prop carries its own two materials, and the
/// loader ignores overrides for glTF assets — see [`crate::level`]).
pub fn potion_level_entities(grid: &TileGrid, points: &[Vec2], asset: &str) -> Vec<LevelEntity> {
    points
        .iter()
        .enumerate()
        .map(|(i, &local)| LevelEntity {
            asset: asset.to_owned(),
            name: Some(potion_name(i)),
            transform: Mat4::from_translation(to_world(grid, local, POTION_Y)).to_cols_array(),
            material_override: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procgen::{DungeonParams, generate};
    use glam::Vec3;

    /// A generated floor, the same way `level.rs`'s tests build one.
    fn dungeon(seed: u64) -> TileGrid {
        generate(seed, &DungeonParams::default())
    }

    /// A generated floor with exactly two rooms — the tightest case the "never the entry
    /// room" rule has: one of the two is the entry's, so every potion on the floor has to
    /// fit in the other one or not exist.
    fn two_rooms(seed: u64) -> TileGrid {
        let grid = generate(
            seed,
            &DungeonParams {
                width: 24,
                height: 16,
                min_rooms: 2,
                max_rooms: 2,
                room_min: 4,
                room_max: 6,
                ..DungeonParams::default()
            },
        );
        assert_eq!(grid.rooms().len(), 2, "the fixture must have two rooms");
        grid
    }

    fn seeds() -> std::ops::Range<u64> {
        0..12
    }

    // -- placement ----------------------------------------------------------------------

    /// The property the cook cache, a replayed seed and every test below rest on.
    #[test]
    fn placement_is_deterministic_and_seed_sensitive() {
        let grid = dungeon(7);
        let a = potion_spawn_points(&grid, 3, potion_seed(7), &[], MIN_POTION_SPACING);
        let b = potion_spawn_points(&grid, 3, potion_seed(7), &[], MIN_POTION_SPACING);
        assert_eq!(a, b, "the same seed must place the same potions");
        assert_eq!(a.len(), 3, "a default floor has room for three");

        let other = potion_spawn_points(&grid, 3, potion_seed(8), &[], MIN_POTION_SPACING);
        assert_ne!(a, other, "a different seed must move them");

        // And the placer is decorrelated from the monster placer it shares an algorithm
        // with: fed the raw seed both would shuffle the rooms identically.
        let grunts = crate::ai::spawn_points(&grid, 6, 12.0, grid.seed());
        let potions = potion_spawn_points(
            &grid,
            3,
            potion_seed(grid.seed()),
            &grunts,
            MIN_POTION_SPACING,
        );
        assert!(
            !potions.is_empty(),
            "the exclusion list must not starve the floor"
        );
    }

    /// Rooms only, and never the room the player wakes up in.
    #[test]
    fn potions_land_in_rooms_and_never_the_entry_room() {
        for seed in seeds() {
            let grid = dungeon(seed);
            let entry = grid.entry();
            let entry_room = grid.room_id_at(entry.0, entry.1);
            let points = potion_spawn_points(
                &grid,
                potions_for_floor(0),
                potion_seed(seed),
                &[],
                MIN_POTION_SPACING,
            );
            assert!(!points.is_empty(), "seed {seed}: nowhere to put a potion");
            for p in &points {
                let (x, z) = collision::tile_of(*p);
                let room = grid.room_at(x, z);
                assert!(
                    room.is_some(),
                    "seed {seed}: potion at tile ({x}, {z}) is not in a room"
                );
                assert_ne!(
                    room.unwrap().id,
                    entry_room,
                    "seed {seed}: potion in the entry room"
                );
                assert!(
                    grid.is_walkable(x, z),
                    "seed {seed}: potion inside solid rock"
                );
            }
        }
    }

    /// The snap is for the *player's* radius, so the player can always stand where the
    /// flask is — which is what makes a 0.6 m pickup radius reachable everywhere.
    #[test]
    fn every_potion_stands_where_the_player_can_stand() {
        for seed in seeds() {
            let grid = dungeon(seed);
            let map = collision::collision(&grid);
            for p in potion_spawn_points(&grid, 4, potion_seed(seed), &[], MIN_POTION_SPACING) {
                assert!(
                    !map.circle_overlaps(p, PLAYER_RADIUS),
                    "seed {seed}: a potion at {p} overlaps geometry"
                );
            }
        }
    }

    /// Spacing holds between potions *and* against the caller's exclusion list.
    #[test]
    fn placement_respects_spacing_and_the_exclusion_list() {
        for seed in seeds() {
            let grid = dungeon(seed);
            // The real caller's exclusion list: the monsters, plus the player's spawn.
            let mut exclude = crate::ai::spawn_points(&grid, 6, 12.0, grid.seed());
            exclude.push(collision::player_spawn_local(&grid));

            let points =
                potion_spawn_points(&grid, 4, potion_seed(seed), &exclude, MIN_POTION_SPACING);
            for (i, a) in points.iter().enumerate() {
                for b in &points[i + 1..] {
                    assert!(
                        (*a - *b).length() >= MIN_POTION_SPACING,
                        "seed {seed}: potions {a} and {b} are {} m apart",
                        (*a - *b).length()
                    );
                }
                for e in &exclude {
                    assert!(
                        (*a - *e).length() >= MIN_POTION_SPACING,
                        "seed {seed}: potion {a} sits {} m from an excluded point",
                        (*a - *e).length()
                    );
                }
            }
        }
    }

    /// A spacing wide enough to cover the floor starves the placement rather than
    /// violating it — fewer points, never a closer one.
    #[test]
    fn an_impossible_spacing_returns_fewer_points_not_worse_ones() {
        let grid = dungeon(3);
        let all = potion_spawn_points(&grid, 5, potion_seed(3), &[], 0.0);
        let spread = potion_spawn_points(&grid, 5, potion_seed(3), &[], 1_000.0);
        assert!(spread.len() < all.len(), "the huge spacing changed nothing");
        assert!(spread.len() <= 1);
    }

    /// Two rooms, one of them the entry's: every potion goes in the other, and asking for
    /// more than it holds yields fewer rather than spilling back into the spawn room.
    #[test]
    fn the_entry_room_is_excluded_even_when_it_is_half_the_floor() {
        for seed in seeds() {
            let grid = two_rooms(seed);
            let entry_room = grid.room_id_at(grid.entry().0, grid.entry().1);
            let other = grid
                .rooms()
                .iter()
                .find(|r| r.id != entry_room)
                .expect("two rooms, one of them not the entry's")
                .id;
            let points = potion_spawn_points(&grid, 8, potion_seed(seed), &[], 2.0);
            assert!(!points.is_empty(), "seed {seed}: the far room fits none");
            for p in &points {
                let (x, z) = collision::tile_of(*p);
                assert_eq!(
                    grid.room_id_at(x, z),
                    other,
                    "seed {seed}: potion at ({x}, {z}) left the one legal room"
                );
            }
        }
    }

    /// A floor with no rooms at all (a hand-authored corridor fixture) places nothing
    /// rather than falling back to corridors.
    #[test]
    fn a_floor_with_no_rooms_places_nothing() {
        let grid = TileGrid::from_rows(&["############", "#E.........#", "############"]);
        assert!(grid.rooms().is_empty(), "the fixture records no rooms");
        assert!(potion_spawn_points(&grid, 3, potion_seed(1), &[], 0.0).is_empty());
    }

    /// Zero potions is a valid floor, and asking for none does no work.
    #[test]
    fn a_zero_count_places_nothing() {
        let grid = dungeon(5);
        assert!(potion_spawn_points(&grid, 0, potion_seed(5), &[], MIN_POTION_SPACING).is_empty());
    }

    /// The floor curve is gentle, monotone, capped — and numbered the same way the run
    /// is, which is the part an off-by-one would hide.
    #[test]
    fn potions_for_floor_scales_gently_and_caps() {
        use crate::game::FIRST_FLOOR;
        assert_eq!(potions_for_floor(FIRST_FLOOR), POTIONS_PER_FLOOR);
        assert_eq!(potions_for_floor(FIRST_FLOOR + 2), POTIONS_PER_FLOOR);
        assert_eq!(potions_for_floor(FIRST_FLOOR + 3), POTIONS_PER_FLOOR + 1);
        // Floor 0 does not exist; it must not read as "one floor above the first".
        assert_eq!(potions_for_floor(0), POTIONS_PER_FLOOR);
        let mut last = 0;
        for floor in 0..64 {
            let n = potions_for_floor(floor);
            assert!(n >= last, "the curve went backwards at floor {floor}");
            assert!(n <= MAX_POTIONS_PER_FLOOR);
            last = n;
        }
        assert_eq!(potions_for_floor(64), MAX_POTIONS_PER_FLOOR);
    }

    // -- pickup -------------------------------------------------------------------------

    /// The pickup radius is a hard edge: just inside collects, just outside does not.
    #[test]
    fn the_pickup_radius_is_a_hard_edge() {
        let at = Vec2::new(4.0, 6.0);
        let def = PotionDef::health_potion();
        let mut inv = Inventory::new();

        let mut items = ItemWorld::new(&[at]);
        let outside = at + Vec2::new(def.pickup_radius + 1e-3, 0.0);
        assert!(
            items.tick(outside, &mut inv).is_empty(),
            "collected too far"
        );
        assert_eq!(inv.potions, 0);
        assert_eq!(items.remaining(), 1);

        let inside = at + Vec2::new(def.pickup_radius - 1e-3, 0.0);
        let events = items.tick(inside, &mut inv);
        assert_eq!(events.len(), 1, "the potion was not collected at the edge");
        assert_eq!(events[0].id, 0);
        assert_eq!(events[0].pos, at);
        assert_eq!(events[0].carried, 1);
        assert_eq!(events[0].name(), "potion_0");
        assert_eq!(inv.potions, 1);
        assert_eq!(items.remaining(), 0);
        assert!(items.is_taken(0));

        // A taken potion never fires again, however long the player stands on it.
        for _ in 0..10 {
            assert!(items.tick(at, &mut inv).is_empty());
        }
        assert_eq!(inv.potions, 1);
    }

    /// Two flasks under one player: both collected, in id order, one event each.
    #[test]
    fn overlapping_potions_are_collected_in_id_order() {
        let mut items = ItemWorld::new(&[Vec2::new(0.0, 0.2), Vec2::new(0.0, -0.2)]);
        let mut inv = Inventory::new();
        let events = items.tick(Vec2::ZERO, &mut inv);
        assert_eq!(events.iter().map(|e| e.id).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(inv.potions, 2);
        assert_eq!(items.remaining(), 0);
    }

    /// A full pocket leaves the flask standing — the rule the cap exists for.
    #[test]
    fn a_full_inventory_leaves_the_potion_on_the_ground() {
        let def = PotionDef::health_potion();
        let at = Vec2::new(1.0, 1.0);
        let mut items = ItemWorld::new(&[at]);
        let mut inv = Inventory {
            potions: def.max_carry,
        };

        assert!(items.tick(at, &mut inv).is_empty(), "picked up while full");
        assert_eq!(inv.potions, def.max_carry, "the count changed");
        assert!(!items.is_taken(0), "the flask was consumed anyway");
        assert_eq!(items.remaining(), 1);

        // Drink one, walk back over it: now it is collectable.
        assert_eq!(items.drink(&mut inv), Some(def.heal));
        let events = items.tick(at, &mut inv);
        assert_eq!(events.len(), 1);
        assert_eq!(inv.potions, def.max_carry);
        assert!(items.is_taken(0));
    }

    /// A pocket that fills mid-tick takes the low ids and leaves the rest standing.
    #[test]
    fn a_pocket_that_fills_mid_tick_leaves_the_remainder() {
        let points: Vec<Vec2> = (0..5)
            .map(|i| Vec2::new(i as f32 * 0.1 - 0.2, 0.0))
            .collect();
        let mut items = ItemWorld::new(&points);
        let mut inv = Inventory::new();
        let events = items.tick(Vec2::ZERO, &mut inv);

        assert_eq!(events.len(), POTION_MAX_CARRY as usize);
        assert_eq!(
            events.iter().map(|e| e.id).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(inv.potions, POTION_MAX_CARRY);
        assert_eq!(items.remaining(), 2, "the rest must still be standing");
        assert!(!items.is_taken(3) && !items.is_taken(4));
    }

    // -- the inventory ------------------------------------------------------------------

    /// The carry loop end to end: three stored, a fourth refused, drinking returns the
    /// heal and frees a slot, and an empty pocket reports nothing.
    #[test]
    fn three_are_stored_a_fourth_is_refused_and_drinking_reports_the_heal() {
        let def = PotionDef::health_potion();
        assert_eq!(def.max_carry, 3, "this test is written against a cap of 3");
        let mut inv = Inventory::new();

        for expected in 1..=def.max_carry {
            assert!(inv.try_pickup(def), "pickup {expected} was refused");
            assert_eq!(inv.potions, expected);
        }
        assert!(inv.is_full(def));
        assert!(!inv.try_pickup(def), "the fourth pickup was allowed");
        assert_eq!(
            inv.potions, def.max_carry,
            "a refused pickup changed the count"
        );

        assert_eq!(inv.drink(def), Some(POTION_HEAL));
        assert_eq!(inv.potions, 2);
        assert_eq!(inv.drink(def), Some(POTION_HEAL));
        assert_eq!(inv.drink(def), Some(POTION_HEAL));
        assert_eq!(inv.potions, 0);
        assert_eq!(inv.drink(def), None, "drank from an empty pocket");
        assert_eq!(inv.potions, 0);
    }

    /// The heal is worth five claws and does not out-heal the warrior's own pool — the
    /// two numbers live in different modules, so nothing but this stops them drifting
    /// into "a potion is a full reset".
    #[test]
    fn the_heal_is_a_meaningful_fraction_of_the_warrior_pool() {
        let warrior = crate::warrior::WarriorController::new();
        let max = warrior.health().max;
        assert!(
            POTION_HEAL > max * 0.2 && POTION_HEAL < max,
            "a potion heals {POTION_HEAL} of {max} — outside the design band"
        );
        let claw = crate::ai::GruntClass::grunt().swing().unwrap().damage;
        assert_eq!(
            (POTION_HEAL / claw).floor(),
            5.0,
            "a potion is meant to buy back five claws"
        );
    }

    /// The controller really does take what `drink` reports.
    #[test]
    fn a_drink_heals_the_warrior_by_what_it_reported() {
        let mut warrior = crate::warrior::WarriorController::new();
        warrior.take_damage([crate::warrior::IncomingHit {
            amount: 55.0,
            direction: Vec2::Y,
            stagger: 0.0,
        }]);
        let hurt = warrior.health().current;

        let items = ItemWorld::new(&[Vec2::ZERO]);
        let mut inv = Inventory { potions: 1 };
        let heal = items.drink(&mut inv).expect("a stored potion");
        let restored = warrior.heal(heal);
        assert_eq!(restored, heal);
        assert_eq!(warrior.health().current, hurt + heal);
        assert_eq!(inv.potions, 0);
    }

    // -- the level seam -----------------------------------------------------------------

    /// The splice: one entity per point, named positionally, standing on the floor at the
    /// world position the simulation's collision-space point maps to.
    #[test]
    fn level_entities_name_and_place_every_point() {
        let grid = dungeon(11);
        let points = potion_spawn_points(&grid, 3, potion_seed(11), &[], MIN_POTION_SPACING);
        let asset = potion_asset_key();
        let entities = potion_level_entities(&grid, &points, &asset);

        assert_eq!(entities.len(), points.len());
        for (i, entity) in entities.iter().enumerate() {
            assert_eq!(entity.name.as_deref(), Some(potion_name(i).as_str()));
            assert_eq!(entity.asset, asset);
            assert!(entity.material_override.is_none());

            let m = Mat4::from_cols_array(&entity.transform);
            let placed = m.transform_point3(Vec3::ZERO);
            let expected = to_world(&grid, points[i], POTION_Y);
            assert!(
                (placed - expected).length() < 1e-5,
                "potion_{i} placed at {placed}, expected {expected}"
            );
            assert_eq!(placed.y, 0.0, "potions stand on the floor");
            // Unit scale: the prop is authored in metres.
            assert!((m.x_axis.length() - 1.0).abs() < 1e-6);
        }
    }

    /// An empty floor writes no entities rather than an empty placeholder.
    #[test]
    fn no_points_means_no_entities() {
        let grid = dungeon(2);
        assert!(potion_level_entities(&grid, &[], &potion_asset_key()).is_empty());
    }

    /// The level references the file the authoring pass actually writes, by the same
    /// normalised key the cook is on — a literal typed here would drift the day the
    /// prop's stem changes.
    #[test]
    fn the_asset_key_is_the_one_the_prop_writer_authors() {
        let key = potion_asset_key();
        assert_eq!(key, "cache/generated/potion.glb");
        assert!(!key.contains('\\'), "the key must be forward-slashed");
        assert_eq!(
            key,
            rigs::rig_asset_path(rigs::POTION_PROP)
                .to_string_lossy()
                .replace('\\', "/")
        );
    }

    /// The names three separate readers agree on.
    #[test]
    fn names_are_positional_and_match_the_placed_list() {
        let items = ItemWorld::new(&[Vec2::ZERO, Vec2::X, Vec2::Y]);
        assert_eq!(items.names(), vec!["potion_0", "potion_1", "potion_2"]);
        assert_eq!(
            items.potions().iter().map(|p| p.id).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(potion_name(7), "potion_7");
    }

    /// A non-finite player position is dropped rather than collecting the whole floor
    /// (`NaN` comparisons are false, so the guard has to be explicit).
    #[test]
    fn a_non_finite_player_collects_nothing() {
        let mut items = ItemWorld::new(&[Vec2::ZERO]);
        let mut inv = Inventory::new();
        assert!(items.tick(Vec2::splat(f32::NAN), &mut inv).is_empty());
        assert_eq!(inv.potions, 0);
        assert!(!items.is_taken(0));
    }
}
