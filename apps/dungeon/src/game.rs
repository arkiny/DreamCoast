//! The game: one [`GameHooks`] implementation driving a warrior through a generated
//! dungeon full of monsters, with a top-down follow camera and a HUD
//! (`docs/game-framework-plan.md` §4).
//!
//! It proves the whole seam — simulation writes into the ECS from
//! [`GameHooks::fixed_update`], the view comes from [`GameHooks::camera`], the visual
//! pose from [`GameHooks::render_update`] and the overlay from [`GameHooks::draw_ui`] —
//! with no renderer changes at all.
//!
//! # What this module is, and is not
//!
//! It is the **wiring**. Every rule it appears to make is made somewhere else and is
//! only *routed* here:
//!
//! | question | answered by |
//! |---|---|
//! | how the warrior moves, swings, rolls, dies | [`crate::warrior`] |
//! | how a monster senses, paths, commits, flinches | [`crate::ai`] |
//! | what a hit does to hit points | [`dreamcoast_game::combat`] |
//! | which clip plays, and what it fades from | [`dreamcoast_game::anim`], through both of the above |
//! | what a clip *looks* like, and when the blade lands | [`crate::rigs`] |
//! | what the world is shaped like | [`crate::procgen`] / [`crate::collision`] |
//!
//! So the interesting content here is **order** — [`DungeonGame::simulate`] documents
//! the one the fixed step runs in and why — and **the split between simulation and
//! presentation**, below.
//!
//! **The grid is the game.** [`DungeonGame`] owns the one [`TileGrid`] the dungeon was
//! generated as. The level writer borrowed it to build the geometry *before* the engine
//! came up ([`crate::level::ensure_dungeon`]), and everything that moves collides
//! against that same instance here — so what you walk into is what was meshed, with no
//! second generate() call to drift.
//!
//! # The run: floors, descent and restart
//!
//! A **run is a seed**, and a floor is a seed derived from it ([`floor_seed`]) — floor 1
//! *is* the run seed, so `--seed 12345` still means the dungeon it always meant. Walking
//! onto the exit tile generates the next floor by the same road `main` took for the
//! first (generate → mesh → `.glb` + `.level`), then asks the engine for it through
//! [`GameHooks::next_level`]; the world is rebuilt wholesale and the cast re-resolves out
//! of the new one ([`DungeonGame::acquire`]). Hit points carry down and monsters get one
//! more per floor ([`grunts_for_floor`]), which is the run's pressure. Dying ends it, and
//! `R` starts the same seed over from floor 1. [`Progression`] is the state machine, and
//! [`DungeonGame::install`] is what a floor change costs.
//!
//! The pairing between the grid and the geometry survives all of that because a floor's
//! geometry is meshed from the grid instance the game is about to play, in that order,
//! every time. The one way to break it is still the engine's **level hot-swap dropdown**:
//! picking a level by hand loads geometry the game did not ask for, and this grid stays
//! whatever floor it was on. The characters are re-acquired and re-snapped to free space,
//! but it will be free space *of the previous floor* — and, since that level is not a
//! continuation of the run, the warrior is rebuilt rather than carried.
//!
//! # Fixed-step discipline, and the single-writer rule
//!
//! Everything that integrates (positions, timers, camera smoothing) advances in
//! `fixed_update` at the engine's fixed `dt`; `camera` and `render_update` only *blend*
//! the two latest states by the frame's `alpha`. That keeps motion framerate-independent
//! and the follow camera free of the jitter a per-frame smoothing filter would add.
//!
//! The two halves write **disjoint** transforms, which is what lets both run without a
//! reconciliation pass:
//!
//! * `fixed_update` writes **bone** locals, via [`crate::characters::ClipSet::apply`] —
//!   the entities the rig's node hierarchy became, under each character's placement root.
//!   Pose application runs at the 60 Hz sim rate rather than per frame: a fade is at most
//!   0.18 s, so the worst visible artefact is one sim step of stale pose on a display
//!   faster than 60 fps, against the cost and the state a second per-frame sampling pass
//!   would need. Moving it to `render_update` later is a pure win and needs no contract
//!   change — the samples are already `(clip, time, alpha)` triples the presentation half
//!   could evaluate.
//! * `render_update` writes **root** transforms — one per character, interpolated. It is
//!   the only writer of those, exactly as it was the only writer of the M1 placeholder's.
//!
//! No entity is written by both, so neither half can lose the other's work.

use std::time::Instant;

use dreamcoast_audio::{AudioSystem, Sfx};
use dreamcoast_game::anim::AnimMachine;
use dreamcoast_game::combat::{
    AttackPhase, BodyCircle, DamageEvent, DeathEvent, Health, IFrames, Team, apply_damage_events,
    tick_iframes,
};
use dreamcoast_game::input::{ActionState, BindingsConfig, InputSnapshot, InputSource};
use glam::{Quat, Vec2, Vec3};
use sandbox::imgui;
use sandbox::scene::{Entity, Events, LocalTransform, MeshInstance, Name, World};
use sandbox::{CameraPose, GameHooks};

use crate::ai::{self, GruntBrain, GruntClass, GruntState, PlayerView};
use crate::characters::{ChildIndex, ClipSample, ClipSet, RigBinding};
use crate::collision::{self, CHARACTER_Y, PLAYER_RADIUS};
use crate::items::{self, Inventory, ItemWorld};
use crate::level::{PLAYER_NAME, grunt_name};
use crate::pathing::Pathfinder;
use crate::procgen::{Tile, TileGrid};
use crate::rigs::{self, Rig};
use crate::warrior::{
    AnimSample, IncomingHit, Target, WarriorController, WarriorCtx, WarriorInput, WarriorState,
};

/// Gameplay actions. The physical keys live in `config/bindings.ron` (data), so a
/// rebind never touches this file.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Action {
    MoveForward,
    MoveBack,
    MoveLeft,
    MoveRight,
    Sprint,
    Attack,
    Dodge,
    /// Drink a carried potion. An **edge**, like the other two buttons: holding Q must not
    /// empty the pocket in three steps.
    Drink,
    /// Start the run again from the first floor. Only read while the warrior is dead —
    /// see [`DungeonGame::restart_requested`].
    Restart,
}

impl Action {
    /// Resolve a binding-file action name. Unknown names are a config error.
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "MoveForward" => Self::MoveForward,
            "MoveBack" => Self::MoveBack,
            "MoveLeft" => Self::MoveLeft,
            "MoveRight" => Self::MoveRight,
            "Sprint" => Self::Sprint,
            "Attack" => Self::Attack,
            "Dodge" => Self::Dodge,
            "Drink" => Self::Drink,
            "Restart" => Self::Restart,
            _ => return None,
        })
    }
}

/// The shipped bindings, embedded so the game runs from any working directory.
const DEFAULT_BINDINGS: &str = include_str!("../config/bindings.ron");

/// Input sources to hold down, as a comma-separated list of the names `bindings.ron`
/// uses, each optionally limited to a step range: `<source>[@<first>-<last>]`.
/// Empty/unset = normal input.
///
/// ```text
/// DUNGEON_HOLD=W,Shift        # both held for the whole run
/// DUNGEON_HOLD=D@0-117        # walk east for 117 steps, then stand still
/// ```
///
/// Headless capture has no keyboard: `--screenshot`/`CAPTURE_SEQ` pump no window events,
/// so a captured frame would always show a character standing on the spawn. This makes
/// the *headless* run drivable — `CAPTURE_SEQ=N DUNGEON_HOLD=W` walks N deterministic
/// fixed steps into whatever is north of the spawn — which is how "walk into a wall
/// without panicking" is checked against the real binary rather than a test harness.
///
/// The step range exists because a *fight* is not a walk: the interesting frames are the
/// ones where the character has arrived and stopped, and a movement key held forever
/// walks it straight back out of the encounter it just reached.
///
/// It names **input sources**, not keys, so a mouse button binds like a key does
/// ([`InputSource::from_name`] is the same parser `bindings.ron` goes through). Held is
/// the right verb for an axis and the wrong one for an edge, though: an attack is
/// `just_pressed`, so a permanently-held `Mouse1` fires exactly one swing, on the first
/// step. [`TAP_ENV`] is the edge form.
const HOLD_ENV: &str = "DUNGEON_HOLD";

/// Input sources to press for exactly one fixed step, as `<source>@<step>` pairs
/// (`DUNGEON_TAP=Mouse1@120,Space@200`). Empty/unset = nothing.
///
/// The edge counterpart of [`HOLD_ENV`], and the thing a combat capture actually needs:
/// a swing is a `just_pressed` edge, so posing one for a screenshot means pressing the
/// button on a *chosen* deterministic step and releasing it on the next. Step numbers
/// count the sim steps this process has run (the HUD's step readout); on the capture path
/// captured frame `f` shows the state after step `f`, so `WARMUP_FRAMES=f CAPTURE_SEQ=1`
/// photographs whatever step `f` produced.
///
/// The step numbers are **computed, not guessed**: `tests::capture_scout` replays a
/// headless run with the same input and prints the steps at which the combo would accept
/// a press, which is exactly the list this variable takes (see that module).
const TAP_ENV: &str = "DUNGEON_TAP";

/// Default monster population of a floor. Six is a floor that is dangerous in twos and
/// threes without ever putting more than [`ai::MAX_GRUNTS`] on one A* workspace.
pub const DEFAULT_GRUNTS: usize = 6;

/// The floor a run starts on. Floors are 1-based because they are shown to the player.
pub const FIRST_FLOOR: u32 = 1;

/// Seconds between the exit tile latching and the floor actually changing.
///
/// The transition is not instant for two reasons, one of each kind: the player needs to
/// see *why* the world is about to vanish (the HUD counts this down), and the next floor
/// is generated, meshed and written at the start of this window — so the one-off cost of
/// building it lands under a banner that already says something is happening rather than
/// as an unexplained hitch. 0.6 s is long enough to read and short enough that walking
/// onto the exit still feels like the thing that did it.
const DESCEND_GRACE: f32 = 0.6;

/// The seed of floor `floor` of the run seeded `run_seed`.
///
/// **Floor 1 is the run seed itself**, unmixed. That is not a special case for its own
/// sake: a run *is* its seed, `--seed 12345` has always meant "play the dungeon 12345
/// generates", and every capture recipe, bug report and golden the game already has
/// names a dungeon that way. Making floor 1 anything else would silently retire all of
/// them. Deeper floors are the same seed pushed through splitmix64's finalizer — the
/// mixer [`crate::procgen::Rng::new`] already trusts to turn a counter into an
/// uncorrelated state — after adding `floor × 2^64/φ`:
///
/// ```text
/// floor 1 → run_seed
/// floor n → mix(run_seed + n · 0x9E37_79B9_7F4A_7C15)
/// ```
///
/// So the whole run is a pure function of one `u64`: floor 7 of seed 12345 is the same
/// dungeon on every machine and every day, and two runs that differ in seed differ on
/// every floor (the additive step keeps neighbouring floors of one run, and identical
/// floors of neighbouring runs, from landing on the same avalanche input).
pub fn floor_seed(run_seed: u64, floor: u32) -> u64 {
    if floor <= FIRST_FLOOR {
        return run_seed;
    }
    let mut z = run_seed.wrapping_add(u64::from(floor).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Floor `floor` of the run seeded `run_seed`, generated.
///
/// The one place a floor's grid comes from — `main` builds the first one with it and the
/// game builds every later one with it, so "the dungeon you start in" and "the dungeon
/// you descend into" cannot drift apart by a parameter.
pub fn floor_grid(run_seed: u64, floor: u32) -> TileGrid {
    crate::procgen::generate(
        floor_seed(run_seed, floor),
        &crate::procgen::DungeonParams::default(),
    )
}

/// How many monsters a floor gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Population {
    /// The default: [`grunts_for_floor`], which scales gently with depth.
    PerFloor,
    /// `--grunts <n>`: this many on every floor, whatever the depth. An explicit count
    /// is an instruction, not a starting point — a capture recipe that asks for twelve
    /// monsters must still get twelve on floor three.
    Fixed(usize),
}

impl Population {
    /// The count this policy puts on `floor`.
    pub fn count(self, floor: u32) -> usize {
        match self {
            Self::PerFloor => grunts_for_floor(floor),
            Self::Fixed(n) => n,
        }
    }
}

/// The default monster count of floor `floor`: one more monster per floor descended,
/// capped at what one A* workspace is sized for.
///
/// `DEFAULT_GRUNTS + (floor − 1)`, not `DEFAULT_GRUNTS + floor`, so **floor 1 is exactly
/// the floor this game has always shipped** — the same six monsters every existing
/// capture and recipe was recorded against (see [`floor_seed`] for the same argument
/// about the geometry). The cap is [`ai::MAX_GRUNTS`] rather than a number of its own:
/// the pathfinder's workspace is what makes twelve the ceiling, so that is where the
/// ceiling should be read from. Floors 7 and deeper are therefore equally populated and
/// get their difficulty from their layout alone — a deliberate v1 limit, not a plateau
/// anyone should read as finished.
pub fn grunts_for_floor(floor: u32) -> usize {
    let extra = floor.saturating_sub(FIRST_FLOOR) as usize;
    (DEFAULT_GRUNTS + extra).min(ai::MAX_GRUNTS)
}

/// Where floor `floor`'s flasks lie, collision space.
///
/// The one place a floor's potions come from — `new` builds the first floor's with it and
/// [`DungeonGame::build_floor`] builds every later one, exactly as [`floor_grid`] is the
/// one place a floor's grid comes from. Every rule in it belongs to [`crate::items`]; what
/// is decided *here* is the two arguments that module deliberately refuses to invent:
///
/// * the **count** is the floor's ([`items::potions_for_floor`]), so a deeper floor is a
///   little better supplied;
/// * the **exclusion list** is this floor's monster spawns, so a flask is never loot the
///   player cannot take without taking the fight. It is the game's own list, threaded in
///   rather than recomputed — a second `ai::spawn_points` call is a second source of
///   truth even when it is deterministic.
///
/// The player's own spawn needs no exclusion: `potion_spawn_points` never places in the
/// entry room at all.
fn floor_potions(grid: &TileGrid, floor: u32, grunt_spawns: &[Vec2]) -> Vec<Vec2> {
    items::potion_spawn_points(
        grid,
        items::potions_for_floor(floor),
        items::potion_seed(grid.seed()),
        grunt_spawns,
        items::MIN_POTION_SPACING,
    )
}

/// How a floor becomes a level the engine can load: mesh the grid, write the `.glb` +
/// `.level` pair, and return the selector [`GameHooks::next_level`] hands back.
///
/// A function pointer rather than a direct call to [`crate::level::ensure_dungeon`] so
/// the progression state machine is exercisable without a filesystem — the tests install
/// one that writes nothing and returns the path the real one would have. Nothing else
/// about the transition changes between the two, which is the point: what the tests
/// drive is the shipping machine with its one side effect stubbed.
///
/// The two placement lists (monsters, potions) are **arguments** rather than something
/// the writer derives, because the game is what owns them: brain `i` and `grunt_<i>`, and
/// [`ItemWorld`] potion `i` and `potion_<i>`, are the same object seen from two sides, and
/// a writer that re-derived either would be a second source of truth for it.
type FloorWriter = fn(&TileGrid, &[Vec2], &[Vec2]) -> anyhow::Result<String>;

/// The shipping [`FloorWriter`]: the same road `main` puts floor 1 on (generator → glb +
/// level → the engine's ordinary level load), taken at runtime for floor `n`.
fn write_floor(grid: &TileGrid, spawns: &[Vec2], potions: &[Vec2]) -> anyhow::Result<String> {
    crate::level::ensure_dungeon(grid, spawns, potions)?;
    Ok(crate::level::dungeon_level_selector(grid.seed()))
}

/// A floor that has been generated and written but is not being played yet.
struct NextFloor {
    floor: u32,
    grid: TileGrid,
    spawns: Vec<Vec2>,
    /// Where this floor's flasks lie, collision space — the list the level was written
    /// from and the one [`ItemWorld`] will be built on.
    potions: Vec<Vec2>,
    /// What [`GameHooks::next_level`] will return — an explicit path, see
    /// [`crate::level::dungeon_level_selector`].
    level: String,
    /// Whether the warrior starts this floor new. A descent carries the run's hit points
    /// down (that is the run's pressure); a restart is a new run and does not.
    fresh_warrior: bool,
}

/// Where the run is between floors.
///
/// The transition spans several frames and two owners — the game generates the floor,
/// the engine rebuilds the world — so "which half has happened" is a state, not a pair
/// of booleans that can disagree:
///
/// ```text
///  Playing ──exit tile──→ Descending{grace, next} ──grace out──→ Handoff{level}
///     ↑                                                              │ next_level()
///     └────────── acquire() resolves the rebuilt cast ──── Awaiting ←─┘  (exactly once)
/// ```
///
/// `Handoff` is the single-shot latch: the level is *taken* out of it when the hook
/// reports it, so a floor cannot be requested twice however many frames pass before the
/// engine gets to the swap. `Awaiting` is the "the world is not mine yet" state — see
/// [`DungeonGame::simulate`] for why nothing is simulated in `Handoff` and why `Awaiting`
/// resolves through the ordinary cast acquisition.
enum Progression {
    /// Playing a floor.
    Playing,
    /// The exit is underfoot and the next floor is already built; this is the grace the
    /// HUD counts down.
    Descending { left: f32, next: Box<NextFloor> },
    /// The next floor is installed in the game and its level is waiting to be handed to
    /// the engine.
    Handoff { level: String },
    /// Handed over. The engine is rebuilding the world; the cast re-resolves out of it.
    Awaiting,
}

/// How far from the entry a monster may spawn, metres of *walking* (see
/// [`ai::spawn_points`]). Twelve metres is past the first doorway on every floor the
/// generator makes, so the player always gets the entry room to themselves.
const GRUNT_MIN_SPAWN_DISTANCE: f32 = 12.0;

/// Time constant of the camera's positional smoothing. Larger = lazier follow.
const CAMERA_SMOOTH_TIME: f32 = 0.14;
/// Downward tilt of the top-down camera, degrees from horizontal. Kept off 90° on
/// purpose: the view basis is built against world +Y, so a perfectly vertical view
/// axis is degenerate (see `sandbox::CameraPose`).
const CAMERA_PITCH_DEG: f32 = 55.0;
/// Eye distance from the followed point, metres.
///
/// Nudged up from the walking skeleton's 12 m once the flat plate became corridors,
/// measured on the same frame of the same run (a sprint north from the spawn, stopped
/// against the corridor's end wall — `tmp/m1_cam12` vs `tmp/m1_wall`):
///
/// * at **12 m** the eye is 9.8 m up and the near wall tops crowd the bottom of the
///   frame; the player ends up in a small pocket of visible floor near the frame edge
///   with the corridor's continuation off-screen;
/// * at **16 m** the eye is 13.1 m up — comfortably over the 4 m walls — and the same
///   moment shows the junction the player is standing in *plus* the rooms either side,
///   which is the information a top-down action game is asking the camera for.
///
/// With the frame's 60° vertical FOV that covers ~27 m of ground along Z, so the
/// largest room the generator places (9x9 tiles = 18 m) still fits — and, now that the
/// floor has monsters on it, so does a grunt closing from across that room.
///
/// What it does **not** fix: a wall between the camera and the player still occludes,
/// because these walls are opaque 4 m slabs. Standing in a corridor you sometimes look
/// at the near wall's top rather than at yourself. The real answers (shorter walls, or
/// fading geometry between eye and player) are camera/art work, not a constant.
const CAMERA_DISTANCE: f32 = 16.0;

/// Height the camera follows at, metres above the floor — chest height rather than the
/// feet, so the character sits nearer the middle of the frame than the bottom of it.
const CAMERA_FOCUS_Y: f32 = 0.9;

/// Sim-step budget, milliseconds. Above this the fixed step is eating into the frame,
/// and the HUD says so (and the log says so once).
const SIM_BUDGET_MS: f32 = 2.0;

/// Eye offset from the followed point: up and back along +Z, so screen-up is world
/// -Z and screen-right is world +X. The camera never yaws, which is exactly why
/// camera-relative movement below can be world-aligned.
pub fn camera_offset() -> Vec3 {
    let (sin, cos) = CAMERA_PITCH_DEG.to_radians().sin_cos();
    Vec3::new(0.0, sin, cos) * CAMERA_DISTANCE
}

/// Framerate-independent exponential approach factor for a step of `dt` with time
/// constant `tau`: `x += (target - x) * blend(dt, tau)`.
fn blend(dt: f32, tau: f32) -> f32 {
    1.0 - (-dt / tau.max(1e-4)).exp()
}

/// An angle folded into `(-PI, PI]`.
fn wrap_pi(angle: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    (angle + PI).rem_euclid(TAU) - PI
}

/// Interpolate two yaws the short way round, so a turn through ±π does not spin the
/// character the long way for one frame.
fn lerp_angle(from: f32, to: f32, alpha: f32) -> f32 {
    from + wrap_pi(to - from) * alpha
}

/// One held source and the step window it is held over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Hold {
    source: InputSource,
    /// Inclusive step window. The default is the whole run.
    first: u64,
    last: u64,
}

impl Hold {
    fn covers(&self, step: u64) -> bool {
        (self.first..=self.last).contains(&step)
    }
}

/// Held sources named by [`HOLD_ENV`], or empty when it is unset.
///
/// Grammar: `<source>[@<first>-<last>]`, comma-separated. A malformed entry is warned
/// about and dropped rather than fatal — a capture recipe is a diagnostic, and a typo in
/// one should not stop the game from starting.
fn held_sources_from_env() -> Vec<Hold> {
    let Ok(spec) = std::env::var(HOLD_ENV) else {
        return Vec::new();
    };
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|entry| match parse_hold(entry) {
            Some(hold) => Some(hold),
            None => {
                tracing::warn!("{HOLD_ENV}: cannot parse '{entry}' — ignored");
                None
            }
        })
        .collect()
}

/// `<source>` or `<source>@<first>-<last>`.
fn parse_hold(entry: &str) -> Option<Hold> {
    let (name, window) = match entry.split_once('@') {
        Some((name, window)) => (name.trim(), Some(window.trim())),
        None => (entry, None),
    };
    let source = InputSource::from_name(name)?;
    let (first, last) = match window {
        None => (0, u64::MAX),
        Some(window) => {
            let (a, b) = window.split_once('-')?;
            (a.trim().parse().ok()?, b.trim().parse().ok()?)
        }
    };
    (first <= last).then_some(Hold {
        source,
        first,
        last,
    })
}

/// `(step, source)` pairs named by [`TAP_ENV`], or empty when it is unset.
fn taps_from_env() -> Vec<(u64, InputSource)> {
    let Ok(spec) = std::env::var(TAP_ENV) else {
        return Vec::new();
    };
    let mut taps: Vec<(u64, InputSource)> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|entry| {
            let (name, step) = entry.split_once('@')?;
            let source = InputSource::from_name(name.trim())?;
            let step = step.trim().parse::<u64>().ok()?;
            Some((step, source))
        })
        .collect();
    let asked = spec.split(',').filter(|s| !s.trim().is_empty()).count();
    if taps.len() != asked {
        tracing::warn!("{TAP_ENV}: expected '<source>@<step>' pairs, got '{spec}'");
    }
    taps.sort_by_key(|&(step, _)| step);
    taps
}

/// One monster's presentation state: the yaw pair `render_update` interpolates.
///
/// The brain keeps its own previous/current *position* (that is the position-sync
/// contract [`crate::ai`] documents) but reports only a current facing, so the yaw pair
/// lives here — on the presentation side, where it belongs, and where nothing in the
/// simulation can read it back.
#[derive(Clone, Copy, Debug, Default)]
struct GruntView {
    prev_yaw: f32,
    yaw: f32,
}

/// One character's animation binding: which entity is its root and which clips are
/// bound to its bones.
struct CharacterAnim {
    root: Entity,
    clips: ClipSet,
}

/// Everything acquired from a freshly loaded level. `None` until the first
/// [`DungeonGame::simulate`] finds the world populated.
struct Cast {
    player: Entity,
    player_anim: CharacterAnim,
    /// One brain per level `grunt_<i>`, in spawn-point order.
    grunts: Vec<GruntBrain>,
    grunt_anims: Vec<CharacterAnim>,
    grunt_views: Vec<GruntView>,
}

pub struct DungeonGame {
    input: ActionState<Action>,
    /// The dungeon: geometry source (already meshed, before bring-up) and collision
    /// world (right here). One instance, so the two cannot disagree.
    ///
    /// Replaced wholesale when the run descends ([`DungeonGame::install`]) — the pairing
    /// survives because the replacement is meshed from *this* grid before its level is
    /// handed over, exactly as `main` does for the first one.
    grid: TileGrid,

    // --- The run ----------------------------------------------------------------------
    /// The run's identity: every floor's seed is derived from it ([`floor_seed`]), so one
    /// `u64` reproduces the whole descent.
    run_seed: u64,
    /// Which floor is being played, 1-based.
    floor: u32,
    /// How many monsters a floor gets.
    population: Population,
    /// Where the run is between floors.
    progression: Progression,
    /// How a generated floor becomes a loadable level (stubbed in tests).
    writer: FloorWriter,
    /// Whether the next cast acquisition starts the warrior fresh. True for a level this
    /// game did not ask for (bring-up, the engine's hot-swap dropdown — neither is a
    /// continuation of anything) and for a restart; false for a descent, which is what
    /// carries hit points down.
    reset_warrior: bool,
    /// Whether this game's grid is a floor of a seeded run at all. False for the
    /// `--generated-room` injection harness, whose grid is a fixture rather than a floor:
    /// descending or restarting out of it would generate a dungeon nobody asked for.
    floors_enabled: bool,
    /// Input sources forced down over a step window (see [`HOLD_ENV`]); empty in a
    /// normal run.
    forced: Vec<Hold>,
    /// One-step presses, by sim step (see [`TAP_ENV`]); empty in a normal run.
    taps: Vec<(u64, InputSource)>,

    // --- The player -------------------------------------------------------------------
    /// The character controller: the authority on the warrior's position, health, combo
    /// clock and animation state. Health lives *here*, not on the ECS entity — see
    /// [`WarriorController`]'s own docs, and [`Self::simulate`] step 6 for what that
    /// means for the damage flow.
    warrior: WarriorController,
    /// Simulation state in fixed steps, **collision space**: previous and current, so the
    /// render frame can interpolate between them.
    prev_pos: Vec2,
    pos: Vec2,
    prev_yaw: f32,
    yaw: f32,
    /// The last step's [`WarriorState`] — the controller reports it per tick rather than
    /// exposing a getter, so the HUD reads the latched copy.
    player_state: WarriorState,

    // --- The monsters -----------------------------------------------------------------
    /// Shared monster data (stats, the one swing, the animation graph).
    grunt_class: GruntClass,
    /// Where the level put them — the same list [`crate::level`] wrote, so index `i` is
    /// `grunt_<i>` in both.
    grunt_spawns: Vec<Vec2>,
    /// One A* workspace for the whole floor (see [`Pathfinder`]).
    finder: Pathfinder,

    // --- The loot ---------------------------------------------------------------------
    /// This floor's flasks: the same list the level was written from, so potion `i` here
    /// is `potion_<i>` there. Replaced with the floor ([`Self::install`]) — a collected
    /// flask does not follow the player down the stairs, the *potion* does.
    items: ItemWorld,
    /// What the player is carrying. **Survives a descent** (that is the point of picking
    /// one up on floor 2 and drinking it on floor 4) and is emptied only by a restart,
    /// with the fresh warrior it belongs to.
    inventory: Inventory,
    /// The mixer handle (docs/game-audio-plan.md M-A2). Constructed as the silent Null
    /// sink; [`Self::enable_audio`] opens the device — tests and headless captures never
    /// touch CoreAudio/WASAPI.
    audio: AudioSystem,
    /// Footstep cadence accumulator (seconds until the next step sound).
    foot_timer: f32,
    /// This floor's torch positions, collision space — the loop-slot emitters
    /// (slot `1 + i`; slot 0 is the ambience bed).
    torch_pts: Vec<Vec2>,
    /// How many torches [`crate::level`] hung on this floor — a HUD readout, not a
    /// simulation input. Latched when the floor is installed rather than recomputed per
    /// frame, and zero on a level this game did not author ([`Self::without_floors`]).
    torches: usize,

    // --- The combat channels ----------------------------------------------------------
    damage: Events<DamageEvent>,
    deaths: Events<DeathEvent>,

    // --- Assets and instances ---------------------------------------------------------
    /// The authored rigs: the source of both the clips and the bone names the level's
    /// instances are bound by ([`crate::characters`]).
    warrior_rig: Rig,
    grunt_rig: Rig,
    cast: Option<Cast>,

    // --- Presentation and readouts ----------------------------------------------------
    /// The point the camera follows (a smoothed trail behind the player), also kept as a
    /// previous/current pair for interpolation.
    prev_focus: Vec3,
    focus: Vec3,
    /// Latched once the player's circle first touches the exit tile — what starts the
    /// descent, and what stops it starting twice on one floor.
    exit_reached: bool,
    /// Fixed steps simulated so far — a cheap "is the sim actually running" readout, and
    /// the clock [`TAP_ENV`] schedules against.
    steps: u64,
    /// The interpolation factor of the frame being drawn, latched by the camera hook
    /// (which is the only hook the frame loop hands it to) so the HUD can report the
    /// rendered position rather than the last simulated one.
    render_alpha: f32,
    /// Length of the last fixed step, seconds — the engine's, not a constant of ours, so
    /// the HUD's derived readouts cannot drift from the loop that produced them.
    last_dt: f32,
    /// Cost of the last fixed step and the worst one so far, milliseconds.
    sim_ms: f32,
    sim_ms_peak: f32,
    /// Whether the over-budget warning has already been logged (once, not per step).
    budget_warned: bool,
}

impl DungeonGame {
    /// Take ownership of the dungeon the level will be written from, and pick this
    /// floor's monster spawns.
    ///
    /// The spawns are chosen **here**, once, because two consumers need the same list and
    /// a second [`ai::spawn_points`] call would be a second source of truth: the level
    /// writer places `grunt_<i>` at point `i` ([`Self::grunt_spawns`]) and the simulation
    /// starts brain `i` there.
    ///
    /// `grid` is **floor 1** of the run, so its seed *is* the run seed ([`floor_seed`]
    /// is the identity there) and no second parameter is needed to say so. Every deeper
    /// floor is derived from it.
    pub fn new(grid: TileGrid, population: Population) -> anyhow::Result<Self> {
        let run_seed = grid.seed();
        let floor = FIRST_FLOOR;
        let grunts = population.count(floor);
        let bindings = BindingsConfig::from_ron(DEFAULT_BINDINGS)
            .map_err(|e| anyhow::anyhow!("dungeon bindings: {e}"))?;
        let map = bindings
            .resolve(Action::from_name)
            .map_err(|e| anyhow::anyhow!("dungeon bindings: {e}"))?;

        // Seeded off the dungeon's own seed, so a floor's monster placement is as
        // reproducible as its walls — and a different seed moves them.
        let grunt_spawns = ai::spawn_points(&grid, grunts, GRUNT_MIN_SPAWN_DISTANCE, grid.seed());
        if grunt_spawns.len() < grunts {
            tracing::warn!(
                "dungeon: seed {} has room for only {} of {grunts} grunts",
                grid.seed(),
                grunt_spawns.len(),
            );
        }
        // The floor's flasks, chosen here for the same reason the monsters are: the level
        // writer places `potion_<i>` at point `i` ([`Self::potion_spawns`]) and
        // [`ItemWorld`] collects point `i`. One list, two readers.
        let potions = floor_potions(&grid, floor, &grunt_spawns);
        let torch_list = crate::level::torch_points(&grid, crate::level::torch_seed(grid.seed()));
        let torch_pts: Vec<Vec2> = torch_list.iter().map(|t| t.pos).collect();
        let torches = torch_list.len();

        let spawn = collision::player_spawn_local(&grid);
        let focus = collision::to_world(&grid, spawn, CAMERA_FOCUS_Y);
        Ok(Self {
            input: ActionState::new(map),
            grid,
            run_seed,
            floor,
            population,
            progression: Progression::Playing,
            writer: write_floor,
            reset_warrior: true,
            floors_enabled: true,
            forced: held_sources_from_env(),
            taps: taps_from_env(),
            warrior: WarriorController::new(),
            prev_pos: spawn,
            pos: spawn,
            prev_yaw: 0.0,
            yaw: 0.0,
            player_state: WarriorState::Idle,
            grunt_class: GruntClass::grunt(),
            grunt_spawns,
            finder: Pathfinder::new(),
            items: ItemWorld::new(&potions),
            inventory: Inventory::new(),
            audio: AudioSystem::new(false, 0),
            foot_timer: 0.0,
            torch_pts,
            torches,
            damage: Events::new(),
            deaths: Events::new(),
            warrior_rig: rigs::warrior(),
            grunt_rig: rigs::grunt(),
            cast: None,
            prev_focus: focus,
            focus,
            exit_reached: false,
            steps: 0,
            render_alpha: 0.0,
            last_dt: 0.0,
            sim_ms: 0.0,
            sim_ms_peak: 0.0,
            budget_warned: false,
        })
    }

    /// The dungeon this game will be played in — what [`crate::level::ensure_dungeon`]
    /// meshes into the `.level` the engine then loads.
    pub fn grid(&self) -> &TileGrid {
        &self.grid
    }

    /// This floor's monster spawn points, collision space — what the level writer places
    /// `grunt_<i>` at.
    pub fn grunt_spawns(&self) -> &[Vec2] {
        &self.grunt_spawns
    }

    /// This floor's potion points, collision space — what the level writer places
    /// `potion_<i>` at, and what this game collects them by.
    ///
    /// Read straight out of [`ItemWorld`] rather than kept alongside it: the runtime and
    /// the level must describe the same flasks, and the cheapest way to guarantee that is
    /// to have only one list.
    pub fn potion_spawns(&self) -> &[Vec2] {
        self.items.positions()
    }

    /// Turn floor progression off: this game's grid is a fixture, not floor 1 of a run.
    ///
    /// The `--generated-room` injection harness is the one caller. Its room has no exit
    /// (entry and exit are the same tile, so the latch never fires) but it *can* be died
    /// in, and a restart there would generate and load a real dungeon — which would
    /// destroy the one thing the harness exists to show. Off is the honest state: there
    /// are no floors here to progress through.
    pub fn without_floors(mut self) -> Self {
        self.floors_enabled = false;
        // The harness level is written by `ensure_generated_room`, which authors one
        // hand-placed light and no torch props at all — so the count derived from its
        // fixture grid in `new` describes a floor that was never written. Zero is what is
        // actually in that level.
        self.torches = 0;
        self.torch_pts.clear();
        self
    }

    /// Open the real output device and start the ambience bed (docs/game-audio-plan.md
    /// M-A2/A3). Called by `main` for interactive runs only — tests and headless
    /// captures keep the silent Null sink from construction, so neither ever touches a
    /// platform audio API. Volume seams: `AUDIO_MASTER` / `AUDIO_SFX` /
    /// `AUDIO_AMBIENCE` (linear gains, default 1.0 / 1.0 / 1.0).
    pub fn enable_audio(&mut self) {
        /// The bank seed is CONTENT identity, not run identity: the same game must
        /// sound the same on every seed — a re-rolled dungeon changes the map, not the
        /// sword.
        const SFX_BANK_SEED: u64 = 0x000D_5C0A_575F;
        let vol = |k: &str| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(1.0)
                .clamp(0.0, 4.0)
        };
        self.audio = AudioSystem::new(true, SFX_BANK_SEED);
        self.audio.set_master(vol("AUDIO_MASTER"));
        self.audio.set_bus_sfx(vol("AUDIO_SFX"));
        self.audio.set_bus_ambience(vol("AUDIO_AMBIENCE"));
        self.audio.loop_flat(0, Sfx::AmbienceLoop, 0.4);
    }

    /// Generate, mesh and write floor `floor` of this run — everything the transition
    /// needs, done *before* the world it belongs to is asked for.
    ///
    /// Deliberately eager: the whole cost of a floor (generation, greedy meshing, the
    /// `.glb` write, the level RON) is paid here, at the start of the descent grace,
    /// rather than on the frame the level is handed over. Re-descending into a seed
    /// already on this machine writes nothing at all — the writer is content-conditional
    /// and the floor's bytes are a pure function of its seed.
    fn build_floor(&self, floor: u32, fresh_warrior: bool) -> anyhow::Result<NextFloor> {
        let started = Instant::now();
        let grid = floor_grid(self.run_seed, floor);
        let spawns = ai::spawn_points(
            &grid,
            self.population.count(floor),
            GRUNT_MIN_SPAWN_DISTANCE,
            grid.seed(),
        );
        let potions = floor_potions(&grid, floor, &spawns);
        let level = (self.writer)(&grid, &spawns, &potions)?;
        tracing::info!(
            "dungeon: floor {floor} of run {} ready — seed {}, {} grunts, {} potions, \
             level '{level}' ({:.1} ms)",
            self.run_seed,
            grid.seed(),
            spawns.len(),
            potions.len(),
            started.elapsed().as_secs_f64() * 1e3,
        );
        Ok(NextFloor {
            floor,
            grid,
            spawns,
            potions,
            level,
            fresh_warrior,
        })
    }

    /// Make `next` the floor being played, and ask for its world.
    ///
    /// Everything scoped to a floor is dropped here rather than carried: the cast (its
    /// entities belong to a world that is about to stop existing), the monster brains and
    /// their spawn list, the A* workspace (sized to the old grid), the exit latch and the
    /// combat queues. What survives is the *run* — the warrior (unless this is a restart:
    /// see [`NextFloor::fresh_warrior`]), the step clock and the cost readouts.
    ///
    /// The cast is cleared **here**, not by testing the cached entity: a rebuilt `World`
    /// restarts its generation counters, so a stale `Entity` can pass `is_alive` in the
    /// new world by landing on a recycled slot. The transition knows the world is going
    /// away; that knowledge is more reliable than any probe of it.
    fn install(&mut self, next: NextFloor) {
        self.floor = next.floor;
        let torch_list =
            crate::level::torch_points(&next.grid, crate::level::torch_seed(next.grid.seed()));
        // Retarget the surviving loop slots to the new floor's brackets and silence the
        // rest — slot indices are 1 + torch index (0 is the ambience bed).
        for slot in (1 + torch_list.len())..=(1 + self.torch_pts.len().max(torch_list.len())) {
            self.audio.loop_stop(slot as u8);
        }
        self.torch_pts = torch_list.iter().map(|t| t.pos).collect();
        self.torches = torch_list.len();
        self.grid = next.grid;
        self.grunt_spawns = next.spawns;
        self.finder = Pathfinder::new();
        self.cast = None;
        self.exit_reached = false;
        self.damage.clear();
        self.deaths.clear();
        self.reset_warrior = next.fresh_warrior;
        self.player_state = WarriorState::Idle;

        // The flasks belong to the floor and are replaced with it. What the player is
        // *carrying* belongs to the run, so it is not touched here — `acquire` empties the
        // pocket in the one branch that builds a new warrior to hold it.
        self.items = ItemWorld::new(&next.potions);

        // The spawn the *level* authors is the entry tile, and `acquire` will snap the
        // player onto it through `nearest_free` when the world arrives. These are the
        // same point, set now so the camera is already looking at the new floor rather
        // than at where the old one used to be.
        let spawn = collision::player_spawn_local(&self.grid);
        self.prev_pos = spawn;
        self.pos = spawn;
        let focus = collision::to_world(&self.grid, spawn, CAMERA_FOCUS_Y);
        self.prev_focus = focus;
        self.focus = focus;

        self.progression = Progression::Handoff { level: next.level };
    }

    /// Start the descent: build the next floor and hold it for the grace window.
    ///
    /// A failure to build is loud and *non-fatal*: the run stays on the floor it is on
    /// (the exit stays latched, so it does not retry every step and flood the log). The
    /// alternative — dying on an `unwrap` because a disk was full — would lose a run for
    /// a reason that has nothing to do with the game.
    fn begin_descent(&mut self) {
        if !self.floors_enabled || !matches!(self.progression, Progression::Playing) {
            return;
        }
        let floor = self.floor + 1;
        match self.build_floor(floor, false) {
            Ok(next) => {
                self.progression = Progression::Descending {
                    left: DESCEND_GRACE,
                    next: Box::new(next),
                }
            }
            Err(e) => tracing::error!(
                "dungeon: cannot build floor {floor}: {e:#} — staying on floor {}",
                self.floor
            ),
        }
    }

    /// Advance the descent grace, and install the next floor when it runs out.
    fn tick_progression(&mut self, dt: f32) {
        let Progression::Descending { left, .. } = &mut self.progression else {
            return;
        };
        *left -= dt;
        if *left > 0.0 {
            return;
        }
        let Progression::Descending { next, .. } =
            std::mem::replace(&mut self.progression, Progression::Playing)
        else {
            unreachable!("just matched Descending");
        };
        tracing::info!(
            "dungeon: descending to floor {} with {:.0}/{:.0} hit points",
            next.floor,
            self.warrior.health().current,
            self.warrior.health().max,
        );
        self.audio.play(Sfx::FloorExit, 0.8, 0.0);
        self.install(*next);
    }

    /// Whether this step's input asks for a restart. Only a dead run may: R is not a
    /// panic button that throws away a floor mid-fight.
    fn restart_requested(&self) -> bool {
        self.floors_enabled
            && self.warrior.is_dead()
            && self.input.just_pressed(Action::Restart)
            && matches!(self.progression, Progression::Playing)
    }

    /// Start the run again: the same seed, from the first floor, with a warrior that has
    /// its hit points back.
    ///
    /// The *same* seed on purpose — a run is its seed, and a death that re-rolled the
    /// dungeon would make "try that again" impossible. `--seed` is how you ask for a
    /// different one.
    fn restart(&mut self) {
        match self.build_floor(FIRST_FLOOR, true) {
            Ok(next) => {
                tracing::info!(
                    "dungeon: restarting run {} from floor {FIRST_FLOOR} after {} steps",
                    self.run_seed,
                    self.steps,
                );
                self.install(next);
            }
            Err(e) => tracing::error!("dungeon: cannot restart run {}: {e:#}", self.run_seed),
        }
    }

    /// Identify a level entity by the scene-graph name the level authored on it.
    ///
    /// Both sides read the same constant, so a renamed or moved character cannot
    /// silently break the lookup — which a spawn-transform match could: it would identify
    /// the player as "the renderable standing where the player starts", so a second
    /// entity placed at the spawn, or a spawn that moved, quietly took over.
    fn find_named(world: &World, name: &str) -> Option<Entity> {
        world
            .iter::<Name>()
            .find(|(_, n)| n.0 == name)
            .map(|(entity, _)| entity)
    }

    /// Resolve the whole cast after a level (re)load: the player, the monsters, their
    /// bone bindings and their per-instance clips.
    ///
    /// Returns `false` while the level has no player entity — the state a headless
    /// single-frame capture and an empty test world are both in — so the caller can skip
    /// the step rather than simulate a character that is not there.
    ///
    /// The player's authored position is snapped through `nearest_free` before it is
    /// believed: the level writer places the character on exactly this point, but a
    /// hot-swapped or hand-edited level need not, and starting the simulation inside rock
    /// is the one state the mover cannot be asked to fix mid-stride.
    ///
    /// This is also where a **floor transition finishes**: the descent cleared the cast
    /// and asked for a new world, and the run resumes on the step this finds the new
    /// world's player. The warrior is rebuilt or kept according to `reset_warrior` —
    /// which is the whole of the carry-over rule, in one branch.
    fn acquire(&mut self, world: &mut World) -> bool {
        if let Some(cast) = &self.cast
            && world.is_alive(cast.player)
        {
            return true;
        }
        let Some(player) = Self::find_named(world, PLAYER_NAME) else {
            return false;
        };

        // One hierarchy sweep for the whole cast (see `ChildIndex`).
        let index = ChildIndex::build(world);
        let player_anim = Self::bind(world, &index, player, &self.warrior_rig, PLAYER_NAME);

        let authored = world
            .get::<LocalTransform>(player)
            .map(|t| t.translation)
            .unwrap_or_else(|| collision::player_spawn(&self.grid));
        let start = collision::collision(&self.grid)
            .nearest_free(collision::to_collision(&self.grid, authored), PLAYER_RADIUS)
            .unwrap_or_else(|| collision::player_spawn_local(&self.grid));

        // The monsters. A level with fewer `grunt_<i>` entities than spawn points (an
        // older `.level` for this seed, hand-edited or written by a different
        // `--grunts`) simply fields fewer monsters: the pairing is by name, so a missing
        // one costs that one and nothing else.
        let mut grunts = Vec::new();
        let mut grunt_anims = Vec::new();
        for (i, &spawn) in self.grunt_spawns.iter().enumerate() {
            let name = grunt_name(i);
            let Some(entity) = Self::find_named(world, &name) else {
                tracing::warn!("dungeon: level has no '{name}' — that monster is skipped");
                continue;
            };
            grunt_anims.push(Self::bind(world, &index, entity, &self.grunt_rig, &name));
            grunts.push(GruntBrain::new(&self.grunt_class, entity, spawn));
            // The ECS half of a monster: what anything that queries the world (a future
            // AoE, a targeting reticle, a save file) sees. The brain and the combat crate
            // own the behaviour; these are the facts.
            world.insert(entity, Health::new(self.grunt_class.max_health));
            world.insert(entity, Team::ENEMY);
            world.insert(entity, BodyCircle::new(self.grunt_class.radius));
        }

        let views = vec![GruntView::default(); grunts.len()];
        tracing::info!(
            "dungeon: cast acquired on floor {} — player #{} ({} clips), {} grunts",
            self.floor,
            player.index(),
            player_anim.clips.len(),
            grunts.len(),
        );

        self.cast = Some(Cast {
            player,
            player_anim,
            grunts,
            grunt_anims,
            grunt_views: views,
        });
        // A new run gets a new warrior; a descent keeps the one that walked down the
        // stairs, hit points and all. `reset_warrior` returns to its default afterwards
        // so an *unrequested* reload (the engine's hot-swap dropdown) is a fresh start
        // again — nothing about that level is a continuation of this run.
        //
        // The pocket goes with the warrior, in the same branch: a run that carried two
        // flasks down four floors has them until it *ends*, and a warrior built fresh has
        // never picked one up. Doing it here rather than in `install` is what makes that
        // one rule instead of two — the hot-swap path never reaches `install`.
        if self.reset_warrior {
            self.warrior = WarriorController::new();
            self.inventory = Inventory::new();
        }
        self.reset_warrior = true;
        if matches!(self.progression, Progression::Awaiting) {
            self.progression = Progression::Playing;
        }
        self.prev_pos = start;
        self.pos = start;
        self.prev_yaw = self.warrior.facing_radians();
        self.yaw = self.prev_yaw;
        let focus = collision::to_world(&self.grid, start, CAMERA_FOCUS_Y);
        self.prev_focus = focus;
        self.focus = focus;
        self.damage.clear();
        self.deaths.clear();
        true
    }

    /// Bind one character's rig to the sub-tree the level built under `root`.
    fn bind(
        world: &World,
        index: &ChildIndex,
        root: Entity,
        rig: &Rig,
        label: &str,
    ) -> CharacterAnim {
        let binding = RigBinding::resolve(world, index, root, rig);
        // Loud but not fatal: an unbound bone costs that bone's motion, and a character
        // that stands in its rest pose is a far better failure than a level that refuses
        // to load (see `characters::RigBinding::resolve`). A *wholly* unbound character
        // is a different fault — the level named a character but placed no rig under it —
        // so it gets its own line rather than hiding as "0/16".
        if binding.is_empty() {
            tracing::warn!("dungeon: '{label}' has no rig sub-tree — it will not animate");
        } else if !binding.is_complete() {
            tracing::warn!(
                "dungeon: '{label}' bound {}/{} rig bones — the rest hold their rest pose",
                binding.resolved(),
                rig.nodes.len(),
            );
        }
        let clips = ClipSet::build(rig, &binding);
        debug_assert!(!clips.is_empty(), "every rig authors at least one clip");
        CharacterAnim { root, clips }
    }

    /// This step's input snapshot: the platform's, plus whatever [`HOLD_ENV`] /
    /// [`TAP_ENV`] force on top of it.
    fn stage_input(&mut self, snapshot: &InputSnapshot) {
        let held: Vec<InputSource> = self
            .forced
            .iter()
            .filter(|hold| hold.covers(self.steps))
            .map(|hold| hold.source)
            .collect();
        let taps: Vec<InputSource> = self
            .taps
            .iter()
            .filter(|&&(step, _)| step == self.steps)
            .map(|&(_, source)| source)
            .collect();
        if held.is_empty() && taps.is_empty() {
            self.input.update(snapshot);
            return;
        }
        let mut forced = snapshot.clone();
        for source in held.iter().chain(&taps) {
            match *source {
                InputSource::Key(vk) => forced.set_key(vk, true),
                InputSource::MouseButton(b) => forced.set_mouse_button(b, true),
                // A wheel notch is a per-frame delta, not a held state; nothing in this
                // game binds one, so forcing it would be inventing a meaning for it.
                InputSource::Wheel(_) => {}
            }
        }
        self.input.update(&forced);
    }

    /// This step's player intent, reduced to what the controller accepts.
    ///
    /// The camera's yaw is fixed, so camera-relative input maps straight onto world axes:
    /// +X right, -Z away. The two buttons are **edges** (`just_pressed`), which is the
    /// contract [`WarriorInput`] states: holding attack must not mash.
    fn warrior_input(&self) -> WarriorInput {
        let stick = self.input.axis2d(
            Action::MoveLeft,
            Action::MoveRight,
            Action::MoveBack,
            Action::MoveForward,
        );
        WarriorInput {
            // Collision space is world XZ up to a translation, so the stick maps
            // component-wise: forward (+stick.y) is -Z.
            move_dir: Vec2::new(stick.x, -stick.y),
            sprint: self.input.pressed(Action::Sprint),
            attack_pressed: self.input.just_pressed(Action::Attack),
            dodge_pressed: self.input.just_pressed(Action::Dodge),
        }
    }

    /// The player as the monsters are allowed to see it (see [`PlayerView`]).
    fn player_view(&self, cast: &Cast) -> PlayerView {
        PlayerView {
            entity: cast.player,
            pos: self.pos,
            radius: PLAYER_RADIUS,
            alive: !self.warrior.is_dead(),
        }
    }

    /// The living monsters, as swing candidates. A corpse is not offered —
    /// [`GruntBrain::as_target`] is the one place that decides it.
    fn targets(&self, cast: &Cast) -> Vec<Target> {
        cast.grunts
            .iter()
            .filter_map(|g| g.as_target(&self.grunt_class))
            .map(|(entity, position, radius, team)| Target {
                entity,
                position,
                radius,
                team,
            })
            .collect()
    }

    /// The player's speed readout, metres/second: how far the last fixed step actually
    /// moved it, over that step's own `dt`.
    ///
    /// Measured rather than read off the class, so what the HUD shows is what happened —
    /// a wall, a swing's movement lock or a dodge's burst all show up here, and none of
    /// them would in `move_speed`.
    fn speed(&self) -> f32 {
        if self.last_dt > 0.0 {
            self.prev_pos.distance(self.pos) / self.last_dt
        } else {
            0.0
        }
    }

    /// The tile the player is standing on (grid coordinates).
    fn player_tile(&self) -> (i32, i32) {
        collision::tile_of(self.pos)
    }

    /// Where the player *is* this rendered frame: the two latest fixed-step positions
    /// blended by the frame's interpolation factor. The camera, the HUD readout and the
    /// character mesh all read this one function, so they cannot disagree.
    fn render_position(&self, alpha: f32) -> Vec3 {
        collision::to_world(&self.grid, self.prev_pos.lerp(self.pos, alpha), CHARACTER_Y)
    }

    /// Monsters still standing, and how many there were.
    fn grunt_tally(&self) -> (usize, usize) {
        match &self.cast {
            Some(cast) => (
                cast.grunts.iter().filter(|g| !g.is_dead()).count(),
                cast.grunts.len(),
            ),
            None => (0, self.grunt_spawns.len()),
        }
    }

    /// One simulation step against one frame of input.
    ///
    /// This *is* the hook body — `fixed_update` forwards straight to it — because the
    /// engine hands games an `InputSnapshot` rather than the platform `Input` whose state
    /// has no public setter. So the whole step is exercisable from a test with no window,
    /// device or swapchain, against the same code the game runs.
    ///
    /// # Order, and why
    ///
    /// 1. **Input**, once per step. Every step of a frame sees the same platform state
    ///    (the window is pumped once per frame), so an edge is reported on the first step
    ///    of the frame it happened in and not repeated by the later ones.
    /// 2. **Acquire** the cast if the level has just loaded.
    /// 3. **[`tick_iframes`]**, first among the combat calls, so a window that expires
    ///    this step stops protecting *this* step (its own docs say so).
    /// 4. **The player swings**, at a position it resolves itself. It reads the monsters'
    ///    start-of-step positions, which is the same instant they will read the player's.
    /// 5. **The monsters swing**, reading the player's *new* position. The asymmetry is
    ///    deliberate and is the one the player benefits from: a step away from a claw is
    ///    seen by the claw, so the escape has to be a real one — while the player's own
    ///    swing cannot be dodged by a monster that has not moved yet. A simultaneous
    ///    resolve would need a double-buffered world for a 16 ms difference nobody can
    ///    perceive.
    /// 6. **Damage applies once**, to everything both sides queued, and only then does
    ///    either side learn what landed:
    ///    * monsters keep their hit points in the ECS, so [`apply_damage_events`] is the
    ///      authority and [`ai::feed_combat`] carries the result back into the brains;
    ///    * the player keeps its hit points in [`WarriorController`] (its docs explain
    ///      why: death has to stop input, cancel the swing and lock the graph, and asking
    ///      the world every step would race whoever wrote it). The player entity
    ///      therefore carries **no `Health` component**, which is not an omission but the
    ///      mechanism: `apply_damage_events` skips a target with no health, so the same
    ///      one call resolves the monsters and passes the player's hits through untouched
    ///      for [`WarriorController::take_damage`] — which honours i-frames itself. No
    ///      event is applied twice and none is dropped.
    /// 7. **Presentation state** — bone poses, the yaw pair, the camera trail — last, on
    ///    a step that has finished deciding what happened.
    fn simulate(&mut self, world: &mut World, snapshot: &InputSnapshot, dt: f32) {
        let started = Instant::now();
        self.stage_input(snapshot);

        // -- 1b. the run's own transitions -------------------------------------------------
        // A dead run restarts on demand, before anything else this step decides anything.
        if self.restart_requested() {
            self.restart();
        }
        // The descent grace runs down at the *top* of a step, so the floor it installs is
        // installed before anything reads the grid — the alternative (ticking it at the
        // end, next to the exit detection that starts it) would swap the grid out from
        // under a step that had already taken the cast out of `self`, and put it back.
        self.tick_progression(dt);
        // Between floors the grid is already the *next* floor's and the world is still
        // the previous floor's, so there is nothing here that could be simulated
        // coherently. `Handoff` is that gap — one frame at most, since the engine polls
        // the hook and swaps before it runs a fixed step (`sandbox::App::frame`). By
        // `Awaiting` the world is the new one, and the acquisition below resolves out of
        // it, which is what ends the transition.
        if matches!(self.progression, Progression::Handoff { .. }) {
            return;
        }

        if !self.acquire(world) {
            return; // level not loaded (or has no player) — nothing to simulate
        }
        let mut cast = self.cast.take().expect("acquire() succeeded");
        self.steps += 1;
        self.last_dt = dt;

        // -- 3. i-frames -----------------------------------------------------------------
        tick_iframes(world, dt);

        // -- 4. the player ---------------------------------------------------------------
        let targets = self.targets(&cast);
        let out = self.warrior.tick(
            self.warrior_input(),
            dt,
            WarriorCtx {
                collision: collision::collision(&self.grid),
                position: self.pos,
                radius: PLAYER_RADIUS,
                attacker: cast.player,
                targets: &targets,
            },
        );
        self.prev_pos = self.pos;
        self.pos = out.position;
        // -- 4b. what the step SOUNDS like (listener = the player, so the player's own
        // actions are centred; spatial pan is for the world around them) ---------------
        if matches!(
            out.state,
            WarriorState::Attacking {
                phase: AttackPhase::Windup,
                ..
            }
        ) && !matches!(
            self.player_state,
            WarriorState::Attacking {
                phase: AttackPhase::Windup,
                ..
            }
        ) {
            self.audio.play(Sfx::SwordSwing, 0.7, 0.0);
        }
        let moved = self.prev_pos.distance(self.pos);
        if moved > 1.0e-4 && !self.warrior.is_dead() {
            self.foot_timer -= dt;
            if self.foot_timer <= 0.0 {
                // Cadence follows actual speed, so a sprint simply steps faster.
                let speed = moved / dt.max(1.0e-6);
                self.foot_timer = (1.4 / speed.max(1.0)).clamp(0.24, 0.55);
                self.audio.play(Sfx::Footstep, 0.45, 0.0);
            }
        } else {
            self.foot_timer = 0.0;
        }
        for hit in &out.hits {
            // Metal-on-target crack, panned a touch toward the push direction.
            self.audio.play(
                Sfx::SwordHit,
                0.85,
                (hit.direction.x * 0.4).clamp(-0.5, 0.5),
            );
        }
        for hit in &out.hits {
            self.damage.send(*hit);
        }
        // The controller is the authority on the window, so it is *assigned*, not
        // refreshed — a window that just closed must close in the ECS too.
        world.insert(cast.player, out.iframes);

        // The mover guarantees a finite, outside-geometry result for finite inputs; this
        // is the tripwire for the day something upstream (a NaN dt, a corrupt level)
        // breaks that assumption, caught here instead of as a vanished player.
        assert!(
            self.pos.is_finite(),
            "dungeon: non-finite player position {:?}",
            self.pos
        );

        // -- 5. the monsters -------------------------------------------------------------
        let view = self.player_view(&cast);
        ai::tick_grunts(
            &self.grid,
            &self.grunt_class,
            &mut self.finder,
            &mut cast.grunts,
            view,
            dt,
            &mut self.damage,
        );

        // -- 6. damage -------------------------------------------------------------------
        // The player's share, read out before the ECS pass (which cannot see it: the
        // player entity carries no `Health` — see the order note above).
        let stagger = self.grunt_class.swing().map_or(0.0, |spec| spec.stagger);
        let player_hits: Vec<IncomingHit> = self
            .damage
            .iter()
            .filter(|event| event.target == cast.player)
            .map(|event| IncomingHit {
                amount: event.amount,
                direction: event.direction,
                stagger,
            })
            .collect();

        apply_damage_events(world, self.damage.iter(), &mut self.deaths);
        ai::feed_combat(&mut cast.grunts, self.damage.iter(), self.deaths.iter());
        for (i, entity) in self.deaths.iter().map(|d| d.entity).enumerate() {
            // Cap the death-cry stack: a whirlwind triple kill is one loud moment, not
            // three clipping ones.
            if i < 2 {
                self.audio.play(Sfx::GruntDeath, 0.75, 0.0);
            }
            Self::retire_corpse(world, entity);
        }
        let report = self.warrior.take_damage(player_hits);
        if report.taken > 0.0 {
            self.audio.play(Sfx::GruntHit, 0.8, 0.0);
        }
        if report.died {
            self.audio.play(Sfx::GruntDeath, 0.9, 0.0);
        }
        if report.died {
            tracing::info!(
                "dungeon: the warrior died after {} steps ({:.1} s)",
                self.steps,
                self.steps as f64 * f64::from(dt),
            );
        }
        self.damage.update();
        self.deaths.update();

        // -- 6b. the loot ----------------------------------------------------------------
        // After damage, so a step that kills the player does not also let the corpse pocket
        // the flask it slid onto; before presentation, so the flask a pickup empties is
        // hidden on the same step the HUD count goes up.
        self.collect_and_drink(world);

        // The torch brackets are the floor's loop emitters: retarget their gain/pan from
        // the player's new position every step (slot 1 + i; the mixer smooths, so 60 Hz
        // targets can't zipper). LOOP_SLOTS bounds the count with slot 0 reserved.
        let listener = [self.pos.x, self.pos.y];
        for (i, p) in self.torch_pts.iter().enumerate().take(31) {
            self.audio
                .loop_at(1 + i as u8, Sfx::TorchLoop, listener, [p.x, p.y], 0.55);
        }

        // -- 7. presentation -------------------------------------------------------------
        self.pose_player(world, &mut cast, &out);
        self.pose_grunts(world, &mut cast);
        self.prev_yaw = self.yaw;
        self.yaw = out.facing_radians;
        self.player_state = out.state;

        self.detect_exit(dt);

        // The camera follows a smoothed trail behind the player, advanced on the same
        // fixed step so the filter is framerate-independent.
        let target = collision::to_world(&self.grid, self.pos, CAMERA_FOCUS_Y);
        self.prev_focus = self.focus;
        self.focus += (target - self.focus) * blend(dt, CAMERA_SMOOTH_TIME);

        self.cast = Some(cast);
        self.record_cost(started);
    }

    /// The item half of a step: collect whatever the player is standing on, and drink if
    /// this step's input asked for it.
    ///
    /// **Both halves are refused to a corpse.** A dead warrior is not offered the drink
    /// (the controller would refuse the heal anyway, but the potion would still be spent —
    /// [`Inventory::drink`] says so), and is not ticked against the flasks at all: a body
    /// sliding onto one must not pocket it.
    ///
    /// The pickup rule itself lives in [`ItemWorld::tick`], which takes the inventory so
    /// that a full pocket leaves the flask standing. What is left here is the half that
    /// module deliberately does not do: making the *visual* go away.
    fn collect_and_drink(&mut self, world: &mut World) {
        if self.warrior.is_dead() {
            return;
        }
        for event in self.items.tick(self.pos, &mut self.inventory) {
            tracing::info!(
                "dungeon: picked up {} at step {} — carrying {}/{}",
                event.name(),
                self.steps,
                event.carried,
                self.items.def().max_carry,
            );
            self.audio.play(Sfx::PotionPickup, 0.7, 0.0);
            Self::hide_entity(world, &event.name());
        }
        if self.input.just_pressed(Action::Drink)
            && let Some(heal) = self.items.drink(&mut self.inventory)
        {
            self.audio.play(Sfx::PotionDrink, 0.8, 0.0);
            let restored = self.warrior.heal(heal);
            tracing::info!(
                "dungeon: drank a potion at step {} — {restored:.0} hit points restored, \
                 {} left",
                self.steps,
                self.inventory.potions,
            );
        }
    }

    /// Make the level entity called `name` — and everything under it — stop drawing.
    ///
    /// A collected flask is *hidden*, not despawned: the entity is the level's, its
    /// sub-tree is what the loader built for the prop's `.glb`, and removing the
    /// [`MeshInstance`] from all of it takes it out of the draw list while leaving the
    /// hierarchy the loader owns intact. Despawning would leave a hole in a structure this
    /// game did not build and cannot rebuild — and there is nothing to gain: a floor has
    /// at most a handful of flasks, and the draw is gone either way.
    ///
    /// This is [`Self::retire_corpse`]'s pattern (drop the components that make a thing
    /// participate, keep the thing) applied to a drawable instead of a combatant. Nothing
    /// re-shows it, because nothing puts a collected potion back.
    fn hide_entity(world: &mut World, name: &str) {
        let Some(root) = Self::find_named(world, name) else {
            // A level that has no such entity is a level written by a different build —
            // the pickup still counted (the rule is the simulation's), there is simply
            // nothing to hide.
            tracing::warn!("dungeon: level has no '{name}' to hide");
            return;
        };
        let index = ChildIndex::build(world);
        for entity in index.subtree(root) {
            world.remove::<MeshInstance>(entity);
        }
    }

    /// A dead monster stops being a body: it keeps its (zeroed) [`Health`] as the record
    /// that it died, and loses the components that make it a *participant*.
    ///
    /// The brain already refuses to offer a corpse as a target
    /// ([`GruntBrain::as_target`]) and separation already ignores one, so this is not
    /// what stops the fight — it is what stops the *next* thing. Anything that finds
    /// combatants by querying the world (an AoE, a targeting reticle, a second monster
    /// type) asks for `Team` + `BodyCircle`, and a corpse that still answers is a corpse
    /// that still soaks hits and blocks a doorway.
    fn retire_corpse(world: &mut World, entity: Entity) {
        world.remove::<Team>(entity);
        world.remove::<BodyCircle>(entity);
        world.remove::<IFrames>(entity);
    }

    /// Commit the player's pose for this step.
    fn pose_player(
        &mut self,
        world: &mut World,
        cast: &mut Cast,
        out: &crate::warrior::WarriorOutput,
    ) {
        let sample = |s: AnimSample, weight: f32| {
            ClipSample::new(s.clip.rig_clip(), s.time, s.looping).with_weight(weight)
        };
        let current = sample(out.anim, out.blend.map_or(1.0, |b| b.alpha));
        let fade = out.blend.map(|b| sample(b.from, 1.0 - b.alpha));
        cast.player_anim.clips.apply(world, current, fade);
    }

    /// Commit every monster's pose, and latch its yaw pair for the render frame.
    ///
    /// The grunt's graph names its clips exactly as the rig does, so the machine's clip
    /// name is the clip key with no mapping (the warrior needs one — see
    /// [`crate::warrior::WarriorClip`] — because its graph is named after swings and its
    /// rig after slots).
    fn pose_grunts(&mut self, world: &mut World, cast: &mut Cast) {
        for (i, grunt) in cast.grunts.iter().enumerate() {
            let (Some(anim), Some(view)) =
                (cast.grunt_anims.get_mut(i), cast.grunt_views.get_mut(i))
            else {
                continue;
            };
            let machine = grunt.anim();
            let (clip, time) = machine.current();
            let alpha = machine.fade().map_or(1.0, |(_, _, a)| a);
            let current = ClipSample::new(clip, time, clip_loops(machine, clip)).with_weight(alpha);
            let fade = machine.fade().map(|(clip, time, alpha)| {
                ClipSample::new(clip, time, clip_loops(machine, clip)).with_weight(1.0 - alpha)
            });
            anim.clips.apply(world, current, fade);

            view.prev_yaw = view.yaw;
            view.yaw = grunt.yaw();
        }
    }

    /// Exit detection, and the start of the descent it now means.
    ///
    /// The latch is what keeps this a *transition* rather than a repeated event: a player
    /// standing on the exit tile touches it on every step of the grace, and only the
    /// first one counts. A grid whose exit is its entry (the injection harness, the
    /// hand-built test arenas) has no exit to reach.
    fn detect_exit(&mut self, dt: f32) {
        if self.exit_reached
            || self.grid.exit() == self.grid.entry()
            || !collision::circle_overlaps_tile(self.pos, PLAYER_RADIUS, self.grid.exit())
        {
            return;
        }
        self.exit_reached = true;
        let (ex, ez) = self.grid.exit();
        tracing::info!(
            "dungeon: exit reached at tile ({ex}, {ez}) after {} steps ({:.1} s)",
            self.steps,
            self.steps as f64 * f64::from(dt),
        );
        self.begin_descent();
    }

    /// Book this step's cost, and complain once if the sim is eating the frame.
    fn record_cost(&mut self, started: Instant) {
        self.sim_ms = started.elapsed().as_secs_f32() * 1e3;
        self.sim_ms_peak = self.sim_ms_peak.max(self.sim_ms);
        if self.sim_ms > SIM_BUDGET_MS && !self.budget_warned {
            self.budget_warned = true;
            let (alive, total) = self.grunt_tally();
            tracing::warn!(
                "dungeon: fixed step took {:.2} ms (budget {SIM_BUDGET_MS:.1} ms) \
                 with {alive}/{total} grunts alive at step {}",
                self.sim_ms,
                self.steps,
            );
        }
    }
}

/// Whether the state playing `clip` in `machine` wraps.
///
/// Read from the graph rather than from a list here: the graph is the file that decides,
/// and a second list would be a second opinion. A clip with no state (unreachable — the
/// machine only ever reports a state's own clip) falls back to one-shot, which clamps,
/// which is the safe end: a locomotion clip played once looks stuck, a one-shot played
/// looping resurrects a corpse.
fn clip_loops(machine: &AnimMachine, clip: &str) -> bool {
    machine
        .def()
        .states
        .iter()
        .find(|s| s.clip == clip)
        .is_some_and(|s| s.looping)
}

impl GameHooks for DungeonGame {
    fn fixed_update(&mut self, world: &mut World, input: &InputSnapshot, dt: f32) {
        self.simulate(world, input, dt);
    }

    /// Push the *rendered* root transforms onto the characters, once per frame, right
    /// before the draw list is built.
    ///
    /// Without this a character draws at its last simulated pose while the camera is
    /// already at the interpolated one, so at high frame rates the player visibly trails
    /// the view by up to a step of travel (~14 cm at sprint). All of it now reads the
    /// same `alpha`, so everything moves as one.
    ///
    /// **Roots only.** Bone locals were written by `fixed_update`; these are the
    /// characters' placement wrappers, whose transform nothing else touches (see the
    /// module docs on the single-writer split).
    ///
    /// Visual-only, per the hook contract: nothing here feeds back into the simulation —
    /// the next `fixed_update` integrates from `self.pos` and from each brain's own
    /// position, never from the ECS.
    fn render_update(&mut self, world: &mut World, alpha: f32) {
        let Some(cast) = self.cast.take() else {
            return;
        };
        let place = |world: &mut World, entity: Entity, translation: Vec3, yaw: f32| {
            if let Some(local) = world.get_mut::<LocalTransform>(entity) {
                local.translation = translation;
                local.rotation = Quat::from_rotation_y(yaw);
            }
        };
        if world.is_alive(cast.player_anim.root) {
            place(
                world,
                cast.player_anim.root,
                self.render_position(alpha),
                lerp_angle(self.prev_yaw, self.yaw, alpha),
            );
        }
        for (i, grunt) in cast.grunts.iter().enumerate() {
            let (Some(anim), Some(view)) = (cast.grunt_anims.get(i), cast.grunt_views.get(i))
            else {
                continue;
            };
            if !world.is_alive(anim.root) {
                continue;
            }
            place(
                world,
                anim.root,
                grunt.render_position(&self.grid, alpha, CHARACTER_Y),
                lerp_angle(view.prev_yaw, view.yaw, alpha),
            );
        }
        self.cast = Some(cast);
    }

    /// Hand over the next floor's level — **exactly once per transition**.
    ///
    /// The level is *taken* out of the `Handoff` state, so however many frames pass
    /// before the engine gets to the swap (and however many times it polls), one descent
    /// asks for one load. The selector is an explicit path to a file this game wrote
    /// moments ago; the engine registers it on the spot (see
    /// [`crate::level::dungeon_level_selector`]). If it fails to load, that error
    /// propagates out of the frame loop and the process stops — which is the right
    /// failure for "the floor I just generated cannot be read".
    fn next_level(&mut self) -> Option<String> {
        let Progression::Handoff { level } = &mut self.progression else {
            return None;
        };
        let level = std::mem::take(level);
        self.progression = Progression::Awaiting;
        tracing::info!("dungeon: floor {} — loading '{level}'", self.floor);
        Some(level)
    }

    fn camera(&mut self, alpha: f32) -> Option<CameraPose> {
        // Blend the two latest sim states — the hook must return the *rendered* pose;
        // the frame loop does not interpolate it. Smoothing itself happened on the
        // fixed step, so what is left here is a pure lerp: no per-frame filtering, no
        // jitter when several frames fall inside one step.
        self.render_alpha = alpha;
        let focus = self.prev_focus.lerp(self.focus, alpha);
        Some(CameraPose::look_at(focus + camera_offset(), focus))
    }

    fn draw_ui(&mut self, ui: &imgui::Ui, world: &World) {
        let dt_ms = ui.io().delta_time * 1000.0;
        ui.window("Dungeon")
            // Below the engine's debug window (which opens at the top-left corner at
            // 320 px tall), so the HUD is a readable sibling rather than a cover.
            .position([16.0, 372.0], imgui::Condition::FirstUseEver)
            .size([360.0, 340.0], imgui::Condition::FirstUseEver)
            .build(|| {
                self.draw_vitals(ui);
                self.draw_progress(ui);
                ui.separator();
                self.draw_combat(ui);
                ui.separator();
                self.draw_world(ui, world);
                ui.separator();
                ui.text(format!("frame     {dt_ms:.2} ms"));
                ui.text(format!(
                    "sim step  {:.3} ms  peak {:.3} ms  ({} steps)",
                    self.sim_ms, self.sim_ms_peak, self.steps
                ));
                ui.text_disabled(
                    "WASD move, Shift sprint, LMB/J attack, Space dodge, Q drink, R restart",
                );
            });
    }
}

impl DungeonGame {
    /// Hit points and the death banner.
    fn draw_vitals(&self, ui: &imgui::Ui) {
        let health = self.warrior.health();
        let fraction = health.fraction();
        // Green → amber → red as the bar empties, so the state is readable from the
        // colour alone at the glance rate a fight allows.
        let colour = if fraction > 0.5 {
            [0.30, 0.78, 0.35, 1.0]
        } else if fraction > 0.25 {
            [0.90, 0.68, 0.20, 1.0]
        } else {
            [0.85, 0.22, 0.20, 1.0]
        };
        let bar = ui.push_style_color(imgui::StyleColor::PlotHistogram, colour);
        imgui::ProgressBar::new(fraction.clamp(0.0, 1.0))
            .size([-1.0, 18.0])
            .overlay_text(format!("{:.0} / {:.0}", health.current, health.max))
            .build(ui);
        bar.end();

        // The pocket, right under the bar it refills: the two numbers are read together
        // ("can I afford this fight?"), so they are shown together. Greyed at zero, which
        // is the state where Q does nothing.
        let def = self.items.def();
        let carried = self.inventory.potions;
        let line = format!(
            "potions   {carried} / {}   ({} on this floor)",
            def.max_carry,
            self.items.remaining()
        );
        if carried == 0 {
            ui.text_disabled(line);
        } else {
            ui.text_colored([0.86, 0.36, 0.38, 1.0], line);
        }

        if self.warrior.is_dead() {
            ui.text_colored([0.92, 0.26, 0.24, 1.0], "YOU DIED");
            if self.floors_enabled {
                ui.text_colored(
                    [0.85, 0.85, 0.85, 1.0],
                    format!(
                        "press R to restart run {} from floor {FIRST_FLOOR}",
                        self.run_seed
                    ),
                );
            } else {
                ui.text_disabled("this harness level has no run to restart");
            }
        }
    }

    /// Where the run is: the floor, and whatever transition is under way.
    ///
    /// The hit-point bar above is the other half of the carry-over story — it does not
    /// refill on a descent, and this line is what says which descent it survived.
    fn draw_progress(&self, ui: &imgui::Ui) {
        ui.text(format!(
            "run       seed {}   floor {}",
            self.run_seed, self.floor
        ));
        match &self.progression {
            Progression::Playing => {}
            Progression::Descending { left, next } => ui.text_colored(
                [0.95, 0.85, 0.45, 1.0],
                // ASCII only: the overlay's font has no glyph for an arrow or an
                // ellipsis, and imgui draws a missing glyph as a question mark.
                format!("descending {:.1} s -> floor {}", left.max(0.0), next.floor),
            ),
            Progression::Handoff { .. } | Progression::Awaiting => ui.text_colored(
                [0.95, 0.85, 0.45, 1.0],
                format!("loading floor {}...", self.floor),
            ),
        }
    }

    /// The combo/attack readout and the monster tally.
    fn draw_combat(&self, ui: &imgui::Ui) {
        let state = match (self.cast.is_some(), self.player_state) {
            (true, WarriorState::Attacking { step, phase }) => {
                format!("attack  combo {}/3  {phase:?}", step + 1)
            }
            (true, other) => format!("{other:?}"),
            (false, _) => "not resolved yet".to_string(),
        };
        ui.text(format!("state     {state}"));
        ui.text(format!("clip      {}", self.warrior.anim_state()));
        if self.warrior.invulnerable() {
            ui.text_colored([0.45, 0.72, 0.95, 1.0], "i-frames  ACTIVE");
        } else {
            ui.text_disabled("i-frames  --");
        }

        let (alive, total) = self.grunt_tally();
        let colour = if alive == 0 {
            [0.45, 0.80, 0.45, 1.0]
        } else {
            [0.88, 0.80, 0.45, 1.0]
        };
        ui.text_colored(colour, format!("grunts    {alive} / {total} alive"));
        if let Some(cast) = &self.cast {
            let chasing = cast
                .grunts
                .iter()
                .filter(|g| matches!(g.state(), GruntState::Chase | GruntState::Attack))
                .count();
            ui.text_disabled(format!("          {chasing} engaged"));
        }
    }

    /// Seed, position, tile, exit — the M1 readouts, kept.
    fn draw_world(&self, ui: &imgui::Ui, world: &World) {
        ui.text(format!(
            "floor     seed {}   {}x{} tiles, {} rooms",
            self.grid.seed(),
            self.grid.width(),
            self.grid.height(),
            self.grid.rooms().len()
        ));
        // The torch count is a *data* readout: it is how many point lights this floor's
        // `.level` carries, not how many the renderer drew. When those two disagree the
        // budget is the renderer's to report — see `crate::level::torch_lights`.
        ui.text_disabled(format!(
            "torches   {} placed ({} potions, {} left)",
            self.torches,
            self.items.potions().len(),
            self.items.remaining(),
        ));
        match &self.cast {
            // A single static headless capture never runs a sim step (the capture path
            // is frame-counted), so "not resolved" is the expected reading there;
            // `CAPTURE_SEQ` does step and resolves it.
            Some(cast) if world.is_alive(cast.player) => {
                ui.text(format!("player    entity #{}", cast.player.index()));
            }
            _ => ui.text_disabled("player    not resolved yet"),
        }
        let p = self.render_position(self.render_alpha);
        let (tx, tz) = self.player_tile();
        ui.text(format!("position  {:.2}, {:.2}, {:.2}", p.x, p.y, p.z));
        ui.text(format!(
            "tile      ({tx}, {tz})  {}",
            tile_label(self.grid.get(tx, tz))
        ));
        ui.text(format!("speed     {:.2} m/s", self.speed()));
        let (ex, ez) = self.grid.exit();
        if self.exit_reached {
            ui.text(format!("exit      REACHED ({ex}, {ez})"));
        } else {
            ui.text_disabled(format!("exit      not reached ({ex}, {ez})"));
        }
    }
}

/// Human-readable tile name for the HUD.
fn tile_label(tile: Tile) -> &'static str {
    match tile {
        Tile::Wall => "wall",
        Tile::Floor => "floor",
        Tile::Door => "door",
        Tile::Entry => "entry",
        Tile::Exit => "exit",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{GRUNT_MAX_HEALTH, GRUNT_RADIUS};
    use crate::characters::fixture::spawn_like_the_level_loader;
    use crate::procgen::{DungeonParams, TILE_SIZE, generate};
    use dreamcoast_game::combat::AttackPhase;
    use dreamcoast_game::input::keys;
    use sandbox::scene::{MaterialHandle, MeshHandle, MeshInstance, propagate_transforms};

    const W: u16 = 0x57;
    const S: u16 = 0x53;
    const A: u16 = 0x41;
    const D: u16 = 0x44;
    const J: u16 = 0x4A;
    const Q: u16 = 0x51;
    const R: u16 = 0x52;
    const SPACE: u16 = 0x20;
    const SHIFT: u16 = 0x10;
    const FIXED_DT: f32 = 1.0 / 60.0;
    /// Seeds every property below is swept over. A fixed list, so a failure is a repro.
    const SEEDS: std::ops::Range<u64> = 0..20;

    fn dungeon(seed: u64) -> TileGrid {
        generate(seed, &DungeonParams::default())
    }

    /// A game on `grid` with no monsters — the M1 shape, for the properties that are
    /// about the mover and the level rather than about a fight.
    ///
    /// Progression is real (the state machine under test is the shipping one) but its
    /// one side effect is not: [`stub_writer`] stands in for the level writer, so a test
    /// that walks onto an exit generates a floor without meshing or writing anything.
    fn game(grid: TileGrid) -> DungeonGame {
        let mut g = DungeonGame::new(grid, Population::Fixed(0)).unwrap();
        g.writer = stub_writer;
        g
    }

    /// A [`FloorWriter`] that writes nothing and returns exactly the selector the real
    /// one would have (`crate::level::write_floor` → `dungeon_level_selector`).
    pub(super) fn stub_writer(
        grid: &TileGrid,
        _spawns: &[Vec2],
        _potions: &[Vec2],
    ) -> anyhow::Result<String> {
        Ok(crate::level::dungeon_level_selector(grid.seed()))
    }

    /// A stand-in for the loaded level: an unnamed geometry entity plus every character
    /// the level would have placed, each spawned the way `level::build_level` spawns one
    /// (see `characters::fixture`), so the rig binding under test is the real one.
    pub(super) fn test_world(game: &DungeonGame) -> World {
        let grid = game.grid();
        let mut world = World::new();
        world
            .spawn_node()
            .with(MeshInstance::new(MeshHandle(0), MaterialHandle(0)))
            .with(LocalTransform::default());
        spawn_like_the_level_loader(
            &mut world,
            &game.warrior_rig,
            PLAYER_NAME,
            collision::player_spawn(grid),
        );
        for (i, &spawn) in game.grunt_spawns().iter().enumerate() {
            spawn_like_the_level_loader(
                &mut world,
                &game.grunt_rig,
                &grunt_name(i),
                collision::to_world(grid, spawn, CHARACTER_Y),
            );
        }
        // The flasks, from the same list the level writer places `potion_<i>` from — so a
        // test pickup hides the entity the shipping game would have hidden.
        let potion_rig = crate::rigs::potion();
        for (i, &at) in game.potion_spawns().iter().enumerate() {
            spawn_like_the_level_loader(
                &mut world,
                &potion_rig,
                &items::potion_name(i),
                collision::to_world(grid, at, items::POTION_Y),
            );
        }
        world
    }

    /// Every entity in `name`'s sub-tree that still draws.
    fn drawn_parts(world: &World, name: &str) -> usize {
        let root = DungeonGame::find_named(world, name).expect("the level placed it");
        ChildIndex::build(world)
            .subtree(root)
            .into_iter()
            .filter(|&e| world.get::<MeshInstance>(e).is_some())
            .count()
    }

    /// Advance the game the way the frame loop does: whole fixed steps, then the
    /// once-per-frame presentation pass that writes the visual transforms.
    fn frame(game: &mut DungeonGame, world: &mut World, input: &InputSnapshot, steps: u32) {
        for _ in 0..steps {
            game.simulate(world, input, FIXED_DT);
        }
        game.render_update(world, 1.0);
    }

    /// The player's placement root, once acquired.
    fn player_root(game: &DungeonGame) -> Entity {
        game.cast.as_ref().expect("cast acquired").player
    }

    // -- configuration -------------------------------------------------------------------

    /// The bindings file must resolve — a typo in it is a startup failure, so catch it
    /// in a test rather than on the player's machine. Every action the controller needs
    /// must be bound, including the two M2 added.
    #[test]
    fn bindings_resolve() {
        let g = game(dungeon(1));
        for action in [
            Action::MoveForward,
            Action::MoveBack,
            Action::MoveLeft,
            Action::MoveRight,
            Action::Sprint,
            Action::Attack,
            Action::Dodge,
            Action::Drink,
            Action::Restart,
        ] {
            assert!(
                !g.input.map().sources(action).is_empty(),
                "{action:?} is unbound"
            );
        }
        // Attack is reachable without a mouse — a headless capture has no mouse either.
        let mut keyboard_only = g;
        keyboard_only
            .input
            .update(&InputSnapshot::default().with_key(J, true));
        assert!(keyboard_only.input.just_pressed(Action::Attack));

        // M3's restart is on R, and it is an edge like the other two buttons: holding it
        // through the death banner must ask once, not once a step.
        keyboard_only
            .input
            .update(&InputSnapshot::default().with_key(R, true));
        assert!(keyboard_only.input.just_pressed(Action::Restart));
        keyboard_only
            .input
            .update(&InputSnapshot::default().with_key(R, true));
        assert!(!keyboard_only.input.just_pressed(Action::Restart));
    }

    /// `DUNGEON_HOLD` / `DUNGEON_TAP` are what make the headless capture drivable, so
    /// their parsers are pinned: keys and mouse buttons both resolve through the same
    /// name table the bindings file uses, and an unknown name is ignored rather than
    /// fatal.
    #[test]
    fn capture_input_specs_resolve_sources_and_keys() {
        assert_eq!(keys::key_vk("W"), Some(W));
        assert_eq!(keys::key_vk("NoSuchKey"), None);
        assert_eq!(InputSource::from_name("W"), Some(InputSource::Key(W)));
        assert_eq!(
            InputSource::from_name("Mouse1"),
            Some(InputSource::MouseButton(0))
        );
        assert_eq!(InputSource::from_name("NoSuchKey"), None);
    }

    /// The `DUNGEON_HOLD` grammar: a bare name is the whole run (the M1 form, which
    /// existing capture recipes still use), `@a-b` is an inclusive step window, and
    /// anything else is rejected rather than silently reinterpreted.
    #[test]
    fn a_hold_spec_parses_bare_names_and_step_windows() {
        let bare = parse_hold("W").expect("a bare name is still valid");
        assert_eq!(bare.source, InputSource::Key(W));
        assert!(bare.covers(0) && bare.covers(u64::MAX));

        let windowed = parse_hold("Mouse1@10-20").expect("a windowed source");
        assert_eq!(windowed.source, InputSource::MouseButton(0));
        assert!(!windowed.covers(9));
        assert!(windowed.covers(10) && windowed.covers(15) && windowed.covers(20));
        assert!(
            !windowed.covers(21),
            "the window is inclusive, not open-ended"
        );

        // A single-step window is legal; a backwards or malformed one is not.
        assert!(parse_hold("W@7-7").is_some_and(|h| h.covers(7) && !h.covers(8)));
        assert!(parse_hold("W@9-2").is_none(), "backwards window");
        assert!(parse_hold("W@10").is_none(), "a window needs both bounds");
        assert!(parse_hold("W@a-b").is_none());
        assert!(parse_hold("NoSuchKey@0-1").is_none());
    }

    // -- input mapping -------------------------------------------------------------------

    /// Holding "forward" must aim the controller toward -Z (away from the camera), at
    /// unit length, and never faster for a diagonal.
    #[test]
    fn forward_input_moves_along_negative_z() {
        let mut g = game(dungeon(1));
        g.input.update(&InputSnapshot::default().with_key(W, true));
        let wish = g.warrior_input().move_dir;
        assert!(wish.y < 0.0 && wish.x == 0.0, "{wish:?}");
        assert!((wish.length() - 1.0).abs() < 1e-4);

        // Collision space is world XZ up to a translation, so -Y here is -Z in the world.
        g.input.update(&InputSnapshot::default().with_key(D, true));
        assert!(g.warrior_input().move_dir.x > 0.0, "D is +X");

        // A diagonal is clamped by the controller, not boosted here.
        g.input
            .update(&InputSnapshot::default().with_key(W, true).with_key(D, true));
        let diag = g.warrior_input().move_dir;
        assert!((diag.length() - 1.0).abs() < 1e-4, "{diag:?}");
    }

    /// Sprint is a speed multiplier the class owns; the input layer only reports that
    /// the modifier is held.
    #[test]
    fn sprint_is_faster_than_walk() {
        let mut g = game(dungeon(1));
        g.input.update(&InputSnapshot::default().with_key(W, true));
        assert!(!g.warrior_input().sprint);
        g.input.update(
            &InputSnapshot::default()
                .with_key(W, true)
                .with_key(SHIFT, true),
        );
        assert!(g.warrior_input().sprint);
        let class = g.warrior.class();
        assert!(class.sprint_speed() > class.move_speed);
    }

    /// Attack and dodge are **edges**: holding the button must not mash. The whole
    /// combo system depends on this, so it is asserted at the seam rather than assumed.
    #[test]
    fn the_attack_and_dodge_bindings_fire_on_an_edge_not_a_hold() {
        let mut g = game(dungeon(1));
        let held = InputSnapshot::default()
            .with_mouse_button(0, true)
            .with_key(SPACE, true);
        g.input.update(&held);
        assert!(g.warrior_input().attack_pressed, "first frame is the edge");
        assert!(g.warrior_input().dodge_pressed);
        g.input.update(&held);
        assert!(!g.warrior_input().attack_pressed, "a hold is not a mash");
        assert!(!g.warrior_input().dodge_pressed);
        // Release, press again: a new edge.
        g.input.update(&InputSnapshot::default());
        g.input.update(&held);
        assert!(g.warrior_input().attack_pressed);
    }

    // -- the seam ------------------------------------------------------------------------

    /// The whole frame, end to end: the warrior is identified by name in a level-like
    /// world, held input walks it, and the result lands on its ECS root transform.
    #[test]
    fn a_held_key_moves_the_player_entity_in_the_ecs() {
        // A seed whose spawn has open floor to the north, so "held W moves" is a
        // statement about the mover and not about which wall happens to be there.
        let grid = dungeon(3);
        let spawn = collision::player_spawn(&grid);
        let mut g = game(grid);
        let mut world = test_world(&g);
        frame(
            &mut g,
            &mut world,
            &InputSnapshot::default().with_key(W, true),
            12,
        );
        let moved = world
            .get::<LocalTransform>(player_root(&g))
            .unwrap()
            .translation;
        assert!(moved.z < spawn.z - 0.5, "walked away from the camera");
        assert!((moved.x - spawn.x).abs() < 1e-3);
        assert_eq!(moved.y, CHARACTER_Y, "stays on the floor");
        // The camera trails the player instead of snapping onto it.
        assert!(g.focus.z > moved.z && g.focus.z <= spawn.z + 1e-3);
    }

    /// An unnamed entity is not the player, however suggestively it is placed — the
    /// lookup is by name, not by "whatever stands at the spawn".
    #[test]
    fn an_unnamed_entity_at_the_spawn_is_not_the_player() {
        let grid = dungeon(3);
        let spawn = collision::player_spawn(&grid);
        let mut g = game(grid);
        let mut world = World::new();
        world
            .spawn_node()
            .with(MeshInstance::new(MeshHandle(0), MaterialHandle(0)))
            .with(LocalTransform {
                translation: spawn,
                ..LocalTransform::default()
            });
        frame(
            &mut g,
            &mut world,
            &InputSnapshot::default().with_key(W, true),
            30,
        );
        assert!(g.cast.is_none());
        assert_eq!(g.steps, 0, "no player, no simulation");
    }

    /// The presentation pass is what moves the character, and it moves it to the
    /// *rendered* (interpolated) position — the same one the camera and the HUD read —
    /// not to the last simulated one. Mid-step, that is a real difference.
    #[test]
    fn render_update_writes_the_interpolated_pose() {
        let grid = dungeon(3);
        let spawn = collision::player_spawn(&grid);
        let mut g = game(grid);
        let mut world = test_world(&g);
        let held = InputSnapshot::default().with_key(W, true);
        for _ in 0..12 {
            g.simulate(&mut world, &held, FIXED_DT);
        }
        let player = player_root(&g);
        // Simulation alone leaves the root where the level authored it.
        assert_eq!(
            world.get::<LocalTransform>(player).unwrap().translation,
            spawn
        );

        g.render_update(&mut world, 0.5);
        let drawn = world.get::<LocalTransform>(player).unwrap().translation;
        assert_eq!(drawn, g.render_position(0.5));
        let (prev, now) = (
            collision::to_world(g.grid(), g.prev_pos, CHARACTER_Y),
            collision::to_world(g.grid(), g.pos, CHARACTER_Y),
        );
        assert!(
            drawn.z > now.z && drawn.z < prev.z,
            "the drawn pose sits strictly between the two sim states"
        );
    }

    /// `fixed_update` owns the bones and `render_update` owns the roots — the
    /// single-writer split the module documents. Neither may touch the other's.
    #[test]
    fn the_two_halves_write_disjoint_transforms() {
        let grid = dungeon(3);
        let mut g = game(grid);
        let mut world = test_world(&g);
        let held = InputSnapshot::default().with_key(W, true);
        frame(&mut g, &mut world, &held, 20);

        let root = player_root(&g);
        let root_before = *world.get::<LocalTransform>(root).unwrap();
        let bones: Vec<Entity> = world
            .iter::<Name>()
            .filter(|(_, n)| n.0 == "sword" || n.0 == "arm_r_lower" || n.0 == "pelvis")
            .map(|(e, _)| e)
            .collect();
        assert_eq!(bones.len(), 3, "the warrior's bones are in the world");
        let bones_before: Vec<LocalTransform> = bones
            .iter()
            .map(|&e| *world.get::<LocalTransform>(e).unwrap())
            .collect();

        // A presentation pass alone: roots move, bones do not.
        g.render_update(&mut world, 0.25);
        assert_ne!(*world.get::<LocalTransform>(root).unwrap(), root_before);
        let bones_after: Vec<LocalTransform> = bones
            .iter()
            .map(|&e| *world.get::<LocalTransform>(e).unwrap())
            .collect();
        assert_eq!(bones_before, bones_after, "render_update touched a bone");

        // A sim step alone: bones move, and the root keeps whatever render_update left.
        let root_now = *world.get::<LocalTransform>(root).unwrap();
        g.simulate(&mut world, &held, FIXED_DT);
        assert_eq!(
            *world.get::<LocalTransform>(root).unwrap(),
            root_now,
            "fixed_update touched a root"
        );
        let bones_stepped: Vec<LocalTransform> = bones
            .iter()
            .map(|&e| *world.get::<LocalTransform>(e).unwrap())
            .collect();
        assert_ne!(bones_after, bones_stepped, "the run cycle did not advance");
    }

    /// The camera sits above and behind its focus, tilted off vertical (a perfectly
    /// vertical view axis is degenerate for the frame loop's look-at), and high enough
    /// to clear the walls it is looking over.
    #[test]
    fn camera_pose_is_tilted_and_above_the_walls() {
        let grid = dungeon(1);
        let spawn = collision::player_spawn(&grid);
        let mut g = game(grid);
        let pose = g.camera(1.0).expect("hook always drives the camera");
        assert!(pose.position.y > crate::meshing::WALL_HEIGHT);
        assert!(pose.position.z > spawn.z);
        let forward = pose.forward();
        assert!(forward.y < -0.5, "camera must look down");
        assert!(forward.y > -0.999, "but not straight down");
    }

    // -- integration properties ------------------------------------------------------

    /// Every seed must spawn the player in free space, on the entry tile — the state
    /// the whole simulation starts from.
    #[test]
    fn the_player_spawns_in_free_space_on_every_seed() {
        for seed in SEEDS {
            let grid = dungeon(seed);
            let mut g = game(grid);
            let mut world = test_world(&g);
            // One idle step is enough to acquire the cast and resolve the spawn.
            frame(&mut g, &mut world, &InputSnapshot::default(), 1);
            assert!(
                !collision::collision(g.grid()).circle_overlaps(g.pos, PLAYER_RADIUS),
                "seed {seed}: spawned inside geometry"
            );
            assert_eq!(g.player_tile(), g.grid().entry(), "seed {seed}");
            assert!(!g.exit_reached, "seed {seed}: spawn is not the exit");
        }
    }

    /// Every seed's exit must be *walkable to* from the spawn: the dungeon is a game
    /// level, not a picture of one.
    #[test]
    fn the_exit_is_reachable_from_the_spawn_on_every_seed() {
        for seed in SEEDS {
            let grid = dungeon(seed);
            let spawn_tile = collision::tile_of(collision::player_spawn_local(&grid));
            let dist = grid.bfs_distances(spawn_tile);
            let (ex, ez) = grid.exit();
            let steps = dist[(ez * grid.width() + ex) as usize];
            assert_ne!(steps, u32::MAX, "seed {seed}: exit unreachable from spawn");
            assert!(
                steps > 0,
                "seed {seed}: exit coincides with the spawn tile — nothing to walk"
            );
            assert!(grid.all_walkable_reachable(), "seed {seed}");
        }
    }

    /// The mover's contract, exercised through the game: hold a direction into a wall
    /// for a full second of fixed steps and the player never ends up inside solid rock,
    /// never leaves the dungeon, and never goes non-finite.
    ///
    /// Run from every seed's spawn in all four directions, so the wall found is whatever
    /// the generator actually put there (some directions are open corridors — those are
    /// the sliding case, and they are checked by the same invariants). Attack and dodge
    /// are held down throughout: a swing's forward step and a roll's burst are the two
    /// motions that do *not* go through ordinary locomotion, and they must respect the
    /// wall too.
    #[test]
    fn walking_into_walls_never_ends_inside_solid() {
        for seed in SEEDS {
            for key in [W, S, D, A] {
                let grid = dungeon(seed);
                let spawn = collision::player_spawn_local(&grid);
                let mut g = game(grid);
                let mut world = test_world(&g);
                let held = InputSnapshot::default()
                    .with_key(key, true)
                    .with_key(
                        SHIFT,
                        // Sprint on half the runs: the fastest step is the one most
                        // likely to tunnel, and it must not.
                        key == W || key == S,
                    )
                    .with_key(J, true)
                    .with_key(SPACE, true);
                for step in 0..60 {
                    g.simulate(&mut world, &held, FIXED_DT);
                    assert!(
                        g.pos.is_finite(),
                        "seed {seed} key {key:#x} step {step}: non-finite"
                    );
                    assert!(
                        !collision::collision(g.grid()).circle_overlaps(g.pos, PLAYER_RADIUS),
                        "seed {seed} key {key:#x} step {step}: inside solid at {:?}",
                        g.pos
                    );
                    let (tx, tz) = g.player_tile();
                    assert!(
                        g.grid().in_bounds(tx, tz) && g.grid().is_walkable(tx, tz),
                        "seed {seed} key {key:#x} step {step}: left the dungeon at ({tx}, {tz})"
                    );
                }
                // A second of walking must have gone *somewhere* unless the spawn is
                // walled in that direction — and even then, the player must not have
                // been pushed backwards through a wall.
                let sprint = g.warrior.class().sprint_speed();
                assert!(
                    g.pos.distance(spawn) < 60.0 * sprint,
                    "seed {seed} key {key:#x}: implausible travel"
                );
            }
        }
    }

    /// Walking onto the exit tile latches the flag the HUD reports — and nothing else
    /// does. Driven by placing the *authored* spawn on the exit's neighbour, so the
    /// detection is exercised through the same simulate() the game runs.
    #[test]
    fn touching_the_exit_latches_the_flag() {
        let grid = dungeon(3);
        let (ex, ez) = grid.exit();
        let mut g = walking_to_the_exit_of(grid);
        let mut world = exit_world(&g);
        frame(&mut g, &mut world, &InputSnapshot::default(), 1);
        assert!(!g.exit_reached, "not on the exit yet");
        assert_ne!(g.player_tile(), (ex, ez));
        assert!(matches!(g.progression, Progression::Playing));

        // Far enough to arrive, not so far that the grace runs out (that is the next
        // test's subject): the latch and the descent it starts are what is asserted here.
        frame(
            &mut g,
            &mut world,
            &InputSnapshot::default().with_key(W, true),
            25,
        );
        assert!(g.exit_reached, "walking onto the exit sets the flag");
        assert!(
            matches!(g.progression, Progression::Descending { .. }),
            "the exit starts the descent"
        );
    }

    /// A game on `grid` whose player will be placed one tile south of the exit, with the
    /// floor writer stubbed — the fixture the progression tests walk.
    fn walking_to_the_exit_of(grid: TileGrid) -> DungeonGame {
        game(grid)
    }

    /// A level-shaped world with the player authored one tile *south* of `g`'s exit, so
    /// holding W walks onto it.
    fn exit_world(g: &DungeonGame) -> World {
        let exit = g.grid().exit_world();
        let mut world = World::new();
        spawn_like_the_level_loader(
            &mut world,
            &g.warrior_rig,
            PLAYER_NAME,
            Vec3::new(exit.x, CHARACTER_Y, exit.z + TILE_SIZE),
        );
        world
    }

    // -- the run: floors, descent, death and restart -----------------------------------

    /// **Floor 1 is the run seed, unmixed** — the property every existing capture recipe,
    /// golden and bug report depends on — and every deeper floor is a different dungeon,
    /// reproducibly.
    #[test]
    fn floor_one_is_the_run_seed_and_deeper_floors_diverge() {
        for run in [0u64, 1, 3, 12345, crate::DEFAULT_SEED, u64::MAX] {
            assert_eq!(floor_seed(run, FIRST_FLOOR), run, "run {run}");
            // Floor 0 does not exist, but a derivation that panicked or wrapped into a
            // different dungeon for it would be a trap for the day one is asked for.
            assert_eq!(floor_seed(run, 0), run, "run {run}");

            let seeds: Vec<u64> = (1..=8).map(|f| floor_seed(run, f)).collect();
            let unique: std::collections::BTreeSet<u64> = seeds.iter().copied().collect();
            assert_eq!(unique.len(), seeds.len(), "run {run}: two floors collide");
            // Deterministic: the same run and floor is the same dungeon, always.
            assert_eq!(
                seeds,
                (1..=8).map(|f| floor_seed(run, f)).collect::<Vec<_>>()
            );
        }

        // Two runs differ on *every* floor — an additive step would otherwise let run
        // `s` floor 2 land on run `s+1` floor 1 and so on down the descent.
        for floor in 1..=8 {
            assert_ne!(floor_seed(7, floor), floor_seed(8, floor), "floor {floor}");
        }

        // And the grid really is the one the seed generates through the ordinary path:
        // `main` and the game must not be able to disagree about floor 1.
        let legacy = generate(crate::DEFAULT_SEED, &DungeonParams::default());
        let first = floor_grid(crate::DEFAULT_SEED, FIRST_FLOOR);
        assert_eq!(first.seed(), legacy.seed());
        assert_eq!(first.to_ascii(), legacy.to_ascii(), "floor 1 moved");
        assert_ne!(
            floor_grid(crate::DEFAULT_SEED, 2).to_ascii(),
            legacy.to_ascii(),
            "floor 2 is the same dungeon over again"
        );
    }

    /// The population rule: one more monster per floor, capped by what one A* workspace
    /// is sized for — and floor 1 still the six the game has always fielded.
    #[test]
    fn the_monster_count_scales_with_depth_and_stops_at_the_pathfinder_cap() {
        assert_eq!(grunts_for_floor(FIRST_FLOOR), DEFAULT_GRUNTS);
        assert_eq!(grunts_for_floor(2), DEFAULT_GRUNTS + 1);
        assert_eq!(grunts_for_floor(6), 11);
        assert_eq!(grunts_for_floor(7), ai::MAX_GRUNTS);
        assert_eq!(grunts_for_floor(500), ai::MAX_GRUNTS, "the cap holds");
        for floor in 1..40 {
            assert!(grunts_for_floor(floor) <= ai::MAX_GRUNTS);
            assert!(grunts_for_floor(floor + 1) >= grunts_for_floor(floor));
        }

        // The policy is what `--grunts` overrides: pinned means pinned, at any depth.
        assert_eq!(Population::PerFloor.count(3), grunts_for_floor(3));
        assert_eq!(Population::Fixed(2).count(3), 2);
        assert_eq!(Population::Fixed(0).count(9), 0);
        // A default game is the per-floor policy on floor 1 = the shipped floor.
        let g = DungeonGame::new(dungeon(3), Population::PerFloor).unwrap();
        assert_eq!(g.floor, FIRST_FLOOR);
        assert_eq!(g.run_seed, 3);
        assert_eq!(g.grunt_spawns().len(), DEFAULT_GRUNTS);
    }

    /// **The transition, step by step.** Walking onto the exit starts a grace; the grace
    /// installs the next floor; the hook offers that floor's level *once*; and the run
    /// only starts playing again when the rebuilt world's cast has been resolved.
    #[test]
    fn the_descent_is_a_graced_single_shot_handoff() {
        let mut g = walking_to_the_exit_of(dungeon(3));
        let mut world = exit_world(&g);
        let held = InputSnapshot::default().with_key(W, true);
        assert!(g.next_level().is_none(), "nothing to load while playing");

        // Walk on. The grace starts the frame the exit latches, and the next floor is
        // built there and then (the stub writer stands in for the meshing).
        let mut steps_in_grace = 0;
        for _ in 0..600 {
            g.simulate(&mut world, &held, FIXED_DT);
            match &g.progression {
                Progression::Descending { left, next } => {
                    steps_in_grace += 1;
                    assert!(*left <= DESCEND_GRACE && *left > -FIXED_DT);
                    assert_eq!(next.floor, 2);
                    assert_eq!(next.grid.seed(), floor_seed(3, 2));
                    assert!(g.next_level().is_none(), "not until the grace is out");
                    // Still playing the floor it is standing on.
                    assert_eq!(g.floor, FIRST_FLOOR);
                    assert_eq!(g.grid().seed(), 3);
                }
                Progression::Handoff { .. } => break,
                Progression::Playing => assert_eq!(steps_in_grace, 0),
                Progression::Awaiting => unreachable!("nothing has been handed over"),
            }
        }
        assert!(
            matches!(g.progression, Progression::Handoff { .. }),
            "the grace never ran out"
        );
        // The grace is a time, not a step count: ~0.6 s of fixed steps, ±the step the
        // exit was touched on.
        let expected = (DESCEND_GRACE / FIXED_DT).round() as i32;
        assert!(
            (steps_in_grace - expected).abs() <= 1,
            "graced for {steps_in_grace} steps, expected ~{expected}"
        );

        // The floor is already installed: new grid, new spawns, no cast, latch cleared.
        assert_eq!(g.floor, 2);
        assert_eq!(g.grid().seed(), floor_seed(3, 2));
        assert_eq!(g.grunt_spawns().len(), 0, "this fixture fields no monsters");
        assert!(g.cast.is_none(), "the old world's entities are gone");
        assert!(!g.exit_reached, "the new floor's exit has not been reached");
        assert_eq!(g.player_tile(), g.grid().entry(), "placed on the new entry");

        // ...and nothing is simulated until the world catches up.
        let steps = g.steps;
        g.simulate(&mut world, &held, FIXED_DT);
        assert_eq!(g.steps, steps, "simulated against a world it has left");

        // The handoff itself: once, with the new floor's own level path.
        assert_eq!(
            g.next_level().as_deref(),
            Some(crate::level::dungeon_level_selector(floor_seed(3, 2)).as_str())
        );
        assert!(matches!(g.progression, Progression::Awaiting));
        for _ in 0..3 {
            assert!(g.next_level().is_none(), "one descent, one load");
        }

        // The engine rebuilds the world; the next step resolves the cast out of it and
        // the run is playing again.
        world = test_world(&g);
        g.simulate(&mut world, &InputSnapshot::default(), FIXED_DT);
        assert!(matches!(g.progression, Progression::Playing));
        assert_eq!(g.steps, steps + 1);
    }

    /// Drive `g` through one descent the way the frame loop does: fixed steps until the
    /// hook asks for a level, the wholesale world rebuild that request causes, then the
    /// step that re-resolves the cast out of it. Returns the level that was requested.
    fn take_the_stairs(g: &mut DungeonGame, world: &mut World) -> String {
        let held = InputSnapshot::default().with_key(W, true);
        for _ in 0..900 {
            g.simulate(world, &held, FIXED_DT);
            if let Some(level) = g.next_level() {
                // What the engine does: a new `World`, built from the level the game
                // just wrote — every cached `Entity` now belongs to a world that is gone.
                *world = test_world(g);
                g.simulate(world, &InputSnapshot::default(), FIXED_DT);
                return level;
            }
        }
        panic!("no transition after fifteen seconds of walking at the exit");
    }

    /// **After the swap**: the cast re-resolves out of the *new* world — player on the
    /// new floor's entry, one brain per new spawn point, in its own body.
    #[test]
    fn the_cast_re_resolves_on_the_new_floor() {
        let mut g = DungeonGame::new(dungeon(3), Population::PerFloor).unwrap();
        g.writer = stub_writer;
        // Start next to the exit rather than at the entry: this is about the arrival.
        let mut world = exit_world(&g);
        let before = g.cast.as_ref().map(|c| c.player);
        assert!(before.is_none());

        take_the_stairs(&mut g, &mut world);

        assert_eq!(g.floor, 2);
        assert!(matches!(g.progression, Progression::Playing));
        let cast = g.cast.as_ref().expect("re-resolved out of the new world");
        assert!(
            world.is_alive(cast.player),
            "the player is a new-world entity"
        );
        assert_eq!(
            world.get::<Name>(cast.player).unwrap().0,
            PLAYER_NAME,
            "resolved by name, as on the first floor"
        );

        // Standing on the new floor's entry, in free space — through `nearest_free`, the
        // same snap the first floor gets.
        assert_eq!(g.player_tile(), g.grid().entry());
        assert!(!collision::collision(g.grid()).circle_overlaps(g.pos, PLAYER_RADIUS));

        // One monster per spawn point of the new floor, each brain in its own body, each
        // standing where the level put it.
        assert_eq!(g.grunt_spawns().len(), grunts_for_floor(2));
        assert_eq!(cast.grunts.len(), g.grunt_spawns.len());
        assert_eq!(cast.grunt_anims.len(), cast.grunts.len());
        assert_eq!(cast.grunt_views.len(), cast.grunts.len());
        for (i, grunt) in cast.grunts.iter().enumerate() {
            assert_eq!(grunt.position(), g.grunt_spawns[i], "grunt {i} misplaced");
            assert_eq!(
                world.get::<Name>(cast.grunt_anims[i].root).unwrap().0,
                grunt_name(i),
                "brain {i} is driving another monster's body"
            );
            assert!(world.is_alive(grunt.entity()));
        }

        // The floor's own geometry is what is being collided against now.
        assert_eq!(g.grid().seed(), floor_seed(3, 2));
        assert!(g.grid().all_walkable_reachable());
    }

    /// **Hit points carry down the stairs.** That is the run's pressure — a floor is not
    /// a checkpoint — and the one thing a descent deliberately does *not* reset.
    #[test]
    fn hit_points_carry_across_a_descent_and_a_restart_returns_them() {
        let mut g = walking_to_the_exit_of(dungeon(3));
        let mut world = exit_world(&g);
        frame(&mut g, &mut world, &InputSnapshot::default(), 1);

        let max = g.warrior.health().max;
        g.warrior.take_damage([IncomingHit {
            amount: 37.0,
            direction: Vec2::Y,
            stagger: 0.0,
        }]);
        let hurt = g.warrior.health().current;
        assert_eq!(hurt, max - 37.0);

        take_the_stairs(&mut g, &mut world);
        assert_eq!(g.floor, 2);
        assert_eq!(
            g.warrior.health().current,
            hurt,
            "the descent healed the warrior"
        );
        assert_eq!(g.warrior.health().max, max);

        // A restart is the other half of the rule: a new run gets a whole warrior.
        g.warrior.kill();
        assert!(g.warrior.is_dead());
        restart(&mut g, &mut world);
        assert_eq!(g.warrior.health().current, max, "restart kept the wounds");
        assert!(!g.warrior.is_dead());
    }

    /// Press R, then run the transition it asks for. Returns the requested level.
    fn restart(g: &mut DungeonGame, world: &mut World) -> String {
        let r = InputSnapshot::default().with_key(R, true);
        g.simulate(world, &r, FIXED_DT);
        let level = g.next_level().expect("R asks for the first floor");
        *world = test_world(g);
        g.simulate(world, &InputSnapshot::default(), FIXED_DT);
        level
    }

    /// **Death, then R**: the same run, from the top — floor 1, its own seed, a whole
    /// warrior — and R does nothing at all while the warrior is alive.
    #[test]
    fn a_dead_run_restarts_on_the_first_floor_of_the_same_seed() {
        let mut g = walking_to_the_exit_of(dungeon(3));
        let mut world = exit_world(&g);
        take_the_stairs(&mut g, &mut world);
        assert_eq!(g.floor, 2);

        // Alive, R is not a panic button: it must not throw the floor away mid-run.
        let r = InputSnapshot::default().with_key(R, true);
        g.simulate(&mut world, &r, FIXED_DT);
        assert!(!g.restart_requested());
        assert!(g.next_level().is_none(), "R restarted a living run");
        assert_eq!(g.floor, 2);

        // Dead, it is the only way out.
        g.warrior.kill();
        g.simulate(&mut world, &InputSnapshot::default(), FIXED_DT);
        let level = restart(&mut g, &mut world);

        assert_eq!(g.run_seed, 3, "a restart re-rolled the run");
        assert_eq!(g.floor, FIRST_FLOOR);
        assert_eq!(g.grid().seed(), 3, "floor 1 is the run seed");
        assert_eq!(level, crate::level::dungeon_level_selector(3));
        assert_eq!(g.warrior.health().current, g.warrior.health().max);
        assert!(!g.exit_reached);
        assert!(matches!(g.progression, Progression::Playing));
        assert_eq!(g.player_tile(), g.grid().entry());
        assert!(g.cast.is_some(), "the cast came back with the first floor");
    }

    /// The harness level (`--generated-room`) is a fixture, not a run: dying in it may
    /// not generate a dungeon nobody asked for.
    #[test]
    fn the_injection_harness_has_no_floors_to_progress_through() {
        let mut g = DungeonGame::new(crate::level::room_collision_grid(), Population::Fixed(0))
            .unwrap()
            .without_floors();
        g.writer = |_, _, _| panic!("the harness must never build a floor");
        let mut world = test_world(&g);
        frame(&mut g, &mut world, &InputSnapshot::default(), 1);

        g.warrior.kill();
        for _ in 0..4 {
            g.simulate(
                &mut world,
                &InputSnapshot::default().with_key(R, true),
                FIXED_DT,
            );
            g.simulate(&mut world, &InputSnapshot::default(), FIXED_DT);
        }
        assert!(g.next_level().is_none(), "the harness asked for a level");
        assert!(matches!(g.progression, Progression::Playing));

        // Its room has no exit to walk onto either: an ASCII fixture with no `X` leaves
        // the exit on the default (0, 0), which in every one of them is the solid corner
        // of the wall ring — so the descent cannot start even before the flag is
        // consulted, in this harness or in the hand-built test arenas.
        assert!(g.grid().is_solid(0, 0));
        assert_eq!(g.grid().exit(), (0, 0));
        assert_ne!(g.grid().exit(), g.grid().entry());
    }

    /// A level swap the game did **not** ask for — the engine's hot-swap dropdown — is
    /// not a continuation of the run: the warrior is rebuilt rather than carried.
    ///
    /// The cast is cleared by hand here for the reason `install` clears it: a rebuilt
    /// `World` restarts its generation counters, so a stale `Entity` can pass `is_alive`
    /// against a recycled slot in the new one. What is under test is the branch, not the
    /// detection.
    #[test]
    fn an_unrequested_level_swap_starts_a_fresh_warrior() {
        let mut g = walking_to_the_exit_of(dungeon(3));
        let mut world = test_world(&g);
        frame(&mut g, &mut world, &InputSnapshot::default(), 1);
        g.warrior.take_damage([IncomingHit {
            amount: 25.0,
            direction: Vec2::Y,
            stagger: 0.0,
        }]);
        assert!(g.warrior.health().current < g.warrior.health().max);

        g.cast = None;
        world = test_world(&g);
        g.simulate(&mut world, &InputSnapshot::default(), FIXED_DT);
        assert_eq!(
            g.warrior.health().current,
            g.warrior.health().max,
            "an unrequested level carried the run's wounds into it"
        );
        assert_eq!(g.floor, FIRST_FLOOR, "and it is not a floor of this run");
    }

    // -- the loot ----------------------------------------------------------------------

    // What the shipping `FloorWriter` was handed, per call, on this test's thread.
    //
    // A `FloorWriter` is a plain `fn` pointer (deliberately — see the type's docs), so a
    // recording stub cannot capture; thread-local state is what is left, and it is correct
    // rather than merely convenient here because the harness runs each test on its own
    // thread.
    thread_local! {
        static WRITTEN: std::cell::RefCell<Vec<(u64, usize, Vec<Vec2>)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// [`stub_writer`] that also books what it was asked to write.
    fn recording_writer(
        grid: &TileGrid,
        spawns: &[Vec2],
        potions: &[Vec2],
    ) -> anyhow::Result<String> {
        WRITTEN.with(|w| {
            w.borrow_mut()
                .push((grid.seed(), spawns.len(), potions.to_vec()))
        });
        stub_writer(grid, spawns, potions)
    }

    fn written() -> Vec<(u64, usize, Vec<Vec2>)> {
        WRITTEN.with(|w| w.borrow().clone())
    }

    /// **The potion list threads from the game into the level, on every floor.**
    ///
    /// The seam the whole pickup rests on: the game chooses the points, the writer places
    /// `potion_<i>` at point `i`, and [`ItemWorld`] collects point `i`. If the two ever
    /// diverge the player walks through a flask that is not there and stands next to one
    /// that cannot be picked up — which is invisible in a unit test of either half.
    #[test]
    fn potion_points_thread_into_every_floor_the_run_writes() {
        WRITTEN.with(|w| w.borrow_mut().clear());
        let mut g = DungeonGame::new(dungeon(3), Population::PerFloor).unwrap();
        g.writer = recording_writer;

        // Floor 1 is `main`'s to write, so what is asserted here is that the game's own
        // list is the floor's rule — that is what `main` hands the writer.
        let floor1 = g.potion_spawns().to_vec();
        assert_eq!(
            floor1,
            floor_potions(g.grid(), FIRST_FLOOR, g.grunt_spawns()),
            "floor 1's potions are not the floor's own"
        );
        assert_eq!(
            floor1.len(),
            items::potions_for_floor(FIRST_FLOOR) as usize,
            "seed 3 has room for the whole floor's flasks"
        );
        assert_eq!(g.items.remaining(), floor1.len());

        // Descend. The floor is built and written up front, potions and all.
        let mut world = exit_world(&g);
        take_the_stairs(&mut g, &mut world);
        assert_eq!(g.floor, 2);
        let calls = written();
        assert_eq!(calls.len(), 1, "one descent, one write");
        let (seed, grunts, floor2) = calls[0].clone();
        assert_eq!(seed, floor_seed(3, 2));
        assert_eq!(grunts, grunts_for_floor(2));
        assert_eq!(
            floor2,
            floor_potions(g.grid(), 2, g.grunt_spawns()),
            "the floor was written from a different list than the rule produces"
        );
        assert_eq!(
            g.potion_spawns(),
            floor2,
            "the runtime is collecting a different floor's flasks"
        );
        assert_ne!(floor1, floor2, "floor 2 reused floor 1's placement");
        // ...and every one of them really is in the world the engine rebuilt.
        for i in 0..floor2.len() {
            assert!(
                DungeonGame::find_named(&world, &items::potion_name(i)).is_some(),
                "the level has no potion_{i}"
            );
        }

        // A restart is the other half: floor 1 again, and its own list again.
        g.warrior.kill();
        g.simulate(&mut world, &InputSnapshot::default(), FIXED_DT);
        restart(&mut g, &mut world);
        let calls = written();
        assert_eq!(calls.len(), 2, "the restart wrote one floor");
        assert_eq!(calls[1].0, 3, "a restart re-rolled the run");
        assert_eq!(calls[1].2, floor1, "floor 1 came back a different floor");
        assert_eq!(g.potion_spawns(), floor1);
        assert_eq!(g.items.remaining(), floor1.len(), "the flasks came back");
    }

    /// An arena with one flask a stride north of the spawn — near enough that holding W
    /// walks onto it, far enough that the first step does not.
    fn with_one_flask() -> (DungeonGame, World, Vec2) {
        let mut g = game(TileGrid::from_rows(&[
            "##########",
            "#........#",
            "#........#",
            "#........#",
            "#...E....#",
            "#........#",
            "#........#",
            "#........#",
            "#........#",
            "##########",
        ]));
        let spawn = collision::player_spawn_local(g.grid());
        // Collision space is world XZ, and W walks toward -y (see `warrior_input`).
        let at = spawn + Vec2::new(0.0, -1.5);
        g.items = ItemWorld::new(&[at]);
        let world = test_world(&g);
        (g, world, at)
    }

    /// **Walking over a flask pockets it and takes it out of the draw list.**
    ///
    /// The integrator's half of the pickup: `items` decides *that* it was collected, this
    /// is what makes the flask stop being drawn. Hidden, not despawned — the entity is the
    /// level's and its sub-tree is the loader's.
    #[test]
    fn walking_over_a_flask_pockets_it_and_hides_the_level_entity() {
        let (mut g, mut world, at) = with_one_flask();
        let held = InputSnapshot::default().with_key(W, true);
        frame(&mut g, &mut world, &InputSnapshot::default(), 1);

        let parts = drawn_parts(&world, "potion_0");
        assert!(parts > 0, "the flask was never drawn to begin with");
        assert_eq!(g.inventory.potions, 0);
        assert!(!g.items.is_taken(0));

        for _ in 0..90 {
            g.simulate(&mut world, &held, FIXED_DT);
            if g.inventory.potions > 0 {
                break;
            }
        }
        assert_eq!(g.inventory.potions, 1, "walked past the flask");
        assert!(g.items.is_taken(0));
        assert_eq!(g.items.remaining(), 0);
        assert!(
            g.pos.distance(at) <= items::PICKUP_RADIUS,
            "collected from {} m away",
            g.pos.distance(at)
        );

        // The entity survives; only its geometry is gone, all of it.
        let root = DungeonGame::find_named(&world, "potion_0").expect("still in the level");
        assert!(world.is_alive(root), "the flask was despawned, not hidden");
        assert_eq!(drawn_parts(&world, "potion_0"), 0, "the flask still draws");
        // ...and nothing else lost its geometry with it.
        assert!(drawn_parts(&world, PLAYER_NAME) > 0, "hid the player too");

        // Standing on it forever does not pocket a second one.
        frame(&mut g, &mut world, &held, 30);
        assert_eq!(g.inventory.potions, 1);
    }

    /// **Q drinks, and the hit points land on the real warrior.**
    ///
    /// End to end through the shipping controller — the heal amount comes out of the same
    /// [`items::PotionDef`] the pickup was judged against, so this also pins that the two
    /// cannot drift. A drink is an *edge*, an empty pocket is a no-op, and a corpse may not
    /// spend a potion it cannot benefit from.
    #[test]
    fn q_drinks_a_carried_potion_and_heals_the_warrior() {
        let (mut g, mut world, _) = with_one_flask();
        let held = InputSnapshot::default().with_key(W, true);
        let drink = InputSnapshot::default().with_key(Q, true);
        let idle = InputSnapshot::default();
        for _ in 0..90 {
            g.simulate(&mut world, &held, FIXED_DT);
            if g.inventory.potions > 0 {
                break;
            }
        }
        assert_eq!(g.inventory.potions, 1);

        // Hurt it by more than one potion is worth, so the heal is not clipped by the cap.
        let max = g.warrior.health().max;
        g.warrior.take_damage([IncomingHit {
            amount: items::POTION_HEAL + 15.0,
            direction: Vec2::Y,
            stagger: 0.0,
        }]);
        let hurt = g.warrior.health().current;

        g.simulate(&mut world, &drink, FIXED_DT);
        assert_eq!(g.inventory.potions, 0, "the potion was not spent");
        assert_eq!(
            g.warrior.health().current,
            hurt + items::POTION_HEAL,
            "the warrior did not get what the flask reported"
        );
        assert!(g.warrior.health().current < max);

        // Holding Q on an empty pocket does nothing, however long it is held.
        let healed = g.warrior.health().current;
        for _ in 0..10 {
            g.simulate(&mut world, &drink, FIXED_DT);
        }
        assert_eq!(g.warrior.health().current, healed);
        assert_eq!(g.inventory.potions, 0);

        // A corpse does not spend one. (Give it a potion by hand: the point is the branch,
        // not how the pocket got filled.)
        g.inventory.potions = 1;
        g.warrior.kill();
        g.simulate(&mut world, &idle, FIXED_DT);
        g.simulate(&mut world, &drink, FIXED_DT);
        assert_eq!(g.inventory.potions, 1, "a corpse drank the last potion");
        assert!(g.warrior.is_dead());
    }

    /// **The pocket is the run's, the flasks are the floor's.** Carrying a potion down the
    /// stairs is the whole point of a cap of three; a restart hands back a warrior who has
    /// never picked one up.
    #[test]
    fn the_pocket_survives_a_descent_and_is_emptied_by_a_restart() {
        let mut g = walking_to_the_exit_of(dungeon(3));
        let mut world = exit_world(&g);
        frame(&mut g, &mut world, &InputSnapshot::default(), 1);
        g.inventory.potions = 2;

        take_the_stairs(&mut g, &mut world);
        assert_eq!(g.floor, 2);
        assert_eq!(g.inventory.potions, 2, "the stairs emptied the pocket");
        assert_eq!(
            g.items.remaining(),
            g.potion_spawns().len(),
            "the new floor's flasks arrived taken"
        );

        g.warrior.kill();
        restart(&mut g, &mut world);
        assert_eq!(g.inventory.potions, 0, "a new run kept the old run's loot");
        assert_eq!(g.warrior.health().current, g.warrior.health().max);
    }

    // -- the fight ---------------------------------------------------------------------

    /// A duel arena: the warrior and one grunt placed a stride apart in open floor, with
    /// the real level shape (named roots, real rigs, real bindings) around them.
    ///
    /// Built rather than found: `spawn_points` deliberately puts monsters rooms away
    /// from the entry, so a fight test that walked to one would be a test of the *level
    /// generator's* geometry. Here the geometry is a fixed grid and the only variables
    /// are the ones under test.
    fn duel() -> (DungeonGame, World, Vec2) {
        // 8x8 tiles of open floor inside a wall ring; entry near one corner.
        let grid = TileGrid::from_rows(&[
            "##########",
            "#........#",
            "#........#",
            "#........#",
            "#...E....#",
            "#........#",
            "#........#",
            "#........#",
            "#........#",
            "##########",
        ]);
        let mut g = game(grid);
        let spawn = collision::player_spawn_local(g.grid());
        // 1.5 m ahead along +Z (the rigs' forward, and the warrior's rest facing): inside
        // the opener's 1.9 m reach and inside the grunt's 1.6 m commit range.
        let enemy = spawn + Vec2::new(0.0, 1.5);
        g.grunt_spawns = vec![enemy];

        let mut world = World::new();
        spawn_like_the_level_loader(
            &mut world,
            &g.warrior_rig,
            PLAYER_NAME,
            collision::to_world(g.grid(), spawn, CHARACTER_Y),
        );
        spawn_like_the_level_loader(
            &mut world,
            &g.grunt_rig,
            &grunt_name(0),
            collision::to_world(g.grid(), enemy, CHARACTER_Y),
        );
        (g, world, enemy)
    }

    /// **The M2 loop, end to end.** 600 fixed steps of a real fight in a real ECS: the
    /// player mashes attack on an edge cadence, a grunt closes and swings back, and the
    /// arithmetic on both sides is the one the data files say it should be.
    ///
    /// What this pins that no unit test can: that the *wiring order* is right. The
    /// grunt's hit points live in the ECS and the player's live in the controller, so a
    /// mis-routed event shows up here as damage landing twice, or not at all.
    #[test]
    fn a_600_tick_fight_kills_a_grunt_and_the_player_survives() {
        let (mut g, mut world, _) = duel();

        // Acquire, then confirm the ECS half of a monster is in place.
        frame(&mut g, &mut world, &InputSnapshot::default(), 1);
        let cast_grunt = g.cast.as_ref().unwrap().grunts[0].entity();
        assert_eq!(
            world.get::<Health>(cast_grunt).unwrap().current,
            GRUNT_MAX_HEALTH
        );
        assert_eq!(world.get::<Team>(cast_grunt).copied(), Some(Team::ENEMY));
        assert_eq!(
            world.get::<BodyCircle>(cast_grunt).map(|b| b.radius),
            Some(GRUNT_RADIUS)
        );

        // What the chain is worth, before anyone swings it: the aligned combo re-times
        // the class's three swings but must not re-price them.
        let chain: Vec<f32> = g.warrior.chain().iter().map(|s| s.damage).collect();
        assert_eq!(chain, vec![12.0, 14.0, 22.0]);
        assert!(
            chain.iter().sum::<f32>() > GRUNT_MAX_HEALTH,
            "the combo cannot kill a grunt at all"
        );

        // Fight it. The cadence is the combo's own, not a fixed period:
        // `AttackState::request` *refuses* an input during windup and buffers only one
        // during the active window, so "press every N steps" fires an opener and throws
        // its links away. Press from idle, then once per *newly opened* active window,
        // releasing in between — an attack is an edge.
        //
        // The chain does not always run to three: a grunt's claw staggers the warrior,
        // and a stagger cancels the swing it interrupts (that is the fight working, and
        // it is why this is an integration test rather than arithmetic). So the loop
        // mashes until the monster is down and counts what actually *landed*.
        let idle = InputSnapshot::default();
        let swing = InputSnapshot::default().with_key(J, true);
        let mut holding = false;
        // Each landed hit as `(hit points removed, hit points the monster had first)`.
        let mut landed: Vec<(f32, f32)> = Vec::new();
        let mut grunt_hp = GRUNT_MAX_HEALTH;
        let mut died_at = None;
        for step in 0..600u32 {
            let ready = match g.warrior.attack_phase() {
                AttackPhase::Idle => true,
                AttackPhase::Active => g.warrior.combo_step() + 1 == landed.len(),
                _ => false,
            };
            let press = ready && !holding && died_at.is_none();
            holding = press;
            g.simulate(&mut world, if press { &swing } else { &idle }, FIXED_DT);

            let now = world.get::<Health>(cast_grunt).unwrap().current;
            if now < grunt_hp {
                landed.push((grunt_hp - now, grunt_hp));
                grunt_hp = now;
            }
            if died_at.is_none() && g.cast.as_ref().unwrap().grunts[0].is_dead() {
                died_at = Some(step);
            }
        }

        let died_at = died_at.expect("the grunt survived ten seconds of combo");
        // Three landed swings, each one a step of the chain — no hit landed twice (the
        // arc is once per target per swing) and none was invented.
        assert_eq!(landed.len(), 3, "landed {landed:?}");
        for &(dealt, before) in &landed {
            // Either a whole chain step, or the killing blow clipped by the hit points
            // that were left — `Health::damage` floors at zero and reports what it took.
            assert!(
                chain.contains(&dealt) || dealt == before,
                "landed {dealt} off {before}, which is neither a chain step nor a kill"
            );
        }
        assert_eq!(
            landed.iter().map(|&(d, _)| d).sum::<f32>(),
            GRUNT_MAX_HEALTH,
            "the three swings did not account for exactly the monster's hit points"
        );
        // The killing blow overkilled rather than left a sliver: health floors at zero.
        let grunt = &g.cast.as_ref().unwrap().grunts[0];
        assert!(grunt.is_dead() && grunt.state() == GruntState::Dead);
        assert_eq!(
            world.get::<Health>(cast_grunt).unwrap().current,
            0.0,
            "the ECS is the monster's health authority"
        );
        // The opener alone cannot kill (12 < 30), so the death cannot land on the first
        // swing's hit frame — which is the tripwire for damage applied twice.
        assert!(
            died_at >= 17,
            "died at step {died_at}, before the opener connected"
        );

        // The player survived, and lost only what a grunt's claw is worth (8 a hit).
        let health = g.warrior.health();
        assert!(!g.warrior.is_dead(), "the player died to one grunt");
        let lost = health.max - health.current;
        assert_eq!(
            lost % 8.0,
            0.0,
            "the player lost {lost}, not a whole number of 8-point claws"
        );
        assert!(lost <= 8.0 * 3.0, "one grunt landed {} hits", lost / 8.0);
        // ...and every point of it came off exactly once: the same events cannot have
        // been applied to a mirror `Health` as well, because there is no mirror.
        assert!(
            world
                .get::<Health>(g.cast.as_ref().unwrap().player)
                .is_none()
        );
        assert_eq!(g.grunt_tally(), (0, 1));
    }

    /// A corpse leaves the fight: it stops being offered as a swing target, drops the
    /// components anything else would find it by, and holds its death pose while the
    /// world keeps ticking.
    #[test]
    fn a_dead_grunt_stops_colliding_and_keeps_its_death_pose() {
        let (mut g, mut world, _) = duel();
        frame(&mut g, &mut world, &InputSnapshot::default(), 1);
        let entity = g.cast.as_ref().unwrap().grunts[0].entity();

        // Kill it outright through the same channel combat uses.
        g.damage.send(DamageEvent::new(
            g.cast.as_ref().unwrap().player,
            entity,
            GRUNT_MAX_HEALTH * 2.0,
            Vec2::Y,
        ));
        g.simulate(&mut world, &InputSnapshot::default(), FIXED_DT);
        assert!(g.cast.as_ref().unwrap().grunts[0].is_dead());

        // Retired from the world: no team, no body.
        assert!(world.get::<Team>(entity).is_none(), "a corpse still fights");
        assert!(
            world.get::<BodyCircle>(entity).is_none(),
            "a corpse still blocks"
        );
        assert!(g.targets(g.cast.as_ref().unwrap()).is_empty());

        // Let the death clip run past its end, then well past it: the pose is reached
        // and then held (the clip clamps, and its last two keys are equal).
        let bones: Vec<Entity> = {
            let index = crate::characters::ChildIndex::build(&world);
            let binding = crate::characters::RigBinding::resolve(
                &world,
                &index,
                g.cast.as_ref().unwrap().grunt_anims[0].root,
                &g.grunt_rig,
            );
            (0..g.grunt_rig.nodes.len())
                .filter_map(|i| binding.entity(i))
                .collect()
        };
        assert!(!bones.is_empty());
        let snapshot = |world: &World| -> Vec<LocalTransform> {
            bones
                .iter()
                .map(|&e| *world.get::<LocalTransform>(e).unwrap())
                .collect()
        };
        frame(&mut g, &mut world, &InputSnapshot::default(), 90); // 1.5 s > the 1.0 s clip
        let corpse = snapshot(&world);
        frame(&mut g, &mut world, &InputSnapshot::default(), 300); // five more seconds
        assert_eq!(snapshot(&world), corpse, "the corpse got back up");

        // And the corpse is still drawn where it fell, not at the origin.
        propagate_transforms(&mut world);
        let root = g.cast.as_ref().unwrap().grunt_anims[0].root;
        let drawn = world.get::<LocalTransform>(root).unwrap().translation;
        assert_eq!(drawn.y, CHARACTER_Y);
        assert!(drawn.is_finite());
    }

    /// A whole floor's worth of monsters runs, and the fixed step stays inside its
    /// budget. Not a benchmark — a tripwire for an accidental O(n^2) that a six-monster
    /// floor would otherwise hide until a twelve-monster one.
    #[test]
    fn a_full_floor_simulates_inside_the_step_budget() {
        let mut g = DungeonGame::new(dungeon(3), Population::Fixed(DEFAULT_GRUNTS)).unwrap();
        assert_eq!(
            g.grunt_spawns().len(),
            DEFAULT_GRUNTS,
            "the floor filled up"
        );
        let mut world = test_world(&g);
        let held = InputSnapshot::default().with_key(W, true);
        frame(&mut g, &mut world, &held, 300);

        let cast = g.cast.as_ref().unwrap();
        assert_eq!(cast.grunts.len(), DEFAULT_GRUNTS);
        assert_eq!(cast.grunt_anims.len(), DEFAULT_GRUNTS);
        assert_eq!(cast.grunt_views.len(), DEFAULT_GRUNTS);
        // Each brain kept its own body.
        for (i, grunt) in cast.grunts.iter().enumerate() {
            assert_eq!(
                world.get::<Name>(cast.grunt_anims[i].root).unwrap().0,
                grunt_name(i)
            );
            assert!(world.is_alive(grunt.entity()));
        }
        // The peak is measured on a debug build here, so the assertion is generous: it
        // exists to catch a step that has become milliseconds *slower*, not to certify
        // the release number (the report measures that on the real binary).
        assert!(
            g.sim_ms_peak < 50.0,
            "a fixed step peaked at {:.2} ms with {DEFAULT_GRUNTS} grunts",
            g.sim_ms_peak
        );
    }

    /// The simulation is deterministic: the same inputs give the same fight, which is
    /// what makes a headless capture reproducible and a bug report a repro.
    #[test]
    fn the_fight_is_deterministic() {
        let run = || {
            let (mut g, mut world, _) = duel();
            let swing = InputSnapshot::default().with_key(J, true);
            let idle = InputSnapshot::default();
            for step in 0..240u32 {
                let input = if step % 15 == 0 { &swing } else { &idle };
                g.simulate(&mut world, input, FIXED_DT);
            }
            let cast = g.cast.as_ref().unwrap();
            (
                g.pos.to_array().map(f32::to_bits),
                g.warrior.health().current.to_bits(),
                cast.grunts[0].position().to_array().map(f32::to_bits),
                cast.grunts[0].is_dead(),
            )
        };
        assert_eq!(run(), run());
    }
}

#[cfg(test)]
mod capture_scout {
    //! Choreography scout for headless captures.
    //!
    //! A capture of a *fight* has to name the fixed step it wants, and the interesting
    //! ones (the frame the blade crosses the arc, the frame a monster falls) are not
    //! guessable: they depend on the generator, the pathfinder and the combo's own
    //! accept/refuse rules. This replays a run with the same deterministic input the
    //! capture will use and prints what happened, so a `DUNGEON_HOLD`/`DUNGEON_TAP`
    //! recipe is read off a simulation rather than found by bisecting screenshots.
    //!
    //! It is not an assertion — nothing here can fail — so it is inert unless asked for:
    //!
    //! ```text
    //! ROOMS=0,1,2  GRUNTS=12  cargo test -p dungeon rooms -- --nocapture
    //!     # which rooms hold a crowd, and where they are relative to the spawn
    //! CHOREO=14,S GRUNTS=12  cargo test -p dungeon choreograph -- --nocapture
    //!     # walk south on seed 14 swinging whenever the combo allows; prints TAPS,
    //!     # the step the player is first hurt, and the step the first corpse drops
    //! ```
    //!
    //! The step numbers it prints are the capture's, exactly: the simulation is
    //! deterministic and the level's extra entities do not reach it (nothing in the game
    //! branches on an entity id).
    use super::*;
    use crate::procgen::{DungeonParams, generate};
    use dreamcoast_game::combat::AttackPhase;

    /// Where a floor's monsters actually stand, grouped by room — the map a capture
    /// recipe is planned from. `ROOMS=<seed>`.
    #[test]
    fn rooms() {
        let Ok(spec) = std::env::var("ROOMS") else {
            return;
        };
        let grunts: usize = std::env::var("GRUNTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_GRUNTS);
        for seed in spec.split(',').filter_map(|s| s.parse::<u64>().ok()) {
            let grid = generate(seed, &DungeonParams::default());
            let spawn = collision::player_spawn_local(&grid);
            let pts = crate::ai::spawn_points(&grid, grunts, 12.0, grid.seed());
            let mut by_room: std::collections::BTreeMap<u16, Vec<Vec2>> = Default::default();
            for p in &pts {
                let (tx, tz) = collision::tile_of(*p);
                by_room.entry(grid.room_id_at(tx, tz)).or_default().push(*p);
            }
            for (room, ps) in &by_room {
                if ps.len() < 2 {
                    continue;
                }
                let centre = ps.iter().fold(Vec2::ZERO, |a, b| a + *b) / ps.len() as f32;
                println!(
                    "seed {seed} room {room}: {} grunts, centre {:?}, delta from spawn {:?}",
                    ps.len(),
                    centre,
                    centre - spawn
                );
            }
        }
    }

    /// A seed whose **stairs are walkable blind** — the recipe an M3 capture needs.
    ///
    /// A descent capture has to reach the exit with `DUNGEON_HOLD`, which is a held
    /// direction and not a route, so the useful seeds are the ones where some held
    /// combination happens to arrive. This sweeps seeds × combinations against the real
    /// simulation and prints the ones that get there alive, with the step the exit
    /// latches and the step the next floor is asked for — which is exactly what
    /// `WARMUP_FRAMES` then needs (the capture path runs one sim step per frame).
    ///
    /// ```text
    /// STAIRS=0-64 cargo test -p dungeon stairs -- --nocapture
    /// ```
    #[test]
    fn stairs() {
        let Ok(spec) = std::env::var("STAIRS") else {
            return;
        };
        let (lo, hi) = spec
            .split_once('-')
            .unwrap_or((spec.as_str(), spec.as_str()));
        let (lo, hi): (u64, u64) = (lo.parse().unwrap(), hi.parse().unwrap());
        const SHIFT: u16 = 0x10;
        let combos: [(&str, &[u16]); 8] = [
            ("W", &[0x57]),
            ("S", &[0x53]),
            ("A", &[0x41]),
            ("D", &[0x44]),
            ("W,A", &[0x57, 0x41]),
            ("W,D", &[0x57, 0x44]),
            ("S,A", &[0x53, 0x41]),
            ("S,D", &[0x53, 0x44]),
        ];
        for seed in lo..=hi {
            for (name, keys) in combos {
                let mut g =
                    DungeonGame::new(floor_grid(seed, FIRST_FLOOR), Population::PerFloor).unwrap();
                g.writer = super::tests::stub_writer;
                let mut world = super::tests::test_world(&g);
                let mut snap = InputSnapshot::default();
                for vk in keys.iter().chain(&[SHIFT]) {
                    snap.set_key(*vk, true);
                }
                let mut latched = None;
                for step in 0..1200u32 {
                    g.simulate(&mut world, &snap, 1.0 / 60.0);
                    if latched.is_none() && g.exit_reached {
                        latched = Some(step);
                    }
                    if let Some(level) = g.next_level() {
                        println!(
                            "seed {seed} hold {name},Shift: exit at step {}, floor 2 requested \
                             at step {step} ('{level}'), {:.0} hp left",
                            latched.unwrap_or(step),
                            g.warrior.health().current,
                        );
                        break;
                    }
                    if g.warrior.is_dead() {
                        break;
                    }
                }
            }
        }
    }

    /// The same recipe for a seed whose stairs are *not* reachable blind — which is most
    /// of them, dungeons being corridors.
    ///
    /// Walks the grid's own BFS route from the entry to the exit, steering with the four
    /// movement keys, and prints the [`HOLD_ENV`] spec that reproduces the walk: a
    /// windowed hold per leg. The route is the map's, the steering is the game's, and the
    /// output is a string a headless capture takes verbatim.
    ///
    /// ```text
    /// STAIRS_ROUTE=20260731 cargo test -p dungeon stairs_route -- --nocapture
    /// ```
    #[test]
    fn stairs_route() {
        let Ok(spec) = std::env::var("STAIRS_ROUTE") else {
            return;
        };
        let seed: u64 = spec.trim().parse().expect("STAIRS_ROUTE=<seed>");
        let (mut g, mut world) = scout_game(seed);
        let route = bfs_route(g.grid(), g.grid().exit());
        let legs = route.len();
        match steer(&mut g, &mut world, &route, 2400, |g| {
            g.next_level().is_some()
        }) {
            Some((hold, step)) => {
                println!("seed {seed}: floor 2 requested at step {step}");
                println!("DUNGEON_HOLD={hold}");
                println!(
                    "exit latched around step {}, {:.0} hp left, {legs} legs",
                    step.saturating_sub((DESCEND_GRACE / (1.0 / 60.0)) as u32),
                    g.warrior.health().current,
                );
            }
            None => println!("seed {seed}: never reached the stairs (or died on the way)"),
        }
    }

    /// The same tool aimed at a **flask**: the recipe an M3 pickup capture needs.
    ///
    /// Walks to the nearest potion of floor `n` and prints the step the pocket fills on,
    /// which is what `WARMUP_FRAMES` wants — the capture path runs one sim step per frame,
    /// so frame `f` photographs the state after step `f`, and the HUD's potion count goes
    /// up on exactly the step printed here.
    ///
    /// ```text
    /// POTION_ROUTE=20260731 cargo test -p dungeon potion_route -- --nocapture
    /// POTION_ROUTE=20260731 FLOOR=3 cargo test -p dungeon potion_route -- --nocapture
    /// ```
    #[test]
    fn potion_route() {
        let Ok(spec) = std::env::var("POTION_ROUTE") else {
            return;
        };
        let seed: u64 = spec.trim().parse().expect("POTION_ROUTE=<seed>");
        let floor: u32 = std::env::var("FLOOR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(FIRST_FLOOR);
        let (mut g, mut world) = scout_game(floor_seed(seed, floor));

        // The nearest flask by *walking* distance, which is the one a route reaches first.
        let grid = g.grid().clone();
        let from = grid.bfs_distances(collision::tile_of(collision::player_spawn_local(&grid)));
        let steps_to = |t: (i32, i32)| from[(t.1 * grid.width() + t.0) as usize];
        let Some((i, target)) = g
            .potion_spawns()
            .iter()
            .map(|&p| collision::tile_of(p))
            .enumerate()
            .filter(|&(_, t)| steps_to(t) != u32::MAX)
            .min_by_key(|&(_, t)| steps_to(t))
        else {
            println!("seed {seed} floor {floor}: no reachable potion");
            return;
        };
        println!(
            "seed {seed} floor {floor}: {} potions, nearest is potion_{i} at tile {target:?} \
             ({} tiles away)",
            g.potion_spawns().len(),
            steps_to(target),
        );

        let route = bfs_route(&grid, target);
        match steer(&mut g, &mut world, &route, 2400, |g| {
            g.inventory.potions > 0
        }) {
            Some((hold, step)) => {
                println!("seed {seed}: potion_{i} pocketed at step {step}");
                println!("DUNGEON_HOLD={hold}");
                println!(
                    "carrying {}/{}, {:.0} hp left",
                    g.inventory.potions,
                    g.items.def().max_carry,
                    g.warrior.health().current,
                );
            }
            None => println!("seed {seed}: never reached potion_{i} (or died on the way)"),
        }
    }

    /// A game on one floor's grid with the writer stubbed, and the level-shaped world that
    /// goes with it — what every route scout starts from.
    fn scout_game(floor_seed: u64) -> (DungeonGame, World) {
        let mut g = DungeonGame::new(
            crate::procgen::generate(floor_seed, &DungeonParams::default()),
            Population::PerFloor,
        )
        .unwrap();
        g.writer = super::tests::stub_writer;
        let world = super::tests::test_world(&g);
        (g, world)
    }

    /// The grid's own route from the player's spawn to `target`: downhill on `target`'s BFS
    /// field, as collision-space waypoints.
    fn bfs_route(grid: &TileGrid, target: (i32, i32)) -> Vec<Vec2> {
        let field = grid.bfs_distances(target);
        let at = |(x, z): (i32, i32)| field[(z * grid.width() + x) as usize];
        let mut tile = collision::tile_of(collision::player_spawn_local(grid));
        let mut route = Vec::new();
        while at(tile) > 0 && route.len() < 4096 {
            tile = grid
                .neighbors4(tile.0, tile.1)
                .filter(|&(x, z)| grid.is_walkable(x, z))
                .min_by_key(|&t| at(t))
                .expect("a walkable neighbour on a connected floor");
            route.push(collision::to_collision(
                grid,
                grid.tile_center(tile.0, tile.1),
            ));
        }
        route
    }

    /// Steer `g` along `route` with the four movement keys until `done` fires, and report
    /// the [`HOLD_ENV`] spec that reproduces the walk together with the step it fired on.
    ///
    /// The route is the map's and the steering is the game's, so what comes out is a string
    /// a headless capture takes verbatim: one windowed hold per leg. `None` means the walk
    /// ran out of budget or the warrior died on the way.
    fn steer(
        g: &mut DungeonGame,
        world: &mut World,
        route: &[Vec2],
        budget: u32,
        mut done: impl FnMut(&mut DungeonGame) -> bool,
    ) -> Option<(String, u32)> {
        const W: u16 = 0x57;
        const S: u16 = 0x53;
        const A: u16 = 0x41;
        const D: u16 = 0x44;
        let mut leg = 0usize;
        // (key name, first step, last step) as the walk produces them.
        let mut held: Vec<(&'static str, u32, u32)> = Vec::new();
        let push =
            |name: &'static str, step: u32, held: &mut Vec<(&'static str, u32, u32)>| match held
                .last_mut()
            {
                Some(last) if last.0 == name && last.2 + 1 == step => last.2 = step,
                _ => held.push((name, step, step)),
            };
        for step in 0..budget {
            while leg < route.len() && route[leg].distance(g.pos) < 0.6 {
                leg += 1;
            }
            let mut snap = InputSnapshot::default();
            if let Some(&target) = route.get(leg) {
                let d = target - g.pos;
                // Collision space is world XZ, and W walks toward -y (see
                // `warrior_input`), so the mapping is the input layer's own, backwards.
                if d.x > 0.4 {
                    snap.set_key(D, true);
                    push("D", step, &mut held);
                } else if d.x < -0.4 {
                    snap.set_key(A, true);
                    push("A", step, &mut held);
                }
                if d.y < -0.4 {
                    snap.set_key(W, true);
                    push("W", step, &mut held);
                } else if d.y > 0.4 {
                    snap.set_key(S, true);
                    push("S", step, &mut held);
                }
            }
            g.simulate(world, &snap, 1.0 / 60.0);
            if done(g) {
                let spec: Vec<String> = held
                    .iter()
                    .map(|(k, a, b)| format!("{k}@{a}-{b}"))
                    .collect();
                return Some((spec.join(","), step));
            }
            if g.warrior.is_dead() {
                println!("died at step {step} on the way");
                return None;
            }
        }
        None
    }

    #[test]
    fn choreograph() {
        let Ok(spec) = std::env::var("CHOREO") else {
            return;
        };
        let mut it = spec.split(',');
        let seed: u64 = it.next().unwrap().parse().unwrap();
        let hold: Vec<u16> = it
            .map(|k| dreamcoast_game::input::keys::key_vk(k).unwrap())
            .collect();
        let grunts: usize = std::env::var("GRUNTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_GRUNTS);
        let mut g = DungeonGame::new(
            generate(seed, &DungeonParams::default()),
            Population::Fixed(grunts),
        )
        .unwrap();
        let mut world = super::tests::test_world(&g);
        let until: u32 = std::env::var("HOLD_UNTIL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(u32::MAX);

        let mut taps: Vec<u32> = Vec::new();
        let mut holding = false;
        let mut swings = 0usize;
        let mut first_hurt = None;
        let mut first_corpse = None;
        for step in 0..900u32 {
            let cast = g.cast.as_ref();
            let near = cast
                .map(|c| {
                    c.grunts
                        .iter()
                        .filter(|x| !x.is_dead())
                        .map(|x| x.position().distance(g.pos))
                        .fold(f32::MAX, f32::min)
                })
                .unwrap_or(f32::MAX);
            // Swing when a live monster is inside the opener's reach, on the combo's
            // own cadence (opener from idle, one link per newly opened active window).
            let ready = match g.warrior.attack_phase() {
                AttackPhase::Idle => near < 1.9,
                AttackPhase::Active => g.warrior.combo_step() + 1 == swings,
                _ => false,
            };
            let press = ready && !holding;
            if press {
                swings += 1;
                taps.push(step);
            }
            holding = press;

            let mut snap = InputSnapshot::default();
            if step <= until {
                for vk in &hold {
                    snap.set_key(*vk, true);
                }
            }
            if press {
                snap.set_mouse_button(0, true);
            }
            g.simulate(&mut world, &snap, 1.0 / 60.0);

            let cast = g.cast.as_ref().unwrap();
            let engaged = cast
                .grunts
                .iter()
                .filter(|x| matches!(x.state(), GruntState::Chase | GruntState::Attack))
                .count();
            let dead = cast.grunts.iter().filter(|x| x.is_dead()).count();
            if first_hurt.is_none() && g.warrior.health().current < 100.0 {
                first_hurt = Some(step);
            }
            if first_corpse.is_none() && dead > 0 {
                first_corpse = Some(step);
            }
            if step % 10 == 0 || press {
                println!(
                    "step {step}{} near {near:.1} eng {engaged} dead {dead} ghp {:.0} php {} phase {:?} clip {} pos {:?}",
                    if press { " PRESS" } else { "" },
                    cast.grunts
                        .iter()
                        .map(|x| world.get::<Health>(x.entity()).map_or(-1.0, |h| h.current))
                        .fold(f32::MAX, f32::min),
                    g.warrior.health().current,
                    g.warrior.attack_phase(),
                    g.warrior.anim_state(),
                    g.pos,
                );
            }
        }
        println!("TAPS {taps:?}");
        println!("FIRST_HURT {first_hurt:?}  FIRST_CORPSE {first_corpse:?}");
    }
}
