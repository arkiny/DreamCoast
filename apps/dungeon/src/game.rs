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
//! The one way to break that pairing is the engine's **level hot-swap dropdown**: it
//! lists every `dungeon_<seed>.level` this machine has generated, and picking a
//! different one loads new geometry while this grid stays the old dungeon. The
//! characters are re-acquired and re-snapped to free space (see
//! [`DungeonGame::acquire`]), but it will be free space *of the previous floor*.
//! Swapping floors properly is M3's progression work (generate → write → load → hand the
//! game the new grid); until then, a seed change means a restart with `--seed`.
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

use dreamcoast_game::anim::AnimMachine;
use dreamcoast_game::combat::{
    BodyCircle, DamageEvent, DeathEvent, Health, IFrames, Team, apply_damage_events, tick_iframes,
};
use dreamcoast_game::input::{ActionState, BindingsConfig, InputSnapshot, InputSource};
use glam::{Quat, Vec2, Vec3};
use sandbox::imgui;
use sandbox::scene::{Entity, Events, LocalTransform, Name, World};
use sandbox::{CameraPose, GameHooks};

use crate::ai::{self, GruntBrain, GruntClass, GruntState, PlayerView};
use crate::characters::{ChildIndex, ClipSample, ClipSet, RigBinding};
use crate::collision::{self, CHARACTER_Y, PLAYER_RADIUS};
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
    grid: TileGrid,
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
    /// Latched once the player's circle first touches the exit tile. Progression to the
    /// next floor is M3; this detects and surfaces it.
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
    pub fn new(grid: TileGrid, grunts: usize) -> anyhow::Result<Self> {
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

        let spawn = collision::player_spawn_local(&grid);
        let focus = collision::to_world(&grid, spawn, CAMERA_FOCUS_Y);
        Ok(Self {
            input: ActionState::new(map),
            grid,
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
            "dungeon: cast acquired — player #{} ({} clips), {} grunts",
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
        self.warrior = WarriorController::new();
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
        for entity in self.deaths.iter().map(|d| d.entity) {
            Self::retire_corpse(world, entity);
        }
        let report = self.warrior.take_damage(player_hits);
        if report.died {
            tracing::info!(
                "dungeon: the warrior died after {} steps ({:.1} s)",
                self.steps,
                self.steps as f64 * f64::from(dt),
            );
        }
        self.damage.update();
        self.deaths.update();

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

    /// Exit detection (this milestone surfaces it; M3 makes it progression).
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
                ui.text_disabled("WASD move, Shift sprint, LMB/J attack, Space dodge");
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

        if self.warrior.is_dead() {
            ui.text_colored([0.92, 0.26, 0.24, 1.0], "YOU DIED");
            // Restart is M3's; saying so here is better than a banner that looks broken.
            ui.text_disabled("restart is not implemented yet (M3)");
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
            "seed      {}   {}x{} tiles, {} rooms",
            self.grid.seed(),
            self.grid.width(),
            self.grid.height(),
            self.grid.rooms().len()
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
    fn game(grid: TileGrid) -> DungeonGame {
        DungeonGame::new(grid, 0).unwrap()
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
        world
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
        let exit_world = grid.exit_world();
        let mut g = game(grid);

        let mut world = World::new();
        spawn_like_the_level_loader(
            &mut world,
            &g.warrior_rig,
            PLAYER_NAME,
            Vec3::new(exit_world.x, CHARACTER_Y, exit_world.z + TILE_SIZE),
        );
        frame(&mut g, &mut world, &InputSnapshot::default(), 1);
        assert!(!g.exit_reached, "not on the exit yet");
        assert_ne!(g.player_tile(), (ex, ez));

        frame(
            &mut g,
            &mut world,
            &InputSnapshot::default().with_key(W, true),
            60,
        );
        assert!(g.exit_reached, "walking onto the exit sets the flag");
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
        let mut g = DungeonGame::new(grid, 0).unwrap();
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
        let mut g = DungeonGame::new(dungeon(3), DEFAULT_GRUNTS).unwrap();
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
        let mut g = DungeonGame::new(generate(seed, &DungeonParams::default()), grunts).unwrap();
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
