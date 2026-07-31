//! The grunt brain: the dungeon's one monster, from spawn to corpse
//! (`docs/game-framework-plan.md` §4.4).
//!
//! A [`GruntBrain`] is a plain struct with a five-state machine
//! ([`Idle`](GruntState::Idle) → [`Chase`](GruntState::Chase) →
//! [`Attack`](GruntState::Attack), plus [`Hit`](GruntState::Hit) and
//! [`Dead`](GruntState::Dead)) and no engine dependencies at all: it moves circles with
//! [`dreamcoast_game::physics`], swings with [`dreamcoast_game::combat`], and picks clips
//! with [`dreamcoast_game::anim`]. Nothing here touches a device, a window, an asset or
//! the render graph, so the whole encounter — twelve monsters, five hundred ticks, one
//! aggro-to-corpse arc — runs in a unit test.
//!
//! # Spaces
//!
//! Every [`Vec2`] is **collision space**: grid-local, `.x` = world X, `.y` = world Z, the
//! space [`crate::collision`] defines and [`crate::pathing`] routes in. World positions
//! appear at exactly one seam, [`GruntBrain::render_position`], which is where the
//! integrator reads a pose to put on a transform.
//!
//! # Position-sync contract
//!
//! **The brain owns the position; the ECS transform is a view of it.**
//!
//! [`GruntBrain::pos`](GruntBrain::position) is the authority — the only thing the next
//! tick integrates from — and the integrator's job is to copy it *out*, once per rendered
//! frame, into a [`LocalTransform`](sandbox::scene::LocalTransform). Nothing reads a
//! transform back into the simulation. This is deliberately the same contract
//! [`crate::game`] already uses for the player (`render_update` is the single writer of
//! the placeholder's transform), for the same two reasons:
//!
//! * **No propagation-order coupling.** Combat resolves against positions that are
//!   *passed in* (see [`BodyCircle`](dreamcoast_game::combat::BodyCircle)); if the brain
//!   read its own position from a transform, every hit would land on either this tick's
//!   or last tick's world matrix depending on where the hook ran.
//! * **Interpolation stays visual.** The brain keeps the previous tick's position too, so
//!   [`render_position`](GruntBrain::render_position) can blend at the frame's `alpha`
//!   without that blended value ever feeding back into the simulation.
//!
//! # A tick, end to end
//!
//! ```text
//! tick_grunts(..)                      // aggro, path, move, swing -> DamageEvent
//!   -> apply_damage_events(..)         // the combat crate decides what lands (iframes!)
//!   -> feed_combat(grunts, dmg, deaths)// the outcome comes back as stagger / death
//!   -> render_update: transform = brain.render_position(..)
//! ```

// As `meshing.rs` and `rigs.rs`: the game loop wires these up in the integration step
// that follows. Until `game.rs` spawns monsters, nothing the binary reaches calls in here,
// and the tests are the only consumer.
#![allow(dead_code)]

use dreamcoast_game::anim::{AnimGraphDef, AnimMachine, Params};
use dreamcoast_game::combat::{
    AttackEvent, AttackPhase, AttackSpec, AttackState, ComboChain, DamageEvent, DeathEvent, Team,
};
use dreamcoast_game::physics::{self, GridCollision};
use glam::{Vec2, Vec3};
use sandbox::scene::{Entity, Events};

use crate::collision;
use crate::pathing::{DEFAULT_MAX_EXPANSIONS, Pathfinder, astar, walk_clear};
use crate::procgen::{Rng, TILE_SIZE, TileGrid};
use crate::rigs::{
    GRUNT_ATTACK_HIT_TIME, GRUNT_ATTACK_LEN, GRUNT_DEATH_LEN, GRUNT_HIT_LEN, GRUNT_IDLE_LEN,
    GRUNT_WALK_LEN,
};

// -------------------------------------------------------------------------------------
// Tuning
// -------------------------------------------------------------------------------------

/// The grunt's body radius, metres.
///
/// Slightly under the player's 0.4 ([`PLAYER_RADIUS`](crate::collision::PLAYER_RADIUS)):
/// a monster that cannot follow the player through its own dungeon's doorways is a bug
/// report, and the cheapest way to never have that conversation is to make the chaser the
/// smaller circle.
pub const GRUNT_RADIUS: f32 = 0.35;

/// Hit points. Three warrior openers (12 each), or two openers and a finisher.
pub const GRUNT_MAX_HEALTH: f32 = 30.0;

/// How far a grunt notices the player, metres — **and only with line of sight**.
///
/// Four tiles. Sight, not hearing: an aggro radius that reaches through walls turns every
/// room into an alarm bell for the two rooms next to it.
pub const AGGRO_RANGE: f32 = 8.0;

/// How far a grunt will chase before losing interest, metres.
///
/// Twice [`AGGRO_RANGE`], so a player who breaks contact at the edge of aggro still has to
/// earn the disengage rather than getting it for one step backwards.
pub const LEASH_RANGE: f32 = 16.0;

/// Chase speed, metres per second.
///
/// Below the warrior's 4.5 walk and well below its 6.98 sprint: a grunt is a threat you
/// have to stand and fight, not one you cannot walk away from. Closing only happens
/// because the dungeon has corners and because they arrive in groups.
pub const CHASE_SPEED: f32 = 2.6;

/// Centre-to-centre distance at which a grunt commits to a swing, metres.
///
/// The same number as the attack's reach ([`GruntClass::grunt`]), which is what makes the
/// commitment readable: the grunt starts its 0.30 s windup exactly when it *could* connect
/// standing still, so stepping back during the windup is always a clean dodge.
pub const ATTACK_RANGE: f32 = 1.6;

/// The swing's three phases, seconds: anticipation, hit window, commitment.
///
/// Hoisted out of [`GruntClass::grunt`] so the claw-timing guard below can be a
/// **compile-time** assertion rather than a test somebody might not run.
pub const ATTACK_WINDUP: f32 = 0.30;
/// See [`ATTACK_WINDUP`].
pub const ATTACK_ACTIVE: f32 = 0.12;
/// See [`ATTACK_WINDUP`].
pub const ATTACK_RECOVERY: f32 = 0.45;

/// The animation and the combat clock must agree about when the claw connects.
///
/// [`crate::rigs`] authors the moment of full extension; this module authors the window
/// damage is dealt in. They live in different files and neither imports the other's
/// intent, so drift here is a build failure rather than a bug report about a monster that
/// hurts you before it has moved.
const _: () = assert!(
    ATTACK_WINDUP <= GRUNT_ATTACK_HIT_TIME
        && GRUNT_ATTACK_HIT_TIME <= ATTACK_WINDUP + ATTACK_ACTIVE,
    "the grunt's hit window does not bracket the claw's full extension"
);

/// Longest a route is trusted before it is recomputed, seconds.
pub const REPATH_INTERVAL: f32 = 0.4;

/// How far the player must move from the tile a route was built for before the cadence is
/// pre-empted, in tiles (Chebyshev).
///
/// The cadence caps the steady-state cost; this catches the case the cadence is bad at —
/// a player sprinting across the route's face, which a 0.4 s-old path chases as a ghost.
/// It cannot fire much faster than the cadence anyway: a full tile is 2 m, which even a
/// sprinting warrior needs 0.29 s to cover.
pub const REPATH_GOAL_TILES: i32 = 1;

/// Hitstun, seconds. Long enough to read as a flinch and to interrupt a windup, short
/// enough that two grunts cannot stun-lock the player out of a fight.
pub const STAGGER_TIME: f32 = 0.25;

/// Distance under which two grunts push each other apart, metres.
pub const SEPARATION_RANGE: f32 = 0.8;

/// Maximum separation speed contributed to one grunt, metres per second.
///
/// The push is summed over neighbours and then **clamped to this**, which is what keeps a
/// pile-up from launching anybody: without the clamp, eleven neighbours at full overlap
/// would contribute 11x the push and the knot would explode.
pub const SEPARATION_SPEED: f32 = 1.6;

/// Grunts one [`tick_grunts`] call is designed for.
///
/// Separation is the only pairwise term and it is O(n^2) — 144 distance tests at 12, well
/// under a microsecond, and simpler than any acceleration structure would be. It is a
/// *design* cap, not an assertion: a larger slice still runs correctly, just quadratically,
/// and [`separation_for`] documents where the line is.
pub const MAX_GRUNTS: usize = 12;

/// How close a grunt must get to a waypoint to consider it reached, metres.
///
/// One body radius: the waypoint is a tile centre, and a circle whose edge touches it has
/// unambiguously arrived. Smaller and a grunt circles a point it can never stand on.
const WAYPOINT_ARRIVE: f32 = GRUNT_RADIUS;

/// Which team a grunt fights for.
pub const GRUNT_TEAM: Team = Team::ENEMY;

// -------------------------------------------------------------------------------------
// Class data
// -------------------------------------------------------------------------------------

/// The grunt as data: its stats, its one swing, and its animation graph.
///
/// One of these is shared by every grunt on the floor — the same split
/// [`AttackState`] makes, where the *state* is per monster and the *data* is not. A second
/// monster type is a second `GruntClass`-shaped value, not a second brain.
#[derive(Clone, Debug)]
pub struct GruntClass {
    /// Starting and maximum hit points.
    pub max_health: f32,
    /// Body radius on the XZ plane, metres.
    pub radius: f32,
    /// Chase speed, metres per second.
    pub move_speed: f32,
    /// Sight range for aggro, metres.
    pub aggro_range: f32,
    /// Chase give-up range, metres.
    pub leash_range: f32,
    /// Centre-to-centre distance that commits a swing, metres.
    pub attack_range: f32,
    /// The melee chain. A grunt has exactly one step: no combo, no link window, no
    /// buffered input — the chain type is still the right shape, and a heavier monster
    /// gets a second step by editing data.
    pub combo: ComboChain,
    /// The animation graph, parsed once and cloned into each brain's machine.
    pub anim: AnimGraphDef,
}

impl Default for GruntClass {
    fn default() -> Self {
        Self::grunt()
    }
}

impl GruntClass {
    /// The shipping grunt.
    ///
    /// | | |
    /// |---|---|
    /// | health | 30 |
    /// | speed | 2.6 m/s |
    /// | swing | 8 damage, 1.6 m reach, 80 degree arc |
    /// | phases | 0.30 windup / 0.12 active / 0.45 recovery |
    ///
    /// The phase split is the whole fight. 0.30 s of windup is a readable tell at the
    /// distance the grunt commits from, and the 0.45 s recovery is the punish window — the
    /// swing costs the grunt 0.87 s of standing still, against a warrior opener that
    /// resolves in 0.40. Trading blows is a losing trade *for the grunt*, which is what
    /// makes a group of them dangerous and a single one not.
    ///
    /// The hit window is where the animation and the combat clock meet: the claw reaches
    /// full extension at [`GRUNT_ATTACK_HIT_TIME`] (0.34 s, authored in
    /// [`crate::rigs`]), and `[windup, windup + active]` = `[0.30, 0.42]` brackets it —
    /// so damage lands on the frame the claw is out, not before the arm has moved. A test
    /// asserts that bracketing, because the two numbers live in different files and would
    /// otherwise drift apart silently.
    pub fn grunt() -> Self {
        Self {
            max_health: GRUNT_MAX_HEALTH,
            radius: GRUNT_RADIUS,
            move_speed: CHASE_SPEED,
            aggro_range: AGGRO_RANGE,
            leash_range: LEASH_RANGE,
            attack_range: ATTACK_RANGE,
            combo: ComboChain::new(vec![AttackSpec {
                name: "claw".to_string(),
                damage: 8.0,
                range: ATTACK_RANGE,
                half_angle_rad: 40f32.to_radians(),
                windup: ATTACK_WINDUP,
                active: ATTACK_ACTIVE,
                recovery: ATTACK_RECOVERY,
                stagger: STAGGER_TIME,
            }]),
            anim: AnimGraphDef::from_ron(GRUNT_ANIM_GRAPH)
                .expect("the compiled-in grunt graph must parse"),
        }
    }

    /// The single swing. A grunt's chain always has exactly one step.
    pub fn swing(&self) -> Option<&AttackSpec> {
        self.combo.get(0)
    }
}

/// The grunt's animation graph.
///
/// Five states over the five clips [`crate::rigs`] authors, and the transition list is in
/// **priority order** — the machine takes the first match, so the interrupts are declared
/// above the ordinary flow:
///
/// 1. `die` — beats everything. It is the **only** non-interruptible edge here: its blend
///    is locked so nothing can be halfway out of a death, and `death` has no outgoing edge
///    at all, which is what makes it terminal.
/// 2. `hit` — beats an attack in progress, which is the animation half of "a stagger
///    interrupts a windup". Deliberately *interruptible*: locking it would mean a grunt
///    staggered two ticks into its windup keeps swinging on screen for the rest of the
///    attack blend while the simulation has already flinched — five frames of the
///    animation disagreeing with the fight.
/// 3. `attack` — beats the locomotion pair, for the same reason.
/// 4. the rest: `attack`/`hit` fall back to `idle` when their clip ends
///    ([`StateDone`](dreamcoast_game::anim::Condition::StateDone)), and `idle`/`walk`
///    swap on the `moving` flag.
///
/// Lengths are authored here *and* overwritten from the `GRUNT_*_LEN` constants when a
/// machine is built ([`grunt_anim_machine`]): the values in the RON keep the file readable
/// and testable on its own, but [`crate::rigs`] is the single source of truth, because it
/// is what actually produces the clips.
pub const GRUNT_ANIM_GRAPH: &str = r#"(
    initial: "idle",
    states: [
        (name: "idle",   clip: "idle",   looping: true,  length: Some(2.0)),
        (name: "walk",   clip: "walk",   looping: true,  length: Some(0.9)),
        (name: "attack", clip: "attack",                 length: Some(0.7)),
        (name: "hit",    clip: "hit",                    length: Some(0.25)),
        (name: "death",  clip: "death",                  length: Some(1.0)),
    ],
    transitions: [
        (from: "any",    to: "death",  condition: Trigger("die"),    fade: 0.08, interruptible: false),
        (from: "any",    to: "hit",    condition: Trigger("hit"),    fade: 0.05),
        (from: "any",    to: "attack", condition: Trigger("attack"), fade: 0.08),
        (from: "attack", to: "idle",   condition: StateDone,         fade: 0.15),
        (from: "hit",    to: "idle",   condition: StateDone,         fade: 0.10),
        (from: "idle",   to: "walk",   condition: Flag("moving"),    fade: 0.12),
        (from: "walk",   to: "idle",   condition: NotFlag("moving"), fade: 0.12),
    ],
)"#;

/// A machine over [`GRUNT_ANIM_GRAPH`] with the real clip lengths installed.
///
/// Panics on a malformed graph, which is correct: the text is a compiled-in constant, so a
/// failure here is a programmer error that a test catches, not a runtime condition a
/// shipped game could hit.
pub fn grunt_anim_machine(def: &AnimGraphDef) -> AnimMachine {
    let mut machine = AnimMachine::new(def.clone()).expect("the grunt graph must be valid");
    for (clip, length) in [
        ("idle", GRUNT_IDLE_LEN),
        ("walk", GRUNT_WALK_LEN),
        ("attack", GRUNT_ATTACK_LEN),
        ("hit", GRUNT_HIT_LEN),
        ("death", GRUNT_DEATH_LEN),
    ] {
        machine.set_clip_length(clip, length);
    }
    machine
}

// -------------------------------------------------------------------------------------
// The player, as the AI sees it
// -------------------------------------------------------------------------------------

/// What a brain is allowed to know about the player.
///
/// Deliberately four fields and no `&World`: the AI cannot read the player's health,
/// stamina, inventory or attack state, so it cannot accidentally become omniscient, and a
/// test can put the player anywhere without building an ECS.
///
/// Note what is *not* here: whether the player is invulnerable. A grunt swings at a
/// dodging player exactly as it swings at a standing one — the roll is negated by
/// [`apply_damage_events`](dreamcoast_game::combat::apply_damage_events), the single place
/// that honours [`IFrames`](dreamcoast_game::combat::IFrames). Letting the AI peek would
/// give it a second opinion on whether a hit landed, and two opinions is one too many.
#[derive(Clone, Copy, Debug)]
pub struct PlayerView {
    /// The player entity, so a hit can name its target.
    pub entity: Entity,
    /// Position in collision space.
    pub pos: Vec2,
    /// Body radius, metres.
    pub radius: f32,
    /// Whether the player is still alive. A dead player is not a target.
    pub alive: bool,
}

// -------------------------------------------------------------------------------------
// The state machine
// -------------------------------------------------------------------------------------

/// Where a grunt is in its life.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GruntState {
    /// Standing still, watching for the player.
    #[default]
    Idle,
    /// Closing on the player along a route.
    Chase,
    /// Committed to a swing.
    Attack,
    /// Staggered by a hit.
    Hit,
    /// Terminal. The corpse stays where it fell and stops interacting with anything.
    Dead,
}

impl GruntState {
    /// Whether the grunt still participates in the simulation (movement, separation,
    /// being a target).
    #[inline]
    pub fn is_active(self) -> bool {
        self != Self::Dead
    }
}

/// One monster's brain.
///
/// Owns its position (see the module's position-sync contract), its combat clock, its
/// animation machine and the route it is following. Constructed once at spawn and stepped
/// by [`tick_grunts`]; it never allocates in the steady state except when a repath returns
/// a longer route than the last one.
pub struct GruntBrain {
    entity: Entity,
    state: GruntState,
    pos: Vec2,
    prev_pos: Vec2,
    facing: Vec2,
    attack: AttackState,
    anim: AnimMachine,
    params: Params,
    /// Remaining smoothed waypoints, collision space; `leg` is the one being walked to.
    route: Vec<Vec2>,
    leg: usize,
    /// Tile the current route was built to reach.
    goal: Option<(i32, i32)>,
    repath_timer: f32,
    stagger: f32,
    last_expansions: usize,
}

impl GruntBrain {
    /// A grunt standing at `pos` (collision space), idle and facing +X.
    pub fn new(class: &GruntClass, entity: Entity, pos: Vec2) -> Self {
        Self {
            entity,
            state: GruntState::Idle,
            pos,
            prev_pos: pos,
            facing: Vec2::X,
            attack: AttackState::new(),
            anim: grunt_anim_machine(&class.anim),
            params: Params::new(),
            route: Vec::new(),
            leg: 0,
            goal: None,
            repath_timer: 0.0,
            stagger: 0.0,
            last_expansions: 0,
        }
    }

    /// The ECS entity this brain drives.
    #[inline]
    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// Position in collision space at the end of the last tick — **the authority**.
    #[inline]
    pub fn position(&self) -> Vec2 {
        self.pos
    }

    /// Unit facing on the XZ plane.
    #[inline]
    pub fn facing(&self) -> Vec2 {
        self.facing
    }

    /// Facing as a Y rotation in radians, for a transform.
    pub fn yaw(&self) -> f32 {
        self.facing.x.atan2(self.facing.y)
    }

    /// Current state.
    #[inline]
    pub fn state(&self) -> GruntState {
        self.state
    }

    /// Whether this grunt is a corpse.
    #[inline]
    pub fn is_dead(&self) -> bool {
        self.state == GruntState::Dead
    }

    /// The animation machine, for the integrator to sample clips from.
    #[inline]
    pub fn anim(&self) -> &AnimMachine {
        &self.anim
    }

    /// The melee clock, for a HUD or a debug overlay.
    #[inline]
    pub fn attack(&self) -> &AttackState {
        &self.attack
    }

    /// Nodes the last repath expanded — the budget readout [`DEFAULT_MAX_EXPANSIONS`] is
    /// tuned against. `0` when the grunt has never had to path.
    #[inline]
    pub fn last_expansions(&self) -> usize {
        self.last_expansions
    }

    /// Waypoints still to walk, in order.
    pub fn route(&self) -> &[Vec2] {
        &self.route[self.leg.min(self.route.len())..]
    }

    /// The world pose to draw this frame: the two latest fixed-step positions blended by
    /// the frame's interpolation factor, lifted into world space at height `y`.
    ///
    /// Visual only. Nothing in the simulation reads it — see the module's position-sync
    /// contract.
    pub fn render_position(&self, grid: &TileGrid, alpha: f32, y: f32) -> Vec3 {
        collision::to_world(grid, self.prev_pos.lerp(self.pos, alpha.clamp(0.0, 1.0)), y)
    }

    /// The tuple the player's [`AttackState::resolve_hits`] wants: who, where, how big,
    /// whose side. A corpse is not offered — it is not a target.
    pub fn as_target(&self, class: &GruntClass) -> Option<(Entity, Vec2, f32, Team)> {
        self.state
            .is_active()
            .then_some((self.entity, self.pos, class.radius, GRUNT_TEAM))
    }

    /// React to having been hit: stagger for `stagger` seconds and abandon the swing.
    ///
    /// Idempotent within a stagger (a second hit refreshes it), and a no-op on a corpse.
    /// **This is the only way in**: it is called by [`feed_combat`] from the damage the
    /// combat crate actually resolved, so a hit negated by iframes cannot flinch a monster.
    pub fn on_damaged(&mut self, stagger: f32) {
        if self.is_dead() {
            return;
        }
        // The swing is dropped, not finished: nothing completed, so no `Finished` event
        // and no chance of the attack clock resuming where it was interrupted.
        self.attack.cancel();
        self.stagger = self.stagger.max(if stagger.is_finite() {
            stagger.max(0.0)
        } else {
            0.0
        });
        self.state = GruntState::Hit;
        self.route.clear();
        self.leg = 0;
        self.goal = None;
        // A route the grunt is no longer on is worth less than an immediate one.
        self.repath_timer = 0.0;
        self.raise("hit");
    }

    /// React to having died. Terminal: nothing moves this brain again.
    pub fn on_died(&mut self) {
        if self.is_dead() {
            return;
        }
        self.attack.cancel();
        self.state = GruntState::Dead;
        self.stagger = 0.0;
        self.route.clear();
        self.leg = 0;
        self.goal = None;
        // Flags as well as triggers: a grunt killed mid-stride leaves `moving` set, and
        // although the death state has no outgoing edge today, a corpse whose animation
        // parameters still say "walking" is a trap laid for whoever edits the graph next.
        self.params.clear();
        self.params.trigger("die");
    }

    /// Raise exactly one animation trigger, clearing any other pending one first.
    ///
    /// The machine takes **one transition per tick** and leaves an unused trigger pending,
    /// so a tick that both staggers and kills a grunt would otherwise fire `die` now and
    /// `hit` next tick — walking the animation straight back out of the death state it is
    /// supposed to be locked into. One trigger at a time makes that unrepresentable.
    fn raise(&mut self, trigger: &str) {
        self.params.clear_triggers();
        self.params.trigger(trigger);
    }
}

// -------------------------------------------------------------------------------------
// Sensing
// -------------------------------------------------------------------------------------

/// Whether `to` is visible from `from` — a single ray against the solid tiles.
///
/// A point ray, with no radius inflation, because this is *sight*: a grunt can see a
/// player through a gap it could not walk through, which is right, and it stops it seeing
/// through a wall, which is the part that matters. Movement asks the other question
/// ([`walk_clear`]).
pub fn line_of_sight<M: physics::SolidMap + ?Sized>(
    map: GridCollision<'_, M>,
    from: Vec2,
    to: Vec2,
) -> bool {
    let seg = to - from;
    let len = seg.length();
    if !seg.is_finite() {
        return false;
    }
    if len <= 1e-6 {
        return true;
    }
    map.raycast(from, seg / len, len).is_none()
}

// -------------------------------------------------------------------------------------
// Separation
// -------------------------------------------------------------------------------------

/// The push-apart velocity one grunt receives from its neighbours, metres per second.
///
/// Read against every other grunt's [`prev_pos`](GruntBrain::prev_pos) — the position it
/// held at the **start** of this tick — rather than its live one. That single choice is
/// what makes separation independent of the order the grunts are stepped in: the
/// alternative, separating against the neighbours already moved earlier in the slice,
/// makes the outcome a function of array order, and array order is not a game rule. It is
/// also why no scratch buffer is needed — `prev_pos` already exists for render
/// interpolation and is, by definition, the snapshot this wants.
///
/// The push is linear in overlap, symmetric between any pair (both sides read the same two
/// start-of-tick positions), and **clamped** to [`SEPARATION_SPEED`] so a knot cannot
/// launch anybody: without the clamp, eleven neighbours at full overlap would contribute
/// 11x the push. Corpses neither push nor are pushed.
///
/// # Cost
///
/// O(n) per grunt, so O(n^2) over a call. Sized for [`MAX_GRUNTS`] (12 → 144 distance
/// tests, well under a microsecond); it stays correct above that and simply stops being
/// free. A floor that wants a hundred monsters wants a spatial hash, and should get one
/// then rather than now.
pub fn separation_for(class: &GruntClass, grunts: &[GruntBrain], index: usize) -> Vec2 {
    let Some(self_grunt) = grunts.get(index).filter(|g| g.state.is_active()) else {
        return Vec2::ZERO;
    };
    let range = SEPARATION_RANGE.max(class.radius * 2.0);
    let mut push = Vec2::ZERO;
    for (other_index, other) in grunts.iter().enumerate() {
        if other_index == index || !other.state.is_active() {
            continue;
        }
        let offset = other.prev_pos - self_grunt.prev_pos;
        let dist = offset.length();
        if dist >= range {
            continue;
        }
        // Exactly coincident grunts have no direction to separate along. Pick a fixed one
        // rather than a random one: reproducible, and the pair breaks symmetry on the next
        // tick anyway. The lower index goes -X so the pair still pushes *apart*.
        let dir = if dist > 1e-4 {
            offset / dist
        } else if other_index > index {
            Vec2::X
        } else {
            -Vec2::X
        };
        push -= dir * (SEPARATION_SPEED * (1.0 - dist / range));
    }
    push.clamp_length_max(SEPARATION_SPEED)
}

// -------------------------------------------------------------------------------------
// The tick
// -------------------------------------------------------------------------------------

/// Step every grunt one fixed timestep.
///
/// `finder` is threaded through rather than owned per monster so that twelve brains share
/// one A* workspace and one heap allocation (see [`Pathfinder`]). `damage` collects this
/// tick's hits; the caller resolves them with
/// [`apply_damage_events`](dreamcoast_game::combat::apply_damage_events) and feeds the
/// outcome back through [`feed_combat`].
///
/// Deterministic and allocation-free: every grunt's `prev_pos` is latched before any of
/// them move, separation reads only those latched positions, and everything after that is
/// a per-grunt function of `(grid, class, player, dt)` plus its own state. The same inputs
/// give the same tick, on any machine, whatever order the slice happens to be in.
pub fn tick_grunts(
    grid: &TileGrid,
    class: &GruntClass,
    finder: &mut Pathfinder,
    grunts: &mut [GruntBrain],
    player: PlayerView,
    dt: f32,
    damage: &mut Events<DamageEvent>,
) {
    if !dt.is_finite() || dt <= 0.0 {
        return;
    }
    // Latch the start-of-tick positions first, for separation *and* for the render
    // interpolation both halves of the frame read (see the position-sync contract).
    for grunt in grunts.iter_mut() {
        grunt.prev_pos = grunt.pos;
    }
    let ctx = TickCtx {
        grid,
        class,
        player,
        dt,
    };
    for index in 0..grunts.len() {
        // Two disjoint borrows in sequence: the whole slice to read the neighbourhood,
        // then one grunt to step it.
        let separation = separation_for(class, grunts, index);
        grunts[index].tick(&ctx, finder, separation, damage);
    }
}

/// Everything a grunt's tick reads and cannot change.
///
/// Bundled rather than passed as six parameters: the split between *what the world is*
/// (here, immutable, shared by every grunt in the call) and *what this grunt does about
/// it* is the module's central shape, and threading it as a context makes that shape
/// visible at every internal call site instead of a wall of arguments.
struct TickCtx<'a> {
    grid: &'a TileGrid,
    class: &'a GruntClass,
    player: PlayerView,
    dt: f32,
}

impl TickCtx<'_> {
    /// The collision handle for this tick's grid.
    #[inline]
    fn map(&self) -> GridCollision<'_, TileGrid> {
        collision::collision(self.grid)
    }
}

impl GruntBrain {
    /// One grunt, one fixed step.
    fn tick(
        &mut self,
        ctx: &TickCtx<'_>,
        finder: &mut Pathfinder,
        separation: Vec2,
        damage: &mut Events<DamageEvent>,
    ) {
        let (class, player, dt) = (ctx.class, ctx.player, ctx.dt);
        if self.is_dead() {
            // A corpse still animates — it is playing its death clip — and does nothing
            // else. No movement, no separation, no params: `on_died` cleared them and
            // nothing may set them again.
            self.anim.tick(dt, &mut self.params);
            return;
        }

        let map = ctx.map();
        let to_player = player.pos - self.pos;
        let dist = to_player.length();
        let sees_player = player.alive && line_of_sight(map, self.pos, player.pos);

        // Steering intent for this tick, in metres per second. Only `Chase` produces one;
        // separation is added to whatever it is, in every live state, so that a grunt
        // rooted in a swing still cannot be stood inside.
        let mut steer = Vec2::ZERO;

        match self.state {
            GruntState::Idle => {
                if sees_player && dist <= class.aggro_range {
                    self.enter_chase();
                }
            }
            GruntState::Chase => {
                if !player.alive || dist > class.leash_range {
                    self.give_up();
                } else if dist <= class.attack_range && sees_player {
                    self.enter_attack(class, to_player);
                } else {
                    steer = self.chase_steer(ctx, finder);
                }
            }
            GruntState::Attack => {
                self.tick_attack(ctx, to_player, dist, sees_player, damage);
            }
            GruntState::Hit => {
                self.stagger -= dt;
                if self.stagger <= 0.0 {
                    self.stagger = 0.0;
                    if player.alive {
                        self.enter_chase();
                    } else {
                        self.give_up();
                    }
                }
            }
            GruntState::Dead => unreachable!("handled above"),
        }

        // Move. `move_circle` is the authority that a grunt never ends inside geometry —
        // waypoints and steering are only ever a request.
        let delta = steer * dt + separation * dt;
        let moved = map.move_circle(self.pos, class.radius, delta);
        let travelled = moved.pos - self.pos;
        self.pos = moved.pos;

        // Face where the body is going while chasing; the attack states aim themselves
        // (see `tick_attack`) and must not be overridden by a separation nudge.
        if self.state == GruntState::Chase && travelled.length_squared() > 1e-8 {
            self.facing = travelled.normalize();
        }

        debug_assert!(
            self.pos.is_finite() && self.facing.is_finite(),
            "grunt {:?}: non-finite state (pos {:?}, facing {:?})",
            self.entity,
            self.pos,
            self.facing
        );

        // The locomotion flag is about the *body*, measured after the move, so a grunt
        // pinned against a wall by a hopeless route plays idle instead of moonwalking.
        let moving = self.state == GruntState::Chase && travelled.length_squared() > 1e-6;
        self.params.set_flag("moving", moving);
        self.anim.tick(dt, &mut self.params);
    }

    /// Enter [`GruntState::Chase`], forcing an immediate repath.
    fn enter_chase(&mut self) {
        self.state = GruntState::Chase;
        self.route.clear();
        self.leg = 0;
        self.goal = None;
        self.repath_timer = 0.0;
    }

    /// Drop back to [`GruntState::Idle`] — the player is dead, out of leash, or
    /// unreachable. All three look the same to a grunt, on purpose.
    fn give_up(&mut self) {
        self.state = GruntState::Idle;
        self.route.clear();
        self.leg = 0;
        self.goal = None;
        self.attack.cancel();
    }

    /// Commit to a swing.
    fn enter_attack(&mut self, class: &GruntClass, to_player: Vec2) {
        self.state = GruntState::Attack;
        self.route.clear();
        self.leg = 0;
        self.goal = None;
        if let Some(dir) = to_player.try_normalize() {
            self.facing = dir;
        }
        if self.attack.request(&class.combo) {
            self.raise("attack");
        }
    }

    /// The chase's steering velocity, repathing on the cadence when it has to.
    fn chase_steer(&mut self, ctx: &TickCtx<'_>, finder: &mut Pathfinder) -> Vec2 {
        let (class, player) = (ctx.class, ctx.player);
        let map = ctx.map();
        self.repath_timer -= ctx.dt;

        // The straight shot, checked first and every tick. When the grunt can simply walk
        // at the player there is no route worth computing, and the chase reads better for
        // it — a monster that string-pulls its way across an empty room looks like it is
        // navigating rather than hunting.
        if walk_clear(map, class.radius, self.pos, player.pos) {
            self.route.clear();
            self.leg = 0;
            self.goal = None;
            // Repath the instant the shot closes, rather than a cadence later.
            self.repath_timer = 0.0;
            return (player.pos - self.pos).normalize_or_zero() * class.move_speed;
        }

        let goal = physics::world_to_tile(player.pos, TILE_SIZE);
        let goal_moved = self.goal.is_none_or(|(gx, gz)| {
            (goal.0 - gx).abs().max((goal.1 - gz).abs()) > REPATH_GOAL_TILES
        });
        let exhausted = self.leg >= self.route.len();
        if self.repath_timer <= 0.0 || goal_moved || exhausted {
            self.repath_timer = REPATH_INTERVAL;
            match finder.find_smoothed(
                ctx.grid,
                class.radius,
                self.pos,
                goal,
                DEFAULT_MAX_EXPANSIONS,
            ) {
                Some(points) => {
                    self.route.clear();
                    self.route.extend_from_slice(points);
                    self.leg = 0;
                    self.goal = Some(goal);
                }
                None => {
                    // No route within budget: the player is behind a sealed door, on the
                    // other side of a collapsed corridor, or simply too far to afford.
                    // Indistinguishable from a grunt's point of view, and all three mean
                    // the same thing.
                    self.give_up();
                    return Vec2::ZERO;
                }
            }
            self.last_expansions = finder.expansions();
        }

        // Consume every waypoint already reached, then steer at the next one.
        while self.leg < self.route.len()
            && (self.route[self.leg] - self.pos).length() <= WAYPOINT_ARRIVE
        {
            self.leg += 1;
        }
        match self.route.get(self.leg) {
            Some(&target) => (target - self.pos).normalize_or_zero() * class.move_speed,
            // Route walked out without arriving: the next tick repaths (`exhausted`).
            None => Vec2::ZERO,
        }
    }

    /// Drive the melee clock for one tick, and decide what happens when it ends.
    fn tick_attack(
        &mut self,
        ctx: &TickCtx<'_>,
        to_player: Vec2,
        dist: f32,
        sees_player: bool,
        damage: &mut Events<DamageEvent>,
    ) {
        let (class, player) = (ctx.class, ctx.player);
        // Aim during the windup only. The grunt tracks a strafing player right up to the
        // moment it is committed, and then it is committed — which is what makes the
        // 0.30 s tell worth reading instead of a homing missile with a delay.
        if self.attack.phase() == AttackPhase::Windup
            && let Some(dir) = to_player.try_normalize()
        {
            self.facing = dir;
        }

        // Resolve every tick rather than only on `HitWindowOpen`: the window spans several
        // ticks and a player who steps *into* the arc mid-window has been hit by any
        // reasonable reading of the fight. Hitting once per swing is guaranteed by
        // `AttackState`'s already-struck set, not by resolving once.
        if player.alive {
            let struck = self.attack.resolve_hits(
                &class.combo,
                self.pos,
                self.facing,
                GRUNT_TEAM,
                [(player.entity, player.pos, player.radius, Team::PLAYER)],
            );
            if let Some(spec) = self.attack.spec(&class.combo) {
                for target in struck {
                    damage.send(DamageEvent::new(
                        self.entity,
                        target,
                        spec.damage,
                        self.facing,
                    ));
                }
            }
        }

        let events = self.attack.tick(&class.combo, ctx.dt);
        if events.iter().any(|e| e == AttackEvent::Finished) {
            // Recovery is over. Swing again if the player is still there to swing at,
            // otherwise go back to closing the distance.
            if !player.alive {
                self.give_up();
            } else if dist <= class.attack_range && sees_player {
                self.enter_attack(class, to_player);
            } else {
                self.enter_chase();
            }
        }
    }
}

// -------------------------------------------------------------------------------------
// Combat feedback
// -------------------------------------------------------------------------------------

/// Feed the tick's resolved combat back into the brains: stagger what was hurt, kill what
/// died.
///
/// Call **after**
/// [`apply_damage_events`](dreamcoast_game::combat::apply_damage_events), with the same
/// damage channel and the death channel it wrote. Deaths are applied last and win: a grunt
/// that was staggered and killed in the same tick is a corpse, not a flinching corpse.
///
/// Grunts carry no [`IFrames`](dreamcoast_game::combat::IFrames), so every damage event
/// addressed to a live grunt is one that landed. If that ever changes — an armoured
/// variant, a parry window — this function is where the filter belongs, because it is
/// already the one place the AI learns what combat decided.
pub fn feed_combat<'a>(
    grunts: &mut [GruntBrain],
    damage: impl IntoIterator<Item = &'a DamageEvent>,
    deaths: impl IntoIterator<Item = &'a DeathEvent>,
) {
    for event in damage {
        if let Some(grunt) = grunts.iter_mut().find(|g| g.entity == event.target) {
            grunt.on_damaged(STAGGER_TIME);
        }
    }
    for event in deaths {
        if let Some(grunt) = grunts.iter_mut().find(|g| g.entity == event.entity) {
            grunt.on_died();
        }
    }
}

// -------------------------------------------------------------------------------------
// Spawn placement
// -------------------------------------------------------------------------------------

/// Deterministic spawn positions for `count` grunts, in **collision space**.
///
/// The rules, in the order they are applied:
///
/// 1. **Rooms only.** Corridors are where the player walks, not where monsters wait; a
///    grunt spawned in a one-tile corridor also has nowhere to spread out to.
/// 2. **Never the entry room.** The player materialises there and must get a moment to
///    understand the floor before anything walks at them.
/// 3. **At least `min_dist_from_entry` metres away**, measured as *walking* distance
///    (breadth-first over the tile graph, the same metric the generator uses to place the
///    exit) rather than as the crow flies. A monster two metres away through a wall is
///    thirty metres of walking away, and it is the walking that decides whether the player
///    meets it in the first ten seconds.
/// 4. **Distinct rooms first.** The eligible rooms are shuffled and then filled
///    round-robin, so eight grunts across five rooms are 2/2/2/1/1, never 8/0/0/0/0.
/// 5. **Free space.** The tile centre is snapped with
///    [`nearest_free`](dreamcoast_game::physics::nearest_free) for [`GRUNT_RADIUS`] and
///    rejected if it still overlaps.
/// 6. **Reachable by the shipping pathfinder.** Every accepted point is confirmed with an
///    [`astar`] from the entry under [`DEFAULT_MAX_EXPANSIONS`]. That is *not* the same
///    check as step 3: breadth-first connectivity says a route exists, this says the
///    monsters' own pathfinder can find it within the budget it will actually run under.
///    A spawn that fails it is a monster that would stand still forever.
///
/// Deterministic in `(grid, count, min_dist_from_entry, rng_seed)` and nothing else.
/// Returns **fewer than `count`** points when the floor has nowhere left to put them —
/// a small dungeon is a valid dungeon, and a caller that needs the exact count should
/// check the length rather than trust it.
pub fn spawn_points(
    grid: &TileGrid,
    count: usize,
    min_dist_from_entry: f32,
    rng_seed: u64,
) -> Vec<Vec2> {
    let mut out = Vec::new();
    if count == 0 {
        return out;
    }
    let entry = grid.entry();
    let map = collision::collision(grid);
    let steps = grid.bfs_distances(entry);
    let min_steps = if min_dist_from_entry.is_finite() {
        (min_dist_from_entry.max(0.0) / TILE_SIZE).ceil() as u32
    } else {
        0
    };
    let entry_room = grid.room_id_at(entry.0, entry.1);
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
                if !grid.is_walkable(x, z) {
                    continue;
                }
                let reach = steps[(z * grid.width() + x) as usize];
                if reach == u32::MAX || reach < min_steps {
                    continue;
                }
                tiles.push((x, z));
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
                let centre = physics::tile_center(x, z, TILE_SIZE);
                let Some(free) = map.nearest_free(centre, GRUNT_RADIUS) else {
                    continue;
                };
                if map.circle_overlaps(free, GRUNT_RADIUS) {
                    continue;
                }
                if astar(grid, entry, (x, z), DEFAULT_MAX_EXPANSIONS).is_none() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procgen::{DungeonParams, generate};
    use dreamcoast_game::combat::{BodyCircle, Health, IFrames, apply_damage_events, tick_iframes};
    use sandbox::scene::World;

    const DT: f32 = 1.0 / 60.0;

    /// The test arena: 14x11 tiles, so 28x22 metres — sized against [`AGGRO_RANGE`]
    /// (8 m = four tiles) rather than against the ASCII, which is the mistake that makes a
    /// hand-drawn fixture quietly untestable.
    ///
    /// Two wall stubs split the middle into three bands. They are one tile thick and leave
    /// a gap at each end, so any two points either side of one are close enough to aggro
    /// through but have to be *walked around* — which is the only geometry that tells a
    /// working chase apart from a grunt drifting toward the player in a straight line.
    ///
    /// ```text
    ///     x 0    5    10
    ///   z 0 ##############
    ///     1 #E...........#
    ///     2 #............#
    ///     3 #..#######...#
    ///     4 #............#
    ///     5 #............#
    ///     6 #............#
    ///     7 #..#######...#
    ///     8 #............#
    ///     9 #..........X.#
    ///    10 ##############
    /// ```
    const ARENA: [&str; 11] = [
        "##############",
        "#E...........#",
        "#............#",
        "#..#######...#",
        "#............#",
        "#............#",
        "#............#",
        "#..#######...#",
        "#............#",
        "#..........X.#",
        "##############",
    ];

    fn arena() -> TileGrid {
        TileGrid::from_rows(&ARENA)
    }

    /// Collision-space centre of a tile.
    fn at(x: i32, z: i32) -> Vec2 {
        physics::tile_center(x, z, TILE_SIZE)
    }

    /// A world with a player entity and `n` grunt entities.
    fn world_with(n: usize) -> (World, Entity, Vec<Entity>) {
        let mut world = World::new();
        let player = world.spawn();
        world.insert(player, Health::new(100.0));
        world.insert(player, Team::PLAYER);
        world.insert(player, BodyCircle::new(0.4));
        world.insert(player, IFrames::default());
        let grunts = (0..n)
            .map(|_| {
                let e = world.spawn();
                world.insert(e, Health::new(GRUNT_MAX_HEALTH));
                world.insert(e, Team::ENEMY);
                world.insert(e, BodyCircle::new(GRUNT_RADIUS));
                e
            })
            .collect();
        (world, player, grunts)
    }

    fn view(entity: Entity, pos: Vec2) -> PlayerView {
        PlayerView {
            entity,
            pos,
            radius: 0.4,
            alive: true,
        }
    }

    /// Step the brains, resolve the damage they produced, and feed the outcome back —
    /// the full loop the module docs describe, so tests exercise the real ordering.
    fn step(
        grid: &TileGrid,
        class: &GruntClass,
        finder: &mut Pathfinder,
        grunts: &mut [GruntBrain],
        world: &mut World,
        player: PlayerView,
    ) -> Vec<DamageEvent> {
        let mut damage: Events<DamageEvent> = Events::new();
        let mut deaths: Events<DeathEvent> = Events::new();
        tick_iframes(world, DT);
        tick_grunts(grid, class, finder, grunts, player, DT, &mut damage);
        apply_damage_events(world, damage.iter(), &mut deaths);
        feed_combat(grunts, damage.iter(), deaths.iter());
        damage.iter().copied().collect()
    }

    // -- data ---------------------------------------------------------------------------

    /// The claw's full extension has to fall inside the hit window, or the grunt damages
    /// you with its arm still cocked. The two numbers live in different files, so nothing
    /// but this test stops them drifting.
    #[test]
    fn the_hit_window_brackets_the_animation_hit_time() {
        let class = GruntClass::grunt();
        let spec = class.swing().expect("the grunt has one swing");
        assert!(
            spec.windup <= GRUNT_ATTACK_HIT_TIME
                && GRUNT_ATTACK_HIT_TIME <= spec.windup + spec.active,
            "claw extends at {GRUNT_ATTACK_HIT_TIME}, window is [{}, {}]",
            spec.windup,
            spec.windup + spec.active
        );
        // And the window is longer than a tick, or the swing can fall between two frames.
        assert!(spec.active > DT, "the hit window can be stepped over");
    }

    /// The compiled-in graph parses, and every clip it names is one the rig authors.
    #[test]
    fn the_anim_graph_matches_the_rig() {
        let class = GruntClass::grunt();
        let machine = grunt_anim_machine(&class.anim);
        for state in &class.anim.states {
            assert!(
                crate::rigs::GRUNT_CLIPS.contains(&state.clip.as_str()),
                "state '{}' names clip '{}', which the rig does not author",
                state.name,
                state.clip
            );
        }
        // The rig's lengths win over the ones authored in the RON.
        assert_eq!(machine.clip_length("attack"), GRUNT_ATTACK_LEN);
        assert_eq!(machine.clip_length("death"), GRUNT_DEATH_LEN);
        assert_eq!(machine.current_state(), "idle");
    }

    // -- aggro --------------------------------------------------------------------------

    #[test]
    fn aggro_needs_range_and_line_of_sight() {
        let grid = arena();
        let class = GruntClass::grunt();
        let mut finder = Pathfinder::new();
        let (mut world, player, ids) = world_with(1);

        // Four metres apart, with the upper stub between them.
        let mut grunts = vec![GruntBrain::new(&class, ids[0], at(4, 4))];
        let blocked = view(player, at(4, 2));
        let map = collision::collision(&grid);
        assert!(
            (blocked.pos - grunts[0].position()).length() < class.aggro_range,
            "the fixture must put the player inside aggro range"
        );
        assert!(
            !line_of_sight(map, grunts[0].position(), blocked.pos),
            "the fixture must put a wall between them"
        );
        for _ in 0..10 {
            step(&grid, &class, &mut finder, &mut grunts, &mut world, blocked);
        }
        assert_eq!(
            grunts[0].state(),
            GruntState::Idle,
            "aggro saw through a wall"
        );

        // The same distance again, this time in the open band: aggro fires at once.
        let seen = view(player, at(6, 5));
        assert!((seen.pos - grunts[0].position()).length() < class.aggro_range);
        step(&grid, &class, &mut finder, &mut grunts, &mut world, seen);
        assert_eq!(grunts[0].state(), GruntState::Chase);
    }

    #[test]
    fn a_player_out_of_range_is_ignored_even_in_the_open() {
        // A long open corridor: clear sight, too far to care.
        let rows: Vec<String> = std::iter::once("#".repeat(22))
            .chain(std::iter::once(format!("#E{}#", ".".repeat(19))))
            .chain(std::iter::once("#".repeat(22)))
            .collect();
        let refs: Vec<&str> = rows.iter().map(String::as_str).collect();
        let grid = TileGrid::from_rows(&refs);
        let class = GruntClass::grunt();
        let mut finder = Pathfinder::new();
        let (mut world, player, ids) = world_with(1);
        let mut grunts = vec![GruntBrain::new(&class, ids[0], at(1, 1))];

        let far = view(player, at(19, 1)); // 36 m away, in plain sight
        let map = collision::collision(&grid);
        assert!(line_of_sight(map, grunts[0].position(), far.pos));
        for _ in 0..10 {
            step(&grid, &class, &mut finder, &mut grunts, &mut world, far);
        }
        assert_eq!(grunts[0].state(), GruntState::Idle);
    }

    // -- chase --------------------------------------------------------------------------

    #[test]
    fn a_chase_closes_the_distance_and_repaths_when_the_player_moves() {
        let grid = arena();
        let class = GruntClass::grunt();
        let mut finder = Pathfinder::new();
        let (mut world, player, ids) = world_with(1);
        let mut grunts = vec![GruntBrain::new(&class, ids[0], at(5, 2))];
        let map = collision::collision(&grid);

        // Aggro in the open band above the stub.
        let mut target = view(player, at(8, 2));
        step(&grid, &class, &mut finder, &mut grunts, &mut world, target);
        assert_eq!(grunts[0].state(), GruntState::Chase);

        // Now the player steps below the stub: six metres away, but only reachable the
        // long way round, so this is a route rather than a straight line.
        target = view(player, at(5, 5));
        assert!(
            !walk_clear(map, class.radius, grunts[0].position(), target.pos),
            "the fixture must force a path"
        );
        step(&grid, &class, &mut finder, &mut grunts, &mut world, target);
        assert_eq!(
            grunts[0].goal,
            Some(physics::world_to_tile(target.pos, TILE_SIZE)),
            "the chase did not build a route"
        );
        assert!(
            grunts[0].last_expansions() > 0,
            "the chase never actually ran A*"
        );

        // Moving the player a long way pre-empts the 0.4 s cadence: the route is rebuilt
        // for the new tile on the very next tick, not up to 24 ticks later.
        let goal_before = grunts[0].goal;
        let moved = view(player, at(2, 5));
        step(&grid, &class, &mut finder, &mut grunts, &mut world, moved);
        assert_ne!(
            grunts[0].goal, goal_before,
            "the route still points at the old tile"
        );
        assert_eq!(
            grunts[0].goal,
            Some(physics::world_to_tile(moved.pos, TILE_SIZE))
        );

        // ...and the chase actually arrives.
        let start_gap = (target.pos - grunts[0].position()).length();
        for _ in 0..400 {
            step(&grid, &class, &mut finder, &mut grunts, &mut world, target);
        }
        let gap = (target.pos - grunts[0].position()).length();
        assert!(
            gap < start_gap - 1.0,
            "the grunt closed only {:.2} m of {start_gap:.2}",
            start_gap - gap
        );
    }

    #[test]
    fn a_chase_gives_up_past_the_leash() {
        let rows: Vec<String> = std::iter::once("#".repeat(24))
            .chain(std::iter::once(format!("#E{}#", ".".repeat(21))))
            .chain(std::iter::once("#".repeat(24)))
            .collect();
        let refs: Vec<&str> = rows.iter().map(String::as_str).collect();
        let grid = TileGrid::from_rows(&refs);
        let class = GruntClass::grunt();
        let mut finder = Pathfinder::new();
        let (mut world, player, ids) = world_with(1);
        let mut grunts = vec![GruntBrain::new(&class, ids[0], at(1, 1))];

        // Aggro at 4 m.
        let near = view(player, at(3, 1));
        step(&grid, &class, &mut finder, &mut grunts, &mut world, near);
        assert_eq!(grunts[0].state(), GruntState::Chase);

        // Now 40 m away — past the 16 m leash.
        let gone = view(player, at(21, 1));
        step(&grid, &class, &mut finder, &mut grunts, &mut world, gone);
        assert_eq!(grunts[0].state(), GruntState::Idle);
        assert!(grunts[0].route().is_empty());
    }

    #[test]
    fn a_dead_player_ends_the_chase() {
        let grid = arena();
        let class = GruntClass::grunt();
        let mut finder = Pathfinder::new();
        let (mut world, player, ids) = world_with(1);
        let mut grunts = vec![GruntBrain::new(&class, ids[0], at(1, 1))];

        let mut target = view(player, at(4, 1));
        step(&grid, &class, &mut finder, &mut grunts, &mut world, target);
        assert_eq!(grunts[0].state(), GruntState::Chase);
        target.alive = false;
        step(&grid, &class, &mut finder, &mut grunts, &mut world, target);
        assert_eq!(grunts[0].state(), GruntState::Idle);
    }

    // -- attack -------------------------------------------------------------------------

    /// One swing produces exactly one damage event, however many ticks its window spans.
    #[test]
    fn a_swing_damages_the_player_exactly_once() {
        let grid = arena();
        let class = GruntClass::grunt();
        let spec = class.swing().unwrap().clone();
        let mut finder = Pathfinder::new();
        let (mut world, player, ids) = world_with(1);
        let mut grunts = vec![GruntBrain::new(&class, ids[0], at(6, 6))];

        // A metre away, standing still: in range, in sight.
        let target = view(player, at(6, 6) + Vec2::new(1.0, 0.0));
        let mut hits = Vec::new();
        for _ in 0..((spec.duration() / DT) as usize + 4) {
            hits.extend(step(
                &grid,
                &class,
                &mut finder,
                &mut grunts,
                &mut world,
                target,
            ));
        }
        assert_eq!(hits.len(), 1, "one swing, {} hits: {hits:?}", hits.len());
        assert_eq!(hits[0].target, player);
        assert_eq!(hits[0].attacker, ids[0]);
        assert_eq!(hits[0].amount, spec.damage);
        assert_eq!(
            world.get::<Health>(player).unwrap().current,
            100.0 - spec.damage
        );
    }

    /// Backing out during the windup is a clean dodge: the grunt is committed and swings
    /// at nothing.
    #[test]
    fn stepping_out_of_the_arc_during_the_windup_whiffs() {
        let grid = arena();
        let class = GruntClass::grunt();
        let mut finder = Pathfinder::new();
        let (mut world, player, ids) = world_with(1);
        let mut grunts = vec![GruntBrain::new(&class, ids[0], at(6, 6))];

        // Two steps: noticing and committing are separate ticks (Idle -> Chase -> Attack).
        let close = view(player, at(6, 6) + Vec2::new(1.0, 0.0));
        step(&grid, &class, &mut finder, &mut grunts, &mut world, close);
        step(&grid, &class, &mut finder, &mut grunts, &mut world, close);
        assert_eq!(grunts[0].state(), GruntState::Attack);

        // Two tiles back, still in the same room, well outside the 1.6 m reach.
        let away = view(player, at(6, 6) + Vec2::new(4.0, 0.0));
        let mut hits = Vec::new();
        for _ in 0..60 {
            hits.extend(step(
                &grid,
                &class,
                &mut finder,
                &mut grunts,
                &mut world,
                away,
            ));
        }
        assert!(hits.is_empty(), "the whiffed swing connected: {hits:?}");
        assert_eq!(world.get::<Health>(player).unwrap().current, 100.0);
        // And the grunt went back to closing rather than swinging at air forever.
        assert_eq!(grunts[0].state(), GruntState::Chase);
    }

    /// Rolling through the swing costs the player nothing — and the AI does not get to
    /// notice and re-swing, because it never learns the hit was negated.
    #[test]
    fn iframes_negate_the_whole_swing() {
        let grid = arena();
        let class = GruntClass::grunt();
        let mut finder = Pathfinder::new();
        let (mut world, player, ids) = world_with(1);
        let mut grunts = vec![GruntBrain::new(&class, ids[0], at(6, 6))];
        let target = view(player, at(6, 6) + Vec2::new(1.0, 0.0));

        // A window long enough to cover the whole swing.
        world.insert(player, IFrames::new(2.0));
        let mut hits = Vec::new();
        for _ in 0..70 {
            hits.extend(step(
                &grid,
                &class,
                &mut finder,
                &mut grunts,
                &mut world,
                target,
            ));
        }
        assert_eq!(hits.len(), 1, "the arc should still report a connection");
        assert_eq!(
            world.get::<Health>(player).unwrap().current,
            100.0,
            "iframes did not absorb the hit"
        );
    }

    // -- hit and death ------------------------------------------------------------------

    #[test]
    fn a_stagger_interrupts_the_windup() {
        let grid = arena();
        let class = GruntClass::grunt();
        let mut finder = Pathfinder::new();
        let (mut world, player, ids) = world_with(1);
        let mut grunts = vec![GruntBrain::new(&class, ids[0], at(6, 6))];
        let target = view(player, at(6, 6) + Vec2::new(1.0, 0.0));

        step(&grid, &class, &mut finder, &mut grunts, &mut world, target);
        step(&grid, &class, &mut finder, &mut grunts, &mut world, target);
        assert_eq!(grunts[0].state(), GruntState::Attack);
        assert_eq!(grunts[0].attack().phase(), AttackPhase::Windup);

        assert_eq!(
            grunts[0].anim().current_state(),
            "attack",
            "the attack trigger never reached the machine"
        );

        grunts[0].on_damaged(STAGGER_TIME);
        assert_eq!(grunts[0].state(), GruntState::Hit);
        assert!(
            !grunts[0].attack().is_attacking(),
            "the swing survived the stagger"
        );

        // The animation follows the brain on the next tick: `hit` outranks `attack` in the
        // graph's transition order, so the reaction plays over the interrupted swing.
        step(&grid, &class, &mut finder, &mut grunts, &mut world, target);
        assert_eq!(grunts[0].anim().current_state(), "hit");

        // Nothing lands while it is staggered, and it recovers into a chase.
        let mut hits = Vec::new();
        let ticks = (STAGGER_TIME / DT).ceil() as usize;
        for _ in 0..ticks {
            hits.extend(step(
                &grid,
                &class,
                &mut finder,
                &mut grunts,
                &mut world,
                target,
            ));
        }
        assert!(hits.is_empty(), "a staggered grunt hit: {hits:?}");
        assert_eq!(world.get::<Health>(player).unwrap().current, 100.0);
        assert_ne!(
            grunts[0].state(),
            GruntState::Hit,
            "the stagger never ended"
        );
    }

    #[test]
    fn death_is_terminal_and_the_corpse_stays_put() {
        let grid = arena();
        let class = GruntClass::grunt();
        let mut finder = Pathfinder::new();
        let (mut world, player, ids) = world_with(1);
        let mut grunts = vec![GruntBrain::new(&class, ids[0], at(6, 6))];
        let target = view(player, at(6, 6) + Vec2::new(1.0, 0.0));

        step(&grid, &class, &mut finder, &mut grunts, &mut world, target);
        let resting = grunts[0].position();
        grunts[0].on_died();
        assert_eq!(grunts[0].state(), GruntState::Dead);
        assert!(grunts[0].is_dead());
        assert!(
            grunts[0].as_target(&class).is_none(),
            "a corpse is still a target"
        );

        let mut hits = Vec::new();
        for _ in 0..200 {
            hits.extend(step(
                &grid,
                &class,
                &mut finder,
                &mut grunts,
                &mut world,
                target,
            ));
        }
        assert_eq!(grunts[0].state(), GruntState::Dead, "the corpse got up");
        assert_eq!(grunts[0].position(), resting, "the corpse drifted");
        assert!(hits.is_empty());
        assert_eq!(grunts[0].anim().current_state(), "death");

        // And nothing can move it afterwards.
        grunts[0].on_damaged(STAGGER_TIME);
        assert_eq!(grunts[0].state(), GruntState::Dead);
    }

    /// A tick that both staggers and kills a grunt must not walk the animation back out of
    /// the death state on the following tick — the trigger-set rule in `raise`.
    #[test]
    fn a_stagger_and_a_death_in_one_tick_stay_dead() {
        let grid = arena();
        let class = GruntClass::grunt();
        let mut finder = Pathfinder::new();
        let (mut world, player, ids) = world_with(1);
        let mut grunts = vec![GruntBrain::new(&class, ids[0], at(6, 6))];
        let target = view(player, at(6, 6) + Vec2::new(1.0, 0.0));

        let damage = [DamageEvent::new(player, ids[0], 999.0, Vec2::X)];
        let deaths = [DeathEvent { entity: ids[0] }];
        feed_combat(&mut grunts, &damage, &deaths);
        assert_eq!(grunts[0].state(), GruntState::Dead);

        for _ in 0..30 {
            step(&grid, &class, &mut finder, &mut grunts, &mut world, target);
        }
        assert_eq!(grunts[0].anim().current_state(), "death");
    }

    // -- separation ---------------------------------------------------------------------

    #[test]
    fn crowded_grunts_push_each_other_apart_and_stay_apart() {
        let grid = arena();
        let class = GruntClass::grunt();
        let mut finder = Pathfinder::new();
        let (mut world, player, ids) = world_with(4);
        // Four grunts nearly on top of each other in the open half of the arena.
        let mut grunts: Vec<GruntBrain> = ids
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                let jitter = Vec2::new(i as f32 * 0.05, (i % 2) as f32 * 0.05);
                GruntBrain::new(&class, e, at(8, 2) + jitter)
            })
            .collect();
        let target = view(player, at(1, 7));

        let mut worst = f32::INFINITY;
        for tick in 0..200 {
            step(&grid, &class, &mut finder, &mut grunts, &mut world, target);
            if tick < 60 {
                continue; // one second for the initial pile to resolve
            }
            for i in 0..grunts.len() {
                for j in (i + 1)..grunts.len() {
                    let d = (grunts[i].position() - grunts[j].position()).length();
                    worst = worst.min(d);
                    assert!(d.is_finite(), "non-finite separation at tick {tick}");
                }
            }
        }
        assert!(
            worst > class.radius,
            "grunts ended up inside each other: closest pair {worst:.3} m"
        );
    }

    #[test]
    fn separation_is_symmetric_and_bounded() {
        let class = GruntClass::grunt();
        let (_world, _player, ids) = world_with(3);
        let mut grunts: Vec<GruntBrain> = ids
            .iter()
            .map(|&e| GruntBrain::new(&class, e, at(4, 4)))
            .collect();

        // Exactly coincident bodies still separate, along the documented fallback axis,
        // and the sum stays inside the clamp.
        let pushes: Vec<Vec2> = (0..grunts.len())
            .map(|i| separation_for(&class, &grunts, i))
            .collect();
        for v in &pushes {
            assert!(
                v.length() <= SEPARATION_SPEED + 1e-4,
                "unbounded push {v:?}"
            );
        }
        assert!(pushes.iter().any(|v| v.length() > 0.0));

        // A pair pushes apart, equally and oppositely — order in the slice is not a rule.
        let mut pair: Vec<GruntBrain> = ids[..2]
            .iter()
            .map(|&e| GruntBrain::new(&class, e, at(4, 4)))
            .collect();
        pair[1].pos += Vec2::new(0.3, 0.0);
        pair[1].prev_pos = pair[1].pos;
        let a = separation_for(&class, &pair, 0);
        let b = separation_for(&class, &pair, 1);
        assert!((a + b).length() < 1e-5, "asymmetric push {a:?} vs {b:?}");
        assert!(a.x < 0.0 && b.x > 0.0, "the pair pushed together");

        // Corpses neither push nor are pushed.
        grunts[0].on_died();
        assert_eq!(separation_for(&class, &grunts, 0), Vec2::ZERO);
        let live = separation_for(&class, &grunts, 1);
        grunts.remove(0);
        assert_eq!(
            live,
            separation_for(&class, &grunts, 0),
            "a corpse still shoved the living"
        );
    }

    // -- spawn placement ----------------------------------------------------------------

    #[test]
    fn spawn_points_are_deterministic_and_legal() {
        for seed in 0..12u64 {
            let grid = generate(seed, &DungeonParams::default());
            let map = collision::collision(&grid);
            let entry_room = grid.room_id_at(grid.entry().0, grid.entry().1);
            let points = spawn_points(&grid, MAX_GRUNTS, 12.0, seed ^ 0xA1);
            assert_eq!(
                points,
                spawn_points(&grid, MAX_GRUNTS, 12.0, seed ^ 0xA1),
                "seed {seed}: spawn placement is not reproducible"
            );
            assert!(!points.is_empty(), "seed {seed}: nowhere to put a monster");
            assert!(points.len() <= MAX_GRUNTS);

            let steps = grid.bfs_distances(grid.entry());
            for &p in &points {
                let (x, z) = physics::world_to_tile(p, TILE_SIZE);
                let room = grid
                    .room_at(x, z)
                    .unwrap_or_else(|| panic!("seed {seed}: spawn at {x},{z} is not in a room"));
                assert_ne!(room.id, entry_room, "seed {seed}: spawn in the entry room");
                assert!(
                    !map.circle_overlaps(p, GRUNT_RADIUS),
                    "seed {seed}: spawn at {x},{z} overlaps geometry"
                );
                let walk = steps[(z * grid.width() + x) as usize] as f32 * TILE_SIZE;
                assert!(
                    walk >= 12.0 - TILE_SIZE,
                    "seed {seed}: spawn only {walk} m from the entry"
                );
                assert!(
                    astar(&grid, grid.entry(), (x, z), DEFAULT_MAX_EXPANSIONS).is_some(),
                    "seed {seed}: spawn is unreachable"
                );
            }
        }
    }

    #[test]
    fn spawn_points_fill_distinct_rooms_first() {
        let mut spread = 0;
        let mut samples = 0;
        for seed in 0..12u64 {
            let grid = generate(seed, &DungeonParams::default());
            let eligible = grid
                .rooms()
                .iter()
                .filter(|r| r.id != grid.room_id_at(grid.entry().0, grid.entry().1))
                .count();
            let points = spawn_points(&grid, 4, 8.0, seed);
            if points.len() < 4 || eligible < 4 {
                continue;
            }
            samples += 1;
            let mut rooms: Vec<u16> = points
                .iter()
                .map(|&p| {
                    let (x, z) = physics::world_to_tile(p, TILE_SIZE);
                    grid.room_id_at(x, z)
                })
                .collect();
            rooms.sort_unstable();
            rooms.dedup();
            assert_eq!(
                rooms.len(),
                4,
                "seed {seed}: 4 grunts landed in {} rooms with {eligible} available",
                rooms.len()
            );
            spread += 1;
        }
        assert!(samples > 0, "no seed produced a testable floor");
        assert_eq!(spread, samples);
    }

    #[test]
    fn spawn_points_degrade_instead_of_failing() {
        let grid = arena();
        // The arena is one room, and it holds the entry — so there is nowhere legal.
        assert!(spawn_points(&grid, 4, 0.0, 1).is_empty());
        // And a zero request is a zero answer, on any floor.
        let dungeon = generate(3, &DungeonParams::default());
        assert!(spawn_points(&dungeon, 0, 0.0, 1).is_empty());
        // An impossible distance requirement empties the candidate set rather than
        // relaxing itself.
        assert!(spawn_points(&dungeon, 4, 1.0e6, 1).is_empty());
    }

    // -- the whole encounter ------------------------------------------------------------

    /// Three grunts, a player walking a scripted route, five hundred ticks.
    ///
    /// The assertions are the ones that catch an entire class of bug rather than one:
    /// nothing goes non-finite, nothing ends up inside a wall, and a full
    /// aggro → chase → attack → hit → death arc is observed rather than assumed.
    #[test]
    fn a_scripted_encounter_runs_clean_from_aggro_to_corpse() {
        let grid = arena();
        let class = GruntClass::grunt();
        let map = collision::collision(&grid);
        let mut finder = Pathfinder::new();
        let (mut world, player_entity, ids) = world_with(3);
        let mut grunts: Vec<GruntBrain> = ids
            .iter()
            .enumerate()
            .map(|(i, &e)| GruntBrain::new(&class, e, at(6, 4 + i as i32)))
            .collect();

        // The player walks down the left edge, into the middle band where the grunts are,
        // then back and stands its ground.
        let waypoints = [at(1, 1), at(1, 5), at(9, 5), at(3, 5), at(5, 5)];
        let player_speed = 4.5;
        let mut player_pos = waypoints[0];
        let mut leg = 1usize;

        let mut seen_chase = false;
        let mut seen_attack = false;
        let mut seen_hit = false;
        let mut seen_dead = false;
        let mut player_hits = 0usize;

        for tick in 0..500 {
            // Walk the player along its route (through the world, not through walls).
            if leg < waypoints.len() {
                let to = waypoints[leg] - player_pos;
                let stepped =
                    map.move_circle(player_pos, 0.4, to.normalize_or_zero() * player_speed * DT);
                player_pos = stepped.pos;
                if to.length() < 0.3 {
                    leg += 1;
                }
            }
            let player = view(player_entity, player_pos);

            // The player swings back once the fight is joined, so grunts actually die.
            let mut damage: Events<DamageEvent> = Events::new();
            let mut deaths: Events<DeathEvent> = Events::new();
            tick_iframes(&mut world, DT);
            tick_grunts(
                &grid,
                &class,
                &mut finder,
                &mut grunts,
                player,
                DT,
                &mut damage,
            );
            if tick > 200 && tick % 20 == 0 {
                for grunt in grunts.iter() {
                    if let Some((entity, pos, radius, _)) = grunt.as_target(&class)
                        && (pos - player_pos).length() <= 2.2
                    {
                        let _ = radius;
                        damage.send(DamageEvent::new(player_entity, entity, 12.0, Vec2::X));
                    }
                }
            }
            apply_damage_events(&mut world, damage.iter(), &mut deaths);
            feed_combat(&mut grunts, damage.iter(), deaths.iter());
            player_hits += damage.iter().filter(|e| e.target == player_entity).count();

            for grunt in grunts.iter() {
                let p = grunt.position();
                assert!(
                    p.is_finite() && grunt.facing().is_finite(),
                    "tick {tick}: non-finite grunt state {p:?}"
                );
                assert!(
                    !map.circle_overlaps(p, class.radius),
                    "tick {tick}: grunt at {p:?} is inside a wall"
                );
                assert!(
                    grunt.yaw().is_finite() && grunt.render_position(&grid, 0.5, 0.35).is_finite()
                );
                match grunt.state() {
                    GruntState::Chase => seen_chase = true,
                    GruntState::Attack => seen_attack = true,
                    GruntState::Hit => seen_hit = true,
                    GruntState::Dead => seen_dead = true,
                    GruntState::Idle => {}
                }
            }
        }

        assert!(seen_chase, "no grunt ever chased");
        assert!(seen_attack, "no grunt ever attacked");
        assert!(seen_hit, "no grunt was ever staggered");
        assert!(seen_dead, "no grunt ever died");
        assert!(player_hits > 0, "the player was never actually threatened");
        assert!(
            grunts.iter().any(GruntBrain::is_dead),
            "the arc never reached a corpse"
        );
        // The repath budget is real: nothing came close to the cap.
        for grunt in &grunts {
            assert!(
                grunt.last_expansions() <= DEFAULT_MAX_EXPANSIONS,
                "repath blew its budget: {}",
                grunt.last_expansions()
            );
        }
        assert!(world.get::<Health>(player_entity).unwrap().current < 100.0);
    }

    /// The same encounter, twice, produces the same trace — no clock, no OS randomness, no
    /// iteration-order dependence anywhere in the brain.
    #[test]
    fn the_encounter_is_reproducible() {
        let run = || {
            let grid = arena();
            let class = GruntClass::grunt();
            let mut finder = Pathfinder::new();
            let (mut world, player_entity, ids) = world_with(3);
            let mut grunts: Vec<GruntBrain> = ids
                .iter()
                .enumerate()
                .map(|(i, &e)| GruntBrain::new(&class, e, at(4 + i as i32 * 2, 5)))
                .collect();
            let mut log = Vec::new();
            for tick in 0..240 {
                let t = tick as f32 * DT;
                let player = view(player_entity, at(6, 5) + Vec2::new(t.sin(), t.cos()));
                step(&grid, &class, &mut finder, &mut grunts, &mut world, player);
                for grunt in grunts.iter() {
                    log.push(format!(
                        "{tick} {:?} {:.6} {:.6} {}",
                        grunt.state(),
                        grunt.position().x,
                        grunt.position().y,
                        grunt.anim().current_state()
                    ));
                }
            }
            log
        };
        let first = run();
        assert_eq!(first, run());
        assert!(first.iter().any(|l| l.contains("Attack")), "trace is inert");
    }
}
