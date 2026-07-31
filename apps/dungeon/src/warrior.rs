//! The warrior: the player character's controller (`docs/game-framework-plan.md` §4.3).
//!
//! This module is the *whole* player character, assembled from parts that each know
//! nothing about the others:
//!
//! * [`dreamcoast_game::combat`] owns the phase clock, the combo rules and the arc test;
//! * [`dreamcoast_game::anim`] owns the state graph — the shipped
//!   `crates/game/assets/warrior_anim.ron` itself, `include_str!`d from the framework
//!   crate rather than copied (see [`WARRIOR_ANIM_GRAPH`]);
//! * [`dreamcoast_game::physics`] owns the mover;
//! * [`crate::rigs`] owns the clips — and therefore the *timing* of everything visible.
//!
//! What is left, and what lives here, is the **glue that has to agree**: which phase
//! locks movement, when a dodge may cancel what, where the arc is tested from, and —
//! the load-bearing one — that the frame the blade crosses the arc is the frame the
//! damage cone opens.
//!
//! # No engine, no window
//!
//! [`WarriorController::tick`] takes a [`WarriorInput`] (already reduced to a direction
//! and three booleans) and a [`WarriorCtx`] (a [`SolidMap`] + the current position +
//! the targets), and returns a [`WarriorOutput`]. It never reads a key, samples a clip,
//! touches the ECS or looks at a clock. So the entire character — combo timing, i-frame
//! windows, hit resolution, death — is exercised by ordinary unit tests, and the
//! integrator is left with a mechanical job: fill the input, route the output.
//!
//! # Coordinates
//!
//! 2D on the **XZ plane**, `Vec2::x` = world X and `Vec2::y` = world Z, exactly like
//! [`dreamcoast_game::physics`]. Positions in and out are **collision space** (the
//! grid-local space of [`crate::collision`]), because that is the space the mover works
//! in; the integrator converts with [`crate::collision::to_world`].
//!
//! [`WarriorOutput::facing_radians`] is the yaw to write on the character's root node:
//! the rigs author **+Z as forward** ([`crate::rigs`]), and a rotation of `θ` about +Y
//! takes `(0, 0, 1)` to `(sin θ, 0, cos θ)` — hence `atan2(x, z)`.
//!
//! # The two reconciliations
//!
//! Wave 1 authored the combat data and the clips independently, so two sets of numbers
//! describe the same three swings. Neither file is edited here (both are locked
//! baselines with their own tests); the controller **derives** the agreement at
//! construction, which is what keeps one source of truth per fact:
//!
//! 1. **Timing** — [`clip_aligned_chain`]. `ClassDef::warrior()` is the authority on
//!    *damage, reach, arc and hit-window length*; the clips are the authority on *when*.
//!    Each swing is re-timed so its windup ends at the clip's authored hit time and the
//!    whole swing lasts exactly one clip (see that function for the numbers).
//! 2. **Naming** — [`WarriorClip`]. The graph names its clips after the *swings*
//!    (`slash_left`), the rigs name them after the *slots* (`attack1`). One enum maps
//!    between them, so a rename in either file fails a test here instead of silently
//!    playing an idle pose mid-combo.
//!
//! # Graph adaptations
//!
//! The graph (not editable from here) defines the vocabulary, and the controller speaks
//! it rather than the other way round:
//!
//! * there is no single `"attack"` trigger — the three swings have **one trigger each**
//!   (`attack_1`/`attack_2`/`attack_3`), which is what lets a linked combo show three
//!   different clips. The controller fires the one for the step that just started, on
//!   the opener and on every [`AttackEvent::ComboAdvanced`].
//! * death is a **flag**, not a trigger (`Flag("dead")`), because "is dead" is a state
//!   the game can answer every tick. There is no `"die"` trigger to fire.
//! * `death` has no outgoing edge, but the graph's wildcard interrupts *would* still
//!   fire from it. Terminality is therefore the controller's job, per the fixture's own
//!   header: [`WarriorController::tick`] stops feeding triggers once dead.

// The controller is now driven by `game.rs`, but it deliberately exposes a wider
// read-only surface than the game loop happens to consume — `dodge_time`,
// `IncomingHit::from_spec`, the class-RON constructor — because it is the *character*,
// not the character's current caller: a HUD, a replay, a second class file and the tests
// below each read a different subset. This silences "never used", not "never checked".
#![allow(dead_code)]

use dreamcoast_game::anim::{AnimError, AnimMachine, Params};
use dreamcoast_game::combat::{
    AttackEvent, AttackPhase, AttackSpec, AttackState, ClassDef, CombatError, ComboChain,
    DamageEvent, Health, IFrames, Team,
};
use dreamcoast_game::physics::{GridCollision, SolidMap};
use glam::Vec2;
use sandbox::scene::Entity;

use crate::rigs::{
    WARRIOR_ATTACK1_HIT_TIME, WARRIOR_ATTACK1_LEN, WARRIOR_ATTACK2_HIT_TIME, WARRIOR_ATTACK2_LEN,
    WARRIOR_ATTACK3_HIT_TIME, WARRIOR_ATTACK3_LEN, WARRIOR_DEATH_LEN, WARRIOR_DODGE_IFRAME_END,
    WARRIOR_DODGE_IFRAME_START, WARRIOR_DODGE_LEN, WARRIOR_HIT_LEN, WARRIOR_IDLE_LEN,
    WARRIOR_RUN_LEN,
};

/// The shipped animation graph — the framework crate's own fixture, re-exported
/// rather than copied or reached for by path.
///
/// This used to `include_str!` `../../../crates/game/assets/warrior_anim.ron`: the
/// right *bytes* (a copy would drift the first time either side is tuned) reached
/// through the wrong seam — a relative path out of one crate's `src` and into
/// another's `assets`, which breaks the day either crate moves and which no
/// `cargo` dependency edge describes. `crates/game` now declares the fixture as
/// what it always was, API surface
/// ([`WARRIOR_ANIM_RON`](dreamcoast_game::anim::WARRIOR_ANIM_RON)), so this is a
/// re-export and the path is gone.
pub const WARRIOR_ANIM_GRAPH: &str = dreamcoast_game::anim::WARRIOR_ANIM_RON;

/// Turn rate while running, radians per second.
///
/// 12 rad/s is a half-turn in ~0.26 s: fast enough that a top-down player feels the
/// character answer the stick immediately, slow enough that the silhouette reads as a
/// body turning rather than a sprite flipping. Attacks and dodges bypass it entirely
/// (they snap), because a committed action must go where it was aimed.
pub const TURN_RATE: f32 = 12.0;

/// Graph parameter names. Spelling lives here once, and
/// `the_controllers_vocabulary_exists_in_the_graph` proves the graph answers each one.
const FLAG_MOVING: &str = "moving";
const FLAG_DEAD: &str = "dead";
const TRIGGER_DODGE: &str = "dodge";
const TRIGGER_HIT: &str = "hit";
/// One trigger per combo step — see the module docs.
const SWING_TRIGGERS: [&str; 3] = ["attack_1", "attack_2", "attack_3"];
/// The graph *state* the hit reaction plays in — the one state the controller has to
/// recognise by name, because it is the only wildcard target it can already be in (see
/// [`WarriorController::take_damage`]). `the_controllers_vocabulary_exists_in_the_graph`
/// proves the graph has it.
const STATE_HIT: &str = "hit";

// --- Clips ---------------------------------------------------------------------------

/// One of the warrior's eight clips: the bridge between the graph's clip names and the
/// rig's.
///
/// Both files are locked baselines with independent tests, and they disagree on naming
/// (`slash_left` vs `attack1`) because each is named after what *it* cares about. This
/// enum is the single place the two vocabularies meet, so the mapping is checked once,
/// at construction, instead of by string comparison at every sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WarriorClip {
    Idle,
    Run,
    /// Combo step 0 — the graph calls it `slash_left`.
    Attack1,
    /// Combo step 1 — the graph calls it `slash_right`.
    Attack2,
    /// Combo step 2 — the graph calls it `overhead`.
    Attack3,
    Dodge,
    /// The hit reaction.
    Hit,
    Death,
}

impl WarriorClip {
    /// Every clip, in [`crate::rigs::WARRIOR_CLIPS`] order.
    pub const ALL: [Self; 8] = [
        Self::Idle,
        Self::Run,
        Self::Attack1,
        Self::Attack2,
        Self::Attack3,
        Self::Dodge,
        Self::Hit,
        Self::Death,
    ];

    /// The clip name used by the **animation graph**.
    pub fn graph_clip(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Run => "run",
            Self::Attack1 => "slash_left",
            Self::Attack2 => "slash_right",
            Self::Attack3 => "overhead",
            Self::Dodge => "dodge",
            Self::Hit => "hit",
            Self::Death => "death",
        }
    }

    /// The clip name used by the **rig asset** (a member of
    /// [`crate::rigs::WARRIOR_CLIPS`]) — what the integrator looks up in the loaded glTF.
    pub fn rig_clip(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Run => "run",
            Self::Attack1 => "attack1",
            Self::Attack2 => "attack2",
            Self::Attack3 => "attack3",
            Self::Dodge => "dodge",
            Self::Hit => "hit",
            Self::Death => "death",
        }
    }

    /// Resolve a graph clip name.
    pub fn from_graph_clip(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.graph_clip() == name)
    }

    /// The authored clip length in seconds — the rig's number, not the graph's
    /// placeholder (see [`WarriorController::with_class`]).
    pub fn length(self) -> f32 {
        match self {
            Self::Idle => WARRIOR_IDLE_LEN,
            Self::Run => WARRIOR_RUN_LEN,
            Self::Attack1 => WARRIOR_ATTACK1_LEN,
            Self::Attack2 => WARRIOR_ATTACK2_LEN,
            Self::Attack3 => WARRIOR_ATTACK3_LEN,
            Self::Dodge => WARRIOR_DODGE_LEN,
            Self::Hit => WARRIOR_HIT_LEN,
            Self::Death => WARRIOR_DEATH_LEN,
        }
    }

    /// Whether the clip wraps (locomotion) or plays once (everything else).
    pub fn looping(self) -> bool {
        matches!(self, Self::Idle | Self::Run)
    }
}

/// What the clips say about one combo step: when the blade crosses the arc, how long the
/// clip runs, and how far the body carries itself while swinging.
#[derive(Clone, Copy, Debug)]
struct SwingClip {
    /// [`crate::rigs`]' authored hit instant, seconds from the start of the swing.
    hit_time: f32,
    /// The clip's full length in seconds.
    length: f32,
    /// Metres travelled along the facing during the hit window — the step into the
    /// swing. Authored per swing rather than as one global constant so a heavier
    /// finisher can lunge further without a code change.
    step: f32,
    /// The clip this swing plays.
    clip: WarriorClip,
}

/// The three swings' clip data, in combo order.
///
/// 0.4 m over a hit window is a shift of weight, not a dash: at the opener's 0.15 s that
/// is 2.7 m/s (under the 4.5 m/s walk), so a swing can close a hair of distance without
/// ever out-running an enemy that is backing off. Uniform across the chain in v1 —
/// the field exists so tuning the finisher is data, not a rewrite.
const WARRIOR_SWINGS: [SwingClip; 3] = [
    SwingClip {
        hit_time: WARRIOR_ATTACK1_HIT_TIME,
        length: WARRIOR_ATTACK1_LEN,
        step: 0.4,
        clip: WarriorClip::Attack1,
    },
    SwingClip {
        hit_time: WARRIOR_ATTACK2_HIT_TIME,
        length: WARRIOR_ATTACK2_LEN,
        step: 0.4,
        clip: WarriorClip::Attack2,
    },
    SwingClip {
        hit_time: WARRIOR_ATTACK3_HIT_TIME,
        length: WARRIOR_ATTACK3_LEN,
        step: 0.4,
        clip: WarriorClip::Attack3,
    },
];

/// Re-time a combo chain against the authored clips.
///
/// The rule, applied per step: **windup ends at the clip's hit time, the hit window
/// keeps its authored length, and whatever the clip has left is recovery.**
///
/// * *Windup = hit time* is the whole point: the damage cone opens on the frame the
///   blade is in front of the character. Any other choice makes the game lie about what
///   it is showing — the two failure modes are "it hit me before the swing" and "the
///   sword went through and nothing happened".
/// * *Active is preserved* because it is a gameplay quantity, not a visual one: it is
///   the number `ClassDef::validate` checks against the fixed step, and shrinking it to
///   fit a clip is how a swing starts falling between two ticks.
/// * *Recovery absorbs the difference*, being the phase whose job is already "the cost
///   of having swung" — and the phase a combo link cancels anyway.
///
/// With `ClassDef::warrior()` and the shipped rig, that is:
///
/// | step | windup (was) | active | recovery (was) | total (was) |
/// |---|---|---|---|---|
/// | `slash_left`  | **0.28** (0.25) | 0.15 | **0.12** (0.35) | **0.55** (0.75) |
/// | `slash_right` | **0.24** (0.28) | 0.16 | **0.10** (0.38) | **0.50** (0.82) |
/// | `overhead`    | **0.46** (0.34) | 0.20 | **0.19** (0.55) | **0.85** (1.09) |
///
/// The clips are quicker than the data assumed, so the chain gets snappier and the
/// finisher keeps the longest commitment — the shape the class documents survives.
///
/// Steps beyond the three authored clips (a longer chain from a modded `ClassDef`) pass
/// through **unchanged**: there is no clip to align them to, and inventing one would be
/// worse than leaving the authored timing alone.
pub fn clip_aligned_chain(combo: &ComboChain) -> ComboChain {
    ComboChain::new(
        combo
            .iter()
            .enumerate()
            .map(|(i, spec)| match WARRIOR_SWINGS.get(i) {
                Some(clip) => AttackSpec {
                    windup: clip.hit_time,
                    recovery: (clip.length - clip.hit_time - spec.active).max(0.0),
                    ..spec.clone()
                },
                None => spec.clone(),
            })
            .collect(),
    )
}

/// Whether the dodge is invulnerable `time` seconds into the roll.
///
/// The window is the clip's own crouch, `[WARRIOR_DODGE_IFRAME_START,
/// WARRIOR_DODGE_IFRAME_END)` = `[0.07, 0.26)`, **not** `DodgeDef::iframes`. The class
/// file says "0.30 s from the start of the roll", authored before the clip existed; the
/// clip says the body is only off the ground between 0.07 and 0.26. Believing the class
/// would make the warrior invulnerable while still standing up, which is the one thing
/// a dodge must not be — so the visible pose wins, exactly as it does for hit windows.
/// The roll still ends vulnerable (0.26 < 0.35), which is the property the class doc
/// actually cares about.
pub fn dodge_iframes_active(time: f32) -> bool {
    (WARRIOR_DODGE_IFRAME_START..WARRIOR_DODGE_IFRAME_END).contains(&time)
}

// --- Inputs and outputs ---------------------------------------------------------------

/// One tick of player intent, already reduced from the platform.
///
/// The controller deliberately cannot see an [`ActionState`](dreamcoast_game::input::ActionState),
/// a key code or a gamepad: the integrator maps its bindings to *this*, so a test drives
/// the character with a struct literal and a rebind never reaches gameplay code.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WarriorInput {
    /// Desired move direction on XZ. Normalized or zero; a longer vector is clamped, a
    /// shorter one is honoured (analog sticks walk).
    pub move_dir: Vec2,
    /// Sprint held.
    pub sprint: bool,
    /// Attack **pressed this tick** (an edge, not a level) — the integrator passes
    /// `just_pressed`. Holding the button must not mash.
    pub attack_pressed: bool,
    /// Dodge pressed this tick, same edge rule.
    pub dodge_pressed: bool,
}

/// A candidate the swing can connect with, in **collision space**.
///
/// Positions are passed in rather than read from components for the reason
/// [`BodyCircle`](dreamcoast_game::combat::BodyCircle) documents: which side of
/// transform propagation a hit is resolved on must be visible at the call site.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    pub entity: Entity,
    /// Centre on the XZ plane, collision space.
    pub position: Vec2,
    /// Body radius in metres.
    pub radius: f32,
    pub team: Team,
}

/// Everything the controller needs from the world for one tick.
#[derive(Clone, Copy)]
pub struct WarriorCtx<'a, M: SolidMap + ?Sized> {
    /// The level, bound to its tile size.
    pub collision: GridCollision<'a, M>,
    /// Where the character is now, collision space. The controller is **not** the owner
    /// of the position: it is handed one and returns the next, so the integrator can
    /// keep the authoritative copy wherever it likes (an ECS transform, an
    /// interpolation pair).
    pub position: Vec2,
    /// Collision radius, metres ([`crate::collision::PLAYER_RADIUS`]).
    pub radius: f32,
    /// The character's own entity — the `attacker` of every [`DamageEvent`] produced.
    pub attacker: Entity,
    /// Who the swing may connect with. Hostility is filtered by [`Team`], so passing
    /// every combatant in the level is correct (just wasteful).
    pub targets: &'a [Target],
}

/// A clip to sample: what [`sandbox::scene::sample_clip`] wants.
///
/// [`looping`](Self::looping) is the `LoopMode` to pass — `Loop` for locomotion,
/// `Clamp` for the one-shots, which is what parks a finished attack on its last pose
/// instead of snapping it back to frame zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimSample {
    pub clip: WarriorClip,
    /// Playback time in seconds, already wrapped/clamped by the state machine.
    pub time: f32,
    /// Whether this clip loops — the `LoopMode` to sample it with.
    pub looping: bool,
}

/// A cross-fade in progress.
///
/// `alpha` is the weight of [`WarriorOutput::anim`], so the integrator's call is
/// `blend_poses(&sample(from), &sample(current), alpha)` — `from` starts at full weight
/// (`alpha == 0`) and hands over as the blend runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimBlend {
    /// The outgoing clip, still playing.
    pub from: AnimSample,
    /// Weight of the *incoming* (current) clip, in `[0, 1]`.
    pub alpha: f32,
}

/// What the character is doing this tick — the HUD's and the AI's readout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarriorState {
    Idle,
    Moving,
    Attacking {
        /// Combo step, 0-based.
        step: usize,
        phase: AttackPhase,
    },
    Dodging,
    /// In hitstun.
    Staggered,
    Dead,
}

/// The result of one [`WarriorController::tick`].
#[derive(Clone, Debug, PartialEq)]
pub struct WarriorOutput {
    /// The new position, collision space — already resolved against the level.
    pub position: Vec2,
    /// Unit facing on XZ.
    pub facing: Vec2,
    /// The same facing as a yaw about +Y, for the root node (`atan2(x, z)`).
    pub facing_radians: f32,
    /// The clip to play.
    pub anim: AnimSample,
    /// The clip to blend it with, if a transition is running.
    pub blend: Option<AnimBlend>,
    /// The invulnerability window to write onto the character's
    /// [`IFrames`] component — **assigned**, not
    /// [`refresh`](IFrames::refresh)ed: this is the authority, and a window that just
    /// closed must close in the ECS too.
    pub iframes: IFrames,
    /// Hits this tick's swing landed. The integrator sends them into its
    /// `Events<DamageEvent>` channel, scaling for crits/buffs on the way if it wants.
    pub hits: Vec<DamageEvent>,
    /// The mover removed motion this tick (walked into a wall).
    pub hit_wall: bool,
    pub state: WarriorState,
}

/// What [`WarriorController::take_damage`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DamageReport {
    /// Hit points actually removed.
    pub taken: f32,
    /// Hits ignored because of i-frames (or because the warrior was already dead).
    pub blocked: usize,
    /// The warrior died on this call — emitted once, like
    /// [`DeathEvent`](dreamcoast_game::combat::DeathEvent).
    pub died: bool,
}

/// One incoming hit.
///
/// `stagger` comes from the *attacker's* [`AttackSpec`], which the defender cannot read
/// — so the integrator carries it across with the [`DamageEvent`] rather than the
/// controller guessing a global hitstun.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IncomingHit {
    pub amount: f32,
    /// Push direction on XZ (for knockback later; carried, not applied).
    pub direction: Vec2,
    /// Hitstun in seconds — [`AttackSpec::stagger`].
    pub stagger: f32,
}

impl IncomingHit {
    /// The hit an [`AttackSpec`] inflicts.
    pub fn from_spec(spec: &AttackSpec, direction: Vec2) -> Self {
        Self {
            amount: spec.damage,
            direction,
            stagger: spec.stagger,
        }
    }
}

/// The character could not be built from the data it was given.
#[derive(Clone, Debug, PartialEq)]
pub enum WarriorError {
    /// The class definition is unusable.
    Class(CombatError),
    /// The animation graph is unusable.
    Anim(AnimError),
    /// The graph names a clip this controller cannot map to a rig clip — a rename on
    /// one side of the fence and not the other.
    UnknownClip(String),
}

impl std::fmt::Display for WarriorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Class(e) => write!(f, "warrior class: {e}"),
            Self::Anim(e) => write!(f, "warrior animation: {e}"),
            Self::UnknownClip(clip) => write!(
                f,
                "warrior animation graph references clip '{clip}', which has no rig clip"
            ),
        }
    }
}

impl std::error::Error for WarriorError {}

// --- The controller --------------------------------------------------------------------

/// A dodge roll in progress.
#[derive(Clone, Copy, Debug)]
struct DodgeState {
    /// Seconds into the roll.
    time: f32,
    /// Direction, fixed at the start — a roll is committed.
    dir: Vec2,
}

/// The player character.
///
/// Owns everything that persists between ticks: the class data, the clip-aligned chain,
/// the phase clock, the animation machine and its parameters, hit points, facing, and
/// the dodge/hitstun timers.
///
/// **Health lives here, not in the ECS.** The controller is the one place that has to
/// know about death (it stops accepting input, cancels the swing, locks the graph), so
/// making the ECS component the authority would mean asking the world a question every
/// tick and racing whoever wrote it. The integrator mirrors [`Self::health`] onto the
/// entity for the HUD and for anything that queries the world.
#[derive(Clone, Debug)]
pub struct WarriorController {
    class: ClassDef,
    /// The class's combo, re-timed against the clips — see [`clip_aligned_chain`].
    chain: ComboChain,
    attack: AttackState,
    machine: AnimMachine,
    params: Params,
    health: Health,
    facing: Vec2,
    dodge: Option<DodgeState>,
    /// Hitstun left, seconds.
    stagger: f32,
    dead: bool,
}

impl Default for WarriorController {
    fn default() -> Self {
        Self::new()
    }
}

impl WarriorController {
    /// The built-in warrior: `ClassDef::warrior()` plus the shipped graph.
    ///
    /// Panics only if those two embedded fixtures are themselves broken, which
    /// `the_builtin_warrior_builds` catches at test time rather than at a player's.
    pub fn new() -> Self {
        Self::with_class(ClassDef::warrior()).expect("the built-in warrior data is valid")
    }

    /// A warrior from a supplied [`ClassDef`] — the seam a class file (or a second
    /// class) arrives through. The class is validated; the graph is the shipped one.
    pub fn with_class(class: ClassDef) -> Result<Self, WarriorError> {
        class.validate().map_err(WarriorError::Class)?;
        let mut machine = AnimMachine::from_ron(WARRIOR_ANIM_GRAPH).map_err(WarriorError::Anim)?;
        // The graph's authored lengths are placeholders for a rig that did not exist
        // when it was written. Replace every one with the clip's real duration, and
        // fail loudly on a clip name neither side knows — a silent fallback here is a
        // combo that visually ends at the wrong time.
        let clips: Vec<String> = machine
            .def()
            .states
            .iter()
            .map(|s| s.clip.clone())
            .collect();
        for clip in clips {
            let mapped = WarriorClip::from_graph_clip(&clip)
                .ok_or_else(|| WarriorError::UnknownClip(clip.clone()))?;
            machine.set_clip_length(&clip, mapped.length());
        }
        let health = class.health();
        Ok(Self {
            chain: clip_aligned_chain(&class.combo),
            class,
            attack: AttackState::new(),
            machine,
            params: Params::new(),
            health,
            // +Z: the rigs' forward, so a warrior that has never moved faces the way its
            // rest pose points.
            facing: Vec2::new(0.0, 1.0),
            dodge: None,
            stagger: 0.0,
            dead: false,
        })
    }

    /// A warrior from class RON text (`assets/warrior.ron`, or a game's own file).
    pub fn from_class_ron(text: &str) -> Result<Self, WarriorError> {
        Self::with_class(ClassDef::from_ron(text).map_err(WarriorError::Class)?)
    }

    /// The class data this warrior was built from (unmodified — the *timing* the
    /// controller runs is [`Self::chain`]).
    pub fn class(&self) -> &ClassDef {
        &self.class
    }

    /// The clip-aligned combo actually being played.
    pub fn chain(&self) -> &ComboChain {
        &self.chain
    }

    /// Hit points.
    pub fn health(&self) -> Health {
        self.health
    }

    /// Whether the warrior is dead — a terminal state, see [`Self::tick`].
    pub fn is_dead(&self) -> bool {
        self.dead
    }

    /// Unit facing on XZ.
    pub fn facing(&self) -> Vec2 {
        self.facing
    }

    /// Facing as a yaw about +Y (`atan2(x, z)`), for the root node.
    pub fn facing_radians(&self) -> f32 {
        self.facing.x.atan2(self.facing.y)
    }

    /// Phase of the swing in progress.
    pub fn attack_phase(&self) -> AttackPhase {
        self.attack.phase()
    }

    /// Combo step being played (0 while idle).
    pub fn combo_step(&self) -> usize {
        self.attack.step()
    }

    /// Whether a swing is in progress.
    pub fn is_attacking(&self) -> bool {
        self.attack.is_attacking()
    }

    /// Whether a dodge roll is in progress.
    pub fn is_dodging(&self) -> bool {
        self.dodge.is_some()
    }

    /// Seconds into the current roll, if any.
    pub fn dodge_time(&self) -> Option<f32> {
        self.dodge.map(|d| d.time)
    }

    /// Hitstun left, seconds.
    pub fn stagger_left(&self) -> f32 {
        self.stagger
    }

    /// Whether damage is currently being ignored (the dodge's i-frame window).
    pub fn invulnerable(&self) -> bool {
        self.dodge.is_some_and(|d| dodge_iframes_active(d.time))
    }

    /// The state machine, for a debug overlay.
    pub fn anim_state(&self) -> &str {
        self.machine.current_state()
    }

    /// One simulation step.
    ///
    /// Order inside the tick, and why:
    ///
    /// 1. **Death short-circuits.** A dead warrior consumes no input and produces no
    ///    hits; only the graph keeps playing (the death clip has to finish).
    /// 2. **Hitstun decays**, then gates the rest: stagger refuses attack and dodge
    ///    input and locks movement.
    /// 3. **Dodge input**, before attack input: when both edges arrive in the same tick
    ///    the defensive option wins, because that is the one the player pressed for a
    ///    reason. A dodge may start from idle or from *recovery* (cancelling it) — never
    ///    from windup or active, which are the frames a swing is committed for.
    /// 4. **Attack input** → [`AttackState::request`], whose own rules decide whether
    ///    this is an opener, a buffered link, or ignored.
    /// 5. **The phase clock advances**, so every decision below reads *this* tick's
    ///    phase; a link fires here and re-aims the character.
    /// 6. **Movement**, from whatever the current state allows.
    /// 7. **Hits resolve**, at the position movement just produced — the swing connects
    ///    from where the body is, not from where it was.
    /// 8. **The graph ticks**, last, on parameters the whole tick has finished setting.
    pub fn tick<M: SolidMap + ?Sized>(
        &mut self,
        input: WarriorInput,
        dt: f32,
        ctx: WarriorCtx<'_, M>,
    ) -> WarriorOutput {
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        let move_dir = sanitize_dir(input.move_dir);

        // 1. Dead is a one-way door: no input, no movement, no swings. The graph is
        //    still driven (the death clip must play), with the triggers cleared so a
        //    wildcard interrupt cannot resurrect the pose.
        if self.dead {
            self.params.clear_triggers();
            self.params.set_flag(FLAG_MOVING, false);
            self.params.set_flag(FLAG_DEAD, true);
            self.machine.tick(dt, &mut self.params);
            return self.output(ctx.position, Vec::new(), false);
        }

        // 2. Hitstun.
        self.stagger = (self.stagger - dt).max(0.0);
        let staggered = self.stagger > 0.0;

        // 3. Dodge input.
        if input.dodge_pressed && !staggered && self.dodge.is_none() && self.can_start_dodge() {
            // Cancels the remainder of a recovery (`can_start_dodge` already refused
            // windup and active), so a committed swing is never rolled out of.
            self.attack.cancel();
            let dir = aim(move_dir, self.facing);
            self.facing = dir;
            self.dodge = Some(DodgeState { time: 0.0, dir });
            self.params.trigger(TRIGGER_DODGE);
        }

        // 4. Attack input.
        if input.attack_pressed && !staggered && self.dodge.is_none() {
            let opener = !self.attack.is_attacking();
            if self.attack.request(&self.chain) && opener {
                self.params.trigger(SWING_TRIGGERS[0]);
                self.facing = aim(move_dir, self.facing);
            }
        }

        // 5. The phase clock. A link starts the next swing *and* re-aims: the player
        //    steers a combo between its steps, which is the only steering a committed
        //    chain gives them.
        for event in self.attack.tick(&self.chain, dt) {
            if let AttackEvent::ComboAdvanced { step } = event {
                if let Some(trigger) = SWING_TRIGGERS.get(step) {
                    self.params.trigger(trigger);
                }
                self.facing = aim(move_dir, self.facing);
            }
        }

        // 6. Movement.
        let (delta, locomotion) = self.movement(move_dir, input.sprint, staggered, dt);
        let moved = ctx.collision.move_circle(ctx.position, ctx.radius, delta);
        if locomotion && move_dir != Vec2::ZERO {
            self.facing = turn_towards(self.facing, move_dir, TURN_RATE * dt);
        }

        // 7. Hit resolution, from the resolved position.
        let hits = self.resolve_hits(moved.pos, ctx.targets, ctx.attacker);

        // 8. The graph.
        self.params
            .set_flag(FLAG_MOVING, locomotion && move_dir != Vec2::ZERO);
        self.params.set_flag(FLAG_DEAD, false);
        self.machine.tick(dt, &mut self.params);

        self.output(moved.pos, hits, moved.hit_any())
    }

    /// Apply incoming damage.
    ///
    /// I-frames block a hit *entirely* — no health loss, no hitstun, no reaction —
    /// which is what makes the dodge worth its recovery tail. A hit that does land
    /// cancels whatever the warrior was doing (swing and roll alike): being hit is the
    /// punishment for having been committed.
    pub fn take_damage(&mut self, hits: impl IntoIterator<Item = IncomingHit>) -> DamageReport {
        let mut report = DamageReport::default();
        for hit in hits {
            if self.dead || self.invulnerable() {
                report.blocked += 1;
                continue;
            }
            let taken = self.health.damage(hit.amount);
            if taken <= 0.0 {
                continue;
            }
            report.taken += taken;
            if self.health.is_dead() {
                self.die();
                report.died = true;
                // Later hits in the same batch find a corpse and are dropped, exactly
                // like `apply_damage_events` — death happens once.
                continue;
            }
            self.attack.cancel();
            self.dodge = None;
            self.stagger = self.stagger.max(hit.stagger.max(0.0));
            // Raised only when the graph can act on it. The wildcard interrupts never
            // restart the state they target, so a second hit *during* the reaction
            // would leave the trigger latched (`Params` keeps an unconsumed trigger —
            // that is what buffers input) and fire it when the flinch ends: a ghost
            // reaction replayed over a player who is already out of hitstun, because
            // the clip (0.30 s) outlives the shorter staggers (0.18 s). The stagger is
            // still refreshed above; only the clip restart is dropped.
            if self.machine.current_state() != STATE_HIT {
                self.params.trigger(TRIGGER_HIT);
            }
        }
        report
    }

    /// Kill outright (a pit, a script). Idempotent.
    pub fn kill(&mut self) {
        if !self.dead {
            self.health.kill();
            self.die();
        }
    }

    /// Restore hit points (a potion). Refused once dead — resurrection is a game
    /// decision, and this controller has no path back out of its terminal state.
    pub fn heal(&mut self, amount: f32) -> f32 {
        if self.dead {
            0.0
        } else {
            self.health.heal(amount)
        }
    }

    /// Enter the terminal state.
    fn die(&mut self) {
        self.dead = true;
        self.attack.cancel();
        self.dodge = None;
        self.stagger = 0.0;
        // Anything still pending would fire from `death` through a wildcard edge.
        self.params.clear_triggers();
        self.params.set_flag(FLAG_MOVING, false);
        self.params.set_flag(FLAG_DEAD, true);
    }

    /// Whether a dodge may start right now: not mid-swing (windup/active are committed),
    /// not already rolling.
    fn can_start_dodge(&self) -> bool {
        matches!(
            self.attack.phase(),
            AttackPhase::Idle | AttackPhase::Recovery
        )
    }

    /// This tick's displacement, and whether it was ordinary locomotion (the only mode
    /// that steers and sets the `moving` flag).
    ///
    /// The lock table, in priority order:
    ///
    /// | state | movement |
    /// |---|---|
    /// | dodging | the roll's burst, in its committed direction |
    /// | staggered | none |
    /// | attack windup | none — the anticipation the player is supposed to read |
    /// | attack active | the swing's authored forward step, along the facing |
    /// | attack recovery, input buffered | none — the chain is already committed |
    /// | attack recovery, nothing buffered | free |
    /// | otherwise | free |
    fn movement(&mut self, move_dir: Vec2, sprint: bool, staggered: bool, dt: f32) -> (Vec2, bool) {
        if let Some(dodge) = self.dodge.as_mut() {
            // The roll's clock is the *class's* `dodge.duration`, and its speed is the
            // matching `dodge.speed()`, so the two always integrate to exactly
            // `dodge.distance` — a modded class that rolls further or slower still
            // covers what it says it does. `WARRIOR_DODGE_LEN` is the clip that draws
            // it; `the_roll_and_its_clip_share_one_clock` is the tripwire for the day
            // the two numbers stop agreeing (the i-frame window is authored in *clip*
            // time, so a divergence would put it in the wrong part of the roll).
            let duration = self.class.dodge.duration;
            // Clamp the last step to the roll's remaining time so the distance covered
            // is the class's, not "the class's, rounded up to a tick".
            let step = dt.min((duration - dodge.time).max(0.0));
            let delta = dodge.dir * self.class.dodge.speed() * step;
            dodge.time += step;
            if dodge.time >= duration {
                self.dodge = None;
            }
            return (delta, false);
        }
        if staggered {
            return (Vec2::ZERO, false);
        }
        match self.attack.phase() {
            AttackPhase::Windup => (Vec2::ZERO, false),
            AttackPhase::Active => {
                let step = WARRIOR_SWINGS
                    .get(self.attack.step())
                    .map_or(0.0, |s| s.step);
                let active = self
                    .attack
                    .spec(&self.chain)
                    .map_or(0.0, |spec| spec.active);
                let speed = if active > 0.0 { step / active } else { 0.0 };
                (self.facing * speed * dt, false)
            }
            AttackPhase::Recovery if self.attack.has_buffered_input() => (Vec2::ZERO, false),
            AttackPhase::Recovery | AttackPhase::Idle => {
                let speed = if sprint {
                    self.class.sprint_speed()
                } else {
                    self.class.move_speed
                };
                (move_dir * speed * dt, true)
            }
        }
    }

    /// Test the arc, once per target per swing, and turn the result into damage.
    fn resolve_hits(
        &mut self,
        origin: Vec2,
        targets: &[Target],
        attacker: Entity,
    ) -> Vec<DamageEvent> {
        if !self.attack.is_hit_window_open() || targets.is_empty() {
            return Vec::new();
        }
        let struck = self.attack.resolve_hits(
            &self.chain,
            origin,
            self.facing,
            Team::PLAYER,
            targets
                .iter()
                .map(|t| (t.entity, t.position, t.radius, t.team)),
        );
        let Some(spec) = self.attack.spec(&self.chain) else {
            return Vec::new();
        };
        struck
            .into_iter()
            .map(|entity| {
                // Push along attacker -> target where that is meaningful, so knockback
                // scatters a crowd instead of shoving it all one way; the facing is the
                // fallback for a target standing exactly on the attacker.
                let direction = targets
                    .iter()
                    .find(|t| t.entity == entity)
                    .map(|t| t.position - origin)
                    .filter(|d| d.length_squared() > 1e-12)
                    .map_or(self.facing, Vec2::normalize);
                DamageEvent::new(attacker, entity, spec.damage, direction)
            })
            .collect()
    }

    /// Assemble the tick's report.
    fn output(&self, position: Vec2, hits: Vec<DamageEvent>, hit_wall: bool) -> WarriorOutput {
        let (clip, time) = self.machine.current();
        let anim = self.sample(clip, time);
        let blend = self.machine.fade().map(|(clip, time, alpha)| AnimBlend {
            from: self.sample(clip, time),
            alpha,
        });
        WarriorOutput {
            position,
            facing: self.facing,
            facing_radians: self.facing_radians(),
            anim,
            blend,
            iframes: self.iframes(),
            hits,
            hit_wall,
            state: self.state(),
        }
    }

    /// The current invulnerability window as the component value to write.
    fn iframes(&self) -> IFrames {
        match self.dodge {
            Some(d) if dodge_iframes_active(d.time) => {
                IFrames::new(WARRIOR_DODGE_IFRAME_END - d.time)
            }
            _ => IFrames::default(),
        }
    }

    /// What the character is doing.
    fn state(&self) -> WarriorState {
        if self.dead {
            return WarriorState::Dead;
        }
        if self.dodge.is_some() {
            return WarriorState::Dodging;
        }
        if self.stagger > 0.0 {
            return WarriorState::Staggered;
        }
        match self.attack.phase() {
            AttackPhase::Idle => {
                if self.params.flag(FLAG_MOVING) {
                    WarriorState::Moving
                } else {
                    WarriorState::Idle
                }
            }
            phase => WarriorState::Attacking {
                step: self.attack.step(),
                phase,
            },
        }
    }

    /// A graph clip name + time as a sample request.
    fn sample(&self, clip: &str, time: f32) -> AnimSample {
        let clip = WarriorClip::from_graph_clip(clip)
            .expect("every graph clip was mapped when the controller was built");
        AnimSample {
            clip,
            time,
            looping: clip.looping(),
        }
    }
}

// --- Small geometry helpers -------------------------------------------------------------

/// A usable move direction: junk becomes zero, over-long input is clamped to unit, and
/// anything shorter is left alone (an analog stick may walk).
fn sanitize_dir(dir: Vec2) -> Vec2 {
    if !dir.is_finite() {
        return Vec2::ZERO;
    }
    if dir.length_squared() > 1.0 {
        dir.normalize()
    } else {
        dir
    }
}

/// Where a committed action points: the stick if it is held, otherwise the way the
/// character already faces. Never zero (the caller's facing is unit).
fn aim(move_dir: Vec2, facing: Vec2) -> Vec2 {
    if move_dir.length_squared() > 1e-12 {
        move_dir.normalize()
    } else {
        facing
    }
}

/// Rotate `from` towards `to` by at most `max_step` radians.
///
/// Angles rather than `Vec2::lerp`, so the turn rate is constant instead of easing out
/// (and a 180° reversal, whose lerp passes through zero length, is well defined).
fn turn_towards(from: Vec2, to: Vec2, max_step: f32) -> Vec2 {
    if to.length_squared() <= 1e-12 || max_step <= 0.0 {
        return from;
    }
    let current = from.y.atan2(from.x);
    let target = to.y.atan2(to.x);
    let delta = wrap_pi(target - current).clamp(-max_step, max_step);
    let angle = current + delta;
    Vec2::new(angle.cos(), angle.sin())
}

/// An angle folded into `(-PI, PI]`.
fn wrap_pi(angle: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    (angle + PI).rem_euclid(TAU) - PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::PLAYER_RADIUS;
    use crate::procgen::TILE_SIZE;
    use crate::rigs::WARRIOR_CLIPS;
    use dreamcoast_game::anim::{ANY_STATE, Condition};
    use sandbox::scene::World;

    const DT: f32 = 1.0 / 60.0;

    /// A string-art level, the same shape `physics`' own fixtures use: row index is
    /// `tz`, column index is `tx`, `#` is solid, and everything outside is solid.
    struct StringMap {
        rows: Vec<Vec<bool>>,
        width: i32,
        height: i32,
    }

    impl StringMap {
        fn new(rows: &[&str]) -> Self {
            let width = rows.iter().map(|r| r.len()).max().unwrap_or(0) as i32;
            let height = rows.len() as i32;
            let rows = rows
                .iter()
                .map(|r| {
                    let mut row: Vec<bool> = r.chars().map(|c| c == '#').collect();
                    row.resize(width as usize, true);
                    row
                })
                .collect();
            Self {
                rows,
                width,
                height,
            }
        }
    }

    impl SolidMap for StringMap {
        fn is_solid(&self, tx: i32, tz: i32) -> bool {
            if tx < 0 || tz < 0 || tx >= self.width || tz >= self.height {
                return true;
            }
            self.rows[tz as usize][tx as usize]
        }
    }

    /// A 10x10-tile room (20 m square) with a pillar off-centre and a wall stub, so a
    /// dodge or a sprint actually meets geometry.
    fn arena() -> StringMap {
        StringMap::new(&[
            "############",
            "#..........#",
            "#..........#",
            "#....#.....#",
            "#..........#",
            "#..........#",
            "#.###......#",
            "#..........#",
            "#..........#",
            "#..........#",
            "#..........#",
            "############",
        ])
    }

    /// Middle of the arena, well clear of every wall.
    fn centre() -> Vec2 {
        Vec2::splat(6.0 * TILE_SIZE + TILE_SIZE * 0.5)
    }

    fn ctx<'a>(
        map: &'a StringMap,
        position: Vec2,
        attacker: Entity,
        targets: &'a [Target],
    ) -> WarriorCtx<'a, StringMap> {
        WarriorCtx {
            collision: GridCollision::new(map, TILE_SIZE),
            position,
            radius: PLAYER_RADIUS,
            attacker,
            targets,
        }
    }

    /// An entity to hang events off.
    fn entity() -> Entity {
        World::new().spawn()
    }

    /// Drive `ticks` steps of identical input, returning the last output. Position is
    /// threaded from each output into the next tick, the way the integrator does it.
    fn run(
        w: &mut WarriorController,
        map: &StringMap,
        pos: &mut Vec2,
        input: WarriorInput,
        ticks: usize,
    ) -> WarriorOutput {
        let me = entity();
        let mut out = None;
        for _ in 0..ticks {
            let o = w.tick(input, DT, ctx(map, *pos, me, &[]));
            *pos = o.position;
            out = Some(o);
        }
        out.expect("at least one tick")
    }

    /// Press attack once (an edge), then release.
    fn press_attack(move_dir: Vec2) -> WarriorInput {
        WarriorInput {
            move_dir,
            attack_pressed: true,
            ..WarriorInput::default()
        }
    }

    // -- construction and data reconciliation ------------------------------------------

    #[test]
    fn the_builtin_warrior_builds() {
        let w = WarriorController::new();
        assert_eq!(w.class().name, "warrior");
        assert_eq!(w.health(), Health::new(100.0));
        assert!(!w.is_dead());
        assert_eq!(w.anim_state(), "idle");
        assert_eq!(w.facing(), Vec2::new(0.0, 1.0));
        assert_eq!(w.facing_radians(), 0.0, "+Z is a yaw of zero");
        // And through the class-file seam.
        let from_file =
            WarriorController::from_class_ron(dreamcoast_game::combat::WARRIOR_CLASS_RON).unwrap();
        assert_eq!(from_file.chain(), w.chain());
    }

    /// **The alignment that matters**: for every swing, the combat hit window opens at
    /// the instant the clip shows the blade crossing the arc, and the swing lasts
    /// exactly as long as the clip that draws it.
    #[test]
    fn every_swing_is_timed_by_its_clip() {
        let w = WarriorController::new();
        let baseline = ClassDef::warrior();
        let expected = [
            (WARRIOR_ATTACK1_HIT_TIME, WARRIOR_ATTACK1_LEN),
            (WARRIOR_ATTACK2_HIT_TIME, WARRIOR_ATTACK2_LEN),
            (WARRIOR_ATTACK3_HIT_TIME, WARRIOR_ATTACK3_LEN),
        ];
        assert_eq!(w.chain().len(), 3);
        for (i, (hit_time, length)) in expected.into_iter().enumerate() {
            let step = w.chain().get(i).unwrap();
            let authored = baseline.combo.get(i).unwrap();
            assert_eq!(
                step.windup, hit_time,
                "step {i}: windup ends at the hit pose"
            );
            assert!(
                (step.duration() - length).abs() < 1e-6,
                "step {i}: swing lasts {} but the clip is {length}",
                step.duration()
            );
            assert_eq!(
                step.active, authored.active,
                "step {i}: hit window preserved"
            );
            assert!(
                step.active > DT,
                "step {i}: a hit window must outlast a fixed step"
            );
            assert!(step.recovery > 0.0, "step {i}: recovery must remain");
            // Everything that is not timing comes straight from the class.
            assert_eq!(step.name, authored.name);
            assert_eq!(step.damage, authored.damage);
            assert_eq!(step.range, authored.range);
            assert_eq!(step.half_angle_rad, authored.half_angle_rad);
            assert_eq!(step.stagger, authored.stagger);
        }
        // The derived chain is still a legal class — the validator the runtime relies on
        // has not been talked out of anything.
        let derived = ClassDef {
            combo: w.chain().clone(),
            ..baseline
        };
        derived.validate().unwrap();
    }

    /// The re-timing is a *rule*, not a table: a chain longer than the three authored
    /// clips passes through untouched.
    #[test]
    fn steps_without_a_clip_keep_their_authored_timing() {
        let mut class = ClassDef::warrior();
        let extra = AttackSpec {
            name: "spin".to_string(),
            windup: 0.5,
            active: 0.1,
            recovery: 0.9,
            ..class.combo.get(0).unwrap().clone()
        };
        class.combo.0.push(extra.clone());
        let w = WarriorController::with_class(class).unwrap();
        assert_eq!(w.chain().get(3).unwrap(), &extra);
    }

    #[test]
    fn a_broken_class_is_refused() {
        let mut class = ClassDef::warrior();
        class.max_health = 0.0;
        assert!(matches!(
            WarriorController::with_class(class).unwrap_err(),
            WarriorError::Class(_)
        ));
        assert!(matches!(
            WarriorController::from_class_ron("not ron").unwrap_err(),
            WarriorError::Anim(_) | WarriorError::Class(_)
        ));
        // The errors say which half broke.
        assert!(
            WarriorError::UnknownClip("ghost".into())
                .to_string()
                .contains("ghost")
        );
    }

    /// The clip map is the fence between two files that name the same eight animations
    /// differently. Both sides must be complete, and the rig names must be the rig's.
    #[test]
    fn clip_names_bridge_the_graph_and_the_rig() {
        let mut rig_names: Vec<&str> = WarriorClip::ALL.iter().map(|c| c.rig_clip()).collect();
        rig_names.sort_unstable();
        let mut authored = WARRIOR_CLIPS.to_vec();
        authored.sort_unstable();
        assert_eq!(
            rig_names, authored,
            "every rig clip has exactly one enum arm"
        );

        let machine = AnimMachine::from_ron(WARRIOR_ANIM_GRAPH).unwrap();
        for state in &machine.def().states {
            let clip = WarriorClip::from_graph_clip(&state.clip)
                .unwrap_or_else(|| panic!("graph clip '{}' is unmapped", state.clip));
            assert_eq!(clip.graph_clip(), state.clip);
            assert_eq!(
                clip.looping(),
                state.looping,
                "{}: loop flag disagrees with the graph",
                state.clip
            );
        }
    }

    /// Construction replaces the graph's placeholder lengths with the rig's real ones —
    /// otherwise the state machine would leave `slash_left` 0.2 s after the swing ended.
    #[test]
    fn construction_installs_the_rig_clip_lengths() {
        let w = WarriorController::new();
        for clip in WarriorClip::ALL {
            assert_eq!(
                w.machine.clip_length(clip.graph_clip()),
                clip.length(),
                "{}",
                clip.graph_clip()
            );
        }
        // ...and those are genuinely different numbers from the fixture's placeholders,
        // so the test above is not vacuous.
        let graph = AnimMachine::from_ron(WARRIOR_ANIM_GRAPH).unwrap();
        assert_ne!(
            graph.clip_length("slash_left"),
            WarriorClip::Attack1.length()
        );
    }

    /// Every flag and trigger the controller speaks must be a word the graph knows.
    /// This is the drift alarm for the two files it cannot edit.
    #[test]
    fn the_controllers_vocabulary_exists_in_the_graph() {
        let graph = AnimMachine::from_ron(WARRIOR_ANIM_GRAPH).unwrap();
        let mut flags = Vec::new();
        let mut triggers = Vec::new();
        for t in &graph.def().transitions {
            match &t.condition {
                Condition::Flag(n) | Condition::NotFlag(n) => flags.push(n.as_str()),
                Condition::Trigger(n) => triggers.push(n.as_str()),
                Condition::StateDone => {}
            }
        }
        for flag in [FLAG_MOVING, FLAG_DEAD] {
            assert!(flags.contains(&flag), "graph has no flag '{flag}'");
        }
        for trigger in SWING_TRIGGERS.iter().chain([&TRIGGER_DODGE, &TRIGGER_HIT]) {
            assert!(
                triggers.contains(trigger),
                "graph has no trigger '{trigger}'"
            );
        }
        // The two adaptations documented in the module header: death is a flag (there is
        // no "die" trigger to fire), and there is no single "attack" trigger.
        assert!(!triggers.contains(&"die"));
        assert!(!triggers.contains(&"attack"));
        // The one state name the controller hard-codes.
        assert!(graph.def().state(STATE_HIT).is_some());
        // ...and the reason it has to: `hit` is the only wildcard target the controller
        // can already be standing in when it wants to re-fire that trigger. Every other
        // interrupt is raised from a state that is provably not its own target.
        let wildcard_targets: Vec<&str> = graph
            .def()
            .transitions
            .iter()
            .filter(|t| t.from.eq_ignore_ascii_case(ANY_STATE))
            .map(|t| t.to.as_str())
            .collect();
        assert!(
            wildcard_targets.contains(&STATE_HIT),
            "{wildcard_targets:?}"
        );
    }

    /// The roll's clock and the clip that draws it are the same 0.35 s.
    ///
    /// The controller runs the roll on `ClassDef::dodge.duration` (so the distance is
    /// always the class's) but reads its i-frames out of the *clip*
    /// (`WARRIOR_DODGE_IFRAME_*`). Those two only describe the same motion while the
    /// numbers agree — this is the alarm for the day one of them moves.
    #[test]
    fn the_roll_and_its_clip_share_one_clock() {
        let dodge = ClassDef::warrior().dodge;
        assert_eq!(dodge.duration, WARRIOR_DODGE_LEN);
        assert!(WARRIOR_DODGE_IFRAME_END < dodge.duration, "ends vulnerable");
        const { assert!(WARRIOR_DODGE_IFRAME_START > 0.0) }; // starts vulnerable
    }

    // -- the combo ----------------------------------------------------------------------

    /// A buffered input links **on the tick the hit window closes** — no recovery is
    /// played. The tick counts are the clip's: windup 0.28 s = 17 steps, active 0.15 s =
    /// 9 more, so the second swing starts on step 26 and not before.
    #[test]
    fn a_buffered_input_links_the_instant_the_window_closes() {
        let map = arena();
        let mut pos = centre();
        let mut w = WarriorController::new();
        let idle = WarriorInput::default();

        run(&mut w, &map, &mut pos, press_attack(Vec2::ZERO), 1);
        assert_eq!(w.combo_step(), 0);
        assert_eq!(w.attack_phase(), AttackPhase::Windup);

        // Ticks 2..=17 finish the windup; the window opens on 17.
        run(&mut w, &map, &mut pos, idle, 16);
        assert_eq!(w.attack_phase(), AttackPhase::Active, "17 steps = 0.2833 s");

        // Press during the active window: buffered, but nothing changes yet.
        run(&mut w, &map, &mut pos, press_attack(Vec2::ZERO), 1);
        assert_eq!(w.combo_step(), 0, "the link waits for the window to close");

        // Steps 19..=25 play the rest of the window out.
        run(&mut w, &map, &mut pos, idle, 7);
        assert_eq!(w.attack_phase(), AttackPhase::Active, "25 steps = 0.4167 s");
        assert_eq!(w.combo_step(), 0);

        // Step 26 crosses 0.43 s = windup + active: the window closes and the buffered
        // input takes over in the same tick, skipping recovery entirely.
        run(&mut w, &map, &mut pos, idle, 1);
        assert_eq!(w.combo_step(), 1);
        assert_eq!(w.attack_phase(), AttackPhase::Windup);
        assert_eq!(w.anim_state(), "slash_right", "and the clip followed");
    }

    /// Mashing through the whole chain plays all three clips in order and then stops:
    /// the finisher has nothing to link into.
    #[test]
    fn the_chain_plays_three_clips_and_ends() {
        let map = arena();
        let mut pos = centre();
        let mut w = WarriorController::new();
        let mut states = vec![w.anim_state().to_string()];
        for _ in 0..200 {
            w.tick(press_attack(Vec2::ZERO), DT, ctx(&map, pos, entity(), &[]));
            pos = w
                .tick(WarriorInput::default(), DT, ctx(&map, pos, entity(), &[]))
                .position;
            let state = w.anim_state().to_string();
            if states.last() != Some(&state) {
                states.push(state);
            }
        }
        let visited: Vec<&str> = states.iter().map(String::as_str).collect();
        assert!(
            visited.starts_with(&["idle", "slash_left", "slash_right", "overhead"]),
            "{visited:?}"
        );
    }

    /// An attack input during windup is refused (the read window cannot be skipped), and
    /// a fresh press after the chain expires opens at step 0 again.
    #[test]
    fn windup_refuses_a_link_and_an_expired_chain_restarts() {
        let map = arena();
        let mut pos = centre();
        let mut w = WarriorController::new();
        run(&mut w, &map, &mut pos, press_attack(Vec2::ZERO), 5);
        run(&mut w, &map, &mut pos, press_attack(Vec2::ZERO), 1);
        assert_eq!(w.combo_step(), 0);
        assert_eq!(w.attack_phase(), AttackPhase::Windup);

        run(&mut w, &map, &mut pos, WarriorInput::default(), 60);
        assert!(!w.is_attacking(), "the chain expired");
        run(&mut w, &map, &mut pos, press_attack(Vec2::ZERO), 1);
        assert_eq!(w.combo_step(), 0);
    }

    // -- movement ------------------------------------------------------------------------

    /// Windup and active lock the stick; the swing's authored forward step is the only
    /// motion, and it is along the **facing**, not the input. Recovery frees the
    /// character again — unless a link is already buffered.
    #[test]
    fn a_swing_locks_movement_except_for_its_authored_step() {
        let map = arena();
        let start = centre();
        let mut pos = start;
        let mut w = WarriorController::new();
        // Attack while pushing +X: the swing snaps to +X, so "forward" and "input" are
        // the same axis and the test measures magnitudes, not directions.
        let push = WarriorInput {
            move_dir: Vec2::X,
            ..WarriorInput::default()
        };
        run(&mut w, &map, &mut pos, press_attack(Vec2::X), 1);
        assert_eq!(w.facing(), Vec2::X);

        // Windup: 16 more steps to 0.2833 s, and the character has not moved a micron.
        run(&mut w, &map, &mut pos, push, 15);
        assert_eq!(w.attack_phase(), AttackPhase::Windup);
        assert_eq!(pos, start, "windup is a full stop");

        // Active: exactly the 0.4 m the swing authors, over the 9 steps of the window.
        run(&mut w, &map, &mut pos, push, 9);
        assert_eq!(w.attack_phase(), AttackPhase::Active);
        let stepped = pos - start;
        assert!(
            (stepped.x - WARRIOR_SWINGS[0].step).abs() < 1e-3 && stepped.y.abs() < 1e-6,
            "stepped {stepped:?}"
        );

        // The step is the swing's, not the stick's: pushing the other way during the
        // window still carries the body forward.
        let mut back = WarriorController::new();
        let mut bpos = start;
        run(&mut back, &map, &mut bpos, press_attack(Vec2::X), 1);
        run(
            &mut back,
            &map,
            &mut bpos,
            WarriorInput {
                move_dir: -Vec2::X,
                ..WarriorInput::default()
            },
            24,
        );
        assert!(bpos.x > start.x, "the swing carries, the stick does not");

        // Recovery with nothing buffered: free movement, at the class's walk speed.
        let before = pos;
        run(&mut w, &map, &mut pos, push, 2);
        assert_eq!(w.attack_phase(), AttackPhase::Recovery);
        assert!(
            ((pos.x - before.x) - 2.0 * DT * ClassDef::warrior().move_speed).abs() < 1e-3,
            "recovery walks at walk speed"
        );
    }

    /// A recovery that is *already linking* stays locked: the next swing is committed,
    /// so the character does not get a free step out of it.
    #[test]
    fn a_buffered_link_keeps_recovery_locked() {
        let map = arena();
        let mut pos = centre();
        let mut w = WarriorController::new();
        run(&mut w, &map, &mut pos, press_attack(Vec2::X), 1);
        run(&mut w, &map, &mut pos, WarriorInput::default(), 16); // -> active
        assert_eq!(w.attack_phase(), AttackPhase::Active);
        // Buffer a link, then let the window close *without* advancing far enough for
        // the link to fire — impossible: the link fires the same tick. So instead check
        // the state directly with a chain whose recovery is longer than a tick.
        run(&mut w, &map, &mut pos, press_attack(Vec2::X), 1);
        assert!(w.attack.has_buffered_input());
        let before = pos;
        // While buffered and still in the window, the stick is ignored (active lock),
        // and the moment the link fires the next windup locks it too.
        run(
            &mut w,
            &map,
            &mut pos,
            WarriorInput {
                move_dir: Vec2::X,
                ..WarriorInput::default()
            },
            9,
        );
        assert_eq!(w.combo_step(), 1);
        assert_eq!(w.attack_phase(), AttackPhase::Windup);
        let travelled = pos.x - before.x;
        assert!(
            travelled < WARRIOR_SWINGS[0].step + 1e-3,
            "only the swing's own step, never the stick's: {travelled}"
        );
    }

    /// Free running turns smoothly at [`TURN_RATE`]; a committed action snaps.
    #[test]
    fn facing_turns_smoothly_but_snaps_on_attack_and_dodge() {
        let map = arena();
        let mut pos = centre();
        let mut w = WarriorController::new();
        // From +Z towards +X is a quarter turn: at 12 rad/s that is ~0.13 s, so one tick
        // must cover a fraction of it and not the whole thing.
        let out = run(
            &mut w,
            &map,
            &mut pos,
            WarriorInput {
                move_dir: Vec2::X,
                ..WarriorInput::default()
            },
            1,
        );
        let turned = wrap_pi(out.facing_radians);
        assert!(
            (turned - TURN_RATE * DT).abs() < 1e-4,
            "one tick of turn = {turned}"
        );
        assert!(turned < std::f32::consts::FRAC_PI_2);

        // An attack aimed the other way snaps in one tick.
        run(&mut w, &map, &mut pos, press_attack(-Vec2::Y), 1);
        assert_eq!(w.facing(), -Vec2::Y);
        // Compared as an *angle*: `-Vec2::Y` is `(-0.0, -1.0)`, and `atan2` reports the
        // negated-zero half-turn as -PI. Same yaw, opposite sign — so the difference is
        // wrapped rather than subtracted, which is what any consumer of a yaw must do.
        assert!(wrap_pi(w.facing_radians() - std::f32::consts::PI).abs() < 1e-5);

        // So does a dodge, and with no stick it keeps the facing it had.
        let mut d = WarriorController::new();
        let mut dpos = centre();
        run(
            &mut d,
            &map,
            &mut dpos,
            WarriorInput {
                move_dir: Vec2::X,
                dodge_pressed: true,
                ..WarriorInput::default()
            },
            1,
        );
        assert_eq!(d.facing(), Vec2::X);
    }

    // -- the dodge -------------------------------------------------------------------------

    /// The i-frame window is the clip's crouch, to the millisecond.
    #[test]
    fn the_iframe_window_is_the_clips_crouch() {
        // The predicate, swept finely across the whole roll.
        let mut t = 0.0f32;
        while t <= WARRIOR_DODGE_LEN {
            assert_eq!(
                dodge_iframes_active(t),
                (WARRIOR_DODGE_IFRAME_START..WARRIOR_DODGE_IFRAME_END).contains(&t),
                "t = {t}"
            );
            t += 0.001;
        }
        assert!(
            dodge_iframes_active(WARRIOR_DODGE_IFRAME_START),
            "inclusive start"
        );
        assert!(
            !dodge_iframes_active(WARRIOR_DODGE_IFRAME_END),
            "exclusive end"
        );
        assert!(!dodge_iframes_active(0.0), "the roll starts vulnerable");
        assert!(
            !dodge_iframes_active(WARRIOR_DODGE_LEN),
            "and ends vulnerable — the class's whole point"
        );

        // And the same window as the tick loop sees it: on within one step of 0.07, off
        // within one step of 0.26.
        let map = arena();
        let mut pos = centre();
        let mut w = WarriorController::new();
        let dodge = WarriorInput {
            dodge_pressed: true,
            ..WarriorInput::default()
        };
        let mut first = None;
        let mut last = None;
        for step in 0..30 {
            let input = if step == 0 {
                dodge
            } else {
                WarriorInput::default()
            };
            let out = w.tick(input, DT, ctx(&map, pos, entity(), &[]));
            pos = out.position;
            let t = (step + 1) as f32 * DT;
            if out.iframes.is_active() {
                first.get_or_insert(t);
                last = Some(t);
                assert!(w.invulnerable());
                assert!(
                    (out.iframes.remaining - (WARRIOR_DODGE_IFRAME_END - t)).abs() < 1e-5,
                    "the window reports its own remainder"
                );
            }
        }
        let first = first.expect("the roll was invulnerable at some point");
        let last = last.unwrap();
        assert!(
            (first - WARRIOR_DODGE_IFRAME_START).abs() <= DT,
            "i-frames opened at {first}, clip says {WARRIOR_DODGE_IFRAME_START}"
        );
        assert!(first >= WARRIOR_DODGE_IFRAME_START, "never early");
        assert!(
            last < WARRIOR_DODGE_IFRAME_END && WARRIOR_DODGE_IFRAME_END - last <= DT,
            "i-frames closed at {last}, clip says {WARRIOR_DODGE_IFRAME_END}"
        );
    }

    /// The roll covers the class's distance in the class's time, and cannot be
    /// re-triggered until it is over.
    #[test]
    fn the_roll_covers_the_class_distance_once() {
        let map = arena();
        let start = centre();
        let mut pos = start;
        let mut w = WarriorController::new();
        let dodge = WarriorInput {
            move_dir: Vec2::X,
            dodge_pressed: true,
            ..WarriorInput::default()
        };
        run(&mut w, &map, &mut pos, dodge, 1);
        assert!(w.is_dodging());
        // Mashing dodge during the roll changes nothing.
        run(&mut w, &map, &mut pos, dodge, 20);
        assert!(!w.is_dodging(), "0.35 s = 21 steps, the roll is over");
        let travelled = (pos - start).length();
        assert!(
            (travelled - ClassDef::warrior().dodge.distance).abs() < 1e-2,
            "rolled {travelled} m"
        );
        assert_eq!(w.anim_state(), "dodge");
    }

    /// A roll may cancel recovery — and only recovery.
    #[test]
    fn a_dodge_cancels_recovery_but_never_a_live_swing() {
        let map = arena();
        let dodge = WarriorInput {
            dodge_pressed: true,
            ..WarriorInput::default()
        };
        // Windup: refused.
        let mut pos = centre();
        let mut w = WarriorController::new();
        run(&mut w, &map, &mut pos, press_attack(Vec2::ZERO), 5);
        assert_eq!(w.attack_phase(), AttackPhase::Windup);
        run(&mut w, &map, &mut pos, dodge, 1);
        assert!(!w.is_dodging(), "windup is committed");
        assert!(w.is_attacking());

        // Active: refused.
        run(&mut w, &map, &mut pos, WarriorInput::default(), 13);
        assert_eq!(w.attack_phase(), AttackPhase::Active);
        run(&mut w, &map, &mut pos, dodge, 1);
        assert!(!w.is_dodging(), "the hit window is committed");
        assert!(w.is_attacking());

        // Recovery: accepted, and the swing is dropped.
        run(&mut w, &map, &mut pos, WarriorInput::default(), 9);
        assert_eq!(w.attack_phase(), AttackPhase::Recovery);
        run(&mut w, &map, &mut pos, dodge, 1);
        assert!(w.is_dodging(), "recovery is cancellable");
        assert!(!w.is_attacking());
        // A cancelled chain restarts at the opener next time.
        assert_eq!(w.combo_step(), 0);
    }

    /// An attack input during a roll is ignored: the roll is committed.
    #[test]
    fn a_roll_refuses_attack_input() {
        let map = arena();
        let mut pos = centre();
        let mut w = WarriorController::new();
        run(
            &mut w,
            &map,
            &mut pos,
            WarriorInput {
                dodge_pressed: true,
                ..WarriorInput::default()
            },
            1,
        );
        run(&mut w, &map, &mut pos, press_attack(Vec2::ZERO), 5);
        assert!(!w.is_attacking());
        assert!(w.is_dodging());
    }

    // -- damage, hitstun and death ------------------------------------------------------

    #[test]
    fn a_hit_staggers_and_cancels_the_swing() {
        let map = arena();
        let mut pos = centre();
        let mut w = WarriorController::new();
        run(&mut w, &map, &mut pos, press_attack(Vec2::ZERO), 10);
        assert!(w.is_attacking());

        let spec = ClassDef::warrior().combo.get(2).unwrap().clone();
        let report = w.take_damage([IncomingHit::from_spec(&spec, Vec2::X)]);
        assert_eq!(report.taken, spec.damage);
        assert!(!report.died);
        assert_eq!(w.health().current, 100.0 - spec.damage);
        assert!(!w.is_attacking(), "being hit costs the swing");
        assert!((w.stagger_left() - spec.stagger).abs() < 1e-6);

        // Locked and playing the reaction while it lasts.
        let held = WarriorInput {
            move_dir: Vec2::X,
            ..WarriorInput::default()
        };
        let before = pos;
        let out = run(&mut w, &map, &mut pos, held, 3);
        assert_eq!(out.state, WarriorState::Staggered);
        assert_eq!(pos, before, "hitstun is a full stop");
        assert_eq!(w.anim_state(), "hit");
        // ...and free again afterwards.
        run(&mut w, &map, &mut pos, held, 20);
        assert_eq!(w.stagger_left(), 0.0);
        assert!(pos.x > before.x);
    }

    /// A second hit *during* the reaction refreshes the hitstun without queueing a
    /// second flinch.
    ///
    /// The regression: the reaction clip (0.30 s) outlives the shorter staggers
    /// (0.18 s), the graph's wildcard interrupts refuse to restart the state they are
    /// already in, and an unconsumed trigger survives the tick — so re-triggering `hit`
    /// mid-reaction used to latch and fire when the flinch ended, replaying it over a
    /// player who was already free to move.
    #[test]
    fn a_second_hit_during_the_reaction_does_not_queue_a_ghost_flinch() {
        let map = arena();
        let mut pos = centre();
        let mut w = WarriorController::new();
        let jab = IncomingHit {
            amount: 5.0,
            direction: Vec2::X,
            stagger: 0.18,
        };
        let idle = WarriorInput::default();
        let held = WarriorInput {
            move_dir: Vec2::X,
            ..WarriorInput::default()
        };

        w.take_damage([jab]);
        run(&mut w, &map, &mut pos, idle, 1);
        assert_eq!(w.anim_state(), STATE_HIT);

        // A second jab most of the way through the 0.30 s clip: hitstun is refreshed...
        run(&mut w, &map, &mut pos, idle, 16); // ~0.28 s in
        assert_eq!(w.anim_state(), STATE_HIT, "still flinching");
        w.take_damage([jab]);
        assert!(
            (w.stagger_left() - jab.stagger).abs() < 1e-6,
            "stun refreshed"
        );

        // ...and once the stagger runs out the character moves, without the clip having
        // been replayed underneath it.
        let before = pos;
        run(&mut w, &map, &mut pos, held, 20);
        assert_eq!(w.stagger_left(), 0.0);
        assert!(pos.x > before.x, "free to move");
        assert_ne!(w.anim_state(), STATE_HIT, "no ghost reaction");
    }

    /// I-frames drop the hit completely: no health, no hitstun, no reaction.
    #[test]
    fn iframes_block_a_hit_outright() {
        let map = arena();
        let mut pos = centre();
        let mut w = WarriorController::new();
        run(
            &mut w,
            &map,
            &mut pos,
            WarriorInput {
                dodge_pressed: true,
                ..WarriorInput::default()
            },
            1,
        );
        // Into the i-frame window (0.07 s = 5 steps).
        run(&mut w, &map, &mut pos, WarriorInput::default(), 5);
        assert!(w.invulnerable());
        let report = w.take_damage([IncomingHit {
            amount: 40.0,
            direction: Vec2::X,
            stagger: 0.5,
        }]);
        assert_eq!(report.taken, 0.0);
        assert_eq!(report.blocked, 1);
        assert_eq!(w.health().current, 100.0);
        assert_eq!(w.stagger_left(), 0.0);
        assert!(
            w.is_dodging(),
            "the roll is not interrupted by what it dodged"
        );
    }

    /// Death is terminal: input does nothing for ever after, and the death clip is what
    /// plays.
    #[test]
    fn death_is_a_one_way_door() {
        let map = arena();
        let mut pos = centre();
        let mut w = WarriorController::new();
        let big = IncomingHit {
            amount: 250.0,
            direction: Vec2::X,
            stagger: 0.4,
        };
        let report = w.take_damage([big, big]);
        assert!(report.died);
        assert_eq!(report.taken, 100.0, "overkill takes what existed");
        assert_eq!(report.blocked, 1, "the second hit found a corpse");
        assert!(w.is_dead());

        // Everything at once, for five seconds.
        let everything = WarriorInput {
            move_dir: Vec2::X,
            sprint: true,
            attack_pressed: true,
            dodge_pressed: true,
        };
        let corpse = pos;
        let target = Target {
            entity: entity(),
            position: pos + Vec2::X,
            radius: 0.5,
            team: Team::ENEMY,
        };
        let targets = [target];
        for _ in 0..300 {
            let out = w.tick(everything, DT, ctx(&map, pos, entity(), &targets));
            pos = out.position;
            assert_eq!(pos, corpse, "a corpse does not move");
            assert!(out.hits.is_empty(), "a corpse does not swing");
            assert_eq!(out.state, WarriorState::Dead);
            assert!(!out.iframes.is_active());
        }
        assert_eq!(w.anim_state(), "death");
        assert_eq!(w.health().current, 0.0);
        assert!(!w.is_attacking() && !w.is_dodging());
        assert_eq!(w.heal(50.0), 0.0, "no resurrection by potion");
        assert_eq!(w.take_damage([big]).taken, 0.0);

        // `kill` is the same door from the other side, and is idempotent.
        let mut k = WarriorController::new();
        k.kill();
        k.kill();
        assert!(k.is_dead() && k.health().is_dead());
    }

    // -- the fight ---------------------------------------------------------------------

    /// A 300-step scripted fight against two dummies: a three-hit chain on the one in
    /// front, nothing at all for the one behind, and the mover's invariants held for the
    /// whole run.
    #[test]
    fn a_scripted_fight_lands_the_chain_and_nothing_else() {
        let map = arena();
        let mut world = World::new();
        let me = world.spawn();
        let front = world.spawn();
        let behind = world.spawn();
        let start = centre();
        let mut pos = start;

        // Both dummies two metres out, on the attack axis. The one behind is at 180°,
        // outside even the finisher's 130° arc.
        let targets = [
            Target {
                entity: front,
                position: start + Vec2::new(2.0, 0.0),
                radius: 0.45,
                team: Team::ENEMY,
            },
            Target {
                entity: behind,
                position: start - Vec2::new(2.0, 0.0),
                radius: 0.45,
                team: Team::ENEMY,
            },
        ];

        let mut w = WarriorController::new();
        let mut damage: Vec<(Entity, f32, usize)> = Vec::new();
        for step in 0..300usize {
            // The script: open the combo facing +X, link once per hit window (never
            // after the finisher starts), then walk a lap and roll.
            let mut input = WarriorInput::default();
            if step == 0 {
                input.move_dir = Vec2::X;
                input.attack_pressed = true;
            } else if w.attack_phase() == AttackPhase::Active && w.combo_step() < 2 {
                input.attack_pressed = true;
            } else if (60..200).contains(&step) {
                input.move_dir = Vec2::new(0.0, -1.0);
                input.sprint = step % 3 == 0;
            } else if step == 200 {
                input.move_dir = Vec2::new(-1.0, 0.0);
                input.dodge_pressed = true;
            } else if step > 200 {
                input.move_dir = Vec2::new(1.0, 1.0).normalize();
            }

            let out = w.tick(input, DT, ctx(&map, pos, me, &targets));
            for hit in &out.hits {
                assert_eq!(hit.attacker, me);
                damage.push((hit.target, hit.amount, w.combo_step()));
            }
            pos = out.position;

            // The mover's contract, every single step.
            assert!(pos.is_finite(), "step {step}: non-finite position");
            assert!(out.facing.is_finite() && out.facing.length() > 0.9);
            assert!(
                !GridCollision::new(&map, TILE_SIZE).circle_overlaps(pos, PLAYER_RADIUS),
                "step {step}: inside solid at {pos:?}"
            );
        }

        // Exactly three hits, one per swing, all on the target in front.
        let landed: Vec<f32> = damage
            .iter()
            .filter(|(e, ..)| *e == front)
            .map(|(_, amount, _)| *amount)
            .collect();
        assert_eq!(
            landed,
            vec![12.0, 14.0, 22.0],
            "the authored chain, in order"
        );
        assert_eq!(damage.len(), 3, "no double hits within a swing: {damage:?}");
        assert!(
            !damage.iter().any(|(e, ..)| *e == behind),
            "the dummy behind was never in the arc"
        );
        // One event per combo step — the already-hit set is what makes that true.
        let steps: Vec<usize> = damage.iter().map(|(.., s)| *s).collect();
        assert_eq!(steps, vec![0, 1, 2]);
        // And the fight actually moved the character around the arena.
        assert!(pos.distance(start) > 1.0, "the script went nowhere");
    }

    /// An ally standing in the same arc is never hit, and neither is a target out of
    /// reach.
    #[test]
    fn the_arc_respects_teams_and_reach() {
        let map = arena();
        let mut world = World::new();
        let (me, ally, far) = (world.spawn(), world.spawn(), world.spawn());
        let start = centre();
        let mut pos = start;
        let targets = [
            Target {
                entity: ally,
                position: start + Vec2::new(1.2, 0.0),
                radius: 0.45,
                team: Team::PLAYER,
            },
            Target {
                entity: far,
                position: start + Vec2::new(6.0, 0.0),
                radius: 0.45,
                team: Team::ENEMY,
            },
        ];
        let mut w = WarriorController::new();
        let mut hits = 0;
        for step in 0..60 {
            let input = if step == 0 {
                press_attack(Vec2::X)
            } else {
                WarriorInput::default()
            };
            let out = w.tick(input, DT, ctx(&map, pos, me, &targets));
            pos = out.position;
            hits += out.hits.len();
        }
        assert_eq!(hits, 0);
    }

    // -- robustness ---------------------------------------------------------------------

    /// Junk `dt` and junk input advance nothing and break nothing.
    #[test]
    fn junk_input_is_survivable() {
        let map = arena();
        let start = centre();
        let mut pos = start;
        let mut w = WarriorController::new();
        let nonsense = WarriorInput {
            move_dir: Vec2::new(f32::NAN, 1.0),
            sprint: true,
            attack_pressed: true,
            dodge_pressed: false,
        };
        for dt in [f32::NAN, -1.0, 0.0, f32::INFINITY] {
            let out = w.tick(nonsense, dt, ctx(&map, pos, entity(), &[]));
            pos = out.position;
            assert!(pos.is_finite());
            assert_eq!(pos, start, "a junk step is not a step");
            assert!(out.facing.is_finite());
        }
        // An over-long stick is clamped rather than turned into a speed boost.
        let mut fast = WarriorController::new();
        let mut fpos = start;
        run(
            &mut fast,
            &map,
            &mut fpos,
            WarriorInput {
                move_dir: Vec2::splat(50.0),
                ..WarriorInput::default()
            },
            6,
        );
        let travelled = (fpos - start).length();
        assert!(
            travelled <= 6.0 * DT * ClassDef::warrior().move_speed + 1e-4,
            "travelled {travelled} m"
        );
    }

    /// The same script twice gives the same character — no clock, no randomness, no
    /// hash-order dependence anywhere in the tick.
    #[test]
    fn the_controller_is_deterministic() {
        let play = || {
            let map = arena();
            let mut world = World::new();
            let (me, dummy) = (world.spawn(), world.spawn());
            let start = centre();
            let mut pos = start;
            let targets = [Target {
                entity: dummy,
                position: start + Vec2::new(1.6, 0.4),
                radius: 0.45,
                team: Team::ENEMY,
            }];
            let mut w = WarriorController::new();
            let mut log = Vec::new();
            for step in 0..240 {
                let input = WarriorInput {
                    move_dir: Vec2::new(
                        ((step / 13) % 3) as f32 - 1.0,
                        ((step / 7) % 3) as f32 - 1.0,
                    ),
                    sprint: step % 5 == 0,
                    attack_pressed: step % 17 == 0,
                    dodge_pressed: step % 53 == 0,
                };
                let out = w.tick(input, DT, ctx(&map, pos, me, &targets));
                pos = out.position;
                log.push(format!(
                    "{step} {:?} {:?} {:.6} {:?} {} {:.6}",
                    out.position.to_array(),
                    out.state,
                    out.facing_radians,
                    out.anim,
                    out.hits.len(),
                    out.iframes.remaining,
                ));
            }
            log
        };
        let first = play();
        assert_eq!(first, play());
        // Non-vacuous: the script fought, rolled and moved.
        assert!(first.iter().any(|l| l.contains("Attacking")));
        assert!(first.iter().any(|l| l.contains("Dodging")));
        assert!(
            first
                .iter()
                .any(|l| l.contains("Attack3") || l.contains("Attack2"))
        );
    }
}
