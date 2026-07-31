//! Animation state machine: data-driven control logic over named clips
//! (`docs/game-framework-plan.md` §4.2).
//!
//! This module answers one question every tick — **which clip should be playing,
//! at what time, blended with what** — and deliberately answers nothing else. It
//! does not load clips, sample poses, blend transforms, or know that a renderer
//! exists; it is decoupled from the scene's animation API as well as from the
//! RHI. The output is at most **two weighted clip samples**:
//!
//! * [`AnimMachine::current`] → `(clip, time)`, the incoming pose;
//! * [`AnimMachine::fade`] → `Some((clip, time, alpha))` while a cross-fade runs,
//!   where `alpha` is the *current* clip's weight and the returned one carries
//!   `1 - alpha`.
//!
//! The integrator maps those clip names to real clips and calls whatever blend
//! API it has. Keeping the seam at *clip names* is what lets pose sampling and
//! the state machine be built, tested and changed independently — and it is why
//! every rule in here can be verified with no window, no device, and no assets.
//!
//! # The pieces
//!
//! * [`AnimGraphDef`] — states, transitions and the initial state, as RON.
//! * [`Condition`] — the expression-free guard set: [`Flag`](Condition::Flag),
//!   [`NotFlag`](Condition::NotFlag), [`Trigger`](Condition::Trigger),
//!   [`StateDone`](Condition::StateDone).
//! * [`Params`] — the flags and triggers the game refills each tick.
//! * [`AnimMachine`] — the runtime. Evaluation order, the one-transition-per-tick
//!   rule, and the fade **retarget rule** are all specified on its type docs;
//!   read them before authoring a graph.
//!
//! ```
//! use dreamcoast_game::anim::{AnimMachine, Params};
//!
//! let mut machine = AnimMachine::from_ron(
//!     r#"(
//!         initial: "idle",
//!         states: [
//!             (name: "idle",   clip: "idle", looping: true,  length: Some(2.0)),
//!             (name: "run",    clip: "run",  looping: true,  length: Some(0.8)),
//!             (name: "attack", clip: "attack_1",             length: Some(0.6)),
//!         ],
//!         transitions: [
//!             (from: "any",    to: "attack", condition: Trigger("attack"),  fade: 0.08,
//!              interruptible: false),
//!             (from: "attack", to: "idle",   condition: StateDone,          fade: 0.15),
//!             (from: "idle",   to: "run",    condition: Flag("moving"),     fade: 0.12),
//!             (from: "run",    to: "idle",   condition: NotFlag("moving"),  fade: 0.12),
//!         ],
//!     )"#,
//! )
//! .unwrap();
//!
//! // The real clip lengths, once the assets are known.
//! machine.set_clip_length("attack_1", 0.72);
//!
//! let mut params = Params::new();
//! params.set_flag("moving", true);      // level: refreshed every tick
//! params.trigger("attack");             // edge: consumed by the transition
//!
//! machine.tick(1.0 / 60.0, &mut params);
//! assert_eq!(machine.current_state(), "attack");
//! let (clip, time) = machine.current();
//! assert_eq!(clip, "attack_1");
//! assert_eq!(time, 0.0);
//! // Fading out of idle: the outgoing clip still carries all the weight.
//! assert_eq!(machine.fade().map(|(c, _, a)| (c, a)), Some(("idle", 0.0)));
//! ```

mod graph;
mod machine;
mod params;

pub use graph::{ANY_STATE, AnimError, AnimGraphDef, AnimStateDef, Condition, TransitionDef};
pub use machine::{AnimMachine, DEFAULT_CLIP_LENGTH};
pub use params::Params;

/// The shipped warrior animation graph (`crates/game/assets/warrior_anim.ron`), as
/// text — the worked example this module is documented against, and the graph
/// `apps/dungeon`'s warrior actually runs.
///
/// Exported as a `pub const` rather than left as a file a consumer reaches for
/// because both readers want *these bytes*, not a copy of them: the crate's own
/// tests parse it, and a game builds its [`AnimMachine`] from it. Before this
/// existed the game `include_str!`d it across the workspace by relative path
/// (`../../../crates/game/assets/…`), which is a build break waiting for the first
/// crate that moves. A fixture with two readers is API surface, so it is declared
/// as API surface.
pub const WARRIOR_ANIM_RON: &str = include_str!("../../assets/warrior_anim.ron");
