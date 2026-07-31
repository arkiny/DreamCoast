//! [`AnimMachine`] — the runtime that walks an [`AnimGraphDef`].

use std::collections::BTreeMap;

use super::graph::{ANY_STATE, AnimError, AnimGraphDef, Condition};
use super::params::Params;

/// Clip length assumed when nothing has told the machine otherwise, in seconds.
///
/// Only matters for wrapping a looping state and for raising
/// [`Condition::StateDone`]; a graph that authors `length` on its states, or an
/// integrator that calls [`AnimMachine::set_clip_length`], never sees it.
pub const DEFAULT_CLIP_LENGTH: f32 = 1.0;

/// A cross-fade in progress: where we came from, and how far along the blend is.
#[derive(Clone, Copy, Debug)]
struct Fade {
    /// State index being faded out.
    from: usize,
    /// Its playback time, which keeps advancing during the blend.
    from_time: f32,
    /// Seconds elapsed of `duration`.
    elapsed: f32,
    /// Total blend length in seconds (always `> 0`; a zero fade is a cut and is
    /// never recorded).
    duration: f32,
    /// Whether another transition may interrupt this blend.
    interruptible: bool,
}

/// The animation state machine: pure control logic over an [`AnimGraphDef`].
///
/// It decides **which clip, at what time, blended with what** — and nothing else.
/// It never loads a clip, samples a pose, or touches the renderer, and it has no
/// dependency on the scene's animation API. The output is at most two weighted
/// clip samples ([`current`](Self::current) plus an optional [`fade`](Self::fade));
/// mapping clip *names* to real clips and blending the two poses is the
/// integrator's job. That seam is what lets the whole machine be unit-tested with
/// no window and no device — and what lets pose sampling evolve independently.
///
/// # One tick
///
/// [`tick`](Self::tick) does exactly two things, in this order:
///
/// 1. **Advance time.** The current state's time grows by `dt * speed`; a looping
///    state wraps, a non-looping one clamps at the clip length and latches
///    [`Condition::StateDone`]. Any fade in progress advances too — including the
///    outgoing state's own playback time, because a crossfade blends two *moving*
///    poses, not a moving one against a freeze-frame.
/// 2. **Evaluate transitions.** The list is scanned **in declaration order** and
///    the **first match wins**; at most **one transition fires per tick**. A
///    transition matches when its `from` is the current state or the wildcard
///    `"any"`, and its condition holds. Because time is advanced first,
///    `StateDone` is visible on the same tick the clip ends.
///
/// Ordering consequences worth knowing:
///
/// * An `"any"` transition declared before a state-specific one **outranks** it.
///   Declare the interrupts (death, hit reactions) at the top of the list and the
///   ordinary flow below.
/// * `"any"` never matches the state it targets, so a wildcard cannot restart the
///   state it is already in — which would otherwise pin that state at time 0
///   forever. An explicitly declared self-transition (`from: "x", to: "x"`) *is*
///   allowed and does restart the state; that is the way to re-fire a looping
///   attack.
/// * A trigger is consumed **only by the transition that takes it**. An unused
///   trigger stays pending (see [`Params::clear_triggers`]).
///
/// # Fades and the retarget rule
///
/// A transition with `fade > 0` blends linearly over that many seconds. When a
/// transition fires **while a fade is still running**, the machine retargets:
///
/// > the in-progress fade's destination — the state that is currently on screen —
/// > becomes the new fade's source, **at its current time**, and the older tail is
/// > dropped.
///
/// So the output is never more than two clips deep. Dropping the tail is the
/// honest trade: keeping it would mean an unbounded stack of decaying poses to
/// sample every frame, and the third pose is already below the weight where
/// anyone can see it. The visible artefact is a small pop *only* when a fade is
/// retargeted very early (the dropped tail still had real weight); a transition
/// that must not be interrupted says so with `interruptible: false`, which locks
/// the machine for the duration of its own fade.
#[derive(Clone, Debug)]
pub struct AnimMachine {
    def: AnimGraphDef,
    /// Clip name → length in seconds. Seeded from the graph's authored lengths.
    lengths: BTreeMap<String, f32>,
    current: usize,
    time: f32,
    done: bool,
    fade: Option<Fade>,
}

impl AnimMachine {
    /// Build a machine from a validated graph, starting in
    /// [`AnimGraphDef::initial`].
    pub fn new(def: AnimGraphDef) -> Result<Self, AnimError> {
        def.validate()?;
        let current = def
            .state_index(&def.initial)
            .ok_or_else(|| AnimError::UnknownState(def.initial.clone()))?;
        let mut lengths = BTreeMap::new();
        for state in &def.states {
            if let Some(length) = state.length {
                // Two states sharing a clip share its length; the last authored
                // value wins, because a clip has exactly one duration.
                lengths.insert(state.clip.clone(), length);
            }
        }
        let mut machine = Self {
            def,
            lengths,
            current,
            time: 0.0,
            done: false,
            fade: None,
        };
        machine.done = machine.advance_state(current, 0.0, 0.0).1;
        Ok(machine)
    }

    /// Build a machine straight from RON graph text.
    pub fn from_ron(text: &str) -> Result<Self, AnimError> {
        Self::new(AnimGraphDef::from_ron(text)?)
    }

    /// The graph this machine walks.
    pub fn def(&self) -> &AnimGraphDef {
        &self.def
    }

    /// Tell the machine how long a clip actually is, overriding any authored
    /// `length`. The integrator calls this once per clip after the asset loads.
    pub fn set_clip_length(&mut self, clip: &str, seconds: f32) {
        if seconds.is_finite() && seconds >= 0.0 {
            self.lengths.insert(clip.to_string(), seconds);
        }
    }

    /// Length in seconds the machine is using for `clip`.
    pub fn clip_length(&self, clip: &str) -> f32 {
        self.lengths
            .get(clip)
            .copied()
            .unwrap_or(DEFAULT_CLIP_LENGTH)
    }

    /// Advance time, then evaluate transitions. See the type docs for the exact
    /// rules. A non-finite or negative `dt` advances nothing.
    pub fn tick(&mut self, dt: f32, params: &mut Params) {
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        self.advance_time(dt);
        self.evaluate(params);
    }

    /// The clip to sample and the time to sample it at — the primary output.
    pub fn current(&self) -> (&str, f32) {
        (self.def.states[self.current].clip.as_str(), self.time)
    }

    /// Name of the state being played.
    pub fn current_state(&self) -> &str {
        self.def.states[self.current].name.as_str()
    }

    /// Playback time within the current state, in seconds.
    pub fn state_time(&self) -> f32 {
        self.time
    }

    /// Whether the current (non-looping) state has reached the end of its clip —
    /// the value [`Condition::StateDone`] reads.
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// The outgoing half of a cross-fade, if one is running:
    /// `(clip, time, alpha)`.
    ///
    /// **`alpha` is the fade's progress in `[0, 1]`, i.e. the weight of the
    /// [`current`](Self::current) clip** — the outgoing clip returned here carries
    /// `1 - alpha`. It starts at 0 (the old pose still fully on screen) and
    /// reaches 1 as the blend completes, at which point this returns `None`.
    pub fn fade(&self) -> Option<(&str, f32, f32)> {
        self.fade.as_ref().map(|f| {
            (
                self.def.states[f.from].clip.as_str(),
                f.from_time,
                (f.elapsed / f.duration).clamp(0.0, 1.0),
            )
        })
    }

    /// Whether a cross-fade is in progress.
    pub fn is_fading(&self) -> bool {
        self.fade.is_some()
    }

    /// Snap back to the initial state with no fade — a respawn or a level swap.
    /// Clip lengths are kept (they describe assets, not gameplay state).
    pub fn reset(&mut self) {
        self.fade = None;
        self.current = self
            .def
            .state_index(&self.def.initial)
            .expect("initial state was validated");
        self.time = 0.0;
        self.done = self.advance_state(self.current, 0.0, 0.0).1;
    }

    /// Playback time and done-ness of `state` after `dt`, given its speed, loop
    /// flag and clip length.
    fn advance_state(&self, state: usize, time: f32, dt: f32) -> (f32, bool) {
        let def = &self.def.states[state];
        let length = self.clip_length(&def.clip);
        let advanced = time + dt * def.speed;
        if def.looping {
            let wrapped = if length > 0.0 {
                advanced.rem_euclid(length)
            } else {
                0.0
            };
            (wrapped, false)
        } else if advanced >= length {
            (length, true)
        } else {
            (advanced, false)
        }
    }

    fn advance_time(&mut self, dt: f32) {
        let (time, done) = self.advance_state(self.current, self.time, dt);
        self.time = time;
        self.done = done;
        if let Some(mut fade) = self.fade.take() {
            fade.from_time = self.advance_state(fade.from, fade.from_time, dt).0;
            fade.elapsed += dt;
            if fade.elapsed < fade.duration {
                self.fade = Some(fade);
            }
        }
    }

    /// Fire at most one transition. Returns whether one was taken.
    fn evaluate(&mut self, params: &mut Params) -> bool {
        // A non-interruptible transition owns the machine until its fade ends.
        if self.fade.is_some_and(|f| !f.interruptible) {
            return false;
        }
        let Some(index) = self.pick_transition(params) else {
            return false;
        };
        let transition = &self.def.transitions[index];
        let to = self
            .def
            .state_index(&transition.to)
            .expect("transition targets were validated");
        let (fade, interruptible) = (transition.fade, transition.interruptible);
        if let Condition::Trigger(name) = &transition.condition {
            params.consume_trigger(name);
        }
        self.enter(to, fade, interruptible);
        true
    }

    /// Index of the first transition, in declaration order, that matches.
    fn pick_transition(&self, params: &Params) -> Option<usize> {
        let current_name = self.def.states[self.current].name.as_str();
        self.def.transitions.iter().position(|t| {
            let wildcard = t.from.eq_ignore_ascii_case(ANY_STATE);
            if !wildcard && t.from != current_name {
                return false;
            }
            let Some(to) = self.def.state_index(&t.to) else {
                return false;
            };
            // A wildcard never restarts the state it targets.
            if wildcard && to == self.current {
                return false;
            }
            self.holds(&t.condition, params)
        })
    }

    fn holds(&self, condition: &Condition, params: &Params) -> bool {
        match condition {
            Condition::Flag(name) => params.flag(name),
            Condition::NotFlag(name) => !params.flag(name),
            Condition::Trigger(name) => params.is_triggered(name),
            Condition::StateDone => self.done,
        }
    }

    /// Enter `to`, starting (or retargeting) a fade from whatever is on screen.
    fn enter(&mut self, to: usize, fade: f32, interruptible: bool) {
        self.fade = if fade > 0.0 {
            // The retarget rule: the source is always the state currently being
            // shown, at its current time, so any older tail is dropped here.
            Some(Fade {
                from: self.current,
                from_time: self.time,
                elapsed: 0.0,
                duration: fade,
                interruptible,
            })
        } else {
            None
        };
        self.current = to;
        let (time, done) = self.advance_state(to, 0.0, 0.0);
        self.time = time;
        self.done = done;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::graph::{AnimStateDef, TransitionDef};

    const DT: f32 = 1.0 / 60.0;

    /// The shipped example graph.
    const WARRIOR_ANIM: &str = include_str!("../../assets/warrior_anim.ron");

    fn state(name: &str, clip: &str, looping: bool, length: f32) -> AnimStateDef {
        AnimStateDef {
            name: name.to_string(),
            clip: clip.to_string(),
            speed: 1.0,
            looping,
            length: Some(length),
        }
    }

    fn edge(from: &str, to: &str, condition: Condition, fade: f32) -> TransitionDef {
        TransitionDef {
            from: from.to_string(),
            to: to.to_string(),
            condition,
            fade,
            interruptible: true,
        }
    }

    /// idle ⇄ run, plus an "any"-state attack that returns to idle when done, and
    /// a locked death state.
    fn graph() -> AnimGraphDef {
        AnimGraphDef {
            initial: "idle".to_string(),
            states: vec![
                state("idle", "idle", true, 2.0),
                state("run", "run", true, 0.8),
                state("attack", "attack_1", false, 0.6),
                state("death", "death", false, 1.5),
            ],
            transitions: vec![
                TransitionDef {
                    interruptible: false,
                    ..edge(ANY_STATE, "death", Condition::Flag("dead".into()), 0.2)
                },
                edge(
                    ANY_STATE,
                    "attack",
                    Condition::Trigger("attack".into()),
                    0.1,
                ),
                edge("attack", "idle", Condition::StateDone, 0.15),
                edge("idle", "run", Condition::Flag("moving".into()), 0.12),
                edge("run", "idle", Condition::NotFlag("moving".into()), 0.12),
            ],
        }
    }

    fn machine() -> AnimMachine {
        AnimMachine::new(graph()).unwrap()
    }

    #[test]
    fn starts_in_the_initial_state() {
        let m = machine();
        assert_eq!(m.current_state(), "idle");
        assert_eq!(m.current(), ("idle", 0.0));
        assert!(m.fade().is_none());
        assert!(!m.is_done());
    }

    #[test]
    fn a_broken_graph_is_rejected_at_construction() {
        let mut def = graph();
        def.initial = "ghost".to_string();
        assert!(AnimMachine::new(def).is_err());
        assert!(AnimMachine::from_ron("not ron").is_err());
    }

    #[test]
    fn looping_time_wraps_and_never_completes() {
        let mut m = machine();
        let mut p = Params::new();
        for _ in 0..300 {
            m.tick(DT, &mut p);
            assert!(m.state_time() < 2.0);
            assert!(!m.is_done(), "a looping state is never done");
        }
        assert_eq!(m.current_state(), "idle");
    }

    #[test]
    fn non_looping_time_clamps_and_raises_state_done() {
        let mut def = graph();
        def.transitions.clear(); // isolate the clock from the graph
        def.initial = "attack".to_string();
        let mut m = AnimMachine::new(def).unwrap();
        let mut p = Params::new();
        for _ in 0..36 {
            m.tick(DT, &mut p); // 0.6s of a 0.6s clip
        }
        assert_eq!(m.state_time(), 0.6, "clamped at the clip length");
        assert!(m.is_done());
        m.tick(DT, &mut p);
        assert!(m.is_done(), "done latches");
    }

    #[test]
    fn speed_scales_the_clock() {
        let mut def = graph();
        def.transitions.clear();
        def.initial = "attack".to_string();
        def.states[2].speed = 2.0;
        let mut m = AnimMachine::new(def).unwrap();
        let mut p = Params::new();
        m.tick(0.1, &mut p);
        assert!((m.state_time() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn a_trigger_transitions_and_is_consumed() {
        let mut m = machine();
        let mut p = Params::new();
        p.trigger("attack");
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "attack");
        assert!(!p.is_triggered("attack"), "the transition consumed it");
        // Playing out the attack returns to idle without re-firing.
        for _ in 0..60 {
            m.tick(DT, &mut p);
        }
        assert_eq!(m.current_state(), "idle");
    }

    #[test]
    fn an_unused_trigger_waits_for_a_reachable_transition() {
        // Only `idle` answers "attack" here, so a trigger fired in `run` waits.
        let mut def = graph();
        def.transitions[1].from = "idle".to_string();
        let mut m = AnimMachine::new(def).unwrap();
        let mut p = Params::new().with_flag("moving", true);
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "run");
        p.trigger("attack");
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "run", "no transition took it");
        assert!(p.is_triggered("attack"), "so it is still pending");
        p.set_flag("moving", false);
        m.tick(DT, &mut p); // run -> idle
        m.tick(DT, &mut p); // idle -> attack, now that it is reachable
        assert_eq!(m.current_state(), "attack");
        assert!(!p.is_triggered("attack"));
    }

    #[test]
    fn flags_drive_the_locomotion_loop_both_ways() {
        let mut m = machine();
        let mut p = Params::new();
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "idle");
        p.set_flag("moving", true);
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "run");
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "run", "NotFlag holds it while moving");
        p.set_flag("moving", false);
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "idle");
    }

    #[test]
    fn any_state_transitions_fire_from_everywhere() {
        for start in ["idle", "run", "attack"] {
            let mut def = graph();
            def.initial = start.to_string();
            let mut m = AnimMachine::new(def).unwrap();
            let mut p = Params::new().with_flag("dead", true);
            m.tick(DT, &mut p);
            assert_eq!(m.current_state(), "death", "from {start}");
        }
    }

    #[test]
    fn a_wildcard_never_restarts_its_own_target() {
        let mut m = machine();
        let mut p = Params::new().with_flag("dead", true);
        m.tick(DT, &mut p); // -> death (fade 0.2, non-interruptible)
        assert_eq!(m.current_state(), "death");
        for _ in 0..60 {
            m.tick(DT, &mut p);
        }
        // Still ticking forward rather than being pinned at 0 by its own edge.
        assert!(m.state_time() > 0.5, "time = {}", m.state_time());
    }

    #[test]
    fn an_explicit_self_transition_does_restart() {
        let def = AnimGraphDef {
            initial: "attack".to_string(),
            states: vec![state("attack", "attack_1", false, 0.6)],
            transitions: vec![edge(
                "attack",
                "attack",
                Condition::Trigger("attack".into()),
                0.05,
            )],
        };
        let mut m = AnimMachine::new(def).unwrap();
        let mut p = Params::new();
        m.tick(0.3, &mut p);
        assert!((m.state_time() - 0.3).abs() < 1e-6);
        p.trigger("attack");
        m.tick(DT, &mut p);
        assert_eq!(m.state_time(), 0.0, "restarted from the top");
        assert!(m.is_fading(), "and blended with its own tail");
    }

    #[test]
    fn declaration_order_decides_ties() {
        let mut def = graph();
        // Both the wildcard death edge (index 0) and the attack edge (index 1)
        // are satisfiable; the earlier declaration wins.
        let mut m = AnimMachine::new(def.clone()).unwrap();
        let mut p = Params::new().with_flag("dead", true);
        p.trigger("attack");
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "death");
        assert!(p.is_triggered("attack"), "the losing edge consumed nothing");

        // Swap the declarations and the answer swaps with them.
        def.transitions.swap(0, 1);
        let mut m = AnimMachine::new(def).unwrap();
        let mut p = Params::new().with_flag("dead", true);
        p.trigger("attack");
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "attack");
    }

    #[test]
    fn at_most_one_transition_fires_per_tick() {
        let def = AnimGraphDef {
            initial: "a".to_string(),
            states: vec![
                state("a", "a", false, 0.0),
                state("b", "b", false, 0.0),
                state("c", "c", false, 0.0),
            ],
            transitions: vec![
                edge("a", "b", Condition::StateDone, 0.0),
                edge("b", "c", Condition::StateDone, 0.0),
            ],
        };
        let mut m = AnimMachine::new(def).unwrap();
        let mut p = Params::new();
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "b", "one hop per tick");
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "c");
    }

    #[test]
    fn a_fade_ramps_linearly_and_then_ends() {
        let mut m = machine();
        let mut p = Params::new();
        p.set_flag("moving", true);
        m.tick(DT, &mut p); // idle -> run, fade 0.12
        let (clip, time, alpha) = m.fade().expect("fading");
        assert_eq!(clip, "idle");
        assert!((time - DT).abs() < 1e-6, "the outgoing clip kept playing");
        assert_eq!(alpha, 0.0, "the new clip has no weight yet");

        m.tick(0.06, &mut p);
        let (_, _, alpha) = m.fade().expect("still fading");
        assert!((alpha - 0.5).abs() < 1e-3, "alpha = {alpha}");

        m.tick(0.06, &mut p);
        assert!(!m.is_fading(), "the blend completed");
        assert_eq!(m.current(), ("run", 0.12));
    }

    #[test]
    fn the_outgoing_clip_keeps_playing_during_a_fade() {
        let mut m = machine();
        let mut p = Params::new();
        for _ in 0..30 {
            m.tick(DT, &mut p); // idle to 0.5s
        }
        p.set_flag("moving", true);
        m.tick(DT, &mut p);
        let (_, first, _) = m.fade().unwrap();
        m.tick(DT, &mut p);
        let (_, second, _) = m.fade().unwrap();
        assert!(second > first, "{second} should be past {first}");
    }

    #[test]
    fn a_zero_fade_is_a_cut() {
        let mut def = graph();
        def.transitions[3].fade = 0.0; // idle -> run
        let mut m = AnimMachine::new(def).unwrap();
        let mut p = Params::new().with_flag("moving", true);
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "run");
        assert!(!m.is_fading());
    }

    #[test]
    fn a_transition_during_a_fade_retargets_from_the_current_clip() {
        // idle --0.12--> run, then interrupt mid-blend with an attack.
        let mut m = machine();
        let mut p = Params::new();
        p.set_flag("moving", true);
        m.tick(DT, &mut p); // -> run, fading from idle
        m.tick(0.04, &mut p);
        assert_eq!(m.fade().unwrap().0, "idle");
        let run_time = m.state_time();

        p.trigger("attack");
        m.tick(DT, &mut p); // -> attack, retargeting the blend
        assert_eq!(m.current_state(), "attack");
        let (clip, time, alpha) = m.fade().expect("a new fade started");
        assert_eq!(clip, "run", "the fade's destination became the new source");
        assert!(
            (time - (run_time + DT)).abs() < 1e-6,
            "at its current time, not from zero"
        );
        assert_eq!(alpha, 0.0, "the new blend starts from scratch");
        // Only ever two clips deep: the idle tail is gone.
        assert_eq!(m.current().0, "attack_1");
    }

    #[test]
    fn a_non_interruptible_fade_locks_the_machine() {
        let mut def = graph();
        def.transitions[3].fade = 0.2; // idle -> run
        def.transitions[3].interruptible = false;
        let mut m = AnimMachine::new(def).unwrap();
        let mut p = Params::new().with_flag("moving", true);
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "run");

        // An attack during that blend is refused...
        p.trigger("attack");
        for _ in 0..6 {
            m.tick(DT, &mut p);
            assert_eq!(m.current_state(), "run", "locked for the fade");
        }
        assert!(p.is_triggered("attack"), "and consumed nothing");
        // ...until the blend finishes, at which point the still-pending trigger
        // fires. The input was buffered, not lost.
        for _ in 0..8 {
            m.tick(DT, &mut p);
        }
        assert_eq!(m.current_state(), "attack");
        assert!(!p.is_triggered("attack"));
    }

    #[test]
    fn clip_lengths_can_be_supplied_at_runtime() {
        let mut m = machine();
        assert_eq!(m.clip_length("attack_1"), 0.6, "authored length");
        assert_eq!(
            m.clip_length("unknown"),
            DEFAULT_CLIP_LENGTH,
            "unknown clips fall back"
        );
        m.set_clip_length("attack_1", 1.25);
        assert_eq!(m.clip_length("attack_1"), 1.25);
        m.set_clip_length("attack_1", f32::NAN);
        assert_eq!(m.clip_length("attack_1"), 1.25, "junk is ignored");

        // The longer clip now takes longer to finish.
        let mut p = Params::new();
        p.trigger("attack");
        m.tick(DT, &mut p);
        for _ in 0..40 {
            m.tick(DT, &mut p);
        }
        assert_eq!(m.current_state(), "attack", "0.67s in, not done yet");
        for _ in 0..40 {
            m.tick(DT, &mut p);
        }
        assert_eq!(m.current_state(), "idle");
    }

    #[test]
    fn junk_and_zero_dt_advance_nothing() {
        let mut m = machine();
        let mut p = Params::new();
        m.tick(f32::NAN, &mut p);
        m.tick(-1.0, &mut p);
        m.tick(0.0, &mut p);
        assert_eq!(m.state_time(), 0.0);
        assert_eq!(m.current_state(), "idle");
    }

    #[test]
    fn reset_returns_to_the_initial_state() {
        let mut m = machine();
        let mut p = Params::new();
        p.trigger("attack");
        m.tick(DT, &mut p);
        m.tick(0.2, &mut p);
        assert_eq!(m.current_state(), "attack");
        m.reset();
        assert_eq!(m.current_state(), "idle");
        assert_eq!(m.state_time(), 0.0);
        assert!(!m.is_fading());
        assert_eq!(m.def().initial, "idle");
    }

    /// The shipped warrior graph must actually drive a warrior: locomotion, the
    /// three-hit chain fed by combo triggers, and a death that locks everything.
    #[test]
    fn the_warrior_fixture_drives_a_whole_moveset() {
        let mut m = AnimMachine::from_ron(WARRIOR_ANIM).unwrap();
        let mut p = Params::new();
        assert_eq!(m.current_state(), "idle");

        p.set_flag("moving", true);
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "run");

        // The chain, as `AttackEvent::ComboAdvanced` would drive it.
        for (trigger, state) in [
            ("attack_1", "slash_left"),
            ("attack_2", "slash_right"),
            ("attack_3", "overhead"),
        ] {
            p.trigger(trigger);
            m.tick(DT, &mut p);
            assert_eq!(m.current_state(), state);
            assert!(m.is_fading(), "{state} blends in");
            // The attack blend is locked, so movement cannot steal it.
            m.tick(0.1, &mut p);
            assert_eq!(m.current_state(), state);
        }
        // The finisher plays out and returns to locomotion.
        for _ in 0..120 {
            m.tick(DT, &mut p);
        }
        assert_eq!(m.current_state(), "run");

        p.set_flag("dead", true);
        m.tick(DT, &mut p);
        assert_eq!(m.current_state(), "death");
        // `death` has no outgoing edge, and a dead controller stops feeding
        // input — so the graph has no way out of it. (The wildcard interrupts
        // above *would* still fire; terminality is the controller's job, not the
        // graph's. See the fixture header.)
        p.clear_triggers();
        for _ in 0..300 {
            m.tick(DT, &mut p);
        }
        assert_eq!(m.current_state(), "death", "dead is a one-way door");
        assert!(m.is_done());
    }

    #[test]
    fn identical_input_sequences_produce_identical_traversals() {
        // A scripted 4-second session: movement toggling, attacks, then death.
        let play = || {
            let mut m = machine();
            let mut p = Params::new();
            let mut log = Vec::new();
            for frame in 0..240 {
                p.set_flag("moving", (frame / 17) % 2 == 0);
                if frame % 23 == 0 {
                    p.trigger("attack");
                }
                if frame == 200 {
                    p.set_flag("dead", true);
                }
                m.tick(DT, &mut p);
                let (clip, time) = m.current();
                log.push(format!(
                    "{frame} {clip} {time:.5} {:?} {}",
                    m.fade().map(|(c, t, a)| format!("{c} {t:.5} {a:.5}")),
                    m.current_state()
                ));
            }
            log
        };
        let first = play();
        assert_eq!(first, play());
        // And it actually exercised the graph rather than sitting in idle.
        assert!(first.iter().any(|l| l.contains("attack_1")));
        assert!(first.last().unwrap().contains("death"));
    }
}
