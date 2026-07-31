//! The melee attack model: one swing's data ([`AttackSpec`]), a chain of them
//! ([`ComboChain`]), and the phase clock that plays them ([`AttackState`]).
//!
//! # The three-phase swing
//!
//! Every attack is **windup → active → recovery**. Windup is the anticipation
//! the player reads to dodge; active is the only phase that can hit anything;
//! recovery is the commitment cost — the reason a mistimed swing is punished.
//! The clock here is the single authority on which phase an attacker is in, so
//! the animation, the hit resolution and the AI all read the same number instead
//! of each keeping a private timer that drifts.
//!
//! # Combo window rule
//!
//! An attack input is accepted, and *buffered*, while the current step is in its
//! **active or recovery** phase. It is ignored during windup: a step that has not
//! yet swung cannot already be queueing its successor, and accepting it there
//! would let a mashing player skip the entire read window.
//!
//! A buffered input **cancels the remaining recovery**: the next step starts at
//! the first tick that finds the state in recovery with an input pending. That is
//! what makes a linked combo feel faster than three separate swings while still
//! playing every hit window in full. If no input arrives, the step plays its
//! recovery out, the window expires, and the chain resets to step 0 — a dropped
//! link never leaves the attacker "half way through" a combo.
//!
//! # No double hits
//!
//! The active window spans many ticks, and the arc is re-tested on each of them.
//! [`AttackState`] therefore keeps the set of entities already struck **by this
//! swing**, cleared when the window opens, so a target inside the arc for eight
//! ticks takes the damage once. See [`AttackState::resolve_hits`].

use dreamcoast_scene::Entity;
use glam::Vec2;
use serde::{Deserialize, Serialize};

use super::Team;
use super::hit::resolve_arc_hits;

/// One swing, as data. RON-friendly: this is what a `ClassDef` file is mostly
/// made of.
///
/// Ranges/angles are on the **XZ plane** (the `physics` convention: `Vec2.x` is
/// world X, `Vec2.y` is world Z).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttackSpec {
    /// Identifier for the step — used by animation graphs and by tuning tools.
    pub name: String,
    /// Hit points removed from each target struck.
    pub damage: f32,
    /// Reach of the arc, in world units, measured from the attacker's centre.
    pub range: f32,
    /// Half the arc's opening angle, in radians. The full swing covers
    /// `2 * half_angle_rad`.
    pub half_angle_rad: f32,
    /// Anticipation, in seconds. Nothing can be hit yet.
    pub windup: f32,
    /// Hit window, in seconds. **Must exceed the fixed timestep** or the swing
    /// can fall between two ticks and hit nothing.
    pub active: f32,
    /// Commitment, in seconds. Cancellable only into the next combo step.
    pub recovery: f32,
    /// Hitstun this swing inflicts, in seconds — read by the *game* when it
    /// reacts to a [`DamageEvent`](super::DamageEvent); this crate only carries
    /// the number.
    #[serde(default)]
    pub stagger: f32,
}

impl Default for AttackSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            damage: 0.0,
            range: 1.0,
            half_angle_rad: std::f32::consts::FRAC_PI_4,
            windup: 0.1,
            active: 0.1,
            recovery: 0.1,
            stagger: 0.0,
        }
    }
}

impl AttackSpec {
    /// Total time from input to idle when the step is *not* linked, in seconds.
    pub fn duration(&self) -> f32 {
        self.windup + self.active + self.recovery
    }

    /// Time from input to the first frame that can connect, in seconds — the
    /// number that decides whether an attack beats a charging enemy.
    pub fn time_to_hit(&self) -> f32 {
        self.windup
    }

    /// The arc's full opening angle in degrees, for tuning UI and logs.
    pub fn arc_degrees(&self) -> f32 {
        (2.0 * self.half_angle_rad).to_degrees()
    }
}

/// An ordered chain of swings — step 0 is the opener, the last step the
/// finisher.
///
/// Serialised **transparently**, so a chain is a plain RON list rather than a
/// nested newtype wrapper:
///
/// ```ron
/// combo: [
///     (name: "slash_left", damage: 12.0, ...),
///     (name: "overhead",   damage: 22.0, ...),
/// ]
/// ```
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ComboChain(pub Vec<AttackSpec>);

impl ComboChain {
    /// A chain from its steps, in order.
    pub fn new(steps: Vec<AttackSpec>) -> Self {
        Self(steps)
    }

    /// Number of steps.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the chain has no steps (an entity that cannot attack).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Borrow step `index`.
    pub fn get(&self, index: usize) -> Option<&AttackSpec> {
        self.0.get(index)
    }

    /// Iterate the steps in order.
    pub fn iter(&self) -> std::slice::Iter<'_, AttackSpec> {
        self.0.iter()
    }

    /// The steps as a slice.
    pub fn as_slice(&self) -> &[AttackSpec] {
        &self.0
    }
}

impl<'a> IntoIterator for &'a ComboChain {
    type Item = &'a AttackSpec;
    type IntoIter = std::slice::Iter<'a, AttackSpec>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Which phase of a swing an attacker is in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AttackPhase {
    /// Not attacking.
    #[default]
    Idle,
    /// Anticipation — committed, but nothing can be hit yet.
    Windup,
    /// The hit window is open.
    Active,
    /// Committed, cancellable only into the next combo step.
    Recovery,
}

/// Something the phase clock did during one [`AttackState::tick`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttackEvent {
    /// The hit window opened; the already-hit set was cleared. Start testing
    /// the arc.
    HitWindowOpen {
        /// Combo step whose window opened.
        step: usize,
    },
    /// The hit window closed. Stop testing the arc.
    HitWindowClosed {
        /// Combo step whose window closed.
        step: usize,
    },
    /// A buffered input linked into the next step, cancelling the remainder of
    /// the previous step's recovery.
    ComboAdvanced {
        /// The step that just started.
        step: usize,
    },
    /// The chain ended (recovery ran out with nothing buffered). The state is
    /// back to [`AttackPhase::Idle`] at step 0.
    Finished,
}

/// Upper bound on events one [`AttackState::tick`] can report.
///
/// The worst case is bounded by data, not by `dt`: a tick can close out the
/// current step (open + close), consume the **one** buffered input (advance),
/// and play the next step out entirely (open + close + finish) — six events. A
/// second link is impossible in the same tick because buffering requires an
/// [`AttackState::request`] call between ticks. Eight leaves headroom.
pub const MAX_ATTACK_EVENTS: usize = 8;

/// Iteration cap for the phase-advance loop. Every iteration either consumes a
/// phase or consumes the single buffered input, both of which are bounded, so
/// this is a belt-and-braces guard against a future edit introducing a cycle.
const MAX_TICK_STEPS: usize = 16;

/// The (at most [`MAX_ATTACK_EVENTS`]) events from one tick, in order, without
/// allocating.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttackEvents {
    buf: [Option<AttackEvent>; MAX_ATTACK_EVENTS],
    len: usize,
}

impl Default for AttackEvents {
    fn default() -> Self {
        Self {
            buf: [None; MAX_ATTACK_EVENTS],
            len: 0,
        }
    }
}

impl AttackEvents {
    fn push(&mut self, event: AttackEvent) {
        debug_assert!(
            self.len < MAX_ATTACK_EVENTS,
            "attack event overflow: {event:?}"
        );
        if self.len < MAX_ATTACK_EVENTS {
            self.buf[self.len] = Some(event);
            self.len += 1;
        }
    }

    /// The events, in the order they happened.
    pub fn iter(&self) -> impl Iterator<Item = AttackEvent> + '_ {
        self.buf[..self.len].iter().filter_map(|e| *e)
    }

    /// How many events this tick produced.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the tick was uneventful (the common case).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether a specific event occurred this tick.
    pub fn contains(&self, event: AttackEvent) -> bool {
        self.iter().any(|e| e == event)
    }

    /// Whether the chain ended this tick.
    pub fn finished(&self) -> bool {
        self.contains(AttackEvent::Finished)
    }
}

impl IntoIterator for AttackEvents {
    type Item = AttackEvent;
    type IntoIter = std::iter::Flatten<
        std::iter::Take<std::array::IntoIter<Option<AttackEvent>, MAX_ATTACK_EVENTS>>,
    >;
    /// By value, still without allocating — the buffer is a `Copy` array.
    fn into_iter(self) -> Self::IntoIter {
        self.buf.into_iter().take(self.len).flatten()
    }
}

/// The per-attacker phase clock: which step, which phase, how far in, whether an
/// input is buffered, and who this swing has already hit.
///
/// **The chain is not stored here.** It is passed to every method that needs it,
/// so one [`ComboChain`] (owned by a [`ClassDef`](super::ClassDef) in a resource)
/// is shared by every entity of that class instead of being cloned per monster.
/// The same split as [`BodyCircle`](super::BodyCircle): state in the component,
/// data outside it.
///
/// Pure logic — no clock, no randomness, no world access — so a whole combo can
/// be driven in a unit test.
#[derive(Clone, Debug, Default)]
pub struct AttackState {
    phase: AttackPhase,
    step: usize,
    time: f32,
    buffered: bool,
    hits: Vec<Entity>,
}

impl AttackState {
    /// An idle attacker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current phase.
    #[inline]
    pub fn phase(&self) -> AttackPhase {
        self.phase
    }

    /// Index of the combo step being played (0 while idle).
    #[inline]
    pub fn step(&self) -> usize {
        self.step
    }

    /// Seconds spent in the current phase.
    #[inline]
    pub fn time_in_phase(&self) -> f32 {
        self.time
    }

    /// Whether a swing is in progress.
    #[inline]
    pub fn is_attacking(&self) -> bool {
        self.phase != AttackPhase::Idle
    }

    /// Whether the arc should be tested this tick.
    #[inline]
    pub fn is_hit_window_open(&self) -> bool {
        self.phase == AttackPhase::Active
    }

    /// Whether an input is waiting to link into the next step.
    #[inline]
    pub fn has_buffered_input(&self) -> bool {
        self.buffered
    }

    /// The spec of the step being played, if any.
    pub fn spec<'a>(&self, chain: &'a ComboChain) -> Option<&'a AttackSpec> {
        if self.is_attacking() {
            chain.get(self.step)
        } else {
            None
        }
    }

    /// Feed an attack input.
    ///
    /// * **Idle** → starts step 0. Returns `true`.
    /// * **Windup** → ignored (see the module docs). Returns `false`.
    /// * **Active / recovery** → buffers the next step, if the chain has one.
    ///   Returns `true` when it was buffered, `false` on the finisher (there is
    ///   nothing to link into).
    ///
    /// Re-requesting while already buffered is idempotent: one input is one
    /// link, so mashing cannot bank a queue of swings.
    pub fn request(&mut self, chain: &ComboChain) -> bool {
        if chain.is_empty() {
            return false;
        }
        match self.phase {
            AttackPhase::Idle => {
                self.begin(0);
                true
            }
            AttackPhase::Windup => false,
            AttackPhase::Active | AttackPhase::Recovery => {
                if self.step + 1 < chain.len() {
                    self.buffered = true;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Advance the clock by `dt` seconds and report what happened.
    ///
    /// Handles a `dt` that crosses several phase boundaries: the leftover time
    /// carries into the next phase, so the clock is frame-rate independent and a
    /// hitch cannot leave a swing stuck. Non-finite or negative `dt` advances
    /// nothing.
    pub fn tick(&mut self, chain: &ComboChain, dt: f32) -> AttackEvents {
        let mut out = AttackEvents::default();
        if self.phase == AttackPhase::Idle {
            return out;
        }
        if chain.is_empty() || chain.get(self.step).is_none() {
            // The chain changed underneath a live swing (class swap, hot reload):
            // end it cleanly rather than playing a step that no longer exists.
            self.finish(&mut out);
            return out;
        }
        let mut remaining = if dt.is_finite() { dt.max(0.0) } else { 0.0 };

        for _ in 0..MAX_TICK_STEPS {
            // A buffered input cancels the rest of recovery, so it is consumed
            // before any time is charged to that phase.
            if self.phase == AttackPhase::Recovery && self.buffered && self.step + 1 < chain.len() {
                let next = self.step + 1;
                self.begin(next);
                out.push(AttackEvent::ComboAdvanced { step: next });
                continue;
            }
            let Some(spec) = chain.get(self.step) else {
                self.finish(&mut out);
                break;
            };
            let duration = match self.phase {
                AttackPhase::Idle => break,
                AttackPhase::Windup => spec.windup,
                AttackPhase::Active => spec.active,
                AttackPhase::Recovery => spec.recovery,
            }
            .max(0.0);

            if self.time + remaining < duration {
                self.time += remaining;
                break;
            }
            remaining -= duration - self.time;
            self.time = 0.0;
            match self.phase {
                AttackPhase::Windup => {
                    self.phase = AttackPhase::Active;
                    self.hits.clear();
                    out.push(AttackEvent::HitWindowOpen { step: self.step });
                }
                AttackPhase::Active => {
                    self.phase = AttackPhase::Recovery;
                    out.push(AttackEvent::HitWindowClosed { step: self.step });
                }
                AttackPhase::Recovery => {
                    self.finish(&mut out);
                    break;
                }
                AttackPhase::Idle => break,
            }
        }
        out
    }

    /// Abort the swing immediately — death, a stagger, a cutscene. No
    /// [`AttackEvent::Finished`] is produced: nothing *completed*.
    pub fn cancel(&mut self) {
        self.phase = AttackPhase::Idle;
        self.step = 0;
        self.time = 0.0;
        self.buffered = false;
        self.hits.clear();
    }

    /// Entities this swing has already struck (cleared when the window opens).
    pub fn hits(&self) -> &[Entity] {
        &self.hits
    }

    /// Whether `entity` was already struck by the swing in progress.
    pub fn already_hit(&self, entity: Entity) -> bool {
        self.hits.contains(&entity)
    }

    /// Record `entity` as struck by the swing in progress, so a later tick of the
    /// same window skips it. Idempotent.
    pub fn mark_hit(&mut self, entity: Entity) {
        if !self.already_hit(entity) {
            self.hits.push(entity);
        }
    }

    /// Test this tick's arc and return the targets struck **for the first time**
    /// by the current swing, marking them as hit.
    ///
    /// Returns empty unless the hit window is open. `targets` yields
    /// `(entity, position on XZ, radius, team)` — the position is passed in
    /// rather than read from a component, see [`BodyCircle`](super::BodyCircle).
    ///
    /// The caller turns the result into
    /// [`DamageEvent`](super::DamageEvent)s; this function deliberately does not,
    /// so that a game can scale damage (crits, buffs) between the two.
    pub fn resolve_hits<I>(
        &mut self,
        chain: &ComboChain,
        origin: Vec2,
        facing: Vec2,
        attacker_team: Team,
        targets: I,
    ) -> Vec<Entity>
    where
        I: IntoIterator<Item = (Entity, Vec2, f32, Team)>,
    {
        if !self.is_hit_window_open() {
            return Vec::new();
        }
        let Some(spec) = chain.get(self.step) else {
            return Vec::new();
        };
        let hits = &self.hits;
        let fresh = resolve_arc_hits(
            origin,
            facing,
            spec,
            targets.into_iter().filter(|(e, ..)| !hits.contains(e)),
            attacker_team,
        );
        self.hits.extend_from_slice(&fresh);
        fresh
    }

    /// Start `step` at its windup.
    fn begin(&mut self, step: usize) {
        self.step = step;
        self.phase = AttackPhase::Windup;
        self.time = 0.0;
        self.buffered = false;
        self.hits.clear();
    }

    /// End the chain: the combo window expired, so the next input opens at
    /// step 0 again.
    fn finish(&mut self, out: &mut AttackEvents) {
        self.cancel();
        out.push(AttackEvent::Finished);
    }
}

#[cfg(test)]
mod tests {
    use super::AttackEvent::*;
    use super::*;
    use dreamcoast_scene::World;

    /// Three steps with easy round numbers: windup 0.2, active 0.1, recovery 0.3.
    fn chain() -> ComboChain {
        ComboChain::new(
            (0..3)
                .map(|i| AttackSpec {
                    name: format!("step{i}"),
                    damage: 10.0 + i as f32,
                    range: 2.0,
                    half_angle_rad: std::f32::consts::FRAC_PI_4,
                    windup: 0.2,
                    active: 0.1,
                    recovery: 0.3,
                    stagger: 0.1,
                })
                .collect(),
        )
    }

    /// Run `dt` repeatedly, collecting every event.
    fn run(state: &mut AttackState, chain: &ComboChain, dt: f32, ticks: usize) -> Vec<AttackEvent> {
        let mut all = Vec::new();
        for _ in 0..ticks {
            all.extend(state.tick(chain, dt));
        }
        all
    }

    #[test]
    fn single_swing_plays_all_three_phases() {
        let chain = chain();
        let mut s = AttackState::new();
        assert!(s.request(&chain));
        assert_eq!(s.phase(), AttackPhase::Windup);
        // 0.2 windup = 12 ticks at 1/60; the window opens on the 12th.
        let events = run(&mut s, &chain, 1.0 / 60.0, 12);
        assert_eq!(events, vec![HitWindowOpen { step: 0 }]);
        assert!(s.is_hit_window_open());
        // 0.1 active = 6 more ticks.
        let events = run(&mut s, &chain, 1.0 / 60.0, 6);
        assert_eq!(events, vec![HitWindowClosed { step: 0 }]);
        assert_eq!(s.phase(), AttackPhase::Recovery);
        // 0.3 recovery = 18 more ticks; nothing buffered, so the chain ends.
        let events = run(&mut s, &chain, 1.0 / 60.0, 18);
        assert_eq!(events, vec![Finished]);
        assert!(!s.is_attacking());
        assert_eq!(s.step(), 0, "expiry resets to the opener");
    }

    #[test]
    fn input_during_windup_is_ignored() {
        let chain = chain();
        let mut s = AttackState::new();
        s.request(&chain);
        s.tick(&chain, 0.05);
        assert_eq!(s.phase(), AttackPhase::Windup);
        assert!(!s.request(&chain), "windup does not accept a link");
        assert!(!s.has_buffered_input());
    }

    #[test]
    fn input_during_active_links_after_the_window_closes() {
        let chain = chain();
        let mut s = AttackState::new();
        s.request(&chain);
        s.tick(&chain, 0.2); // -> active
        assert!(s.is_hit_window_open());
        assert!(s.request(&chain), "active is inside the combo window");
        assert!(s.has_buffered_input());
        // The window still plays out in full before the link happens.
        let events = s.tick(&chain, 0.1);
        assert_eq!(
            events.iter().collect::<Vec<_>>(),
            vec![HitWindowClosed { step: 0 }, ComboAdvanced { step: 1 }]
        );
        assert_eq!(s.step(), 1);
        assert_eq!(s.phase(), AttackPhase::Windup);
        assert!(
            !s.has_buffered_input(),
            "the buffer holds exactly one input"
        );
    }

    #[test]
    fn input_during_recovery_cancels_the_rest_of_it() {
        let chain = chain();
        let mut s = AttackState::new();
        s.request(&chain);
        s.tick(&chain, 0.3); // windup + active -> recovery
        assert_eq!(s.phase(), AttackPhase::Recovery);
        assert!(s.request(&chain));
        // 0.25s of recovery are still owed, but the link fires on the next tick.
        let events = s.tick(&chain, 1.0 / 60.0);
        assert!(events.contains(ComboAdvanced { step: 1 }));
        assert_eq!(s.step(), 1);
        assert_eq!(s.phase(), AttackPhase::Windup);
    }

    #[test]
    fn input_after_the_window_expires_restarts_at_step_zero() {
        let chain = chain();
        let mut s = AttackState::new();
        s.request(&chain);
        run(&mut s, &chain, 1.0 / 60.0, 40); // whole swing, nothing buffered
        assert!(!s.is_attacking());
        // Too late to link: this is a fresh opener.
        assert!(s.request(&chain));
        assert_eq!(s.step(), 0);
    }

    #[test]
    fn the_whole_chain_links_and_then_ends() {
        let chain = chain();
        let mut s = AttackState::new();
        s.request(&chain);
        let mut steps_seen = vec![0usize];
        for _ in 0..200 {
            // Mash: request every tick. Only active/recovery ticks take it.
            s.request(&chain);
            for e in s.tick(&chain, 1.0 / 60.0) {
                if let ComboAdvanced { step } = e {
                    steps_seen.push(step);
                }
            }
            if !s.is_attacking() {
                break;
            }
        }
        assert_eq!(steps_seen, vec![0, 1, 2], "each step links exactly once");
        // The finisher has nothing to link into, so mashing cannot extend it.
        assert!(!s.is_attacking());
    }

    #[test]
    fn a_huge_dt_crosses_several_phases_in_one_tick() {
        let chain = chain();
        let mut s = AttackState::new();
        s.request(&chain);
        s.request(&chain); // ignored during windup
        let events = s.tick(&chain, 5.0);
        assert_eq!(
            events.iter().collect::<Vec<_>>(),
            vec![
                HitWindowOpen { step: 0 },
                HitWindowClosed { step: 0 },
                Finished
            ]
        );
        assert!(!s.is_attacking());
    }

    #[test]
    fn a_huge_dt_reports_at_most_one_link() {
        let chain = chain();
        let mut s = AttackState::new();
        s.request(&chain);
        s.tick(&chain, 0.2); // -> active
        s.request(&chain); // one buffered input
        let events = s.tick(&chain, 10.0);
        assert_eq!(
            events.iter().collect::<Vec<_>>(),
            vec![
                HitWindowClosed { step: 0 },
                ComboAdvanced { step: 1 },
                HitWindowOpen { step: 1 },
                HitWindowClosed { step: 1 },
                Finished
            ]
        );
        assert!(events.len() <= MAX_ATTACK_EVENTS);
    }

    #[test]
    fn leftover_time_carries_across_phase_boundaries() {
        let chain = chain();
        let mut s = AttackState::new();
        s.request(&chain);
        // 0.25 = the 0.2 windup plus 0.05 into the active window.
        s.tick(&chain, 0.25);
        assert_eq!(s.phase(), AttackPhase::Active);
        assert!((s.time_in_phase() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn zero_and_junk_dt_do_nothing() {
        let chain = chain();
        let mut s = AttackState::new();
        s.request(&chain);
        assert!(s.tick(&chain, 0.0).is_empty());
        assert!(s.tick(&chain, -1.0).is_empty());
        assert!(s.tick(&chain, f32::NAN).is_empty());
        assert_eq!(s.time_in_phase(), 0.0);
        assert_eq!(s.phase(), AttackPhase::Windup);
    }

    #[test]
    fn ticking_while_idle_is_free_and_silent() {
        let chain = chain();
        let mut s = AttackState::new();
        assert!(s.tick(&chain, 1.0).is_empty());
        assert!(!s.is_attacking());
    }

    #[test]
    fn an_empty_chain_cannot_attack() {
        let empty = ComboChain::default();
        let mut s = AttackState::new();
        assert!(!s.request(&empty));
        assert!(!s.is_attacking());
        assert!(s.tick(&empty, 1.0).is_empty());
    }

    #[test]
    fn a_chain_that_shrinks_under_a_live_swing_ends_cleanly() {
        let chain = chain();
        let mut s = AttackState::new();
        s.request(&chain);
        s.tick(&chain, 0.2);
        let shorter = ComboChain::new(Vec::new());
        assert!(s.tick(&shorter, 1.0 / 60.0).contains(Finished));
        assert!(!s.is_attacking());
    }

    #[test]
    fn cancel_drops_the_swing_without_finishing_it() {
        let chain = chain();
        let mut s = AttackState::new();
        s.request(&chain);
        s.tick(&chain, 0.2);
        s.cancel();
        assert!(!s.is_attacking());
        assert_eq!(s.step(), 0);
        assert!(s.tick(&chain, 1.0).is_empty());
    }

    #[test]
    fn multi_frame_window_hits_each_target_once() {
        let chain = chain();
        let mut world = World::new();
        let victim = world.spawn();
        let bystander = world.spawn();
        let mut s = AttackState::new();
        s.request(&chain);
        s.tick(&chain, 0.2); // window opens
        assert!(s.is_hit_window_open());

        let targets = || {
            [
                (victim, Vec2::new(1.0, 0.0), 0.4, Team::ENEMY),
                (bystander, Vec2::new(9.0, 0.0), 0.4, Team::ENEMY),
            ]
        };
        let mut total = Vec::new();
        // Six ticks of a 0.1s window, the victim standing still inside the arc.
        for _ in 0..6 {
            total.extend(s.resolve_hits(&chain, Vec2::ZERO, Vec2::X, Team::PLAYER, targets()));
            s.tick(&chain, 1.0 / 60.0);
        }
        assert_eq!(total, vec![victim], "one hit per target per swing");

        // The next swing's window is a fresh slate.
        s.request(&chain);
        s.tick(&chain, 1.0 / 60.0); // recovery -> link
        s.tick(&chain, 0.2); // windup -> active
        assert!(s.is_hit_window_open());
        let again = s.resolve_hits(&chain, Vec2::ZERO, Vec2::X, Team::PLAYER, targets());
        assert_eq!(again, vec![victim]);
    }

    #[test]
    fn hits_resolve_only_while_the_window_is_open() {
        let chain = chain();
        let mut world = World::new();
        let victim = world.spawn();
        let mut s = AttackState::new();
        let targets = || [(victim, Vec2::new(1.0, 0.0), 0.4, Team::ENEMY)];

        // Idle.
        assert!(
            s.resolve_hits(&chain, Vec2::ZERO, Vec2::X, Team::PLAYER, targets())
                .is_empty()
        );
        // Windup.
        s.request(&chain);
        s.tick(&chain, 0.1);
        assert!(
            s.resolve_hits(&chain, Vec2::ZERO, Vec2::X, Team::PLAYER, targets())
                .is_empty()
        );
        // Active.
        s.tick(&chain, 0.1);
        assert_eq!(
            s.resolve_hits(&chain, Vec2::ZERO, Vec2::X, Team::PLAYER, targets()),
            vec![victim]
        );
        // Recovery.
        s.tick(&chain, 0.1);
        assert!(
            s.resolve_hits(&chain, Vec2::ZERO, Vec2::X, Team::PLAYER, targets())
                .is_empty()
        );
    }

    #[test]
    fn mark_hit_is_idempotent() {
        let mut world = World::new();
        let e = world.spawn();
        let mut s = AttackState::new();
        s.mark_hit(e);
        s.mark_hit(e);
        assert_eq!(s.hits(), &[e]);
        assert!(s.already_hit(e));
    }

    #[test]
    fn spec_reports_the_live_step() {
        let chain = chain();
        let mut s = AttackState::new();
        assert!(s.spec(&chain).is_none());
        s.request(&chain);
        assert_eq!(s.spec(&chain).unwrap().name, "step0");
        s.tick(&chain, 0.3);
        s.request(&chain);
        s.tick(&chain, 1.0 / 60.0);
        assert_eq!(s.spec(&chain).unwrap().name, "step1");
    }

    #[test]
    fn spec_arithmetic() {
        let spec = &chain().0[0];
        assert!((spec.duration() - 0.6).abs() < 1e-6);
        assert!((spec.time_to_hit() - 0.2).abs() < 1e-6);
        assert!((spec.arc_degrees() - 90.0).abs() < 1e-4);
    }

    #[test]
    fn identical_input_sequences_produce_identical_clocks() {
        let chain = chain();
        // A repeating press pattern, long enough to link, drop, and restart.
        let script = [false, true, false, false, true, true, false, false, false];
        let play = || {
            let mut s = AttackState::new();
            let mut log: Vec<String> = Vec::new();
            for frame in 0..120 {
                if script[frame % script.len()] {
                    s.request(&chain);
                }
                let events = s.tick(&chain, 1.0 / 60.0);
                log.push(format!(
                    "{frame} {:?} step={} t={:.6} buffered={} {:?}",
                    s.phase(),
                    s.step(),
                    s.time_in_phase(),
                    s.has_buffered_input(),
                    events.iter().collect::<Vec<_>>(),
                ));
            }
            log
        };
        let first = play();
        assert_eq!(first, play());
        // And the run actually exercised the machine rather than idling.
        assert!(first.iter().any(|l| l.contains("ComboAdvanced")));
        assert!(first.iter().any(|l| l.contains("Finished")));
    }

    #[test]
    fn chain_iterates_and_measures() {
        let c = chain();
        assert_eq!(c.len(), 3);
        assert!(!c.is_empty());
        assert_eq!(c.iter().count(), 3);
        assert_eq!((&c).into_iter().count(), 3);
        assert_eq!(c.as_slice().len(), 3);
        assert!(c.get(3).is_none());
        assert!(ComboChain::default().is_empty());
    }

    #[test]
    fn event_list_reports_its_contents() {
        let chain = chain();
        let mut s = AttackState::new();
        s.request(&chain);
        let events = s.tick(&chain, 0.2);
        assert_eq!(events.len(), 1);
        assert!(!events.is_empty());
        assert!(events.contains(HitWindowOpen { step: 0 }));
        assert!(!events.finished());
        assert_eq!(events.into_iter().count(), 1);
        assert!(AttackEvents::default().is_empty());
    }
}
