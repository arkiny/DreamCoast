//! [`Params`] — the game's side of the conversation with an [`AnimMachine`](super::AnimMachine).

use std::collections::BTreeSet;

/// The parameter set a graph's conditions are evaluated against: named flags and
/// named triggers.
///
/// **Flags are level, triggers are edge.** A flag describes a state of the world
/// the game keeps up to date every tick (`"moving"`, `"airborne"`, `"dead"`); a
/// trigger describes an event that happened once (`"attack"`, `"hit"`) and is
/// consumed by the transition that acts on it. Anything the game can answer every
/// tick should be a flag — triggers exist for things that have no "still
/// happening" state to read.
///
/// Both are stored in ordered sets, so iteration is deterministic and two runs of
/// the same input produce the same graph traversal.
///
/// ```
/// use dreamcoast_game::anim::Params;
///
/// let mut params = Params::new();
/// params.set_flag("moving", true);
/// assert!(params.flag("moving"));
///
/// params.trigger("attack");
/// assert!(params.is_triggered("attack"));
/// assert!(params.consume_trigger("attack"));
/// assert!(!params.is_triggered("attack"), "a trigger fires once");
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Params {
    /// Flags currently set. Absent means false, so clearing is a removal.
    flags: BTreeSet<String>,
    /// Triggers waiting to be consumed.
    triggers: BTreeSet<String>,
}

impl Params {
    /// An empty parameter set: every flag false, no trigger pending.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set or clear `name`.
    pub fn set_flag(&mut self, name: &str, value: bool) {
        if value {
            if !self.flags.contains(name) {
                self.flags.insert(name.to_string());
            }
        } else {
            self.flags.remove(name);
        }
    }

    /// Set `name` (chainable, for building a set in one expression).
    #[must_use]
    pub fn with_flag(mut self, name: &str, value: bool) -> Self {
        self.set_flag(name, value);
        self
    }

    /// Whether `name` is set.
    pub fn flag(&self, name: &str) -> bool {
        self.flags.contains(name)
    }

    /// Fire the one-shot `name`. Firing twice before it is consumed still yields
    /// one transition — a trigger is a latch, not a counter.
    pub fn trigger(&mut self, name: &str) {
        if !self.triggers.contains(name) {
            self.triggers.insert(name.to_string());
        }
    }

    /// Whether `name` is pending.
    pub fn is_triggered(&self, name: &str) -> bool {
        self.triggers.contains(name)
    }

    /// Consume `name`, reporting whether it was pending. The machine calls this
    /// for the trigger of a transition it takes.
    pub fn consume_trigger(&mut self, name: &str) -> bool {
        self.triggers.remove(name)
    }

    /// Drop every pending trigger.
    ///
    /// A trigger deliberately **survives** a tick that did not use it (an attack
    /// pressed one tick before the state that answers it becomes reachable still
    /// fires, which is the input-buffering behaviour players expect). A game that
    /// wants strict same-tick semantics calls this at the end of its tick
    /// instead.
    pub fn clear_triggers(&mut self) {
        self.triggers.clear();
    }

    /// Drop every flag and every pending trigger — e.g. on a level swap, so
    /// stale state cannot cross the transition.
    pub fn clear(&mut self) {
        self.flags.clear();
        self.triggers.clear();
    }

    /// The set flags, in name order.
    pub fn flags(&self) -> impl Iterator<Item = &str> {
        self.flags.iter().map(String::as_str)
    }

    /// The pending triggers, in name order.
    pub fn triggers(&self) -> impl Iterator<Item = &str> {
        self.triggers.iter().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_are_level_triggered() {
        let mut p = Params::new();
        assert!(!p.flag("moving"));
        p.set_flag("moving", true);
        p.set_flag("moving", true); // idempotent
        assert!(p.flag("moving"));
        p.set_flag("moving", false);
        assert!(!p.flag("moving"));
        assert_eq!(p.flags().count(), 0);
    }

    #[test]
    fn triggers_are_one_shot_latches() {
        let mut p = Params::new();
        assert!(!p.consume_trigger("attack"));
        p.trigger("attack");
        p.trigger("attack"); // still one latch
        assert!(p.consume_trigger("attack"));
        assert!(!p.consume_trigger("attack"));
    }

    #[test]
    fn an_unused_trigger_survives_the_tick() {
        let mut p = Params::new();
        p.trigger("attack");
        // Nothing consumed it; it is still pending next tick (input buffering).
        assert!(p.is_triggered("attack"));
        p.clear_triggers();
        assert!(!p.is_triggered("attack"));
    }

    #[test]
    fn iteration_is_name_ordered() {
        let p = Params::new()
            .with_flag("zulu", true)
            .with_flag("alpha", true)
            .with_flag("mike", true);
        assert_eq!(p.flags().collect::<Vec<_>>(), vec!["alpha", "mike", "zulu"]);
    }

    #[test]
    fn clear_drops_everything() {
        let mut p = Params::new().with_flag("moving", true);
        p.trigger("attack");
        p.clear();
        assert!(!p.flag("moving"));
        assert!(!p.is_triggered("attack"));
        assert_eq!(p.triggers().count(), 0);
        assert_eq!(p, Params::new());
    }
}
