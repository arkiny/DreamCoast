//! The graph as data: states, transitions, conditions, and their RON form.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// The reserved `from` name that matches every state.
pub const ANY_STATE: &str = "any";

/// Something went wrong loading or validating an animation graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnimError {
    /// The file could not be read or written.
    Io(String),
    /// The text is not valid RON for this type.
    Ron(String),
    /// A transition (or the initial state) names a state that does not exist.
    UnknownState(String),
    /// The graph parsed but is not usable (see the message).
    Invalid(String),
}

impl std::fmt::Display for AnimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "anim graph I/O failed: {msg}"),
            Self::Ron(msg) => write!(f, "anim graph is not valid RON: {msg}"),
            Self::UnknownState(name) => write!(f, "anim graph references unknown state '{name}'"),
            Self::Invalid(msg) => write!(f, "anim graph is invalid: {msg}"),
        }
    }
}

impl std::error::Error for AnimError {}

/// One node: a clip, how fast it plays, and whether it wraps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimStateDef {
    /// State name, unique within the graph and referenced by transitions.
    pub name: String,
    /// Clip identifier handed back by [`AnimMachine::current`](super::AnimMachine::current).
    /// The integrator maps it to an actual clip; this crate never resolves it.
    pub clip: String,
    /// Playback rate multiplier. Must be `> 0`.
    #[serde(default = "unit_speed")]
    pub speed: f32,
    /// Whether the clip wraps at its end (`true`) or clamps and raises
    /// [`Condition::StateDone`] (`false`).
    #[serde(default)]
    pub looping: bool,
    /// Optional authored clip length in seconds. Seeds the machine's clip-length
    /// table so a graph is usable (and testable) before any real clip is loaded;
    /// [`AnimMachine::set_clip_length`](super::AnimMachine::set_clip_length)
    /// overrides it with the asset's true duration at runtime.
    #[serde(default)]
    pub length: Option<f32>,
}

fn unit_speed() -> f32 {
    1.0
}

fn interruptible_by_default() -> bool {
    true
}

/// A transition guard. Expression-free by design: every variant is a leaf, so a
/// graph file can be read top to bottom without evaluating anything, and the
/// machine cannot become a scripting language by accident. Compound logic is
/// expressed as several transitions, in the order they should be tried.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Condition {
    /// A named boolean the game sets and clears, true while set.
    Flag(String),
    /// The negation of [`Flag`](Self::Flag) — true while the flag is *not* set.
    /// Present because "leave the run state when the player stops moving" is a
    /// leaf question, not a compound one, and expressing it as a second flag the
    /// game must keep in sync is how flags drift apart.
    NotFlag(String),
    /// A named one-shot, **consumed** by the transition that uses it.
    Trigger(String),
    /// The current state is non-looping and has reached the end of its clip.
    StateDone,
}

impl Condition {
    /// The parameter name this condition reads, if any.
    pub fn param(&self) -> Option<&str> {
        match self {
            Self::Flag(name) | Self::NotFlag(name) | Self::Trigger(name) => Some(name),
            Self::StateDone => None,
        }
    }
}

/// One edge: when to leave `from` for `to`, and how long to blend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionDef {
    /// Source state name, or [`ANY_STATE`] (`"any"`, case-insensitive) to match
    /// every state.
    pub from: String,
    /// Destination state name.
    pub to: String,
    /// The guard.
    pub condition: Condition,
    /// Cross-fade length in seconds. `0` cuts.
    #[serde(default)]
    pub fade: f32,
    /// Whether another transition may fire while **this** transition's fade is
    /// still running. `false` locks the machine for `fade` seconds — the way to
    /// say "this blend must be seen", e.g. entering a death state.
    #[serde(default = "interruptible_by_default")]
    pub interruptible: bool,
}

/// A whole animation graph, as data.
///
/// ```ron
/// (
///     initial: "idle",
///     states: [
///         (name: "idle", clip: "idle", looping: true, length: Some(2.0)),
///         (name: "attack", clip: "attack_1", length: Some(0.75)),
///     ],
///     transitions: [
///         (from: "any",    to: "attack", condition: Trigger("attack"), fade: 0.06),
///         (from: "attack", to: "idle",   condition: StateDone,         fade: 0.15),
///     ],
/// )
/// ```
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AnimGraphDef {
    /// The state the machine starts (and [`reset`](super::AnimMachine::reset)s) in.
    pub initial: String,
    /// The states.
    pub states: Vec<AnimStateDef>,
    /// The transitions, **in evaluation order** — see
    /// [`AnimMachine::tick`](super::AnimMachine::tick).
    #[serde(default)]
    pub transitions: Vec<TransitionDef>,
}

impl AnimGraphDef {
    /// Index of the state called `name`.
    pub fn state_index(&self, name: &str) -> Option<usize> {
        self.states.iter().position(|s| s.name == name)
    }

    /// Borrow the state called `name`.
    pub fn state(&self, name: &str) -> Option<&AnimStateDef> {
        self.state_index(name).map(|i| &self.states[i])
    }

    /// Parse RON text, rejecting anything [`validate`](Self::validate) refuses.
    pub fn from_ron(text: &str) -> Result<Self, AnimError> {
        let parsed: Self = ron::from_str(text).map_err(|e| AnimError::Ron(e.to_string()))?;
        parsed.validate()?;
        Ok(parsed)
    }

    /// Serialise to pretty RON text (hand-editable, stable field order).
    pub fn to_ron(&self) -> Result<String, AnimError> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(|e| AnimError::Ron(e.to_string()))
    }

    /// Load a graph file.
    pub fn load_ron(path: impl AsRef<Path>) -> Result<Self, AnimError> {
        let text =
            std::fs::read_to_string(path).map_err(|e| AnimError::Io(format!("read: {e}")))?;
        Self::from_ron(&text)
    }

    /// Write a graph file.
    pub fn save_ron(&self, path: impl AsRef<Path>) -> Result<(), AnimError> {
        std::fs::write(path, self.to_ron()?).map_err(|e| AnimError::Io(format!("write: {e}")))
    }

    /// Check every invariant the machine relies on, so
    /// [`AnimMachine::tick`](super::AnimMachine::tick) can index without
    /// `Option`s and a broken graph fails at load rather than mid-fight.
    pub fn validate(&self) -> Result<(), AnimError> {
        if self.states.is_empty() {
            return Err(AnimError::Invalid("graph has no states".to_string()));
        }
        for (i, state) in self.states.iter().enumerate() {
            if state.name.trim().is_empty() {
                return Err(AnimError::Invalid(format!("state {i} has no name")));
            }
            if state.name.eq_ignore_ascii_case(ANY_STATE) {
                return Err(AnimError::Invalid(format!(
                    "state {i} is named '{ANY_STATE}', which is reserved for transitions"
                )));
            }
            if state.clip.trim().is_empty() {
                return Err(AnimError::Invalid(format!(
                    "state '{}' has no clip",
                    state.name
                )));
            }
            if !state.speed.is_finite() || state.speed <= 0.0 {
                return Err(AnimError::Invalid(format!(
                    "state '{}': speed must be finite and > 0",
                    state.name
                )));
            }
            if let Some(length) = state.length
                && (!length.is_finite() || length < 0.0)
            {
                return Err(AnimError::Invalid(format!(
                    "state '{}': length must be finite and >= 0",
                    state.name
                )));
            }
            if self.states[..i].iter().any(|s| s.name == state.name) {
                return Err(AnimError::Invalid(format!(
                    "duplicate state name '{}'",
                    state.name
                )));
            }
        }
        if self.state_index(&self.initial).is_none() {
            return Err(AnimError::UnknownState(self.initial.clone()));
        }
        for (i, t) in self.transitions.iter().enumerate() {
            if !t.from.eq_ignore_ascii_case(ANY_STATE) && self.state_index(&t.from).is_none() {
                return Err(AnimError::UnknownState(t.from.clone()));
            }
            if self.state_index(&t.to).is_none() {
                return Err(AnimError::UnknownState(t.to.clone()));
            }
            if !t.fade.is_finite() || t.fade < 0.0 {
                return Err(AnimError::Invalid(format!(
                    "transition {i} ({} -> {}): fade must be finite and >= 0",
                    t.from, t.to
                )));
            }
            if t.condition.param().is_some_and(|p| p.trim().is_empty()) {
                return Err(AnimError::Invalid(format!(
                    "transition {i} ({} -> {}): condition parameter is empty",
                    t.from, t.to
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> AnimGraphDef {
        AnimGraphDef {
            initial: "idle".to_string(),
            states: vec![
                AnimStateDef {
                    name: "idle".to_string(),
                    clip: "idle".to_string(),
                    speed: 1.0,
                    looping: true,
                    length: Some(2.0),
                },
                AnimStateDef {
                    name: "attack".to_string(),
                    clip: "attack_1".to_string(),
                    speed: 1.2,
                    looping: false,
                    length: Some(0.75),
                },
            ],
            transitions: vec![
                TransitionDef {
                    from: ANY_STATE.to_string(),
                    to: "attack".to_string(),
                    condition: Condition::Trigger("attack".to_string()),
                    fade: 0.06,
                    interruptible: false,
                },
                TransitionDef {
                    from: "attack".to_string(),
                    to: "idle".to_string(),
                    condition: Condition::StateDone,
                    fade: 0.15,
                    interruptible: true,
                },
            ],
        }
    }

    #[test]
    fn ron_round_trips() {
        let g = graph();
        let text = g.to_ron().unwrap();
        let parsed = AnimGraphDef::from_ron(&text).unwrap();
        assert_eq!(parsed, g);
        assert_eq!(parsed.to_ron().unwrap(), text, "serialisation is stable");
    }

    #[test]
    fn hand_written_ron_takes_the_documented_defaults() {
        let text = r#"(
            initial: "idle",
            states: [
                (name: "idle", clip: "idle", looping: true),
                (name: "run", clip: "run", speed: 1.4, looping: true),
            ],
            transitions: [
                (from: "idle", to: "run", condition: Flag("moving")),
                (from: "run", to: "idle", condition: NotFlag("moving"), fade: 0.1),
            ],
        )"#;
        let g = AnimGraphDef::from_ron(text).unwrap();
        assert_eq!(g.state("idle").unwrap().speed, 1.0, "speed defaults to 1");
        assert_eq!(g.state("idle").unwrap().length, None);
        assert_eq!(g.transitions[0].fade, 0.0, "fade defaults to a cut");
        assert!(
            g.transitions[0].interruptible,
            "transitions are interruptible unless they say otherwise"
        );
    }

    #[test]
    fn lookups_by_name() {
        let g = graph();
        assert_eq!(g.state_index("attack"), Some(1));
        assert_eq!(g.state("attack").unwrap().clip, "attack_1");
        assert!(g.state("nope").is_none());
    }

    #[test]
    fn conditions_report_their_parameter() {
        assert_eq!(Condition::Flag("a".into()).param(), Some("a"));
        assert_eq!(Condition::NotFlag("b".into()).param(), Some("b"));
        assert_eq!(Condition::Trigger("c".into()).param(), Some("c"));
        assert_eq!(Condition::StateDone.param(), None);
    }

    #[test]
    fn validation_rejects_broken_graphs() {
        /// One way to break a graph.
        type Break = Box<dyn Fn(&mut AnimGraphDef)>;

        let cases: Vec<(&str, Break)> = vec![
            (
                "no states",
                Box::new(|g: &mut AnimGraphDef| g.states.clear()),
            ),
            (
                "unnamed state",
                Box::new(|g: &mut AnimGraphDef| g.states[0].name.clear()),
            ),
            (
                "reserved name",
                Box::new(|g: &mut AnimGraphDef| g.states[1].name = "Any".to_string()),
            ),
            (
                "clipless state",
                Box::new(|g: &mut AnimGraphDef| g.states[0].clip.clear()),
            ),
            (
                "zero speed",
                Box::new(|g: &mut AnimGraphDef| g.states[0].speed = 0.0),
            ),
            (
                "nan length",
                Box::new(|g: &mut AnimGraphDef| g.states[0].length = Some(f32::NAN)),
            ),
            (
                "duplicate names",
                Box::new(|g: &mut AnimGraphDef| g.states[1].name = "idle".to_string()),
            ),
            (
                "negative fade",
                Box::new(|g: &mut AnimGraphDef| g.transitions[0].fade = -1.0),
            ),
            (
                "empty parameter",
                Box::new(|g: &mut AnimGraphDef| {
                    g.transitions[0].condition = Condition::Flag(" ".to_string())
                }),
            ),
        ];
        for (label, break_it) in cases {
            let mut g = graph();
            break_it(&mut g);
            assert!(g.validate().is_err(), "{label} should not validate");
        }
    }

    #[test]
    fn validation_rejects_dangling_state_references() {
        let mut g = graph();
        g.initial = "nowhere".to_string();
        assert_eq!(
            g.validate().unwrap_err(),
            AnimError::UnknownState("nowhere".to_string())
        );

        let mut g = graph();
        g.transitions[1].from = "ghost".to_string();
        assert_eq!(
            g.validate().unwrap_err(),
            AnimError::UnknownState("ghost".to_string())
        );

        let mut g = graph();
        g.transitions[0].to = "ghost".to_string();
        assert!(g.validate().is_err());
    }

    #[test]
    fn file_round_trip_and_error_text() {
        let dir = std::env::temp_dir().join(format!("dreamcoast-game-anim-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("graph.ron");
        graph().save_ron(&path).unwrap();
        assert_eq!(AnimGraphDef::load_ron(&path).unwrap(), graph());
        std::fs::remove_dir_all(&dir).ok();

        assert!(matches!(
            AnimGraphDef::load_ron(dir.join("missing.ron")).unwrap_err(),
            AnimError::Io(_)
        ));
        assert!(matches!(
            AnimGraphDef::from_ron("not ron").unwrap_err(),
            AnimError::Ron(_)
        ));
        assert!(
            AnimError::UnknownState("x".into())
                .to_string()
                .contains("unknown state 'x'")
        );
        assert!(AnimError::Io("y".into()).to_string().contains("I/O"));
        assert!(AnimError::Invalid("z".into()).to_string().contains('z'));
    }
}
