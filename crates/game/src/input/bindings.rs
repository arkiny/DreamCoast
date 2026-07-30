//! Data-driven bindings: `bindings.ron` ⇄ [`ActionMap`].

use std::collections::BTreeMap;
use std::hash::Hash;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::action_map::ActionMap;
use super::source::{BindingError, InputSource};

/// Bindings as plain data: action *name* → source names.
///
/// The framework cannot know a game's action enum, so the on-disk form is all
/// strings and the game supplies the name → action resolver ([`Self::resolve`]).
/// That keeps `bindings.ron` hand-editable and lets a title ship remappable
/// controls without any framework change.
///
/// ```ron
/// (
///     actions: {
///         "MoveForward": ["W", "Up"],
///         "Attack": ["Mouse1"],
///         "ZoomIn": ["WheelUp"],
///     },
/// )
/// ```
///
/// A [`BTreeMap`] (not a hash map) keeps the serialized order stable, so a
/// rewritten config diffs cleanly and the round-trip is byte-reproducible.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BindingsConfig {
    /// Action name → the source names bound to it.
    pub actions: BTreeMap<String, Vec<String>>,
}

impl BindingsConfig {
    /// An empty config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a source name to an action (chainable).
    #[must_use]
    pub fn with(mut self, action: &str, source: &str) -> Self {
        self.bind(action, source);
        self
    }

    /// Append a source name to an action.
    pub fn bind(&mut self, action: &str, source: &str) {
        self.actions
            .entry(action.to_string())
            .or_default()
            .push(source.to_string());
    }

    /// Parse RON text.
    pub fn from_ron(text: &str) -> Result<Self, BindingError> {
        ron::from_str(text).map_err(|e| BindingError::Ron(e.to_string()))
    }

    /// Serialize to pretty RON text (hand-editable).
    pub fn to_ron(&self) -> Result<String, BindingError> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(|e| BindingError::Ron(e.to_string()))
    }

    /// Load a `bindings.ron` file.
    pub fn load_ron(path: impl AsRef<Path>) -> Result<Self, BindingError> {
        let text =
            std::fs::read_to_string(path).map_err(|e| BindingError::Io(format!("read: {e}")))?;
        Self::from_ron(&text)
    }

    /// Write a `bindings.ron` file.
    pub fn save_ron(&self, path: impl AsRef<Path>) -> Result<(), BindingError> {
        std::fs::write(path, self.to_ron()?).map_err(|e| BindingError::Io(format!("write: {e}")))
    }

    /// Resolve this config into a usable [`ActionMap`].
    ///
    /// `action_of` maps an action *name* to the game's enum; returning `None`
    /// rejects the file with [`BindingError::UnknownAction`] rather than silently
    /// dropping a binding a player typed. Unparsable source names likewise fail
    /// loudly with [`BindingError::UnknownSource`].
    pub fn resolve<A, F>(&self, mut action_of: F) -> Result<ActionMap<A>, BindingError>
    where
        A: Copy + Eq + Hash,
        F: FnMut(&str) -> Option<A>,
    {
        let mut map = ActionMap::new();
        for (name, sources) in &self.actions {
            let action =
                action_of(name).ok_or_else(|| BindingError::UnknownAction(name.clone()))?;
            for source in sources {
                let parsed =
                    InputSource::from_name(source).ok_or_else(|| BindingError::UnknownSource {
                        action: name.clone(),
                        source: source.clone(),
                    })?;
                map.bind(action, parsed);
            }
        }
        Ok(map)
    }

    /// Build a config from a live map, so a remapping UI can write the player's
    /// bindings back out. `name_of` is the inverse of `resolve`'s resolver.
    pub fn from_action_map<A, F>(map: &ActionMap<A>, mut name_of: F) -> Result<Self, BindingError>
    where
        A: Copy + Eq + Hash,
        F: FnMut(A) -> Option<String>,
    {
        let mut config = Self::new();
        for (action, sources) in map.iter() {
            let name = name_of(action)
                .ok_or_else(|| BindingError::UnknownAction("<unnamed action>".to_string()))?;
            let entry = config.actions.entry(name).or_default();
            for source in sources {
                entry.push(source.name());
            }
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{InputSnapshot, WheelDirection};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum Action {
        MoveForward,
        MoveLeft,
        Attack,
        ZoomIn,
    }

    impl Action {
        fn from_name(name: &str) -> Option<Self> {
            match name {
                "MoveForward" => Some(Self::MoveForward),
                "MoveLeft" => Some(Self::MoveLeft),
                "Attack" => Some(Self::Attack),
                "ZoomIn" => Some(Self::ZoomIn),
                _ => None,
            }
        }

        fn name(self) -> String {
            match self {
                Self::MoveForward => "MoveForward",
                Self::MoveLeft => "MoveLeft",
                Self::Attack => "Attack",
                Self::ZoomIn => "ZoomIn",
            }
            .to_string()
        }
    }

    fn sample() -> BindingsConfig {
        BindingsConfig::new()
            .with("MoveForward", "W")
            .with("MoveForward", "Up")
            .with("MoveLeft", "A")
            .with("Attack", "Mouse1")
            .with("Attack", "Space")
            .with("ZoomIn", "WheelUp")
    }

    #[test]
    fn ron_text_round_trips() {
        let config = sample();
        let text = config.to_ron().unwrap();
        let parsed = BindingsConfig::from_ron(&text).unwrap();
        assert_eq!(parsed, config);
        // Serialization is stable, so the second pass is byte-identical.
        assert_eq!(parsed.to_ron().unwrap(), text);
    }

    #[test]
    fn hand_written_ron_parses() {
        let text = r#"(
            actions: {
                "MoveForward": ["W", "Up"],
                "Attack": ["Mouse1"],
            },
        )"#;
        let config = BindingsConfig::from_ron(text).unwrap();
        assert_eq!(config.actions["MoveForward"], vec!["W", "Up"]);
        assert_eq!(config.actions["Attack"], vec!["Mouse1"]);
    }

    #[test]
    fn resolves_to_a_working_map() {
        let map = sample().resolve(Action::from_name).unwrap();
        assert_eq!(
            map.sources(Action::MoveForward),
            &[InputSource::Key(0x57), InputSource::Key(0x26)]
        );
        assert_eq!(
            map.sources(Action::Attack),
            &[InputSource::MouseButton(0), InputSource::Key(0x20)]
        );
        assert_eq!(
            map.sources(Action::ZoomIn),
            &[InputSource::Wheel(WheelDirection::Up)]
        );

        let snap = InputSnapshot::default().with_key(0x26, true);
        assert!(map.is_active(Action::MoveForward, &snap));
        assert!(!map.is_active(Action::MoveLeft, &snap));
    }

    #[test]
    fn map_to_config_to_map_is_lossless() {
        let map = sample().resolve(Action::from_name).unwrap();
        let config = BindingsConfig::from_action_map(&map, |a| Some(a.name())).unwrap();
        let text = config.to_ron().unwrap();
        let back = BindingsConfig::from_ron(&text)
            .unwrap()
            .resolve(Action::from_name)
            .unwrap();
        for action in [
            Action::MoveForward,
            Action::MoveLeft,
            Action::Attack,
            Action::ZoomIn,
        ] {
            assert_eq!(back.sources(action), map.sources(action), "{action:?}");
        }
    }

    #[test]
    fn unknown_names_fail_loudly() {
        let err = BindingsConfig::new()
            .with("Fly", "W")
            .resolve(Action::from_name)
            .unwrap_err();
        assert_eq!(err, BindingError::UnknownAction("Fly".to_string()));

        let err = BindingsConfig::new()
            .with("Attack", "Frobnicate")
            .resolve(Action::from_name)
            .unwrap_err();
        assert_eq!(
            err,
            BindingError::UnknownSource {
                action: "Attack".to_string(),
                source: "Frobnicate".to_string(),
            }
        );
        assert!(err.to_string().contains("Frobnicate"));

        assert!(matches!(
            BindingsConfig::from_ron("not ron").unwrap_err(),
            BindingError::Ron(_)
        ));
        assert!(matches!(
            BindingsConfig::from_action_map(&sample().resolve(Action::from_name).unwrap(), |_| {
                None
            })
            .unwrap_err(),
            BindingError::UnknownAction(_)
        ));
    }

    #[test]
    fn file_round_trip() {
        let dir =
            std::env::temp_dir().join(format!("dreamcoast-game-bindings-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bindings.ron");
        sample().save_ron(&path).unwrap();
        assert_eq!(BindingsConfig::load_ron(&path).unwrap(), sample());
        std::fs::remove_dir_all(&dir).ok();

        assert!(matches!(
            BindingsConfig::load_ron(dir.join("missing.ron")).unwrap_err(),
            BindingError::Io(_)
        ));
    }

    #[test]
    fn empty_config_resolves_to_an_empty_map() {
        let map: ActionMap<Action> = BindingsConfig::new().resolve(Action::from_name).unwrap();
        assert!(map.is_empty());
        assert!(map.sources(Action::Attack).is_empty());
    }
}
