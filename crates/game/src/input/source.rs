//! A single physical input a game action can be bound to.

use std::fmt;

use dreamcoast_platform::InputSnapshot;
use dreamcoast_platform::keys::{key_name, key_vk};

/// Which way the wheel turned. The wheel has no held state, so it is exposed as a
/// pair of momentary sources: each is "active" only on the frame the wheel moved
/// that way, which makes it behave like a key tap under the edge detector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WheelDirection {
    /// Wheel rolled away from the user (positive notches).
    Up,
    /// Wheel rolled toward the user (negative notches).
    Down,
}

/// One physical input: a key, a mouse button, or a wheel direction.
///
/// Actions bind to a *list* of these, so `Jump` can be `Space` **or** `Mouse3`
/// without the game knowing which one fired.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputSource {
    /// A keyboard key, as a Win32 virtual-key code (see [`super::keys`]).
    Key(u16),
    /// A mouse button: 0 = left, 1 = right, 2 = middle.
    MouseButton(u8),
    /// A wheel direction — momentary, active only on the frame it moved.
    Wheel(WheelDirection),
}

impl InputSource {
    /// A key source by name (`"W"`, `"Space"`), for terse binding tables.
    pub fn key(name: &str) -> Option<Self> {
        key_vk(name).map(Self::Key)
    }

    /// Parse a binding source name.
    ///
    /// Accepts every key name [`key_vk`] understands, the mouse buttons
    /// (`"Mouse1"`/`"MouseLeft"`/`"LMB"`, `"Mouse2"`/`"MouseRight"`/`"RMB"`,
    /// `"Mouse3"`/`"MouseMiddle"`/`"MMB"`), and `"WheelUp"` / `"WheelDown"`
    /// (`"ScrollUp"` / `"ScrollDown"`). Case-insensitive.
    pub fn from_name(name: &str) -> Option<Self> {
        let name = name.trim();
        match name.to_ascii_lowercase().as_str() {
            "mouseleft" | "lmb" => return Some(Self::MouseButton(0)),
            "mouseright" | "rmb" => return Some(Self::MouseButton(1)),
            "mousemiddle" | "mmb" => return Some(Self::MouseButton(2)),
            "wheelup" | "scrollup" => return Some(Self::Wheel(WheelDirection::Up)),
            "wheeldown" | "scrolldown" => return Some(Self::Wheel(WheelDirection::Down)),
            lower => {
                // `Mouse<N>`: 1-based to match how players count buttons.
                if let Some(digits) = lower.strip_prefix("mouse") {
                    if let Ok(n) = digits.parse::<u8>()
                        && n >= 1
                    {
                        return Some(Self::MouseButton(n - 1));
                    }
                    return None;
                }
            }
        }
        Self::key(name)
    }

    /// The canonical name for this source — the spelling a config serializes to.
    ///
    /// Always parses back to the same source, including for unnamed key codes
    /// (which fall back to the `vk:0x..` escape hatch).
    pub fn name(self) -> String {
        match self {
            Self::Key(vk) => match key_name(vk) {
                Some(name) => name.to_string(),
                None => format!("vk:0x{vk:02X}"),
            },
            Self::MouseButton(b) => format!("Mouse{}", b as u16 + 1),
            Self::Wheel(WheelDirection::Up) => "WheelUp".to_string(),
            Self::Wheel(WheelDirection::Down) => "WheelDown".to_string(),
        }
    }

    /// Whether this source is active in the given frame.
    pub fn is_active(self, snapshot: &InputSnapshot) -> bool {
        match self {
            Self::Key(vk) => snapshot.key(vk),
            Self::MouseButton(b) => snapshot.mouse_button(b),
            Self::Wheel(WheelDirection::Up) => snapshot.wheel() > 0.0,
            Self::Wheel(WheelDirection::Down) => snapshot.wheel() < 0.0,
        }
    }

    /// The source's analog magnitude this frame.
    ///
    /// Digital sources report `1.0` while held; a wheel source reports how far the
    /// wheel turned in its direction (in notches), so a scroll-bound action can
    /// drive a continuous value instead of a tap.
    pub fn analog(self, snapshot: &InputSnapshot) -> f32 {
        match self {
            Self::Wheel(WheelDirection::Up) => snapshot.wheel().max(0.0),
            Self::Wheel(WheelDirection::Down) => (-snapshot.wheel()).max(0.0),
            other => {
                if other.is_active(snapshot) {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

impl fmt::Display for InputSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

/// What can go wrong turning binding *data* into a usable [`super::ActionMap`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingError {
    /// A source name in the config matched no key, mouse button, or wheel axis.
    UnknownSource {
        /// The action the bad source was listed under.
        action: String,
        /// The unparsable source name.
        source: String,
    },
    /// An action name the game's resolver did not recognize.
    UnknownAction(String),
    /// The RON text could not be parsed or written.
    Ron(String),
    /// The bindings file could not be read or written.
    Io(String),
}

impl fmt::Display for BindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSource { action, source } => {
                write!(
                    f,
                    "unknown input source '{source}' bound to action '{action}'"
                )
            }
            Self::UnknownAction(a) => write!(f, "unknown action '{a}'"),
            Self::Ron(e) => write!(f, "bindings RON: {e}"),
            Self::Io(e) => write!(f, "bindings file: {e}"),
        }
    }
}

impl std::error::Error for BindingError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_keys_buttons_and_wheel() {
        assert_eq!(InputSource::from_name("W"), Some(InputSource::Key(0x57)));
        assert_eq!(
            InputSource::from_name("Space"),
            Some(InputSource::Key(0x20))
        );
        assert_eq!(
            InputSource::from_name("Mouse1"),
            Some(InputSource::MouseButton(0))
        );
        assert_eq!(
            InputSource::from_name("mouse3"),
            Some(InputSource::MouseButton(2))
        );
        assert_eq!(
            InputSource::from_name("RMB"),
            Some(InputSource::MouseButton(1))
        );
        assert_eq!(
            InputSource::from_name("WheelUp"),
            Some(InputSource::Wheel(WheelDirection::Up))
        );
        assert_eq!(
            InputSource::from_name("scrolldown"),
            Some(InputSource::Wheel(WheelDirection::Down))
        );
        assert_eq!(
            InputSource::from_name("Mouse0"),
            None,
            "buttons are 1-based"
        );
        assert_eq!(InputSource::from_name("Mousey"), None);
        assert_eq!(InputSource::from_name("Nonsense"), None);
    }

    #[test]
    fn every_source_kind_round_trips_through_its_name() {
        for source in [
            InputSource::Key(0x57),
            InputSource::Key(0x20),
            InputSource::Key(0xA0),
            InputSource::Key(0x71),
            InputSource::Key(0x0C), // unnamed code -> vk: escape hatch
            InputSource::MouseButton(0),
            InputSource::MouseButton(2),
            InputSource::Wheel(WheelDirection::Up),
            InputSource::Wheel(WheelDirection::Down),
        ] {
            let name = source.name();
            assert_eq!(InputSource::from_name(&name), Some(source), "{name}");
        }
    }

    #[test]
    fn activity_reads_the_snapshot() {
        let snap = InputSnapshot::default()
            .with_key(0x57, true)
            .with_mouse_button(1, true)
            .with_wheel(2.5);
        assert!(InputSource::Key(0x57).is_active(&snap));
        assert!(!InputSource::Key(0x53).is_active(&snap));
        assert!(InputSource::MouseButton(1).is_active(&snap));
        assert!(!InputSource::MouseButton(0).is_active(&snap));
        assert!(InputSource::Wheel(WheelDirection::Up).is_active(&snap));
        assert!(!InputSource::Wheel(WheelDirection::Down).is_active(&snap));
    }

    #[test]
    fn wheel_analog_reports_notches_per_direction() {
        let up = InputSnapshot::default().with_wheel(3.0);
        assert_eq!(InputSource::Wheel(WheelDirection::Up).analog(&up), 3.0);
        assert_eq!(InputSource::Wheel(WheelDirection::Down).analog(&up), 0.0);

        let down = InputSnapshot::default().with_wheel(-1.5);
        assert_eq!(InputSource::Wheel(WheelDirection::Up).analog(&down), 0.0);
        assert_eq!(InputSource::Wheel(WheelDirection::Down).analog(&down), 1.5);

        let held = InputSnapshot::default().with_key(0x57, true);
        assert_eq!(InputSource::Key(0x57).analog(&held), 1.0);
        assert_eq!(InputSource::Key(0x41).analog(&held), 0.0);
    }
}
