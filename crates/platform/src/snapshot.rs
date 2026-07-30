//! A plain-data capture of one frame of platform input.

use std::fmt;

use crate::keys::key_name;

/// Mouse buttons the platform layer tracks: left, right, middle.
pub const MOUSE_BUTTON_COUNT: usize = 3;

/// Number of virtual-key slots the platform layer keeps.
pub const KEY_COUNT: usize = 256;

/// One frame of keyboard + mouse state, detached from the window.
///
/// This is the engine's **input test seam**. [`crate::Input`] can only be filled by a
/// real window's event pump (its setters are crate-private), so everything downstream
/// of it consumes snapshots instead: gameplay code takes `&InputSnapshot`, and tests
/// build the frames they want by hand ([`InputSnapshot::default`] plus the `with_*`
/// builders) with no window, device, or swapchain in sight.
///
/// It lives in the **platform** crate, next to the `Input` it captures, because that
/// is the layer that defines the key/button encoding — so a consumer that only wants
/// "one frame of input" (e.g. the engine's game-hook signature) does not have to
/// depend on the game framework to name the type. `dreamcoast_game::input`
/// re-exports it, so gameplay code can keep importing it from there.
///
/// It is also the frame boundary. The platform state mutates while messages are
/// pumped; a snapshot is taken once per frame and stays still for the whole
/// simulation step, so two systems reading input in the same frame cannot
/// disagree.
#[derive(Clone, PartialEq)]
pub struct InputSnapshot {
    keys: [bool; KEY_COUNT],
    buttons: [bool; MOUSE_BUTTON_COUNT],
    mouse_pos: (i32, i32),
    mouse_delta: (i32, i32),
    wheel: f32,
    captured: bool,
}

impl Default for InputSnapshot {
    fn default() -> Self {
        Self {
            keys: [false; KEY_COUNT],
            buttons: [false; MOUSE_BUTTON_COUNT],
            mouse_pos: (0, 0),
            mouse_delta: (0, 0),
            wheel: 0.0,
            captured: false,
        }
    }
}

impl InputSnapshot {
    /// Capture the platform's current input state.
    ///
    /// Call once per frame, after the window has pumped its messages and before
    /// the simulation step runs.
    pub fn capture(input: &crate::Input) -> Self {
        let mut keys = [false; KEY_COUNT];
        for (vk, slot) in keys.iter_mut().enumerate() {
            *slot = input.key_down(vk as u16);
        }
        let mut buttons = [false; MOUSE_BUTTON_COUNT];
        for (i, slot) in buttons.iter_mut().enumerate() {
            *slot = input.mouse_button(i);
        }
        Self {
            keys,
            buttons,
            mouse_pos: input.mouse_position(),
            mouse_delta: input.mouse_delta(),
            wheel: input.wheel_delta(),
            captured: input.captured(),
        }
    }

    /// Whether the given Win32 virtual-key code is held this frame.
    #[inline]
    pub fn key(&self, vk: u16) -> bool {
        self.keys[(vk & 0xFF) as usize]
    }

    /// Whether a mouse button is held: 0 = left, 1 = right, 2 = middle.
    #[inline]
    pub fn mouse_button(&self, button: u8) -> bool {
        self.buttons.get(button as usize).copied().unwrap_or(false)
    }

    /// Cursor position in client-area pixels.
    #[inline]
    pub fn mouse_position(&self) -> (i32, i32) {
        self.mouse_pos
    }

    /// Cursor movement this frame in pixels (raw motion while the pointer is
    /// captured, otherwise the in-window position difference).
    #[inline]
    pub fn mouse_delta(&self) -> (i32, i32) {
        self.mouse_delta
    }

    /// Accumulated wheel movement this frame, in notches (+up).
    #[inline]
    pub fn wheel(&self) -> f32 {
        self.wheel
    }

    /// Whether the pointer is captured (hidden + pinned for mouse look).
    #[inline]
    pub fn captured(&self) -> bool {
        self.captured
    }

    /// Set a key's held state (test/builder use).
    pub fn set_key(&mut self, vk: u16, down: bool) {
        self.keys[(vk & 0xFF) as usize] = down;
    }

    /// Set a mouse button's held state (test/builder use).
    pub fn set_mouse_button(&mut self, button: u8, down: bool) {
        if let Some(slot) = self.buttons.get_mut(button as usize) {
            *slot = down;
        }
    }

    /// Builder form of [`Self::set_key`].
    #[must_use]
    pub fn with_key(mut self, vk: u16, down: bool) -> Self {
        self.set_key(vk, down);
        self
    }

    /// Builder form of [`Self::set_mouse_button`].
    #[must_use]
    pub fn with_mouse_button(mut self, button: u8, down: bool) -> Self {
        self.set_mouse_button(button, down);
        self
    }

    /// Builder: cursor position in client-area pixels.
    #[must_use]
    pub fn with_mouse_position(mut self, x: i32, y: i32) -> Self {
        self.mouse_pos = (x, y);
        self
    }

    /// Builder: cursor movement for this frame.
    #[must_use]
    pub fn with_mouse_delta(mut self, dx: i32, dy: i32) -> Self {
        self.mouse_delta = (dx, dy);
        self
    }

    /// Builder: wheel movement for this frame, in notches (+up).
    #[must_use]
    pub fn with_wheel(mut self, notches: f32) -> Self {
        self.wheel = notches;
        self
    }

    /// Builder: pointer-capture state.
    #[must_use]
    pub fn with_captured(mut self, captured: bool) -> Self {
        self.captured = captured;
        self
    }

    /// The virtual-key codes held this frame (ascending).
    pub fn held_keys(&self) -> impl Iterator<Item = u16> + '_ {
        self.keys
            .iter()
            .enumerate()
            .filter(|(_, down)| **down)
            .map(|(vk, _)| vk as u16)
    }
}

/// Prints only what is actually held — a 256-entry key array is unreadable.
impl fmt::Debug for InputSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let keys: Vec<String> = self
            .held_keys()
            .map(|vk| match key_name(vk) {
                Some(name) => name.to_string(),
                None => format!("vk:0x{vk:02X}"),
            })
            .collect();
        let buttons: Vec<usize> = (0..MOUSE_BUTTON_COUNT)
            .filter(|b| self.buttons[*b])
            .collect();
        f.debug_struct("InputSnapshot")
            .field("keys", &keys)
            .field("buttons", &buttons)
            .field("mouse_pos", &self.mouse_pos)
            .field("mouse_delta", &self.mouse_delta)
            .field("wheel", &self.wheel)
            .field("captured", &self.captured)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all_idle() {
        let snap = InputSnapshot::default();
        assert!(snap.held_keys().next().is_none());
        assert!(!snap.key(0x57));
        assert!(!snap.mouse_button(0));
        assert_eq!(snap.mouse_delta(), (0, 0));
        assert_eq!(snap.wheel(), 0.0);
        assert!(!snap.captured());
    }

    #[test]
    fn builders_set_every_field() {
        let snap = InputSnapshot::default()
            .with_key(0x57, true)
            .with_mouse_button(1, true)
            .with_mouse_position(320, 240)
            .with_mouse_delta(-4, 7)
            .with_wheel(-2.0)
            .with_captured(true);
        assert!(snap.key(0x57));
        assert!(!snap.key(0x41));
        assert!(snap.mouse_button(1));
        assert!(!snap.mouse_button(0));
        assert_eq!(snap.mouse_position(), (320, 240));
        assert_eq!(snap.mouse_delta(), (-4, 7));
        assert_eq!(snap.wheel(), -2.0);
        assert!(snap.captured());
        assert_eq!(snap.held_keys().collect::<Vec<_>>(), vec![0x57]);
    }

    /// The platform layer masks key codes to 8 bits; the snapshot must agree so a
    /// binding can never index out of the table.
    #[test]
    fn key_index_is_masked_like_the_platform_layer() {
        let snap = InputSnapshot::default().with_key(0x157, true);
        assert!(snap.key(0x57));
        assert!(snap.key(0x157));
    }

    #[test]
    fn out_of_range_mouse_button_is_inert() {
        let snap = InputSnapshot::default().with_mouse_button(9, true);
        assert!(!snap.mouse_button(9));
        assert!((0..3).all(|b| !snap.mouse_button(b)));
    }

    #[test]
    fn captures_the_platform_default_state() {
        // `Input`'s setters are crate-private, so an untouched default is the only
        // state constructible from outside — enough to pin the field wiring.
        let platform = crate::Input::default();
        let snap = InputSnapshot::capture(&platform);
        assert_eq!(snap, InputSnapshot::default());
    }

    #[test]
    fn debug_lists_held_keys_by_name() {
        let snap = InputSnapshot::default()
            .with_key(0x57, true)
            .with_key(0x10, true);
        let text = format!("{snap:?}");
        assert!(text.contains('W'), "{text}");
        assert!(text.contains("Shift"), "{text}");
    }
}
