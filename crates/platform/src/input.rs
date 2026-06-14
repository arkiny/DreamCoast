//! Per-frame keyboard and mouse state, accumulated from Win32 messages.
//!
//! The window's message handler feeds raw events in via the `pub(crate)`
//! setters; consumers read the snapshot through the public accessors. Mouse
//! delta is measured relative to the position at the start of the current frame
//! (see [`Input::begin_frame`]).

/// A snapshot of input state for the current frame.
#[derive(Clone)]
pub struct Input {
    keys: [bool; 256],
    buttons: [bool; 3],
    pos: (i32, i32),
    frame_start_pos: (i32, i32),
    has_mouse: bool,
    wheel: f32,
    chars: Vec<char>,
}

impl Input {
    /// Whether the given Win32 virtual-key code is currently held down.
    #[inline]
    pub fn key_down(&self, vk: u16) -> bool {
        self.keys[(vk & 0xFF) as usize]
    }

    /// Whether a mouse button is held: 0 = left, 1 = right, 2 = middle.
    #[inline]
    pub fn mouse_button(&self, button: usize) -> bool {
        self.buttons.get(button).copied().unwrap_or(false)
    }

    /// Cursor position in client-area pixels.
    #[inline]
    pub fn mouse_position(&self) -> (i32, i32) {
        self.pos
    }

    /// Cursor movement since the start of the current frame, in pixels.
    #[inline]
    pub fn mouse_delta(&self) -> (i32, i32) {
        (
            self.pos.0 - self.frame_start_pos.0,
            self.pos.1 - self.frame_start_pos.1,
        )
    }

    /// Accumulated mouse wheel delta this frame (in notches; +up).
    #[inline]
    pub fn wheel_delta(&self) -> f32 {
        self.wheel
    }

    /// Characters typed this frame (from `WM_CHAR`).
    #[inline]
    pub fn chars(&self) -> &[char] {
        &self.chars
    }

    /// Latch the frame's starting cursor position and clear per-frame
    /// accumulators (wheel, typed characters). Call once before pumping messages.
    pub(crate) fn begin_frame(&mut self) {
        self.frame_start_pos = self.pos;
        self.wheel = 0.0;
        self.chars.clear();
    }

    pub(crate) fn add_wheel(&mut self, delta: f32) {
        self.wheel += delta;
    }

    pub(crate) fn push_char(&mut self, ch: char) {
        self.chars.push(ch);
    }

    pub(crate) fn set_key(&mut self, vk: usize, down: bool) {
        if let Some(slot) = self.keys.get_mut(vk & 0xFF) {
            *slot = down;
        }
    }

    pub(crate) fn set_button(&mut self, button: usize, down: bool) {
        if let Some(slot) = self.buttons.get_mut(button) {
            *slot = down;
        }
    }

    pub(crate) fn set_mouse_pos(&mut self, x: i32, y: i32) {
        if !self.has_mouse {
            self.has_mouse = true;
            self.frame_start_pos = (x, y);
        }
        self.pos = (x, y);
    }
}

impl Default for Input {
    fn default() -> Self {
        Self {
            keys: [false; 256],
            buttons: [false; 3],
            pos: (0, 0),
            frame_start_pos: (0, 0),
            has_mouse: false,
            wheel: 0.0,
            chars: Vec::new(),
        }
    }
}
