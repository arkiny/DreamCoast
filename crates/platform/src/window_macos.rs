//! A thin Cocoa/AppKit window backed by a `CAMetalLayer`.
//!
//! We talk to AppKit directly (rather than via a windowing crate) to keep the
//! engine in control of its own render loop, mirroring the Win32 backend. The
//! window owns an `NSWindow` whose content view is layer-backed by a
//! `CAMetalLayer`; the Metal RHI backend renders into that layer (it sets the
//! layer's `device`/`pixelFormat` and pulls drawables from it).
//!
//! Event handling is a non-blocking pump: each frame we drain pending `NSEvent`s
//! (`untilDate: distantPast`) into [`Input`] and forward them to the app so the
//! window chrome (move/resize/close) keeps working.
//!
//! Keyboard events are translated from macOS hardware key codes into **Win32 virtual-key codes**
//! ([`mac_keycode_to_vk`]) because that is the vocabulary [`Input`] and every consumer of it
//! speak, on every platform.

use dreamcoast_core::EngineError;
use objc2::rc::Retained;
use objc2::{ClassType, MainThreadMarker, MainThreadOnly, msg_send};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSEvent, NSEventMask,
    NSEventType, NSWindow, NSWindowStyleMask,
};
use objc2_foundation::{NSDate, NSDefaultRunLoopMode, NSPoint, NSRect, NSSize, NSString};
use objc2_quartz_core::CAMetalLayer;

/// macOS virtual key code for Escape (used to request close, like the Win32 path).
const KEY_ESCAPE: u16 = 53;

/// Win32 virtual-key codes, as the rest of the engine speaks them.
///
/// The Win32 window feeds these straight out of `wParam`, so naming them here lets the macOS
/// translation table below read as key names instead of bare hex — and lets a reviewer check it
/// against the Win32 header line by line.
mod vk {
    /// Letters and top-row digits take the ASCII code of the *uppercase* character
    /// (`VK_W == b'W' == 0x57`, `VK_1 == b'1' == 0x31`) — that is the Win32 definition, not a
    /// coincidence we lean on.
    pub const fn ascii(c: u8) -> u16 {
        c as u16
    }

    pub const BACK: u16 = 0x08;
    pub const TAB: u16 = 0x09;
    pub const RETURN: u16 = 0x0D;
    pub const SHIFT: u16 = 0x10;
    pub const CONTROL: u16 = 0x11;
    /// `VK_MENU` — the Alt key. macOS Option occupies the same physical slot.
    pub const MENU: u16 = 0x12;
    /// `VK_CAPITAL` — Caps Lock.
    pub const CAPITAL: u16 = 0x14;
    pub const ESCAPE: u16 = 0x1B;
    pub const SPACE: u16 = 0x20;
    /// `VK_PRIOR` — Page Up.
    pub const PRIOR: u16 = 0x21;
    /// `VK_NEXT` — Page Down.
    pub const NEXT: u16 = 0x22;
    pub const END: u16 = 0x23;
    pub const HOME: u16 = 0x24;
    pub const LEFT: u16 = 0x25;
    pub const UP: u16 = 0x26;
    pub const RIGHT: u16 = 0x27;
    pub const DOWN: u16 = 0x28;
    pub const INSERT: u16 = 0x2D;
    /// `VK_DELETE` — forward delete, *not* backspace (that is [`BACK`]).
    pub const DELETE: u16 = 0x2E;
    /// Left/right OS ("Windows") key; macOS Command lives in that slot.
    pub const LWIN: u16 = 0x5B;
    pub const RWIN: u16 = 0x5C;
    /// `VK_APPS` — the context-menu key.
    pub const APPS: u16 = 0x5D;
    /// `VK_NUMPAD0`; the keypad digits run `NUMPAD0 + n`.
    pub const NUMPAD0: u16 = 0x60;
    pub const MULTIPLY: u16 = 0x6A;
    pub const ADD: u16 = 0x6B;
    pub const SUBTRACT: u16 = 0x6D;
    pub const DECIMAL: u16 = 0x6E;
    pub const DIVIDE: u16 = 0x6F;
    /// `VK_F1`; the function keys run `F1 + n` up to `VK_F24 == 0x87`.
    pub const F1: u16 = 0x70;
    pub const NUMLOCK: u16 = 0x90;
    pub const VOLUME_MUTE: u16 = 0xAD;
    pub const VOLUME_DOWN: u16 = 0xAE;
    pub const VOLUME_UP: u16 = 0xAF;
    /// `VK_OEM_1` — `;` `:` on a US layout.
    pub const OEM_1: u16 = 0xBA;
    /// `=` `+`.
    pub const OEM_PLUS: u16 = 0xBB;
    pub const OEM_COMMA: u16 = 0xBC;
    pub const OEM_MINUS: u16 = 0xBD;
    pub const OEM_PERIOD: u16 = 0xBE;
    /// `VK_OEM_2` — `/` `?`.
    pub const OEM_2: u16 = 0xBF;
    /// `VK_OEM_3` — `` ` `` `~`.
    pub const OEM_3: u16 = 0xC0;
    /// `VK_OEM_4` — `[` `{`.
    pub const OEM_4: u16 = 0xDB;
    /// `VK_OEM_5` — `\` `|`.
    pub const OEM_5: u16 = 0xDC;
    /// `VK_OEM_6` — `]` `}`.
    pub const OEM_6: u16 = 0xDD;
    /// `VK_OEM_7` — `'` `"`.
    pub const OEM_7: u16 = 0xDE;
    /// `VK_OEM_102` — the extra `<` `>` key on ISO keyboards (absent from ANSI US).
    pub const OEM_102: u16 = 0xE2;
}

/// Translate a macOS hardware key code (`kVK_ANSI_*` / `kVK_*`) into the Win32 virtual-key code the
/// whole engine speaks, so keyboard controls behave identically on both platforms. Without this,
/// macOS reports e.g. `W == 0x0D` while the fly camera checks `VK_W == 0x57`, so WASD does nothing.
///
/// Covers the full ANSI US layout: letters, top-row digits, `F1`..`F20`, arrows, the editing and
/// navigation cluster, both sides of every modifier, the OEM punctuation keys, and the keypad.
///
/// **Unmapped keys are dropped** (`None`), never passed through. A macOS key code and a Win32 VK
/// live in different spaces, so passing one through aliases it onto an unrelated key: `kVK_Return`
/// (0x24) would land on `VK_HOME`, `kVK_ANSI_V` (0x09) on `VK_TAB`, `kVK_ANSI_Y` (0x10) on
/// `VK_SHIFT`. Dropping loses a key the engine could not have addressed anyway; aliasing fires the
/// *wrong* action. Currently dropped: the `fn` key (no Win32 equivalent), the JIS-only keys
/// (Yen/Underscore/Eisu/Kana — every plausible Win32 code for them collides with an ANSI key), and
/// keypad `=` (Win32 has no keypad-equals VK).
///
/// Intentional many-to-one shares (both platforms do the same thing):
/// * left/right Shift, Control and Option collapse to `VK_SHIFT`/`VK_CONTROL`/`VK_MENU` — Win32's
///   `WM_KEYDOWN` `wParam` reports exactly these generic codes, and the sided `VK_LSHIFT`..
///   `VK_RMENU` (0xA0..0xA5) never appear there either. Matching that is parity; inventing sided
///   codes here would make macOS the odd one out.
/// * keypad Enter shares `VK_RETURN` with Return. On Win32 both send `VK_RETURN` (the keypad one
///   only differs by the extended-key bit in `lParam`, which this layer does not carry).
///   `VK_SEPARATOR` (0x6C) is *not* the keypad Enter — it is a locale key almost no layout emits.
fn mac_keycode_to_vk(kc: u16) -> Option<u16> {
    let vk = match kc {
        // --- Letters (kVK_ANSI_A .. kVK_ANSI_Z, in hardware order) ---
        0x00 => vk::ascii(b'A'),
        0x0B => vk::ascii(b'B'),
        0x08 => vk::ascii(b'C'),
        0x02 => vk::ascii(b'D'),
        0x0E => vk::ascii(b'E'),
        0x03 => vk::ascii(b'F'),
        0x05 => vk::ascii(b'G'),
        0x04 => vk::ascii(b'H'),
        0x22 => vk::ascii(b'I'),
        0x26 => vk::ascii(b'J'),
        0x28 => vk::ascii(b'K'),
        0x25 => vk::ascii(b'L'),
        0x2E => vk::ascii(b'M'),
        0x2D => vk::ascii(b'N'),
        0x1F => vk::ascii(b'O'),
        0x23 => vk::ascii(b'P'),
        0x0C => vk::ascii(b'Q'),
        0x0F => vk::ascii(b'R'),
        0x01 => vk::ascii(b'S'),
        0x11 => vk::ascii(b'T'),
        0x20 => vk::ascii(b'U'),
        0x09 => vk::ascii(b'V'),
        0x0D => vk::ascii(b'W'),
        0x07 => vk::ascii(b'X'),
        0x10 => vk::ascii(b'Y'),
        0x06 => vk::ascii(b'Z'),

        // --- Top-row digits (note macOS orders 5/6 and 7/8/9 oddly) ---
        0x1D => vk::ascii(b'0'),
        0x12 => vk::ascii(b'1'),
        0x13 => vk::ascii(b'2'),
        0x14 => vk::ascii(b'3'),
        0x15 => vk::ascii(b'4'),
        0x17 => vk::ascii(b'5'),
        0x16 => vk::ascii(b'6'),
        0x1A => vk::ascii(b'7'),
        0x1C => vk::ascii(b'8'),
        0x19 => vk::ascii(b'9'),

        // --- Editing / whitespace ---
        0x24 => vk::RETURN,  // kVK_Return
        0x30 => vk::TAB,     // kVK_Tab
        0x31 => vk::SPACE,   // kVK_Space
        0x33 => vk::BACK,    // kVK_Delete is backspace on macOS
        0x35 => vk::ESCAPE,  // kVK_Escape
        0x75 => vk::DELETE,  // kVK_ForwardDelete
        0x72 => vk::INSERT,  // kVK_Help sits in the Insert position on full-size keyboards
        0x6E => vk::APPS,    // kVK_ContextualMenu
        0x0A => vk::OEM_102, // kVK_ISO_Section (ISO layouts only; no ANSI key collides)

        // --- Navigation ---
        0x73 => vk::HOME,  // kVK_Home
        0x77 => vk::END,   // kVK_End
        0x74 => vk::PRIOR, // kVK_PageUp
        0x79 => vk::NEXT,  // kVK_PageDown
        0x7B => vk::LEFT,  // kVK_LeftArrow
        0x7C => vk::RIGHT, // kVK_RightArrow
        0x7D => vk::DOWN,  // kVK_DownArrow
        0x7E => vk::UP,    // kVK_UpArrow

        // --- Modifiers (delivered as FlagsChanged, see `modifier_key_state`) ---
        0x38 | 0x3C => vk::SHIFT,   // kVK_Shift / kVK_RightShift
        0x3B | 0x3E => vk::CONTROL, // kVK_Control / kVK_RightControl
        0x3A | 0x3D => vk::MENU,    // kVK_Option / kVK_RightOption -> Alt
        0x37 => vk::LWIN,           // kVK_Command
        0x36 => vk::RWIN,           // kVK_RightCommand
        0x39 => vk::CAPITAL,        // kVK_CapsLock

        // --- OEM punctuation (US layout) ---
        0x1B => vk::OEM_MINUS,  // kVK_ANSI_Minus
        0x18 => vk::OEM_PLUS,   // kVK_ANSI_Equal
        0x21 => vk::OEM_4,      // kVK_ANSI_LeftBracket
        0x1E => vk::OEM_6,      // kVK_ANSI_RightBracket
        0x2A => vk::OEM_5,      // kVK_ANSI_Backslash
        0x29 => vk::OEM_1,      // kVK_ANSI_Semicolon
        0x27 => vk::OEM_7,      // kVK_ANSI_Quote
        0x2B => vk::OEM_COMMA,  // kVK_ANSI_Comma
        0x2F => vk::OEM_PERIOD, // kVK_ANSI_Period
        0x2C => vk::OEM_2,      // kVK_ANSI_Slash
        0x32 => vk::OEM_3,      // kVK_ANSI_Grave

        // --- Function keys (F1..F20; macOS has no F21+) ---
        0x7A => vk::F1,
        0x78 => vk::F1 + 1,
        0x63 => vk::F1 + 2,
        0x76 => vk::F1 + 3,
        0x60 => vk::F1 + 4,
        0x61 => vk::F1 + 5,
        0x62 => vk::F1 + 6,
        0x64 => vk::F1 + 7,
        0x65 => vk::F1 + 8,
        0x6D => vk::F1 + 9,
        0x67 => vk::F1 + 10,
        0x6F => vk::F1 + 11,
        0x69 => vk::F1 + 12,
        0x6B => vk::F1 + 13,
        0x71 => vk::F1 + 14,
        0x6A => vk::F1 + 15,
        0x40 => vk::F1 + 16,
        0x4F => vk::F1 + 17,
        0x50 => vk::F1 + 18,
        0x5A => vk::F1 + 19,

        // --- Keypad ---
        0x52 => vk::NUMPAD0,
        0x53 => vk::NUMPAD0 + 1,
        0x54 => vk::NUMPAD0 + 2,
        0x55 => vk::NUMPAD0 + 3,
        0x56 => vk::NUMPAD0 + 4,
        0x57 => vk::NUMPAD0 + 5,
        0x58 => vk::NUMPAD0 + 6,
        0x59 => vk::NUMPAD0 + 7,
        0x5B => vk::NUMPAD0 + 8,
        0x5C => vk::NUMPAD0 + 9,
        0x41 => vk::DECIMAL,  // kVK_ANSI_KeypadDecimal
        0x43 => vk::MULTIPLY, // kVK_ANSI_KeypadMultiply
        0x45 => vk::ADD,      // kVK_ANSI_KeypadPlus
        0x4E => vk::SUBTRACT, // kVK_ANSI_KeypadMinus
        0x4B => vk::DIVIDE,   // kVK_ANSI_KeypadDivide
        0x4C => vk::RETURN,   // kVK_ANSI_KeypadEnter (shares VK_RETURN, as on Win32)
        // kVK_ANSI_KeypadClear occupies the Num Lock position; PC keyboards call it Num Lock.
        0x47 => vk::NUMLOCK,

        // --- Media keys present on Apple keyboards ---
        0x48 => vk::VOLUME_UP,
        0x49 => vk::VOLUME_DOWN,
        0x4A => vk::VOLUME_MUTE,

        // Everything else (fn, JIS-only keys, keypad `=`, unassigned codes) is dropped rather
        // than aliased onto an unrelated VK slot — see the note on this function.
        _ => return None,
    };
    Some(vk)
}

/// Modifier-flag bits carried by `NSEvent.modifierFlags`.
mod ns_flags {
    /// Device-*independent* bits (AppKit `NSEventModifierFlag*`): "some Shift is down", etc.
    pub const CAPS_LOCK: u64 = 1 << 16;
    pub const SHIFT: u64 = 1 << 17;
    pub const CONTROL: u64 = 1 << 18;
    pub const OPTION: u64 = 1 << 19;
    pub const COMMAND: u64 = 1 << 20;
    /// Device-*dependent* bits (IOKit `NX_DEVICE*KEYMASK`) — the only place the side of the key
    /// is reported. Synthesized events may leave them clear, hence the fallback below.
    pub const LCOMMAND: u64 = 0x0000_0008;
    pub const RCOMMAND: u64 = 0x0000_0010;
}

/// Resolve a `FlagsChanged` event into the VK slot it owns and whether that key is now down.
///
/// macOS does not send KeyDown/KeyUp for modifiers: pressing Shift produces a `FlagsChanged`
/// event carrying the key code plus the *new* modifier mask, so the up/down edge has to be read
/// out of the mask. Returns `None` for any key code that is not a modifier we map (e.g. `fn`).
///
/// Shift/Control/Option read the device-independent bit, which is exactly right because both
/// sides share one VK slot (holding both and releasing one must keep the slot down). Command is
/// sided (`VK_LWIN`/`VK_RWIN`), so it needs the device-dependent bits. Caps Lock reports the
/// *lock* state rather than the physical hold — that is what macOS exposes, and it matches how
/// the key actually behaves.
fn modifier_key_state(kc: u16, flags: u64) -> Option<(u16, bool)> {
    let down = match kc {
        0x38 | 0x3C => flags & ns_flags::SHIFT != 0,
        0x3B | 0x3E => flags & ns_flags::CONTROL != 0,
        0x3A | 0x3D => flags & ns_flags::OPTION != 0,
        0x39 => flags & ns_flags::CAPS_LOCK != 0,
        0x37 => {
            // Left Command: its device bit, or Command-without-a-side (fall back to left).
            flags & ns_flags::LCOMMAND != 0
                || (flags & ns_flags::COMMAND != 0
                    && flags & (ns_flags::LCOMMAND | ns_flags::RCOMMAND) == 0)
        }
        0x36 => flags & ns_flags::RCOMMAND != 0,
        _ => return None,
    };
    Some((mac_keycode_to_vk(kc)?, down))
}

/// An open application window.
pub struct Window {
    window: Retained<NSWindow>,
    layer: Retained<CAMetalLayer>,
    should_close: bool,
    resized: bool,
    /// Client-area size in physical pixels (backing-scaled).
    size: (u32, u32),
    /// Backing scale factor (points -> pixels).
    scale: f64,
    input: crate::Input,
    /// Pointer-lock (fly-camera capture): cursor hidden + disassociated from mouse motion,
    /// so the raw per-event deltas keep flowing with no screen-edge limit.
    captured: bool,
    /// Whether Escape requests close (the dev-tool default). A game that owns Escape
    /// (pause menu) turns this off and drives shutdown through its own quit path.
    close_on_escape: bool,
}

// Pointer-lock plumbing: freezing the on-screen cursor while the hardware deltas keep
// arriving is a CoreGraphics service (there is no AppKit equivalent).
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGAssociateMouseAndMouseCursorPosition(connected: i32) -> i32;
}

impl Window {
    /// Create and show a window with the given title and client-area size (in
    /// points; the backing layer is scaled to physical pixels).
    pub fn new(title: &str, width: u32, height: u32) -> Result<Self, EngineError> {
        let mtm = MainThreadMarker::new().ok_or_else(|| {
            EngineError::Platform("window must be created on the main thread".into())
        })?;

        let app = NSApplication::sharedApplication(mtm);
        app.setActivationPolicy(NSApplicationActivationPolicy::Regular);

        let content_rect = NSRect::new(
            NSPoint::new(0.0, 0.0),
            NSSize::new(width as f64, height as f64),
        );
        let style = NSWindowStyleMask::Titled
            | NSWindowStyleMask::Closable
            | NSWindowStyleMask::Resizable
            | NSWindowStyleMask::Miniaturizable;

        let window = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(
                NSWindow::alloc(mtm),
                content_rect,
                style,
                NSBackingStoreType::Buffered,
                false,
            )
        };
        // Closing the window just orders it out (we detect it via `isVisible`);
        // without this AppKit would over-release the window object.
        unsafe { window.setReleasedWhenClosed(false) };
        window.setTitle(&NSString::from_str(title));
        window.center();

        // Layer-back the content view with a CAMetalLayer.
        let layer: Retained<CAMetalLayer> = unsafe { msg_send![CAMetalLayer::class(), new] };
        let scale = window.backingScaleFactor();
        let px = (width as f64 * scale, height as f64 * scale);
        layer.setContentsScale(scale);
        layer.setDrawableSize(NSSize::new(px.0, px.1));
        let view = window
            .contentView()
            .ok_or_else(|| EngineError::Platform("NSWindow has no content view".into()))?;
        view.setWantsLayer(true);
        view.setLayer(Some(&layer));

        window.makeKeyAndOrderFront(None);
        app.activate();

        Ok(Self {
            window,
            layer,
            should_close: false,
            resized: false,
            size: (px.0 as u32, px.1 as u32),
            scale,
            input: crate::Input::default(),
            captured: false,
            close_on_escape: true,
        })
    }

    /// Drain pending Cocoa events into input state. Non-blocking.
    pub fn pump_events(&mut self) {
        self.input.begin_frame();
        let Some(mtm) = MainThreadMarker::new() else {
            return;
        };
        let app = NSApplication::sharedApplication(mtm);

        loop {
            // `distantPast` makes this return immediately when the queue is empty.
            let event: Option<Retained<NSEvent>> = unsafe {
                app.nextEventMatchingMask_untilDate_inMode_dequeue(
                    NSEventMask::Any,
                    Some(&NSDate::distantPast()),
                    NSDefaultRunLoopMode,
                    true,
                )
            };
            let Some(event) = event else { break };
            self.handle_event(&event);
            app.sendEvent(&event);
        }

        // The window may have been closed (ordered out) by the chrome's close
        // button; reflect that, and pick up any resize.
        if !self.window.isVisible() {
            self.should_close = true;
        }
        self.update_size();
    }

    fn handle_event(&mut self, event: &NSEvent) {
        let ty = event.r#type();
        match ty {
            NSEventType::KeyDown => {
                let kc = event.keyCode();
                if let Some(vk) = mac_keycode_to_vk(kc) {
                    self.input.set_key(vk as usize, true);
                }
                if kc == KEY_ESCAPE && self.close_on_escape {
                    self.should_close = true;
                }
                // Feed typed characters (for text input / ImGui), skipping
                // control characters.
                if let Some(chars) = event.characters() {
                    for ch in chars.to_string().chars() {
                        if !ch.is_control() {
                            self.input.push_char(ch);
                        }
                    }
                }
            }
            NSEventType::KeyUp => {
                if let Some(vk) = mac_keycode_to_vk(event.keyCode()) {
                    self.input.set_key(vk as usize, false);
                }
            }
            // Modifiers never arrive as KeyDown/KeyUp on macOS — the press *and* the release
            // both come through here, with the new modifier mask attached.
            NSEventType::FlagsChanged => {
                let flags = event.modifierFlags().0 as u64;
                if let Some((vk, down)) = modifier_key_state(event.keyCode(), flags) {
                    self.input.set_key(vk as usize, down);
                }
            }
            NSEventType::LeftMouseDown => self.input.set_button(0, true),
            NSEventType::LeftMouseUp => self.input.set_button(0, false),
            NSEventType::RightMouseDown => self.input.set_button(1, true),
            NSEventType::RightMouseUp => self.input.set_button(1, false),
            NSEventType::OtherMouseDown => self.input.set_button(2, true),
            NSEventType::OtherMouseUp => self.input.set_button(2, false),
            NSEventType::MouseMoved
            | NSEventType::LeftMouseDragged
            | NSEventType::RightMouseDragged
            | NSEventType::OtherMouseDragged => {
                // `locationInWindow` is in window points, origin bottom-left;
                // convert to top-left physical pixels to match the Win32 path.
                let p = event.locationInWindow();
                let h = self.size.1 as f64 / self.scale;
                let x = (p.x * self.scale) as i32;
                let y = ((h - p.y) * self.scale) as i32;
                self.input.set_mouse_pos(x, y);
                // Raw hardware deltas (points -> physical px) for the pointer-locked fly look:
                // they keep flowing while the cursor itself is frozen by the capture.
                let (dx, dy) = (event.deltaX(), event.deltaY());
                self.input
                    .add_raw_delta((dx * self.scale) as f32, (dy * self.scale) as f32);
            }
            NSEventType::ScrollWheel => {
                let dy = event.scrollingDeltaY();
                self.input.add_wheel(dy as f32);
            }
            _ => {}
        }
    }

    /// Recompute the physical drawable size from the content view; set the
    /// resize flag and the layer's drawable size when it changes.
    fn update_size(&mut self) {
        let Some(view) = self.window.contentView() else {
            return;
        };
        let bounds = view.bounds();
        self.scale = self.window.backingScaleFactor();
        let px = (
            (bounds.size.width * self.scale) as u32,
            (bounds.size.height * self.scale) as u32,
        );
        if px != self.size && px.0 > 0 && px.1 > 0 {
            self.size = px;
            self.resized = true;
            self.layer
                .setDrawableSize(NSSize::new(px.0 as f64, px.1 as f64));
        }
    }

    /// Whether the window has been asked to close (close button or ESC).
    #[inline]
    pub fn should_close(&self) -> bool {
        self.should_close
    }

    /// Whether Escape requests close (default true — the dev-tool behaviour). A game
    /// that owns Escape (pause menu) turns this off; the close button and quit still work.
    pub fn set_close_on_escape(&mut self, on: bool) {
        self.close_on_escape = on;
    }

    /// Current client-area size in physical pixels.
    #[inline]
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Take the "was resized since last checked" flag, clearing it.
    pub fn take_resized(&mut self) -> bool {
        std::mem::take(&mut self.resized)
    }

    /// Current input snapshot.
    #[inline]
    pub fn input(&self) -> &crate::Input {
        &self.input
    }

    /// The `CAMetalLayer` backing this window, for Metal swapchain creation.
    #[inline]
    pub fn metal_layer(&self) -> Retained<CAMetalLayer> {
        self.layer.clone()
    }

    /// Pointer lock for the fly camera: `true` hides the cursor and freezes it in place
    /// (mouse motion keeps arriving as raw deltas — see `Input::mouse_delta`), so the look
    /// never stops at a screen edge; `false` restores the normal cursor. Idempotent.
    pub fn set_cursor_captured(&mut self, on: bool) {
        if self.captured == on {
            return;
        }
        self.captured = on;
        self.input.set_captured(on);
        unsafe {
            CGAssociateMouseAndMouseCursorPosition(i32::from(!on));
            if on {
                objc2_app_kit::NSCursor::hide();
            } else {
                objc2_app_kit::NSCursor::unhide();
            }
        }
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        // Never leave the user's cursor hidden/frozen past the window's lifetime.
        self.set_cursor_captured(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// The mappings that existed before the table was completed. The sandbox's camera and
    /// main-loop toggles are hard-coded to these codes, so they are frozen: a change here
    /// silently re-binds WASD/QE, sprint, the fly-camera toggle or the screenshot key.
    const LEGACY: &[(u16, u16)] = &[
        (0x00, 0x41), // A
        (0x01, 0x53), // S
        (0x02, 0x44), // D
        (0x0D, 0x57), // W
        (0x0C, 0x51), // Q
        (0x0E, 0x45), // E
        (0x38, 0x10), // Shift        -> VK_SHIFT (sprint)
        (0x3C, 0x10), // right Shift
        (0x3A, 0x12), // Option       -> VK_MENU (pointer-lock release)
        (0x3D, 0x12), // right Option
        (0x2E, 0x4D), // M            -> pointer-lock latch
        (0x30, 0x09), // Tab          -> fly-camera toggle
        (0x35, 0x1B), // Escape       -> quit
        (0x78, 0x71), // F2           -> screenshot
    ];

    #[test]
    fn legacy_mappings_are_unchanged() {
        for &(kc, vk) in LEGACY {
            assert_eq!(
                mac_keycode_to_vk(kc),
                Some(vk),
                "mac 0x{kc:02X} must still map to VK 0x{vk:02X}"
            );
        }
    }

    #[test]
    fn letters_and_digits() {
        // Win32 letter/digit VKs are the ASCII uppercase codes.
        for (kc, ch) in [
            (0x00u16, b'A'),
            (0x06, b'Z'),
            (0x09, b'V'),
            (0x10, b'Y'),
            (0x22, b'I'),
            (0x2D, b'N'),
            (0x0D, b'W'),
        ] {
            assert_eq!(
                mac_keycode_to_vk(kc),
                Some(ch as u16),
                "letter {}",
                ch as char
            );
        }
        for (kc, ch) in [
            (0x1Du16, b'0'),
            (0x12, b'1'),
            (0x17, b'5'),
            (0x16, b'6'),
            (0x1A, b'7'),
            (0x19, b'9'),
        ] {
            assert_eq!(
                mac_keycode_to_vk(kc),
                Some(ch as u16),
                "digit {}",
                ch as char
            );
        }
        // Every letter and every digit must be reachable exactly once.
        let mapped: Vec<u16> = (0..=0xFFu16).filter_map(mac_keycode_to_vk).collect();
        for vk in (b'A' as u16..=b'Z' as u16).chain(b'0' as u16..=b'9' as u16) {
            assert_eq!(
                mapped.iter().filter(|&&v| v == vk).count(),
                1,
                "VK 0x{vk:02X} must come from exactly one key code"
            );
        }
    }

    #[test]
    fn arrows_and_navigation() {
        assert_eq!(mac_keycode_to_vk(0x7B), Some(0x25)); // Left
        assert_eq!(mac_keycode_to_vk(0x7E), Some(0x26)); // Up
        assert_eq!(mac_keycode_to_vk(0x7C), Some(0x27)); // Right
        assert_eq!(mac_keycode_to_vk(0x7D), Some(0x28)); // Down
        assert_eq!(mac_keycode_to_vk(0x73), Some(0x24)); // Home
        assert_eq!(mac_keycode_to_vk(0x77), Some(0x23)); // End
        assert_eq!(mac_keycode_to_vk(0x74), Some(0x21)); // Page Up
        assert_eq!(mac_keycode_to_vk(0x79), Some(0x22)); // Page Down
        assert_eq!(mac_keycode_to_vk(0x72), Some(0x2D)); // Help -> Insert
        assert_eq!(mac_keycode_to_vk(0x75), Some(0x2E)); // forward Delete
    }

    #[test]
    fn editing_keys_split_backspace_from_delete() {
        // The macOS "delete" key is backspace; forward delete is a separate code. Getting this
        // backwards is the classic port bug.
        assert_eq!(mac_keycode_to_vk(0x33), Some(0x08), "backspace -> VK_BACK");
        assert_eq!(
            mac_keycode_to_vk(0x75),
            Some(0x2E),
            "fwd delete -> VK_DELETE"
        );
        assert_eq!(mac_keycode_to_vk(0x24), Some(0x0D), "Return -> VK_RETURN");
        assert_eq!(mac_keycode_to_vk(0x31), Some(0x20), "Space -> VK_SPACE");
    }

    #[test]
    fn modifiers_use_the_generic_win32_codes() {
        assert_eq!(mac_keycode_to_vk(0x38), Some(0x10)); // Shift
        assert_eq!(mac_keycode_to_vk(0x3C), Some(0x10)); // right Shift
        assert_eq!(mac_keycode_to_vk(0x3B), Some(0x11)); // Control
        assert_eq!(mac_keycode_to_vk(0x3E), Some(0x11)); // right Control
        assert_eq!(mac_keycode_to_vk(0x3A), Some(0x12)); // Option -> Alt
        assert_eq!(mac_keycode_to_vk(0x3D), Some(0x12)); // right Option
        assert_eq!(mac_keycode_to_vk(0x39), Some(0x14)); // Caps Lock
        assert_eq!(mac_keycode_to_vk(0x37), Some(0x5B)); // Command -> VK_LWIN
        assert_eq!(mac_keycode_to_vk(0x36), Some(0x5C)); // right Command -> VK_RWIN
        // Nothing produces the sided VK_LSHIFT..VK_RMENU block: Win32's WM_KEYDOWN does not
        // either, so producing them here would break parity.
        let mapped: Vec<u16> = (0..=0xFFu16).filter_map(mac_keycode_to_vk).collect();
        for vk in 0xA0..=0xA5u16 {
            assert!(!mapped.contains(&vk), "VK 0x{vk:02X} must not be produced");
        }
    }

    #[test]
    fn punctuation_and_function_keys() {
        assert_eq!(mac_keycode_to_vk(0x1B), Some(0xBD)); // - VK_OEM_MINUS
        assert_eq!(mac_keycode_to_vk(0x18), Some(0xBB)); // = VK_OEM_PLUS
        assert_eq!(mac_keycode_to_vk(0x21), Some(0xDB)); // [
        assert_eq!(mac_keycode_to_vk(0x1E), Some(0xDD)); // ]
        assert_eq!(mac_keycode_to_vk(0x2A), Some(0xDC)); // \
        assert_eq!(mac_keycode_to_vk(0x29), Some(0xBA)); // ;
        assert_eq!(mac_keycode_to_vk(0x27), Some(0xDE)); // '
        assert_eq!(mac_keycode_to_vk(0x2B), Some(0xBC)); // ,
        assert_eq!(mac_keycode_to_vk(0x2F), Some(0xBE)); // .
        assert_eq!(mac_keycode_to_vk(0x2C), Some(0xBF)); // /
        assert_eq!(mac_keycode_to_vk(0x32), Some(0xC0)); // `

        assert_eq!(mac_keycode_to_vk(0x7A), Some(0x70)); // F1
        assert_eq!(mac_keycode_to_vk(0x6F), Some(0x7B)); // F12
        assert_eq!(mac_keycode_to_vk(0x5A), Some(0x83)); // F20
        // F1..F20 each reachable exactly once.
        let mapped: Vec<u16> = (0..=0xFFu16).filter_map(mac_keycode_to_vk).collect();
        for vk in 0x70..=0x83u16 {
            assert_eq!(
                mapped.iter().filter(|&&v| v == vk).count(),
                1,
                "VK 0x{vk:02X}"
            );
        }
    }

    #[test]
    fn keypad() {
        for n in 0..10u16 {
            // Keypad 0..9 in hardware order: 0,1..7 are contiguous, then 8 and 9 jump.
            let kc = match n {
                0..=7 => 0x52 + n,
                8 => 0x5B,
                _ => 0x5C,
            };
            assert_eq!(mac_keycode_to_vk(kc), Some(0x60 + n), "keypad {n}");
        }
        assert_eq!(mac_keycode_to_vk(0x41), Some(0x6E)); // VK_DECIMAL
        assert_eq!(mac_keycode_to_vk(0x43), Some(0x6A)); // VK_MULTIPLY
        assert_eq!(mac_keycode_to_vk(0x45), Some(0x6B)); // VK_ADD
        assert_eq!(mac_keycode_to_vk(0x4E), Some(0x6D)); // VK_SUBTRACT
        assert_eq!(mac_keycode_to_vk(0x4B), Some(0x6F)); // VK_DIVIDE
        assert_eq!(mac_keycode_to_vk(0x47), Some(0x90)); // Clear -> VK_NUMLOCK
        // Keypad Enter is VK_RETURN on Win32, not VK_SEPARATOR (0x6C).
        assert_eq!(mac_keycode_to_vk(0x4C), Some(0x0D));
    }

    #[test]
    fn untranslatable_keys_are_dropped_not_aliased() {
        for kc in [
            0x3F,  // fn — no Win32 equivalent
            0x51,  // keypad `=` — Win32 has no keypad-equals VK
            0x5D,  // JIS Yen
            0x5E,  // JIS underscore
            0x5F,  // JIS keypad comma
            0x66,  // JIS Eisu
            0x68,  // JIS Kana
            0x42,  // unassigned
            0x44,  // unassigned
            0x46,  // unassigned
            0x4D,  // unassigned
            0x6C,  // unassigned
            0x7F,  // unassigned
            0x100, // out of the hardware range entirely
            0xFFFF,
        ] {
            assert_eq!(
                mac_keycode_to_vk(kc),
                None,
                "mac 0x{kc:02X} must be dropped"
            );
        }
    }

    /// Two hardware keys sharing a VK slot means one of them fires the other's action. Only the
    /// documented shares are allowed — all of which Win32 itself collapses the same way.
    #[test]
    fn no_unintended_vk_collisions() {
        let mut by_vk: BTreeMap<u16, Vec<u16>> = BTreeMap::new();
        for kc in 0..=0xFFFFu16 {
            if let Some(vk) = mac_keycode_to_vk(kc) {
                by_vk.entry(vk).or_default().push(kc);
            }
        }
        let intentional: &[(u16, &[u16])] = &[
            (0x10, &[0x38, 0x3C]), // VK_SHIFT   — both Shifts
            (0x11, &[0x3B, 0x3E]), // VK_CONTROL — both Controls
            (0x12, &[0x3A, 0x3D]), // VK_MENU    — both Options
            (0x0D, &[0x24, 0x4C]), // VK_RETURN  — Return + keypad Enter
        ];
        for (vk, codes) in by_vk {
            if codes.len() == 1 {
                continue;
            }
            let expected = intentional
                .iter()
                .find(|(v, _)| *v == vk)
                .unwrap_or_else(|| panic!("unintended collision on VK 0x{vk:02X}: {codes:02X?}"))
                .1;
            assert_eq!(codes, expected, "VK 0x{vk:02X} shared by unexpected keys");
        }
    }

    #[test]
    fn modifier_flags_drive_the_up_down_edge() {
        // Press left Shift, then release it.
        assert_eq!(
            modifier_key_state(0x38, ns_flags::SHIFT),
            Some((0x10, true))
        );
        assert_eq!(modifier_key_state(0x38, 0), Some((0x10, false)));
        // Both Shifts share the slot: releasing one while the other is held keeps it down
        // (the device-independent bit stays set).
        assert_eq!(
            modifier_key_state(0x3C, ns_flags::SHIFT),
            Some((0x10, true))
        );

        assert_eq!(
            modifier_key_state(0x3A, ns_flags::OPTION),
            Some((0x12, true)),
            "Option -> VK_MENU, the pointer-lock release chord"
        );
        assert_eq!(
            modifier_key_state(0x3B, ns_flags::CONTROL),
            Some((0x11, true))
        );
        assert_eq!(
            modifier_key_state(0x39, ns_flags::CAPS_LOCK),
            Some((0x14, true))
        );

        // Command is sided, so it reads the device-dependent bits.
        let l = ns_flags::COMMAND | ns_flags::LCOMMAND;
        let r = ns_flags::COMMAND | ns_flags::RCOMMAND;
        assert_eq!(modifier_key_state(0x37, l), Some((0x5B, true)));
        assert_eq!(modifier_key_state(0x36, l), Some((0x5C, false)));
        assert_eq!(modifier_key_state(0x36, r), Some((0x5C, true)));
        assert_eq!(modifier_key_state(0x37, r), Some((0x5B, false)));
        // Sideless Command (synthesized events) falls back to left rather than vanishing.
        assert_eq!(
            modifier_key_state(0x37, ns_flags::COMMAND),
            Some((0x5B, true))
        );

        // Non-modifier key codes never come through this path.
        assert_eq!(modifier_key_state(0x00, ns_flags::SHIFT), None, "A");
        assert_eq!(modifier_key_state(0x3F, 0), None, "fn");
    }
}
