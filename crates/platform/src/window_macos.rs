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

/// Translate a macOS hardware key code (`kVK_ANSI_*` / `kVK_*`) into the Win32 virtual-key code the
/// input layer + camera controller speak, so keyboard controls behave identically to the Windows
/// path. Without this, macOS reports e.g. `W == 13` while the fly camera checks `VK_W == 0x57`, so
/// WASD silently does nothing. Only the keys the app actually queries are mapped; anything else is
/// passed through (the app never checks those slots).
fn mac_keycode_to_vk(kc: u16) -> u16 {
    match kc {
        0x00 => 0x41,        // A
        0x01 => 0x53,        // S
        0x02 => 0x44,        // D
        0x0D => 0x57,        // W
        0x0C => 0x51,        // Q
        0x0E => 0x45,        // E
        0x38 | 0x3C => 0x10, // Shift (left / right) -> VK_SHIFT
        0x3A | 0x3D => 0x12, // Option (left / right) -> VK_MENU (pointer-lock release chord)
        0x2E => 0x4D,        // M (pointer-lock latch toggle)
        0x30 => 0x09,        // Tab -> VK_TAB
        0x35 => 0x1B,        // Escape -> VK_ESCAPE
        0x78 => 0x71,        // F2
        other => other,
    }
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
                self.input.set_key(mac_keycode_to_vk(kc) as usize, true);
                if kc == KEY_ESCAPE {
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
                let kc = event.keyCode();
                self.input.set_key(mac_keycode_to_vk(kc) as usize, false);
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
