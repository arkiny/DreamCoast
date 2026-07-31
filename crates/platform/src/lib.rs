//! Platform layer: native windowing and input.
//!
//! Hand-rolled per OS (no third-party windowing crate), matching the engine's
//! "own the render loop" philosophy:
//!   - Windows: Win32 (`window.rs`).
//!   - macOS: Cocoa/AppKit + a `CAMetalLayer`-backed view (`window_macos.rs`).
//!
//! The rest of the engine depends on [`Window`] for a surface to render into and
//! [`Input`] for user interaction. [`Input`] is platform-agnostic (the per-OS
//! window modules feed it raw events through its `pub(crate)` setters).
//!
//! Because those setters are crate-private, nothing outside this crate can *build*
//! an `Input` — so [`InputSnapshot`] is the transferable, constructible frame of the
//! same state ([`keys`] is the name ↔ code table both sides speak). Engine APIs that
//! hand a frame of input to a consumer (e.g. the sandbox's game hooks) take the
//! snapshot, which is what makes those consumers testable without a window.

mod input;
pub use input::Input;

pub mod keys;
mod snapshot;
pub use snapshot::{InputSnapshot, KEY_COUNT, MOUSE_BUTTON_COUNT};

#[cfg(windows)]
mod window;
#[cfg(windows)]
pub use window::Window;

#[cfg(target_os = "macos")]
mod window_macos;
#[cfg(target_os = "macos")]
pub use window_macos::Window;
