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

mod input;
pub use input::Input;

#[cfg(windows)]
mod window;
#[cfg(windows)]
pub use window::Window;

#[cfg(target_os = "macos")]
mod window_macos;
#[cfg(target_os = "macos")]
pub use window_macos::Window;
