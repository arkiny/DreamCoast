//! Platform layer: native Win32 windowing and input.
//!
//! Windows-only by design (see `docs/ROADMAP.md`). The rest of the engine depends on
//! [`Window`] for a surface to render into and [`Input`] for user interaction.

mod input;
mod window;

pub use input::Input;
pub use window::Window;
