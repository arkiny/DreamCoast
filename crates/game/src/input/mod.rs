//! Input: frame snapshots, action bindings, and edge detection
//! (`docs/game-framework-plan.md` §2.2).
//!
//! The platform layer offers **polled state only** — `key_down(vk)`,
//! `mouse_button(i)`, `mouse_delta()` — with no notion of "this frame" versus
//! "last frame", so every consumer that wanted a *tap* hand-rolled its own
//! `*_prev` booleans. This module replaces that pattern with one implementation:
//!
//! 1. [`InputSnapshot::capture`] freezes a frame of platform state as plain data.
//! 2. An [`ActionMap`] binds the game's own action enum to physical sources —
//!    keys by name, mouse buttons, wheel directions — from code or from RON.
//! 3. [`ActionState::update`] consumes one snapshot per frame and answers
//!    [`pressed`](ActionState::pressed),
//!    [`just_pressed`](ActionState::just_pressed),
//!    [`just_released`](ActionState::just_released),
//!    [`axis2d`](ActionState::axis2d).
//!
//! Gameplay code never touches the window: it reads snapshots. That is what makes
//! this testable — the tests in this module drive multi-frame input sequences with
//! no window, device, or swapchain.
//!
//! ```
//! use dreamcoast_game::input::{ActionState, BindingsConfig, InputSnapshot};
//!
//! #[derive(Clone, Copy, PartialEq, Eq, Hash)]
//! enum Action {
//!     MoveForward,
//!     MoveBack,
//!     MoveLeft,
//!     MoveRight,
//!     Attack,
//! }
//!
//! fn action_of(name: &str) -> Option<Action> {
//!     Some(match name {
//!         "MoveForward" => Action::MoveForward,
//!         "MoveBack" => Action::MoveBack,
//!         "MoveLeft" => Action::MoveLeft,
//!         "MoveRight" => Action::MoveRight,
//!         "Attack" => Action::Attack,
//!         _ => return None,
//!     })
//! }
//!
//! let config = BindingsConfig::from_ron(
//!     r#"(actions: {
//!         "MoveForward": ["W"],
//!         "MoveBack": ["S"],
//!         "MoveLeft": ["A"],
//!         "MoveRight": ["D"],
//!         "Attack": ["Mouse1", "Space"],
//!     })"#,
//! )
//! .unwrap();
//! let mut input = ActionState::new(config.resolve(action_of).unwrap());
//!
//! // Per frame: `InputSnapshot::capture(window.input())` in the real loop.
//! input.update(&InputSnapshot::default().with_key(0x57, true).with_mouse_button(0, true));
//!
//! let move_dir = input.axis2d(
//!     Action::MoveLeft,
//!     Action::MoveRight,
//!     Action::MoveBack,
//!     Action::MoveForward,
//! );
//! assert_eq!(move_dir, glam::Vec2::new(0.0, 1.0));
//! assert!(input.just_pressed(Action::Attack));
//! ```

mod action_map;
mod bindings;
pub mod keys;
mod snapshot;
mod source;

pub use action_map::{ActionMap, ActionState};
pub use bindings::BindingsConfig;
pub use snapshot::{InputSnapshot, KEY_COUNT, MOUSE_BUTTON_COUNT};
pub use source::{BindingError, InputSource, WheelDirection};
