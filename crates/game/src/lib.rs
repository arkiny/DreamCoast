//! `dreamcoast-game` — the reusable **game framework** layer (see
//! `docs/game-framework-plan.md` §1).
//!
//! This crate sits between the engine crates and an actual game binary: it is
//! **engine-dependent but game-independent**. Anything here must be useful to any
//! title built on DreamCoast, not to one scene — game-specific rules (dungeon
//! generation, class definitions, HUD layout) belong in the game app instead.
//!
//! It is deliberately **RHI-free and renderer-free**: it speaks `glam`, the
//! platform input state, and plain data. That keeps it unit-testable without a
//! window, a device, or a swapchain, and keeps the renderer's golden-image gates
//! insulated from gameplay work.
//!
//! Modules land per milestone: M0 opened with [`input`] (action mapping + edge
//! detection) and [`physics`] (grid collision, raycasts, combat shapes); M2 adds
//! [`combat`] (health, the melee phase clock, the damage flow) and [`anim`] (a
//! data-driven animation state machine). Camera follows in a later milestone and
//! is not stubbed out ahead of time.

pub mod anim;
pub mod combat;
pub mod input;
pub mod physics;
