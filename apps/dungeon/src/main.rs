//! `dungeon` — the game binary (`docs/game-framework-plan.md` §2.4).
//!
//! It links the engine as a library and injects gameplay through `sandbox::GameHooks`;
//! the render path, the frame loop and the golden-image gates stay in `sandbox`,
//! untouched. Every engine CLI flag and env var still applies, so the same headless
//! capture the engine uses (`--screenshot-clean out.png`) also captures this game —
//! which is how the seam is verified.
//!
//! Run from the workspace root (the engine resolves level and asset paths relative to
//! the working directory):
//!
//! ```text
//! cargo run -p dungeon
//! cargo run -p dungeon -- --screenshot-clean tmp/dungeon.png
//! ```

mod game;
mod level;

fn main() -> anyhow::Result<()> {
    // The level is authored in code (see `level.rs`) and written where the engine's
    // loader looks. Doing it before bring-up means a fresh checkout needs no extra step.
    level::ensure_level_file()?;

    sandbox::main_entry(sandbox::GameConfig {
        level: Some(level::LEVEL_NAME.to_owned()),
        hooks: Some(Box::new(game::DungeonGame::new()?)),
    })
}
