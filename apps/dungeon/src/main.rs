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
//!
//! # M1 static-geometry injection proof: a runtime-GENERATED room, entering the scene
//! # as a real asset file so it collects the per-mesh SDF / GDF / GI / reflection bakes.
//! # `DEBUG_VIEW=9` renders distance-field AO — the generated walls and pillars
//! # occluding there is the proof that the bake saw them (see `level.rs`).
//! cargo run -p dungeon --release -- --generated-room --screenshot-clean tmp/room_ao.png
//! ```

mod game;
mod level;
// Landed ahead of their consumer: the M1 integration step swaps `level::room_meshes`
// for the real generator+mesher output. Until then the modules are test-only (each
// carries its own `#![allow(dead_code)]`).
mod meshing;
mod procgen;

/// Whether `--generated-room` was passed: load the generated-geometry level instead of
/// the walking skeleton. Parsed here rather than in the engine — unknown arguments pass
/// through the engine's own scan untouched, so a game is free to add its own.
fn generated_room_requested() -> bool {
    std::env::args().skip(1).any(|a| a == "--generated-room")
}

fn main() -> anyhow::Result<()> {
    // Logging first: level generation below runs *before* engine bring-up, and its
    // report (mesh/triangle counts, generation time) is exactly what the M1 risk gate
    // wants to read — so it has to reach the same log stream the engine uses.
    sandbox::init_logging();

    // Levels are authored in code (see `level.rs`) and written where the engine's loader
    // is pointed. Doing it before bring-up means a fresh checkout needs no extra step.
    let level = if generated_room_requested() {
        level::ensure_generated_room()?
    } else {
        level::ensure_level_file()?
    };

    sandbox::main_entry(sandbox::GameConfig {
        level: Some(level.to_owned()),
        // This game keeps its levels in its own directory, so the engine's built-in
        // `.level` files are neither written into it nor listed in its hot-swap menu.
        levels_dir: Some(level::GENERATED_DIR.into()),
        hooks: Some(Box::new(game::DungeonGame::new()?)),
    })
}
