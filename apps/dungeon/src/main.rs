//! `dungeon` — the game binary (`docs/game-framework-plan.md` §3).
//!
//! It links the engine as a library and injects gameplay through `sandbox::GameHooks`;
//! the render path, the frame loop and the golden-image gates stay in `sandbox`,
//! untouched. Every engine CLI flag and env var still applies, so the same headless
//! capture the engine uses (`--screenshot-clean out.png`) also captures this game —
//! which is how the seam is verified.
//!
//! **A run is a seed.** The default run generates a dungeon, meshes it, writes it as a
//! `.glb` + `.level` pair and plays it:
//!
//! ```text
//! generate(seed) ─┬─→ mesh_chunks → .glb + .level → the engine's normal level load
//!                 └─→ DungeonGame (the same TileGrid instance) → collision, HUD, exit
//! ```
//!
//! One grid, two consumers: the geometry is written from it before bring-up and the
//! player collides against it during play, so the walls you see and the walls you hit
//! cannot drift apart (`main` owns the ordering; the game owns the grid).
//!
//! Run from the workspace root (the engine resolves level and asset paths relative to
//! the working directory):
//!
//! ```text
//! cargo run -p dungeon                       # the default seed
//! cargo run -p dungeon -- --seed 12345       # any u64
//! cargo run -p dungeon -- --screenshot-clean tmp/dungeon.png
//!
//! # Headless, driven: DUNGEON_HOLD holds keys down for a capture sequence, which has
//! # no keyboard of its own (see `game.rs`).
//! CAPTURE_SEQ=120 DUNGEON_HOLD=W,Shift cargo run -p dungeon --release -- \
//!     --screenshot tmp/walk.png
//!
//! # M1 static-geometry injection proof: the minimal runtime-GENERATED room, entering
//! # the scene as a real asset file so it collects the per-mesh SDF / GDF / GI /
//! # reflection bakes. `DEBUG_VIEW=9` renders distance-field AO — generated geometry
//! # occluding there is the proof that the bake saw it (see `level.rs`).
//! cargo run -p dungeon --release -- --generated-room --screenshot-clean tmp/room_ao.png
//! ```
//!
//! Unknown arguments pass through the engine's own scan untouched, which is what lets a
//! game add its own flags without the engine knowing about them.

mod collision;
mod game;
mod level;
mod meshing;
mod procgen;

use procgen::DungeonParams;

/// The dungeon generated when no `--seed` is given.
///
/// A fixed constant rather than a clock or an OS random source: the default run must be
/// the *same* dungeon on every machine and every day, so a screenshot, a bug report and
/// a golden capture all describe one place. Pass `--seed <u64>` for any other.
const DEFAULT_SEED: u64 = 20260731;

/// Parse `--seed <u64>`. A malformed or missing value is a hard error — silently
/// playing a different dungeon than the one asked for is worse than not starting.
fn seed_from_args() -> anyhow::Result<u64> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = match arg.strip_prefix("--seed") {
            Some("") => args.next(),                                 // --seed <n>
            Some(rest) => rest.strip_prefix('=').map(str::to_owned), // --seed=<n>
            None => continue,
        };
        let value = value.ok_or_else(|| anyhow::anyhow!("--seed needs a value (a u64)"))?;
        return value
            .parse::<u64>()
            .map_err(|e| anyhow::anyhow!("--seed '{value}': {e}"));
    }
    Ok(DEFAULT_SEED)
}

/// Whether `--generated-room` was passed: load the minimal injection-proof level
/// instead of a generated dungeon.
fn generated_room_requested() -> bool {
    std::env::args().skip(1).any(|a| a == "--generated-room")
}

fn main() -> anyhow::Result<()> {
    // This game's SDF sign policy: the probe-based inversion decision. The legacy
    // voxel-majority flip coin-flips on this game's open-shell chunk meshes (zero
    // enclosed volume, >60% half-space contamination) and measurably inverts room
    // air to solid — an 80/255 AO plateau with a hard chunk-seam edge. The engine
    // default stays legacy until the open-shell contamination work lands (the PT
    // gates show the legacy flips still net-help IntelSponza); this is a per-app
    // default, not a per-scene patch, and an explicit env setting still wins.
    if std::env::var_os("P_SDF_SIGN_PROBE").is_none() {
        // SAFETY: single-threaded here — before logging, jobs, and engine bring-up
        // spawn any thread.
        unsafe { std::env::set_var("P_SDF_SIGN_PROBE", "1") };
    }

    // Logging first: generation and meshing below run *before* engine bring-up, and
    // their report (mesh/triangle counts, generation time) is exactly what the M1 risk
    // gate wants to read — so it has to reach the same log stream the engine uses.
    sandbox::init_logging();

    // The game owns the grid; writing the level *borrows* it. That ordering is the
    // single-instance guarantee in code form — there is no second `generate()` call to
    // drift, and no way to mesh one dungeon and play another.
    let (game, level) = if generated_room_requested() {
        let game = game::DungeonGame::new(level::room_collision_grid())?;
        let level = level::ensure_generated_room()?.to_owned();
        (game, level)
    } else {
        let seed = seed_from_args()?;
        let game = game::DungeonGame::new(procgen::generate(seed, &DungeonParams::default()))?;
        let level = level::ensure_dungeon(game.grid())?;
        (game, level)
    };

    sandbox::main_entry(sandbox::GameConfig {
        level: Some(level),
        // This game keeps its levels in its own directory, so the engine's built-in
        // `.level` files are neither written into it nor listed in its hot-swap menu.
        levels_dir: Some(level::GENERATED_DIR.into()),
        hooks: Some(Box::new(game)),
    })
}
