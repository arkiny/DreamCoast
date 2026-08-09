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
//! **A run is more than one dungeon.** The seed identifies the *run*: floor 1 is the seed
//! itself and every deeper floor is derived from it (`game::floor_seed`). Walking onto
//! the exit runs the pipeline above again, at runtime, for the next floor and hands the
//! engine the level it wrote (`sandbox::GameHooks::next_level`) — so everything below
//! about generation, meshing and cook-caching is as true of floor 7 as of floor 1. The
//! only thing `main` owns is the first floor, because it is the only one that must exist
//! before the engine comes up.
//!
//! Run from the workspace root (the engine resolves level and asset paths relative to
//! the working directory):
//!
//! ```text
//! cargo run -p dungeon                       # the default seed, 6 monsters
//! cargo run -p dungeon -- --seed 12345       # any u64
//! cargo run -p dungeon -- --grunts 12        # any monster count (0 = an empty floor)
//! cargo run -p dungeon -- --screenshot-clean tmp/dungeon.png
//!
//! # Headless, driven: DUNGEON_HOLD holds input sources down for a capture sequence,
//! # which has no keyboard of its own, and DUNGEON_TAP presses one for a single step
//! # (an attack is an edge — see `game.rs`).
//! CAPTURE_SEQ=120 DUNGEON_HOLD=W,Shift cargo run -p dungeon --release -- \
//!     --screenshot tmp/walk.png
//! WARMUP_FRAMES=210 CAPTURE_SEQ=2 DUNGEON_HOLD=W DUNGEON_TAP=Mouse1@205 \
//!     cargo run -p dungeon --release -- --screenshot tmp/swing.png
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

mod ai;
mod characters;
mod collision;
mod game;
mod hud;
mod items;
mod level;
mod meshing;
mod pathing;
mod procgen;
mod rigs;
mod warrior;

/// The dungeon generated when no `--seed` is given.
///
/// A fixed constant rather than a clock or an OS random source: the default run must be
/// the *same* dungeon on every machine and every day, so a screenshot, a bug report and
/// a golden capture all describe one place. Pass `--seed <u64>` for any other.
///
/// It seeds the **run**, not one dungeon: floor 1 is this seed itself and every deeper
/// floor is derived from it (`game::floor_seed`), so one number reproduces a whole
/// descent — and the floor the default run opens on is byte-for-byte the one it has
/// always opened on.
const DEFAULT_SEED: u64 = 20260731;

/// Parse `--<flag> <value>` / `--<flag>=<value>` as a `u64`, or `default` when absent.
///
/// A malformed or missing value is a hard error — silently playing a different dungeon
/// (or a different number of monsters) than the one asked for is worse than not starting.
/// Unknown flags are skipped here and fall through to the engine's own scan, which is
/// what lets a game add flags the engine knows nothing about.
fn u64_arg(flag: &str, default: u64) -> anyhow::Result<u64> {
    Ok(opt_u64_arg(flag)?.unwrap_or(default))
}

/// As [`u64_arg`], but reporting **whether the flag was there** rather than substituting
/// a default — the difference between "six monsters" and "however many this floor is
/// worth", which is what `--grunts` decides (see `game::Population`).
fn opt_u64_arg(flag: &str) -> anyhow::Result<Option<u64>> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = match arg.strip_prefix(flag) {
            Some("") => args.next(),                                 // --flag <n>
            Some(rest) => rest.strip_prefix('=').map(str::to_owned), // --flag=<n>
            None => continue,
        };
        let value = value.ok_or_else(|| anyhow::anyhow!("{flag} needs a value (a u64)"))?;
        return value
            .parse::<u64>()
            .map(Some)
            .map_err(|e| anyhow::anyhow!("{flag} '{value}': {e}"));
    }
    Ok(None)
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

    // This game's shadow policy: VIRTUAL SHADOW MAPS (docs/vsm-shadows-plan.md, V3
    // complete): page-granular caching around the follow camera, dynamic objects
    // invalidating only the pages they overlap, mm-scale texels at the warrior. CSM is
    // DEPRECATED for game content — its camera-frustum split scheme needed per-scene
    // coverage contracts (CSM_NEAR/FAR) that VSM's receiver-driven page marking makes
    // moot; the engine keeps the CSM/legacy paths for the anchor scenes and as the
    // `VSM=0` fallback seam (set `CSM=4 CSM_ATLAS=4096 CSM_NEAR=6 CSM_FAR=55` manually
    // to reproduce the pre-VSM look). Same per-app-default seam as the SDF sign probe
    // above: the engine default (and the gallery anchor) stays legacy, and an explicit
    // env setting still wins.
    if std::env::var_os("VSM").is_none() {
        // SAFETY: single-threaded here — before logging, jobs, and engine bring-up
        // spawn any thread.
        unsafe { std::env::set_var("VSM", "1") };
    }

    // This game's motion-vector policy: VELOCITY ON. The default tier renders below
    // the output resolution (TAAU upscale), and TAAU without a velocity target can
    // only camera-reproject — every game-driven mover (warrior, grunts: rigid node
    // rigs pushed by `fixed_update`) then smears a ghost trail whenever it walks.
    // The engine's velocity pass + TAAU's per-pixel velocity reprojection remove
    // exactly that; the engine default stays off (byte-identical gallery anchor),
    // this is the same per-app-default seam as VSM above, and an explicit env
    // setting still wins (`P_VELOCITY=0` reproduces the ghosting for an A/B).
    if std::env::var_os("P_VELOCITY").is_none() {
        // SAFETY: single-threaded here — before logging, jobs, and engine bring-up
        // spawn any thread.
        unsafe { std::env::set_var("P_VELOCITY", "1") };
    }

    // Logging first: generation and meshing below run *before* engine bring-up, and
    // their report (mesh/triangle counts, generation time) is exactly what the M1 risk
    // gate wants to read — so it has to reach the same log stream the engine uses.
    sandbox::init_logging();

    // The characters, first: both levels below reference the two rig `.glb`s by path, so
    // they have to exist before a `.level` naming them is written (and long before the
    // engine's loader tries to cook one).
    rigs::ensure_rigs()?;

    // The game owns the grid *and* the two placement lists (monsters, potions); writing
    // the level *borrows* all three. That ordering is the single-instance guarantee in
    // code form — there is no second `generate()`, `spawn_points()` or
    // `potion_spawn_points()` call to drift, and no way to mesh one dungeon and play
    // another. (Torches are the exception and are derived inside the writer: nothing
    // simulates them, so the level is their only consumer — see `level::torch_points`.)
    let (game, level) = if generated_room_requested() {
        // The injection harness is a room, not an encounter: no monsters (see
        // `level::room_level_data`), and no run to descend through or restart.
        let game =
            game::DungeonGame::new(level::room_collision_grid(), game::Population::Fixed(0))?
                .without_floors();
        let level = level::ensure_generated_room()?.to_owned();
        (game, level)
    } else {
        let seed = u64_arg("--seed", DEFAULT_SEED)?;
        // `--tiles N`: the GDF-scaling stress seam (gdf-scale-follow-plan.md U0).
        // Exported as env so every floor the game generates later matches (the same
        // single-source rule the seed follows).
        let tiles = u64_arg("--tiles", 40)?.clamp(16, 512);
        if tiles != 40 {
            // SAFETY(env): single-threaded startup, before any engine thread exists.
            unsafe { std::env::set_var("DUNGEON_TILES", tiles.to_string()) };
        }
        // An explicit `--grunts` pins the count on every floor; without it the count is
        // the floor's own (`game::grunts_for_floor`), which on floor 1 is the shipped
        // six — so an unflagged run starts exactly as it always has.
        let population = match opt_u64_arg("--grunts")? {
            Some(n) => game::Population::Fixed(n as usize),
            None => game::Population::PerFloor,
        };
        // Floor 1 of the run. Deeper floors are generated by the game itself, through
        // this same function and the same writer — `main` only owns the first one
        // because it is the only one that has to exist before the engine comes up.
        let grid = game::floor_grid(seed, game::FIRST_FLOOR);
        let game = game::DungeonGame::new(grid, population)?;
        let level = level::ensure_dungeon(game.grid(), game.grunt_spawns(), game.potion_spawns())?;
        (game, level)
    };

    // Interactive runs get the real output device; anything headless (screenshot
    // captures, driven CAPTURE_SEQ walks) or `NO_AUDIO=1` keeps the silent Null sink so
    // no capture or test path ever touches a platform audio API.
    let mut game = game;
    let headless = std::env::args().any(|a| a.starts_with("--screenshot"))
        || std::env::var_os("CAPTURE_SEQ").is_some();
    if !headless && std::env::var("NO_AUDIO").ok().as_deref() != Some("1") {
        game.enable_audio();
    }

    sandbox::main_entry(sandbox::GameConfig {
        level: Some(level),
        // This game keeps its levels in its own directory, so the engine's built-in
        // `.level` files are neither written into it nor listed in its hot-swap menu.
        levels_dir: Some(level::GENERATED_DIR.into()),
        hooks: Some(Box::new(game)),
        // The HUD's typeface (game-ui-plan.md M-U2): Alegreya SC, embedded under the
        // SIL OFL 1.1 (license copy beside the file; THIRD_PARTY_LICENSES.md entry).
        // Two atlas sizes: HUD body and overlay titles.
        ui_fonts: vec![sandbox::UiFont {
            bytes: include_bytes!("../assets/fonts/AlegreyaSC-Regular.ttf"),
            sizes_px: vec![
                crate::hud::FONT_BODY_PX,
                crate::hud::FONT_BODY_LG_PX,
                crate::hud::FONT_TITLE_PX,
                crate::hud::FONT_TITLE_LG_PX,
            ],
        }],
    })
}
