//! The game: one [`GameHooks`] implementation driving the player through a generated
//! dungeon, a top-down follow camera and a HUD (`docs/game-framework-plan.md` §3).
//!
//! It proves the whole seam — simulation writes into the ECS from
//! [`GameHooks::fixed_update`], the view comes from [`GameHooks::camera`], the visual
//! pose from [`GameHooks::render_update`] and the overlay from [`GameHooks::draw_ui`] —
//! with no renderer changes at all.
//!
//! **The grid is the game.** [`DungeonGame`] owns the one [`TileGrid`] the dungeon was
//! generated as. The level writer borrowed it to build the geometry *before* the engine
//! came up ([`crate::level::ensure_dungeon`]), and the mover collides against that same
//! instance here — so what you walk into is what was meshed, with no second generate()
//! call to drift.
//!
//! The one way to break that pairing is the engine's **level hot-swap dropdown**: it
//! lists every `dungeon_<seed>.level` this machine has generated, and picking a
//! different one loads new geometry while this grid stays the old dungeon. The player is
//! re-acquired and re-snapped to free space (see [`DungeonGame::acquire_player`]), but it
//! will be free space *of the previous floor*. Swapping floors properly is M3's
//! progression work (generate → write → load → hand the game the new grid); until then,
//! a seed change means a restart with `--seed`.
//!
//! **Fixed-step discipline.** Everything that integrates (velocity, camera smoothing)
//! advances in `fixed_update` at the engine's fixed `dt`, and `camera` /
//! `render_update` only *blend* the two latest states by the frame's `alpha`. That keeps
//! motion framerate-independent and the follow camera free of the jitter a per-frame
//! smoothing filter would add.

use dreamcoast_game::input::{ActionState, BindingsConfig, InputSnapshot, keys};
use glam::{Vec2, Vec3};
use sandbox::imgui;
use sandbox::scene::{Entity, LocalTransform, Name, World};
use sandbox::{CameraPose, GameHooks};

use crate::collision::{self, PLAYER_RADIUS};
use crate::level::PLAYER_NAME;
use crate::procgen::{Tile, TileGrid};

/// Gameplay actions. The physical keys live in `config/bindings.ron` (data), so a
/// rebind never touches this file.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Action {
    MoveForward,
    MoveBack,
    MoveLeft,
    MoveRight,
    Sprint,
}

impl Action {
    /// Resolve a binding-file action name. Unknown names are a config error.
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "MoveForward" => Self::MoveForward,
            "MoveBack" => Self::MoveBack,
            "MoveLeft" => Self::MoveLeft,
            "MoveRight" => Self::MoveRight,
            "Sprint" => Self::Sprint,
            _ => return None,
        })
    }
}

/// The shipped bindings, embedded so the game runs from any working directory.
const DEFAULT_BINDINGS: &str = include_str!("../config/bindings.ron");

/// Keys to hold down for the whole run, as a comma-separated list of the same names the
/// bindings file uses (`DUNGEON_HOLD=W,Shift`). Empty/unset = normal input.
///
/// Headless capture has no keyboard: `--screenshot`/`CAPTURE_SEQ` pump no window events,
/// so a captured frame would always show a player standing on the spawn. This makes the
/// *headless* run drivable — `CAPTURE_SEQ=N DUNGEON_HOLD=W` walks N deterministic fixed
/// steps into whatever is north of the spawn — which is how the "walk into a wall
/// without panicking" sanity check is run against the real binary rather than a test
/// harness. Game-side and off by default; the engine knows nothing about it.
const HOLD_KEYS_ENV: &str = "DUNGEON_HOLD";

/// Ground speed, metres/second.
const WALK_SPEED: f32 = 4.5;
const SPRINT_SPEED: f32 = 8.5;
/// Time constant of the velocity ramp (seconds to ~63% of the target speed): the
/// placeholder accelerates instead of snapping, which is what makes the interpolated
/// follow camera worth having.
const ACCEL_TIME: f32 = 0.09;
/// Time constant of the camera's positional smoothing. Larger = lazier follow.
const CAMERA_SMOOTH_TIME: f32 = 0.14;
/// Downward tilt of the top-down camera, degrees from horizontal. Kept off 90° on
/// purpose: the view basis is built against world +Y, so a perfectly vertical view
/// axis is degenerate (see `sandbox::CameraPose`).
const CAMERA_PITCH_DEG: f32 = 55.0;
/// Eye distance from the followed point, metres.
///
/// Nudged up from the walking skeleton's 12 m once the flat plate became corridors,
/// measured on the same frame of the same run (a sprint north from the spawn, stopped
/// against the corridor's end wall — `tmp/m1_cam12` vs `tmp/m1_wall`):
///
/// * at **12 m** the eye is 9.8 m up and the near wall tops crowd the bottom of the
///   frame; the player ends up in a small pocket of visible floor near the frame edge
///   with the corridor's continuation off-screen;
/// * at **16 m** the eye is 13.1 m up — comfortably over the 4 m walls — and the same
///   moment shows the junction the player is standing in *plus* the rooms either side,
///   which is the information a top-down action game is asking the camera for.
///
/// With the frame's 60° vertical FOV that covers ~27 m of ground along Z, so the
/// largest room the generator places (9x9 tiles = 18 m) still fits.
///
/// What it does **not** fix: a wall between the camera and the player still occludes,
/// because these walls are opaque 4 m slabs. Standing in a corridor you sometimes look
/// at the near wall's top rather than at yourself. The real answers (shorter walls, or
/// fading geometry between eye and player) are camera/art work for M2+, not a constant.
const CAMERA_DISTANCE: f32 = 16.0;

/// Eye offset from the followed point: up and back along +Z, so screen-up is world
/// -Z and screen-right is world +X. The camera never yaws, which is exactly why
/// camera-relative movement below can be world-aligned.
pub fn camera_offset() -> Vec3 {
    let (sin, cos) = CAMERA_PITCH_DEG.to_radians().sin_cos();
    Vec3::new(0.0, sin, cos) * CAMERA_DISTANCE
}

/// Framerate-independent exponential approach factor for a step of `dt` with time
/// constant `tau`: `x += (target - x) * blend(dt, tau)`.
fn blend(dt: f32, tau: f32) -> f32 {
    1.0 - (-dt / tau.max(1e-4)).exp()
}

/// Virtual-key codes named by [`HOLD_KEYS_ENV`], or empty when it is unset.
fn held_keys_from_env() -> Vec<u16> {
    let Ok(spec) = std::env::var(HOLD_KEYS_ENV) else {
        return Vec::new();
    };
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|name| match keys::key_vk(name) {
            Some(vk) => Some(vk),
            None => {
                tracing::warn!("{HOLD_KEYS_ENV}: unknown key name '{name}' ignored");
                None
            }
        })
        .collect()
}

pub struct DungeonGame {
    input: ActionState<Action>,
    /// The dungeon: geometry source (already meshed, before bring-up) and collision
    /// world (right here). One instance, so the two cannot disagree.
    grid: TileGrid,
    /// Keys forced down every step (see [`HOLD_KEYS_ENV`]); empty in a normal run.
    forced_keys: Vec<u16>,
    /// The player placeholder's entity, resolved lazily from the loaded level.
    player: Option<Entity>,
    /// Simulation state, in fixed steps: previous and current, so the render frame can
    /// interpolate between them. World space, `y` at [`PLAYER_RADIUS`].
    prev_position: Vec3,
    position: Vec3,
    velocity: Vec3,
    /// The point the camera follows (a smoothed trail behind `position`), also kept as
    /// a previous/current pair for interpolation.
    prev_focus: Vec3,
    focus: Vec3,
    /// Latched once the player's circle first touches the exit tile. Progression to the
    /// next floor is M3; M1 detects and surfaces it.
    exit_reached: bool,
    /// Fixed steps simulated so far — a cheap "is the sim actually running" readout.
    steps: u64,
    /// The interpolation factor of the frame being drawn, latched by the camera hook
    /// (which is the only hook the frame loop hands it to) so the HUD can report the
    /// rendered position rather than the last simulated one.
    render_alpha: f32,
}

impl DungeonGame {
    /// Take ownership of the dungeon the level was written from.
    pub fn new(grid: TileGrid) -> anyhow::Result<Self> {
        let bindings = BindingsConfig::from_ron(DEFAULT_BINDINGS)
            .map_err(|e| anyhow::anyhow!("dungeon bindings: {e}"))?;
        let map = bindings
            .resolve(Action::from_name)
            .map_err(|e| anyhow::anyhow!("dungeon bindings: {e}"))?;
        let spawn = collision::player_spawn(&grid);
        Ok(Self {
            input: ActionState::new(map),
            grid,
            forced_keys: held_keys_from_env(),
            player: None,
            prev_position: spawn,
            position: spawn,
            velocity: Vec3::ZERO,
            prev_focus: spawn,
            focus: spawn,
            exit_reached: false,
            steps: 0,
            render_alpha: 0.0,
        })
    }

    /// The dungeon this game will be played in — what [`crate::level::ensure_dungeon`]
    /// meshes into the `.level` the engine then loads.
    pub fn grid(&self) -> &TileGrid {
        &self.grid
    }

    /// Identify the player placeholder in a freshly loaded level, by the scene-graph
    /// name the level authored on it ([`PLAYER_NAME`]).
    ///
    /// Both sides read the same constant, so a renamed or moved placeholder cannot
    /// silently break the lookup — which a spawn-transform match could: it would
    /// identify the player as "the renderable standing where the player starts", so a
    /// second entity placed at the spawn, or a spawn that moved, quietly took over.
    fn find_player(world: &World) -> Option<Entity> {
        world
            .iter::<Name>()
            .find(|(_, name)| name.0 == PLAYER_NAME)
            .map(|(entity, _)| entity)
    }

    /// Re-resolve the player after a level (re)load, seeding the sim state from the
    /// entity's authored transform so the camera does not lurch.
    ///
    /// The authored position is snapped through `nearest_free` before it is believed:
    /// the level writer places the placeholder on exactly this point, but a hot-swapped
    /// or hand-edited level need not, and starting the simulation inside rock is the one
    /// state the mover cannot be asked to fix mid-stride.
    fn acquire_player(&mut self, world: &World) -> Option<Entity> {
        if let Some(e) = self.player
            && world.is_alive(e)
        {
            return Some(e);
        }
        let entity = Self::find_player(world)?;
        let authored = world
            .get::<LocalTransform>(entity)
            .map(|t| t.translation)
            .unwrap_or_else(|| collision::player_spawn(&self.grid));
        let local = collision::to_collision(&self.grid, authored);
        let start = collision::collision(&self.grid)
            .nearest_free(local, PLAYER_RADIUS)
            .map(|p| collision::to_world(&self.grid, p, PLAYER_RADIUS))
            .unwrap_or_else(|| collision::player_spawn(&self.grid));
        self.player = Some(entity);
        self.prev_position = start;
        self.position = start;
        self.velocity = Vec3::ZERO;
        self.prev_focus = start;
        self.focus = start;
        Some(entity)
    }

    /// Desired ground velocity from this step's input. The camera's yaw is fixed, so
    /// camera-relative input maps straight onto world axes: +X right, -Z away.
    fn wish_velocity(&self) -> Vec3 {
        let stick: Vec2 = self.input.axis2d(
            Action::MoveLeft,
            Action::MoveRight,
            Action::MoveBack,
            Action::MoveForward,
        );
        // Clamp (not normalize): diagonals must not be faster, but a partially held
        // direction must not be boosted to full speed either.
        let stick = if stick.length_squared() > 1.0 {
            stick.normalize()
        } else {
            stick
        };
        let speed = if self.input.pressed(Action::Sprint) {
            SPRINT_SPEED
        } else {
            WALK_SPEED
        };
        Vec3::new(stick.x, 0.0, -stick.y) * speed
    }

    /// The player's speed readout, metres/second.
    fn speed(&self) -> f32 {
        self.velocity.length()
    }

    /// The tile the player is standing on (grid coordinates).
    fn player_tile(&self) -> (i32, i32) {
        collision::tile_of(collision::to_collision(&self.grid, self.position))
    }

    /// Where the player *is* this rendered frame: the two latest fixed-step positions
    /// blended by the frame's interpolation factor. The camera, the HUD readout and the
    /// placeholder mesh all read this one function, so they cannot disagree.
    fn render_position(&self, alpha: f32) -> Vec3 {
        self.prev_position.lerp(self.position, alpha)
    }

    /// One simulation step against one frame of input.
    ///
    /// This *is* the hook body — `fixed_update` forwards straight to it — because the
    /// engine hands games an `InputSnapshot` rather than the platform `Input` whose
    /// state has no public setter. So the whole step is exercisable from a test with no
    /// window, device or swapchain (see below), against the same code the game runs.
    fn simulate(&mut self, world: &mut World, snapshot: &InputSnapshot, dt: f32) {
        // One snapshot per step. Every step of a frame sees the same platform state
        // (the window is pumped once per frame), so an edge is reported on the first
        // step of the frame it happened in and not repeated by the later ones.
        if self.forced_keys.is_empty() {
            self.input.update(snapshot);
        } else {
            let mut forced = snapshot.clone();
            for &vk in &self.forced_keys {
                forced.set_key(vk, true);
            }
            self.input.update(&forced);
        }

        if self.acquire_player(world).is_none() {
            return; // level not loaded (or has no placeholder) — nothing to simulate
        }
        self.steps += 1;

        // Velocity ramp, then move against the grid. The mover is 2D on the XZ plane in
        // the grid's own space (see `collision`), so this is the one place the two
        // coordinate systems meet.
        let wish = self.wish_velocity();
        self.velocity += (wish - self.velocity) * blend(dt, ACCEL_TIME);
        self.prev_position = self.position;

        let local = collision::to_collision(&self.grid, self.position);
        let delta = Vec2::new(self.velocity.x, self.velocity.z) * dt;
        let moved = collision::collision(&self.grid).move_circle(local, PLAYER_RADIUS, delta);
        // A blocked axis loses its velocity rather than keeping it pressed into the
        // wall: without this, walking into a wall for a second and then turning away
        // would release a second's worth of stored speed. The unblocked axis is
        // untouched, which is what makes sliding along a wall cost nothing.
        if moved.hit_x {
            self.velocity.x = 0.0;
        }
        if moved.hit_z {
            self.velocity.z = 0.0;
        }
        self.position = collision::to_world(&self.grid, moved.pos, PLAYER_RADIUS);

        // The mover guarantees a finite, outside-geometry result for finite inputs; this
        // is the tripwire for the day something upstream (a NaN dt, a corrupt level)
        // breaks that assumption, caught here instead of as a vanished player.
        assert!(
            self.position.is_finite() && self.velocity.is_finite(),
            "dungeon: non-finite player state (pos {:?}, vel {:?})",
            self.position,
            self.velocity
        );

        // Exit detection (M1 surfaces it; M3 makes it progression).
        if !self.exit_reached
            && self.grid.exit() != self.grid.entry()
            && collision::circle_overlaps_tile(moved.pos, PLAYER_RADIUS, self.grid.exit())
        {
            self.exit_reached = true;
            let (ex, ez) = self.grid.exit();
            tracing::info!(
                "dungeon: exit reached at tile ({ex}, {ez}) after {} steps ({:.1} s)",
                self.steps,
                self.steps as f64 * f64::from(dt),
            );
        }

        // Note there is no ECS write here: `render_update` is the single writer of the
        // placeholder's transform, so the mesh draws at the *interpolated* pose the
        // camera is using and the two can never disagree.

        // The camera follows a smoothed trail behind the player, advanced on the same
        // fixed step so the filter is framerate-independent.
        self.prev_focus = self.focus;
        self.focus += (self.position - self.focus) * blend(dt, CAMERA_SMOOTH_TIME);
    }
}

impl GameHooks for DungeonGame {
    fn fixed_update(&mut self, world: &mut World, input: &InputSnapshot, dt: f32) {
        self.simulate(world, input, dt);
    }

    /// Push the *rendered* player position onto the placeholder's transform, once per
    /// frame, right before the draw list is built.
    ///
    /// Without this the mesh draws at the last simulated pose while the camera is
    /// already at the interpolated one, so at high frame rates the player visibly trails
    /// the view by up to a step of travel (~14 cm at sprint). Both now read
    /// [`Self::render_position`] with the same `alpha`, so they move as one.
    ///
    /// Visual-only, per the hook contract: nothing here feeds back into the simulation —
    /// the next `fixed_update` integrates from `self.position`, never from the ECS.
    fn render_update(&mut self, world: &mut World, alpha: f32) {
        let Some(player) = self.player.filter(|&e| world.is_alive(e)) else {
            return;
        };
        let rendered = self.render_position(alpha);
        if let Some(local) = world.get_mut::<LocalTransform>(player) {
            local.translation = rendered;
        }
    }

    fn camera(&mut self, alpha: f32) -> Option<CameraPose> {
        // Blend the two latest sim states — the hook must return the *rendered* pose;
        // the frame loop does not interpolate it. Smoothing itself happened on the
        // fixed step, so what is left here is a pure lerp: no per-frame filtering, no
        // jitter when several frames fall inside one step.
        self.render_alpha = alpha;
        let focus = self.prev_focus.lerp(self.focus, alpha);
        Some(CameraPose::look_at(focus + camera_offset(), focus))
    }

    fn draw_ui(&mut self, ui: &imgui::Ui, world: &World) {
        let dt_ms = ui.io().delta_time * 1000.0;
        ui.window("Dungeon")
            // Below the engine's debug window (which opens at the top-left corner at
            // 320 px tall), so the HUD is a readable sibling rather than a cover.
            .position([16.0, 372.0], imgui::Condition::FirstUseEver)
            .size([280.0, 220.0], imgui::Condition::FirstUseEver)
            .build(|| {
                ui.text(format!("seed      {}", self.grid.seed()));
                ui.text(format!(
                    "dungeon   {}x{} tiles, {} rooms",
                    self.grid.width(),
                    self.grid.height(),
                    self.grid.rooms().len()
                ));
                ui.separator();
                match self.player {
                    Some(e) if world.is_alive(e) => {
                        ui.text(format!("player entity  #{}", e.index()));
                    }
                    // A single static headless capture never runs a sim step (the
                    // capture path is frame-counted), so this is the expected reading
                    // there; `CAPTURE_SEQ` does step and resolves it.
                    _ => ui.text_disabled("player not resolved yet"),
                }
                let p = self.render_position(self.render_alpha);
                let (tx, tz) = self.player_tile();
                ui.text(format!("position  {:.2}, {:.2}, {:.2}", p.x, p.y, p.z));
                ui.text(format!(
                    "tile      ({tx}, {tz})  {}",
                    tile_label(self.grid.get(tx, tz))
                ));
                ui.text(format!("speed     {:.2} m/s", self.speed()));
                let (ex, ez) = self.grid.exit();
                if self.exit_reached {
                    ui.text(format!("exit      REACHED ({ex}, {ez})"));
                } else {
                    ui.text_disabled(format!("exit      not reached ({ex}, {ez})"));
                }
                ui.text(format!("frame     {dt_ms:.2} ms"));
                ui.text(format!("sim steps {}", self.steps));
                ui.separator();
                ui.text_disabled("WASD move, Shift sprint");
            });
    }
}

/// Human-readable tile name for the HUD.
fn tile_label(tile: Tile) -> &'static str {
    match tile {
        Tile::Wall => "wall",
        Tile::Floor => "floor",
        Tile::Door => "door",
        Tile::Entry => "entry",
        Tile::Exit => "exit",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procgen::{DungeonParams, TILE_SIZE, generate};
    use sandbox::scene::{MaterialHandle, MeshHandle, MeshInstance};

    const W: u16 = 0x57;
    const S: u16 = 0x53;
    const D: u16 = 0x44;
    const SHIFT: u16 = 0x10;
    const FIXED_DT: f32 = 1.0 / 60.0;
    /// Seeds every property below is swept over. A fixed list, so a failure is a repro.
    const SEEDS: std::ops::Range<u64> = 0..20;

    fn dungeon(seed: u64) -> TileGrid {
        generate(seed, &DungeonParams::default())
    }

    fn game(grid: TileGrid) -> DungeonGame {
        DungeonGame::new(grid).unwrap()
    }

    /// A stand-in for the loaded level: an unnamed geometry entity plus the *named*
    /// player placeholder at the authored spawn — i.e. exactly what the level loader
    /// produces from `level.rs`.
    fn test_world(grid: &TileGrid) -> World {
        let mut world = World::new();
        let renderable = |i: u32| MeshInstance::new(MeshHandle(i), MaterialHandle(i));
        world
            .spawn_node()
            .with(renderable(0))
            .with(LocalTransform::default());
        world
            .spawn_node()
            .named(PLAYER_NAME)
            .with(renderable(1))
            .with(LocalTransform {
                translation: collision::player_spawn(grid),
                ..LocalTransform::default()
            });
        world
    }

    /// Advance the game the way the frame loop does: whole fixed steps, then the
    /// once-per-frame presentation pass that writes the visual transform.
    fn frame(game: &mut DungeonGame, world: &mut World, input: &InputSnapshot, steps: u32) {
        for _ in 0..steps {
            game.simulate(world, input, FIXED_DT);
        }
        game.render_update(world, 1.0);
    }

    /// The bindings file must resolve — a typo in it is a startup failure, so catch it
    /// in a test rather than on the player's machine.
    #[test]
    fn bindings_resolve() {
        assert!(DungeonGame::new(dungeon(1)).is_ok());
    }

    /// The whole frame, end to end: the placeholder is identified by name in a
    /// level-like world, held input accelerates it, and the result lands on its ECS
    /// transform.
    #[test]
    fn a_held_key_moves_the_player_entity_in_the_ecs() {
        // A seed whose spawn has open floor to the north, so "held W moves" is a
        // statement about the mover and not about which wall happens to be there.
        let grid = dungeon(3);
        let spawn = collision::player_spawn(&grid);
        let mut g = game(grid);
        let mut world = test_world(g.grid());
        frame(
            &mut g,
            &mut world,
            &InputSnapshot::default().with_key(W, true),
            12,
        );
        let player = g.player.expect("placeholder identified from the level");
        let moved = world.get::<LocalTransform>(player).unwrap().translation;
        assert!(moved.z < spawn.z - 0.5, "walked away from the camera");
        assert_eq!(moved.x, spawn.x);
        assert_eq!(moved.y, PLAYER_RADIUS, "stays on the floor");
        // The camera trails the player instead of snapping onto it.
        assert!(g.focus.z > moved.z && g.focus.z <= spawn.z);
    }

    /// An unnamed placeholder is not the player, however suggestively it is placed —
    /// the lookup is by name, not by "whatever stands at the spawn".
    #[test]
    fn an_unnamed_entity_at_the_spawn_is_not_the_player() {
        let grid = dungeon(3);
        let spawn = collision::player_spawn(&grid);
        let mut g = game(grid);
        let mut world = World::new();
        world
            .spawn_node()
            .with(MeshInstance::new(MeshHandle(0), MaterialHandle(0)))
            .with(LocalTransform {
                translation: spawn,
                ..LocalTransform::default()
            });
        frame(
            &mut g,
            &mut world,
            &InputSnapshot::default().with_key(W, true),
            30,
        );
        assert!(g.player.is_none());
        assert_eq!(g.steps, 0, "no player, no simulation");
    }

    /// The presentation pass is what moves the mesh, and it moves it to the *rendered*
    /// (interpolated) position — the same one the camera and the HUD read — not to the
    /// last simulated one. Mid-step, that is a real difference.
    #[test]
    fn render_update_writes_the_interpolated_pose() {
        let grid = dungeon(3);
        let spawn = collision::player_spawn(&grid);
        let mut g = game(grid);
        let mut world = test_world(g.grid());
        let held = InputSnapshot::default().with_key(W, true);
        for _ in 0..12 {
            g.simulate(&mut world, &held, FIXED_DT);
        }
        let player = g.player.unwrap();
        // Simulation alone leaves the mesh where the level authored it.
        assert_eq!(
            world.get::<LocalTransform>(player).unwrap().translation,
            spawn
        );

        g.render_update(&mut world, 0.5);
        let drawn = world.get::<LocalTransform>(player).unwrap().translation;
        assert_eq!(drawn, g.prev_position.lerp(g.position, 0.5));
        assert!(
            drawn.z > g.position.z && drawn.z < g.prev_position.z,
            "the drawn pose sits strictly between the two sim states"
        );
    }

    /// Holding "forward" must move the placeholder toward -Z and ramp up, not teleport.
    #[test]
    fn forward_input_moves_along_negative_z() {
        let mut g = game(dungeon(1));
        g.input.update(&InputSnapshot::default().with_key(W, true));
        let wish = g.wish_velocity();
        assert!(wish.z < 0.0 && wish.x == 0.0);
        assert!((wish.length() - WALK_SPEED).abs() < 1e-4);
    }

    /// Sprint is a speed multiplier on the same direction.
    #[test]
    fn sprint_is_faster_than_walk() {
        let mut g = game(dungeon(1));
        g.input.update(
            &InputSnapshot::default()
                .with_key(W, true)
                .with_key(SHIFT, true),
        );
        assert!((g.wish_velocity().length() - SPRINT_SPEED).abs() < 1e-4);
    }

    /// The camera sits above and behind its focus, tilted off vertical (a perfectly
    /// vertical view axis is degenerate for the frame loop's look-at), and high enough
    /// to clear the walls it is looking over.
    #[test]
    fn camera_pose_is_tilted_and_above_the_walls() {
        let grid = dungeon(1);
        let spawn = collision::player_spawn(&grid);
        let mut g = game(grid);
        let pose = g.camera(1.0).expect("hook always drives the camera");
        assert!(pose.position.y > crate::meshing::WALL_HEIGHT);
        assert!(pose.position.z > spawn.z);
        let forward = pose.forward();
        assert!(forward.y < -0.5, "camera must look down");
        assert!(forward.y > -0.999, "but not straight down");
    }

    // -- integration properties ------------------------------------------------------

    /// Every seed must spawn the player in free space, on the entry tile — the state
    /// the whole simulation starts from.
    #[test]
    fn the_player_spawns_in_free_space_on_every_seed() {
        for seed in SEEDS {
            let grid = dungeon(seed);
            let mut g = game(grid);
            let mut world = test_world(g.grid());
            // One idle step is enough to acquire the placeholder and resolve the spawn.
            frame(&mut g, &mut world, &InputSnapshot::default(), 1);
            let local = collision::to_collision(g.grid(), g.position);
            assert!(
                !collision::collision(g.grid()).circle_overlaps(local, PLAYER_RADIUS),
                "seed {seed}: spawned inside geometry"
            );
            assert_eq!(g.player_tile(), g.grid().entry(), "seed {seed}");
            assert!(!g.exit_reached, "seed {seed}: spawn is not the exit");
        }
    }

    /// Every seed's exit must be *walkable to* from the spawn: the dungeon is a game
    /// level, not a picture of one.
    #[test]
    fn the_exit_is_reachable_from_the_spawn_on_every_seed() {
        for seed in SEEDS {
            let grid = dungeon(seed);
            let spawn_tile = collision::tile_of(collision::to_collision(
                &grid,
                collision::player_spawn(&grid),
            ));
            let dist = grid.bfs_distances(spawn_tile);
            let (ex, ez) = grid.exit();
            let steps = dist[(ez * grid.width() + ex) as usize];
            assert_ne!(steps, u32::MAX, "seed {seed}: exit unreachable from spawn");
            assert!(
                steps > 0,
                "seed {seed}: exit coincides with the spawn tile — nothing to walk"
            );
            assert!(grid.all_walkable_reachable(), "seed {seed}");
        }
    }

    /// The mover's contract, exercised through the game: hold a direction into a wall
    /// for a full second of fixed steps and the player never ends up inside solid rock,
    /// never leaves the dungeon, and never goes non-finite.
    ///
    /// Run from every seed's spawn in all four directions, so the wall found is whatever
    /// the generator actually put there (some directions are open corridors — those are
    /// the sliding case, and they are checked by the same invariants).
    #[test]
    fn walking_into_walls_never_ends_inside_solid() {
        for seed in SEEDS {
            for key in [W, S, D, 0x41 /* A */] {
                let grid = dungeon(seed);
                let mut g = game(grid);
                let mut world = test_world(g.grid());
                let held = InputSnapshot::default().with_key(key, true).with_key(
                    SHIFT,
                    // Sprint on half the runs: the fastest step is the one most likely
                    // to tunnel, and it must not.
                    key == W || key == S,
                );
                for step in 0..60 {
                    g.simulate(&mut world, &held, FIXED_DT);
                    let local = collision::to_collision(g.grid(), g.position);
                    assert!(
                        g.position.is_finite(),
                        "seed {seed} key {key:#x} step {step}: non-finite"
                    );
                    assert!(
                        !collision::collision(g.grid()).circle_overlaps(local, PLAYER_RADIUS),
                        "seed {seed} key {key:#x} step {step}: inside solid at {:?}",
                        g.position
                    );
                    let (tx, tz) = g.player_tile();
                    assert!(
                        g.grid().in_bounds(tx, tz) && g.grid().is_walkable(tx, tz),
                        "seed {seed} key {key:#x} step {step}: left the dungeon at ({tx}, {tz})"
                    );
                }
                // A second of walking must have gone *somewhere* unless the spawn is
                // walled in that direction — and even then, the player must not have
                // been pushed backwards through a wall.
                assert!(
                    g.position.distance(collision::player_spawn(g.grid())) < 60.0 * WALK_SPEED,
                    "seed {seed} key {key:#x}: implausible travel"
                );
            }
        }
    }

    /// Walking onto the exit tile latches the flag the HUD reports — and nothing else
    /// does. Driven by teleporting the *authored* spawn to the exit's neighbour, so the
    /// detection is exercised through the same simulate() the game runs.
    #[test]
    fn touching_the_exit_latches_the_flag() {
        let grid = dungeon(3);
        let (ex, ez) = grid.exit();
        let exit_world = grid.exit_world();
        let mut g = game(grid);

        // Stand the player one tile short of the exit, then walk onto it.
        let mut world = World::new();
        world
            .spawn_node()
            .named(PLAYER_NAME)
            .with(MeshInstance::new(MeshHandle(1), MaterialHandle(1)))
            .with(LocalTransform {
                translation: exit_world + Vec3::new(0.0, PLAYER_RADIUS, TILE_SIZE),
                ..LocalTransform::default()
            });
        frame(&mut g, &mut world, &InputSnapshot::default(), 1);
        assert!(!g.exit_reached, "not on the exit yet");
        assert_ne!(g.player_tile(), (ex, ez));

        frame(
            &mut g,
            &mut world,
            &InputSnapshot::default().with_key(W, true),
            60,
        );
        assert!(g.exit_reached, "walking onto the exit sets the flag");
    }

    /// `DUNGEON_HOLD` is what makes the headless capture drivable, so its parser is
    /// pinned: known names resolve, unknown ones are ignored rather than fatal.
    #[test]
    fn hold_key_spec_resolves_names() {
        assert_eq!(keys::key_vk("W"), Some(W));
        assert_eq!(keys::key_vk("Shift"), Some(SHIFT));
        assert_eq!(keys::key_vk("NoSuchKey"), None);
    }
}
