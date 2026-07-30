//! The game: one [`GameHooks`] implementation driving a player placeholder, a
//! top-down follow camera and a HUD.
//!
//! This is the walking skeleton for the dungeon crawler (`docs/game-framework-plan.md`
//! §2.4): it proves the whole seam — simulation writes into the ECS from
//! [`GameHooks::fixed_update`], the view comes from [`GameHooks::camera`], and the
//! overlay comes from [`GameHooks::draw_ui`] — with no renderer changes at all.
//!
//! **Fixed-step discipline.** Everything that integrates (velocity, camera smoothing)
//! advances in `fixed_update` at the engine's fixed `dt`, and `camera` only *blends*
//! the two latest states by the frame's `alpha`. That keeps motion framerate-
//! independent and the follow camera free of the jitter a per-frame smoothing filter
//! would add.

use dreamcoast_game::input::{ActionState, BindingsConfig, InputSnapshot};
use glam::{Vec2, Vec3};
use sandbox::imgui;
use sandbox::platform::Input;
use sandbox::scene::{Entity, LocalTransform, MeshInstance, World};
use sandbox::{CameraPose, GameHooks};

use crate::level::{GROUND_HALF, PLAYER_RADIUS, PLAYER_SPAWN};

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
const CAMERA_DISTANCE: f32 = 12.0;

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

pub struct DungeonGame {
    input: ActionState<Action>,
    /// The player placeholder's entity, resolved lazily from the loaded level.
    player: Option<Entity>,
    /// Simulation state, in fixed steps: previous and current, so the render frame can
    /// interpolate between them.
    prev_position: Vec3,
    position: Vec3,
    velocity: Vec3,
    /// The point the camera follows (a smoothed trail behind `position`), also kept as
    /// a previous/current pair for interpolation.
    prev_focus: Vec3,
    focus: Vec3,
    /// Fixed steps simulated so far — a cheap "is the sim actually running" readout.
    steps: u64,
    /// The interpolation factor of the frame being drawn, latched by the camera hook
    /// (which is the only hook the frame loop hands it to) so the HUD can report the
    /// rendered position rather than the last simulated one.
    render_alpha: f32,
}

impl DungeonGame {
    pub fn new() -> anyhow::Result<Self> {
        let bindings = BindingsConfig::from_ron(DEFAULT_BINDINGS)
            .map_err(|e| anyhow::anyhow!("dungeon bindings: {e}"))?;
        let map = bindings
            .resolve(Action::from_name)
            .map_err(|e| anyhow::anyhow!("dungeon bindings: {e}"))?;
        Ok(Self {
            input: ActionState::new(map),
            player: None,
            prev_position: PLAYER_SPAWN,
            position: PLAYER_SPAWN,
            velocity: Vec3::ZERO,
            prev_focus: PLAYER_SPAWN,
            focus: PLAYER_SPAWN,
            steps: 0,
            render_alpha: 0.0,
        })
    }

    /// Identify the player placeholder in a freshly loaded level.
    ///
    /// Entities instantiated from a `.level` procedural asset carry no `Name` (only
    /// imported glTF roots get one), so there is nothing to look up by. The stable
    /// identity available today is the authored spawn transform: the placeholder is
    /// the renderable whose local translation is [`PLAYER_SPAWN`]. `level.rs` writes
    /// that transform from the same constant, so the two cannot drift.
    fn find_player(world: &World) -> Option<Entity> {
        world
            .query2::<MeshInstance, LocalTransform>()
            .into_iter()
            .find(|(_, _, local)| local.translation.abs_diff_eq(PLAYER_SPAWN, 1e-4))
            .map(|(entity, _, _)| entity)
    }

    /// Re-resolve the player after a level (re)load, seeding the sim state from the
    /// entity's authored transform so the camera does not lurch.
    fn acquire_player(&mut self, world: &World) -> Option<Entity> {
        if let Some(e) = self.player
            && world.is_alive(e)
        {
            return Some(e);
        }
        let entity = Self::find_player(world)?;
        let start = world
            .get::<LocalTransform>(entity)
            .map(|t| t.translation)
            .unwrap_or(PLAYER_SPAWN);
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

    /// Where the player *is* this rendered frame: the two latest fixed-step positions
    /// blended by the frame's interpolation factor.
    ///
    /// Note the engine draws the placeholder mesh from its ECS transform, which only
    /// the fixed step writes — so the mesh sits at the last simulated position while
    /// this reports the sub-step one. They differ by at most one step of travel
    /// (~14 cm at sprint), and closing it would need a per-frame hook that can write
    /// the ECS — which the M0 seam deliberately does not expose (an M1 question).
    fn render_position(&self, alpha: f32) -> Vec3 {
        self.prev_position.lerp(self.position, alpha)
    }

    /// One simulation step against an already-captured input snapshot.
    ///
    /// The hook itself receives the *platform* `Input`, whose key state has no public
    /// setter, so a test cannot fabricate one. Taking a snapshot here — the framework's
    /// own testable input type — keeps the whole step exercisable without a window
    /// (see the tests below) and costs the real path nothing.
    fn simulate(&mut self, world: &mut World, snapshot: &InputSnapshot, dt: f32) {
        // One snapshot per step. Every step of a frame sees the same platform state
        // (the window is pumped once per frame), so an edge is reported on the first
        // step of the frame it happened in and not repeated by the later ones.
        self.input.update(snapshot);

        let Some(player) = self.acquire_player(world) else {
            return; // level not loaded (or has no placeholder) — nothing to simulate
        };
        self.steps += 1;

        // Velocity ramp, then integrate position on the ground plane.
        let wish = self.wish_velocity();
        self.velocity += (wish - self.velocity) * blend(dt, ACCEL_TIME);
        self.prev_position = self.position;
        self.position += self.velocity * dt;

        // Stay on the floor plate: stop dead at the edge instead of sliding off it.
        let limit = GROUND_HALF - PLAYER_RADIUS;
        for axis in [0usize, 2] {
            if self.position[axis].abs() > limit {
                self.position[axis] = self.position[axis].clamp(-limit, limit);
                self.velocity[axis] = 0.0;
            }
        }
        self.position.y = PLAYER_RADIUS;

        // Write the sim result into the ECS. This runs before transform propagation,
        // so it lands in this frame's draw list.
        if let Some(local) = world.get_mut::<LocalTransform>(player) {
            local.translation = self.position;
        }

        // The camera follows a smoothed trail behind the player, advanced on the same
        // fixed step so the filter is framerate-independent.
        self.prev_focus = self.focus;
        self.focus += (self.position - self.focus) * blend(dt, CAMERA_SMOOTH_TIME);
    }
}

impl GameHooks for DungeonGame {
    fn fixed_update(&mut self, world: &mut World, input: &Input, dt: f32) {
        self.simulate(world, &InputSnapshot::capture(input), dt);
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
            .size([260.0, 150.0], imgui::Condition::FirstUseEver)
            .build(|| {
                match self.player {
                    Some(e) if world.is_alive(e) => {
                        ui.text(format!("player entity  #{}", e.index()));
                    }
                    // Headless single-frame captures never run a sim step (the capture
                    // path is frame-counted), so this is the expected reading there.
                    _ => ui.text_disabled("player not resolved yet"),
                }
                let p = self.render_position(self.render_alpha);
                ui.text(format!("position  {:.2}, {:.2}, {:.2}", p.x, p.y, p.z));
                ui.text(format!("speed     {:.2} m/s", self.speed()));
                ui.text(format!("frame     {dt_ms:.2} ms"));
                ui.text(format!("sim steps {}", self.steps));
                ui.separator();
                ui.text_disabled("WASD move, Shift sprint");
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sandbox::scene::{MaterialHandle, MeshHandle};

    const W: u16 = 0x57;
    const SHIFT: u16 = 0x10;
    const FIXED_DT: f32 = 1.0 / 60.0;

    /// A stand-in for the loaded level: a floor entity plus the player placeholder at
    /// the authored spawn, both renderable — i.e. exactly what the level loader
    /// produces (and, notably, with no `Name` on either).
    fn test_world() -> World {
        let mut world = World::new();
        let renderable = |i: u32| MeshInstance::new(MeshHandle(i), MaterialHandle(i));
        world
            .spawn_node()
            .with(renderable(0))
            .with(LocalTransform::default());
        world.spawn_node().with(renderable(1)).with(LocalTransform {
            translation: PLAYER_SPAWN,
            ..LocalTransform::default()
        });
        world
    }

    /// The bindings file must resolve — a typo in it is a startup failure, so catch it
    /// in a test rather than on the player's machine.
    #[test]
    fn bindings_resolve() {
        assert!(DungeonGame::new().is_ok());
    }

    /// The whole fixed step, end to end: the placeholder is identified in a level-like
    /// world, held input accelerates it, and the result lands on its ECS transform.
    #[test]
    fn a_held_key_moves_the_player_entity_in_the_ecs() {
        let mut game = DungeonGame::new().unwrap();
        let mut world = test_world();
        let held = InputSnapshot::default().with_key(W, true);
        for _ in 0..30 {
            game.simulate(&mut world, &held, FIXED_DT);
        }
        let player = game.player.expect("placeholder identified from the level");
        let moved = world.get::<LocalTransform>(player).unwrap().translation;
        assert!(moved.z < PLAYER_SPAWN.z - 1.0, "walked away from camera");
        assert_eq!(moved.x, PLAYER_SPAWN.x);
        assert_eq!(moved.y, PLAYER_RADIUS, "stays on the floor");
        // The camera trails the player instead of snapping onto it.
        assert!(game.focus.z > moved.z && game.focus.z < PLAYER_SPAWN.z);
    }

    /// Releasing input brings the placeholder to rest (the ramp works both ways) and
    /// the floor plate is a hard boundary.
    #[test]
    fn player_stops_and_stays_on_the_plate() {
        let mut game = DungeonGame::new().unwrap();
        let mut world = test_world();
        let held = InputSnapshot::default()
            .with_key(W, true)
            .with_key(SHIFT, true);
        for _ in 0..600 {
            game.simulate(&mut world, &held, FIXED_DT);
        }
        assert_eq!(game.position.z, -(GROUND_HALF - PLAYER_RADIUS));
        for _ in 0..60 {
            game.simulate(&mut world, &InputSnapshot::default(), FIXED_DT);
        }
        assert!(game.speed() < 0.01, "velocity ramps back down to rest");
    }

    /// Holding "forward" must move the placeholder toward -Z and ramp up, not teleport.
    #[test]
    fn forward_input_moves_along_negative_z() {
        let mut game = DungeonGame::new().unwrap();
        game.input
            .update(&InputSnapshot::default().with_key(W, true));
        let wish = game.wish_velocity();
        assert!(wish.z < 0.0 && wish.x == 0.0);
        assert!((wish.length() - WALK_SPEED).abs() < 1e-4);
    }

    /// Sprint is a speed multiplier on the same direction.
    #[test]
    fn sprint_is_faster_than_walk() {
        let mut game = DungeonGame::new().unwrap();
        game.input.update(
            &InputSnapshot::default()
                .with_key(W, true)
                .with_key(SHIFT, true),
        );
        assert!((game.wish_velocity().length() - SPRINT_SPEED).abs() < 1e-4);
    }

    /// The camera sits above and behind its focus, tilted off vertical (a perfectly
    /// vertical view axis is degenerate for the frame loop's look-at).
    #[test]
    fn camera_pose_is_tilted_and_above() {
        let mut game = DungeonGame::new().unwrap();
        let pose = game.camera(1.0).expect("hook always drives the camera");
        assert!(pose.position.y > PLAYER_SPAWN.y);
        assert!(pose.position.z > PLAYER_SPAWN.z);
        let forward = pose.forward();
        assert!(forward.y < -0.5, "camera must look down");
        assert!(forward.y > -0.999, "but not straight down");
    }
}
