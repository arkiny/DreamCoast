//! Game-injection seam (game-framework M0, `docs/game-framework-plan.md` §2).
//!
//! The sandbox *is* the engine — the render path, the fixed-timestep loop and the
//! frame assembly all live in this crate — so a game does not fork it. It links the
//! sandbox as a **library**, implements [`GameHooks`], and gets three call-ins on the
//! existing frame:
//!
//! | hook | when | what it may do |
//! |------|------|----------------|
//! | [`GameHooks::fixed_update`] | inside the fixed-step sim loop, after `advance_spin`/`advance_animation` | mutate the ECS `World` (game simulation) |
//! | [`GameHooks::render_update`] | once per rendered frame, after the sim loop, **before** transform propagation | write *visual-only* interpolated transforms |
//! | [`GameHooks::camera`] | at camera resolve, before the view/projection matrices are built | replace the Orbit/Fly pose for this frame |
//! | [`GameHooks::draw_ui`] | inside the ImGui frame, after the debug window closes | draw game UI/HUD |
//!
//! **Every method has a no-op default**, and with no hooks installed the frame loop
//! takes exactly the paths it took before this module existed — that byte-identity
//! (golden/gallery anchor unchanged) is the landing contract for the seam.
//!
//! This module is RHI-free: it speaks only `glam`, the ECS `World`, platform input
//! and `imgui`.

use dreamcoast_core::glam::{Quat, Vec3};
use dreamcoast_gui::imgui;
use dreamcoast_platform::InputSnapshot;
use dreamcoast_scene::World;

/// A camera pose a game hands back to the frame loop, replacing the built-in
/// Orbit/Fly camera for that frame.
///
/// **Conventions** (they are the frame loop's, not a new set):
/// - Right-handed, world **+Y is up**; `rotation` maps the canonical camera basis
///   (forward `-Z`, up `+Y`) into world space. The view matrix is built as a look-at
///   from `position` toward `position + forward()` with world `+Y` up, so **roll
///   around the view axis is not applied** — the frame loop has a single up-vector
///   convention and this type does not widen it.
/// - Because up is world `+Y`, a *perfectly* vertical view direction is degenerate
///   (the look-at basis collapses). A top-down game camera should keep a few degrees
///   of tilt, exactly as the built-in overhead inset view does.
/// - `fov_y_radians` is optional; `None` keeps the frame's default vertical FOV. When
///   set it drives the scene projection, the view descriptor, the shadow-cascade fit
///   and the screen-space AO projection scale from this one value (single source).
///
/// **Interpolation.** The hook receives the frame's `alpha` (see
/// [`GameHooks::camera`]) and is expected to return the *rendered* pose — i.e. the
/// game's own previous/current sim states already blended by `alpha`. The frame loop
/// does not interpolate the returned pose.
///
/// **Temporal state.** The previous-frame view matrices that feed motion vectors and
/// the TAAU/denoiser history are latched at the end of the frame from whatever
/// view-projection the frame actually used, so a hook pose is captured like any other
/// camera: reprojection stays consistent, and installing/removing the override
/// between frames does not desynchronize it. A *discontinuous* jump (a teleport,
/// or a camera cut) is still a real camera cut for the temporal passes — history
/// reprojection will miss for a frame or two, the same as any hard cut.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPose {
    /// Eye position in world space.
    pub position: Vec3,
    /// Orientation: rotates camera-local forward `-Z` / up `+Y` into world space.
    pub rotation: Quat,
    /// Vertical field of view in radians. `None` = keep the frame's default.
    pub fov_y_radians: Option<f32>,
}

impl Default for CameraPose {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            fov_y_radians: None,
        }
    }
}

impl CameraPose {
    /// Pose looking from `eye` toward `target` (world `+Y` up), default FOV.
    ///
    /// `target` must not be directly above/below `eye` (see the up-vector note on the
    /// type); keep a small tilt for an overhead camera.
    pub fn look_at(eye: Vec3, target: Vec3) -> Self {
        let forward = (target - eye).normalize_or_zero();
        let rotation = if forward == Vec3::ZERO {
            Quat::IDENTITY
        } else {
            // `look_to_rh` builds a world->view rotation; the camera's object-space
            // orientation is its inverse.
            Quat::from_mat4(&dreamcoast_core::glam::Mat4::look_to_rh(
                Vec3::ZERO,
                forward,
                Vec3::Y,
            ))
            .inverse()
        };
        Self {
            position: eye,
            rotation,
            fov_y_radians: None,
        }
    }

    /// Set the vertical FOV (radians) on this pose.
    pub fn with_fov_y(mut self, fov_y_radians: f32) -> Self {
        self.fov_y_radians = Some(fov_y_radians);
        self
    }

    /// Unit view direction in world space (`rotation * -Z`).
    pub fn forward(&self) -> Vec3 {
        (self.rotation * Vec3::NEG_Z).normalize_or_zero()
    }

    /// The look-at target the frame loop uses: a unit step along [`Self::forward`].
    pub fn focus(&self) -> Vec3 {
        self.position + self.forward()
    }
}

/// The game-side callbacks the sandbox frame loop invokes. All methods default to
/// no-ops, so a game implements only what it needs.
///
/// Installed via [`crate::GameConfig::hooks`] (or [`crate::App::set_hooks`]) and owned
/// by the app for the process lifetime; `&mut self` means a hook is free to keep game
/// state (dungeon grid, player handle, timers) in the implementor.
pub trait GameHooks {
    /// One simulation step, called from inside the fixed-timestep loop with the
    /// engine's `FIXED_DT` (1/60 s), after the built-in `advance_spin` /
    /// `advance_animation` and **before** transform propagation — so any
    /// `LocalTransform` written here is picked up by this frame's draw list.
    ///
    /// May run zero, one, or several times per rendered frame (the loop consumes
    /// whole steps from an accumulator, capped against a stall backlog).
    ///
    /// `input` is **one frame** of platform state, captured once before the loop, so
    /// every step of a frame sees the same snapshot and edge detection across steps is
    /// the game's business. It is an [`InputSnapshot`] rather than the live
    /// `platform::Input` on purpose: `Input`'s setters are crate-private, so a game
    /// could not build one — a snapshot it can, which is what lets a whole fixed step
    /// be exercised in a unit test with no window, device, or swapchain.
    ///
    /// Headless capture (`--screenshot[-clean]`) bypasses the accumulator to stay
    /// frame-counted and byte-identical; there this runs once per captured frame on
    /// the `CAPTURE_SEQ` path (which is the capture path's deterministic sim step)
    /// and not at all for a single static capture.
    fn fixed_update(&mut self, world: &mut World, input: &InputSnapshot, dt: f32) {
        let _ = (world, input, dt);
    }

    /// One *rendered* frame's presentation pass: called once per frame after the
    /// fixed-step loop has consumed whatever whole steps were due, and immediately
    /// **before** transform propagation and draw-list assembly — so what it writes is
    /// what this frame draws.
    ///
    /// Its reason to exist is the gap between simulation rate and frame rate.
    /// [`Self::fixed_update`] leaves the ECS holding the *last simulated* pose, while
    /// [`Self::camera`] returns a pose interpolated by `alpha`; at 1/60 s steps and
    /// sprint speed those differ by a step of travel (~14 cm), which reads as the world
    /// sliding under a lagging character. This hook is where a game closes that: write
    /// `prev.lerp(current, alpha)` into the visual `LocalTransform`s.
    ///
    /// **Contract: visual-only.** `alpha` is a render-time quantity, so anything
    /// derived from it is presentation, not state. Simulation state must live in the
    /// implementor and advance solely in `fixed_update`; a value that this hook writes
    /// and the next `fixed_update` reads back would make the simulation depend on the
    /// frame rate — the exact coupling the fixed step exists to prevent. (Writing an
    /// interpolated `LocalTransform` is safe precisely because the next step recomputes
    /// it from the game's own `prev`/`current`, never from the ECS.)
    ///
    /// `alpha` is the same `[0, 1)` factor `camera` receives — the fraction of a fixed
    /// step not yet simulated — so the mesh and the camera interpolate in lockstep. In
    /// headless capture mode it is exactly `1.0` (the capture path is frame-counted, not
    /// interpolated), which makes this hook a no-op there by construction.
    fn render_update(&mut self, world: &mut World, alpha: f32) {
        let _ = (world, alpha);
    }

    /// Resolve this frame's camera. `Some` replaces the built-in Orbit/Fly pose for
    /// the frame; `None` falls through to the existing behavior (and leaves the
    /// interactive fly camera driving, so a game can hand control back at any time).
    ///
    /// `alpha` is the render interpolation factor in `[0, 1)` — the fraction of a
    /// fixed step not yet simulated — so a game blends its own previous/current
    /// camera state by it (see [`CameraPose`]).
    ///
    /// The headless diagnostic overrides (`CAM_EYE` / `CAM_TARGET` / `CAM_EYE_END`)
    /// still win over a hook pose, so a game scene can be posed for a capture.
    fn camera(&mut self, alpha: f32) -> Option<CameraPose> {
        let _ = alpha;
        None
    }

    /// Draw game UI. Called inside the engine's ImGui frame, after the debug window
    /// has been closed, so a game window/HUD is a sibling of it rather than nested.
    /// Skipped entirely for `--screenshot-clean` (no ImGui frame is started at all).
    fn draw_ui(&mut self, ui: &imgui::Ui, world: &World) {
        let _ = (ui, world);
    }
}
