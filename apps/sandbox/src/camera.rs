//! Camera controllers (Phase 12 Stage 0).
//!
//! Two modes share the frame loop's `(eye, focus)` contract:
//! - **Orbit** — the legacy whole-scene framing driven by `App::angle`. It stays
//!   the default and is the *only* mode used for headless `--screenshot-clean`
//!   captures, so the parity baseline remains byte-identical.
//! - **Fly** — an interactive WASD + mouse-look free camera for inspecting scenes
//!   and (Stage D) driving across streaming chunk boundaries. Never active during a
//!   headless capture.
//!
//! This module is RHI-free: it speaks only `glam` + `platform::Input`.

use dreamcoast_core::glam::Vec3;
use dreamcoast_platform::Input;

// Win32 virtual-key codes for the fly controls.
const VK_W: u16 = 0x57;
const VK_A: u16 = 0x41;
const VK_S: u16 = 0x53;
const VK_D: u16 = 0x44;
const VK_Q: u16 = 0x51;
const VK_E: u16 = 0x45;
const VK_SHIFT: u16 = 0x10;

/// Which camera drives the frame.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum CameraMode {
    /// Legacy whole-scene orbit (the screenshot/parity baseline).
    #[default]
    Orbit,
    /// Interactive WASD + mouse-look free camera.
    Fly,
}

/// A yaw/pitch free-fly camera. Angles are radians; `yaw = 0` looks toward +X,
/// `pitch` is elevation (clamped to ±89° to avoid gimbal flip at the poles).
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlyCamera {
    pub(crate) position: Vec3,
    yaw: f32,
    pitch: f32,
    /// Base translation speed (world units/sec), adjustable via the mouse wheel.
    speed: f32,
}

const PITCH_LIMIT: f32 = 1.553_343; // 89° in radians
const MOUSE_SENSITIVITY: f32 = 0.0025; // radians per pixel of mouse delta

impl FlyCamera {
    /// Seed the fly camera so it looks from `eye` toward `focus` with no visible
    /// jump when toggling out of orbit mode. `speed` scales with scene size.
    pub(crate) fn from_look(eye: Vec3, focus: Vec3, speed: f32) -> Self {
        let dir = (focus - eye).normalize_or_zero();
        let pitch = dir.y.clamp(-1.0, 1.0).asin();
        let yaw = dir.z.atan2(dir.x);
        Self {
            position: eye,
            yaw,
            pitch,
            speed: speed.max(0.01),
        }
    }

    /// Unit forward vector from the current yaw/pitch.
    pub(crate) fn forward(&self) -> Vec3 {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp)
    }

    /// The look-at target a unit ahead of the eye (feeds the reflection probe/focus).
    pub(crate) fn focus(&self) -> Vec3 {
        self.position + self.forward()
    }

    /// Advance the camera from this frame's input. Plain mouse move = mouse-look
    /// (gated by `look_enabled`, false while the UI owns the cursor); WASD =
    /// planar move, Q/E = down/up, Shift = sprint, wheel = adjust base speed.
    pub(crate) fn update(&mut self, input: &Input, dt: f32, look_enabled: bool) {
        // Free look on plain mouse movement (no button chord). The caller passes
        // `look_enabled = false` while ImGui wants the cursor, so hovering /
        // dragging a UI window doesn't spin the camera underneath it.
        if look_enabled {
            let (dx, dy) = input.mouse_delta();
            self.yaw += dx as f32 * MOUSE_SENSITIVITY;
            self.pitch =
                (self.pitch - dy as f32 * MOUSE_SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        }

        // Wheel adjusts base speed multiplicatively (1 notch ≈ ±15%).
        let wheel = input.wheel_delta();
        if wheel != 0.0 {
            self.speed = (self.speed * 1.15f32.powf(wheel)).clamp(0.01, 1000.0);
        }

        let forward = self.forward();
        // Camera-right on the ground plane; up is world +Y.
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let mut motion = Vec3::ZERO;
        if input.key_down(VK_W) {
            motion += forward;
        }
        if input.key_down(VK_S) {
            motion -= forward;
        }
        if input.key_down(VK_D) {
            motion += right;
        }
        if input.key_down(VK_A) {
            motion -= right;
        }
        if input.key_down(VK_E) {
            motion += Vec3::Y;
        }
        if input.key_down(VK_Q) {
            motion -= Vec3::Y;
        }
        let sprint = if input.key_down(VK_SHIFT) { 4.0 } else { 1.0 };
        self.position += motion.normalize_or_zero() * self.speed * sprint * dt;
    }
}
