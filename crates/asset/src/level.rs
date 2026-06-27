//! Scene / level description (Phase 12 item 2, the Phase 13 Stage-E foundation).
//!
//! A `LevelData` is the **declarative** description of a scene — entities (each
//! referencing a cooked asset by its logical key, plus a world transform and an
//! optional material override), lights, a camera, and the environment. It carries
//! no GPU handles; the runtime resolves the asset references and builds the scene.
//!
//! It serializes into the same `.dcasset` chunk container as everything else (a
//! `CHUNK_LEVEL`, see [`crate::dcasset::write_level`] / [`read_level`]) so a level
//! is just another cooked asset. Wiring it to drive the live render is Phase 13.

/// A whole scene: entities + lights + camera + environment.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct LevelData {
    pub entities: Vec<Entity>,
    pub lights: Vec<Light>,
    pub camera: Camera,
    pub environment: Environment,
}

/// One placed instance of a cooked asset.
#[derive(Clone, Debug, PartialEq)]
pub struct Entity {
    /// Logical asset key — the same stable reference the mesh cook is keyed on
    /// (e.g. `assets/model.glb`), resolved to a `.dcasset` at load.
    pub asset: String,
    /// World transform, column-major (`glam::Mat4::to_cols_array` order).
    pub transform: [f32; 16],
    /// Optional per-instance material override (else the asset's own material).
    pub material_override: Option<MaterialOverride>,
}

/// Per-instance material scalar overrides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialOverride {
    pub base_color_factor: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
}

/// Light type tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightKind {
    Directional,
    Point,
}

/// A scene light. `vec` is the direction (directional) or position (point).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    pub kind: LightKind,
    pub vec: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

/// The level's default camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub fov_y_deg: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: [0.0, 1.0, 3.0],
            target: [0.0; 3],
            fov_y_deg: 45.0,
            znear: 0.05,
            zfar: 100.0,
        }
    }
}

/// Environment / sky + sun.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Environment {
    pub sun_dir: [f32; 3],
    pub sun_intensity: f32,
    pub sky_tint: [f32; 3],
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            sun_dir: [-0.4, -1.0, -0.3],
            sun_intensity: 3.0,
            sky_tint: [0.6, 0.7, 0.9],
        }
    }
}
