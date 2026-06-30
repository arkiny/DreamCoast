//! Scene / level description (Phase 12 item 2, the Phase 13 Stage-E foundation).
//!
//! A `LevelData` is the **declarative** description of a scene — entities (each
//! referencing a cooked asset by its logical key, plus a world transform and an
//! optional material override), lights, a camera, and the environment. It carries
//! no GPU handles; the runtime resolves the asset references and builds the scene.
//!
//! It serializes two ways from this one model (single source of truth): **RON text**
//! (Phase 12 Stage C — [`load_ron`] / [`save_ron`], human-authored `.level` files)
//! and the binary `.dcasset` `CHUNK_LEVEL` (see [`crate::dcasset::write_level`] /
//! [`read_level`], the Stage E cooked form). The data model is identical; only the
//! container differs.

use std::path::Path;

use dreamcoast_core::EngineError;
use serde::{Deserialize, Serialize};

/// A whole scene: entities + lights + camera + environment.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct LevelData {
    pub entities: Vec<Entity>,
    pub lights: Vec<Light>,
    pub camera: Camera,
    pub environment: Environment,
}

impl LevelData {
    /// Load a level from a RON text file (Stage C authored `.level`).
    pub fn load_ron(path: impl AsRef<Path>) -> Result<Self, EngineError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| EngineError::Asset(format!("level read: {e}")))?;
        ron::from_str(&text).map_err(|e| EngineError::Asset(format!("level parse: {e}")))
    }

    /// Serialize this level to RON text (pretty-printed for hand-editing).
    pub fn to_ron(&self) -> Result<String, EngineError> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(|e| EngineError::Asset(format!("level serialize: {e}")))
    }

    /// Save this level to a RON text file.
    pub fn save_ron(&self, path: impl AsRef<Path>) -> Result<(), EngineError> {
        std::fs::write(path, self.to_ron()?)
            .map_err(|e| EngineError::Asset(format!("level write: {e}")))
    }
}

/// One placed instance of a cooked asset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialOverride {
    pub base_color_factor: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
}

/// Light type tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightKind {
    Directional,
    Point,
}

/// A scene light. `vec` is the direction (directional) or position (point).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Light {
    pub kind: LightKind,
    pub vec: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

/// The level's default camera.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Environment {
    pub sun_dir: [f32; 3],
    pub sun_intensity: f32,
    /// Per-channel gain on the procedural sky radiance — the env capture that drives the visible
    /// sky plus the IBL irradiance/prefilter cubes (hence the IBL and SW-RT GI ambient). A value of
    /// `(1, 1, 1)` is neutral; warming it (e.g. `(1.2, 1.05, 0.8)`) takes the blue out of a high-sun
    /// sky's ambient without tinting the direct sun. The `SKY_WB` env var overrides it.
    pub sky_white_balance: [f32; 3],
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            sun_dir: [-0.4, -1.0, -0.3],
            sun_intensity: 3.0,
            sky_white_balance: [1.0, 1.0, 1.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ron_roundtrip() {
        let level = LevelData {
            entities: vec![
                Entity {
                    asset: "assets/Lantern.glb".into(),
                    transform: [
                        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0,
                        1.0,
                    ],
                    material_override: None,
                },
                Entity {
                    asset: "sphere".into(),
                    transform: [0.0; 16],
                    material_override: Some(MaterialOverride {
                        base_color_factor: [0.95, 0.64, 0.54, 1.0],
                        metallic: 1.0,
                        roughness: 0.35,
                    }),
                },
            ],
            lights: vec![Light {
                kind: LightKind::Directional,
                vec: [-0.4, -1.0, -0.3],
                color: [1.0, 0.95, 0.9],
                intensity: 3.0,
            }],
            camera: Camera::default(),
            environment: Environment::default(),
        };
        let text = level.to_ron().expect("serialize");
        let parsed: LevelData = ron::from_str(&text).expect("parse");
        assert_eq!(parsed, level);
    }
}
