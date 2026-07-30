//! The dungeon's level — authored **from code**, serialized to the `.level` RON the
//! engine's declarative loader consumes.
//!
//! Why from code rather than a hand-written file: gameplay needs the same numbers the
//! scene is built from (where the player starts, how far the floor reaches), and a
//! duplicated literal in a data file is drift waiting to happen. The constants below
//! are the single source of truth — [`level_data`] places the geometry from them and
//! [`crate::game`] simulates against them.
//!
//! **Placement:** the engine resolves a level by *stem* against the fixed directory
//! `apps/sandbox/levels` (relative to the working directory), so that is where this
//! writes. [`ensure_level_file`] only touches the file when the content differs, so a
//! re-run is a no-op for the level cook cache.

use std::path::{Path, PathBuf};

use dreamcoast_asset::LevelData;
use dreamcoast_asset::level::{Camera, Entity, Environment, Light, LightKind, MaterialOverride};
use glam::{Mat4, Vec3};

/// The level stem handed to `GameConfig::level` (the file is `<stem>.level`).
pub const LEVEL_NAME: &str = "dungeon";

/// The directory the engine's level loader scans, relative to the working directory.
const LEVELS_DIR: &str = "apps/sandbox/levels";

/// Half-extent of the floor plate, metres. The player is clamped inside it.
pub const GROUND_HALF: f32 = 20.0;

/// Player placeholder radius, metres (a sphere resting on the floor).
pub const PLAYER_RADIUS: f32 = 0.5;

/// Where the player placeholder starts. Its Y is the sphere's centre, so the ball sits
/// exactly on the floor. Also the key the game uses to find its entity (see
/// [`crate::game::DungeonGame::find_player`]).
pub const PLAYER_SPAWN: Vec3 = Vec3::new(0.0, PLAYER_RADIUS, 0.0);

/// Landmark blocks: `(centre, half_extent, colour)`. The procedural `cube` asset is a
/// 2 m cube, so a uniform scale of `s` yields a `2s` m block whose centre must sit at
/// `y = s` to rest on the floor.
const LANDMARKS: [(Vec3, f32, [f32; 4]); 3] = [
    (Vec3::new(-6.0, 1.0, -6.0), 1.0, [0.45, 0.47, 0.52, 1.0]),
    (Vec3::new(7.5, 1.5, -1.0), 1.5, [0.52, 0.44, 0.36, 1.0]),
    (Vec3::new(1.0, 2.0, -11.0), 2.0, [0.38, 0.40, 0.46, 1.0]),
];

/// Direction the sun *travels* (the level convention the loader negates into a
/// direction-toward-light). Deliberately low in the sky: a top-down camera reads
/// depth from cast shadows, and a near-vertical sun leaves none.
const SUN_DIR: [f32; 3] = [-0.55, -0.75, -0.35];
/// Sun radiance on this level's own scale (the same arbitrary scale `sponza.level`
/// authors on, not lux — the auto-exposure meters whatever it is given).
const SUN_INTENSITY: f32 = 3.5;

/// The torch: one point light near the tall landmark. The engine's level lighting
/// carries at most four point lights (see the plan's R1 risk), so one is well inside.
const TORCH_POS: [f32; 3] = [7.5, 3.4, 1.2];

/// A column-major translation + uniform-scale transform, the form `Entity::transform`
/// takes (`glam::Mat4::to_cols_array` order).
fn trs(t: Vec3, scale: f32) -> [f32; 16] {
    (Mat4::from_translation(t) * Mat4::from_scale(Vec3::splat(scale))).to_cols_array()
}

fn solid(base_color_factor: [f32; 4], roughness: f32) -> Option<MaterialOverride> {
    Some(MaterialOverride {
        base_color_factor,
        metallic: 0.0,
        roughness,
    })
}

/// Build the walking-skeleton level: a floor plate, the player placeholder, three
/// landmark blocks, one sun and one torch.
pub fn level_data() -> LevelData {
    let mut entities = vec![
        // Floor. The procedural `ground` asset is a unit (±1 m) quad on y = 0, so the
        // scale is the half-extent in metres.
        Entity {
            asset: "ground".into(),
            transform: trs(Vec3::ZERO, GROUND_HALF),
            material_override: solid([0.34, 0.33, 0.31, 1.0], 0.9),
        },
        // Player placeholder: a unit-radius sphere scaled to `PLAYER_RADIUS`.
        Entity {
            asset: "sphere".into(),
            transform: trs(PLAYER_SPAWN, PLAYER_RADIUS),
            material_override: solid([0.80, 0.28, 0.18, 1.0], 0.45),
        },
    ];
    entities.extend(LANDMARKS.iter().map(|&(centre, half, colour)| Entity {
        asset: "cube".into(),
        transform: trs(centre, half),
        material_override: solid(colour, 0.75),
    }));

    LevelData {
        entities,
        lights: vec![
            Light {
                kind: LightKind::Directional,
                vec: SUN_DIR,
                color: [1.0, 0.96, 0.90],
                intensity: SUN_INTENSITY,
            },
            Light {
                kind: LightKind::Point,
                vec: TORCH_POS,
                color: [1.0, 0.62, 0.28],
                intensity: 12.0,
            },
        ],
        // The authored view matches the follow camera's rest pose, so the level still
        // frames sensibly if the game ever hands the camera back to the engine.
        camera: Camera {
            position: (PLAYER_SPAWN + crate::game::camera_offset()).to_array(),
            target: PLAYER_SPAWN.to_array(),
            fov_y_deg: 60.0,
            znear: 0.05,
            zfar: 100.0,
        },
        environment: Environment {
            sun_dir: SUN_DIR,
            sun_intensity: SUN_INTENSITY,
            sky_white_balance: [1.0, 1.0, 1.0],
        },
        deforms: Vec::new(),
    }
}

/// Path of the `.level` file this game loads.
pub fn level_path() -> PathBuf {
    Path::new(LEVELS_DIR).join(format!("{LEVEL_NAME}.level"))
}

/// Write the level next to the engine's built-in ones, unless an identical file is
/// already there (an unchanged file keeps the level cook cache warm).
pub fn ensure_level_file() -> anyhow::Result<PathBuf> {
    let path = level_path();
    let ron = level_data().to_ron()?;
    let current = std::fs::read_to_string(&path).ok();
    if current.as_deref() != Some(ron.as_str()) {
        std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
        std::fs::write(&path, &ron)?;
        tracing::info!("dungeon: wrote level '{}'", path.display());
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The authored level must round-trip through the RON the engine parses, and the
    /// player placeholder must be findable at the spawn the game simulates from.
    #[test]
    fn level_round_trips_and_places_the_player() {
        let level = level_data();
        let parsed: LevelData = ron::from_str(&level.to_ron().unwrap()).unwrap();
        assert_eq!(parsed, level);

        let player = level
            .entities
            .iter()
            .find(|e| e.asset == "sphere")
            .expect("player placeholder");
        let translation = Mat4::from_cols_array(&player.transform).w_axis.truncate();
        assert_eq!(translation, PLAYER_SPAWN);
    }
}
