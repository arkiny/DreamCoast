//! The dungeon's levels — authored **from code**, serialized to the `.level` RON the
//! engine's declarative loader consumes.
//!
//! Why from code rather than a hand-written file: gameplay needs the same numbers the
//! scene is built from (where the player starts, how far the floor reaches), and a
//! duplicated literal in a data file is drift waiting to happen. The constants below
//! are the single source of truth — [`level_data`] places the geometry from them and
//! [`crate::game`] simulates against them.
//!
//! Two levels live here:
//!
//! * **`dungeon`** — the M0 walking skeleton: a floor plate, the player placeholder and
//!   a few landmark blocks, all built from the engine's procedural `ground`/`sphere`/
//!   `cube` assets.
//! * **`dungeon_room`** — the M1 **static-geometry injection proof**: a room whose walls
//!   and pillars are *generated at runtime* and enter the scene as a real asset file.
//!   See [Static-geometry injection](self#static-geometry-injection).
//!
//! **Placement:** both are written to [`GENERATED_DIR`] (derived data — the whole
//! directory is regenerated from this file), and the game points
//! `GameConfig::levels_dir` at it. Writing is content-conditional, so a re-run that
//! changes nothing leaves every file — and therefore every cook-cache entry — alone.
//!
//! # Static-geometry injection
//!
//! M1 needs runtime-generated dungeon geometry to be **fully static** scene geometry:
//! visible to the per-mesh SDF/GDF, the surface cache, GI, reflections and the TLAS,
//! exactly like a level-loaded asset. The engine already has one road that leads there
//! — cook → content-hash-keyed `.dcasset` → level instantiation → static bakes — and
//! nothing along it is glTF-specific except its entrance. So instead of opening a
//! second injection path into the middle of `App::new` (a setup hook uploading meshes
//! straight into the registries), the generator **writes a `.glb`** plus a `.level` that
//! references it, and rides the existing road:
//!
//! ```text
//! generator → dreamcoast_asset::save_glb → cache/generated/<name>.glb
//!                                        + cache/generated/<name>.level
//!                                          → the engine's normal level load
//!                                            → cook (.dcasset) → per-mesh SDF bake
//!                                              → GDF / surface cache / GI / reflections
//! ```
//!
//! What that buys, none of which a setup hook would get for free:
//!
//! * **Cook caching keyed by generator output.** The cook key is a hash of the source
//!   bytes, and the writer is deterministic, so the bytes are a pure function of the
//!   generator (seed, version, parameters). An unchanged generator is a cache hit; a
//!   changed one re-cooks. There is no separate "generator version + seed" key to
//!   invent, and so none to get wrong.
//! * **Every static bake, for free and forever.** The geometry arrives through the same
//!   door as an authored asset, so a bake added to the level path later applies to it
//!   without anyone remembering to wire up a second path.
//! * **Inspectability.** The generated `.glb` is a real file, openable in any glTF
//!   viewer when a bake looks wrong.

use std::path::{Path, PathBuf};

use dreamcoast_asset::level::{Camera, Entity, Environment, Light, LightKind, MaterialOverride};
use dreamcoast_asset::{GlbMaterial, GlbMesh, LevelData, MeshVertex};
use glam::{Mat4, Vec3};

/// Where this game's generated levels + assets are written, relative to the working
/// directory. Derived data (gitignored, alongside the cooked-asset cache) — nothing in
/// it is hand-edited, and deleting it costs only a regenerate + re-cook.
pub const GENERATED_DIR: &str = "cache/generated";

/// The walking-skeleton level's stem (the file is `<stem>.level`).
pub const LEVEL_NAME: &str = "dungeon";

/// The generated-geometry level's stem — the M1 static-injection proof.
pub const ROOM_LEVEL_NAME: &str = "dungeon_room";

/// Scene-graph name the player placeholder is authored with. Gameplay finds its entity
/// by this (`crate::game::DungeonGame::find_player`) rather than by matching a spawn
/// transform, so moving the spawn cannot silently break the lookup.
pub const PLAYER_NAME: &str = "player";

/// Half-extent of the floor plate, metres. The player is clamped inside it.
pub const GROUND_HALF: f32 = 20.0;

/// Player placeholder radius, metres (a sphere resting on the floor).
pub const PLAYER_RADIUS: f32 = 0.5;

/// Where the player placeholder starts. Its Y is the sphere's centre, so the ball sits
/// exactly on the floor.
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

/// The identity transform — generated geometry is authored in world space already.
fn identity() -> [f32; 16] {
    Mat4::IDENTITY.to_cols_array()
}

fn solid(base_color_factor: [f32; 4], roughness: f32) -> Option<MaterialOverride> {
    Some(MaterialOverride {
        base_color_factor,
        metallic: 0.0,
        roughness,
    })
}

/// The player placeholder: a unit-radius sphere scaled to [`PLAYER_RADIUS`], named so
/// gameplay can find it. Shared by both levels — the player is the same in each.
fn player_entity() -> Entity {
    Entity {
        asset: "sphere".into(),
        name: Some(PLAYER_NAME.into()),
        transform: trs(PLAYER_SPAWN, PLAYER_RADIUS),
        material_override: solid([0.80, 0.28, 0.18, 1.0], 0.45),
    }
}

/// This game's sun + torch.
fn lights() -> Vec<Light> {
    vec![
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
    ]
}

/// The sky the sun above agrees with (one source for direct light and the IBL/GI ambient).
fn environment() -> Environment {
    Environment {
        sun_dir: SUN_DIR,
        sun_intensity: SUN_INTENSITY,
        sky_white_balance: [1.0, 1.0, 1.0],
    }
}

/// The follow camera's rest pose, so a level still frames sensibly if the game ever
/// hands the view back to the engine.
fn rest_camera() -> Camera {
    Camera {
        position: (PLAYER_SPAWN + crate::game::camera_offset()).to_array(),
        target: PLAYER_SPAWN.to_array(),
        fov_y_deg: 60.0,
        znear: 0.05,
        zfar: 100.0,
    }
}

/// Build the walking-skeleton level: a floor plate, the player placeholder, three
/// landmark blocks, one sun and one torch.
pub fn level_data() -> LevelData {
    let mut entities = vec![
        // Floor. The procedural `ground` asset is a unit (±1 m) quad on y = 0, so the
        // scale is the half-extent in metres.
        Entity {
            asset: "ground".into(),
            name: None,
            transform: trs(Vec3::ZERO, GROUND_HALF),
            material_override: solid([0.34, 0.33, 0.31, 1.0], 0.9),
        },
        player_entity(),
    ];
    entities.extend(LANDMARKS.iter().map(|&(centre, half, colour)| Entity {
        asset: "cube".into(),
        name: None,
        transform: trs(centre, half),
        material_override: solid(colour, 0.75),
    }));

    LevelData {
        entities,
        lights: lights(),
        camera: rest_camera(),
        environment: environment(),
        deforms: Vec::new(),
    }
}

// --- Generated geometry -------------------------------------------------------------

/// Interior half-extent of the generated room, metres (wall inner faces sit at ±this).
const ROOM_HALF: f32 = 8.0;
/// Wall thickness and height, metres.
const WALL_THICKNESS: f32 = 0.5;
const WALL_HEIGHT: f32 = 4.0;
/// Half-width of the doorway cut into the -Z wall, metres.
const DOOR_HALF_WIDTH: f32 = 1.5;
/// Square pillars: half-extent and their offset from the room centre, metres.
const PILLAR_HALF: f32 = 0.35;
const PILLAR_OFFSET: f32 = 4.5;
/// Floor slab thickness, metres — the walkable surface is its top face at y = 0.
const FLOOR_THICKNESS: f32 = 0.5;

/// Quads per box-face edge. Boxes are the primitive here, but a chunk merged from a real
/// tile grid has interior vertices, so the faces are subdivided: it keeps the vertex and
/// triangle counts (and therefore the SDF-bake timing) representative rather than a
/// 12-triangle toy.
const FACE_SUBDIV: u32 = 2;

/// The generated room's geometry, as one [`GlbMesh`] per material group — the same
/// granularity a chunk-merged tile grid produces (the engine bakes one SDF per unique
/// mesh and issues one draw per instance, so chunk-sized meshes are what keeps both
/// counts sane).
pub fn room_meshes() -> (Vec<GlbMesh>, Vec<GlbMaterial>) {
    let outer = ROOM_HALF + WALL_THICKNESS;

    // Floor slab: top face flush with y = 0, so the player's ground plane needs no
    // offset and the walls stand on it.
    let mut floor = Builder::new("room_floor", 0);
    floor.add_box(
        Vec3::new(-outer, -FLOOR_THICKNESS, -outer),
        Vec3::new(outer, 0.0, outer),
    );

    // Four walls. The -Z wall is split around a doorway, which is what makes the AO read
    // unambiguous: light spills through a gap only real geometry can shape.
    let mut walls = Builder::new("room_walls", 1);
    walls.add_box(
        Vec3::new(-outer, 0.0, ROOM_HALF),
        Vec3::new(outer, WALL_HEIGHT, outer),
    );
    walls.add_box(
        Vec3::new(-outer, 0.0, -outer),
        Vec3::new(-ROOM_HALF, WALL_HEIGHT, outer),
    );
    walls.add_box(
        Vec3::new(ROOM_HALF, 0.0, -outer),
        Vec3::new(outer, WALL_HEIGHT, outer),
    );
    // -Z wall, in two segments either side of the doorway.
    walls.add_box(
        Vec3::new(-ROOM_HALF, 0.0, -outer),
        Vec3::new(-DOOR_HALF_WIDTH, WALL_HEIGHT, -ROOM_HALF),
    );
    walls.add_box(
        Vec3::new(DOOR_HALF_WIDTH, 0.0, -outer),
        Vec3::new(ROOM_HALF, WALL_HEIGHT, -ROOM_HALF),
    );

    // Four pillars — free-standing occluders well clear of the walls, so their contact
    // shadows and their ambient occlusion are attributable to them alone.
    let mut pillars = Builder::new("room_pillars", 2);
    for &sx in &[-1.0f32, 1.0] {
        for &sz in &[-1.0f32, 1.0] {
            let c = Vec3::new(sx * PILLAR_OFFSET, 0.0, sz * PILLAR_OFFSET);
            pillars.add_box(
                c - Vec3::new(PILLAR_HALF, 0.0, PILLAR_HALF),
                c + Vec3::new(PILLAR_HALF, WALL_HEIGHT, PILLAR_HALF),
            );
        }
    }

    let materials = vec![
        GlbMaterial {
            name: "floor_stone".into(),
            base_color_factor: [0.34, 0.33, 0.31, 1.0],
            metallic: 0.0,
            roughness: 0.9,
            double_sided: false,
        },
        GlbMaterial {
            name: "wall_stone".into(),
            base_color_factor: [0.44, 0.43, 0.41, 1.0],
            metallic: 0.0,
            roughness: 0.85,
            double_sided: false,
        },
        GlbMaterial {
            name: "pillar_stone".into(),
            base_color_factor: [0.52, 0.50, 0.46, 1.0],
            metallic: 0.0,
            roughness: 0.8,
            double_sided: false,
        },
    ];
    (
        vec![floor.finish(), walls.finish(), pillars.finish()],
        materials,
    )
}

/// The level that places the generated room: the `.glb` at identity (it is authored in
/// world space) plus the same player placeholder the walking skeleton uses.
pub fn room_level_data(asset: &str) -> LevelData {
    LevelData {
        entities: vec![
            Entity {
                asset: asset.into(),
                name: Some("room".into()),
                transform: identity(),
                material_override: None,
            },
            player_entity(),
        ],
        lights: lights(),
        camera: rest_camera(),
        environment: environment(),
        deforms: Vec::new(),
    }
}

/// Accumulates boxes into one indexed triangle mesh.
struct Builder {
    name: String,
    material: usize,
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
}

impl Builder {
    fn new(name: &str, material: usize) -> Self {
        Self {
            name: name.to_owned(),
            material,
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    fn finish(self) -> GlbMesh {
        GlbMesh {
            name: self.name,
            vertices: self.vertices,
            indices: self.indices,
            material: self.material,
        }
    }

    /// Append an axis-aligned box with outward-facing, per-face normals.
    fn add_box(&mut self, min: Vec3, max: Vec3) {
        let e = max - min;
        let (x, y, z) = (
            Vec3::new(e.x, 0.0, 0.0),
            Vec3::new(0.0, e.y, 0.0),
            Vec3::new(0.0, 0.0, e.z),
        );
        // (face origin, du, dv) with `du × dv` pointing outward — which also makes the
        // quad winding below counter-clockwise seen from outside, i.e. front-facing.
        for (origin, du, dv) in [
            (min + x, y, z), // +X
            (min, z, y),     // -X
            (min + y, z, x), // +Y
            (min, x, z),     // -Y
            (min + z, x, y), // +Z
            (min, y, x),     // -Z
        ] {
            self.add_face(origin, du, dv);
        }
    }

    /// One planar quad, subdivided [`FACE_SUBDIV`]² times. UVs are the face-local
    /// position in metres (a world-planar projection — the generated materials carry no
    /// textures today, but a metre-scale UV is what a tiling one will want).
    fn add_face(&mut self, origin: Vec3, du: Vec3, dv: Vec3) {
        let n = FACE_SUBDIV.max(1);
        let normal = du.cross(dv).normalize();
        let (lu, lv) = (du.length(), dv.length());
        let base = self.vertices.len() as u32;
        let row = n + 1;
        for j in 0..row {
            for i in 0..row {
                let (fu, fv) = (i as f32 / n as f32, j as f32 / n as f32);
                self.vertices.push(MeshVertex {
                    pos: (origin + du * fu + dv * fv).to_array(),
                    normal: normal.to_array(),
                    uv: [fu * lu, fv * lv],
                });
            }
        }
        for j in 0..n {
            for i in 0..n {
                let a = base + j * row + i;
                let (b, c, d) = (a + 1, a + row + 1, a + row);
                self.indices.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
    }
}

// --- Files --------------------------------------------------------------------------

/// Path of the walking-skeleton `.level`.
pub fn level_path() -> PathBuf {
    Path::new(GENERATED_DIR).join(format!("{LEVEL_NAME}.level"))
}

/// Path of the generated room's `.level`.
pub fn room_level_path() -> PathBuf {
    Path::new(GENERATED_DIR).join(format!("{ROOM_LEVEL_NAME}.level"))
}

/// Path of the generated room's `.glb` — the asset the level references.
pub fn room_asset_path() -> PathBuf {
    Path::new(GENERATED_DIR).join(format!("{ROOM_LEVEL_NAME}.glb"))
}

/// Write `level` to `path` unless an identical file is already there.
///
/// The skip is not an optimization — it is the contract that makes a re-run free. The
/// level cook keys on the RON bytes, so rewriting identical content would still hit the
/// cache, but leaving the file alone keeps "nothing changed" honestly observable.
fn write_level_if_changed(path: &Path, level: &LevelData) -> anyhow::Result<()> {
    let ron = level.to_ron()?;
    if std::fs::read_to_string(path).ok().as_deref() == Some(ron.as_str()) {
        return Ok(());
    }
    std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
    std::fs::write(path, &ron)?;
    tracing::info!("dungeon: wrote level '{}'", path.display());
    Ok(())
}

/// Write the walking-skeleton level. Returns its stem (what `GameConfig::level` takes).
pub fn ensure_level_file() -> anyhow::Result<&'static str> {
    write_level_if_changed(&level_path(), &level_data())?;
    Ok(LEVEL_NAME)
}

/// Generate the room, write it as a `.glb` + a `.level` referencing it, and return the
/// level stem.
///
/// This is the M1 static-geometry injection path end to end (see the module docs): the
/// mesh never touches an engine registry directly — it becomes a file, and the engine's
/// ordinary level load cooks it, instantiates it and bakes it like any authored asset.
/// A dungeon generator swaps [`room_meshes`] for its own chunk meshes and changes
/// nothing else here.
pub fn ensure_generated_room() -> anyhow::Result<&'static str> {
    let (meshes, materials) = room_meshes();
    let asset = room_asset_path();
    let started = std::time::Instant::now();
    dreamcoast_asset::save_glb(&asset, &meshes, &materials)?;
    let tris: usize = meshes.iter().map(|m| m.indices.len() / 3).sum();
    let verts: usize = meshes.iter().map(|m| m.vertices.len()).sum();
    tracing::info!(
        "dungeon: generated room '{}' — {} meshes, {tris} triangles, {verts} vertices, \
         written in {:.1} ms",
        asset.display(),
        meshes.len(),
        started.elapsed().as_secs_f64() * 1e3,
    );

    // The level references the asset by the same cwd-relative string the engine resolves
    // and keys its cook cache on, so the key stays stable across runs and machines.
    let asset_key = asset.to_string_lossy().replace('\\', "/");
    write_level_if_changed(&room_level_path(), &room_level_data(&asset_key))?;
    Ok(ROOM_LEVEL_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The authored level must round-trip through the RON the engine parses, and the
    /// player placeholder must be findable by the name gameplay looks up.
    #[test]
    fn level_round_trips_and_names_the_player() {
        let level = level_data();
        let parsed: LevelData = ron::from_str(&level.to_ron().unwrap()).unwrap();
        assert_eq!(parsed, level);

        let player = level
            .entities
            .iter()
            .find(|e| e.name.as_deref() == Some(PLAYER_NAME))
            .expect("player placeholder is named");
        let translation = Mat4::from_cols_array(&player.transform).w_axis.truncate();
        assert_eq!(translation, PLAYER_SPAWN);
    }

    /// The generated room must be outward-facing: a mis-wound face flips the SDF sign
    /// for a whole half-space, which is how a generated chunk turns into phantom solid
    /// in the GI field instead of into a wall.
    #[test]
    fn generated_room_is_outward_facing() {
        let (meshes, materials) = room_meshes();
        assert_eq!(meshes.len(), 3);
        for mesh in &meshes {
            assert!(mesh.material < materials.len());
            assert!(mesh.indices.len().is_multiple_of(3));
            for tri in mesh.indices.chunks_exact(3) {
                let p: Vec<Vec3> = tri
                    .iter()
                    .map(|&i| Vec3::from(mesh.vertices[i as usize].pos))
                    .collect();
                let geometric = (p[1] - p[0]).cross(p[2] - p[0]);
                assert!(
                    geometric.length() > 1e-6,
                    "{}: degenerate triangle",
                    mesh.name
                );
                // The winding's normal must agree with the authored vertex normal — that
                // agreement is what the SDF bake's sign convention rests on.
                let authored = Vec3::from(mesh.vertices[tri[0] as usize].normal);
                assert!(
                    geometric.normalize().dot(authored) > 0.99,
                    "{}: face winding disagrees with its normal",
                    mesh.name
                );
            }
        }
    }

    /// Enough geometry to be a real measurement, and the walls must actually enclose the
    /// spawn (an open box would make the AO proof meaningless).
    #[test]
    fn generated_room_encloses_the_spawn() {
        let (meshes, _) = room_meshes();
        let tris: usize = meshes.iter().map(|m| m.indices.len() / 3).sum();
        assert!(
            (200..2000).contains(&tris),
            "unexpected triangle count {tris}"
        );

        let walls = meshes.iter().find(|m| m.name == "room_walls").unwrap();
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for v in &walls.vertices {
            min = min.min(Vec3::from(v.pos));
            max = max.max(Vec3::from(v.pos));
        }
        assert!(min.x < -ROOM_HALF && max.x > ROOM_HALF);
        assert!(min.z < -ROOM_HALF && max.z > ROOM_HALF);
        assert_eq!(max.y, WALL_HEIGHT);
        // The player spawns inside, on the floor the slab's top face defines.
        assert!(PLAYER_SPAWN.x.abs() < ROOM_HALF && PLAYER_SPAWN.z.abs() < ROOM_HALF);
    }

    /// The room level must reference the generated asset by a path the engine's glTF test
    /// accepts — otherwise it reads as a procedural primitive name and the load fails.
    #[test]
    fn room_level_references_the_generated_glb() {
        let key = room_asset_path().to_string_lossy().replace('\\', "/");
        let level = room_level_data(&key);
        let room = &level.entities[0];
        assert!(room.asset.ends_with(".glb"), "{}", room.asset);
        assert_eq!(room.transform, identity());
        assert_eq!(
            level.entities[1].name.as_deref(),
            Some(PLAYER_NAME),
            "the player rides along on the generated level too"
        );
        let parsed: LevelData = ron::from_str(&level.to_ron().unwrap()).unwrap();
        assert_eq!(parsed, level);
    }

    /// The generator is deterministic — the property the cook cache key rests on.
    #[test]
    fn generation_is_deterministic() {
        let (a, am) = room_meshes();
        let (b, bm) = room_meshes();
        assert_eq!(a, b);
        assert_eq!(am, bm);
        assert_eq!(
            dreamcoast_asset::write_glb(&a, &am).unwrap(),
            dreamcoast_asset::write_glb(&b, &bm).unwrap()
        );
    }
}
