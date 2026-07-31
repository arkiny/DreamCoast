//! The dungeon's levels — authored **from code**, serialized to the `.level` RON the
//! engine's declarative loader consumes.
//!
//! Why from code rather than a hand-written file: gameplay needs the same numbers the
//! scene is built from (where the player starts, where the walls are), and a duplicated
//! literal in a data file is drift waiting to happen. Here the [`TileGrid`] is that
//! single source — the geometry below is meshed from it and [`crate::game`] collides
//! against the very same instance (see [`crate::main`]).
//!
//! Two levels live here:
//!
//! * **`dungeon_<seed>`** — the game: a seeded dungeon, meshed into per-chunk geometry
//!   ([`ensure_dungeon`]). One file per seed, so re-running a seed you have already
//!   played is a pure cache hit and the engine's hot-swap dropdown lists the seeds you
//!   have generated.
//! * **`dungeon_room`** — the M1 **static-geometry injection proof** kept from the seam
//!   work ([`ensure_generated_room`], `--generated-room`): a single hand-built room whose
//!   walls and pillars are generated at runtime. It stays because it is the *minimal*
//!   repro of the injection road — three meshes, no generator involved — so when a bake
//!   looks wrong it tells you whether the dungeon or the road is at fault.
//!
//! The M0 walking-skeleton level (a flat ground plate, three landmark blocks, the player
//! clamped to the plate's rectangle) is **gone**: every part of it that was load-bearing
//! now has a stronger replacement — the seam it proved is proved by the dungeon itself,
//! and its ground-rectangle clamp was replaced by real grid collision. Keeping a second
//! "level authored from constants" would have meant a second spawn convention to keep in
//! sync with gameplay, which is exactly the drift this module exists to prevent.
//!
//! **Placement:** everything is written to [`GENERATED_DIR`] (derived data — the whole
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
//! generator → mesher → dreamcoast_asset::save_glb → cache/generated/<name>.glb
//!                                                 + cache/generated/<name>.level
//!                                                   → the engine's normal level load
//!                                                     → cook (.dcasset) → per-mesh SDF bake
//!                                                       → GDF / surface cache / GI / reflections
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

use dreamcoast_asset::level::{Camera, Entity, Environment, Light, LightKind};
use dreamcoast_asset::{GlbMaterial, GlbMesh, LevelData, MeshVertex};
use glam::{Mat4, Vec2, Vec3};

use crate::collision::{CHARACTER_Y, player_spawn, to_world};
use crate::meshing::{ChunkMesh, MeshParams, mesh_chunks, mesh_stats};
use crate::procgen::TileGrid;
use crate::rigs;

/// Where this game's generated levels + assets are written, relative to the working
/// directory. Derived data (gitignored, alongside the cooked-asset cache) — nothing in
/// it is hand-edited, and deleting it costs only a regenerate + re-cook.
pub const GENERATED_DIR: &str = "cache/generated";

/// The generated-room proof level's stem — the M1 static-injection harness.
pub const ROOM_LEVEL_NAME: &str = "dungeon_room";

/// Scene-graph name the player character is authored with. Gameplay finds its entity by
/// this (`crate::game::DungeonGame::find_player`) rather than by matching a spawn
/// transform, so moving the spawn cannot silently break the lookup.
pub const PLAYER_NAME: &str = "player";

/// Scene-graph name of the `i`-th monster (`grunt_0`, `grunt_1`, …).
///
/// Positional, and deliberately so: [`crate::ai::spawn_points`] is a deterministic
/// function of the seed, this writer places point `i` under this name, and gameplay
/// re-acquires point `i`'s brain by it. One ordered list, three readers, no matching
/// heuristic — a monster cannot end up driving a different monster's body.
pub fn grunt_name(index: usize) -> String {
    format!("grunt_{index}")
}

/// Direction the sun *travels* (the level convention the loader negates into a
/// direction-toward-light). Deliberately low in the sky: a top-down camera reads
/// depth from cast shadows, and a near-vertical sun leaves none.
const SUN_DIR: [f32; 3] = [-0.55, -0.75, -0.35];
/// Sun radiance on this level's own scale (the same arbitrary scale `sponza.level`
/// authors on, not lux — the auto-exposure meters whatever it is given).
const SUN_INTENSITY: f32 = 3.5;

/// Height of the torch above the floor, metres — eye level, well under the 4 m ceiling
/// line so it lights the walls around the spawn rather than the wall tops.
const TORCH_HEIGHT: f32 = 2.4;

/// A column-major translation + uniform-scale transform, the form `Entity::transform`
/// takes (`glam::Mat4::to_cols_array` order).
fn trs(t: Vec3, scale: f32) -> [f32; 16] {
    (Mat4::from_translation(t) * Mat4::from_scale(Vec3::splat(scale))).to_cols_array()
}

/// The identity transform — generated geometry is authored in world space already.
fn identity() -> [f32; 16] {
    Mat4::IDENTITY.to_cols_array()
}

/// A character placement: a rig's `.glb` at `spawn`, at its authored (metre) scale.
///
/// The `.glb` path is the same cwd-relative string [`rigs::ensure_rigs`] writes and the
/// engine cooks against, so the cook key is stable across runs and machines — and,
/// because six grunts reference one asset, the loader imports and uploads it once and
/// instantiates it six times (`level::build_level`'s per-asset cache).
///
/// No `material_override`: the loader ignores it for glTF assets (it is the procedural
/// primitives' knob), and the rigs carry their own four/two materials — the plate, the
/// blade and the crest that make a knight read from 16 m up (see [`crate::rigs`]).
fn character_entity(asset: &str, name: String, spawn: Vec3) -> Entity {
    Entity {
        asset: asset.into(),
        name: Some(name),
        transform: trs(spawn, 1.0),
        material_override: None,
    }
}

/// The cwd-relative asset key for a rig's `.glb`, normalised to forward slashes so the
/// cook key is identical on every platform.
fn rig_asset_key(name: &str) -> String {
    rigs::rig_asset_path(name)
        .to_string_lossy()
        .replace('\\', "/")
}

/// The player and the monsters, as level entities.
///
/// `grunt_spawns` is [`crate::ai::spawn_points`]' output in **collision space**; it is
/// passed in rather than recomputed here for the same reason the grid is
/// ([`ensure_dungeon`]): the game owns the one list, and a second call to a deterministic
/// function is still a second source of truth.
fn character_entities(grid: &TileGrid, grunt_spawns: &[Vec2]) -> Vec<Entity> {
    let mut out = vec![character_entity(
        &rig_asset_key(rigs::WARRIOR_RIG),
        PLAYER_NAME.into(),
        player_spawn(grid),
    )];
    let grunt = rig_asset_key(rigs::GRUNT_RIG);
    out.extend(grunt_spawns.iter().enumerate().map(|(i, &local)| {
        character_entity(&grunt, grunt_name(i), to_world(grid, local, CHARACTER_Y))
    }));
    out
}

/// This game's sun + one torch, the torch placed at `torch` (the spawn, so the room the
/// player opens their eyes in is lit from inside rather than only by the sky).
fn lights(torch: Vec3) -> Vec<Light> {
    vec![
        Light {
            kind: LightKind::Directional,
            vec: SUN_DIR,
            color: [1.0, 0.96, 0.90],
            intensity: SUN_INTENSITY,
        },
        Light {
            kind: LightKind::Point,
            vec: [torch.x, TORCH_HEIGHT, torch.z],
            color: [1.0, 0.62, 0.28],
            intensity: 8.0,
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
fn rest_camera(focus: Vec3) -> Camera {
    Camera {
        position: (focus + crate::game::camera_offset()).to_array(),
        target: focus.to_array(),
        fov_y_deg: 60.0,
        znear: 0.05,
        // Matching the engine's own frame projection, which uses the fixed
        // `CLUSTER_Z_NEAR`/`CLUSTER_Z_FAR` (0.05 / 100 m) constants rather than these
        // fields — a level that authored a larger `zfar` here would be describing a
        // view distance the renderer does not have. 100 m covers an 80 m dungeon from
        // any point inside it; it does *not* cover a bird's-eye diagnostic camera
        // parked 95 m up, where the far half of the map clips away (see the M1 report).
        zfar: 100.0,
    }
}

// --- The dungeon ---------------------------------------------------------------------

/// Material slots in the generated `.glb`, in write order.
///
/// Two, not one: the floor and the walls are the dungeon's only two readable surfaces,
/// and a top-down camera looking at a single grey solid has no silhouette to read a room
/// from. The split is by **face orientation** (see [`chunk_glb_meshes`]) — no tagging
/// pass, no second mesher mode.
const MAT_FLOOR: usize = 0;
const MAT_WALL: usize = 1;

/// The dungeon's stone. Placeholder factors until the AI-generated base/MR/normal
/// textures land in M2 — chosen so the *shape* reads now:
///
/// * the walls are lighter and noticeably smoother than the floor, so a lit wall face
///   separates from the floor it stands on even where both are in shadow;
/// * the floor is the rougher, darker, very slightly warmer of the two, which is what
///   keeps a big open room from reading as a bright plate;
/// * both are dielectric (`metallic = 0`) with high roughness, so the surface cache and
///   the reflection path have nothing sharp to resolve — cheap, and honest for stone.
fn dungeon_materials() -> Vec<GlbMaterial> {
    vec![
        GlbMaterial {
            name: "dungeon_floor_stone".into(),
            base_color_factor: [0.24, 0.23, 0.22, 1.0],
            metallic: 0.0,
            roughness: 0.92,
            double_sided: false,
        },
        GlbMaterial {
            name: "dungeon_wall_stone".into(),
            base_color_factor: [0.56, 0.55, 0.52, 1.0],
            metallic: 0.0,
            roughness: 0.72,
            double_sided: false,
        },
    ]
}

/// Split one chunk's triangles by facing into the meshes the `.glb` carries.
///
/// A [`GlbMesh`] holds one material, so a chunk that has both floor and wall faces
/// becomes two meshes — and therefore two glTF nodes, two draws, two SDF bakes. That is
/// still `O(chunks)`, which is the property the chunking exists for, and it keeps each
/// piece its own drawable for culling.
///
/// Horizontal faces (floors, and ceilings when enabled) are the ones whose normal has a
/// Y component; everything else is a wall. Classifying by the geometry rather than by a
/// tag keeps `meshing.rs` a pure geometry producer.
///
/// Vertices are re-indexed in first-use order and triangles kept in the mesher's order,
/// so the output is a pure function of the chunk — no hashing, no sorting, nothing that
/// could reorder between runs (the cook cache keys on these bytes).
fn chunk_glb_meshes(chunk: &ChunkMesh) -> Vec<GlbMesh> {
    let (cx, cz) = chunk.chunk_coord;
    [(MAT_FLOOR, "floor", true), (MAT_WALL, "wall", false)]
        .into_iter()
        .filter_map(|(material, suffix, horizontal)| {
            let mut remap = vec![u32::MAX; chunk.vertices.len()];
            let mut vertices: Vec<MeshVertex> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            for tri in chunk.indices.chunks_exact(3) {
                if (chunk.vertices[tri[0] as usize].normal[1] != 0.0) != horizontal {
                    continue;
                }
                for &i in tri {
                    if remap[i as usize] == u32::MAX {
                        remap[i as usize] = vertices.len() as u32;
                        vertices.push(chunk.vertices[i as usize]);
                    }
                    indices.push(remap[i as usize]);
                }
            }
            (!indices.is_empty()).then(|| GlbMesh {
                name: format!("chunk_{cx}_{cz}_{suffix}"),
                vertices,
                indices,
                material,
            })
        })
        .collect()
}

/// The dungeon's geometry: the grid, greedy-meshed per chunk, as `.glb` meshes.
pub fn dungeon_meshes(grid: &TileGrid) -> (Vec<GlbMesh>, Vec<GlbMaterial>) {
    let chunks = mesh_chunks(grid, &MeshParams::default());
    let meshes = chunks.iter().flat_map(chunk_glb_meshes).collect();
    (meshes, dungeon_materials())
}

/// The level that places a generated dungeon: the `.glb` at identity (it is authored in
/// world space), the warrior on the entry tile, and one grunt per spawn point.
pub fn dungeon_level_data(grid: &TileGrid, asset: &str, grunt_spawns: &[Vec2]) -> LevelData {
    let spawn = player_spawn(grid);
    let mut entities = vec![Entity {
        asset: asset.into(),
        name: Some("dungeon".into()),
        transform: identity(),
        material_override: None,
    }];
    entities.extend(character_entities(grid, grunt_spawns));
    LevelData {
        entities,
        lights: lights(spawn),
        camera: rest_camera(spawn),
        environment: environment(),
        deforms: Vec::new(),
    }
}

/// The level stem for a seed. One file per seed keeps a replayed seed a pure cache hit
/// (nothing is rewritten, so nothing re-cooks) and makes the engine's hot-swap dropdown
/// a list of the dungeons you have generated.
///
/// Floors key by seed and nothing else, because a floor *is* a seed
/// ([`crate::game::floor_seed`]): descending re-derives one, so a floor already visited
/// on this machine — this run's or an earlier one's — writes bytes that are already
/// there and re-cooks nothing.
pub fn dungeon_level_name(seed: u64) -> String {
    format!("dungeon_{seed}")
}

/// The selector a *running* game hands [`sandbox::GameHooks::next_level`] for a seed:
/// the `.level`'s cwd-relative path, forward-slashed.
///
/// A path rather than the stem [`ensure_dungeon`] returns, because the two resolve
/// differently at runtime: a stem is matched against the levels the engine *discovered
/// at bring-up*, and a floor written after that was not there to discover, so it would
/// resolve only by the accident of having been played before. An explicit path is
/// registered on the spot (`sandbox::level::resolve_selection`) and is therefore the
/// only spelling that is correct for a floor the game just generated.
pub fn dungeon_level_selector(seed: u64) -> String {
    generated_path(&format!("{}.level", dungeon_level_name(seed)))
        .to_string_lossy()
        .replace('\\', "/")
}

/// Mesh `grid`, write it as a `.glb` + a `.level` referencing it, and return the level
/// stem (what `GameConfig::level` takes).
///
/// Takes the grid by reference on purpose: the caller (and therefore the game) owns the
/// one [`TileGrid`] instance, so the geometry written here and the collision the player
/// walks against are the same dungeon by construction, not by regenerating from the same
/// seed twice and hoping. `grunt_spawns` is threaded the same way — the game's own list,
/// not a second [`crate::ai::spawn_points`] call.
///
/// **[`rigs::ensure_rigs`] must have run first**: the level references the two character
/// `.glb`s by path, and the engine's loader resolves them at load time. `crate::main`
/// owns that ordering (the same way it owns "generate before mesh").
///
/// One `.level` per seed, but its *contents* also carry the grunt count, so replaying a
/// seed with a different `--grunts` rewrites the level (a cheap re-cook of the RON) and
/// leaves the far larger geometry `.glb` untouched.
pub fn ensure_dungeon(grid: &TileGrid, grunt_spawns: &[Vec2]) -> anyhow::Result<String> {
    let started = std::time::Instant::now();
    let (meshes, materials) = dungeon_meshes(grid);
    let stats = mesh_stats(&mesh_chunks(grid, &MeshParams::default()));
    let bytes = dreamcoast_asset::write_glb(&meshes, &materials)?;
    let meshed = started.elapsed();

    let name = dungeon_level_name(grid.seed());
    let asset = generated_path(&format!("{name}.glb"));
    let wrote = write_if_changed(&asset, &bytes)?;
    tracing::info!(
        "dungeon: seed {} — {}x{} tiles, {} rooms, {} walkable; {} chunks → {} meshes, \
         {} triangles, {} vertices ({:.2} MB glb, {}) in {:.1} ms",
        grid.seed(),
        grid.width(),
        grid.height(),
        grid.rooms().len(),
        grid.walkable_count(),
        stats.chunks,
        meshes.len(),
        stats.triangles,
        stats.vertices,
        bytes.len() as f64 / (1024.0 * 1024.0),
        if wrote { "written" } else { "unchanged" },
        meshed.as_secs_f64() * 1e3,
    );

    // The level references the asset by the same cwd-relative string the engine resolves
    // and keys its cook cache on, so the key stays stable across runs and machines.
    let asset_key = asset.to_string_lossy().replace('\\', "/");
    write_level_if_changed(
        &generated_path(&format!("{name}.level")),
        &dungeon_level_data(grid, &asset_key, grunt_spawns),
    )?;
    Ok(name)
}

// --- The generated-room proof harness ------------------------------------------------

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

/// The proof room's collision grid: a 10x10 ring of rock around an 8x8 tile interior.
///
/// Not a decoration — the room's walls really do stand at ±[`ROOM_HALF`] = ±8 m, and a
/// 10-tile grid centred on the origin puts its interior tiles at exactly -8..8 m, so the
/// player collides with the harness room's walls where they are drawn. (The four pillars
/// are 0.7 m posts, far smaller than a tile, and are deliberately *not* in the grid: the
/// harness exists to test the injection road, and tile collision does not represent
/// sub-tile props — that is the deferred SDF-prop track.)
pub fn room_collision_grid() -> TileGrid {
    TileGrid::from_rows(&[
        "##########",
        "#........#",
        "#........#",
        "#........#",
        "#...E....#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "##########",
    ])
}

/// The generated room's geometry, as one [`GlbMesh`] per material group.
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
/// world space) plus the warrior on the harness grid's entry tile.
///
/// No grunts: the harness exists to test the *static-geometry injection road*, and a
/// monster in it would be a second thing to explain when a bake looks wrong.
pub fn room_level_data(asset: &str) -> LevelData {
    let grid = room_collision_grid();
    let spawn = player_spawn(&grid);
    let mut entities = vec![Entity {
        asset: asset.into(),
        name: Some("room".into()),
        transform: identity(),
        material_override: None,
    }];
    entities.extend(character_entities(&grid, &[]));
    LevelData {
        entities,
        lights: lights(spawn),
        camera: rest_camera(spawn),
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

/// A path inside [`GENERATED_DIR`].
fn generated_path(file: &str) -> PathBuf {
    Path::new(GENERATED_DIR).join(file)
}

/// Path of the generated room's `.level`.
pub fn room_level_path() -> PathBuf {
    generated_path(&format!("{ROOM_LEVEL_NAME}.level"))
}

/// Path of the generated room's `.glb` — the asset the level references.
pub fn room_asset_path() -> PathBuf {
    generated_path(&format!("{ROOM_LEVEL_NAME}.glb"))
}

/// Write `bytes` to `path` unless an identical file is already there; reports whether it
/// wrote.
///
/// The skip is not an optimization — it is the contract that makes a re-run free. The
/// cook keys on content, so rewriting identical bytes would still hit the cache, but
/// leaving the file alone keeps "nothing changed" honestly observable (an unchanged
/// mtime, an untouched cache directory).
fn write_if_changed(path: &Path, bytes: &[u8]) -> anyhow::Result<bool> {
    if std::fs::read(path).ok().as_deref() == Some(bytes) {
        return Ok(false);
    }
    std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
    std::fs::write(path, bytes)?;
    Ok(true)
}

/// Write `level` to `path` unless an identical file is already there.
fn write_level_if_changed(path: &Path, level: &LevelData) -> anyhow::Result<()> {
    if write_if_changed(path, level.to_ron()?.as_bytes())? {
        tracing::info!("dungeon: wrote level '{}'", path.display());
    }
    Ok(())
}

/// Generate the proof room, write it as a `.glb` + a `.level` referencing it, and return
/// the level stem.
///
/// This is the M1 static-geometry injection path end to end (see the module docs): the
/// mesh never touches an engine registry directly — it becomes a file, and the engine's
/// ordinary level load cooks it, instantiates it and bakes it like any authored asset.
/// [`ensure_dungeon`] is the same road with the real generator's meshes on it.
///
/// As [`ensure_dungeon`], this places the warrior and so needs [`rigs::ensure_rigs`] to
/// have run first; `crate::main` calls it for both paths.
pub fn ensure_generated_room() -> anyhow::Result<&'static str> {
    let (meshes, materials) = room_meshes();
    let asset = room_asset_path();
    let started = std::time::Instant::now();
    let bytes = dreamcoast_asset::write_glb(&meshes, &materials)?;
    let wrote = write_if_changed(&asset, &bytes)?;
    let tris: usize = meshes.iter().map(|m| m.indices.len() / 3).sum();
    let verts: usize = meshes.iter().map(|m| m.vertices.len()).sum();
    tracing::info!(
        "dungeon: generated room '{}' — {} meshes, {tris} triangles, {verts} vertices, \
         {} in {:.1} ms",
        asset.display(),
        meshes.len(),
        if wrote { "written" } else { "unchanged" },
        started.elapsed().as_secs_f64() * 1e3,
    );

    let asset_key = asset.to_string_lossy().replace('\\', "/");
    write_level_if_changed(&room_level_path(), &room_level_data(&asset_key))?;
    Ok(ROOM_LEVEL_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::to_collision;
    use crate::procgen::{DungeonParams, generate};
    use std::collections::BTreeSet;

    fn test_grid(seed: u64) -> TileGrid {
        generate(seed, &DungeonParams::default())
    }

    /// Every triangle's winding must agree with its shading normal *in the direction its
    /// producer intends*, uniformly.
    ///
    /// `sign` is the expected sign of `cross(p1-p0, p2-p0) · normal`. Both producers
    /// here wind counter-clockwise about the normal (`+1`, the glTF front-face
    /// convention): the virtual-geometry G-buffer producer backface-culls single-sided
    /// materials per triangle, so a backwards quad is invisible even though it still
    /// lights, shadows and occludes. A mesh that *mixes* the two is worse than either —
    /// the per-mesh SDF signs by averaged vertex normals (`crates/asset/src/sdf.rs`), so
    /// a mis-wound face flips the sign for a whole half-space and generated geometry
    /// becomes phantom solid in the GI field instead of a wall.
    fn assert_consistent_winding(meshes: &[GlbMesh], materials: &[GlbMaterial], sign: f32) {
        for mesh in meshes {
            assert!(mesh.material < materials.len(), "{}", mesh.name);
            assert!(mesh.indices.len().is_multiple_of(3), "{}", mesh.name);
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
                let authored = Vec3::from(mesh.vertices[tri[0] as usize].normal);
                assert!(
                    geometric.normalize().dot(authored) * sign > 0.99,
                    "{}: face winding disagrees with its normal",
                    mesh.name
                );
            }
        }
    }

    /// The dungeon's `.glb` conversion must preserve the mesher's geometry exactly: same
    /// triangle count, every triangle sorted into exactly one material group, nothing
    /// invented and nothing dropped.
    #[test]
    fn chunk_conversion_partitions_the_mesher_output() {
        let grid = test_grid(3);
        let chunks = mesh_chunks(&grid, &MeshParams::default());
        let stats = mesh_stats(&chunks);
        let (meshes, materials) = dungeon_meshes(&grid);
        assert!(!meshes.is_empty());
        assert_eq!(
            meshes.iter().map(|m| m.indices.len() / 3).sum::<usize>(),
            stats.triangles,
            "no triangle gained or lost in the split"
        );
        assert_consistent_winding(&meshes, &materials, 1.0);

        // Every floor mesh is horizontal, every wall mesh vertical — the property the
        // material assignment rests on.
        for mesh in &meshes {
            let horizontal = mesh.material == MAT_FLOOR;
            assert_eq!(mesh.name.ends_with("floor"), horizontal, "{}", mesh.name);
            for v in &mesh.vertices {
                assert_eq!(v.normal[1] != 0.0, horizontal, "{}", mesh.name);
            }
        }

        // One node per chunk per material group, each named for its chunk so a bake or a
        // draw can be traced back to a place on the grid.
        let names: BTreeSet<&str> = meshes.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names.len(), meshes.len(), "chunk mesh names are unique");
        assert!(meshes.len() <= chunks.len() * 2);
    }

    /// The dungeon level must reference the generated asset by a path the engine's glTF
    /// test accepts, and place the warrior exactly where gameplay will start it.
    #[test]
    fn dungeon_level_places_the_player_on_the_spawn() {
        let grid = test_grid(3);
        let level = dungeon_level_data(&grid, "cache/generated/dungeon_3.glb", &[]);
        assert!(level.entities[0].asset.ends_with(".glb"));
        assert_eq!(level.entities[0].transform, identity());

        let player = &level.entities[1];
        assert_eq!(player.name.as_deref(), Some(PLAYER_NAME));
        assert_eq!(player.asset, rig_asset_key(rigs::WARRIOR_RIG));
        let placed = Mat4::from_cols_array(&player.transform).w_axis.truncate();
        assert_eq!(placed, player_spawn(&grid));
        assert_eq!(placed.y, CHARACTER_Y, "the warrior stands on the floor");
        assert_eq!(
            crate::collision::tile_of(to_collision(&grid, placed)),
            grid.entry(),
            "the warrior stands on the entry tile"
        );

        let parsed: LevelData = ron::from_str(&level.to_ron().unwrap()).unwrap();
        assert_eq!(parsed, level);
    }

    /// **Spawn points round-trip into the level**, on every seed: one `grunt_<i>` entity
    /// per point, in order, at that point's world position, referencing the grunt rig —
    /// and each one standing in free space on a walkable tile.
    ///
    /// This is the seam the whole monster wiring rests on: `ai::spawn_points` chooses,
    /// the writer places, and `game::acquire` re-pairs brain `i` with `grunt_<i>`. A
    /// reordering anywhere in that chain puts a brain in someone else's body.
    #[test]
    fn spawn_points_round_trip_into_level_entities() {
        use crate::ai::{GRUNT_RADIUS, spawn_points};
        use crate::collision::collision;

        for seed in [1u64, 3, 7, 11, 20260731] {
            let grid = test_grid(seed);
            let spawns = spawn_points(&grid, 6, 12.0, grid.seed());
            assert!(!spawns.is_empty(), "seed {seed}: no room for any monster");

            let level = dungeon_level_data(&grid, "cache/generated/x.glb", &spawns);
            // [geometry, player, grunt_0 ..]
            assert_eq!(level.entities.len(), 2 + spawns.len(), "seed {seed}");

            let grunt_asset = rig_asset_key(rigs::GRUNT_RIG);
            for (i, &local) in spawns.iter().enumerate() {
                let entity = &level.entities[2 + i];
                assert_eq!(entity.name.as_deref(), Some(grunt_name(i).as_str()));
                assert_eq!(entity.asset, grunt_asset, "seed {seed}: grunt {i} asset");
                let placed = Mat4::from_cols_array(&entity.transform).w_axis.truncate();
                assert_eq!(placed, to_world(&grid, local, CHARACTER_Y));
                // ...and back the other way, which is what `game::acquire` relies on.
                assert_eq!(to_collision(&grid, placed), local, "seed {seed}: grunt {i}");

                let (tx, tz) = crate::collision::tile_of(local);
                assert!(grid.is_walkable(tx, tz), "seed {seed}: grunt {i} in rock");
                assert!(
                    !collision(&grid).circle_overlaps(local, GRUNT_RADIUS),
                    "seed {seed}: grunt {i} overlaps geometry"
                );
                assert_ne!(
                    grid.room_id_at(tx, tz),
                    grid.room_id_at(grid.entry().0, grid.entry().1),
                    "seed {seed}: grunt {i} spawned in the entry room"
                );
            }

            // The names are unique, so the lookup that pairs them cannot collide.
            let names: BTreeSet<&str> = level
                .entities
                .iter()
                .filter_map(|e| e.name.as_deref())
                .collect();
            assert_eq!(names.len(), level.entities.len(), "seed {seed}");

            let parsed: LevelData = ron::from_str(&level.to_ron().unwrap()).unwrap();
            assert_eq!(parsed, level, "seed {seed}: level RON round-trip");
        }
    }

    /// The monster count is level *content*, so two counts on one seed are two different
    /// files — which is what makes `write_if_changed` rewrite the level (and only the
    /// level: the geometry `.glb` is untouched) when `--grunts` changes.
    #[test]
    fn the_grunt_count_changes_the_level_but_not_the_geometry() {
        use crate::ai::spawn_points;
        let grid = test_grid(3);
        let few = spawn_points(&grid, 2, 12.0, grid.seed());
        let many = spawn_points(&grid, 6, 12.0, grid.seed());
        assert!(many.len() > few.len());
        assert_eq!(&many[..few.len()], &few[..], "the list only grows");

        let a = dungeon_level_data(&grid, "x.glb", &few).to_ron().unwrap();
        let b = dungeon_level_data(&grid, "x.glb", &many).to_ron().unwrap();
        assert_ne!(a, b);
        // Same seed, same geometry bytes, whatever the population.
        let (meshes, materials) = dungeon_meshes(&grid);
        assert_eq!(
            dreamcoast_asset::write_glb(&meshes, &materials).unwrap(),
            dreamcoast_asset::write_glb(&meshes, &materials).unwrap()
        );
    }

    /// A floor is named by its seed, and the *running* game asks for it by path.
    ///
    /// The two spellings are not interchangeable at runtime — a stem resolves only
    /// against the levels the engine discovered at bring-up, and a floor generated after
    /// that was not among them — so the selector must stay an explicit, forward-slashed
    /// path inside the generated directory, whatever the platform.
    #[test]
    fn a_floor_is_named_by_its_seed_and_selected_by_path() {
        assert_eq!(dungeon_level_name(20260731), "dungeon_20260731");
        let selector = dungeon_level_selector(20260731);
        assert_eq!(selector, "cache/generated/dungeon_20260731.level");
        assert!(!selector.contains('\\'), "{selector}");
        // What `sandbox::level::resolve_selection` calls an explicit path: it contains a
        // separator *and* ends in `.level`, so it is registered rather than looked up.
        assert!(selector.contains('/') && selector.ends_with(".level"));
        assert!(selector.starts_with(GENERATED_DIR));

        // Different floors of one run are different files; the same floor revisited is
        // the same file (which is what makes a revisit a pure cache hit).
        let (a, b) = (
            crate::game::floor_seed(20260731, 1),
            crate::game::floor_seed(20260731, 2),
        );
        assert_ne!(dungeon_level_selector(a), dungeon_level_selector(b));
        assert_eq!(dungeon_level_selector(b), dungeon_level_selector(b));
        assert_eq!(
            dungeon_level_selector(a),
            dungeon_level_selector(20260731),
            "floor 1 is the run seed's own level — the one every recipe names"
        );
    }

    /// Same seed, byte-identical `.glb` — the property the cook cache (and therefore the
    /// SDF bake cache) keys on. Different seeds must not collide.
    #[test]
    fn generation_is_deterministic() {
        let bytes = |seed| {
            let grid = test_grid(seed);
            let (m, mat) = dungeon_meshes(&grid);
            dreamcoast_asset::write_glb(&m, &mat).unwrap()
        };
        assert_eq!(bytes(3), bytes(3));
        assert_ne!(bytes(3), bytes(4));

        let (a, am) = room_meshes();
        let (b, bm) = room_meshes();
        assert_eq!(a, b);
        assert_eq!(am, bm);
    }

    /// The proof room's harness grid must agree with the room's *drawn* walls: its
    /// interior is exactly the room interior, so the player stops where the wall is.
    #[test]
    fn the_room_harness_grid_matches_the_room_geometry() {
        let grid = room_collision_grid();
        assert_eq!((grid.width(), grid.height()), (10, 10));
        // Interior tiles span -ROOM_HALF..ROOM_HALF on both axes.
        assert_eq!(grid.tile_edge_x(1), -ROOM_HALF);
        assert_eq!(grid.tile_edge_x(grid.width() - 1), ROOM_HALF);
        assert_eq!(grid.tile_edge_z(1), -ROOM_HALF);
        assert_eq!(grid.tile_edge_z(grid.height() - 1), ROOM_HALF);
        assert!(grid.all_walkable_reachable());

        let spawn = player_spawn(&grid);
        assert!(spawn.x.abs() < ROOM_HALF && spawn.z.abs() < ROOM_HALF);
    }

    /// The proof room is still outward-facing and still encloses its spawn.
    #[test]
    fn generated_room_is_outward_facing_and_encloses_the_spawn() {
        let (meshes, materials) = room_meshes();
        assert_eq!(meshes.len(), 3);
        assert_consistent_winding(&meshes, &materials, 1.0);
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

        let level = room_level_data("cache/generated/dungeon_room.glb");
        let parsed: LevelData = ron::from_str(&level.to_ron().unwrap()).unwrap();
        assert_eq!(parsed, level);
        assert_eq!(level.entities[1].name.as_deref(), Some(PLAYER_NAME));
    }
}
