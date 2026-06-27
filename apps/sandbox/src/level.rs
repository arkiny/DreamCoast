//! Declarative levels (Phase 12 Stage C): instantiate an `asset::LevelData` into the
//! ECS, plus the sandbox's built-in authored levels.
//!
//! A level is a flat list of placed entities, each referencing an asset by a logical
//! key — either a glTF file path (imported via Stage B and normalized to unit size)
//! or a procedural primitive (`sphere` / `cube`). The same `LevelData` model is the
//! single source of truth shared with the binary `.dcasset` cook (Stage E). Levels
//! render through the rasterizer + captured-cube IBL (the GDF/HW-RT path is
//! gallery-only), so switching levels needs no GDF rebuild.

use std::collections::HashMap;
use std::path::Path;

use dreamcoast_asset::cook::load_or_cook_level;
use dreamcoast_asset::level::{Entity as LevelEntity, LightKind, MaterialOverride};
use dreamcoast_asset::{GltfScene, LevelData, load_gltf_scene, unit_cube, uv_sphere};
use dreamcoast_core::glam::{Mat4, Vec3};
use dreamcoast_scene::{LocalTransform, MeshInstance, Name, Parent, World, instantiate_gltf};
use rhi::{Device, Texture};

use crate::NO_TEXTURE;
use crate::registry::{
    MaterialDesc, MaterialRegistry, MeshRegistry, PrimitiveHandles, gltf_bounds, upload_gltf_scene,
};

/// The world-space AABB of a placed scene (metres), for framing the camera.
pub(crate) type Bounds = (Vec3, Vec3);

/// Expand `[min, max]` by the 8 corners of a local AABB transformed by `place`.
fn expand_bounds(min: &mut Vec3, max: &mut Vec3, place: Mat4, lmin: Vec3, lmax: Vec3) {
    for &x in &[lmin.x, lmax.x] {
        for &y in &[lmin.y, lmax.y] {
            for &z in &[lmin.z, lmax.z] {
                let p = place.transform_point3(Vec3::new(x, y, z));
                *min = min.min(p);
                *max = max.max(p);
            }
        }
    }
}

/// A mesh's local AABB.
fn mesh_bounds(mesh: &dreamcoast_asset::MeshData) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for v in &mesh.vertices {
        let p = Vec3::from(v.pos);
        min = min.min(p);
        max = max.max(p);
    }
    (min, max)
}

/// Load a `.level` through the cook (Phase 12 Stage E): the RON cooks to a binary
/// `.dcasset` on first load and is read from that cache thereafter (no RON re-parse).
/// The cache key is the level's file name (stable, cwd-independent).
pub(crate) fn load(path: &Path) -> anyhow::Result<LevelData> {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy())
        .unwrap_or_default();
    let key = format!("levels/{name}");
    let (level, outcome) = load_or_cook_level(path, &key, &crate::app::cooked_cache_dir())?;
    tracing::info!("level '{}' ({outcome:?})", path.display());
    Ok(level)
}

/// A level's authored camera as `(eye, target)`, or `None` if it's the default (the
/// renderer then frames the scene by its AABB instead).
pub(crate) fn level_camera(level: &LevelData) -> Option<(Vec3, Vec3)> {
    use dreamcoast_asset::level::Camera;
    (level.camera != Camera::default()).then(|| {
        (
            Vec3::from(level.camera.position),
            Vec3::from(level.camera.target),
        )
    })
}

/// Whether an asset key names a glTF file (vs a procedural primitive).
fn is_gltf(asset: &str) -> bool {
    let a = asset.to_ascii_lowercase();
    a.ends_with(".glb") || a.ends_with(".gltf")
}

/// Decompose a column-major `[f32; 16]` world matrix into a `LocalTransform`.
fn local_from_cols(transform: &[f32; 16]) -> LocalTransform {
    let (scale, rotation, translation) =
        Mat4::from_cols_array(transform).to_scale_rotation_translation();
    LocalTransform {
        translation,
        rotation,
        scale,
    }
}

fn desc_from_override(ov: Option<MaterialOverride>) -> MaterialDesc {
    match ov {
        Some(o) => MaterialDesc {
            base_color: o.base_color_factor,
            metallic: o.metallic,
            roughness: o.roughness,
            tex: [NO_TEXTURE; 4],
        },
        None => MaterialDesc {
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.6,
            tex: [NO_TEXTURE; 4],
        },
    }
}

/// Instantiate a `LevelData` into `world`, uploading geometry/materials/textures into
/// the registries. glTF assets are imported + normalized to unit size, then placed by
/// the entity transform; procedural assets spawn a single entity with the override
/// material. The same glTF file referenced by several entities is imported once.
pub(crate) fn build_level(
    device: &Device,
    level: &LevelData,
    world: &mut World,
    meshes: &mut MeshRegistry,
    materials: &mut MaterialRegistry,
    textures: &mut Vec<Texture>,
    // World-space offset applied to every entity (Stage D chunk placement; Vec3::ZERO
    // for a standalone level).
    origin: Vec3,
) -> anyhow::Result<Option<Bounds>> {
    // Cache each glTF asset's import + uploaded handles so a row of the same model
    // (e.g. lanterns) uploads once.
    let mut gltf_cache: HashMap<String, (GltfScene, PrimitiveHandles)> = HashMap::new();
    let mut bmin = Vec3::splat(f32::MAX);
    let mut bmax = Vec3::splat(f32::MIN);

    for ent in &level.entities {
        // Assets are placed at their authored (native, 1 unit = 1 m) scale by the entity
        // transform — no auto-normalization, so a building stays building-sized.
        let place = Mat4::from_translation(origin) * Mat4::from_cols_array(&ent.transform);
        if is_gltf(&ent.asset) {
            if !gltf_cache.contains_key(&ent.asset) {
                let gscene = load_gltf_scene(&ent.asset)?;
                let handles = upload_gltf_scene(device, &gscene, meshes, materials, textures)?;
                gltf_cache.insert(ent.asset.clone(), (gscene, handles));
            }
            let (gscene, handles) = &gltf_cache[&ent.asset];
            let imported = instantiate_gltf(world, gscene, handles);
            let root = world.spawn();
            world.insert(root, local_from_cols(&place.to_cols_array()));
            world.insert(root, Name(ent.asset.clone()));
            world.insert(imported, Parent(root));
            if let Some((lmin, lmax)) = gltf_bounds(gscene) {
                expand_bounds(&mut bmin, &mut bmax, place, lmin, lmax);
            }
        } else {
            let mesh = match ent.asset.as_str() {
                "sphere" => uv_sphere(48, 32),
                "cube" => unit_cube(),
                // A unit (2×2 m) ground quad on y=0; the entity transform scales it.
                "ground" => crate::mesh::ground_mesh(1.0, 0.0),
                other => {
                    return Err(anyhow::anyhow!("level: unknown procedural asset '{other}'"));
                }
            };
            let (lmin, lmax) = mesh_bounds(&mesh);
            expand_bounds(&mut bmin, &mut bmax, place, lmin, lmax);
            let mesh_handle = meshes.upload(device, &mesh)?;
            let material = materials.add(desc_from_override(ent.material_override));
            world
                .spawn_node()
                .with(MeshInstance::new(mesh_handle, material))
                .with(local_from_cols(&place.to_cols_array()));
        }
    }
    Ok((bmin.x <= bmax.x).then_some((bmin, bmax)))
}

/// Discover the `.level` files in `dir`, writing the built-in levels first if missing
/// (so the files exist for hot-swap + hand editing). Returns the sorted path list.
pub(crate) fn ensure_level_files(dir: &Path) -> anyhow::Result<Vec<String>> {
    std::fs::create_dir_all(dir)?;
    for (name, builder) in [
        ("gallery.level", gallery_level as fn() -> LevelData),
        ("lanterns.level", lanterns_level as fn() -> LevelData),
        ("sponza.level", sponza_level as fn() -> LevelData),
    ] {
        let path = dir.join(name);
        if !path.exists() {
            builder().save_ron(&path)?;
        }
    }
    let mut paths: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "level"))
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    paths.sort();
    Ok(paths)
}

/// A column-major translation + uniform-scale transform as a `[f32; 16]`.
///
/// Transforms are in **metres** (the engine convention: 1 unit = 1 m). Assets load at
/// their authored native scale, so a level scales mis-authored assets here — e.g. the
/// Khronos Avocado is ~6 cm and the Lantern ~26 m, both placed to a sensible size.
fn trs(x: f32, y: f32, z: f32, s: f32) -> [f32; 16] {
    (Mat4::from_translation(Vec3::new(x, y, z)) * Mat4::from_scale(Vec3::splat(s))).to_cols_array()
}

/// A flat grey ground patch of `half` metres, centred on the origin (the floor a level
/// brings itself now that the hardcoded ground is gallery-only).
fn ground_entity(half: f32) -> LevelEntity {
    LevelEntity {
        asset: "ground".into(),
        transform: trs(0.0, 0.0, 0.0, half),
        material_override: Some(MaterialOverride {
            base_color_factor: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.9,
        }),
    }
}

/// The migrated gallery, in declarative form: the avocado glTF + chrome/copper
/// spheres + a red cube (mirrors the hardcoded gallery's layout/materials).
pub(crate) fn gallery_level() -> LevelData {
    use dreamcoast_asset::level::{Camera, Environment};
    LevelData {
        entities: vec![
            ground_entity(6.0),
            LevelEntity {
                // The 6 cm avocado scaled up to ~1 m to sit with the procedural spheres.
                asset: "assets/model.glb".into(),
                transform: trs(0.0, 0.0, 0.0, 18.0),
                material_override: None,
            },
            LevelEntity {
                asset: "sphere".into(),
                transform: trs(-1.7, 0.75, 0.5, 0.75),
                material_override: Some(MaterialOverride {
                    base_color_factor: [0.95, 0.96, 0.97, 1.0],
                    metallic: 1.0,
                    roughness: 0.08,
                }),
            },
            LevelEntity {
                asset: "sphere".into(),
                transform: trs(1.9, 0.5, -0.4, 0.5),
                material_override: Some(MaterialOverride {
                    base_color_factor: [0.95, 0.64, 0.54, 1.0],
                    metallic: 1.0,
                    roughness: 0.35,
                }),
            },
            LevelEntity {
                asset: "cube".into(),
                transform: trs(0.0, 0.45, -2.0, 0.45),
                material_override: Some(MaterialOverride {
                    base_color_factor: [0.85, 0.25, 0.2, 1.0],
                    metallic: 0.0,
                    roughness: 0.5,
                }),
            },
        ],
        lights: vec![],
        camera: Camera::default(),
        environment: Environment::default(),
    }
}

/// A row of Lantern instances — exercises instancing the same glTF hierarchy several
/// times from a level file.
pub(crate) fn lanterns_level() -> LevelData {
    use dreamcoast_asset::level::{Camera, Environment, Light};
    // The Lantern is authored at ~26 m; scale it to a ~4 m street-lamp size.
    let lantern = |x: f32| LevelEntity {
        asset: "assets/Lantern.glb".into(),
        transform: trs(x, 0.0, 0.0, 0.15),
        material_override: None,
    };
    LevelData {
        entities: vec![
            // A 16 m ground patch (tiles exactly with the demo world's 16 m chunk spacing).
            ground_entity(8.0),
            lantern(-4.0),
            lantern(0.0),
            lantern(4.0),
        ],
        lights: vec![Light {
            kind: LightKind::Directional,
            vec: [-0.4, -1.0, -0.3],
            color: [1.0, 0.95, 0.9],
            intensity: 3.0,
        }],
        camera: Camera::default(),
        environment: Environment::default(),
    }
}

/// A single Sponza instance — the large multi-material asset used for the Stage E
/// cooked-level verification. (Sponza is fetched locally via tools/fetch-sponza;
/// loading this level errors cleanly if it's absent.)
pub(crate) fn sponza_level() -> LevelData {
    use dreamcoast_asset::level::{Camera, Environment, Light};
    LevelData {
        entities: vec![LevelEntity {
            asset: "assets/Sponza/Sponza.gltf".into(),
            transform: trs(0.0, 0.0, 0.0, 1.0),
            material_override: None,
        }],
        // Lights live in the level asset, serialized by kind (directional `vec` is the
        // travel direction). A directional sun into the open courtyard plus warm/cool
        // point lights down the nave — a test bed for the lighting model + future GI.
        lights: vec![
            Light {
                kind: LightKind::Directional,
                vec: [0.3, -0.9, 0.25],
                color: [1.0, 0.96, 0.9],
                intensity: 4.0,
            },
            Light {
                kind: LightKind::Point,
                vec: [1.0, 1.2, -2.0],
                color: [1.0, 0.6, 0.3], // warm, near the camera in the nave
                intensity: 50.0,
            },
            Light {
                kind: LightKind::Point,
                vec: [8.0, 1.5, -3.0],
                color: [1.0, 0.6, 0.3], // warm, down the nave
                intensity: 60.0,
            },
            Light {
                kind: LightKind::Point,
                vec: [-4.0, 1.5, 3.0],
                color: [0.4, 0.5, 1.0], // cool fill across the courtyard
                intensity: 40.0,
            },
        ],
        // The iconic Sponza atrium angle: standing in the open courtyard looking down
        // the length, the draped arcades receding on both sides.
        camera: Camera {
            position: [-6.0, 2.0, 0.0],
            target: [8.0, 3.0, -5.0],
            fov_y_deg: 60.0,
            znear: 0.05,
            zfar: 100.0,
        },
        environment: Environment::default(),
    }
}
