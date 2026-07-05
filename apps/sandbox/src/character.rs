//! Skinned / static character overlay on a loaded level (Phase 13 Stage E).
//!
//! `--level` mode builds the scene via [`crate::level::build_level`], which wires only
//! static geometry (`instantiate_gltf`, no skin cache). This module adds characters *on
//! top* of an already-loaded level: it instantiates a glTF/FBX scene into the same ECS
//! world + registries, parents it under a wrapper transform (placement), optionally
//! attaches an [`AnimationPlayer`], and appends its skinned primitives to the frame
//! loop's skin-cache list (`gltf_skinned`) so the existing
//! `advance_animation`/`update_palettes`/`patch_scene` path drives them.
//!
//! Opt-in (`SPONZA_CHARS=1`, default off → the level renders byte-unchanged). Used to
//! verify the ufbx FBX importer + GPU skin cache against Intel Sponza: an **animated
//! skinned VoxelCharacter** (glTF) plus the **static knight** (FBX geometry — its FBX
//! carries no skin weights, so it renders its bind pose; see docs/phase-13-fbx-knight.md).

use dreamcoast_core::glam::{Quat, Vec3};
use dreamcoast_scene::{AnimationClip, AnimationPlayer, LocalTransform, Name, Parent, World};
use rhi::{Device, Texture};
use tracing::{info, warn};

use dreamcoast_asset::GltfScene;

use crate::registry::{MaterialRegistry, MeshRegistry, upload_gltf_scene};
use crate::skin::{self, SkinnedMesh};

/// Where + how big a character stands in the level (world metres; `scale` also converts
/// a source authored in cm — e.g. the knight FBX — to metres).
pub(crate) struct Placement {
    pub translation: Vec3,
    pub rotation_y_deg: f32,
    pub scale: f32,
}

impl Placement {
    /// Parse `"x,y,z,rotDeg,scale"` from an env override, falling back to `self`.
    fn with_env(mut self, var: &str) -> Self {
        if let Ok(s) = std::env::var(var) {
            let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            if v.len() == 5 {
                self.translation = Vec3::new(v[0], v[1], v[2]);
                self.rotation_y_deg = v[3];
                self.scale = v[4];
            } else {
                warn!("character: {var} must be 'x,y,z,rotDeg,scale' — ignoring '{s}'");
            }
        }
        self
    }
}

/// Instantiate `gscene` into the level's world/registries, place it, optionally play
/// animation clip `anim`, and append any skinned primitives to `skinned`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn overlay(
    device: &Device,
    world: &mut World,
    meshes: &mut MeshRegistry,
    materials: &mut MaterialRegistry,
    textures: &mut Vec<Texture>,
    gscene: &GltfScene,
    place: &Placement,
    anim: Option<usize>,
    label: &str,
    skinned: &mut Vec<SkinnedMesh>,
) -> anyhow::Result<()> {
    let handles = upload_gltf_scene(device, gscene, meshes, materials, textures)?;
    let (imported, node_map) = dreamcoast_scene::instantiate_gltf_mapped(world, gscene, &handles);

    // Wrapper root carrying the world placement; the whole import (incl. the joint nodes)
    // inherits it via `propagate_transforms`, so a skinned character's world-space skinned
    // vertices — and a static one's drawables — land at this transform.
    let root = world.spawn();
    world.insert(
        root,
        LocalTransform {
            translation: place.translation,
            rotation: Quat::from_rotation_y(place.rotation_y_deg.to_radians()),
            scale: Vec3::splat(place.scale),
        },
    );
    world.insert(root, Name(format!("character:{label}")));
    world.insert(imported, Parent(root));

    if let Some(idx) = anim {
        match gscene.animations.get(idx) {
            Some(a) => {
                let clip = AnimationClip::from_gltf(a, &node_map);
                if clip.is_empty() {
                    info!("character '{label}': anim {idx} has no channels for this rig");
                } else {
                    let dur = clip.duration;
                    let player = world.spawn();
                    world.insert(player, AnimationPlayer::new(clip));
                    info!("character '{label}': playing anim {idx} ({dur:.2}s)");
                }
            }
            None => warn!(
                "character '{label}': no anim {idx} ({} available)",
                gscene.animations.len()
            ),
        }
    }

    let mut sk = skin::build_skinned_meshes(device, gscene, &handles, &node_map, meshes)?;
    let n = sk.len();
    skinned.append(&mut sk);
    info!("character '{label}': {n} skinned primitive(s)");
    Ok(())
}

/// The default placements for the Intel Sponza verification scene (camera at ~+X looking
/// down the −X nave): the animated VoxelCharacter and the static knight stand a few metres
/// ahead on the floor, facing the camera. Tunable via `CHAR_VOXEL` / `CHAR_KNIGHT`
/// (`"x,y,z,rotDeg,scale"`).
pub(crate) fn voxel_placement() -> Placement {
    Placement {
        translation: Vec3::new(2.5, 0.0, -0.6),
        rotation_y_deg: -90.0,
        scale: 1.0, // glb authored in metres (~1.4 m)
    }
    .with_env("CHAR_VOXEL")
}

#[cfg(feature = "fbx")]
pub(crate) fn knight_placement() -> Placement {
    Placement {
        translation: Vec3::new(1.5, 0.0, 1.2),
        rotation_y_deg: -90.0,
        // The FBX importer bakes units into a static mesh's geometry (ModifyGeometry), so
        // the knight already loads in metres (~1.93 m) — no cm→m scale needed here.
        scale: 1.0,
    }
    .with_env("CHAR_KNIGHT")
}
