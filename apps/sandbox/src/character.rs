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

// ============================================================================
// Alembic vertex-cache playback (docs/alembic-usd-import.md, A4).
//
// The Intel Sponza knight animation is a baked vertex cache (its FBX has no skin
// weights). `read_vertex_cache` decodes it to per-frame positions per mesh part; this
// plays it back by updating each mesh's host-visible vertex buffer every frame — no new
// shader/RHI, the normal static mesh pipeline draws the deformed geometry. Deterministic
// (CPU-driven from the fixed-timestep clock) so headless captures reproduce. NOTE: a
// single VB per mesh is rewritten each frame; correct for the deterministic screenshot,
// real-time playback should double-buffer per frame-in-flight (documented follow-up).
// ============================================================================

use std::rc::Rc;

use dreamcoast_asset::alembic::{VcMesh, VertexCache};
use dreamcoast_asset::{Material, MeshData, MeshVertex};
use dreamcoast_scene::MeshInstance;

use crate::registry::{GpuMesh, MaterialDesc, representative_albedo};

/// Plays a decoded Alembic [`VertexCache`]: holds the cache + each part's GPU mesh, and
/// rewrites their vertex buffers to the current frame each tick.
pub(crate) struct VertexCachePlayer {
    cache: VertexCache,
    /// GPU mesh per `cache.meshes` entry (parallel); `None` for an empty/degenerate part.
    gpu: Vec<Option<Rc<GpuMesh>>>,
    time: f32,
}

/// Per-vertex normals from a position array + triangle indices (area-weighted face
/// normals accumulated then normalized) — recomputed each frame so the deforming surface
/// shades correctly.
fn compute_normals(pos: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut n = vec![[0f32; 3]; pos.len()];
    for t in indices.chunks_exact(3) {
        let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cr = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        for &i in t {
            let m = &mut n[i as usize];
            m[0] += cr[0];
            m[1] += cr[1];
            m[2] += cr[2];
        }
    }
    for v in &mut n {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if l > 1e-8 {
            *v = [v[0] / l, v[1] / l, v[2] / l];
        } else {
            *v = [0.0, 1.0, 0.0];
        }
    }
    n
}

/// Build a mesh part's `MeshVertex` list for one frame (positions + recomputed normals,
/// no UVs — the cache carries none).
fn frame_vertices(m: &VcMesh, frame: usize) -> Vec<MeshVertex> {
    let f = frame.min(m.frames.len().saturating_sub(1));
    let pos = &m.frames[f];
    let nrm = compute_normals(pos, &m.indices);
    pos.iter()
        .zip(&nrm)
        .map(|(p, n)| MeshVertex {
            pos: *p,
            normal: *n,
            uv: [0.0, 0.0],
        })
        .collect()
}

/// `MeshVertex` slice → tightly-packed little-endian bytes (matches the 32-byte layout the
/// vertex buffer expects).
fn vertex_bytes(verts: &[MeshVertex]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(verts.len() * 32);
    for v in verts {
        for f in v.pos.iter().chain(&v.normal).chain(&v.uv) {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
    }
    bytes
}

impl VertexCachePlayer {
    /// Advance the clock by `dt` and rewrite every part's vertex buffer to the new frame.
    pub(crate) fn update(&mut self, dt: f32) -> anyhow::Result<()> {
        if self.cache.num_frames == 0 {
            return Ok(());
        }
        self.time += dt;
        let frame = ((self.time * self.cache.fps) as usize) % self.cache.num_frames;
        for (m, g) in self.cache.meshes.iter().zip(&self.gpu) {
            let Some(g) = g else { continue };
            if m.frames.is_empty() {
                continue;
            }
            g.vbuf.write(&vertex_bytes(&frame_vertices(m, frame)))?;
        }
        Ok(())
    }
}

/// Overlay a decoded Alembic vertex cache on the level: upload each part's frame-0
/// geometry, spawn a drawable per part under a placement wrapper, and return the player
/// that animates them. The cache's parts are pre-assembled in one metre/Y-up space, so a
/// single wrapper transform places the whole character.
#[allow(clippy::too_many_arguments)]
pub(crate) fn overlay_vcache(
    device: &Device,
    world: &mut World,
    meshes: &mut MeshRegistry,
    materials: &mut MaterialRegistry,
    cache: VertexCache,
    place: &Placement,
) -> anyhow::Result<VertexCachePlayer> {
    let root = world.spawn();
    world.insert(
        root,
        LocalTransform {
            translation: place.translation,
            rotation: Quat::from_rotation_y(place.rotation_y_deg.to_radians()),
            scale: Vec3::splat(place.scale),
        },
    );
    world.insert(root, Name("vcache-knight".to_owned()));

    // One shared brushed-metal material for the whole knight (the cache carries no textures).
    let base = [0.58, 0.58, 0.60, 1.0];
    let material = materials.add(MaterialDesc {
        base_color: base,
        metallic: 0.6,
        roughness: 0.45,
        tex: [crate::NO_TEXTURE; 4],
        albedo: representative_albedo(None, base),
        alpha_cutoff: 0.0,
        kind: dreamcoast_asset::MaterialKind::Opaque,
        two_sided: true,
    });

    let mut gpu = Vec::with_capacity(cache.meshes.len());
    let mut parts = 0usize;
    for m in &cache.meshes {
        if m.frames.is_empty() || m.indices.is_empty() {
            gpu.push(None);
            continue;
        }
        let md = MeshData {
            vertices: frame_vertices(m, 0),
            indices: m.indices.clone(),
            material: Material::default(),
        };
        let handle = meshes.upload(device, &md)?;
        gpu.push(Some(meshes.get(handle)));
        let e = world.spawn();
        world.insert(e, LocalTransform::IDENTITY);
        world.insert(e, MeshInstance::new(handle, material));
        world.insert(e, Parent(root));
        parts += 1;
    }
    info!(
        "vcache knight: {parts} parts, {} frames @ {} fps",
        cache.num_frames, cache.fps
    );
    // `ABC_START_S` seeds the playback clock (seconds) so headless captures can sample
    // different animation phases (the capture is otherwise deterministic at one frame).
    let time = std::env::var("ABC_START_S")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0);
    Ok(VertexCachePlayer { cache, gpu, time })
}

/// Default placement for the Alembic knight in Intel Sponza (metres, feet at y=0). Tunable
/// via `CHAR_KNIGHT_ABC` (`"x,y,z,rotDeg,scale"`).
pub(crate) fn knight_abc_placement() -> Placement {
    Placement {
        translation: Vec3::new(3.5, 0.0, 0.0),
        rotation_y_deg: 90.0,
        scale: 1.0,
    }
    .with_env("CHAR_KNIGHT_ABC")
}
