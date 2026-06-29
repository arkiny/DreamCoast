//! Full glTF scene import (Phase 12 Stage B).
//!
//! Unlike [`crate::load_gltf`] (which reads only the first primitive of the first
//! mesh), [`load_gltf_scene`] returns the **whole** node hierarchy, **every**
//! primitive of every mesh, all materials, and a shared, deduplicated image list —
//! the data the scene crate needs to instantiate a faithful entity sub-tree.
//!
//! RHI-agnostic: plain CPU data. Node transforms are kept as **TRS** (glTF's native
//! form) so the scene's `LocalTransform` is exact; images are referenced by index so
//! the renderer can upload each one once.

use std::path::Path;

use dreamcoast_core::EngineError;

use crate::{ImageData, MeshVertex, image_to_rgba8};

/// One node in the glTF hierarchy. `children`/`mesh`/`skin` are indices into
/// [`GltfScene::nodes`] / [`GltfScene::meshes`] / [`GltfScene::skins`]. Transform is
/// TRS (rotation is the `[x, y, z, w]` quaternion).
pub struct GltfNode {
    pub name: Option<String>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub children: Vec<usize>,
    pub mesh: Option<usize>,
    /// Index into [`GltfScene::skins`] if this node's mesh is skinned.
    pub skin: Option<usize>,
}

/// One primitive: geometry + an optional material index (into [`GltfScene::materials`]).
/// `joints`/`weights` (per-vertex, parallel to `vertices`) are present only on skinned
/// primitives; they are kept off the GPU [`MeshVertex`] (CPU-only skinning side data).
pub struct GltfPrimitive {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub material: Option<usize>,
    /// Per-vertex joint indices (into the owning node's skin's `joints`), 4 per vertex.
    pub joints: Option<Vec<[u16; 4]>>,
    /// Per-vertex skin weights, 4 per vertex (parallel to `joints`).
    pub weights: Option<Vec<[f32; 4]>>,
    /// Morph targets (per-vertex position/normal deltas), in target order; empty if
    /// the primitive has no morph targets. The animation's morph-weight channel
    /// (parallel order) blends them: `vertex += Σ wᵢ · targetᵢ.delta` (CPU side data).
    pub morph_targets: Vec<MorphTarget>,
}

/// One morph target's per-vertex deltas (parallel to a primitive's `vertices`).
/// `normals` is present only when the target supplies normal deltas.
pub struct MorphTarget {
    pub positions: Vec<[f32; 3]>,
    pub normals: Option<Vec<[f32; 3]>>,
}

/// A skin: the joint nodes it animates plus their inverse-bind matrices (column-major
/// `[f32; 16]`). `inverse_bind[i]` pairs with `joints[i]`; empty means identity per
/// joint (the glTF default when the accessor is omitted).
pub struct GltfSkin {
    pub joints: Vec<usize>,
    pub inverse_bind: Vec<[f32; 16]>,
}

/// glTF `alphaMode` — how a material's base-color alpha is interpreted. Preserved from
/// import (the single source) so the renderer can route OPAQUE/MASK/BLEND differently.
/// `MASK` carries the alpha-test cutoff; `BLEND` covers both surface decals and true
/// transparency (glTF has no decal flag — see [`classify_material`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}

/// How the renderer should treat a material, derived from [`AlphaMode`] + authoring hints.
/// glTF lacks a decal flag (decals and true transparents both author as `BLEND`), so this
/// is the engine's own routing tag: `Decal` modifies an already-lit surface's G-buffer
/// (albedo tint), `Transparent` is a real OVER-blended surface (track B, currently still
/// drawn as opaque). See [`classify_material`] — the one place this is decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MaterialKind {
    #[default]
    Opaque,
    Masked,
    Decal,
    Transparent,
}

/// Decide a material's [`MaterialKind`] from its `alphaMode` and name. **This is the single,
/// isolated home for decal/transparent classification** so the heuristic can be replaced
/// wholesale later (an authoring convention, a `KHR_*` extension, or scene metadata) without
/// touching the renderer. Short-term: a `BLEND` material whose name contains "decal" is a
/// surface decal (the Intel Sponza `dirt_decal` convention); any other `BLEND` is treated as
/// `Transparent` (track B — still opaque-fallback in the renderer for now).
pub fn classify_material(name: Option<&str>, alpha_mode: AlphaMode) -> MaterialKind {
    match alpha_mode {
        AlphaMode::Opaque => MaterialKind::Opaque,
        AlphaMode::Mask => MaterialKind::Masked,
        AlphaMode::Blend => {
            if name.is_some_and(|n| n.to_ascii_lowercase().contains("decal")) {
                MaterialKind::Decal
            } else {
                MaterialKind::Transparent
            }
        }
    }
}

/// A material with its texture slots referenced by **image index** (into
/// [`GltfScene::images`]) so shared images upload once.
pub struct GltfMaterial {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3],
    pub base_color: Option<usize>,
    pub metallic_roughness: Option<usize>,
    pub normal: Option<usize>,
    pub emissive: Option<usize>,
    /// Alpha-test cutoff for `alphaMode: MASK` materials (fragments with base-color alpha
    /// below this are discarded). `0.0` means no alpha test — `OPAQUE`/`BLEND` carry none.
    /// Single source for the renderer's masked cutout + masked shadows.
    pub alpha_cutoff: f32,
    /// glTF `alphaMode`, preserved from import (the single source for OPAQUE/MASK/BLEND routing).
    pub alpha_mode: AlphaMode,
    /// Engine routing tag derived once at import via [`classify_material`] (`alpha_mode` + name).
    pub kind: MaterialKind,
}

/// Keyframe interpolation mode for an animation sampler (glTF's three modes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    /// Hold the previous keyframe's value until the next.
    Step,
    /// Linear blend between adjacent keyframes (spherical for rotations).
    Linear,
    /// Cubic Hermite spline; outputs are `[in-tangent, value, out-tangent]` per key.
    CubicSpline,
}

/// One node-animation channel: the keyframe times plus the typed outputs for a
/// single TRS property of a single target node. For `CubicSpline` the output vector
/// has `3 * times.len()` entries (in-tangent, value, out-tangent per key).
pub struct GltfChannel {
    /// Index into [`GltfScene::nodes`] of the animated node.
    pub target_node: usize,
    pub interpolation: Interpolation,
    pub times: Vec<f32>,
    pub data: ChannelData,
}

/// The typed output samples of a [`GltfChannel`] (which property it drives).
pub enum ChannelData {
    Translation(Vec<[f32; 3]>),
    /// `[x, y, z, w]` quaternions.
    Rotation(Vec<[f32; 4]>),
    Scale(Vec<[f32; 3]>),
    /// Morph-target weights — `num_targets` values per keyframe, flattened
    /// (`times.len() × num_targets`), in target order (Stage C).
    Weights(Vec<f32>),
}

/// One animation clip: a set of node-TRS channels and its total duration (the
/// largest keyframe time across channels), in seconds.
pub struct GltfAnimation {
    pub name: Option<String>,
    pub channels: Vec<GltfChannel>,
    pub duration: f32,
}

/// A whole imported glTF scene: node hierarchy + per-mesh primitives + materials +
/// shared images + animation clips. `roots` are the top-level nodes of the default
/// scene.
pub struct GltfScene {
    pub nodes: Vec<GltfNode>,
    pub roots: Vec<usize>,
    /// Indexed by glTF mesh index; each entry is that mesh's primitives.
    pub meshes: Vec<Vec<GltfPrimitive>>,
    pub materials: Vec<GltfMaterial>,
    pub images: Vec<ImageData>,
    /// Animation clips (node TRS tracks); empty for a static scene.
    pub animations: Vec<GltfAnimation>,
    /// Skins (joint sets + inverse-bind matrices); empty for an unskinned scene.
    pub skins: Vec<GltfSkin>,
}

impl GltfScene {
    /// Total primitive count across all meshes (debug/logging).
    pub fn primitive_count(&self) -> usize {
        self.meshes.iter().map(Vec::len).sum()
    }
}

/// Load the entire glTF/GLB scene: every node, primitive, material, and image.
pub fn load_gltf_scene(path: impl AsRef<Path>) -> Result<GltfScene, EngineError> {
    let (doc, buffers, images_raw) =
        gltf::import(path).map_err(|e| EngineError::Asset(format!("gltf import: {e}")))?;

    let images: Vec<ImageData> = images_raw.iter().map(image_to_rgba8).collect();

    let materials: Vec<GltfMaterial> = doc
        .materials()
        .map(|m| {
            let pbr = m.pbr_metallic_roughness();
            let src =
                |info: Option<gltf::texture::Info>| info.map(|i| i.texture().source().index());
            // Preserve alphaMode (single source) and derive the engine routing tag once.
            let alpha_mode = match m.alpha_mode() {
                gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                gltf::material::AlphaMode::Mask => AlphaMode::Mask,
                gltf::material::AlphaMode::Blend => AlphaMode::Blend,
            };
            // Only MASK is alpha-tested; OPAQUE/BLEND carry no cutoff. glTF's default cutoff
            // is 0.5 when MASK omits `alphaCutoff`.
            let alpha_cutoff = match alpha_mode {
                AlphaMode::Mask => m.alpha_cutoff().unwrap_or(0.5),
                _ => 0.0,
            };
            let kind = classify_material(m.name(), alpha_mode);
            GltfMaterial {
                base_color_factor: pbr.base_color_factor(),
                metallic_factor: pbr.metallic_factor(),
                roughness_factor: pbr.roughness_factor(),
                emissive_factor: m.emissive_factor(),
                base_color: src(pbr.base_color_texture()),
                metallic_roughness: src(pbr.metallic_roughness_texture()),
                normal: m.normal_texture().map(|n| n.texture().source().index()),
                emissive: src(m.emissive_texture()),
                alpha_cutoff,
                alpha_mode,
                kind,
            }
        })
        .collect();

    let meshes: Vec<Vec<GltfPrimitive>> = doc
        .meshes()
        .map(|mesh| {
            mesh.primitives()
                .map(|prim| read_primitive(&prim, &buffers))
                .collect()
        })
        .collect();

    let nodes: Vec<GltfNode> = doc
        .nodes()
        .map(|n| {
            let (translation, rotation, scale) = n.transform().decomposed();
            GltfNode {
                name: n.name().map(str::to_owned),
                translation,
                rotation,
                scale,
                children: n.children().map(|c| c.index()).collect(),
                mesh: n.mesh().map(|m| m.index()),
                skin: n.skin().map(|s| s.index()),
            }
        })
        .collect();

    let skins: Vec<GltfSkin> = doc
        .skins()
        .map(|s| {
            let reader = s.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
            let inverse_bind = reader
                .read_inverse_bind_matrices()
                .map(|it| it.map(flatten_mat4).collect())
                .unwrap_or_default();
            GltfSkin {
                joints: s.joints().map(|j| j.index()).collect(),
                inverse_bind,
            }
        })
        .collect();

    let roots = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .map(|s| s.nodes().map(|n| n.index()).collect())
        .unwrap_or_default();

    let animations = doc
        .animations()
        .map(|anim| read_animation(&anim, &buffers))
        .collect();

    Ok(GltfScene {
        nodes,
        roots,
        meshes,
        materials,
        images,
        animations,
        skins,
    })
}

/// Flatten glTF's `[[f32; 4]; 4]` (column-major columns) into a flat `[f32; 16]`.
fn flatten_mat4(m: [[f32; 4]; 4]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for (c, col) in m.iter().enumerate() {
        out[c * 4..c * 4 + 4].copy_from_slice(col);
    }
    out
}

/// Read one animation clip's node-TRS channels. Morph-target-weight channels are
/// skipped (Stage C); duration is the largest keyframe time across channels.
fn read_animation(anim: &gltf::Animation, buffers: &[gltf::buffer::Data]) -> GltfAnimation {
    use gltf::animation::Interpolation as Gi;
    use gltf::animation::util::ReadOutputs;

    let mut duration = 0.0f32;
    let mut channels = Vec::new();
    for ch in anim.channels() {
        let interpolation = match ch.sampler().interpolation() {
            Gi::Step => Interpolation::Step,
            Gi::Linear => Interpolation::Linear,
            Gi::CubicSpline => Interpolation::CubicSpline,
        };
        let reader = ch.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
        let Some(times) = reader.read_inputs().map(|i| i.collect::<Vec<f32>>()) else {
            continue;
        };
        if let Some(&last) = times.last() {
            duration = duration.max(last);
        }
        let data = match reader.read_outputs() {
            Some(ReadOutputs::Translations(t)) => ChannelData::Translation(t.collect()),
            Some(ReadOutputs::Rotations(r)) => ChannelData::Rotation(r.into_f32().collect()),
            Some(ReadOutputs::Scales(s)) => ChannelData::Scale(s.collect()),
            Some(ReadOutputs::MorphTargetWeights(w)) => {
                ChannelData::Weights(w.into_f32().collect())
            }
            None => continue,
        };
        channels.push(GltfChannel {
            target_node: ch.target().node().index(),
            interpolation,
            times,
            data,
        });
    }
    GltfAnimation {
        name: anim.name().map(str::to_owned),
        channels,
        duration,
    }
}

/// Read one primitive's geometry (positions/normals/uvs/indices) into a
/// [`GltfPrimitive`]. Missing normals default to +Y, missing uvs to 0, missing
/// indices to a trivial 0..n sequence (matching [`crate::load_gltf`]).
fn read_primitive(prim: &gltf::Primitive, buffers: &[gltf::buffer::Data]) -> GltfPrimitive {
    let reader = prim.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .map(Iterator::collect)
        .unwrap_or_default();
    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map(Iterator::collect)
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|t| t.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
    let indices: Vec<u32> = reader
        .read_indices()
        .map(|i| i.into_u32().collect())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());

    let vertices = positions
        .iter()
        .enumerate()
        .map(|(i, p)| MeshVertex {
            pos: *p,
            normal: normals[i],
            uv: uvs[i],
        })
        .collect();

    // Skinning side data (CPU-only; off the GPU `MeshVertex`). Present only when the
    // primitive carries the JOINTS_0/WEIGHTS_0 attributes.
    let joints = reader.read_joints(0).map(|j| j.into_u16().collect());
    let weights = reader.read_weights(0).map(|w| w.into_f32().collect());

    // Morph targets: per-vertex position (+ optional normal) deltas, in target order.
    let morph_targets = reader
        .read_morph_targets()
        .map(|(pos, nrm, _tangent)| MorphTarget {
            positions: pos.map(Iterator::collect).unwrap_or_default(),
            normals: nrm.map(Iterator::collect),
        })
        .collect();

    GltfPrimitive {
        vertices,
        indices,
        material: prim.material().index(),
        joints,
        weights,
        morph_targets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_opaque_and_mask_ignore_name() {
        // OPAQUE/MASK never become decals regardless of name.
        assert_eq!(
            classify_material(None, AlphaMode::Opaque),
            MaterialKind::Opaque
        );
        assert_eq!(
            classify_material(Some("dirt_decal"), AlphaMode::Opaque),
            MaterialKind::Opaque
        );
        assert_eq!(
            classify_material(None, AlphaMode::Mask),
            MaterialKind::Masked
        );
        assert_eq!(
            classify_material(Some("foliage_decal"), AlphaMode::Mask),
            MaterialKind::Masked
        );
    }

    #[test]
    fn classify_blend_decal_by_name() {
        // BLEND + "decal" in the name → surface decal (case-insensitive, substring).
        assert_eq!(
            classify_material(Some("dirt_decal"), AlphaMode::Blend),
            MaterialKind::Decal
        );
        assert_eq!(
            classify_material(Some("Decal_Grime_01"), AlphaMode::Blend),
            MaterialKind::Decal
        );
    }

    #[test]
    fn classify_blend_without_decal_is_transparent() {
        // BLEND without the decal hint → transparent (track B; opaque-fallback for now).
        assert_eq!(
            classify_material(Some("glass"), AlphaMode::Blend),
            MaterialKind::Transparent
        );
        assert_eq!(
            classify_material(None, AlphaMode::Blend),
            MaterialKind::Transparent
        );
    }
}
