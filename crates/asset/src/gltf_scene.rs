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

/// One node in the glTF hierarchy. `children`/`mesh` are indices into
/// [`GltfScene::nodes`] / [`GltfScene::meshes`]. Transform is TRS (rotation is the
/// `[x, y, z, w]` quaternion).
pub struct GltfNode {
    pub name: Option<String>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub children: Vec<usize>,
    pub mesh: Option<usize>,
}

/// One primitive: geometry + an optional material index (into [`GltfScene::materials`]).
pub struct GltfPrimitive {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub material: Option<usize>,
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
}

/// A whole imported glTF scene: node hierarchy + per-mesh primitives + materials +
/// shared images. `roots` are the top-level nodes of the default scene.
pub struct GltfScene {
    pub nodes: Vec<GltfNode>,
    pub roots: Vec<usize>,
    /// Indexed by glTF mesh index; each entry is that mesh's primitives.
    pub meshes: Vec<Vec<GltfPrimitive>>,
    pub materials: Vec<GltfMaterial>,
    pub images: Vec<ImageData>,
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
            GltfMaterial {
                base_color_factor: pbr.base_color_factor(),
                metallic_factor: pbr.metallic_factor(),
                roughness_factor: pbr.roughness_factor(),
                emissive_factor: m.emissive_factor(),
                base_color: src(pbr.base_color_texture()),
                metallic_roughness: src(pbr.metallic_roughness_texture()),
                normal: m.normal_texture().map(|n| n.texture().source().index()),
                emissive: src(m.emissive_texture()),
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
            }
        })
        .collect();

    let roots = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .map(|s| s.nodes().map(|n| n.index()).collect())
        .unwrap_or_default();

    Ok(GltfScene {
        nodes,
        roots,
        meshes,
        materials,
        images,
    })
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

    GltfPrimitive {
        vertices,
        indices,
        material: prim.material().index(),
    }
}
