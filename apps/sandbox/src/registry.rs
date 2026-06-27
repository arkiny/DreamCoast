//! Renderer-side registries that back the scene's opaque handles (Phase 12 P2).
//!
//! `dreamcoast-scene` references geometry/materials by `MeshHandle` / `MaterialHandle`
//! (plain indices, no GPU types). These registries own the actual GPU mesh buffers
//! and material descriptors those handles point at, and [`build_scene`] resolves a
//! `World`'s draw list into the renderer's [`SceneObject`] draw records.
//!
//! Meshes are shared via `Rc` so multiple instances of the same geometry (e.g. two
//! spheres) upload once and each `SceneObject` just holds a cheap clone — the P2
//! generalization the Stage B glTF importer reuses for N-primitive / M-material
//! assets.

use std::rc::Rc;

use dreamcoast_asset::MeshData;
use dreamcoast_scene::{MaterialHandle, MeshHandle, World};
use rhi::{Buffer, Device};

use crate::SceneObject;
use crate::mesh::upload_mesh;

/// An uploaded mesh: vertex/index buffers + their counts.
pub(crate) struct GpuMesh {
    pub(crate) vbuf: Buffer,
    pub(crate) ibuf: Buffer,
    pub(crate) index_count: u32,
    pub(crate) vertex_count: u32,
}

/// Owns every uploaded [`GpuMesh`], addressed by [`MeshHandle`].
#[derive(Default)]
pub(crate) struct MeshRegistry {
    meshes: Vec<Rc<GpuMesh>>,
}

impl MeshRegistry {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Upload `mesh` to the GPU and return its handle. Each call is one upload; call
    /// once per unique geometry and reuse the handle across instances.
    pub(crate) fn upload(
        &mut self,
        device: &Device,
        mesh: &MeshData,
    ) -> anyhow::Result<MeshHandle> {
        let (vbuf, ibuf, index_count) = upload_mesh(device, mesh)?;
        let handle = MeshHandle(self.meshes.len() as u32);
        self.meshes.push(Rc::new(GpuMesh {
            vbuf,
            ibuf,
            index_count,
            vertex_count: mesh.vertices.len() as u32,
        }));
        Ok(handle)
    }

    /// Resolve a handle to a shared [`GpuMesh`] (cheap `Rc` clone).
    pub(crate) fn get(&self, handle: MeshHandle) -> Rc<GpuMesh> {
        self.meshes[handle.0 as usize].clone()
    }
}

/// A material's shading parameters (mirrors the glTF metallic-roughness model). `tex`
/// holds bindless indices for base-color / metallic-roughness / normal / emissive
/// (`NO_TEXTURE` if absent).
#[derive(Clone, Copy)]
pub(crate) struct MaterialDesc {
    pub(crate) base_color: [f32; 4],
    pub(crate) metallic: f32,
    pub(crate) roughness: f32,
    pub(crate) tex: [u32; 4],
}

/// Owns every material descriptor, addressed by [`MaterialHandle`].
#[derive(Default)]
pub(crate) struct MaterialRegistry {
    materials: Vec<MaterialDesc>,
}

impl MaterialRegistry {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Register a material and return its handle.
    pub(crate) fn add(&mut self, desc: MaterialDesc) -> MaterialHandle {
        let handle = MaterialHandle(self.materials.len() as u32);
        self.materials.push(desc);
        handle
    }

    /// Resolve a handle to its descriptor.
    pub(crate) fn get(&self, handle: MaterialHandle) -> &MaterialDesc {
        &self.materials[handle.0 as usize]
    }
}

/// Materialize a `World`'s draw list into the renderer's [`SceneObject`] records by
/// resolving each drawable's mesh/material handles against the registries. Order
/// matches the draw list (deterministic), so the produced list is stable frame to
/// frame and across the rasterizer / RT / GDF consumers.
pub(crate) fn build_scene(
    world: &World,
    meshes: &MeshRegistry,
    materials: &MaterialRegistry,
) -> Vec<SceneObject> {
    world
        .draw_list()
        .into_iter()
        .map(|d| {
            let mat = materials.get(d.material);
            SceneObject {
                mesh: meshes.get(d.mesh),
                transform: d.world,
                base_color: mat.base_color,
                metallic: mat.metallic,
                roughness: mat.roughness,
                tex: mat.tex,
                casts_shadow: d.casts_shadow,
            }
        })
        .collect()
}
