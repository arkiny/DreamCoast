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

use std::collections::HashMap;
use std::rc::Rc;

use dreamcoast_asset::{GltfScene, Material, MaterialKind, MeshData, MeshVertex};
use dreamcoast_core::glam::{Mat4, Quat, Vec3};
use dreamcoast_scene::{MaterialHandle, MeshHandle, World};
use rhi::{Buffer, Device, Format, Texture};

use crate::mesh::{upload_geometry, upload_mesh, upload_texture};
use crate::{NO_TEXTURE, SceneObject};

/// An uploaded mesh: vertex/index buffers + their counts.
pub(crate) struct GpuMesh {
    pub(crate) vbuf: Buffer,
    pub(crate) ibuf: Buffer,
    pub(crate) index_count: u32,
    pub(crate) vertex_count: u32,
    /// Local-space AABB (min, max) over the mesh vertices, computed once at upload. Frustum
    /// culling transforms these 8 corners by the instance transform to get a world AABB (cheap
    /// per frame vs re-scanning vertices). Empty mesh ⇒ a degenerate zero AABB (never culled).
    pub(crate) local_aabb: [[f32; 3]; 2],
}

/// The CPU-side geometry kept alongside each [`GpuMesh`] so the scalable-GI fuse
/// ([`crate::fuse`]) can rebuild the world-space triangle soup straight from the
/// draw list. Sponza-scale memory (~10 MB) is accepted — the GDF bake needs it.
pub(crate) struct MeshCpu {
    pub(crate) vertices: Vec<MeshVertex>,
    pub(crate) indices: Vec<u32>,
}

/// Owns every uploaded [`GpuMesh`], addressed by [`MeshHandle`]. The parallel `cpu`
/// vector holds each mesh's CPU geometry in the same index order (the fuse reads it).
#[derive(Default)]
pub(crate) struct MeshRegistry {
    meshes: Vec<Rc<GpuMesh>>,
    cpu: Vec<MeshCpu>,
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
        Ok(self.push(
            vbuf,
            ibuf,
            index_count,
            mesh.vertices.clone(),
            mesh.indices.clone(),
        ))
    }

    /// Upload raw vertex/index slices (a glTF primitive) and return its handle.
    pub(crate) fn upload_geometry(
        &mut self,
        device: &Device,
        vertices: &[MeshVertex],
        indices: &[u32],
    ) -> anyhow::Result<MeshHandle> {
        let (vbuf, ibuf, index_count) = upload_geometry(device, vertices, indices)?;
        Ok(self.push(vbuf, ibuf, index_count, vertices.to_vec(), indices.to_vec()))
    }

    fn push(
        &mut self,
        vbuf: Buffer,
        ibuf: Buffer,
        index_count: u32,
        vertices: Vec<MeshVertex>,
        indices: Vec<u32>,
    ) -> MeshHandle {
        let handle = MeshHandle(self.meshes.len() as u32);
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for v in &vertices {
            for i in 0..3 {
                mn[i] = mn[i].min(v.pos[i]);
                mx[i] = mx[i].max(v.pos[i]);
            }
        }
        // Degenerate (empty mesh) ⇒ min > max ⇒ the frustum test treats it as always-visible.
        let local_aabb = if vertices.is_empty() {
            [[0.0; 3], [0.0; 3]]
        } else {
            [mn, mx]
        };
        self.meshes.push(Rc::new(GpuMesh {
            vbuf,
            ibuf,
            index_count,
            vertex_count: vertices.len() as u32,
            local_aabb,
        }));
        self.cpu.push(MeshCpu { vertices, indices });
        handle
    }

    /// Resolve a handle to a shared [`GpuMesh`] (cheap `Rc` clone).
    pub(crate) fn get(&self, handle: MeshHandle) -> Rc<GpuMesh> {
        self.meshes[handle.0 as usize].clone()
    }

    /// Resolve a handle to its CPU geometry (the fuse's single geometry source).
    pub(crate) fn cpu(&self, handle: MeshHandle) -> &MeshCpu {
        &self.cpu[handle.0 as usize]
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
    /// Alpha-test cutoff for `alphaMode: MASK` materials; `0.0` = opaque (no test). The single
    /// value driving both the G-buffer cutout and the masked-shadow alpha test (see the
    /// `gbuffer`/`shadow` passes), packed into the push constants' spare `mr_factor.w` slot.
    pub(crate) alpha_cutoff: f32,
    /// Representative linear albedo for the GDF/SW-RT GI: the base-color texture's
    /// linear average × factor, or the factor's RGB when untextured (see
    /// [`representative_albedo`]). The single source the fuse tags onto every triangle.
    pub(crate) albedo: [f32; 3],
    /// Renderer routing tag (classified once at glTF import). `Opaque` for procedural/level
    /// materials; only glTF imports can be `Decal`/`Transparent`. Drives the deferred-decal
    /// pass split (decals modify the G-buffer's albedo instead of overwriting it as opaque).
    pub(crate) kind: MaterialKind,
}

/// A material's representative linear albedo for the GDF/GI bake. `tex_average` is the
/// base-color texture's linear-space average (`None` when untextured); the result is
/// that average modulated by the base-color factor, falling back to the factor's RGB.
/// One definition so every site (gallery, glTF, level) derives the same value.
pub(crate) fn representative_albedo(tex_average: Option<[f32; 3]>, factor: [f32; 4]) -> [f32; 3] {
    match tex_average {
        Some(a) => [a[0] * factor[0], a[1] * factor[1], a[2] * factor[2]],
        None => [factor[0], factor[1], factor[2]],
    }
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
            let mesh = meshes.get(d.mesh);
            // World AABB = the 8 local-AABB corners transformed by the instance matrix, reduced to
            // min/max. Cheap (8 points) and conservative — a frustum-vs-AABB test on it never culls a
            // visible object (only fully-outside ones), so culling stays image-identical.
            let [lmn, lmx] = mesh.local_aabb;
            let mut wmn = dreamcoast_core::glam::Vec3::splat(f32::INFINITY);
            let mut wmx = dreamcoast_core::glam::Vec3::splat(f32::NEG_INFINITY);
            for cx in [lmn[0], lmx[0]] {
                for cy in [lmn[1], lmx[1]] {
                    for cz in [lmn[2], lmx[2]] {
                        let w = d.world.transform_point3(dreamcoast_core::glam::vec3(cx, cy, cz));
                        wmn = wmn.min(w);
                        wmx = wmx.max(w);
                    }
                }
            }
            SceneObject {
                mesh,
                transform: d.world,
                world_aabb: [wmn, wmx],
                base_color: mat.base_color,
                metallic: mat.metallic,
                roughness: mat.roughness,
                tex: mat.tex,
                alpha_cutoff: mat.alpha_cutoff,
                kind: mat.kind,
                casts_shadow: d.casts_shadow,
                skin: None,
                morph: None,
            }
        })
        .collect()
}

/// Per glTF mesh index, the `(mesh, material)` handle of each of its primitives.
pub(crate) type PrimitiveHandles = Vec<Vec<(MeshHandle, MaterialHandle)>>;

/// Upload a whole imported [`GltfScene`] into the registries (Phase 12 Stage B, the
/// P2 multi-material generalization): every primitive's geometry → a `MeshHandle`,
/// every material → a `MaterialHandle` (with shared images deduplicated by index and
/// colour space). Returns, per glTF mesh index, the `(mesh, material)` handle of each
/// primitive.
pub(crate) fn upload_gltf_scene(
    device: &Device,
    scene: &GltfScene,
    meshes: &mut MeshRegistry,
    materials: &mut MaterialRegistry,
    textures: &mut Vec<Texture>,
) -> anyhow::Result<PrimitiveHandles> {
    // The scene's texture table is already cooked (block-compressed where eligible) by
    // `cook::load_or_cook_gltf_scene`, so upload just pushes each `TexData` to the GPU —
    // no per-slot compression here. Dedup by (image index, sRGB) so a shared texture
    // uploads once; `srgb` only selects the RGBA8 format (BC data carries its own).
    let mut image_cache: HashMap<(usize, bool), u32> = HashMap::new();
    let mut resolve =
        |textures: &mut Vec<Texture>, slot: Option<usize>, srgb: bool| -> anyhow::Result<u32> {
            let Some(idx) = slot else {
                return Ok(NO_TEXTURE);
            };
            if let Some(&bindless) = image_cache.get(&(idx, srgb)) {
                return Ok(bindless);
            }
            let format = if srgb {
                Format::Rgba8Srgb
            } else {
                Format::Rgba8Unorm
            };
            let bindless = upload_texture(device, textures, &scene.images[idx], format)?;
            image_cache.insert((idx, srgb), bindless);
            Ok(bindless)
        };

    // Materials: base-color / emissive are sRGB; metallic-roughness / normal linear.
    let mut material_handles: Vec<MaterialHandle> = Vec::with_capacity(scene.materials.len());
    for m in &scene.materials {
        let tex = [
            resolve(textures, m.base_color, true)?,
            resolve(textures, m.metallic_roughness, false)?,
            resolve(textures, m.normal, false)?,
            resolve(textures, m.emissive, true)?,
        ];
        // Representative albedo from the base-color image's linear average × factor
        // (the GDF/GI single source); untextured → the factor's RGB.
        let albedo = representative_albedo(
            m.base_color.map(|i| scene.images[i].average_linear()),
            m.base_color_factor,
        );
        material_handles.push(materials.add(MaterialDesc {
            base_color: m.base_color_factor,
            metallic: m.metallic_factor,
            roughness: m.roughness_factor,
            tex,
            albedo,
            alpha_cutoff: m.alpha_cutoff,
            kind: m.kind,
        }));
    }
    // Fallback for primitives with no material (glTF default material).
    let default_material = materials.add(MaterialDesc {
        base_color: Material::default().base_color_factor,
        metallic: Material::default().metallic_factor,
        roughness: Material::default().roughness_factor,
        tex: [NO_TEXTURE; 4],
        albedo: representative_albedo(None, Material::default().base_color_factor),
        alpha_cutoff: 0.0,
        kind: MaterialKind::Opaque,
    });

    let mut per_mesh: Vec<Vec<(MeshHandle, MaterialHandle)>> =
        Vec::with_capacity(scene.meshes.len());
    for primitives in &scene.meshes {
        let mut row = Vec::with_capacity(primitives.len());
        for prim in primitives {
            let mesh = meshes.upload_geometry(device, &prim.vertices, &prim.indices)?;
            let material = prim
                .material
                .map(|i| material_handles[i])
                .unwrap_or(default_material);
            row.push((mesh, material));
        }
        per_mesh.push(row);
    }
    Ok(per_mesh)
}

/// The world-space AABB of an imported glTF scene at its **native** (authored) scale,
/// walking the node hierarchy with accumulated transforms. Returns `None` if the scene
/// has no geometry. The engine treats glTF native units as **metres** (1 unit = 1 m) —
/// assets are placed at this scale, not rescaled, so a building stays building-sized.
pub(crate) fn gltf_bounds(scene: &GltfScene) -> Option<(Vec3, Vec3)> {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    let mut stack: Vec<(usize, Mat4)> = scene.roots.iter().map(|&r| (r, Mat4::IDENTITY)).collect();
    while let Some((idx, parent)) = stack.pop() {
        let n = &scene.nodes[idx];
        let local = Mat4::from_scale_rotation_translation(
            Vec3::from(n.scale),
            Quat::from_array(n.rotation),
            Vec3::from(n.translation),
        );
        let world = parent * local;
        if let Some(mesh_idx) = n.mesh {
            for prim in &scene.meshes[mesh_idx] {
                for v in &prim.vertices {
                    let p = world.transform_point3(Vec3::from(v.pos));
                    min = min.min(p);
                    max = max.max(p);
                }
            }
        }
        for &child in &n.children {
            stack.push((child, world));
        }
    }
    (min.x <= max.x).then_some((min, max))
}
