//! GPU vertex skinning (animation Stage B.2 — vertex pulling).
//!
//! The bind-pose vertex buffer is uploaded once (the mesh registry already holds it);
//! per frame only a small joint **palette** (`joint_world × inverse_bind`, one
//! `float4x4` per joint) is rebuilt from the ECS joint transforms and written into a
//! per-frame-in-flight storage buffer. The skinning vertex shader (`gbuffer.slang`
//! `vsMainSkinned`) reads each vertex's joints/weights + the palette from bindless
//! storage buffers (by `SV_VertexID`) and deforms the bind pose — so no per-frame
//! vertex re-upload and no change to the vertex layout / pipelines beyond a second
//! vertex entry point. Skinned vertices land in scene/world space, so a skinned
//! drawable's model matrix is the identity (its glTF mesh-node transform is ignored,
//! per the glTF spec).

use std::rc::Rc;

use dreamcoast_asset::GltfScene;
use dreamcoast_core::glam::Mat4;
use dreamcoast_scene::{Entity, MaterialHandle, MeshHandle, World, WorldTransform};
use rhi::{Device, StorageBuffer, StorageBufferDesc};

use crate::FRAMES_IN_FLIGHT;
use crate::registry::{GpuMesh, MeshRegistry};

/// One skinned primitive: the static per-vertex joints/weights storage buffers, a
/// per-fif joint-palette ring, and the skin binding needed to rebuild the palette.
pub(crate) struct SkinnedMesh {
    /// The bind-pose mesh `build_scene` references; drawables are matched to it by
    /// `Rc` identity (the draw keeps using this bind-pose vertex buffer — the vertex
    /// shader skins it).
    base_mesh: Rc<GpuMesh>,
    /// Bindless indices of the static per-vertex joints (`uint4`) / weights (`float4`).
    joints_idx: u32,
    weights_idx: u32,
    joint_count: u32,
    /// The skin's joint entities (parallel to `inverse_bind`), for the per-frame palette.
    joint_entities: Vec<Entity>,
    inverse_bind: Vec<Mat4>,
    /// Per-fif palette storage buffers + their bindless indices; written each frame.
    palette_bufs: Vec<StorageBuffer>,
    palette_idx: Vec<u32>,
    /// Reused per-frame palette byte scratch (`joint_count × 64`).
    palette_scratch: Vec<u8>,
    // Keep the static buffers resident.
    _joints_buf: StorageBuffer,
    _weights_buf: StorageBuffer,
}

fn storage_desc(size: usize, stride: u32) -> StorageBufferDesc {
    StorageBufferDesc {
        size: size as u64,
        stride,
        indirect: false,
    }
}

/// Build the GPU-skinning table for an imported glTF scene: one [`SkinnedMesh`] per
/// (skinned node, primitive-with-weights). `node_map` is the node-index → entity map
/// from `instantiate_gltf_mapped`; `prim_handles` the per-primitive renderer handles.
pub(crate) fn build_skinned_meshes(
    device: &Device,
    gscene: &GltfScene,
    prim_handles: &[Vec<(MeshHandle, MaterialHandle)>],
    node_map: &[Option<Entity>],
    registry: &MeshRegistry,
) -> anyhow::Result<Vec<SkinnedMesh>> {
    let mut skinned = Vec::new();
    for node in &gscene.nodes {
        let (Some(mesh_idx), Some(skin_idx)) = (node.mesh, node.skin) else {
            continue;
        };
        let skin = &gscene.skins[skin_idx];
        let joint_entities: Vec<Entity> = skin
            .joints
            .iter()
            .filter_map(|&n| node_map.get(n).copied().flatten())
            .collect();
        if joint_entities.len() != skin.joints.len() {
            continue; // a joint node was not instantiated — skip this skin
        }
        let inverse_bind: Vec<Mat4> = (0..skin.joints.len())
            .map(|i| {
                skin.inverse_bind
                    .get(i)
                    .map(Mat4::from_cols_array)
                    .unwrap_or(Mat4::IDENTITY)
            })
            .collect();
        let joint_count = joint_entities.len() as u32;

        for (prim_idx, prim) in gscene.meshes[mesh_idx].iter().enumerate() {
            let (Some(joints), Some(weights)) = (prim.joints.as_ref(), prim.weights.as_ref())
            else {
                continue;
            };
            // Static per-vertex joints (4 × u32) + weights (4 × f32).
            let mut jbytes = Vec::with_capacity(joints.len() * 16);
            for jv in joints {
                for &j in jv {
                    jbytes.extend_from_slice(&(j as u32).to_le_bytes());
                }
            }
            let mut wbytes = Vec::with_capacity(weights.len() * 16);
            for wv in weights {
                for &w in wv {
                    wbytes.extend_from_slice(&w.to_le_bytes());
                }
            }
            let joints_buf =
                device.create_storage_buffer_init(&storage_desc(jbytes.len(), 16), &jbytes)?;
            let weights_buf =
                device.create_storage_buffer_init(&storage_desc(wbytes.len(), 16), &wbytes)?;

            // Per-fif palette ring (one float4x4 per joint), seeded to zero.
            let palette_size = joint_count as usize * 64;
            let zeros = vec![0u8; palette_size];
            let mut palette_bufs = Vec::with_capacity(FRAMES_IN_FLIGHT);
            let mut palette_idx = Vec::with_capacity(FRAMES_IN_FLIGHT);
            for _ in 0..FRAMES_IN_FLIGHT {
                let pb =
                    device.create_storage_buffer_init(&storage_desc(palette_size, 64), &zeros)?;
                palette_idx.push(pb.storage_index());
                palette_bufs.push(pb);
            }

            skinned.push(SkinnedMesh {
                base_mesh: registry.get(prim_handles[mesh_idx][prim_idx].0),
                joints_idx: joints_buf.storage_index(),
                weights_idx: weights_buf.storage_index(),
                joint_count,
                joint_entities: joint_entities.clone(),
                inverse_bind: inverse_bind.clone(),
                palette_bufs,
                palette_idx,
                palette_scratch: Vec::with_capacity(palette_size),
                _joints_buf: joints_buf,
                _weights_buf: weights_buf,
            });
        }
    }
    Ok(skinned)
}

/// Rebuild each skin's joint palette (`joint_world × inverse_bind`, column-major) from
/// the current ECS transforms and write it into this frame's palette buffer. Call
/// after the transforms propagate and after this slot's fence has been waited.
pub(crate) fn update_palettes(
    skinned: &mut [SkinnedMesh],
    world: &World,
    fif: usize,
) -> anyhow::Result<()> {
    for sk in skinned.iter_mut() {
        sk.palette_scratch.clear();
        for (&je, ib) in sk.joint_entities.iter().zip(&sk.inverse_bind) {
            let joint_world = world
                .get::<WorldTransform>(je)
                .map(|w| w.0)
                .unwrap_or(Mat4::IDENTITY);
            // Column-major bytes — matches the shader's explicit column-major M*v.
            for f in (joint_world * *ib).to_cols_array() {
                sk.palette_scratch.extend_from_slice(&f.to_le_bytes());
            }
        }
        sk.palette_bufs[fif].write(&sk.palette_scratch)?;
    }
    Ok(())
}

/// Tag each skinned drawable with this frame's skin indices + reset its model matrix
/// to the identity (skinned vertices are already in world space). The drawable keeps
/// its bind-pose mesh; the G-buffer pass draws it with the skinned pipeline.
pub(crate) fn patch_scene(skinned: &[SkinnedMesh], scene: &mut [crate::SceneObject], fif: usize) {
    for obj in scene.iter_mut() {
        if let Some(sk) = skinned.iter().find(|s| Rc::ptr_eq(&obj.mesh, &s.base_mesh)) {
            obj.skin = Some([
                sk.joints_idx,
                sk.weights_idx,
                sk.palette_idx[fif],
                sk.joint_count,
            ]);
            obj.transform = Mat4::IDENTITY;
        }
    }
}
