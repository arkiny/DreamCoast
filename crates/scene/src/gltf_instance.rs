//! Instantiating an imported glTF scene as an ECS entity sub-tree (Phase 12 Stage B).
//!
//! Each glTF node becomes one entity carrying the node's [`LocalTransform`] (+ `Name`
//! and `Parent` link), preserving the hierarchy so [`crate::propagate_transforms`]
//! moves a whole sub-tree when an ancestor moves. A node's mesh contributes a
//! [`MeshInstance`]: a single-primitive mesh attaches directly to the node entity; a
//! multi-primitive mesh (e.g. Sponza's one mesh of 25 material groups) spawns one
//! child entity per primitive so each gets its own material.

use dreamcoast_asset::GltfScene;
use glam::{Quat, Vec3};

use crate::components::{MaterialHandle, MeshHandle, MeshInstance, Name};
use crate::ecs::{Entity, World};
use crate::transform::{Children, LocalTransform, Parent};

/// Instantiate `scene` into `world`, returning the root entity of the new sub-tree
/// (a synthetic root if the glTF has multiple top-level nodes). `primitive_handles`
/// is indexed by glTF mesh index and gives the renderer-resolved (mesh, material)
/// handle of each of that mesh's primitives — the seam between this RHI-agnostic
/// crate and the renderer's registries.
pub fn instantiate_gltf(
    world: &mut World,
    scene: &GltfScene,
    primitive_handles: &[Vec<(MeshHandle, MaterialHandle)>],
) -> Entity {
    instantiate_gltf_mapped(world, scene, primitive_handles).0
}

/// Like [`instantiate_gltf`], but also returns the node-index → entity map (indexed
/// by glTF node index), so animation channels (which target node indices) can be
/// resolved to the entities created here.
pub fn instantiate_gltf_mapped(
    world: &mut World,
    scene: &GltfScene,
    primitive_handles: &[Vec<(MeshHandle, MaterialHandle)>],
) -> (Entity, Vec<Option<Entity>>) {
    let mut map = vec![None; scene.nodes.len()];
    let roots: Vec<Entity> = scene
        .roots
        .iter()
        .map(|&r| spawn_node(world, scene, primitive_handles, r, None, &mut map))
        .collect();

    let root = match roots.as_slice() {
        [single] => *single,
        _ => {
            // Wrap multiple top-level nodes under one transformable root.
            let root = world.spawn();
            world.insert(root, LocalTransform::IDENTITY);
            world.insert(root, Name("gltf-root".to_owned()));
            for &child in &roots {
                world.insert(child, Parent(root));
            }
            world.insert(root, Children(roots));
            root
        }
    };
    (root, map)
}

fn spawn_node(
    world: &mut World,
    scene: &GltfScene,
    primitive_handles: &[Vec<(MeshHandle, MaterialHandle)>],
    node_idx: usize,
    parent: Option<Entity>,
    map: &mut [Option<Entity>],
) -> Entity {
    let node = &scene.nodes[node_idx];
    let entity = world.spawn();
    map[node_idx] = Some(entity);
    world.insert(
        entity,
        LocalTransform {
            translation: Vec3::from(node.translation),
            rotation: Quat::from_array(node.rotation),
            scale: Vec3::from(node.scale),
        },
    );
    if let Some(name) = &node.name {
        world.insert(entity, Name(name.clone()));
    }
    if let Some(parent) = parent {
        link_child(world, parent, entity);
    }

    if let Some(mesh_idx) = node.mesh {
        let prims = &primitive_handles[mesh_idx];
        if let [(mesh, material)] = prims.as_slice() {
            // Single primitive: the node entity is the drawable.
            world.insert(entity, MeshInstance::new(*mesh, *material));
        } else {
            // Multiple primitives share the node's transform via a child each.
            for &(mesh, material) in prims {
                let prim_entity = world.spawn();
                world.insert(prim_entity, LocalTransform::IDENTITY);
                link_child(world, entity, prim_entity);
                world.insert(prim_entity, MeshInstance::new(mesh, material));
            }
        }
    }

    for &child in &node.children {
        spawn_node(world, scene, primitive_handles, child, Some(entity), map);
    }
    entity
}

/// Set `child`'s [`Parent`] and append it to `parent`'s [`Children`].
fn link_child(world: &mut World, parent: Entity, child: Entity) {
    world.insert(child, Parent(parent));
    match world.get_mut::<Children>(parent) {
        Some(children) => children.0.push(child),
        None => world.insert(parent, Children(vec![child])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{WorldTransform, propagate_transforms};
    use dreamcoast_asset::{GltfMaterial, GltfNode, GltfPrimitive, GltfScene};

    // A 2-node hierarchy: parent at x=10 with a 1-primitive mesh, child at y=5.
    fn two_node_scene() -> GltfScene {
        GltfScene {
            nodes: vec![
                GltfNode {
                    name: Some("parent".into()),
                    translation: [10.0, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                    children: vec![1],
                    mesh: Some(0),
                    skin: None,
                },
                GltfNode {
                    name: Some("child".into()),
                    translation: [0.0, 5.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                    children: vec![],
                    mesh: Some(0),
                    skin: None,
                },
            ],
            roots: vec![0],
            meshes: vec![vec![GltfPrimitive {
                vertices: vec![],
                indices: vec![],
                material: Some(0),
                joints: None,
                weights: None,
                morph_targets: vec![],
            }]],
            materials: vec![GltfMaterial {
                base_color_factor: [1.0; 4],
                metallic_factor: 0.0,
                roughness_factor: 1.0,
                emissive_factor: [0.0; 3],
                base_color: None,
                metallic_roughness: None,
                normal: None,
                emissive: None,
                alpha_cutoff: 0.0,
                alpha_mode: dreamcoast_asset::AlphaMode::Opaque,
                kind: dreamcoast_asset::MaterialKind::Opaque,
            }],
            images: vec![],
            animations: vec![],
            skins: vec![],
        }
    }

    #[test]
    fn preserves_hierarchy_and_propagates() {
        let mut world = World::new();
        let handles = vec![vec![(MeshHandle(0), MaterialHandle(0))]];
        let root = instantiate_gltf(&mut world, &two_node_scene(), &handles);
        assert_eq!(world.get::<Name>(root).unwrap().0, "parent");
        // The child carries the parent's translation composed with its own.
        propagate_transforms(&mut world);
        let child = world.get::<Children>(root).unwrap().0[0];
        let child_world = world.get::<WorldTransform>(child).unwrap().0;
        assert_eq!(child_world.w_axis.truncate(), Vec3::new(10.0, 5.0, 0.0));
        // Drawables: one per node (both have the single-primitive mesh).
        assert_eq!(world.draw_list().len(), 2);
    }
}
