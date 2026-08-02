//! Flattening the ECS into a render-ready draw list.

use glam::Mat4;

use crate::components::{MaterialHandle, MeshHandle, MeshInstance};
use crate::ecs::{Entity, World};
use crate::transform::WorldTransform;

/// One resolved draw: a world matrix + the mesh/material handles + render flags.
/// The renderer materializes this into actual draw calls (and TLAS instances).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Drawable {
    /// The ECS entity this draw came from — the STABLE cross-frame identity of the
    /// drawable (generational, so a died-and-respawned slot never aliases). Temporal
    /// consumers (motion vectors, caches) key per-object history on it; a draw-list
    /// index is NOT stable once anything spawns or despawns mid-run.
    pub entity: Entity,
    pub world: Mat4,
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub casts_shadow: bool,
}

impl World {
    /// The frame's draw list: one [`Drawable`] per entity carrying both a
    /// [`WorldTransform`] and a [`MeshInstance`], in `MeshInstance` insertion order
    /// (deterministic — the renderer relies on this for stable TLAS instance order).
    ///
    /// Call [`crate::propagate_transforms`] first so world matrices are current.
    pub fn draw_list(&self) -> Vec<Drawable> {
        self.query2::<MeshInstance, WorldTransform>()
            .into_iter()
            .map(|(e, mi, wt)| Drawable {
                entity: e,
                world: wt.0,
                mesh: mi.mesh,
                material: mi.material,
                casts_shadow: mi.casts_shadow,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::MeshInstance;
    use crate::transform::{LocalTransform, propagate_transforms};
    use glam::Vec3;

    #[test]
    fn draw_list_follows_spawn_order() {
        let mut w = World::new();
        for i in 0..3u32 {
            let e = w.spawn();
            w.insert(e, LocalTransform::trs(Vec3::new(i as f32, 0.0, 0.0), 1.0));
            w.insert(e, MeshInstance::new(MeshHandle(i), MaterialHandle(i)));
        }
        propagate_transforms(&mut w);
        let dl = w.draw_list();
        let meshes: Vec<u32> = dl.iter().map(|d| d.mesh.0).collect();
        assert_eq!(meshes, vec![0, 1, 2]);
        assert_eq!(dl[2].world.w_axis.x, 2.0);
    }
}
