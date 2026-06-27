//! The transform hierarchy, expressed as ECS components (not a separate tree).
//!
//! Authoring sets [`LocalTransform`] (+ optional [`Parent`]); [`propagate_transforms`]
//! derives the absolute [`WorldTransform`] for every entity. A root (no `Parent`)
//! has `world == local`.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

use crate::ecs::{Entity, World};

/// An entity's transform relative to its parent (or to the world, if it is a root).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalTransform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl LocalTransform {
    /// The identity transform (no translation/rotation, unit scale).
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Translation + uniform scale, no rotation — the common case for placing props.
    /// Built from the same primitives as `from_translation * from_scale`, so the
    /// resulting matrix is bit-identical to that product.
    pub fn trs(translation: Vec3, uniform_scale: f32) -> Self {
        Self {
            translation,
            rotation: Quat::IDENTITY,
            scale: Vec3::splat(uniform_scale),
        }
    }

    /// This transform as a column-major matrix (`T * R * S`).
    #[inline]
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

impl Default for LocalTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// The entity's absolute world matrix, written by [`propagate_transforms`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldTransform(pub Mat4);

/// Links a child entity to its parent. Absence of this component marks a root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parent(pub Entity);

/// A parent's ordered child list (maintained by the node builder). Optional — the
/// authoritative link is [`Parent`]; `Children` is a convenience for top-down walks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Children(pub Vec<Entity>);

// Guards against a malformed parent cycle turning the walk-up into an infinite loop.
const MAX_HIERARCHY_DEPTH: usize = 256;

/// Compute every entity's [`WorldTransform`] from its [`LocalTransform`] chain.
///
/// Each node multiplies its ancestors' locals (root → leaf). For the typical
/// shallow scene this whole-world recompute is trivial; dirty-tracking is a later
/// optimization. Entities without a `LocalTransform` are skipped (they keep any
/// existing `WorldTransform`).
pub fn propagate_transforms(world: &mut World) {
    // Snapshot the local matrices and parent links without holding storage borrows
    // across the write-back (WorldTransform lives in a different storage).
    let locals: HashMap<Entity, Mat4> = world
        .iter::<LocalTransform>()
        .map(|(e, lt)| (e, lt.matrix()))
        .collect();

    let mut results: Vec<(Entity, Mat4)> = Vec::with_capacity(locals.len());
    for (&entity, &local) in &locals {
        let mut world_mat = local;
        let mut cursor = entity;
        let mut depth = 0;
        while let Some(Parent(parent)) = world.get::<Parent>(cursor).copied() {
            // An ancestor missing a LocalTransform contributes identity.
            if let Some(&parent_local) = locals.get(&parent) {
                world_mat = parent_local * world_mat;
            }
            cursor = parent;
            depth += 1;
            if depth >= MAX_HIERARCHY_DEPTH {
                break;
            }
        }
        results.push((entity, world_mat));
    }

    for (entity, world_mat) in results {
        world.insert(entity, WorldTransform(world_mat));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trs_matrix_matches_translation_times_scale() {
        // The whole no-regression argument hinges on this equality.
        let t = Vec3::new(1.0, 2.0, -3.0);
        let s = 0.45f32;
        let lt = LocalTransform::trs(t, s);
        let expected = Mat4::from_translation(t) * Mat4::from_scale(Vec3::splat(s));
        assert_eq!(lt.matrix(), expected);
    }

    #[test]
    fn root_world_equals_local() {
        let mut w = World::new();
        let e = w.spawn();
        let lt = LocalTransform::trs(Vec3::new(5.0, 0.0, 0.0), 2.0);
        w.insert(e, lt);
        propagate_transforms(&mut w);
        assert_eq!(w.get::<WorldTransform>(e).unwrap().0, lt.matrix());
    }

    #[test]
    fn child_composes_parent() {
        let mut w = World::new();
        let parent = w.spawn();
        let parent_lt = LocalTransform::trs(Vec3::new(10.0, 0.0, 0.0), 1.0);
        w.insert(parent, parent_lt);
        let child = w.spawn();
        let child_lt = LocalTransform::trs(Vec3::new(0.0, 5.0, 0.0), 1.0);
        w.insert(child, child_lt);
        w.insert(child, Parent(parent));
        propagate_transforms(&mut w);
        let expected = parent_lt.matrix() * child_lt.matrix();
        assert_eq!(w.get::<WorldTransform>(child).unwrap().0, expected);
        // Moving the parent moves the child.
        assert_eq!(
            w.get::<WorldTransform>(child).unwrap().0.w_axis,
            (parent_lt.matrix() * child_lt.matrix()).w_axis
        );
    }
}
