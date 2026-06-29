//! The transform hierarchy, expressed as ECS components (not a separate tree).
//!
//! Authoring sets [`LocalTransform`] (+ optional [`Parent`]); [`propagate_transforms`]
//! derives the absolute [`WorldTransform`] for every entity. A root (no `Parent`)
//! has `world == local`.

use std::collections::HashMap;

use dreamcoast_jobs::JobSystem;
use glam::{Mat4, Quat, Vec3};

use crate::ecs::{Entity, World, WorldCell};
use crate::schedule::Access;

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

/// Constant angular velocity applied to an entity's [`LocalTransform`] each sim
/// step by [`advance_spin`] — the engine's simplest piece of per-frame motion,
/// used to exercise the fixed-timestep → parallel-propagate → draw pipeline with
/// actual moving objects.
#[derive(Clone, Copy, Debug)]
pub struct Spin {
    /// Rotation axis (normalised on use).
    pub axis: Vec3,
    /// Angular speed in radians per second.
    pub speed: f32,
}

/// Advance every [`Spin`] entity's [`LocalTransform`] rotation by one `dt` step.
///
/// A real ECS system: reads `Spin`, writes `LocalTransform`. Deterministic given
/// the same `dt` sequence (the engine drives it from the fixed-timestep
/// accumulator), so headless capture sequences reproduce exactly. Call
/// [`propagate_transforms`] / [`propagate_transforms_parallel`] afterwards to push
/// the new locals out to `WorldTransform`.
pub fn advance_spin(world: &mut World, dt: f32) {
    // Snapshot the per-entity delta rotations without holding the Spin storage
    // borrow across the LocalTransform write-back.
    let deltas: Vec<(Entity, Quat)> = world
        .iter::<Spin>()
        .map(|(e, s)| (e, Quat::from_axis_angle(s.axis.normalize(), s.speed * dt)))
        .collect();
    for (e, dq) in deltas {
        if let Some(lt) = world.get_mut::<LocalTransform>(e) {
            // Left-multiply: spin in the entity's parent space; renormalise to keep
            // the quaternion unit over long runs.
            lt.rotation = (dq * lt.rotation).normalize();
        }
    }
}

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

/// Resolve one entity's world matrix from snapshot data (its own local + the
/// ancestor chain). Pure function of the snapshots, so it is safe to evaluate in
/// parallel and gives the **bit-identical** result of the sequential walk (same
/// `parent_local * world_mat` fold order).
fn resolve_world(
    entity: Entity,
    local: Mat4,
    locals: &HashMap<Entity, Mat4>,
    parents: &HashMap<Entity, Entity>,
) -> Mat4 {
    let mut world_mat = local;
    let mut cursor = entity;
    let mut depth = 0;
    while let Some(&parent) = parents.get(&cursor) {
        if let Some(&parent_local) = locals.get(&parent) {
            world_mat = parent_local * world_mat;
        }
        cursor = parent;
        depth += 1;
        if depth >= MAX_HIERARCHY_DEPTH {
            break;
        }
    }
    world_mat
}

/// Parallel [`propagate_transforms`]: snapshot (single-threaded) → resolve every
/// entity's world matrix in parallel on `jobs` → write back (single-threaded).
///
/// Each entity's matrix is an independent pure function of the immutable snapshots
/// (no `World` access in the parallel region — `World` is `!Sync`), so the result
/// is **bit-identical** to the sequential version and to the single-threaded run
/// regardless of worker count. The draw list reads `WorldTransform` by entity in
/// `MeshInstance` insertion order, so write-back order does not affect it.
pub fn propagate_transforms_parallel(world: &mut World, jobs: &JobSystem) {
    let locals: HashMap<Entity, Mat4> = world
        .iter::<LocalTransform>()
        .map(|(e, lt)| (e, lt.matrix()))
        .collect();
    let parents: HashMap<Entity, Entity> = world.iter::<Parent>().map(|(e, p)| (e, p.0)).collect();

    // Stable input vector for the parallel pass (order is irrelevant to results).
    let mut results: Vec<(Entity, Mat4)> = locals.iter().map(|(&e, &m)| (e, m)).collect();
    // ~256 entities/chunk: enough work per job to amortise scheduling on big
    // scenes, while tiny scenes stay in one or two chunks.
    jobs.parallel_for(&mut results, 256, |_, (entity, mat)| {
        *mat = resolve_world(*entity, *mat, &locals, &parents);
    });

    for (entity, world_mat) in results {
        world.insert(entity, WorldTransform(world_mat));
    }
}

/// [`propagate_transforms`] packaged as a schedule system: reads `LocalTransform`
/// and `Parent`, writes `WorldTransform`. Register it with
/// [`SystemSchedule::add`](crate::SystemSchedule::add). The body is the sequential
/// resolve over a [`WorldCell`]; intra-system data-parallelism is provided
/// separately by [`propagate_transforms_parallel`] (the two share `resolve_world`,
/// so they agree bit-for-bit).
pub fn propagate_transforms_system() -> (Access, impl Fn(&WorldCell) + Send + Sync) {
    let access = Access::new()
        .reads::<LocalTransform>()
        .reads::<Parent>()
        .writes::<WorldTransform>();
    let run = |cell: &WorldCell| {
        let locals: HashMap<Entity, Mat4> = cell
            .collect_read::<LocalTransform>()
            .into_iter()
            .map(|(e, lt)| (e, lt.matrix()))
            .collect();
        let parents: HashMap<Entity, Entity> = cell
            .collect_read::<Parent>()
            .into_iter()
            .map(|(e, p)| (e, p.0))
            .collect();
        for (&entity, &local) in &locals {
            let world_mat = resolve_world(entity, local, &locals, &parents);
            cell.insert(entity, WorldTransform(world_mat));
        }
    };
    (access, run)
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
    fn advance_spin_moves_and_is_deterministic() {
        let build = || {
            let mut w = World::new();
            let e = w.spawn();
            w.insert(e, LocalTransform::IDENTITY);
            w.insert(
                e,
                Spin {
                    axis: Vec3::Y,
                    speed: 1.0,
                },
            );
            (w, e)
        };
        // Same dt sequence → identical end rotation (determinism).
        let step = |w: &mut World, e: Entity, n: usize| {
            for _ in 0..n {
                advance_spin(w, 1.0 / 60.0);
            }
            propagate_transforms(w);
            w.get::<WorldTransform>(e).unwrap().0
        };
        let (mut a, ea) = build();
        let (mut b, eb) = build();
        let ra = step(&mut a, ea, 120);
        let rb = step(&mut b, eb, 120);
        assert_eq!(ra, rb, "spin is deterministic for the same dt sequence");
        // And it actually moved (not identity).
        assert_ne!(ra, Mat4::IDENTITY, "spin produced motion");
    }

    #[test]
    fn parallel_matches_sequential_bit_for_bit() {
        use dreamcoast_jobs::JobSystem;
        // Build a multi-level hierarchy with assorted transforms.
        let build = || {
            let mut w = World::new();
            let mut prev: Option<Entity> = None;
            for i in 0..2000u32 {
                let e = w.spawn();
                let lt = LocalTransform {
                    translation: Vec3::new(i as f32 * 0.1, (i % 7) as f32, -(i as f32) * 0.05),
                    rotation: Quat::from_rotation_y(i as f32 * 0.01),
                    scale: Vec3::splat(1.0 + (i % 3) as f32 * 0.25),
                };
                w.insert(e, lt);
                // Chain every 4th entity to the previous to make real ancestry.
                if i % 4 != 0
                    && let Some(p) = prev
                {
                    w.insert(e, Parent(p));
                }
                prev = Some(e);
            }
            w
        };

        let mut seq = build();
        propagate_transforms(&mut seq);

        let mut par = build();
        let js = JobSystem::new(Some(4));
        propagate_transforms_parallel(&mut par, &js);

        // Every entity's world matrix must match exactly (bit-identical).
        let seq_list: Vec<(Entity, Mat4)> = seq
            .iter::<WorldTransform>()
            .map(|(e, w)| (e, w.0))
            .collect();
        for (e, m) in seq_list {
            assert_eq!(par.get::<WorldTransform>(e).unwrap().0, m);
        }
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
