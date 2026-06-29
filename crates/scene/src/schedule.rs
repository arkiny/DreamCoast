//! Parallel **system schedule** (Phase 15 M3).
//!
//! A system is a function over the world plus a declared component [`Access`] set
//! (which components it reads / writes). The [`SystemSchedule`] groups systems
//! into **batches** of mutually non-conflicting systems and runs each batch in
//! parallel on the [`dreamcoast_jobs`] work-stealing pool; conflicting systems
//! fall into later batches, so a writer always runs before a reader of the same
//! component. This is the CPU analogue of the render graph: declared dependencies,
//! automatic ordering, parallel where independent.
//!
//! **Determinism (engine rule 8).** Batches are formed in registration order and
//! systems within a batch touch disjoint storages, so the world state after a run
//! is independent of worker count or execution interleaving — headless golden
//! images are unaffected.
//!
//! **Soundness.** Within a batch every system's writes are disjoint and no read
//! overlaps another's write, so the [`WorldCell`](crate::ecs::WorldCell) each
//! system receives accesses a disjoint set of storages. All accessed storages are
//! pre-created (single-threaded) before the parallel region, so no worker mutates
//! the shared storage map. See `WorldCell` for the full argument.

use std::any::TypeId;

use dreamcoast_jobs::JobSystem;

use crate::ecs::{World, WorldCell};

/// Resolves a type-erased storage pointer for one component from a `&mut World`
/// (creating the storage if absent). One per declared component; see [`Access`].
type StorageGetter = fn(&mut World) -> (TypeId, *mut ());

/// A system's declared component access — the contract the scheduler relies on to
/// decide what may run in parallel.
#[derive(Clone, Default)]
pub struct Access {
    reads: Vec<TypeId>,
    writes: Vec<TypeId>,
    // One storage-pointer getter per declared component (read or write), in
    // declaration order. The scheduler calls these once while holding the
    // exclusive `&mut World` to resolve each system's disjoint pointer table
    // (also creating the storage if absent) before any parallel work.
    getters: Vec<StorageGetter>,
}

impl Access {
    /// An empty access set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a read of component `T`.
    pub fn reads<T: 'static>(mut self) -> Self {
        self.reads.push(TypeId::of::<T>());
        self.getters.push(World::storage_ptr::<T>);
        self
    }

    /// Declare a write of component `T`.
    pub fn writes<T: 'static>(mut self) -> Self {
        self.writes.push(TypeId::of::<T>());
        self.getters.push(World::storage_ptr::<T>);
        self
    }

    /// Two systems conflict if one writes a component the other reads or writes.
    fn conflicts_with(&self, other: &Access) -> bool {
        self.writes
            .iter()
            .any(|w| other.writes.contains(w) || other.reads.contains(w))
            || other.writes.iter().any(|w| self.reads.contains(w))
    }
}

type SystemFn = Box<dyn Fn(&WorldCell) + Send + Sync>;

struct System {
    access: Access,
    run: SystemFn,
}

/// An ordered set of systems run in conflict-respecting parallel batches.
#[derive(Default)]
pub struct SystemSchedule {
    systems: Vec<System>,
}

impl SystemSchedule {
    /// An empty schedule.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a system with its declared [`Access`]. The body receives a
    /// [`WorldCell`] scoped to that access. Systems run in registration order
    /// except where parallelism is safe.
    pub fn add<F>(&mut self, access: Access, run: F)
    where
        F: Fn(&WorldCell) + Send + Sync + 'static,
    {
        self.systems.push(System {
            access,
            run: Box::new(run),
        });
    }

    /// Number of registered systems.
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// Whether the schedule has no systems.
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// Partition systems into batches: each system joins the current batch unless
    /// it conflicts with a member already in it, in which case it opens a new
    /// batch. Registration order is preserved, making batching deterministic.
    fn batches(&self) -> Vec<Vec<usize>> {
        let mut batches: Vec<Vec<usize>> = Vec::new();
        let mut current: Vec<usize> = Vec::new();
        for (i, sys) in self.systems.iter().enumerate() {
            let conflicts = current
                .iter()
                .any(|&j| sys.access.conflicts_with(&self.systems[j].access));
            if conflicts {
                batches.push(std::mem::take(&mut current));
            }
            current.push(i);
        }
        if !current.is_empty() {
            batches.push(current);
        }
        batches
    }

    /// Run every system once, in dependency order, parallelising independent
    /// systems across `jobs`.
    pub fn run(&mut self, world: &mut World, jobs: &JobSystem) {
        // Resolve every system's disjoint storage-pointer table up front, while we
        // hold the exclusive `&mut World` (so write pointers are legitimately
        // mutable and storages are created here, never in the parallel region).
        // Box-backed storage addresses are stable across later HashMap growth, so
        // pointers resolved early stay valid.
        let ptr_tables: Vec<Vec<(TypeId, *mut ())>> = self
            .systems
            .iter()
            .map(|sys| sys.access.getters.iter().map(|g| g(world)).collect())
            .collect();

        let batches = self.batches();
        let systems = &self.systems;

        for batch in batches {
            if batch.len() == 1 {
                // Single system: run inline on the calling thread (no scope cost).
                let idx = batch[0];
                let sys = &systems[idx];
                let cell = WorldCell::new(
                    ptr_tables[idx].clone(),
                    sys.access.reads.clone(),
                    sys.access.writes.clone(),
                );
                (sys.run)(&cell);
            } else {
                let ptr_tables = &ptr_tables;
                jobs.scope(|s| {
                    for &idx in &batch {
                        let sys = &systems[idx];
                        // Each concurrent system gets its own cell over a disjoint
                        // storage set (batch invariant) → sound.
                        let cell = WorldCell::new(
                            ptr_tables[idx].clone(),
                            sys.access.reads.clone(),
                            sys.access.writes.clone(),
                        );
                        s.spawn(move |_| (sys.run)(&cell));
                    }
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::World;

    // Distinct component types so systems can declare disjoint access.
    #[derive(Clone, Copy)]
    struct A(u32);
    #[derive(Clone, Copy)]
    struct B(u32);
    #[derive(Clone, Copy)]
    struct C(u32);

    fn jobs() -> JobSystem {
        JobSystem::new(Some(4))
    }

    #[test]
    fn disjoint_writers_run_and_produce_correct_state() {
        let mut world = World::new();
        let mut entities = Vec::new();
        for i in 0..1000u32 {
            let e = world.spawn();
            world.insert(e, A(i));
            world.insert(e, B(0));
            world.insert(e, C(0));
            entities.push(e);
        }

        let mut sched = SystemSchedule::new();
        // Two systems writing *different* components from the same read → no
        // conflict → same batch → run in parallel.
        sched.add(Access::new().reads::<A>().writes::<B>(), |cell| {
            for (e, a) in cell.collect_read::<A>() {
                cell.insert(e, B(a.0 * 2));
            }
        });
        sched.add(Access::new().reads::<A>().writes::<C>(), |cell| {
            for (e, a) in cell.collect_read::<A>() {
                cell.insert(e, C(a.0 + 1));
            }
        });

        let js = jobs();
        sched.run(&mut world, &js);

        for (i, &e) in entities.iter().enumerate() {
            assert_eq!(world.get::<B>(e).unwrap().0, i as u32 * 2);
            assert_eq!(world.get::<C>(e).unwrap().0, i as u32 + 1);
        }
    }

    #[test]
    fn read_write_conflict_serializes_in_order() {
        // sys0 writes B; sys1 reads B and writes C. They conflict → different
        // batches → sys1 must observe sys0's write.
        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, A(5));
        world.insert(e, B(0));
        world.insert(e, C(0));

        let mut sched = SystemSchedule::new();
        sched.add(Access::new().reads::<A>().writes::<B>(), move |cell| {
            let a = cell.get_copy::<A>(e).unwrap();
            cell.insert(e, B(a.0 * 10));
        });
        sched.add(Access::new().reads::<B>().writes::<C>(), move |cell| {
            let b = cell.get_copy::<B>(e).unwrap();
            cell.insert(e, C(b.0 + 7));
        });

        let js = jobs();
        sched.run(&mut world, &js);

        assert_eq!(world.get::<B>(e).unwrap().0, 50);
        assert_eq!(world.get::<C>(e).unwrap().0, 57); // sees B's write → ordered
    }

    #[test]
    fn writer_feeds_propagate_through_schedule() {
        // End-to-end M3 path: a system that writes LocalTransform, then the real
        // propagate system reads it and writes WorldTransform. They conflict on
        // LocalTransform → different batches → propagate observes the write.
        use crate::transform::{LocalTransform, WorldTransform, propagate_transforms_system};
        use glam::{Mat4, Vec3};

        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, LocalTransform::IDENTITY);

        let mut sched = SystemSchedule::new();
        // Move the entity by writing LocalTransform.
        sched.add(Access::new().writes::<LocalTransform>(), move |cell| {
            cell.insert(e, LocalTransform::trs(Vec3::new(3.0, 0.0, 0.0), 2.0));
        });
        // Then propagate (reads LocalTransform + Parent, writes WorldTransform).
        let (access, run) = propagate_transforms_system();
        sched.add(access, run);

        let js = jobs();
        sched.run(&mut world, &js);

        let expected = LocalTransform::trs(Vec3::new(3.0, 0.0, 0.0), 2.0).matrix();
        assert_eq!(world.get::<WorldTransform>(e).unwrap().0, expected);
        assert_ne!(world.get::<WorldTransform>(e).unwrap().0, Mat4::IDENTITY);
    }

    #[test]
    fn batching_groups_independent_and_splits_conflicts() {
        let mut sched = SystemSchedule::new();
        sched.add(Access::new().writes::<A>(), |_| {}); // 0
        sched.add(Access::new().writes::<B>(), |_| {}); // 1 (indep of 0)
        sched.add(Access::new().reads::<A>().writes::<C>(), |_| {}); // 2 (reads A → conflicts 0)
        let batches = sched.batches();
        assert_eq!(batches, vec![vec![0, 1], vec![2]]);
    }
}
