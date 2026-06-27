//! Minimal from-scratch ECS: generational entities + per-component **sparse-set**
//! storage.
//!
//! Sparse-set (not archetype) is the deliberate first-implementation choice:
//! add/remove are O(1) and the code stays simple, which matters for a scene of a
//! handful-to-thousands of entities where iteration is not the bottleneck. Dense
//! arrays still give cache-friendly iteration. Archetype storage (faster
//! multi-component iteration, more complex) is a measurement-driven later option.
//!
//! Single-threaded / `!Send` by design — no interior locking, no system scheduler
//! (out of scope for Phase 12).

use std::any::{Any, TypeId};
use std::collections::HashMap;

/// A generational entity id: `index` addresses a slot, `generation` distinguishes
/// reused slots so a stale [`Entity`] copy can't alias a freshly-spawned one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Entity {
    index: u32,
    generation: u32,
}

impl Entity {
    /// The slot index (stable storage key).
    #[inline]
    pub fn index(self) -> u32 {
        self.index
    }

    /// The generation stamp for this id.
    #[inline]
    pub fn generation(self) -> u32 {
        self.generation
    }
}

struct EntityMeta {
    generation: u32,
    alive: bool,
}

/// One component type's storage. `sparse[entity.index] -> dense slot`; `dense` and
/// `entities` run in parallel and preserve **insertion order** (no swap on insert),
/// which is what lets the draw list be deterministic.
struct SparseSet<T> {
    sparse: Vec<Option<u32>>,
    entities: Vec<Entity>,
    dense: Vec<T>,
}

impl<T> SparseSet<T> {
    fn new() -> Self {
        Self {
            sparse: Vec::new(),
            entities: Vec::new(),
            dense: Vec::new(),
        }
    }

    fn insert(&mut self, e: Entity, value: T) {
        let i = e.index as usize;
        if i >= self.sparse.len() {
            self.sparse.resize(i + 1, None);
        }
        match self.sparse[i] {
            Some(slot) => {
                self.dense[slot as usize] = value;
                self.entities[slot as usize] = e;
            }
            None => {
                let slot = self.dense.len() as u32;
                self.sparse[i] = Some(slot);
                self.entities.push(e);
                self.dense.push(value);
            }
        }
    }

    fn get(&self, e: Entity) -> Option<&T> {
        let slot = (*self.sparse.get(e.index as usize)?)? as usize;
        // Generation guard: the dense entry must still belong to this exact id.
        if self.entities[slot] == e {
            Some(&self.dense[slot])
        } else {
            None
        }
    }

    fn get_mut(&mut self, e: Entity) -> Option<&mut T> {
        let slot = (*self.sparse.get(e.index as usize)?)? as usize;
        if self.entities[slot] == e {
            Some(&mut self.dense[slot])
        } else {
            None
        }
    }

    /// Remove `e`'s component via swap-remove (O(1); does NOT preserve order —
    /// only used for despawn, never mid-frame on the draw set).
    fn remove(&mut self, e: Entity) -> Option<T> {
        let i = e.index as usize;
        let slot = (*self.sparse.get(i)?)? as usize;
        if self.entities[slot] != e {
            return None;
        }
        let last = self.dense.len() - 1;
        self.entities.swap(slot, last);
        self.dense.swap(slot, last);
        self.sparse[i] = None;
        if slot != last {
            let moved = self.entities[slot];
            self.sparse[moved.index as usize] = Some(slot as u32);
        }
        self.entities.pop();
        self.dense.pop()
    }

    fn iter(&self) -> impl Iterator<Item = (Entity, &T)> {
        self.entities.iter().copied().zip(self.dense.iter())
    }
}

/// Type-erased view of a [`SparseSet`] so the world can drop an entity's components
/// across every storage without knowing their concrete types.
trait ErasedStorage: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn remove_entity(&mut self, e: Entity);
}

impl<T: 'static> ErasedStorage for SparseSet<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn remove_entity(&mut self, e: Entity) {
        self.remove(e);
    }
}

/// The ECS container: owns entity allocation and all component storages.
#[derive(Default)]
pub struct World {
    metas: Vec<EntityMeta>,
    free: Vec<u32>,
    storages: HashMap<TypeId, Box<dyn ErasedStorage>>,
}

impl World {
    /// An empty world.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a fresh entity, reusing a freed slot (with a bumped generation) when
    /// available.
    pub fn spawn(&mut self) -> Entity {
        if let Some(index) = self.free.pop() {
            let meta = &mut self.metas[index as usize];
            meta.generation += 1;
            meta.alive = true;
            Entity {
                index,
                generation: meta.generation,
            }
        } else {
            let index = self.metas.len() as u32;
            self.metas.push(EntityMeta {
                generation: 0,
                alive: true,
            });
            Entity {
                index,
                generation: 0,
            }
        }
    }

    /// Whether `e` is the live occupant of its slot.
    pub fn is_alive(&self, e: Entity) -> bool {
        self.metas
            .get(e.index as usize)
            .is_some_and(|m| m.alive && m.generation == e.generation)
    }

    /// A builder handle for spawning an entity and attaching components fluently.
    pub fn spawn_node(&mut self) -> crate::NodeRef<'_> {
        let e = self.spawn();
        crate::NodeRef::new(self, e)
    }

    /// Despawn `e`, dropping every component it owns. The slot is recycled with a
    /// bumped generation so old ids become dangling.
    pub fn despawn(&mut self, e: Entity) {
        if !self.is_alive(e) {
            return;
        }
        for storage in self.storages.values_mut() {
            storage.remove_entity(e);
        }
        let meta = &mut self.metas[e.index as usize];
        meta.alive = false;
        self.free.push(e.index);
    }

    fn storage<T: 'static>(&self) -> Option<&SparseSet<T>> {
        self.storages
            .get(&TypeId::of::<T>())
            .and_then(|b| b.as_any().downcast_ref::<SparseSet<T>>())
    }

    fn storage_mut<T: 'static>(&mut self) -> &mut SparseSet<T> {
        self.storages
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(SparseSet::<T>::new()))
            .as_any_mut()
            .downcast_mut::<SparseSet<T>>()
            .expect("storage type matches TypeId")
    }

    /// Attach (or replace) component `value` on `e`.
    pub fn insert<T: 'static>(&mut self, e: Entity, value: T) {
        self.storage_mut::<T>().insert(e, value);
    }

    /// Remove component `T` from `e`, returning it if present.
    pub fn remove<T: 'static>(&mut self, e: Entity) -> Option<T> {
        self.storage_mut::<T>().remove(e)
    }

    /// Borrow `e`'s component `T`.
    pub fn get<T: 'static>(&self, e: Entity) -> Option<&T> {
        self.storage::<T>().and_then(|s| s.get(e))
    }

    /// Mutably borrow `e`'s component `T`.
    pub fn get_mut<T: 'static>(&mut self, e: Entity) -> Option<&mut T> {
        self.storage_mut::<T>().get_mut(e)
    }

    /// Iterate `(entity, &T)` over all entities carrying `T`, in insertion order.
    pub fn iter<T: 'static>(&self) -> impl Iterator<Item = (Entity, &T)> {
        self.storage::<T>().into_iter().flat_map(|s| s.iter())
    }

    /// Collect `(entity, &A, &B)` for every entity carrying both — driven by `A`'s
    /// insertion order. Used by the draw list and transform propagation.
    pub fn query2<A: 'static, B: 'static>(&self) -> Vec<(Entity, &A, &B)> {
        let (Some(a), Some(b)) = (self.storage::<A>(), self.storage::<B>()) else {
            return Vec::new();
        };
        a.iter()
            .filter_map(|(e, av)| b.get(e).map(|bv| (e, av, bv)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_insert_get() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, 42u32);
        assert_eq!(w.get::<u32>(e), Some(&42));
        assert_eq!(w.get::<i8>(e), None);
    }

    #[test]
    fn generation_guards_stale_ids() {
        let mut w = World::new();
        let e0 = w.spawn();
        w.insert(e0, 1u32);
        w.despawn(e0);
        let e1 = w.spawn(); // reuses the slot with a bumped generation
        assert_eq!(e0.index(), e1.index());
        assert_ne!(e0.generation(), e1.generation());
        assert!(!w.is_alive(e0));
        assert!(w.is_alive(e1));
        // The stale id must not read the recycled slot's data (despawn cleared it).
        assert_eq!(w.get::<u32>(e0), None);
        assert_eq!(w.iter::<u32>().count(), 0);
    }

    #[test]
    fn iter_preserves_insertion_order() {
        let mut w = World::new();
        for i in 0..4u32 {
            let e = w.spawn();
            w.insert(e, i);
        }
        let order: Vec<u32> = w.iter::<u32>().map(|(_, v)| *v).collect();
        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn query2_intersects_both_components() {
        let mut w = World::new();
        let a = w.spawn();
        w.insert(a, 1u32);
        w.insert(a, 1.5f32);
        let b = w.spawn();
        w.insert(b, 2u32); // no f32 -> excluded
        let c = w.spawn();
        w.insert(c, 3u32);
        w.insert(c, 3.5f32);
        let got: Vec<(u32, f32)> = w
            .query2::<u32, f32>()
            .iter()
            .map(|(_, a, b)| (**a, **b))
            .collect();
        assert_eq!(got, vec![(1, 1.5), (3, 3.5)]);
    }
}
