//! Typed **singleton store** — one value per type, keyed by [`TypeId`].
//!
//! Components answer "per entity"; a lot of game state is per *world* instead
//! (elapsed time, the input snapshot, the RNG, an [`Events`](crate::Events)
//! channel). Modelling those as a component on a dummy entity is a well-known
//! anti-pattern — they get swept into queries and the draw list — so they live here.
//!
//! **Why this is a standalone struct and not a field of [`World`](crate::World).**
//! `World` hands out type-erased storage pointers to [`WorldCell`](crate::WorldCell)
//! so systems can run in parallel over disjoint components; that soundness argument
//! rests on every accessible item being a `SparseSet` the scheduler has resolved and
//! proven disjoint. A resource map inside `World` would either need the same
//! per-type access declarations in [`Access`](crate::Access) (a scheduler feature,
//! not an M0 one) or be unreachable from a `WorldCell` anyway — and in the meantime
//! it would widen the `unsafe` surface for zero gain. Keeping it separate means the
//! app owns `World` and `Resources` side by side, the parallel region borrows only
//! what it declared, and folding resources into the schedule later is an additive
//! change rather than a rewrite of the aliasing argument.
//!
//! Like `World`, this is single-threaded and `!Send` by design: values are plain
//! `Box<dyn Any>` with no interior locking.

use std::any::{Any, TypeId};
use std::collections::HashMap;

/// A map holding at most one value of each type.
#[derive(Default)]
pub struct Resources {
    map: HashMap<TypeId, Box<dyn Any>>,
}

impl Resources {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `value`, replacing and returning any previous value of the same type.
    pub fn insert<T: 'static>(&mut self, value: T) -> Option<T> {
        self.map
            .insert(TypeId::of::<T>(), Box::new(value))
            .and_then(|old| old.downcast::<T>().ok().map(|b| *b))
    }

    /// Borrow the `T` resource, or `None` if it was never inserted.
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
    }

    /// Mutably borrow the `T` resource.
    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.map
            .get_mut(&TypeId::of::<T>())
            .and_then(|b| b.downcast_mut::<T>())
    }

    /// Mutably borrow the `T` resource, inserting `T::default()` first if absent.
    /// The ergonomic path for lazily-created singletons such as event channels.
    pub fn get_or_default<T: 'static + Default>(&mut self) -> &mut T {
        self.map
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(T::default()))
            .downcast_mut::<T>()
            .expect("resource type matches TypeId")
    }

    /// Remove and return the `T` resource.
    pub fn remove<T: 'static>(&mut self) -> Option<T> {
        self.map
            .remove(&TypeId::of::<T>())
            .and_then(|b| b.downcast::<T>().ok().map(|b| *b))
    }

    /// Whether a `T` resource is present.
    pub fn contains<T: 'static>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<T>())
    }

    /// Number of stored resources.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the store holds nothing.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Drop every resource (e.g. on a level swap).
    pub fn clear(&mut self) {
        self.map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(PartialEq, Debug)]
    struct Clock(f32);
    #[derive(PartialEq, Debug, Default)]
    struct Score(u32);

    #[test]
    fn insert_get_remove_roundtrip() {
        let mut r = Resources::new();
        assert!(!r.contains::<Clock>());
        assert!(r.get::<Clock>().is_none());
        assert!(r.insert(Clock(1.0)).is_none());
        assert!(r.contains::<Clock>());
        assert_eq!(r.get::<Clock>(), Some(&Clock(1.0)));
        assert_eq!(r.remove::<Clock>(), Some(Clock(1.0)));
        assert!(!r.contains::<Clock>());
        assert!(r.remove::<Clock>().is_none());
    }

    #[test]
    fn insert_replaces_and_returns_previous() {
        let mut r = Resources::new();
        r.insert(Clock(1.0));
        assert_eq!(r.insert(Clock(2.0)), Some(Clock(1.0)));
        assert_eq!(r.get::<Clock>(), Some(&Clock(2.0)));
        assert_eq!(r.len(), 1, "same type must not create a second slot");
    }

    #[test]
    fn types_are_independent() {
        let mut r = Resources::new();
        r.insert(Clock(1.0));
        r.insert(Score(7));
        assert_eq!(r.len(), 2);
        assert_eq!(r.get::<Score>(), Some(&Score(7)));
        r.remove::<Clock>();
        assert!(r.contains::<Score>());
        assert!(!r.contains::<Clock>());
    }

    #[test]
    fn get_mut_writes_through() {
        let mut r = Resources::new();
        r.insert(Score(1));
        r.get_mut::<Score>().unwrap().0 += 41;
        assert_eq!(r.get::<Score>(), Some(&Score(42)));
        assert!(r.get_mut::<Clock>().is_none());
    }

    #[test]
    fn get_or_default_inserts_once() {
        let mut r = Resources::new();
        r.get_or_default::<Score>().0 = 5;
        assert_eq!(r.len(), 1);
        // Second call must return the *existing* value, not a fresh default.
        assert_eq!(r.get_or_default::<Score>().0, 5);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn holds_an_event_channel() {
        // The intended pairing: event channels are world singletons.
        use crate::Events;
        let mut r = Resources::new();
        r.get_or_default::<Events<u32>>().send(3);
        assert_eq!(r.get::<Events<u32>>().unwrap().len(), 1);
        r.get_mut::<Events<u32>>().unwrap().update();
        assert!(r.get::<Events<u32>>().unwrap().is_empty());
    }

    #[test]
    fn clear_drops_everything() {
        let mut r = Resources::new();
        r.insert(Clock(1.0));
        r.insert(Score(2));
        r.clear();
        assert!(r.is_empty());
    }
}
