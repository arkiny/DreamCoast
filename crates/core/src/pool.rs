//! A generational-index pool, the foundation for GPU resource handles.
//!
//! Handles are small `Copy` values (index + generation) rather than pointers,
//! so they can be passed around freely and validated on access. When a slot is
//! freed and later reused its generation is bumped, which makes stale handles
//! detectable instead of silently aliasing a new resource. The RHI crates reuse
//! this for buffers, textures, pipelines, and so on.

use std::marker::PhantomData;

/// A lightweight, typed reference to a value stored in a [`Pool`].
///
/// The `T` parameter is purely a compile-time tag so handles to different
/// resource kinds cannot be mixed up; it carries no runtime cost.
pub struct Handle<T> {
    index: u32,
    generation: u32,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Handle<T> {
    /// The slot index this handle refers to.
    #[inline]
    pub fn index(self) -> u32 {
        self.index
    }

    /// The generation this handle was minted at.
    #[inline]
    pub fn generation(self) -> u32 {
        self.generation
    }
}

// Manual impls so the `T` tag does not impose its own bounds (derive would
// require `T: Clone`, `T: PartialEq`, etc.).
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Handle<T> {}
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}
impl<T> Eq for Handle<T> {}
impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Handle({}, gen {})", self.index, self.generation)
    }
}

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// A generational arena: stable typed handles over a contiguous backing store.
pub struct Pool<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    len: usize,
}

impl<T> Pool<T> {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
        }
    }

    /// Number of live entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the pool holds no live entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert a value, returning a handle that refers to it.
    pub fn insert(&mut self, value: T) -> Handle<T> {
        self.len += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.value = Some(value);
            Handle {
                index,
                generation: slot.generation,
                _marker: PhantomData,
            }
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 0,
                value: Some(value),
            });
            Handle {
                index,
                generation: 0,
                _marker: PhantomData,
            }
        }
    }

    /// Remove the value a handle refers to, returning it if the handle is live.
    ///
    /// The slot's generation is bumped so any other handle to it becomes stale.
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let value = slot.value.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(handle.index);
        self.len -= 1;
        Some(value)
    }

    /// Borrow the value a handle refers to, if the handle is still live.
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        let slot = self.slots.get(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.value.as_ref()
    }

    /// Mutably borrow the value a handle refers to, if the handle is still live.
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.value.as_mut()
    }
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove() {
        let mut pool = Pool::new();
        let a = pool.insert(10u32);
        let b = pool.insert(20u32);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.get(a), Some(&10));
        assert_eq!(pool.get(b), Some(&20));
        assert_eq!(pool.remove(a), Some(10));
        assert_eq!(pool.get(a), None);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn stale_handle_after_reuse() {
        let mut pool = Pool::new();
        let a = pool.insert(1u32);
        pool.remove(a);
        // Reuses slot 0 with a bumped generation.
        let b = pool.insert(2u32);
        assert_eq!(a.index(), b.index());
        assert_ne!(a.generation(), b.generation());
        assert_eq!(pool.get(a), None, "stale handle must not resolve");
        assert_eq!(pool.get(b), Some(&2));
    }
}
