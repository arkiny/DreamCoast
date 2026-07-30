//! Double-buffered **event channel** — the one-way fan-out sibling of components.
//!
//! Gameplay code needs "something happened" messages (damage dealt, door opened)
//! that any number of readers consume without the sender knowing about them.
//! Storing those as components would leak them into the draw list and into every
//! query; an [`Events<T>`] channel keeps them out of the ECS storages entirely.
//!
//! **Lifetime: exactly one update cycle.** [`send`](Events::send) appends to the
//! current buffer, [`iter`](Events::iter) / [`drain`](Events::drain) read that same
//! buffer, and [`update`](Events::update) — called once per fixed-step tick, at the
//! tick boundary — retires it. An event sent during tick *N* is therefore visible
//! for the rest of tick *N* and gone in tick *N+1*, regardless of system order, so
//! a reader can never miss-then-double-read one. The retired buffer is not freed but
//! swapped into `previous` and reused as next tick's write buffer (that is the
//! "double" in double-buffered): steady-state event traffic allocates zero times.
//!
//! Ordering is send order, which is deterministic given a deterministic system
//! order — the same rule the draw list relies on.
//!
//! A channel is normally stored as a singleton in [`Resources`](crate::Resources):
//! `resources.insert(Events::<DamageEvent>::new())`.

/// A single-producer-agnostic, multi-reader event queue for one event type `T`.
///
/// See the module docs for the lifetime contract. `T` has no bounds: any type may
/// be an event.
pub struct Events<T> {
    /// Events sent since the last [`update`](Events::update) — what readers see.
    current: Vec<T>,
    /// Last cycle's buffer. Kept only so its allocation can be recycled as the next
    /// cycle's `current`; its contents are never visible to readers.
    previous: Vec<T>,
}

impl<T> Events<T> {
    /// An empty channel.
    pub fn new() -> Self {
        Self {
            current: Vec::new(),
            previous: Vec::new(),
        }
    }

    /// Queue `event` for readers in the current cycle.
    pub fn send(&mut self, event: T) {
        self.current.push(event);
    }

    /// Borrow this cycle's events in send order. Non-consuming, so several readers
    /// can each see the full set.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.current.iter()
    }

    /// Take this cycle's events in send order, emptying the channel. Use when a
    /// single consumer owns the events (it must run after every sender); prefer
    /// [`iter`](Events::iter) when there are multiple readers.
    pub fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.current.drain(..)
    }

    /// End the cycle: retire the current buffer and recycle the previous one as the
    /// (empty) write buffer. Events sent before this call are no longer readable.
    pub fn update(&mut self) {
        std::mem::swap(&mut self.current, &mut self.previous);
        // `current` now holds the cycle-before-last's events; drop them, keeping the
        // allocation.
        self.current.clear();
    }

    /// Number of events readable this cycle.
    pub fn len(&self) -> usize {
        self.current.len()
    }

    /// Whether no event is readable this cycle.
    pub fn is_empty(&self) -> bool {
        self.current.is_empty()
    }

    /// Drop every queued event immediately (both buffers), e.g. on a level swap so
    /// stale events cannot cross the transition.
    pub fn clear(&mut self) {
        self.current.clear();
        self.previous.clear();
    }
}

impl<T> Default for Events<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Damage(u32);

    #[test]
    fn send_then_read_same_cycle() {
        let mut ev = Events::new();
        assert!(ev.is_empty());
        ev.send(Damage(1));
        ev.send(Damage(2));
        assert_eq!(ev.len(), 2);
        // Non-consuming: two readers see the same events, in send order.
        let a: Vec<Damage> = ev.iter().copied().collect();
        let b: Vec<Damage> = ev.iter().copied().collect();
        assert_eq!(a, vec![Damage(1), Damage(2)]);
        assert_eq!(a, b);
    }

    #[test]
    fn update_clears_after_exactly_one_cycle() {
        let mut ev = Events::new();
        ev.send(Damage(1));
        ev.update(); // end of the cycle that sent it
        assert!(ev.is_empty(), "events must not survive their update");
        assert_eq!(ev.iter().count(), 0);
        // A second update on an already-empty channel is harmless.
        ev.update();
        assert_eq!(ev.len(), 0);
    }

    #[test]
    fn buffers_swap_without_leaking_old_events() {
        let mut ev = Events::new();
        ev.send(Damage(1));
        ev.update();
        ev.send(Damage(2));
        // Only this cycle's event is visible — the swapped-in buffer was cleared.
        assert_eq!(ev.iter().copied().collect::<Vec<_>>(), vec![Damage(2)]);
        ev.update();
        ev.send(Damage(3));
        assert_eq!(ev.iter().copied().collect::<Vec<_>>(), vec![Damage(3)]);
    }

    #[test]
    fn drain_consumes_current_buffer() {
        let mut ev = Events::new();
        ev.send(Damage(1));
        ev.send(Damage(2));
        let taken: Vec<Damage> = ev.drain().collect();
        assert_eq!(taken, vec![Damage(1), Damage(2)]);
        assert!(ev.is_empty());
        // Draining does not disturb the cycle: a later send is still readable.
        ev.send(Damage(3));
        assert_eq!(ev.len(), 1);
    }

    #[test]
    fn clear_drops_both_buffers() {
        let mut ev = Events::new();
        ev.send(Damage(1));
        ev.update();
        ev.send(Damage(2));
        ev.clear();
        assert!(ev.is_empty());
        // The retired buffer was cleared too, so the next update cannot resurrect it.
        ev.update();
        assert!(ev.is_empty());
    }
}
