//! Bounded lock-free SPSC command ring (docs/game-audio-plan.md §2).
//!
//! The audio callback must never allocate or take a lock: `std::sync::mpsc` allocates
//! a node per send and frees it on the receiving (callback) side, so it fails the
//! invariant even though it "feels" lock-free. This ring is a fixed power-of-two
//! buffer with monotonic head/tail counters — the producer owns `tail`, the consumer
//! owns `head`, and each reads the other's counter with `Acquire` against its own
//! `Release` store, the textbook single-producer/single-consumer contract.
//!
//! Overflow policy is DROP (push returns `false`): a burst of game commands beyond the
//! ring capacity loses the newest commands rather than blocking the game thread or
//! growing without bound. The capacity is sized so that only a pathological spam frame
//! can hit it, and the drop count is the caller's `DIAG_AUDIO` warning signal.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Shared<T, const N: usize> {
    buf: UnsafeCell<[MaybeUninit<T>; N]>,
    /// Next slot the consumer will read. Owned by the consumer, `Release`-published.
    head: AtomicUsize,
    /// Next slot the producer will write. Owned by the producer, `Release`-published.
    tail: AtomicUsize,
}

// SAFETY: the buffer is only touched through the SPSC protocol below — the producer
// writes slot `tail` strictly before publishing `tail + 1`, the consumer reads slot
// `head` strictly after observing `tail > head` (Acquire), and each index is owned by
// exactly one side. `T: Send` because values cross threads by copy.
unsafe impl<T: Send + Copy, const N: usize> Sync for Shared<T, N> {}
unsafe impl<T: Send + Copy, const N: usize> Send for Shared<T, N> {}

/// Producer half — lives on the game thread.
pub struct Producer<T: Send + Copy, const N: usize>(Arc<Shared<T, N>>);
/// Consumer half — lives on the audio callback.
pub struct Consumer<T: Send + Copy, const N: usize>(Arc<Shared<T, N>>);

/// Build a ring. `N` must be a power of two (compile-time assert).
pub fn ring<T: Send + Copy, const N: usize>() -> (Producer<T, N>, Consumer<T, N>) {
    const { assert!(N.is_power_of_two()) };
    let shared = Arc::new(Shared {
        buf: UnsafeCell::new([const { MaybeUninit::uninit() }; N]),
        head: AtomicUsize::new(0),
        tail: AtomicUsize::new(0),
    });
    (Producer(shared.clone()), Consumer(shared))
}

impl<T: Send + Copy, const N: usize> Producer<T, N> {
    /// Push a command; `false` = ring full, command dropped (see the module policy).
    pub fn push(&mut self, v: T) -> bool {
        let s = &*self.0;
        let tail = s.tail.load(Ordering::Relaxed);
        if tail.wrapping_sub(s.head.load(Ordering::Acquire)) == N {
            return false;
        }
        // SAFETY: slot `tail % N` is outside the consumer's published window
        // (`head..tail`), and only this single producer writes slots.
        unsafe {
            (*s.buf.get())[tail % N].write(v);
        }
        s.tail.store(tail.wrapping_add(1), Ordering::Release);
        true
    }
}

impl<T: Send + Copy, const N: usize> Consumer<T, N> {
    /// Pop the oldest command, if any. Wait-free; no allocation.
    pub fn pop(&mut self) -> Option<T> {
        let s = &*self.0;
        let head = s.head.load(Ordering::Relaxed);
        if s.tail.load(Ordering::Acquire) == head {
            return None;
        }
        // SAFETY: `tail > head` (Acquire) guarantees the producer fully wrote this slot
        // before publishing, and only this single consumer reads slots.
        let v = unsafe { (*s.buf.get())[head % N].assume_init_read() };
        s.head.store(head.wrapping_add(1), Ordering::Release);
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_order_and_capacity() {
        let (mut tx, mut rx) = ring::<u32, 8>();
        for i in 0..8 {
            assert!(tx.push(i));
        }
        assert!(!tx.push(99), "ninth push must report full");
        for i in 0..8 {
            assert_eq!(rx.pop(), Some(i));
        }
        assert_eq!(rx.pop(), None);
        for round in 0..5u32 {
            for i in 0..6 {
                assert!(tx.push(round * 10 + i));
            }
            for i in 0..6 {
                assert_eq!(rx.pop(), Some(round * 10 + i));
            }
        }
    }

    #[test]
    fn cross_thread_stream() {
        let (mut tx, mut rx) = ring::<u64, 256>();
        let producer = std::thread::spawn(move || {
            let mut sent = 0u64;
            while sent < 10_000 {
                if tx.push(sent) {
                    sent += 1;
                }
            }
        });
        let mut expect = 0u64;
        while expect < 10_000 {
            if let Some(v) = rx.pop() {
                assert_eq!(v, expect);
                expect += 1;
            }
        }
        producer.join().unwrap();
    }
}
