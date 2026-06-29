//! A from-scratch **Chase-Lev work-stealing deque** (lock-free).
//!
//! This is the single concurrency primitive every higher layer (worker pool,
//! scope, task graph) is built on, and the **only unsafe-heavy module** in the
//! crate — deliberately isolated here with focused tests.
//!
//! Topology: one *owner* thread holds a [`Worker`] and does `push`/`pop` at the
//! **bottom** (LIFO, cache-hot, no contention). Any number of *thief* threads
//! hold cloned [`Stealer`]s and `steal` from the **top** (FIFO). Owner and
//! thieves only race on the very last element, resolved by a single CAS on the
//! top index.
//!
//! The algorithm + memory orderings follow Lê, Pop, Cohen & Nardelli,
//! *"Correct and Efficient Work-Stealing for Weak Memory Models"* (PPoPP'13).
//!
//! **Reclamation.** `grow` retires the old backing buffer instead of freeing it,
//! because a thief may still be mid-`steal` reading from it. Retired buffers (and
//! the live one) are freed only when the whole deque is dropped, at which point no
//! thread is concurrently accessing it.

use std::alloc::{Layout, alloc, dealloc};
use std::cell::UnsafeCell;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicIsize, AtomicPtr, Ordering, fence};

/// Result of a [`Stealer::steal`] attempt.
pub enum Steal<T> {
    /// A job was taken.
    Success(T),
    /// The deque was observed empty.
    Empty,
    /// Lost a race (the owner or another thief took the slot); caller may retry.
    Retry,
}

/// A power-of-two ring buffer of `T`, addressed by a monotonically increasing
/// index masked into range. Holds no length of its own — the deque's `top`/
/// `bottom` indices define which slots are live.
struct Buffer<T> {
    ptr: *mut T,
    cap: usize, // always a power of two
}

impl<T> Buffer<T> {
    fn alloc(cap: usize) -> *mut Buffer<T> {
        debug_assert!(cap.is_power_of_two());
        let layout = Layout::array::<T>(cap).expect("deque buffer layout");
        // SAFETY: cap > 0 so the layout is non-zero; we check the result for null.
        let ptr = unsafe { alloc(layout) } as *mut T;
        assert!(!ptr.is_null(), "deque buffer allocation failed");
        Box::into_raw(Box::new(Buffer { ptr, cap }))
    }

    #[inline]
    unsafe fn slot(&self, index: isize) -> *mut T {
        // Masking with cap-1 is the modulo for a power-of-two capacity.
        unsafe { self.ptr.add((index as usize) & (self.cap - 1)) }
    }

    #[inline]
    unsafe fn write(&self, index: isize, value: T) {
        unsafe { ptr::write(self.slot(index), value) }
    }

    /// Bitwise read of a slot. The caller is responsible for ensuring exactly one
    /// consumer ever *commits* (keeps) any given slot's value; a racing reader
    /// that loses its CAS must `mem::forget` the returned value so it is not
    /// double-dropped.
    #[inline]
    unsafe fn read(&self, index: isize) -> T {
        unsafe { ptr::read(self.slot(index)) }
    }

    /// Free the backing allocation. Does **not** drop any contained `T`.
    unsafe fn free(raw: *mut Buffer<T>) {
        // SAFETY: `raw` came from `Box::into_raw(Buffer)`; reconstitute and drop
        // the box, then free the inner array allocation it pointed at.
        let b = unsafe { Box::from_raw(raw) };
        let layout = Layout::array::<T>(b.cap).expect("deque buffer layout");
        unsafe { dealloc(b.ptr as *mut u8, layout) }
    }
}

/// State shared by the single [`Worker`] and all its [`Stealer`]s.
struct Inner<T> {
    /// FIFO steal end. Only ever increased (by the owner on the last-element race,
    /// or by a successful thief).
    top: AtomicIsize,
    /// LIFO push/pop end, owned exclusively by the [`Worker`] thread.
    bottom: AtomicIsize,
    /// The live backing buffer; swapped on growth.
    buffer: AtomicPtr<Buffer<T>>,
    /// Buffers retired by `grow`, kept alive until the deque is dropped so an
    /// in-flight `steal` never reads freed memory. Only contended on growth/drop.
    retired: Mutex<Vec<*mut Buffer<T>>>,
}

// SAFETY: access to `buffer`/`top`/`bottom` is synchronised by the atomics and the
// Chase-Lev protocol; `retired` is behind a Mutex. `T` only moves between threads
// through the deque, so `T: Send` is the only requirement.
unsafe impl<T: Send> Send for Inner<T> {}
unsafe impl<T: Send> Sync for Inner<T> {}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        // No concurrent access at drop: drop the still-live elements [top, bottom)
        // then free every buffer (live + retired).
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        let live = self.buffer.load(Ordering::Relaxed);
        // SAFETY: exclusive access; `live` is a valid Buffer pointer.
        let buf = unsafe { &*live };
        let mut i = t;
        while i < b {
            // SAFETY: slots in [top, bottom) hold initialised values we now own.
            unsafe { ptr::drop_in_place(buf.slot(i)) };
            i += 1;
        }
        // SAFETY: each pointer was produced by Buffer::alloc and is freed once.
        unsafe { Buffer::free(live) };
        for raw in self.retired.lock().unwrap().drain(..) {
            unsafe { Buffer::free(raw) };
        }
    }
}

/// The owner endpoint: single-threaded `push`/`pop` at the bottom. `!Sync` — only
/// the owning thread may touch it — but `Send` so it can be moved onto a worker
/// thread.
pub struct Worker<T> {
    inner: Arc<Inner<T>>,
    // Make Worker !Sync (it owns the bottom index exclusively).
    _not_sync: UnsafeCell<()>,
}

// SAFETY: a Worker can be moved to another thread (Send) as long as T is Send; it
// must never be shared (no Sync impl — provided by the UnsafeCell field).
unsafe impl<T: Send> Send for Worker<T> {}

/// A thief endpoint: cloneable, `Send + Sync`, `steal`s from the top.
pub struct Stealer<T> {
    inner: Arc<Inner<T>>,
}

unsafe impl<T: Send> Send for Stealer<T> {}
unsafe impl<T: Send> Sync for Stealer<T> {}

impl<T> Clone for Stealer<T> {
    fn clone(&self) -> Self {
        Stealer {
            inner: Arc::clone(&self.inner),
        }
    }
}

const MIN_CAP: usize = 16;

/// Create a fresh deque, returning its owner [`Worker`]. Call [`Worker::stealer`]
/// to mint thief handles.
pub fn new<T>() -> Worker<T> {
    let inner = Arc::new(Inner {
        top: AtomicIsize::new(0),
        bottom: AtomicIsize::new(0),
        buffer: AtomicPtr::new(Buffer::<T>::alloc(MIN_CAP)),
        retired: Mutex::new(Vec::new()),
    });
    Worker {
        inner,
        _not_sync: UnsafeCell::new(()),
    }
}

impl<T> Worker<T> {
    /// Mint a thief handle that steals from this deque.
    pub fn stealer(&self) -> Stealer<T> {
        Stealer {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Push a value onto the bottom (owner only). Grows the buffer if full.
    pub fn push(&self, value: T) {
        let inner = &self.inner;
        let b = inner.bottom.load(Ordering::Relaxed);
        let t = inner.top.load(Ordering::Acquire);
        let mut buf = inner.buffer.load(Ordering::Relaxed);
        // SAFETY: `buf` is the live buffer; only the owner mutates it.
        let cap = unsafe { (*buf).cap } as isize;
        if b - t > cap - 1 {
            // Full: grow, retiring the old buffer (a thief may still read it).
            buf = self.grow(buf, t, b);
        }
        // SAFETY: slot is within the (possibly grown) buffer; owner-exclusive write.
        unsafe { (*buf).write(b, value) };
        // Publish the value before the index that exposes it to thieves.
        fence(Ordering::Release);
        inner.bottom.store(b + 1, Ordering::Relaxed);
    }

    /// Pop a value from the bottom (owner only), or `None` if empty.
    pub fn pop(&self) -> Option<T> {
        let inner = &self.inner;
        let b = inner.bottom.load(Ordering::Relaxed) - 1;
        let buf = inner.buffer.load(Ordering::Relaxed);
        inner.bottom.store(b, Ordering::Relaxed);
        // Order the bottom store before the top load: without this a concurrent
        // thief and the owner could both believe they took the last element.
        fence(Ordering::SeqCst);
        let t = inner.top.load(Ordering::Relaxed);

        if t <= b {
            // Non-empty.
            // SAFETY: slot b holds an initialised value.
            let value = unsafe { (*buf).read(b) };
            if t == b {
                // Last element: race a thief for it via CAS on top.
                if inner
                    .top
                    .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
                    .is_err()
                {
                    // Lost: the thief took it. Our read was a bitwise copy — forget
                    // it so the value isn't dropped twice.
                    mem::forget(value);
                    inner.bottom.store(b + 1, Ordering::Relaxed);
                    return None;
                }
                // Won the last element; reset to the canonical empty state.
                inner.bottom.store(b + 1, Ordering::Relaxed);
            }
            Some(value)
        } else {
            // Empty: restore bottom.
            inner.bottom.store(b + 1, Ordering::Relaxed);
            None
        }
    }

    /// Double the buffer, copying live elements [t, b) into it, and retire the old
    /// buffer for deferred reclamation. Returns the new live buffer pointer.
    #[cold]
    fn grow(&self, old: *mut Buffer<T>, t: isize, b: isize) -> *mut Buffer<T> {
        // SAFETY: owner-exclusive; `old` is the live buffer.
        let old_cap = unsafe { (*old).cap };
        let new = Buffer::<T>::alloc(old_cap * 2);
        let mut i = t;
        while i < b {
            // SAFETY: copy each live slot bitwise from old to new; ownership moves
            // with the buffer swap, so neither side drops.
            unsafe {
                let v = (*old).read(i);
                (*new).write(i, v);
            }
            i += 1;
        }
        self.inner.buffer.store(new, Ordering::Release);
        self.inner.retired.lock().unwrap().push(old);
        new
    }
}

impl<T> Stealer<T> {
    /// Attempt to steal one value from the top. Returns [`Steal::Empty`] if the
    /// deque looked empty or [`Steal::Retry`] if a race was lost.
    pub fn steal(&self) -> Steal<T> {
        let inner = &self.inner;
        let t = inner.top.load(Ordering::Acquire);
        // Ensure the top load is ordered before the bottom load (mirror of pop's
        // fence) so we observe a consistent [top, bottom) window.
        fence(Ordering::SeqCst);
        let b = inner.bottom.load(Ordering::Acquire);

        if t < b {
            // Non-empty: read then try to claim the slot by advancing top.
            let buf = inner.buffer.load(Ordering::Acquire);
            // SAFETY: slot t is initialised and within the buffer.
            let value = unsafe { (*buf).read(t) };
            if inner
                .top
                .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                Steal::Success(value)
            } else {
                // Lost the race: forget our bitwise copy and let the caller retry.
                mem::forget(value);
                Steal::Retry
            }
        } else {
            Steal::Empty
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    #[test]
    fn push_pop_lifo() {
        let w = new::<i32>();
        for i in 0..1000 {
            w.push(i);
        }
        for i in (0..1000).rev() {
            assert_eq!(w.pop(), Some(i));
        }
        assert!(w.pop().is_none());
    }

    #[test]
    fn grows_past_initial_capacity() {
        let w = new::<usize>();
        let n = MIN_CAP * 8 + 3;
        for i in 0..n {
            w.push(i);
        }
        let mut seen = 0;
        while w.pop().is_some() {
            seen += 1;
        }
        assert_eq!(seen, n);
    }

    #[test]
    fn steal_fifo_from_other_thread() {
        let w = new::<i32>();
        let s = w.stealer();
        for i in 0..100 {
            w.push(i);
        }
        // FIFO: a single thief with no contention sees 0,1,2,... in order.
        let h = thread::spawn(move || {
            let mut got = Vec::new();
            while got.len() < 100 {
                match s.steal() {
                    Steal::Success(v) => got.push(v),
                    Steal::Empty => break,
                    Steal::Retry => continue,
                }
            }
            got
        });
        let got = h.join().unwrap();
        assert_eq!(got, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn concurrent_owner_and_thieves_lose_nothing() {
        // Owner pushes N, four thieves + the owner drain; every value taken
        // exactly once (counted), none lost or duplicated.
        const N: usize = 100_000;
        let w = new::<usize>();
        let taken = Arc::new(AtomicUsize::new(0));
        let sum = Arc::new(AtomicUsize::new(0));

        let mut thieves = Vec::new();
        for _ in 0..4 {
            let s = w.stealer();
            let taken = Arc::clone(&taken);
            let sum = Arc::clone(&sum);
            thieves.push(thread::spawn(move || {
                loop {
                    match s.steal() {
                        Steal::Success(v) => {
                            taken.fetch_add(1, Ordering::Relaxed);
                            sum.fetch_add(v, Ordering::Relaxed);
                        }
                        Steal::Empty => {
                            if taken.load(Ordering::Relaxed) >= N {
                                break;
                            }
                        }
                        Steal::Retry => {}
                    }
                }
            }));
        }

        for i in 0..N {
            w.push(i);
            // Interleave owner pops to exercise the last-element race.
            if i % 3 == 0
                && let Some(v) = w.pop()
            {
                taken.fetch_add(1, Ordering::Relaxed);
                sum.fetch_add(v, Ordering::Relaxed);
            }
        }
        while let Some(v) = w.pop() {
            taken.fetch_add(1, Ordering::Relaxed);
            sum.fetch_add(v, Ordering::Relaxed);
        }
        for h in thieves {
            h.join().unwrap();
        }

        assert_eq!(taken.load(Ordering::Relaxed), N, "every value taken once");
        assert_eq!(
            sum.load(Ordering::Relaxed),
            (0..N).sum::<usize>(),
            "no value lost or duplicated"
        );
    }

    #[test]
    fn drops_remaining_elements_once() {
        // Values that count their own drops; leftover deque contents must drop
        // exactly once at deque drop (no leak, no double free).
        struct Dropper(Arc<AtomicUsize>);
        impl Drop for Dropper {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        {
            let w = new::<Dropper>();
            for _ in 0..50 {
                w.push(Dropper(Arc::clone(&drops)));
            }
            // Take a few; the rest drop with the deque.
            for _ in 0..10 {
                drop(w.pop());
            }
            assert_eq!(drops.load(Ordering::Relaxed), 10);
        }
        assert_eq!(drops.load(Ordering::Relaxed), 50);
    }
}
