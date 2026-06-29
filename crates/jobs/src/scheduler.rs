//! The work-stealing scheduler: a pool of worker threads each owning a
//! [`deque::Worker`], a shared injector queue for jobs submitted from non-worker
//! threads, and the [`JobSystem`] handle that drives them.
//!
//! A job spawned *from inside* a worker is pushed to that worker's own deque
//! (cache-hot, contention-free); idle workers steal across deques. Jobs submitted
//! from outside (the main thread) go to the injector. Idle workers park on a
//! condvar with a short timeout, so a missed wake-up self-heals within ~1ms
//! rather than deadlocking.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::deque::{self, Steal, Stealer};

/// A unit of work. `'static` because deques outlive any single submission; the
/// [`Scope`](crate::Scope) API re-introduces borrowed lifetimes safely on top.
pub type Job = Box<dyn FnOnce() + Send + 'static>;

/// State shared by the [`JobSystem`] and every worker thread.
pub(crate) struct Shared {
    pub(crate) stealers: Vec<Stealer<Job>>,
    pub(crate) injector: Mutex<VecDeque<Job>>,
    sleep: Mutex<()>,
    wake: Condvar,
    shutdown: AtomicBool,
}

impl Shared {
    /// Wake parked workers (called after new work is published).
    pub(crate) fn wake_all(&self) {
        // Hold the lock so a worker that just decided to park (but hasn't yet
        // waited) doesn't miss this notification.
        let _g = self.sleep.lock().unwrap();
        self.wake.notify_all();
    }

    /// Pull a job from the injector, then try to steal from every worker. Used by
    /// non-worker threads (e.g. the main thread draining a scope).
    pub(crate) fn find_global(&self) -> Option<Job> {
        if let Some(job) = self.injector.lock().unwrap().pop_front() {
            return Some(job);
        }
        for s in &self.stealers {
            loop {
                match s.steal() {
                    Steal::Success(job) => return Some(job),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        None
    }
}

/// Per-worker-thread context, referenced through a thread-local so a job running
/// on a worker can push child jobs to that worker's local deque.
pub(crate) struct WorkerThread {
    index: usize,
    local: deque::Worker<Job>,
    pub(crate) shared: Arc<Shared>,
}

thread_local! {
    static CONTEXT: std::cell::Cell<*const WorkerThread> = const { std::cell::Cell::new(std::ptr::null()) };
}

/// The current worker context, if the calling thread is a pool worker.
pub(crate) fn current_worker() -> Option<&'static WorkerThread> {
    let p = CONTEXT.with(|c| c.get());
    if p.is_null() {
        None
    } else {
        // SAFETY: the pointer was set to a `WorkerThread` living for the whole
        // worker-loop call and cleared before the loop returns; it is only ever
        // read on that same thread.
        Some(unsafe { &*p })
    }
}

impl WorkerThread {
    /// Push a job to this worker's local deque.
    pub(crate) fn push(&self, job: Job) {
        self.local.push(job);
    }

    /// Find one job: local first (LIFO, hot), then the injector, then steal.
    fn find(&self) -> Option<Job> {
        if let Some(job) = self.local.pop() {
            return Some(job);
        }
        if let Some(job) = self.shared.injector.lock().unwrap().pop_front() {
            return Some(job);
        }
        let n = self.shared.stealers.len();
        for offset in 1..=n {
            let idx = (self.index + offset) % n;
            loop {
                match self.shared.stealers[idx].steal() {
                    Steal::Success(job) => return Some(job),
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }
        None
    }

    fn run_loop(&self) {
        CONTEXT.with(|c| c.set(self as *const _));
        while !self.shared.shutdown.load(Ordering::Acquire) {
            if let Some(job) = self.find() {
                job();
            } else {
                // Nothing to do: park briefly. The timeout bounds the cost of any
                // missed wake-up to ~1ms instead of risking a lost-wakeup hang.
                let guard = self.shared.sleep.lock().unwrap();
                if !self.shared.shutdown.load(Ordering::Acquire) {
                    let _ = self
                        .shared
                        .wake
                        .wait_timeout(guard, Duration::from_millis(1));
                }
            }
        }
        CONTEXT.with(|c| c.set(std::ptr::null()));
    }
}

/// A handle to a running work-stealing thread pool.
pub struct JobSystem {
    pub(crate) shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl JobSystem {
    /// Start a pool. `num_threads` is the count of *worker* threads; pass `None`
    /// to default to `available_parallelism() - 1` (the calling thread is the
    /// other participant). Always clamped to at least 1.
    pub fn new(num_threads: Option<usize>) -> Self {
        let n = num_threads
            .unwrap_or_else(|| {
                thread::available_parallelism()
                    .map(|p| p.get().saturating_sub(1))
                    .unwrap_or(1)
            })
            .max(1);

        // Build all deques first so every worker can hold every other's stealer.
        let workers: Vec<deque::Worker<Job>> = (0..n).map(|_| deque::new::<Job>()).collect();
        let stealers: Vec<Stealer<Job>> = workers.iter().map(|w| w.stealer()).collect();

        let shared = Arc::new(Shared {
            stealers,
            injector: Mutex::new(VecDeque::new()),
            sleep: Mutex::new(()),
            wake: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });

        let mut threads = Vec::with_capacity(n);
        for (index, local) in workers.into_iter().enumerate() {
            let shared = Arc::clone(&shared);
            threads.push(
                thread::Builder::new()
                    .name(format!("dc-worker-{index}"))
                    .spawn(move || {
                        let wt = WorkerThread {
                            index,
                            local,
                            shared,
                        };
                        wt.run_loop();
                    })
                    .expect("spawn worker thread"),
            );
        }

        JobSystem { shared, threads }
    }

    /// Number of worker threads.
    pub fn num_workers(&self) -> usize {
        self.shared.stealers.len()
    }

    /// Submit a fire-and-forget job. Completion is not tracked — use
    /// [`JobSystem::scope`] or [`JobSystem::parallel_for`] when you need to wait.
    pub fn spawn<F: FnOnce() + Send + 'static>(&self, f: F) {
        submit(&self.shared, Box::new(f));
    }

    /// Run `f`, which may spawn child jobs into the passed [`Scope`]; all such
    /// jobs are guaranteed complete before this returns. The calling thread helps
    /// execute pending work while waiting, so a scope never blocks the whole pool.
    pub fn scope<'scope, F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Scope<'scope>) -> R,
    {
        let scope = Scope {
            shared: Arc::clone(&self.shared),
            pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            _marker: std::marker::PhantomData,
        };
        let result = f(&scope);
        scope.wait();
        result
    }

    /// Apply `f` to every element of `items` in parallel, split into chunks of
    /// `grain` (clamped to ≥1). Returns after all chunks finish.
    pub fn parallel_for<T, F>(&self, items: &mut [T], grain: usize, f: F)
    where
        T: Send,
        F: Fn(usize, &mut T) + Send + Sync,
    {
        let grain = grain.max(1);
        let f = &f;
        self.scope(|s| {
            let mut base = 0usize;
            for chunk in items.chunks_mut(grain) {
                let start = base;
                base += chunk.len();
                s.spawn(move |_| {
                    for (i, item) in chunk.iter_mut().enumerate() {
                        f(start + i, item);
                    }
                });
            }
        });
    }
}

impl Drop for JobSystem {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.wake_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

/// Route a job to the current worker's local deque (if on a worker) or the
/// shared injector, then wake a sleeper.
fn submit(shared: &Arc<Shared>, job: Job) {
    if let Some(wt) = current_worker()
        && Arc::ptr_eq(&wt.shared, shared)
    {
        wt.push(job);
    } else {
        shared.injector.lock().unwrap().push_back(job);
    }
    shared.wake_all();
}

/// A structured-concurrency scope. Jobs spawned here borrow from `'scope` and are
/// all joined before [`JobSystem::scope`] returns.
pub struct Scope<'scope> {
    shared: Arc<Shared>,
    pending: Arc<std::sync::atomic::AtomicUsize>,
    // Invariant over 'scope so borrowed captures can't outlive the scope.
    _marker: std::marker::PhantomData<&'scope mut &'scope ()>,
}

// SAFETY: all fields are Send+Sync (Arc + atomic); the PhantomData carries only a
// lifetime. Sharing `&Scope` across worker threads is what lets nested jobs spawn.
unsafe impl Sync for Scope<'_> {}

impl<'scope> Scope<'scope> {
    /// Spawn a job that may borrow data living at least as long as `'scope`. The
    /// job receives a `&Scope` so it can spawn further child jobs (the scope
    /// reference is reconstructed per job rather than captured, which keeps the
    /// borrow checker happy for recursive scheduling).
    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce(&Scope<'scope>) + Send + 'scope,
    {
        self.pending.fetch_add(1, Ordering::Relaxed);
        let shared = Arc::clone(&self.shared);
        let pending = Arc::clone(&self.pending);
        let body: Box<dyn FnOnce() + Send + 'scope> = Box::new(move || {
            let scope = Scope {
                shared: Arc::clone(&shared),
                pending: Arc::clone(&pending),
                _marker: std::marker::PhantomData,
            };
            f(&scope);
            pending.fetch_sub(1, Ordering::Release);
        });
        // SAFETY: we extend the job's lifetime to 'static only for storage in the
        // deque. `wait()` below does not return until `pending` hits zero, i.e.
        // every such job has run, so no job outlives the borrowed `'scope` data.
        let job: Job = unsafe {
            std::mem::transmute::<Box<dyn FnOnce() + Send + 'scope>, Box<dyn FnOnce() + Send>>(body)
        };
        submit(&self.shared, job);
    }

    /// Block until every job spawned into this scope has completed, executing
    /// pending work on the calling thread meanwhile (cooperative — never idles
    /// the caller while jobs remain).
    fn wait(&self) {
        while self.pending.load(Ordering::Acquire) != 0 {
            if let Some(job) = self.find_one() {
                job();
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Find one job to help with: prefer the calling worker's own deque, else go
    /// global. Mirrors the worker loop's order for locality.
    fn find_one(&self) -> Option<Job> {
        if let Some(wt) = current_worker()
            && Arc::ptr_eq(&wt.shared, &self.shared)
            && let Some(job) = wt.local.pop()
        {
            return Some(job);
        }
        self.shared.find_global()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn parallel_for_touches_every_element() {
        let js = JobSystem::new(Some(4));
        let mut data: Vec<usize> = (0..10_000).collect();
        js.parallel_for(&mut data, 64, |_, x| *x *= 2);
        for (i, x) in data.iter().enumerate() {
            assert_eq!(*x, i * 2);
        }
    }

    #[test]
    fn scope_joins_all_children() {
        let js = JobSystem::new(Some(4));
        let counter = AtomicUsize::new(0);
        js.scope(|s| {
            for _ in 0..1000 {
                s.spawn(|_| {
                    counter.fetch_add(1, Ordering::Relaxed);
                });
            }
        });
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn nested_scopes_and_spawns() {
        let js = JobSystem::new(Some(3));
        let counter = AtomicUsize::new(0);
        js.scope(|s| {
            for _ in 0..10 {
                s.spawn(|s| {
                    // Nested spawning from inside a job, via the passed scope.
                    s.spawn(|_| {
                        counter.fetch_add(1, Ordering::Relaxed);
                    });
                    counter.fetch_add(1, Ordering::Relaxed);
                });
            }
        });
        assert_eq!(counter.load(Ordering::Relaxed), 20);
    }

    #[test]
    fn single_worker_still_makes_progress() {
        // 1 worker: the calling thread must cooperatively drain, no deadlock.
        let js = JobSystem::new(Some(1));
        let sum = AtomicUsize::new(0);
        js.scope(|s| {
            let sum = &sum;
            for i in 0..500 {
                s.spawn(move |_| {
                    sum.fetch_add(i, Ordering::Relaxed);
                });
            }
        });
        assert_eq!(sum.load(Ordering::Relaxed), (0..500).sum::<usize>());
    }
}
