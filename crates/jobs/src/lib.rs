//! `dreamcoast-jobs` — the engine's **work-stealing job system**, the CPU-side
//! spine of multithreading (as the render graph is the GPU-pass spine).
//!
//! Layers, bottom-up:
//! - [`deque`] — a from-scratch lock-free Chase-Lev work-stealing deque (the only
//!   unsafe-heavy module).
//! - [`JobSystem`] — a pool of worker threads over those deques, plus a
//!   [`Scope`] for structured `'scope`-borrowed parallelism and `parallel_for`.
//! - [`TaskGraph`] — dependency-ordered execution of jobs (CPU render graph).
//!
//! **Determinism.** Steal order and root scheduling are insertion-ordered, and
//! reductions must be order-independent, so headless runs reproduce regardless of
//! worker count (see the engine determinism rule).
//!
//! RHI-independent: depends only on `std` (+ `dreamcoast-core` for logging
//! conventions). ECS system scheduling and parallel RHI recording build on top.

mod deque;
mod graph;
mod scheduler;

pub use graph::{TaskGraph, TaskId};
pub use scheduler::{Job, JobSystem, Scope};

use std::sync::OnceLock;

/// Process-wide job system, for code that wants a shared pool without threading a
/// handle through every call. Initialised on first [`global`] access (or via
/// [`init_global`]).
static GLOBAL: OnceLock<JobSystem> = OnceLock::new();

/// Initialise the global [`JobSystem`] with a specific worker count. Returns
/// `false` if it was already initialised (the existing pool is kept).
pub fn init_global(num_threads: Option<usize>) -> bool {
    GLOBAL.set(JobSystem::new(num_threads)).is_ok()
}

/// The global [`JobSystem`], created with the default worker count on first use.
pub fn global() -> &'static JobSystem {
    GLOBAL.get_or_init(|| JobSystem::new(None))
}
