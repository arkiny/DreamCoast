//! Progress reporting for the startup cook (asset → `.dcasset`, SDF/albedo bake, vgeo LOD DAG).
//!
//! Cooking runs on the [job system](dreamcoast_jobs) worker threads; the CALLING thread stays free
//! to report progress. [`parallel_cook`] runs the per-item work on a scoped background thread (which
//! fans the items out across the job pool) while the caller drives a [`ProgressSink`] every ~33 ms —
//! so the main thread can pump the window + draw an ImGui loading frame instead of freezing at
//! "Not Responding" for a multi-minute cold cook (Intel New Sponza: hundreds of per-mesh SDF bakes +
//! cluster DAGs).
//!
//! The sink is injected so headless/CI callers get a plain terminal bar ([`TermProgress`]) while an
//! interactive run gets the graphical [`crate::loading::LoadingScreen`]. Both see the same
//! `(label, done, total)` ticks.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// Below this item count the phase finishes before the first tick — run it inline, no reporting.
const MIN_ITEMS_FOR_BAR: usize = 16;
const BAR_WIDTH: usize = 24;

/// A progress consumer, ticked on the cook's driving thread. `done`/`total` are item counts.
pub(crate) trait ProgressSink {
    fn tick(&mut self, label: &str, done: usize, total: usize);
}

/// Terminal sink (headless / CI default): a TTY gets an in-place `\r` bar on stderr; redirected
/// output gets a `tracing` line every 10 % (so a log file isn't spammed with control codes).
pub(crate) struct TermProgress {
    tty: bool,
    last_decile: usize,
}

impl TermProgress {
    pub(crate) fn new() -> Self {
        Self {
            tty: std::io::stderr().is_terminal(),
            last_decile: 0,
        }
    }
}

impl ProgressSink for TermProgress {
    fn tick(&mut self, label: &str, done: usize, total: usize) {
        let done = done.min(total);
        let pct = if total == 0 { 100 } else { done * 100 / total };
        if self.tty {
            let filled = pct * BAR_WIDTH / 100;
            let bar: String = (0..BAR_WIDTH)
                .map(|i| if i < filled { '#' } else { '-' })
                .collect();
            let mut err = std::io::stderr();
            let _ = write!(
                err,
                "\r  cook: {label:<22} [{bar}] {pct:3}% ({done}/{total})"
            );
            if done == total {
                let _ = writeln!(err);
            }
            let _ = err.flush();
        } else {
            let decile = pct / 10;
            if decile > self.last_decile || done == total {
                self.last_decile = decile;
                tracing::info!("cook {label}: {pct}% ({done}/{total})");
            }
        }
    }
}

/// Apply `f` to every element of `items` in parallel on the job-system workers, reporting to `sink`.
/// A drop-in for a sequential cook `for` loop where each item is independent (per-mesh SDF/albedo
/// bake, per-mesh cluster DAG). The work runs on a scoped thread (so the caller can render a loading
/// frame between ticks); `grain` is the [`parallel_for`](dreamcoast_jobs::JobSystem::parallel_for)
/// chunk size (1 for coarse, expensive-per-item work).
pub(crate) fn parallel_cook<T, F>(
    label: &str,
    items: &mut [T],
    grain: usize,
    f: F,
    sink: &mut dyn ProgressSink,
) where
    T: Send,
    F: Fn(usize, &mut T) + Send + Sync,
{
    let total = items.len();
    if total < MIN_ITEMS_FOR_BAR {
        dreamcoast_jobs::global().parallel_for(items, grain, |i, item| f(i, item));
        return;
    }
    let counter = AtomicUsize::new(0);
    // `std::thread::scope` lets the worker borrow `items`/`f`/`counter` (no 'static needed) while the
    // owner thread stays free to tick the sink. The worker calls `parallel_for`, so all pool workers
    // (plus this scoped thread) do the actual cooking.
    std::thread::scope(|s| {
        let counter = &counter;
        let f = &f;
        let worker = s.spawn(move || {
            dreamcoast_jobs::global().parallel_for(items, grain, |i, item| {
                f(i, item);
                counter.fetch_add(1, Ordering::Relaxed);
            });
        });
        while !worker.is_finished() {
            sink.tick(label, counter.load(Ordering::Relaxed), total);
            std::thread::sleep(Duration::from_millis(33));
        }
        let _ = worker.join();
        sink.tick(label, total, total);
    });
}
