//! Terminal progress bars for the startup cook (asset → `.dcasset`, SDF/albedo bake, vgeo LOD DAG).
//!
//! Cooking runs on the [job system](dreamcoast_jobs) worker threads; a separate monitor thread
//! polls a shared atomic counter and renders progress so a long cold cook (Intel New Sponza:
//! hundreds of per-mesh SDF bakes + cluster DAGs) shows a live percentage instead of a silent
//! multi-minute stall. On a TTY it draws an in-place `\r` bar on stderr; when output is redirected
//! (headless capture, CI) it logs a line every 10% so the file isn't spammed with control codes.
//!
//! [`parallel_cook`] is the common entry point — a drop-in parallel `for` over a slice with a bar;
//! [`with_progress`] wraps any closure that increments the counter itself (for loops that need
//! custom scheduling, e.g. a dedup-then-bake pass).

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Below this item count a bar is just noise (the phase finishes before the first poll) — run the
/// work directly with no monitor.
const MIN_ITEMS_FOR_BAR: usize = 16;
const BAR_WIDTH: usize = 24;

/// Render one progress line. TTY → in-place `\r` bar on stderr; non-TTY → nothing (the caller logs
/// at deciles instead). `done` is clamped to `total`.
fn draw_bar(label: &str, done: usize, total: usize) {
    let done = done.min(total);
    let pct = if total == 0 { 100 } else { done * 100 / total };
    let filled = pct * BAR_WIDTH / 100;
    let bar: String = (0..BAR_WIDTH)
        .map(|i| if i < filled { '#' } else { '-' })
        .collect();
    let mut err = std::io::stderr();
    let _ = write!(err, "\r  cook: {label:<22} [{bar}] {pct:3}% ({done}/{total})");
    let _ = err.flush();
}

/// Spawn the monitor thread. It polls `counter` every 100 ms and renders until `stop` is set, then
/// draws a final 100% frame. Returns the join handle so [`with_progress`] can wait for the last
/// frame before returning (so the bar never lingers half-drawn).
fn spawn_monitor(
    label: String,
    total: usize,
    counter: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let tty = std::io::stderr().is_terminal();
        let mut last_decile = 0usize;
        loop {
            let done = counter.load(Ordering::Relaxed);
            if tty {
                draw_bar(&label, done, total);
            } else {
                let pct = if total == 0 { 100 } else { done.min(total) * 100 / total };
                let decile = pct / 10;
                if decile > last_decile {
                    last_decile = decile;
                    tracing::info!("cook {label}: {pct}% ({}/{total})", done.min(total));
                }
            }
            if stop.load(Ordering::Acquire) {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        if tty {
            draw_bar(&label, total, total);
            let _ = writeln!(std::io::stderr());
        } else {
            tracing::info!("cook {label}: done ({total}/{total})");
        }
    })
}

/// Run `work` (which increments the passed counter as items complete) while a monitor thread renders
/// a progress bar for `total` items. Small phases (`< MIN_ITEMS_FOR_BAR`) skip the monitor.
pub(crate) fn with_progress<F>(label: &str, total: usize, work: F)
where
    F: FnOnce(&AtomicUsize),
{
    if total < MIN_ITEMS_FOR_BAR {
        work(&AtomicUsize::new(0));
        return;
    }
    let counter = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let monitor = spawn_monitor(label.to_string(), total, Arc::clone(&counter), Arc::clone(&stop));
    work(&counter);
    stop.store(true, Ordering::Release);
    let _ = monitor.join();
}

/// Apply `f` to every element of `items` in parallel on the job-system workers, with a progress bar.
/// A drop-in for a sequential cook `for` loop where each item is independent (per-mesh SDF/albedo
/// bake, per-mesh cluster DAG). `grain` is the [`parallel_for`](dreamcoast_jobs::JobSystem::parallel_for)
/// chunk size (1 for coarse, expensive-per-item work).
pub(crate) fn parallel_cook<T, F>(label: &str, items: &mut [T], grain: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Send + Sync,
{
    with_progress(label, items.len(), |counter| {
        dreamcoast_jobs::global().parallel_for(items, grain, |i, item| {
            f(i, item);
            counter.fetch_add(1, Ordering::Relaxed);
        });
    });
}
