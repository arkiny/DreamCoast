//! Shared engine infrastructure: logging, error types, and resource handles.
//!
//! This crate carries no GPU dependencies; every other crate builds on it. In
//! particular it re-exports [`glam`] so the whole workspace shares one math
//! library version.

pub use glam;

mod error;
mod pool;

pub use error::EngineError;
pub use pool::{Handle, Pool};

use std::fs::File;
use std::sync::Once;
use tracing_subscriber::fmt::writer::MakeWriterExt;
use tracing_subscriber::{EnvFilter, fmt};

static LOGGING_INIT: Once = Once::new();

/// Initialize global `tracing` logging.
///
/// Idempotent: safe to call more than once (subsequent calls are no-ops). The
/// log level is read from the `RUST_LOG` environment variable, defaulting to
/// `info` when unset.
///
/// If `DREAMCOAST_LOG_FILE` is set, every log line is mirrored to that file (in
/// addition to stdout). GPU capture tools such as RenderDoc launch the app with
/// stdout/stderr redirected away, so a crash or returned error would otherwise
/// be invisible; the file sink (combined with the panic hook installed here)
/// gives a postmortem of the real failure. A panic message is also written via
/// `tracing::error!`, so it lands in the file too.
pub fn init_logging() {
    LOGGING_INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

        if let Some(path) = std::env::var_os("DREAMCOAST_LOG_FILE") {
            match File::create(&path) {
                Ok(file) => {
                    // Tee stdout + the file. The closure re-`try_clone`s the file
                    // handle per write (the `MakeWriter` contract); no ANSI codes so
                    // the file stays plain text.
                    let writer = std::io::stdout
                        .and(move || file.try_clone().expect("clone DREAMCOAST_LOG_FILE handle"));
                    fmt()
                        .with_env_filter(filter)
                        .with_target(true)
                        .with_thread_ids(false)
                        .with_ansi(false)
                        .with_writer(writer)
                        .init();
                    install_panic_hook();
                    tracing::info!("logging to file: {}", path.to_string_lossy());
                    return;
                }
                Err(e) => {
                    eprintln!("could not open DREAMCOAST_LOG_FILE: {e}; logging to stdout only");
                }
            }
        }

        fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_thread_ids(false)
            .init();
        install_panic_hook();
    });
}

/// Route panics through `tracing::error!` (so they reach the log file) while
/// preserving the default hook's stderr backtrace.
fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        tracing::error!("panic: {info}");
        default(info);
    }));
}
