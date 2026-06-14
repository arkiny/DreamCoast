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

use std::sync::Once;
use tracing_subscriber::{EnvFilter, fmt};

static LOGGING_INIT: Once = Once::new();

/// Initialize global `tracing` logging.
///
/// Idempotent: safe to call more than once (subsequent calls are no-ops). The
/// log level is read from the `RUST_LOG` environment variable, defaulting to
/// `info` when unset.
pub fn init_logging() {
    LOGGING_INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_thread_ids(false)
            .init();
    });
}
