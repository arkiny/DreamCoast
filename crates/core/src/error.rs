//! Top-level engine error type.
//!
//! Backend- and subsystem-specific errors (RHI, shader, asset) will be added as
//! variants as those crates come online in later phases.

use thiserror::Error;

/// The crate-wide error type for engine operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EngineError {
    /// A platform / OS-level operation failed (windowing, input, etc.).
    #[error("platform error: {0}")]
    Platform(String),

    /// Shader loading or compilation failed.
    #[error("shader error: {0}")]
    Shader(String),

    /// A graphics backend (RHI) operation failed. Wired up in Phase 1.
    #[error("rhi error: {0}")]
    Rhi(String),
}
