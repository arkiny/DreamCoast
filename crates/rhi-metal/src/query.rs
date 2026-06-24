//! Timestamp query heap (Phase 9 M1) — stub.
//!
//! Metal GPU timestamp sampling (`MTLCounterSampleBuffer`) is not yet wired up on
//! this in-progress backend. This stub keeps the cross-backend profiling API
//! compiling and runnable on macOS; [`Self::read`] returns zero ticks (the
//! sandbox profiler shows 0 ms on Metal until real sampling lands).

use crate::Result;

/// A stub timestamp query heap of `count` slots.
pub struct MetalQueryHeap {
    count: u32,
}

impl MetalQueryHeap {
    pub(crate) fn new(count: u32) -> Result<Self> {
        Ok(Self { count })
    }

    /// Number of timestamp slots.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Nanoseconds per tick — 0 until real sampling is implemented.
    pub fn period_ns(&self) -> f32 {
        0.0
    }

    /// Zero ticks (no real timestamps yet).
    pub fn read(&self) -> Vec<u64> {
        vec![0; self.count as usize]
    }
}
