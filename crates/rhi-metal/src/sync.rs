//! Metal synchronization primitives.
//!
//! - [`MetalFence`] is a CPU↔GPU fence: it remembers the command buffer a submit
//!   committed and blocks on `waitUntilCompleted`. This is the simplest correct
//!   mapping; an `MTLSharedEvent`-based version can replace it if finer-grained
//!   signaling is needed.
//! - [`MetalSemaphore`] is a GPU↔GPU ordering token. Within a single Metal queue
//!   submission order already guarantees ordering, so for the single-queue path it
//!   is a no-op (like the D3D12 backend's semaphore). Real cross-queue events
//!   arrive with async compute in M5.

use std::cell::RefCell;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;

use crate::Result;

/// A CPU-GPU fence backed by the committed command buffer's completion.
pub struct MetalFence {
    cmd: RefCell<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
}

impl MetalFence {
    pub(crate) fn new(_signaled: bool) -> Self {
        // No stored command buffer == already signaled (`wait` returns at once).
        Self {
            cmd: RefCell::new(None),
        }
    }

    /// Attach the command buffer a submit just committed.
    pub(crate) fn set(&self, cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>) {
        *self.cmd.borrow_mut() = Some(cmd);
    }

    pub fn wait(&self) -> Result<()> {
        if let Some(cmd) = self.cmd.borrow().as_ref() {
            cmd.waitUntilCompleted();
        }
        Ok(())
    }

    pub fn reset(&self) -> Result<()> {
        *self.cmd.borrow_mut() = None;
        Ok(())
    }
}

/// A GPU-GPU ordering token (no-op on the single-queue path).
pub struct MetalSemaphore;

impl MetalSemaphore {
    pub(crate) fn new() -> Self {
        Self
    }
}
