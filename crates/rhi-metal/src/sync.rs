//! Metal synchronization primitives.
//!
//! - [`MetalFence`] is a CPU↔GPU fence: it remembers the command buffer a submit
//!   committed and blocks on `waitUntilCompleted`. This is the simplest correct
//!   mapping; an `MTLSharedEvent`-based version can replace it if finer-grained
//!   signaling is needed.
//! - [`MetalSemaphore`] is a GPU↔GPU ordering token backed by an `MTLSharedEvent`.
//!   Within a single Metal queue submission order already guarantees ordering, so
//!   for the single-queue path the semaphore is unused (like the D3D12 backend's
//!   no-op semaphore). The async-compute path (M5) uses it as a real cross-queue
//!   event: the compute queue signals a monotonically increasing value and the
//!   graphics queue waits on it (see `MetalComputeQueue::submit` /
//!   `MetalQueue::submit_async`).

use std::cell::{Cell, RefCell};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLDevice, MTLEvent, MTLSharedEvent};

use crate::{Result, rhi_err};

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
        // Take ownership before waiting so the completed command buffer (and the
        // drawable retained by its presentation) is released as soon as the wait
        // returns. Merely borrowing it here leaves the previous frame alive until
        // `reset`; if nextDrawable fails before reset, the drawable pool can remain
        // permanently exhausted.
        if let Some(cmd) = self.cmd.borrow_mut().take() {
            cmd.waitUntilCompleted();
        }
        Ok(())
    }

    pub fn reset(&self) -> Result<()> {
        *self.cmd.borrow_mut() = None;
        Ok(())
    }
}

/// A GPU-GPU ordering token. Backed by an `MTLSharedEvent` so the async-compute
/// path can signal from the compute queue and wait on the graphics queue; on the
/// single-queue path it is simply never signaled/waited.
pub struct MetalSemaphore {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// Last value signaled on `event` for this token (monotonic). The single
    /// shared event per token means a fresh value is bumped on each signal.
    value: Cell<u64>,
}

impl MetalSemaphore {
    pub(crate) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let event = device
            .newSharedEvent()
            .ok_or_else(|| rhi_err("newSharedEvent failed"))?;
        Ok(Self {
            event,
            value: Cell::new(0),
        })
    }

    /// The backing event, as the `MTLEvent` view the command-buffer
    /// signal/wait methods take.
    pub(crate) fn event(&self) -> &ProtocolObject<dyn MTLEvent> {
        ProtocolObject::from_ref(&*self.event)
    }

    /// Reserve the next signal value (post-increment) for an upcoming
    /// `encodeSignalEvent`.
    pub(crate) fn next_value(&self) -> u64 {
        let v = self.value.get() + 1;
        self.value.set(v);
        v
    }

    /// The most recent value reserved by [`Self::next_value`] (the value a waiter
    /// should block on).
    pub(crate) fn current_value(&self) -> u64 {
        self.value.get()
    }
}
