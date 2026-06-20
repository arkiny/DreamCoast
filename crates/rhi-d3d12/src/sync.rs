//! Fence (binary-semantics emulation) and a no-op semaphore.
//!
//! The facade expects Vulkan-style binary fences and semaphores. D3D12 has only
//! a monotonic `ID3D12Fence`, so [`D3d12Fence`] tracks a target value the GPU
//! will reach and `wait()` blocks until then; `reset()` is a no-op. Semaphores
//! map to nothing (ordering within the single DIRECT queue is implicit).

use std::cell::Cell;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::{D3D12_FENCE_FLAG_NONE, ID3D12Fence};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::PCWSTR;

use crate::device::DeviceShared;
use crate::instance::d3d_err;

/// A CPU↔GPU fence emulating binary-fence semantics over a monotonic counter.
pub struct D3d12Fence {
    #[allow(dead_code)] // keeps the device (and thus the fence) alive
    device: Rc<DeviceShared>,
    fence: ID3D12Fence,
    event: HANDLE,
    counter: Cell<u64>,
    target: Cell<u64>,
}

impl D3d12Fence {
    pub(crate) fn new(device: Rc<DeviceShared>, signaled: bool) -> Result<Self, EngineError> {
        unsafe {
            let fence: ID3D12Fence = device
                .device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(d3d_err)?;
            let event = CreateEventW(None, false, false, PCWSTR::null()).map_err(d3d_err)?;
            // Signaled => target 0 (GetCompletedValue starts at 0, so wait passes
            // immediately). Unsignaled => target 1 (blocks until first signal).
            let target = if signaled { 0 } else { 1 };
            Ok(Self {
                device,
                fence,
                event,
                counter: Cell::new(0),
                target: Cell::new(target),
            })
        }
    }

    pub(crate) fn raw(&self) -> &ID3D12Fence {
        &self.fence
    }

    /// Reserve the next signal value (called by the queue on submit).
    pub(crate) fn next_value(&self) -> u64 {
        let value = self.counter.get() + 1;
        self.counter.set(value);
        value
    }

    /// Record the value the GPU will reach for this fence.
    pub(crate) fn set_target(&self, value: u64) {
        self.target.set(value);
    }

    /// Block until the GPU reaches the current target value.
    pub fn wait(&self) -> Result<(), EngineError> {
        unsafe {
            let target = self.target.get();
            if self.fence.GetCompletedValue() < target {
                self.fence
                    .SetEventOnCompletion(target, self.event)
                    .map_err(d3d_err)?;
                WaitForSingleObject(self.event, INFINITE);
            }
            Ok(())
        }
    }

    /// No-op: binary semantics are emulated via the monotonic target.
    pub fn reset(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

impl Drop for D3d12Fence {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

/// A no-op semaphore. D3D12 needs no GPU↔GPU semaphore for the single-queue
/// triangle; this exists only to satisfy the facade's Vulkan-shaped surface.
pub struct D3d12Semaphore;

impl D3d12Semaphore {
    pub(crate) fn new() -> Self {
        Self
    }
}
