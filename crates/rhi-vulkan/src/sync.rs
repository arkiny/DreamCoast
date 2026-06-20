//! Fences and binary semaphores for frame synchronization.

use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;

use crate::device::DeviceShared;
use crate::vk_err;

/// A fence for CPU↔GPU synchronization (one per frame in flight).
pub struct VulkanFence {
    device: Arc<DeviceShared>,
    fence: vk::Fence,
}

impl VulkanFence {
    pub(crate) fn new(device: Arc<DeviceShared>, signaled: bool) -> Result<Self, EngineError> {
        let flags = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };
        let ci = vk::FenceCreateInfo::default().flags(flags);
        let fence = unsafe { device.device.create_fence(&ci, None).map_err(vk_err)? };
        Ok(Self { device, fence })
    }

    pub(crate) fn raw(&self) -> vk::Fence {
        self.fence
    }

    /// Block until the fence is signaled.
    pub fn wait(&self) -> Result<(), EngineError> {
        unsafe {
            self.device
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(vk_err)
        }
    }

    /// Reset the fence to the unsignaled state.
    pub fn reset(&self) -> Result<(), EngineError> {
        unsafe {
            self.device
                .device
                .reset_fences(&[self.fence])
                .map_err(vk_err)
        }
    }
}

impl Drop for VulkanFence {
    fn drop(&mut self) {
        unsafe { self.device.device.destroy_fence(self.fence, None) };
    }
}

/// A binary semaphore for GPU↔GPU ordering (acquire/submit/present).
pub struct VulkanSemaphore {
    device: Arc<DeviceShared>,
    semaphore: vk::Semaphore,
}

impl VulkanSemaphore {
    pub(crate) fn new(device: Arc<DeviceShared>) -> Result<Self, EngineError> {
        let ci = vk::SemaphoreCreateInfo::default();
        let semaphore = unsafe { device.device.create_semaphore(&ci, None).map_err(vk_err)? };
        Ok(Self { device, semaphore })
    }

    pub(crate) fn raw(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for VulkanSemaphore {
    fn drop(&mut self) {
        unsafe { self.device.device.destroy_semaphore(self.semaphore, None) };
    }
}
