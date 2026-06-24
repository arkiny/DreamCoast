//! GPU timestamp queries for per-pass profiling (Phase 9 M1).
//!
//! A [`VulkanQueryHeap`] wraps a `VK_QUERY_TYPE_TIMESTAMP` query pool. The render
//! graph writes a timestamp at each pass boundary (`vkCmdWriteTimestamp`); the
//! host reads the raw ticks back ([`Self::read`]) once the frame's fence has
//! signalled and converts to nanoseconds via [`Self::period_ns`]
//! (`VkPhysicalDeviceLimits::timestampPeriod`). The pool must be reset
//! (`vkCmdResetQueryPool`) each frame before it is written.

use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;

use crate::device::DeviceShared;
use crate::vk_err;

/// A timestamp query pool of `count` queries.
pub struct VulkanQueryHeap {
    device: Arc<DeviceShared>,
    pool: vk::QueryPool,
    count: u32,
    period_ns: f32,
}

impl VulkanQueryHeap {
    pub(crate) fn new(device: Arc<DeviceShared>, count: u32) -> Result<Self, EngineError> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count);
        let pool = unsafe { device.device.create_query_pool(&info, None) }.map_err(vk_err)?;
        // `timestampPeriod` is the number of nanoseconds a timestamp tick spans.
        let props = unsafe {
            device
                .instance
                .instance
                .get_physical_device_properties(device.physical_device)
        };
        Ok(Self {
            device,
            pool,
            count,
            period_ns: props.limits.timestamp_period,
        })
    }

    pub(crate) fn raw(&self) -> vk::QueryPool {
        self.pool
    }

    /// Number of timestamp slots in the pool.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Nanoseconds per timestamp tick (multiply tick deltas by this).
    pub fn period_ns(&self) -> f32 {
        self.period_ns
    }

    /// Read all `count` raw timestamp ticks. Call only after the submission that
    /// wrote them has completed (e.g. after the frame fence); unavailable queries
    /// leave their slot unchanged (the call returns `NOT_READY`, ignored here).
    pub fn read(&self) -> Vec<u64> {
        let mut data = vec![0u64; self.count as usize];
        unsafe {
            let _ = self.device.device.get_query_pool_results(
                self.pool,
                0,
                &mut data,
                vk::QueryResultFlags::TYPE_64,
            );
        }
        data
    }
}

impl Drop for VulkanQueryHeap {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_query_pool(self.pool, None);
        }
    }
}
