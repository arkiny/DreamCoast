//! Logical device, queue, command pool, and resource creation.

use std::sync::Arc;

use ash::vk;
use engine_core::EngineError;
use rhi_types::{GraphicsPipelineDesc, SwapchainDesc};

use crate::instance::{InstanceShared, VulkanInstance};
use crate::pipeline::VulkanGraphicsPipeline;
use crate::swapchain::VulkanSwapchain;
use crate::sync::{VulkanFence, VulkanSemaphore};
use crate::vk_err;
use crate::{VulkanCommandBuffer, command};

/// Device-level objects shared (via `Arc`) by every GPU resource so each can
/// destroy itself before the device is torn down.
pub(crate) struct DeviceShared {
    pub instance: Arc<InstanceShared>,
    pub device: ash::Device,
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub queue: vk::Queue,
    pub physical_device: vk::PhysicalDevice,
    pub command_pool: vk::CommandPool,
}

impl DeviceShared {
    pub(crate) fn new(instance: &VulkanInstance) -> Result<Self, EngineError> {
        unsafe {
            let qfi = instance.queue_family_index;
            let priorities = [1.0f32];
            let queue_ci = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(qfi)
                .queue_priorities(&priorities);

            let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
            let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true);
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features13);

            let device_ci = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_ci))
                .enabled_extension_names(&device_extensions)
                .push_next(&mut features2);

            let raw = &instance.shared.instance;
            tracing::debug!("creating logical device (qfi={qfi})");
            let device = raw
                .create_device(instance.physical_device, &device_ci, None)
                .map_err(vk_err)?;
            tracing::debug!("logical device created");
            let queue = device.get_device_queue(qfi, 0);
            let swapchain_loader = ash::khr::swapchain::Device::new(raw, &device);

            let pool_ci = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(qfi);
            let command_pool = device.create_command_pool(&pool_ci, None).map_err(vk_err)?;
            tracing::debug!("command pool created");

            Ok(Self {
                instance: instance.shared.clone(),
                device,
                swapchain_loader,
                queue,
                physical_device: instance.physical_device,
                command_pool,
            })
        }
    }
}

impl Drop for DeviceShared {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

/// A logical Vulkan device: the factory for swapchains, pipelines, command
/// buffers, and synchronization primitives.
pub struct VulkanDevice {
    pub(crate) shared: Arc<DeviceShared>,
}

impl VulkanDevice {
    /// Create a swapchain for the window surface.
    pub fn create_swapchain(&self, desc: &SwapchainDesc) -> Result<VulkanSwapchain, EngineError> {
        VulkanSwapchain::new(self.shared.clone(), desc)
    }

    /// Create the triangle graphics pipeline (dynamic rendering, no vertex input).
    pub fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc,
    ) -> Result<VulkanGraphicsPipeline, EngineError> {
        VulkanGraphicsPipeline::new(self.shared.clone(), desc)
    }

    /// Allocate a primary command buffer from the device's pool.
    pub fn create_command_buffer(&self) -> Result<VulkanCommandBuffer, EngineError> {
        command::VulkanCommandBuffer::new(self.shared.clone())
    }

    /// Create a fence, optionally already signaled.
    pub fn create_fence(&self, signaled: bool) -> Result<VulkanFence, EngineError> {
        VulkanFence::new(self.shared.clone(), signaled)
    }

    /// Create a binary semaphore.
    pub fn create_semaphore(&self) -> Result<VulkanSemaphore, EngineError> {
        VulkanSemaphore::new(self.shared.clone())
    }

    /// The device's graphics+present queue.
    pub fn queue(&self) -> VulkanQueue {
        VulkanQueue {
            shared: self.shared.clone(),
        }
    }

    /// Block until the device is idle (used before teardown / swapchain rebuild).
    pub fn wait_idle(&self) -> Result<(), EngineError> {
        unsafe { self.shared.device.device_wait_idle().map_err(vk_err) }
    }
}

/// The device's queue. Submits command buffers and presents swapchain images.
pub struct VulkanQueue {
    pub(crate) shared: Arc<DeviceShared>,
}

impl VulkanQueue {
    /// Submit one command buffer, waiting on `wait` (at color-output) and
    /// signaling `signal` and `fence` on completion.
    pub fn submit(
        &self,
        cmd: &VulkanCommandBuffer,
        wait: &VulkanSemaphore,
        signal: &VulkanSemaphore,
        fence: &VulkanFence,
    ) -> Result<(), EngineError> {
        unsafe {
            let wait_semaphores = [wait.raw()];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [signal.raw()];
            let command_buffers = [cmd.raw()];
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            self.shared
                .device
                .queue_submit(self.shared.queue, &[submit], fence.raw())
                .map_err(vk_err)
        }
    }

    /// Present a swapchain image, waiting on `wait`. Returns `true` if the
    /// swapchain is out-of-date/suboptimal and should be recreated.
    pub fn present(
        &self,
        swapchain: &VulkanSwapchain,
        image_index: u32,
        wait: &VulkanSemaphore,
    ) -> Result<bool, EngineError> {
        unsafe {
            let wait_semaphores = [wait.raw()];
            let swapchains = [swapchain.raw()];
            let indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&indices);
            match self
                .shared
                .swapchain_loader
                .queue_present(self.shared.queue, &present_info)
            {
                Ok(suboptimal) => Ok(suboptimal),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(true),
                Err(e) => Err(vk_err(e)),
            }
        }
    }
}
