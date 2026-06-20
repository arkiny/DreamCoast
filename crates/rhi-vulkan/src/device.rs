//! Logical device, queue, command pool, and resource creation.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::{
    BufferDesc, Extent2D, GraphicsPipelineDesc, MemoryRequirements, RenderTargetDesc,
    SwapchainDesc, TextureDesc,
};

use crate::buffer::VulkanBuffer;
use crate::depth::VulkanDepthBuffer;
use crate::instance::{InstanceShared, VulkanInstance};
use crate::pipeline::VulkanGraphicsPipeline;
use crate::render_target::{self, VulkanRenderTarget, VulkanTransientHeap};
use crate::swapchain::VulkanSwapchain;
use crate::sync::{VulkanFence, VulkanSemaphore};
use crate::texture::VulkanTexture;
use crate::vk_err;
use crate::{VulkanCommandBuffer, command};

/// Size of the bindless sampled-image table.
pub(crate) const BINDLESS_COUNT: u32 = 1024;

/// Device-level objects shared (via `Arc`) by every GPU resource so each can
/// destroy itself before the device is torn down.
pub(crate) struct DeviceShared {
    pub instance: Arc<InstanceShared>,
    pub device: ash::Device,
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub queue: vk::Queue,
    pub physical_device: vk::PhysicalDevice,
    pub command_pool: vk::CommandPool,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    // Bindless: one big SAMPLED_IMAGE array (binding 0) + an immutable sampler
    // (binding 1), in a single descriptor set bound for every bindless pipeline.
    pub bindless_pool: vk::DescriptorPool,
    pub bindless_layout: vk::DescriptorSetLayout,
    pub bindless_set: vk::DescriptorSet,
    pub bindless_sampler: vk::Sampler,
    bindless_next: AtomicU32,
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
            // Descriptor indexing for bindless sampled images.
            let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
                .runtime_descriptor_array(true)
                .shader_sampled_image_array_non_uniform_indexing(true)
                .descriptor_binding_partially_bound(true)
                .descriptor_binding_sampled_image_update_after_bind(true);
            // SV_VertexID full-screen-triangle shaders (triangle/post/blur) compile
            // to SPIR-V using the DrawParameters capability.
            let mut features11 =
                vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut features13)
                .push_next(&mut features12)
                .push_next(&mut features11);

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

            let mem_props = raw.get_physical_device_memory_properties(instance.physical_device);

            let (bindless_pool, bindless_layout, bindless_set, bindless_sampler) =
                create_bindless(&device)?;

            Ok(Self {
                instance: instance.shared.clone(),
                device,
                swapchain_loader,
                queue,
                physical_device: instance.physical_device,
                command_pool,
                mem_props,
                bindless_pool,
                bindless_layout,
                bindless_set,
                bindless_sampler,
                bindless_next: AtomicU32::new(0),
            })
        }
    }

    /// Find a memory type index satisfying `type_bits` and `props`.
    pub(crate) fn find_memory_type(
        &self,
        type_bits: u32,
        props: vk::MemoryPropertyFlags,
    ) -> Result<u32, EngineError> {
        for i in 0..self.mem_props.memory_type_count {
            let suitable = (type_bits & (1 << i)) != 0;
            let has_props = self.mem_props.memory_types[i as usize]
                .property_flags
                .contains(props);
            if suitable && has_props {
                return Ok(i);
            }
        }
        Err(EngineError::Rhi("no suitable memory type".into()))
    }

    /// Register a sampled image view in the bindless table, returning its index.
    pub(crate) fn register_sampled_image(&self, view: vk::ImageView) -> u32 {
        let index = self.bindless_next.fetch_add(1, Ordering::Relaxed);
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let infos = [image_info];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.bindless_set)
            .dst_binding(0)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&infos);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        index
    }

    /// Record + submit a one-time command buffer and wait for completion.
    pub(crate) fn immediate_submit(
        &self,
        record: impl FnOnce(vk::CommandBuffer),
    ) -> Result<(), EngineError> {
        unsafe {
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self
                .device
                .allocate_command_buffers(&alloc)
                .map_err(vk_err)?[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(vk_err)?;
            record(cmd);
            self.device.end_command_buffer(cmd).map_err(vk_err)?;

            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(vk_err)?;
            self.device
                .queue_submit(self.queue, &[submit], fence)
                .map_err(vk_err)?;
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(vk_err)?;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &cmds);
            Ok(())
        }
    }
}

/// Build the bindless descriptor pool, set layout, set, and immutable sampler.
fn create_bindless(
    device: &ash::Device,
) -> Result<
    (
        vk::DescriptorPool,
        vk::DescriptorSetLayout,
        vk::DescriptorSet,
        vk::Sampler,
    ),
    EngineError,
> {
    unsafe {
        let sampler_ci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(vk::LOD_CLAMP_NONE);
        let sampler = device.create_sampler(&sampler_ci, None).map_err(vk_err)?;
        let immutable = [sampler];

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(BINDLESS_COUNT)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .immutable_samplers(&immutable),
        ];
        let flags = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::empty(),
        ];
        let mut flags_ci =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        let layout_ci = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&bindings)
            .push_next(&mut flags_ci);
        let layout = device
            .create_descriptor_set_layout(&layout_ci, None)
            .map_err(vk_err)?;

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(BINDLESS_COUNT),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(1),
        ];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let pool = device
            .create_descriptor_pool(&pool_ci, None)
            .map_err(vk_err)?;

        let layouts = [layout];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let set = device.allocate_descriptor_sets(&alloc).map_err(vk_err)?[0];

        Ok((pool, layout, set, sampler))
    }
}

impl Drop for DeviceShared {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device
                .destroy_descriptor_pool(self.bindless_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.bindless_layout, None);
            self.device.destroy_sampler(self.bindless_sampler, None);
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

    /// Create a host-visible buffer.
    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<VulkanBuffer, EngineError> {
        VulkanBuffer::new(self.shared.clone(), desc)
    }

    /// Create a sampled 2D texture, upload `pixels`, and register it in the
    /// bindless table.
    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        pixels: &[u8],
    ) -> Result<VulkanTexture, EngineError> {
        VulkanTexture::new(self.shared.clone(), desc, pixels)
    }

    /// Create a depth buffer sized to `extent`.
    pub fn create_depth_buffer(&self, extent: Extent2D) -> Result<VulkanDepthBuffer, EngineError> {
        VulkanDepthBuffer::new(self.shared.clone(), extent)
    }

    /// Create an offscreen color render target (attachment + bindless sampled).
    pub fn create_render_target(
        &self,
        desc: &RenderTargetDesc,
    ) -> Result<VulkanRenderTarget, EngineError> {
        VulkanRenderTarget::new(self.shared.clone(), desc)
    }

    /// Memory footprint of an aliasable render target (for graph alias planning).
    pub fn render_target_memory(
        &self,
        desc: &RenderTargetDesc,
    ) -> Result<MemoryRequirements, EngineError> {
        render_target::render_target_memory(&self.shared, desc)
    }

    /// Create a transient heap of `size` bytes for aliased render targets.
    pub fn create_transient_heap(&self, size: u64) -> Result<VulkanTransientHeap, EngineError> {
        VulkanTransientHeap::new(self.shared.clone(), size)
    }

    /// Create a render target aliased into `heap` at `offset`.
    pub fn create_aliased_target(
        &self,
        heap: &VulkanTransientHeap,
        offset: u64,
        desc: &RenderTargetDesc,
    ) -> Result<VulkanRenderTarget, EngineError> {
        VulkanRenderTarget::new_aliased(self.shared.clone(), heap, offset, desc)
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
