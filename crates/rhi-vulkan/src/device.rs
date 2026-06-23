//! Logical device, queue, command pool, and resource creation.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::{
    BufferDesc, CubemapDesc, Extent2D, GraphicsPipelineDesc, MemoryRequirements, ReadbackLayout,
    RenderTargetDesc, SwapchainDesc, TextureDesc,
};

use crate::buffer::VulkanBuffer;
use crate::cubemap::VulkanCubemap;
use crate::depth::VulkanDepthBuffer;
use crate::instance::{InstanceShared, VulkanInstance};
use crate::pipeline::VulkanGraphicsPipeline;
use crate::render_target::{self, VulkanRenderTarget, VulkanTransientHeap};
use crate::swapchain::VulkanSwapchain;
use crate::sync::{VulkanFence, VulkanSemaphore};
use crate::texture::VulkanTexture;
use crate::vk_err;
use crate::{VulkanCommandBuffer, command};

/// Size of the bindless sampled-image (2D) table.
pub(crate) const BINDLESS_COUNT: u32 = 1024;
/// Size of the bindless cubemap table (binding 2 / register space 1). Its index
/// space is separate from the 2D table and starts at 0.
pub(crate) const CUBE_COUNT: u32 = 64;
/// Size of the bindless storage-image (UAV) table (binding 3). Compute writes
/// these; its index space is separate and 0-based (Phase 7).
pub(crate) const STORAGE_IMAGE_COUNT: u32 = 64;
/// Size of the bindless storage-buffer (UAV) table (binding 4). Compute/vertex
/// read-write these; separate 0-based index space (Phase 7).
pub(crate) const STORAGE_BUFFER_COUNT: u32 = 64;

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
    cube_next: AtomicU32,
    storage_image_next: AtomicU32,
    storage_buffer_next: AtomicU32,
    // Per-frame globals (set 1): one UNIFORM_BUFFER_DYNAMIC binding, written once
    // to point at the app's globals buffer; the per-frame slice is selected by a
    // dynamic offset at bind time. Only PBR pipelines bind this set.
    pub globals_pool: vk::DescriptorPool,
    pub globals_layout: vk::DescriptorSetLayout,
    pub globals_set: vk::DescriptorSet,
    // Async compute (Phase 7): a queue from a dedicated compute family (when one
    // exists) for work that overlaps the graphics queue, plus its command pool.
    // `graphics_family`/`compute_family` drive CONCURRENT sharing for cross-queue
    // storage buffers. When no dedicated family exists, the compute queue is the
    // graphics queue (no real overlap, but the cross-queue path still works).
    pub graphics_family: u32,
    pub compute_family: u32,
    pub compute_queue: vk::Queue,
    pub compute_command_pool: vk::CommandPool,
    pub has_dedicated_compute: bool,
    // Hardware ray tracing (Phase 8): true when the acceleration-structure +
    // ray-query + ray-tracing-pipeline extensions are enabled.
    pub has_raytracing: bool,
    // Acceleration-structure extension function loader (`vkCmdBuildAcceleration
    // StructuresKHR` etc.); `Some` only when `has_raytracing` (Phase 8).
    pub accel_loader: Option<ash::khr::acceleration_structure::Device>,
}

impl DeviceShared {
    pub(crate) fn new(instance: &VulkanInstance) -> Result<Self, EngineError> {
        unsafe {
            let qfi = instance.queue_family_index;
            let priorities = [1.0f32];

            // Find a dedicated async-compute family (COMPUTE without GRAPHICS); fall
            // back to the graphics family if none exists.
            let raw_inst = &instance.shared.instance;
            let families =
                raw_inst.get_physical_device_queue_family_properties(instance.physical_device);
            let dedicated_compute = families.iter().enumerate().find_map(|(i, f)| {
                let i = i as u32;
                let has_compute = f.queue_flags.contains(vk::QueueFlags::COMPUTE);
                let has_graphics = f.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                (has_compute && !has_graphics && i != qfi).then_some(i)
            });
            let compute_family = dedicated_compute.unwrap_or(qfi);
            let has_dedicated_compute = dedicated_compute.is_some();

            // One queue from the graphics family, plus one from a distinct compute
            // family when available.
            let gfx_ci = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(qfi)
                .queue_priorities(&priorities);
            let cmp_ci = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(compute_family)
                .queue_priorities(&priorities);
            let queue_cis = if has_dedicated_compute {
                vec![gfx_ci, cmp_ci]
            } else {
                vec![gfx_ci]
            };

            // Hardware ray tracing (Phase 8): enable the acceleration-structure +
            // ray-query + ray-tracing-pipeline extensions when the physical device
            // supports all of them (plus the deferred-host-operations dependency).
            // Gated so devices without RT still create a valid logical device.
            let rt_extensions = [
                ash::khr::acceleration_structure::NAME,
                ash::khr::ray_tracing_pipeline::NAME,
                ash::khr::ray_query::NAME,
                ash::khr::deferred_host_operations::NAME,
            ];
            let supported_exts = raw_inst
                .enumerate_device_extension_properties(instance.physical_device)
                .unwrap_or_default();
            let has_raytracing = rt_extensions.iter().all(|name| {
                supported_exts
                    .iter()
                    .any(|e| e.extension_name_as_c_str() == Ok(name))
            });

            let mut device_extensions = vec![ash::khr::swapchain::NAME.as_ptr()];
            if has_raytracing {
                for name in &rt_extensions {
                    device_extensions.push(name.as_ptr());
                }
            }
            let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true);
            // Descriptor indexing for bindless sampled images + storage (Phase 7).
            // Buffer device address (vertex/index/AS addresses) is needed for ray
            // tracing (Phase 8); enabled only when RT is available.
            let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
                .runtime_descriptor_array(true)
                .shader_sampled_image_array_non_uniform_indexing(true)
                .descriptor_binding_partially_bound(true)
                .descriptor_binding_sampled_image_update_after_bind(true)
                .descriptor_binding_storage_image_update_after_bind(true)
                .descriptor_binding_storage_buffer_update_after_bind(true)
                .buffer_device_address(has_raytracing);
            // SV_VertexID full-screen-triangle shaders (triangle/post/blur) compile
            // to SPIR-V using the DrawParameters capability.
            let mut features11 =
                vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
            // Compute writes to storage images whose SPIR-V `OpTypeImage` carries no
            // format qualifier (Slang emits `Unknown` for `RWTexture2D<float4>`).
            let base_features = vk::PhysicalDeviceFeatures::default()
                .shader_storage_image_write_without_format(true)
                // The particle draw's vertex stage reads a storage buffer (UAV).
                .vertex_pipeline_stores_and_atomics(true);
            // RT feature structs (Phase 8) — only chained when the extensions are
            // enabled, else validation rejects the unsupported feature structs.
            let mut accel_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true)
                // Lets the TLAS descriptor (bindless binding 5) be updated after
                // bind, so switching the bound scene (e.g. Cornell toggle) doesn't
                // invalidate in-flight command buffers (Phase 8 M4).
                .descriptor_binding_acceleration_structure_update_after_bind(true);
            let mut ray_query_features =
                vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
            let mut rt_pipeline_features =
                vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
                    .ray_tracing_pipeline(true);

            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .features(base_features)
                .push_next(&mut features13)
                .push_next(&mut features12)
                .push_next(&mut features11);
            if has_raytracing {
                features2 = features2
                    .push_next(&mut accel_features)
                    .push_next(&mut ray_query_features)
                    .push_next(&mut rt_pipeline_features);
            }

            let device_ci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_cis)
                .enabled_extension_names(&device_extensions)
                .push_next(&mut features2);

            let raw = &instance.shared.instance;
            tracing::debug!(
                "creating logical device (gfx qfi={qfi}, compute qfi={compute_family}, \
                 dedicated={has_dedicated_compute}, raytracing={has_raytracing})"
            );
            let device = raw
                .create_device(instance.physical_device, &device_ci, None)
                .map_err(vk_err)?;
            tracing::debug!("logical device created");
            let queue = device.get_device_queue(qfi, 0);
            let compute_queue = device.get_device_queue(compute_family, 0);
            let swapchain_loader = ash::khr::swapchain::Device::new(raw, &device);
            let accel_loader =
                has_raytracing.then(|| ash::khr::acceleration_structure::Device::new(raw, &device));

            let pool_ci = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(qfi);
            let command_pool = device.create_command_pool(&pool_ci, None).map_err(vk_err)?;
            let compute_pool_ci = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(compute_family);
            let compute_command_pool = device
                .create_command_pool(&compute_pool_ci, None)
                .map_err(vk_err)?;
            tracing::debug!("command pools created");

            let mem_props = raw.get_physical_device_memory_properties(instance.physical_device);

            let (bindless_pool, bindless_layout, bindless_set, bindless_sampler) =
                create_bindless(&device, has_raytracing)?;
            let (globals_pool, globals_layout, globals_set) = create_globals(&device)?;

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
                cube_next: AtomicU32::new(0),
                storage_image_next: AtomicU32::new(0),
                storage_buffer_next: AtomicU32::new(0),
                globals_pool,
                globals_layout,
                globals_set,
                graphics_family: qfi,
                compute_family,
                compute_queue,
                compute_command_pool,
                has_dedicated_compute,
                has_raytracing,
                accel_loader,
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

    /// Register a cubemap (CUBE) image view in the bindless table (binding 2),
    /// returning its index in the (separate, 0-based) cube index space.
    pub(crate) fn register_sampled_cube(&self, view: vk::ImageView) -> u32 {
        let index = self.cube_next.fetch_add(1, Ordering::Relaxed);
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let infos = [image_info];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.bindless_set)
            .dst_binding(2)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&infos);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        index
    }

    /// Register a storage-image (UAV) view in the bindless table (binding 3),
    /// returning its index in the separate 0-based storage-image space. Compute
    /// writes it (image layout GENERAL).
    pub(crate) fn register_storage_image(&self, view: vk::ImageView) -> u32 {
        let index = self.storage_image_next.fetch_add(1, Ordering::Relaxed);
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL);
        let infos = [image_info];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.bindless_set)
            .dst_binding(3)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&infos);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        index
    }

    /// Register a storage-buffer (UAV) in the bindless table (binding 4),
    /// returning its index in the separate 0-based storage-buffer space.
    pub(crate) fn register_storage_buffer(&self, buffer: vk::Buffer, range: u64) -> u32 {
        let index = self.storage_buffer_next.fetch_add(1, Ordering::Relaxed);
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(range)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.bindless_set)
            .dst_binding(4)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&info);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        index
    }

    /// Write the scene TLAS into the bindless set (binding 5) for inline ray
    /// query (Phase 8 M3). The acceleration-structure write goes through a
    /// `pNext` struct; the main write's `descriptor_count` must still be set.
    pub(crate) fn register_tlas(&self, tlas: vk::AccelerationStructureKHR) {
        let accels = [tlas];
        let mut as_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&accels);
        let mut write = vk::WriteDescriptorSet::default()
            .dst_set(self.bindless_set)
            .dst_binding(5)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut as_write);
        write.descriptor_count = 1;
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
    }

    /// Point the globals set (set 1) at `buffer`; `range` is one frame's slice
    /// size (the per-frame offset is applied as a dynamic offset at bind time).
    pub(crate) fn set_globals_buffer(&self, buffer: vk::Buffer, range: u64) {
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(range)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.globals_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .buffer_info(&info);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
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
    has_raytracing: bool,
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
            // Trilinear: prefilter cubemaps sample a roughness mip via SampleLevel.
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(vk::LOD_CLAMP_NONE);
        let sampler = device.create_sampler(&sampler_ci, None).map_err(vk_err)?;
        let immutable = [sampler];

        // Sampled images/samplers/cubes are read by fragment AND compute (Phase 7
        // compute passes sample textures). Storage image/buffer are written by
        // compute; the storage buffer is also read by the vertex stage (particle
        // vertex-pull).
        let sampled_stages = vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE;
        let dynamic = vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
        let mut bindings = vec![
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(BINDLESS_COUNT)
                .stage_flags(sampled_stages),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(sampled_stages)
                .immutable_samplers(&immutable),
            // Binding 2: cubemap (CUBE-view) array. Its own 0-based index space
            // (shader samples it as `g_cubes[]`, separate from `g_textures[]`).
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(CUBE_COUNT)
                .stage_flags(sampled_stages),
            // Binding 3: storage-image (UAV) array, written by compute. 0-based.
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(STORAGE_IMAGE_COUNT)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 4: storage-buffer (UAV) array. Read-write in compute, read in
            // the vertex stage (particle draw). 0-based.
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(STORAGE_BUFFER_COUNT)
                .stage_flags(
                    vk::ShaderStageFlags::COMPUTE
                        | vk::ShaderStageFlags::VERTEX
                        | vk::ShaderStageFlags::FRAGMENT,
                ),
        ];
        let mut flags = vec![
            dynamic,
            vk::DescriptorBindingFlags::empty(),
            dynamic,
            dynamic,
            dynamic,
        ];
        // Binding 5: scene TLAS (single acceleration structure) for inline ray
        // query, only on RT-capable devices (the descriptor type needs the
        // acceleration-structure extension). Shaders that don't trace drop it.
        if has_raytracing {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(5)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
            // UPDATE_AFTER_BIND so the bound scene's TLAS can be swapped at runtime
            // (Cornell toggle) without invalidating in-flight command buffers.
            flags.push(dynamic);
        }
        let mut flags_ci =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        let layout_ci = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&bindings)
            .push_next(&mut flags_ci);
        let layout = device
            .create_descriptor_set_layout(&layout_ci, None)
            .map_err(vk_err)?;

        let mut pool_sizes = vec![
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(BINDLESS_COUNT + CUBE_COUNT), // bindings 0 + 2
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(STORAGE_IMAGE_COUNT), // binding 3
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(STORAGE_BUFFER_COUNT), // binding 4
        ];
        if has_raytracing {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1), // binding 5
            );
        }
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

/// Build the globals descriptor pool, set layout (one dynamic UBO at binding 0),
/// and set. Bound as set 1 by PBR pipelines.
fn create_globals(
    device: &ash::Device,
) -> Result<
    (
        vk::DescriptorPool,
        vk::DescriptorSetLayout,
        vk::DescriptorSet,
    ),
    EngineError,
> {
    unsafe {
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];
        let layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let layout = device
            .create_descriptor_set_layout(&layout_ci, None)
            .map_err(vk_err)?;

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
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

        Ok((pool, layout, set))
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
            self.device.destroy_descriptor_pool(self.globals_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.globals_layout, None);
            self.device.destroy_sampler(self.bindless_sampler, None);
            self.device.destroy_command_pool(self.command_pool, None);
            if self.has_dedicated_compute {
                self.device
                    .destroy_command_pool(self.compute_command_pool, None);
            }
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

    /// Create a compute pipeline (Phase 7).
    pub fn create_compute_pipeline(
        &self,
        desc: &rhi_types::ComputePipelineDesc,
    ) -> Result<crate::pipeline::VulkanComputePipeline, EngineError> {
        crate::pipeline::VulkanComputePipeline::new(self.shared.clone(), desc)
    }

    /// Build the scene's acceleration structures (BLAS per mesh + one TLAS) in a
    /// one-shot graphics-queue submission (static scene, Phase 8 M2).
    pub fn build_raytracing_scene(
        &self,
        geometries: &[(&VulkanBuffer, &VulkanBuffer, rhi_types::BlasGeometry)],
        instances: &[rhi_types::TlasInstance],
    ) -> Result<crate::accel::VulkanRaytracingScene, EngineError> {
        crate::accel::VulkanRaytracingScene::build(self.shared.clone(), geometries, instances)
    }

    /// Register the scene TLAS in the bindless set so shaders can trace it
    /// (Phase 8 M3). Call once after building a static scene.
    pub fn bind_tlas(&self, scene: &crate::accel::VulkanRaytracingScene) {
        self.shared.register_tlas(scene.tlas());
    }

    /// Allocate a primary command buffer from the device's pool.
    pub fn create_command_buffer(&self) -> Result<VulkanCommandBuffer, EngineError> {
        command::VulkanCommandBuffer::new(self.shared.clone())
    }

    /// Create a host-visible buffer.
    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<VulkanBuffer, EngineError> {
        VulkanBuffer::new(self.shared.clone(), desc)
    }

    /// Create a device-local storage buffer (UAV) for compute (Phase 7).
    pub fn create_storage_buffer(
        &self,
        desc: &rhi_types::StorageBufferDesc,
    ) -> Result<crate::buffer::VulkanStorageBuffer, EngineError> {
        crate::buffer::VulkanStorageBuffer::new(self.shared.clone(), desc)
    }

    /// Create a storage buffer seeded with host data (Phase 8: RT geometry +
    /// instance table read by the path tracer).
    pub fn create_storage_buffer_init(
        &self,
        desc: &rhi_types::StorageBufferDesc,
        data: &[u8],
    ) -> Result<crate::buffer::VulkanStorageBuffer, EngineError> {
        crate::buffer::VulkanStorageBuffer::new_init(self.shared.clone(), desc, data)
    }

    /// Register the per-frame globals buffer with the globals descriptor set.
    /// `slice_size` is one frame's slice (selected via dynamic offset at bind).
    pub fn set_globals_buffer(&self, buffer: &VulkanBuffer, slice_size: u64) {
        self.shared.set_globals_buffer(buffer.raw(), slice_size);
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

    pub fn create_cubemap(&self, desc: &CubemapDesc) -> Result<VulkanCubemap, EngineError> {
        VulkanCubemap::new(self.shared.clone(), desc)
    }

    /// CPU memory layout for reading a swapchain image back to the host. Vulkan
    /// packs rows tightly (`row_pitch = width * 4`).
    pub fn swapchain_readback_layout(&self, swapchain: &VulkanSwapchain) -> ReadbackLayout {
        let e = swapchain.extent();
        let row_pitch = e.width * 4;
        ReadbackLayout {
            width: e.width,
            height: e.height,
            row_pitch,
            size: row_pitch as u64 * e.height as u64,
        }
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

    /// Allocate a command buffer on the async-compute queue family (Phase 7).
    pub fn create_compute_command_buffer(&self) -> Result<VulkanCommandBuffer, EngineError> {
        command::VulkanCommandBuffer::new_compute(self.shared.clone())
    }

    /// The async-compute queue (a dedicated compute family when one exists).
    pub fn compute_queue(&self) -> VulkanComputeQueue {
        VulkanComputeQueue {
            shared: self.shared.clone(),
        }
    }

    /// Whether a dedicated async-compute queue family is in use (else the compute
    /// queue aliases the graphics queue and there is no real overlap).
    pub fn has_async_compute(&self) -> bool {
        self.shared.has_dedicated_compute
    }

    /// Whether hardware ray tracing is available (acceleration-structure +
    /// ray-query + ray-tracing-pipeline extensions enabled) (Phase 8).
    pub fn has_raytracing(&self) -> bool {
        self.shared.has_raytracing
    }

    /// Block until the device is idle (used before teardown / swapchain rebuild).
    pub fn wait_idle(&self) -> Result<(), EngineError> {
        unsafe { self.shared.device.device_wait_idle().map_err(vk_err) }
    }
}

/// The device's async-compute queue (Phase 7).
pub struct VulkanComputeQueue {
    pub(crate) shared: Arc<DeviceShared>,
}

impl VulkanComputeQueue {
    /// Submit async-compute work, signaling `signal` on completion (the graphics
    /// queue waits on it). No wait, no fence: frame pacing is carried transitively
    /// by the graphics fence the graphics submit signals.
    pub fn submit(
        &self,
        cmd: &VulkanCommandBuffer,
        signal: &VulkanSemaphore,
    ) -> Result<(), EngineError> {
        unsafe {
            let command_buffers = [cmd.raw()];
            let signal_semaphores = [signal.raw()];
            let submit = vk::SubmitInfo::default()
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            self.shared
                .device
                .queue_submit(self.shared.compute_queue, &[submit], vk::Fence::null())
                .map_err(vk_err)
        }
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

    /// Submit, additionally waiting on `compute_wait` (the async-compute queue's
    /// completion semaphore) at the vertex stage (where the particle draw reads the
    /// compute-written buffer), before signaling `signal`/`fence` (Phase 7).
    pub fn submit_async(
        &self,
        cmd: &VulkanCommandBuffer,
        wait: &VulkanSemaphore,
        compute_wait: &VulkanSemaphore,
        signal: &VulkanSemaphore,
        fence: &VulkanFence,
    ) -> Result<(), EngineError> {
        unsafe {
            let wait_semaphores = [wait.raw(), compute_wait.raw()];
            let wait_stages = [
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::VERTEX_SHADER,
            ];
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

    /// Submit one command buffer with no semaphore sync, signaling `fence` on
    /// completion. For one-off startup work (e.g. IBL cubemap generation).
    pub fn submit_oneshot(
        &self,
        cmd: &VulkanCommandBuffer,
        fence: &VulkanFence,
    ) -> Result<(), EngineError> {
        unsafe {
            let command_buffers = [cmd.raw()];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
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
