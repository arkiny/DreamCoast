//! Hardware ray-tracing pipeline + shader binding table (Phase 8 M5).
//!
//! Builds a ray-tracing pipeline from one raygen / one miss / one closest-hit
//! shader (three groups: two general + one triangle hit group) and packs the
//! group handles into a shader binding table. The pipeline reuses the shared
//! bindless descriptor set (set 0, with the scene TLAS at binding 5) plus a push
//! constant range visible to the RT stages, so the path tracer's bindless
//! resources are reached exactly as they are from the inline compute path.
//!
//! `MaxTraceRecursionDepth` is 1: the bounce loop re-issues `TraceRay` from
//! raygen (not recursively), and shadow rays use an inline `RayQuery` inside the
//! hit shader, so no nested pipeline trace is ever needed.

use std::ffi::CString;
use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::RaytracingPipelineDesc;

use crate::device::DeviceShared;
use crate::pipeline::create_shader_module;
use crate::vk_err;

/// A host-visible, device-addressable buffer holding the shader binding table.
struct SbtBuffer {
    device: Arc<DeviceShared>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl SbtBuffer {
    fn new(device: Arc<DeviceShared>, size: u64) -> Result<Self, EngineError> {
        unsafe {
            let ci = vk::BufferCreateInfo::default()
                .size(size.max(1))
                .usage(
                    vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = device.device.create_buffer(&ci, None).map_err(vk_err)?;
            let req = device.device.get_buffer_memory_requirements(buffer);
            let mem_type = device.find_memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type)
                .push_next(&mut flags);
            let memory = device
                .device
                .allocate_memory(&alloc, None)
                .map_err(vk_err)?;
            device
                .device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(vk_err)?;
            Ok(Self {
                device,
                buffer,
                memory,
            })
        }
    }

    fn device_address(&self) -> u64 {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe { self.device.device.get_buffer_device_address(&info) }
    }
}

impl Drop for SbtBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_buffer(self.buffer, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}

/// A ray-tracing pipeline and its shader binding table. Holds the three strided
/// address regions (raygen / miss / hit) that `trace_rays` passes to
/// `vkCmdTraceRaysKHR`.
pub struct VulkanRaytracingPipeline {
    device: Arc<DeviceShared>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    _sbt: SbtBuffer,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,
    callable_region: vk::StridedDeviceAddressRegionKHR,
}

fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

impl VulkanRaytracingPipeline {
    pub(crate) fn new(
        device: Arc<DeviceShared>,
        desc: &RaytracingPipelineDesc,
    ) -> Result<Self, EngineError> {
        let loader = device
            .rt_pipeline_loader
            .as_ref()
            .ok_or_else(|| EngineError::Rhi("ray tracing pipeline not available".into()))?;

        unsafe {
            let rgen = create_shader_module(&device.device, desc.raygen_bytes)?;
            let miss = create_shader_module(&device.device, desc.miss_bytes)?;
            let chit = create_shader_module(&device.device, desc.closesthit_bytes)?;

            let rgen_name =
                CString::new(desc.raygen_entry).map_err(|e| EngineError::Rhi(e.to_string()))?;
            let miss_name =
                CString::new(desc.miss_entry).map_err(|e| EngineError::Rhi(e.to_string()))?;
            let chit_name =
                CString::new(desc.closesthit_entry).map_err(|e| EngineError::Rhi(e.to_string()))?;

            // Stage 0 = raygen, 1 = miss, 2 = closest hit.
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                    .module(rgen)
                    .name(&rgen_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::MISS_KHR)
                    .module(miss)
                    .name(&miss_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                    .module(chit)
                    .name(&chit_name),
            ];

            // Group 0 = raygen (general), 1 = miss (general), 2 = triangle hit group.
            let groups = [
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                    .general_shader(0)
                    .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                    .general_shader(1)
                    .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                    .general_shader(vk::SHADER_UNUSED_KHR)
                    .closest_hit_shader(2)
                    .any_hit_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
            ];

            // Layout: shared bindless set (0) + push constants for the RT stages.
            let set_layouts = [device.bindless_layout];
            let push_ranges = [vk::PushConstantRange::default()
                .stage_flags(
                    vk::ShaderStageFlags::RAYGEN_KHR
                        | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                        | vk::ShaderStageFlags::MISS_KHR,
                )
                .offset(0)
                .size(desc.push_constant_size)];
            let mut layout_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            if desc.push_constant_size > 0 {
                layout_ci = layout_ci.push_constant_ranges(&push_ranges);
            }
            let layout = device
                .device
                .create_pipeline_layout(&layout_ci, None)
                .map_err(vk_err)?;

            let pipeline_ci = vk::RayTracingPipelineCreateInfoKHR::default()
                .stages(&stages)
                .groups(&groups)
                .max_pipeline_ray_recursion_depth(1)
                .layout(layout);
            let pipeline = loader
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[pipeline_ci],
                    None,
                )
                .map_err(|(_, e)| vk_err(e))?[0];

            // Shader modules are no longer needed once the pipeline exists.
            device.device.destroy_shader_module(rgen, None);
            device.device.destroy_shader_module(miss, None);
            device.device.destroy_shader_module(chit, None);

            // ---- Shader binding table ----
            // Three single-record regions (raygen / miss / hit). Each record holds
            // one group handle; the record stride is the handle size rounded up to
            // the handle alignment, and each region starts on a base-alignment
            // boundary (Vulkan SBT requirements).
            let handle_size = device.rt_handle_size as u64;
            let handle_align = device.rt_handle_alignment.max(1) as u64;
            let base_align = device.rt_base_alignment.max(1) as u64;
            let stride = align_up(handle_size, handle_align);
            let region_size = align_up(stride, base_align);

            let group_count = 3u32;
            let handles = loader
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    group_count,
                    (handle_size * group_count as u64) as usize,
                )
                .map_err(vk_err)?;

            let sbt_size = region_size * 3;
            let sbt = SbtBuffer::new(device.clone(), sbt_size)?;
            // Map and write the three handles, each at its region base.
            let ptr = device
                .device
                .map_memory(sbt.memory, 0, sbt_size, vk::MemoryMapFlags::empty())
                .map_err(vk_err)? as *mut u8;
            std::ptr::write_bytes(ptr, 0, sbt_size as usize);
            for group in 0..3usize {
                let src = handles.as_ptr().add(group * handle_size as usize);
                let dst = ptr.add(group * region_size as usize);
                std::ptr::copy_nonoverlapping(src, dst, handle_size as usize);
            }
            device.device.unmap_memory(sbt.memory);

            let base = sbt.device_address();
            let raygen_region = vk::StridedDeviceAddressRegionKHR::default()
                .device_address(base)
                .stride(region_size)
                .size(region_size);
            let miss_region = vk::StridedDeviceAddressRegionKHR::default()
                .device_address(base + region_size)
                .stride(stride)
                .size(region_size);
            let hit_region = vk::StridedDeviceAddressRegionKHR::default()
                .device_address(base + region_size * 2)
                .stride(stride)
                .size(region_size);
            let callable_region = vk::StridedDeviceAddressRegionKHR::default();

            Ok(Self {
                device,
                pipeline,
                layout,
                _sbt: sbt,
                raygen_region,
                miss_region,
                hit_region,
                callable_region,
            })
        }
    }

    pub(crate) fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub(crate) fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    pub(crate) fn regions(
        &self,
    ) -> (
        &vk::StridedDeviceAddressRegionKHR,
        &vk::StridedDeviceAddressRegionKHR,
        &vk::StridedDeviceAddressRegionKHR,
        &vk::StridedDeviceAddressRegionKHR,
    ) {
        (
            &self.raygen_region,
            &self.miss_region,
            &self.hit_region,
            &self.callable_region,
        )
    }
}

impl Drop for VulkanRaytracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.layout, None);
        }
    }
}
