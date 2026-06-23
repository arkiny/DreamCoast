//! Hardware ray-tracing acceleration structures (Phase 8).
//!
//! Builds bottom-level (BLAS, one per mesh) and a single top-level (TLAS, over
//! instances) acceleration structure for a static scene, in one graphics-queue
//! one-shot submission (`immediate_submit`). Compaction / refit / dynamic
//! rebuild are out of scope (static-scene assumption) — see the phase doc.

use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::TlasInstance;

use crate::buffer::VulkanBuffer;
use crate::device::DeviceShared;
use crate::vk_err;

/// A device-local buffer with a device address, used for AS storage, scratch,
/// and the TLAS instance array. Minimal owner with its own allocation.
struct RtBuffer {
    device: Arc<DeviceShared>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl RtBuffer {
    fn new(
        device: Arc<DeviceShared>,
        size: u64,
        usage: vk::BufferUsageFlags,
        host_visible: bool,
    ) -> Result<Self, EngineError> {
        unsafe {
            let usage = usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
            let ci = vk::BufferCreateInfo::default()
                .size(size.max(1))
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = device.device.create_buffer(&ci, None).map_err(vk_err)?;
            let req = device.device.get_buffer_memory_requirements(buffer);
            let props = if host_visible {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            } else {
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            };
            let mem_type = device.find_memory_type(req.memory_type_bits, props)?;
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

    /// Map and write `data` (host-visible buffers only).
    fn write(&self, data: &[u8]) -> Result<(), EngineError> {
        unsafe {
            let ptr = self
                .device
                .device
                .map_memory(
                    self.memory,
                    0,
                    data.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(vk_err)? as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            self.device.device.unmap_memory(self.memory);
        }
        Ok(())
    }
}

impl Drop for RtBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_buffer(self.buffer, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}

/// A bottom-level acceleration structure (one mesh). Its backing buffer must
/// outlive the TLAS that references it, so the scene owns it.
struct Blas {
    handle: vk::AccelerationStructureKHR,
    _buffer: RtBuffer,
}

/// A built scene's acceleration structures: N BLAS + one TLAS. Owns all backing
/// memory; lives for the scene. The TLAS handle is bound in the shader (M3+).
pub struct VulkanRaytracingScene {
    device: Arc<DeviceShared>,
    blases: Vec<Blas>,
    tlas: vk::AccelerationStructureKHR,
    _tlas_buffer: RtBuffer,
    _instance_buffer: RtBuffer,
}

impl VulkanRaytracingScene {
    /// The TLAS handle, for descriptor binding (Phase 8 M3 — inline trace).
    pub(crate) fn tlas(&self) -> vk::AccelerationStructureKHR {
        self.tlas
    }

    /// Build all BLAS + the TLAS for a static scene. `geometries` pairs each
    /// mesh's vertex/index buffers with its counts; `instances` reference a BLAS
    /// by index and place it with a 3x4 transform (Phase 8 M2).
    pub(crate) fn build(
        device: Arc<DeviceShared>,
        geometries: &[(&VulkanBuffer, &VulkanBuffer, rhi_types::BlasGeometry)],
        instances: &[TlasInstance],
    ) -> Result<Self, EngineError> {
        let accel = device
            .accel_loader
            .as_ref()
            .ok_or_else(|| EngineError::Rhi("ray tracing not available".into()))?;

        // ---- BLAS: size queries + AS creation (host side, no commands yet) ----
        struct BlasBuild {
            geo: vk::AccelerationStructureGeometryKHR<'static>,
            range: vk::AccelerationStructureBuildRangeInfoKHR,
            scratch_size: u64,
            handle: vk::AccelerationStructureKHR,
            buffer: RtBuffer,
            device_address: u64,
        }
        let mut builds: Vec<BlasBuild> = Vec::with_capacity(geometries.len());
        for (vbuf, ibuf, g) in geometries {
            let prim_count = g.index_count / 3;
            let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: vbuf.device_address(),
                })
                .vertex_stride(g.vertex_stride as u64)
                .max_vertex(g.vertex_count.saturating_sub(1))
                .index_type(vk::IndexType::UINT32)
                .index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: ibuf.device_address(),
                });
            let geo = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                .flags(vk::GeometryFlagsKHR::OPAQUE);

            let geos = [geo];
            let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&geos);
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            unsafe {
                accel.get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &[prim_count],
                    &mut sizes,
                );
            }
            let buffer = RtBuffer::new(
                device.clone(),
                sizes.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
                false,
            )?;
            let create_info = vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(buffer.buffer)
                .size(sizes.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
            let handle = unsafe {
                accel
                    .create_acceleration_structure(&create_info, None)
                    .map_err(vk_err)?
            };
            build_info = build_info.dst_acceleration_structure(handle);
            let device_address = unsafe {
                accel.get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(handle),
                )
            };
            // `geo`/`range` are stored to drive the deferred command recording; the
            // geometry's address fields are plain integers so the 'static lifetime
            // is sound (no borrows held).
            let geo_static: vk::AccelerationStructureGeometryKHR<'static> =
                unsafe { std::mem::transmute(geo) };
            builds.push(BlasBuild {
                geo: geo_static,
                range: vk::AccelerationStructureBuildRangeInfoKHR::default()
                    .primitive_count(prim_count),
                scratch_size: sizes.build_scratch_size,
                handle,
                buffer,
                device_address,
            });
        }

        // One scratch buffer sized to the largest BLAS build (builds run serially).
        let max_scratch = builds.iter().map(|b| b.scratch_size).max().unwrap_or(0);
        let blas_scratch = RtBuffer::new(
            device.clone(),
            max_scratch,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            false,
        )?;

        // ---- TLAS: instance array + size query + AS creation ----
        let mut instance_data: Vec<vk::AccelerationStructureInstanceKHR> =
            Vec::with_capacity(instances.len());
        for inst in instances {
            let blas = &builds[inst.blas_index as usize];
            let mut matrix = [0.0f32; 12];
            matrix.copy_from_slice(&inst.transform);
            instance_data.push(vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix },
                instance_custom_index_and_mask: vk::Packed24_8::new(inst.custom_index, inst.mask),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas.device_address,
                },
            });
        }
        let instance_bytes = std::mem::size_of_val(instance_data.as_slice());
        let instance_buffer = RtBuffer::new(
            device.clone(),
            instance_bytes as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            true,
        )?;
        instance_buffer.write(unsafe {
            std::slice::from_raw_parts(instance_data.as_ptr() as *const u8, instance_bytes)
        })?;

        let tlas_geo = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_buffer.device_address(),
                    }),
            })
            .flags(vk::GeometryFlagsKHR::OPAQUE);
        let tlas_geos = [tlas_geo];
        let tlas_prims = instances.len() as u32;
        let mut tlas_build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&tlas_geos);
        let mut tlas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            accel.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &tlas_build,
                &[tlas_prims],
                &mut tlas_sizes,
            );
        }
        let tlas_buffer = RtBuffer::new(
            device.clone(),
            tlas_sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            false,
        )?;
        let tlas = unsafe {
            accel
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(tlas_buffer.buffer)
                        .size(tlas_sizes.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
                    None,
                )
                .map_err(vk_err)?
        };
        tlas_build = tlas_build.dst_acceleration_structure(tlas);
        let tlas_scratch = RtBuffer::new(
            device.clone(),
            tlas_sizes.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            false,
        )?;

        // ---- Record + submit: build all BLAS, barrier, build TLAS, barrier ----
        device.immediate_submit(|cmd| unsafe {
            for b in &builds {
                let geos = [b.geo];
                let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .dst_acceleration_structure(b.handle)
                    .geometries(&geos)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: blas_scratch.device_address(),
                    });
                let range = [b.range];
                let ranges: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] = &[&range];
                accel.cmd_build_acceleration_structures(cmd, &[info], ranges);
                // Serialize successive BLAS builds (shared scratch) + make writes
                // visible to the TLAS build that reads them.
                accel_memory_barrier(&device.device, cmd);
            }
            let tlas_range = [
                vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(tlas_prims)
            ];
            let tlas_ranges: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] = &[&tlas_range];
            let tlas_info = tlas_build.scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: tlas_scratch.device_address(),
            });
            accel.cmd_build_acceleration_structures(cmd, &[tlas_info], tlas_ranges);
            accel_memory_barrier(&device.device, cmd);
        })?;

        let blases = builds
            .into_iter()
            .map(|b| Blas {
                handle: b.handle,
                _buffer: b.buffer,
            })
            .collect();

        Ok(Self {
            device,
            blases,
            tlas,
            _tlas_buffer: tlas_buffer,
            _instance_buffer: instance_buffer,
        })
    }
}

/// A global memory barrier covering acceleration-structure build read/write,
/// inserted between successive builds and after the TLAS build.
fn accel_memory_barrier(device: &ash::Device, cmd: vk::CommandBuffer) {
    unsafe {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(
                vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                    | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
                | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

impl Drop for VulkanRaytracingScene {
    fn drop(&mut self) {
        if let Some(accel) = self.device.accel_loader.as_ref() {
            unsafe {
                accel.destroy_acceleration_structure(self.tlas, None);
                for b in &self.blases {
                    accel.destroy_acceleration_structure(b.handle, None);
                }
            }
        }
    }
}
