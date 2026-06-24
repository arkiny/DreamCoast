//! Hardware ray-tracing acceleration structures (Phase 8, inline path).
//!
//! Builds bottom-level (BLAS, one per mesh) and a single top-level (TLAS, over
//! instances) acceleration structure for a static scene, in one graphics-queue
//! one-shot submission (`commit` + `waitUntilCompleted`). Mirrors the role of
//! `rhi-vulkan/src/accel.rs` / `rhi-d3d12/src/accel.rs`. Compaction / refit /
//! dynamic rebuild are out of scope (static-scene assumption).
//!
//! The TLAS is encoded into the bindless argument buffer's `tlas` slot
//! (`device.rs::bind_tlas`) so the inline `RayQuery` path tracer
//! (`rt_path.slang` / `rt_trace.slang`) can `TraceRayInline(g.tlas, ...)`.

use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSArray;
use objc2_metal::{
    MTLAccelerationStructure, MTLAccelerationStructureCommandEncoder,
    MTLAccelerationStructureGeometryDescriptor, MTLAccelerationStructureInstanceDescriptorType,
    MTLAccelerationStructureInstanceOptions, MTLAccelerationStructureTriangleGeometryDescriptor,
    MTLAccelerationStructureUserIDInstanceDescriptor, MTLBuffer, MTLCommandBuffer,
    MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLIndexType,
    MTLInstanceAccelerationStructureDescriptor, MTLPackedFloat3, MTLPackedFloat4x3,
    MTLPrimitiveAccelerationStructureDescriptor, MTLResourceID, MTLResourceOptions,
};
use rhi_types::TlasInstance;

use crate::device::DeviceShared;
use crate::resources::MetalBuffer;
use crate::{Result, rhi_err};

/// A built scene's acceleration structures: N BLAS + one TLAS. Owns all backing
/// objects for the scene's lifetime. The TLAS is bound in the shader (the inline
/// `RayQuery` path); the BLASes must outlive the TLAS that references them.
pub struct MetalRaytracingScene {
    blases: Vec<Retained<ProtocolObject<dyn MTLAccelerationStructure>>>,
    tlas: Retained<ProtocolObject<dyn MTLAccelerationStructure>>,
    /// The instance-descriptor buffer the TLAS build read; kept alive for parity
    /// with the other backends (Metal copies it during the build, but holding it is
    /// cheap and matches the Vulkan/D3D12 ownership model).
    _instance_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl MetalRaytracingScene {
    /// The TLAS's argument-buffer resource handle, for binding into the bindless
    /// table's `tlas` slot (the inline path traces `g.tlas`).
    pub(crate) fn tlas_resource_id(&self) -> MTLResourceID {
        self.tlas.gpuResourceID()
    }

    /// Every acceleration structure (TLAS + all BLAS) for residency. When the TLAS
    /// is traced through the bindless argument buffer, Metal requires the instance
    /// AS *and* each referenced primitive AS to be made resident (`useResource`).
    pub(crate) fn acceleration_structures(
        &self,
    ) -> impl Iterator<Item = &Retained<ProtocolObject<dyn MTLAccelerationStructure>>> {
        std::iter::once(&self.tlas).chain(self.blases.iter())
    }

    /// Build all BLAS + the TLAS for a static scene. `geometries` pairs each mesh's
    /// vertex/index buffers with its counts; `instances` reference a BLAS by index
    /// and place it with a row-major 3x4 transform. Metal needs no special buffer
    /// usage for AS geometry inputs (unlike Vulkan device-address / D3D12).
    pub(crate) fn build(
        shared: &Rc<DeviceShared>,
        geometries: &[(&MetalBuffer, &MetalBuffer, rhi_types::BlasGeometry)],
        instances: &[TlasInstance],
    ) -> Result<Self> {
        let device = &shared.device;

        // ---- BLAS: descriptor + size query + AS creation (one scratch per build) ----
        // The geometry descriptor + its NSArray are retained in `BlasBuild` so they
        // outlive the deferred build command recording below (the build references
        // them; if they were dropped after the loop the BLAS would build empty).
        struct BlasBuild {
            descriptor: Retained<MTLPrimitiveAccelerationStructureDescriptor>,
            accel: Retained<ProtocolObject<dyn MTLAccelerationStructure>>,
            scratch: Retained<ProtocolObject<dyn MTLBuffer>>,
            _tri: Retained<MTLAccelerationStructureTriangleGeometryDescriptor>,
            _geos: Retained<NSArray<MTLAccelerationStructureGeometryDescriptor>>,
        }
        let mut builds: Vec<BlasBuild> = Vec::with_capacity(geometries.len());
        for (vbuf, ibuf, g) in geometries {
            let tri = MTLAccelerationStructureTriangleGeometryDescriptor::descriptor();
            tri.setVertexBuffer(Some(&vbuf.buffer));
            tri.setVertexStride(g.vertex_stride as usize);
            tri.setIndexBuffer(Some(&ibuf.buffer));
            tri.setIndexType(MTLIndexType::UInt32);
            tri.setTriangleCount((g.index_count / 3) as usize);
            // Mark opaque (matches Vulkan's GeometryFlagsKHR::OPAQUE): the inline
            // `RayQuery` does a single `next()`, which auto-commits only opaque hits.
            tri.setOpaque(true);

            let geo: &MTLAccelerationStructureGeometryDescriptor = &tri;
            let geos = NSArray::from_slice(&[geo]);
            let descriptor = MTLPrimitiveAccelerationStructureDescriptor::descriptor();
            descriptor.setGeometryDescriptors(Some(&geos));

            let sizes = device.accelerationStructureSizesWithDescriptor(&descriptor);
            let accel = device
                .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
                .ok_or_else(|| rhi_err("newAccelerationStructureWithSize (BLAS) failed"))?;
            let scratch = device
                .newBufferWithLength_options(
                    sizes.buildScratchBufferSize.max(1),
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| rhi_err("BLAS scratch alloc failed"))?;
            builds.push(BlasBuild {
                descriptor,
                accel,
                scratch,
                _tri: tri,
                _geos: geos,
            });
        }

        // ---- TLAS: instance array (user-ID descriptors) + size query + AS ----
        // `CommittedInstanceID()` in the shader maps to Metal's
        // `get_committed_user_instance_id()`, so the instances must carry an explicit
        // `userID` (= `custom_index`) via the UserID descriptor type.
        let stride = std::mem::size_of::<MTLAccelerationStructureUserIDInstanceDescriptor>();
        let instance_buffer = device
            .newBufferWithLength_options(
                (stride * instances.len()).max(stride),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| rhi_err("instance buffer alloc failed"))?;
        {
            let base = instance_buffer.contents().as_ptr()
                as *mut MTLAccelerationStructureUserIDInstanceDescriptor;
            for (i, inst) in instances.iter().enumerate() {
                let desc = MTLAccelerationStructureUserIDInstanceDescriptor {
                    transformationMatrix: packed_4x3(&inst.transform),
                    options: MTLAccelerationStructureInstanceOptions::DisableTriangleCulling,
                    mask: inst.mask as u32,
                    intersectionFunctionTableOffset: 0,
                    accelerationStructureIndex: inst.blas_index,
                    userID: inst.custom_index,
                };
                unsafe { base.add(i).write(desc) };
            }
        }

        let blas_array: Vec<_> = builds.iter().map(|b| b.accel.clone()).collect();
        let instanced = NSArray::from_retained_slice(&blas_array);
        let tlas_descriptor = MTLInstanceAccelerationStructureDescriptor::descriptor();
        tlas_descriptor
            .setInstanceDescriptorType(MTLAccelerationStructureInstanceDescriptorType::UserID);
        tlas_descriptor.setInstanceCount(instances.len());
        tlas_descriptor.setInstanceDescriptorBuffer(Some(&instance_buffer));
        tlas_descriptor.setInstancedAccelerationStructures(Some(&instanced));

        let tlas_sizes = device.accelerationStructureSizesWithDescriptor(&tlas_descriptor);
        let tlas = device
            .newAccelerationStructureWithSize(tlas_sizes.accelerationStructureSize)
            .ok_or_else(|| rhi_err("newAccelerationStructureWithSize (TLAS) failed"))?;
        let tlas_scratch = device
            .newBufferWithLength_options(
                tlas_sizes.buildScratchBufferSize.max(1),
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or_else(|| rhi_err("TLAS scratch alloc failed"))?;

        // ---- Record + submit: build all BLAS, then the TLAS (separate encoders so
        // the TLAS build, which references the BLASes, is ordered after them), then
        // block until the GPU finishes (one-shot static build). ----
        let cmd = shared
            .queue
            .commandBuffer()
            .ok_or_else(|| rhi_err("commandBuffer for AS build failed"))?;
        {
            let enc = cmd
                .accelerationStructureCommandEncoder()
                .ok_or_else(|| rhi_err("accelerationStructureCommandEncoder (BLAS) failed"))?;
            for b in &builds {
                enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                    &b.accel,
                    &b.descriptor,
                    &b.scratch,
                    0,
                );
            }
            enc.endEncoding();
        }
        {
            let enc = cmd
                .accelerationStructureCommandEncoder()
                .ok_or_else(|| rhi_err("accelerationStructureCommandEncoder (TLAS) failed"))?;
            enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                &tlas,
                &tlas_descriptor,
                &tlas_scratch,
                0,
            );
            enc.endEncoding();
        }
        cmd.commit();
        cmd.waitUntilCompleted();

        let blases = builds.into_iter().map(|b| b.accel).collect();
        Ok(Self {
            blases,
            tlas,
            _instance_buffer: instance_buffer,
        })
    }
}

/// Transpose a row-major 3x4 object-to-world transform (12 floats, translation in
/// the last column) into Metal's column-major `MTLPackedFloat4x3`.
fn packed_4x3(m: &[f32; 12]) -> MTLPackedFloat4x3 {
    let col = |c: usize| MTLPackedFloat3 {
        x: m[c],
        y: m[c + 4],
        z: m[c + 8],
    };
    MTLPackedFloat4x3 {
        columns: [col(0), col(1), col(2), col(3)],
    }
}
