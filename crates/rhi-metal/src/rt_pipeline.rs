//! Metal Shader Converter ray-tracing pipeline (Phase 8 / M7).
//!
//! The converter maps DXR's `DispatchRays` model onto an indirect ray-dispatch
//! compute kernel plus visible/intersection function tables. This module
//! implements the first DreamCoast shape: one raygen shader, one miss shader, and
//! one opaque-triangle closest-hit group, with no local root signature data.

use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSDictionary, NSString};
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineDescriptor, MTLComputePipelineState,
    MTLDevice, MTLFunction, MTLFunctionHandle, MTLIntersectionFunctionSignature,
    MTLIntersectionFunctionTable, MTLIntersectionFunctionTableDescriptor, MTLLinkedFunctions,
    MTLPipelineOption, MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSize, MTLTexture,
    MTLVisibleFunctionTable, MTLVisibleFunctionTableDescriptor,
};
use rhi_types::RaytracingPipelineDesc;

use crate::accel::RT_PIPELINE_INTERSECTION_FUNCTION_OFFSET;
use crate::device::{
    BINDLESS_COUNT, CUBE_COUNT, DeviceShared, STORAGE_BUFFER_COUNT, STORAGE_IMAGE_COUNT,
    resource_id_bits,
};
use crate::{Result, rhi_err};

const RT_STORAGE_IMAGE_BASE: usize = (BINDLESS_COUNT + CUBE_COUNT) as usize;
const RT_CUBE_BASE: usize = BINDLESS_COUNT as usize;
const RT_STORAGE_BUFFER_BASE: usize = RT_STORAGE_IMAGE_BASE + STORAGE_IMAGE_COUNT as usize;
const RT_TLAS_SLOT: usize = RT_STORAGE_BUFFER_BASE + STORAGE_BUFFER_COUNT as usize;
const RT_DESCRIPTOR_TABLE_SLOTS: usize = RT_TLAS_SLOT + 1;

const TLAB_DESCRIPTOR_TABLE_OFFSET: usize = 0;
const TLAB_PUSH_CONSTANT_OFFSET: usize = 8;
const RT_PUSH_CONSTANT_BYTES: usize = 128;
const TLAB_BYTES: usize = TLAB_PUSH_CONSTANT_OFFSET + RT_PUSH_CONSTANT_BYTES;

const SBT_RECORD_STRIDE: u64 = 64;
const SBT_RECORDS: u64 = 3;
const INTERSECTION_FUNCTION_HANDLE: u64 = RT_PIPELINE_INTERSECTION_FUNCTION_OFFSET as u64;
const MISS_SHADER_HANDLE: u64 = 1;
const HIT_SHADER_HANDLE: u64 = 2;
const RAYGEN_SHADER_HANDLE: u64 = 3;

const THREADS_PER_GROUP: MTLSize = MTLSize {
    width: 8,
    height: 8,
    depth: 1,
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IrDescriptorTableEntry {
    gpu_va: u64,
    texture_view_id: u64,
    metadata: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IrShaderIdentifier {
    intersection_shader_handle: u64,
    shader_handle: u64,
    local_root_signature_samplers_buffer: u64,
    pad0: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IrVirtualAddressRange {
    start_address: u64,
    size_in_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IrVirtualAddressRangeAndStride {
    start_address: u64,
    size_in_bytes: u64,
    stride_in_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IrDispatchRaysDescriptor {
    ray_generation_shader_record: IrVirtualAddressRange,
    miss_shader_table: IrVirtualAddressRangeAndStride,
    hit_group_table: IrVirtualAddressRangeAndStride,
    callable_shader_table: IrVirtualAddressRangeAndStride,
    width: u32,
    height: u32,
    depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IrDispatchRaysArgument {
    dispatch_rays_desc: IrDispatchRaysDescriptor,
    grs: u64,
    res_desc_heap: u64,
    smp_desc_heap: u64,
    visible_function_table: u64,
    intersection_function_table: u64,
    intersection_function_tables: u64,
}

pub struct MetalRaytracingPipeline {
    #[allow(dead_code)]
    shared: Rc<DeviceShared>,
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    visible_table: Retained<ProtocolObject<dyn MTLVisibleFunctionTable>>,
    intersection_table: Retained<ProtocolObject<dyn MTLIntersectionFunctionTable>>,
    sbt: Retained<ProtocolObject<dyn MTLBuffer>>,
    descriptor_table: Retained<ProtocolObject<dyn MTLBuffer>>,
    tlab: Retained<ProtocolObject<dyn MTLBuffer>>,
    dispatch_args: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl MetalRaytracingPipeline {
    pub(crate) fn new(shared: Rc<DeviceShared>, desc: &RaytracingPipelineDesc) -> Result<Self> {
        if desc.push_constant_size as usize > RT_PUSH_CONSTANT_BYTES {
            return Err(rhi_err(
                "Metal RT pipeline push constants exceed converter TLAB",
            ));
        }

        let raygen =
            crate::pipeline::load_function(&shared.device, desc.raygen_bytes, desc.raygen_entry)?;
        let ray_dispatch = match desc.metal_ray_dispatch_bytes {
            Some(bytes) => crate::pipeline::load_function(
                &shared.device,
                bytes,
                desc.metal_ray_dispatch_entry.unwrap_or("RaygenIndirection"),
            )?,
            None => raygen.clone(),
        };
        let miss =
            crate::pipeline::load_function(&shared.device, desc.miss_bytes, desc.miss_entry)?;
        let closesthit = crate::pipeline::load_function(
            &shared.device,
            desc.closesthit_bytes,
            desc.closesthit_entry,
        )?;
        let intersection = match desc.metal_intersection_bytes {
            Some(bytes) => Some(crate::pipeline::load_function(
                &shared.device,
                bytes,
                desc.metal_intersection_entry
                    .unwrap_or("irconverter.wrapper.intersection.function.triangle"),
            )?),
            None => None,
        };

        let linked = MTLLinkedFunctions::linkedFunctions();
        let mut functions = vec![raygen.clone(), miss.clone(), closesthit.clone()];
        if let Some(intersection) = intersection.as_ref() {
            functions.push(intersection.clone());
        }
        let linked_functions = NSArray::from_retained_slice(&functions);
        linked.setFunctions(Some(&linked_functions));
        let raygen_group_functions = NSArray::from_retained_slice(std::slice::from_ref(&raygen));
        let miss_group_functions = NSArray::from_retained_slice(std::slice::from_ref(&miss));
        let closesthit_group_functions =
            NSArray::from_retained_slice(std::slice::from_ref(&closesthit));
        let raygen_group = NSString::from_str("rayGen");
        let miss_group = NSString::from_str("miss");
        let closesthit_group = NSString::from_str("closestHit");
        let groups = NSDictionary::from_slices(
            &[&*raygen_group, &*miss_group, &*closesthit_group],
            &[
                &*raygen_group_functions,
                &*miss_group_functions,
                &*closesthit_group_functions,
            ],
        );
        linked.setGroups(Some(&groups));

        let pd = MTLComputePipelineDescriptor::new();
        pd.setComputeFunction(Some(&ray_dispatch));
        pd.setLinkedFunctions(Some(&linked));
        pd.setMaxCallStackDepth(8);
        let state = shared
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &pd,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|e| rhi_err(format!("new RT compute pipeline failed: {e}")))?;

        let visible_table = create_visible_table(&state, &raygen, &miss, &closesthit)?;
        let intersection_table = create_intersection_table(&state, intersection.as_deref())?;

        let sbt = shared
            .device
            .newBufferWithLength_options(
                (SBT_RECORD_STRIDE * SBT_RECORDS) as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| rhi_err("RT SBT buffer alloc failed"))?;
        init_sbt(&sbt);

        let descriptor_table = shared
            .device
            .newBufferWithLength_options(
                RT_DESCRIPTOR_TABLE_SLOTS * std::mem::size_of::<IrDescriptorTableEntry>(),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| rhi_err("RT descriptor table alloc failed"))?;
        let tlab = shared
            .device
            .newBufferWithLength_options(TLAB_BYTES, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| rhi_err("RT TLAB alloc failed"))?;
        let dispatch_args = shared
            .device
            .newBufferWithLength_options(
                std::mem::size_of::<IrDispatchRaysArgument>(),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| rhi_err("RT dispatch args alloc failed"))?;

        Ok(Self {
            shared,
            state,
            visible_table,
            intersection_table,
            sbt,
            descriptor_table,
            tlab,
            dispatch_args,
        })
    }

    pub(crate) fn state(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.state
    }

    pub(crate) fn threads_per_group(&self) -> MTLSize {
        THREADS_PER_GROUP
    }

    pub(crate) fn encode_dispatch(
        &self,
        shared: &DeviceShared,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        push_constants: &[u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        if push_constants.len() > RT_PUSH_CONSTANT_BYTES {
            return Err(rhi_err("Metal RT push constant block too large"));
        }

        self.write_descriptor_table(shared)?;
        self.write_tlab(push_constants);
        self.write_dispatch_args(width, height);
        self.mark_resources_resident(shared, enc);

        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.tlab), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.dispatch_args), 0, 3);
        }
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: width as usize,
                height: height as usize,
                depth: 1,
            },
            THREADS_PER_GROUP,
        );
        Ok(())
    }

    fn write_descriptor_table(&self, shared: &DeviceShared) -> Result<()> {
        let ptr = self.descriptor_table.contents().as_ptr() as *mut IrDescriptorTableEntry;
        unsafe {
            std::ptr::write_bytes(ptr, 0, RT_DESCRIPTOR_TABLE_SLOTS);
        }

        {
            let sampled_textures = shared.sampled_textures();
            for (i, texture) in sampled_textures
                .iter()
                .take(BINDLESS_COUNT as usize)
                .enumerate()
            {
                if let Some(texture) = texture {
                    unsafe {
                        *ptr.add(i) = IrDescriptorTableEntry {
                            texture_view_id: resource_id_bits(texture.gpuResourceID()),
                            ..Default::default()
                        };
                    }
                }
            }
        }

        {
            let cube_textures = shared.cube_textures();
            for (i, texture) in cube_textures.iter().take(CUBE_COUNT as usize).enumerate() {
                if let Some(texture) = texture {
                    unsafe {
                        *ptr.add(RT_CUBE_BASE + i) = IrDescriptorTableEntry {
                            texture_view_id: resource_id_bits(texture.gpuResourceID()),
                            ..Default::default()
                        };
                    }
                }
            }
        }

        {
            let storage_images = shared.storage_images();
            for (i, texture) in storage_images
                .iter()
                .take(STORAGE_IMAGE_COUNT as usize)
                .enumerate()
            {
                if let Some(texture) = texture {
                    unsafe {
                        *ptr.add(RT_STORAGE_IMAGE_BASE + i) = IrDescriptorTableEntry {
                            texture_view_id: resource_id_bits(texture.gpuResourceID()),
                            ..Default::default()
                        };
                    }
                }
            }
        }

        {
            let storage_buffers = shared.storage_buffers();
            // Enumerate over the slot vector directly so `i` is the true slot index (freed slots
            // are `None` and skipped — their RT descriptor entry stays zeroed).
            for (i, slot) in storage_buffers
                .iter()
                .take(STORAGE_BUFFER_COUNT as usize)
                .enumerate()
            {
                if let Some(buffer) = slot {
                    unsafe {
                        *ptr.add(RT_STORAGE_BUFFER_BASE + i) = IrDescriptorTableEntry {
                            gpu_va: buffer.gpuAddress(),
                            metadata: buffer.length() as u64,
                            ..Default::default()
                        };
                    }
                }
            }
        }

        let tlas = shared
            .rt_tlas_binding()
            .ok_or_else(|| rhi_err("RT pipeline trace without a bound TLAS"))?;
        unsafe {
            *ptr.add(RT_TLAS_SLOT) = IrDescriptorTableEntry {
                gpu_va: tlas.header.gpuAddress(),
                ..Default::default()
            };
        }

        Ok(())
    }

    fn write_tlab(&self, push_constants: &[u8]) {
        let base = self.tlab.contents().as_ptr() as *mut u8;
        let table_va = self.descriptor_table.gpuAddress();
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&table_va as *const u64).cast::<u8>(),
                base.add(TLAB_DESCRIPTOR_TABLE_OFFSET),
                std::mem::size_of::<u64>(),
            );
            std::ptr::write_bytes(
                base.add(TLAB_PUSH_CONSTANT_OFFSET),
                0,
                RT_PUSH_CONSTANT_BYTES,
            );
            std::ptr::copy_nonoverlapping(
                push_constants.as_ptr(),
                base.add(TLAB_PUSH_CONSTANT_OFFSET),
                push_constants.len(),
            );
        }
    }

    fn write_dispatch_args(&self, width: u32, height: u32) {
        let sbt_base = self.sbt.gpuAddress();
        let descriptor_table_va = self.descriptor_table.gpuAddress();
        let args = IrDispatchRaysArgument {
            dispatch_rays_desc: IrDispatchRaysDescriptor {
                ray_generation_shader_record: IrVirtualAddressRange {
                    start_address: sbt_base,
                    size_in_bytes: SBT_RECORD_STRIDE,
                },
                miss_shader_table: IrVirtualAddressRangeAndStride {
                    start_address: sbt_base + SBT_RECORD_STRIDE,
                    size_in_bytes: SBT_RECORD_STRIDE,
                    stride_in_bytes: SBT_RECORD_STRIDE,
                },
                hit_group_table: IrVirtualAddressRangeAndStride {
                    start_address: sbt_base + SBT_RECORD_STRIDE * 2,
                    size_in_bytes: SBT_RECORD_STRIDE,
                    stride_in_bytes: SBT_RECORD_STRIDE,
                },
                callable_shader_table: IrVirtualAddressRangeAndStride::default(),
                width,
                height,
                depth: 1,
            },
            grs: self.tlab.gpuAddress(),
            res_desc_heap: descriptor_table_va,
            smp_desc_heap: 0,
            visible_function_table: resource_id_bits(self.visible_table.gpuResourceID()),
            intersection_function_table: resource_id_bits(self.intersection_table.gpuResourceID()),
            intersection_function_tables: 0,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&args as *const IrDispatchRaysArgument).cast::<u8>(),
                self.dispatch_args.contents().as_ptr() as *mut u8,
                std::mem::size_of::<IrDispatchRaysArgument>(),
            );
        }
    }

    fn mark_resources_resident(
        &self,
        shared: &DeviceShared,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) {
        use_resource(enc, &self.sbt, MTLResourceUsage::Read);
        use_resource(enc, &self.descriptor_table, MTLResourceUsage::Read);
        use_resource(enc, &self.tlab, MTLResourceUsage::Read);
        use_resource(enc, &self.dispatch_args, MTLResourceUsage::Read);

        let visible: &ProtocolObject<dyn MTLResource> =
            ProtocolObject::from_ref(&*self.visible_table);
        enc.useResource_usage(visible, MTLResourceUsage::Read);
        let intersection: &ProtocolObject<dyn MTLResource> =
            ProtocolObject::from_ref(&*self.intersection_table);
        enc.useResource_usage(intersection, MTLResourceUsage::Read);

        if let Some(tlas) = shared.rt_tlas_binding() {
            use_resource(enc, &tlas.header, MTLResourceUsage::Read);
            use_resource(enc, &tlas.contributions, MTLResourceUsage::Read);
        }
        for tex in shared.resident_textures().iter() {
            let res: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**tex);
            enc.useResource_usage(res, MTLResourceUsage::Read);
        }
        for tex in shared.storage_resident_textures().iter() {
            let res: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**tex);
            enc.useResource_usage(res, MTLResourceUsage::Read | MTLResourceUsage::Write);
        }
        for buf in shared.storage_buffers().iter().flatten() {
            let res: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**buf);
            enc.useResource_usage(res, MTLResourceUsage::Read | MTLResourceUsage::Write);
        }
        for accel in shared.rt_acceleration_structures().iter() {
            let res: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**accel);
            enc.useResource_usage(res, MTLResourceUsage::Read);
        }
    }
}

fn create_visible_table(
    state: &ProtocolObject<dyn MTLComputePipelineState>,
    raygen: &ProtocolObject<dyn MTLFunction>,
    miss: &ProtocolObject<dyn MTLFunction>,
    closesthit: &ProtocolObject<dyn MTLFunction>,
) -> Result<Retained<ProtocolObject<dyn MTLVisibleFunctionTable>>> {
    let vd = MTLVisibleFunctionTableDescriptor::visibleFunctionTableDescriptor();
    unsafe { vd.setFunctionCount(RAYGEN_SHADER_HANDLE as usize + 1) };
    let table = state
        .newVisibleFunctionTableWithDescriptor(&vd)
        .ok_or_else(|| rhi_err("newVisibleFunctionTable failed"))?;
    let raygen_handle = function_handle(state, raygen, "raygen")?;
    let miss_handle = function_handle(state, miss, "miss")?;
    let hit_handle = function_handle(state, closesthit, "closest-hit")?;
    unsafe {
        table.setFunction_atIndex(Some(&miss_handle), MISS_SHADER_HANDLE as usize);
        table.setFunction_atIndex(Some(&hit_handle), HIT_SHADER_HANDLE as usize);
        table.setFunction_atIndex(Some(&raygen_handle), RAYGEN_SHADER_HANDLE as usize);
    }
    Ok(table)
}

fn function_handle(
    state: &ProtocolObject<dyn MTLComputePipelineState>,
    function: &ProtocolObject<dyn MTLFunction>,
    label: &str,
) -> Result<Retained<ProtocolObject<dyn MTLFunctionHandle>>> {
    state
        .functionHandleWithFunction(function)
        .ok_or_else(|| rhi_err(format!("RT pipeline missing {label} function handle")))
}

fn create_intersection_table(
    state: &ProtocolObject<dyn MTLComputePipelineState>,
    intersection: Option<&ProtocolObject<dyn MTLFunction>>,
) -> Result<Retained<ProtocolObject<dyn MTLIntersectionFunctionTable>>> {
    let id = MTLIntersectionFunctionTableDescriptor::intersectionFunctionTableDescriptor();
    id.setFunctionCount(INTERSECTION_FUNCTION_HANDLE as usize + 1);
    let table = state
        .newIntersectionFunctionTableWithDescriptor(&id)
        .ok_or_else(|| rhi_err("newIntersectionFunctionTable failed"))?;
    if let Some(intersection) = intersection {
        let handle = function_handle(state, intersection, "intersection-wrapper")?;
        table.setFunction_atIndex(Some(&handle), INTERSECTION_FUNCTION_HANDLE as usize);
    } else {
        unsafe {
            table.setOpaqueTriangleIntersectionFunctionWithSignature_atIndex(
                MTLIntersectionFunctionSignature::TriangleData,
                INTERSECTION_FUNCTION_HANDLE as usize,
            );
        }
    }
    Ok(table)
}

fn init_sbt(sbt: &ProtocolObject<dyn MTLBuffer>) {
    let ptr = sbt.contents().as_ptr() as *mut u8;
    unsafe {
        std::ptr::write_bytes(ptr, 0, (SBT_RECORD_STRIDE * SBT_RECORDS) as usize);
        write_shader_identifier(ptr, 0, RAYGEN_SHADER_HANDLE);
        write_shader_identifier(ptr, SBT_RECORD_STRIDE as usize, MISS_SHADER_HANDLE);
        write_shader_identifier(ptr, (SBT_RECORD_STRIDE * 2) as usize, HIT_SHADER_HANDLE);
    }
}

unsafe fn write_shader_identifier(base: *mut u8, offset: usize, shader_handle: u64) {
    let id = IrShaderIdentifier {
        shader_handle,
        ..Default::default()
    };
    unsafe {
        std::ptr::copy_nonoverlapping(
            (&id as *const IrShaderIdentifier).cast::<u8>(),
            base.add(offset),
            std::mem::size_of::<IrShaderIdentifier>(),
        );
    }
}

fn use_resource(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    usage: MTLResourceUsage,
) {
    let res: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(buffer);
    enc.useResource_usage(res, usage);
}
