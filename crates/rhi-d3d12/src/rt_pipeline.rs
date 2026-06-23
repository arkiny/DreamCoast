//! Hardware ray-tracing pipeline (state object) + shader binding table on
//! D3D12 / DXR (Phase 8 M5).
//!
//! Mirrors the Vulkan backend: one raygen / one miss / one closest-hit shader
//! (a single triangle hit group) assembled into a ray-tracing state object, with
//! a three-record shader binding table (raygen / miss / hit). The state object's
//! global root signature reuses the bindless layout (descriptor table over the
//! shared heap + 32-bit root constants at `b0`), so `DispatchRays` reaches the
//! same bindless resources the compute path uses. `MaxTraceRecursionDepth` is 1
//! (the bounce loop re-issues `TraceRay` from raygen; shadows use inline RayQuery).

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::RaytracingPipelineDesc;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMPARISON_FUNC_NEVER, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_DISPATCH_RAYS_DESC,
    D3D12_DXIL_LIBRARY_DESC, D3D12_EXPORT_DESC, D3D12_FILTER_MIN_MAG_MIP_LINEAR,
    D3D12_GLOBAL_ROOT_SIGNATURE, D3D12_GPU_VIRTUAL_ADDRESS_RANGE,
    D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_TYPE_UPLOAD, D3D12_HIT_GROUP_DESC, D3D12_HIT_GROUP_TYPE_TRIANGLES,
    D3D12_MEMORY_POOL_UNKNOWN, D3D12_RAYTRACING_PIPELINE_CONFIG, D3D12_RAYTRACING_SHADER_CONFIG,
    D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_NONE,
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_ROOT_CONSTANTS, D3D12_ROOT_DESCRIPTOR_TABLE,
    D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0, D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
    D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_NONE, D3D12_SHADER_BYTECODE, D3D12_SHADER_VISIBILITY_ALL,
    D3D12_STATE_OBJECT_DESC, D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, D3D12_STATE_SUBOBJECT,
    D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE,
    D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG,
    D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG,
    D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK, D3D12_STATIC_SAMPLER_DESC,
    D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12Device5,
    ID3D12Resource, ID3D12RootSignature, ID3D12StateObject, ID3D12StateObjectProperties,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::core::{Interface, PCWSTR};

use crate::device::DeviceShared;
use crate::instance::d3d_err;
use crate::pipeline::{bindless_ranges, serialize_and_create};

/// Shader identifier size (`D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES`).
const ID_SIZE: usize = 32;
/// Each SBT record is padded to 64 bytes so every region (raygen / miss / hit)
/// starts on a `D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT` (64) boundary.
const RECORD: u64 = 64;

/// A NUL-terminated UTF-16 string for the D3D12 wide-string APIs.
fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// A ray-tracing state object + its shader binding table.
pub struct D3d12RaytracingPipeline {
    #[allow(dead_code)]
    device: Rc<DeviceShared>,
    state_object: ID3D12StateObject,
    root_signature: ID3D12RootSignature,
    _sbt: ID3D12Resource,
    raygen: D3D12_GPU_VIRTUAL_ADDRESS_RANGE,
    miss: D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE,
    hit: D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE,
}

impl D3d12RaytracingPipeline {
    pub(crate) fn new(
        device: Rc<DeviceShared>,
        desc: &RaytracingPipelineDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let device5: ID3D12Device5 = device.device.cast().map_err(d3d_err)?;
            let root_signature = create_global_root_signature(&device, desc.push_constant_size)?;

            // ---- Export names + per-entry DXIL libraries ----
            let rg_export = wide(desc.raygen_entry);
            let ms_export = wide(desc.miss_entry);
            let ch_export = wide(desc.closesthit_entry);
            let hit_group_name = wide("HitGroup");

            let rg_desc = D3D12_EXPORT_DESC {
                Name: PCWSTR(rg_export.as_ptr()),
                ..Default::default()
            };
            let ms_desc = D3D12_EXPORT_DESC {
                Name: PCWSTR(ms_export.as_ptr()),
                ..Default::default()
            };
            let ch_desc = D3D12_EXPORT_DESC {
                Name: PCWSTR(ch_export.as_ptr()),
                ..Default::default()
            };
            let rg_lib = D3D12_DXIL_LIBRARY_DESC {
                DXILLibrary: shader_bytecode(desc.raygen_bytes),
                NumExports: 1,
                pExports: &rg_desc as *const _ as *mut _,
            };
            let ms_lib = D3D12_DXIL_LIBRARY_DESC {
                DXILLibrary: shader_bytecode(desc.miss_bytes),
                NumExports: 1,
                pExports: &ms_desc as *const _ as *mut _,
            };
            let ch_lib = D3D12_DXIL_LIBRARY_DESC {
                DXILLibrary: shader_bytecode(desc.closesthit_bytes),
                NumExports: 1,
                pExports: &ch_desc as *const _ as *mut _,
            };

            // ---- Hit group (closest hit only) ----
            let hit_group = D3D12_HIT_GROUP_DESC {
                HitGroupExport: PCWSTR(hit_group_name.as_ptr()),
                Type: D3D12_HIT_GROUP_TYPE_TRIANGLES,
                ClosestHitShaderImport: PCWSTR(ch_export.as_ptr()),
                ..Default::default()
            };

            // ---- Shader / pipeline config + global root signature ----
            let shader_config = D3D12_RAYTRACING_SHADER_CONFIG {
                MaxPayloadSizeInBytes: desc.max_payload_size,
                MaxAttributeSizeInBytes: desc.max_attribute_size,
            };
            let pipeline_config = D3D12_RAYTRACING_PIPELINE_CONFIG {
                MaxTraceRecursionDepth: 1,
            };
            let global_rs = D3D12_GLOBAL_ROOT_SIGNATURE {
                pGlobalRootSignature: ManuallyDrop::new(Some(root_signature.clone())),
            };

            let subobjects = [
                subobject(D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &rg_lib),
                subobject(D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &ms_lib),
                subobject(D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &ch_lib),
                subobject(D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &hit_group),
                subobject(
                    D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG,
                    &shader_config,
                ),
                subobject(
                    D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG,
                    &pipeline_config,
                ),
                subobject(D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &global_rs),
            ];
            let so_desc = D3D12_STATE_OBJECT_DESC {
                Type: D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
                NumSubobjects: subobjects.len() as u32,
                pSubobjects: subobjects.as_ptr(),
            };
            let state_object: ID3D12StateObject =
                device5.CreateStateObject(&so_desc).map_err(d3d_err)?;

            // `D3D12_GLOBAL_ROOT_SIGNATURE` wraps the root signature in a
            // `ManuallyDrop<Option<..>>`; release our extra ref now that the state
            // object holds its own (the field doesn't own the original `clone`).
            ManuallyDrop::into_inner(global_rs.pGlobalRootSignature);

            // ---- Shader binding table: raygen / miss / hit records ----
            let props: ID3D12StateObjectProperties = state_object.cast().map_err(d3d_err)?;
            let rg_id = props.GetShaderIdentifier(PCWSTR(rg_export.as_ptr()));
            let ms_id = props.GetShaderIdentifier(PCWSTR(ms_export.as_ptr()));
            let hg_id = props.GetShaderIdentifier(PCWSTR(hit_group_name.as_ptr()));

            let sbt = create_upload_buffer(&device, RECORD * 3)?;
            let mut ptr = std::ptr::null_mut();
            sbt.Map(0, None, Some(&mut ptr)).map_err(d3d_err)?;
            std::ptr::write_bytes(ptr as *mut u8, 0, (RECORD * 3) as usize);
            copy_id(rg_id, ptr as *mut u8, 0);
            copy_id(ms_id, ptr as *mut u8, RECORD as usize);
            copy_id(hg_id, ptr as *mut u8, (RECORD * 2) as usize);
            sbt.Unmap(0, None);

            let base = sbt.GetGPUVirtualAddress();
            let raygen = D3D12_GPU_VIRTUAL_ADDRESS_RANGE {
                StartAddress: base,
                SizeInBytes: ID_SIZE as u64,
            };
            let miss = D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE {
                StartAddress: base + RECORD,
                SizeInBytes: ID_SIZE as u64,
                StrideInBytes: ID_SIZE as u64,
            };
            let hit = D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE {
                StartAddress: base + RECORD * 2,
                SizeInBytes: ID_SIZE as u64,
                StrideInBytes: ID_SIZE as u64,
            };

            Ok(Self {
                device,
                state_object,
                root_signature,
                _sbt: sbt,
                raygen,
                miss,
                hit,
            })
        }
    }

    pub(crate) fn state_object(&self) -> &ID3D12StateObject {
        &self.state_object
    }

    pub(crate) fn root_signature(&self) -> &ID3D12RootSignature {
        &self.root_signature
    }

    /// The full dispatch-rays descriptor for a `width` x `height` trace.
    pub(crate) fn dispatch_desc(&self, width: u32, height: u32) -> D3D12_DISPATCH_RAYS_DESC {
        D3D12_DISPATCH_RAYS_DESC {
            RayGenerationShaderRecord: self.raygen,
            MissShaderTable: self.miss,
            HitGroupTable: self.hit,
            CallableShaderTable: D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE::default(),
            Width: width,
            Height: height,
            Depth: 1,
        }
    }
}

/// Build a state subobject pointing at `desc`.
fn subobject<T>(
    ty: windows::Win32::Graphics::Direct3D12::D3D12_STATE_SUBOBJECT_TYPE,
    desc: &T,
) -> D3D12_STATE_SUBOBJECT {
    D3D12_STATE_SUBOBJECT {
        Type: ty,
        pDesc: desc as *const T as *const c_void,
    }
}

/// Copy a 32-byte shader identifier into the SBT buffer at `offset`.
unsafe fn copy_id(id: *const c_void, dst: *mut u8, offset: usize) {
    unsafe { std::ptr::copy_nonoverlapping(id as *const u8, dst.add(offset), ID_SIZE) };
}

fn shader_bytecode(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr() as *const c_void,
        BytecodeLength: bytes.len(),
    }
}

/// Global root signature for the RT pipeline: the shared bindless descriptor
/// table (5 ranges, space1) + 32-bit root constants (`b0`) + the static sampler,
/// identical to the compute root signature so `DispatchRays` binds the same way.
fn create_global_root_signature(
    device: &DeviceShared,
    push_constant_size: u32,
) -> Result<ID3D12RootSignature, EngineError> {
    unsafe {
        let ranges = bindless_ranges();
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: ranges.len() as u32,
                        pDescriptorRanges: ranges.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: push_constant_size / 4,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MipLODBias: 0.0,
            MaxAnisotropy: 0,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            BorderColor: D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
            MinLOD: 0.0,
            MaxLOD: f32::MAX,
            ShaderRegister: 0,
            RegisterSpace: 1,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        serialize_and_create(
            device,
            &D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 1,
                pStaticSamplers: &sampler,
                Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            },
        )
    }
}

/// Create a host-visible UPLOAD buffer for the shader binding table.
fn create_upload_buffer(device: &DeviceShared, size: u64) -> Result<ID3D12Resource, EngineError> {
    unsafe {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let rdesc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut res: Option<ID3D12Resource> = None;
        device
            .device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &rdesc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut res,
            )
            .map_err(d3d_err)?;
        res.ok_or_else(|| EngineError::Rhi("SBT buffer null".into()))
    }
}
