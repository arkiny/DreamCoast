//! Hardware ray-tracing acceleration structures on D3D12 / DXR (Phase 8).
//!
//! Mirrors the Vulkan backend: BLAS per mesh + one TLAS over instances, built in
//! one DIRECT-queue one-shot submission (static scene). Uses `ID3D12Device5` /
//! `ID3D12GraphicsCommandList4` (cast from the base interfaces, guaranteed when
//! DXR Tier >= 1.1 is reported by `has_raytracing`).

use std::mem::ManuallyDrop;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::{BlasGeometry, TlasInstance};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC,
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    D3D12_ELEMENTS_LAYOUT_ARRAY, D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE, D3D12_HEAP_FLAG_NONE,
    D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_UPLOAD,
    D3D12_MEMORY_POOL_UNKNOWN,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL, D3D12_RAYTRACING_GEOMETRY_DESC,
    D3D12_RAYTRACING_GEOMETRY_DESC_0, D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
    D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC, D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
    D3D12_RAYTRACING_INSTANCE_DESC, D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_UAV, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_NONE,
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATES, D3D12_RESOURCE_UAV_BARRIER,
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12Device5, ID3D12GraphicsCommandList,
    ID3D12GraphicsCommandList4, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
};
use windows::core::Interface;

use crate::device::DeviceShared;
use crate::instance::d3d_err;

/// Create a committed DEFAULT/UPLOAD-heap buffer for AS storage, scratch, or the
/// instance array.
fn create_buffer(
    device: &DeviceShared,
    size: u64,
    upload: bool,
    uav: bool,
    initial: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, EngineError> {
    unsafe {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: if upload {
                D3D12_HEAP_TYPE_UPLOAD
            } else {
                D3D12_HEAP_TYPE_DEFAULT
            },
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let desc = D3D12_RESOURCE_DESC {
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
            Flags: if uav {
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_FLAG_NONE
            },
        };
        let mut res: Option<ID3D12Resource> = None;
        device
            .device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, initial, None, &mut res)
            .map_err(d3d_err)?;
        res.ok_or_else(|| EngineError::Rhi("AS buffer null".into()))
    }
}

/// A built scene's acceleration structures: N BLAS + one TLAS (Phase 8 M2). Owns
/// the AS buffers (which must outlive any trace that uses the TLAS); scratch and
/// instance buffers are released after the build completes.
pub struct D3d12RaytracingScene {
    _blas_buffers: Vec<ID3D12Resource>,
    tlas: ID3D12Resource,
}

impl D3d12RaytracingScene {
    /// GPU virtual address of the TLAS, for the `RaytracingAccelerationStructure`
    /// SRV bound in the shader (Phase 8 M3 — inline trace).
    pub(crate) fn tlas_gpu_va(&self) -> u64 {
        unsafe { self.tlas.GetGPUVirtualAddress() }
    }

    pub(crate) fn build(
        device: Rc<DeviceShared>,
        geometries: &[(
            &crate::buffer::D3d12Buffer,
            &crate::buffer::D3d12Buffer,
            BlasGeometry,
        )],
        instances: &[TlasInstance],
    ) -> Result<Self, EngineError> {
        unsafe {
            let device5: ID3D12Device5 = device.device.cast().map_err(d3d_err)?;

            // ---- BLAS: geometry descs + prebuild sizes + AS/scratch buffers ----
            struct BlasBuild {
                geo: D3D12_RAYTRACING_GEOMETRY_DESC,
                buffer: ID3D12Resource,
                scratch: ID3D12Resource,
            }
            let mut builds: Vec<BlasBuild> = Vec::with_capacity(geometries.len());
            for (vbuf, ibuf, g) in geometries {
                let vbuf = vbuf.resource();
                let ibuf = ibuf.resource();
                let geo = D3D12_RAYTRACING_GEOMETRY_DESC {
                    Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                    Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                    Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                        Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                            Transform3x4: 0,
                            IndexFormat: DXGI_FORMAT_R32_UINT,
                            VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                            IndexCount: g.index_count,
                            VertexCount: g.vertex_count,
                            IndexBuffer: ibuf.GetGPUVirtualAddress(),
                            VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                                StartAddress: vbuf.GetGPUVirtualAddress(),
                                StrideInBytes: g.vertex_stride as u64,
                            },
                        },
                    },
                };
                let geos = [geo];
                let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                    Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                    NumDescs: 1,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        pGeometryDescs: geos.as_ptr(),
                    },
                };
                let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
                device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut info);
                let buffer = create_buffer(
                    &device,
                    info.ResultDataMaxSizeInBytes,
                    false,
                    true,
                    D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                )?;
                let scratch = create_buffer(
                    &device,
                    info.ScratchDataSizeInBytes,
                    false,
                    true,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )?;
                builds.push(BlasBuild {
                    geo,
                    buffer,
                    scratch,
                });
            }

            // ---- TLAS: instance descs (UPLOAD) + prebuild + AS/scratch ----
            let mut instance_descs: Vec<D3D12_RAYTRACING_INSTANCE_DESC> =
                Vec::with_capacity(instances.len());
            for inst in instances {
                let blas_va = builds[inst.blas_index as usize]
                    .buffer
                    .GetGPUVirtualAddress();
                // `Transform` is a flat row-major 3x4 (12 floats), matching our layout.
                let mut desc = D3D12_RAYTRACING_INSTANCE_DESC {
                    Transform: inst.transform,
                    ..Default::default()
                };
                // _bitfield: InstanceID (24) | InstanceMask (8).
                desc._bitfield1 = (inst.custom_index & 0x00FF_FFFF) | ((inst.mask as u32) << 24);
                // _bitfield: HitGroupIndex (24) | Flags (8).
                desc._bitfield2 =
                    (D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE.0 as u32) << 24;
                desc.AccelerationStructure = blas_va;
                instance_descs.push(desc);
            }
            let instance_bytes = std::mem::size_of_val(instance_descs.as_slice());
            let instance_buffer = create_buffer(
                &device,
                instance_bytes as u64,
                true,
                false,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            // Upload instance descs into the mapped UPLOAD buffer.
            {
                let mut ptr = std::ptr::null_mut();
                instance_buffer
                    .Map(0, None, Some(&mut ptr))
                    .map_err(d3d_err)?;
                std::ptr::copy_nonoverlapping(
                    instance_descs.as_ptr() as *const u8,
                    ptr as *mut u8,
                    instance_bytes,
                );
                instance_buffer.Unmap(0, None);
            }
            let tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                NumDescs: instances.len() as u32,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    InstanceDescs: instance_buffer.GetGPUVirtualAddress(),
                },
            };
            let mut tlas_info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
            device5.GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &mut tlas_info);
            let tlas = create_buffer(
                &device,
                tlas_info.ResultDataMaxSizeInBytes,
                false,
                true,
                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            )?;
            let tlas_scratch = create_buffer(
                &device,
                tlas_info.ScratchDataSizeInBytes,
                false,
                true,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?;

            // ---- Record + submit: build all BLAS, UAV barrier, build TLAS ----
            device.immediate_submit(|list: &ID3D12GraphicsCommandList| {
                let list4: ID3D12GraphicsCommandList4 =
                    list.cast().expect("CommandList4 (DXR available)");
                for b in &builds {
                    let geos = [b.geo];
                    let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                        NumDescs: 1,
                        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                            pGeometryDescs: geos.as_ptr(),
                        },
                    };
                    let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                        DestAccelerationStructureData: b.buffer.GetGPUVirtualAddress(),
                        Inputs: inputs,
                        SourceAccelerationStructureData: 0,
                        ScratchAccelerationStructureData: b.scratch.GetGPUVirtualAddress(),
                    };
                    list4.BuildRaytracingAccelerationStructure(&desc, None);
                    uav_barrier(list, &b.buffer);
                }
                let tlas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: tlas.GetGPUVirtualAddress(),
                    Inputs: tlas_inputs,
                    SourceAccelerationStructureData: 0,
                    ScratchAccelerationStructureData: tlas_scratch.GetGPUVirtualAddress(),
                };
                list4.BuildRaytracingAccelerationStructure(&tlas_desc, None);
                uav_barrier(list, &tlas);
            })?;

            let blas_buffers = builds.into_iter().map(|b| b.buffer).collect();
            // `instance_buffer` / scratch buffers drop here (GPU is idle).
            Ok(Self {
                _blas_buffers: blas_buffers,
                tlas,
            })
        }
    }
}

/// A UAV barrier on `resource`, serializing AS builds and making writes visible
/// to the TLAS build / later traces.
fn uav_barrier(list: &ID3D12GraphicsCommandList, resource: &ID3D12Resource) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    };
    unsafe { list.ResourceBarrier(&[barrier]) };
}
