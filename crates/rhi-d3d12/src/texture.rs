//! Sampled 2D textures: DEFAULT-heap resource + UPLOAD copy + bindless SRV.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::TextureDesc;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_UPLOAD, D3D12_MEMORY_POOL_UNKNOWN,
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
    D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_TEXTURE_LAYOUT_UNKNOWN, ID3D12GraphicsCommandList,
    ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

use crate::device::DeviceShared;
use crate::instance::d3d_err;
use crate::to_dxgi_format;

/// A device-local sampled texture registered in the bindless SRV heap.
pub struct D3d12Texture {
    #[allow(dead_code)]
    device: Rc<DeviceShared>,
    #[allow(dead_code)] // kept alive; referenced by the bindless SRV
    resource: ID3D12Resource,
    index: u32,
}

impl D3d12Texture {
    pub(crate) fn new(
        device: Rc<DeviceShared>,
        desc: &TextureDesc,
        pixels: &[u8],
    ) -> Result<Self, EngineError> {
        unsafe {
            let format = to_dxgi_format(desc.format);

            // CPU-generated mip chain (identical bytes across backends — the
            // cross-backend-parity rule; see rhi_types::generate_mip_chain).
            let levels =
                rhi_types::generate_mip_chain(pixels, desc.width, desc.height, desc.format);
            let mip_levels = levels.len() as u32;

            // DEFAULT-heap texture (full mip chain) in COPY_DEST.
            let default_heap = heap(D3D12_HEAP_TYPE_DEFAULT);
            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: desc.width as u64,
                Height: desc.height,
                DepthOrArraySize: 1,
                MipLevels: mip_levels as u16,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                ..Default::default()
            };
            let mut tex: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &default_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &tex_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut tex,
                )
                .map_err(d3d_err)?;
            let resource = tex.ok_or_else(|| EngineError::Rhi("texture was null".into()))?;

            // Copyable footprints for every subresource (mip).
            let n = mip_levels as usize;
            let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
            let mut num_rows = vec![0u32; n];
            let mut row_sizes = vec![0u64; n];
            let mut total = 0u64;
            device.device.GetCopyableFootprints(
                &tex_desc,
                0,
                mip_levels,
                0,
                Some(footprints.as_mut_ptr()),
                Some(num_rows.as_mut_ptr()),
                Some(row_sizes.as_mut_ptr()),
                Some(&mut total),
            );

            // UPLOAD buffer holding every mip level, each row-padded to its footprint.
            let upload_heap = heap(D3D12_HEAP_TYPE_UPLOAD);
            let upload_desc = buffer_desc(total);
            let mut upload: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &upload_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &upload_desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut upload,
                )
                .map_err(d3d_err)?;
            let upload = upload.ok_or_else(|| EngineError::Rhi("upload buffer null".into()))?;

            let mut ptr: *mut c_void = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut ptr)).map_err(d3d_err)?;
            let base = ptr as *mut u8;
            for (((fp, &src_pitch), &rows), level) in footprints
                .iter()
                .zip(row_sizes.iter())
                .zip(num_rows.iter())
                .zip(levels.iter())
            {
                let dst_pitch = fp.Footprint.RowPitch as usize;
                let src_pitch = src_pitch as usize;
                let rows = rows as usize;
                let dst = base.add(fp.Offset as usize);
                for row in 0..rows {
                    std::ptr::copy_nonoverlapping(
                        level.as_ptr().add(row * src_pitch),
                        dst.add(row * dst_pitch),
                        src_pitch,
                    );
                }
            }
            upload.Unmap(0, None);

            // Copy each mip upload -> its texture subresource, then transition all to read.
            device.immediate_submit(|list| {
                for (mip, fp) in footprints.iter().enumerate() {
                    let dst = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: ManuallyDrop::new(Some(std::mem::transmute_copy(&resource))),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: mip as u32,
                        },
                    };
                    let src = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: ManuallyDrop::new(Some(std::mem::transmute_copy(&upload))),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: *fp,
                        },
                    };
                    list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                }
                transition(
                    list,
                    &resource,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                );
            })?;

            let index = device.register_texture(&resource, desc.format);
            Ok(Self {
                device,
                resource,
                index,
            })
        }
    }

    pub fn bindless_index(&self) -> u32 {
        self.index
    }
}

pub(crate) fn heap(
    ty: windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE,
) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: ty,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    }
}

pub(crate) fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    }
}

pub(crate) fn transition(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
    after: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    unsafe { list.ResourceBarrier(&[barrier]) };
}
