//! Sampled 2D textures: DEFAULT-heap resource + UPLOAD copy + bindless SRV.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::rc::Rc;

use engine_core::EngineError;
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

            // DEFAULT-heap texture in COPY_DEST.
            let default_heap = heap(D3D12_HEAP_TYPE_DEFAULT);
            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: desc.width as u64,
                Height: desc.height,
                DepthOrArraySize: 1,
                MipLevels: 1,
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

            // Copyable footprint for the upload layout.
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut num_rows = 0u32;
            let mut row_size = 0u64;
            let mut total = 0u64;
            device.device.GetCopyableFootprints(
                &tex_desc,
                0,
                1,
                0,
                Some(&mut footprint),
                Some(&mut num_rows),
                Some(&mut row_size),
                Some(&mut total),
            );

            // UPLOAD buffer with row-padded pixel data.
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
            let dst_pitch = footprint.Footprint.RowPitch as usize;
            let src_pitch = row_size as usize;
            let base = (ptr as *mut u8).add(footprint.Offset as usize);
            for row in 0..num_rows as usize {
                std::ptr::copy_nonoverlapping(
                    pixels.as_ptr().add(row * src_pitch),
                    base.add(row * dst_pitch),
                    src_pitch,
                );
            }
            upload.Unmap(0, None);

            // Copy upload -> texture, then transition to shader-readable.
            device.immediate_submit(|list| {
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: ManuallyDrop::new(Some(std::mem::transmute_copy(&resource))),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: ManuallyDrop::new(Some(std::mem::transmute_copy(&upload))),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: footprint,
                    },
                };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
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

fn heap(ty: windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: ty,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    }
}

fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
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

fn transition(
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
