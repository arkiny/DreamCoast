//! A 3D (volume) texture, usable as both a compute-writable storage volume
//! (bindless `storage_volumes[]`) and a trilinear-sampled volume (bindless
//! `volumes[]`). Phase 11 Stage B distance fields. Its current resource state is
//! tracked so the caller can barrier between baking (UNORDERED_ACCESS) and sampling
//! (NON_PIXEL_SHADER_RESOURCE), mirroring the 2D storage render target.

use std::cell::Cell;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::VolumeDesc;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_READBACK,
    D3D12_HEAP_TYPE_UPLOAD, D3D12_MEMORY_POOL_UNKNOWN, D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_TEXTURE3D,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATES,
    D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    D3D12_TEXTURE_LAYOUT_UNKNOWN, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use crate::device::DeviceShared;
use crate::instance::d3d_err;
use crate::texture::{buffer_desc, heap, transition};
use crate::to_dxgi_format;

/// A device-local 3D texture registered in both bindless volume tables.
pub struct D3d12Volume {
    #[allow(dead_code)] // keeps the GPU resource alive while its views are bound
    resource: ID3D12Resource,
    sampled_index: u32,
    storage_index: u32,
    state: Cell<D3D12_RESOURCE_STATES>,
}

impl D3d12Volume {
    pub(crate) fn new(device: Rc<DeviceShared>, desc: &VolumeDesc) -> Result<Self, EngineError> {
        unsafe {
            let heap_props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CPUPageProperty: Default::default(),
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            };
            let res_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
                Alignment: 0,
                Width: desc.width.max(1) as u64,
                Height: desc.height.max(1),
                DepthOrArraySize: desc.depth.max(1) as u16,
                MipLevels: 1,
                Format: to_dxgi_format(desc.format),
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            };
            // Created shader-readable; the caller transitions to UNORDERED_ACCESS
            // before the bake pass and back to NON_PIXEL_SHADER_RESOURCE to sample.
            let initial = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            let mut res: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &heap_props,
                    D3D12_HEAP_FLAG_NONE,
                    &res_desc,
                    initial,
                    None,
                    &mut res,
                )
                .map_err(d3d_err)?;
            let resource = res.ok_or_else(|| EngineError::Rhi("volume null".into()))?;

            let sampled_index = device.register_volume(&resource, desc.format);
            let storage_index =
                device.register_storage_volume(&resource, desc.format, desc.depth.max(1));
            Ok(Self {
                resource,
                sampled_index,
                storage_index,
                state: Cell::new(initial),
            })
        }
    }

    /// Create a 3D volume seeded with host `data` (Phase 12 M2: a CPU-baked SDF
    /// uploaded instead of a GPU bake). `data` is `width*height*depth` voxels in
    /// `x + dim*(y + dim*z)` order. Allocates the texture in COPY_DEST, copies the
    /// bytes through an UPLOAD buffer (respecting the 256-byte row pitch across all
    /// height*depth rows), and transitions to NON_PIXEL_SHADER_RESOURCE to sample.
    pub(crate) fn new_init(
        device: Rc<DeviceShared>,
        desc: &VolumeDesc,
        data: &[u8],
    ) -> Result<Self, EngineError> {
        unsafe {
            let res_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
                Alignment: 0,
                Width: desc.width.max(1) as u64,
                Height: desc.height.max(1),
                DepthOrArraySize: desc.depth.max(1) as u16,
                MipLevels: 1,
                Format: to_dxgi_format(desc.format),
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            };
            let mut res: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &heap(D3D12_HEAP_TYPE_DEFAULT),
                    D3D12_HEAP_FLAG_NONE,
                    &res_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut res,
                )
                .map_err(d3d_err)?;
            let resource = res.ok_or_else(|| EngineError::Rhi("volume null".into()))?;

            // Copyable footprint of the single 3D subresource: padded row pitch, the
            // row count per depth slice, the unpadded source row size, and total size.
            let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut num_rows = 0u32;
            let mut row_size = 0u64;
            let mut total = 0u64;
            device.device.GetCopyableFootprints(
                &res_desc,
                0,
                1,
                0,
                Some(&mut fp),
                Some(&mut num_rows),
                Some(&mut row_size),
                Some(&mut total),
            );

            let mut upload: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &heap(D3D12_HEAP_TYPE_UPLOAD),
                    D3D12_HEAP_FLAG_NONE,
                    &buffer_desc(total),
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut upload,
                )
                .map_err(d3d_err)?;
            let upload = upload.ok_or_else(|| EngineError::Rhi("upload buffer null".into()))?;

            // Fill the upload buffer: every (slice, row) tightly-packed source row
            // copied to its padded destination row. For a 3D texture the slices are
            // contiguous, so height*depth rows of `RowPitch` cover the whole volume.
            let dst_pitch = fp.Footprint.RowPitch as usize;
            let src_pitch = row_size as usize;
            let rows = num_rows as usize * desc.depth.max(1) as usize;
            let mut ptr: *mut c_void = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut ptr)).map_err(d3d_err)?;
            let base = (ptr as *mut u8).add(fp.Offset as usize);
            for row in 0..rows {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(row * src_pitch),
                    base.add(row * dst_pitch),
                    src_pitch.min(data.len().saturating_sub(row * src_pitch)),
                );
            }
            upload.Unmap(0, None);

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
                        PlacedFootprint: fp,
                    },
                };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                transition(
                    list,
                    &resource,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                );
            })?;

            let sampled_index = device.register_volume(&resource, desc.format);
            let storage_index =
                device.register_storage_volume(&resource, desc.format, desc.depth.max(1));
            Ok(Self {
                resource,
                sampled_index,
                storage_index,
                state: Cell::new(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            })
        }
    }

    /// Read the volume back to host memory (Phase 12 item 3) as `w*h*d*bpp` tightly
    /// packed bytes (`x + dim*(y + dim*z)` order). Copies the texture into a READBACK
    /// buffer through the 256-byte-aligned footprint, then de-pads each row.
    pub(crate) fn read_back(
        &self,
        device: &DeviceShared,
        w: u32,
        _h: u32,
        d: u32,
        bpp: u32,
    ) -> Result<Vec<u8>, EngineError> {
        unsafe {
            let res_desc = self.resource.GetDesc();
            let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut num_rows = 0u32;
            let mut row_size = 0u64;
            let mut total = 0u64;
            device.device.GetCopyableFootprints(
                &res_desc,
                0,
                1,
                0,
                Some(&mut fp),
                Some(&mut num_rows),
                Some(&mut row_size),
                Some(&mut total),
            );

            let mut readback: Option<ID3D12Resource> = None;
            device
                .device
                .CreateCommittedResource(
                    &heap(D3D12_HEAP_TYPE_READBACK),
                    D3D12_HEAP_FLAG_NONE,
                    &buffer_desc(total),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .map_err(d3d_err)?;
            let readback =
                readback.ok_or_else(|| EngineError::Rhi("readback buffer null".into()))?;

            let prior = self.state.get();
            device.immediate_submit(|list| {
                transition(
                    list,
                    &self.resource,
                    prior,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                );
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: ManuallyDrop::new(Some(std::mem::transmute_copy(&readback))),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: fp,
                    },
                };
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: ManuallyDrop::new(Some(std::mem::transmute_copy(&self.resource))),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                transition(
                    list,
                    &self.resource,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    prior,
                );
            })?;

            // Map and de-pad: tight row = w*bpp; padded row = footprint RowPitch; rows
            // = num_rows per slice × depth slices, laid contiguously.
            let mut ptr: *mut c_void = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut ptr)).map_err(d3d_err)?;
            let base = (ptr as *const u8).add(fp.Offset as usize);
            let dst_pitch = (w * bpp) as usize;
            let src_pitch = fp.Footprint.RowPitch as usize;
            let rows = num_rows as usize * d as usize;
            let mut out = vec![0u8; dst_pitch * rows];
            for row in 0..rows {
                std::ptr::copy_nonoverlapping(
                    base.add(row * src_pitch),
                    out.as_mut_ptr().add(row * dst_pitch),
                    dst_pitch,
                );
            }
            readback.Unmap(0, None);
            Ok(out)
        }
    }

    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    pub(crate) fn state(&self) -> D3D12_RESOURCE_STATES {
        self.state.get()
    }

    pub(crate) fn set_state(&self, state: D3D12_RESOURCE_STATES) {
        self.state.set(state);
    }

    /// `volumes[]` (SRV) index for trilinear sampling.
    pub fn sampled_index(&self) -> u32 {
        self.sampled_index
    }

    /// `storage_volumes[]` (UAV) index for compute writes.
    pub fn storage_index(&self) -> u32 {
        self.storage_index
    }
}
