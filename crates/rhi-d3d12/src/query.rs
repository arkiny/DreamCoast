//! GPU timestamp queries for per-pass profiling (Phase 9 M1).
//!
//! A [`D3d12QueryHeap`] wraps an `ID3D12QueryHeap` of timestamp queries plus a
//! READBACK buffer the resolved ticks land in. The render graph writes a
//! timestamp at each pass boundary (`EndQuery`), resolves them into the readback
//! buffer (`ResolveQueryData`), and the host reads the raw ticks back
//! ([`Self::read`]) once the frame's fence has signalled, converting to
//! nanoseconds via [`Self::period_ns`] (`1e9 / GetTimestampFrequency`).
//! Unlike Vulkan there is no reset step.

use std::ffi::c_void;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_TYPE_READBACK, D3D12_MEMORY_POOL_UNKNOWN, D3D12_QUERY_HEAP_DESC,
    D3D12_QUERY_HEAP_TYPE_TIMESTAMP, D3D12_RANGE, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12QueryHeap, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

use crate::device::DeviceShared;
use crate::instance::d3d_err;

/// A timestamp query heap of `count` queries + its readback buffer.
pub struct D3d12QueryHeap {
    #[allow(dead_code)]
    device: Rc<DeviceShared>,
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    count: u32,
    period_ns: f32,
}

impl D3d12QueryHeap {
    pub(crate) fn new(device: Rc<DeviceShared>, count: u32) -> Result<Self, EngineError> {
        unsafe {
            let desc = D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                Count: count,
                NodeMask: 0,
            };
            let mut heap: Option<ID3D12QueryHeap> = None;
            device
                .device
                .CreateQueryHeap(&desc, &mut heap)
                .map_err(d3d_err)?;
            let heap = heap.ok_or_else(|| EngineError::Rhi("query heap was null".into()))?;

            // READBACK buffer the resolved 64-bit ticks copy into (one u64/query).
            let readback = create_readback(&device, count as u64 * 8)?;

            // Ticks-per-second on the graphics queue → nanoseconds per tick.
            let freq = device.queue.GetTimestampFrequency().map_err(d3d_err)?;
            let period_ns = if freq == 0 { 0.0 } else { 1.0e9 / freq as f32 };

            Ok(Self {
                device,
                heap,
                readback,
                count,
                period_ns,
            })
        }
    }

    pub(crate) fn heap(&self) -> &ID3D12QueryHeap {
        &self.heap
    }

    pub(crate) fn readback(&self) -> &ID3D12Resource {
        &self.readback
    }

    /// Number of timestamp slots.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Nanoseconds per timestamp tick (multiply tick deltas by this).
    pub fn period_ns(&self) -> f32 {
        self.period_ns
    }

    /// Read all `count` raw timestamp ticks from the readback buffer. Call only
    /// after the submission that resolved them has completed (e.g. after the
    /// frame fence).
    pub fn read(&self) -> Vec<u64> {
        let mut data = vec![0u64; self.count as usize];
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let range = D3D12_RANGE {
                Begin: 0,
                End: self.count as usize * 8,
            };
            if self.readback.Map(0, Some(&range), Some(&mut ptr)).is_ok() {
                std::ptr::copy_nonoverlapping(
                    ptr as *const u64,
                    data.as_mut_ptr(),
                    self.count as usize,
                );
                self.readback
                    .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            }
        }
        data
    }
}

/// Create a READBACK-heap buffer of `size` bytes (CPU reads GPU-copied data).
fn create_readback(device: &DeviceShared, size: u64) -> Result<ID3D12Resource, EngineError> {
    unsafe {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let res_desc = D3D12_RESOURCE_DESC {
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
        };
        let mut res: Option<ID3D12Resource> = None;
        device
            .device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &res_desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut res,
            )
            .map_err(d3d_err)?;
        res.ok_or_else(|| EngineError::Rhi("readback buffer was null".into()))
    }
}
