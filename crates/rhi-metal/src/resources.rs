//! GPU resource types for the Metal backend.
//!
//! M2 fills in buffers ([`MetalBuffer`]) and graphics pipelines
//! ([`MetalGraphicsPipeline`]); the remaining types are placeholders whose
//! contents and behavior arrive in later milestones (textures/depth in M3, render
//! targets/cubemaps/heaps in M4, storage buffers in M5). Methods that the facade
//! forwards but that aren't implemented yet are stubbed with `unimplemented!`.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLRenderPipelineState};

/// Metal buffer-argument index that Slang assigns to the push-constant block
/// (`[[buffer(0)]]`) when no globals/bindless arguments precede it. The globals
/// (M4) and bindless (M3) paths shift this and will revisit the convention.
pub(crate) const PUSH_CONSTANT_INDEX: usize = 0;

/// Buffer index the vertex descriptor binds the (single) vertex buffer at. Placed
/// at the top of Metal's 0..=30 buffer range so it never collides with the
/// low-index argument buffers (push constants, globals).
pub(crate) const VERTEX_BUFFER_INDEX: usize = 30;

/// A host-visible buffer (vertex / index / uniform / readback). Backed by a
/// shared-storage `MTLBuffer` so the CPU can write it directly each frame. (M2)
pub struct MetalBuffer {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    size: u64,
}

impl MetalBuffer {
    pub(crate) fn new(buffer: Retained<ProtocolObject<dyn MTLBuffer>>, size: u64) -> Self {
        Self { buffer, size }
    }

    pub fn write(&self, data: &[u8]) -> crate::Result<()> {
        self.write_at(0, data)
    }

    pub fn write_at(&self, offset: u64, data: &[u8]) -> crate::Result<()> {
        if offset + data.len() as u64 > self.size {
            return Err(crate::rhi_err("buffer write_at out of bounds"));
        }
        // Shared storage: `contents()` is a CPU-visible pointer to the buffer's
        // memory; copy into it (no flush needed for shared mode).
        let dst = self.buffer.contents().as_ptr() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst.add(offset as usize), data.len());
        }
        Ok(())
    }

    /// Copy out of the buffer into `dst` (clamped to its size). Shared storage, so
    /// the CPU sees GPU writes once the command buffer that wrote it has completed.
    pub fn read_into(&self, dst: &mut [u8]) -> crate::Result<()> {
        let n = dst.len().min(self.size as usize);
        let src = self.buffer.contents().as_ptr() as *const u8;
        unsafe { std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), n) };
        Ok(())
    }
}

/// A device-local storage buffer (UAV) for compute. (M5)
pub struct MetalStorageBuffer;

impl MetalStorageBuffer {
    pub fn storage_index(&self) -> u32 {
        unimplemented!("Metal storage buffers: milestone M5")
    }
}

/// A sampled 2D texture registered in the bindless table. (M3)
pub struct MetalTexture;

impl MetalTexture {
    pub fn bindless_index(&self) -> u32 {
        unimplemented!("Metal textures: milestone M3")
    }
}

/// A depth buffer for the mesh / shadow passes. (M3)
pub struct MetalDepthBuffer;

impl MetalDepthBuffer {
    pub fn bindless_index(&self) -> u32 {
        unimplemented!("Metal depth buffers: milestone M3")
    }
}

/// An offscreen color render target. (M4)
pub struct MetalRenderTarget;

impl MetalRenderTarget {
    pub fn bindless_index(&self) -> u32 {
        unimplemented!("Metal render targets: milestone M4")
    }

    pub fn storage_index(&self) -> Option<u32> {
        unimplemented!("Metal render targets: milestone M4")
    }
}

/// A render-target cubemap for IBL. (M4)
pub struct MetalCubemap;

impl MetalCubemap {
    pub fn bindless_index(&self) -> u32 {
        unimplemented!("Metal cubemaps: milestone M4")
    }

    pub fn mip_levels(&self) -> u32 {
        unimplemented!("Metal cubemaps: milestone M4")
    }

    pub fn mip_size(&self, _mip: u32) -> u32 {
        unimplemented!("Metal cubemaps: milestone M4")
    }
}

/// A heap that transient render targets alias into. (M4)
pub struct MetalTransientHeap;

/// A compiled graphics pipeline (`MTLRenderPipelineState`). (M2)
pub struct MetalGraphicsPipeline {
    pub(crate) state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

/// A compute pipeline (`MTLComputePipelineState`). (M5)
pub struct MetalComputePipeline;
