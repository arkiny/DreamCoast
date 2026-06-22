//! GPU resource types for the Metal backend.
//!
//! These are placeholders for M0 (the empty-window / clear milestone): the types
//! exist so the `rhi` facade's `Metal` variant type-checks, but their contents
//! and behavior are filled in by later milestones (buffers/pipelines in M2,
//! textures/depth in M3, render targets/cubemaps/heaps in M4, storage buffers in
//! M5). Methods that the facade forwards are stubbed with `unimplemented!`.

/// A host-visible buffer (vertex / index / uniform / readback). (M2)
pub struct MetalBuffer;

impl MetalBuffer {
    pub fn write(&self, _data: &[u8]) -> crate::Result<()> {
        unimplemented!("Metal buffers: milestone M2")
    }

    pub fn write_at(&self, _offset: u64, _data: &[u8]) -> crate::Result<()> {
        unimplemented!("Metal buffers: milestone M2")
    }

    pub fn read_into(&self, _dst: &mut [u8]) -> crate::Result<()> {
        unimplemented!("Metal readback: milestone M6")
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

/// A graphics pipeline (`MTLRenderPipelineState`). (M2)
pub struct MetalGraphicsPipeline;

/// A compute pipeline (`MTLComputePipelineState`). (M5)
pub struct MetalComputePipeline;
