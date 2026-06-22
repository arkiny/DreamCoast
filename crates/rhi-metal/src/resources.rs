//! GPU resource types for the Metal backend.
//!
//! M2 fills in buffers ([`MetalBuffer`]) and graphics pipelines
//! ([`MetalGraphicsPipeline`]); the remaining types are placeholders whose
//! contents and behavior arrive in later milestones (textures/depth in M3, render
//! targets/cubemaps/heaps in M4, storage buffers in M5). Methods that the facade
//! forwards but that aren't implemented yet are stubbed with `unimplemented!`.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDepthStencilState, MTLHeap, MTLRenderPipelineState, MTLTexture};

/// Metal buffer-argument index that Slang assigns to the push-constant block.
/// The bindless `ParameterBlock` carries the `[[vk::binding(0, 0)]]` pin (see
/// `bindless.slang`), which keeps the push constant at `[[buffer(0)]]` even with
/// the argument buffer present.
pub(crate) const PUSH_CONSTANT_INDEX: usize = 0;

/// Buffer index of the bindless argument buffer (the `ParameterBlock<Bindless>`)
/// for pipelines **without** a globals UBO. Slang assigns it `[[buffer(1)]]`,
/// right after the push constant; verified via the Metal target reflection for
/// `mesh.slang` / `imgui.slang` / `post` / `blur` / `capture` / `irradiance` /
/// `prefilter`.
pub(crate) const BINDLESS_BUFFER_INDEX: usize = 1;

/// Buffer index of the per-frame globals UBO for `uses_globals` pipelines (the
/// deferred PBR lighting pass). Slang lays the globals `ConstantBuffer` at
/// `[[buffer(1)]]`, which pushes the bindless argument buffer to
/// [`BINDLESS_BUFFER_INDEX_WITH_GLOBALS`]; verified via the Metal target
/// reflection for `pbr.slang` (`pc`=buffer(0), `globals`=buffer(1), block=buffer(2)).
pub(crate) const GLOBALS_BUFFER_INDEX: usize = 1;

/// Buffer index of the bindless argument buffer for `uses_globals` pipelines: one
/// past the globals UBO (see [`GLOBALS_BUFFER_INDEX`]).
pub(crate) const BINDLESS_BUFFER_INDEX_WITH_GLOBALS: usize = 2;

/// Buffer index the vertex descriptor binds the (single) vertex buffer at. Placed
/// at the top of Metal's 0..=30 buffer range so it never collides with the
/// low-index argument buffers (push constants, bindless table, globals).
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

/// A sampled 2D texture registered in the bindless argument buffer. The `MTLTexture`
/// itself lives in the device's resident list (kept alive + made resident for the
/// app's lifetime in M3); this handle just carries its bindless slot. (M3)
pub struct MetalTexture {
    index: u32,
}

impl MetalTexture {
    pub(crate) fn new(index: u32) -> Self {
        Self { index }
    }

    /// Index of this texture in the bindless argument buffer.
    pub fn bindless_index(&self) -> u32 {
        self.index
    }
}

/// A depth buffer for the mesh / shadow passes. Registered in the bindless table
/// (its slot is reused for shadow-map sampling in M4). (M3)
pub struct MetalDepthBuffer {
    pub(crate) texture: Retained<ProtocolObject<dyn MTLTexture>>,
    index: u32,
}

impl MetalDepthBuffer {
    pub(crate) fn new(texture: Retained<ProtocolObject<dyn MTLTexture>>, index: u32) -> Self {
        Self { texture, index }
    }

    pub fn bindless_index(&self) -> u32 {
        self.index
    }
}

/// An offscreen color render target: an `MTLTexture` usable both as a color
/// attachment (render-graph passes write it) and a bindless sampled texture
/// (later passes read it via `g.textures[index]`). Registered in the texture
/// table; its residency for sampling is toggled by the render graph's
/// `rt_to_sampled` / `rt_to_render_target` hooks (Metal tracks the write→read
/// hazard across encoders itself). (M4)
pub struct MetalRenderTarget {
    pub(crate) texture: Retained<ProtocolObject<dyn MTLTexture>>,
    index: u32,
    /// Storage-image (UAV) index — `None` until compute lands in M5.
    storage_index: Option<u32>,
}

impl MetalRenderTarget {
    pub(crate) fn new(
        texture: Retained<ProtocolObject<dyn MTLTexture>>,
        index: u32,
        storage_index: Option<u32>,
    ) -> Self {
        Self {
            texture,
            index,
            storage_index,
        }
    }

    pub fn bindless_index(&self) -> u32 {
        self.index
    }

    pub fn storage_index(&self) -> Option<u32> {
        self.storage_index
    }
}

/// A render-target cubemap for IBL: a 6-face, optionally mipped `MTLTextureType::Cube`
/// usable both as a per-(face, mip) color attachment (the IBL generation passes
/// write it) and a bindless `TextureCube` (`g.cubes[index]`). Registered in the
/// cube table; residency toggled by `cube_to_color` / `cube_to_sampled`. (M4)
pub struct MetalCubemap {
    pub(crate) texture: Retained<ProtocolObject<dyn MTLTexture>>,
    index: u32,
    size: u32,
    mip_levels: u32,
}

impl MetalCubemap {
    pub(crate) fn new(
        texture: Retained<ProtocolObject<dyn MTLTexture>>,
        index: u32,
        size: u32,
        mip_levels: u32,
    ) -> Self {
        Self {
            texture,
            index,
            size,
            mip_levels,
        }
    }

    pub fn bindless_index(&self) -> u32 {
        self.index
    }

    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    /// Edge length of `mip` (`size >> mip`, at least 1).
    pub fn mip_size(&self, mip: u32) -> u32 {
        (self.size >> mip).max(1)
    }
}

/// A placement heap that transient render targets alias into at graph-computed
/// offsets (`MTLHeapType::Placement` maps Vulkan's offset model 1:1). (M4)
pub struct MetalTransientHeap {
    pub(crate) heap: Retained<ProtocolObject<dyn MTLHeap>>,
}

/// A compiled graphics pipeline (`MTLRenderPipelineState`). (M2 + M3 bindless/depth)
pub struct MetalGraphicsPipeline {
    pub(crate) state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    /// Whether to bind the device's bindless argument buffer + make its textures
    /// resident before draws (M3). Mirrors `is_bindless()` on the other backends.
    pub(crate) bindless: bool,
    /// Whether the pipeline binds the per-frame globals UBO (the deferred PBR
    /// lighting pass). When set, the globals buffer binds at [`GLOBALS_BUFFER_INDEX`]
    /// and the bindless block shifts to [`BINDLESS_BUFFER_INDEX_WITH_GLOBALS`].
    pub(crate) uses_globals: bool,
    /// Depth-stencil state (compare + write) when the pipeline does depth testing;
    /// bound alongside the pipeline. `None` for depth-less passes (triangle, ImGui).
    pub(crate) depth_stencil: Option<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
}

/// A compute pipeline (`MTLComputePipelineState`). (M5)
pub struct MetalComputePipeline;
