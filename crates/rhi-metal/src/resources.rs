//! GPU resource types for the Metal backend.
//!
//! M2 fills in buffers ([`MetalBuffer`]) and graphics pipelines
//! ([`MetalGraphicsPipeline`]); the remaining types are placeholders whose
//! contents and behavior arrive in later milestones (textures/depth in M3, render
//! targets/cubemaps/heaps in M4, storage buffers in M5). Methods that the facade
//! forwards but that aren't implemented yet are stubbed with `unimplemented!`.

use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLComputePipelineState, MTLDepthStencilState, MTLHeap, MTLRenderPipelineState,
    MTLSize, MTLTexture,
};

use crate::device::DeviceShared;

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

/// A device-local (`Private`) storage buffer (UAV) for compute. Its 8-byte GPU
/// address is written into the bindless argument buffer's storage-buffer region
/// (`g.storage_buffers[storage_index]`); compute writes it and the particle / cull
/// draw passes read it in their vertex stage. Kept permanently resident on the
/// device (made `useResource` on every bindless compute/graphics encoder). (M5)
pub struct MetalStorageBuffer {
    /// The device, so `drop` can return the bindless slot to the free-list.
    shared: Rc<DeviceShared>,
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    index: u32,
}

impl MetalStorageBuffer {
    pub(crate) fn new(
        shared: Rc<DeviceShared>,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        index: u32,
    ) -> Self {
        Self {
            shared,
            buffer,
            index,
        }
    }

    /// Index of this buffer in the bindless storage-buffer table.
    pub fn storage_index(&self) -> u32 {
        self.index
    }

    /// Host-write the buffer's contents (used for the per-frame skin joint palette).
    /// Only valid for a host-visible (`StorageModeShared`) buffer — i.e. one created
    /// via `create_storage_buffer_init`; a `Private` buffer's `contents()` is not
    /// CPU-addressable. Bounds-checked against the allocated length.
    pub fn write(&self, data: &[u8]) -> crate::Result<()> {
        if data.len() > self.buffer.length() {
            return Err(crate::rhi_err("storage buffer write out of bounds"));
        }
        let dst = self.buffer.contents().as_ptr() as *mut u8;
        if dst.is_null() {
            return Err(crate::rhi_err("storage buffer is not host-visible"));
        }
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };
        Ok(())
    }

    /// Host-read the buffer's contents into `dst` (HZB cull stats readback). Only valid
    /// for a host-visible (`StorageModeShared`) buffer; the caller must have synced the
    /// GPU writes first. Bounds-checked against the allocated length.
    pub fn read_into(&self, dst: &mut [u8]) -> crate::Result<()> {
        if dst.len() > self.buffer.length() {
            return Err(crate::rhi_err("storage buffer read out of bounds"));
        }
        let src = self.buffer.contents().as_ptr() as *const u8;
        if src.is_null() {
            return Err(crate::rhi_err("storage buffer is not host-visible"));
        }
        unsafe { std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }
}

impl Drop for MetalStorageBuffer {
    fn drop(&mut self) {
        // Return the bindless slot + drop the device's strong ref to the buffer. Safe: the handoff
        // contract defers this Drop until the referencing frames retire.
        self.shared.free_storage_buffer(self.index);
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

    /// Debug name (Phase 9 M2) — no-op on the Metal stub.
    pub fn set_name(&self, _name: &str) {}
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

    /// Debug name (Phase 9 M2) — no-op on the Metal stub.
    pub fn set_name(&self, _name: &str) {}
}

/// A 3D (volume) texture for Phase 11 Stage B distance fields. A single `Private`
/// 3D `MTLTexture` registered in both bindless volume tables — `storage_volumes[]`
/// (UAV) for the SDF bake / GDF merge compute writes and `volumes[]` (SRV) for
/// trilinear sampling by the SW ray marcher (the Vulkan single-view / D3D12 SRV+UAV
/// mirror). Residency is toggled per use by `volume_to_storage` / `volume_to_sampled`
/// (see `command.rs`), so it is never both UAV-resident and sampled-resident.
pub struct MetalVolume {
    pub(crate) texture: Retained<ProtocolObject<dyn MTLTexture>>,
    sampled_index: u32,
    storage_index: u32,
}

impl MetalVolume {
    pub(crate) fn new(
        texture: Retained<ProtocolObject<dyn MTLTexture>>,
        sampled_index: u32,
        storage_index: u32,
    ) -> Self {
        Self {
            texture,
            sampled_index,
            storage_index,
        }
    }

    /// `volumes[]` (sampled) bindless index.
    pub fn sampled_index(&self) -> u32 {
        self.sampled_index
    }

    /// `storage_volumes[]` (UAV) bindless index.
    pub fn storage_index(&self) -> u32 {
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

    /// Debug name (Phase 9 M2) — no-op on the Metal stub.
    pub fn set_name(&self, _name: &str) {}
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

/// A compiled mesh-shader pipeline (`MTLRenderPipelineState` built from object/mesh/fragment
/// functions) plus the per-stage threadgroup sizes MSL needs at draw time (Phase 14). Bound
/// with `bind_mesh_pipeline`, drawn with `draw_mesh_tasks`.
pub struct MetalMeshPipeline {
    pub(crate) state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    /// Object (task) stage threadgroup size; `(1,1,1)` for a mesh-only pipeline.
    pub(crate) object_threads: MTLSize,
    /// Mesh stage threadgroup size (the shader's `[numthreads]`).
    pub(crate) mesh_threads: MTLSize,
    /// Whether to bind the bindless argument buffer to the object/mesh stages.
    pub(crate) bindless: bool,
    /// Whether the pipeline binds the per-frame globals UBO.
    pub(crate) uses_globals: bool,
    /// Depth-stencil state (compare + write) when depth testing; bound with the pipeline.
    pub(crate) depth_stencil: Option<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
}

/// A compiled compute pipeline (`MTLComputePipelineState`) plus the threadgroup
/// size from the shader's `[numthreads]` (MSL kernels don't bake it in, unlike
/// SPIR-V/DXIL), and whether it binds the device's bindless argument buffer. (M5)
pub struct MetalComputePipeline {
    pub(crate) state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Threads per threadgroup; combined with the `dispatch(x, y, z)` threadgroup
    /// counts to call `dispatchThreadgroups:threadsPerThreadgroup:`.
    pub(crate) threads_per_group: MTLSize,
    /// Bind the bindless argument buffer + make storage resources resident.
    pub(crate) bindless: bool,
    /// Bind the per-frame globals UBO (Stage C7 SSR reprojection reads
    /// `globals.prev_view_proj`). When set, the globals buffer binds at
    /// [`GLOBALS_BUFFER_INDEX`] and the bindless block shifts to
    /// [`BINDLESS_BUFFER_INDEX_WITH_GLOBALS`], mirroring the graphics path.
    pub(crate) uses_globals: bool,
}
