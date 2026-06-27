//! Metal instance, logical device, and queues.

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::rc::Rc;

use dreamcoast_platform::Window;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAccelerationStructure, MTLBuffer, MTLCommandBuffer, MTLCommandQueue,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLOrigin, MTLPixelFormat, MTLRegion, MTLResourceID, MTLResourceOptions,
    MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter,
    MTLSamplerState, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};
use objc2_quartz_core::CAMetalLayer;
use rhi_types::{
    BackendKind, BufferDesc, ComputePipelineDesc, CubemapDesc, Extent2D, GraphicsPipelineDesc,
    InstanceDesc, MemoryRequirements, RenderTargetDesc, StorageBufferDesc, SwapchainDesc,
    TextureDesc,
};

use crate::command::MetalCommandBuffer;
use crate::resources::{
    MetalBuffer, MetalComputePipeline, MetalCubemap, MetalDepthBuffer, MetalGraphicsPipeline,
    MetalRenderTarget, MetalStorageBuffer, MetalTexture, MetalTransientHeap,
};
use crate::swapchain::MetalSwapchain;
use crate::sync::{MetalFence, MetalSemaphore};
use crate::{Result, bytes_per_pixel, pixel_format, rhi_err};

/// Size of the bindless sampled-texture table. Matches `BINDLESS_COUNT` in
/// rhi-vulkan / rhi-d3d12 and the `Bindless.textures[1024]` array in
/// `bindless.slang`; the shared sampler occupies the slot just past it.
pub(crate) const BINDLESS_COUNT: u32 = 1024;

/// Size of the bindless cubemap table. Matches `CUBE_COUNT` in rhi-vulkan /
/// rhi-d3d12 and the `Bindless.cubes[64]` array in `bindless.slang`. The cubes
/// follow the sampler in the argument buffer, so cube `i` lives at handle slot
/// `BINDLESS_COUNT + 1 + i` (Slang lays the struct out textures, samp, cubes).
pub(crate) const CUBE_COUNT: u32 = 64;

/// Size of the bindless storage-image (UAV) table. Matches `STORAGE_IMAGE_COUNT`
/// in rhi-vulkan / rhi-d3d12 and `Bindless.storage_images[64]`; storage image `i`
/// lives at handle slot `STORAGE_IMAGE_BASE + i` (M5).
pub(crate) const STORAGE_IMAGE_COUNT: u32 = 64;

/// Size of the bindless storage-buffer (UAV) table. Matches `STORAGE_BUFFER_COUNT`
/// in rhi-vulkan / rhi-d3d12 and `Bindless.storage_buffers[64]`; storage buffer `i`
/// lives at handle slot `STORAGE_BUFFER_BASE + i` (M5).
pub(crate) const STORAGE_BUFFER_COUNT: u32 = 64;

/// First argument-buffer slot of the storage-image region. Mirrors the
/// `Bindless { textures[1024], samp, cubes[64], storage_images[64],
/// storage_buffers[64] }` layout: textures `0..BINDLESS_COUNT`, sampler at
/// `BINDLESS_COUNT`, cubes next, then storage images, then storage buffers.
pub(crate) const STORAGE_IMAGE_BASE: u32 = BINDLESS_COUNT + 1 + CUBE_COUNT;

/// First argument-buffer slot of the storage-buffer region (just past the
/// storage images).
pub(crate) const STORAGE_BUFFER_BASE: u32 = STORAGE_IMAGE_BASE + STORAGE_IMAGE_COUNT;

/// Argument-buffer slot of the scene TLAS (the `Bindless.tlas` member, declared
/// last in `bindless.slang`, after the storage buffers). Present only on RT-capable
/// devices; written by [`MetalDevice::bind_tlas`] (Phase 8).
pub(crate) const TLAS_SLOT: u32 = STORAGE_BUFFER_BASE + STORAGE_BUFFER_COUNT;

/// Size of the bindless sampled-volume (3D SRV) table. Matches `VOLUME_COUNT` in
/// rhi-vulkan / rhi-d3d12 and `Bindless.volumes[64]` in `bindless.slang` (Phase 11
/// Stage B distance fields, trilinear-sampled by the SW ray marcher).
pub(crate) const VOLUME_COUNT: u32 = 64;

/// Size of the bindless storage-volume (3D UAV) table. Matches
/// `STORAGE_VOLUME_COUNT` and `Bindless.storage_volumes[64]`; the SDF bake / GDF
/// merge compute shaders write these.
pub(crate) const STORAGE_VOLUME_COUNT: u32 = 64;

/// First argument-buffer slot of the sampled-volume region. The volumes follow the
/// TLAS in declaration order, and Slang's Metal target keeps the `tlas` member in
/// the argument-buffer struct even for shaders that never trace (it does not compact
/// unused members), so `volumes` lands one slot past the TLAS — matching the MSL
/// layout `... storage_buffers[64], tlas, volumes[64], storage_volumes[64]`.
pub(crate) const VOLUME_BASE: u32 = TLAS_SLOT + 1;

/// First argument-buffer slot of the storage-volume region (just past the sampled
/// volumes).
pub(crate) const STORAGE_VOLUME_BASE: u32 = VOLUME_BASE + VOLUME_COUNT;

/// Total number of 8-byte handle slots in the bindless argument buffer: textures,
/// sampler, cubes, storage images, storage buffers, the TLAS slot, then the sampled
/// + storage volume tables (Phase 11 Stage B).
pub(crate) const ARG_BUFFER_SLOTS: u32 = STORAGE_VOLUME_BASE + STORAGE_VOLUME_COUNT;

type MetalTextureHandle = Retained<ProtocolObject<dyn MTLTexture>>;
type SampledTextureSlots = Vec<Option<MetalTextureHandle>>;
type CubeTextureSlots = Vec<Option<MetalTextureHandle>>;
type StorageImageSlots = Vec<Option<MetalTextureHandle>>;

#[derive(Clone)]
pub(crate) struct RtTlasBinding {
    pub header: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub contributions: Retained<ProtocolObject<dyn MTLBuffer>>,
}

/// Device state shared (via `Rc`) by every resource created from a device, so the
/// `MTLDevice` / command queue / layer outlive the resources that reference them.
pub(crate) struct DeviceShared {
    // Creates pipelines, buffers, textures, and samplers.
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// A second command queue for async compute (M5). Apple GPUs overlap compute
    /// with graphics across queues; cross-queue ordering uses an `MTLSharedEvent`
    /// (see [`MetalComputeQueue::submit`] / [`MetalQueue::submit_async`]).
    pub compute_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub layer: Retained<CAMetalLayer>,
    /// The bindless `ParameterBlock<Bindless>` argument buffer. Tier-2 layout: an
    /// array of 8-byte `MTLResourceID` handles — texture slots `0..BINDLESS_COUNT`,
    /// the shared sampler at slot `BINDLESS_COUNT`. Shared storage so the CPU writes
    /// handles directly (Apple Silicon argument buffers tier 2).
    pub arg_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    /// Next free bindless texture slot. `Cell`: the Metal backend is single-threaded
    /// (`Rc`, not `Arc`), so no atomics are needed.
    tex_next: Cell<u32>,
    /// Next free bindless cube slot (0-based; the handle lands at argument-buffer
    /// slot `BINDLESS_COUNT + 1 + index`).
    cube_next: Cell<u32>,
    /// Next free bindless storage-image slot (0-based; handle at
    /// `STORAGE_IMAGE_BASE + index`).
    storage_img_next: Cell<u32>,
    /// Next free bindless storage-buffer slot (0-based; handle at
    /// `STORAGE_BUFFER_BASE + index`).
    storage_buf_next: Cell<u32>,
    /// Next free bindless sampled-volume slot (0-based; handle at
    /// `VOLUME_BASE + index`). Phase 11 Stage B.
    volume_next: Cell<u32>,
    /// Next free bindless storage-volume (UAV) slot (0-based; handle at
    /// `STORAGE_VOLUME_BASE + index`). Phase 11 Stage B.
    storage_volume_next: Cell<u32>,
    /// The per-frame globals UBO (camera/lights/shadow/IBL), set once via
    /// [`MetalDevice::set_globals_buffer`]; bound at [`GLOBALS_BUFFER_INDEX`] with a
    /// per-draw byte offset for `uses_globals` pipelines.
    globals: RefCell<Option<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    /// Textures that must be made resident (`useResource`) while the bindless table
    /// is bound. Static sampled textures (`create_texture`) stay here for the app's
    /// lifetime; render targets / cubemaps / shadow maps are toggled in and out by
    /// the render graph's `*_to_sampled` / `*_to_render_target` transition hooks, so
    /// a resource is never made resident while it is an attachment in the same pass.
    resident: RefCell<Vec<Retained<ProtocolObject<dyn MTLTexture>>>>,
    /// Sampled 2D texture slots by bindless index. The M7 Metal Shader Converter
    /// RT-pipeline path writes its own descriptor table, so it needs slot-indexed
    /// texture objects in addition to the current residency list.
    sampled_textures: RefCell<SampledTextureSlots>,
    /// Cubemap slots by bindless cube index, for the converter descriptor table's
    /// cube SRV range.
    cube_textures: RefCell<CubeTextureSlots>,
    /// Storage images (UAV) currently in compute-write state: made resident with
    /// `Read | Write` on bindless compute encoders. Toggled by the render graph's
    /// `rt_to_storage` (enter UAV) / `storage_to_sampled` (back to sampled `Read`)
    /// hooks, so a storage image is never both a UAV-resident and sampled-resident.
    storage_resident: RefCell<Vec<Retained<ProtocolObject<dyn MTLTexture>>>>,
    /// Storage image slots by bindless UAV index. The M7 Metal Shader Converter
    /// root signature uses an explicit descriptor table, so the RT-pipeline path
    /// needs the texture object for each registered UAV slot instead of only the
    /// current resident set.
    storage_images: RefCell<StorageImageSlots>,
    /// Every storage buffer ever created. They are persistent (seeded on the GPU,
    /// never reallocated), so they stay permanently resident — made `useResource`
    /// (`Read | Write` on compute, `Read` on the particle/cull draw vertex stage)
    /// on every bindless encoder.
    storage_buffers: RefCell<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    /// Acceleration structures (TLAS + BLAS) bound via [`MetalDevice::bind_tlas`].
    /// They must be made resident (`useResource`) on the inline path tracer's
    /// compute encoder, since the TLAS is reached indirectly through the bindless
    /// argument buffer (Metal requires the instance AS *and* each referenced
    /// primitive AS to be resident). Permanent for the scene's lifetime. (Phase 8)
    rt_resident: RefCell<Vec<Retained<ProtocolObject<dyn MTLAccelerationStructure>>>>,
    /// Converter ABI binding for the currently bound TLAS. Unlike the M6 inline
    /// path, Metal Shader Converter descriptor tables point at a small GPU header
    /// that wraps the `MTLResourceID` plus instance-hit-group contributions.
    rt_tlas: RefCell<Option<RtTlasBinding>>,
}

impl DeviceShared {
    /// Write a resource handle into the bindless argument buffer at `slot`.
    fn write_handle(&self, slot: u32, id: MTLResourceID) {
        let n = std::mem::size_of::<MTLResourceID>();
        // Shared storage: `contents()` is a CPU pointer into the buffer's memory.
        unsafe {
            let dst = (self.arg_buffer.contents().as_ptr() as *mut u8).add(slot as usize * n);
            std::ptr::copy_nonoverlapping((&id as *const MTLResourceID).cast::<u8>(), dst, n);
        }
    }

    /// Register a texture in the bindless table, returning its slot index. When
    /// `resident`, it is also tracked for `useResource` (sampled textures); depth
    /// attachments pass `false`.
    fn register(&self, texture: Retained<ProtocolObject<dyn MTLTexture>>, resident: bool) -> u32 {
        let index = self.tex_next.get();
        self.tex_next.set(index + 1);
        self.write_handle(index, texture.gpuResourceID());
        {
            let mut textures = self.sampled_textures.borrow_mut();
            if textures.len() <= index as usize {
                textures.resize_with(index as usize + 1, || None);
            }
            textures[index as usize] = Some(texture.clone());
        }
        if resident {
            self.resident.borrow_mut().push(texture);
        }
        index
    }

    /// Register a cubemap in the bindless cube table, returning its 0-based cube
    /// index. The handle lands at argument-buffer slot `BINDLESS_COUNT + 1 + index`
    /// (textures, then the sampler, then the cubes — see `bindless.slang`). Not made
    /// resident here; the `cube_to_sampled` hook does that before it is sampled.
    fn register_cube(&self, texture: Retained<ProtocolObject<dyn MTLTexture>>) -> u32 {
        let index = self.cube_next.get();
        self.cube_next.set(index + 1);
        // The owning MetalCubemap keeps the texture alive; the argument buffer just
        // records its 8-byte handle.
        self.write_handle(BINDLESS_COUNT + 1 + index, texture.gpuResourceID());
        let mut cubes = self.cube_textures.borrow_mut();
        if cubes.len() <= index as usize {
            cubes.resize_with(index as usize + 1, || None);
        }
        cubes[index as usize] = Some(texture);
        index
    }

    /// Register a storage image (UAV) in the bindless storage-image table,
    /// returning its 0-based index (handle at `STORAGE_IMAGE_BASE + index`). Like a
    /// cube it is not made resident here; `rt_to_storage` does that before a compute
    /// pass writes it.
    fn register_storage_image(&self, texture: &Retained<ProtocolObject<dyn MTLTexture>>) -> u32 {
        let index = self.storage_img_next.get();
        self.storage_img_next.set(index + 1);
        self.write_handle(STORAGE_IMAGE_BASE + index, texture.gpuResourceID());
        let mut images = self.storage_images.borrow_mut();
        if images.len() <= index as usize {
            images.resize_with(index as usize + 1, || None);
        }
        images[index as usize] = Some(texture.clone());
        index
    }

    /// Register a storage buffer (UAV) in the bindless storage-buffer table,
    /// returning its 0-based index. Tier-2 argument buffers encode a buffer entry as
    /// its 8-byte GPU virtual address (the MSL `device T*`), not an `MTLResourceID`.
    /// The buffer is kept permanently resident.
    fn register_storage_buffer(&self, buffer: &Retained<ProtocolObject<dyn MTLBuffer>>) -> u32 {
        let index = self.storage_buf_next.get();
        self.storage_buf_next.set(index + 1);
        let slot = STORAGE_BUFFER_BASE + index;
        let n = std::mem::size_of::<u64>();
        let addr = buffer.gpuAddress();
        unsafe {
            let dst = (self.arg_buffer.contents().as_ptr() as *mut u8).add(slot as usize * n);
            std::ptr::copy_nonoverlapping((&addr as *const u64).cast::<u8>(), dst, n);
        }
        self.storage_buffers.borrow_mut().push(buffer.clone());
        index
    }

    /// Register a 3D volume texture in the bindless sampled-volume table
    /// (`volumes[]`, handle at `VOLUME_BASE + index`), returning its 0-based index.
    /// Like a cube it is not made resident here; `volume_to_sampled` does that before
    /// the SW ray marcher samples it. The owning `MetalVolume` keeps the texture
    /// alive (the argument buffer just records its 8-byte handle). Phase 11 Stage B.
    fn register_volume(&self, texture: &Retained<ProtocolObject<dyn MTLTexture>>) -> u32 {
        let index = self.volume_next.get();
        self.volume_next.set(index + 1);
        self.write_handle(VOLUME_BASE + index, texture.gpuResourceID());
        index
    }

    /// Register a 3D volume texture in the bindless storage-volume (UAV) table
    /// (`storage_volumes[]`, handle at `STORAGE_VOLUME_BASE + index`), returning its
    /// 0-based index. Made resident with `Read | Write` by `volume_to_storage` before
    /// a bake/merge compute pass writes it. Phase 11 Stage B.
    fn register_storage_volume(&self, texture: &Retained<ProtocolObject<dyn MTLTexture>>) -> u32 {
        let index = self.storage_volume_next.get();
        self.storage_volume_next.set(index + 1);
        self.write_handle(STORAGE_VOLUME_BASE + index, texture.gpuResourceID());
        index
    }

    /// Move a storage image into (`storage = true`) or out of (`false`) the
    /// compute-write resident set. Idempotent. Called by `rt_to_storage` /
    /// `storage_to_sampled`.
    pub(crate) fn set_storage_resident(
        &self,
        texture: &Retained<ProtocolObject<dyn MTLTexture>>,
        storage: bool,
    ) {
        let mut list = self.storage_resident.borrow_mut();
        let ptr = Retained::as_ptr(texture);
        let pos = list.iter().position(|t| Retained::as_ptr(t) == ptr);
        match (storage, pos) {
            (true, None) => list.push(texture.clone()),
            (false, Some(i)) => {
                list.swap_remove(i);
            }
            _ => {}
        }
    }

    /// Add or remove `texture` from the resident set (idempotent). Called by the
    /// render graph's transition hooks: `*_to_sampled` makes a target resident
    /// before a sampling pass, `*_to_render_target` drops it before it is written as
    /// an attachment (Metal forbids `useResource` on the current render target).
    pub(crate) fn set_resident(
        &self,
        texture: &Retained<ProtocolObject<dyn MTLTexture>>,
        resident: bool,
    ) {
        let mut list = self.resident.borrow_mut();
        let ptr = Retained::as_ptr(texture);
        let pos = list.iter().position(|t| Retained::as_ptr(t) == ptr);
        match (resident, pos) {
            (true, None) => list.push(texture.clone()),
            (false, Some(i)) => {
                list.swap_remove(i);
            }
            _ => {}
        }
    }

    /// The per-frame globals UBO, if one has been set.
    pub(crate) fn globals_buffer(&self) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        self.globals.borrow().clone()
    }

    /// The sampled textures to make resident before a bindless draw.
    pub(crate) fn resident_textures(
        &self,
    ) -> std::cell::Ref<'_, Vec<Retained<ProtocolObject<dyn MTLTexture>>>> {
        self.resident.borrow()
    }

    /// Registered sampled 2D textures by bindless texture index.
    pub(crate) fn sampled_textures(&self) -> std::cell::Ref<'_, SampledTextureSlots> {
        self.sampled_textures.borrow()
    }

    /// Registered cubemaps by bindless cube index.
    pub(crate) fn cube_textures(&self) -> std::cell::Ref<'_, CubeTextureSlots> {
        self.cube_textures.borrow()
    }

    /// The storage images (UAV) to make `Read | Write`-resident before a bindless
    /// compute dispatch.
    pub(crate) fn storage_resident_textures(
        &self,
    ) -> std::cell::Ref<'_, Vec<Retained<ProtocolObject<dyn MTLTexture>>>> {
        self.storage_resident.borrow()
    }

    /// Registered storage images by storage-image index (converter descriptor table).
    pub(crate) fn storage_images(&self) -> std::cell::Ref<'_, StorageImageSlots> {
        self.storage_images.borrow()
    }

    /// The storage buffers (UAV), all permanently resident.
    pub(crate) fn storage_buffers(
        &self,
    ) -> std::cell::Ref<'_, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>> {
        self.storage_buffers.borrow()
    }

    /// The acceleration structures to make resident before tracing through the
    /// bindless `g.tlas` (Phase 8 inline path).
    pub(crate) fn rt_acceleration_structures(
        &self,
    ) -> std::cell::Ref<'_, Vec<Retained<ProtocolObject<dyn MTLAccelerationStructure>>>> {
        self.rt_resident.borrow()
    }

    pub(crate) fn rt_tlas_binding(&self) -> Option<RtTlasBinding> {
        self.rt_tlas.borrow().clone()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RtAccelerationStructureHeader {
    acceleration_structure_id: u64,
    address_of_instance_contributions: u64,
    pad0: [u64; 4],
    pad1: [u32; 3],
}

pub(crate) fn resource_id_bits(id: MTLResourceID) -> u64 {
    unsafe { std::mem::transmute::<MTLResourceID, u64>(id) }
}

fn create_rt_tlas_binding(
    shared: &Rc<DeviceShared>,
    scene: &crate::MetalRaytracingScene,
) -> RtTlasBinding {
    let header = shared
        .device
        .newBufferWithLength_options(
            std::mem::size_of::<RtAccelerationStructureHeader>(),
            MTLResourceOptions::StorageModeShared,
        )
        .expect("RT pipeline TLAS header alloc failed");
    let contribution_count = scene.instance_count().max(1);
    let contributions = shared
        .device
        .newBufferWithLength_options(
            contribution_count * std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
        .expect("RT pipeline TLAS contribution alloc failed");

    let header_value = RtAccelerationStructureHeader {
        acceleration_structure_id: resource_id_bits(scene.tlas_resource_id()),
        address_of_instance_contributions: contributions.gpuAddress(),
        ..Default::default()
    };
    unsafe {
        std::ptr::copy_nonoverlapping(
            (&header_value as *const RtAccelerationStructureHeader).cast::<u8>(),
            header.contents().as_ptr() as *mut u8,
            std::mem::size_of::<RtAccelerationStructureHeader>(),
        );
        std::ptr::write_bytes(
            contributions.contents().as_ptr() as *mut u8,
            0,
            contribution_count * std::mem::size_of::<u32>(),
        );
    }

    RtTlasBinding {
        header,
        contributions,
    }
}

/// A Metal instance: owns the system `MTLDevice` and the window's `CAMetalLayer`.
pub struct MetalInstance {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    layer: Retained<CAMetalLayer>,
}

impl MetalInstance {
    /// Create an instance bound to `window`'s Metal layer.
    pub fn new(window: &Window, _desc: &InstanceDesc) -> Result<Self> {
        let device =
            MTLCreateSystemDefaultDevice().ok_or_else(|| rhi_err("no Metal-capable device"))?;
        let layer = window.metal_layer();
        layer.setDevice(Some(&device));
        Ok(Self { device, layer })
    }

    /// Create a logical device (allocates the command queue and the bindless table).
    pub fn create_device(&self) -> Result<MetalDevice> {
        let queue = self
            .device
            .newCommandQueue()
            .ok_or_else(|| rhi_err("newCommandQueue failed"))?;
        let compute_queue = self
            .device
            .newCommandQueue()
            .ok_or_else(|| rhi_err("newCommandQueue (compute) failed"))?;

        // Bindless argument buffer: one 8-byte handle per slot, laid out to match
        // the `Bindless { textures[1024], samp, cubes[64], storage_images[64],
        // storage_buffers[64] }` struct (texture/sampler/cube/storage-image entries
        // are MTLResourceIDs; storage-buffer entries are GPU addresses, also 8 bytes).
        // Shared storage = CPU-writable.
        let handle_size = std::mem::size_of::<MTLResourceID>();
        let arg_len = ARG_BUFFER_SLOTS as usize * handle_size;
        let arg_buffer = self
            .device
            .newBufferWithLength_options(arg_len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| rhi_err("bindless argument buffer alloc failed"))?;

        // One shared trilinear / repeat sampler (matches the Vulkan immutable
        // sampler; the bindless table holds a single `samp`).
        let sd = MTLSamplerDescriptor::new();
        sd.setMinFilter(MTLSamplerMinMagFilter::Linear);
        sd.setMagFilter(MTLSamplerMinMagFilter::Linear);
        sd.setMipFilter(MTLSamplerMipFilter::Linear);
        sd.setSAddressMode(MTLSamplerAddressMode::Repeat);
        sd.setTAddressMode(MTLSamplerAddressMode::Repeat);
        // Required because the sampler is encoded into the bindless argument buffer
        // via `gpuResourceID()` below. Without this, `gpuResourceID()` is invalid for
        // argument-buffer use and Metal shader validation (MTL_SHADER_VALIDATION=1)
        // faults on the sampler slot.
        sd.setSupportArgumentBuffers(true);
        let sampler = self
            .device
            .newSamplerStateWithDescriptor(&sd)
            .ok_or_else(|| rhi_err("newSamplerState failed"))?;

        let shared = Rc::new(DeviceShared {
            device: self.device.clone(),
            queue,
            compute_queue,
            layer: self.layer.clone(),
            arg_buffer,
            sampler,
            tex_next: Cell::new(0),
            cube_next: Cell::new(0),
            storage_img_next: Cell::new(0),
            storage_buf_next: Cell::new(0),
            volume_next: Cell::new(0),
            storage_volume_next: Cell::new(0),
            globals: RefCell::new(None),
            resident: RefCell::new(Vec::new()),
            sampled_textures: RefCell::new(Vec::new()),
            cube_textures: RefCell::new(Vec::new()),
            storage_resident: RefCell::new(Vec::new()),
            storage_images: RefCell::new(Vec::new()),
            storage_buffers: RefCell::new(Vec::new()),
            rt_resident: RefCell::new(Vec::new()),
            rt_tlas: RefCell::new(None),
        });
        // The sampler sits at the slot just past the texture array (Slang assigns it
        // id `BINDLESS_COUNT` in the argument-buffer struct).
        shared.write_handle(BINDLESS_COUNT, shared.sampler.gpuResourceID());

        Ok(MetalDevice { shared })
    }

    pub fn backend(&self) -> BackendKind {
        BackendKind::Metal
    }
}

/// A Metal logical device: the factory for GPU resources.
#[derive(Clone)]
pub struct MetalDevice {
    pub(crate) shared: Rc<DeviceShared>,
}

impl MetalDevice {
    pub fn create_swapchain(&self, desc: &SwapchainDesc) -> Result<MetalSwapchain> {
        MetalSwapchain::new(self.shared.clone(), desc)
    }

    pub fn queue(&self) -> MetalQueue {
        MetalQueue {
            shared: self.shared.clone(),
        }
    }

    pub fn compute_queue(&self) -> MetalComputeQueue {
        MetalComputeQueue {
            shared: self.shared.clone(),
        }
    }

    pub fn create_command_buffer(&self) -> Result<MetalCommandBuffer> {
        Ok(MetalCommandBuffer::new(
            self.shared.clone(),
            self.shared.queue.clone(),
        ))
    }

    /// Create a timestamp query heap (Phase 9 profiling — stub on Metal).
    pub fn create_query_heap(&self, count: u32) -> Result<crate::query::MetalQueryHeap> {
        crate::query::MetalQueryHeap::new(count)
    }

    /// A command buffer that records onto the dedicated compute queue (M5 async
    /// compute). Used for the particle sim that overlaps the graphics frame.
    pub fn create_compute_command_buffer(&self) -> Result<MetalCommandBuffer> {
        Ok(MetalCommandBuffer::new(
            self.shared.clone(),
            self.shared.compute_queue.clone(),
        ))
    }

    pub fn create_fence(&self, signaled: bool) -> Result<MetalFence> {
        Ok(MetalFence::new(signaled))
    }

    pub fn create_semaphore(&self) -> Result<MetalSemaphore> {
        MetalSemaphore::new(&self.shared.device)
    }

    /// Apple GPUs overlap compute on a dedicated queue with graphics; cross-queue
    /// ordering is handled by an `MTLSharedEvent` (M5).
    pub fn has_async_compute(&self) -> bool {
        true
    }

    /// Hardware ray tracing (Phase 8): true on Apple GPUs that support the
    /// `metal::raytracing` inline ray-query API (Apple7+ / Metal 3). The inline
    /// `RayQuery` path tracer traces the bindless `g.tlas`.
    pub fn has_raytracing(&self) -> bool {
        self.shared.device.supportsRaytracing()
    }

    /// Whether a DXR-style ray-tracing *pipeline* (raygen/miss/closesthit + SBT,
    /// `trace_rays`) is available via Metal Shader Converter's kernel raygen +
    /// visible-function-table ABI.
    pub fn supports_rt_pipeline(&self) -> bool {
        self.has_raytracing()
    }

    /// Build the scene's BLAS (one per geometry) + a single TLAS and return the
    /// owning [`MetalRaytracingScene`]. Bind it once with [`Self::bind_tlas`].
    pub fn build_raytracing_scene(
        &self,
        geometries: &[(&MetalBuffer, &MetalBuffer, rhi_types::BlasGeometry)],
        instances: &[rhi_types::TlasInstance],
    ) -> Result<crate::MetalRaytracingScene> {
        crate::MetalRaytracingScene::build(&self.shared, geometries, instances)
    }

    /// Register a built scene's TLAS in the bindless argument buffer (`g.tlas`) so
    /// the inline path tracer can trace it, and keep the scene's acceleration
    /// structures resident for the compute encoder.
    pub fn bind_tlas(&self, scene: &crate::MetalRaytracingScene) {
        self.shared
            .write_handle(TLAS_SLOT, scene.tlas_resource_id());
        let binding = create_rt_tlas_binding(&self.shared, scene);
        *self.shared.rt_tlas.borrow_mut() = Some(binding);
        let mut resident = self.shared.rt_resident.borrow_mut();
        for accel in scene.acceleration_structures() {
            resident.push(accel.clone());
        }
    }

    pub fn wait_idle(&self) -> Result<()> {
        // Metal has no device-wide idle; commit an empty buffer and block on it.
        if let Some(cb) = self.shared.queue.commandBuffer() {
            cb.commit();
            cb.waitUntilCompleted();
        }
        Ok(())
    }

    // ---- Implemented in later milestones (M2+) -----------------------------

    pub fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc,
    ) -> Result<MetalGraphicsPipeline> {
        crate::pipeline::build(&self.shared.device, desc)
    }

    /// Compile a compute pipeline (`MTLComputePipelineState`) from the metallib
    /// blob + the shader's threadgroup size (M5).
    pub fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDesc,
    ) -> Result<MetalComputePipeline> {
        crate::pipeline::build_compute(&self.shared.device, desc)
    }

    pub fn create_raytracing_pipeline(
        &self,
        desc: &rhi_types::RaytracingPipelineDesc,
    ) -> Result<crate::MetalRaytracingPipeline> {
        crate::rt_pipeline::MetalRaytracingPipeline::new(self.shared.clone(), desc)
    }

    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<MetalBuffer> {
        // All these buffers are host-visible (per-frame dynamic upload / readback),
        // so shared storage gives the CPU a direct pointer via `contents()`.
        let buffer = self
            .shared
            .device
            .newBufferWithLength_options(desc.size as usize, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| rhi_err("newBufferWithLength failed"))?;
        Ok(MetalBuffer::new(buffer, desc.size))
    }

    /// Create a device-local (`Private`) read-write storage buffer (UAV) and
    /// register it in the bindless storage-buffer table. Seeded on the GPU (a
    /// compute init dispatch), not from the host, so `Private` is fine; `indirect`
    /// needs no special Metal flag (any buffer can be a `drawIndexedPrimitives`
    /// indirect source). The buffer is kept resident for the device's lifetime (M5).
    pub fn create_storage_buffer(&self, desc: &StorageBufferDesc) -> Result<MetalStorageBuffer> {
        let buffer = self
            .shared
            .device
            .newBufferWithLength_options(
                desc.size.max(1) as usize,
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or_else(|| rhi_err("storage buffer alloc failed"))?;
        let index = self.shared.register_storage_buffer(&buffer);
        Ok(MetalStorageBuffer::new(buffer, index))
    }

    /// Host-seeded storage buffer (Phase 8 RT geometry + per-instance table read by
    /// the path tracer). Apple Silicon is unified-memory, so a `Shared` buffer is
    /// CPU-writable directly (no staging blit); register it in the bindless
    /// storage-buffer table like [`Self::create_storage_buffer`].
    pub fn create_storage_buffer_init(
        &self,
        desc: &StorageBufferDesc,
        data: &[u8],
    ) -> Result<MetalStorageBuffer> {
        let len = (desc.size.max(1) as usize).max(data.len());
        let buffer = self
            .shared
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| rhi_err("storage buffer (init) alloc failed"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buffer.contents().as_ptr() as *mut u8,
                data.len(),
            );
        }
        let index = self.shared.register_storage_buffer(&buffer);
        Ok(MetalStorageBuffer::new(buffer, index))
    }

    /// Store the per-frame globals UBO. `slice_size` is unused on Metal (the
    /// per-draw byte offset is passed explicitly to `set_globals`); the buffer is
    /// bound at [`crate::resources::GLOBALS_BUFFER_INDEX`] for `uses_globals`
    /// pipelines.
    pub fn set_globals_buffer(&self, buffer: &MetalBuffer, _slice_size: u64) {
        *self.shared.globals.borrow_mut() = Some(buffer.buffer.clone());
    }

    /// Create a sampled 2D texture, upload `pixels`, and register it in the bindless
    /// argument buffer. Shared storage lets the CPU fill it via `replaceRegion`
    /// directly (Apple Silicon — no staging buffer / blit needed).
    pub fn create_texture(&self, desc: &TextureDesc, pixels: &[u8]) -> Result<MetalTexture> {
        // Full mip chain so minified material textures are trilinear-filtered (the
        // shared sampler is `MipFilter::Linear`); the ray tracer selects an explicit
        // mip via ray cones. Mips are CPU-generated by the shared `generate_mip_chain`
        // (identical bytes across all backends — the cross-backend-parity rule) and
        // each level uploaded via `replaceRegion` (Apple Silicon Shared storage —
        // no staging/blit needed).
        let bpp = bytes_per_pixel(desc.format);
        let levels = rhi_types::generate_mip_chain(pixels, desc.width, desc.height, desc.format);
        let td = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                pixel_format(desc.format),
                desc.width as usize,
                desc.height as usize,
                true,
            )
        };
        td.setUsage(MTLTextureUsage::ShaderRead);
        td.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .shared
            .device
            .newTextureWithDescriptor(&td)
            .ok_or_else(|| rhi_err("newTexture failed"))?;

        for (mip, level) in levels.iter().enumerate() {
            let w = (desc.width >> mip).max(1) as usize;
            let h = (desc.height >> mip).max(1) as usize;
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width: w,
                    height: h,
                    depth: 1,
                },
            };
            let ptr = NonNull::new(level.as_ptr() as *mut c_void)
                .ok_or_else(|| rhi_err("create_texture: null pixel pointer"))?;
            unsafe {
                texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(region, mip, ptr, w * bpp);
            }
        }

        let index = self.shared.register(texture, true);
        Ok(MetalTexture::new(index))
    }

    /// Create a depth buffer (`Depth32Float`) usable as a render attachment, and
    /// reserve a bindless slot (its handle is written so the M4 shadow pass can
    /// sample it; it is not made resident here since M3 only uses it as a target).
    pub fn create_depth_buffer(&self, extent: Extent2D) -> Result<MetalDepthBuffer> {
        let td = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::Depth32Float,
                extent.width as usize,
                extent.height as usize,
                false,
            )
        };
        td.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        td.setStorageMode(MTLStorageMode::Private);
        let texture = self
            .shared
            .device
            .newTextureWithDescriptor(&td)
            .ok_or_else(|| rhi_err("depth newTexture failed"))?;
        let index = self.shared.register(texture.clone(), false);
        Ok(MetalDepthBuffer::new(texture, index))
    }

    /// Build the texture descriptor for an offscreen color target (render
    /// attachment + bindless sampled, `Private` storage). Shared by the owned,
    /// memory-query, and heap-aliased paths so their size/alignment agree.
    fn render_target_descriptor(&self, desc: &RenderTargetDesc) -> Retained<MTLTextureDescriptor> {
        let td = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                pixel_format(desc.format),
                desc.width.max(1) as usize,
                desc.height.max(1) as usize,
                false,
            )
        };
        let mut usage = MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead;
        if desc.storage {
            // Compute-writable (Phase 7): also gets a storage-image bindless slot in
            // `create_render_target` / `create_aliased_target` (M5).
            usage |= MTLTextureUsage::ShaderWrite;
        }
        td.setUsage(usage);
        td.setStorageMode(MTLStorageMode::Private);
        td
    }

    /// Create a 3D (volume) texture (Phase 11 Stage B distance fields), registered in
    /// both bindless volume tables: `storage_volumes[]` (UAV) for the SDF bake / GDF
    /// merge compute writes and `volumes[]` (SRV) for trilinear sampling by the SW ray
    /// marcher. `Private` storage (GPU-only — the bake seeds it, no host upload);
    /// `ShaderRead | ShaderWrite` for the sampled + UAV uses. One `MTLTexture`,
    /// registered in both tables (the Vulkan single-view / D3D12 SRV+UAV mirror).
    /// Residency is toggled per use by `volume_to_storage` / `volume_to_sampled`,
    /// like the 2D storage render target.
    pub fn create_volume(
        &self,
        desc: &rhi_types::VolumeDesc,
    ) -> Result<crate::resources::MetalVolume> {
        let td = MTLTextureDescriptor::new();
        td.setTextureType(MTLTextureType::Type3D);
        td.setPixelFormat(pixel_format(desc.format));
        // The dimension setters are `unsafe` in objc2-metal (the 2D path bakes them
        // into the `texture2DDescriptor…` constructor instead); a 3D texture has no
        // such convenience constructor, so set them directly.
        unsafe {
            td.setWidth(desc.width.max(1) as usize);
            td.setHeight(desc.height.max(1) as usize);
            td.setDepth(desc.depth.max(1) as usize);
            td.setMipmapLevelCount(1);
        }
        td.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        td.setStorageMode(MTLStorageMode::Private);
        let texture = self
            .shared
            .device
            .newTextureWithDescriptor(&td)
            .ok_or_else(|| rhi_err("volume newTexture failed"))?;
        let sampled_index = self.shared.register_volume(&texture);
        let storage_index = self.shared.register_storage_volume(&texture);
        Ok(crate::resources::MetalVolume::new(
            texture,
            sampled_index,
            storage_index,
        ))
    }

    /// Create a 3D volume seeded with host `data` (Phase 12 M2: a CPU-baked SDF
    /// uploaded instead of a GPU bake). `data` is `width*height*depth` voxels in
    /// `x + dim*(y + dim*z)` order. `Shared` storage lets the CPU fill it via
    /// `replaceRegion` directly (Apple Silicon unified memory — no staging/blit).
    pub fn create_volume_init(
        &self,
        desc: &rhi_types::VolumeDesc,
        data: &[u8],
    ) -> Result<crate::resources::MetalVolume> {
        let bpp = bytes_per_pixel(desc.format);
        let (w, h, d) = (
            desc.width.max(1) as usize,
            desc.height.max(1) as usize,
            desc.depth.max(1) as usize,
        );
        let td = MTLTextureDescriptor::new();
        td.setTextureType(MTLTextureType::Type3D);
        td.setPixelFormat(pixel_format(desc.format));
        unsafe {
            td.setWidth(w);
            td.setHeight(h);
            td.setDepth(d);
            td.setMipmapLevelCount(1);
        }
        td.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        td.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .shared
            .device
            .newTextureWithDescriptor(&td)
            .ok_or_else(|| rhi_err("volume newTexture failed"))?;

        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: w,
                height: h,
                depth: d,
            },
        };
        let ptr = NonNull::new(data.as_ptr() as *mut c_void)
            .ok_or_else(|| rhi_err("create_volume_init: null data pointer"))?;
        unsafe {
            texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                region,
                0,
                0,
                ptr,
                w * bpp,
                w * h * bpp,
            );
        }

        let sampled_index = self.shared.register_volume(&texture);
        let storage_index = self.shared.register_storage_volume(&texture);
        Ok(crate::resources::MetalVolume::new(
            texture,
            sampled_index,
            storage_index,
        ))
    }

    /// Create an offscreen color render target (color attachment + bindless
    /// sampled) with its own dedicated allocation.
    pub fn create_render_target(&self, desc: &RenderTargetDesc) -> Result<MetalRenderTarget> {
        let td = self.render_target_descriptor(desc);
        let texture = self
            .shared
            .device
            .newTextureWithDescriptor(&td)
            .ok_or_else(|| rhi_err("render target newTexture failed"))?;
        let index = self.shared.register(texture.clone(), false);
        // A storage (UAV) target also gets a storage-image bindless slot so compute
        // can write `g.storage_images[storage_index]` (M5).
        let storage_index = desc
            .storage
            .then(|| self.shared.register_storage_image(&texture));
        Ok(MetalRenderTarget::new(texture, index, storage_index))
    }

    /// Create a render-target cubemap (6 faces, `mip_levels` each) usable as a
    /// per-(face, mip) attachment and a bindless `TextureCube`.
    pub fn create_cubemap(&self, desc: &CubemapDesc) -> Result<MetalCubemap> {
        let size = desc.size.max(1);
        let mip_levels = desc.mip_levels.max(1);
        let td = unsafe {
            MTLTextureDescriptor::textureCubeDescriptorWithPixelFormat_size_mipmapped(
                pixel_format(desc.format),
                size as usize,
                mip_levels > 1,
            )
        };
        unsafe { td.setMipmapLevelCount(mip_levels as usize) };
        td.setTextureType(MTLTextureType::TypeCube);
        td.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        td.setStorageMode(MTLStorageMode::Private);
        let texture = self
            .shared
            .device
            .newTextureWithDescriptor(&td)
            .ok_or_else(|| rhi_err("cubemap newTexture failed"))?;
        let index = self.shared.register_cube(texture.clone());
        Ok(MetalCubemap::new(texture, index, size, mip_levels))
    }

    pub fn swapchain_readback_layout(
        &self,
        swapchain: &MetalSwapchain,
    ) -> rhi_types::ReadbackLayout {
        let extent = swapchain.extent_2d();
        rhi_types::ReadbackLayout {
            width: extent.width,
            height: extent.height,
            row_pitch: extent.width * 4,
            size: (extent.width * extent.height * 4) as u64,
        }
    }

    /// Memory footprint of an aliasable render target, for the graph's transient
    /// heap planning. Uses the same descriptor as `create_aliased_target` so the
    /// size/alignment match the placement allocation.
    pub fn render_target_memory(&self, desc: &RenderTargetDesc) -> Result<MemoryRequirements> {
        let td = self.render_target_descriptor(desc);
        let sa = self
            .shared
            .device
            .heapTextureSizeAndAlignWithDescriptor(&td);
        Ok(MemoryRequirements {
            size: sa.size as u64,
            alignment: sa.align as u64,
        })
    }

    /// Create a placement heap of `size` bytes that transient targets alias into at
    /// graph-computed offsets. `Placement` maps Vulkan's explicit-offset model 1:1;
    /// `Tracked` lets Metal insert the aliasing/RAW hazards automatically, so the
    /// graph's `aliasing_barrier` / `rt_to_*` hooks can stay no-ops.
    pub fn create_transient_heap(&self, size: u64) -> Result<MetalTransientHeap> {
        let hd = MTLHeapDescriptor::new();
        hd.setType(MTLHeapType::Placement);
        hd.setStorageMode(MTLStorageMode::Private);
        hd.setHazardTrackingMode(MTLHazardTrackingMode::Tracked);
        hd.setSize(size.max(1) as usize);
        let heap = self
            .shared
            .device
            .newHeapWithDescriptor(&hd)
            .ok_or_else(|| rhi_err("newHeapWithDescriptor failed"))?;
        Ok(MetalTransientHeap { heap })
    }

    /// Create a render target aliased into `heap` at `offset` (placement heap).
    pub fn create_aliased_target(
        &self,
        heap: &MetalTransientHeap,
        offset: u64,
        desc: &RenderTargetDesc,
    ) -> Result<MetalRenderTarget> {
        let td = self.render_target_descriptor(desc);
        let texture = unsafe {
            heap.heap
                .newTextureWithDescriptor_offset(&td, offset as usize)
        }
        .ok_or_else(|| rhi_err("heap newTextureWithDescriptor_offset failed"))?;
        let index = self.shared.register(texture.clone(), false);
        let storage_index = desc
            .storage
            .then(|| self.shared.register_storage_image(&texture));
        Ok(MetalRenderTarget::new(texture, index, storage_index))
    }
}

/// The graphics / present queue.
pub struct MetalQueue {
    shared: Rc<DeviceShared>,
}

impl MetalQueue {
    pub fn submit(
        &self,
        cmd: &MetalCommandBuffer,
        _wait: &MetalSemaphore,
        _signal: &MetalSemaphore,
        fence: &MetalFence,
    ) -> Result<()> {
        let committed = cmd.commit();
        fence.set(committed);
        Ok(())
    }

    /// Submit the graphics command buffer so it GPU-waits on the async compute
    /// queue's `compute_wait` event before running. Metal can only encode a wait
    /// into a command buffer's command stream (no queue-level wait), and the
    /// graphics buffer is already fully recorded by now — so the wait goes on a
    /// tiny *leading* command buffer committed to the graphics queue first. Command
    /// buffers in one queue execute in commit order, so the real graphics buffer
    /// does not start until the leading wait resolves (compute finished writing the
    /// particle buffer the draw's vertex stage reads). `wait` (image-available) and
    /// `signal` (render-finished) are unused, as on the single-queue path.
    pub fn submit_async(
        &self,
        cmd: &MetalCommandBuffer,
        _wait: &MetalSemaphore,
        compute_wait: &MetalSemaphore,
        _signal: &MetalSemaphore,
        fence: &MetalFence,
    ) -> Result<()> {
        if let Some(waiter) = self.shared.queue.commandBuffer() {
            waiter.encodeWaitForEvent_value(compute_wait.event(), compute_wait.current_value());
            waiter.commit();
        }
        let committed = cmd.commit();
        fence.set(committed);
        Ok(())
    }

    pub fn submit_oneshot(&self, cmd: &MetalCommandBuffer, fence: &MetalFence) -> Result<()> {
        let committed = cmd.commit();
        fence.set(committed);
        Ok(())
    }

    /// Presentation is recorded onto the frame's command buffer (via
    /// `transition_to_present` + `commit`), so this is a no-op; the swapchain
    /// never needs an out-of-band recreate signal on Metal.
    pub fn present(
        &self,
        _swapchain: &MetalSwapchain,
        _image_index: u32,
        _wait: &MetalSemaphore,
    ) -> Result<bool> {
        let _ = &self.shared;
        Ok(false)
    }
}

/// The dedicated async-compute queue (M5).
pub struct MetalComputeQueue {
    shared: Rc<DeviceShared>,
}

impl MetalComputeQueue {
    /// Submit `cmd` (recorded onto the compute queue) and signal `signal`'s shared
    /// event with a fresh monotonic value, so the graphics queue's `submit_async`
    /// can wait on it. No wait/fence here — frame pacing is handled transitively by
    /// the graphics submit's fence.
    pub fn submit(&self, cmd: &MetalCommandBuffer, signal: &MetalSemaphore) -> Result<()> {
        let _ = &self.shared;
        cmd.commit_signaling(signal.event(), signal.next_value());
        Ok(())
    }
}
