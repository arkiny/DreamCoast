//! Metal instance, logical device, and queues.

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::rc::Rc;

use dreamcoast_platform::Window;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor, MTLHeapType, MTLOrigin, MTLPixelFormat,
    MTLRegion, MTLResourceID, MTLResourceOptions, MTLSamplerAddressMode, MTLSamplerDescriptor,
    MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLSamplerState, MTLSize, MTLStorageMode,
    MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
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

/// Device state shared (via `Rc`) by every resource created from a device, so the
/// `MTLDevice` / command queue / layer outlive the resources that reference them.
pub(crate) struct DeviceShared {
    // Creates pipelines, buffers, textures, and samplers.
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
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
        index
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

        // Bindless argument buffer: BINDLESS_COUNT texture handles + one sampler
        // handle + CUBE_COUNT cube handles, each an 8-byte MTLResourceID, in that
        // order (matches the `Bindless { textures, samp, cubes }` struct layout).
        // Shared storage = CPU-writable.
        let handle_size = std::mem::size_of::<MTLResourceID>();
        let arg_len = (BINDLESS_COUNT as usize + 1 + CUBE_COUNT as usize) * handle_size;
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
        let sampler = self
            .device
            .newSamplerStateWithDescriptor(&sd)
            .ok_or_else(|| rhi_err("newSamplerState failed"))?;

        let shared = Rc::new(DeviceShared {
            device: self.device.clone(),
            queue,
            layer: self.layer.clone(),
            arg_buffer,
            sampler,
            tex_next: Cell::new(0),
            cube_next: Cell::new(0),
            globals: RefCell::new(None),
            resident: RefCell::new(Vec::new()),
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
        Ok(MetalCommandBuffer::new(self.shared.clone()))
    }

    pub fn create_compute_command_buffer(&self) -> Result<MetalCommandBuffer> {
        Ok(MetalCommandBuffer::new(self.shared.clone()))
    }

    pub fn create_fence(&self, signaled: bool) -> Result<MetalFence> {
        Ok(MetalFence::new(signaled))
    }

    pub fn create_semaphore(&self) -> Result<MetalSemaphore> {
        Ok(MetalSemaphore::new())
    }

    /// Metal supports multiple command queues; async compute lands in M5.
    pub fn has_async_compute(&self) -> bool {
        false
    }

    /// Metal ray tracing (Phase 8) is out of scope for the M0–M5 parity effort.
    pub fn has_raytracing(&self) -> bool {
        false
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

    /// M4: returns an inert placeholder so the sandbox's *unconditional* Phase-7
    /// setup links and the deferred (non-compute) scene runs. Actual compute
    /// **execution** is M5 — `bind_compute_pipeline` / `dispatch` still
    /// `unimplemented!`, and the sandbox gates every dispatch off on Metal.
    pub fn create_compute_pipeline(
        &self,
        _desc: &ComputePipelineDesc,
    ) -> Result<MetalComputePipeline> {
        Ok(MetalComputePipeline)
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

    /// M4: inert placeholder (see [`Self::create_compute_pipeline`]) — the deferred
    /// scene creates the Phase-7 storage buffers unconditionally but never reads or
    /// writes them on Metal. Real storage buffers (UAV + bindless slot) land in M5.
    pub fn create_storage_buffer(&self, _desc: &StorageBufferDesc) -> Result<MetalStorageBuffer> {
        Ok(MetalStorageBuffer)
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
        let td = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                pixel_format(desc.format),
                desc.width as usize,
                desc.height as usize,
                false,
            )
        };
        td.setUsage(MTLTextureUsage::ShaderRead);
        td.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .shared
            .device
            .newTextureWithDescriptor(&td)
            .ok_or_else(|| rhi_err("newTexture failed"))?;

        let bytes_per_row = desc.width as usize * bytes_per_pixel(desc.format);
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: desc.width as usize,
                height: desc.height as usize,
                depth: 1,
            },
        };
        let ptr = NonNull::new(pixels.as_ptr() as *mut c_void)
            .ok_or_else(|| rhi_err("create_texture: null pixel pointer"))?;
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(region, 0, ptr, bytes_per_row);
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
            // Compute-writable (Phase 7); the storage bindless slot lands in M5.
            usage |= MTLTextureUsage::ShaderWrite;
        }
        td.setUsage(usage);
        td.setStorageMode(MTLStorageMode::Private);
        td
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
        Ok(MetalRenderTarget::new(texture, index, None))
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
        let sa = self.shared.device.heapTextureSizeAndAlignWithDescriptor(&td);
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
        let texture = unsafe { heap.heap.newTextureWithDescriptor_offset(&td, offset as usize) }
            .ok_or_else(|| rhi_err("heap newTextureWithDescriptor_offset failed"))?;
        let index = self.shared.register(texture.clone(), false);
        Ok(MetalRenderTarget::new(texture, index, None))
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

    pub fn submit_async(
        &self,
        cmd: &MetalCommandBuffer,
        _wait: &MetalSemaphore,
        _compute_wait: &MetalSemaphore,
        _signal: &MetalSemaphore,
        fence: &MetalFence,
    ) -> Result<()> {
        // Real cross-queue overlap arrives in M5; for now run it inline.
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

/// The async-compute queue (M5). For now it shares the single queue.
pub struct MetalComputeQueue {
    shared: Rc<DeviceShared>,
}

impl MetalComputeQueue {
    pub fn submit(&self, cmd: &MetalCommandBuffer, _signal: &MetalSemaphore) -> Result<()> {
        let _ = &self.shared;
        cmd.commit();
        Ok(())
    }
}
