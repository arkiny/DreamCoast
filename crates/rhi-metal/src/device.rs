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
    MTLOrigin, MTLPixelFormat, MTLRegion, MTLResourceID, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLSamplerState, MTLSize,
    MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
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
    /// Sampled textures that must be made resident (`useResource`) while the bindless
    /// table is bound. Depth attachments are registered for their slot but kept out
    /// of this list (they are render targets, not sampled, in M3).
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
        // handle, each an 8-byte MTLResourceID. Shared storage = CPU-writable.
        let handle_size = std::mem::size_of::<MTLResourceID>();
        let arg_len = (BINDLESS_COUNT as usize + 1) * handle_size;
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

    pub fn create_compute_pipeline(
        &self,
        _desc: &ComputePipelineDesc,
    ) -> Result<MetalComputePipeline> {
        unimplemented!("Metal compute pipelines: milestone M5")
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

    pub fn create_storage_buffer(&self, _desc: &StorageBufferDesc) -> Result<MetalStorageBuffer> {
        unimplemented!("Metal storage buffers: milestone M5")
    }

    pub fn set_globals_buffer(&self, _buffer: &MetalBuffer, _slice_size: u64) {
        // Implemented with the PBR globals path in M4.
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

    pub fn create_render_target(&self, _desc: &RenderTargetDesc) -> Result<MetalRenderTarget> {
        unimplemented!("Metal render targets: milestone M4")
    }

    pub fn create_cubemap(&self, _desc: &CubemapDesc) -> Result<MetalCubemap> {
        unimplemented!("Metal cubemaps: milestone M4")
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

    pub fn render_target_memory(&self, _desc: &RenderTargetDesc) -> Result<MemoryRequirements> {
        unimplemented!("Metal transient aliasing: milestone M4")
    }

    pub fn create_transient_heap(&self, _size: u64) -> Result<MetalTransientHeap> {
        unimplemented!("Metal transient heap: milestone M4")
    }

    pub fn create_aliased_target(
        &self,
        _heap: &MetalTransientHeap,
        _offset: u64,
        _desc: &RenderTargetDesc,
    ) -> Result<MetalRenderTarget> {
        unimplemented!("Metal transient aliasing: milestone M4")
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
