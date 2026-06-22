//! Metal instance, logical device, and queues.

use std::rc::Rc;

use dreamcoast_platform::Window;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};
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
use crate::{Result, rhi_err};

/// Device state shared (via `Rc`) by every resource created from a device, so the
/// `MTLDevice` / command queue / layer outlive the resources that reference them.
pub(crate) struct DeviceShared {
    // Used from M2 onward to create pipelines, buffers, and textures.
    #[allow(dead_code)]
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub layer: Retained<CAMetalLayer>,
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

    /// Create a logical device (allocates the command queue).
    pub fn create_device(&self) -> Result<MetalDevice> {
        let queue = self
            .device
            .newCommandQueue()
            .ok_or_else(|| rhi_err("newCommandQueue failed"))?;
        Ok(MetalDevice {
            shared: Rc::new(DeviceShared {
                device: self.device.clone(),
                queue,
                layer: self.layer.clone(),
            }),
        })
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
        _desc: &GraphicsPipelineDesc,
    ) -> Result<MetalGraphicsPipeline> {
        unimplemented!("Metal graphics pipelines: milestone M2")
    }

    pub fn create_compute_pipeline(
        &self,
        _desc: &ComputePipelineDesc,
    ) -> Result<MetalComputePipeline> {
        unimplemented!("Metal compute pipelines: milestone M5")
    }

    pub fn create_buffer(&self, _desc: &BufferDesc) -> Result<MetalBuffer> {
        unimplemented!("Metal buffers: milestone M2")
    }

    pub fn create_storage_buffer(&self, _desc: &StorageBufferDesc) -> Result<MetalStorageBuffer> {
        unimplemented!("Metal storage buffers: milestone M5")
    }

    pub fn set_globals_buffer(&self, _buffer: &MetalBuffer, _slice_size: u64) {
        // Implemented with the PBR globals path in M4.
    }

    pub fn create_texture(&self, _desc: &TextureDesc, _pixels: &[u8]) -> Result<MetalTexture> {
        unimplemented!("Metal textures: milestone M3")
    }

    pub fn create_depth_buffer(&self, _extent: Extent2D) -> Result<MetalDepthBuffer> {
        unimplemented!("Metal depth buffers: milestone M3")
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
