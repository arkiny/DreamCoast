//! Metal swapchain: a thin wrapper over the window's `CAMetalLayer`.
//!
//! Metal has no explicit swapchain object — the `CAMetalLayer` vends drawables.
//! We model "acquire" as `nextDrawable` (stashing the current drawable so the
//! command buffer can render to and present it) and report a single image index,
//! since only one drawable is in hand at a time.

use std::cell::RefCell;
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSSize;
use objc2_metal::MTLPixelFormat;
use objc2_quartz_core::CAMetalDrawable;
use rhi_types::{Extent2D, Format, SwapchainDesc};

use crate::device::DeviceShared;
use crate::sync::MetalSemaphore;
use crate::{Result, pixel_format};

pub struct MetalSwapchain {
    shared: Rc<DeviceShared>,
    format: Format,
    extent: Extent2D,
    /// The drawable handed out by the most recent `acquire_next_image`.
    current: RefCell<Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>>,
}

impl MetalSwapchain {
    pub(crate) fn new(shared: Rc<DeviceShared>, desc: &SwapchainDesc) -> Result<Self> {
        configure_layer(&shared, desc);
        Ok(Self {
            shared,
            format: desc.format,
            extent: desc.extent,
            current: RefCell::new(None),
        })
    }

    pub fn acquire_next_image(&self, _signal: &MetalSemaphore) -> Result<Option<u32>> {
        match self.shared.layer.nextDrawable() {
            Some(drawable) => {
                *self.current.borrow_mut() = Some(drawable);
                Ok(Some(0))
            }
            None => Ok(None),
        }
    }

    pub fn recreate(&mut self, desc: &SwapchainDesc) -> Result<()> {
        configure_layer(&self.shared, desc);
        self.format = desc.format;
        self.extent = desc.extent;
        *self.current.borrow_mut() = None;
        Ok(())
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn extent_2d(&self) -> Extent2D {
        self.extent
    }

    pub fn image_count(&self) -> u32 {
        1
    }

    /// The drawable acquired this frame (for the command buffer to render/present).
    pub(crate) fn current_drawable(&self) -> Option<Retained<ProtocolObject<dyn CAMetalDrawable>>> {
        self.current.borrow().clone()
    }
}

/// Apply the swapchain's format / size to the shared `CAMetalLayer`.
fn configure_layer(shared: &DeviceShared, desc: &SwapchainDesc) {
    let layer = &shared.layer;
    layer.setPixelFormat(pixel_format_for_swapchain(desc.format));
    // Allow blitting the drawable into a readback buffer (screenshots). Drawables
    // are framebuffer-only by default, which forbids using them as a copy source.
    layer.setFramebufferOnly(false);
    layer.setDrawableSize(NSSize::new(
        desc.extent.width as f64,
        desc.extent.height as f64,
    ));
}

/// CAMetalLayer only supports a few color pixel formats; map ours to one.
fn pixel_format_for_swapchain(format: Format) -> MTLPixelFormat {
    pixel_format(format)
}
