//! Native Metal backend (macOS) for the DreamCoast RHI.
//!
//! Mirrors the structure of `rhi-vulkan` / `rhi-d3d12`: a set of concrete types
//! (`MetalDevice`, `MetalSwapchain`, `MetalCommandBuffer`, …) that the `rhi`
//! enum-dispatch facade wraps in its `Metal` variant. The whole crate is gated to
//! macOS; on other targets it is empty (the facade only depends on it under
//! `cfg(target_os = "macos")`).
//!
//! Built incrementally per the Metal milestones (M0 = device + swapchain clear).
#![cfg(target_os = "macos")]

mod accel;
mod command;
mod device;
mod pipeline;
mod resources;
mod rt_pipeline;
mod swapchain;
mod sync;

pub use accel::MetalRaytracingScene;
pub use command::MetalCommandBuffer;
pub use device::{MetalComputeQueue, MetalDevice, MetalInstance, MetalQueue};
pub use resources::{
    MetalBuffer, MetalComputePipeline, MetalCubemap, MetalDepthBuffer, MetalGraphicsPipeline,
    MetalRenderTarget, MetalStorageBuffer, MetalTexture, MetalTransientHeap,
};
pub use rt_pipeline::MetalRaytracingPipeline;
pub use swapchain::MetalSwapchain;
pub use sync::{MetalFence, MetalSemaphore};

use dreamcoast_core::EngineError;
use objc2_metal::MTLPixelFormat;
use rhi_types::Format;

/// Shorthand for results across the backend.
pub(crate) type Result<T> = std::result::Result<T, EngineError>;

/// Build an `EngineError::Rhi` from a message.
pub(crate) fn rhi_err(msg: impl Into<String>) -> EngineError {
    EngineError::Rhi(msg.into())
}

/// Map an engine [`Format`] to its Metal pixel format.
pub(crate) fn pixel_format(format: Format) -> MTLPixelFormat {
    match format {
        Format::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        Format::Bgra8Srgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        Format::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        Format::Rgba8Srgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        Format::Rgba16Float => MTLPixelFormat::RGBA16Float,
        Format::Rg16Float => MTLPixelFormat::RG16Float,
        Format::Depth32Float => MTLPixelFormat::Depth32Float,
    }
}

/// Bytes per pixel for a [`Format`], for computing a texture upload's row pitch.
pub(crate) fn bytes_per_pixel(format: Format) -> usize {
    match format {
        Format::Bgra8Unorm
        | Format::Bgra8Srgb
        | Format::Rgba8Unorm
        | Format::Rgba8Srgb
        | Format::Rg16Float
        | Format::Depth32Float => 4,
        Format::Rgba16Float => 8,
    }
}
