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
mod query;
mod resources;
mod rt_pipeline;
mod swapchain;
mod sync;

pub use accel::MetalRaytracingScene;
pub use command::MetalCommandBuffer;
pub use device::{MetalComputeQueue, MetalDevice, MetalInstance, MetalQueue};
pub use query::MetalQueryHeap;
pub use resources::{
    MetalBuffer, MetalComputePipeline, MetalCubemap, MetalDepthBuffer, MetalGraphicsPipeline,
    MetalRenderTarget, MetalStorageBuffer, MetalTexture, MetalTransientHeap, MetalVolume,
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
        // Single-channel 32-bit float: signed distance for the Phase 11 volume
        // (`Texture3D<float>` / `RWTexture3D<float>` in `bindless.slang`).
        Format::R32Float => MTLPixelFormat::R32Float,
        // BCn block compression (Phase 12 M3). Supported on Apple Silicon GPUs.
        Format::Bc1Srgb => MTLPixelFormat::BC1_RGBA_sRGB,
        Format::Bc1Unorm => MTLPixelFormat::BC1_RGBA,
        Format::Bc5Unorm => MTLPixelFormat::BC5_RGUnorm,
        Format::Bc3Srgb => MTLPixelFormat::BC3_RGBA_sRGB,
        Format::Bc3Unorm => MTLPixelFormat::BC3_RGBA,
        Format::Bc4Unorm => MTLPixelFormat::BC4_RUnorm,
        Format::Bc7Srgb => MTLPixelFormat::BC7_RGBAUnorm_sRGB,
        Format::Bc7Unorm => MTLPixelFormat::BC7_RGBAUnorm,
    }
}

/// Bytes per pixel for a [`Format`], for computing a texture upload's row pitch.
/// Block-compressed formats have no per-pixel size; callers use the block-aware
/// upload path instead (`Format::upload_pitch`).
pub(crate) fn bytes_per_pixel(format: Format) -> usize {
    match format {
        Format::Bgra8Unorm
        | Format::Bgra8Srgb
        | Format::Rgba8Unorm
        | Format::Rgba8Srgb
        | Format::Rg16Float
        | Format::Depth32Float
        | Format::R32Float => 4,
        Format::Rgba16Float => 8,
        Format::Bc1Srgb
        | Format::Bc1Unorm
        | Format::Bc5Unorm
        | Format::Bc3Srgb
        | Format::Bc3Unorm
        | Format::Bc4Unorm
        | Format::Bc7Srgb
        | Format::Bc7Unorm => {
            unreachable!("block-compressed formats use the block upload path")
        }
    }
}
