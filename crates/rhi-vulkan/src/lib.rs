//! Vulkan backend for the engine RHI, built on `ash`.
//!
//! Implements the Phase 1 minimal slice: instance/surface/device, swapchain,
//! a graphics pipeline using dynamic rendering (Vulkan 1.3), command recording,
//! and per-frame synchronization — enough to draw a triangle. The `rhi` facade
//! wraps these concrete types in enum-dispatch variants.
//!
//! Object lifetimes are managed with `Arc`: an [`InstanceShared`] outlives the
//! [`DeviceShared`] that references it, and every GPU resource holds an
//! `Arc<DeviceShared>` so it can destroy itself in `Drop` before the device.

use std::ffi::CStr;

use ash::vk;
use engine_core::EngineError;

mod buffer;
mod command;
mod depth;
mod device;
mod instance;
mod pipeline;
mod render_target;
mod swapchain;
mod sync;
mod texture;

pub use buffer::VulkanBuffer;
pub use command::VulkanCommandBuffer;
pub use depth::VulkanDepthBuffer;
pub use device::{VulkanDevice, VulkanQueue};
pub use instance::VulkanInstance;
pub use pipeline::VulkanGraphicsPipeline;
pub use render_target::{VulkanRenderTarget, VulkanTransientHeap};
pub use swapchain::VulkanSwapchain;
pub use sync::{VulkanFence, VulkanSemaphore};
pub use texture::VulkanTexture;

/// Map a Vulkan result code into the engine error type.
fn vk_err(e: vk::Result) -> EngineError {
    EngineError::Rhi(format!("vulkan: {e:?}"))
}

/// Translate a backend-agnostic format into its Vulkan equivalent.
fn to_vk_format(format: rhi_types::Format) -> vk::Format {
    match format {
        rhi_types::Format::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        rhi_types::Format::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
        rhi_types::Format::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        rhi_types::Format::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        rhi_types::Format::Depth32Float => vk::Format::D32_SFLOAT,
    }
}

/// The full-color-image subresource range used throughout the slice.
fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Debug-utils callback: forwards validation messages to `tracing`.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*callback_data).p_message) }.to_string_lossy();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        tracing::error!(target: "vulkan", "{message}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        tracing::warn!(target: "vulkan", "{message}");
    } else {
        tracing::debug!(target: "vulkan", "{message}");
    }
    vk::FALSE
}
