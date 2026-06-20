//! Direct3D 12 backend for the engine RHI, built on `windows` (windows-rs).
//!
//! Implements the same Phase 1 minimal slice as `rhi-vulkan` so the `rhi` facade
//! can dispatch to either. D3D12 COM objects are reference-counted and released
//! automatically on `Drop`, so unlike Vulkan there is no manual teardown; we
//! still share an `Arc<DeviceShared>` for parity and to keep device/queue alive.
//!
//! The facade's synchronization surface is Vulkan-shaped (two semaphores + a
//! binary fence). D3D12 has only a monotonic `ID3D12Fence` and no semaphores, so
//! [`D3d12Semaphore`] is a no-op and [`D3d12Fence`] emulates binary-fence
//! semantics with a monotonic counter + a Win32 event (see `sync.rs`).

use rhi_types::Format;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
};

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

pub use buffer::D3d12Buffer;
pub use command::D3d12CommandBuffer;
pub use depth::D3d12DepthBuffer;
pub use device::{D3d12Device, D3d12Queue};
pub use instance::D3d12Instance;
pub use pipeline::D3d12GraphicsPipeline;
pub use render_target::{D3d12RenderTarget, D3d12TransientHeap};
pub use swapchain::D3d12Swapchain;
pub use sync::{D3d12Fence, D3d12Semaphore};
pub use texture::D3d12Texture;

/// Render/RTV format (includes sRGB write conversion).
fn to_dxgi_format(format: Format) -> DXGI_FORMAT {
    match format {
        Format::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        Format::Bgra8Srgb => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        Format::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        Format::Rgba8Srgb => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        Format::Depth32Float => DXGI_FORMAT_D32_FLOAT,
    }
}

/// Swapchain buffer format. Flip-model swapchains disallow `_SRGB` formats, so
/// the buffer is created as UNORM and the sRGB-ness is applied via the RTV and
/// the pipeline's RTV format ([`to_dxgi_format`]).
fn to_dxgi_swapchain_format(format: Format) -> DXGI_FORMAT {
    match format {
        Format::Bgra8Unorm | Format::Bgra8Srgb => DXGI_FORMAT_B8G8R8A8_UNORM,
        Format::Rgba8Unorm | Format::Rgba8Srgb => DXGI_FORMAT_R8G8B8A8_UNORM,
        Format::Depth32Float => DXGI_FORMAT_D32_FLOAT,
    }
}
