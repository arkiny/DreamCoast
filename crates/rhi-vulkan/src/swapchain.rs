//! Swapchain creation, image acquisition, and recreation on resize.

use std::sync::Arc;

use ash::vk;
use dreamcoast_core::EngineError;
use rhi_types::{Extent2D, PresentMode, SwapchainDesc};

use crate::device::DeviceShared;
use crate::sync::VulkanSemaphore;
use crate::{color_subresource_range, to_vk_format, vk_err};

/// A window swapchain plus its color image views.
pub struct VulkanSwapchain {
    device: Arc<DeviceShared>,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    format: vk::Format,
    extent: vk::Extent2D,
}

impl VulkanSwapchain {
    pub(crate) fn new(
        device: Arc<DeviceShared>,
        desc: &SwapchainDesc,
    ) -> Result<Self, EngineError> {
        let (swapchain, images, views, format, extent) =
            build(&device, desc, vk::SwapchainKHR::null())?;
        Ok(Self {
            device,
            swapchain,
            images,
            views,
            format,
            extent,
        })
    }

    /// Rebuild against a new size (e.g. after a window resize), reusing the old
    /// swapchain as a hint and destroying it afterward.
    pub fn recreate(&mut self, desc: &SwapchainDesc) -> Result<(), EngineError> {
        unsafe { self.device.device.device_wait_idle().map_err(vk_err)? };
        let old = self.swapchain;
        let (swapchain, images, views, format, extent) = build(&self.device, desc, old)?;
        self.destroy_resources();
        unsafe {
            self.device.swapchain_loader.destroy_swapchain(old, None);
        }
        self.swapchain = swapchain;
        self.images = images;
        self.views = views;
        self.format = format;
        self.extent = extent;
        Ok(())
    }

    /// Acquire the next image. Returns `Some(index)` to render, or `None` when
    /// the swapchain is out-of-date and must be recreated (the semaphore is left
    /// unsignaled in that case, so it is safe to reuse next frame). A merely
    /// suboptimal swapchain still returns `Some` — recreation is then driven by
    /// the present result, after the acquired semaphore has been consumed.
    pub fn acquire_next_image(&self, signal: &VulkanSemaphore) -> Result<Option<u32>, EngineError> {
        unsafe {
            match self.device.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                signal.raw(),
                vk::Fence::null(),
            ) {
                Ok((index, _suboptimal)) => Ok(Some(index)),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(None),
                Err(e) => Err(vk_err(e)),
            }
        }
    }

    pub(crate) fn raw(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    pub(crate) fn image(&self, index: u32) -> vk::Image {
        self.images[index as usize]
    }

    pub(crate) fn view(&self, index: u32) -> vk::ImageView {
        self.views[index as usize]
    }

    pub(crate) fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// The swapchain color format (for matching the pipeline's attachment).
    pub fn format(&self) -> rhi_types::Format {
        match self.format {
            vk::Format::B8G8R8A8_UNORM => rhi_types::Format::Bgra8Unorm,
            vk::Format::B8G8R8A8_SRGB => rhi_types::Format::Bgra8Srgb,
            vk::Format::R8G8B8A8_UNORM => rhi_types::Format::Rgba8Unorm,
            _ => rhi_types::Format::Rgba8Srgb,
        }
    }

    /// Current extent in pixels.
    pub fn extent_2d(&self) -> Extent2D {
        Extent2D::new(self.extent.width, self.extent.height)
    }

    /// Number of images in the swapchain.
    pub fn image_count(&self) -> u32 {
        self.images.len() as u32
    }

    fn destroy_resources(&self) {
        unsafe {
            for &view in &self.views {
                self.device.device.destroy_image_view(view, None);
            }
        }
    }
}

impl Drop for VulkanSwapchain {
    fn drop(&mut self) {
        self.destroy_resources();
        unsafe {
            self.device
                .swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

type BuildResult = (
    vk::SwapchainKHR,
    Vec<vk::Image>,
    Vec<vk::ImageView>,
    vk::Format,
    vk::Extent2D,
);

fn build(
    device: &DeviceShared,
    desc: &SwapchainDesc,
    old: vk::SwapchainKHR,
) -> Result<BuildResult, EngineError> {
    unsafe {
        let surface_loader = &device.instance.surface_loader;
        let surface = device.instance.surface;
        let pd = device.physical_device;

        let caps = surface_loader
            .get_physical_device_surface_capabilities(pd, surface)
            .map_err(vk_err)?;
        let formats = surface_loader
            .get_physical_device_surface_formats(pd, surface)
            .map_err(vk_err)?;
        let present_modes = surface_loader
            .get_physical_device_surface_present_modes(pd, surface)
            .map_err(vk_err)?;

        // Prefer the requested format; otherwise take the first available.
        let wanted = to_vk_format(desc.format);
        let surface_format = formats
            .iter()
            .find(|f| f.format == wanted && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .copied()
            .unwrap_or(formats[0]);

        let present_mode = choose_present_mode(&present_modes, desc.present_mode);

        // current_extent of 0xFFFFFFFF means "pick your own size".
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: desc
                    .extent
                    .width
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: desc
                    .extent
                    .height
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };

        let mut image_count = desc.image_count.max(caps.min_image_count);
        if caps.max_image_count != 0 {
            image_count = image_count.min(caps.max_image_count);
        }

        let ci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            // TRANSFER_SRC lets us copy a rendered image back for screenshots.
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old);

        let swapchain = device
            .swapchain_loader
            .create_swapchain(&ci, None)
            .map_err(vk_err)?;
        let images = device
            .swapchain_loader
            .get_swapchain_images(swapchain)
            .map_err(vk_err)?;
        tracing::debug!(
            "swapchain {}x{} fmt={:?} present={:?} images={}",
            extent.width,
            extent.height,
            surface_format.format,
            present_mode,
            images.len()
        );

        let mut views = Vec::with_capacity(images.len());
        for &image in &images {
            let view_ci = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .subresource_range(color_subresource_range());
            views.push(
                device
                    .device
                    .create_image_view(&view_ci, None)
                    .map_err(vk_err)?,
            );
        }

        Ok((swapchain, images, views, surface_format.format, extent))
    }
}

fn choose_present_mode(
    available: &[vk::PresentModeKHR],
    wanted: PresentMode,
) -> vk::PresentModeKHR {
    let target = match wanted {
        PresentMode::Fifo => vk::PresentModeKHR::FIFO,
        PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
    };
    if available.contains(&target) {
        target
    } else {
        // FIFO is guaranteed to be available by the spec.
        vk::PresentModeKHR::FIFO
    }
}
