//! Graphics pipeline construction (`MTLRenderPipelineState`).
//!
//! Builds an `MTLRenderPipelineState` from a [`GraphicsPipelineDesc`]: each stage
//! is a standalone `.metallib` blob turned into an `MTLLibrary` (via a
//! `dispatch_data_t`), the entry function is looked up by name (Slang preserves
//! `vsMain`/`fsMain`), and the color/depth formats, blending, and vertex layout
//! are translated to their Metal equivalents.

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLDevice, MTLFunction, MTLLibrary,
    MTLRenderPipelineDescriptor, MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction,
};
use rhi_types::{BlendMode, GraphicsPipelineDesc, VertexLayout};

use crate::resources::{MetalGraphicsPipeline, VERTEX_BUFFER_INDEX};
use crate::{Result, pixel_format, rhi_err};

/// Compile a graphics pipeline from per-stage metallib blobs + render state.
pub(crate) fn build(
    device: &ProtocolObject<dyn MTLDevice>,
    desc: &GraphicsPipelineDesc,
) -> Result<MetalGraphicsPipeline> {
    let vs = load_function(device, desc.vertex_bytes, desc.vertex_entry)?;
    let fs = load_function(device, desc.fragment_bytes, desc.fragment_entry)?;

    let pd = MTLRenderPipelineDescriptor::new();
    pd.setVertexFunction(Some(&vs));
    pd.setFragmentFunction(Some(&fs));

    // Color attachments (one per format; MRT shares one blend mode).
    let attachments = pd.colorAttachments();
    for (i, format) in desc.color_formats.iter().enumerate() {
        let attach = unsafe { attachments.objectAtIndexedSubscript(i) };
        attach.setPixelFormat(pixel_format(*format));
        if matches!(desc.blend, BlendMode::AlphaBlend) {
            attach.setBlendingEnabled(true);
            attach.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
            attach.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            attach.setRgbBlendOperation(MTLBlendOperation::Add);
            attach.setSourceAlphaBlendFactor(MTLBlendFactor::One);
            attach.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            attach.setAlphaBlendOperation(MTLBlendOperation::Add);
        }
    }

    if let Some(depth) = desc.depth_format {
        pd.setDepthAttachmentPixelFormat(pixel_format(depth));
    }

    if let Some(vd) = vertex_descriptor(desc.vertex_layout) {
        pd.setVertexDescriptor(Some(&vd));
    }

    let state = device
        .newRenderPipelineStateWithDescriptor_error(&pd)
        .map_err(|e| rhi_err(format!("newRenderPipelineState failed: {e}")))?;
    Ok(MetalGraphicsPipeline { state })
}

/// Create an `MTLLibrary` from a `.metallib` blob and fetch `entry` from it.
fn load_function(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
    entry: &str,
) -> Result<Retained<ProtocolObject<dyn MTLFunction>>> {
    // `newLibraryWithData:` wants a dispatch_data_t; `from_bytes` copies the blob
    // so the (static) shader bytes need not outlive the call.
    let data = DispatchData::from_bytes(bytes);
    let library = device
        .newLibraryWithData_error(&data)
        .map_err(|e| rhi_err(format!("newLibraryWithData failed: {e}")))?;
    let name = NSString::from_str(entry);
    library
        .newFunctionWithName(&name)
        .ok_or_else(|| rhi_err(format!("entry function '{entry}' not found in metallib")))
}

/// Build the Metal vertex descriptor for a layout, or `None` for `VertexLayout::None`
/// (vertices synthesized from the vertex id). Attributes mirror the Vulkan backend
/// and all source from the single buffer at [`VERTEX_BUFFER_INDEX`].
fn vertex_descriptor(layout: VertexLayout) -> Option<Retained<MTLVertexDescriptor>> {
    // (format, offset) per attribute, plus the buffer stride.
    let (attrs, stride): (&[(MTLVertexFormat, usize)], usize) = match layout {
        VertexLayout::None => return None,
        VertexLayout::ImGui => (
            &[
                (MTLVertexFormat::Float2, 0),
                (MTLVertexFormat::Float2, 8),
                (MTLVertexFormat::UChar4Normalized, 16),
            ],
            20,
        ),
        VertexLayout::Mesh => (
            &[
                (MTLVertexFormat::Float3, 0),
                (MTLVertexFormat::Float3, 12),
                (MTLVertexFormat::Float2, 24),
            ],
            32,
        ),
        VertexLayout::MeshPosition => (&[(MTLVertexFormat::Float3, 0)], 32),
        VertexLayout::MeshPosNormal => (
            &[(MTLVertexFormat::Float3, 0), (MTLVertexFormat::Float3, 12)],
            32,
        ),
    };

    let vd = MTLVertexDescriptor::new();
    let vd_attrs = vd.attributes();
    for (location, (format, offset)) in attrs.iter().enumerate() {
        let a = unsafe { vd_attrs.objectAtIndexedSubscript(location) };
        a.setFormat(*format);
        unsafe {
            a.setOffset(*offset);
            a.setBufferIndex(VERTEX_BUFFER_INDEX);
        }
    }
    let vd_layouts = vd.layouts();
    let buf = unsafe { vd_layouts.objectAtIndexedSubscript(VERTEX_BUFFER_INDEX) };
    unsafe { buf.setStride(stride) };
    buf.setStepFunction(MTLVertexStepFunction::PerVertex);
    Some(vd)
}
