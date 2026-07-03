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
    MTLBlendFactor, MTLBlendOperation, MTLColorWriteMask, MTLCompareFunction,
    MTLDepthStencilDescriptor, MTLDevice, MTLFunction, MTLLibrary, MTLMeshRenderPipelineDescriptor,
    MTLPipelineOption, MTLRenderPipelineDescriptor, MTLSize, MTLVertexDescriptor, MTLVertexFormat,
    MTLVertexStepFunction,
};
use rhi_types::{
    BlendMode, ComputePipelineDesc, DepthCompare, GraphicsPipelineDesc, MeshPipelineDesc,
    VertexLayout,
};

use crate::resources::{
    MetalComputePipeline, MetalGraphicsPipeline, MetalMeshPipeline, VERTEX_BUFFER_INDEX,
};
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

    // Color attachments (one per format). Opaque/AlphaBlend share one mode across all MRT;
    // DecalAlbedo is per-attachment (RT0 blends RGB, the rest are write-masked off).
    let attachments = pd.colorAttachments();
    for (i, format) in desc.color_formats.iter().enumerate() {
        let attach = unsafe { attachments.objectAtIndexedSubscript(i) };
        attach.setPixelFormat(pixel_format(*format));
        match desc.blend {
            BlendMode::Opaque => {}
            BlendMode::AlphaBlend => {
                attach.setBlendingEnabled(true);
                attach.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
                attach.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
                attach.setRgbBlendOperation(MTLBlendOperation::Add);
                attach.setSourceAlphaBlendFactor(MTLBlendFactor::One);
                attach.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
                attach.setAlphaBlendOperation(MTLBlendOperation::Add);
            }
            // Deferred-decal preset (see `BlendMode::DecalAlbedo`): attachment 0 alpha-blends
            // its RGB into the G-buffer albedo with a write mask of RGB only (A = baked AO is
            // preserved); every other attachment is masked off so the decal leaves the rest of
            // the G-buffer (normal / metallic / roughness / world-pos) untouched.
            BlendMode::DecalAlbedo => {
                if i == 0 {
                    // RT0 albedo: alpha-blend RGB, write RGB (A = baked AO preserved).
                    attach.setBlendingEnabled(true);
                    attach.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
                    attach.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
                    attach.setRgbBlendOperation(MTLBlendOperation::Add);
                    attach.setWriteMask(
                        MTLColorWriteMask::Red | MTLColorWriteMask::Green | MTLColorWriteMask::Blue,
                    );
                } else if i == 2 {
                    // RT2 material: alpha-blend the decal roughness into G only (write mask Green);
                    // metallic (R) and AO (B) stay the underlying surface's. (A4)
                    attach.setBlendingEnabled(true);
                    attach.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
                    attach.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
                    attach.setRgbBlendOperation(MTLBlendOperation::Add);
                    attach.setWriteMask(MTLColorWriteMask::Green);
                } else {
                    // RT1 normal, RT3 world-pos: untouched.
                    attach.setWriteMask(MTLColorWriteMask::empty());
                }
            }
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

    // Depth test/write is render *state* (an `MTLDepthStencilState`), separate from
    // the pipeline's depth attachment format; build one for depth-testing passes so
    // `bind_graphics_pipeline` can set it. The default `Less` maps to `LessEqual` here
    // — the pre-existing mapping kept so every current pipeline stays byte-identical —
    // and `Equal` maps to `Equal` for the depth-pre-pass base pass (matches VK/DX EQUAL).
    let depth_stencil = if desc.depth_test {
        let dsd = MTLDepthStencilDescriptor::new();
        dsd.setDepthCompareFunction(match desc.depth_compare {
            DepthCompare::Less => MTLCompareFunction::LessEqual,
            DepthCompare::Equal => MTLCompareFunction::Equal,
        });
        dsd.setDepthWriteEnabled(desc.depth_write);
        Some(
            device
                .newDepthStencilStateWithDescriptor(&dsd)
                .ok_or_else(|| rhi_err("newDepthStencilState failed"))?,
        )
    } else {
        None
    };

    Ok(MetalGraphicsPipeline {
        state,
        bindless: desc.bindless,
        uses_globals: desc.uniform_buffer,
        depth_stencil,
    })
}

/// Compile a mesh-shader pipeline (Phase 14) from object/mesh/fragment metallib blobs. The
/// legacy `MTLRenderPipelineDescriptor` mesh path (`setObjectFunction`/`setMeshFunction`,
/// macOS 13+) produces a normal `MTLRenderPipelineState` — bound like a graphics pipeline,
/// drawn with `drawMeshThreadgroups`. M0 uses mesh+fragment only (no object stage), opaque,
/// no depth.
pub(crate) fn build_mesh(
    device: &ProtocolObject<dyn MTLDevice>,
    desc: &MeshPipelineDesc,
) -> Result<MetalMeshPipeline> {
    let pd = MTLMeshRenderPipelineDescriptor::new();
    if let Some(object_bytes) = desc.object_bytes {
        let object = load_function(device, object_bytes, desc.object_entry)?;
        unsafe { pd.setObjectFunction(Some(&object)) };
    }
    let mesh = load_function(device, desc.mesh_bytes, desc.mesh_entry)?;
    let fragment = load_function(device, desc.fragment_bytes, desc.fragment_entry)?;
    unsafe {
        pd.setMeshFunction(Some(&mesh));
        pd.setFragmentFunction(Some(&fragment));
    }

    let attachments = pd.colorAttachments();
    for (i, format) in desc.color_formats.iter().enumerate() {
        let attach = unsafe { attachments.objectAtIndexedSubscript(i) };
        attach.setPixelFormat(pixel_format(*format));
    }
    if let Some(depth) = desc.depth_format {
        pd.setDepthAttachmentPixelFormat(pixel_format(depth));
    }

    let state = device
        .newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
            &pd,
            MTLPipelineOption::None,
            None,
        )
        .map_err(|e| rhi_err(format!("newRenderPipelineState (mesh) failed: {e}")))?;

    let size = |t: [u32; 3]| MTLSize {
        width: t[0].max(1) as usize,
        height: t[1].max(1) as usize,
        depth: t[2].max(1) as usize,
    };
    Ok(MetalMeshPipeline {
        state,
        object_threads: size(desc.object_threads),
        mesh_threads: size(desc.mesh_threads),
        bindless: desc.bindless,
        uses_globals: desc.uniform_buffer,
    })
}

/// Compile a compute pipeline from a metallib blob + the shader's threadgroup size.
pub(crate) fn build_compute(
    device: &ProtocolObject<dyn MTLDevice>,
    desc: &ComputePipelineDesc,
) -> Result<MetalComputePipeline> {
    let cs = load_function(device, desc.compute_bytes, desc.compute_entry)?;
    let state = device
        .newComputePipelineStateWithFunction_error(&cs)
        .map_err(|e| rhi_err(format!("newComputePipelineState failed: {e}")))?;
    let [x, y, z] = desc.threads_per_group;
    Ok(MetalComputePipeline {
        state,
        threads_per_group: MTLSize {
            width: x.max(1) as usize,
            height: y.max(1) as usize,
            depth: z.max(1) as usize,
        },
        bindless: desc.bindless,
        uses_globals: desc.uniform_buffer,
    })
}

/// Create an `MTLLibrary` from a `.metallib` blob and fetch `entry` from it.
pub(crate) fn load_function(
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
    // (attribute location, format, offset) per attribute, plus the buffer stride.
    // The location is explicit (not the array index) so a layout can skip an
    // attribute the shader doesn't consume — e.g. `MeshPositionUv` provides
    // locations 0 and 2 with no 1, matching the shadow VS's `[[attribute]]` set.
    let (attrs, stride): (&[(usize, MTLVertexFormat, usize)], usize) = match layout {
        VertexLayout::None => return None,
        VertexLayout::ImGui => (
            &[
                (0, MTLVertexFormat::Float2, 0),
                (1, MTLVertexFormat::Float2, 8),
                (2, MTLVertexFormat::UChar4Normalized, 16),
            ],
            20,
        ),
        VertexLayout::Mesh => (
            &[
                (0, MTLVertexFormat::Float3, 0),
                (1, MTLVertexFormat::Float3, 12),
                (2, MTLVertexFormat::Float2, 24),
            ],
            32,
        ),
        VertexLayout::MeshPosition => (&[(0, MTLVertexFormat::Float3, 0)], 32),
        VertexLayout::MeshPosNormal => (
            &[
                (0, MTLVertexFormat::Float3, 0),
                (1, MTLVertexFormat::Float3, 12),
            ],
            32,
        ),
        // Position (attribute 0) + uv (attribute 1), normal skipped. The shadow
        // VS omits NORMAL, so Slang packs uv at attribute 1 over the 32-byte buffer.
        VertexLayout::MeshPositionUv => (
            &[
                (0, MTLVertexFormat::Float3, 0),
                (1, MTLVertexFormat::Float2, 24),
            ],
            32,
        ),
    };

    let vd = MTLVertexDescriptor::new();
    let vd_attrs = vd.attributes();
    for (location, format, offset) in attrs.iter().copied() {
        let a = unsafe { vd_attrs.objectAtIndexedSubscript(location) };
        a.setFormat(format);
        unsafe {
            a.setOffset(offset);
            a.setBufferIndex(VERTEX_BUFFER_INDEX);
        }
    }
    let vd_layouts = vd.layouts();
    let buf = unsafe { vd_layouts.objectAtIndexedSubscript(VERTEX_BUFFER_INDEX) };
    unsafe { buf.setStride(stride) };
    buf.setStepFunction(MTLVertexStepFunction::PerVertex);
    Some(vd)
}
