//! Root signature (empty or bindless) + graphics pipeline state (DXIL).

use std::ffi::c_void;
use std::rc::Rc;

use engine_core::EngineError;
use rhi_types::{BlendMode, GraphicsPipelineDesc, PrimitiveTopology, VertexLayout};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1, D3D12_BLEND_DESC, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_ONE,
    D3D12_BLEND_OP_ADD, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_ZERO, D3D12_COLOR_WRITE_ENABLE_ALL,
    D3D12_COMPARISON_FUNC_ALWAYS, D3D12_COMPARISON_FUNC_NEVER,
    D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF, D3D12_CULL_MODE_NONE, D3D12_DEPTH_STENCIL_DESC,
    D3D12_DEPTH_WRITE_MASK_ZERO, D3D12_DESCRIPTOR_RANGE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_FILL_MODE_SOLID, D3D12_FILTER_MIN_MAG_MIP_LINEAR,
    D3D12_GRAPHICS_PIPELINE_STATE_DESC, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
    D3D12_INPUT_ELEMENT_DESC, D3D12_INPUT_LAYOUT_DESC, D3D12_LOGIC_OP_NOOP,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, D3D12_RASTERIZER_DESC, D3D12_RENDER_TARGET_BLEND_DESC,
    D3D12_ROOT_CONSTANTS, D3D12_ROOT_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER,
    D3D12_ROOT_PARAMETER_0, D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
    D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, D3D12_SHADER_BYTECODE,
    D3D12_SHADER_VISIBILITY_ALL, D3D12_SHADER_VISIBILITY_PIXEL,
    D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK, D3D12_STATIC_SAMPLER_DESC,
    D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12SerializeRootSignature, ID3D12PipelineState,
    ID3D12RootSignature,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32G32_FLOAT, DXGI_SAMPLE_DESC,
};
use windows::core::s;

use crate::device::DeviceShared;
use crate::instance::d3d_err;
use crate::to_dxgi_format;

/// A graphics pipeline: its root signature and pipeline state object.
pub struct D3d12GraphicsPipeline {
    #[allow(dead_code)]
    device: Rc<DeviceShared>,
    root_signature: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    bindless: bool,
}

impl D3d12GraphicsPipeline {
    pub(crate) fn new(
        device: Rc<DeviceShared>,
        desc: &GraphicsPipelineDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let root_signature = create_root_signature(&device, desc)?;

            // ImGui input layout (only when requested).
            let imgui_elems = [
                input_elem(s!("POSITION"), DXGI_FORMAT_R32G32_FLOAT, 0),
                input_elem(s!("TEXCOORD"), DXGI_FORMAT_R32G32_FLOAT, 8),
                input_elem(s!("COLOR"), DXGI_FORMAT_R8G8B8A8_UNORM, 16),
            ];
            let input_layout = match desc.vertex_layout {
                VertexLayout::None => D3D12_INPUT_LAYOUT_DESC::default(),
                VertexLayout::ImGui => D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: imgui_elems.as_ptr(),
                    NumElements: imgui_elems.len() as u32,
                },
            };

            let topology = match desc.topology {
                PrimitiveTopology::TriangleList => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            };

            let mut pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root_signature),
                VS: shader_bytecode(desc.vertex_bytes),
                PS: shader_bytecode(desc.fragment_bytes),
                RasterizerState: rasterizer_default(),
                BlendState: blend_state(desc.blend),
                DepthStencilState: depth_disabled(),
                SampleMask: u32::MAX,
                InputLayout: input_layout,
                PrimitiveTopologyType: topology,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            pso_desc.RTVFormats[0] = to_dxgi_format(desc.color_format);

            let pso: ID3D12PipelineState = device
                .device
                .CreateGraphicsPipelineState(&pso_desc)
                .map_err(d3d_err)?;

            Ok(Self {
                device,
                root_signature,
                pso,
                bindless: desc.bindless,
            })
        }
    }

    pub(crate) fn root_signature(&self) -> &ID3D12RootSignature {
        &self.root_signature
    }

    pub(crate) fn pso(&self) -> &ID3D12PipelineState {
        &self.pso
    }

    pub(crate) fn is_bindless(&self) -> bool {
        self.bindless
    }
}

fn create_root_signature(
    device: &DeviceShared,
    desc: &GraphicsPipelineDesc,
) -> Result<ID3D12RootSignature, EngineError> {
    unsafe {
        let flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        // Non-bindless (triangle): empty root signature.
        if !desc.bindless {
            return serialize_and_create(
                device,
                &D3D12_ROOT_SIGNATURE_DESC {
                    NumParameters: 0,
                    pParameters: std::ptr::null(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: flags,
                },
            );
        }

        // Bindless: SRV table (t0, unbounded-ish) + 32-bit root constants (b0) + static sampler (s0).
        let srv_range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            // Unbounded range (shader declares `Texture2D g_textures[]`).
            NumDescriptors: u32::MAX,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        };
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &srv_range,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: desc.push_constant_size / 4,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MipLODBias: 0.0,
            MaxAnisotropy: 0,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            BorderColor: D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
            MinLOD: 0.0,
            MaxLOD: f32::MAX,
            ShaderRegister: 0,
            RegisterSpace: 0,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        };
        serialize_and_create(
            device,
            &D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 1,
                pStaticSamplers: &sampler,
                Flags: flags,
            },
        )
    }
}

unsafe fn serialize_and_create(
    device: &DeviceShared,
    rs_desc: &D3D12_ROOT_SIGNATURE_DESC,
) -> Result<ID3D12RootSignature, EngineError> {
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        D3D12SerializeRootSignature(
            rs_desc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut blob,
            Some(&mut error),
        )
        .map_err(d3d_err)?;
        let blob = blob.ok_or_else(|| EngineError::Rhi("root signature blob was null".into()))?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize());
        device.device.CreateRootSignature(0, bytes).map_err(d3d_err)
    }
}

fn input_elem(
    semantic: windows::core::PCSTR,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    offset: u32,
) -> D3D12_INPUT_ELEMENT_DESC {
    D3D12_INPUT_ELEMENT_DESC {
        SemanticName: semantic,
        SemanticIndex: 0,
        Format: format,
        InputSlot: 0,
        AlignedByteOffset: offset,
        InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
        InstanceDataStepRate: 0,
    }
}

fn shader_bytecode(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr() as *const c_void,
        BytecodeLength: bytes.len(),
    }
}

fn rasterizer_default() -> D3D12_RASTERIZER_DESC {
    D3D12_RASTERIZER_DESC {
        FillMode: D3D12_FILL_MODE_SOLID,
        CullMode: D3D12_CULL_MODE_NONE,
        FrontCounterClockwise: false.into(),
        DepthBias: 0,
        DepthBiasClamp: 0.0,
        SlopeScaledDepthBias: 0.0,
        DepthClipEnable: true.into(),
        MultisampleEnable: false.into(),
        AntialiasedLineEnable: false.into(),
        ForcedSampleCount: 0,
        ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
    }
}

fn blend_state(mode: BlendMode) -> D3D12_BLEND_DESC {
    let (enable, src, dst) = match mode {
        BlendMode::Opaque => (false, D3D12_BLEND_ONE, D3D12_BLEND_ZERO),
        BlendMode::AlphaBlend => (true, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA),
    };
    let rt = D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: enable.into(),
        LogicOpEnable: false.into(),
        SrcBlend: src,
        DestBlend: dst,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
    };
    D3D12_BLEND_DESC {
        AlphaToCoverageEnable: false.into(),
        IndependentBlendEnable: false.into(),
        RenderTarget: [rt; 8],
    }
}

fn depth_disabled() -> D3D12_DEPTH_STENCIL_DESC {
    D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: false.into(),
        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
        DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        StencilEnable: false.into(),
        ..Default::default()
    }
}
