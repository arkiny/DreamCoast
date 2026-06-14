//! Empty root signature + graphics pipeline state (DXIL, no input layout).

use std::ffi::c_void;
use std::rc::Rc;

use engine_core::EngineError;
use rhi_types::{GraphicsPipelineDesc, PrimitiveTopology};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1, D3D12_BLEND_DESC, D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD,
    D3D12_BLEND_ZERO, D3D12_COLOR_WRITE_ENABLE_ALL, D3D12_COMPARISON_FUNC_ALWAYS,
    D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF, D3D12_CULL_MODE_NONE, D3D12_DEPTH_STENCIL_DESC,
    D3D12_DEPTH_WRITE_MASK_ZERO, D3D12_FILL_MODE_SOLID, D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    D3D12_LOGIC_OP_NOOP, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, D3D12_RASTERIZER_DESC,
    D3D12_RENDER_TARGET_BLEND_DESC, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, D3D12_SHADER_BYTECODE,
    D3D12SerializeRootSignature, ID3D12PipelineState, ID3D12RootSignature,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use crate::device::DeviceShared;
use crate::instance::d3d_err;
use crate::to_dxgi_format;

/// A graphics pipeline: its root signature and pipeline state object.
pub struct D3d12GraphicsPipeline {
    #[allow(dead_code)] // kept alive; referenced by the PSO and bound at draw time
    device: Rc<DeviceShared>,
    root_signature: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

impl D3d12GraphicsPipeline {
    pub(crate) fn new(
        device: Rc<DeviceShared>,
        desc: &GraphicsPipelineDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let root_signature = create_root_signature(&device)?;

            let _ = match desc.topology {
                PrimitiveTopology::TriangleList => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            };

            let mut pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                // Borrow the root signature without AddRef (kept alive in `self`).
                pRootSignature: std::mem::transmute_copy(&root_signature),
                VS: shader_bytecode(desc.vertex_bytes),
                PS: shader_bytecode(desc.fragment_bytes),
                RasterizerState: rasterizer_default(),
                BlendState: blend_opaque(),
                DepthStencilState: depth_disabled(),
                SampleMask: u32::MAX,
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
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
            })
        }
    }

    pub(crate) fn root_signature(&self) -> &ID3D12RootSignature {
        &self.root_signature
    }

    pub(crate) fn pso(&self) -> &ID3D12PipelineState {
        &self.pso
    }
}

fn create_root_signature(device: &DeviceShared) -> Result<ID3D12RootSignature, EngineError> {
    unsafe {
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: 0,
            pParameters: std::ptr::null(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        D3D12SerializeRootSignature(
            &rs_desc,
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

fn blend_opaque() -> D3D12_BLEND_DESC {
    let rt = D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: false.into(),
        LogicOpEnable: false.into(),
        SrcBlend: D3D12_BLEND_ONE,
        DestBlend: D3D12_BLEND_ZERO,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ZERO,
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
