//! Root signature (empty or bindless) + graphics pipeline state (DXIL).

use std::ffi::c_void;
use std::rc::Rc;

use dreamcoast_core::EngineError;
use rhi_types::{BlendMode, DepthCompare, GraphicsPipelineDesc, PrimitiveTopology, VertexLayout};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1, D3D12_BLEND_DESC, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_ONE,
    D3D12_BLEND_OP_ADD, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_ZERO, D3D12_COLOR_WRITE_ENABLE_ALL,
    D3D12_COLOR_WRITE_ENABLE_BLUE, D3D12_COLOR_WRITE_ENABLE_GREEN, D3D12_COLOR_WRITE_ENABLE_RED,
    D3D12_COMPARISON_FUNC_ALWAYS, D3D12_COMPARISON_FUNC_EQUAL, D3D12_COMPARISON_FUNC_LESS,
    D3D12_COMPARISON_FUNC_NEVER, D3D12_COMPUTE_PIPELINE_STATE_DESC,
    D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF, D3D12_CULL_MODE_NONE, D3D12_DEPTH_STENCIL_DESC,
    D3D12_DEPTH_WRITE_MASK_ALL, D3D12_DEPTH_WRITE_MASK_ZERO, D3D12_DESCRIPTOR_RANGE,
    D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_DESCRIPTOR_RANGE_TYPE_UAV, D3D12_FILL_MODE_SOLID,
    D3D12_FILTER_ANISOTROPIC, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, D3D12_INPUT_ELEMENT_DESC, D3D12_INPUT_LAYOUT_DESC,
    D3D12_LOGIC_OP_NOOP, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, D3D12_RASTERIZER_DESC,
    D3D12_RENDER_TARGET_BLEND_DESC, D3D12_ROOT_CONSTANTS, D3D12_ROOT_DESCRIPTOR,
    D3D12_ROOT_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0,
    D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, D3D12_ROOT_PARAMETER_TYPE_CBV,
    D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT, D3D12_ROOT_SIGNATURE_FLAG_NONE,
    D3D12_SHADER_BYTECODE, D3D12_SHADER_VISIBILITY_ALL,
    D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK, D3D12_STATIC_SAMPLER_DESC,
    D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12SerializeRootSignature,
    ID3D12PipelineState, ID3D12RootSignature,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT,
    DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
};
use windows::core::s;

use crate::device::{
    BINDLESS_COUNT, CUBE_COUNT, DeviceShared, STORAGE_BUFFER_BASE, STORAGE_BUFFER_COUNT,
    STORAGE_IMAGE_BASE, STORAGE_IMAGE_COUNT, STORAGE_VOLUME_BASE, STORAGE_VOLUME_COUNT, TLAS_SLOT,
    VOLUME_BASE, VOLUME_COUNT,
};
use crate::instance::d3d_err;
use crate::to_dxgi_format;
use rhi_types::ComputePipelineDesc;

/// A graphics pipeline: its root signature and pipeline state object.
pub struct D3d12GraphicsPipeline {
    #[allow(dead_code)]
    device: Rc<DeviceShared>,
    root_signature: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    bindless: bool,
    uniform_buffer: bool,
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
            let mesh_elems = [
                input_elem(s!("POSITION"), DXGI_FORMAT_R32G32B32_FLOAT, 0),
                input_elem(s!("NORMAL"), DXGI_FORMAT_R32G32B32_FLOAT, 12),
                input_elem(s!("TEXCOORD"), DXGI_FORMAT_R32G32_FLOAT, 24),
            ];
            // Same interleaved mesh buffer, POSITION + TEXCOORD only (normal
            // skipped): the shadow VS reads position for depth and uv for
            // alpha-cutout. Non-contiguous in `mesh_elems`, so its own array.
            let mesh_pos_uv_elems = [
                input_elem(s!("POSITION"), DXGI_FORMAT_R32G32B32_FLOAT, 0),
                input_elem(s!("TEXCOORD"), DXGI_FORMAT_R32G32_FLOAT, 24),
            ];
            let input_layout = match desc.vertex_layout {
                VertexLayout::None => D3D12_INPUT_LAYOUT_DESC::default(),
                VertexLayout::ImGui => D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: imgui_elems.as_ptr(),
                    NumElements: imgui_elems.len() as u32,
                },
                VertexLayout::Mesh => D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: mesh_elems.as_ptr(),
                    NumElements: mesh_elems.len() as u32,
                },
                // Same interleaved mesh buffer, only POSITION (shadow pass).
                VertexLayout::MeshPosition => D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: mesh_elems.as_ptr(),
                    NumElements: 1,
                },
                // Position + normal, uv skipped (capture pass).
                VertexLayout::MeshPosNormal => D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: mesh_elems.as_ptr(),
                    NumElements: 2,
                },
                // Position + uv, normal skipped (shadow pass).
                VertexLayout::MeshPositionUv => D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: mesh_pos_uv_elems.as_ptr(),
                    NumElements: mesh_pos_uv_elems.len() as u32,
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
                DepthStencilState: depth_state(
                    desc.depth_test,
                    desc.depth_write,
                    desc.depth_compare,
                ),
                SampleMask: u32::MAX,
                InputLayout: input_layout,
                PrimitiveTopologyType: topology,
                NumRenderTargets: desc.color_formats.len() as u32,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            for (i, &fmt) in desc.color_formats.iter().enumerate() {
                pso_desc.RTVFormats[i] = to_dxgi_format(fmt);
            }
            pso_desc.DSVFormat = match desc.depth_format {
                Some(f) => to_dxgi_format(f),
                None => DXGI_FORMAT_UNKNOWN,
            };

            let pso: ID3D12PipelineState = device
                .device
                .CreateGraphicsPipelineState(&pso_desc)
                .map_err(d3d_err)?;

            Ok(Self {
                device,
                root_signature,
                pso,
                bindless: desc.bindless,
                uniform_buffer: desc.uniform_buffer,
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

    pub(crate) fn uses_uniform(&self) -> bool {
        self.uniform_buffer
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

        // Bindless: one table over the shared heap with four ranges (textures, cubes,
        // storage images, storage buffers), all in register space 1 via the shared
        // `ParameterBlock` (see `bindless_ranges`) — plus 32-bit root constants (b0),
        // an optional globals CBV (b1), and a static sampler (`s0,space1`, inside the
        // block). Visibility ALL: the vertex stage reads storage buffers (particle/
        // cull vertex-pull) and the pixel stage reads textures.
        let ranges = bindless_ranges();
        // params[0] = bindless table, params[1] = 32-bit root constants (b0).
        // params[2] = root CBV (b1) for the per-frame globals, when opted in.
        let mut params = vec![
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: ranges.len() as u32,
                        pDescriptorRanges: ranges.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
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
        if desc.uniform_buffer {
            params.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 1,
                        RegisterSpace: 0,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
        }
        let samplers = bindless_static_samplers();
        serialize_and_create(
            device,
            &D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: flags,
            },
        )
    }
}

/// The shared static samplers inside the bindless `ParameterBlock` (space1): `s0` clamp
/// (cubes/volumes/G-buffer) + `s1` wrap (tiling material textures — floor tiles, brick
/// courses with glTF UVs > 1). Order matches the `samp` / `samp_wrap` declaration in
/// `bindless.slang`.
fn bindless_static_samplers() -> [D3D12_STATIC_SAMPLER_DESC; 2] {
    let base = D3D12_STATIC_SAMPLER_DESC {
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
        RegisterSpace: 1, // inside the shared ParameterBlock
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    // 1b: opt-in anisotropic filtering for the wrap sampler (grazing material surfaces). `P_ANISO=<N>`
    // (clamped to D3D12's [1,16]) switches it to ANISOTROPIC; unset (or <=1) keeps the linear filter
    // => byte-identical and no DX≡VK risk by default. Anisotropy is driver-dependent => opt-in only.
    let aniso = std::env::var("P_ANISO")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(1.0);
    let wrap = if aniso > 1.0 {
        D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_ANISOTROPIC,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            MaxAnisotropy: (aniso.round() as u32).clamp(1, 16),
            ShaderRegister: 1,
            ..base
        }
    } else {
        D3D12_STATIC_SAMPLER_DESC {
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            ShaderRegister: 1,
            ..base
        }
    };
    [base, wrap]
}

/// The four bindless descriptor-table ranges, shared verbatim by the graphics and
/// compute root signatures. Every shader binds these through a single Slang
/// `ParameterBlock` (`bindless.slang`), which the DXIL backend places in register
/// **space 1** (space0 holds the loose `b0` push constants + the optional `b1`
/// globals CBV, so the block — a whole space — takes the next one). The block packs
/// its members into consecutive registers within that space:
///   * 2D textures     — `t0,space1`    (1024 SRVs)
///   * cubemaps        — `t1024,space1` (64 SRVs, right after the textures)
///   * storage images  — `u0,space1`    (64 UAVs)
///   * storage buffers — `u64,space1`   (64 UAVs, right after the storage images)
///
/// The shared sampler is the static sampler at `s0,space1` (see the root-signature
/// builders). Each range keeps its own heap region (the `*_BASE` offsets), so the
/// device binds — and the shader indexes — each array 0-based.
///
/// A fifth range covers the scene TLAS SRV at `t{BINDLESS_COUNT+CUBE_COUNT},
/// space1` (Phase 8). It is always present so every bindless root signature stays
/// uniform; non-RT pipelines simply never reference it (and the slot stays empty
/// on devices without a built scene).
pub(crate) fn bindless_ranges() -> [D3D12_DESCRIPTOR_RANGE; 7] {
    [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: BINDLESS_COUNT,
            BaseShaderRegister: 0,
            RegisterSpace: 1,
            OffsetInDescriptorsFromTableStart: 0,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: CUBE_COUNT,
            BaseShaderRegister: BINDLESS_COUNT,
            RegisterSpace: 1,
            OffsetInDescriptorsFromTableStart: BINDLESS_COUNT,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: STORAGE_IMAGE_COUNT,
            BaseShaderRegister: 0,
            RegisterSpace: 1,
            OffsetInDescriptorsFromTableStart: STORAGE_IMAGE_BASE,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: STORAGE_BUFFER_COUNT,
            // Storage buffers share the UAV register space with the storage images,
            // so they start at `u{STORAGE_IMAGE_COUNT}` (the shader packs them after).
            BaseShaderRegister: STORAGE_IMAGE_COUNT,
            RegisterSpace: 1,
            OffsetInDescriptorsFromTableStart: STORAGE_BUFFER_BASE,
        },
        // Scene TLAS SRV at t{BINDLESS_COUNT+CUBE_COUNT}, space1 (Phase 8).
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: BINDLESS_COUNT + CUBE_COUNT,
            RegisterSpace: 1,
            OffsetInDescriptorsFromTableStart: TLAS_SLOT,
        },
        // Sampled 3D volumes SRV at t{BINDLESS_COUNT+CUBE_COUNT+1}, space1 (Phase 11).
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: VOLUME_COUNT,
            BaseShaderRegister: BINDLESS_COUNT + CUBE_COUNT + 1,
            RegisterSpace: 1,
            OffsetInDescriptorsFromTableStart: VOLUME_BASE,
        },
        // Storage 3D volumes UAV at u{STORAGE_IMAGE_COUNT+STORAGE_BUFFER_COUNT},
        // space1 (Phase 11). Shares the UAV register space after the storage buffers.
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: STORAGE_VOLUME_COUNT,
            BaseShaderRegister: STORAGE_IMAGE_COUNT + STORAGE_BUFFER_COUNT,
            RegisterSpace: 1,
            OffsetInDescriptorsFromTableStart: STORAGE_VOLUME_BASE,
        },
    ]
}

/// A D3D12 root signature is capped at **64 DWORDs** total. Inline 32-bit root
/// constants cost `push_size / 4` DWORDs; the bindless descriptor table costs 1 and
/// the optional globals CBV costs 2. When the push block alone would leave no room
/// for those (e.g. a 256-byte / 64-DWORD block + the 1-DWORD table = 65 > 64), the
/// block is spilled into a **root CBV** at `b0` — 2 DWORDs regardless of size,
/// uploaded per-dispatch from a ring — instead of inline root constants. This is the
/// same "large per-pass constants live in a bound constant buffer" split reference
/// engines use. The DXIL binds `b0` as a cbuffer either way (Slang maps the push
/// constant to `register(b0, space0)`), so the shader bytecode is byte-identical and
/// the delivered bytes — hence the rendered output — are unchanged.
const ROOT_SIG_MAX_DWORDS: u32 = 64;

/// Whether `desc`'s push block must spill to a root CBV to fit the 64-DWORD budget.
/// Only meaningful for bindless pipelines (the non-bindless path has no root params).
pub(crate) fn compute_push_via_cbv(desc: &ComputePipelineDesc) -> bool {
    if !desc.bindless {
        return false;
    }
    let table = 1; // param[0]: bindless descriptor table
    let globals_cbv = if desc.uniform_buffer { 2 } else { 0 }; // param[2]: root CBV (b1)
    table + desc.push_constant_size / 4 + globals_cbv > ROOT_SIG_MAX_DWORDS
}

/// A compute pipeline: its (bindless) root signature and compute PSO.
pub struct D3d12ComputePipeline {
    #[allow(dead_code)]
    device: Rc<DeviceShared>,
    root_signature: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    bindless: bool,
    uniform_buffer: bool,
    /// `true` ⇒ push constants bind as a root CBV (param 1) fed from the command
    /// buffer's upload ring, not inline 32-bit constants. See [`compute_push_via_cbv`].
    push_via_cbv: bool,
}

/// A mesh-shader pipeline (Phase 14 virtual geometry). Placeholder for the Windows-box
/// follow-up: it must build a mesh-shader PSO (SM6.5, `D3D12_PIPELINE_STATE_STREAM` with
/// MS/PS). Until then [`D3d12Device::capabilities`] reports `mesh_shader: false`, so nothing
/// constructs this on D3D12 (the smokes gate on it and are verified on Metal). DX≡VK parity
/// is the tracked follow-up.
pub struct D3d12MeshPipeline {
    _priv: (),
}

impl D3d12ComputePipeline {
    pub(crate) fn new(
        device: Rc<DeviceShared>,
        desc: &ComputePipelineDesc,
    ) -> Result<Self, EngineError> {
        unsafe {
            let root_signature = create_compute_root_signature(&device, desc)?;
            let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root_signature),
                CS: shader_bytecode(desc.compute_bytes),
                ..Default::default()
            };
            let pso: ID3D12PipelineState = device
                .device
                .CreateComputePipelineState(&pso_desc)
                .map_err(d3d_err)?;
            Ok(Self {
                device,
                root_signature,
                pso,
                bindless: desc.bindless,
                uniform_buffer: desc.uniform_buffer,
                push_via_cbv: compute_push_via_cbv(desc),
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

    pub(crate) fn uses_uniform(&self) -> bool {
        self.uniform_buffer
    }

    /// Whether push constants must be uploaded through the root-CBV ring (param 1)
    /// rather than set as inline 32-bit root constants. See [`compute_push_via_cbv`].
    pub(crate) fn push_via_cbv(&self) -> bool {
        self.push_via_cbv
    }
}

fn create_compute_root_signature(
    device: &DeviceShared,
    desc: &ComputePipelineDesc,
) -> Result<ID3D12RootSignature, EngineError> {
    unsafe {
        if !desc.bindless {
            return serialize_and_create(
                device,
                &D3D12_ROOT_SIGNATURE_DESC {
                    NumParameters: 0,
                    pParameters: std::ptr::null(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
                },
            );
        }
        let ranges = bindless_ranges();
        // params[0] = bindless table, params[1] = the push block at b0, and the optional
        // params[2] = root CBV (b1) for the per-frame globals — the same layout as the
        // graphics root signature, so a compute pass can opt into the globals UBO
        // (Stage C7 reflection reprojection) via `uniform_buffer`.
        //
        // params[1] is inline 32-bit root constants when the block fits the 64-DWORD root
        // budget, else a root CBV (b0) that the command buffer feeds from its upload ring
        // — see `compute_push_via_cbv`. Both bind the same `b0` cbuffer the DXIL declares,
        // so the shader is unaffected; only the delivery mechanism (and DWORD cost) differs.
        let push_param = if compute_push_via_cbv(desc) {
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            }
        } else {
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
            }
        };
        let mut params = vec![
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: ranges.len() as u32,
                        pDescriptorRanges: ranges.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            push_param,
        ];
        if desc.uniform_buffer {
            params.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 1,
                        RegisterSpace: 0,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
        }
        let samplers = bindless_static_samplers();
        serialize_and_create(
            device,
            &D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            },
        )
    }
}

pub(crate) unsafe fn serialize_and_create(
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
    // Deferred-decal preset (see `BlendMode::DecalAlbedo`): per-RT independent blend —
    // RT0 alpha-blends RGB into the G-buffer albedo (write mask RGB preserves A = baked AO),
    // every other RT is write-masked off so the decal leaves normal / metallic / roughness /
    // world-pos untouched. Needs `IndependentBlendEnable = TRUE`.
    if matches!(mode, BlendMode::DecalAlbedo) {
        let rt0 = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            LogicOpEnable: false.into(),
            SrcBlend: D3D12_BLEND_SRC_ALPHA,
            DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: (D3D12_COLOR_WRITE_ENABLE_RED.0
                | D3D12_COLOR_WRITE_ENABLE_GREEN.0
                | D3D12_COLOR_WRITE_ENABLE_BLUE.0) as u8,
        };
        let masked = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: false.into(),
            LogicOpEnable: false.into(),
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: 0,
        };
        // RT2 material: alpha-blend the decal roughness into G only (R metallic + B AO stay the
        // surface's). (A4)
        let rt2 = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            LogicOpEnable: false.into(),
            SrcBlend: D3D12_BLEND_SRC_ALPHA,
            DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_GREEN.0 as u8,
        };
        let mut render_target = [masked; 8];
        render_target[0] = rt0;
        render_target[2] = rt2;
        return D3D12_BLEND_DESC {
            AlphaToCoverageEnable: false.into(),
            IndependentBlendEnable: true.into(),
            RenderTarget: render_target,
        };
    }

    let (enable, src, dst) = match mode {
        BlendMode::Opaque => (false, D3D12_BLEND_ONE, D3D12_BLEND_ZERO),
        BlendMode::AlphaBlend => (true, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA),
        BlendMode::DecalAlbedo => unreachable!("handled above"),
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

fn depth_state(test: bool, write: bool, compare: DepthCompare) -> D3D12_DEPTH_STENCIL_DESC {
    D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: test.into(),
        DepthWriteMask: if write {
            D3D12_DEPTH_WRITE_MASK_ALL
        } else {
            D3D12_DEPTH_WRITE_MASK_ZERO
        },
        DepthFunc: if test {
            match compare {
                DepthCompare::Less => D3D12_COMPARISON_FUNC_LESS,
                DepthCompare::Equal => D3D12_COMPARISON_FUNC_EQUAL,
            }
        } else {
            D3D12_COMPARISON_FUNC_ALWAYS
        },
        StencilEnable: false.into(),
        ..Default::default()
    }
}
