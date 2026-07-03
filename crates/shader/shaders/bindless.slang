// Shared bindless resource table, declared as a Slang `ParameterBlock`.
//
// A `ParameterBlock<T>` compiles, from one source, to:
//   * Metal  — an **argument buffer** (the M3 enabler: loose unbounded
//              `Texture2D g_textures[]` globals cannot compile to `metallib`,
//              "flexible array member not at end of struct"; a bounded array
//              inside a ParameterBlock can).
//   * Vulkan — a **descriptor set**.
//   * D3D12  — a **descriptor table** (its own register space).
//
// Emitted layout, verified with `slangc` reflection + spirv-asm (Slang 2026.11):
//   * SPIR-V : descriptor **set 0** — binding 0 = `textures[]`, binding 1 = `samp`
//              (`OpDecorate ... DescriptorSet 0 / Binding 0|1`). Byte-identical to
//              the previous loose-global layout, so the shared Vulkan `bindless_set`
//              / `bindless_layout` is UNCHANGED (rhi-vulkan untouched).
//   * Metal  : argument buffer at `[[buffer(1)]]`. Push constants stay
//              `[[buffer(0)]]` (PUSH_CONSTANT_INDEX); the vertex buffer at index 30.
//
// The `[[vk::binding(0, 0)]]` on the block pins it to set 0 even when the push
// constant declares `register(b0, space0)` (which otherwise reserves space 0 and
// bumps the block to set 1). It does not perturb the push constant — it stays a
// real SPIR-V `PushConstant` and Metal `[[buffer(0)]]`.
//
// A ParameterBlock occupies an *entire* descriptor set, so the sampler lives
// inside the block (it cannot be a loose binding in the same set). On Vulkan that
// still resolves to set 0 / binding 1 (matching the previous immutable sampler);
// on D3D12 the sampler moves from a static sampler into the bindless table — a
// root-signature tweak to validate on the Windows box.
//
// M3 declared only `textures` + `samp`. M4 folds in the `cubes[]` array (used by
// `pbr` and the IBL gen shaders), which previously lived as a separate loose
// `g_cubes[]` at `space1`. M5 folds in the read-write storage arrays used by the
// compute shaders (`post_compute`, `particle_sim`/`particle_draw`, `cull`/
// `cull_draw`), which previously lived as loose `g_storage_images[]` (binding 3)
// and `g_storage_buffers[]` (binding 4, `space1`).
//
// Adding members is layout-safe for shaders that don't use them: Slang drops the
// unused binding (e.g. `mesh`/`imgui`/`gbuffer` keep set 0 / bindings 0,1), so
// `rhi-vulkan` is untouched. In a globals-using shader (`pbr`) the block still
// pins to **set 0** via `[[vk::binding(0,0)]]` even with a loose globals UBO at
// set 1 — matching the previous loose-global layout exactly (Vulkan unchanged).
// On Metal the loose globals takes `[[buffer(1)]]`, pushing this argument buffer
// to `[[buffer(2)]]` for globals shaders (non-globals shaders keep it at
// `[[buffer(1)]]`); verified via `slangc` reflection on this macOS box.
//
// Members are assigned sequential set-0 bindings in declaration order (Slang does
// not compact unused leading members away — verified in M4), so the order here
// must stay textures(0) / samp(1) / cubes(2) / storage_images(3) /
// storage_buffers(4) to match the rhi-vulkan / rhi-d3d12 descriptor layout.

// Counts must match BINDLESS_COUNT / CUBE_COUNT / STORAGE_IMAGE_COUNT /
// STORAGE_BUFFER_COUNT in rhi-vulkan / rhi-d3d12 (`device.rs`). Each array has its
// own 0-based index space (its binding), separate from the others.
struct Bindless
{
    Texture2D           textures[1024];      // binding 0 — sampled images
    SamplerState        samp;                // binding 1 — shared sampler
    TextureCube         cubes[64];           // binding 2 — sampled cubemaps
    RWTexture2D<float4> storage_images[64];  // binding 3 — storage images (UAV)
    RWByteAddressBuffer storage_buffers[2048]; // binding 4 — storage buffers (UAV)
    // binding 5 — scene TLAS for hardware ray tracing (Phase 8). Present only on
    // RT-capable devices; shaders that don't trace drop it (Slang omits the unused
    // binding), so non-RT pipelines/layouts are unaffected.
    RaytracingAccelerationStructure tlas;
    // binding 6 — sampled 3D volume textures (Phase 11 Stage B: baked distance
    // fields, trilinear-sampled by the SW ray marcher). SRVs, after the TLAS.
    Texture3D<float> volumes[64];
    // binding 7 — read-write 3D volume textures (UAV) the SDF bake compute writes.
    RWTexture3D<float> storage_volumes[64];
    // binding 8 — REPEAT/wrap sampler for tiling material textures (floor tiles, brick
    // courses) whose glTF UVs exceed [0,1]. Declared LAST so it only appends a binding
    // (Vulkan binding 8 / D3D12 s1,space1 / Metal last arg-buffer slot) — existing layout
    // unchanged. `samp` (CLAMP) stays for cubes (seamless) and volumes (no SDF edge wrap).
    SamplerState samp_wrap;
};

[[vk::binding(0, 0)]]
ParameterBlock<Bindless> g;

// Linear → sRGB OETF (gamma encode). The swapchain/backbuffer is a UNORM surface,
// not sRGB: GPU capture & overlay layers (RenderDoc, OBS, Steam, …) force
// VK_IMAGE_USAGE_STORAGE onto swapchain images so they can read them back, and an
// sRGB surface format can never support storage — so an sRGB backbuffer makes
// those layers fail (RenderDoc hard-crashes at vkCreateSwapchainKHR). With a UNORM
// surface the hardware no longer encodes on write, so every pass that writes the
// backbuffer applies this transfer function explicitly. ImGui is the exception:
// its vertex colors are authored in gamma space and pass straight through, which
// is correct on a UNORM surface (and was subtly too bright on the old sRGB one).
float3 linear_to_srgb(float3 c)
{
    float3 lo = c * 12.92;
    float3 hi = 1.055 * pow(max(c, float3(0.0)), 1.0 / 2.4) - 0.055;
    return lerp(lo, hi, step(float3(0.0031308), c));
}

// NaN/Inf guard (레퍼런스 엔진 SW GI `MakeFinite`): an IEEE float is non-finite iff its exponent
// field is all ones. Zero out non-finite components so one bad sample can't poison a
// temporal-accumulation buffer — an EMA blend would otherwise spread the NaN every frame
// (the "black creeps in from frame 2" failure). The asuint bit test survives the optimizer
// (native isnan/isinf often get folded away). Finite values pass through unchanged, so a
// scene that never produces a NaN (the gallery) is byte-identical.
bool3 is_finite3(float3 v)
{
    return (asuint(v) & 0x7F800000u) != 0x7F800000u;
}
float3 make_finite(float3 v)
{
    return select(is_finite3(v), v, float3(0.0, 0.0, 0.0));
}

// Normalize that never returns NaN: a zero-length vector (e.g. a flat SDF gradient in an
// empty region → `normalize(0)`) yields `fallback` instead of NaN. For a non-degenerate
// vector this is bit-identical to `normalize(v)`.
float3 safe_normalize(float3 v, float3 fallback)
{
    return dot(v, v) > 1e-20 ? normalize(v) : fallback;
}
