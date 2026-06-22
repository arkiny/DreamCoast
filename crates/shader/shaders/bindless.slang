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
// `g_cubes[]` at `space1`. Storage arrays join in M5 when the compute shaders
// migrate.
//
// Adding `cubes` is layout-safe for the M3 shaders: Slang drops the unused cube
// binding from `mesh`/`imgui`/`gbuffer` (their SPIR-V stays set 0 / bindings 0,1),
// so `rhi-vulkan` is untouched. In a globals-using shader (`pbr`) the block still
// pins to **set 0** via `[[vk::binding(0,0)]]` even with a loose globals UBO at
// set 1 — matching the previous loose-global layout exactly (Vulkan unchanged).
// On Metal the loose globals takes `[[buffer(1)]]`, pushing this argument buffer
// to `[[buffer(2)]]` for globals shaders (non-globals shaders keep it at
// `[[buffer(1)]]`); verified via `slangc` reflection on this macOS box.

// Counts must match BINDLESS_COUNT / CUBE_COUNT in rhi-vulkan / rhi-d3d12
// (`device.rs`). Cubes share the descriptor set but have their own 0-based index
// space (binding 2), separate from `textures[]` (binding 0).
struct Bindless
{
    Texture2D    textures[1024];
    SamplerState samp;
    TextureCube  cubes[64];
};

[[vk::binding(0, 0)]]
ParameterBlock<Bindless> g;
