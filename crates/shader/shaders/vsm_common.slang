// Virtual shadow map shared layout (docs/vsm-shadows-plan.md V1, adapted from UE5's
// VirtualShadowMaps — see docs/research/vsm_arch.txt for the reference mechanisms).
//
// One directional sun = VSM_LEVELS independent clipmap levels; level i is an 8k²-virtual
// ortho shadow map covering radius 2^(i+2) metres around the (per-level page-snapped)
// clipmap origin. Each level is 64×64 pages of 128² texels. The page table maps virtual
// pages to pages in one shared physical pool; sampling is a single table fetch (V1 walks
// coarser LEVELS on a miss — the UE LODOffset pointer propagation lands with caching).
//
// ## Orientation contract (the froxel-mirror lesson, made structural)
// Every VSM matrix is FLIP-FREE (D3D clip orientation), on every backend. The depth
// raster's vertex shader applies the Vulkan clip-Y flip itself (`flip_y` push) so the
// HARDWARE viewport lands each caster on the same virtual texel row on every backend —
// after that single flip, marking, allocation, rasterization and sampling all share
// `vsm_ndc_to_texel` below with no per-backend branches anywhere.
//
// ## Depth convention
// Ortho clip z in [0,1], z = 0 at the light. The pool stores `asuint(z)` written with
// `InterlockedMin` (asuint is monotonic for non-negative floats), so the value kept is
// the occluder NEAREST the light; a receiver is shadowed when `stored < receiver - bias`.

#ifndef VSM_COMMON_INCLUDED
#define VSM_COMMON_INCLUDED

// Keep in lockstep with `VSM_*` consts in apps/sandbox/src/vsm.rs (single source pair).
#define VSM_LEVELS 6u
#define VSM_PAGE_SIZE 128u          // texels per page side
#define VSM_TABLE_DIM 64u           // pages per level side (virtual = 64*128 = 8192)
#define VSM_VIRTUAL_SIZE (VSM_PAGE_SIZE * VSM_TABLE_DIM)
#define VSM_PAGES_PER_LEVEL (VSM_TABLE_DIM * VSM_TABLE_DIM)
// Physical pool geometry: pages packed row-major into a buffer, POOL_ROW pages per row.
#define VSM_POOL_PAGES 512u
// Page-table entry: bits [0:15] physical page index, bit 30 = mapped-for-render,
// bit 31 = mapped (valid to sample). Unallocated = 0.
#define VSM_PTE_MAPPED 0x80000000u
#define VSM_PTE_RENDER 0x40000000u
#define VSM_PTE_PAGE_MASK 0xffffu
// Request flags (one u32 per page-table entry).
#define VSM_REQ_ALLOCATE 1u

// Per-frame constants buffer layout (host-written; see `VsmSystem::upload`):
//   [0..VSM_LEVELS)   float4x4 world_to_clip[i]   (flip-free ortho, row-major mul(M,v))
//   then per level     float4  params[i]: x = texel size (world m), y = 1/depth-range (unused
//                      V1 spare), zw = spare
//   then               uint4   misc: x = level count, y = pool pages, z/w = spare
//   then               float4  origin: xyz = clipmap origin (camera focus), w spare
// All offsets in BYTES for ByteAddressBuffer loads.
#define VSM_CONST_MATS_OFF 0u
#define VSM_CONST_PARAMS_OFF (VSM_LEVELS * 64u)
#define VSM_CONST_MISC_OFF (VSM_CONST_PARAMS_OFF + VSM_LEVELS * 16u)
#define VSM_CONST_ORIGIN_OFF (VSM_CONST_MISC_OFF + 16u)
#define VSM_CONST_SIZE (VSM_CONST_ORIGIN_OFF + 16u)

float3 vsm_load_origin(RWByteAddressBuffer consts)
{
    return asfloat(consts.Load3(VSM_CONST_ORIGIN_OFF));
}

float4x4 vsm_load_world_to_clip(RWByteAddressBuffer consts, uint level)
{
    // The host writes glam `to_cols_array` = COLUMN-major floats. `float4x4(a,b,c,d)`
    // builds with the operands as ROWS, so transpose to get `mul(M, v)` row semantics —
    // the raster's push matrix goes through the engine's cbuffer convention and must
    // land every caster on exactly the texels marking/sampling derive from this load
    // (the V1 bring-up bug: without the transpose, marking+sampling agreed with each
    // other but disagreed with the raster, and only ~12% of fragments hit mapped pages).
    uint o = VSM_CONST_MATS_OFF + level * 64u;
    float4 c0 = asfloat(consts.Load4(o + 0u));
    float4 c1 = asfloat(consts.Load4(o + 16u));
    float4 c2 = asfloat(consts.Load4(o + 32u));
    float4 c3 = asfloat(consts.Load4(o + 48u));
    return transpose(float4x4(c0, c1, c2, c3));
}

float4 vsm_load_params(RWByteAddressBuffer consts, uint level)
{
    return asfloat(consts.Load4(VSM_CONST_PARAMS_OFF + level * 16u));
}

// The clipmap level whose radius (2^(i+2) m) covers a receiver at distance `d` from the
// clipmap origin: i = floor(log2(max(d, 2))) - 1, clamped. Shared by marking + sampling
// (the two MUST agree; a sampler asking a level marking never requested reads unmapped
// and walks coarser).
uint vsm_level_for_distance(float d)
{
    float l = floor(log2(max(d, 2.0))) - 1.0;
    return (uint)clamp(l, 0.0, float(VSM_LEVELS - 1u));
}

// Flip-free NDC -> virtual texel coordinates (D3D orientation: ndc.y up = row 0 at top).
// The single source for marking, the raster PS reverse-check and sampling.
float2 vsm_ndc_to_texel(float2 ndc)
{
    return float2((ndc.x * 0.5 + 0.5) * float(VSM_VIRTUAL_SIZE),
                  (0.5 - ndc.y * 0.5) * float(VSM_VIRTUAL_SIZE));
}

// Page-table entry byte offset for (level, page.xy).
uint vsm_pte_offset(uint level, uint2 page)
{
    return (level * VSM_PAGES_PER_LEVEL + page.y * VSM_TABLE_DIM + page.x) * 4u;
}

// Physical pool byte offset of a texel inside a mapped page.
uint vsm_pool_offset(uint physical_page, uint2 texel_in_page)
{
    return (physical_page * VSM_PAGE_SIZE * VSM_PAGE_SIZE
            + texel_in_page.y * VSM_PAGE_SIZE + texel_in_page.x)
        * 4u;
}

#endif // VSM_COMMON_INCLUDED
