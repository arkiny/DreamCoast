// VSM page-management compute (plan V1): clear, mark from the G-buffer, allocate.
// All state lives in bindless storage buffers; pass wiring in apps/sandbox/src/vsm.rs.

#include "bindless.slang"
#include "vsm_common.slang"

struct PushConstants
{
    uint consts_buf;   // per-frame constants (matrices/params/misc)
    uint table_buf;    // page table: VSM_LEVELS * 64*64 u32 entries
    uint request_buf;  // request flags, same shape as the table
    uint pool_buf;     // physical pool texels (u32 depth bits)

    uint counter_buf;  // [0] = next free physical page, [1] = overflow count (stats)
    uint position_index; // G-buffer world-position texture (marking only)
    uint screen_w;       // G-buffer extent (marking only)
    uint screen_h;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

// --- Clear: page table + request flags + the allocator counter -------------------------
// One thread per page-table entry (thread 0 also resets the counters). V1 has no caching,
// so every frame starts from an empty table; the pool needs no clear — freshly allocated
// pages are cleared by csAlloc below before any raster touches them.
[shader("compute")]
[numthreads(64, 1, 1)]
void csClear(uint3 tid: SV_DispatchThreadID)
{
    uint entries = VSM_LEVELS * VSM_PAGES_PER_LEVEL;
    if (tid.x >= entries)
        return;
    g.storage_buffers[pc.table_buf].Store(tid.x * 4u, 0u);
    g.storage_buffers[pc.request_buf].Store(tid.x * 4u, 0u);
    if (tid.x == 0u)
    {
        g.storage_buffers[pc.counter_buf].Store(0u, 0u);
        g.storage_buffers[pc.counter_buf].Store(4u, 0u);
    }
}

// --- Mark: one thread per screen pixel ---------------------------------------------------
// Reads the world-position G-buffer (w = 0 -> background), picks the receiver's clipmap
// level analytically and ORs an allocation request for the page it projects into.
// InterlockedOr rather than a plain store of an idempotent value: same effect, and clean
// under both APIs' validation (docs/research/vsm_arch.txt, marking notes).
[shader("compute")]
[numthreads(8, 8, 1)]
void csMark(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.screen_w || tid.y >= pc.screen_h)
        return;
    float4 pos = g.textures[pc.position_index].Load(int3(tid.xy, 0));
    if (pos.w < 0.5)
        return; // sky / no geometry
    RWByteAddressBuffer consts = g.storage_buffers[pc.consts_buf];
    uint level = vsm_level_for_distance(length(pos.xyz - vsm_load_origin(consts)));
    // The level chosen by distance always contains the receiver (radius covers it), so a
    // single level's request suffices; the sampler walks coarser only on pool overflow.
    float4 clip = mul(vsm_load_world_to_clip(consts, level), float4(pos.xyz, 1.0));
    float2 texel = vsm_ndc_to_texel(clip.xy);
    if (min(texel.x, texel.y) < 0.0 || max(texel.x, texel.y) >= float(VSM_VIRTUAL_SIZE))
        return;
    uint2 page = uint2(texel) / VSM_PAGE_SIZE;
    g.storage_buffers[pc.request_buf].InterlockedOr(vsm_pte_offset(level, page), VSM_REQ_ALLOCATE);
}

// --- Allocate: one thread per page-table entry -------------------------------------------
// Requested entries pop a physical page from the linear counter and clear its 128² texels
// to "far" (asuint(1.0)) before writing the table entry. The pop order is GPU-scheduling
// dependent, but the virtual->content mapping is not: every virtual page renders and
// samples its own content wherever it landed, so images stay deterministic and DX≡VK.
// Pool exhaustion increments the overflow counter and leaves the entry unmapped — the
// sampler falls back to a coarser level (and a coarser level's page may itself be absent,
// which shades fully lit; the stat is the loud signal to grow the pool).
[shader("compute")]
[numthreads(64, 1, 1)]
void csAlloc(uint3 tid: SV_DispatchThreadID)
{
    uint entries = VSM_LEVELS * VSM_PAGES_PER_LEVEL;
    if (tid.x >= entries)
        return;
    uint off = tid.x * 4u;
    if (g.storage_buffers[pc.request_buf].Load(off) == 0u)
        return;
    uint page;
    g.storage_buffers[pc.counter_buf].InterlockedAdd(0u, 1u, page);
    if (page >= VSM_POOL_PAGES)
    {
        uint _unused;
        g.storage_buffers[pc.counter_buf].InterlockedAdd(4u, 1u, _unused);
        return;
    }
    RWByteAddressBuffer pool = g.storage_buffers[pc.pool_buf];
    uint base = page * VSM_PAGE_SIZE * VSM_PAGE_SIZE * 4u;
    for (uint t = 0u; t < VSM_PAGE_SIZE * VSM_PAGE_SIZE; ++t)
        pool.Store(base + t * 4u, 0x3f800000u); // 1.0f = far
    g.storage_buffers[pc.table_buf].Store(off, VSM_PTE_MAPPED | VSM_PTE_RENDER | page);
}
