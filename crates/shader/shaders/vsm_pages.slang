// VSM page management (plan V2: persistent page cache + clipmap scrolling; V1 history in
// git). All state lives in bindless storage buffers; pass wiring in apps/sandbox/src/vsm.rs.
//
// The UE-shaped frame protocol (docs/research/vsm_cache.txt): the page TABLE is rebuilt
// every frame FROM the physical-page metadata — physical pages persist, their virtual
// address slides by the host-computed per-level scroll offset, and only pages that are
// new or invalidated get the RENDER flag (the depth raster's PS writes nowhere else, so
// a static scene's steady state re-renders zero texels).
//
//   csClear  — table/requests/per-frame stats to zero (metadata + pool CONTENT persist).
//   csMark   — receiver pages from the G-buffer world position -> request flags.
//   csUpdate — one thread per PHYSICAL page: scroll its virtual address, free it when it
//              scrolled out / its level was invalidated / it aged out; survivors rewrite
//              their PT entry (MAPPED, no RENDER = cached).
//   csAlloc  — one thread per PT entry: requested-but-unmapped pops the free list (then
//              virgin pages), clears the page texels to far, writes metadata + a
//              MAPPED|RENDER entry.
//
// counter buffer: [0] pages given RENDER this frame (stat), [1] overflowed requests,
// [2] spare, [3] virgin high-water (never-yet-used pool pages; PERSISTS).
// freelist buffer: [0] count, then page indices (PERSISTS across frames).

#include "bindless.slang"
#include "vsm_common.slang"

struct PushConstants
{
    uint consts_buf;   // per-frame constants (matrices/params/misc/origin/scroll)
    uint table_buf;    // page table: VSM_LEVELS * 64*64 u32 entries
    uint request_buf;  // request flags, same shape as the table
    uint pool_buf;     // physical pool texels (u32 depth bits)

    uint counter_buf;  // stats + virgin high-water (see header)
    uint meta_buf;     // physical-page metadata (VSM_META_STRIDE per page)
    uint freelist_buf; // [0] count, then free physical page indices
    uint position_index; // G-buffer world-position texture (marking only)

    uint screen_w; // G-buffer extent (marking only)
    uint screen_h;
    uint2 _pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

// --- Clear: page table + request flags + per-frame stats --------------------------------
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
        g.storage_buffers[pc.counter_buf].Store(0u, 0u); // rendered-page stat
        g.storage_buffers[pc.counter_buf].Store(4u, 0u); // overflow stat
        // [3] virgin high-water and the free list PERSIST — they are the cache.
    }
}

// --- Mark: one thread per screen pixel ---------------------------------------------------
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
    float4 clip = mul(vsm_load_world_to_clip(consts, level), float4(pos.xyz, 1.0));
    float2 texel = vsm_ndc_to_texel(clip.xy);
    if (min(texel.x, texel.y) < 0.0 || max(texel.x, texel.y) >= float(VSM_VIRTUAL_SIZE))
        return;
    uint2 page = uint2(texel) / VSM_PAGE_SIZE;
    g.storage_buffers[pc.request_buf].InterlockedOr(vsm_pte_offset(level, page), VSM_REQ_ALLOCATE);
}

// --- Update: one thread per PHYSICAL page (the cache heart) ------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void csUpdate(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= VSM_POOL_PAGES)
        return;
    RWByteAddressBuffer meta = g.storage_buffers[pc.meta_buf];
    uint mo = tid.x * VSM_META_STRIDE;
    if ((meta.Load(mo + 8u) & VSM_META_VALID) == 0u)
        return;

    RWByteAddressBuffer consts = g.storage_buffers[pc.consts_buf];
    uint level;
    uint2 page;
    vsm_meta_unpack_addr(meta.Load(mo), level, page);
    int3 scroll = vsm_load_scroll(consts, level);
    uint2 misc_zw = vsm_load_misc_zw(consts); // z = frame_no, w = max_age
    int2 np = int2(page) - scroll.xy;         // the level slid; its content stays put

    bool drop = (uint(scroll.z) & VSM_LEVEL_INVALIDATE) != 0u || np.x < 0 || np.y < 0
             || np.x >= int(VSM_TABLE_DIM) || np.y >= int(VSM_TABLE_DIM);
    uint last = meta.Load(mo + 4u);
    uint requested = 0u;
    if (!drop)
    {
        requested =
            g.storage_buffers[pc.request_buf].Load(vsm_pte_offset(level, uint2(np)));
        if (requested != 0u)
            last = misc_zw.x;
        else if (misc_zw.x - last > misc_zw.y)
            drop = true; // aged out
    }
    if (drop)
    {
        meta.Store(mo + 8u, 0u);
        RWByteAddressBuffer freelist = g.storage_buffers[pc.freelist_buf];
        uint slot;
        freelist.InterlockedAdd(0u, 1u, slot);
        freelist.Store(4u + slot * 4u, tid.x);
        return;
    }
    meta.Store(mo + 0u, vsm_meta_pack_addr(level, uint2(np)));
    meta.Store(mo + 4u, last);
    // Cached: mapped for sampling, NOT flagged for rendering — the raster PS discards
    // here, which is the whole point of V2.
    g.storage_buffers[pc.table_buf].Store(vsm_pte_offset(level, uint2(np)),
                                          VSM_PTE_MAPPED | tid.x);
}

// --- Allocate: one thread per page-table entry -------------------------------------------
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
    if ((g.storage_buffers[pc.table_buf].Load(off) & VSM_PTE_MAPPED) != 0u)
        return; // cached page survived csUpdate — nothing to do

    RWByteAddressBuffer counter = g.storage_buffers[pc.counter_buf];
    RWByteAddressBuffer freelist = g.storage_buffers[pc.freelist_buf];
    // Pop the free list first (recycled pages), then virgin pages, else overflow. The
    // count is decremented optimistically and can dip negative under contention; a
    // non-positive PREVIOUS value means empty, and the underflow is undone. Frees
    // (csUpdate) and pops (here) are in separate barrier-ordered dispatches, so pushes
    // never race pops.
    uint page = 0xffffffffu;
    uint prev;
    freelist.InterlockedAdd(0u, 0xffffffffu, prev); // -= 1
    if (int(prev) > 0)
    {
        page = freelist.Load(4u + (prev - 1u) * 4u);
    }
    else
    {
        freelist.InterlockedAdd(0u, 1u, prev); // undo the underflow
        uint virgin;
        counter.InterlockedAdd(12u, 1u, virgin);
        if (virgin < VSM_POOL_PAGES)
        {
            page = virgin;
        }
        else
        {
            counter.InterlockedAdd(12u, 0xffffffffu, virgin); // pin the high-water at cap
            uint _o;
            counter.InterlockedAdd(4u, 1u, _o); // overflow stat
            return;
        }
    }

    // New content: clear the page to far before any caster writes land.
    RWByteAddressBuffer pool = g.storage_buffers[pc.pool_buf];
    uint base = page * VSM_PAGE_SIZE * VSM_PAGE_SIZE * 4u;
    for (uint t = 0u; t < VSM_PAGE_SIZE * VSM_PAGE_SIZE; ++t)
        pool.Store(base + t * 4u, 0x3f800000u); // 1.0f = far

    uint level = tid.x / VSM_PAGES_PER_LEVEL;
    uint rem = tid.x % VSM_PAGES_PER_LEVEL;
    uint2 vpage = uint2(rem % VSM_TABLE_DIM, rem / VSM_TABLE_DIM);
    RWByteAddressBuffer meta = g.storage_buffers[pc.meta_buf];
    uint mo = page * VSM_META_STRIDE;
    RWByteAddressBuffer consts = g.storage_buffers[pc.consts_buf];
    meta.Store(mo + 0u, vsm_meta_pack_addr(level, vpage));
    meta.Store(mo + 4u, vsm_load_misc_zw(consts).x);
    meta.Store(mo + 8u, VSM_META_VALID);
    g.storage_buffers[pc.table_buf].Store(off, VSM_PTE_MAPPED | VSM_PTE_RENDER | page);
    uint _r;
    counter.InterlockedAdd(0u, 1u, _r); // rendered-page stat
}
