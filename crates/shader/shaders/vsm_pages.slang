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
// [2] pages freed this frame (stat), [3] free-list pop cursor (per frame).
// freelist buffer: [0] count, then free page indices - REBUILT every frame from the
// metadata by csCompact (the UE PackAvailablePages shape). A persistent mutable stack
// popped with a decrement-and-undo pattern corrupts under contention (phantom pops
// double-assign physical pages - the V3 bring-up collapse); rebuilding from truth each
// frame plus a monotonic pop cursor is race-free by construction.

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
        g.storage_buffers[pc.counter_buf].Store(0u, 0u);  // rendered-page stat
        g.storage_buffers[pc.counter_buf].Store(4u, 0u);  // overflow stat
        g.storage_buffers[pc.counter_buf].Store(8u, 0u);  // freed stat
        g.storage_buffers[pc.counter_buf].Store(12u, 0u); // free-list pop cursor
        g.storage_buffers[pc.freelist_buf].Store(0u, 0u); // rebuilt by csCompact below
    }
    // Per-level RENDER-page bounds (VSM_BOUNDS_BASE rows): reset to the empty
    // sentinel (min > max). csUpdate/csAlloc Min/Max every page they flag RENDER;
    // the depth VS culls whole casters against the result — same frame, no
    // stale-readback shadow holes.
    if (tid.x < VSM_LEVELS)
    {
        uint b = VSM_BOUNDS_BASE + tid.x * 16u;
        g.storage_buffers[pc.counter_buf].Store(b + 0u, 0xFFFFFFFFu); // min x
        g.storage_buffers[pc.counter_buf].Store(b + 4u, 0xFFFFFFFFu); // min y
        g.storage_buffers[pc.counter_buf].Store(b + 8u, 0u);          // max x
        g.storage_buffers[pc.counter_buf].Store(b + 12u, 0u);         // max y
    }
}

// A page just flagged RENDER on `level` grows that level's caster-cull bounds.
void vsm_bounds_grow(uint level, uint2 vpage)
{
    uint b = VSM_BOUNDS_BASE + level * 16u;
    uint _o;
    g.storage_buffers[pc.counter_buf].InterlockedMin(b + 0u, vpage.x, _o);
    g.storage_buffers[pc.counter_buf].InterlockedMin(b + 4u, vpage.y, _o);
    g.storage_buffers[pc.counter_buf].InterlockedMax(b + 8u, vpage.x, _o);
    g.storage_buffers[pc.counter_buf].InterlockedMax(b + 12u, vpage.y, _o);
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
    float d = length(pos.xyz - vsm_load_origin(consts));
    uint level = vsm_level_for_distance(d);
    // Inside the blend band the sampler dithers between this level and the next
    // coarser one, so request BOTH (the shared-fn contract: a sampler asking a level
    // marking never requested reads unmapped and walks coarser).
    uint last = vsm_level_in_blend_band(d) ? min(level + 1u, VSM_LEVELS - 1u) : level;
    for (uint l = level; l <= last; ++l)
    {
        float4 clip = mul(vsm_load_world_to_clip(consts, l), float4(pos.xyz, 1.0));
        float2 texel = vsm_ndc_to_texel(clip.xy);
        if (min(texel.x, texel.y) < 0.0 || max(texel.x, texel.y) >= float(VSM_VIRTUAL_SIZE))
            continue;
        uint2 page = uint2(texel) / VSM_PAGE_SIZE;
        g.storage_buffers[pc.request_buf].InterlockedOr(vsm_pte_offset(l, page),
                                                        VSM_REQ_ALLOCATE);
    }
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
    uint2 page; // (R1) ABSOLUTE page coords, wrapped to the address field's 14 bits
    vsm_meta_unpack_addr(meta.Load(mo), level, page);
    int4 origin = vsm_load_scroll(consts, level); // xy = absolute origin, w = epoch
    uint2 misc_zw = vsm_load_misc_zw(consts);     // z = frame_no, w = max_age
    // Window-relative position re-derived FROM SCRATCH each frame (modular difference
    // in the 14-bit space): there is no incremental delta chain a skipped or aborted
    // frame could desync, and a whole-table jump just fails the window test below.
    //
    // AXIS HANDEDNESS: texel space inverts Y against light-space up (`vsm_ndc_to_texel`
    // puts row 0 at ndc.y = +1), so when the window's centre moves +1 page along light
    // up, a cached page's table ROW moves +1 (its column moves -1 for +right). The
    // original V2 slide used the X sign for both axes — invisible while the pre-merge
    // invalidation cap kept the cache dead, and the actual corruption behind the
    // 'cached shadows land on the wrong geometry' regressions when it first woke up.
    uint2 rel = uint2((page.x - uint(origin.x)) & 0x3fffu, (page.y + uint(origin.y)) & 0x3fffu);
    int2 np = int2(rel);

    // Epoch mismatch = the level was invalidated some frame after this page rendered
    // — evaluated EVERY frame, so a swallowed invalidation frame changes nothing (the
    // absolute-state principle, same as the addressing above).
    bool drop = meta.Load(mo + 12u) != uint(origin.w) || rel.x >= VSM_TABLE_DIM
             || rel.y >= VSM_TABLE_DIM;
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
        meta.Store(mo + 8u, 0u); // csCompact folds it into the free list below
        uint _f;
        g.storage_buffers[pc.counter_buf].InterlockedAdd(8u, 1u, _f); // freed stat
        return;
    }
    // (R1) The stored address is absolute — nothing to rewrite as the window slides.
    meta.Store(mo + 4u, last);

    // Dynamic-object invalidation (V3, the track's core): a cached page overlapped by any
    // mover sphere (old OR new footprint) clears to far and re-renders this frame. The
    // sphere is projected into the level's texel space (ortho rows are scaled-orthogonal:
    // |row 0| = 1/half, so texels-per-metre = VIRTUAL·|row0|/2) and tested against the
    // page's texel rect — conservative in Z (a sphere behind the wall still invalidates;
    // correctness over precision, the page just re-renders).
    bool rerender = false;
    uint inval_count = min(vsm_load_inval_count(consts), VSM_MAX_INVAL);
    if (inval_count != 0u)
    {
        float4x4 wc = vsm_load_world_to_clip(consts, level);
        float texel_per_m = float(VSM_VIRTUAL_SIZE) * length(wc[0].xyz) * 0.5;
        float2 pmin = float2(np * int2(VSM_PAGE_SIZE, VSM_PAGE_SIZE));
        float2 pmax = pmin + float(VSM_PAGE_SIZE);
        for (uint i = 0u; i < inval_count && !rerender; ++i)
        {
            float4 s = vsm_load_inval_sphere(consts, i);
            float4 c = mul(wc, float4(s.xyz, 1.0));
            float2 ct = vsm_ndc_to_texel(c.xy);
            float rt = s.w * texel_per_m;
            float2 closest = clamp(ct, pmin, pmax);
            float2 d = closest - ct;
            rerender = dot(d, d) <= rt * rt;
        }
    }
    if (rerender)
    {
        RWByteAddressBuffer pool = g.storage_buffers[pc.pool_buf];
        uint base = tid.x * VSM_PAGE_SIZE * VSM_PAGE_SIZE * 4u;
        for (uint t = 0u; t < VSM_PAGE_SIZE * VSM_PAGE_SIZE; ++t)
            pool.Store(base + t * 4u, 0x3f800000u); // 1.0f = far
        g.storage_buffers[pc.table_buf].Store(vsm_pte_offset(level, uint2(np)),
                                              VSM_PTE_MAPPED | VSM_PTE_RENDER | tid.x);
        uint _r;
        g.storage_buffers[pc.counter_buf].InterlockedAdd(0u, 1u, _r); // rendered-page stat
        vsm_bounds_grow(level, uint2(np));
        return;
    }
    // Cached: mapped for sampling, NOT flagged for rendering — the raster PS discards
    // here, which is the whole point of V2.
    g.storage_buffers[pc.table_buf].Store(vsm_pte_offset(level, uint2(np)),
                                          VSM_PTE_MAPPED | tid.x);
}

// --- Stats: snapshot the frame's counter words into the per-FIF consts diag tail ---------
// One thread, dispatched LAST in the pages pass (after csAlloc): the host harvests the
// tail when this FIF slot cycles back, giving frame-accurate pages/overflow stats — the
// raw counter readback races whatever the in-flight GPU is mid-writing (observed as
// phantom 0/partial "page collapse" blips that polluted a whole investigation).
[shader("compute")]
[numthreads(1, 1, 1)]
void csStats(uint3 tid: SV_DispatchThreadID)
{
    RWByteAddressBuffer counter = g.storage_buffers[pc.counter_buf];
    RWByteAddressBuffer consts = g.storage_buffers[pc.consts_buf];
    consts.Store(VSM_CONST_DIAG_OFF + 4u, counter.Load(0u));
    consts.Store(VSM_CONST_DIAG_OFF + 8u, counter.Load(4u));
}

// --- Compact: rebuild the free list from the metadata (one thread per physical page) ----
[shader("compute")]
[numthreads(64, 1, 1)]
void csCompact(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= VSM_POOL_PAGES)
        return;
    if ((g.storage_buffers[pc.meta_buf].Load(tid.x * VSM_META_STRIDE + 8u) & VSM_META_VALID)
        != 0u)
        return;
    RWByteAddressBuffer freelist = g.storage_buffers[pc.freelist_buf];
    uint slot;
    freelist.InterlockedAdd(0u, 1u, slot);
    freelist.Store(4u + slot * 4u, tid.x);
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
    // Race-free pop: the list was rebuilt from the metadata this frame (csCompact) and
    // its count no longer moves; a monotonic cursor hands each claimer a unique slot.
    uint cursor;
    counter.InterlockedAdd(12u, 1u, cursor);
    if (cursor >= freelist.Load(0u))
    {
        uint _o;
        counter.InterlockedAdd(4u, 1u, _o); // overflow stat (pool exhausted)
        return;
    }
    uint page = freelist.Load(4u + cursor * 4u);

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
    // (R1) Store the ABSOLUTE page address (window position resolved against the
    // level's absolute origin, wrapped): csUpdate re-derives the window position from
    // it every frame. Y folds with the OPPOSITE sign — texel rows run against light-
    // space up (see csUpdate's axis-handedness note).
    int4 origin_a = vsm_load_scroll(consts, level);
    uint2 apage = uint2((vpage.x + uint(origin_a.x)) & 0x3fffu,
                        (vpage.y - uint(origin_a.y)) & 0x3fffu);
    meta.Store(mo + 0u, vsm_meta_pack_addr(level, apage));
    meta.Store(mo + 4u, vsm_load_misc_zw(consts).x);
    meta.Store(mo + 8u, VSM_META_VALID);
    meta.Store(mo + 12u, uint(origin_a.w)); // birth epoch (see csUpdate's drop test)
    g.storage_buffers[pc.table_buf].Store(off, VSM_PTE_MAPPED | VSM_PTE_RENDER | page);
    uint _r;
    counter.InterlockedAdd(0u, 1u, _r); // rendered-page stat
    vsm_bounds_grow(level, vpage);
}
