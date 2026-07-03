// Phase 14 (virtual geometry) M0 — 64-bit atomic-max capability smoke.
//
// The visibility-buffer pillar records `(depth << 32) | payload` into an R64 target
// with a single `atomicMax`, so the highest-depth (nearest, after inversion) writer
// wins per pixel without a lock. This shader exercises exactly that primitive against
// a bindless storage buffer, so `--atomic64-test` can prove the path end-to-end (GPU
// atomic → CPU readback) before any of the real cluster machinery exists.
//
// ── Cross-backend 64-bit atomicMax (the M0 pivot) ────────────────────────────────
// Slang's built-in `InterlockedMax(uint64_t)` lowers to `atomic_fetch_max_explicit`
// on the Metal target, but Metal's stdlib has NO 64-bit overload of that name (it is
// 32-bit int/uint only). Metal spells the 64-bit min/max differently —
// `atomic_max_explicit` (void return, `device ulong*` only), gated by
// `__HAVE_ATOMIC_ULONG_MIN_MAX__` which is defined for `__METAL_VERSION__` >= 300
// (Apple M3). So Metal fully supports it; only Slang's symbol choice is wrong. We emit
// the correct Metal call directly via `__intrinsic_asm` under `RT_METAL_TARGET` (the
// same build.rs-injected Metal gate `rt_common.slang` uses), and keep the portable
// `InterlockedMax64` on SPIR-V/D3D12 (both lower to `OpAtomicUMax` / SM6.6 atomics).
//
// `$0` is the ByteAddressBuffer's device base pointer, `$1` a u64-ELEMENT index (the
// intrinsic does `(atomic_ulong*)base + elem`), `$2` the value. The SPIR-V/D3D12 path
// takes the same `elem` as a u64 index and converts to the BYTE offset it wants.
#include "bindless.slang"

struct PushConstants
{
    uint buf_index; // bindless storage-buffer slot holding the R64 target
    uint count;     // number of participating threads (packed values written)
    uint slots;     // number of u64 slots in the target buffer (write wraps mod slots)
    uint pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

#ifdef RT_METAL_TARGET
void atomicMaxU64(RWByteAddressBuffer buf, uint elem, uint64_t val)
{
    __intrinsic_asm "atomic_max_explicit((device atomic_ulong*)$0 + $1, $2, memory_order_relaxed)";
}
#else
void atomicMaxU64(RWByteAddressBuffer buf, uint elem, uint64_t val)
{
    uint64_t prev;
    buf.InterlockedMax64(elem * 8, val, prev);
}
#endif

// Each thread packs `(hi << 32) | lo` and atomic-maxes it into slot `tid % slots`.
// `hi = tid * 2654435761 (Knuth) & 0xFFFF` scatters the winning value so the CPU-side
// expected max is a non-trivial function of `count` (not just `count-1`), catching a
// silently-dropped high word. Determinism: max is commutative/associative, so the
// result is independent of thread order — bit-identical run to run and across backends.
[shader("compute")]
[numthreads(64, 1, 1)]
void csAtomicMax(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.count)
        return;

    uint hi = (i * 2654435761u) & 0xFFFFu;
    uint lo = i;
    uint64_t packed = (uint64_t(hi) << 32) | uint64_t(lo);
    atomicMaxU64(g.storage_buffers[pc.buf_index], i % pc.slots, packed);
}
