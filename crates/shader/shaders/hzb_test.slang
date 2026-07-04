// Shared conservative Hi-Z (HZB) occlusion test — single source of truth for the
// "is this bounding sphere fully behind every occluder over its screen footprint" query.
//
// This is a verbatim port of the P7 grid cull test in `cull.slang::csCullHzb` (which passes
// the Sponza-atrium conservatism gate, `docs/hzb-occlusion-culling.md` §4). It is factored
// here so the Phase-14 vgeo two-phase occlusion cut (`vgeo_scene_cut.slang`) reuses the SAME
// byte-verified math instead of rolling its own — the M4b divergence (D3D12 over-culled
// silhouette clusters, reverted `26cbab0`) came from a self-rolled cut-side test, so sharing
// the proven one is what keeps the class from recurring.
//
// Requires `bindless.slang` (the `g` globals) to be #included by the caller first. The HZB is a
// max-reduced depth pyramid (standard-Z near=0 far=1; empty/background = 1.0 far so it never
// occludes) whose mips live in CONSECUTIVE bindless sampled slots (`hzb_base + mip`).
//
// Conservative by construction: the caller passes a bounding SPHERE; we test its world-space
// AABB (⊇ sphere), any ambiguity (behind near plane, off-screen edge, co-planar) keeps it
// visible, and the 4-tap corner-texel max at the AABB-covering mip bounds the whole footprint.

// Project a world point through a column-major view_proj (`vp[i]` = column i) to screen UV
// (0..1) + NDC depth (0..1, standard-Z). Returns false when behind the camera (w <= 0). The
// V flip (`0.5 - 0.5*ndc.y`) is backend-uniform: the cull matrix is the no-Y-flip view_proj on
// every backend (Vulkan's clip-space flip is baked into the scene render, so its depth image
// has the same top-down orientation) — see the cull.slang note.
bool hzbProject(float4 vp[4], float3 wp, out float2 uv, out float ndc_z)
{
    float4 c = vp[0] * wp.x + vp[1] * wp.y + vp[2] * wp.z + vp[3];
    uv = float2(0, 0);
    ndc_z = 0;
    if (c.w <= 1e-6)
        return false;
    float3 ndc = c.xyz / c.w;
    uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    ndc_z = ndc.z;
    return true;
}

// True iff the world bounding sphere (`center`, `radius`) is provably occluded by the HZB.
// `vp` = unjittered no-Y-flip view_proj; `hzb_base`/`levels`/`w`/`h` describe the pyramid.
bool hzbOccluded(float4 vp[4], uint hzb_base, uint hzb_levels, uint hzb_w, uint hzb_h,
                 float3 center, float radius)
{
    // Screen AABB from the 8 world corners (per-axis half-extent = the sphere radius: a
    // conservative over-estimate that never shrinks the footprint). Track the AABB in UV and
    // the NEAREST NDC depth over the corners.
    float2 uv_min = float2(1e9, 1e9);
    float2 uv_max = float2(-1e9, -1e9);
    float z_near = 1e9;
    bool any_front = false;
    [unroll]
    for (int cx = -1; cx <= 1; cx += 2)
        [unroll] for (int cy = -1; cy <= 1; cy += 2)
            [unroll] for (int cz = -1; cz <= 1; cz += 2)
            {
                float3 wp = center + float3(cx, cy, cz) * radius;
                float2 uv;
                float ndc_z;
                if (hzbProject(vp, wp, uv, ndc_z))
                {
                    any_front = true;
                    uv_min = min(uv_min, uv);
                    uv_max = max(uv_max, uv);
                    z_near = min(z_near, ndc_z);
                }
            }
    // Only cull on a fully-projected, on-screen, in-front AABB (never on a partial projection —
    // that is the conservative fallback that guarantees no false cull).
    bool safe = any_front && uv_min.x >= 0.0 && uv_min.y >= 0.0 && uv_max.x <= 1.0 &&
                uv_max.y <= 1.0 && z_near >= 0.0;
    if (!safe)
        return false;

    // Coarsest mip whose texel size covers the AABB extent, then the 4-tap (2x2) test on the
    // AABB's CORNER texels: a texel-sized AABB can still straddle a texel boundary, so a single
    // center tap misses the neighbour texel's depths and can false-cull at a depth discontinuity.
    // If the corner texels span more than 2 per axis (boundary-aligned edge case), go one coarser.
    float2 ext_px = (uv_max - uv_min) * float2(hzb_w, hzb_h);
    float max_px = max(ext_px.x, ext_px.y);
    int mip = int(ceil(log2(max(max_px, 1.0))));
    mip = clamp(mip, 0, int(hzb_levels) - 1);
    uint mw = max(hzb_w >> uint(mip), 1u);
    uint mh = max(hzb_h >> uint(mip), 1u);
    int2 t0 = int2(clamp(int(uv_min.x * float(mw)), 0, int(mw) - 1),
                   clamp(int(uv_min.y * float(mh)), 0, int(mh) - 1));
    int2 t1 = int2(clamp(int(uv_max.x * float(mw)), 0, int(mw) - 1),
                   clamp(int(uv_max.y * float(mh)), 0, int(mh) - 1));
    if ((t1.x - t0.x > 1 || t1.y - t0.y > 1) && mip + 1 < int(hzb_levels))
    {
        mip += 1;
        mw = max(hzb_w >> uint(mip), 1u);
        mh = max(hzb_h >> uint(mip), 1u);
        t0 = int2(clamp(int(uv_min.x * float(mw)), 0, int(mw) - 1),
                  clamp(int(uv_min.y * float(mh)), 0, int(mh) - 1));
        t1 = int2(clamp(int(uv_max.x * float(mw)), 0, int(mw) - 1),
                  clamp(int(uv_max.y * float(mh)), 0, int(mh) - 1));
    }
    float hzb_max = g.textures[hzb_base + uint(mip)].Load(int3(t0, 0)).r;
    hzb_max = max(hzb_max, g.textures[hzb_base + uint(mip)].Load(int3(t1.x, t0.y, 0)).r);
    hzb_max = max(hzb_max, g.textures[hzb_base + uint(mip)].Load(int3(t0.x, t1.y, 0)).r);
    hzb_max = max(hzb_max, g.textures[hzb_base + uint(mip)].Load(int3(t1, 0)).r);
    // Occluded iff the nearest point is farther than every occluder in the region. Epsilon keeps
    // a co-planar cluster visible (no z-fight false cull).
    return z_near > hzb_max + 1e-5;
}
