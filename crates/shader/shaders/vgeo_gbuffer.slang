// Phase 14 renderer integration — visibility buffer → Phase-6 G-buffer MRT.
//
// This is the M6 material resolve wired for the REAL deferred renderer: instead of shading the
// resolved attributes itself (the self-contained viewer's `vgeo_resolve.slang`), it writes the four
// deferred G-buffer render targets — albedo / world-normal / material / world-position — plus
// SV_Depth, in the exact format the existing `gbuffer.slang` fill produces. The downstream deferred
// pipeline (shadow / GDF / PBR lighting / tonemap) then consumes it UNCHANGED, so virtual geometry
// becomes a drop-in G-buffer producer rather than a debug view.
//
// SCENE resolve (unified per-frame pass): one full-screen resolve for the whole scene. The per-pixel
// payload stores a WORK INDEX `t`; `work[t] = (instance, global cluster)` recovers the instance, and
// the per-object transform + material come from the instance table (not push constants) — the Nanite
// `VisibleCluster.InstanceId → GetInstanceSceneData` indirection (docs/phase-14-vgeo-unified-pass.md).
// The material logic below mirrors `gbuffer.slang`'s `fsMain` byte-for-byte so a resolved surface
// shades identically to rasterizing the same triangles directly.
#include "vgeo_scene_common.slang"

static const uint NO_TEXTURE = 0xFFFFFFFF;

struct PushConstants
{
    uint vtx_buf;    //  0: vertex pool (32 B/vertex: pos[3], normal[3], uv[2])
    uint remap_buf;  //  4: cluster_vertices (u32)
    uint tri_buf;    //  8: cluster_triangles (u32, 3/tri)
    uint rec_buf;    // 12: GpuCluster records (96 B)
    uint vis_buf;    // 16: R64 visibility buffer (u64/pixel)
    uint work_buf;   // 20: work list (uint2: instance, global cluster)
    uint inst_buf;   // 24: instance table (SceneInstance, 256 B)
    uint width;      // 28
    uint height;     // 32
    float mip_bias;  // 36: per-frame global texture LOD bias (folds into the SampleGrad gradient)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

struct GBufferOut
{
    float4 albedo : SV_Target0;   // rgb base color (linear), a = AO
    float4 normal : SV_Target1;   // xyz world-space normal
    float4 material : SV_Target2; // r metallic, g roughness, b AO
    float4 position : SV_Target3; // xyz world position
    float depth : SV_Depth;       // window-space depth (matches the raster the resolve reads)
};

struct VSOut
{
    float4 pos : SV_Position;
};

// Full-screen triangle from the vertex id — no vertex buffer.
[shader("vertex")]
VSOut vsMain(uint vid: SV_VertexID)
{
    VSOut o;
    float2 uv = float2((vid << 1) & 2, vid & 2);
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// Screen-space TBN from world-position + uv derivatives. Same math as `gbuffer.slang`, but the
// derivatives are passed in ANALYTICALLY (not `ddx`/`ddy`): in a full-screen visibility resolve the
// 2×2 quad straddles different triangles, so hardware derivatives are garbage at triangle edges.
float3 apply_normal_map(float3 n, float3 dp1, float3 dp2, float2 duv1, float2 duv2,
                        float3 tangent_normal)
{
    float3 dp2perp = cross(dp2, n);
    float3 dp1perp = cross(n, dp1);
    float3 t = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 b = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = rsqrt(max(dot(t, t), dot(b, b)));
    float3x3 tbn = float3x3(t * invmax, b * invmax, n);
    return normalize(mul(tangent_normal, tbn));
}

[shader("fragment")]
GBufferOut fsGBuffer(VSOut i)
{
    uint x = (uint)i.pos.x;
    uint y = (uint)i.pos.y;
    GBufferOut o;
    if (x >= pc.width || y >= pc.height)
        discard;
    uint px = y * pc.width + x;
    uint64_t packed = g.storage_buffers[pc.vis_buf].Load<uint64_t>(px * 8);
    if (packed == 0)
        discard; // no geometry → leave the cleared G-buffer (position.a == 0)

    uint payload = (uint)(packed & 0xFFFFFFFFu);
    uint t = payload >> 7;          // WORK INDEX (not a raw cluster)
    uint tri = payload & 0x7Fu;

    // Recover the instance + global cluster from the work list, then its transform + material.
    uint2 w = loadWork(pc.work_buf, t);
    InstanceData inst = loadInstance(pc.inst_buf, w.x);

    uint4 rec = g.storage_buffers[pc.rec_buf].Load4(w.y * 96);
    uint voff = rec.x, toff = rec.z;

    uint3 li = g.storage_buffers[pc.tri_buf].Load3((toff + tri * 3) * 4);
    float4 clip[3];
    float3 nrm[3];
    float2 uvs[3];
    float3 wpos[3];
    for (int k = 0; k < 3; ++k)
    {
        uint remap = g.storage_buffers[pc.remap_buf].Load((voff + li[k]) * 4);
        uint b = remap * 32;
        float3 p = asfloat(g.storage_buffers[pc.vtx_buf].Load3(b));
        float3 nl = asfloat(g.storage_buffers[pc.vtx_buf].Load3(b + 12));
        uvs[k] = asfloat(g.storage_buffers[pc.vtx_buf].Load2(b + 24));
        wpos[k] = mul(inst.model, float4(p, 1.0)).xyz;
        nrm[k] = mul((float3x3)inst.model, nl);
        clip[k] = mul(inst.mvp, float4(p, 1.0)); // keep HOMOGENEOUS — do NOT divide by w here
    }

    // Perspective-correct barycentric via a HOMOGENEOUS (pre-divide) solve. A per-vertex `xy/w`
    // projection is invalid when a vertex crosses the near plane (clip.w <= 0) — the `max(w,eps)`
    // clamp then places the vertex at a garbage screen position, warping the barycentric into an
    // affine-looking UV streak on large grazing triangles (corridor walls/vaults). Instead solve for
    // the true barycentric `lam` (sum 1) such that the interpolated clip point projects to this
    // pixel's NDC (u,v):  lam·clip.x = u·(lam·clip.w),  lam·clip.y = v·(lam·clip.w),  sum(lam)=1.
    // The first two rows are the vectors A=(clip.x - u·clip.w) and B=(clip.y - v·clip.w); `lam` is
    // orthogonal to both (their cross product), renormalized to sum 1. No division by any per-vertex
    // w, so w<=0 vertices are handled exactly like the HW rasterizer. `lam` is the TRUE 3D
    // barycentric, so world pos / normal / uv interpolate linearly (Schied & Dachsbacher, "Deferred
    // Attribute Interpolation", HPG 2015 — the same pre-divide edge math Nanite uses).
    float u = 2.0 * (float(x) + 0.5) / float(pc.width) - 1.0;
    float v = 1.0 - 2.0 * (float(y) + 0.5) / float(pc.height);
    float3 W = float3(clip[0].w, clip[1].w, clip[2].w);
    float3 A = float3(clip[0].x, clip[1].x, clip[2].x) - u * W;
    float3 B = float3(clip[0].y, clip[1].y, clip[2].y) - v * W;
    float3 L = cross(A, B);
    float s = L.x + L.y + L.z;
    float rs = (abs(s) < 1e-20) ? 0.0 : 1.0 / s;
    float3 lam = L * rs;

    // Analytic screen-space derivatives of the barycentric (∂lam/∂pixel), for correct texture mip
    // gradients and TBN in the full-screen resolve — hardware ddx/ddy would sample across triangle
    // boundaries. dA/du = -W, dB/dv = -W → dL/du = -cross(W,B), dL/dv = cross(W,A); then
    // ∂lam/∂u = (dL/du - lam·sum(dL/du))·rs, chained by ∂u/∂px = 2/width, ∂v/∂py = -2/height.
    float3 Lu = -cross(W, B);
    float3 Lv = cross(W, A);
    float3 dlam_du = (Lu - lam * (Lu.x + Lu.y + Lu.z)) * rs;
    float3 dlam_dv = (Lv - lam * (Lv.x + Lv.y + Lv.z)) * rs;
    float dpx = 2.0 / float(pc.width);
    float dpy = -2.0 / float(pc.height);

    float3 world_pos = lam.x * wpos[0] + lam.y * wpos[1] + lam.z * wpos[2];
    float3 world_normal = lam.x * nrm[0] + lam.y * nrm[1] + lam.z * nrm[2];
    float2 uv = lam.x * uvs[0] + lam.y * uvs[1] + lam.z * uvs[2];

    // EXACT window-space depth for SV_Depth: the interpolated clip point P = Σ lam·clip projects to
    // z = P.z/P.w — full float32 precision, identical to what the HW rasterizer writes for the mesh
    // fill. (Reconstructing from the packed 32-bit key instead loses ~8 bits through `float(uint>2^24)`
    // and rounds differently than the mesh path → the two fight in the depth composite, letting the
    // wall behind a grazing arch soffit bleed through as streaks.) This is view-independent of the
    // Y-flip (z row is unaffected), so it composites correctly against the mesh-fill depth on DX≡VK.
    float pz = lam.x * clip[0].z + lam.y * clip[1].z + lam.z * clip[2].z;
    float pw = lam.x * clip[0].w + lam.y * clip[1].w + lam.z * clip[2].w;
    float exact_depth = pz / pw;

    // Analytic UV / world-position screen derivatives (fold `mip_bias` into the gradient scale, since
    // SampleGrad takes gradients not a bias: LOD += bias ⇔ grad *= exp2(bias)).
    float grad_scale = exp2(pc.mip_bias);
    float2 uv_dx = (dlam_du.x * uvs[0] + dlam_du.y * uvs[1] + dlam_du.z * uvs[2]) * dpx * grad_scale;
    float2 uv_dy = (dlam_dv.x * uvs[0] + dlam_dv.y * uvs[1] + dlam_dv.z * uvs[2]) * dpy * grad_scale;
    float3 wp_dx = (dlam_du.x * wpos[0] + dlam_du.y * wpos[1] + dlam_du.z * wpos[2]) * dpx;
    float3 wp_dy = (dlam_dv.x * wpos[0] + dlam_dv.y * wpos[1] + dlam_dv.z * wpos[2]) * dpy;

    // ── Material resolve — mirrors gbuffer.slang fsMain, but samples with SampleGrad(analytic uv
    // gradients) instead of implicit derivatives (the visibility-resolve mip-selection fix). ──────
    float cutoff = inst.mr.w;
    float3 albedo = inst.base_color.rgb;
    if (inst.tex.x != NO_TEXTURE)
    {
        float4 bc = g.textures[inst.tex.x].SampleGrad(g.samp_wrap, uv, uv_dx, uv_dy);
        albedo *= bc.rgb;
        float a = bc.a * inst.base_color.a;
        if (cutoff > 0.0 && a < cutoff)
            discard;
    }
    float metallic = inst.mr.x;
    float roughness = inst.mr.y;
    if (inst.tex.y != NO_TEXTURE)
    {
        float3 mr = g.textures[inst.tex.y].SampleGrad(g.samp_wrap, uv, uv_dx, uv_dy).rgb;
        roughness *= mr.g;
        metallic *= mr.b;
    }
    float3 n = normalize(world_normal);
    if (inst.tex.z != NO_TEXTURE)
    {
        float3 tn = g.textures[inst.tex.z].SampleGrad(g.samp_wrap, uv, uv_dx, uv_dy).rgb * 2.0 - 1.0;
        n = apply_normal_map(n, wp_dx, wp_dy, uv_dx, uv_dy, tn);
    }

    o.albedo = float4(albedo, 1.0);
    o.normal = float4(n, 0.0);
    o.material = float4(metallic, saturate(roughness), 1.0, 0.0);
    o.position = float4(world_pos, 1.0);
    // Reconstruct window-space depth from the visibility buffer's packed depth key (nearest→max).
    o.depth = exact_depth;
    return o;
}
