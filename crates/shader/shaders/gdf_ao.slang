// Phase 11 Stage C2 — GDF ambient occlusion into the deferred render.
//
// A full-screen compute pass: for every shaded G-buffer pixel it reconstructs the
// world-space surface point from the depth buffer, reads the world normal, and
// marches the world-space scene GDF (C1) for ambient occlusion. The result is an
// AO factor in [0,1] written to a storage image that the deferred lighting pass
// (pbr.slang) multiplies into the ambient / IBL term — the first time the GDF
// affects the real render.
//
// World position comes from depth, NOT the G-buffer position MRT: the rasterizer
// writes object-space positions (the model matrix is folded only into the clip
// position), so transformed objects' G-buffer positions don't line up with the
// world-space GDF. Reconstructing `inv_view_proj * (ndc, depth)` yields the true
// world point for every object uniformly, exactly like the C1 primary-ray path.
// The world normal IS correct in the G-buffer because the sample scene's transforms
// are translation + uniform scale (direction preserved).

#include "bindless.slang"
#include "clipmap.slang"

struct AoPush
{
    float4x4 inv_view_proj; // clip -> world (surface reconstruction)
    uint depth_index;       // sampled G-buffer depth
    uint normal_index;      // sampled G-buffer world normal
    uint gdf_sampled;       // volumes[] (SRV) index of the scene GDF (legacy; clipmap below)
    uint out_index;         // storage-image index (AO output)
    uint width;
    uint height;
    uint flip_y;            // bit0: 1 = Vulkan (clip-space Y down)
    uint pad0;
    float4 aabb_min;        // xyz GDF world min, w = ground plane Y
    float4 aabb_max;        // xyz GDF world max, w = sample distance clamp
    float4 params;          // x = AO reach (world units), y = strength, z = bias, w = AO floor
    uint4 clip;             // Stage B clipmap: x = descriptor index, y = level count
};

[[vk::push_constant]]
ConstantBuffer<AoPush> pc : register(b0, space0);

#define GDF_MIN (pc.aabb_min.xyz)
#define GDF_MAX (pc.aabb_max.xyz)
#define GROUND_Y (pc.aabb_min.w)

// Distance to the baked scene geometry via the clipmap (single level ⇒ identical to the
// legacy single-volume sample, clamped). Outside every level, a large distance — the
// empty bounding box must never read as a solid occluder.
float geo_inside(float3 p)
{
    ClipMap c;
    c.desc = pc.clip.x;
    c.count = pc.clip.y;
    return cm_geo_inside(c, p, pc.aabb_max.w);
}

// Occlusion distance: baked geometry unioned with the analytic ground plane (y=0),
// matching the C1 trace's `scene_occ`.
float scene_occ(float3 p)
{
    return min(geo_inside(p), p.y - GROUND_Y);
}

// Marched ambient occlusion (Inigo Quilez 5-tap), with the sampling reach scaled to
// world units (`params.x`) since this scene is large, not the Stage-B unit cube.
float calc_ao(float3 p, float3 n)
{
    float reach = pc.params.x;
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++)
    {
        float h = 0.001 + reach * (float(i) + 1.0) / 5.0;
        float d = scene_occ(p + n * h);
        occ += (h - d) * sca;
        sca *= 0.92;
    }
    // Exponential (Beer-Lambert-like) attenuation rather than `saturate(1 - k·occ)`: occlusion
    // asymptotes toward — but never reaches — zero, so a visible concave corner (which still sees
    // most of the sky) keeps a soft AO instead of crushing to pure black. The linear form clamped
    // to 0 in corners and, with weak indirect fill, burned the recesses black.
    float ao = exp(-pc.params.y * max(occ, 0.0));
    // Remap to [floor, 1]: even a fully-occluded contact (a column meeting the floor) keeps at
    // least `floor` of the ambient, so deep contacts read as soft shade rather than near-black —
    // the single-bounce GDF GI under-fills those recesses, and a metric reach makes the contact
    // band wide, so without a floor the column bases crush. `AO_FLOOR` tunes it (0 = no floor).
    return pc.params.w + (1.0 - pc.params.w) * ao;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(pc.width, pc.height);
    float depth = g.textures[pc.depth_index].SampleLevel(g.samp, uv, 0.0).r;

    // Far plane (no geometry): nothing to occlude.
    if (depth >= 1.0)
    {
        g.storage_images[pc.out_index][tid.xy] = float4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    // Reconstruct the world-space surface point. Same NDC convention as pbr.slang's
    // skybox reconstruction (the Y flip folds the backend clip-space handedness),
    // but with the sampled depth instead of the far plane.
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
    float4 hpos = mul(pc.inv_view_proj, float4(ndc, depth, 1.0));
    float3 world_pos = hpos.xyz / hpos.w;

    float3 n = normalize(g.textures[pc.normal_index].SampleLevel(g.samp, uv, 0.0).xyz);
    // Two-sided facing contract (F6M, flip_y bit1 = content scenes): orient toward the viewer
    // exactly as the lighting pass does (pbr.slang `pc.two_sided`), or a back-face pixel of
    // thin double-sided geometry (leaf card, cloth fold) marches its AO cone INTO the sheet
    // and pins at the floor. Same-pixel near-plane unprojection (standard-Z: 0 = near) = a
    // point toward the camera on this pixel's view ray, so no camera position in the push.
    if ((pc.flip_y & 2u) != 0u)
    {
        float4 hnear = mul(pc.inv_view_proj, float4(ndc, 0.0, 1.0));
        if (dot(n, hnear.xyz / hnear.w - world_pos) < 0.0)
            n = -n;
    }

    // Offset along the normal by the bias so the surface's own voxel doesn't
    // self-occlude through trilinear/voxelisation error.
    float ao = calc_ao(world_pos + n * pc.params.z, n);

    g.storage_images[pc.out_index][tid.xy] = float4(ao, ao, ao, 1.0);
}
