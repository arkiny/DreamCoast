// Per-pixel motion vectors (velocity G-buffer channel — pipeline re-baseline PR-2).
//
// Rasterizes the opaque scene into a single RG16Float velocity target that stores, per
// pixel, the screen-space motion of that surface point between the PREVIOUS and CURRENT
// frame: `motion = curr_ndc.xy - prev_ndc.xy` (in NDC, i.e. clip.xy / clip.w). Downstream
// consumers (velocity-aware TAAU here; motion blur / reprojection later) read it to
// reproject each pixel by its own motion instead of the camera-only reprojection.
//
// Two design invariants (both canonical, see docs/velocity-motion-vectors.md):
//   • JITTER-EXCLUDED. The velocity must not carry the TAA sub-pixel camera jitter, or the
//     jitter would leak into the reprojection and re-introduce shimmer. So this pass uses
//     the UNJITTERED current + previous view-projection (`mvp` / `prev_mvp` here are built
//     from the stable, jitter-free projection), NOT the jittered scene mvp the G-buffer
//     fill uses. The delta of two unjittered clip positions is jitter-free by construction.
//   • Y-FLIP CANCELS. Vulkan flips clip-space Y; but the motion vector is a *difference* of
//     two NDC positions computed with the *same* projection convention, so any Y-flip
//     applied to both terms cancels in the subtraction. We store the raw NDC delta and let
//     the consumer add it in its own UV convention — identical bytes on all three backends.
//
// The previous-frame clip position is reconstructed from the object's PREVIOUS model
// transform (`prev_model`) fed by the animation/transform system's single prev-pose source
// (per-object prev model matrix; skinning uses the prev-frame joint palette, morph the prev
// weights). Static geometry passes `prev_model == model` and `prev_mvp` = last frame's
// unjittered view-proj, so a moving camera still produces correct camera motion.

#include "bindless.slang"

struct VSInput
{
    float3 pos : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
};

struct VSOutput
{
    float4 clip : SV_Position; // current unjittered clip (rasterization position)
    float4 cur_clip : POSITION0;
    float4 prev_clip : POSITION1;
};

struct PushConstants
{
    float4x4 mvp;         // current UNJITTERED view_proj * model (clip for rasterization)
    float4x4 prev_mvp;    // previous UNJITTERED view_proj * prev_model
    // GPU skinning: joints / weights / palette (this frame) + joint_count. `vsMainSkinned`
    // reads them; the previous-frame palette lives in `skin_prev.x`.
    uint4 skin;           // x joints, y weights, z palette(curr), w joint_count
    uint4 skin_prev;      // x palette(prev), yzw unused
    // GPU morph: deltas / weights(curr) + target_count + vertex_count; prev weights in
    // `morph_prev.x`. `vsMainMorphed` reads them.
    uint4 morph;          // x deltas, y weights(curr), z target_count, w vertex_count
    uint4 morph_prev;     // x weights(prev), yzw unused
    // Baked vertex-cache (deform): the PREVIOUS-frame position storage buffer. The current
    // positions come from the bound frame vertex buffer (`input.pos`), so no current index is
    // needed. `vsMainDeform` reads `deform_prev.x`.
    uint4 deform_prev;    // x prev-frame positions (tight float3/vertex), yzw unused
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

// NDC (perspective-divided) motion in xy: current minus previous. Guard w>0 so a vertex
// behind the camera (w<=0) doesn't produce a garbage divide; such pixels are off-screen
// anyway. Stored as the raw NDC delta (consumer applies its UV convention).
float2 motion_ndc(float4 cur_clip, float4 prev_clip)
{
    float2 cur_ndc = cur_clip.xy / max(cur_clip.w, 1e-6);
    float2 prev_ndc = prev_clip.xy / max(prev_clip.w, 1e-6);
    return cur_ndc - prev_ndc;
}

// Column-major M*v against a palette matrix at joint `j` (matches gbuffer.slang layout).
float4 joint_mul(uint palette_index, uint j, float4 v)
{
    RWByteAddressBuffer palette = g.storage_buffers[palette_index];
    uint base = j * 64;
    float4 c0 = palette.Load<float4>(base + 0);
    float4 c1 = palette.Load<float4>(base + 16);
    float4 c2 = palette.Load<float4>(base + 32);
    float4 c3 = palette.Load<float4>(base + 48);
    return c0 * v.x + c1 * v.y + c2 * v.z + c3 * v.w;
}

// Linear-blend skinning of the bind-pose position by a given palette (curr or prev).
float4 skin_pos(uint palette_index, uint4 j, float4 w, float4 bind_pos)
{
    return w.x * joint_mul(palette_index, j.x, bind_pos)
         + w.y * joint_mul(palette_index, j.y, bind_pos)
         + w.z * joint_mul(palette_index, j.z, bind_pos)
         + w.w * joint_mul(palette_index, j.w, bind_pos);
}

// Morph-blend the bind-pose position by a given per-target weights buffer (curr or prev).
float3 morph_pos(uint deltas_index, uint weights_index, uint target_count, uint vertex_count,
                 uint vid, float3 base_pos)
{
    RWByteAddressBuffer deltas = g.storage_buffers[deltas_index];
    RWByteAddressBuffer weights = g.storage_buffers[weights_index];
    float3 pos = base_pos;
    for (uint t = 0; t < target_count; ++t)
    {
        float w = weights.Load<float>(t * 4);
        if (w != 0.0)
        {
            uint bb = (t * vertex_count + vid) * 32;
            pos += w * deltas.Load<float4>(bb + 0).xyz;
        }
    }
    return pos;
}

// Static / Spin / node-animation velocity: prev clip from the prev model transform baked
// into `prev_mvp`, current clip from `mvp` (both unjittered).
[shader("vertex")]
VSOutput vsMain(VSInput input)
{
    VSOutput o;
    float4 p = float4(input.pos, 1.0);
    o.cur_clip = mul(pc.mvp, p);
    o.prev_clip = mul(pc.prev_mvp, p);
    o.clip = o.cur_clip;
    return o;
}

// Skinned velocity: current pose from the current palette, previous pose from the prev-frame
// palette (`skin_prev.x`) — so the motion follows joint animation, not just the object node.
[shader("vertex")]
VSOutput vsMainSkinned(VSInput input, uint vid : SV_VertexID)
{
    RWByteAddressBuffer joints_buf = g.storage_buffers[pc.skin.x];
    RWByteAddressBuffer weights_buf = g.storage_buffers[pc.skin.y];
    uint4 j = joints_buf.Load<uint4>(vid * 16);
    float4 w = weights_buf.Load<float4>(vid * 16);
    float4 bind_pos = float4(input.pos, 1.0);

    float4 cur_pos = skin_pos(pc.skin.z, j, w, bind_pos);
    float4 prev_pos = skin_pos(pc.skin_prev.x, j, w, bind_pos);

    VSOutput o;
    o.cur_clip = mul(pc.mvp, cur_pos);
    o.prev_clip = mul(pc.prev_mvp, prev_pos);
    o.clip = o.cur_clip;
    return o;
}

// Morphed velocity: current pose from the current weights, previous pose from the prev-frame
// weights (`morph_prev.x`) — so the motion follows the blendshape animation.
[shader("vertex")]
VSOutput vsMainMorphed(VSInput input, uint vid : SV_VertexID)
{
    uint tc = pc.morph.z;
    uint vc = pc.morph.w;
    float3 cur_pos = morph_pos(pc.morph.x, pc.morph.y, tc, vc, vid, input.pos);
    float3 prev_pos = morph_pos(pc.morph.x, pc.morph_prev.x, tc, vc, vid, input.pos);

    VSOutput o;
    o.cur_clip = mul(pc.mvp, float4(cur_pos, 1.0));
    o.prev_clip = mul(pc.prev_mvp, float4(prev_pos, 1.0));
    o.clip = o.cur_clip;
    return o;
}

// Baked vertex-cache (deform) velocity: current position from the bound frame vertex buffer
// (`input.pos`, this frame's ring VB), previous position from the prev-frame position storage
// (`deform_prev.x`, tightly-packed float3/vertex) by `SV_VertexID` — so the motion follows the
// baked surface deformation, not just the static placement node / camera.
[shader("vertex")]
VSOutput vsMainDeform(VSInput input, uint vid : SV_VertexID)
{
    RWByteAddressBuffer prev = g.storage_buffers[pc.deform_prev.x];
    float3 prev_pos = prev.Load<float3>(vid * 12);

    VSOutput o;
    o.cur_clip = mul(pc.mvp, float4(input.pos, 1.0));
    o.prev_clip = mul(pc.prev_mvp, float4(prev_pos, 1.0));
    o.clip = o.cur_clip;
    return o;
}

[shader("fragment")]
float2 fsMain(VSOutput input) : SV_Target0
{
    return motion_ndc(input.cur_clip, input.prev_clip);
}

// ---------------------------------------------------------------------------------------
// Velocity debug visualization (DEBUG_VIEW=11). A full-screen compute pass that reads the
// RG16Float velocity target and writes a color-coded HDR the tonemap displays: motion is
// mapped to hue/brightness (direction → red/green channels, magnitude → intensity) so the
// user can see which pixels move and how fast. `scale` normalizes NDC motion to a visible
// range (a full-screen sweep is ~2.0 in NDC; small motions need amplification).
struct VizPush
{
    uint velocity_index; // sampled: the RG16Float velocity target
    uint out_index;      // storage image: HDR viz output
    uint width;
    uint height;
    float scale;         // amplification of the NDC motion for display
    float pad0, pad1, pad2;
};

[[vk::push_constant]]
ConstantBuffer<VizPush> vpc : register(b0, space0);

[shader("compute")]
[numthreads(8, 8, 1)]
void csViz(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= vpc.width || tid.y >= vpc.height)
        return;
    float2 uv = (float2(tid.xy) + 0.5) / float2(vpc.width, vpc.height);
    float2 mv = g.textures[vpc.velocity_index].SampleLevel(g.samp, uv, 0.0).xy;
    float2 s = mv * vpc.scale;
    // Encode: R = +x / -x split, G = +y / -y split, B = magnitude — a readable, standard
    // motion-vector colour key (positive motion warm, magnitude brightens).
    float mag = length(s);
    float3 col = float3(0.5 + 0.5 * clamp(s.x, -1.0, 1.0), 0.5 + 0.5 * clamp(s.y, -1.0, 1.0),
                        clamp(mag, 0.0, 1.0));
    g.storage_images[vpc.out_index][tid.xy] = float4(col, 1.0);
}
