// Directional shadow map: a depth-only pass that rasterizes the scene from the
// light's point of view. The vertex shader transforms world-space positions by
// the light's view-projection (`light_mvp`); the fragment stage writes nothing
// (depth only) except, for masked materials, an alpha-test discard so cutout
// geometry (foliage, chains) punches holes in the shadow too. A later full-screen
// pass (pbr.slang) samples the stored depth to shadow the directional sun. The
// model matrix is identity (the sandbox orbits the camera), so input positions are
// already world-space.

#include "bindless.slang"

static const uint NO_TEXTURE = 0xFFFFFFFF;

// POSITION + TEXCOORD only (NORMAL is omitted — the shadow pass is depth + an
// alpha-test discard, neither of which needs the normal). Declaring only the
// consumed attributes keeps every backend's input signature to two elements, so
// the `MeshPositionUv` vertex layout (pos@0, uv@24 over the shared 32-byte mesh
// buffer) matches exactly with no "attribute not consumed" validation warning.
struct VSInput
{
    float3 pos : POSITION;
    float2 uv : TEXCOORD;
};

struct VSOutput
{
    float4 clip : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushConstants
{
    float4x4 light_mvp;  // light view-projection (model is identity)
    uint base_color_tex; // base-color bindless index (NO_TEXTURE if absent)
    float alpha_cutoff;  // |cutoff| = alpha-test threshold (0 = opaque). Sign selects the
                         // G-buffer's hard-vs-hashed mode; the shadow always uses |cutoff|.
    float2 _pad;
    // GPU skinning (Stage B.2b): bindless storage-buffer indices for joints / weights /
    // palette + joint count. Only read by `vsMainSkinned`; unused on the static path.
    uint4 skin;
    // GPU morph (Stage C optimization): per-target per-vertex deltas + per-frame weights.
    // Only read by `vsMainMorphed`; unused on the static / skinned paths.
    uint4 morph; // x deltas, y weights, z target_count, w vertex_count
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[shader("vertex")]
VSOutput vsMain(VSInput input)
{
    VSOutput o;
    o.clip = mul(pc.light_mvp, float4(input.pos, 1.0));
    o.uv = input.uv;
    return o;
}

// Transform `v` by joint `j`'s palette matrix as an explicit column-major M*v (same
// layout the engine packs the push matrices with — matches gbuffer.slang's skinning).
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

// Skinned shadow vertex shader: skin the bind-pose position (vertex pulling), then
// transform by the light view-projection. The palette places the vertex in world
// space, so `light_mvp` is the bare light view-projection.
[shader("vertex")]
VSOutput vsMainSkinned(VSInput input, uint vid : SV_VertexID)
{
    RWByteAddressBuffer joints_buf = g.storage_buffers[pc.skin.x];
    RWByteAddressBuffer weights_buf = g.storage_buffers[pc.skin.y];
    uint4 j = joints_buf.Load<uint4>(vid * 16);
    float4 w = weights_buf.Load<float4>(vid * 16);
    uint p = pc.skin.z;

    float4 bind_pos = float4(input.pos, 1.0);
    float4 pos = w.x * joint_mul(p, j.x, bind_pos) + w.y * joint_mul(p, j.y, bind_pos)
               + w.z * joint_mul(p, j.z, bind_pos) + w.w * joint_mul(p, j.w, bind_pos);

    VSOutput o;
    o.clip = mul(pc.light_mvp, pos);
    o.uv = input.uv;
    return o;
}

// Morphed shadow vertex shader: blend the bind-pose position with the active morph
// targets (vertex pulling, matching gbuffer.slang's `vsMainMorphed`), then transform by
// the light view-projection so the cast shadow follows the morph. Morphed verts are in
// local space, so `light_mvp` already folds in the node transform.
[shader("vertex")]
VSOutput vsMainMorphed(VSInput input, uint vid : SV_VertexID)
{
    RWByteAddressBuffer deltas = g.storage_buffers[pc.morph.x];
    RWByteAddressBuffer weights = g.storage_buffers[pc.morph.y];
    uint target_count = pc.morph.z;
    uint vertex_count = pc.morph.w;

    float3 pos = input.pos;
    for (uint t = 0; t < target_count; ++t)
    {
        float w = weights.Load<float>(t * 4);
        if (w != 0.0)
        {
            uint base = (t * vertex_count + vid) * 32;
            pos += w * deltas.Load<float4>(base + 0).xyz;
        }
    }

    VSOutput o;
    o.clip = mul(pc.light_mvp, float4(pos, 1.0));
    o.uv = input.uv;
    return o;
}

[shader("fragment")]
void fsMain(VSOutput input)
{
    // Opaque (cutoff 0) never touches the texture -> depth-only, byte-identical to the
    // pre-mask pass. Masked / foliage materials discard sub-cutoff texels so the shadow has
    // the same holes as the lit geometry (uses the wrap sampler to match the G-buffer). The
    // G-buffer may render this leaf with HASHED alpha (signalled by a negative cutoff), but the
    // shadow map isn't TAA-resolved, so it always uses the |cutoff| hard test for a crisp,
    // noise-free cutout shadow.
    float cutoff = abs(pc.alpha_cutoff);
    if (cutoff > 0.0 && pc.base_color_tex != NO_TEXTURE)
    {
        float a = g.textures[pc.base_color_tex].Sample(g.samp_wrap, input.uv).a;
        if (a < cutoff)
            discard;
    }
}
