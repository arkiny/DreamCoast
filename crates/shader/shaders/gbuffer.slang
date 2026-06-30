// Deferred G-buffer fill: rasterizes the mesh and writes albedo, world-space
// normal, material (metallic/roughness/AO), and world position into four MRT
// color attachments. A later full-screen pass (pbr.slang) reads these to do the
// lighting. The model matrix is identity (the sandbox orbits the camera), so
// input positions/normals are already world-space.
//
// Bindless resources (`g.textures[]` / `g.samp`) come from the shared
// `ParameterBlock` in `bindless.slang` (Metal argument buffer / Vulkan
// descriptor set 0 / D3D12 table). No globals here, so the block lands at set 0
// — byte-identical SPIR-V to the previous loose-global form (Vulkan untouched).
// A texture index of 0xFFFFFFFF means "absent — use the scalar factor".

#include "bindless.slang"

static const uint NO_TEXTURE = 0xFFFFFFFF;

struct VSInput
{
    float3 pos : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
};

struct VSOutput
{
    float4 clip : SV_Position;
    float3 world_pos : POSITION0;
    float3 world_normal : NORMAL0;
    float2 uv : TEXCOORD0;
};

struct GBufferOut
{
    float4 albedo : SV_Target0;   // rgb base color (linear), a = AO
    float4 normal : SV_Target1;   // xyz world-space normal
    float4 material : SV_Target2; // r metallic, g roughness, b AO
    float4 position : SV_Target3; // xyz world position
};

struct PushConstants
{
    float4x4 mvp;
    float4 base_color_factor;
    float4 mr_factor;     // x metallic, y roughness, z texture mip LOD bias (TAAU), w alpha cutoff
    uint4 tex;            // x base_color, y metallic_roughness, z normal, w emissive
    float4x4 model;       // object -> world (for world-space position + normal outputs)
    // GPU skinning (Stage B.2): bindless storage-buffer indices for the per-vertex
    // joints (uint4) + weights (float4) and the joint palette (float4x4 per joint).
    // Only read by `vsMainSkinned`; 0 / unused on the non-skinned path.
    uint4 skin;           // x joints, y weights, z palette, w joint_count
    // GPU morph (Stage C optimization): the static per-target per-vertex pos/normal
    // deltas (two float4 per vertex/target) + the small per-frame weights buffer. Only
    // read by `vsMainMorphed`; 0 / unused on the non-morphed path.
    uint4 morph;          // x deltas, y weights, z target_count, w vertex_count
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[shader("vertex")]
VSOutput vsMain(VSInput input)
{
    VSOutput o;
    o.clip = mul(pc.mvp, float4(input.pos, 1.0));
    // World-space position + normal (the model matrix is translation + uniform scale,
    // so the 3x3 transforms the normal direction correctly once re-normalized in fsMain).
    o.world_pos = mul(pc.model, float4(input.pos, 1.0)).xyz;
    o.world_normal = mul((float3x3)pc.model, input.normal);
    o.uv = input.uv;
    return o;
}

// Transform `v` by joint `j`'s palette matrix, done as an explicit column-major
// `M*v` (the palette is packed column-major — cols at +0/+16/+32/+48 — the same
// layout the engine packs the push `mvp`/`model` with, so no Load<float4x4> /
// matrix-majorness ambiguity).
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

// GPU vertex-pulling skinning: read this vertex's joints/weights from bindless
// storage buffers (by vertex id) and blend the per-joint transformed bind-pose
// position/normal — linear-blend skinning, blending the transformed points (= the
// weighted joint matrices applied), which avoids any matrix construction. The palette
// already places the vertex in scene/world space, so `pc.model` is identity for a
// skinned draw.
[shader("vertex")]
VSOutput vsMainSkinned(VSInput input, uint vid : SV_VertexID)
{
    RWByteAddressBuffer joints_buf = g.storage_buffers[pc.skin.x];
    RWByteAddressBuffer weights_buf = g.storage_buffers[pc.skin.y];
    uint4 j = joints_buf.Load<uint4>(vid * 16);    // 4 x u32
    float4 w = weights_buf.Load<float4>(vid * 16); // 4 x f32
    uint p = pc.skin.z;

    float4 bind_pos = float4(input.pos, 1.0);
    float4 bind_nrm = float4(input.normal, 0.0); // direction (w = 0)

    float4 pos = w.x * joint_mul(p, j.x, bind_pos) + w.y * joint_mul(p, j.y, bind_pos)
               + w.z * joint_mul(p, j.z, bind_pos) + w.w * joint_mul(p, j.w, bind_pos);
    float4 nrm = w.x * joint_mul(p, j.x, bind_nrm) + w.y * joint_mul(p, j.y, bind_nrm)
               + w.z * joint_mul(p, j.z, bind_nrm) + w.w * joint_mul(p, j.w, bind_nrm);

    VSOutput o;
    o.clip = mul(pc.mvp, pos);
    o.world_pos = mul(pc.model, pos).xyz;
    o.world_normal = mul((float3x3)pc.model, nrm.xyz);
    o.uv = input.uv;
    return o;
}

// GPU vertex-pulling morph (Stage C optimization): blend the bind-pose position +
// normal with the active morph targets — `vertex = base + Σ wᵢ·deltaᵢ` — reading each
// target's per-vertex pos/normal delta from a static bindless storage buffer (by vertex
// id) and the weights from a small per-frame buffer. This moves the per-vertex blend
// the CPU path (`apply_morph`) did onto the GPU, so the CPU does no per-vertex work and
// the bind-pose buffer is uploaded once. Morphed vertices stay in the mesh node's local
// space (unlike skinning), so `pc.model`/`pc.mvp` apply normally. Each delta is stored
// as two `float4` (pos.xyz, nrm.xyz) = 32 bytes, matching the skinning `Load<float4>`
// access pattern (16-byte aligned on every backend).
[shader("vertex")]
VSOutput vsMainMorphed(VSInput input, uint vid : SV_VertexID)
{
    RWByteAddressBuffer deltas = g.storage_buffers[pc.morph.x];
    RWByteAddressBuffer weights = g.storage_buffers[pc.morph.y];
    uint target_count = pc.morph.z;
    uint vertex_count = pc.morph.w;

    float3 pos = input.pos;
    float3 nrm = input.normal;
    for (uint t = 0; t < target_count; ++t)
    {
        float w = weights.Load<float>(t * 4);
        if (w != 0.0)
        {
            uint base = (t * vertex_count + vid) * 32;
            pos += w * deltas.Load<float4>(base + 0).xyz;
            nrm += w * deltas.Load<float4>(base + 16).xyz;
        }
    }

    VSOutput o;
    o.clip = mul(pc.mvp, float4(pos, 1.0));
    o.world_pos = mul(pc.model, float4(pos, 1.0)).xyz;
    o.world_normal = mul((float3x3)pc.model, nrm); // fsMain re-normalizes
    o.uv = input.uv;
    return o;
}

// Build a tangent frame from screen-space derivatives (no per-vertex tangents),
// then transform a tangent-space normal-map sample into world space.
float3 apply_normal_map(float3 n, float3 world_pos, float2 uv, float3 tangent_normal)
{
    float3 dp1 = ddx(world_pos);
    float3 dp2 = ddy(world_pos);
    float2 duv1 = ddx(uv);
    float2 duv2 = ddy(uv);

    float3 dp2perp = cross(dp2, n);
    float3 dp1perp = cross(n, dp1);
    float3 t = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 b = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = rsqrt(max(dot(t, t), dot(b, b)));
    float3x3 tbn = float3x3(t * invmax, b * invmax, n);
    return normalize(mul(tangent_normal, tbn));
}

// PCG integer hash -> a well-distributed uint from a uint seed (Jarzynski & Olano 2020).
// Pure integer ops, so it is bit-identical on SPIR-V and DXIL (the cross-backend gate).
uint pcg_hash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hashed/stochastic-alpha threshold in [0,1) keyed to a WORLD-SPACE point. World position is
// identical on every backend (no screen-space SV_Position Y-flip ambiguity), so the dither is
// DX≡VK-deterministic. Quantizing to ~0.25 mm decorrelates neighbouring pixels (spatial blue
// noise) AND makes the camera's sub-pixel TAA jitter resample a different threshold each frame,
// so the temporal accumulator resolves the binary keep/discard into smooth fractional coverage
// — soft, translucent foliage edges in a deferred pipeline with no sorting (Wyman & McGuire
// 2017, "Hashed Alpha Testing", resolved by TAA rather than MSAA).
float hashed_alpha_threshold(float3 world_pos)
{
    int3 q = int3(floor(world_pos * 4096.0)); // 1/4096 m ≈ 0.244 mm cells
    uint3 u = uint3(q);                        // two's-complement reinterpret (backend-stable)
    uint h = pcg_hash(u.x ^ pcg_hash(u.y ^ pcg_hash(u.z)));
    return float(h & 0xFFFFFFu) * (1.0 / float(0x1000000)); // [0,1)
}

[shader("fragment")]
GBufferOut fsMain(VSOutput input)
{
    // Material textures use the REPEAT/wrap sampler: floor tiles / brick courses tile via
    // glTF UVs > 1 (CLAMP would smear one edge texel across the whole surface). Untextured
    // or in-[0,1] UVs (the gallery avocado) are unaffected → byte-identical.
    // `mip_bias` (mr_factor.z) is the DLSS/FSR2-style negative LOD bias = log2(render_scale) when
    // rendering at a reduced internal resolution for TAAU: it pulls in sharper mips so the temporal
    // upscaler can reconstruct full-res texture detail (otherwise the low-res derivatives pick
    // blurry mips and the result stays soft). 0 on the native path = byte-identical to Sample().
    float mip_bias = pc.mr_factor.z;
    // Alpha-test cutoff (mr_factor.w) routes three ways by SIGN (one push field, no layout change):
    //   cutoff == 0 : opaque   -> the branch never runs, only `.rgb` is used (whole gallery is
    //                             byte-identical to the pre-mask path).
    //   cutoff  > 0 : MASK / foliage cutout -> hard discard below the threshold (crisp edges).
    //   cutoff  < 0 : foliage HASHED alpha  -> stochastic discard against a world-space hash so
    //                             TAA resolves the leaf edge into a soft translucent silhouette.
    float cutoff = pc.mr_factor.w;
    float3 albedo = pc.base_color_factor.rgb;
    if (pc.tex.x != NO_TEXTURE)
    {
        float4 bc = g.textures[pc.tex.x].SampleBias(g.samp_wrap, input.uv, mip_bias);
        albedo *= bc.rgb;
        float a = bc.a * pc.base_color_factor.a;
        if (cutoff > 0.0 && a < cutoff)
            discard;
        else if (cutoff < 0.0 && a < hashed_alpha_threshold(input.world_pos))
            discard;
    }

    float metallic = pc.mr_factor.x;
    float roughness = pc.mr_factor.y;
    if (pc.tex.y != NO_TEXTURE)
    {
        // glTF metallic-roughness: G = roughness, B = metallic.
        float3 mr = g.textures[pc.tex.y].SampleBias(g.samp_wrap, input.uv, mip_bias).rgb;
        roughness *= mr.g;
        metallic *= mr.b;
    }

    float3 n = normalize(input.world_normal);
    if (pc.tex.z != NO_TEXTURE)
    {
        float3 tn = g.textures[pc.tex.z].SampleBias(g.samp_wrap, input.uv, mip_bias).rgb * 2.0 - 1.0;
        n = apply_normal_map(n, input.world_pos, input.uv, tn);
    }

    GBufferOut o;
    o.albedo = float4(albedo, 1.0);
    o.normal = float4(n, 0.0);
    o.material = float4(metallic, saturate(roughness), 1.0, 0.0);
    o.position = float4(input.world_pos, 1.0);
    return o;
}

// Deferred surface-decal fragment shader (decals A3). A decal is a mesh laid coplanar on
// an already-shaded opaque surface; it must TINT the surface's albedo, not replace it with
// its own (diffuse-less) lighting inputs. So it writes ONLY RT0 = float4(albedo, alpha):
// the `DecalAlbedo` blend state alpha-blends the RGB into the G-buffer albedo (write mask
// RGB preserves RT0.a = the surface's baked AO) and write-masks RT1/RT2/RT3 off, leaving the
// underlying normal / metallic / roughness / world-pos intact. That is what stops the Intel
// Sponza `dirt_decal` (BLEND, no MR texture → metallicFactor 1.0) from painting the stone as
// opaque black metal — the stone keeps its own metalness/normal and only its colour is dusted.
// Decal G-buffer outputs: RT0 albedo (rgb tint, a = blend alpha) + RT2 material (g = roughness,
// a = blend alpha). The `DecalAlbedo` blend alpha-blends RT0.rgb and RT2.g and masks everything
// else (RT0.a AO, RT2.r metallic + RT2.b AO, RT1 normal, RT3 world-pos), so the decal only tints
// the albedo and nudges the roughness — the surface keeps its own metalness/normal/AO/position.
struct DecalOut
{
    float4 albedo : SV_Target0;
    float4 material : SV_Target2;
};

[shader("fragment")]
DecalOut fsDecal(VSOutput input)
{
    float mip_bias = pc.mr_factor.z;
    float3 albedo = pc.base_color_factor.rgb;
    float alpha = pc.base_color_factor.a;
    if (pc.tex.x != NO_TEXTURE)
    {
        float4 bc = g.textures[pc.tex.x].SampleBias(g.samp_wrap, input.uv, mip_bias);
        albedo *= bc.rgb;
        alpha *= bc.a;
    }
    // Roughness = factor × metallic-roughness texture's G (glTF packs roughness there). Dusty
    // decals are rough, so this raises the covered surface's roughness (A4). Written to RT2.g only.
    float roughness = pc.mr_factor.y;
    if (pc.tex.y != NO_TEXTURE)
        roughness *= g.textures[pc.tex.y].SampleBias(g.samp_wrap, input.uv, mip_bias).g;

    DecalOut o;
    o.albedo = float4(albedo, alpha);
    o.material = float4(0.0, saturate(roughness), 0.0, alpha);
    return o;
}
