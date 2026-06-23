// Path tracer via the hardware ray-tracing *pipeline* (Phase 8 M5).
//
// Reproduces the inline path tracer (`rt_path.slang`) using raygen / miss /
// closest-hit shaders dispatched through a shader binding table, so the two RHI
// ray-tracing abstractions (inline ray query vs. full pipeline + SBT) can be
// cross-checked against each other for a pixel-near match.
//
// The bounce loop lives in the raygen shader: each iteration issues one
// `TraceRay`, the closest-hit shader fills a small payload with the surface's
// emitted + direct light, the diffuse attenuation, and the next bounce ray; the
// miss shader fills it with the sky and terminates. Shadow rays use an inline
// `RayQuery` inside closest-hit, so the binding table stays at one raygen / one
// miss / one hit group and `MaxTraceRecursionDepth` stays at 1.
//
// Diffuse-only, same determinism scheme (PCG hash per pixel + sample) and the
// same push constants / accumulation buffer as the inline tracer.

#include "bindless.slang"

static const float PI = 3.14159265358979;
static const uint MAX_BOUNCES = 4;

struct PushConstants
{
    float4x4 inv_view_proj; // clip -> world
    float4 cam_pos;         // xyz camera world position
    float4 sun;             // xyz direction toward the sun, w = intensity
    uint out_index;         // bindless storage-image index (averaged output)
    uint accum_index;       // bindless storage-buffer index (float4 sum per pixel)
    uint inst_index;        // bindless storage-buffer index (instance table)
    uint frame;             // accumulation frame (0 = first, clears the sum)
    uint width;
    uint height;
    uint flip_y;            // bit0: 1 = Vulkan (Y down); bit1: 1 = Cornell box mode
    uint spp;               // samples per dispatch
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

// Bounce-loop state carried between the raygen loop and the hit/miss shaders.
struct Payload
{
    float3 radiance;    // light emitted + directly lit at this hit (or sky on miss)
    float3 atten;       // multiply throughput by this for the next bounce (albedo)
    float3 next_origin; // spawn point of the next bounce ray
    float3 next_dir;    // cosine-weighted diffuse bounce direction
    uint rng;           // PCG state, advanced by the hit shader's bounce sampling
    uint done;          // 1 = terminate the path (miss or absorbed)
    uint primary;       // 1 = camera ray (miss shows the sun disk), 0 = bounce
};

// 32-byte per-instance record (matches the host packing in main.rs).
struct Instance
{
    uint vtx;      // bindless storage index of the 32-byte/vertex buffer
    uint idx;      // bindless storage index of the u32 index buffer
    uint pad0;
    uint pad1;
    float4 albedo; // rgb base color, a = emissive scale
};

// PCG hash RNG (deterministic per seed -> identical across backends).
uint pcg(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}
float rnd(inout uint s)
{
    return float(pcg(s) & 0xFFFFFFu) / float(0x1000000);
}

// Gradient sky + HDR sun disk. `with_sun` adds the bright sun disk; it is only
// enabled for primary (camera) rays. Diffuse bounce rays use `with_sun = false`
// because the sun's contribution is already added analytically as a direct light
// (the shadow-ray term) — sampling the disk again would double-count the sun and
// inject fireflies (the standard "sun is a delta light for diffuse" treatment).
float3 sky(float3 dir, float3 sun_dir, float sun_i, bool with_sun)
{
    float up = dir.y;
    float3 zenith = float3(0.04, 0.16, 0.55);
    float3 horizon = float3(0.50, 0.66, 0.92);
    float3 ground = float3(0.16, 0.15, 0.14);
    float3 col = up >= 0.0 ? lerp(horizon, zenith, pow(saturate(up), 0.45))
                           : lerp(horizon, ground, saturate(-up * 6.0));
    if (with_sun)
    {
        float c = saturate(dot(dir, sun_dir));
        col += sun_i * (pow(c, 4000.0) * 6.0 + pow(c, 32.0) * 0.4);
    }
    return col;
}

// Cosine-weighted hemisphere sample about `n`.
float3 cosine_hemisphere(float3 n, inout uint rng)
{
    float u1 = rnd(rng);
    float u2 = rnd(rng);
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    float3 t = normalize(abs(n.x) > 0.9 ? cross(n, float3(0, 1, 0)) : cross(n, float3(1, 0, 0)));
    float3 b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}

// Read an instance record from the table.
Instance load_instance(uint id)
{
    uint base = id * 32u;
    Instance inst;
    inst.vtx = g.storage_buffers[pc.inst_index].Load(base + 0u);
    inst.idx = g.storage_buffers[pc.inst_index].Load(base + 4u);
    inst.pad0 = 0u;
    inst.pad1 = 0u;
    inst.albedo = asfloat(g.storage_buffers[pc.inst_index].Load4(base + 16u));
    return inst;
}

// Barycentric-interpolated, normalized surface normal for a hit triangle.
float3 hit_normal(Instance inst, uint prim, float2 bary)
{
    uint i0 = g.storage_buffers[inst.idx].Load(prim * 12u + 0u);
    uint i1 = g.storage_buffers[inst.idx].Load(prim * 12u + 4u);
    uint i2 = g.storage_buffers[inst.idx].Load(prim * 12u + 8u);
    // Vertex layout: position @0 (float3), normal @12 (float3), 32-byte stride.
    float3 n0 = asfloat(g.storage_buffers[inst.vtx].Load3(i0 * 32u + 12u));
    float3 n1 = asfloat(g.storage_buffers[inst.vtx].Load3(i1 * 32u + 12u));
    float3 n2 = asfloat(g.storage_buffers[inst.vtx].Load3(i2 * 32u + 12u));
    float b0 = 1.0 - bary.x - bary.y;
    return normalize(n0 * b0 + n1 * bary.x + n2 * bary.y);
}

// ---- Miss: sky (or black for the enclosed Cornell box) ----
[shader("miss")]
void msMain(inout Payload p)
{
    bool cornell = (pc.flip_y & 2u) != 0u;
    float3 sun_dir = normalize(pc.sun.xyz);
    // Sun disk only on the primary ray; bounces get the sun analytically.
    p.radiance = cornell ? float3(0, 0, 0)
                         : sky(WorldRayDirection(), sun_dir, pc.sun.w, p.primary != 0u);
    p.atten = float3(0, 0, 0);
    p.done = 1u;
}

// ---- Closest hit: emitted + direct sun (inline shadow ray) + next bounce ----
[shader("closesthit")]
void chMain(inout Payload p, BuiltInTriangleIntersectionAttributes attr)
{
    bool cornell = (pc.flip_y & 2u) != 0u;
    float3 sun_dir = normalize(pc.sun.xyz);

    Instance inst = load_instance(InstanceID());
    float3 n = hit_normal(inst, PrimitiveIndex(), attr.barycentrics);
    float3 wdir = WorldRayDirection();
    if (dot(n, wdir) > 0.0)
        n = -n; // face the incoming ray (two-sided)
    float3 hit = WorldRayOrigin() + wdir * RayTCurrent();
    float3 albedo = inst.albedo.rgb;

    // Emissive surfaces add light directly.
    float3 radiance = albedo * inst.albedo.a;

    // Direct sun light with a hardware shadow ray (outdoor mode only; the Cornell
    // box is lit purely by its emissive ceiling via the bounce loop).
    float ndl = max(dot(n, sun_dir), 0.0);
    if (!cornell && ndl > 0.0)
    {
        RayDesc sray;
        sray.Origin = hit + n * 0.002;
        sray.Direction = sun_dir;
        sray.TMin = 0.001;
        sray.TMax = 10000.0;
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
        sq.TraceRayInline(g.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, sray);
        sq.Proceed();
        if (sq.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
            radiance += (albedo / PI) * pc.sun.w * ndl;
    }

    p.radiance = radiance;
    // Cosine-weighted diffuse bounce (BRDF*cos/pdf = albedo).
    p.atten = albedo;
    p.next_origin = hit + n * 0.002;
    p.next_dir = cosine_hemisphere(n, p.rng);
    p.done = 0u;
}

// ---- Ray generation: per-pixel sampling + bounce loop + accumulation ----
[shader("raygeneration")]
void rgMain()
{
    uint2 tid = DispatchRaysIndex().xy;
    if (tid.x >= pc.width || tid.y >= pc.height)
        return;

    bool cornell = (pc.flip_y & 2u) != 0u;
    float sy = (pc.flip_y & 1u) != 0u ? 1.0 : -1.0;
    uint pixel = tid.y * pc.width + tid.x;

    float3 sun_dir = normalize(pc.sun.xyz);
    float3 sum = float3(0, 0, 0);
    for (uint s = 0; s < pc.spp; s++)
    {
        // Deterministic seed per pixel + global sample index (same on both APIs).
        uint rng = pixel * 9781u + (pc.frame * pc.spp + s) * 26699u + 1u;
        rng = pcg(rng);
        // Sub-pixel jitter for anti-aliasing.
        float2 jitter = float2(rnd(rng), rnd(rng)) - 0.5;
        float2 uv = (float2(tid) + 0.5 + jitter) / float2(pc.width, pc.height);
        float2 ndc = float2(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * sy);
        float4 world = mul(pc.inv_view_proj, float4(ndc, 1.0, 1.0));
        float3 dir = normalize(world.xyz / world.w - pc.cam_pos.xyz);

        // Bounce loop: one TraceRay per bounce, state carried in the payload.
        float3 radiance = float3(0, 0, 0);
        float3 throughput = float3(1, 1, 1);
        RayDesc ray;
        ray.Origin = pc.cam_pos.xyz;
        ray.Direction = dir;
        ray.TMin = 0.001;
        ray.TMax = 10000.0;
        for (uint bounce = 0; bounce < MAX_BOUNCES; bounce++)
        {
            Payload p;
            p.radiance = float3(0, 0, 0);
            p.atten = float3(0, 0, 0);
            p.next_origin = float3(0, 0, 0);
            p.next_dir = float3(0, 0, 0);
            p.rng = rng;
            p.done = 0u;
            p.primary = bounce == 0u ? 1u : 0u;
            TraceRay(g.tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, p);
            rng = p.rng;
            radiance += throughput * p.radiance;
            if (p.done != 0u)
                break;
            throughput *= p.atten;
            ray.Origin = p.next_origin;
            ray.Direction = p.next_dir;
            ray.TMin = 0.001;
            ray.TMax = 10000.0;
        }
        sum += radiance;
    }
    sum /= float(pc.spp);

    // Progressive accumulation: frame 0 clears, later frames add this frame's mean.
    float4 prev = pc.frame == 0u
                      ? float4(0, 0, 0, 0)
                      : asfloat(g.storage_buffers[pc.accum_index].Load4(pixel * 16u));
    float4 acc = prev + float4(sum, 1.0);
    g.storage_buffers[pc.accum_index].Store4(pixel * 16u, asuint(acc));

    g.storage_images[pc.out_index][tid] = float4(acc.rgb / acc.w, 1.0);
}
