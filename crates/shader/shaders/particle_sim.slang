// GPU particle simulation (Phase 7): one compute thread per particle integrates
// a simple fountain (gravity + ground bounce) in-place in a storage buffer, and
// respawns dead particles from a hashed initial velocity. The buffer is a raw
// byte-address UAV in the bindless storage-buffer table; the draw pass reads the
// same buffer in its vertex stage.
//
// Particle layout (32 bytes): float3 pos; float life; float3 vel; float seed.

struct PushConstants
{
    uint buffer_index; // bindless storage-buffer index of the particle array
    uint count;        // number of particles
    float dt;          // timestep (seconds)
    float time;        // elapsed time (varies respawn hashes)
    uint init;         // 1 = seed all particles (first dispatch)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[[vk::binding(4, 0)]]
RWByteAddressBuffer g_storage_buffers[] : register(u0, space1);

// Integer hash -> 3 pseudo-random floats in [0,1).
float3 hash3(uint n)
{
    n = (n << 13) ^ n;
    n = n * (n * n * 15731u + 789221u) + 1376312589u;
    uint3 k = uint3(n, n * 16807u, n * 48271u);
    return float3(k & 0x7fffffffu) / float(0x7fffffff);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.count)
        return;

    uint base = i * 32;
    RWByteAddressBuffer buf = g_storage_buffers[pc.buffer_index];

    float3 pos = buf.Load<float3>(base + 0);
    float life = buf.Load<float>(base + 12);
    float3 vel = buf.Load<float3>(base + 16);

    if (pc.init != 0 || life <= 0.0)
    {
        // Respawn from the emitter with a hashed cone velocity. Mixing the time
        // gives each respawn a fresh direction.
        float3 r = hash3(i * 2654435761u ^ uint(pc.time * 60.0) * 40503u);
        pos = float3(0.0, 0.05, 0.0);
        vel = float3((r.x - 0.5) * 2.4, 2.6 + r.y * 2.0, (r.z - 0.5) * 2.4);
        life = 1.2 + r.z * 1.8;
        if (pc.init != 0)
        {
            // Seed particles spread along their parabola so the very first frame
            // already shows a full fountain (matters for headless screenshots).
            float t0 = hash3(i * 374761393u + 1u).x * life;
            pos += vel * t0 + 0.5 * float3(0.0, -5.0, 0.0) * t0 * t0;
            vel.y -= 5.0 * t0;
            life -= t0;
        }
    }
    else
    {
        vel.y -= 5.0 * pc.dt; // gravity
        pos += vel * pc.dt;
        life -= pc.dt;
        if (pos.y < 0.0)
        {
            pos.y = 0.0;
            vel.y = -vel.y * 0.45; // damped bounce
        }
    }

    buf.Store<float3>(base + 0, pos);
    buf.Store<float>(base + 12, life);
    buf.Store<float3>(base + 16, vel);
}
