// Directional shadow map: a depth-only pass that rasterizes the scene from the
// light's point of view. The vertex shader transforms world-space positions by
// the light's view-projection (`light_mvp`); the fragment stage writes nothing
// (depth only). A later full-screen pass (pbr.slang) samples the stored depth to
// shadow the directional sun. The model matrix is identity (the sandbox orbits
// the camera), so input positions are already world-space.

// Only POSITION is consumed; the mesh buffer's normal/uv are skipped via the
// 32-byte stride. Declaring just POSITION keeps the input signature matching the
// position-only vertex layout on both backends.
struct VSInput
{
    float3 pos : POSITION;
};

struct PushConstants
{
    float4x4 light_mvp; // light view-projection (model is identity)
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc : register(b0, space0);

[shader("vertex")]
float4 vsMain(VSInput input) : SV_Position
{
    return mul(pc.light_mvp, float4(input.pos, 1.0));
}

[shader("fragment")]
void fsMain()
{
    // Depth-only: nothing to write.
}
