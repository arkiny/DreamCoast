// Phase 14 (virtual geometry) M0 — mesh-shader pipeline capability smoke.
//
// The cluster rasterizer's HW path emits meshlets from a mesh shader (a task/object
// stage dispatches one threadgroup per visible cluster; the mesh stage emits its
// ~128 tris). This shader is the minimal bring-up: a single mesh threadgroup emits
// one hardcoded RGB triangle so `--mesh-shader-test` can prove the RHI mesh pipeline
// (`create_mesh_pipeline` + `draw_mesh_tasks`) works on each backend before any
// cluster data exists. The amplification (object/task) stage and per-cluster payload
// arrive in M2, where they are actually exercised.
//
// Metal requires mesh outputs to be assigned as a WHOLE STRUCT (`verts[i] = v`), not
// field-by-field (error E56104) — SPIR-V is lenient, so we write the whole struct to
// stay portable. One vertex per lane keeps the emit branch-free.

struct VertexOut
{
    float4 pos : SV_Position;
    // Digit-less semantic on purpose: Slang's Metal mesh-stage codegen keeps the trailing
    // digit (`user(COLOR0)`) while the fragment stage strips it (`user(COLOR)`), so a
    // `COLOR0` interpolant fails mesh↔fragment interstage linking at pipeline creation
    // ("Fragment input user(COLOR) mismatching mesh shader output"). `COLOR` matches on both.
    float3 color : COLOR;
};

[shader("mesh")]
[numthreads(3, 1, 1)]
[outputtopology("triangle")]
void meshMain(
    uint lane : SV_GroupThreadID,
    // The `indices` / `vertices` output modifiers already imply the `out` direction; the
    // explicit `out` is a Slang 2026.10.2 → DXC 1.9 codegen hazard (Slang emits a malformed
    // `out indices out uint3` that DXC rejects). Dropping it makes Slang warn E54001 but emit
    // valid DXIL, and SPIR-V / Metal are unaffected.
    indices uint3 tris[1],
    vertices VertexOut verts[3])
{
    SetMeshOutputCounts(3, 1);

    const float2 positions[3] = { float2(0.0, 0.6), float2(0.6, -0.6), float2(-0.6, -0.6) };
    const float3 colors[3] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) };

    VertexOut v;
    v.pos = float4(positions[lane], 0.0, 1.0);
    v.color = colors[lane];
    verts[lane] = v;

    if (lane == 0)
        tris[0] = uint3(0, 1, 2);
}

[shader("fragment")]
float4 fragMain(VertexOut i) : SV_Target
{
    return float4(i.color, 1.0);
}
