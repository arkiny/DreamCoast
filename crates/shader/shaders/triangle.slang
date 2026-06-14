// A hardcoded RGB triangle. No vertex buffers: positions and colors are indexed
// by SV_VertexID. Compiled to both SPIR-V and DXIL by build.rs; actually drawn
// starting in Phase 1.

struct VSOut
{
    float4 pos : SV_Position;
    float3 color : COLOR0;
};

[shader("vertex")]
VSOut vsMain(uint vid : SV_VertexID)
{
    float2 positions[3] = { float2(0.0, 0.5), float2(0.5, -0.5), float2(-0.5, -0.5) };
    float3 colors[3] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) };

    VSOut o;
    o.pos = float4(positions[vid], 0.0, 1.0);
    o.color = colors[vid];
    return o;
}

[shader("fragment")]
float4 fsMain(VSOut i) : SV_Target
{
    return float4(i.color, 1.0);
}
