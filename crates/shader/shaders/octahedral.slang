// Octahedral direction <-> square mapping for the screen-space radiance probes.
//
// The full-sphere octahedral map folds the unit sphere into the square [-1,1]^2 (area-
// preserving enough for low-res probes; equal-area refinement is a later option). A probe
// stores its incoming radiance as an OCT x OCT atlas tile; the gather decodes each texel
// to a world direction (trace) and re-encodes a normal to look it up (integrate). Matches
// the standard `UnitVectorToOctahedron` / `OctahedronToUnitVector` convention.

// Square coordinate in [-1,1]^2 -> unit direction.
float3 oct_to_dir(float2 oct)
{
    float3 n = float3(oct.x, oct.y, 1.0 - abs(oct.x) - abs(oct.y));
    float t = max(-n.z, 0.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

// Unit direction -> square coordinate in [-1,1]^2.
float2 dir_to_oct(float3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 oct = n.xy;
    if (n.z <= 0.0)
    {
        oct = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return oct;
}

// Texel index [0,res) in one probe tile -> octahedral square coord (texel center).
float2 oct_texel_to_square(uint2 texel, uint res)
{
    float2 uv = (float2(texel) + 0.5) / float(res);
    return uv * 2.0 - 1.0;
}

// Direction -> fractional texel coordinate [0,res] within a probe tile (for bilinear).
float2 oct_dir_to_texel(float3 dir, uint res)
{
    float2 sq = dir_to_oct(dir);
    float2 uv = sq * 0.5 + 0.5;
    return uv * float(res);
}
