// Shared GGX reflection sampling for the stochastic reflection track.
//
// The trace (`gdf_reflect.slang`) casts ONE GGX-importance-sampled ray per pixel; the spatial
// ratio-estimator resolve (`reflect_resolve.slang`) reconstructs each neighbour's ray direction +
// pdf from the SAME deterministic per-pixel jitter (so it can reweight borrowed neighbour samples
// by `pdf_p(dir)/pdf_q` under the centre pixel's own lobe — Stachowiak's ratio estimator, the same
// scheme `ssr_resolve.slang` uses for screen traces).
//
// SINGLE SOURCE: the sampler + rng here MUST stay bit-identical to `gdf_reflect.slang`'s inline
// `sample_ggx`/`pcg_hash`/`rnd01`/`onb` and the ray-gen jitter (`rng = tid.x*1973 + tid.y*9277 +
// frame*26699`), or the resolve reconstructs the wrong direction. `gdf_reflect` keeps its own inline
// copy (so its compiled output — the gallery byte-identical anchor — is untouched); this file is the
// canonical definition the resolve includes. Any change to the sampler must be mirrored in both.
// `ggx_reflect_pdf` matches `ssr_resolve.slang`'s pdf so both resolves weight identically.

#ifndef REFLECT_GGX_INCLUDED
#define REFLECT_GGX_INCLUDED

static const float REFL_GGX_PI = 3.14159265359;

uint refl_pcg_hash(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}
float refl_rnd01(inout uint s) { return float(refl_pcg_hash(s) & 0xFFFFFFu) / float(0x1000000); }
void refl_onb(float3 n, out float3 t, out float3 b)
{
    t = normalize(abs(n.x) > 0.9 ? cross(n, float3(0, 1, 0)) : cross(n, float3(1, 0, 0)));
    b = cross(n, t);
}
// GGX NDF importance sample: a microfacet half-vector around the surface normal. Bit-identical to
// `gdf_reflect.slang::sample_ggx`.
float3 refl_sample_ggx(float3 n, float roughness, inout uint rng)
{
    float a = max(roughness * roughness, 1e-3);
    float u1 = refl_rnd01(rng), u2 = refl_rnd01(rng);
    float phi = 2.0 * REFL_GGX_PI * u1;
    float ct = sqrt((1.0 - u2) / (1.0 + (a * a - 1.0) * u2));
    float st = sqrt(saturate(1.0 - ct * ct));
    float3 t, bvec;
    refl_onb(n, t, bvec);
    return normalize(t * (st * cos(phi)) + bvec * (st * sin(phi)) + n * ct);
}

// Reconstruct the exact reflection ray direction the trace cast for pixel (px, py) at `frame`.
// Mirrors `gdf_reflect.slang` ray-gen: `rng = px*1973 + py*9277 + frame*26699 | 1`, GGX half-vector,
// `reflect(-v, h)`, with the upper-hemisphere fallback. `v` = normalized surface->camera direction.
float3 refl_ggx_dir(float3 n, float3 v, float roughness, uint px, uint py, uint frame)
{
    uint rng = (px * 1973u + py * 9277u + frame * 26699u) | 1u;
    float3 h = refl_sample_ggx(n, roughness, rng);
    float3 dir = reflect(-v, h);
    if (dot(dir, n) <= 0.0)
        dir = reflect(-v, n); // keep the ray in the upper hemisphere
    return dir;
}

float refl_ggx_d(float ndh, float alpha)
{
    float a2 = alpha * alpha;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (REFL_GGX_PI * d * d + 1e-9);
}
// pdf of a reflected direction `l` under GGX NDF importance sampling at this surface (n, v,
// roughness). Bit-identical to `ssr_resolve.slang::ggx_reflect_pdf`.
float refl_ggx_pdf(float3 n, float3 v, float3 l, float roughness)
{
    float3 h = normalize(v + l);
    float ndh = max(dot(n, h), 1e-4);
    float vdh = max(dot(v, h), 1e-4);
    float alpha = max(roughness, 0.04);
    alpha *= alpha;
    return refl_ggx_d(ndh, alpha) * ndh / (4.0 * vdh);
}

#endif // REFLECT_GGX_INCLUDED
