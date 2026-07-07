# Bent-normal AO ⟷ skylight integration

Goal: move AO/skylight occlusion from a **scalar** multiply to a **directional (bent-normal)**
integration, matching the reference engine's DFAO → skylight coupling. The AO stage already
produces the *unoccluded average direction* (bent normal); the diffuse skylight is sampled along
that direction (not the surface normal) and attenuated by the sky-visibility scalar. Adds GTAO
multi-bounce energy return and bent-normal specular occlusion.

Trademark-free: the reference engine is cited generically ("reference engine"). All new paths are
content-only opt-in seams; the gallery golden `af70c1a5` stays byte-identical.

## Key insight (why this is nearly free)

Our indoor skylight occlusion already projects a **scalar directional sky-visibility** onto SH
band-0/1 at each GI probe (`gi_volume.slang`):

```
vis = hit ? 0.0 : 1.0;         // 1 on a sky escape, 0 on a geometry hit
sv0 += vis * SH_C0;
sv1 += vis * (SH_C1 * dir.y);
sv2 += vis * (SH_C1 * dir.z);
sv3 += vis * (SH_C1 * dir.x);
```

The band-1 vector `(sv3, sv1, sv2)` is exactly `∫ V(ω)·ω dω` — the visibility-weighted average
unoccluded direction. That is **the DFAO bent normal by construction**. Reference DFAO builds the
identical quantity as `Σ ConeVisibility·ConeDir` and defines `SkyVisibility = length(BentNormal)`
(`DistanceFieldScreenGridLighting.usf:404-417`, `DistanceFieldLightingPost.usf:382`).

So the bent normal costs **zero extra tracing** — it is recovered from SH coefficients we already
sample. The `gi_skyvis` image is `Rgba16Float` and currently uses only `.r` (the scalar V); the
free `.gba` channels carry the world-space bent normal.

## Reference-engine grounding (verbatim, extracted from the source tree)

**Skylight sampled along the bent normal** — `SkyLightingDiffuseShared.ush:41-48,79-81,130,133`:

```
SkyVisibility        = length(BentNormal);
NormalizedBentNormal = BentNormal / max(SkyVisibility, 1e-5);
BentNormalWeightFactor = SkyVisibility;                                   // more bent normal in corners
SkyLightingNormal    = lerp(NormalizedBentNormal, WorldNormal, BentNormalWeightFactor);
DotProductFactor     = lerp(dot(NormalizedBentNormal, WorldNormal), 1, BentNormalWeightFactor);
SkyDiffuseLookUpMul  = SkyVisibility * DotProductFactor;
SkyDiffuseLookUpAdd  = (1 - SkyVisibility) * OcclusionTint;
DiffuseLookup        = GetSkySHDiffuse(SkyLightingNormal) * SkyColor;
Lighting += (SkyDiffuseLookUpMul * DiffuseLookup + SkyDiffuseLookUpAdd) * DiffuseColor;
```

Note this reduces to a plain `irradiance(n) * V + tint*(1-V)` when the bent normal aligns with `n`
and has length V — i.e. **our current `occlude_sky_diffuse` is the isotropic special case**.

**GTAO multi-bounce** — `DeferredShadingCommon.ush:1364`:

```
half3 AOMultiBounce(half3 BaseColor, half AO) {
    half3 a = 2.0404 * BaseColor - 0.3324;
    half3 b = -4.7951 * BaseColor + 0.6417;
    half3 c = 2.7552 * BaseColor + 0.6903;
    return max(AO, ((AO * a + b) * AO + c) * AO);
}
```

**Specular occlusion (scalar)** — `ReflectionEnvironmentShared.ush:131`:

```
half GetSpecularOcclusion(half NoV, half RoughnessSq, half AO) {
    return saturate( pow( NoV + AO, RoughnessSq ) - 1 + AO );
}
```

**Bent-normal cone-cone specular occlusion** — `SkyLightingShared.ush:9-19,25-64`:

```
float ApproximateConeConeIntersection(float ArcLength0, float ArcLength1, float AngleBetweenCones) {
    float AngleDifference = abs(ArcLength0 - ArcLength1);
    return smoothstep(0, 1, 1 - saturate((AngleBetweenCones - AngleDifference)
                                         / (ArcLength0 + ArcLength1 - AngleDifference)));
}
// ReflectionConeAngle = max(Roughness,0.1)*PI; UnoccludedAngle = length(BentNormal)*PI*InvStrength;
// AngleBetween = acos(dot(BentNormal, ReflVec)/length(BentNormal));
// IndirectSpecularOcclusion = ApproximateConeConeIntersection(ReflectionConeAngle, UnoccludedAngle, AngleBetween);
```

## Stages (each: content-only seam, gallery byte-identical, own commit)

1. **Producer** — `gdf_gi.slang` writes the bent normal (SH band-1 vector, normalized; fall back to
   the surface normal when the vector is degenerate) into `gi_skyvis.gba`. No pixel change (nothing
   reads `.gba` yet) → content + gallery byte-identical. `P_BENT_NORMAL` (gdf_gi `flip_y` bit1,
   default on) can zero it for A/B.
2. **Consumer** — `pbr.slang` reads the bent normal, builds the reference `SkyLightingNormal` +
   `DotProductFactor`, samples the irradiance cube along it, scales by `V*dotFactor` and adds the
   existing neutral tint leak. Scalar fallback (bent≈0, e.g. the screen-probe producer or gallery)
   is byte-identical to the current `occlude_sky_diffuse`. Gallery (skyvis unbound) untouched.
3. **GTAO multi-bounce** — apply `AOMultiBounce(albedo, ao)` to the diffuse AO term. Opt-in default
   off (recolors gallery AO<1 pixels), `P_AO_MULTIBOUNCE`.
4. **Specular occlusion** — `GetSpecularOcclusion` + bent-normal cone-cone occlusion on the specular
   ambient. Opt-in default off; the reflection track's "AO does not multiply mirror specular" rule
   stands — this is a physically distinct bent-normal term, gated separately.
5. **Reconcile** — the bent normal *reuses* the SH sky-vis volume (no second computation). Docs +
   DX≡VK Windows follow-up.

## Gates

- Gallery golden `af70c1a5` byte-identical (all stages).
- Path-tracer parity (`P8_PATHTRACE=1`, `tools/rt-compare.py`) is the lighting success metric;
  indoor crevice/curtain directionality should move toward the PT reference.
- Determinism (run-to-run bit-identical); integer-hash cross-backend safety preserved.
- DX≡VK Windows parity is a follow-up (Metal-verified here).
