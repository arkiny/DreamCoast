# GI 디버그 뷰 표현 규칙 (distance-field/probe cache 위 GI 시각화)

상위: [gi-radiance-cache.md](gi-radiance-cache.md), [lumen-parity-swrt.md](lumen-parity-swrt.md).
목적: 카메라를 GDF 안으로 날려 hit마다 GI 상태(월드 radiance cache의 indirect irradiance)를 칠하는
플라이 가능한 디버그 뷰의 **표현(presentation)** 규칙을, 레퍼런스 렌더러가 distance-field/surface-cache/
world-probe 위에서 쓰는 유용한 관례에 맞춘다. 아래는 레퍼런스 소스에서 확인한 내용만 일반화해 기술한다.

## Modes (레퍼런스가 제공하는 GI/cache 시각화 모드)

레퍼런스는 화면 뷰를 그대로 재추적하는 "cone/ray march" 뷰와, cache 자체를 구슬(sphere)로 그리는
"probe" 뷰 두 계열을 가진다. 각 모드가 보여주는 것:

- **Scene(cache 최종 라이팅)** — GDF/카드를 재추적해 hit을 surface-cache 최종 라이팅(direct+indirect)으로
  칠한 "최종 GI가 반영된 씬" 뷰. lit radiance 이므로 tonemap 처리.
- **Surface-cache 커버리지** — 위와 같으나 cache가 없거나(coverage miss) cull된 표면을 형광색으로 마킹.
  레퍼런스 라벨: "Pink = missing surface-cache coverage, Yellow = culled surface-cache".
- **Albedo / Normal / Emissive / Direct-only / Indirect-only** — 커서 아래 카드를 뜯어 채널별로 표시.
  albedo=raw, normal=`(n+1)*0.5`, direct/indirect/emissive는 각 atlas를 그대로. 체크박스로 채널 토글.
- **Geometry Normals** — 재추적한 표면 노멀 자체(라이팅 없음, tonemap/sRGB 미적용).
- **World radiance-cache probe (radiance)** — clipmap probe를 구슬로 그리고, 구슬 노멀 방향으로 probe
  옥타헤드럴 radiance를 샘플해 칠함. 즉 "각 probe가 무엇을 보는가"의 방향별 표시.
- **World radiance-cache probe (irradiance)** — 위와 같으나 저장물이 irradiance일 때, 구슬 노멀로
  irradiance를 샘플 후 **작은 상수 albedo(≈0.3)를 곱해** "lit clay 구슬" 느낌으로 표시.
- **Probe sky-visibility** — probe당 sky 가시성(스칼라)을 grayscale 구슬로.
- **Num-frames-accumulated(수렴/우선순위 히트맵)** — probe/타일의 누적 프레임 수 등 수치장을,
  씬을 회색으로 깔고 그 위에 red-intensity 히트맵으로 lerp(scene_grey → red).

## Presentation conventions (색/노출/배경 관례)

- **lit radiance는 tonemap** — 씬과 색이 맞도록 후처리 tonemap을 재사용(eye-adaptation × color-grading LUT,
  아니면 `LinearToSrgb`). 라이팅 값은 `* PreExposure`를 곱해 노출 정합.
- **material 채널은 tonemap 미적용** — albedo만 sRGB, normal은 원시(`(n+1)*0.5`), tonemap 안 함.
- **shape("clay")** — 형상만 볼 땐 라이팅을 빼고 반-람베르트(예: `0.5*dot(L,N)+0.5`)로 중립 클레이. irradiance를
  "lit"하게 보이려면 **raw irradiance에 작은 중립 albedo를 곱함**(구슬 예시 ≈0.3) — 실제 텍스처 albedo가 아님.
- **수치장(numeric field)은 false-color/heatmap** — coverage/우선순위/누적프레임은 히트맵. 씬을 dim시켜
  깔고 red 강도로 lerp하거나(공간 문맥 유지), `frac(dist/100)`·`steps/100` 류의 램프.
- **direct vs indirect split** — 최종 라이팅을 direct-only / indirect-only 채널로 분리해 각각 별 모드로.
- **overlays / 마커** — probe는 구슬 인스턴스로 위치 표시. **invalid probe = red**(또는 magenta/pink),
  culled = bright green, adaptive probe = cyan `(0,1,1)`, world-offset이 걸린 probe = green 틴트 가산.
  surface-cache miss = pink, culled tile = yellow. clipmap 레벨은 레벨별 인덱스로 각각 그려 분리.
- **디버그 텍스트** — 커서 아래 값을 shader-print로 출력(예: LinearDiffuseColor), 예산 초과는
  white→yellow→red 글자색으로 경고.

## 실전 기본값 (defaults)

- **노출/게인** — 디버그 뷰도 씬과 동일 노출(`PreExposure` 곱 + 동일 tonemap)로 색 정합. tonemap off일 땐 sRGB.
- **miss 배경** — ray가 아무 것도 못 맞히면 sky를 tonemap해서 배경으로. radiance-cache가 켜져 있으면
  miss 방향은 cache/sky radiance로 채워 "구멍" 대신 하늘.
- **둥근 타일 클리핑** — overview 타일은 12px 라운드 보더로 클립(전체화면 뷰는 클립 안 함).

## DreamCoast 매핑 (world-radiance-cache-on-GDF 뷰에 먼저 넣을 2–4 모드)

우리 뷰는 카메라를 GDF로 march해 hit마다 clipmap world-probe의 indirect irradiance를 칠한다. 위 관례를
바탕으로 처음 넣을 값 있는 모드:

1. **Mode 0 — raw indirect irradiance (grayscale/linear)**: hit에서 clipmap probe irradiance를 그대로.
   albedo 미곱, tonemap 대신 `PreExposure` 곱 후 sRGB — GI 세기/누수를 albedo에 가려지지 않게 본다. (기준점)
2. **Mode 1 — irradiance × neutral clay albedo ("lit clay")**: 동일 irradiance에 중립 상수 albedo(≈0.3)를
   곱해 lit 느낌 + 형상 판독. 레퍼런스의 irradiance-probe 표시와 동일 관례(작은 상수 albedo).
3. **Mode 2 — clipmap-level 색상 코딩**: hit을 커버하는 clipmap 레벨을 레벨별 틴트로. probe 밀도/레벨 전환
   경계와 커버리지 구멍을 즉시 확인 — GDF march와 clipmap 정합 버그 잡기에 가장 실전적.
4. **Mode 3 — direct/indirect split (선택)**: 최종 라이팅을 direct-only vs indirect-only로 토글.
   실내가 어두운 게 GI 부재인지 direct 부재인지 분리. + coverage-miss는 **pink**, 무효 probe는 **red**로 마킹.

miss=sky tonemap 배경, 모든 모드 `PreExposure` 정합, mode 0/1 기본 제공. clipmap 색코딩(2)과 split(3)은
버그 진단용 후속. (레퍼런스에서 확인된 규칙만 반영.)

---

## Implemented — GI-on-distance-field view (`P_WRC_VIZ`)

`wrc_view.slang` + `GiSystem::record_wrc_view`: a full-screen debug pass that marches the
camera ray into the scene global distance field (reusing the shared `bs_scene_march` /
`bs_scene_normal`) and, at each hit, reconstructs the world radiance cache's stored indirect
irradiance for the hit normal (`wrc_irradiance` in `wrc_common.slang`). It runs its own world
radiance-cache update first (so the cache is populated), then paints it onto the distance-field
geometry — showing the GI state spatially, flyable.

- `P_WRC_VIZ=1` — enable (replaces the tonemap source, like the other GDF debug views).
- `P_WRC_VIZ_MODE` — 0 = raw indirect irradiance (grayscale), 1 = irradiance × neutral clay
  albedo (lit-clay look, the default).
- `P_WRC_VIZ_GAIN` — sets the mid-point of the tone compression (default 1.0).

The cache holds high-dynamic-range lighting (sun-lit surfaces are orders of magnitude brighter
than shadowed ones), so a single exposure crushes shadows to black and blows highlights to pure
white — the GI structure is lost. The view compresses the **luminance** with a Reinhard curve
(asymptotes to 1, so nothing clips to pure white) while **preserving chroma**, so a bright neutral
wall reads light-grey and the coloured indirect bounces stay legible everywhere (e.g. the curtain
colours bleeding onto the floor). The host tonemap runs with exposure 1 on top; a subtle
normal-based form cue keeps the blobby distance-field surfaces reading as 3D shapes.

**No pure black — matched to the reference source.** Two things made the earlier view read as
dead-black holes: (1) ray MISSES painted a near-black background, and (2) the world cache stores
BOUNCE-ONLY radiance — rays escaping to the sky contribute 0 (the real pipeline adds sky/IBL in a
separate pass), so a surface lit mostly by skylight integrates to ~0. Verified against the
reference visualization source: on a miss it paints the **sky radiance**, and at a hit it adds the
skylight **weighted by how much the surface sees the sky** (an OCCLUDED skylight, not a flat fill),
plus a small "skylight leaking" floor so fully enclosed surfaces are dim rather than pure black.
Matched here: the world cache now stores the per-direction **sky visibility** in its tile alpha
(the trace already knows whether each ray escaped to the sky); the view reconstructs the
cosine-weighted sky visibility with `wrc_gather` and adds `sky · max(skyvis, SKY_LEAK) · SKY_STRENGTH`
— open surfaces get the sky, enclosed interior stays correctly darker (not a uniform wash), and the
`SKY_LEAK` floor keeps it from ever reading pure black. Misses paint the sky gradient.

Notes: the look reflects the actual data resolution — the distance field (~48³) resolves as
blobby surfaces and the world radiance cache (16³ probes/level) makes the GI blocky; deep interior
reads dark where the coarse 1-bounce cache has little GI (the honest state). Gallery byte-identical
with no env (opt-in); deterministic (the cache update has no RNG). Metal-verified; Windows DX≡VK
pending.
