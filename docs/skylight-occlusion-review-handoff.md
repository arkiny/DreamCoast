# 핸드오프: 차폐부 스카이라이트 누출 재점검 (AO · 그림자 · 스카이라이트 오클루전)

DreamCoast 엔진(Rust/raw-RHI, macOS=Metal, repo: `/Users/arkiny/GitRepos/DreamCoast`)에서 **차폐된
곳(특히 Intel Sponza 사자부조 relief 크레비스)에 스카이라이트(IBL diffuse)가 과하게 들어오는** 현상을
근본원인 관점에서 재점검한다. 참고 엔진 소스는 `/Users/arkiny/GitRepos/UnrealEngine-1`. **트레이드마크명
금지 — 문서/주석/커밋엔 "reference engine"으로만 표기** ([[dreamcoast-no-trademark-names]]).

## 관찰 (현상)
- 실내 차폐부(사자부조 relief의 오목한 홈, 기둥 뒤, 아치 안쪽)가 물리적으로 하늘을 거의 못 보는데도
  **밝은 스카이라이트가 그대로 들어와 평평/밝게** 보인다. 오클루전이 부족해 입체감·그늘이 약하다.
- 즉 **diffuse 스카이라이트(IBL irradiance)가 차폐부에서 충분히 감쇠되지 않는다**는 의심.

## 목표 (한 줄)
AO · 그림자 · 스카이라이트 오클루전이 **어떻게 계산·합성되는지 다시 확인**하고, 차폐부에서 스카이라이트가
새는 **근본원인을 파악**한 뒤(추측 금지, 근거 기반), reference engine 대비 부족한 지점을 규명한다. 이 문서는
**조사·진단용 핸드오프**이며, 곧바로 구현하지 말고 먼저 원인을 측정·확정할 것.

## 현재 파이프라인 (재점검 대상 — file:line)

### 1. AO 두 계층 (near × far, DIFFUSE 앰비언트에만 곱함)
- `crates/shader/shaders/gdf_ao.slang` — **원거리 distance-field AO**(DFAO 상당). GDF 콘/마치, 5-tap IQ AO,
  지수 감쇠 + `AO_FLOOR` 바닥. `apps/sandbox/src/gi.rs::record_ao`. env `P11_GDF_AO`, `AO_REACH`(월드 단위
  reach), `AO_STRENGTH`, `AO_FLOOR`, 해상도 `P_AO_RES_DIV`(Apple 타일 quarter-res + joint-bilateral 업샘플).
  **핵심 확인: `reach`(gi.rs:711)가 접촉(~0.5m급)만 커버하는지 — relief 홈 스케일보다 짧으면 홈 전체를 못
  어둡게 함.**
- `crates/shader/shaders/gtao.slang` — **근거리 스크린 AO(HBAO-lite obscurance)**, env `SSAO`
  (+`SSAO_RADIUS/INTENSITY/BIAS/POWER`), 정수-해시 회전(DX≡VK 결정적) + separable depth-aware blur.
- 합성: `pbr.slang`에서 `gdf_ao *= ssao`(near×far), 그리고 **DIFFUSE 앰비언트에만** 곱함(스페큘러 반사는
  자체 오클루전). `pbr.slang::ambient_ibl` 마지막 `kd * diffuse * diff_ao + specular` 참고.

### 2. 스카이라이트 오클루전 (SH-L1 sky-vis + bent normal, 이번 세션 작업)
- **생산**: `crates/shader/shaders/gi_volume.slang` — GI 프로브가 스칼라 방향성 sky-visibility(hit=0/miss=1)를
  SH band-0/1(4 vol)로 투영. **`GI_VOL_DIM = 32`(gi.rs:37)** — ★ 여기가 유력한 병목: 37m 씬을 32³로 나누면
  ≈1.2m/복셀 → 사자부조의 ~10–30cm 크레비스를 전혀 해상 못함 → 오목한 relief도 "하늘 잘 보이는" 프로브를
  읽어 V가 높게 남음 → 스카이라이트가 샌다.
- **재구성**: `gdf_gi.slang` — SH-L1에서 코사인-가중 hemispherical sky-vis `V(n)` + **bent normal**(band-1
  벡터 = ∫V·ω, 이번 세션)를 `gi_skyvis` 이미지(.r=V, .gba=bent)로 출력.
- **적용**: `pbr.slang::occlude_sky_diffuse_bent` — IBL diffuse(irradiance cube)를 bent normal 방향으로 샘플 +
  `V·dotFactor`로 감쇠 + 차폐분은 neutral OcclusionTint leak(`skyvis_tint`/`P_SKYVIS_TINT`) + MinOcclusion
  바닥(`skyvis_min_occ`/`P_SKYVIS_MIN_OCC`, =1이면 오클루전 off). sky_vis=1 & bent=0 → 정확한 no-op(갤러리
  앵커). 배경/도출은 [[dreamcoast-permesh-df-plan]](SH-L1 sky-vis) + [[dreamcoast-bent-normal-ao-skylight]].

### 3. 그림자 (DIRECT 태양만, 스카이라이트는 미적용)
- `pbr.slang` — CSM 아틀라스 / 단일 shadow map(`csm_params`, `sun_shadow()`), 캐시드 shadow(A6). **태양 직접광**만
  그림자 처리. **스카이라이트(앰비언트 IBL diffuse)는 shadow map으로 가려지지 않음** — 오직 AO(위 1)와 sky-vis
  (위 2)로만 감쇠. 그래서 태양-그림자 크레비스도 AO/sky-vis가 약하면 스카이라이트가 가득 든다.

## 유력한 근본원인 가설 (측정으로 확정할 것 — 추측 금지)
1. **★ sky-vis 볼륨 32³ 과소해상(최우선)**: relief 크레비스가 프로브 격자보다 훨씬 작아 V가 안 떨어진다.
   메모리 [[dreamcoast-permesh-df-plan]]에도 "coarse 32³ 볼륨" 한계가 반복 등장. reference engine은 **per-pixel
   DFAO bent-normal**(distance-field 트레이스, 프로브 격자보다 훨씬 고해상)으로 스카이라이트를 크레비스 스케일로
   가린다 — 우리는 저해상 볼륨.
2. **AO reach가 접촉만 커버**: `gdf_ao`가 ~0.5m급이면 relief 홈 전체(수십 cm~m)를 못 어둡게 함. 스카이라이트에
   적용되는 far-field 오클루전이 사실상 coarse sky-vis뿐.
3. **MinOcclusion/Tint leak가 과함**: `skyvis_min_occ`/`skyvis_tint` 기본값이 차폐부를 너무 밝게 남길 가능성.
4. **bent-normal 스카이라이트가 과밝게**: 이번 세션 도입분이 차폐부에서 bent 방향으로 밝은 하늘을 끌어오지
   않는지 A/B(`P_BENT_NORMAL=0`) 확인.
5. **스카이라이트가 태양-그림자와 무관**: 설계상 맞지만, 실내에선 sky-vis/AO가 유일한 방어선이라 이들이 약하면
   바로 누출.

## 조사 절차 (방법론 — 이 세션에서 검증됨)
**추측 말고 측정 + reference 소스 정밀 추출.** 
1. **재현/측정 (Metal)**: 사자부조가 보이는 카메라로 캡처하고 진단 뷰로 성분 분리.
   - `DEBUG_VIEW=13`(sky-vis V), `14`(bent normal), `9`(gdf_ao), `6`(material AO), `8`(IBL ambient), `10`(GI).
   - relief 크레비스 픽셀의 **V가 실제로 높은지**(→ 원인 1 확정), gdf_ao가 홈을 덮는지(원인 2) 수치로 확인.
   - 레버 A/B(코드 변경 없이): `AO_REACH`↑ / `AO_STRENGTH`↑ / `P_SKYVIS_MIN_OCC=0` / `P_BENT_NORMAL=0` /
     `GI_VOL_DIM` 상향(있으면) → 어느 것이 크레비스를 어둡게 하는지로 원인 순위 매김.
   - 반복 리트: `EV100=11 LEVEL=sponza_intel CAM_EYE=... CAM_TARGET=...`(사자부조 프레이밍은 메모리
     [[dreamcoast-intel-sponza-asset]]의 lion-view 캡처 cmd 참고; RELEASE 빌드 + GI 워밍업 필요).
2. **reference engine 추출 (Explore 에이전트, verbatim file:line)**: 
   - 차폐부 스카이라이트를 무엇으로 가리는가 — **DFAO(distance-field AO) bent-normal의 해상도/스케일**, GTAO의
     스카이라이트 적용, per-pixel vs 프로브-볼륨. `DistanceFieldAmbientOcclusion*.usf`,
     `PostProcessAmbientOcclusion.usf`(GTAO), 스카이라이트가 AO/bent-normal로 감쇠되는 지점
     (`SkyLightingShared.ush`/`SkyLightingDiffuseShared.ush` — 이번 세션에서 이미 일부 추출, 메모리 참조).
   - 특히 "실내 크레비스에서 diffuse 스카이라이트를 어떤 **해상도**로 가리는가"(우리 32³ 볼륨 vs 그들의 per-pixel
     DF/GTAO)에 초점.
3. **원인 확정 후** 방향 제안(구현은 별도): 예) per-pixel sky-vis(AO 패스가 sky-vis도 산출 / screen-space 또는
   DF-traced), 또는 sky-vis 볼륨 고해상/클립맵, 또는 AO reach·스카이라이트 결합 강화. 각 방향의 gallery
   byte-identical seam + PT 잔차 영향 명시.

## 게이트 / 검증
- **gallery 골든 `af70c1a5` byte-identical** 필수 — 모든 신규 경로는 content-only(gi_volume/AO on) seam,
  갤러리는 스칼라·no-op 레거시 유지. (`python3 tools/golden-image.py --only gallery --backend metal`)
- **경로-트레이서 패리티**(`P8_PATHTRACE=1` … 단, 콘텐츠 씬은 HW 패스트레이서 BLAS 없음 → 갤러리만 유효.
  콘텐츠는 육안 + 성분 분리 + reference 정합으로 판단). 차폐부 스카이라이트가 물리적으로 옳게 줄어드는지.
- 과다 오클루전(차폐부가 새까매짐) 주의 — MinOcclusion/tint로 조절, 실내 GI 바운스로 채우는지 병행 확인.
- DX≡VK Windows 파리티는 후속(이번 라인 Metal 검증).

## 참고 (메모리·문서)
- [[dreamcoast-permesh-df-plan]] — SH-L1 sky-vis + **skylight occlusion v2**(현재 적용분)의 배경·한계(coarse
  32³ 볼륨, 크레비스 미해상)가 이미 상세. **필독.**
- [[dreamcoast-bent-normal-ao-skylight]] — 이번 세션 bent-normal AO/skylight(V·dotFactor, tint, min_occ) +
  GTAO 멀티바운스 + 스페큘러 오클루전. UE DFAO/GTAO/skylight verbatim 추출 결과 포함.
- [[dreamcoast-gdf-ao-flicker]] — gdf_ao 깊이 load/store 픽스(차폐부 AO 신뢰성).
- `docs/ao-skylight-bent-normal.md`, `docs/ao-skylight-handoff.md`(이 트랙의 원 핸드오프).

## 규칙 (CLAUDE.md)
근본원인 수정 · opt-in seam · 기본 byte-identical · 3백엔드 파리티(Metal 후 DX≡VK) · 단일 진실원 · 상용
트레이드마크명 금지("reference engine"). 커밋 끝 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
