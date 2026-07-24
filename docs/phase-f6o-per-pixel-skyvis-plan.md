# F6O 계획서 — 픽셀당 스카이-가시성 (bent-normal), 프로브-해상도 블록 근본 수리

> 상태: **계획 — 착수 전 승인 대기**. 선행: F6N(소비-측 소프트 witness = 실측 무효·되돌림,
> `phase-f6n-stochastic-occupancy-plan.md` §4c에 층 정정 기록). 본 페이즈 = **올바른 층(프로듀서)**
> 수리. 참고: 상용 렌더러("reference engine")의 화면-공간 bent-normal / 거리장 스카이 오클루전
> 기법(상표명 문서·주석·커밋 금지 — [[dreamcoast-no-trademark-names]]).

## 0. 요약

**근본(실측 확정, `phase-f6n` §4c)**: 사용자의 "검은 Block"(부조 주변 벽의 하드 사각 암부)은
**스카이-가시성(V) 필드의 프로브-해상도 양자화**다. DEBUG_VIEW 항별 분리로 확정:
- DEBUG_VIEW=10(GI 볼륨)·9(GDF AO)·6(AO): 매끄러움, 블록 없음.
- **DEBUG_VIEW=8(IBL 앰비언트): 블록 존재.**

V는 `gi_volume.slang`이 **이진 per-ray 마치**(1=하늘 탈출, 0=히트)로 **프로브당** 계산해 SH 볼륨에
저장하고, `pbr.slang`이 `skyvis_index` 이미지 + bent normal로 IBL diffuse를 오클루드한다
(`occlude_sky_diffuse_bent`). 소비-측 수리 2종(F6N 소프트 witness, trapped-fill)은 **둘 다 실측
무효** → 블록은 소비가 아니라 **저장된 V 필드**(프로브 해상도) 탓. fineOFF=큰 블록/fineON=작은
블록 = 해상도 양자화 지문(카메라-앵커 fine 볼륨이라 뷰-의존).

**핵심 관찰**: DreamCoast의 소비(`occlude_sky_diffuse_bent`, pbr.slang:204-253)는 **이미
reference engine의 bent-normal 스카이 diffuse와 동형**이다(SkyLightingNormal=lerp(bent,n,V),
V·DotProductFactor·tint leak). 유일한 갭은 **bent+V의 출처**: DreamCoast는 프로브 볼륨(블록),
reference engine은 **픽셀당** 화면-공간/거리장 트레이스. 즉 **소비는 이미 맞고, 프로듀서만 프로브
→ 픽셀당으로 바꾸면 근본 해결**.

**본 페이즈**: skyvis 이미지(V + bent normal)를 **픽셀당 GDF 헤미스피어 트레이스**로 생성한다
(reference engine의 ray-sampled bent-normal 방식). 소비(pbr)는 **무변경**(이미 bent+V 계약).
gdf_ao(픽셀당 GDF 트레이스)가 이미 매끄러움(DEBUG_VIEW=9 블록 무)이 이 기전을 입증한다.

**비목표**: GI 볼륨(E)·GI 방식 자체(별도)·반투명·HW-RT 경로. 커버리지-α 필드는 **얇은 지오메트리**용
별도 트랙(본 블록은 두꺼운-벽 해상도라 부적합 — §4c).

## 0.5 Fable 검증 반영 — 설계 정정 (REVISE → 권위 있는 개정)

초안의 "소비 무변경" 전제는 **부분적으로 틀렸다**(Fable). 아래가 개정 확정:

1. **채널 순서**: 출력은 `float4(V, bent.xyz)`(pbr.slang:726 `sv.r`=V·`sv.gba`=bent; 전 프로듀서
   동일 — gdf_gi.slang:415/554/631). 초안의 `(bent.xyz, V)`는 오류.
2. **`|bent|` = 유효도(validity), V 아님(F6L 계약)**: pbr.slang:167-174·198·239-241은 `|bent|`을
   **신뢰도**로 읽어 낮으면 스칼라 경로로 페이드. 픽셀당 프로듀서가 `|bent|=V`로 쓰면 실내(V≈0.05-0.3)
   에서 방향성 수리가 **자기-비활성화**. 게다가 벡터 denoise가 |bent|<V로 축소(삼각부등식). ⇒ **V와
   유효도를 분리**: V는 `.r`(코사인-가중 스칼라 가시성), bent(`.gba`)는 **방향 × 유효도**(수렴 직접
   추정은 유효도≈1). 증분 1은 **스칼라 V만**(bent=0 → 소비자 스칼라 폴백 `irradiance(n)·V`), 방향성
   bent는 증분 1b(유효도 인코딩·denoise 별도 설계).
3. **V = 코사인-가중** 헤미스피어(pbr.slang:139-141·SH 재구성과 정합). 균일-헤미스피어 금지.
4. **볼륨은 계속 생성**: sdf_cache_light.slang:365-462·gdf_reflect.slang:1261/1421/1450이 sky-vis
   **볼륨**을 월드-공간에서 샘플 — 스크린 이미지 소비 불가. 픽셀당 V는 **디퍼드 스카이라이트(pbr)에만**
   공급, 볼륨은 세계-공간 소비자용으로 **유지**. §5 "볼륨 은퇴"·§3 "비용 상계"는 철회(별도 절감 없음).
5. **증분-1 게이트 = 진단-전용**: V 재분포가 봉인 스카이 캘리브를 연다(F6H: V 수리→sky 항 개방→bias
   +40, tint 0.15/sky_gain 6.0이 붕괴-V에 적합). ⇒ 증분-1 knob-ON은 **게이트 면제·진단**(판정 =
   매칭 EV11에서 블록 소멸). improved-or-neutral은 **증분-2(재캘리브) 후** 적용. **two-master 문제**:
   tint/min_occ가 픽셀당 V(디퍼드)와 볼륨 V(캐시·반사)를 동시에 섬김 → 증분-2가 두 분포를 함께 정합.
6. **별도 `gdf_skyvis.slang`**(골든-게이트 gdf_ao 불변 — FP-민감 셰이더 편집 금지 선례). 스카이 픽셀
   `float4(1,0,0,1)`(depth≥1) 계약 복제. **해상도**: 픽셀당 V는 실루엣 불연속 → gdf_gi_upsample 조인트-
   바이래터럴 업샘플 경유 또는 풀-res. 커버리지 경계(클립맵 밖 → V=1 vs 볼륨 AABB 클램프) 거동 명시.
   히어로 `P_SKYVIS_BENT_FLOOR=0.25` 재검증(V 재분포).
7. **기전 증거 정정**: "gdf_ao 매끄러움"은 약함(5-탭 법선-방향 휴리스틱). 강한 증거 = **갤러리 gdf_gi
   레이-마치 경로**(동일 GDF 마치·denoise·블록 무). 단 N=1~4 이진 레이는 고분산 → denoise 의존,
   클립맵 레벨-심 밴딩·필드 결함(F6H 비수밀 부호)은 잔존 가능(증분 2 예상).

## 1. reference-engine 기법 (수학, 무-상표)

ray-sampled bent normal (화면-공간 bent-normal 셰이더의 핵심):
```
UnoccludedSum = 0; Bent = 0; Vsum = 0;
for k in 0..N:                       // 픽셀당 헤미스피어 방향(코사인/균일 + 블루노이즈 지터)
    dir = sample_hemisphere(n, rand_k)
    visible = trace(worldpos, dir) escapes-to-sky ? 1 : 0   // GDF 마치
    Bent += dir * visible
    Vsum += visible
V = Vsum / N                          // 스칼라 스카이 가시성
if (length(Bent) > 0) Bent = normalize(Bent) * V   // 방향 × V  (= DreamCoast F6L 계약과 동일)
```
- `length(Bent) == V`, 방향 = 평균 미차폐 방향. **DreamCoast F6L bent 계약(방향×유효도)과 정확히
  일치** → pbr 소비 무변경.
- 몇 개(N=1~4) 레이 + **블루노이즈 + 시간 누적 + 공간 필터**로 저비용·저노이즈(reference engine
  방식). DreamCoast는 gdf temporal + à-trous denoiser를 **이미 보유**(gdf_temporal/gdf_atrous) —
  재사용.

## 2. 방법 (사전 등록)

### 2a. 프로듀서 — 픽셀당 sky-vis 패스
- **재사용**: `gdf_bounce.slang`의 GDF 마치(gdf_gi가 쓰는 것과 동일), gdf_ao의 픽셀→월드 재구성 +
  two-sided facing 계약(F6M) 그대로.
- **신규 셰이더** `gdf_skyvis.slang`(또는 gdf_ao를 방향성으로 확장): 픽셀당 N 헤미스피어 레이 →
  위 bent 누적 → `float4(bent.xyz, V)` 출력. `bs.sky_term`/albedo 불요(가시성만).
- **denoise**: 출력 bent+V를 기존 gdf temporal + à-trous에 통과(반경/가중은 스카이-vis용으로 조정).
- **출력**: 현 `skyvis_index`(vol_b가 쓰던 이미지)를 **이 패스 출력으로 대체**. 볼륨 sky-vis
  (gi_volume 쓰기)는 knob OFF 시 유지(앵커), ON 시 이 패스가 skyvis를 공급.

### 2b. knob / 스테이징
- `P_SKYVIS_PP`(기본 OFF): 픽셀당 sky-vis ON. OFF = 볼륨 sky-vis(현 배송, 바이트-불변 앵커).
- **증분 1**: 픽셀당 **V + bent** 생성·denoise·skyvis 공급. 블록 소멸 확인(시각 + DEBUG_VIEW=8).
- **증분 2**: 스카이라이트 **재캘리브**(V 분포가 프로브-평균 → 픽셀-정확으로 바뀌며 절대 V가 이동 →
  tint/min_occ/sky_gain 재정합). F6G 재개방 후보.
- N(레이 수)·reach·denoise 반경은 **게이트·PROFILE_GPU 측정으로** 결정(스칼라 튜닝 금지).

### 2c. 무엇을 고치는가
- **블록**: 픽셀당 트레이스라 V가 화면 해상도 → 프로브-격자 양자화 소멸(gdf_ao 매끄러움이 입증).
- **뷰-의존**: 픽셀당이라 카메라-앵커 프로브 볼륨 재중심과 무관 → "다가가면 형태 변화" 소멸.
- **누출 blob**: 픽셀당 방향성 가시성이 프로브-누출을 대체.

## 3. 게이트 정책
- **knob OFF = 바이트-불변**(gallery/sc_viz/gdf_ao SHA + PT block64 20.74/27.21). 신규 패스는
  OFF 시 **미실행**(skyvis는 볼륨 공급) → 앵커 무변경 필수.
- **knob ON**: PT block64 게이트(sunlit 20.74/interior 27.21) **improved-or-neutral**. 블록이
  IBL 앰비언트(라이팅) 항이라 block64에 **직접 반영** — F6N(소비 inert)과 달리 게이트가 민감.
- **시각 A/B**: 부조 near/far 재캡처(eye -11/-2,1.2,0.79, target -15.84,1.2,0.79) — 블록·blob·
  뷰-시프트 소멸 확인. near vs fineOFF `mean|Δ|`(F6N 베이스라인 13.3)이 **의미 있게 하락**해야 함
  (픽셀당 V가 뷰-안정).
- **PROFILE_GPU**: 신규 패스 ms(레이 수·denoise) 측정·보고(house rule #2). 볼륨 sky-vis 업데이트
  절감분과 상계.
- **셔머/결정론**: CAPTURE_SEQ 정적 셔머 무회귀(시간 누적 도입 주의 — AE FIXED_DT 레시피 준수,
  [[dreamcoast-golden-gdf-ao-determinism]] 프레임-0 링버퍼 함정).
- **DX≡VK**: Metal 로컬 + Windows 보류 태그.
- **착지 규약**: V 재캘리브가 게이트를 악화시키면(F6H/F6M 크러치-캘리브 개방 클래스) 기본 OFF·knob
  잔존 + 히어로 옵트인. 개선/중립이면 기본 ON 후보(수치 기록, 버짓 하향-전용 래칫).

## 4. 함정
- **[[dreamcoast-golden-gdf-ao-determinism]]**: 시간 누적 도입 → 프레임-0 미초기화 Private 링버퍼
  read = 골든 플립. gather reset 가드 + FIXED_DT AE + 재시드 규약 준수.
- **stale-shader pull 함정**: .slang 편집 후 'Compiling dreamcoast-shader' 확인(재컴파일 안 되면
  구 바이트코드) — [[dreamcoast-f4b-hierarchical-cache]].
- **sample_texture_lod(…,0.0) ≠ 밉0** — 트레이스에 텍스처 쓰면 주의(F6M 사고).
- **PT 게이트 AE 커플링** — 같은-행 비교만 유효.
- **재캘리브 상쇄-착지**(F6I §0 판독 규칙): 총량 정합+스펙트럼 오프=상쇄 의심 / V≈5+E급감=프로브
  갇힘. 항별 오라클(EV11 고정)로 귀속.
- **성능**: 픽셀당 트레이스는 볼륨 대비 비쌈 — 레이 수 최소화 + 시간 누적 필수. 무측정 기본 ON 금지.

## 4d. 랜딩 기록 — 증분 1 (2026-07-24)

**구현**: `gdf_skyvis.slang`(코사인 헤미스피어 N 레이 GDF 가시성 마치, `bs_trace_bounce` escape
재사용, `float4(V,0,0,1)`, 스카이 픽셀 `float4(1,0,0,1)`, F6M facing flip) + gi.rs 파이프라인/
`record_skyvis`(풀-res, ray_max/bias=record_gi 정합) + push.rs `gdf_skyvis_push`(160B) + main.rs
`P_SKYVIS_PP` 배선(디퍼드 `skyvis` = `skyvis_pp_out.or(gi_skyvis_out)`, 볼륨 유지). build.rs Job
등록. **함정 수리**: `surface_cache.slang` include 누락 → metallib E30015(bs_shade_hit 참조) → 추가.

**검증(진단-전용, 매칭 EV11 near 부조)**:
- **블록 소멸 확정**: DEBUG_VIEW=8(격리 sky-vis 항) before(프로브-볼륨 하드 사각 블록) →
  after(P_SKYVIS_PP=8/16, **블록 완전 소멸·매끄러운 벽·부조 접점 음영 물리적**). 최종 이미지도
  동일 — 사용자 "검은 Block" 근본 해결. 뷰-의존도 소멸(픽셀당이라 카메라-앵커 무관).
- **OFF 바이트-앵커**: `P_SKYVIS_PP` 미설정 → 패스 미기록 → `None.or(gi_skyvis_out)`=원본 →
  구조적 바이트-동일(게이트 확인 중).
- **관찰(예상된 V 재분포)**: after가 다소 어두움 — 픽셀당 V가 프로브-평균보다 정확·고대비 →
  스카이 항 재분포(Fable #4·F6H 클래스). ⇒ **증분 2 재캘리브 필요**(tint/min_occ/sky_gain,
  two-master). 증분-1은 진단-전용이므로 게이트 improved-or-neutral은 증분-2 후 적용.

**다음**: 증분 1b(방향성 bent — 방향×유효도, 4채널 denoise) · 증분 2(스카이라이트 재캘리브 +
게이트) · 성능(PROFILE_GPU: 16 레이 풀-res 비쌈 → 레이 수↓ + 시간 누적).

## 4e. 랜딩 기록 — 노이즈/성능 완성 + TAAU 발견 (2026-07-24)

- **성능**: full-res PP=16=564ms → 저해상 트레이스 + 스텝-캡 + **full-res 업샘플 패스 제거**(Retina
  고정비용 범인, 디퍼드 UV bilinear로 대체) → ~30fps. 잠재 bent 버그(.a=1→bent(0,0,1)) 수리.
- **노이즈 스택**: 시간 누적(gdf_temporal 재사용·별도 저해상 히스토리·V=순수기하라 조명리셋 불요) +
  저해상 5×5 엣지-인지 denoise(0.16ms) + **분산-유도 볼륨-V 블렌드**(고분산=에일리어싱/노이즈→볼륨,
  저분산→per-pixel). Fable F6O 버그 수리: 시간누적에 `temporal_flip`(지터 서브픽셀 히스토리 계약).
- **"문 speckle"의 진짜 정체(Fable 진단·자체 확증) = TAAU 지터**, F6O 무관: 라이브 Apple 티어
  RS=0.67→TAAU+지터 vs 정적 RS=1→off의 불일치. **PP=0·DEBUG_VIEW=1(알베도)에도 speckle → 1차
  가시성 상류.** 서브픽셀 하늘-틈을 지터가 뒤집는데 TAAU 클램프가 밝은 클러스터 통합 실패.
- **TAAU 안티-플리커 랜딩**(`taau.slang`, `P_TAAU_ANTIFLICKER` = flip_y bit2, **기본 OFF**):
  current-history 휘도차 크면 alpha 감쇠(Lottes/Karis). 문 HF 5.19→3.67(지터-off 3.92↓);
  knob OFF는 5.191 = 원 베이스라인 정확 복원. **기본 OFF 사유**: 콘텐츠 전용 게이팅만으론
  sc_viz(mean 0.818/max 230)·gdf_ao(mean 0.032/max 47) 골든이 변한다(둘 다 티어-res=TAAU 활성으로
  캡처). 그런데 **PT 게이트는 RENDER_SCALE=1(TAAU off)이라 이 변화를 측정하지 못함** → 개선/회귀를
  판정할 게이트가 없어 기본 플립 정당화 불가(P_SDF_OPEN_UNSIGNED 선례). 갤러리/PT는 불변 PASS.
  **차기: 티어-res 셔머/HF 품질 게이트 신설 → 기본 ON 재심.**
- **함정(재시도 금지)**: in-TAAU 네이버후드 클램프 2종 — 이웃 평균 당기기(HF 6.59)·플리커-적응 박스
  타이트닝(4.87) — 은 **지터된 현재 프레임을 공간 참조**하므로 오히려 악화. 잔여 제거는 안정화된
  출력을 참조하는 별도 공간 post-blur 또는 supersampling 몫(미착수).

## 5. 후속
- 커버리지-α 필드(얇은-지오메트리 부분 투과율 — 별도 트랙, 본 블록엔 부적합).
- HW-RT 스카이-vis(High 티어 opt-in, DXR≡VK 파리티).
- 볼륨 sky-vis 은퇴(픽셀당이 전면 대체 시) — GI 볼륨 E는 유지(별도 항).
