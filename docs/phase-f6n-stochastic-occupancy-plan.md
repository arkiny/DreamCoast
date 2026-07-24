# F6N 계획서 — 확률적-점유: 소비-측 α-가중(소프트) 위트니스

> 상태: **계획 — 착수 전 승인 대기**. 선행 봉인: F6I §0(재시도-금지 원장)·F6L(유효도 계약)·
> F6M(α-가중 위트니스 사전 등록, §2-4 line 97). 재개 조건 = F6I §0 ① "확률적-점유 필드
> 표현". 본 페이즈 = 그 트랙의 **소비-측 첫 증분**(필드 바이너리 유지, 소비 가중만 연속화).

## 0. 요약

**동기(신규 진단, 2026-07-24 세션)**: `LEVEL=sponza_intel` 인터렉티브에서 사용자가 **부조
(사자머리 fountain) 밑 음영이 카메라가 다가가면 형태가 바뀐다**고 보고. 우상단 벽에 **하드-엣지
검은 사각 블록**도 관측. 격리 결과 두 증상은 하나의 근본에서 나온다:

1. **소비-측 이진 위트니스** (`gdf_gi.slang` `gv_level_occ`, line 230-231): 점유-가중
   트라이리니어가 8-코너 프로브를 `min(cm_geo_inside(probe), probe.y-GROUND_Y) > 0`의
   **bool**로 통째 채택/거부. 셰이딩 점이 셀을 가로지르며 코너 프로브 하나가 witness→비-witness로
   뒤집히면 트라이리니어 가중이 급변 → **축-정렬 하드 사각 seam**("검은 Block").
2. **카메라-앵커 fine 볼륨** (`P_GI_VOL_CLIP`, 기본 ON): fine 레벨이 초기 카메라에 중심을 두고
   `gi_fine_recenter`(dead-zone ≈ half·0.5 ≈ 3.6m)로 재중심 → 위 블록/누출 패턴 전체가 카메라와
   함께 이동 → "다가가면 음영 형태 변화". coarse 레벨(월드-앵커)은 안정적이나 블록이 더 크다.

**정량(동일 카메라 A/B, near, EV100=11·RENDER_SCALE=1·WARMUP=192)**: fine ON vs
`P_GI_VOL_CLIP=0` = **mean|Δ|=13.3 / max=141 / 30% 픽셀 >15/255**. 부조 좌·우 벽 코너에 **밝은
누출 blob**(프로브가 sunlit 바닥/개구부 irradiance를 그늘진 코너로 샘 — witness라도 셰이딩 점에
대한 **가시성 가중이 없음**). `P_GI_VOL_OCC`(점유 가중)는 이미 기본 ON이나 "지오메트리 안 프로브
통째 거부"만 하고 코너 누출·하드 seam은 못 막는다.

**본 페이즈 목표**: 이진 위트니스의 **SDF 항을 연속 α-가중**(`smoothstep(0, band, sdf)`)으로 대체.
셀 경계에서 개방-측 코너(0<sdf<band)가 램프되어 하드 seam이 밴드-폭 그라디언트가 되고, 표면에
바짝 붙은 개방 프로브가 부분 down-weight되어 누출 blob이 완화된다. `band → 0`이면 `step` =
**정확히 레거시 이진**(바이트-불변 앵커).

**정직 단서(Fable 검증 반영)**: (a) `smoothstep`은 **`sdf ≤ 0`(내장)에서 정확히 0** — 내장
프로브는 얕든 깊든 이진과 동일하게 0 기여. 따라서 램프는 **개방-측 한쪽**만 완화하고, F6L의
내장-프로브 하드-블랙(gi_volume.slang이 명시한 "witness 재정규화가 복셀-스케일 검은 상자로 집중")
클래스는 **불변**. 사용자의 "검은 Block"이 완화될지는 **플립하는 코너가 개방 셸에 있는지**에 달렸다
— 3-사이트 배선 전 §2d 격리 스텝으로 확인 후 주장 조정. (b) 본 α는 F6M §2-4/§4가 등록한 **베이크
커버리지 필드**의 α가 아니라 **SDF-거리 프록시**다. 사전 등록의 정신(이진 위트니스를 연속으로)은
계승하되, 등록된 커버리지-α 구현 그 자체는 아님을 명시(다음 페이즈).

**비목표(다음 페이즈 — "will not fix" 명시)**: (a) **커버리지-α 필드**(베이크 시 복셀 삼각형-커버리지
→ R8 α, 투과율 마치 — F6M §4, 진짜 근본·수 주 규모·스카이-체인 재캘리브 동반), (b) fine 볼륨
카메라-앵커의 뷰-의존 자체 제거(설계상 근접-디테일 목적), (b2) **fine↔coarse 박스-면 하드 스위치**
(gdf_reflect:417-426 등 — 축-정렬 카메라-추종 seam, 밴드로 안 닿음, 모티브 증상에 동반 기여),
(c) 내장-프로브 하드-블랙(F6L 클래스, smoothstep가 sdf≤0에서 0이라 불변), (d) 반투명,
(e) gdf_reflect 법선 플립.

## 1. 근본 귀속 (코드)

`gdf_gi.slang` `gv_level_occ` (line 178-296), 8-코너 루프:
```
bool witness = min(cm_geo_inside(occ_clip, probe, pc.aabb_max.w), probe.y - GROUND_Y) > 0.0;
...
if (witness) c[k][ch] += w * cv;   // E (SH r/g/b)
if (witness) sc[k]    += w * sv;   // V + bent (sky-vis SH)
if (witness) wsum     += w;
```
- `cm_geo_inside`(clipmap.slang:72)는 **월드-미터 SDF 값**(>0 = 개방, <0 = 내장, 클립맵 밖 = 1e3).
  `occ_val = min(cm_geo_inside, probe.y - GROUND_Y)`는 "개방 & 지상" 여유(미터).
- 이진 `> 0`이 셀 경계에서 계단 → 하드 seam. 소비-측이 **필드의 바이너리성을 그대로 계단으로
  투영**한다(F6I §0가 지목한 "이진 SDF 결정론적 팬텀"의 소비 발현).

**lockstep 사이트(단일소스, 전부 동일 패턴 `min(geo_inside, p.y-GROUND_Y) > 0`)**:
- `gdf_gi.slang:230` — `gv_level_occ` (fine/coarse SH E·V·bent 소비). **주 타겟.**
- `gdf_gi.slang:485` — csMain 내 coarse-전용 블록(동일 로직 인라인).
- `gdf_reflect.slang:460` — `sample_gi_irradiance_valid` (반사 폴스루 GI).
세 곳을 **동시에** 연속화(rule #1 일반화·#4 단일소스). 미적용 사이트가 남으면 소비자 간 드리프트.

**명시적 제외 사이트(Fable #5 — 코너 게이트 아님, 손대지 말 것)**: `gdf_ao.slang:60`,
`gdf_trace.slang:107/113`, `gi_volume.slang:101-102`의 `min(geo_inside, p.y-GROUND_Y)`와 is_ground
분류자(gi_volume:288, gdf_trace:249, gdf_reflect:1388)는 **마치/인젝터 거리함수**이지 트라이리니어
코너 게이트가 아니다. 마치를 소프트화하는 것은 **다음 페이즈 커버리지-α 투과율 마치**의 설계이므로
본 증분에서 제외. surface_cache.slang의 "witness"는 무관한 오클루더 라우팅. grep 확인: 위트니스-게이트
트라이리니어 소비자는 정확히 위 3곳뿐.

## 2. 방법 (사전 등록)

### 2a. 소프트 위트니스 헬퍼 (지상 항 이진 유지 — Fable 결함 #1 반영)
레거시 `witness = min(cm_geo_inside, probe.y-GROUND_Y) > 0`은 **(sdf>0) AND (지상>0)**의 곱이다.
지상 항까지 램프하면 바닥 프로브 행이 씬 전역(개방 아트리움 포함) `ow≈0.5`로 편향되므로 — seam과
무관한 전역 변화 — **지상 항은 이진 게이트로 유지, SDF 항만** 소프트화:
```
// sdf     = cm_geo_inside(occ_clip, probe, pc.aabb_max.w)   (월드 미터, >0=개방)
// ground  = probe.y - GROUND_Y
// band_m  = 0 → step(레거시 이진, 바이트-불변) ; > 0 → 표면 밴드 위 연속 램프
float occ_soft(float sdf, float ground, float band_m) {
    if (ground <= 0.0) return 0.0;                       // 지상 게이트 = 하드(레거시 보존)
    return band_m > 0.0 ? smoothstep(0.0, band_m, sdf)
                        : (sdf > 0.0 ? 1.0 : 0.0);       // band=0 = 정확히 레거시 witness
}
```
누적:
```
float ow = occ_soft(sdf, ground, band_m);
c[k][ch] += w * ow * cv;   sc[k] += w * ow * sv;   wsum += w * ow;   // (cu[]/wu 언게이트는 불변)
```
- **바이트-불변 구현(Fable 결함 #3 반영)**: 통합 `w*ow*cv`는 witness 분기를 없애고 곱 2개를 더해
  **명령 스트림이 달라진다**(F6L/F6M의 knob-off 코드젠-드리프트 = max 1 LSB 재시드 클래스). 두 경로:
  (i) **`band_m > 0.0`에 동적-유니폼 분기**를 두고 else 가지에 **레거시 루프 본문을 그대로** 유지
  (gi_importance 선례) → SHA 불변 보장, 또는 (ii) 통합 경로 + **sc_viz/gdf_ao 재시드 사전승인**
  (실델타 ≤ 1 LSB 확인). **(i) 채택**(갤러리 앵커 무재시드 우선). 또한 `0.0 * cv`는 텍셀이 비-유한일
  때 NaN — band>0 곱 경로는 이진 분기가 "미사용"으로 건너뛰던 값을 곱하므로, 볼륨 텍셀 유한성 가정을
  주석에 못박음(프레임-0 미초기화 Private 선례).
- 언게이트 폴백(`vol_occ` bit1, `cu[]/scu[]/wu`)은 **그대로 유지**: 전 코너 내장(각 ow=0)일 때만
  발동 — F6L all-rejected 하드-블랙 의미 보존. **주의(Fable #2 방향)**: ow≤1이라 wsum이 단조 감소
  → 이진보다 **더 많은** 점이 1e-4 문턱을 넘어 언게이트로 전환하고, 그 전환은 불연속 → wsum≈1e-4
  등고선을 따라 **신규 seam**을 만들 수 있음(§4 함정).

### 2b. knob / 배선 (밴드 = 복셀 단위 — Fable 결함 #4 반영)
- 신규 env `P_GI_OCC_SOFT`(기본 **0.0 = 하드**). **밴드는 월드-미터가 아니라 복셀 배수**로 해석:
  `gv_level_occ`는 fine·coarse를 한 헬퍼로 서비스하는데 단일 월드 밴드는 fine 복셀엔 ~1배, coarse
  복셀엔 극소 분수 → 레벨 간 불일치. 셰이더가 **그 레벨의 복셀 크기**(`vox = mean((lmax-lmin)/dims)`)
  로 환산: `band_m = P_GI_OCC_SOFT * vox`. 단일 dimensionless knob이 자동으로 per-level 스케일.
- `quality.rs` env_f32, `main.rs`에서 읽어 `GiPush`/reflect push에 `occ_soft_vox: f32` 슬롯 전달
  (스페어 슬롯, rule #3). reflect(`scene_occ`)도 동일 단일소스 값(드리프트 금지).
- 1차 후보 = **1.0 복셀**(물리 근거: 커버리지 밴드 ≈ 복셀 폭). 스칼라 튜닝 금지 — A/B는 검증용,
  게이트 수치로만 확정.

### 2c. 무엇을 고치는가 (기전 — Fable #1 반영, 편측성 정직화)
- **하드 seam**: 개방-측 코너(0<sdf<band)가 밴드 위 램프 → 그라디언트. **단, 플립 코너가 개방 셸에
  있을 때만** — §2d 격리로 사용자 "검은 Block"이 이 클래스인지 먼저 확인(내장-집중 블랙박스면
  본 증분은 무효, 커버리지-α 필드 몫).
- **누출 blob**: 표면에 바짝 붙은 **개방** 프로브(sdf 작은 양수)가 부분 down-weight → sunlit
  irradiance 유입 감소. 완전 해소는 **셰이딩 점 방향 가시성**(다음 페이즈)이라 정직히 명시.
- **뷰-의존**: seam/blob이 그라디언트화되면 카메라 이동 시 이동이 **덜 가시적**. 앵커·fine-box 하드
  스위치(§5 비목표)는 불변 — 근본 제거 아님, 지각 완화.

### 2e. 구현 스코프 (증분 1 vs 1b — push 예산 제약)
- **증분 1(본 커밋)**: gdf_gi 2개 사이트(`gv_level_occ` :230, csMain coarse 블록 :485). 밴드는
  `GiPush.vol_occ` **bits 16-31**에 고정소수점 인코딩(`band_q = round(P_GI_OCC_SOFT·256)`; bits 0-1은
  기존 플래그). GiPush는 **256B cap 만석**(gi.rs:178)이라 새 필드 불가 — 비트 패킹이 유일한 무-레이아웃
  경로. `band_q=0`이면 vol_occ 불변 → **바이트-불변 앵커 보존**.
- **증분 1b(같은 페이즈, 후속 커밋)**: reflect 폴스루 `sample_gi_irradiance_valid`(gdf_reflect:460).
  reflect push의 `flip_y`(bits 0-27: Yflip·vol·sv·fine-box·fallback 만석)·`max_steps`(bits 0-23,30
  만석)에 깨끗한 16비트 여유가 없어, push를 240→256으로 키우고 파이프라인 레이아웃을 갱신하는 직교
  배선 필요. reflect-GI 폴스루는 **2차 소비자(반사 전용)**로 사용자 아티팩트·block64 게이트에 거의
  불가시. **knob 기본 OFF에선 드리프트 없음**(둘 다 하드 witness) — 드리프트는 knob ON·글로시 표면
  에서만 발현하며 1b에서 측정·정합 후에만 기본-ON 후보. rule #4는 기본-OFF로 중화.

### 2d. 배선 전 격리 (Fable #1)
3-사이트 배선 전, 부조 우상단 블록 영역에서 플립 코너의 `sdf`(cm_geo_inside)를 덤프(임시
`DEBUG_VIEW` 또는 storage 덤프)해 **개방-셸(0<sdf<vox) 대 내장(sdf<0)** 판별. 개방-셸이면 진행,
내장-집중이면 본 증분 무효 판정하고 커버리지-α 필드로 직행(정직 기록).

## 3. 게이트 정책

착수 전 러너/예산 확정(F6B·F6M 레시피 준수):
- **갤러리 앵커 SHA 불변**(`65d04ceca2c4dbff…`) — knob OFF는 전부 바이트-불변 **필수**(콘텐츠-게이팅
  확인). 위반 시 band=0 경로가 레거시와 안 맞는 것 → 수정.
- **PT 잔차 게이트**(`python tools/golden-image.py --only sponza_pt_sunlit --only
  sponza_pt_interior`, 기본 backend metal): 현 예산 **sunlit 20.74 / interior 27.21**(F6M
  재기준선). knob ON 캡처가 **block64_avg improved-or-neutral**이어야 기본 ON 자격. `pt_black_frac`
  ·`masked_avg`(scatter 섀도)도 회귀 감시.
- **sc_viz / gdf_ao**: 소비 변경이 표면캐시 lit·AO에 닿을 수 있음 — 구조 무변화 확인 후 필요 시
  `--update --save-png` 재시드(§4 스테일-PNG 함정 준수).
- **시각 A/B 정량 기준(Fable #4a 반영)**: PT 게이트 2개 카메라는 부조의 국소·재중심-의존 seam에
  **둔감**할 수 있음(neutral+육안만으론 무용한 밴드도 기본-ON될 위험). 따라서 **기본-ON 자격에
  near/far A/B `mean|Δ|` 개선을 정량 기준으로 추가**: 부조 near 캡처(eye `-11,1.2,0.79`, target
  `-15.84,1.2,0.79`)의 knob ON vs fine-OFF `mean|Δ|`가 **베이스라인 13.3 대비 유의하게 감소**해야
  한다(즉 소프트 witness 결과가 월드-앵커 coarse에 더 근접 = 뷰-의존 시프트 축소). far도 병행.
- **PROFILE_GPU 비용 라인(Fable #4b, house rule #2)**: 추가 비용 = 이미 지불한 `cm_geo_inside` +
  코너당 smoothstep 1회(사소)이나, `PROFILE_GPU=1`로 gdf_gi 패스 ms를 knob OFF/ON 측정·보고.
- **DX≡VK**: Metal 로컬 검증만 가능(Windows RTX 동결) — 셰이더 변경이 순수 산술·백엔드 무관이므로
  파리티 리스크 낮으나, **Windows 재검증 보류 태그** 부착(F6 관행).

**착지 판정(사전 규약, 캘리브-결합 경고)**: 선행 아크(F6H unsigned·F6M 플립)에서 **물리 수리가
크러치 캘리브를 개방해 게이트가 옳게 기각**한 전례가 반복됐다. 소프트 위트니스가 E/V를 올려 스카이
항 과주입을 유발하면 block64가 악화될 수 있다. 그 경우:
- **기본 OFF·knob 잔존**으로 랜딩(P_SDF_OPEN_UNSIGNED / P_GDF_FACING_FLIP 선례) + 히어로/레시피
  옵트인. 기본 ON 재심 = 커버리지-α 필드 + 스카이 재캘리브(F6G 재개방) 시점.
- 게이트 개선 또는 중립이면 **기본 ON** 자격 — 단 band 값과 수치를 본 문서에 기록.
- 어느 경우든 **버짓 무단 상향 금지**: 하향이면 `--update` 래칫, 상향-사유는 문서화 후 재시드.

## 4. 함정 (선행 아크 계승)

- **`sample_texture_lod(tex,uv,0.0)` ≠ 밉 0** — 본 페이즈는 텍스처 밉과 무관하나, 셰이더 편집 중
  실수 방지 위해 명시(F6M 사고 클래스).
- **tools/goldens PNG 스테일** — 러너 SHA 진실은 manifest. main 재현 SHA가 일치하면 PNG-기반
  "max 115"류 diff는 허상. `--update --save-png`로 재생성.
- **PT 게이트 AE 커플링** — PT lit_mean은 라스터-미터링 AE 공유. knob이 라스터 노출을 바꾸면 같은-행
  내 비교만 유효(F4B/F6M 재확인).
- **언게이트 폴백 상호작용(양방향)** — (a) 전-코너 내장이면 각 ow=0 → wsum≈0 유지 → bit1 폴백 정상
  발동(smoothstep 하한 sdf≤0→0이 보장). (b) **역방향(Fable #2)**: ow≤1이라 wsum 단조 감소 → 이진보다
  더 많은 점이 `wsum≤1e-4`를 넘어 언게이트로 전환, 이 게이트↔언게이트 스위치는 불연속 → wsum≈1e-4
  등고선을 따라 **신규 seam** 가능. 소규모 예상이나 A/B에서 감시.
- **injector-fill 결합(Fable #5)** — F6L trapped-probe fill(gi_volume.slang `GV_FILL_*`)은 이진
  소비자의 집중 거동에 캘리브됨. fill + bit1 폴백 + 소프트 witness **3층 상호작용 미검증** — `P_GI_
  TRAPPED_FILL` ON/OFF 교차 A/B로 확인.
- **fine-box 하드 스위치 존속(Fable #5)** — `sample_gi_irradiance_valid`(gdf_reflect:417-426) 및
  gv_level_occ 호출부의 fine↔coarse 박스-면 하드 전환은 축-정렬·카메라-추종 seam이며 **밴드로 안 닿음**
  (§5 비목표). A/B 판정 시 이 seam을 소프트-witness 실패로 오독 말 것.
- **`P_GI_VOL_OCC=0`/`P_GI_VOLUME=0` 밝아짐은 경로-전환 착시**(F6L §2) — 판정 금지.
- **EV11 vs AE 도메인 분리** — 귀속은 EV11 고정노출, 지각은 AE. 섞지 말 것.
- **재시도-금지 원장(F6I §0) 비저촉 확인** — 본 변경은 이진 부호 전략·해상도·sky-fill·τ-radiance·
  방향-로테이션·E-마치 밴드 **어느 것도 아님**(전부 필드/마치 측). 소비-측 α-가중은 원장 미수록·
  F6M §2-4 사전 등록 항목. path ②(SH 블러 = 증상 완화)와도 구분: 이건 프로브 커버리지의 물리적
  부분-점유 가중(path ① 소비 발현).

## 4b. 랜딩 기록 — 증분 1 실측 결과 (2026-07-24)

**게이트(knob OFF, 기본 배송 구성): ALL PASS · 완전 바이트-불변.**
- SHA 골든 불변: gallery `65d04ceca2c4dbff` · sc_viz `8769d3e2ec257b54` · gdf_ao `7c2d75b9ecf5b308`.
- PT block64 = **sunlit 20.438 / interior 26.908**(F6M와 정확히 일치) ≤ 예산 20.74/27.21.
- 즉 `band_q=0`의 값-동일 산술이 **동일 바이트코드**를 생성 — §2a 코드젠-드리프트 우려는 실측상
  발생하지 않았고(재시드 불요), 바이트-불변 앵커가 보장됨.

**knob ON(band=1.0·2.0 voxel): 이 아티팩트에 대해 측정-확정 무효(inert).**
- 정량(near, vs 뷰-안정 fineOFF 기준): binary **13.30** → soft1 **13.32** / soft2 **13.26** —
  뷰-의존 시프트 **불변**(§3 기본-ON 기준 미달). soft가 binary를 움직인 양은 **2.0/2.4 mean|Δ|**뿐.
- 시각(부조 near): 하드 사각 블록·좌측 base 누출 blob **잔존**(엣지만 미세하게 덜 crisp).
- **근본 사유(Fable #1 실측 확증)**: 블록 엣지는 **벽에 내장된(sdf<0) 코너 프로브의 완전 거부**로
  생긴다. `smoothstep(0,band,sdf)`는 sdf≤0에서 정확히 0이라 **내장 코너를 못 건드리고**, 얇은
  개방-셸(0<sdf<band)만 램프한다. 이진 SDF에는 내장 프로브의 부분-커버리지 정보가 없으므로
  **소비-측 SDF-거리 프록시로는 블록/누출을 못 고친다**. §2d 격리의 답 = 이 씬의 블록은
  **내장-집중 클래스**(개방-셸 클래스 아님) → 본 증분 무효 판정.

**결론**: 증분 1은 **바이트-안전하나 단독으로는 무효**. 진짜 근본은 §5의 **커버리지-α 필드**
(필드 자체를 비-이진화 — 얇은 벽 내장 프로브도 부분 α → 부분 기여). 그 α는 본 증분이 깐 소비
지점(`occ_soft`/`wow`, `vol_occ` 비트)에 그대로 주입된다(SDF 프록시 → 베이크 α 교체). PT-ON
block64는 미측정(증분이 **기본 OFF로 랜딩** — 배송 구성은 바이트-동일 OFF 경로로 이미 ALL PASS;
ON block64는 기본-ON 승격만 게이트하는데 시각·정량이 이미 승격을 기각).

**코드 처분(사용자 결정 대기)**: (a) `P_GI_OCC_SOFT` 기본-OFF knob으로 랜딩(커버리지-α 소비
스캐폴딩·바이트-안전) 또는 (b) 증분 되돌리고 커버리지-α 필드 페이즈로 직행. 어느 쪽이든 본 실측
기록은 유지.

## 4c. 아티팩트 재-국소화 — 층 정정 (2026-07-24, 실측)

증분-1 무효 후 **아티팩트의 실제 출처를 항별 분리**(near 카메라, DEBUG_VIEW):
- **DEBUG_VIEW=10 (GDF GI 볼륨)**: 매끄러움, 블록 **없음**.
- **DEBUG_VIEW=9 (GDF AO) / =6 (AO)**: 매끄러움, 블록 **없음**.
- **DEBUG_VIEW=8 (IBL 앰비언트)**: 하드 사각 **블록 여기 존재**.

**판정**: 사용자의 "검은 Block"은 **스카이-가시성(V) 필드**의 항이다(GI 볼륨 아님). V는
`gi_volume.slang`이 **이진 per-ray 마치**(1=하늘 탈출, 0=지오메트리 히트, `scene_occ`=binary SDF)로
프로브당 계산해 SH로 저장하고, pbr.slang이 `skyvis_index` 이미지로 소비한다. 소비-측 수리(소프트
witness·trapped-fill) **둘 다 실측 inert(2.0/0.9 mean|Δ|)** → 블록은 **소비가 아니라 프로듀서(저장된
V 필드)** 측. fineOFF=큰 블록 / fineON=작은 블록 = **프로브-해상도 양자화**의 지문(카메라-앵커라
뷰-의존).

**함의(트랙 재조준)**: 증분-1(소비 witness)은 **잘못된 층**이었다(실측 확증). 올바른 층 =
`gi_volume.slang`의 **V·태양 가시성 마치**(등록된 커버리지-α 설계가 정확히 여기). 단, 추가 주의:
이 씬의 블록은 **두꺼운 벽/조각의 프로브-해상도 양자화**로 보이며, 커버리지-α(얇은-지오메트리 부분
투과율)는 **얇은 지오메트리엔 유효하나 두꺼운-벽 해상도 블록엔 무효일 수 있다**(불투명 벽은 α=1).
그렇다면 이 블록의 진짜 레버는 (i) **V 필드 볼륨-공간 SH 블러**(F6I §0 path ② — 증상 완화 명시),
(ii) **per-pixel 스크린-공간 V 정제**(G버퍼 법선 + SDF 콘 탭, 프로브-해상도 탈피 — 신규 기법),
(iii) 해상도 상향(F6I §0 봉인). **다음 페이즈 착수 전 세 레버의 저비용 A/B로 유효 레버부터 실측 확정**
(증분-1 교훈: 층·기전을 실측으로 못박기 전 대형 구현 금지).

## 5. 후속 (사전 등록)

- **커버리지-α 필드**(진짜 근본, F6M §4): 베이크 복셀 삼각형-커버리지 → R8 α(pack_like, ~+14MB,
  340MB 천장 재시도 아님). **V·태양 가시성만** 투과율 마치 `T *= (1-α)^(ds/vox)`, radiance 첫-히트
  유지(τ-radiance 기각 준수). 착지 = sky_gain 6.0/tint 재캘리브(F6G 재개방, t≈0.10 후보), 심 기본
  ON 재심. §1-3(석재 바운스)·V² 크러시·dv13 붕괴가 이 트랙에서 해소.
- **fine 볼륨 뷰-의존 완화**: 토로이달 히스토리 재사용(재중심 주석의 예고) 또는 월드-스냅 앵커 강화 —
  근접 디테일 유지하며 이동 시프트 축소.
- **천·잎 반투명**(PT 먼저), **gdf_reflect 법선 플립**(T4/C4 스페큘러 판).
