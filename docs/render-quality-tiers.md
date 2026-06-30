# RenderQuality 티어 (Phase 10 Stage D)

> **렌더링 아키텍처 결정 (2026-06-29): 모든 티어는 SW-RT, HW-RT는 별도 옵션.**
> `RenderQuality{Low,Med,High}` **세 티어 모두 GDF 소프트웨어 레이트레이싱(SW-RT)** 을 기본 렌더 경로로
> 쓴다 — AO·diffuse GI(DDGI 라디언스 캐시 볼륨)·하이브리드 SW-RT 반사·surface cache. 티어는 그 SW-RT의
> 샘플 수/해상도/디노이즈 강도만 조절한다(HW-RT를 끌어오지 않는다).
> **하드웨어 레이트레이싱(DXR / VK_KHR)** 은 티어와 무관한 **명시적 옵션**으로 분리한다:
> `--raytracing` 플래그(레거시 `P8_PATHTRACE` env도 유효)로 path tracer 전체 렌더를 켠다. 이는 SW-RT의
> 정합 기준(ground truth)이자 HW-RT 데모 경로이며, 기본 렌더는 언제나 SW-RT다. (`P8_RT_DEBUG` RT-디버그
> 뷰, `P8_CORNELL` 코넬박스 씬도 같은 부류의 분리된 opt-in.)
>
> **상태: 계획.** 트랙 전반에 흩어진 품질 노브를 단일 `RenderQuality{Low,Med,High}` enum 한 곳으로
> 묶어 런타임/플랫폼별로 선택만 하면 되게 한다(저사양=저티어 폴백). 각 기능은 이미 "기본 off/저비용 +
> env·플래그 seam"으로 설계돼 있으므로(`SHADOW_SOFTNESS`/`SOFT_SHADOWS`, `P11_*`, 셰이더 상수 블록),
> **티어는 그 seam을 *선택*만 하는 얇은 레이어**다(중복 로직 금지, 단일 진실 공급원).
>
> 선행 컨텍스트: `docs/ROADMAP.md`(Stage D), `docs/shadow-reflection-quality.md`(소프트섀도우=옵트인 티어),
> `docs/reflection-sdf-resolution.md`(SDF 해상도=잔차 레버 아님, 종료). 원칙: `CLAUDE.md` 5원칙.

## 검증된 기준선 (cold-start, 2026-06-27)

| 항목 | 결과 | 게이트 |
|---|---|---|
| `cargo clippy -p sandbox --all-targets -- -D warnings` | clean | ✅ |
| `cargo fmt --check` | clean | ✅ |
| 기본 씬 PT 잔차 (d3d12, raster vs `P8_PATHTRACE`) | **6.039/ch** (off>8 26.2%, off>32 1.22%) | =6.04 ✅ |
| DX≡VK (raster d3d12 vs vulkan) | **0.000/ch** (off>8 0%) | ≤0.001 ✅ |

> 이 기준선이 `Medium` 티어 = 미설정 기본의 no-reg 타깃이다.

## 설계 — 단일 진실 공급원

- **enum + 프리셋 테이블 한 곳(Rust).** `apps/sandbox/src/quality.rs` 신설:
  `RenderQuality{Low,Med,High}` + `QualityPreset{...}` + `fn preset(q) -> QualityPreset`.
  티어→노브 매핑은 **이 테이블에만** 존재.
- **선택**: env `RENDER_QUALITY=low|med|high`, 미설정 → `Med`(현재 기본과 바이트 동일).
- **해상 순서 (seam 유지)**: `티어 프리셋 → 개별 env 오버라이드`. 각 노브는 기존
  `std::env::var(...).unwrap_or(<하드코딩>)`를 `.unwrap_or(preset.x)`로 바꾸기만 한다.
  **개별 env가 항상 우선** → 기존 `P11_*`/`SHADOW_*` seam 그대로 동작.
- 중복 금지: 노브는 여전히 `App` 필드 한 곳에 저장되고 소비처는 무변경. 티어는 *기본값 공급자*일 뿐.

## low/med/high 노브 매핑표

| 노브 | env 오버라이드 | **Low** (저사양) | **Med** (=현재 기본, no-reg) | **High** (품질) |
|---|---|---|---|---|
| GI 샘플/px | `P11_GI_SPP` | 4 | **8** | 16 |
| GI 디노이즈 | `P11_GI_DENOISE` | on | **on** | on |
| 반사 hit 캐시 (C8g) | `P11_REFLECT_CACHE` | **off** | **on** | on |
| GI 서피스 캐시 (C8b3) | `P11_SURFACE_CACHE` | off | **off** | **on** |
| SSR 모드 | `P11_SSR_STOCHASTIC` | **stochastic half-res** (~4× 저렴) | **full mirror** | full mirror |
| 반사 max-roughness | `P11_REFLECT_MAX_ROUGHNESS` | 0.3 | **0.5** | 0.6 |
| GDF AO | `P11_GDF_AO` | off | **off** | **on** |
| 파이어플라이 클램프 | `P11_FIREFLY_CLAMP` | on | **on** | on |
| 소프트 그림자 | `SHADOW_SOFTNESS` | 0 (하드 3×3) | **0 (하드 3×3)** | **0.03 (소프트 on)** |
| 소프트-PCF/blocker 탭 | `shadow.w` (신규 런타임 슬롯) | (소프트 경로 미사용) | (소프트 경로 미사용) | 16 / 16 |
| GDF 해상도 | `P11_GDF_DIM` | 48 | **48** | 48 |

### 매핑 근거
- **Med = 현재 기본**: no-reg 게이트가 강제. 미설정 출력이 바이트 동일해야 함.
- **Low**: 무거운 기능 off로 폴백 — 반사 hit 캐시 off, SSR stochastic half-res(미러 대신),
  GI spp 절반, 반사 max-rough 낮춰 GDF 폴백 비중↓. 잔차 약간↑·코스트↓ 허용.
- **High**: 옵트인 무거운 기능 on — GI 서피스 캐시(멀티바운스), GDF AO, GI spp 2×,
  미적 소프트 그림자. 잔차 동등/개선·코스트↑.

### 소프트 그림자 (High) — 정직한 트레이드오프
하드 3×3 PCF는 *가장 싸면서 PT에 가장 가깝다*(PT 태양 디스크가 near-sharp,
`shadow-reflection-quality.md` Phase 1 측정). 따라서 소프트 그림자는 **미적 품질** 항목이지
패리티 레버가 아니다. 사용자 결정으로 **High에서만 소프트 on**(미적 우선):
- High의 그림자 영역 PT 잔차는 **소폭 상승**할 수 있음(부드러움이 PT보다 넓음) → 측정 후 정직 보고.
- **DX≡VK 주의**: 소프트 경로는 페넘브라 가장자리(V-flip×IGN 회전)로 **0.0165/ch**(문서화된 옵트인
  허용치). 즉 **High 티어는 ≤0.001 게이트를 만족하지 못함** — 이는 사전 합의된 옵트인 소프트 경로의
  알려진 한계. **Low/Med는 하드 경로라 0.001 유지.** 게이트 보고 시 티어별로 구분 표기.

## 측정으로 기각된 항목 — 티어 레버에서 제외 (중복 구현 방지)
- **`P11_GDF_DIM` 상향**: 반사 잔차 레버 아님(48→128 ≈0.02/ch, `reflection-sdf-resolution.md` 종료).
  전 티어 48 고정. **큰 씬 커버리지(Stage B) 전용 노브로만 보존** — 반사 선명도 목적 재시도 금지.
- **`CARD_TILE` 32→64**: 효과 없음(blob 모양 지배, 메모리 `engine-backlog` C8h). 32 고정, 티어 노브 아님.
- **소프트 그림자를 패리티 목적으로 사용**: 하드가 더 정확. High 소프트는 *미적* 목적에 한정.

## 셰이더 작업 — 소프트-PCF 탭 런타임화 (Phase 1 포함)
현재 `crates/shader/shaders/pbr.slang`의 `SHADOW_BLOCKER_SAMPLES`/`SHADOW_PCF_SAMPLES`(둘 다 16)는
컴파일타임 `static const`. 티어로 바꾸려면 런타임 값이 필요:
- 여유 슬롯 **`shadow.w`** 에 소프트-PCF 탭 카운트(기본 16)를 싣는다(`shadow.x/y/z`는 bias/texel/softness).
- pbr.slang의 blocker/PCF 루프 바운드를 `min(shadow.w, 16)`로(POISSON16 배열이 16 상한).
- 하드 3×3 폴백(softness==0)은 무변경 → **Low/Med 출력 무변경**(no-reg).
- 미래 모바일/Low-소프트 티어가 탭을 8로 내릴 수 있는 seam 확보(셰이더 재빌드 불필요).
- 품질 상수(POISSON 배열, SEARCH_UV, MAX_PENUMBRA)는 셰이더 한 블록 유지 — 단일 소스.

## Phase 분할

- **Phase 1 ✅ (`4df8e63`) — 티어 스캐폴드 + 노브 배선 (셰이더 탭 포함).**
  `quality.rs`(enum+프리셋+`env_bool`), 전 노브의 `unwrap_or` 기본값을 프리셋에서 공급(개별 env 우선),
  소프트-PCF 탭을 `shadow.w`로 런타임화(pbr.slang). UI에 현재 티어 표시.
  - 검증: 미설정 ≡ Med ≡ 기존(바이트 동일, no-reg 6.039/ch), DX≡VK 0.000(하드 경로). clippy/fmt clean.
- **Phase 2 ✅ — 티어 특성화(measure).** 아래 [측정 결과](#측정-결과-phase-2-d3d12-rtx-2070-super) 참조.
  데이터가 매핑을 검증(튜닝 불필요): Low=싼 폴백, High=게임 미적 품질.
- **Phase 3 ✅ — 런타임 UI 전환 + 플랫폼 기본 seam.**
  ImGui 콤보(`low|med|high`)가 프리셋을 라이브 노브에 재적용(생성시와 동일한 capability 게이트 유지;
  매 프레임 그래프 재빌드라 즉시 반영). 수동 선택은 시작 env 오버라이드를 대체(env는 *초기 상태*만 시드).
  `RenderQuality::platform_default()` = 미설정 시 기본 티어 seam(현재 Med; 향후 per-GPU 선택이 꽂히는 지점,
  GPU perf-tier 룩업 없이 가짜 감지 안 함 — 정직). 시작 시 활성 티어 `info!` 로그(헤드리스 캡처/프로파일 관측).

각 Phase는 자체 검증 커밋·푸시. 수치는 정직 보고.

## 측정 결과 (Phase 2, d3d12 RTX 2070 SUPER)

| 티어 | PT 잔차/ch | DX≡VK/ch | GPU 코스트 | gdf_gi |
|---|---|---|---|---|
| **Low** | **6.019** | 0.000 | **3.96 ms** (−29%) | 2.06 ms (spp4) |
| **Med** | **6.039** | 0.000 | **5.58 ms** (기준) | 3.48 ms (spp8) |
| **High** | **6.722** | **0.009** (max 32) | **11.74 ms** (+110%) | 9.40 ms (spp16+캐시), +gdf_ao 0.04 ms |

### 정직한 해석 (이 작은 테스트 씬 한정)
1. **Low가 Med보다 PT 잔차 미세하게 더 좋고(6.019<6.039) 29% 싸다.** 이 씬은 GDF 지배라 무거운 반사
   기능(reflect_cache·풀미러 SSR·spp8)이 PT 패리티를 못 올림 — `reflection-sdf-resolution.md`의 "SW-RT
   실용 바닥" 재확인. Low는 손해 없는 저사양 폴백.
2. **High PT 잔차는 *나빠짐*(6.722).** 미적 소프트섀도우가 near-sharp PT 태양과 벌어짐(예측대로). High는
   *게임 미적 품질*(소프트섀도우·멀티바운스 GI 캐시·AO·2× GI)용이지 이 마이크로벤치 패리티용이 아님.
3. **DX≡VK**: Low/Med=0.000(결정적 하드섀도우). **High=0.009(max 32)** — 소프트섀도우 페넘브라
   가장자리(V-flip×IGN 회전), 사전 합의된 옵트인 한계로 **≤0.001 게이트 밖**(설계상 허용). **Low의
   stochastic SSR은 이 씬서 거의 miss라 발산 안 함(0.000) — 온스크린 반사 많은 씬에선 잠재 발산**(scene-dependent).
4. **매핑 튜닝 불필요**: 데이터가 설계 의도(Low=코스트↓·동등 품질, High=2× 코스트로 게임 품질)를 확인.
   이 테스트 씬은 PT-패리티가 노브에 둔감 — 티어 가치는 *코스트 스케일링(Low)*과 *미적 품질(High, 풍부한 씬)*.

## 검증 (전 Phase 공통)
- `RENDER_QUALITY={low,med,high}` 각각: `tools/rt-compare.py` PT 잔차 + `PROFILE_GPU` 코스트 +
  DX≡VK(Low/Med ≤0.001, High는 소프트 경로 0.0165 표기).
- Med = 미설정 = 현재 6.039/ch 바이트 동일(no-reg).
- clippy `-D warnings` clean, fmt clean.

## 후속(계획) — AO 파라미터화 + 베이크드 occlusion 캡처

> **상태: 계획.** 현재 GDF AO는 `P11_GDF_AO`(on/off, High 전용) **하나만** 티어에 들어와 있고, 강도/범위
> 노브(`AO_STRENGTH`/`AO_REACH`/`AO_FLOOR`, [`apps/sandbox/src/gi.rs`])는 **env-only**다. 기본값은
> `strength 2.0 / reach (diag·0.07).min(0.5) / floor 0.3`로 한 번 상향(`1.5/–/0.4`이 실내에서 너무
> 옅었음 — DEBUG_VIEW=9 기준 darkened<200 픽셀 1.4%→31.9%). 모두 **push-constant**라 백엔드 동일값 =
> DX≡VK-safe. 이걸 정식 파라미터로 끌어올리는 게 후속 작업.

### (1) 런타임/티어 노브로 승격
- `QualityPreset`에 `ao_strength`/`ao_reach`/`ao_floor` 추가, `gi.rs`의 `.unwrap_or(<하드코딩>)`을
  `.unwrap_or(preset.ao_*)`로 교체(개별 env 항상 우선 — 기존 seam 유지). Low/Med/High별 강도 매핑.
- ImGui 슬라이더(strength/floor) → 라이브 노브 재적용(소프트 그림자 탭과 같은 패턴). 매 프레임 그래프
  재빌드라 즉시 반영. 셰이더 재빌드 불필요(이미 push-constant).
- no-reg 주의: **Med의 AO 기본을 바꾸면 GDF-AO 씬 출력이 변한다**(이번 2.0/0.3 상향이 그 예). 티어
  승격 시 Med 기본을 무엇으로 고정할지(현 2.0/0.3 유지 권장) 명시하고 그 시점 베이스라인 재캡처.

### (2) 머티리얼 **베이크드 occlusion**(glTF `occlusionTexture`) 캡처 — 현재 **미구현**
> **조사 결과(2026-06-30):** 엔진은 glTF occlusion을 **임포트하지 않고**(`GltfMaterial`에 occlusion
> 필드 없음 — base_color/MR/normal/emissive만), G-buffer는 **베이크드 AO를 1.0으로 하드코딩**한다
> (`gbuffer.slang`: `albedo.a=1.0`, `material.b=1.0`). 게다가 **Intel Sponza 3종(main 28 / 사이프러스 3
> / 커튼 4 머티리얼)에 `occlusionTexture`가 0개**라 캡처할 데이터 자체가 없다 → 현재 화면 AO는 **전적으로
> 런타임 GDF AO**. 그래서 이번 "AO 진하게"도 GDF 노브로 처리한 게 맞는 레버였다.
- 후속 구현(occlusion을 저작한 에셋이 들어올 때 의미):
  - `GltfMaterial`/`MaterialDesc`/`SceneObject`에 occlusion 텍스처 인덱스(+`strength`) 캡처. 단일소스
    임포트(`gltf_scene.rs`)에 한정. 텍스처 슬롯 1개 추가(현 `tex[4]` 확장 또는 emissive와 패킹).
  - ORM 패킹 인지: glTF는 보통 occlusion을 **MR 텍스처의 R채널**에 패킹(occlusionTexture가 MR과 같은
    이미지를 가리킴). 임포트 시 동일 이미지면 R=occlusion으로 함께 읽기(추가 업로드 없음).
  - `gbuffer.slang fsMain`: `ao_baked = lerp(1, occTex.r, strength)`를 RT0.a / RT2.b에 기록(현 하드코딩
    1.0 대체). PBR 라이팅이 이미 RT의 AO를 ambient에 곱하므로 **베이크드 AO × 런타임 GDF AO**가 자연 합성.
  - no-reg: occlusion 텍스처 **없는** 머티리얼은 1.0 폴백 → 기존 씬(Intel Sponza 포함) **바이트 동일**.
    DX≡VK: 텍스처 샘플 1개 추가(좌표 동일) — 데칼/마스크와 같은 크로스백엔드 표면, 게이트로 검증.
- 우선순위: (1) 노브 승격이 먼저(즉효·저위험). (2) 베이크드 occlusion은 **해당 에셋이 생길 때** 착수
  (지금 Intel Sponza엔 소스가 없어 가치 0).
