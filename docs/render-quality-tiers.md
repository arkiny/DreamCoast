# RenderQuality 티어 (Phase 11 Stage D)

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

- **Phase 1 — 티어 스캐폴드 + 노브 배선 (셰이더 탭 포함).**
  `quality.rs`(enum+프리셋+선택), 전 노브의 `unwrap_or` 기본값을 프리셋에서 공급(개별 env 우선),
  소프트-PCF 탭을 `shadow.w`로 런타임화(pbr.slang). UI에 현재 티어 표시.
  - 검증: 미설정 ≡ Med ≡ 현재(바이트 동일, no-reg 6.039/ch), DX≡VK 0.000(하드 경로).
- **Phase 2 — 티어 특성화(measure & tune).**
  Low/Med/High 각각 PT 잔차 + `PROFILE_GPU` 코스트 + DX≡VK 측정, 매핑 수치를 데이터로 확정/조정.
  High 소프트섀도우의 잔차/게이트 영향 정직 보고. 표를 측정값으로 갱신.
- (선택) **Phase 3 — RenderQuality{...} 프레임당 UI/런타임 전환** + 플랫폼 자동 선택 훅(저사양 폴백).

각 Phase는 자체 검증 커밋·푸시. 수치는 정직 보고.

## 검증 (전 Phase 공통)
- `RENDER_QUALITY={low,med,high}` 각각: `tools/rt-compare.py` PT 잔차 + `PROFILE_GPU` 코스트 +
  DX≡VK(Low/Med ≤0.001, High는 소프트 경로 0.0165 표기).
- Med = 미설정 = 현재 6.039/ch 바이트 동일(no-reg).
- clippy `-D warnings` clean, fmt clean.
