# GI-fidelity wave-1 이후 성능 재측정 (DX/VK, 다각도) — 최적화 착수 기준선

상위: [sponza-perf.md](sponza-perf.md)(Sponza HD 60fps 달성 — DX 12.6 / VK 16.6ms, **48³ scene SDF**
시절) · [gi-fidelity-roadmap.md](gi-fidelity-roadmap.md)(품질 트랙). 이 문서는 **GI-fidelity wave-1
(per-mesh SDF direct-sample 승격 `a7896b8`/`c34b0e5` + AO 패스 추가 + F1–F6)이 성능에 미친 영향**을
측정-우선 원칙으로 재귀속한 기준선이다. 이후 모든 최적화 우선순위는 이 표가 지배한다.

## 측정 조건
- HW: RTX 2070 SUPER, 1280×720(native, `render_scale`=1), `RENDER_QUALITY=med`.
- 빌드: `97246dc` release. `PROFILE_GPU=1` per-pass GPU 타임스탬프.
- 방법: `WARMUP_FRAMES=45`(콜드 SDF 베이크 + settle 통과) + 정적 카메라 62프레임 평균, 콜드/스파이크 제거.
- 앵글: **demo**(authored level view) / **nave**(`CAM_EYE=-14,2,0 CAM_TARGET=14,2,0`) /
  **overview**(`CAM_EYE=10,12,8 CAM_TARGET=0,2,0`). 하네스: `scratchpad/measure.py`.

## 핵심 결과 — Sponza가 60fps에서 크게 후퇴 (품질과 맞바꿈)

| | DX demo | VK demo | (perf-track 당시) |
|---|---:|---:|---:|
| **프레임 총합** | **27.2ms (37fps)** | **33.1ms (30fps)** | DX 12.6 / VK 16.6 (48³ SDF) |

wave-1은 **의도적으로 성능↔품질을 교환**했다: per-mesh SDF direct-sample(170³ 하이브리드 아틀라스)로
reflect/AO/GI march가 고정밀 필드를 돌게 되어 `gdf_reflect`가 3→10.7ms로 뛰고, 신규 AO 스택
(`gdf_ao`+`ssao`+`gi_volume` ≈ 5.6ms)이 추가됐다. 이 문서는 그 비용을 정확히 계량한다.

## 패스별 비용 — Sponza Med (ms, 62프레임 평균)

| 패스 | 전선 | DX demo | DX nave | DX over | VK demo | demo/over |
|---|---|---:|---:|---:|---:|---:|
| **gdf_reflect** | B(per-px march) | **10.74** | 7.16 | 3.94 | **13.46** | 2.7× |
| **sdf_cache_light** | B(캐시, 뷰독립) | **6.79** | 6.77 | 6.82 | **8.52** | 1.0× |
| **gdf_ao** | B(per-px march) | **3.15** | 1.35 | 0.64 | **3.78** | 4.9× |
| gi_volume | B(뷰독립) | 1.64 | 1.63 | 1.64 | 2.19 | 1.0× |
| ssao | B(per-px) | 0.82 | 0.93 | 0.93 | 0.91 | 0.9× |
| gdf_atrous ×2 | B(디노이저) | 0.80 | 0.78 | 0.79 | 0.80 | — |
| reflect_temporal | B | 0.72 | 0.73 | 0.74 | 0.70 | — |
| gbuffer | A(지오) | 0.85 | 0.89 | 0.94 | 0.91 | — |
| shadow | A(지오) | 0.83 | 0.78 | 0.79 | 0.92 | — |
| gdf_gi_upsample ×2 | B | 0.20 | 0.20 | 0.21 | 0.20 | — |
| gi/reflect_temporal 외 소계 | 기타 | <0.5 | | | | |
| **총합** | | **27.20** | **21.70** | **18.09** | **33.07** | |

### 귀속 (DX demo)
- **Top-3 = gdf_reflect 39.5% + sdf_cache_light 25.0% + gdf_ao 11.6% = 76%.** 나머지 24%는 gi_volume·
  ssao·디노이저·지오·기타.
- **전선 (B) GDF SW-RT 스택이 여전히 프레임의 ~90%.** (A) 지오(shadow+gbuffer) ≈ 1.7ms.
- **뷰 의존 계층**(demo≫overview): gdf_reflect 2.7×, gdf_ao **4.9×** — 화면 내 GI 픽셀 수에 비례
  (실내 데모가 최악, 탑다운 overview가 최선). = **내부 렌더 스케일 + TAAU**(qhd-perf 트랙)가 직접 공략.
- **뷰 독립 계층**: sdf_cache_light·gi_volume — 스케일 낮춰도 안 줄어듦. = **async-compute / 상각**만이 레버.

### 백엔드 격차 (VK/DX)
| 패스 | DX | VK | VK/DX |
|---|---:|---:|---:|
| gdf_reflect | 10.74 | 13.46 | 1.25× |
| sdf_cache_light | 6.79 | 8.52 | 1.25× |
| gdf_ao | 3.15 | 3.78 | 1.20× |
| gi_volume | 1.64 | 2.19 | 1.34× |

VK가 GDF 컴퓨트를 구조적으로 ~1.2–1.3× 느리게 도는 기존 격차 재확인(sponza-perf/qhd-perf와 동일). VK
헤드룸의 정공법은 **async-compute로 뷰독립 캐시 relight를 오버랩**(qhd-perf Stage 6에서 VK +33% 실증, opt-in).

## SCREEN_PROBE GI — 아직 성능 미최적화 (기본화 불가)

`SCREEN_PROBE=1`(최근 `97246dc`가 **패리티만** 고침)은 현재 큰 회귀:

| | 기본 GDF GI | SCREEN_PROBE | Δ |
|---|---:|---:|---:|
| DX demo | 27.2ms | **51.7ms (19fps)** | +24.5 |
| VK demo | 33.1ms | **79.3ms (13fps)** | +46.2 |

- **`screen_probe_trace` 단일 패스가 26.6ms(DX)/48.5ms(VK)** — 회귀 전부. gi_volume+gdf_gi(1.7ms)를
  대체하지만 트레이스가 미최적. 통합/필터(irradiance/integrate/filter)는 합쳐 ~0.3ms로 저렴.
- 결론: 정확성(패리티)은 회복됐으나 **per-probe 프리인테그레이션/프로브 밀도(P2/P5) 최적화 재적용 전에는
  opt-in 유지**. 기본 GDF GI 경로가 여전히 3배 빠름.

## 참고 — Gallery(레퍼런스, 바이트 동일 앵커)
DX 10.5ms / VK 14.5ms. 지배 = `gdf_gi` 7.5(DX)/11.0(VK)ms(풀해상, 갤러리는 하프해상 안 씀 = 앵커).
콘텐츠 최적화 대상 아님(패리티 앵커).

## 최적화 후보 (측정이 지시하는 우선순위)

1. **`gdf_reflect` (40%, 10.7/13.5ms)** — 반해상 + 96-step per-mesh SDF march. 레버: reflect march-step
   캡↓ / cone-trace LOD(`cone_k`)로 원거리 스텝 절감 / atlas mip march. 뷰의존이라 내부 스케일에도 반응.
   품질은 PT 잔차로 검증(반사는 육안 민감 → 조심).
2. **`sdf_cache_light` (25%, 6.8/8.5ms, 뷰독립)** — VK 바닥. 레버: **async-compute 오버랩**(VK 실증) /
   relight 주기·피드백 추가 상각.
3. **`gdf_ao` (12%, 3.2ms, 뷰의존 4.9×)** — 신규 GDF AO march. 레버: **하프해상 트레이스 + bilateral
   upsample**(GI와 동일 패턴, `gdf_gi_upsample` 재사용) / step↓ / GI 트레이스와 융합(한 번의 march로 AO+GI).
4. **경로 병합 검토** — gdf_ao·gi_volume·gdf_reflect가 모두 같은 per-mesh SDF를 독립 march. 공유 march /
   레이 재사용으로 중복 제거 여지(근본 원인 = 세 패스가 같은 필드를 3번 훑음).

### 두 갈래 60fps 경로
- **(A) 품질 보존 (native HD)**: gdf_reflect step/cone + gdf_ao 하프해상 + async 캐시 → wave-1 회귀의
  절반 회수 목표. 각 항목 RenderQuality 노브 결속 + PT 잔차 정직 보고.
- **(B) 해상도 (이미 구축)**: 내부 렌더 스케일 + TAAU(qhd-perf 트랙). 뷰의존 per-px 패스(reflect/ao)가
  직접 축소 → DX 0.5 스케일 ≈ 60fps+ 를 거의-네이티브 품질로. wave-1 품질을 그대로 유지하는 가장 빠른 길.

## 게이트 (CLAUDE.md 5원칙 준수)
착수 시: PROFILE_GPU before/after, 양 백엔드, 갤러리 무회귀(바이트 동일 앵커), DX≡VK ≤0.001, 품질 PT
잔차 재측정, fmt+clippy(-D warnings), Vulkan 검증 클린.
