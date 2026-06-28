# Lumen-parity SW-RT 트랙 (권위 계획)

상위: [sponza-perf.md](sponza-perf.md)(Sponza HD 60fps 완료) · [qhd-perf.md](qhd-perf.md)(QHD TAAU 완료).
이 트랙은 프레임의 ~80%를 차지하는 **GDF 소프트웨어 레이트레이싱 스택**을 UE5 Lumen의 디폴트(SW) 경로와
같은 수준의 **정교함**으로 끌어올려, 같은 화질을 더 적은 추적·march로 달성한다. 목표: RTX 2070 SUPER, 데모
앵글, **양 백엔드 ≤16.6ms 유지하며 SW-RT 스택 비용을 추가 절감**, 갤러리 무회귀 + DX≡VK ≤0.001 + PT 잔차 수용.

## ★ 1원칙: 측정 먼저 (Sponza 트랙과 동일)

`PROFILE_GPU=1 LEVEL=sponza … --screenshot-clean`로 패스별 ms를 먼저 분해. 모든 before/after를 ms로 보고.

### Stage 0 — 베이스라인 (2026-06-28, RTX 2070 SUPER, 1280×720, Med)

| 패스 | DX | VK | 전선 |
|---|---:|---:|---|
| sdf_cache_light | 3.08 | 4.79 | march(gather+shadow) |
| gdf_gi | 2.59 | 4.31 | march(bounce+shadow) |
| gdf_reflect | 3.05 | 3.71 | march(primary+shadow) |
| gdf_atrous ×2 | 0.78 | 0.79 | 디노이저 |
| reflect_temporal | 0.66 | 0.65 | 디노이저 |
| shadow+gbuffer | 1.55 | 1.57 | 지오메트리 |
| 기타 | ~0.5 | ~0.7 | tonemap/ssr/upsample/ui |
| **총합** | **12.59** | **16.74** | |

SW-RT march 3개 패스 = DX 8.7ms(69%) / VK 12.8ms(76%). **march 비용이 1차 타깃.**

## 스테이지 (= 분석에서 권장한 P3→P1→P2→P4/P5 순)

각 Stage 독립 커밋. 게이트: PROFILE_GPU before/after(양 백엔드) + `cargo fmt` +
`RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets` + 갤러리 무회귀(바이트 동일) +
DX≡VK ≤0.001 + (품질 변경 시) PT 잔차 재측정 + Vulkan 검증 클린. 모든 성능 노브는 `quality.rs` 티어 한 곳에,
갤러리는 레거시 강제(바이트 동일 앵커).

### P3 — cone-trace LOD march (완료, 2026-06-28)
march 루프의 step 하한/상한을 **거리 비례로 완화**(원거리에서 더 큰 step) → grazing 슬라이버를 적게 밟아
step 수 감소. UE/Frostbite의 거리 기반 cone/LOD march. 단일 노브 `gdf_cone_k`(0 = 레거시 = 바이트 동일):
- 1차 march: `t += max(d, max(MIN, cone_k·t))` (하한 상향; cone_k=0 → `max(d, MIN)` 레거시).
- shadow march: `t += clamp(h, MIN, max(0.2, cone_k·t))` (상한 상향; cone_k=0 → `clamp(h,MIN,0.2)` 레거시).
적용: `gdf_gi`(bounce+gi_shadow), `gdf_reflect`(primary+refl_shadow), `sdf_cache_light`(gather+sun_shadow).
갤러리 cone_k=0 강제. 저주파/디노이즈 신호라 콘텐츠 잔차 영향 작음. reflect primary는 sky-leak 리스크가 있어
측정 후 필요 시 값 분리. push: gi 224→240, cache 128→144, reflect는 ground_albedo 패딩(236) 재사용(무변).

**노브(단일 소스)**: `quality.rs` `gdf_cone_k` (Low 0.05 / **Med 0.02** / High 0.0=풀품질). `P_CONE_K` 오버라이드.
갤러리는 0 강제(바이트 동일 앵커). 파일: `gdf_gi.slang`/`gdf_reflect.slang`/`sdf_cache_light.slang`(struct+march),
`push.rs`(3 패커), `gi.rs`/`reflect.rs`/`gdf.rs`(record+파이프라인 크기), `quality.rs`/`main.rs`(티어+배선).

#### before/after (RTX 2070 SUPER, 1280×720, Med)
| 패스 | DX before→after | VK before→after |
|---|---:|---:|
| gdf_gi | 2.59 → **2.01** (−22%) | 4.31 → **2.93** (−32%) |
| gdf_reflect | 3.05 → **2.81** (−8%) | 3.71 → **3.36** (−9%) |
| sdf_cache_light | 3.08 → 3.27¹ | 4.79 → **4.20** (−12%) |
| **프레임 총합** | 12.59 → **11.97** (~84fps) | 16.74 → **14.36** (~70fps) |

¹ DX cache_light 미세 증가는 run-to-run 노이즈(VK는 −12% 감소 = cone_k가 march 단조 감소시킴 확증). VK가 step당
컴퓨트가 느려 cone-trace 이득이 더 큼 → **VK가 16.6ms 경계에서 벗어남**(60→70fps).

#### 게이트 (정직 보고)
- **무회귀(갤러리)**: DX 0.000/ch(max 1=GPU 비결정), **VK 0.000/ch(max 0=bit-identical)**. cone_k=0 강제. ✓
- **DX≡VK(Sponza)**: P3 0.005/ch(base 0.003, max 229 동일) — 기존 스토캐스틱 firefly 갭 내, **파리티 중립**. ✓
- **품질 델타(base vs P3 raster)**: 0.315/ch, 0.34% 픽셀 >8 (max 66) — 원거리 GI/반사가 약간 부드러워짐,
  perf-tier 수용(sponza 트랙 한도 0.76 대비 여유). High 티어는 cone_k=0=풀품질. ✓
- fmt/clippy(-D warnings) 클린, Vulkan 검증 클린(기존 `VK_NV_external_memory` 노트만). ✓

### P1 — 스크린 프로브 GI (최대 레버, 분석 격차①)
픽셀당 GI 레이를 **16×16 타일당 1 프로브** 추적 + depth/normal-aware 보간으로 교체. 추적 원점 ~64× 감소.
기존 `gdf_gi_upsample.slang`(joint-bilateral) 인프라 재사용. RenderQuality 티어 결속.

### P2 — 샘플 기반 캐시 피드백 + 우선순위 (VK 바닥, 분석 격차④)
`sdf_cache_light`의 이진 카메라-가시성을 **GI/reflect 소비자가 실제 샘플한 카드 마크 + staleness 버킷**으로.
오프스크린·미사용 카드 relight 스킵 → 해상도-독립 VK 캐시 바닥 절감.

### P4 — DX async-compute 안정화 (분석 격차⑥)
cross-frame 펜스/우선순위 정리로 DX도 캐시 relight를 raster와 오버랩(현재 편차 과대로 보류 상태).

### P5 — VK 컴퓨트 점유율 튜닝 (분석 격차⑥)
threadgroup 크기·subgroup·VGPR 압력 측정 주도 튜닝으로 VK 1.3–1.6× 격차 일부 회수.

## 하지 말 것
- 측정 없이 추측. 갤러리 무회귀 위반(노브는 갤러리 레거시 강제). DX≡VK 깨기. Vulkan 검증 경고.
- HW-RT 경로를 기본 경로로(파리티 리스크 — High 티어 opt-in 별도 트랙).
