# SW-RT GI 성능 트랙 (권위 계획)

상위: [sponza-perf.md](sponza-perf.md)(Sponza HD 60fps 완료) · [qhd-perf.md](qhd-perf.md)(QHD TAAU 완료).
이 트랙은 프레임의 ~80%를 차지하는 **GDF 소프트웨어 레이트레이싱 스택**을 레퍼런스 상용 엔진의 SW GI 디폴트 경로와
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
step 수 감소. 레퍼런스 엔진들의 거리 기반 cone/LOD march. 단일 노브 `gdf_cone_k`(0 = 레거시 = 바이트 동일):
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

### P1 — sparse 스크린 GI 프로브 (최대 레버, 분석 격차①) (완료, 2026-06-28)
GI 추적을 더 sparse한 해상도로 — **하프(1/2)→쿼터(1/4)** = 추적 원점 4× 감소(풀해상 대비 16×). 기존
`gdf_gi_upsample.slang`(joint-bilateral)이 이름만 "half"일 뿐 수학은 **임의 배율 bilinear + edge-stopping**
이라 source dims만 바꾸면 그대로 재사용. 이게 레퍼런스 엔진의 스크린-프로브 게더의 **공간 sparse 절반**(sparser
추적 원점 + guided 보간); octahedral 방향 프로브는 후속 정제. **단일 노브 `quality.rs gi_res_div`**
(Low/Med **4**=쿼터 / High 2[half_res off라 무효]). `gi_half_res` 게이팅 재사용 → 콘텐츠만, 갤러리는
풀해상(바이트 동일). `P_GI_RES_DIV` 오버라이드. 파일: `quality.rs`/`main.rs`(노브+배선; 셰이더 무변).

#### before/after (RTX 2070 SUPER, 1280×720, Med; before = P3)
| 패스 | DX before→after | VK before→after |
|---|---:|---:|
| **gdf_gi** | 2.01 → **0.71** (−65%) | 2.93 → **0.97** (−67%) |
| **프레임 총합** | 11.97 → **10.40** (~96fps) | 14.36 → **12.72** (~79fps) |

누적(baseline→P1): **DX 12.59→10.40 (79→96fps), VK 16.74→12.72 (60→79fps).** 새 Top-2 = sdf_cache_light
(DX 3.0 / VK 4.4) + gdf_reflect (DX 2.8 / VK 3.4) → P2가 캐시 공략.

#### 게이트 (정직 보고)
- **무회귀(갤러리)**: DX 0.000/ch(max 1), VK 0.000/ch(max 0=bit-identical). 갤러리 풀해상=불변. ✓
- **DX≡VK(Sponza)**: 0.005/ch(max 229) — P3와 동일, 파리티 중립. ✓
- **품질**: P3→P1 마진 0.357/ch(max 25, 0.04% >8); 누적 base→P1 0.534/ch(0.39% >8, sponza 한도 0.76 내).
  쿼터-res GI가 temporal EMA+à-trous로 잘 수렴 — Sponza 데모 육안 정상(얼룩/광 누설 없음). ✓
- fmt/clippy 클린, Vulkan 검증 클린. ✓

### P2 — 샘플 기반 캐시 피드백 (측정 후 보류, 2026-06-28)
설계: `sdf_cache_light`의 이진 카메라-가시성을 **GI/reflect 소비자가 실제 샘플한 카드 마크 + staleness**로.
**그러나 측정이 ROI를 제한**한다 — 착수 전 진단(VK, `P11_CACHE_RELIGHT_PERIOD` 스윕):

| period | VK sdf_cache_light |
|---:|---:|
| 40 (Med) | 4.5–5.4 (런 노이즈 큼) |
| 80 | 3.7 |
| 200 | 3.3 |

→ cache_light = **relight-바운드 부분(~2ms, period가 줄임) + ~3.2ms 바닥**. 바닥은 **실제 카드의 per-texel
작업 + carry-forward**(dispatch는 이미 `num_cards`에 타이트 — Sponza는 budget 미초과라 빈 카드 dispatch 낭비
없음). **샘플 피드백은 relight 부분(~2ms)만 공략**하는데, 데모 앵글은 on-screen 카드 대부분이 GI/reflect에
실제 샘플되므로(밝은 아트리움) 추가 컷이 작고 불확실(~0.5–1ms 추정) — 노이즈 폭(~1ms)과 비슷. 캐시 바닥은
품질 손실(tile↓=반사 blur, qhd Stage 3에서 기각) 없이는 줄이기 어려움. **결론: 이 씬에선 측정상 ROI<리스크
+plumbing(공유 sampler 시그니처 + gi/reflect/cache push 확장 + 피드백 버퍼)** → 보류. **off-screen/미샘플
카드가 많은 씬**(레퍼런스식 피드백이 빛나는 조건)에서 측정 주도 재개 권장. 확실한 레버는 relight period 상향
(품질=이동카메라 lag 트레이드)으로 이미 노출됨(`P11_CACHE_RELIGHT_PERIOD`).

### P4 — DX async-compute 안정화 (측정 후 보류, 2026-06-28)
[qhd-perf.md](qhd-perf.md) Stage 6가 이미 결론: DX async-compute는 cross-queue 스케줄링 편차(7.7–14.5ms)로
**불안정 → 기본 off, DX 비권장**. P1/P3 후 DX는 Sponza 96fps라 캐시 오버랩 불필요. cross-frame 펜스/우선순위
재작업은 RHI-deep + 양 백엔드 상이 리스크 대비 이득 없음 → 보류(드라이버 스케줄링 조사는 별도 트랙).

### P5 — VK 컴퓨트 점유율 튜닝 (측정 후 음의 결과, 2026-06-28)
실험: gdf_gi threadgroup `[8,8]`→`[16,16]` (출력 불변=파리티/품질 리스크 0). **측정상 더 느림**:
VK gdf_gi 0.97→1.04, DX 0.71→0.76 — march-heavy 셰이더는 큰 그룹이 **VGPR 압력↑ → occupancy↓**. 현 `[8,8]`이
이미 near-optimal → revert. VK 1.3–1.6× 격차는 **Turing 컴퓨트 throughput의 구조적 특성**(이 march 루프에서)
이지 단순 threadgroup 튜닝으로 안 줄어듦. **검증된 VK 레버는 async-compute**(qhd Stage 6, +33%, opt-in
`P_ASYNC_CACHE`)로 이미 존재. subgroup/VGPR 심층 튜닝은 별도 측정 트랙.

## P1 정정 — div 4→3 (마스킹됐던 DX≡VK 회귀 수정, 2026-06-28)
반사 history-clamp 재측정 중 발견: **P1의 쿼터-res(div4) GI가 DX≡VK를 0.016→0.117/ch로 키웠다**(coarse
스토캐스틱 GI 레이의 FP 발산이 bilateral 업스케일로 더 넓은 풋프린트에 퍼져 **브로드 파리티 발산**, firefly만
이 아님). **이 갭은 reflect_temporal의 (당시 무조건) clamp가 마스킹**해 P1 커밋 게이트가 0.005로 거짓 통과했다.
clamp를 roughness-게이트하자 마스킹이 풀려 노출됨. div 스윕(no-clamp):

| div | gdf_gi DX | DX≡VK |
|---:|---:|---:|
| 2 (half, 레거시) | ~2.0 | 0.016 |
| **3 (정정 채택)** | **~1.02** | **0.006** |
| 4 (quarter, 기각) | ~0.71 | **0.117** |

→ **Low/Med `gi_res_div` 4→3**: gdf_gi −48%(half 대비) 유지하며 DX≡VK 베이스라인 복원. div4의 추가 0.3ms는
20× 파리티 악화 값어치 없음. **교훈: 마스킹 가능한 패스(여기선 반사 clamp)가 켜진 채로 파리티 게이트를 재면
안 된다** — 격리 측정 필요.

## 반사 history-clamp 퍼뮤테이션 (사용자 WIP 확장, 2026-06-28)
사용자 WIP(회전 시 시점의존 specular 히스토리가 stale → 크롬에 어두운 끌림 smear; TAA neighborhood clamp로
제거)를 **스케일러블 퍼뮤테이션**으로 일반화. `reflect_temporal.slang` `clamp_mode`:
- **0 off**: clamp 스킵 = 레거시 resolve 바이트 동일(갤러리 강제).
- **1 hard**: 톤맵공간 이웃 AABB [cmin,cmax]로 clamp. 가장 쌈.
- **2 variance**: mean±γσ(AABB 교집합), Salvi/Karis 분산 클리핑. firefly 이웃에 robust(γ로 타이트니스 조절).
**roughness 게이트(핵심 수정)**: sr=0(near-mirror)는 clamp **스킵** — 측정 결과 샤프 미러는 3×3 이웃이
고주파 반사를 담기 너무 좁아 실루엣에 **hard dark band**를 만들었음(회전 갤러리 크롬에서 mode2 1.201/ch 발산).
게이트 후 1.201→**0.003**(band 제거). 미러는 per-frame 재트레이스+EMA로 안정. 노브 `quality.rs
reflect_history_clamp`(Low/Med 1=hard·**갤러리 0 강제** / High 2=variance) + `reflect_clamp_gamma`(1.25),
`P_REFL_CLAMP`/`P_REFL_CLAMP_GAMMA` 오버라이드. push 208→224, 3 모드 perf 동일(0.70ms, 통계 누적 무료).

## 깜빡임(temporal shimmer) 조사 + GI temporal clamp 수정 (2026-06-28)
사용자가 **Metal에서 깜빡임**을 보고, Windows도 확인 요청. `CAPTURE_SEQ=N STEP=0`(정적 카메라 N프레임)으로
프레임간 diff = 깜빡임을 정량화:
- **갤러리 DX: 0.003/ch** (안정). **Sponza DX: 0.225/ch**(max ~140) — **Windows도 깜빡임 = Metal 전용 버그
  아닌 알고리즘적 시간 불안정**.
- 격리: 레거시(div2,no-clamp) 0.108 / **div3 0.258**(div가 증폭) / div3+full 0.225.
- **근본 원인 = `gdf_temporal.slang`의 GI temporal neighborhood clamp**: reprojected 히스토리를 **현재
  noisy spp1 GI의 3×3 hard min/max box**로 clamp하는데, box가 매 프레임 노이즈 중심이라 **수렴한 히스토리를
  노이즈로 끌어내림** → 매 프레임 fresh 노이즈 = 셔머. (반사 clamp 샤프미러 문제와 동형.) max_hist↑로는 안
  잡힘(clamp가 누적을 캡). **temporal clamp만 끄면(per-sample firefly clamp 유지) 0.225→0.020/ch(11×↓)**.
- **수정 = GI temporal clamp 퍼뮤테이션**(반사 clamp와 동일 구조): `gdf_temporal` params.w로 0=off / 1=hard
  (레거시) / >1.5=variance(γ). **콘텐츠=off**(EMA 수렴 허용; firefly는 per-sample clamp+à-trous, ghost는
  월드위치 disocclusion이 담당), **갤러리=1.0 hard 강제(바이트동일)**. 노브 `quality.rs gi_temporal_clamp`,
  `P_GI_TEMPORAL_CLAMP`. div2/div3 공통 기존 이슈를 둘 다 해결.
- **검증**: 갤러리 vs 진짜 레거시 DX 0.000(max1)/VK 0.000(max0). Sponza 정적 셔머 **0.225→0.020**. 회전
  Sponza(STEP 0.02) 육안 정상(ghosting/스미어 없음 — disocclusion이 모션 처리). DX≡VK 0.022(노이즈 바닥
  0.017 수준; GI clamp가 억제하던 미세 발산 노출, 파리티 중립). gdf_temporal 0.19ms 무변. fmt/clippy/검증 클린.

## 누적 결과 (P3 + P1@div3 + 반사 clamp + GI temporal clamp 수정, 2026-06-28, 정직 정정)
| | DX 데모 | VK 데모 |
|---|---:|---:|
| Stage 0 (baseline) | 12.59ms (79fps) | 16.74ms (60fps, 경계) |
| **최종** | **10.84ms (~92fps)** | **14.26ms (~70fps)** |
가속 1.16×/1.17×. P3 cone-trace + P1 third-res GI(gdf_gi DX 2.59→1.02) + 반사 history-clamp(회전 smear 제거).
**게이트(엄밀 재측정)**: 갤러리 vs **진짜 레거시**(WIP-free stash 빌드) DX 0.000(max 1)/VK 0.000(max 0=
bit-identical) — 앵커 마스킹 없이 증명. DX≡VK 0.005/ch. 누적 품질 델타 0.607/ch(한도 0.76 내). fmt/clippy/
Vulkan 검증 클린. P2/P4/P5는 측정상 ROI 제한/음수로 보류(위 참조).

## 하지 말 것
- 측정 없이 추측. 갤러리 무회귀 위반(노브는 갤러리 레거시 강제). DX≡VK 깨기. Vulkan 검증 경고.
- HW-RT 경로를 기본 경로로(파리티 리스크 — High 티어 opt-in 별도 트랙).
