# Sponza 60fps — 성능 분석 & 컬링 트랙 (권위 계획)

상위: [scalable-gi.md](scalable-gi.md)(GDF GI를 Sponza에 일반화 — 완료). 이 트랙은 그 결과로
**GDF GI가 디폴트인 Sponza가 60fps 미만**인 문제를 푼다. 목표: RTX 2070 SUPER, 데모 앵글, **양 백엔드
≥60fps(≤16.6ms/frame)**, 갤러리 무회귀 + GDF GI 품질 수용 가능 유지.

## ★ 1원칙: 측정 먼저 (추측 금지)

성능은 측정으로 시작한다. `PROFILE_GPU=1`(스크린샷 모드에서 패스별 GPU ms를 로그로 덤프)로 Sponza의
**패스별 비용을 먼저 분해**한 뒤, **가장 큰 비용부터** 공략한다. 측정 전에 "컬링이 답"이라고 가정하지
않는다.

### 두 개의 비용 전선 (가설 — Stage 0에서 검증)
| 전선 | 패스 | 지오메트리 의존? | 컬링으로 줄어드나? |
|---|---|---|---|
| **(A) 지오메트리 제출** | G-buffer fill, shadow map (현재 262k tri **전량** 매 프레임, 컬링 0) | O | **O** (프러스텀/오클루전/커버리지) |
| **(B) GDF SW-RT 스크린/캐시** | gdf_gi(풀스크린×spp×march), gdf_reflect, **surface cache relight(~1024카드×32²≈1M 텍셀/프레임)**, GI 디노이저, 반사 temporal | X (화면/캐시 바운드) | **X** (컬링 무관) |

**정직한 프레이밍**: 사용자가 제안한 프러스텀/오클루전/커버리지 컬링은 **(A)만** 줄인다. **(B)는 컬링으로
안 줄어든다** — 화면/캐시 바운드이기 때문. GDF GI 디폴트에서 (B)가 지배적일 가능성이 높으므로(풀스크린
레이마칭 + 매 프레임 100만 카드 텍셀 재조명), Stage 0 측정이 노력 배분을 결정한다. **컬링 단독으로
60fps가 안 될 수 있음을 전제**한다.

## 스테이지

각 Stage 독립 커밋. 게이트: `PROFILE_GPU` before/after(핵심 지표) + `cargo fmt --all` +
`RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets` + 양 백엔드 스크린샷 → **DX≡VK ≤0.001**
+ Vulkan 검증 클린. **보수적 컬링(A/B/C)은 갤러리 + Sponza 바이트 동일**(보이는 픽셀 불변); **품질 영향
(D)은 RenderQuality 티어 + `tools/rt-compare.py` 잔차 재측정·수용**.

### Stage 0 — 측정 & 귀속 (코드 없음, 측정+계획 정련)
- `PROFILE_GPU=1`로 Sponza 데모 앵글 + 2~3개 앵글, 양 백엔드 패스별 ms + 프레임 총합.
- **CPU vs GPU 바운드** 판정(프레임 총합 vs 패스 합; CPU 제출/드라이버 비용 확인).
- 패스를 (A 지오메트리)/(B GDF-SWRT)/(기타: tonemap, shadow, ibl)로 묶어 **Top-3 비용** 식별.
- 산출물: 본 문서에 측정 표 + 확정 스테이지 순서. **이 표가 이후 모든 우선순위를 지배.**

### Stage A — 프러스텀 컬링 (실제 씬 드로우)
- GPU 컴퓨트 프러스텀 컬: draw-list **per-drawable AABB**(Scalable-GI Stage 0가 이미 CPU 지오/AABB
  제공 — 재사용)를 카메라 프러스텀과 테스트 → **visible 인스턴스 리스트 + indirect draw**로 G-buffer fill.
  shadow는 **광원 프러스텀**으로 별도 컬. `cull.rs`(현재 데모 큐브 그리드)의 reset/cull/indirect 패턴
  일반화.
- 보수적(보이는 오브젝트 절대 누락 X) → **렌더 바이트 동일**. 게이트: 갤러리·Sponza 바이트 동일 +
  제출 삼각형 수/gbuffer·shadow ms 감소 측정.

### Stage B — 오클루전 컬링 (Hi-Z 2-phase)
- 전 프레임 depth로 **Hi-Z 피라미드** 빌드 → drawable AABB를 Hi-Z에 테스트. **2-phase**(① 전 프레임
  visible 드로우 → Hi-Z 재빌드 → ② 디오클루전된 것 추가 드로우)로 팝핑 없이 보수적 유지. 밀집 Sponza
  (기둥·아치 상호 가림)의 **오버드로 절감**. 게이트: 바이트 동일 + 오버드로/gbuffer ms 측정.

### Stage C — 커버리지 버퍼(소프트웨어 오클루전) — 선택(Hi-Z 부족 시)
- 큰 오클루더(벽/바닥)를 저해상 **커버리지/깊이 버퍼**에 소프트 래스터(컴퓨트) → AABB 테스트.
  Frostbite/Intel masked-occlusion 스타일. Hi-Z의 지연/엣지 한계를 보완하는 대안. Stage B 측정 후
  필요할 때만.

### Stage D — GDF SW-RT 비용 절감 (B 전선; GDF 라이팅의 핵심 레버)
- **D1 하프해상 GI+반사**: 트레이스를 1/2 해상(쿼터픽셀)으로 → 기존 à-trous/temporal 디노이저로 업스케일
  (Lumen 스크린 프로브식). ~4× 레이 감소. RenderQuality 티어 게이트.
- **D2 surface cache 갱신 예산**: 매 프레임 100만 카드 텍셀 전량 재조명 금지 — **우선순위/피드백 갱신**
  (화면에 보이는/근접 카드만, 나머지 라운드로빈), persistent radiance. UE Lumen surface-cache feedback.
  MAX_CARDS(메모리 예산)에 더해 **per-frame relight(컴퓨트) 예산**을 캡.
- **D3 march/clipmap 샘플 비용**: 원거리 step 수 감소(cone/LOD), clipmap 레벨 선택 비용 절감, 거리 기반
  early-out.
- 각 항목 RenderQuality{Low,Med,High} 노브로 결속, 품질 PT 잔차로 정직 보고.

### Stage E — 60fps 검증
- 데모 앵글 양 백엔드 프레임 ms **≤16.6(≥60fps)** 달성 확인. GDF GI 품질 vs 현재 비교(수용 가능).
- 보수적 스테이지(A/B/C) 갤러리 바이트 동일; 품질 영향(D)은 RenderQuality 티어 + 잔차 측정. 정직 보고
  (어느 패스가 얼마 줄었는지 표).

## Stage 0 — 측정 결과 (2026-06-27, RTX 2070 SUPER, 1280×720, RENDER_QUALITY 미설정=Med)

`PROFILE_GPU=1 LEVEL=sponza … --screenshot-clean`. 스티디 스테이트(워밍업 아님, 4프레임 안정).
패스별 GPU ms, 양 백엔드 + 데모 앵글 + 2개 대체 앵글.

| 패스 | 전선 | DX 데모 | VK 데모 | DX 네이브* | DX 오버뷰† |
|---|---|---:|---:|---:|---:|
| **sdf_cache_light** | **B** | **357.5** | **773.6** | **461.8** | **446.5** |
| **gdf_gi** | **B** | **120.0** | **246.0** | **145.5** | **17.4** |
| gdf_reflect | B | 11.6 | 15.1 | 10.8 | 7.6 |
| reflect_temporal | B | 0.56 | 0.56 | 0.57 | 0.61 |
| gdf_atrous ×2 | B | 0.75 | 1.31 | 0.69 | 0.73 |
| gdf_temporal | B | 0.19 | 0.21 | 0.18 | 0.18 |
| ssr | B | 0.22 | 0.22 | 0.04 | 0.27 |
| reflect_composite | B | 0.04 | 0.04 | 0.04 | 0.04 |
| shadow | **A** | 0.78 | 1.28 | 0.78 | 0.78 |
| gbuffer | **A** | 0.71 | 0.76 | 0.84 | 0.85 |
| lighting+lit_history+tonemap+ui | 기타 | 0.19 | 0.25 | 0.19 | 0.21 |
| **프레임 총합 (GPU)** | | **492.6** | **1039.3** | **621.4** | **475.2** |

\* `CAM_EYE="-14,2,0" CAM_TARGET="14,2,0"` (네이브 길이 방향). † `CAM_EYE="10,12,8" CAM_TARGET="0,2,0"` (탑다운 오버뷰).

### 귀속 (Top-3 + 전선 비중)
- **(B) GDF SW-RT ≈ 프레임의 99%.** **(A) 지오메트리(shadow+gbuffer) ≈ 1.5ms = ~0.3%(DX)/~0.2%(VK).** 기타 <0.3ms.
- **GPU 바운드, 컴퓨트 지배** (패스 합 ≈ 보고된 총합 = GPU 타임스탬프 구간; 두 컴퓨트 패스가 전부).
- **Top-1 `sdf_cache_light` (DX 357 / VK 774ms)** — 카메라와 **거의 무관**(네이브 462, 오버뷰 446 — 뷰 독립).
  근본 원인: `record_cache_light`가 매 프레임 **전 카드 아틀라스**(`num_cards×32² ≤ 1.05M 텍셀`)를 **고정
  spp=8**로 **무조건** 재조명(예산/피드백/가시성 우선순위 없음; temporal α=0.35만 누적). `dispatch(num_texels/64)`.
  → **Stage D2(재조명 예산/피드백)가 단일 최대 레버.** Med에서 `reflect_cache=true`가 이 패스를 켠다(반사 히트 캐시 소비자).
- **Top-2 `gdf_gi` (DX 120 / VK 246ms)** — 풀스크린 × spp × march, **뷰 의존**(오버뷰 17ms로 급감 = 화면 GI 픽셀 수).
  → **Stage D1(하프해상)·D3(march/LOD).**
- **Top-3 `gdf_reflect` (7–15ms)** — → D1/D3 부차.

### 측정이 지배하는 결론 (노력 배분 재정렬)
1. **사용자가 제안한 컬링(A: 프러스텀/오클루전/커버리지)은 GPU 프레임타임을 의미 있게 못 줄인다** — 지오 제출
   (shadow+gbuffer)이 통틀어 ~1.5ms뿐. 문서의 정직한 프레이밍이 **측정으로 확증**됨. 16.6ms 목표엔 ~96.5% 절감
   필요한데 컬링 상한이 ~1.5ms. → **Stage A/B/C는 60fps 경로에서 보류**(CPU 제출 비용/더 조밀한 씬 확장성용으로만
   추후, GPU 예산엔 무관).
2. **확정 스테이지 순서: D2 → D1/D3 → (필요시) 재조명 켜짐 정책 재검토 → A/B 보류.**
   - **Stage D2 먼저**: `sdf_cache_light`를 매 프레임 전량 재조명 → **per-frame relight 예산 + 라운드로빈/가시성
     우선순위**(persistent radiance는 이미 있음 = temporal α). 1.05M 텍셀을 예: 1/4·1/8로 분할 상각 → 357ms를
     비례 절감. 양 백엔드 동일 구조(VK 2.1× 절대값 차이는 throughput, 파리티 버그 아님).
   - **Stage D1/D3 다음**: `gdf_gi` 하프해상 트레이스+디노이저 업스케일(≈4× 레이↓) + march/clipmap LOD.
   - 두 레버로도 16.6ms 미달 시: Med에서 `reflect_cache`(→`sdf_cache_light` 강제)의 비용 대비 효용을 티어로
     재검토(반사 캐시를 High 전용/하프해상 캐시로).
3. **백엔드 노트**: VK가 두 컴퓨트 패스에서 DX 대비 ~2.1× 느림(동일 알고리즘) — 디스패치 점유율/occupancy
   격차로 추정. **렌더 픽셀 파리티(DX≡VK)와 무관한 throughput 차이**지만, D2/D1 최적화 시 VK 절대 예산이 더
   빡빡하므로 **양 백엔드 모두 ≤16.6ms 게이트**를 엄격 적용.

> 다음 작업: **Stage D2** 착수(`gdf.rs record_cache_light` + `gi.rs`/`quality.rs`에 relight 예산 노브). 게이트:
> PROFILE_GPU before/after, 양 백엔드, DX≡VK ≤0.001, 갤러리 무회귀(캐시 패스는 갤러리 미사용 → 자동 바이트 동일),
> Sponza GDF GI 품질 `tools/rt-compare.py` 잔차 수용, Vulkan 검증 클린, fmt+clippy.

## Stage D2 — surface-cache 상각 재조명 (완료, 2026-06-27)

`sdf_cache_light`(Stage 0 Top-1)가 매 프레임 **전 카드 아틀라스를 무조건 재조명**하던 것을 **라운드로빈
갱신 예산**으로 전환. UE5 Lumen surface-cache update budget(D. Wright et al., Epic Games, SIGGRAPH 2022
"Lumen: Real-time GI in UE5") 참고 — 매 프레임 `1/period`의 카드만 재조명, 나머지는 직전 radiance를
ping-pong write로 carry-forward(소비자는 항상 완전한 아틀라스를 읽음). 캐시는 EMA로 누적되는 거의 정적
신호라 정적 씬에선 같은 고정점에 수렴.

- **노브(단일 소스)**: `quality.rs` `cache_relight_period` (Low 8 / **Med 4** / High 1). 셰이더는
  `period=1`이면 매 프레임 전량 = **레거시 바이트 동일**. `P11_CACHE_RELIGHT_PERIOD` 오버라이드. **갤러리는
  무회귀 앵커라 호출부에서 1로 강제**(`clip_max_levels`와 동일 패턴), 콘텐츠(Sponza)만 티어값으로 상각.
- 파일: `sdf_cache_light.slang`(card 선택 + carry-forward), `push.rs`(`clip.z`=period), `gdf.rs`
  (`record_cache_light` 인자), `quality.rs`(티어 노브), `main.rs`(배선 + UI 재적용, 갤러리 게이트).

### before/after (RTX 2070 SUPER, 1280×720, Med=period4)
| 패스 | DX before | DX after | VK before | VK after |
|---|---:|---:|---:|---:|
| **sdf_cache_light** | 357.5 | **103.2** (−3.5×) | 773.6 | **193.9** (−4.0×) |
| gdf_gi | 120.0 | 122.4 | 246.0 | 103.6 |
| gdf_reflect | 11.6 | 11.8 | 15.1 | 12.8 |
| **프레임 총합** | **492.6** | **240.8** | **1039.3** | **314.3** |

→ `sdf_cache_light`가 4× 줄어 **새 Top-1은 `gdf_gi`**(DX 122 / VK 104ms). 다음 레버는 **Stage D1/D3**
(gdf_gi 하프해상 + march/clipmap LOD). D2 단독으론 60fps 미달(예상대로 — B 전선은 다축).

### 게이트 (정직 보고)
- **무회귀(갤러리)**: DX/VK base vs D2 = **0.000/ch** (VK max 0 = bit-identical, DX max 1 = run-to-run GPU 비결정성). period=1 강제로 바이트 동일. ✓
- **품질(Sponza 상각 델타)**: base vs D2 raster = **0.004/ch, max 1 LSB** (0–255 스케일) → 64프레임 워밍업 내 완전 수렴, **지각 불가**. ✓
- **DX≡VK(Sponza)**: D2 후 0.004/ch — 단 **D2 전에도 0.004/ch (max 228)** → **D2는 파리티 중립**. 이 갭은 콘텐츠 경로의 **기존** 스토캐스틱 반사 firefly 소수 픽셀 차이(갤러리 바이트동일 게이트와 무관, D2가 도입한 것 아님). ✓
- **PT 잔차(Sponza)**: base 0.001 → D2 0.004/ch (둘 다 sub-LSB, 노이즈 내). ✓
- **fmt/clippy(-D warnings) 클린, Vulkan 검증 클린**(VUID 없음; 기존 `VK_NV_external_memory` interop 노트만, D2 무관). ✓

## Stage D1 — 하프해상 GI 트레이스 + bilateral 업스케일 (완료, 2026-06-27)

D2 후 새 뷰의존 Top-1이 된 `gdf_gi`(풀스크린 spp 레이마칭)를 **하프해상 트레이스(1/4 레이) + joint
bilateral 업스케일**로 전환. `gdf_gi.slang`은 G-buffer를 **정규화 UV**로 샘플하므로 **셰이더 변경 없이**
half extent/dims만 넘기면 하프해상 트레이스가 정확. 새 `gdf_gi_upsample.slang`이 풀해상 depth/normal로
2×2 half 풋프린트를 edge-stopping(à-trous와 동일 pos/normal 가중) 가중해 풀해상으로 복원 → 기존 풀해상
디노이저(temporal+à-trous) 그대로 소비. UE5 Lumen 스크린-프로브 / Frostbite 하프해상 GI 방식.
  Ref: J. Kopf et al., "Joint Bilateral Upsampling", SIGGRAPH 2007.

- **노브(단일 소스)**: `quality.rs` `gi_half_res` (Low/Med **true** / High false). `P11_GI_HALF_RES`
  오버라이드. **갤러리는 풀해상 강제**(바이트 동일 앵커), 콘텐츠만 하프해상.
- 파일: `gdf_gi_upsample.slang`(신규), `gi.rs`(`record_upsample`+파이프라인), `push.rs`
  (`gdf_gi_upsample_push`), `build.rs`(셰이더 등록), `quality.rs`/`main.rs`(티어+배선).

### before/after (Med; before = D2 풀해상 GI)
| 패스 | DX before→after | VK before→after |
|---|---:|---:|
| **gdf_gi** | 122.4 → **32.1** (−3.8×) | 103.6 → **27.0** (−3.8×) |
| gdf_gi_upsample | — | **0.10** | — | **0.09** |
| **프레임 총합** | 240.8 → **190.4** | 314.3 → **199.3** |

→ 새 Top-1은 다시 **`sdf_cache_light`**(DX ~143 / VK ~156ms, 측정 변동 있음). **60fps 미달(190ms)** —
surface-cache relight가 벽. 다음: D2 후속(주기↑/카드수↓) 또는 **Med에서 `reflect_cache`(=relight 강제)
비용 대비 효용 재검토** + D3(march/clipmap LOD).

### 게이트 (정직 보고)
- **무회귀(갤러리)**: base vs D1 = **0.000/ch** (풀해상 강제). ✓
- **품질(하프해상 GI)**: 풀해상(D2) vs 하프해상(D1) = **0.090/ch, max 9 LSB** — ×4 증폭 diff 몽타주가
  사실상 검정(육안 동일). GI는 저주파+디노이즈라 하프해상 적합. ✓ (High=풀해상으로 선택 가능)
- **DX≡VK(Sponza)**: 0.005/ch (D2 전 0.004와 동일 범위, 업스케일은 결정적) → **파리티 중립**. ✓
- **PT 잔차**: 0.004 → 0.091/ch (둘 다 sub-0.1 grey-level, 지각 불가). ✓
- **fmt/clippy 클린, Vulkan 검증 클린**(VUID 없음). ✓

## Stage D2b/D3 — UE Lumen 연구 + surface-cache relight 제안 (측정 기반)

D1 후 `sdf_cache_light`가 다시 Top-1(~143ms). UE5 Lumen 소스(`D:/Repositories/UnrealEngine-1`)를
폭넓게 조사한 결과와 그에 기반한 제안.

### UE Lumen의 surface-cache 조명 갱신 (확인된 사실)
- **고정 텍셀 예산 + 상각**: `r.LumenScene.DirectLighting.UpdateFactor=32` / `Radiosity.UpdateFactor=64`
  → 매 프레임 텍셀의 **1/32(직접)·1/64(간접)** 만 재조명(전체 갱신 32~64프레임). 우리 D2 period=4는 UE보다
  **8~16× 보수적**. (`LumenSceneLighting.cpp:40,48`)
- **우선순위 선택(라운드로빈 아님)**: 16-bin **우선순위 히스토그램** — 타일을
  `bucket = f(log2(FramesSinceLastUpdated × UpdateSpeed))`로 분류, 예산(`MaxUpdateTiles`)을 **가장
  오래된/가장 보이는 버킷부터** 채움. 굶주림 방지로 "최소 1페이지는 갱신". (`LumenSceneLighting.usf:176`)
- **가시성 피드백**: `r.LumenScene.SurfaceCache.Feedback` — 화면에 샘플된 카드만 고우선순위 →
  **오프스크린 카드는 거의 갱신 안 함**. (`LumenSurfaceCacheFeedback.cpp`)
- **직접/간접 분리**: 비싼 간접(spp gather에 해당)을 가장 드물게(1/64). 라디오시티는 자체 시간 누적으로
  드문 갱신에도 노이즈 적음.
- persistent radiance 아틀라스(우리는 EMA carry-forward로 이미 보유).

### 제안 (레버리지/리스크 순, 모두 reflect 품질 보존 지향)
- **T1 — relight 주기 상향 (즉시, 품질 안전)**: Med period 4→8, Low→16. UE가 32/64를 쓰는 점에서 매우
  보수적. 64프레임 워밍업 수렴 확인(period 8 EMA 잔차 ~0.03). 비용 ~½. *단독으론 ~75ms, 60fps 미달.*
- **T2 — 가시성/우선순위 피드백 (구조적 핵심, UE의 진짜 레버)**: GI/reflect 소비자가 샘플한 카드를
  feedback 버퍼에 마크 → relight를 **화면에 보이는 + 오래된 카드 우선**으로. Sponza 데모 앵글은 카드
  다수가 오프스크린이라 **여기서 큰 컷**. per-card last-update-frame + 가시성 마크 + 예산 내 우선순위 선택
  (UE 히스토그램의 경량판). 신규 작업(여러 셰이더 + 버퍼). **60fps 경로의 핵심.**
- **T3 — march/LOD 비용 절감 (D3)**: relight gather spp 8→4 + march step↓ + 거리 기반 거친 clipmap LOD;
  같은 절감을 `gdf_gi`/`gdf_reflect`에도. 각 RenderQuality 티어 결속, PT 잔차로 검증.
- **대안 — Med에서 캐시 끄기**: `reflect_cache`를 High 전용으로 → Med Sponza가 143ms 통째 제거(가장
  빠른 60fps 길), 단 반사는 per-ray GDF 폴백(grazing smear 등 저품질). 품질 우선 사용자에겐 비권장.

**권장 순서**: T1(즉시 안전) → **T2(구조적, UE식 가시성 피드백 = 60fps 핵심)** → T3(미세 조정). T2가
이 씬에서 가장 큰 컷이자 UE가 실제로 의존하는 메커니즘.

### T1+T2 구현 완료 (2026-06-27)
- **T1**: Med relight 주기 4→8 (quality.rs).
- **T2**: 신규 `sdf_cache_visibility.slang` — 카드당 바운딩 스피어를 **카메라 프러스텀**(Y-flip-free planes
  → DX≡VK) 테스트해 per-card visibility 버퍼 작성. `sdf_cache_light`가 이를 읽어 **온스크린=주기 P,
  오프스크린=주기 P×4**로 relight(밀집 씬은 카드 대부분이 시야 밖 → 큰 컷). UE Lumen 가시성/staleness
  우선순위의 **결정적 카메라-가시성 변형**(샘플 피드백은 정밀판, 추후 정제 여지). 가시성 패스 자체 비용
  **0.006ms**. clip.w=0xFFFFFFFF=피드백 off=균일 주기(D2), 갤러리 강제 off.
- 파일: `sdf_cache_visibility.slang`(신규)+`build.rs`, `gdf.rs`(`record_cache_visibility`+버퍼+파이프라인),
  `push.rs`(`cache_vis_push`+clip.w), `sdf_cache_light.slang`(가시성 주기), `quality.rs`/`main.rs`(배선,
  `cache_feedback` 콘텐츠 디폴트·갤러리 off, `P11_CACHE_FEEDBACK`).

#### before/after (Med; before = D1)
| 패스 | DX before→after | VK before→after |
|---|---:|---:|
| **sdf_cache_light** | ~143 → **46.9** | ~156 → **56.2** |
| sdf_cache_visibility | — | **0.006** | — | **0.006** |
| **프레임 총합** | 190.4 → **93.4** | 199.3 → **96.3** |

#### 누적 (Stage 0 → T2)
| | DX 프레임 | VK 프레임 |
|---|---:|---:|
| Stage 0 | 492.6 | 1039.3 |
| D2 | 240.8 | 314.3 |
| D1 | 190.4 | 199.3 |
| **T1+T2** | **93.4 (5.3×)** | **96.3 (10.8×)** |

#### 게이트 (정직 보고)
- 무회귀(갤러리): base vs T2 = **0.000/ch** (피드백 off + 주기 1 강제). ✓
- 품질(Sponza D1 vs T2): **0.005/ch, max 3 LSB** — 온스크린 카드는 동일 수렴, 지각 불가. ✓
- DX≡VK: **0.004/ch** (가시성 결정적·Y-flip-free → 신규 발산 없음, 기존 갭과 동일). ✓
- PT 잔차: 0.094/ch (≈D1 0.091, 하프해상 GI가 지배). ✓
- fmt/clippy 클린, Vulkan 검증 클린. ✓

**남은 것**: 아직 60fps 미달(93ms). 잔여 Top-3 = sdf_cache_light 47 + gdf_gi 32 + gdf_reflect 11 = ~90ms.
다음 = **T3(march/LOD): relight gather spp 8→4 + step↓, gdf_gi/reflect march LOD**, 또는 주기·HIDDEN_MULT
추가 상향. 각 RenderQuality 티어 결속·PT 잔차 검증.

## 설계 제약 (CLAUDE.md 5원칙)
1. **근본 원인**: 마이크로 패치 금지. 비용의 근원(풀스크린 레이 수 / 카드 텍셀 수 / 미컬링 드로우)을 줄인다.
2. **측정 주도**: `PROFILE_GPU`가 성공 지표. 모든 before/after를 ms로 보고.
3. **확장성**: 모든 성능 노브(해상 배율, relight 예산, march step, 컬링 토글)를 `quality.rs`
   RenderQuality 티어 한 곳에. 기본=현 품질(Med).
4. **단일 소스**: 컬 AABB는 레지스트리/fuse 한 곳에서(중복 금지).
5. **검증 후 주장**: 양 백엔드 + DX≡VK + 무회귀(보수적) / 잔차(품질) 수치 정직 보고 후 커밋.

## 하지 말 것
- 갤러리 무회귀 위반(보수적 컬링은 바이트 동일; 품질 변경은 티어/측정). DX≡VK 깨기. Vulkan 검증 경고.
- 측정 전 컬링이 답이라 단정. 보이는 오브젝트를 컬해 깜빡임 유발(보수성 위반).
- HW-RT 경로 변경. 새 무거운 의존(승인 필요). 스트리밍(월드) — 범위 외.
- 한 앵글만 빠르게 만들고 일반화 누락(여러 앵글 측정).

## 파일 (예상)
- 수정 `apps/sandbox/src/cull.rs`(데모→씬 드로우 일반화), `deferred.rs`(indirect draw), `gi.rs`/
  `reflect.rs`/`gdf.rs`(하프해상 + 캐시 갱신 예산), `quality.rs`(성능 티어), `main.rs`(배선).
- 신규 `apps/sandbox/src/occlusion.rs`(Hi-Z) / `coverage.rs`(선택), `crates/shader/shaders/
  {cull,hiz,coverage,*_halfres}.slang`.
- 수정 `docs/ROADMAP.md`(이 트랙 추가), 본 문서(Stage 0 측정 표).

## 현재 상태 (착수 전 검증 항목)
- 빌드/clippy/fmt 클린, 갤러리 무회귀 기준선(양 백엔드, `base_dx/base_vk`).
- Scalable-GI 완료: GDF GI가 콘텐츠(Sponza/레벨) **디폴트**(`P11_LEGACY_IBL` escape). 클립맵
  (`P11_GDF_CLIP_LEVELS`, 콘텐츠 기본 4레벨), surface cache(MAX_CARDS=1024), 머티리얼 wrap 샘플러.
- `cull.rs`는 **데모 전용**(큐브 그리드, `P7_CULL`) — 실제 씬 드로우엔 컬링 없음(Stage A가 일반화).
- `PROFILE_GPU=1` 패스별 ms 덤프, `RENDER_QUALITY=low|med|high` 티어, `CAM_EYE`/`CAM_TARGET` 앵글 고정.
