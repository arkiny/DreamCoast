# 반사 품질 v2 — 서피스캐시 + Global Distance Field 선명도

> 목표: SW-RT(및 HWRT 하이브리드) 반사의 **선명도/색 정확도**를 레퍼런스 엔진 수준으로 끌어올린다.
> 상위/형제: [reflection-gi-quality.md](reflection-gi-quality.md),
> [reflection-sdf-resolution.md](reflection-sdf-resolution.md)(해상도-기각 측정),
> [surface-cache-mip-handoff.md](surface-cache-mip-handoff.md)(MIP+멀티카드 cone, `17d7666`),
> [gdf-reference-alignment.md](gdf-reference-alignment.md)(표현/업데이트 격차 로드맵).
> 브랜치 기준: `feature/hwrt-hybrid-reflections`(Phase 16 A–E 완료 위에서 이어감).
> 검증 우선순위(사용자 결정 2026-07-08): **대형 콘텐츠 씬(Intel Sponza) 우선** — cross-card·커버리지
> 문제가 실제로 드러나는 씬에서 선명도 레버를 측정.

## 0. 한 줄 요약
반사 선명도의 천장은 **GDF voxel 해상도가 아니다**(측정 기각). 진짜 레버는 (1) **리졸브측 ratio
estimator**(흐리지 않고 노이즈에서 선명 복원), (2) **검증된 screen-trace-first**(온스크린=풀해상 정확색),
(3) **서피스캐시 색 캡처 + 적응 해상도**(반사되는 색·디테일), (4) **GDF expand-surface bias**(해상도가
아닌 정밀도로 실루엣/leak 정리). 전부 opt-in·갤러리 byte-identical·DX≡VK.

## 1. 진단 — 현재 선명도 천장 (코드 근거)

- **서피스캐시 해상도 한계**: 카드당 고정 32² 텍셀(`CARD_TILE=32`, `gdf.rs:186`), u/v 축이 AABB
  half-extent라 **카드 텍셀 월드 크기가 오브젝트 크기에 비례**(큰 오브젝트=거침, `fuse.rs:326-347`).
  `sample_surface_cache_cone`(`surface_cache.slang:125-209`)의 cone-MIP + 멀티카드 fit-가중 평균은
  cross-card 튐을 없애려 **의도적으로 흐린다**(설계 주석 `surface_cache.slang:117-124`). → SurfaceCache
  모드 선명도 = 캐시-해상도 한계(트레이스 한계 아님).
- **카드 커버리지 부족**: `MAX_CARDS=1024`(`fuse.rs:30`) ÷ 6 ≈ 170 drawable. 대형 씬(449 drawable)은
  초과분이 coarse dense 필드 fallback(`fuse.rs:292-301`). 데맨드 페이징 없음.
- **온스크린 정확 경로가 HWRT 전용**: 진짜 선명한 경로 = screen-color-at-hit(`gdf_reflect.slang:452-493`,
  이전 프레임 `lit_hist` 풀해상 재투영). SSR은 hit-validation이 없어 미러에서 폐기
  (`reflect_composite.slang:92-98`) → HWRT 없으면 미러가 캐시-해상도로 강등.
- **글로시 = GGX 1레이/프레임**(`gdf_reflect.slang:364-377`) + 시간적 EMA 의존; 카메라 모션/노출 변화 시
  history 거부(`reflect_temporal.slang:225-256`).
- **색 소실**: 서피스캐시 캡처가 coarse albedo 볼륨을 읽음 → 얇은 천/커튼 색 소실
  (`gdf-reference-alignment.md` 원 과제 "C").

### 기각된 경로 (재시도 금지)
- **GDF voxel 해상도 상향(선명도 목적)**: 48³→128³ 잔차 ~0.02/ch, 무의미(`reflection-sdf-resolution.md`
  Step 1). 클립맵/해상도를 "반사 선명도" 목적으로 재시도하지 말 것. (클립맵은 대형 씬 *커버리지/메모리*
  트랙 D3에서만 정당.)

## 2. 레퍼런스 엔진 메커니즘 (소스 대조, generic 명명)

우선순위순, 각 항목이 우리 어느 트랙으로 매핑되는지 표기.

1. **Spatial reconstruction ratio estimator** → Track A1. 각 픽셀이 이웃 레이를 그 hit 지점으로 재투영,
   목적 픽셀의 BRDF `weight = D_GGX(α², H)/RayPDF`로 재가중해 재사용. 이웃 hit-distance는 중심 hit로
   클램프(접촉 보존). 커널 반경은 roughness 비례, 미러 픽셀은 재구성 skip. **흐리지 않고** 적은 노이즈
   레이에서 near-mirror 선명 복원 = 캐시-해상도-한계 룩을 정면으로 공격.
2. **VNDF 중요도 샘플 + cone-angle = 1/PDF를 trace에 전달** → Track A2. mirror(roughness<ε)는
   `reflect(V,N)`, 그 외 VNDF 샘플 `reflect(V,H)`; cone-angle=1/PDF가 SDF mip 선택과 리졸브 커널
   크기를 함께 구동 → 글로시↔미러가 분기가 아닌 연속.
3. **Screen-trace-first 계층 + trace compaction** → Track B. 스크린 트레이스(HZB, 이전 프레임 씬컬러,
   ~50 iter, thickness) → 메시 SDF(근거리) → global SDF(원거리) → distant screen → radiance cache.
   단계 사이 미완료 레이만 compaction → 풀해상(downsample=1) 감당. 온스크린 hit=정확 풀해상 색.
4. **Demand-driven 적응 카드 해상도** → Track C2. ResLevel 티어(8…2048, 9단계), 기본은 거리·texel-density
   구동(`MaxProjectedSize = TexelDensityScale·Extent/ViewerDistance`, 밀도 노브 ~0.2), + **스크린 피드백**
   (샘플 시 타일당 desired ResLevel 기록 → 해시 유니크화 → MinPageHits≈16 승급)으로 실제 반사되는
   표면만 고해상 페이지로 over-allocate. 미사용 페이지는 N프레임 후 evict.
5. **Expand-surface bias** → Track D1. hit 판정 `d < ExpandAmount`, `ExpandAmount = voxelExtent · scale`,
   scale은 coverage로 0.6(미커버)~1.0(커버). 반사 레이는 **max-distance 모드**(ray-time 모드는 과occlusion
   → 디퓨즈 GI용). 얇은 벽 leak·자기교차 제거 → 반사 실루엣 정리. **해상도 아닌 정밀도.**
6. **Pre-lit Final Lighting atlas**: `(Direct+Indirect)·Albedo/π + Emissive`, hit=1 bilinear 탭.
   → 우리 relight로 **이미 보유**(`sdf_cache_light.slang`); 다수 레이를 감당케 해 1~3을 가능케 함. (유지)
7. **mesh-card 색 캡처(ortho 래스터)** → Track C1. 카드 축으로 메시를 ortho 래스터해 albedo/normal/
   emissive/depth를 스크래치에 그린 뒤 물리 아틀라스로 복사(+1텍셀 dilation, bilinear bleed 방지).
8. **BC 압축 material atlas**(BC7 albedo/BC5 normal/BC6H emissive) → Track C4(선택). 같은 메모리로 더 큰
   유효 해상도.

## 3. 계획 — 4 트랙

원칙(CLAUDE.md): 근본원인·최적화 우선·확장성(RenderQuality 노브)·단일 소스·검증. 전 단계 **opt-in seam +
기본 byte-identical(갤러리 `af70c1a5`) + DX≡VK ≤0.001**(Metal 우선, Windows 후속).

### Track A — 반사 리졸브 (레버 최고, 캐시/GDF 구조 변경 불필요)

#### A1. Spatial reconstruction ratio estimator  ← **최우선 착수**
현재 `reflect_composite.slang`의 roughness 블러(`:46-71`)를 BRDF-재가중 이웃 재사용으로 교체.
- 변경점: `reflect_composite.slang`(또는 신규 `reflect_resolve.slang`) — 이웃 N개를 그 hit 지점 방향으로
  재구성, `weight = D_GGX(α²,H)/PDF`(현 픽셀 BRDF)로 가중; 이웃 hit-dist를 중심 hit로 클램프; 미러 skip.
- 입력 요구: 트레이스가 **hit-distance**를 출력(이미 `.w`에 있음, `gdf_reflect.slang:642-655`) + PDF/
  roughness. PDF는 A2 전에는 roughness에서 유도.
- seam: `P_REFLECT_RESOLVE=1`(콘텐츠), 갤러리는 기존 블러 경로 유지 → byte-identical.
- 기대: half-res 트레이스에서도 near-mirror 선명 복원, blockiness↓ 없이 블러 제거.

#### A2. VNDF 중요도 샘플 + cone=1/PDF plumbing
`gdf_reflect.slang:364-377`의 GGX 샘플을 VNDF로, PDF를 push로 리졸브(A1)와 SDF mip 선택에 전달.
- 변경점: `gdf_reflect.slang` ray-gen; `sample_surface_cache_cone`의 `sample_radius`를 cone=1/PDF에서 유도
  (현 `0.05+t·0.08` 대체 실험 — 튜닝 필요).
- seam: `P_REFLECT_VNDF=1`. 갤러리 mirror 경로 불변.

#### A1 구현 완료 (Metal, opt-in `P_REFLECT_RESOLVE`, 기본 off)
신규 트레이스-res stateless 패스 `reflect_resolve.slang`(트레이스↔upsample 사이). 이웃의 GGX 레이를
공유 include `reflect_ggx.slang`로 **재구성**(content 결정적 jitter) → `pdf_p/pdf_q` 재가중(검증된
`ssr_resolve` 수식과 동일). near-mirror(≤0.125)는 passthrough. resolve가 돌면 `reflect_temporal`의 박스
평균을 끔(`spatial_off`, re-blur 방지). gdf_reflect **무변경**(240B push 유지) → 갤러리 자동 byte-identical.

**검증 (Metal, RELEASE):**
- **갤러리 `af70c1a5` byte-identical PASS** (opt-in seam; 갤러리는 패스 미실행).
- `sponza_intel_chromeball`: flicker 바닥 OFF-vs-OFF 0.013/255(사실상 결정적), ON 결정성 0.16/255,
  OFF-vs-ON 3.32/255(픽셀 23.9%>4). **에너지 보존**(평균 113.89 vs 113.88), **black hole/NaN 없음**.
  차이 히트맵: **크롬볼=완전 검정(passthrough 확인)**, 거친 돌바닥 글로시 성분만 확산 변경, 아티팩트 무.
- 비용: `reflect_resolve` **~1.84ms**(half-res; gdf_reflect 5.5ms·temporal 2.07ms 대비 ~30% 추가).

**결정적 검증 — mid-glossy 금속 (`sponza_intel_glossyball`, roughness 0.3):** 새 테스트 레벨(크롬볼의
글로시 형제, roughness 0.3 = 미러 임계 0.125 초과 → GGX 샘플 경로). **OFF: 글로시 볼 반사가 심하게
노이지/얼룩(half-res 단일레이 노이즈; 풀-res 박스평균이 못 줄임 — 인접 풀-res 픽셀이 같은 트레이스 샘플
공유). ON: 노이즈 소거된 깨끗한 글로시 반사, 색 피처(드레이프·아치)는 올바른 위치 보존.** HF-noise
1.96→1.75(−11%), 볼 크롭 평균 변화 5.5/255, black hole 없음. → **A1은 트레이스-res에서 서로 다른 이웃
레이를 `pdf_p/pdf_q`로 결합해 실제 분산을 줄이는 게 핵심**(박스평균은 upsample 후라 새 샘플이 없어 무력).

**결론:** mid-glossy 금속에서 **명확한 승리**(노이지 → 깨끗). 앞선 chromeball/knight가 온건했던 건
near-mirror passthrough + 거친 표면이라 글로시가 드물었기 때문(씬 한계, 기법 한계 아님). **후속: (a) 커널
비용(~1.8ms) 최적화 후 default-on 검토, (b) DX≡VK Windows 파리티, (c) Track B 위에서 재이득(더 많은
노이지 레이 → ratio estimator 값↑).** 기본 off로 무회귀 랜딩 가능.

### Track B — trace 계층 (온스크린 정확도, HWRT 없이 미러 선명)

#### B1. 검증된 screen-trace-first
GDF march 앞에 hit-validation HZB 스크린 트레이스를 1차 바운스로. 온스크린 hit=이전 프레임 `lit_hist`
풀해상 색(HWRT screen-color-at-hit의 SW 등가), 미스만 GDF march로 fallback.
- 변경점: `gdf_reflect.slang`에 스크린 마치 프리패스(depth/HZB + `prev_view_proj` 검증, HWRT 블록
  `:452-493`의 검증 로직 재사용); `reflect_composite`의 무검증 SSR 폐기 로직 제거.
- seam: `P_REFLECT_SCREEN_TRACE=1`. 갤러리(작은 씬, SSR 미스 지배)는 off → byte-identical.
- 주의: SSR은 **검증 없이 GDF와 블렌드하지 말 것**(레퍼런스는 하드 스위치) — 검증 실패 시 GDF로.

#### B2. Trace compaction → 풀해상 트레이스 감당
단계 사이 미완료 레이만 compaction 버퍼로 진행 → `P_REFLECT_RES_DIV`를 1(풀해상)로 낮춰도 비용 유지.
- 변경점: `reflect.rs` 디스패치에 compaction 패스; `gdf_reflect`를 스크린-미스 레이에만.
- seam: `P_REFLECT_COMPACT=1`. 성능 인에이블러(정확도 불변).

### Track C — 서피스캐시 품질 (반사되는 색·디테일의 천장)

#### C1. mesh-card 색 캡처 (원 과제 "C")
`sdf_cache_capture.slang`가 coarse albedo 볼륨 대신 **드로어블 메시 삼각형 albedo+opacity** 캡처
(메시당 수백 tri 최근접). 커튼/얇은 천 색 복원.
- 변경점: `sdf_cache_capture.slang`(현 GDF sphere-trace 캡처 `:93-120`) → 메시 tri 최근접 albedo;
  카드에 opacity 채널 추가; `fuse.rs` 카드 레코드/`gdf.rs` 버퍼.
- seam: `P11_CARD_MESH_CAPTURE=1`. 갤러리는 기존 stamped 색 유지.
- 검증: `bleed.py` 커튼 색 번짐 복원(빨강 옆 바닥 R−B↑), DX≡VK.

#### C2. 적응 카드 해상도 (고정 32² → ResLevel 티어 + 피드백)
2단계로 분할:
- **C2a 거리 구동 ResLevel**: 카드별 `res = clamp(RoundUpPow2(TexelDensity·Extent/dist), MIN, MAX)`;
  아틀라스를 가변-타일(서브-할당 bin, ≤코어는 공유 페이지 sub-alloc, ≥는 풀 페이지)로. `CARD_TILE` 고정
  제거. → `fuse.rs`/`gdf.rs` 아틀라스 할당, `surface_cache.slang` 인덱싱.
- **C2b 스크린 피드백 승급**: 샘플 시 타일당 desired res 기록 → 해시 유니크화 → 임계 히트 후 다음 프레임
  고해상 페이지. → 신규 피드백 버퍼/컴팩션 패스(`P11_CACHE_FEEDBACK` 이미 존재, 확장).
- seam: `P11_CACHE_ADAPTIVE_RES=1`. 갤러리는 고정 32² → byte-identical.
- 주의: 큰 구조 변경 → C2a 먼저 랜딩·검증 후 C2b.

#### C0. 반사 드리프트 (진단 완료, 2026-07-08) — relight 수렴 레이트
**증상(사용자 보고)**: content 씬(글로시 볼/바닥)에서 반사가 종료 전쯤(글로벌 씬 정착 후)까지 느리게
크리프. **A1과 무관·기존 이슈**(전부 A1 off에서 측정). **원인 격리**(sponza_intel_glossyball, 160→200
프레임 드리프트):
- **이산 버그 아님** — period-2·오토익스포저·A3 reflect-skip 재사용·수렴 freeze 래치 전부 기각(끄나 켜나
  동일), 프레임당 율 단조 감소(느린 수렴 꼬리).
- **LEGACY_IBL(정적 큐브 반사)**: 볼 드리프트 2.93→1.24(반토막) → SW-RT 반사가 지배적 원인.
- **지배 레버 = `cache_relight_period`**(content 기본 **40** — 각 카드 40프레임마다 relight): period 1로
  바닥 드리프트 0.461→0.297(−36%). **`gi_volume_period`(4→1)·`gi_spp`(4→8) 단독은 거의 무효.**
- **볼 잔여 드리프트**(relight=1로도 ~2.0 불변): 볼은 **오프스크린** 면을 반사 → `sdf_cache_light.slang`
  `HIDDEN_MULT=8`로 오프스크린 카드가 8× 드물게 relight → 오래 크리프. env 노브 없음(셰이더 상수).
**대응 방향(우선순위)**: (1) content relight period를 RenderQuality 티어 노브로 낮춤(40→8~16; 바닥 개선,
relight 비용↑). (2) **오프스크린 페널티 완화 / 반사-보이는 카드 relight 우선순위**(= C2b 피드백; 볼 잔여의
근본). (3) 완전 동적에선 카메라 이동마다 재수렴하므로 이건 amortized 동적 GI의 본질 → 티어로 노출.

#### C3. 카드 커버리지 스케일
`MAX_CARDS`를 drawable 수에 스케일(또는 LRU 데맨드 페이징). 현 `P11_REFLECT_HQ`가 전-drawable 카드를
주지만 메모리 큼 → RenderQuality 티어 노브로.
- 변경점: `fuse.rs:30` 예산 + residency; `gdf.rs` 버퍼 크기.
- seam: `P_CACHE_BUDGET=<n>` / 티어.

#### C4. (선택) BC 압축 material atlas
같은 메모리로 유효 해상도↑. RHI BC UAV 지원 필요 → 후순위.

### Track D — GDF 정밀도 (해상도 아님, 타깃 수정)

#### D1. Expand-surface bias (covered/not-covered, 반사=max-distance)
`gdf_reflect.slang` sphere-march(`:426-445`) hit 판정에 expand-amount = voxelExtent·scale(0.6~1.0,
coverage 구동), 반사 레이 max-distance 모드. 얇은 벽 leak·자기교차 제거.
- 변경점: `gdf_trace.slang`/`gdf_reflect.slang` march hit 조건; coverage는 초기엔 상수(0.8) 후 D2에서 실제.
- seam: `P_GDF_EXPAND=1`. 갤러리 march 불변(scale=1이면 근사 동일 — 검증).

#### D2. (후속) narrow-band + coverage atlas → 얇은 천 hit
±influence(≈4 voxel) narrow-band 인코딩 + 반해상 coverage atlas로 two-sided 얇은 지오를 hit 가능케.
`gdf-reference-alignment.md` C/그리고 thin-cloth 과제와 합류.

#### D3. (스케일러빌리티 후속) sparse page table 클립맵
`gdf-reference-alignment.md` G4. **선명도 레버 아님**(측정) — 대형 씬 커버리지/메모리 트랙. 여기선 추진 X,
Phase 10 Stage B에서.

## 4. 착수 순서 (권장)
```
A1  →  B1  →  A2 / B2  →  C1  →  C2a → C2b  →  C3  →  D1   (D2/D3/C4 후속)
```
근거: A1/B1이 캐시·GDF 구조를 안 건드리고 최대 선명도 이득(리졸브+온스크린). 이어 캐시 색/해상도(C),
마지막 GDF 정밀도(D1). 각 단계 독립 커밋·검증.

## 5. 검증 (대형 콘텐츠 씬 우선)
- **1차 지표(대형 씬)**: `LEVEL=sponza_intel_chromeball EV100=11 WARMUP_FRAMES=100`, RELEASE.
  볼 crop(화면 (0.38,0.30)-(0.62,0.72)) 중심 blockiness = mean |Δ3px 이웃|(현 ~3.24, `17d7666` 이후) ↓,
  cross-card 튐 해소. 커튼 색 번짐 `bleed.py`(C1).
- **게이트**: 갤러리 `af70c1a5` **byte-identical**(opt-in seam), `tools/golden-image.py --only gallery
  --backend metal`. 콘텐츠 sha 골든은 flicker로 비결정 → 관대(avg/ch) PNG 비교로.
- **패리티/비용**: `rt-compare.py` 잔차(크롬/글로시 뷰), **DX≡VK ≤0.001**(Metal 검증 후 Windows 후속),
  `PROFILE_GPU`로 각 신규 패스 비용, 무회귀.
- 정직 보고: 어느 트랙이 blockiness/잔차를 얼마 줄였는지 수치로.

## 6. 확장성 seam (RenderQuality 티어)
- 캐시: `CARD_TILE`/ResLevel MAX·`MAX_CARDS`·relight period를 티어 노브(low=32² 고정·170카드,
  high=적응 res·전-drawable·피드백).
- 반사: `RES_DIV`(low=4 → high=1+compaction), resolve 샘플 수, screen-trace on/off.
- GDF: expand on/off, D2/D3 대형 씬만. 단일 소스에서 선택.

## 7. 규칙
근본원인·opt-in seam·기본 byte-identical·3백엔드 파리티(Metal 검증 후 DX≡VK Windows)·상용 트레이드마크명
금지(문서/주석/커밋 = "reference engine"). 커밋 끝에
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
