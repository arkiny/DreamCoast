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

### Track A4 — 반사 디노이저 (variance-guided, 레퍼런스 대조 완료 2026-07-08)

레퍼런스 디노이저 체인 = **resolve(ratio estimator=A1) → temporal → single variance-guided bilateral**
(à-trous 아님, 1패스), 전부 풀해상 + Reinhard 톤맵 공간. 우리는 resolve(A1)+temporal(EMA·virtual-image·
clamp)은 있으나 **per-pixel 분산**과 **variance-guided 스페이셜 패스**가 없다. 소스 확정 수치:
temporal `MaxFramesAccumulated=12`(near-mirror는 `lerp(2,12,sat(R/0.05))`), clamp γ=1σ(YCoCg 5×5),
disocclusion thresh 0.03·depth; spatial KernelRadius=8·`sat(R·8)`, NumSamples 4(→8 disoccl.),
DepthWeightScale=10000, gate `StdDev/max(Luma,0.1)>0.5`.

착수 순서(각 opt-in·갤러리 byte-identical):
- **A4a — temporal per-pixel 2차 모멘트 → 분산.** `reflect_temporal`에 M2(휘도²)를 radiance와 같은 alpha로
  EMA, `Var=M2−Luma(rad)²` 출력. 버퍼: accum(rgb,len)+pos(xyz,valid) → len을 pos.w로 옮기고 accum.a=M2
  (8스칼라 유지, 3rd 버퍼 불필요). **핵심 전제**(아래 스페이셜을 적응형으로).
- **A4b — variance-guided bilateral 스페이셜 패스(신규 `reflect_spatial.slang`, temporal 뒤 풀해상).**
  Poisson ~4탭, 반경 `K·sat(rough·8)`(미러 0=passthrough), edge stop 3종(depth `exp2(−10000·(planeΔ/depth)²)`
  · normal `1−sat(angle/lobeHalfAngle)`, `lobeHalfAngle=atan(0.75·R²/0.25)` · 휘도 `exp2(−|ΔLuma|/max(StdDev,eps))`),
  `StdDev/max(Luma,0.1)>0.5`인 픽셀만. content opt-in `P_REFLECT_SPATIAL`; 갤러리 미실행.
- **A4c+ (후속 안정화)** — 적응형 프레임카운터(`frames=min(frames·conf+1,max)`, `conf=0.75·sat(1−|clamp
  Δ|/ext)+0.25`) · clamp을 YCoCg 분산클램프로 · dual-hypothesis reproject(surface 폴백+에러선택) ·
  disocclusion boost(탭2×·로브4×·휘도stop off·heavy tonemap) · grazing depth 완화(`/clamp(N·V,0.1,1)`).
  이 중 적응형 카운터·dual-hypothesis는 **반사 드리프트(C0) 안정화에도 기여**.

#### A4 구현 완료 + 검증 (opt-in `P_REFLECT_SPATIAL`, 기본 off)
A4a(temporal per-pixel 2차 모멘트→분산, moment ping-pong 버퍼, out.a=StdDev) + A4b(신규
`reflect_spatial.slang` variance-guided bilateral, 8-tap golden-disk, depth `exp2(−10000·(planeΔ/depth)²)`·
normal lobe·휘도 `exp2(−|ΔL|/StdDev)`, `StdDev/Luma>0.5` 게이트). 갤러리 `af70c1a5` byte-identical
(temporal push 224→240B, denoise off=out.a=1.0), clippy/fmt clean, ~0.35ms.

**검증에서 드러난 핵심 (디버그 확인):** 분산 계산은 정상(볼 상대분산 0.37, 28%>게이트)이나 **A4b가
현 파이프라인에서 거의 무효**. 두 원인:
1. **수렴된 스크린샷은 StdDev 낮음→passthrough**(정상: 수렴 픽셀 안 흐림).
2. **고정 per-pixel jitter → temporal 분산이 노이즈를 못 잡음**. 우리 content 노이즈는 **공간적**(이웃
   픽셀 간, A1이 처리)이지 temporal이 아님. 커브드 볼은 depth/normal edge stop이 곡률에 막혀 collapse.
**결론:** A4는 정확하나 **고정-jitter 아키텍처에선 값이 제한적**. A4의 진짜 역할 = 아래 A5(블루노이즈/
frame-varying)의 **수렴 전제** — frame-varying 스토캐스틱에서 temporal 분산이 실제 노이즈가 되어 A4b가
작동. 즉 A4는 A5를 위해 필요한 기반.

### Track A5 — 블루노이즈 스토캐스틱 반사 (frame-varying + 저불일치)

**동기(사용자 통찰):** 고정 per-pixel jitter는 **영구적 1-샘플 dead end** — 프레임을 더 줘도 개선 없음.
레퍼런스는 **per-frame 블루노이즈 jitter + temporal 누적**으로 진짜 몬테카를로 적분. **검증:** frame-varying
(백색)+A1+A4는 수렴하나 residual mottle + sparkle 0.62/255(고정 0.2 대비↑) — **백색이라 시간축 분포가
나빠서**. 블루노이즈/저불일치면 빠르고 sparkle 없이 수렴.

**설계(구현 예정):**
- **저불일치 스토캐스틱 시퀀스** — per-pixel 공간 오프셋(hash; 공간은 A1이 재가중) + **per-frame R2 additive
  (Roberts 일반 황금비 `(0.7548777, 0.5698403)`) Cranley-Patterson 회전** → 시간축 저불일치(빠른 수렴).
  (후속: spatiotemporal blue-noise 텍스처(STBN)로 공간 분포까지 blue.)
- **공유 `reflect_ggx.slang`에 `refl_ggx_dir(..., stochastic)`** — stochastic=0은 현 백색(gdf 인라인과
  bit-match), =1은 블루노이즈. gdf_reflect content가 이걸 호출(gallery는 인라인 유지=byte-identical),
  A1 resolve도 같은 flag로 재구성 → 항상 일치.
- **번들 opt-in `P_REFLECT_STOCHASTIC`** = frame-varying + 블루노이즈 + A1 + A4를 함께 켬(개별 조합의
  불일치 방지). max_steps bit30에 stochastic flag(push 무증가; bit31=content).

#### A5 구현 완료 + 검증 (opt-in `P_REFLECT_STOCHASTIC`, 기본 off)
공유 `reflect_ggx.slang` `refl_ggx_dir(...,stochastic)`(0=백색/gdf 인라인 bit-match, 1=R2 저불일치
블루노이즈); gdf_reflect content가 호출(gallery 인라인 유지), A1 resolve도 같은 flag로 재구성. 번들이
frame-varying+A1+A4를 함께 켬. 갤러리 `af70c1a5` byte-identical, clippy/fmt clean.

**검증 결과 (글로시볼, 정직):** A5 HF-noise 1.855 / sparkle 0.586 — **white frame-varying+A1+A4
(1.84/0.62)과 사실상 동일, 이득 안 보임.** 이미지도 세 경우(고정+A1 / white-fv / A5) 모두 깨끗하고 유사.
**원인 = 잔여가 jitter 노이즈가 아니라 서피스캐시/GI 수렴 mottle**(C0 드리프트와 같은 근본). 디노이저·jitter는
없는 노이즈를 못 줄임 — **병목이 상류(캐시/GI 수렴)**.

**전략적 결론 (중요):** A1/A4/A5는 정확한 레퍼런스급 디노이저 인프라이고 A5는 고정-jitter dead end를
올바르게 탈출(사용자 통찰)하지만, **현 content 파이프라인의 가시적 잔여는 서피스캐시/GI 수렴이 지배** →
디노이저 이득이 가려짐. **다음 최고 레버 = Track C(서피스캐시 색 캡처 + 반사-보이는 카드 relight 우선순위
+ GI 수렴 가속)**, 이게 드리프트(C0)도 함께 해결. A1/A4/A5는 그 위에서 값이 드러남(수렴된 GI + 러프 로브 +
카메라 이동).

### Track A6 — SSR 피드백 감쇠 (구현 완료) + 글로시 샘플 기근 진단 (2026-07-08)

**SSR 진동 (사용자 격리로 확정):** 글로시 볼의 매 프레임 움직임은 SSR — `P11_REFLECT_MAX_ROUGHNESS=0.1`
(볼에서 SSR 제외)로 정지 확인. **UE 심층 분석 결론:** UE도 **같은 피드백 루프 보유**(스크린 트레이스가
반사 포함 완전 합성 prev 씬컬러를 읽음 — `ScreenSpaceRayTracingInput` 캡처는 리플렉션 합성 후) — 안정성은
루프 제거가 아니라 **감쇠 3중주**: ① 반사에 α≈1/12 포화 running mean(**결정타** — 피드백 맵을 수축으로;
variance clamp 단독은 진폭만 제한, rate 못 제한 → limit cycle이 클램프 밴드 안에서 생존), ② per-hit
`MaxRayIntensity=40` 클램프 / Karis `rcp(1+L)`, ③ variance clamp(①과 함께). miss는 블렌드가 아닌
**하드 핸드오프**(compaction).

**구현 (converge 번들에 통합, 갤러리 byte-identical):** converge 모드가 stochastic SSR 강제(plain 경로는
EMA 없는 gain-1 루프), SSR jitter K=12 주기 순환(host가 `frame%12` 전달, 셰이더 무변경), `ssr_resolve`
running mean α=1/(1+N) N→12(N은 pos `.w` 겸용, `params.y<0` seam), variance clamp 기본 on.
+ **A1 resolve를 Reinhard 톤맵 공간 평균으로 수정**(선형 평균은 밝은 HDR 히트 하나가 지배 → 흰 사각
스파클; 레퍼런스는 resolve/temporal/spatial 전부 이 공간).

**잔여 진단 — 글로시 샘플 기근 (스크린샷 확정):** 화면 채운 rough 0.3 미러 클로즈업의 잔여 스페클 =
**~0.1 레이/디스플레이픽셀**(0.67 스케일 × div2) vs 레퍼런스 실효 ~5(풀해상+5..64샘플 재구성) — **~50×
언더샘플**. 해상도(div6→2)로도 안 사라짐(사용자 확인) = 구조 문제. apple 티어 div=6은 과함(대형 박스).

**다음 정식 트랙 (사용자 승인):**
- **B2' 글로시 샘플 밀도** — trace compaction(스크린 히트 조기 종료로 예산 확보) + 글로시 픽셀 고밀도.
- **러프-프리필터 스플릿** — roughness 상한 이상은 스토캐스틱 대신 **cone-필터 캐시/radiance-cache 경로**
  (구조적 노이즈 0; 기존 `sample_surface_cache_cone` 재사용, 배선만). 레퍼런스도 RadianceCache.MaxRoughness
  이상은 트레이스 대신 프리필터 프로브.
- 이어서 Track C(캐시 색/해상도 — 저주파 얼룩), B1(검증 스크린트레이스 하드 핸드오프 — SSR 블렌드 구조
  자체 대체).

#### 러프-프리필터 스플릿 구현 완료 (opt-in `P_REFLECT_PREFILTER=<thresh>`, `=1` → 기본 0.4)
roughness ≥ 임계 픽셀은 스토캐스틱 GGX 대신 **결정적 미러 레이 1개 + roughness 구동 콘 풋프린트**로
캐시 MIP를 샘플(`slope = tan(2θ_h) ≈ 2α`, α=r²; analytic 폴백의 `albedo_cone`도 동일 slope). 배선:
임계가 `max_steps` bits 16..23(roughness×255, 0=off; SW 스텝 캡은 bits 0..15로 마스크 축소 — 티어 최대
256이라 무손실), resolve push의 `stochastic` bits 16..23 동승 — 프리필터 픽셀은 resolve passthrough +
이웃 차용 금지(GGX draw가 아님). HWRT 경로는 비트 미설정(max_steps가 lit_hist 인덱스). push 무성장.

**검증 (glossyball, Metal RELEASE, converge 스택):**
- **갤러리 `af70c1a5` byte-identical PASS**, clippy/fmt clean.
- **프레임간 플리커(볼 크롭): 스토캐스틱 0.368 → 프리필터 0.067/255 (5.5× 안정)** — 잔여는 TAAU 디더
  + 캐시 수렴 꼬리. 구조적 노이즈 0 설계 확인.
- **임계 0.25 + `P11_REFLECT_HQ=1`(전-drawable 카드): 볼(r=0.3) 반사가 워시아웃 얼룩 → 알아볼 수 있는
  홀 반사(아치·커튼·바닥 구조)로 변모.** HQ 단독(스토캐스틱 유지)은 여전히 얼룩 — 구조 복원의 주역은
  프리필터 경로. 즉 현 샘플레이트(~0.1레이/px)에서 스토캐스틱+디노이저 체인은 수렴해도 '얼룩'으로
  수렴하고, 결정적 콘 경로가 캐시가 가진 구조를 그대로 통과시킨다.
- **잔여 한계(예상대로 Track C):** 콘 slope를 2α로 넓혀도 hf 불변 — 잔여 블록/블롭은 (a) 카드 미커버
  히트의 per-voxel albedo analytic 폴백(기본 카드예산에서 지배적, HQ로 크게 해소), (b) 32² 카드 해상도.
  둘 다 캐시 커버리지/해상도 = Track C 과제. SSR 기여는 배제 확인(`P11_REFLECT_MAX_ROUGHNESS=0.1` 무변화).
- 기본 임계 0.4(레퍼런스 등가)에선 볼(0.3)은 스토캐스틱 유지 — 그 밴드는 B2' 샘플 밀도가 담당.

#### B2' 글로시 샘플 밀도 구현 완료 (opt-in `P_REFLECT_GLOSSY_SPP=<K>`, 기본 1)
스토캐스틱 글로시 밴드(mirror < r < 프리필터) 픽셀이 프레임당 **K개의 GGX 레이**를 트레이스 —
동일 저불일치 시퀀스를 전진(`seq = frame·K + s`, R2 회전이 K프레임치를 한 프레임에 소화)하고 톤맵
공간(A1과 동일)에서 평균. A1 resolve는 이웃의 K레이 전부 재구성해 ratio 가중 합산(`Σw_s·L̄`).
배선: `max_steps` bits 24..29 = K−1(SW 콘텐츠 전용), resolve push bits 8..15. spp==1은 톤맵
왕복 없이 기존 단일샘플 경로 그대로(비트 호환). 트레이스+셰이드 블록을 per-sample 루프로 랩.

**검증 (glossyball K=8 + HQ 카드, Metal RELEASE):** 얼룩 대부분 소거(hf 7.96→7.62 최저),
프레임간 플리커 0.368→**0.118**(3× 안정), 갤러리 byte-identical PASS, clippy/fmt clean.
남은 미관: 반사가 다소 milky — 광각 로브 레이의 스카이 이스케이프(반사 오클루전 부재) 의심, 후속.

**비용 (Metal 타이머, 상대 지표):** gdf_reflect K=1 6.2ms → **K=8 45.6ms(K 선형)** — 예상대로
스크린-히트 조기 종료(B2' 후반) 없이는 과함. ⚠️**신규 발견: `P11_REFLECT_HQ`(전-drawable 449카드)가
K=1에서도 30.5ms** — `sample_surface_cache_cone`이 히트당 **전 카드 선형 루프**(O(num_cards))라서.
K=8+HQ = 245ms. 카드 룩업 가속(공간 해시/per-object 인덱스)이 Track C의 선행 과제로 승격되어야 함.

#### B2' 스크린-히트 조기 종료 구현 완료 (opt-in `P_REFLECT_SCREEN_HIT=1`)
신규 `SCREEN_HIT` permutation(`gdf_reflect_screen_cs`, globals UBO 바인드): per-ray 스크린 마치
프리패스(24스텝 지수 간격, `prev_view_proj` 투영 + 현재 depth 검증 — HWRT B.2와 동일 허용 법칙,
`vtol = 0.01·diag + t·0.02`) — 검증된 온스크린 히트는 **prev 프레임 풀해상 lit 히스토리 색**을 읽고
GDF 마치 + 카드 셰이드를 통째로 스킵. lit_hist 인덱스는 이 permutation 한정 `params.y`(콘텐츠에서
죽은 상수 albedo 폴백 슬롯) 오버로드, push 무성장. 프리필터 픽셀은 제외(콘 평균이어야 함). 카메라
이동 시 검증 실패 → GDF 폴백(스미어 없음).

**검증/측정 (glossyball, Metal RELEASE, converge 스택):**
- **품질 격변: 볼(r=0.3) 반사가 milky 워시 → 진짜 글로시 홀 반사**(커튼·아치·바닥, 올바른 색/콘트라스트)
  — HWRT 트랙의 "screen-color-at-hit이 선명함의 핵심"이 SW 경로에서 재현됨.
- **비용: 풀 스택(`SPP=8 + SCREEN_HIT + PREFILTER=1`) gdf_reflect 10.1ms, `SPP=4`면 6.4ms =
  K=1 베이스라인(6.2ms)과 동급.** 프리필터가 러프 바닥을 K루프에서 제거(45.6→10.1의 대부분),
  스크린-히트가 볼 레이 비용 지불. resolve 1.4→0.18ms(passthrough 확대).
- 프레임간 플리커 0.319(vs 스토캐스틱 베이스 0.368, K8-GDF전용 0.118) — **잔여 = lit 히스토리가
  반사를 포함한 합성 색이라는 피드백**(A6에서 규명한 것과 동일 구조; HWRT B.2도 같은 속성). 근본
  해소는 B1(검증 스크린트레이스 하드 핸드오프 + 감쇠) 몫.
- 갤러리 `af70c1a5` byte-identical PASS, clippy/fmt clean.

**권장 검증 스택(현재):** `P_CACHE_CONVERGE=32 P_GI_STABLE=1 P_REFLECT_STOCHASTIC=1
P_REFLECT_GLOSSY_SPP=4..8 P_REFLECT_SCREEN_HIT=1 P_REFLECT_PREFILTER=1 P_CACHE_GRID=1`

### Track C 선행 — 카드 룩업 그리드 가속 구현 완료 (opt-in `P_CACHE_GRID=1`)
호스트가 캐시 빌드 시 카드 위로 균일 월드 그리드(최장축 64셀)를 1회 구축: 각 카드의 영향
볼륨(카드 평면 ± u/v + trace_depth 안쪽) AABB를 **셰이더가 적용할 수 있는 최대 수용 톨러런스로
팽창**해 겹치는 모든 셀에 삽입 → 셀 조회 = 선형 스캔이 수락했을 카드의 초집합(오름차순 유지 =
FP 합 순서 동일) → **결과 동일, O(cell) 비용**. 버퍼 2개(헤더+per-cell (offset,count) / 인덱스 풀),
인덱스는 `cache.x` 스페어 비트(cards | (cells+1)<<8 | (pool+1)<<16) — push 무성장. 반사 전용
(GI/relight 컨슈머는 기존 스캔 유지).

**동반 근본 수정:** 그리드 모드에서 t-증가 수용항을 `min(t·0.03, 0.012·diag)`로 **캡** — 언바운드
수용 성장이 최악치 팽창(카드당 ±4.7%·diag)을 강제해 풀이 6.1M 엔트리(셀 평균 47카드)로 비대해지던
근본. 캡 후 팽창 2.3%·diag. 캡 너머 수락되던 원거리 카드는 score≈0(align/(1+4d))로 콘 평균만
오염하던 것들 — 그리드 ON/OFF 이미지 diff **0.03/255**(run-to-run 노이즈 0.06 이하 = 실질 동일).

**측정 (glossyball, Metal, K=1):** HQ(div3, 2694카드) 30.5→**16.5ms**(−46%), 기본(div6, 1024카드)
6.2→**4.7ms**(−24%). 풀 스택(SPP8+SCREEN_HIT+PREFILTER+GRID): 9.0ms, +HQ 21.6ms(트랙 시작점
K8+HQ 245ms 대비 **11×**). ⚠️앞선 "HQ 30ms = 카드 루프" 귀속은 절반만 정확 — **HQ는 트레이스
해상도도 div6→3(4× 픽셀)으로 올림**(캐시 OFF에도 21.9ms). 카드 루프 몫이 그리드로 제거된 것.
갤러리 byte-identical PASS, clippy/fmt/unit tests clean.

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

#### C1 구현 완료 (opt-in `P11_CARD_MESH_CAPTURE=1`, 2026-07-08)
캡처가 GDF 히트를 **드로어블의 실제 삼각형에 투영**(closest-point-on-triangle, Ericson)해 보간 UV로
base-color 텍스처를 샘플(+LOD = 카드 텍셀 풋프린트 매칭) — 카드가 드로어블당 단일 스탬프 색 대신
**per-texel 텍스처 디테일 + opacity(.w)** 를 담음. 지오는 HWRT 히트라이팅과 동일 레이아웃의 통합
버퍼(RT 능력과 무관하게 빌드) + 카드→인스턴스 맵(table row + world→object 3x4). 수락 반경 = 카드
텍셀 4배(초과 시 기존 스탬프 폴백). 캡처는 1회성이라 brute-force 최근접 스캔으로 충분.
**검증:** 아틀라스 diff 5.65/255(내용 실변화), 볼 반사 크롭 diff 0.79(구조적) — 시각 효과는 32² 카드
해상도에 제한됨 = **C2 적응 해상도가 페어**. 갤러리 byte-identical(센티널 시 .w=0 유지), clippy/fmt
clean. 레거시 `bleed.py`는 부재 — 커튼 색 번짐 정량화는 C2 후 재측정.

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

#### C0-fix 구현 완료 — CONVERGE 모드 (opt-in `P_CACHE_CONVERGE=<K>` + `P_GI_STABLE=1`)
**"지속적으로 변경됨"(수렴 불가)의 수학적 근본(레퍼런스 소스 대조로 확정):** 우리 relight/GI 볼륨은
**매 업데이트 새 백색잡음 레이 + 고정 alpha EMA** — 이는 분산 바닥 `α/(2−α)·σ²`가 영원히 남는 AR(1)
정상과정이라 **원리적으로 수렴 불가**(A2 freeze 래치가 IntelSponza에서 절대 arm 안 되던 이유).
레퍼런스의 수렴 메커니즘(소스 인용 M1~M11): **running mean α=1/(1+N)**(NumFramesAccumulated 아틀라스,
cap 4~12) + **결정적/주기적 방향 세트**(radiosity는 16-strata 완전 반구/업데이트 + `%MaxFrames` 순환
placement; radiance cache는 **jitter를 아예 끄고**("we want stable lighting") 고정 방향 + idempotent
덮어쓰기) + **수축 피드백**(albedo/π<1 — 우리도 이미 만족) + 잔여는 의도적 ±1-step 디더뿐.

**구현:** `sdf_cache_light.slang` — `params.y<0` seam(K=−params.y): **고정 per-texel 방향**(M5, frame 항
제거) + **running mean α=1/(1+N)**(M1/M2, N=radiance `.w`의 빈 채널 — 모든 컨슈머 Load3/pos.w만 읽음
확인) + host가 **spp를 K로 상향**(M4, relight당 완전 추정 = idempotent) + **period ≤2**(오프스크린
320프레임 지평선 제거 — freeze가 steady-state 비용을 0으로 만드니 amortization 불필요). `gi_volume`은
셰이더 무변경 — host가 고정 jitter 인덱스 0 전달(M5). 갤러리 byte-identical(`af70c1a5` PASS).

**측정 (glossyball, 40프레임 윈도우 볼/바닥 드리프트):**
| | 볼 | 바닥 |
|---|---|---|
| 베이스라인(영구, 감쇠 안 함) | ~2.1 | ~0.46 |
| CONVERGE 240v280 | 0.819 | 0.150 |
| CONVERGE 280v320 | **0.676 (감쇠!)** | **0.089 (감쇠!)** |

**결론: 수렴 달성** — 잔여가 윈도우마다 감쇠(지수 꼬리 = reflect_temporal 64프레임 히스토리가 빠지는 중,
유한)이고 베이스라인처럼 영구 지속하지 않음. per-frame 0.09 글로벌은 TAAU/디더 바닥(레퍼런스도 의도적
디더로 byte-identical 아님). AE 무관(격리 확인). **후속:** freeze-arm 시 reflect 히스토리 리셋(꼬리 단축),
K/period 티어 튜닝, 동적 씬 invalidation(epoch은 sun/sky만 — 오브젝트 이동 미포함), DX≡VK Windows.

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
