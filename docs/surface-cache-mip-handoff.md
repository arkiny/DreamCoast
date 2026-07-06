# 핸드오프: 서피스캐시 MIP 계층 + cone-LOD (대형 씬 clean chrome)

DreamCoast 엔진(Rust/raw-RHI, macOS=Metal, repo: /Users/arkiny/GitRepos/DreamCoast)에서
SW-RT 반사 품질 개선을 이어간다. 참고용 언리얼(Lumen) 소스가 /Users/arkiny/GitRepos/UnrealEngine-1 에 있다.

## 목표 (한 줄)
스폰자 크롬구가 blocky/color-bleed로 나오는 마지막 원인 = **서피스캐시에 MIP 계층이 없고 반사가 cone-LOD로
샘플하지 않아서**다. Lumen처럼 **서피스캐시 atlas의 MIP 피라미드를 만들고, 반사 히트가 ray-cone footprint로
MIP 레벨을 선택**하게 하여 극도로 축소된(demagnified) 먼 반사가 coarse-MIP 평균으로 매끄럽게 나오게 한다.

## 착수 전 필수
- git: 브랜치 `feature/lumen-style-reflection-gi` (origin/main `1efba75`에서 분기, 푸시됨, 현재 HEAD `f1fa851`).
  origin/main은 로컬 옛 main과 disjoint 히스토리 → `git pull` 금지. 이 브랜치에서 계속 작업.
- 메모리 읽기: `dreamcoast-reflection-gi-fix.md`(전체 조사 요약 — 반드시 읽을 것),
  `dreamcoast-permesh-sdf-direct-sample.md`, `dreamcoast-vgeo-metal-atomic64.md`.
- 게이트: gallery 골든 `af70c1a5`가 항상 byte-identical
  (`python3 tools/golden-image.py --only gallery --backend metal`).

## 지금까지 확정된 진단 (이번 트랙에서)
- **왜 갤러리 크롬은 깨끗한데 스폰자는 blocky인가**: 갤러리는 작은 씬(소수 drawable, 전부 서피스캐시 카드
  보유, 반사 대상이 가깝고 크게 비침). 스폰자는 대형 나브(449 drawable, `MAX_CARDS=1024`÷6=170개만 카드).
  크롬구는 **먼 나브를 극도로 축소해서** 반사 → 고주파. coarse 48³ GDF march의 far 히트가 부정확 →
  서피스캐시 lookup 실패 → **coarse per-voxel `albedo_at`(blocky red)로 fallback**.
  볼 중심 분류: cached 56% / **analytic 43%** / miss 0% — 그 43%가 blocky의 정체.
- **Lumen은 반사에서 per-voxel albedo를 절대 안 읽는다** (에이전트가 UE 소스로 확인):
  coarse global-SDF 히트에서도 global-SDF 페이지의 object-grid로 메시 카드를 찾아 서피스캐시의
  **FINAL LIT radiance**를 읽고, cone SampleRadius + **캐시 MIP/page LOD** + bilinear로 매끄럽게 하며,
  진짜 miss면 radiance cache로 (albedo로 안 떨어짐).
- **이미 커밋한 것 (`f1fa851`, 전부 content-gated `extra_tol>0`, gallery byte-identical)**:
  (1) `gdf_reflect.slang`의 캐시 tolerance를 ray 거리 비례(`+ t*0.03`)로 → far 히트가 cached lit
  radiance를 읽음. (2) `sample_surface_cache`에 **bilinear** 2×2 카드-텍셀 보간(uncaptured skip+renorm).
  **BUT** 극도 demagnification에선 marginal(중심 blockiness 5.1→5.0). blockiness가 **카드 간 불연속**
  (인접 반사 픽셀이 서로 다른 카드 히트)이라 카드 내부 bilinear로는 부족 → **MIP가 결정적**.

## 이번 작업 = 서피스캐시 MIP 계층 + cone-LOD

### 참조 (Lumen 소스, /Users/arkiny/GitRepos/UnrealEngine-1)
- `Engine/Shaders/Private/Lumen/SurfaceCache/LumenSurfaceCacheSampling.ush`:
  `ComputeSurfaceCacheSample` (라인 ~99-190) = page 해상도 레벨 추출 + **cone 기반 DesiredResLevel**:
  `SampleResolution = max(Card.LocalExtent) / max(SampleRadius,1); DesiredResLevel =
  clamp(log2(SampleResolution)+bias, MIN, MAX)`. `SampleSurfaceCacheAtlas`(~222-233) = GatherRed/Green/Blue
  + bilinear TexelWeights. 즉 **cone SampleRadius로 MIP 레벨을 고르고 bilinear**.
- `Engine/Shaders/Private/Lumen/LumenReflectionTracing.usf` (~591-596): ray-cone
  `SpreadAngle=View.EyeToPixelSpreadAngle; PropagateRayCone(...); SampleRadius = ConeStartRadius +
  TanConeAngle * HitDistance`.
- `Engine/Shaders/Private/Lumen/LumenSoftwareRayTracing.ush`: `ConeTraceMeshSDFsAndInterpolateFromCards`
  (~573-632), `EvaluateGlobalDistanceFieldHit`(~637-763)에서 SampleRadius/SurfaceCacheBias를 카드 샘플에 전달.

### DreamCoast 관련 파일
- `crates/shader/shaders/surface_cache.slang` — `sample_surface_cache(p,n,cards_index,cache_pos_index,
  cache_rad_index,num_cards,tile,extra_tol,found)`. 지금은 flat atlas(카드당 `tile²` 텍셀, tile=32)에서
  best 카드 찾아 bilinear 텍셀 읽음. **여기에 MIP 레벨 인자 + MIP 샘플링을 추가**.
- `crates/shader/shaders/sdf_cache_light.slang` — 매 프레임 캐시 radiance atlas(mip0)를 relight로 채움
  (`cache_rad_write`). MIP 생성 패스는 별도.
- `apps/sandbox/src/gdf.rs` — `build_surface_cache`(atlas 할당, `card_tile`=32, `cache_radiance[3]` ping-pong
  storage buffers `num_cards*tile²*16B`), `record_cache_light`/`record_cache_async`(relight). `CARD_TILE=32`.
- `apps/sandbox/src/fuse.rs` — `MAX_CARDS=1024`, `CARDS_PER_DRAWABLE=6`, 카드 capture.
- `apps/sandbox/src/reflect.rs` — `record_gdf_reflect` (반사 패스; 여기서 cone SampleRadius를 push로 전달).
- `apps/sandbox/src/gi.rs` — GI gather도 `sample_surface_cache`를 쓴다(primary consumer, extra_tol=0).

### 구현 스케치 (권장 순서, 각 단계 gallery byte-identical + 커밋)
1. **Atlas에 MIP 저장 공간 추가**: 카드당 `tile²` → `tile² * 4/3`(mip0=tile², mip1=(tile/2)², ...). 
   `cache_radiance` 버퍼 크기를 늘리고, 카드 c의 mip L 텍셀 base 오프셋 계산 헬퍼(`mip_base(c, L, tile)`).
   (또는 mip별 별도 버퍼. flat + 오프셋이 index 계산은 단순.)
2. **MIP 생성 compute 패스** (`sdf_cache_mipgen.slang` 신규): relight가 mip0을 채운 뒤, mip1..N을
   2×2 평균 다운샘플로 생성(uncaptured 텍셀은 가중 제외 + renorm — bilinear과 동일 규칙). radiance +
   pos(capt.w) 둘 다 mip 필요(유효성). `gdf.rs`에서 relight 뒤 record.
3. **cone SampleRadius plumbing**: `gdf_reflect.slang`에서 히트 시 `sample_radius = base + t * cone_slope`
   (base = 픽셀 footprint, cone_slope ~ `EyeToPixelSpreadAngle` 유사; 지금 tolerance의 `t*0.03`와 별개로
   LOD용). `sample_surface_cache`에 `sample_radius` 인자로 넘김.
4. **`sample_surface_cache`에서 MIP 선택**: best 카드 확정 후
   `res = max(len(ua),len(va)) / max(sample_radius, eps); mip = clamp(log2(tile/res_in_texels)+bias, 0, maxmip)`
   (Lumen `log2(CardExtent/SampleRadius)` 형태). 그 mip의 `tile>>mip` 그리드에서 bilinear(2단계면 trilinear
   optional). content(extra_tol>0)만; gallery(extra_tol==0)는 mip0 nearest 그대로 → **byte-identical**.
5. **feedback(선택)**: Lumen은 desired res를 feedback 버퍼에 써서 다음 프레임 그 페이지를 고해상도로.
   1차 구현은 생략 가능(전 카드 full mip 생성).

### 게이트 / 검증
- **gallery `af70c1a5` byte-identical 필수** — 모든 신규 경로는 content-only(extra_tol>0 또는 별도 seam).
  primary GI gather(extra_tol=0)와 gallery는 mip0-nearest 레거시 경로 유지.
- 검증 씬: `LEVEL=sponza_intel_chromeball EV100=11 WARMUP_FRAMES=100`, RELEASE 빌드, gitignored
  Intel Sponza 에셋 필요. 볼 crop(화면 (0.38,0.30)-(0.62,0.72))으로 중심 blockiness 비교.
  blockiness 지표: 볼 중심(r<0.09H) 픽셀의 mean |Δ 3px 이웃| (현재 ~5.0 → MIP로 낮아져야 함).
  target 참고: `RENDER_SCALE=1 P_TAAU_JITTER=0`은 별개(TAAU) — 반사 blockiness는 이 지표로.
- cone LOD가 과하면 near/face-on 반사가 흐려짐(over-blur) — bias 튜닝. near 반사(볼 상부 아치)는 선명 유지,
  far/center만 coarse.
- 성능: MIP 생성은 mip0 텍셀의 ~1/3 추가. `PROFILE_GPU=1`로 relight+mipgen 비용 측정.

## 규칙 (CLAUDE.md)
근본원인 수정·opt-in seam·기본 byte-identical·3백엔드 파리티(Metal 검증 후 DX≡VK Windows 후속)·
상용 트레이드마크명 금지(문서/주석/커밋엔 "reference engine"). 커밋 끝에
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## 구현 완료 (commit `17d7666`) — 결과 + 진단 정정

**결과 (Metal, RELEASE):** gallery `af70c1a5` byte-identical; `sponza_intel_chromeball` 볼 중심
blockiness(mean |Δ3px|) **5.70 → 3.24 (−43%)**, 블록 스페클이 매끄러운 축소 반사로 해소, near
반사(상단 아치·바닥)는 유지. 비용: mipgen ~0.5ms, multi-card는 gdf_reflect에 사실상 0 추가.

**진단 정정 (핸드오프 전제였던 "MIP가 결정적"은 부분적):** CLASSIFY diag로 볼 중심이 이미 **100%
cached**임을 확인(f1fa851 `+t*0.03` tolerance가 far 히트를 analytic→cached로 전환 완료). 따라서 잔여
블록은 카드 내부 해상도가 아니라 **카드 간(cross-card) 불연속** — 부정확한 far 히트가 인접 픽셀마다 다른
카드를 스쳐 색이 튐. 카드 내부 MIP만으로는 5.70→5.59(2%)에 그치고, 전역 MIP_BIAS는 near까지 흐리며,
max-mip 강제(카드당 1색)는 5.36으로 오히려 악화(= cross-card가 벽임을 증명).

**결정적 레버 = MULTI-CARD CONE ACCUMULATION** (Lumen cone 샘플의 나머지 절반): `sample_surface_cache_cone`이
히트를 덮는 **모든 카드**를 fit 가중(`score=align/(1+dist·4)`)으로 각자의 cone-MIP에서 누적 → cross-card
점프를 평균 → 5.70→**3.24**. MIP 레벨은 tolerance와 **분리된** ray-cone `sample_radius=0.05+t·0.08`로
선택(tolerance를 넓히면 오히려 scatter 증가); near/face-on은 카드 1개 mip0로 수렴(선명 유지).

구현: `sdf_cache_mipgen.slang`(2×2 평균, uncaptured 가중제외+renorm, coverage `.w` 상향 전파) → `gdf.rs
cache_rad_mips`(레벨 1..=`mip_levels`, `float4(rgb,coverage)`) relight mip0에서 매프레임 생성
(`record_cache_mipgen`, 레벨별 barrier 체인); `card_mip_sample` 헬퍼가 per-card MIP bilinear; gdf_reflect가
mip_index+max_mip를 cache-push tile 슬롯 여유비트에 팩. content 전용 seam(`extra_tol>0 && mip_index!=sentinel`);
primary 컨슈머는 미변경 `sample_surface_cache` wrapper(sentinel→legacy) → gallery byte-identical. 옵트아웃
`P11_REFLECT_MIP=0`.

## 이 트랙의 남은 후속 (MIP 이후)
- grazing-ring reddening: 볼 바깥 링이 프레임 진행하며 warm 밴드 누적 — 반사 analytic relight가 primary보다
  sun-dominated warm. 반사 relight의 sun/sky 밸런스를 primary와 일치시키는 별건.
- 잔여 저주파 색 mottle = 실제 배너/커튼 콘텐츠(자연스러움).
- DX≡VK Windows 파리티(이번 트랙 12커밋 전부 Metal만 검증).
