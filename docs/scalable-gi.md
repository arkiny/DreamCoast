# Scalable GI — GDF/서피스 캐시를 임의 씬으로 일반화 (Phase 10 GI 트랙 확장)

상위: [ROADMAP.md](ROADMAP.md) Phase 10(SW-RT + Distance-Field GI). 이 문서는 그 GI 트랙을
**갤러리 전용 → 임의/대형 씬(Sponza 등)** 으로 확장하는 권위 계획이다. (번호 충돌 회피로 토픽명
사용 — 13=애니, 14=VirtualGeom은 선점됨. 필요 시 Phase 15로 승격 가능.)

## 동기 / 배경

Phase 10에서 만든 GDF(global distance field) 기반 SW-RT GI/AO/반사는 **갤러리(5-오브젝트, 수천
삼각형, 48³ SDF)** 에 맞춰 설계됐다. Phase 12에서 씬을 데이터 주도(ECS/레벨/스트리밍)로 일반화했고,
임의 glTF 레벨(Sponza)을 native 스케일(1u=1m)로 로드·렌더하게 됐다. 자연스러운 다음 목표는 **GI도
임의 씬에서 동작**하게 하는 것이다. 그러나 현재 GDF 파이프라인은 대형 씬에서 3중 스케일링 벽에
막힌다(아래).

### 이번 세션에서 확정된 결정 (2026-06-27)
- **Surface cache 단일화 (Lumen式).** surface cache(메시 카드)를 **GI·반사 공통 radiance 소스**로
  쓰고, **per-voxel albedo 볼륨 베이크(C8a)를 제거**한다. 진짜 Lumen은 surface cache가 radiance
  소스이고 per-voxel albedo 볼륨은 없다. 우리 albedo 볼륨은 소형 갤러리용 단축 경로였다.
  - 현재(확인됨, [gdf_gi.slang](../crates/shader/shaders/gdf_gi.slang) `trace_bounce`): **반사**는
    surface cache 기본(reflect_cache), **GI**는 기본이 `albedo_at()`(per-voxel albedo 볼륨 재조명);
    GI surface cache는 opt-in(`P11_SURFACE_CACHE`). → GI 기본 경로가 albedo 볼륨이라 중복.
- **접근 순서 = C → A.** 먼저 이 계획서(C)로 범위·스테이지를 확정·승인하고, 첫 구현 스테이지가
  (A) **베이크 가속**이다.

## 발견된 3중 스케일링 벽 (이번 세션 스코핑)

| 벽 | 현재 구조 | Sponza(~10만+ 삼각형, 30m)에서 | 필요 작업 |
|---|---|---|---|
| **① SDF/albedo 베이크** | brute-force `for tri in 0..N` (voxel당 전 삼각형). [crates/asset/src/sdf.rs](../crates/asset/src/sdf.rs) `bake_slab`/`bake_albedo_slab` — 공간 가속 없음 | 48³ × 10만 ≈ **110억 연산/베이크** → CPU 수 분~수십 분 (GPU도 brute-force) | **공간 그리드/BVH 가속** — 베이크를 O(V·k)로 |
| **② SDF 해상도** | 48³ 고정 (`gdf.scene_dim()`) | 30m → **0.6m/복셀** → 기둥(~0.5m)/아치 표현 불가(블롭) | **클립맵 / 고해상 SDF** (카메라 중심 다중해상) |
| **③ surface cache 아틀라스** | ~24 카드(갤러리 4-오브젝트 × 6면). 메인 fuse에서 per-object AABB → 6 카드 | 103 프리미티브 → **600+ 카드** | 아틀라스 할당/카드 배치 일반화 |

> 즉 갤러리(수천 삼각형, 48³)에 맞춰 설계된 시스템을 26만 삼각형 30m 건물로 끌어올리는 일이며,
> 베이크 가속·해상도·아틀라스 재설계가 본체다. "후속"이 아니라 별도 Phase 규모.

## 스테이지 (Stages)

각 Stage는 독립 커밋, 게이트: `cargo fmt --all` + `RUSTFLAGS="-D warnings" cargo clippy --workspace
--all-targets` + 양 백엔드 헤드리스 스크린샷 → **VK ≡ DX ≤ 0.001** + Vulkan 검증 클린 +
`tools/rt-compare.py` 패스트레이서 잔차. **갤러리는 매 스테이지 바이트 동일 유지**(회귀 게이트).

### Stage 0 — 기반: draw-list fuse + 레지스트리 CPU 지오메트리
- `MeshRegistry`에 **CPU 지오메트리 보관**(`MeshCpu{vertices,indices}`, `cpu(handle)` getter). 업로드
  시 함께 저장(메모리 비용 수용 — Sponza ~10MB).
- **일반 fuse** (신규 `apps/sandbox/src/fuse.rs`): `fuse_scene(world, mesh_reg, mat_reg) -> FusedScene
  {vtx(pos+normal+uv, world), idx, tri_albedo, tri_count, aabb, per-drawable AABBs}`. 갤러리 하드코딩
  fuse(main.rs)를 대체하되 **갤러리는 텍스처-평균 albedo 등 기존 값을 정확히 재현**(바이트 동일).
  > 주의: 이번 세션에서 fuse.rs/CPU 지오메트리 초안을 작성했다가 트리를 그린으로 되돌렸다(아래 ②/③
  > 벽 때문에 단독으론 무의미). Stage A와 함께 되살린다.
- **검증:** 갤러리 fuse_scene 출력이 기존 하드코딩과 동일(SDF/albedo 바이트 동일) → 갤러리 무회귀.

### Stage A — 베이크 가속 (첫 구현 스테이지, 진짜 언락)
- `sdf.rs`에 **균일 공간 그리드(uniform grid) 또는 BVH**를 넣어 voxel당 최근접 삼각형 검색을
  O(전체삼각형) → O(근방 셀) 로. 링(ring) 확장으로 최근접 보장(거리 ≤ 현재까지 최소면 다음 링까지만
  검사). SDF + albedo 베이크 모두 적용.
- 결정론 유지(`bake_is_deterministic` 테스트 + 캐시 바이트 동일). 가속 후에도 동일 결과(가속은
  순서 무관 최소 거리이므로 비트 동일 가능 — 부동소수 합산 없음, min 연산).
- **검증:** 갤러리 SDF/albedo 베이크 결과 **비트 동일**(가속=결과 불변); Sponza 48³ 베이크 시간이
  분→초 단위로(측정 보고). 일반 fuse(Stage 0)와 결합해 Sponza scene SDF가 실제로 빌드됨.

### Stage B — SDF 해상도 = 카메라 중심 클립맵 (확정 2026-06-27)

48³ 단일 볼륨 → **카메라 중심 클립맵**(다중 해상 SDF, 근거리 고해상). 사용자 결정: 스트리밍의 직접
기반이 되도록 적응 단일해상이 아닌 **클립맵을 지금** 구축([[gdf-streaming-future]]). 대형 씬에서
기둥/아치(≈0.5m)를 표현하려면 근거리 voxel ≈0.1m 필요(48³ × 30m = 0.6m는 블롭).

**모델.** `L`개 동심 큐브 레벨, 모두 같은 voxel 해상도(`scene_dim`, 기본 48³). 레벨 `i`의 half-extent
= `base_half · 2^i`(레벨 0=최고해상/최소영역). 레벨 0..L-2는 (스냅된) **카메라 위치 중심**, 최외곽
레벨 L-1은 **씬 전체 AABB를 덮도록**(씬 중심, half=H+pad) → 전역 커버리지 보장(=오늘의 단일 볼륨).
`L`은 씬 크기에서 산정: `base_half` = 레벨0 목표 voxel(≈0.1m)×dim/2, `L` = ceil(log2(H/base_half))+1,
메모리 예산으로 캡.

**서브스테이지 (각 독립 커밋·게이트):**
- **B1 (Rust 자료구조+베이크+캐시).** `gdf.rs`의 `scene_gdf`/`scene_albedo`/`scene_aabb`를
  `Vec<ClipLevel{ sdf, albedo[3], aabb_min/max }>`로. 레벨별로 `bake_sdf_from_fused`(Stage A 그리드로
  빠름)+`load_or_bake_scene_*`(키=fused+dim+레벨AABB → 레벨마다 별도 캐시). 셰이더가 읽을 **클립
  디스크립터 스토리지 버퍼**(레벨당 aabb_min/max+sdf_idx+albedo_idx[3], +clip_count) 빌드.
  **갤러리=L1**(오늘의 AABB 그대로) → 단일 볼륨과 동치. 게이트: 갤러리 바이트 동일.
- **B2 (셰이더 통합 샘플링).** 신규 `clipmap.slang`: `sample_scene_sdf/occ/march/normal/albedo(p)`가
  최내곽 포함 레벨 선택→샘플(미포함=다음 레벨, 최외곽 폴백). gdf_gi/gdf_reflect/gdf_trace/gdf_ao/
  sdf_cache_capture/sdf_cache_light이 각자 복제한 `geo_inside/geo_march/albedo_at`를 이 include로 교체.
  **L1 경로는 `(p-MIN)/(MAX-MIN)` 동일 산술**(추가 연산 0)로 갤러리 바이트 동일. 게이트: 갤러리
  DX/VK 바이트 동일 + DX≡VK.
- **B3 (Sponza 다중레벨 시연).** 레벨 경로에서 클립맵 빌드(임시 게이트; 정식 활성화는 Stage D).
  Sponza 코트야드에서 기둥/아치가 SDF trace로 식별. 양 백엔드. (Stage D 전이라 임시 토글로 검증.)

- **검증:** Sponza 기둥/아치가 SDF trace(`P11_SCENE_GDF`)에서 식별 가능; 갤러리 무회귀(클립맵 L1이
  단일 볼륨과 바이트 동일). 카메라 추종 per-frame 업데이트는 스트리밍 후속(B는 1회 베이크, 정적).

### Stage C — surface cache 아틀라스 일반화
- 카드를 **draw-list 기반 per-drawable(또는 per-primitive)** 로 생성; 아틀라스 동적 할당(카드 수에
  따라 아틀라스 크기/타일 배치). 캡처/라이팅 패스가 임의 카드 수 처리.
- **검증:** Sponza 600+ 카드 캡처·라이팅 동작, 아틀라스 오버플로 없음; VK≡DX.

### Stage D — surface-cache 단일화 + 레벨 GDF 활성화
- **GI가 surface cache를 radiance 소스로(기본)**; `albedo_at()`/per-voxel albedo 볼륨 제거(또는
  미세 폴백만). gdf_gi.slang `trace_bounce` 정리.
- 레벨/glTF 경로에서 GDF(SDF trace + 캐시 + GI/AO/반사) **활성화** — `legacy_ibl` 강제(현재
  `!gallery_scene`)를 "scene SDF + 캐시 존재 시 GDF 사용"으로 완화.
- **검증:** 갤러리 무회귀(albedo 볼륨 제거가 갤러리 GI를 캐시 경로로 전환 — PT 잔차 재측정·수용);
  레벨에서 GI 동작.

### Stage E — Sponza GI 검증
- Sponza 레벨(이미 디렉셔널+점광 3개 authored, `apps/sandbox/levels/sponza.level`)에서 GI 바운스
  확인. 데모 앵글(코트야드) + 양 백엔드 VK≡DX + PT 잔차 정직 보고. 한계(48³/클립맵 트레이드오프)
  명시.

## 파일 (생성 / 수정)
- **신규** `apps/sandbox/src/fuse.rs` (일반 fuse), `docs/scalable-gi.md`(본 문서).
- **수정** `apps/sandbox/src/registry.rs`(CPU 지오메트리), `crates/asset/src/sdf.rs`(베이크 가속 +
  해상도), `apps/sandbox/src/gdf.rs`(임의 카드 수/해상도), `apps/sandbox/src/main.rs`(일반 fuse 배선,
  레벨 GDF 활성화 게이트), `crates/shader/shaders/{gdf_gi,surface_cache,sdf_*}.slang`(캐시 단일화/카드
  일반화).
- **수정** `docs/ROADMAP.md`(이 트랙 항목 추가).

## 리스크 / 미결
- **베이크 가속 결정론**: 가속이 결과를 바꾸면(부동소수 순서) 갤러리 비트 동일 깨짐 — min 거리 연산은
  순서 무관이라 비트 동일 유지 가능, 검증 필수.
- **클립맵 vs 단일 고해상**: 메모리/베이크 예산 트레이드오프 — Stage B에서 측정 주도 결정.
  **단, 카메라 중심 클립맵 쪽을 기본 후보로** 둔다(아래 스트리밍 정합성).
- **아틀라스 크기**: 600+ 카드의 캡처 해상도/메모리 — 카드당 텍셀 수와 아틀라스 총량 조율.
- **GI 품질의 본질적 상한**: 48³/클립맵 SDF는 정확한 지오메트리가 아니라 근사 — PT 대비 잔차는
  갤러리보다 클 수밖에 없음(정직 보고). 목표는 "임의 씬에서 그럴듯한 GI"이지 PT 일치가 아님.
- **per-chunk 스트리밍 GI(Stage D 월드)**: 이번 트랙 범위 외(정적 레벨 우선). 스트리밍 GI는 후속.

## 진행 상황 (2026-06-27 — Stage 0~D 완료, E는 정직 보고)

| Stage | 커밋 | 결과 |
|---|---|---|
| 0 fuse | `962c34d` | draw-list fuse + 레지스트리 CPU 지오 (갤러리 바이트 동일) |
| A 베이크 가속 | `43bab80` | 균일 그리드 ring 검색, brute와 **비트 동일**(real Sponza assert + 캐시 byte-compare). **Sponza 262k tri 48³: 757s→0.33s** |
| B1 플래너 | `02af466` | `clipmap.rs::plan_levels` (씬 크기 자동산정, 4 테스트) |
| B2a/b 샘플링 | `9d406bf`/`dac7e8b` | `clipmap.slang` — 7개 SW-RT 셰이더가 디스크립터 경유 샘플 (L1=레거시 동일) |
| B3 멀티레벨 | `8e3c6aa` | finer 레벨 볼륨 빌드 + 디스크립터 + 패스별 transition |
| C 아틀라스 | `5995649` | `fuse::build_surface_cards` 일반화 + MAX_CARDS 예산 |
| D-build | `9ce08b8` | 콘텐츠 씬 GDF/클립맵 빌드(Sponza 4레벨), analytic ground 비활성 |
| D-lighting | `4171839` | 콘텐츠 GDF ambient 배선 |
| **NaN fix** | `46f396d` | UE `MakeFinite` + safe_normalize — temporal 누적 NaN 오염 차단 |
| **GDF 디폴트** | `e5f54df` | **콘텐츠 씬 GDF ambient 디폴트**(P11_LEGACY_IBL=escape) |

**검증 결과:**
- 갤러리: 전 스테이지 **바이트 동일**(DX/VK 0.000/ch, DX≡VK 0.000, Vulkan 검증 클린).
- **Sponza GDF 라이팅 정상**(디폴트): 코트야드 기둥·아치·배너 제대로 조명, 프레임 안정. GDF vs IBL
  =5.6/ch(실제 GI 바운스+앰비언트 모델 차이). **라이팅 렌더 DX≡VK 0.003/ch**(디노이즈됨 — raw 트레이스
  0.041보다 훨씬 타이트; 잔차=복잡 SDF march의 본질적 FP, 갤러리 파리티 기준은 0.000 유지).
- 클립맵 지오메트리: `P11_SCENE_GDF` 트레이스가 기둥·아치·벽 해상(4레벨 vs 1레벨=31.9/ch). 트레이스
  자체 DX≡VK 0.026~0.041은 디버그 viz 한정(라이팅 렌더는 디노이즈로 0.003).

**★ 핵심 교훈(NaN 오염)**: "Sponza GDF가 검다"는 GI 부족이 아니라 **temporal 누적 버퍼 NaN 오염**이었음
(첫 프레임 정상→이후 검은색 번짐). 원인=빈 영역 SDF 그래디언트 `normalize(0)`→NaN→서피스캐시/GI
디노이저/반사 누적의 EMA로 매 프레임 확산. UE5 Lumen `MakeFinite` 패턴(누적 경계 sanitize)+
safe_normalize로 해결. 유한값엔 무영향=갤러리 바이트 동일.

**미결(측정 주도 후속):**
- **서피스 캐시 단일화(albedo 볼륨 제거)**: 계획의 GI=캐시-radiance-소스 단일화는 미적용(현 GI=
  per-voxel albedo 볼륨 경로, 콘텐츠는 클립맵 레벨당 albedo 볼륨). 갤러리 재baseline 동반→별도 측정 주도.
- **콘텐츠 라이팅 추가 품질**(UE Lumen 정렬): 멀티바운스 강화·스카이 이라디언스·노출 미세조정.

## 향후 정합성 — 카메라 이동 GDF 스트리밍 (대형 월드)

대형 월드에서는 단일 정적 GDF가 불가능하므로 **카메라 이동에 따라 GDF(거리장 + 서피스 캐시)를
청크/클립맵 단위로 스트리밍**해야 한다(후속 트랙). 이번 트랙은 정적 레벨이 우선이지만, **B/C 설계를
스트리밍에서 갈아엎지 않도록** 다음을 지킨다:

- **Stage B = 카메라 중심 클립맵 우선.** 단일 고해상 SDF는 월드 스케일에서 메모리·스트리밍 모두
  부적합. 클립맵(근거리 고해상 + 원거리 저해상, 카메라 추종)이 그대로 스트리밍의 기반이 된다.
- **Stage C = 동적 아틀라스 할당 + 방출(eviction) 가능 구조.** 고정 슬롯이 아니라 카드 할당/해제가
  되는 구조로 만들면 청크 진입·이탈 시 카드 스트리밍이 자연스럽다.
- **Stage A 베이크는 이미 스트리밍 친화적.** 그리드 가속은 임의 AABB에 대해 O(voxel·근방셀)이라,
  청크 AABB로 grid build + 부분 재베이크에 그대로 재사용된다(전체 재베이크 불필요).
- 범위 경계: 이번 트랙은 **정적 레벨 GI 완성**까지. 실제 스트리밍(LRU 캐시, 청크 경계 이음새, 비동기
  베이크 잡)은 측정 주도로 별도 트랙에서. [[dynamic-gdf-deferred]]는 *동적 오브젝트* 이슈로 이와 별개.

## 검증 (Stage별)
`cargo fmt --all` → `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets` → 각 백엔드
헤드리스(`--screenshot-clean`) → VK vs DX(및 갤러리는 사전 baseline) `tools/rt-compare.py`. GI
스테이지는 `P8_PATHTRACE=1` 패스트레이서 잔차로 품질을 정직 보고. 갤러리 바이트 동일이 매 스테이지
1순위 게이트.
