# 네이티브 Alembic + USD 애니메이션 임포트 (from-scratch)

상위: [ROADMAP.md](ROADMAP.md) · 선행: [phase-13-fbx-knight.md](phase-13-fbx-knight.md)(FBX 임포터).

## 동기
Intel New Sponza **knight**의 애니메이션은 FBX에 없다(스킨 웨이트 유실). Intel은 애니메이션을
**Alembic(.abc, 베이크된 vertex cache)** 와 **USD(스켈레탈)** 로 제공한다. 사용자 결정(2026-07-05):
**RHI·FBX와 같은 from-scratch 철학으로 네이티브 임포터를 직접 구현**(외부 변환 툴 미사용, Rust 크레이트
없음 → 포맷을 직접 파싱). 순서: **Alembic vertex-cache 먼저** → **USD 후속**.

## 트랙 A — Alembic vertex-cache 임포트 (먼저)

### 포맷 분석 (knight_ANIM_001.rnd.abc, 이미 리버스 완료분)
- **Ogawa 컨테이너**(magic `Ogawa`): 헤더 16B(=`Ogawa`+frozen `0xff`+ver u16 + **루트 그룹 오프셋 u64**,
  파일 끝 근처). 노드 2종:
  - **Group**@off: `u64 numChildren` + `numChildren × u64` 자식 오프셋. 자식 값의 **MSB(bit63) set = DATA,
    clear = GROUP**; 하위 63비트가 오프셋. `0`/`0xFFFF..` = null.
  - **Data**@off: `u64 size` + `size` 바이트.
- **Alembic 스키마**(Ogawa 위): 아카이브 → 오브젝트 계층(각 오브젝트=그룹) → 프로퍼티 컴파운드.
  - **PolyMesh** 오브젝트: `.geom` 컴파운드에 `P`(정점 위치, animated), `.faceIndices`·`.faceCounts`
    (토폴로지, constant), 선택 `N`/`uv`. 상위에 **Xform**(오브젝트 로컬→월드) 있음.
  - **Array 프로퍼티 샘플**: 프로퍼티 그룹의 자식이 샘플별 `[dims-data, array-data]` 쌍(=2×numSamples).
    array-data 노드 = `[16B spookyhash key][ POD 배열 바이트 ]`. **자식 순서 = 프레임 순서.**
- **이 파일 실측:** 30개 PolyMesh 파트 × **300 프레임**. 최대 파트 208350 정점(샘플 2500216B=16+208350×12).
  P는 **로컬 [-1,1] 공간**(오브젝트 Xform 별도). vert0가 프레임 간 이동 확인(애니메이션 有). FBX와 정점
  수 불일치(52639 vs 52701) → **FBX 토폴로지 재사용 불가, Alembic 토폴로지도 읽어야 함.**

### 구현 (Stages)
- **A1 — Ogawa 컨테이너 리더** (`crates/asset/src/alembic/ogawa.rs`): 그룹/데이터 트리 파서(오프셋
  기반 랜덤 액세스, 메모리맵 또는 시크). 검증: 노드 카운트가 Python 탐색과 일치.
- **A2 — Alembic 스키마 디코드**: 오브젝트 계층 + 프로퍼티 헤더 파싱(AbcCoreOgawa 포맷) → PolyMesh별
  `P`(샘플 배열) + `faceIndices`/`faceCounts` + Xform + TimeSampling(fps). 최소 스키마만(PolyMesh/Xform).
- **A3 — 중립 vertex-cache 타입 + 쿡**: `VertexCache { meshes: Vec<{ topology, frames: Vec<Vec<[f32;3]>> }>,
  fps }` → 삼각화 + 좌표 변환(Alembic Y-up RH → 엔진). `.dcasset` 청크로 쿡(CHUNK_VCACHE, [[cooked-asset-policy]]).
- **A4 — 런타임 재생**: 프레임 인덱스로 정점 버퍼 갱신(더블버퍼/모션벡터, dirty-skip). 대용량(30메시×300f
  ×~350K정점≈1.26GB)이라 **스트리밍/LOD 예산** 필요 — 서브셋 프레임/데시메이션 seam. 기존 스킨-캐시 소비처
  패턴 재사용(모든 소비처가 갱신된 VB 공유).
- **A5 — 검증**: knight abc가 Intel Sponza에서 애니메이션 재생, DX≡VK(결정적 CPU 디코드 + 정적 업로드),
  무회귀. 좌표/스케일 정합(Xform 적용 후 ~1.9m).

## 트랙 B — USD 애니메이션 임포트 ✅ DONE (2026-07-05)

> **⚠ 실측 정정 (2026-07-05):** `pkg_e_knight_anim/knight_USD_PREVIEW_SURFACE_ANIM_002_1.usd`
> (665MB, ASCII `#usda 1.0`)는 **UsdSkel(스켈레탈)이 아니라 베이크된 포인트 캐시**였다. 135개
> `def Mesh` prim 각각이 `point3f[] points.timeSamples`(프레임별 정점 위치) +
> `faceVertexIndices`/`faceVertexCounts`(토폴로지) + `extent`만 가진다. 전체 파일에
> `SkelRoot`/`Skeleton`/`primvars:skel`/`SkelAnimation` = **0건**. 즉 이 USD는 Alembic과 동일한
> 베이크 변형 캐시(리그 없음)이고, 진짜 스킨/리그는 Maya `.ma`(USD 아님)에만 있다.
> **사용자 결정(2026-07-05): 포인트 캐시로 피벗** — USD 임포터는 기존 중립 `VertexCache`(Alembic과
> 동일 타입)를 산출하고, USD/Alembic 두 소스를 **하나의 vertex-cache 쿡 경로**로 통합한다.

- **아키텍처(사용자, 2026-07-05): 애셋은 별도 애셋으로 쿡되고, 레벨은 쿡된 애셋을 로드한다.** 이
  요구는 그대로 달성됨 — `.abc`/`.usda` → `CHUNK_VCACHE` `.dcasset` → 레벨이 CacheHit 로드.
- **B1 — ASCII USD (`.usda`) 파서** ✅ (`crates/asset/src/usd.rs`, 무의존): prim 트리
  (`def`/`over`/`class` + `{ }` 바디) + typed attr + `.timeSamples` dict. **`points`/
  `faceVertexIndices`/`faceVertexCounts`만 실체화**하고 나머지(extent/primvars/xformOp/메타데이터)는
  brace-matching으로 빠르게 스킵 → point float만 파싱. 변환은 무시(이 애셋의 xformOpOrder =
  `[translate:pivot, !invert!translate:pivot]` = net identity, 루트 `!resetXformStack!` → 이미 조립된
  단일 metre 공간, Alembic 파트와 동일). **실측: 665MB → 2.8s 디코드, 135메시/300프레임@24fps/122k tri.**
- **B2 — USD Mesh 포인트캐시 → 중립 `VertexCache`** ✅: `crate::vcache::{VcMesh, VertexCache}`로
  타입을 중립화(Alembic도 재수출). UsdSkel→GltfScene 스킨 경로는 이 애셋에 스켈레톤이 없어 불가 →
  포인트 캐시로 산출, 기존 `character::overlay_vcache` + `VertexCachePlayer` 재사용.
- **B3 — 별도 애셋 쿡 (통합)** ✅: `dcasset::{write,read}_vcache`(`CHUNK_VCACHE`, VERSION 8) +
  `cook::load_or_cook_vcache(path,key,cache)`. 확장자로 디코더 디스패치(`.abc`→alembic, `.usd(a)`→usd).
  무효화 키는 **소스 파일의 len+mtime+path**(665MB/1.4GB를 매 시작 해시하지 않음 → CacheHit 빠름).
  레벨 오버레이(`KNIGHT_USD=1`/`KNIGHT_ABC=1`)가 **쿡된 애셋을 로드**(실시간 디코드 금지).
- **B4 — 검증** ✅: Crytek Sponza에서 USD knight 렌더(쿡 애셋), 두번째 로드 CacheHit, DX≡VK
  diff 0.10%>8 (베이스라인 no-knight 0.01%>8 + 씬 GI의 알려진 backend 1-LSB, 지오메트리 결정적).
  쿡 `.dcasset` = 223MB. clippy/fmt 클린, 기본 경로 무회귀.
- **companion A3** ✅: Alembic vertex cache도 동일 `load_or_cook_vcache` 경로로 쿡(더 이상 매 시작
  1.4GB 실시간 디코드 아님). 두 소스가 하나의 쿡 아키텍처 공유.

## 리스크
- **스키마 리버스 정확도**: Alembic property 헤더/타입 인코딩을 정확히 디코드해야(오독=쓰레기). Python
  탐색으로 각 단계 실측 대조(정점 수·bbox·프레임 델타).
- **런타임 메모리/대역폭**: vertex cache 1.26GB → 프레임 서브셋/스트리밍/데시메이션 예산 필수.
- **USD 규모**: `.usdc` Crate 포맷 + UsdSkel은 방대 — 최소 서브셋으로 스코프 절단 유지.
- **결정성/양 백엔드**: CPU 디코드는 결정적, 업로드는 정적 → DX≡VK 자명하게 유지.
