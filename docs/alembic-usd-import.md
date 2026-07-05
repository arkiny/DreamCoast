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

## 트랙 B — USD 애니메이션 임포트 (다음)

> **knight USD는 ASCII `.usda` (`#usda 1.0`)** — 바이너리 Crate 아님(확인됨). 텍스트 파싱이라
> 훨씬 다룰 만하다. `pkg_e_knight_anim/knight_USD_PREVIEW_SURFACE_ANIM_002_1.usd` (665MB ASCII, UsdSkel).

- **아키텍처 결정(사용자, 2026-07-05): 애셋은 레벨에 쿡되는 게 아니라 별도 애셋으로 쿡되고, 레벨은
  쿡된 애셋을 참조/로드한다.** 기존 glTF→`.dcasset` 쿡 패턴(`load_or_cook_gltf_scene`: 레벨이 소스
  경로 참조 → 첫 로드 시 `cache/dcasset/`로 쿡 → 이후 CacheHit)을 **애니메이션 모델**로 일반화.
- **B1 — ASCII USD (`.usda`) 파서** (`crates/asset/src/usd/`, from-scratch): prim/property/metadata +
  **time samples** 서브셋. USD 문법(중괄호 prim 계층, `def`/`over`, typed attrs, `.timeSamples`).
- **B2 — UsdSkel 서브셋 → `GltfScene`**: `Skeleton`(joints, bindTransforms, restTransforms) +
  `SkelAnimation`(joint translations/rotations/scales time-samples) + skinning primvars
  (`primvars:skel:jointIndices`/`jointWeights` + `geomBindTransform`) → 기존 중립 타입
  (`GltfScene` skins+animations+per-vertex joints/weights). **기존 스킨 캐시 경로 재사용**
  ([[fbx-importer-stage-e]] `skin::build_skinned_meshes` + `AnimationPlayer`).
- **B3 — 별도 애셋 쿡**: `load_or_cook_usd(path,key,cache) -> 쿡된 스킨드 모델 .dcasset`(신규 청크:
  스켈레톤/스킨/클립/메시). **레벨/오버레이는 쿡된 애셋을 로드**(USD 실시간 디코드 금지 — 맵처럼 CacheHit).
- **B4 — 검증**: knight가 쿡된 Intel Sponza에서 **스킨 애니메이션**, DX≡VK, 쿡 애셋 CacheHit 빠른 로드.
- **companion A3**: Alembic vertex cache도 같은 "별도 애셋 쿡 → 레벨 참조" 패턴으로 쿡(현재 A4는 매
  시작 1.4GB 실시간 디코드). 두 애니메이션 애셋(skinned USD/FBX, vcache abc)이 동일 쿡 아키텍처 공유.

## 리스크
- **스키마 리버스 정확도**: Alembic property 헤더/타입 인코딩을 정확히 디코드해야(오독=쓰레기). Python
  탐색으로 각 단계 실측 대조(정점 수·bbox·프레임 델타).
- **런타임 메모리/대역폭**: vertex cache 1.26GB → 프레임 서브셋/스트리밍/데시메이션 예산 필수.
- **USD 규모**: `.usdc` Crate 포맷 + UsdSkel은 방대 — 최소 서브셋으로 스코프 절단 유지.
- **결정성/양 백엔드**: CPU 디코드는 결정적, 업로드는 정적 → DX≡VK 자명하게 유지.
