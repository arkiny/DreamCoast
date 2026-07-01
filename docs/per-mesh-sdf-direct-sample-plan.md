# 다음 세션 프롬프트 — per-mesh SDF **직접 샘플** (dense 48³ 합성 폐기, 정밀 히트)

> 이 문서 = 다음 세션 콜드스타트 작업 지시. 그대로 붙여 시작할 수 있게 자기완결적으로 작성.

## 빌드 디렉티브 (최우선, 메모리 `dreamcoast-build-to-quality`)
**한계효용으로 기능을 제약하지 말 것.** 특정 씬에서 가시 변화가 작아도 되돌리지 말 것. GI/거리장은
**레퍼런스 충실도 + path-tracer 패리티**로 측정하고, **최적화된 고품질 재사용 라이브러리**로 구현한다.
기존 하드 게이트(갤러리 바이트 동일·DX≡VK·결정론·heavy=opt-in)는 *품질* 게이트이므로 유지.

## 작업 (한 줄)
per-mesh 거리장을 **하나의 dense 48³ 클립맵 그리드로 재-베이크(합성)하지 말고**, per-mesh SDF 볼륨을
**그대로 인스턴스로 두고 쿼리 시 직접 샘플**한다(mesh-SDF instances). 이걸 **콘텐츠 기본값**으로 만들고,
**dense-합성 경로는 DEPRECATED** 처리(옵트아웃 + WARN)한다. 레퍼런스 상용 엔진의 SW 경로와 동일한 구조.

## 이번 세션에서 확정된 근본 원인 (반복·재진단 금지)
per-mesh DF는 유니크 메시마다 **~5cm 타깃 복셀**로 잘 굽지만(`crates/asset/src/sdf.rs`
`MESH_SDF_TARGET_VOXEL=0.05`, `mesh_sdf_dim`), 그걸 **레벨당 48³ dense 클립맵 그리드로 다시 리샘플**해서
합성한다(런타임 로그: `scene SDF 48³ (composed from N per-mesh DFs)`; `SCENE_DIM=48`
in `apps/sandbox/src/gdf.rs`). → 샘플하는 필드가 48³(~0.28–0.76m @ 37m 씬)라서 **per-mesh 해상도를
합성 단계에서 버린다.** 증상 2개(측정으로 확정):
1. **얇은 지오메트리 관통** — 사자 부조/얇은 벽/트레이서리가 복셀보다 작아 DF 등가면이 안 생겨 SW-RT
   마치(gdf_gi/ao/reflect + GI-on-DF 뷰)가 관통 → "뚫려 보임".
2. **서페이스-캐시 카드 등록 체커보드** — 고해상 2D 카드는 실제 메시 표면에 등록돼 있는데, DF 등가면이
   실제 표면과 어긋나(합성 48³ 오차) DF-march 히트에서 카드를 샘플하면 엉뚱한 텍셀 → 플라이드 노이즈.

레퍼런스는 per-mesh SDF를 **고해상 희소(sparse brick) 구조로 유지하고 직접 샘플**(또는 HW-RT)하기 때문에
정밀하다. dense 재-베이크 자체가 우리 병목. (대안 A = 하드웨어 RT 히트도 유효하나, 이번 작업은 B = 직접 샘플.)

## 프로젝트 (콜드스타트)
DreamCoast — 순수 Vulkan(ash)/D3D12(windows-rs)/Metal(objc2)를 직접 깐 Rust 엔진(wgpu/프레임워크 없음).
하나의 hand-rolled 바인드리스 RHI 뒤 3 백엔드. 딥퍼드 PBR + SW-RT GI/AO/reflect(베이크된 글로벌 거리장
GDF + 클립맵) + 메시-카드 서페이스 캐시 + 스크린-공간 라디언스 프로브 GI(`SCREEN_PROBE=1`). 먼저
`DreamCoast/CLAUDE.md`. 루트 `/Users/arkiny/GitRepos`, 엔진 `DreamCoast/`. **브랜치: `main`에서 새 피처
브랜치를 따서 작업**(현재 GI/DF 작업은 main에 머지됨). 레퍼런스 소스 `/Users/arkiny/GitRepos/UnrealEngine-1`
(대조용; 산출물엔 상표명 금지).

## 하드 룰
- **갤러리 바이트 동일 앵커 = `af70c1a5c8db49661d2c7926140c1309c28fda04c82cc1ab8aa6638d588b2b74`**
  (`--screenshot-clean`, 이번 세션에서 sky-on-miss로 `dba9ff7c…`→`af70c1a5…` 재기준선). **주의: 이전
  문서/커밋의 `dba9ff7c…`는 폐기됨.** 매 변경 1순위 무회귀 게이트(SHA-256). 의도적 라이팅 변경만
  path-tracer 잔차로 검증 후 재기준선.
- **DX≡VK ≤0.001 avg/ch** (Windows RTX 2070 SUPER, **현재 동결** → Metal 구현·검증 + 보류 명시). 푸시
  레이아웃 후행 스칼라/스페어로 안전하게(256B Vulkan 한계 유의).
- **상표명 금지**: 제3자 제품/소스 식별자(Unreal/UE/Lumen/Nanite/Epic 등)를 문서/주석/커밋에 쓰지 말 것.
- 근본원인·단일소스·heavy opt-in·**verify-then-claim**. **정확도 1순위 = path-tracer 패리티.**

## 현재 코드 (먼저 읽기)
- **합성 샘플러(교체 대상)**: `crates/shader/shaders/clipmap.slang` — `cm_geo_march` / `cm_geo_inside` /
  `cm_albedo`가 클립맵 레벨 볼륨(dense 48³)을 트라이리니어 샘플. **모든 SW-RT 소비자
  (gdf_gi/gdf_reflect/gdf_ao/gdf_trace/surface_cache/gdf_bounce/wrc/screen_probe)가 이 `cm_*`를 통해
  거리장을 읽는다** → 여기만 바꾸면 소비자 전부가 새 필드를 쓴다(단일소스 지렛대).
- **per-mesh 베이크/합성 (Rust)**: `crates/asset/src/sdf.rs`(`mesh_sdf_dim`, `MESH_SDF_TARGET_VOXEL=0.05`,
  `mesh_local_aabb_padded`, `encode_*`, 캐시), `apps/sandbox/src/compose.rs`(`ComposeObject`,
  `mesh_world_radius`, `DEFAULT_MIN_MESH_RADIUS`), `apps/sandbox/src/main.rs` ~1327(`use_permesh`,
  이제 `P11_PERMESH_GDF` 기본 ON = 콘텐츠 기본; dense 합성으로 감), `apps/sandbox/src/gdf.rs`
  (`SCENE_DIM=48`, `VOLUME_DIM=64`, 클립맵 디스크립터 빌드, `clip_descriptor`, `clip_level_volumes`).
- **클립맵 디스크립터 포맷**: `clipmap.slang` 주석 — 스토리지 버퍼, 48바이트/레벨
  `{ float4 aabb_min, float4 aabb_max, uint4 (sdf_idx, albedo_r, albedo_g, albedo_b) }`, finest→coarsest.
- 문서: `docs/per-mesh-distance-fields.md`(베이크 아키텍처 계획·현 상태), `docs/gdf-reference-alignment.md`,
  `docs/reflection-sdf-resolution.md`(해상도는 레버 아님 — 단, 그건 dense 그리드 해상도 얘기), `docs/scalable-gi.md`.
- 메모리: `dreamcoast-permesh-df-plan`(전체사 + "per-mesh 기본 승격/fused deprecated" 업데이트), `dreamcoast-build-to-quality`,
  `dreamcoast-screen-probe-gi`, `dreamcoast-no-trademark-names`, `dreamcoast-verification-split`.

## 설계 — mesh-SDF **직접 샘플** (스테이지)
목표: `cm_geo_march/inside/albedo`를, "쿼리점 `p`를 포함/근접하는 **per-mesh SDF 인스턴스**들을 찾아
각 인스턴스 로컬 공간에서 고해상 SDF를 직접 샘플하고 min"으로 바꾼다. dense 48³ 재-베이크 제거.

- **P0 — 인스턴스 스토리지.** per-mesh SDF 볼륨을 (합성하지 말고) 그대로 바인드리스에 유지.
  유니크 메시당 1개 SDF 볼륨(이미 `sdf.rs`가 굽고 캐시). 인스턴스 = `{ mesh_sdf_idx, albedo_idx[3],
  world→local 역변환(3x4 or 4x4), local AABB, mesh_sdf_dim }`. 인스턴스 배열을 스토리지 버퍼로.
  (주의: 바인드리스 볼륨 슬롯 한계 — `Bindless.volumes[…]` 개수 확인; 426 유니크 메시면 슬롯 부족 가능 →
  볼륨 아틀라스(3D 타일) 또는 슬롯 재사용/스트리밍 설계 필요. **이게 핵심 난제**. 옵션:
  (a) 모든 per-mesh SDF를 하나의 큰 3D 아틀라스에 타일로 팩 + 인스턴스가 타일 오프셋 참조,
  (b) 볼륨 배열 상한을 올림, (c) 상위 N개만 인스턴스·나머지는 coarse 폴백.)
- **P1 — 공간 가속 구조.** `p`에서 후보 인스턴스를 빠르게 찾기: 카메라-추종 **균일 그리드(클립 셀)마다
  겹치는 인스턴스 인덱스 리스트**(build 시 인스턴스 AABB를 셀에 분배) 또는 경량 BVH. `cm_*`가 이 리스트를
  순회. 레퍼런스의 mesh-SDF culling 대조(`LumenMeshSDFCulling`, 상표명 미표기).
- **P2 — 샘플러 교체.** `clipmap.slang`에 `cm_geo_march/inside`를 인스턴스-직접-샘플로 구현(또는 새
  include `mesh_sdf_sample.slang` + `cm_*`가 위임). `p` → 후보 인스턴스별 local `p'` 변환 → 고해상 SDF
  트라이리니어 → world-scale로 되돌려 min. `cm_albedo`도 히트 인스턴스의 albedo 볼륨에서. **단일소스**라
  모든 소비자 자동 승계. 기존 dense 경로는 `clip_count==0`/플래그로 폴백 유지.
- **P3 — 기본값 승격 + dense 합성 deprecated.** 직접-샘플을 콘텐츠 기본으로. dense 48³ 합성은
  옵트아웃(`P11_DENSE_GDF=1` 류) + WARN 로그(우리가 fused→per-mesh 할 때 쓴 패턴 재사용,
  `main.rs`의 `use_permesh` 승격 커밋 `c34b0e5` 참고). 갤러리는 dense 유지 여부 결정(갤러리는 단순 씬 —
  앵커 안정성 위해 dense 유지가 안전. 콘텐츠만 직접-샘플).
- **P4 (선택) — 정밀 히트 검증.** GI-on-DF 뷰(`P_SC_VIZ`)의 카드 등록 체커보드가 사라지는지 + 사자부조
  관통이 메워지는지 시각 확인(이번 세션 뷰가 회귀 테스트). 서페이스-캐시 게더 히트도 정밀해져 플라이드 제거.

## 측정 / 게이트 (스테이지마다)
`cargo fmt` → `RUSTFLAGS="-D warnings" cargo clippy -p sandbox -p dreamcoast-asset --all-targets`
→ **path-tracer 잔차 보고**(gallery `P8_PATHTRACE=1` vs raster, `tools/rt-compare.py`; 콘텐츠는 PT 미지원
→ 시각/정성) → **갤러리 바이트 동일**(SHA `af70c1a5…`) → 결정론(run-to-run 바이트 동일) → DX≡VK(동결
보류 명시) → `PROFILE_GPU`(Metal 타이머 이제 동작 — `feat(rhi-metal)` GPU timestamps).
- 씬: `gallery`(앵커), `EV100=11 LEVEL=sponza_intel`, `sponza_hero`(히어로). RELEASE, 64프레임 warmup
  (per-mesh 첫 쿡 느림, 캐시됨). GI-on-DF 뷰 회귀: `P_SC_VIZ=1 P11_CACHE_RELIGHT_PERIOD=<기본>`.

## 하지 말 것
- dense 48³ 그리드 해상도만 올려서 때우기(근본 아님). per-mesh 직접 샘플이 목표.
- 갤러리 앵커(`af70c1a5…`) 무단 변경(라이팅 개선이면 PT 잔차 검증 후 재기준선). 옛 `dba9ff7c…` 참조 금지.
- 단일소스 위반(각 소비자가 따로 거리장 샘플). `clipmap.slang cm_*` 한 곳에서 교체.
- 바인드리스 볼륨 슬롯 한계 무시(P0의 아틀라스/스트리밍 설계가 선결).
- path-tracer 패리티 없이 단정. 상표명 산출물 사용. heavy 기본 ON으로 강제(첫 쿡 비용 — 콘텐츠 기본이되
  옵트아웃 seam 유지).

## 배경 (이번 세션 산출물, main에 머지됨)
GI-on-distance-field 비주얼라이저(`P_WRC_VIZ` 월드-캐시 소스 / `P_SC_VIZ` 서페이스-캐시 고해상 소스),
occluded skylight + no-pure-black(레퍼런스 소스 검증), 서페이스-캐시 sky-on-miss 시드(캐시 전반),
뷰 검정 채움(가시성 게이팅 off)·깜빡임 픽스(느린 EMA)·성능(period=1 제거 + 반사 스킵 → 2.8→11.9fps),
per-mesh DF 콘텐츠 기본 승격 + fused deprecated, Metal GPU 타이머. 관련 커밋: `25ad5cf`,`131be3d`,
`4983bb4`,`da901e1`,`8592cdb`,`c34b0e5`.
