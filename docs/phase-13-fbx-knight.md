# Phase 13 Stage E — FBX 임포터 (ufbx) + Intel Sponza knight 검증

상위: [phase-13-animation-skinning.md](phase-13-animation-skinning.md) Stage E · [ROADMAP.md](ROADMAP.md) Phase 13.

## 동기 / 목표

Phase 13 Stage A–D(glTF 애니메이션 + GPU 스킨 캐시 + RT 스킨드 BLAS)는 완료됐고, 남은 것은
**Stage E — 업계 표준 FBX 임포터**다. 이를 실제 프로덕션 에셋으로 검증한다: Intel New Sponza의
**animated knight 팩**(`D:\Assets\intelsponza\pkg_e_knight_anim.zip`)을 임포트해 **Intel Sponza
레벨 안에서 스킨 애니메이션되어 렌더**되게 하고, 기존 검증 정책(VK ≡ DX, 무회귀)을 지킨다.

knight 팩에서 우리가 소비 가능한 유일한 메시 포맷은 **FBX**다(Alembic 1.4GB·USD 665MB는 미지원):
- `Exports/FBX/Knight_USD_002.fbx` (7.6 MB) — 메시 + 스켈레톤 + 스킨(bind pose).
- `Exports/FBX/Knight_Animation_Data_Only_002.fbx` (4.6 MB) — 애니메이션 커브만(본 이름으로 매칭).
- `Textures/*.png` — brushed_metal / cloth / leather / shield / sword (FBX 머티리얼이 참조).

둘 다 **binary FBX**(Kaydara FBX Binary, Maya 2022 저작). ufbx가 binary/ascii·전 버전을 커버한다.

## 사용자 결정 (이번 세션)
- **FBX 백엔드 = ufbx 만.** Autodesk FBX SDK seam은 차후 optional feature로 연기.
- **연결 메커니즘 = 공식 ufbx 바인딩을 Cargo git 의존성(핀된 rev)으로.** 사용자 선호("git 모듈
  방식")에 맞춰 crates.io 미경유 + 소스 미커밋. 바인딩 크레이트가 `ufbx.c`를 vendor해 `cc`로 컴파일
  (bindgen/libclang 불필요), **사전 생성된 안전 바인딩**이라 수동 FFI 위험 없음.
  - 핀: `ufbx = { git = "https://github.com/ufbx/ufbx-rust", rev = "b608754e3d51...", optional = true }`
    (crate 0.11.2, **vendored ufbx C v0.23.0** = 최신). `fbx` feature로 게이트(기본 off).
  - **라이선스 = MIT OR PDDL-1.0**(퍼블릭 도메인) — 벤더/재배포 제약 없음. 미커밋은 관례상 선택이지
    강제 아님(cf. Sponza = 독점 → 강제 미커밋).
  - C 컴파일러: 로컬 VS 2022/2019 → `cc`가 cl.exe 자동 탐지(rustc host = windows-msvc). 검증 완료:
    `cargo build -p dreamcoast-asset --features fbx`가 ufbx C 컴파일·링크 성공.

## 설계 원칙 적용 (CLAUDE.md)
1. **근본 원인·일반화:** FBX를 glTF와 **동일 중립 타입**([`GltfScene`](../crates/asset/src/gltf_scene.rs))으로
   매핑 → 다운스트림(스키닝/캐시/RT/GDF)은 포맷 무관. 임포터는 *파서*만 다르다.
2. **최적화:** 임포트는 로드 1회(차후 `.dcasset` 쿡으로 승격 가능, 본 Stage 범위 밖). skin_cache는 기존
   *skin once, consume many* 경로 재사용(신규 GPU 비용 0).
3. **확장성:** 캐릭터 오버레이는 기본 off + env/CLI seam(무회귀). 스키닝 품질 노브는 기존
   `RenderQuality` seam(Stage F, 후속).
4. **단일 소스:** FBX/glTF가 같은 `GltfScene`·같은 `AnimationClip`·같은 `build_skinned_meshes`.
   좌표/단위 정합은 ufbx 로드 옵션 한 곳에서.
5. **검증 후 주장:** 각 서브스테이지 VK ≡ DX + 기존 레벨 무회귀를 숫자로 보고.

## 서브스테이지

### E1 — ufbx 바인딩 연결 ✅
- `crates/asset/Cargo.toml` — `ufbx`를 **핀된 Cargo git 의존성 + optional**으로 추가, `[features] fbx`로
  게이트(기본 off = glTF-only 빌드 무영향, C 컴파일러 불요). 바인딩 크레이트가 `cc`로 ufbx.c를
  자동 컴파일(별도 build.rs 불요 — 크레이트 자체 build.rs가 처리).
- **검증 완료:** `cargo build -p dreamcoast-asset --features fbx` = ufbx C v0.23.0 컴파일·링크 성공.
  기본 빌드(feature off)는 dep 미포함.

### E2 — `load_fbx_scene` → `GltfScene`
- `crates/asset/src/fbx.rs`(신규, `#[cfg(feature = "fbx")]` 또는 무조건) — 최소 수동 FFI(bindgen 미사용):
  `ufbx_load_file` + 필요한 struct/accessor만 선언.
- **좌표/단위 정합(최고 위험):** `ufbx_load_opts.target_unit_meters = 1.0` +
  `target_axes = ufbx_axes_right_handed_y_up`(glTF 관례) → 지오메트리가 **미터·Y-up·RH**로 나와
  엔진(1u=1m, Y-up)과 자동 정합(FBX cm/Z-up 변환을 ufbx가 처리). 회귀 시 수동 보정 fallback 기록.
- `load_fbx_scene(mesh_path, anim_path: Option<_>) -> GltfScene`:
  - 노드 계층 → `GltfNode`(TRS), 메시 프리미티브 → `GltfPrimitive`(pos/normal/uv + `MeshVertex`).
  - **스킨 클러스터** → per-vertex `joints[4]`/`weights[4]`(상위 4 클램프 + 정규화, 드롭 비율 로그) +
    `GltfSkin{joints, inverse_bind}`.
  - **머티리얼** → `GltfMaterial`, 참조 PNG 텍스처를 `TexData::Rgba8`로 로드(경로 해석은 FBX 상대 +
    팩의 `Textures/`).
  - **애니메이션**(별도 anim FBX): `ufbx_anim_stack`의 커브를 노드 TRS 채널로 샘플 → `GltfAnimation`.
    본 이름으로 mesh FBX 노드에 매칭(인덱스 아닌 이름).
- **검증:** knight 임포트가 노드/프리미티브/스킨/클립 수를 로그, rest 포즈가 정상 실루엣(단독 씬
  `--scene-fbx`로 한 컷), VK ≡ DX.

### E3 — 레벨 위 스킨드 캐릭터 오버레이
- 문제: `level::build_level`은 `instantiate_gltf`만 호출 → 스키닝/애니메이션 미와이어링.
- 해법: 레벨 로드 후 **캐릭터를 별도 스킨드 인스턴스로 오버레이**하는 seam 추가
  (`CHARACTER=knight` env 또는 `--character <key>` CLI). `instantiate_gltf_mapped` +
  `AnimationPlayer` + `skin::build_skinned_meshes`(기존 `--scene-gltf` 경로 재사용).
- 배치: Sponza 나브 중앙에 knight를 세우는 트랜스폼(레벨 카메라 프레이밍 안). 기본 off = 무회귀.
- **검증:** `--level sponza_intel CHARACTER=knight` 헤드리스 캡처에 knight가 애니메이션되어 등장,
  기존 `sponza_intel`(캐릭터 off) 바이트 무회귀.

### E4 — 검증 마무리
- 고정 클립 `t`(결정적) 헤드리스: 양 백엔드 스크린샷 → **VK ≡ DX ≤ 0.001/ch**.
- 기존 레벨(gallery/sponza_intel) 무회귀. Vulkan 검증 / D3D12 디버그 클린. `PROFILE_GPU`로
  skin_cache ms. `tools/rt-compare.py`는 GDF 갤러리 전용이라 레벨엔 비적용(육안 + DX≡VK가 게이트).

## 파일 (생성 / 수정)
- **신규** `tools/fetch-ufbx.ps1`, `tools/fetch-ufbx.sh`.
- **신규** `crates/asset/build.rs`, `crates/asset/src/fbx.rs`.
- **수정** `crates/asset/Cargo.toml`(build-dep `cc`, feature `fbx`), `crates/asset/src/lib.rs`(re-export).
- **수정** `apps/sandbox/src/main.rs`(캐릭터 오버레이 seam), `apps/sandbox/src/app.rs`(CLI, 필요 시).
- **수정** `.gitignore`(`/tools/ufbx/`), `tools/README.md`.
- **수정** `docs/ROADMAP.md`(Phase 13 Stage E ✅ 반영, 랜딩 시).

## 구현 발견 (검증 중 확정)

- **knight FBX는 스킨 웨이트가 없음.** `Knight_USD_002.fbx`·`Knight_Animation_Data_Only_002.fbx`
  둘 다 `skin_clusters=0`(USD→FBX 익스포트가 스킨 클러스터 유실). 스킨/애니 원본은 665MB USD·Maya .ma
  (스킨)·1.4GB Alembic(vertex cache)뿐 — 셋 다 미지원. **→ knight는 정적 지오메트리로만 검증**(963-노드
  스켈레톤·122k tri·8 머티리얼·정확한 1.93m 임포트 확인). 애니메이션 스키닝은 **VoxelCharacter**로 검증.
- **knight FBX는 텍스처 참조도 없음**(`textures=0`; USD→FBX가 텍스처 유실, `FbxLambert` base color 값만
  보존). → 8 머티리얼 색상으로 렌더(무텍스처). 금속 재질이 하늘을 반사해 창백하게 보이는 건 예상된 한계.
- **ufbx space_conversion을 스킨 유무로 분기(핵심).** ufbx는 지오메트리 재작성과 스키닝 보존을 동시에 못함:
  - 스킨 有 → `AdjustTransforms`(+`HelperNodes`): 스킨 클러스터 보존. 지오메트리는 소스 cm로 남고, 각
    cluster `geometry_to_bone`가 cm→m을 실어 팔레트·정점 출력이 미터 월드에 안착.
  - 스킨 無(정적) → `ModifyGeometry`: 축/단위를 지오메트리에 직접 베이크 → 정점+트랜스폼이 **같은 미터**
    (정적 드로우 자기일관). knight는 이 경로로 미터 정합(placement scale=1.0).
  - `load_fbx_scene`는 스킨드 옵션으로 1회 로드 후 `skin_clusters==0`이면 정적 옵션으로 **1회 재로드**.
- **VoxelCharacter**: FBX는 스킨 완벽(15 joints·749 weights)하나 **애니 클립은 glb에만**(Idle/Run/Die).
  → E3 애니 검증은 **glb**(기존 glTF 경로)로 오버레이. FBX 스킨 임포트는 fbx_probe로 별도 확인(≡ glb).

## E3 결과
- `SPONZA_CHARS=1 LEVEL=<level>`: 레벨 로드 후 VoxelCharacter(glb, Idle 스킨 애니) + knight(FBX 정적)
  오버레이. 배치 env `CHAR_VOXEL`/`CHAR_KNIGHT`=`"x,y,z,rotDeg,scale"`, 애니 `CHAR_ANIM`.
- 검증: Sponza 나브에 VoxelCharacter가 스키닝+Idle로, knight(방패+검)가 정적으로 렌더. **DX≡VK diff
  패널 black**(0.24/ch = 레벨 IBL 1-LSB 수준, 지각 불가). 기본(SPONZA_CHARS 미설정) 무회귀.

## 리스크 / 미결
- **좌표/단위:** ufbx `target_axes`/`target_unit_meters`로 정합. Maya cm/Y-up이 기대치. 실측 검증 필수.
- **2파일 anim 매칭:** 본 이름 불일치 시 클립 미해결 → 로그로 진단, 이름 정규화 검토.
- **스킨 클러스터 부재 가능성:** `Knight_USD_002.fbx`가 스킨 웨이트를 담는지 ufbx로 확인(없으면 rest만).
- **머티리얼/텍스처 경로:** FBX 임베디드 텍스처 아님(외부 PNG) → 팩 `Textures/` 상대 해석.
- **4 영향치 초과:** 상위 4 클램프 + 정규화(드롭 비율 로그), 8 영향치는 범위 밖(후속).
- **결정성:** 임포트는 CPU 결정적, skin_cache는 기존 결정적 컴퓨트 → VK ≡ DX 비트 동일 기대.
