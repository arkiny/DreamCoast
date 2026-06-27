# Phase 14 — 스켈레탈 애니메이션 + GPU 스키닝/스킨 캐시 (세부 계획)

상위: [ROADMAP.md](ROADMAP.md) Phase 14.

## 동기 / 배경

DreamCoast는 정적 메시(glTF 첫 프리미티브, `crates/asset/src/lib.rs:64`)만 렌더한다. **범용 게임용
엔진**이라는 위상([phase-13-scene-graph.md](phase-13-scene-graph.md))에서 **스켈레탈 애니메이션**은
캐릭터/생물/기계 등 거의 모든 동적 콘텐츠의 필수 기반이다. 이 Phase는 (1) RHI 비의존 **애니메이션
데이터 모델 + CPU 포즈 샘플링/블렌딩**(스켈레톤·클립·포즈·본 팔레트), (2) **GPU 스키닝 컴퓨트 + GPU
스킨 캐시**(스킨된 정점을 프레임당 한 번 계산해 모든 소비처가 공유), (3) glTF에 더해 **업계 표준 FBX
임포터**(서드파티 백엔드 + 별도 설치 스크립트)를 추가한다.

### 왜 GPU 스킨 캐시인가 (정점 셰이더 스키닝이 아니라)

전통적 스키닝은 **정점 셰이더 안에서** 본 매트릭스를 곱한다. 단일 forward 패스에는 충분하지만,
DreamCoast는 **하나의 지오메트리를 여러 소비처가 읽는** 구조다:

- 디퍼드 **G-buffer** 채우기 패스 (`deferred.rs`)
- **섀도우 깊이** 패스 (라이트 공간 재변환)
- 하드웨어 **패스트레이서**의 BLAS/TLAS (`rt.rs`) — *정답 레퍼런스*
- 소프트웨어 RT **GDF** 보셀화/머지 (`gdf.rs`, 동적 오브젝트가 GDF에 들어갈 때)

정점 셰이더 스키닝은 (a) **패스마다 같은 스키닝을 재계산**하고(섀도우 + G-buffer = 2배), 무엇보다
(b) **레이트레이싱에 줄 정점 버퍼가 없다** — RT는 trace할 구체적인 (스킨된) 월드 정점 버퍼가 있어야
BLAS를 빌드/리핏한다. **GPU 스킨 캐시**는 스킨된 정점을 **프레임당 한 번 컴퓨트로 계산해 버퍼에 캐시**
하고, 모든 소비처가 그 **단일 버퍼**를 읽게 한다(UE의 *GPU Skin Cache*와 동일 동기). 이것이 레이트레이싱
엔진에서 스킨 캐시가 존재하는 이유이자, 본 엔진의 멀티 소비처 구조에 정확히 들어맞는 설계다.

> **엔진 위상:** 이 스키닝/캐시는 데모 한 컷이 아니라 **재사용 가능한 프로덕션 경로**로 설계한다
> (CLAUDE.md Engineering Rules). 품질 노브는 [render-quality-tiers.md](render-quality-tiers.md)의
> `RenderQuality` 티어에 접속할 seam으로 둔다.

### 사용자 결정 (이번 세션)
- **FBX 백엔드 = 둘 다 (seam).** **ufbx**(MIT, 단일 .c)를 **기본 백엔드**로, **Autodesk FBX SDK**를
  옵션 feature flag(`fbx-sdk`)로. 각각 별도의 `tools/` fetch 스크립트(서드파티 → minimal-install·
  gitignored 방침에 따라 소스/바이너리는 미커밋, 스크립트로 확보).

## 아키텍처

### 신규 크레이트 `crates/anim` (`dreamcoast-anim`)
RHI 비의존 — `glam` + `dreamcoast-asset` + `dreamcoast-core`에만 의존(`crates/scene`·`crates/render`가
파사드에만 의존하는 구조를 미러링). 애니메이션 *로직*을 소유하고 **본 팔레트(`Vec<Mat4>`)**라는 평면
산출물을 낸다. GPU 리소스는 샌드박스가 소유하고 팔레트를 스토리지 버퍼로 업로드한다. 모듈:

- `skeleton.rs` — `Skeleton { joints: Vec<Joint> }`; `Joint { parent: Option<u16>, local_bind: Transform,
  inverse_bind: Mat4, name }`. 본 계층(부모는 항상 자식보다 앞 인덱스 = topological, 단일 패스 전파).
- `clip.rs` — `AnimationClip { duration, channels: Vec<Channel> }`; `Channel { joint: u16,
  path: Translation|Rotation|Scale, sampler: Keyframes, interp: Step|Linear|CubicSpline }`. glTF/FBX
  공통 중립 표현.
- `pose.rs` — `Pose { local: Vec<Transform> }`(per-joint 로컬 TRS); `sample(clip, t) -> Pose`,
  `blend(a, b, w) -> Pose`(쿼터니언 nlerp/slerp), `compute_palette(&Skeleton, &Pose) -> Vec<Mat4>`
  = `global_joint * inverse_bind`(top-down 전파). **이 팔레트가 스키닝의 단일 소스**.
- `player.rs` — 재생 상태(`time`, `speed`, `looping`), `advance(dt)`. (Phase 13 도입 시 `AnimationPlayer`
  ECS 컴포넌트로 승격 — 아래 Phase 13 시너지 참조.)

`Transform { translation: Vec3, rotation: Quat, scale: Vec3 }`는 `crates/scene`의 `LocalTransform`과
동일 표현 — 한 곳(`dreamcoast-core` 또는 `anim`)에 정의하고 양 크레이트가 재노출(중복 정의 금지,
Engineering Rule 4).

### 정점 레이아웃 — 스킨 영향치는 별도 스트림
현재 `VertexLayout::Mesh`(32B: pos12+normal12+uv8, `rhi-types/src/lib.rs:198`)는 **렌더되는(스킨된
출력) 정점**으로 그대로 둔다 — 모든 기존 PSO(G-buffer/섀도우/캡처)가 무변경. 스키닝에 필요한 본
인덱스/가중치는 **병렬 별도 스트림**으로:

- `SkinInfluence { joints: [u16; 4], weights: [u8; 4] /* unorm */ }` = **12B**(또는 가중치 f32×4 = 24B).
  스킨 캐시 **컴퓨트만** 읽는다. → `VertexLayout` 확장 불필요(PSO 처닝 0), UE의 *separate skin weight
  stream*과 동일 관례. 4 영향치 초과는 1차 범위 외(정규화 후 상위 4개 클램프, 드롭 비율 로그).

스킨 캐시 **출력 버퍼**는 `VertexLayout::Mesh`(32B)와 **바이트 동일**한 레이아웃으로 쓴다 → 다운스트림
래스터 패스는 정적 메시와 **구별 없이** 같은 파이프라인으로 그린다(스킨 캐시는 정점 버퍼 *바인딩만* 교체).

### GPU 스키닝 컴퓨트 `skinning.slang`
바인드리스 스토리지 입력 — 정적(rest) 정점 SBB(32B), 스킨 영향치 SBB(12B), **본 팔레트 SBB**(per-instance
`float4x4` 배열). push: `vertex_count`, 입력/출력 base offset, palette base. per-vertex:

1. 4개 `(joint, weight)` 로드, `skin_mat = Σ weightᵢ · palette[jointᵢ]`.
2. `pos' = skin_mat · float4(pos, 1)`(point), `normal' = (float3x3)skin_mat · normal`(rigid+uniform
   스케일 가정; 비균등 스케일의 inverse-transpose는 리스크 항목 참조).
3. 32B `Mesh` 레이아웃 정점을 **출력 스킨 VB**에 기록. (uv는 패스스루 복사.)

per-instance 1 디스패치(차후 인스턴스 테이블로 배치 가능). **결정적**(부동소수 순서 고정) → VK ≡ DX
비트 동일(결정성 컴퓨트, 본 엔진의 결정적 래스터/GI 경로와 동일 정책).

### GPU 스킨 캐시 (프레임 그래프 패스)
- **persistent per-instance 출력 정점 버퍼**. 프레임 그래프에 **`skin_cache` 컴퓨트 패스**를 *모든*
  지오메트리 소비처 **앞**에 1회 스케줄(`crates/render`). 이후 G-buffer/섀도우/RT/GDF가 **같은 캐시 VB**를
  읽는다 → *skin once, consume many*.
- **더블 버퍼(현재/이전 프레임)** → **모션 벡터**(TAA·시공간 디노이즈·RT 재투영)용. 모션 벡터/RT가
  불필요하면 단일 버퍼(opt-in, Engineering Rule 2 비용 회피).
- **dirty 스킵**: 포즈가 바뀐 인스턴스만 재스킨; 정지 포즈 인스턴스는 디스패치 생략.
- **신규 RHI surface**: 출력 버퍼는 *컴퓨트 기록*(UAV/storage) + *정점 버퍼*(래스터) + *AS 입력*(RT BLAS)
  세 용도를 겸한다 → `BufferUsage`에 `Vertex | Storage | AccelInput` 조합 + 백엔드별 상태 전이
  (D3D12 리소스 상태 / Vulkan 배리어)를 추가(리스크 항목).

### RT 통합 (Phase 8 게이트)
- 스킨 캐시 출력 VB로 매 프레임 **스킨된 BLAS 빌드/리핏**(refit = 빠른 갱신, 토폴로지 불변 → 주기적
  full rebuild만) + **TLAS 갱신**. 패스트레이서/GDF가 애니메이션을 본다. — 스킨 캐시가 RT에 정점
  버퍼를 제공하는 바로 그 지점.
- 더블 버퍼 모션 벡터를 RT 디노이저/TAA에 공급.

## 선행조건 / 의존

- **Phase 8 (HW RT) — 완료.** Stage D(스킨된 BLAS/TLAS)만 의존. Stage A–C는 RT 없이 래스터 경로로
  독립 출하 가능.
- **Phase 13 (씬 그래프) — 시너지, 하드 의존 아님.** Stage A–C는 **단일 스킨 메시 인스턴스**를 샌드박스
  하드코딩 씬에 추가해 독립 검증 가능. 다중 인스턴스 저작·재생 상태는 Phase 13 ECS의
  `AnimationPlayer`/`SkinnedMesh` 컴포넌트로 자연 승격(둘 중 어느 Phase가 먼저든 무관, 교차 참조만 기록).
- **`crates/scene`의 `Transform`** 표현 공유(중복 정의 금지). Phase 13 미구현 시 `anim`이 `core`에 정의
  → Phase 13이 재사용.
- **외부 의존 승인(minimal-install 방침):** `cc`(ufbx C 컴파일), ufbx 소스 벤더링, (옵션) Autodesk FBX
  SDK — Stage E에서 사용자 승인 대상.

## FBX 임포터 (Stage E) — 백엔드 seam + 설치 스크립트

glTF와 **동일한 중립 산출물**(`SkinnedModel`, 아래 파일)로 매핑 → 임포터 백엔드는 *파서*만 다르고
다운스트림(스키닝/캐시/RT)은 포맷 무관.

### 기본 백엔드: ufbx (MIT, 단일 .c)
- `tools/fetch-ufbx.ps1` / `tools/fetch-ufbx.sh` — 핀된 태그/커밋의 `ufbx.c`+`ufbx.h`를 `tools/ufbx/`
  (gitignored)로 fetch + SHA256 검증. 게이팅 없는 공개 다운로드.
- `crates/asset/build.rs`(또는 신규 `crates/fbx`)가 **`cc` 크레이트**로 `ufbx.c`를 컴파일(SDK 설치 불필요,
  "설치"는 소스 fetch뿐). FFI 바인딩은 최소 수동 선언(필요 함수만; bindgen 미사용 = 의존 0 유지).
- 스키닝/애니메이션/blend shape를 모두 읽어 `SkinnedModel`로 매핑.

### 옵션 백엔드: Autodesk FBX SDK (feature `fbx-sdk`)
- `tools/fetch-fbxsdk.ps1` — 게이트된 다운로드(Autodesk 계정/EULA, 수백 MB, 플랫폼별 바이너리)라 **완전
  자동화 불가** → 스크립트는 (a) 수동 다운로드 URL/EULA 안내, (b) 받은 설치본을 `tools/fbxsdk/`
  (gitignored)에 배치, (c) `FBXSDK_ROOT` 설정 검증까지 담당. `--features fbx-sdk` 일 때만 링크.
- **CI/기본 빌드는 FBX SDK 불요**(feature off). ufbx로 전 기능 커버, SDK는 호환성 보강용 옵션.

### .gitignore / tools 컨벤션
`tools/ufbx/`, `tools/fbxsdk/`를 `.gitignore`에 추가(기존 `tools/slang/`·`tools/vulkan-layers/`와 동일
패턴). `tools/README.md`에 두 fetch 스크립트 등재.

## 단계별 (Stages)

각 Stage는 독립 커밋, 게이트: `cargo fmt --all` + `RUSTFLAGS="-D warnings" cargo clippy --workspace
--all-targets` + 양 백엔드 헤드리스 실행 + Vulkan 검증 클린 + 양쪽 스크린샷 → **VK ≡ DX**. 결정적
시점(고정 클립 `t`)으로 헤드리스 캡처해 회귀 비교.

### Stage A — 애니메이션 데이터 모델 + CPU 포즈 샘플링 (`crates/anim`)
- `crates/anim` 생성: `skeleton`/`clip`/`pose`/`player`. 클립 샘플(Step/Linear/CubicSpline 보간) →
  `Pose` → `compute_palette`. 쿼터니언 블렌딩(nlerp).
- **검증(GPU 신규 surface 0):** 본 팔레트를 **CPU에서** rest 정점에 적용해 정적 메시로 업로드 → 기존
  `Mesh` 파이프라인으로 렌더. 포즈 수학 증명(특정 `t`에서 기대 실루엣). 단위 테스트: 항등 포즈 = rest,
  단일 본 회전 = 강체 회전. VK ≡ DX(정적 업로드라 자명).

### Stage B — glTF 스킨드 임포트
- `crates/asset` 확장: `load_gltf_skinned(path) -> SkinnedModel` — 스킨(inverse bind 행렬, joint 노드),
  스킨 영향치(joints[4] u16 + weights[4]), 스켈레톤 계층, `Vec<AnimationClip>`(TRS 샘플러). 기존
  `load_gltf`는 정적 폴백으로 유지.
- 테스트 애셋은 런타임 fetch(`tools/fetch-assets`에 CC0/로열티프리 스킨 glTF 추가 — 예: glTF-Sample-Assets
  `CesiumMan` / `RiggedFigure` / `Fox`). 미커밋(기존 애셋 방침).
- **검증(애셋 구동):** rest 포즈가 정확히 로드(Stage A CPU 스킨으로 한 컷 렌더), 클립 메타(채널 수/
  duration) 로그 확인. VK ≡ DX.

### Stage C — GPU 스키닝 컴퓨트 + 스킨 캐시 패스
- `skinning.slang` + 본 팔레트 SBB + per-instance 스킨 출력 VB. 프레임 그래프 `skin_cache` 패스를
  G-buffer/섀도우 **앞**에 스케줄; 두 패스가 캐시 VB를 정점 버퍼로 바인드.
- **신규 RHI**: `BufferUsage` `Vertex | Storage`(+ Stage D에서 `AccelInput`) 조합 + 컴퓨트-기록→정점-읽기
  상태 전이(양 백엔드). 더블 버퍼 + 모션 벡터 + dirty 스킵.
- **검증:** GPU 스킨 출력이 Stage A의 **CPU 스킨과 픽셀 일치**(같은 `t`), 결정적 컴퓨트 → **VK ≡ DX
  비트 동일**, 정적 씬 무회귀. `PROFILE_GPU`로 skin_cache ms 측정.

### Stage D — RT 스킨드 BLAS/TLAS (Phase 8 게이트)
- 스킨 캐시 출력 VB로 매 프레임 BLAS 리핏(주기적 rebuild) + TLAS 갱신. 패스트레이서/GDF가 애니메이션
  반영. 더블 버퍼 모션 벡터를 RT 재투영/TAA에 연결.
- **검증:** 고정 `t` 포즈에서 **래스터 vs 패스트레이서 잔차**(`tools/rt-compare.py`)가 정적 메시 수준으로
  수렴(스킨된 지오메트리 일관), TLAS 갱신 후 검증 클린, VK ≡ DX(결정적 빌드 입력).

### Stage E — FBX 임포터 (ufbx 기본 + FBX SDK 옵션 seam)
- `tools/fetch-ufbx.{ps1,sh}` + `cc` 빌드 → `load_fbx_skinned(path) -> SkinnedModel`(glTF와 동일 중립
  타입). 스키닝/클립/blend shape 매핑.
- `--features fbx-sdk` + `tools/fetch-fbxsdk.ps1`(게이트 다운로드 안내 + 배치 + `FBXSDK_ROOT`).
  `.gitignore`/`tools/README.md` 갱신.
- **검증:** 동일 캐릭터의 glTF/FBX가 **같은 스켈레톤·클립으로 같은 렌더**(중립 타입 일관), ufbx-only
  기본 빌드 클린(SDK 불요), `fbx-sdk` feature 빌드도 클린. VK ≡ DX.

### Stage F — 확장성 / RenderQuality 통합 (선택)
- 스키닝 품질 노브(스킨 갱신 레이트, 최대 영향치 수, 본 LOD)를 `quality.rs` `RenderQuality{low,med,high}`에
  접속(`P14_*` env seam, 기존 `P11_*`/`SHADOW_*` 관례). 저티어 = 저빈도/저영향치 폴백.
- **범위 외(후속 기록):** 애니메이션 그래프/블렌드 트리/스테이트 머신, IK, blend shape의 **GPU** 적용,
  루트 모션, 본 어태치먼트.

## 파일 (생성 / 수정)
- **신규** `crates/anim/{Cargo.toml, src/lib.rs, skeleton.rs, clip.rs, pose.rs, player.rs}`; 워크스페이스
  `Cargo.toml` members + `[workspace.dependencies] dreamcoast-anim` 추가.
- **신규** `crates/shader/shaders/skinning.slang`.
- **신규** `tools/fetch-ufbx.ps1`, `tools/fetch-ufbx.sh`, `tools/fetch-fbxsdk.ps1`.
- **수정** `crates/asset/src/lib.rs` — `SkinnedModel`/`SkinInfluence`, `load_gltf_skinned`(Stage B),
  `load_fbx_skinned`(Stage E); `crates/asset/build.rs` — ufbx `cc` 컴파일(Stage E).
- **수정** `crates/rhi-types/src/lib.rs` — `BufferUsage`에 `Vertex|Storage|AccelInput` 조합(Stage C/D);
  `crates/rhi-vulkan`·`crates/rhi-d3d12` — 해당 usage/상태 전이.
- **수정** `crates/render` — `skin_cache` 컴퓨트 패스 + 다운스트림 정점 버퍼 재바인딩.
- **수정** `apps/sandbox/src/{main.rs, mesh.rs, deferred.rs, rt.rs}` — 스킨 메시 인스턴스 + 팔레트 업로드
  + skin_cache 와이어링; `rt.rs` 스킨된 BLAS 리핏; `quality.rs`(Stage F).
- **수정** `tools/fetch-assets.{ps1,sh}` — 스킨 테스트 애셋(Stage B); `tools/README.md` — fetch 스크립트.
- **수정** `.gitignore` — `tools/ufbx/`, `tools/fbxsdk/`.
- **수정** `docs/ROADMAP.md` — Phase 14 추가, 워크스페이스 구조에 `crates/anim`, 기술 스택에 FBX 임포터.

## Engineering Rules 적용 (CLAUDE.md)
1. **근본 원인:** 스키닝을 정점 셰이더가 아닌 **스킨 캐시**로 — 멀티 소비처/레이트레이싱 구조의 정답
   (패스별 재계산·RT 정점 부재라는 근본 한계 제거), 모든 메시/각도에 일반.
2. **최적화:** *skin once, consume many*; dirty 스킵; 더블 버퍼·모션 벡터는 opt-in; 12B 영향치 스트림;
   BLAS는 rebuild가 아닌 refit. `PROFILE_GPU`로 비용 측정.
3. **확장성:** 품질 노브를 `RenderQuality` 티어 seam으로(Stage F), 기능은 기본 off + env 폴백.
4. **단일 소스:** 본 팔레트는 `compute_palette` 한 곳; 스킨 출력 VB는 모든 GPU 소비처의 단일 소스;
   FBX/glTF는 같은 `SkinnedModel`; `Transform`은 `anim`/`scene` 공유 1정의.
5. **검증 후 주장:** 각 Stage VK ≡ DX + 패스트레이서 잔차 + 정적 무회귀를 숫자로 보고.

## 리스크 / 미결
- **노멀 매트릭스(비균등 스케일):** 스키닝 행렬의 3×3는 강체+균등 스케일에서만 정확. 비균등 스케일 본은
  per-bone inverse-transpose가 필요 — 1차는 강체+균등 가정(대다수 캐릭터 리그)으로 스킵, 아티팩트 관측 시
  보강. 문서화된 한계.
- **RHI 버퍼 다용도(Vertex|Storage|AccelInput):** 컴퓨트 기록 → 정점 읽기 → BLAS 입력의 상태 전이를 두
  백엔드(D3D12 리소스 상태 / Vulkan 배리어)에서 통일 — Stage C/D 세부 결정.
- **외부 의존(승인 필요):** `cc` + ufbx 벤더링(Stage E). FBX SDK는 게이트 다운로드라 자동화 불가 + 옵션
  feature로 격리(CI 불요).
- **TLAS 갱신 비용:** 애니메이션 BLAS 매 프레임 리핏 + TLAS 갱신의 cadence(refit N프레임 / rebuild) —
  Stage D에서 측정 기반 결정.
- **4 영향치 초과:** 상위 4개 클램프+정규화(드롭 비율 로그); 8 영향치는 범위 외(후속).
- **Phase 13 순서:** 재생 상태/다중 인스턴스는 ECS 컴포넌트가 자연스럽지만 하드 의존 아님 — 단일 인스턴스로
  독립 출하, ECS 승격은 교차 참조로 기록.

## 검증 (Stage별)
`cargo fmt --all` → `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets` → 각 백엔드
**고정 클립 `t`** 헤드리스 캡처 후 PNG 비교:
`VK_LOADER_LAYERS_DISABLE="~implicit~" cargo run -q -p sandbox -- --backend vulkan|d3d12
--screenshot-clean tmp/x.png`. Stage A↔C는 **CPU 스킨 vs GPU 스킨** 픽셀 일치, Stage C는 **VK ≡ DX
비트 동일**(결정적 컴퓨트), Stage D는 **래스터 vs 패스트레이서** 잔차(`tools/rt-compare.py`)를 정적
메시 수준으로 수렴. 전 Stage Vulkan 검증 / D3D12 디버그 클린.
