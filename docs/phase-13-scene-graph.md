# Phase 13 — 씬 그래프 + 레벨 스트리밍 (세부 계획)

상위: [ROADMAP.md](ROADMAP.md) Phase 13.

## 동기 / 배경

DreamCoast에는 성숙한 **렌더 그래프**(`crates/render`) — GPU 패스 DAG("어떻게 그리는가") — 가 있지만,
**씬 그래프**("무엇이 어디에 존재하는가"의 공간·논리 표현)는 없다. 현재 샌드박스는
`apps/sandbox/src/main.rs`에 평면 `Vec<SceneObject>`(101–115행, 576–633행)로 씬을 하드코딩한다.
각 오브젝트는 메시 + 월드 공간 `Mat4` + 머티리얼이며, **계층도, 로컬 트랜스폼도, 레벨 개념도 없다.**
게다가 `dreamcoast_asset::load_gltf`(`crates/asset/src/lib.rs:64`)는 **첫 메시의 첫 프리미티브만**
읽고 glTF 노드 계층을 통째로 버린다 — 보유 애셋으로 확인: `Lantern.glb`는 4노드 / 3메시(부모
"Lantern" → 자식 Body·Chain·Lantern)인데 지금은 Body 하나만 렌더된다.

이 Phase는 (1) 래스터라이저·HW 레이트레이싱·GPU 컬링에 **per-instance 월드 트랜스폼을 공급하는 단일
소스**가 되는 변환 계층 **씬 그래프**와, (2) 선언적 **레벨** 포맷 + **스트리밍 레벨 그래프**(실제 게임에서
쓰일 법한 콘텐츠 호스팅)를 추가한다. 테스트는 보유 애셋(Avocado / BoomBox / Lantern)으로 구동한다.

### 사용자 결정 (이번 세션)
- **레벨 범위** = 레벨 파일 **+ 스트리밍 그래프**(완전한 게임형 야심, Stage D).
- **기존 하드코딩 씬** = 파일로 통째 대체가 아니라 **씬 그래프로 마이그레이션**(그래프로 재구성하고 픽셀
  동일성 회귀 검증).

## 아키텍처

### 신규 크레이트 `crates/scene` (`dreamcoast-scene`)
RHI 비의존 — `glam` + `dreamcoast-asset` + `dreamcoast-core`에만 의존(`crates/render`가 `rhi` 파사드에만
의존하는 구조를 그대로 미러링). 씬/레벨 *로직*을 소유하고 평면 드로우 리스트를 산출한다. GPU 리소스는
전부 샌드박스가 소유하고 핸들 → 버퍼를 해석한다. 모듈 구성: `transform.rs`, `graph.rs`(arena),
`node.rs`, `gltf_instance.rs`(Stage B), `level.rs`(Stage C), `world.rs`(Stage D).

### 핵심 설계 선택 (기각 대안 포함)
- **Arena/SoA 트리, ECS 아님.** `SceneGraph { nodes: Vec<Node> }`를 generational `NodeId(u32)`로 키잉.
  인덱스 기반 부모/자식(`Rc` 순환 없음) → 캐시 친화적, 직렬화 가능, 단일 스레드 `!Send` 모델에 부합.
  *ECS 기각*: 학습용 렌더러엔 과하고, 사용자가 명시적으로 씬 그래프를 요청.
- **로컬 `Transform { translation, rotation: Quat, scale }` → 캐시된 `world: Mat4`, dirty-flag 전파.**
  `update_world_transforms()`는 dirty 서브트리만 재계산하는 top-down 단일 패스. 작은 씬에선 전체
  재계산이 trivial fallback.
- **노드 페이로드는 `NodeKind` enum**: `Empty | MeshInstance { mesh, material } | Light { .. } |
  Camera { .. }`. 메시/머티리얼은 **핸들**(`MeshHandle(u32)`, `MaterialHandle(u32)`)로 참조 —
  샌드박스 소유 레지스트리의 인덱스. 이 핸들이 `crates/scene`를 RHI 타입에서 분리하는 이음매.
- **드로우 리스트 추출**: `scene.draw_list() -> Vec<Drawable { world, mesh, material, flags }>`가
  오늘의 `Vec<SceneObject>`를 대체. `RtSystem`(TLAS 인스턴스)와 `CullSystem`이 같은 리스트를 소비 →
  그래프가 per-instance 트랜스폼의 단일 소스.

## 선행조건 (의존 Stage 전/내부에서 처리)

- **P1 — 자유 비행 카메라 (Stage D 차단).** 샌드박스 카메라는 **궤도 전용**, 씬 전체를 프레이밍하는
  angle 구동(`main.rs:772–776`)이라 청크 경계를 가로질러 주행할 수 없는데 Stage D 스트리밍이 이를 요구.
  WASD + 마우스룩 비행 카메라 추가(`platform::Input` 브리지에 휠 + char 큐 이미 존재; 마우스 델타만
  추가). 작고 자기완결적 → **Stage 0**으로 전진 배치해 Stage B/C 씬 검수에도 활용. 궤도 카메라는
  토글로 유지(헤드리스 스크린샷 베이스라인이 고정 각도 → Stage A 회귀가 바이트 동일 유지).
- **P2 — 레지스트리 기반 다중 머티리얼 업로드 (Stage B 차단, Stage A에서 확립).** 현재 `upload_mesh` /
  `upload_texture`(`apps/sandbox/src/mesh.rs`)는 메시 하나 + per-object `tex` 인덱스를 인라인 처리.
  N 프리미티브 / M 머티리얼 glTF(Lantern)는 레지스트리 기반 업로드 루프(텍스처 dedup, 머티리얼별
  `MaterialHandle` 하나)가 필요. Stage A가 `MeshRegistry` / `MaterialRegistry` 도입 시 이 경로를
  일반화해 Stage B 임포터가 재사용하게 한다.
- **P3 — Phase 12 `.dcasset` 컨테이너 (레벨/월드 *바이너리* 쿡만 차단).** 레벨/월드의 쿡된 바이너리
  형태(아래 직렬화 참조)는 Phase 12의 `.dcasset` 청크 컨테이너를 재사용하는데, 그 골격(M1 메시 직렬화)이
  **계획됐으나 미구현**(`docs/phase-12-asset-pipeline.md`). RON **텍스트** 직렬화는 선행조건 없이
  Stage C/D에서 출하; 바이너리 쿡은 Phase 12 M1 뒤로 순서화. 새 선행조건 추가가 아니라 교차 Phase
  순서 기록일 뿐.

## 직렬화 & 차후 애셋화 (Phase 12 교차)

레벨(Stage C)과 월드(Stage D)는 처음부터 직렬화 가능하게 설계해, Phase 12가 메시/SDF에 하듯 차후 엔진
애셋으로 쿡할 수 있게 한다.

- **지금 (Stage C/D): RON 텍스트, 왕복 serde.** 데이터 모델에 `serde::{Serialize, Deserialize}` 유도 —
  `Transform`, `Level`, 엔티티 레코드(`asset_ref`, `Transform`, 머티리얼 오버라이드), `Light`,
  `CameraDesc`, `Environment`, `World`(청크 리스트 + 배치 + 그래프 인접). 샌드박스가 `.level` /
  `.world`를 **로드·저장 양쪽**. 모델을 지금 serde-ready로 두면 차후 바이너리 쿡은 *데이터 모델* 변경이
  아니라 *포맷* 변경이 된다.
- **차후 (Phase 12 결속): 쿡된 바이너리 `.dclevel` / `.dcworld`.** Phase 12 `.dcasset` 컨테이너에
  씬/레벨 **청크 타입** 추가: 엔티티 리스트(애셋을 경로가 아닌 **source hash**로 참조), 트랜스폼,
  머티리얼 오버라이드, 라이트, 카메라, 환경, 그리고 (월드) 청크 그래프(인접 + 월드 공간 배치 + 스트리밍
  반경) 저장. `source_hash` + cook 파라미터로 키잉, 크로스백엔드 바이트 동일 — Phase 12 캐싱 모델과 동일.
  **Phase 12 신규 마일스톤(M3 "씬/레벨 청크")** 또는 Phase 13 **Stage E**로 안착, Phase 12 M1 이후.
  `docs/ROADMAP.md`(Phase 12·13)와 `docs/phase-12-asset-pipeline.md`에 교차 참조 추가.

## 단계별 (Stages)

각 Stage는 독립 커밋, 게이트: `cargo fmt --all` + `RUSTFLAGS="-D warnings" cargo clippy --workspace
--all-targets` + 양 백엔드 실행 + Vulkan 검증 클린(`VK_LOADER_LAYERS_DISABLE="~implicit~"`) +
양쪽 스크린샷 → **VK ≡ DX**.

### Stage 0 — 자유 비행 카메라 (선행조건 P1)
- 샌드박스에 WASD + 마우스룩 비행 카메라(yaw/pitch + position), `platform::Input`(마우스 델타 추가)
  구동. 궤도 카메라는 선택 가능하게 유지 → 헤드리스 스크린샷 베이스라인 고정 각도, Stage A 회귀 바이트
  동일.
- **검증:** 카메라 비행; 궤도 모드 스크린샷이 현재 베이스라인과 불변, VK ≡ DX.

### Stage A — 씬 그래프 코어 + 샌드박스 씬 마이그레이션
- `crates/scene` 생성: `Transform`, `Node`, `NodeId`, `SceneGraph` arena, `update_world_transforms`
  (dirty 전파), `NodeKind`, `Drawable`, `draw_list()`.
- 샌드박스가 `MeshRegistry` / `MaterialRegistry`(기존 `Buffer`/index-count/`tex` 데이터의 `Vec` 인덱스
  스토어) 소유. 씬 노드는 이들 핸들을 보유.
- 오늘의 5-오브젝트 씬(로드 모델 + 크롬 구 + 코퍼 구 + 빨강 큐브 + 지면)을 **씬 그래프로 프로그래밍
  구성**; 프레임 루프는 `scene.draw_list()`를 순회. `RtSystem` / `CullSystem`도 같은 리스트로 공급(당장
  RT는 정적 유지).
- **검증(회귀 게이트):** 기본 씬이 마이그레이션 전과 **바이트 동일** 렌더(양 백엔드) — 이 Stage는 순수
  데이터 경로 리팩터. RT 패스트레이서도 일치 유지.

### Stage B — 전체 glTF 계층 임포트 → 씬 서브트리
- `crates/asset` 확장: `load_gltf_scene(path) -> GltfScene` — **모든** 노드(TRS 포함), 메시/프리미티브,
  머티리얼, 텍스처 반환. 기존 단일 메시 `load_gltf`는 폴백/큐브 경로용으로 유지.
- `crates/scene`: `instantiate_gltf(&mut SceneGraph, &GltfScene, registries) -> NodeId` — 한 루트
  아래 서브트리 구성, 부모/자식 TRS 보존, 구별되는 메시/머티리얼 등록.
- 샌드박스가 **Lantern.glb** 로드 → 4노드 계층이 3개의 정확히 배치된 드로어블이 됨(오늘 로더는 Chain +
  Lantern을 누락). Avocado/BoomBox는 단일 노드 서브트리.
- **검증(애셋 구동):** Lantern이 **3개 서브메시 전부**를 올바른 상대 배치로 렌더; 부모 "Lantern" 노드
  회전 시 자식 3개가 함께 이동(계층 증명). VK ≡ DX.

### Stage C — 선언적 레벨 포맷 + 로더
- 레벨 파일을 **RON**으로(`ron` 크레이트 — 작고 순수 Rust; minimal-install 방침에 따라 사용자 승인
  대상; 새 의존이 싫으면 `serde_json` 폴백). 레벨 = 엔티티(애셋 ref + `Transform` + 선택 머티리얼
  오버라이드), 라이트(태양 + 포인트), 카메라, 환경(기존 `Globals` IBL/sky/exposure 노브). 데이터 모델은
  serde `Serialize`/`Deserialize` 유도(직렬화 참조) → 샌드박스가 레벨을 **로드·저장 양쪽**.
- `crates/scene` `level.rs`: `Level` 구조 + `load_level(path)` / `save_level(path)` → 엔티티를 씬
  그래프에 인스턴스화(각 엔티티는 Stage B로 glTF 서브트리 로드+배치).
- 샌드박스: 보유 애셋으로 `.level` 2개 저작 — `gallery.level`(마이그레이션된 기본 씬)과
  `lanterns.level`(Lantern 인스턴스 한 줄). 런타임 **레벨 전환/리로드** UI 드롭다운(그래프 재구성 + GPU
  리소스 재업로드; TLAS 재빌드).
- **검증:** 각 레벨이 올바른 씬으로 로드; UI에서 핫스왑; VK ≡ DX.

### Stage D — 레벨 그래프 + 스트리밍
- `World` = 레벨 청크의 그래프: 노드 = 레벨(각자 월드 공간 원점 오프셋), 엣지 = 인접/포털 연결.
  `.world` 파일이 청크 + 배치 + 연결성 나열.
- 스트리밍: 매 프레임, 카메라 위치 기준으로 반경/그래프 거리 안의 청크 **로드**, 밖의 청크 **언로드**.
  단일 스레드 엔진 → 동기 로드를 프레임당 한 청크로 예산화해 히치 방지(비동기 로드는 범위 외로 문서화).
  GPU 리소스는 언로드 시 per-chunk 리소스 arena로 해제(공유 메시는 refcount).
- 샌드박스: Lantern으로 채운 청크 3개를 한 줄로 둔 `.world`; 카메라를 날리면 청크가 스트림 인/아웃; UI가
  로드된 청크 집합 + 작은 ImGui 그래프 시각화 표시.
- `World` 데이터 모델 serde-ready(청크 리스트 + 배치 + 그래프 인접 + 스트리밍 반경); 샌드박스가 `.world`를
  **로드·저장 양쪽**.
- **검증:** 카메라 주행이 청크를 올바르게 로드/언로드(GPU 누수 없음); VK ≡ DX; 검증 클린.

### Stage E — 쿡된 바이너리 레벨/월드 (Phase 12 M1 이후; Phase 12 결속)
- Phase 12 `.dcasset` 컨테이너에 씬/레벨 **청크 타입** 추가; `.level`/`.world` → 바이너리 쿡,
  `source_hash` + cook 파라미터로 키잉, 크로스백엔드 바이트 동일. 런타임은 쿡된 형태를 직접 로드(RON
  재파싱 없음). **Phase 12 M3**으로 안착할 수도 — Phase 12 M1 구현 시 결정. **P3** 게이트.
- **검증:** 쿡된 로드가 RON 경로와 동일 렌더(양 백엔드); 바이트 동일 캐시.

## 파일 (생성 / 수정)
- **신규** `crates/scene/{Cargo.toml, src/lib.rs, transform.rs, graph.rs, node.rs, gltf_instance.rs,
  level.rs, world.rs}`; 워크스페이스 `Cargo.toml` members에 추가.
- **신규** `apps/sandbox/levels/{gallery.level, lanterns.level}` + `worlds/demo.world`.
- **수정** `crates/asset/src/lib.rs` — `load_gltf_scene` / `GltfScene` 추가(Stage B).
- **수정** `apps/sandbox/src/main.rs` — 자유 비행 카메라(Stage 0); `Vec<SceneObject>` 구성 →
  씬 그래프 구성 + 레지스트리 + `draw_list()` 소비(Stage A/C/D); 레벨/월드 전환+저장 UI.
- **수정** `apps/sandbox/src/{rt.rs, cull.rs, mesh.rs}` — 레지스트리 기반 다중 머티리얼 업로드(P2);
  인스턴스를 `draw_list()`로 공급; 씬 변경 시 TLAS 재빌드.
- **수정** `docs/ROADMAP.md` — Phase 12 뒤에 **Phase 13 — 씬 그래프 + 레벨 스트리밍**(🧪 계획) 추가,
  워크스페이스 구조 섹션에 `crates/scene` 기재, Phase 12 ↔ 13 직렬화 교차 참조(쿡된 레벨/월드 청크) 기록.
- **수정** `docs/phase-12-asset-pipeline.md` — 차후 씬/레벨 청크(Stage E / M3) 명기.

## 리스크 / 미결
- **RT/cull 결합:** `RtSystem`은 씬 순서를 미러링해 TLAS 인스턴스 테이블을 빌드 → 동적 드로우 리스트는
  씬/레벨 변경 시 TLAS 재빌드를 의미. Stage A는 씬을 정적으로 유지해 RT 유지; 동적 재빌드는 C/D에서
  안착하고 각 Stage마다 PT 레퍼런스 대비 재검증.
- **언로드 시 리소스 수명 (Stage D):** 청크의 비공유 GPU 리소스만 해제하도록 refcount / per-chunk arena
  필요 — Stage D에서 상세.
- **`ron` 의존성:** Stage C에서 사용자 승인 필요(대안: `serde_json` 또는 손수 만든 최소 파서).
- **glTF 애니메이션 / 스키닝:** 계층이 잠금 해제하는 자연스러운 후속이나 Phase 13 **범위 외**(향후 작업
  으로 기록).

## 검증 (Stage별)
`cargo fmt --all` → `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets` → 각 백엔드
헤드리스 실행 후 PNG Read:
`VK_LOADER_LAYERS_DISABLE="~implicit~" cargo run -q -p sandbox -- --backend vulkan|d3d12
--screenshot-clean tmp/x.png`, VK vs DX(및 Stage A에선 마이그레이션 전 베이스라인) 차분을
`tools/rt-compare.py`로. Stage B는 부모 노드 회전 토글로 계층 확인; Stage D는 청크 경계를 가로지르는
카메라 주행으로 로드/언로드를 검증 오류 없이 확인.
