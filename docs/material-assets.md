# 머티리얼 애셋화 — 공유 가능한 1급 머티리얼/텍스처 애셋 (세부 계획)

상위: [ROADMAP.md](ROADMAP.md) Phase 11(에셋 파이프라인) · Phase 12(씬 그래프).
가로지르는 토픽 — 쿡 컨테이너([phase-11-asset-pipeline.md](phase-11-asset-pipeline.md))와 런타임
레지스트리/씬([phase-12-scene-graph.md](phase-12-scene-graph.md)) 양쪽에 결속한다.

> **현재 위치 (착수 타이밍):** Phase 11 ✅ 완료 → `.dcasset` 청크 컨테이너가 닫혀 **토대는 준비됨**
> (`CHUNK_MATERIAL` 추가는 즉시 가능). Phase 12 🚧 진행 중 → 런타임/레벨 결속이 진행 중인 씬 그래프 작업에
> 올라탄다: 키 dedup은 **Stage B**(레지스트리 기반 다중 머티리얼 업로드), Material Instance는 **Stage C**
> (`.level` RON)/**Stage E**(바이너리 레벨). 따라서 본 작업은 **Phase 12와 함께 추진**하는 컴패니언 트랙이다
> (닫힌 Phase 11의 잔여 작업이 아님).

> **번호 주의:** 본 문서는 *현재* 문서 번호(에셋=Phase 11, 씬 그래프=Phase 12)를 쓴다. 코드의 env
> 변수·커밋은 옛 번호(`P12_*`)를 유지하므로([doc-phase-renumbering] 참조) 신규 opt-in 시드도 그 관례를
> 따라 `P12_*`로 명명한다.

## 동기 / 배경

머티리얼이 **메시에 종속**되어 있어 공유가 불가능하다. 게임에서는 한 머티리얼(예: "광택 구리",
"벽돌")을 수백 오브젝트가 공유하고, 변형은 인스턴스 단위 파라미터 오버라이드로 처리하는 게 정석인데,
현 구조는 오브젝트마다 머티리얼·텍스처를 복제한다.

| 계층 | 현재 | 한계 |
|---|---|---|
| 데이터 모델 | `Material`이 `MeshData`에 **소유됨** (1 메시 = 1 머티리얼) — `crates/asset/src/lib.rs:147` | 머티리얼 독립 정체성·재사용 단위 없음 |
| 쿡 컨테이너 | mesh `.dcasset` = 메시 청크 + 텍스처 청크를 **한 덩어리로 접착** — `crates/asset/src/dcasset/mesh.rs` | 머티리얼·텍스처를 메시와 분리해 공유 불가 |
| 텍스처 공유 | glTF import **한 건 내부**에서만 image index로 dedup — `apps/sandbox/src/registry.rs:205` | 애셋 간 공유 안 됨, 중복 디스크/VRAM |
| 런타임 | `MaterialRegistry`/`MaterialHandle` **이미 존재**(인덱스 모델) — `apps/sandbox/src/registry.rs:139` | **키 기반 dedup 없음** → 동일 머티리얼이 매번 새 슬롯 |
| 레벨 오버라이드 | `MaterialOverride` = **스칼라(색/metal/rough)만** — `crates/asset/src/level.rs:63` | 텍스처·베이스 머티리얼 참조 불가 |

요지: 런타임 인덱스 테이블(`MaterialHandle`)은 이미 "공유 가능한" 형태인데, **저작·쿡 쪽에 머티리얼이라는
애셋 단위가 없어서** 모든 오브젝트가 자기 머티리얼을 들고 다닌다. 이 갭을 메우는 게 본 작업이다.

> **이 엔진의 위상:** DreamCoast는 장기적으로 직접 게임 개발에 쓸 범용 엔진이다. 따라서 머티리얼은
> 지오메트리와 분리된 **콘텐츠-주소 1급 애셋**으로 두고, 메시는 슬롯으로 참조하며, 변형은 인스턴스
> 파라미터 오버라이드로 처리한다(상용 엔진 정석).

## 상용 엔진 레퍼런스

세 엔진이 같은 결론에 수렴한다 — **머티리얼은 지오메트리와 분리된 콘텐츠-주소 애셋, 메시는 슬롯으로
참조, 텍스처는 전역 공유, 변형은 인스턴스 파라미터 오버라이드.**

- **Unreal** — `UMaterial`(파라미터/셰이더 그래프 정의) → `UMaterialInstance`(부모 머티리얼 + 파라미터
  오버라이드만; 텍스처 복제 X). 게임 에셋 대부분이 *인스턴스*다. 메시는 머티리얼 **슬롯**을 선언하고,
  컴포넌트 단위로 슬롯을 오버라이드한다.
- **Unity** — `Shader`(프로그램) → `Material`(`.mat`, 공유 애셋) → `MeshRenderer.sharedMaterial[]`
  슬롯에 배정. 인스턴스 단위 변형은 `MaterialPropertyBlock`(텍스처를 늘리지 않음).
- **glTF** — 문서 스코프 `materials[]` 배열, primitive가 `material:<index>`로 참조, 머티리얼이
  `textures`/`images[]`를 참조. **이미 인덱스 그래프 모델**이며, 우리 `GltfScene`도 import 시점엔 이렇게
  분리해 들고 있다가 `.dcasset` 경계에서 다시 합쳐져 버린다.

**공통 패턴 5가지** (본 설계의 기준):
1. 머티리얼 = **독립 애셋**, 안정 키/콘텐츠 해시로 식별.
2. 메시 = **슬롯**으로 머티리얼 참조(메시는 슬롯 수만 선언).
3. 텍스처 = **전역 공유 애셋**, 콘텐츠-주소 dedup.
4. **Material Instance** = 베이스 머티리얼 + 파라미터 오버라이드(텍스처 복제 X).
5. 런타임 **머티리얼 라이브러리**가 동일 머티리얼을 한 번만 업로드.

## 아키텍처 / 핵심 설계

**핵심 아이디어 한 줄:** 오늘 mesh `.dcasset` 하나에 뭉쳐 있는 것을, **서로를 논리 키로 참조하는 독립
`.dcasset`들(texture / material / mesh / level)의 그래프로 분해한다.** 새 파일 포맷이 아니라 기존
컨테이너에 청크 타입과 키 정책만 추가 — [cooked-asset-policy]("모든 애셋은 같은 `.dcasset` 컨테이너,
타입별 청크 확장, 별도 로더 금지")를 그대로 따른다.

### 데이터 모델 (asset 크레이트 — 단일 소스)

`Material`을 메시에서 떼어내 독립 애셋으로 승격하고, 텍스처는 **임베드된 `TexData`가 아니라 키 참조**로
바꾼다(전역 dedup의 핵심):

```rust
// 텍스처를 데이터가 아니라 "참조"로 — 콘텐츠-해시 키로 해석
pub struct TextureRef { pub key: String }

pub struct MaterialAsset {
    pub key: String,                                 // 안정 식별자 = 쿡 키
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3],
    pub base_color:         Option<TextureRef>,      // 임베드 X → 참조
    pub metallic_roughness: Option<TextureRef>,
    pub normal:             Option<TextureRef>,
    pub emissive:           Option<TextureRef>,
    pub alpha: AlphaMode, pub double_sided: bool,    // 확장 여지(지금은 기본값)
}

// 메시는 슬롯만 선언 — glTF/UE 모델
pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices:  Vec<u32>,
    pub material_slot: String,                        // 기본 머티리얼 키(멀티-슬롯은 후속)
}
```

- `Material`(현 임베드형, `lib.rs:120`)의 색공간 규약(base/emissive=sRGB, mr/normal=linear)은
  `MaterialAsset`이 그대로 승계한다.
- `representative_albedo`(GI용 대표 알베도, `registry.rs:130`)는 **단일 정의 유지** — 텍스처 참조가
  resolve된 뒤 동일 함수로 계산.
- *멀티-슬롯 메시 기각(1차)*: glTF primitive 1개 = 슬롯 1개로 시작. N-슬롯은 Phase 12 Stage B
  멀티-프리미티브 임포트와 함께 후속(슬롯 배열로 자연 확장).

> **알파 모드 — MASK 구현됨 (2026-06-28, `cb2e1ca`).** 위 `alpha: AlphaMode`의 **MASK(알파 테스트)**
> 부분은 런타임 렌더 경로에 먼저 들어갔다(에셋-그래프 쿡과 독립). 임포터가 `alphaMode`/`alphaCutoff`를
> 읽어 `GltfMaterial.alpha_cutoff`(MASK=cutoff, 기본 0.5; OPAQUE/BLEND=0)에 보존 → 단일 값으로
> `MaterialDesc`→`SceneObject`→푸시상수(`mr_factor.w`) 전파. `gbuffer.slang`(컷아웃)과 `shadow.slang`
> (마스크드 그림자 구멍) 양쪽이 `alpha*factor.a < cutoff`면 discard. cutoff 0(불투명)은 텍스처 샘플/discard
> 미진입 = **바이트 동일**. cull=NONE이라 마스크드 폴리지는 양면. **BLEND(진짜 알파 블렌딩)는 후속**
> — 현재 BLEND는 opaque로 폴백한다. 이 폴백이 Intel Sponza의 `dirt_decal`(BLEND 데칼)을 검은 불투명
> 금속으로 만든 원인(RenderDoc 확정). 트랙 A = **deferred 데칼** [deferred-decals.md](deferred-decals.md),
> 트랙 B = 포워드 투명(glass). `alphaMode`를 cutoff뿐 아니라 `AlphaMode`/`MaterialKind`로 보존하는 게
> 데칼/투명 분류의 전제(A1).
> **텍스처 알파 주의:** 마스크드 base_color는 알파가 필요 → 라이브 glTF 경로는 RGBA8 무압축이라 보존되지만,
> 쿡 BCn(`P12_TEX_COMPRESS`)이 마스크드 컬러에 BC1(1-bit 알파)을 쓰면 컷아웃이 깨진다 → 마스크드 컬러는
> 무압축/BC7로 라우팅하는 게 후속 과제(현재 쿡은 머티리얼 alpha-mode 비인지). [cooked-asset-policy] 참조.

### 애셋 그래프 / 쿡

기존 `.dcasset` 컨테이너에 청크 타입만 추가하고, **각 애셋을 독립 키로 쿡**한다:

- **텍스처 애셋** — 콘텐츠 해시를 키로 자체 `.dcasset`(`CHUNK_TEXTURE` 단일). BCn 정책(컬러→BC1/BC7,
  노멀→BC5, 데이터 텍스처 무압축)은 그대로 승계. 키가 콘텐츠 해시라서 **전역 dedup이 공짜**.
- **머티리얼 애셋** — 신규 `CHUNK_MATERIAL`(factors + 텍스처 키들 + alpha/double_sided). 자체 `.dcasset`.
- **메시 애셋** — 지오메트리 + 머티리얼 슬롯 키. 텍스처·머티리얼 임베드 제거(`mesh.rs`의 `encode_mesh`가
  factors를 인라인하던 것을 슬롯 키 참조로 대체).
- 무효화 키 `{version, source_hash, cook_params_hash}`는 그대로 재사용(`dcasset/mod.rs:78`). `VERSION`
  **bump 필수**(레이아웃 변경).

glTF importer는 이미 `materials[]`/`images[]`를 분리해 들고 있으므로(`gltf_scene.rs`), 쿡 단계에서
**그 분리를 `.dcasset`까지 보존**만 하면 된다. glTF image index → 콘텐츠 해시 키 매핑이 dedup 지점.

> **기각: 단일 번들 유지.** 메시에 머티리얼·텍스처를 계속 임베드하면 공유가 원천 불가 — 본 작업의 목적
> 자체와 모순. 분해(graph-of-assets)가 유일하게 목적에 맞는 구조.

### Material Instance (UE 패턴)

레벨의 스칼라 전용 `MaterialOverride`(`level.rs:63`)를 일반화한다:

```rust
pub struct MaterialInstance {
    pub base: String,                                 // 베이스 머티리얼 키
    pub scalar_overrides: Vec<(ParamId, f32)>,        // metallic, roughness, …
    pub vector_overrides: Vec<(ParamId, [f32; 4])>,   // base_color_factor, emissive, …
    pub texture_overrides: Vec<(Slot, TextureRef)>,   // 후속 단계
}
```

`Entity.material_override`(`level.rs:51`)가 이를 참조 — 동일 베이스를 공유하면서 인스턴스별로
색/거칠기만 다르게(텍스처는 복제되지 않음). RON `.level` 저작(`load_ron`/`save_ron`)과 `CHUNK_LEVEL`
바이너리 양쪽에 동일 모델로 직렬화(단일 소스).

### 런타임

- **`TextureRegistry`** — 현 per-scene `image_cache`(`registry.rs:205`)를 전역
  `HashMap<TextureKey, bindless u32>`로 승격 → 애셋 간 텍스처 공유.
- **`MaterialRegistry`** — `HashMap<MaterialKey, MaterialHandle>` 추가(`registry.rs:139`) → 동일
  머티리얼 한 번만 등록. `MaterialDesc`/`MaterialHandle`/`MeshInstance`(`scene/components.rs`)는 이미
  올바른 인덱스 모델이라 변경 최소 — **이 핸들 이음매가 변경 없이 그대로 작동**한다.
- 인스턴스 오버라이드는 베이스 `MaterialDesc`를 복제 후 파라미터만 덮어쓴 **파생 핸들**로 등록(텍스처
  bindless 인덱스는 공유).

### 확장성 (RenderQuality / 셰이더 트랙과의 접속)

머티리얼 파라미터·슬롯 정의를 한 곳(asset 크레이트)에 모아 후속 확장(샘플러 wrap/scale, 알파 모드,
셰이더 변형/static switch, SSS/clearcoat 등 Phase 24 머티리얼 그래프)을 한 지점에서 받게 둔다. BCn 압축
티어(`P12_TEX_COMPRESS`)·`RenderQuality`와 동일한 "기본 off + env/플래그 seam" 원칙을 따른다.

## 단계별 구현 (검증·호환성 우선)

각 단계는 **렌더 출력 byte-identical 유지 + DX≡VK 0.000**을 합격 게이트로 둔다. 신규 경로는 opt-in 시드
`P12_MATERIAL_ASSETS`(기본 off = 기존 갤러리 경로 무회귀) 뒤에 격리하고, 픽셀 회귀 확인 후 기본화한다.

| 단계 | 내용 | 검증 게이트 |
|---|---|---|
| **M1 — 머티리얼 분리** | `Material`을 `MeshData`에서 분리(`MaterialAsset` + 키), 메시는 슬롯 참조. 런타임 빌드 경로는 동일 `MaterialDesc` 생성 → 출력 불변. `mesh.rs` 청크를 슬롯 키 참조로 전환(`VERSION` bump). | `tools/rt-compare.py` 무회귀, DX≡VK 0.000, asset 단위 테스트(roundtrip) |
| **M2 — 텍스처 전역 dedup** | 텍스처를 콘텐츠-해시 독립 `.dcasset`로 분리 + 전역 `TextureRegistry`. glTF image→키 매핑. | `.dcasset` 총량 감소(중복 텍스처 1회), 출력 불변, DX≡VK 0.000 |
| **M3 — 머티리얼 애셋 + 키 dedup** | `CHUNK_MATERIAL` 코덱 + `MaterialRegistry` 키 dedup + RON 머티리얼 라이브러리 저작(`.level`과 동일 방식의 `.material`/`.matlib`). | 동일 머티리얼 N오브젝트 = **1 슬롯** 확인, roundtrip 테스트, 결정적 쿡(바이트 동일) |
| **M4 — Material Instance 일반화** | `MaterialInstance`(베이스 + 스칼라/벡터→텍스처 오버라이드). `Entity.material_override` 결속, RON + `CHUNK_LEVEL` 양쪽. | 인스턴스 오버라이드 path-tracer 패리티, RON↔바이너리 roundtrip |

> **마이그레이션 안전망:** M1에서 레거시 임베드 경로를 `P12_MATERIAL_ASSETS` off로 보존하고, 신규
> 분해 경로가 픽셀 동일임을 확인한 뒤 기본 전환(전환기 공존, 영구 이중 표현 금지 — 씬 그래프 ECS
> 마이그레이션과 동일 원칙).

## 검증 전략

- **양 백엔드 동일:** 모든 단계 DX≡VK ≤ 0.001(목표 0.000) — 쿡은 순수 CPU·결정적이라 크로스백엔드
  바이트 동일이 보장(`dcasset/mod.rs` 규약).
- **패스트레이서 패리티:** 머티리얼/인스턴스 변경마다 `raster.png` vs `P8_PATHTRACE=1` 잔차 비교
  (`tools/rt-compare.py`). 머티리얼 분리가 셰이딩 입력을 바꾸지 않음을 잔차 무변동으로 확인.
- **단위 테스트:** `crates/asset`에 청크 roundtrip + 결정성 테스트(기존 `dcasset/mesh.rs`·`level.rs`
  테스트 패턴 답습).
- **dedup 실측:** 동일 머티리얼/텍스처 다수 배치 씬에서 레지스트리 슬롯 수·`.dcasset` 총량이 1회분으로
  줄어드는지 확인(본 작업의 성공 지표).
- clippy `-D warnings` 클린, `cargo fmt`.

## 범위 외 (후속)

- **셰이더 그래프 / static switch / 고급 셰이딩**(SSS·헤어·클로스) — Phase 24 머티리얼 그래프. 본 작업은
  metallic-roughness 고정 모델의 *애셋화*만.
- **머티리얼 핫리로드** — 셰이더 핫리로드([shader-system-todo])와 함께 후속.
- **샘플러 상태**(wrap/filter/anisotropy)의 머티리얼 노출 — 데이터 모델에 자리만 두고 1차 미구현.
- **에디터 머티리얼 서브에디터** — `apps/editor` E4 트랙.

## 미결 사항 (구현 착수 시 확정)

- **머티리얼 키 스킴** — 논리 경로(`"materials/copper"`) vs 콘텐츠 해시 vs 둘 다(텍스처=콘텐츠 해시,
  머티리얼=논리 키가 자연스러움). 텍스처는 dedup 목적상 콘텐츠 해시 유력.
- **RON 라이브러리 단위** — 머티리얼 1개 = 파일 1개(`.material`) vs 라이브러리 묶음(`.matlib`). `.level`
  관례와 일관성 고려.
- **`ParamId` 표현** — enum(고정 metallic-roughness 슬롯) vs 문자열(그래프 확장 대비). 1차는 enum,
  Phase 24에서 문자열로 일반화 여지.
