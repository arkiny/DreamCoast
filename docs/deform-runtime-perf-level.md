# 변형-메시 실시간화 · 성능 회복 · 선언적 레벨 애셋 (3-트랙 계획)

상위: [ROADMAP.md](ROADMAP.md) · 선행: [alembic-usd-import.md](alembic-usd-import.md)(vertex-cache 임포트/쿡),
[phase-13-animation-skinning.md](phase-13-animation-skinning.md)(skin/morph 링·prev-pose 패턴),
[velocity-motion-vectors.md](velocity-motion-vectors.md)(모션벡터 패스),
[sponza-perf.md](sponza-perf.md)·[swrt-gi-perf-track.md](swrt-gi-perf-track.md)(perf 레버).

브랜치: `feat/deform-runtime-perf` (main `5c4f8c7`에서 분기). 각 Phase = 독립 검증 커밋.

## 구현 상태 (2026-07-06)
- **Phase A ✅** `9568f76` — 일반 `deform` 모듈(`character.rs`→`deform.rs`, `DeformPlayer`) + per-FIF
  VB 링 더블버퍼(morph CPU 경로 선례). `MeshRegistry::upload_geometry_aabb`(deform 파트 always-visible
  AABB, 대모션 컬링 정확). DX≡VK 0.142/ch.
- **Phase B ✅** `2eb3db4`(RHI) + `bde854c`(sandbox) — 베이크드-정점 모션벡터(`vsMainDeform` + per-fif
  prev-position bindless storage 링, `P_VELOCITY=1`일 때만 할당). **발견+수정한 선재 버그:** velocity
  패스가 `DepthCompare::Less`를 써서 VK/D3D12(strict LESS, Metal은 LessEqual 매핑)에서 동일-깊이 표면을
  전부 reject → Windows에서 모션벡터 패스가 **아무것도 렌더 안 함**(Metal에서만 동작). 백엔드 균일한
  `DepthCompare::LessEqual` 추가. 변형 knight 모션 확인(전엔 0), lit+TAAU DX≡VK 0.158/ch 0.03%>8.
- **Phase C ✅** `5a44856` — vcache를 `.level` 1급 `deforms` 엔티티로 승격. `LevelData.deforms`
  (`#[serde(default)]`), `CHUNK_LEVEL` codec v8→**v9**(deforms 추가, v8 prefix 호환 = EOF→0),
  `build_level`가 쿡+spawn, `deform::spawn`이 full `LocalTransform` 수용. `KNIGHT_USD/ABC` env 해킹
  제거, 신규 `sponza_knight.level`. DX≡VK 0.070/ch. 핫스왑(`load_level`) 플레이어 재생성.
- **Phase D ✅** `ed4f163` — 프레임 데시메이션 메모리 예산. `VertexCache::decimate`(균등 서브샘플, duration
  보존), `load_or_cook_vcache(max_frames)` 키에 fold, `quality::deform_max_frames`(env `DEFORM_MAX_FRAMES`,
  기본 0=무예산). `DEFORM_MAX_FRAMES=100`: knight 300→100f@8fps, 쿡 223MB→**76MB**, DX≡VK 0.130/ch.
  **윈도우 스트리밍(offset table + 프레임 창)은 후속** — 이 데시메이션이 결정적·항상-정확한 coarse 예산.
- **Phase E ✅** `79282e9` — VK 1080p 60fps perf 회복. RTX2070S 헤드리스 `PROFILE_GPU` 재측정(메모리
  프로파일 stale — top이 gdf_reflect 아닌 **gdf_ao**, sdf_cache_light은 Med에 부재). Med 3-knob 리튠
  (사용자 승인): `ao_res_div 1→2`(gdf_ao 6.7–8.1→1.7–2.1ms, 반해상+guided upsample), `ssao off`(중복
  2차 AO 제거 ~2.6–2.8ms), `gi_volume_period 1→4`(QualityPreset 신규 필드, view-independent DDGI 암토화).
  reflect는 div2 유지(선명). **결과: DX 24→~15.8ms(~63fps), VK 25.1→~16.5ms(~60fps)**, DX≡VK 0.093/ch.
  gallery PT 앵커 불변(gallery_preset resolve). med legacy-lock 테스트 → 리튠 baseline lock으로 전환.
  reflect_res_div=4는 parity 0.463(더 큼)이라 미채택. **후속:** VK가 60fps 경계(worst ~59fps) — 여유
  위해 gi_volume 추가 amortize / VK async overlap 여지. 실기 인터랙티브 vsync 확인 권장.


## 배경 (직전 세션, main 5c4f8c7)
- 네이티브 애니메이션 임포트 트랙 종료: FBX(ufbx) + Alembic(.abc) + 네이티브 ASCII USD(.usda) 포인트
  캐시. 두 소스 → 중립 `crate::vcache::VertexCache` → 하나의 쿡 `cook::load_or_cook_vcache`
  (`CHUNK_VCACHE` v8, 무효화 키 = len+mtime+path) → 별도 `.dcasset` → 레벨 CacheHit 로드.
- 현재 재생 경로: [`apps/sandbox/src/character.rs`](../apps/sandbox/src/character.rs) `VertexCachePlayer`
  + `overlay_vcache`. env `KNIGHT_USD=1`/`KNIGHT_ABC=1`로 LEVEL 모드에서 오버레이(임시 해킹).
- 핵심 사실: knight USD는 UsdSkel 아님 = 베이크된 포인트 캐시(리그 없음). 진짜 리그는 Maya `.ma`
  에만 → 스코프 밖.

## 현재 코드의 정확한 상태 (착수 전 실측)
- `VertexCachePlayer::update(dt)`가 **파트당 host-visible VB 하나**를 매 프레임 `g.vbuf.write(...)`
  로 재기록. 프레임 인덱스 = `(time*fps) as usize % num_frames`. 정적 씬 리스트가 이미 그 VB를
  가리키므로(핸들 경유) 별도 patch 없음. → **인플라이트 프레임과 CPU write 레이스**(헤드리스는 결정적
  이라 무해).
- **정확히 재사용할 선례가 있다:** [`apps/sandbox/src/morph.rs`](../apps/sandbox/src/morph.rs)의
  CPU-morph 경로(`CpuMorphMesh`)가 이미 `ring: Vec<Rc<GpuMesh>>`(FRAMES_IN_FLIGHT), `patch_scene`에서
  `obj.mesh = ring[fif].clone()` 스왑, prev-pose 링을 구현. vcache의 1a는 이 패턴의 직역이다.
- **모션벡터 갭(1b):** CPU-morph도 vcache도 현재 velocity 패스에서 **표준 `pipeline`**(비-skin/비-morph)
  으로 그려진다 → prev_mvp가 `prev_view_proj * prev.transform`뿐. 정점이 베이크되어 바뀌었어도 velocity는
  카메라/노드 이동만 반영, **표면 변형 속도 = 0**. 즉 이건 vcache만의 문제가 아니라 "베이크드-정점" 경로
  공통 갭. 1b는 이 경로에 prev-position을 공급하는 일반 인프라로 설계한다.
- `SceneObject`는 `mesh: Rc<GpuMesh>` + `skin/morph: Option<[u32;4]>` 슬롯 보유. velocity 패스는
  `obj.mesh.vbuf`를 vertex 스트림으로 draw. skin/morph velocity는 prev 팔레트/weights를 **bindless
  storage index**로 읽어 prev 위치 재구성 → 같은 메커니즘을 baked-vertex에 확장 가능.
- `FRAMES_IN_FLIGHT = 2`. VB는 `BufferUsage::Vertex` + `.write()`(host-visible) 지원.
- `CHUNK_VCACHE` 코덱은 **전 프레임을 메모리로 디코드**(offset table 없음) → 스트리밍(1c)엔 포맷 확장 필요.

---

## 실행 순서 (의존성 반영)
사용자 트랙 번호(1/2/3)는 우선순위, 실제 착수는 의존성 순:

1. **Phase A** = 1a + 1d — per-FIF 더블버퍼 + knight-전용→일반 `deform` 모듈. (실시간 정확성의 토대)
2. **Phase B** = 1b — 변형 표면 모션벡터(베이크드-정점 공통 인프라). Phase A의 링 위에 얹음.
3. **Phase C** = 3 — vcache를 `.level` 1급 엔티티로 승격, env 해킹 제거. Phase A의 일반 API 소비.
4. **Phase D** = 1c — 메모리/스트리밍 예산(포맷 확장 + 프레임 윈도우). 가장 무겁고 독립적.
5. **Phase E** = 2 — VK 1080p 60fps 성능 회복(측정 주도). 실기(RTX 2070S) 반복 필요.

각 Phase는 앞 Phase에 무회귀. Phase A~C가 "실시간 재생 + 선언 레벨"을 눈으로 검증 가능한 상태로 만들고,
D/E가 예산·성능을 붙인다.

---

## Phase A — per-FIF 더블버퍼 + 일반 `deform` 모듈 (작업 1a·1d)

### A1. `deform` 모듈 분리 (1d)
- `character.rs`의 vertex-cache 부분(`VertexCachePlayer`, `overlay_vcache`, `compute_normals`,
  `frame_vertices`, `vertex_bytes`, `knight_abc_placement`)을 새 파일
  **`apps/sandbox/src/deform.rs`**로 이동. knight 명칭 제거 → cloth/destruction 재사용 가능한 일반명:
  - `VertexCachePlayer` → `DeformPlayer`
  - `overlay_vcache(...)` → `deform::spawn(world, meshes, materials, cache, place, material_desc)`
    (재질을 인자화 — knight 브러시드메탈은 호출측 기본값). knight 명은 호출측(레벨/env)에만.
- `character.rs`는 skin/morph 오버레이(glTF/FBX) 전용으로 유지. `main.rs`/`mod` 참조 갱신.
- **무회귀**: 순수 이동 + 이름변경, 동작 불변. 헤드리스 knight 캡처가 이동 전후 byte-동일.

### A2. per-FIF VB 링 (1a)
- `DeformPlayer` 파트별 상태를 `CpuMorphMesh` 선례대로 재구성:
  ```
  struct DeformPart {
      base_mesh: Rc<GpuMesh>,        // 씬 리스트가 처음 참조하는 프레임-0 메시 (Rc identity 매칭용)
      ring: Vec<Rc<GpuMesh>>,        // FRAMES_IN_FLIGHT 개 VB (frustum-cull 제외 AABB)
      // 1b에서 prev-pos storage 추가
  }
  ```
- `DeformPlayer::update(fif, dt)`: 프레임 인덱스 계산 후 **`ring[fif]`만** write(현 프레임 정점).
- `DeformPlayer::patch_scene(scene, prev, fif)`: `morph::patch_scene`와 동형 —
  `Rc::ptr_eq(obj.mesh, base_mesh)`인 드로어블을 `obj.mesh = ring[fif].clone()`으로 스왑.
- `main.rs` 프레임 루프(현 `vc.update(FIXED_DT)` 자리, ~3800): `update(fif, FIXED_DT)` +
  `patch_scene(&mut scene, &mut prev_scene, fif)` 호출(skin/morph 패치와 같은 블록, inline-경로 전용).
- **cull 상호작용**: ring 메시 AABB는 morph처럼 항상-보임(`[-1e9,1e9]`) — 변형으로 프레임별 bounds가
  변하므로 프러스텀 컬 제외(정확성 우선; 예산은 1c). 히스테리시스는 없음.

### A3. 검증 (Phase A)
- `LEVEL=sponza KNIGHT_USD=1` 헤드리스 DX/VK 캡처 + `tools/rt-compare.py` → DX≡VK가 기존
  (0.10%>8 = no-knight 0.01% + 씬 GI 1-LSB) 수준 유지, 결정적 지오메트리에 divergence 추가 없음.
- 인터랙티브(비-headless)에서 실시간 재생 시 **틀림/깜빡임(레이스) 소멸** 육안 확인.
- `KNIGHT_ABC=1`도 동일. no-knight 레벨(gallery/sponza) byte-무회귀.
- `rtk proxy cargo clippy -p sandbox --all-targets -- -D warnings` + `cargo fmt` 클린.
- **커밋**: `refactor(sandbox): general deform module + per-FIF vertex-cache double-buffer`

---

## Phase B — 변형 표면 모션벡터 (작업 1b)

### 설계 결정
베이크드-정점(vcache, 그리고 후속으로 CPU-morph)은 정점이 프레임마다 통째로 바뀌므로, velocity 패스가
**현 프레임 위치와 이전 프레임 위치를 둘 다** 봐야 per-vertex 모션을 낸다. 엔진의 bindless-first 관례
(skin/morph velocity가 prev를 storage index로 읽음)에 맞춰:
- 파트별로 **prev-position bindless storage 링**을 둔다(`ring`과 병렬, FRAMES_IN_FLIGHT).
  `update`가 `ring[fif]` VB를 쓸 때 같은 위치 배열을 `pos_storage[fif]`(host storage)에도 쓴다.
- 새 velocity VS 엔트리 **`vsMainDeform`**(`crates/shader/shaders/velocity.slang`): `SV_VertexID`로
  `prev_positions[vid]`를 읽어 `prev_ndc = prev_view_proj * prevpos`, 현 위치는 vertex 스트림에서.
  → `cur_ndc.xy - prev_ndc.xy`가 표면 변형까지 반영.
- `SceneObject`에 `deform: Option<[u32;2]>`(cur_pos_idx 불필요 시 prev_pos_idx + vert_count) 슬롯 추가,
  또는 기존 `morph` 슬롯 재해석 회피 위해 신규 슬롯. `velocity::PrevPose`에 `deform_prev: u32` 추가.
- `patch_scene`가 `obj.deform = Some([prev_pos_idx, vert_count])`, `prev.transform = IDENTITY`(위치가
  절대 월드공간이면) 설정. velocity 패스는 `obj.deform.is_some()`일 때 `deform_pipeline` 바인드.
- push block: 기존 192B에 deform prev index가 이미 `skin_prev`/`morph_prev` 슬롯과 유사하게 들어가도록
  packer 확장(또는 morph_prev 자리 재사용 검토).

### 대안 (기록)
- 2-스트림(현 VB + prev VB) velocity 파이프라인: `VertexLayout` 단일 스트림 가정을 깨야 함 →
  bindless storage 안이 관례에 부합, 채택.
- prev-pos storage는 velocity 전용(추가 ~파트당 vert×12B×FIF). 예산은 1c에서 함께 계량.

### B 검증
- `DEBUG_VIEW=11`(velocity 컬러코드) 헤드리스에서 knight 변형부에 **0이 아닌 모션** 확인(기존엔 0).
- TAA on(`P_TAAU_FORCE` 등) 상태에서 변형 표면의 고스팅/스미어가 줄어드는지 육안 + rt-compare 잔차.
- DX≡VK(prev-pos storage 결정적), no-deform 씬 무회귀(velocity 패스 자체 opt-in 유지).
- **커밋**: `feat(sandbox): per-vertex motion vectors for baked vertex-cache deforms`

---

## Phase C — vcache를 선언적 `.level` 애셋으로 승격 (작업 3)

### C1. LevelData 스키마 확장 (`crates/asset/src/level.rs`)
- `LevelData`에 신규 필드 `pub deforms: Vec<DeformEntity>`(기본 빈 vec, `#[serde(default)]`로 하위호환):
  ```
  pub struct DeformEntity {
      pub source: String,              // 쿡 소스 경로: "assets/Knight/knight.usda" | "*.abc"
      pub transform: [f32; 16],        // placement (기존 Entity와 동일 규약)
      pub material_override: Option<MaterialOverride>,
  }
  ```
- RON 라운드트립 유닛테스트에 deforms 케이스 추가.

### C2. 쿡 포맷 (`crates/asset/src/dcasset/level.rs` + `cook/level.rs`)
- `CHUNK_LEVEL` write/read에 deforms 직렬화 추가 → `dcasset::VERSION` 범프(레벨 캐시 재쿡, 무해).
- `load_or_cook_level`은 변화 없음(스키마만 확장). deforms 없는 기존 레벨은 빈 vec.

### C3. build_level 배선 (`apps/sandbox/src/level.rs`)
- `build_level` 시그니처 확장: deform 소스를 `load_or_cook_vcache`로 쿡→로드 후 `deform::spawn`,
  생성된 `DeformPlayer`들을 반환(예: 반환형을 `(Option<Bounds>, Vec<DeformPlayer>)`로).
- `main.rs` LEVEL 경로: 반환된 players를 `gltf_vcache`(현 단일 Option) 대신 **`Vec<DeformPlayer>`**로
  보관, 프레임 루프가 전부 update/patch. **env `KNIGHT_USD`/`KNIGHT_ABC` 및 `overlay_vcache` 직접호출
  제거** — 대신 `sponza.level`(또는 신규 `knight.level`)이 deforms로 선언.
- placement 헬퍼(`knight_abc_placement`)의 기본값은 데모 레벨의 deforms 항목으로 이관.

### C4. 데모 레벨
- `ensure_level_files`에 knight를 실은 레벨을 추가하거나 기존 `sponza.level`에 deforms 1건 추가
  (assets 없으면 로드시 클린 에러 — 기존 Sponza 규약과 동일). assets는 gitignored이므로 레벨 RON만 커밋.

### C 검증
- deforms 실은 레벨 로드 → knight 재생, 2번째 로드 CacheHit(vcache + level 둘 다), DX≡VK.
- env 해킹 제거 후에도 동일 결과. deforms 없는 레벨(gallery 등) byte-무회귀.
- 신규 asset 크레이트 테스트(RON 라운드트립 + dcasset 라운드트립) 통과.
- `rtk proxy cargo clippy -p dreamcoast-asset -p sandbox --all-targets -- -D warnings` + fmt 클린.
- **커밋**: `feat(level): declarative vertex-cache deform entities (remove env overlay hack)`

---

## Phase D — 메모리/스트리밍 예산 (작업 1c)

쿡 USD `.dcasset` 223MB, abc 1.26GB를 통째 상주 → 게임 예산 밖. 품질 파라미터는 한 곳(스칼라빌리티
설정)으로 모아 RenderQuality 티어화(CLAUDE.md 3).

### D1. 예산 파라미터 (한 곳)
- `apps/sandbox/config/scalability.ron` 또는 신규 `DeformQuality` 블록에:
  `frame_stride`(프레임 데시메이션: 매 N번째 프레임만 상주/재생, 사이는 보간 or nearest),
  `max_resident_frames`(스트리밍 윈도우 크기), `decimate`(정점 데시메이션 비율 seam).
- 소비처는 개별 env(`DEFORM_FRAME_STRIDE` 등) → 프리셋 순으로 resolve(기존 quality.rs 관례).

### D2. 스트리밍 리더 (포맷 확장)
- `CHUNK_VCACHE`에 **per-frame(파트별) offset table** 추가(VERSION 범프): 스트리밍 리더가 프레임 f의
  위치 블록만 seek/read. 전체 디코드 API는 유지(하위호환, 티어 off = 현행).
- 런타임: `DeformPlayer`가 전 프레임 `Vec<Vec<[f32;3]>>` 상주 대신 **프레임 윈도우**(현재±리드어헤드)만
  보유, 재생 진행에 따라 백그라운드(잡 시스템)로 다음 창을 로드. 예산 초과 시 stride로 폴백.
- 1d 원칙: cloth/destruction도 같은 스트리밍 예산 재사용(파트 구조 일반).

### D3. 측정/검증
- 상주 메모리(abc 1.26GB → 윈도우×stride로 목표 예산, 예: <150MB) 실측 로그.
- stride/윈도우가 재생 품질에 주는 영향 육안 + DX≡VK(디코드 결정적, 업로드 정적).
- 헤드리스 캡처는 결정적 유지(고정 클록 → 고정 프레임/윈도우).
- **커밋(분할 가능)**: `feat(asset): streaming vertex-cache frame windows + decimation budget`

> 리스크: 스트리밍은 잡-시스템 + IO 레이턴시가 헤드리스 결정성/DX≡VK를 흔들 수 있음 → 헤드리스는
> 동기 로드 경로(윈도우 프리로드)로 고정, 스트리밍은 인터랙티브에만. 이 갈림을 D2에서 명시 설계.

---

## Phase E — VK 1080p 60fps 성능 회복 (작업 2)

메모리: GI-fidelity wave-1 이후 Sponza 12.6→27ms(DX)/16.6→33ms(VK), VK ~47fps 바닥 고착. top =
gdf_reflect 40% + sdf_cache_light 25% + gdf_ao 12%; SCREEN_PROBE 2× slow(trace 미최적); VK floor =
view-independent(cache+gi_volume) → gi_volume amortization / VK async overlap 필요.
도구: `PROFILE_GPU=1`, `tools/measure.py`, RON 런타임 리로드, 레버 = quality.rs Med 티어.

### E1. 다각도 재측정 (기준선)
- `PROFILE_GPU=1` + `measure.py`로 **1080p** DX/VK 다각도 패스별 ms 재수집(현재 값이 이전 해상도 기준일 수
  있음). top-N 패스 확정 + `docs/`에 표로 기록.

### E2. DX 60fps 확정 → VK 공략
- Med 티어 레버부터: gdf_reflect `reflect_res_div`/`reflect_max_steps`/`reflect_half_res`,
  sdf_cache_light `cache_relight_period`/`cache_relight_spp`, gdf_ao `ao_res_div`. **정확도 유지**하며
  탭/대역폭/디스패치 절감(rt-compare 잔차로 품질 가드).
- VK 바닥 = view-independent: **gi_volume 업데이트 amortization**(프레임 분할 갱신) + **VK async compute
  overlap**(cache/gi_volume를 그래픽스와 겹침). 구조적 floor를 깨는 게 핵심.

### E3. 검증
- 1080p **양 백엔드 60fps**(측정 로그). DX≡VK ≤0.001/ch 유지, gallery PT 잔차 무회귀(품질 유지).
- 레버 변경은 quality.rs/scalability.ron 한 곳, 티어 경계 명확.
- **커밋(분할)**: `perf(gi): recover 1080p 60fps (VK gi_volume amortization + async overlap)`

> ⚠ 이 Phase는 **실기(RTX 2070S) 반복 측정**이 필수 — 헤드리스 캡처만으론 프레임타임/async overlap을
> 검증 못 함. 사용자 환경에서 measure.py를 돌리며 진행. (다른 Phase는 헤드리스로 대부분 검증 가능.)

---

## 관례 / 게이트 (전 Phase 공통)
- clippy 게이트(rtk 훅이 `-- -D warnings`를 망가뜨림):
  `rtk proxy cargo clippy -p dreamcoast-asset -p sandbox --all-targets -- -D warnings`.
- `cargo fmt` 클린. asset 크레이트 유닛테스트(`cargo test -p dreamcoast-asset`) 통과.
- 헤드리스 DX≡VK: `cargo run -p sandbox --release -- --backend d3d12|vulkan --screenshot-clean out.png`
  → `python tools/rt-compare.py dx.png vk.png diff.png`. 반복은 `LEVEL=sponza`(Crytek, 빠른 CacheHit).
  Intel Sponza(`LEVEL=sponza_intel`)는 로드 ~5–9분.
- DX≡VK 기준: no-knight 베이스라인 ~0.95 avg / 0.01%>8 = 씬 GI 알려진 backend 1-LSB(수용). 결정적
  지오메트리는 divergence 추가 없어야 함.
- assets/ gitignored(커밋 금지). knight.usda는 `D:/Assets/intelsponza/pkg_e_knight_anim.zip`에서 추출됨.
- 커밋 메시지 끝: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## 리스크 요약
- **1b prev-pos storage 슬롯 배선**: `SceneObject`/`PrevPose`/push packer/velocity.slang을 함께 손대야
  함 → skin/morph 선례를 정확히 따라 회귀 최소화.
- **1c 스트리밍 결정성**: IO/잡 레이턴시가 헤드리스 DX≡VK를 흔들 위험 → 헤드리스=동기 프리로드로 고정.
- **작업 2 실기 의존**: 프레임타임/async는 사용자 GPU에서만 검증 가능.
- **포맷 VERSION 범프 2회**(CHUNK_LEVEL deforms, CHUNK_VCACHE offset table): 캐시 재쿡만 유발, 데이터
  손실 없음. cooked-asset-policy 준수(같은 컨테이너 확장).
</content>
</invoke>
