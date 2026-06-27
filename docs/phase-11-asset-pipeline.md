# Phase 11 — 에셋 파이프라인 / 쿠킹된 에셋 (세부 계획 / 스텁)

상위: [ROADMAP.md](ROADMAP.md) Phase 11. **크로스컷팅 엔진 인프라** — 특정 렌더 기법이 아니라
`crates/asset`의 자산 직렬화 계층이다. Phase 10(Distance-Field GI)의 GDF 베이크 영속화가 직접적 동기였으나,
**메시 지오메트리 자체를 포함해** 가공된 자산을 하나의 바이너리로 저장/로드하는 실질적 에셋 개념이다.

> 목표: 매 실행마다 glTF를 재파싱하고 SDF를 재베이크하지 않는다. **가공된 메시 + 베이크 데이터(SDF,
> 향후 BVH/라이트맵 등)를 하나의 직렬화된 `.dcasset`으로 cook → 저장 → 런타임 직접 로드.**

## 동기 / 배경
- 현재 `crates/asset`은 런타임에 `gltf`+`image`로 glTF→`MeshData`를 파싱한다(즉석, 영속화 없음).
- Phase 10 Stage B의 per-mesh SDF 베이크는 비싸다(brute-force 점→삼각형). 한 번 베이크하고 캐시해야 한다.
- 사용자 요청: SDF만이 아니라 **메시까지 함께 직렬화하는 실질적 에셋**. → 별도 Phase로 승격.

## 포맷 `.dcasset` (가칭) — 청크 컨테이너
- **헤더** {magic, version, source_hash, flags, chunk_count + 청크 디렉터리(타입·오프셋·크기)}.
- **메시 청크** {정점(32B 레이아웃: pos/normal/uv), 인덱스(u32), material params, AABB,
  (후속) tangents/LOD/meshlet}.
- **SDF 볼륨 청크**(옵션) {dims(x,y,z), aabb_min/max, format=R16F, voxels} — 첫 베이크 페이로드.
- **확장 청크**(후속) BVH, 라이트맵, 미리 계산 프로브 등 — 같은 컨테이너에 타입 태그로 추가.
- 청크 디렉터리 기반이라 **앞으로 데이터를 더 붙여도 포맷 호환** 유지가 쉽다.

## 무효화 / 결정성
- **키:** `source_hash`(소스 glTF 바이트 해시) + cook 파라미터(SDF 해상도 등) + `version`. 불일치/부재 → 재쿡.
- **결정성:** voxel·메시 바이트는 **크로스백엔드 바이트 동일** 원칙(밉 체인 규칙과 동일 — CPU 또는 결정적
  컴퓨트 베이크). 백엔드가 달라도 동일 `.dcasset`을 만들어야 패리티 게이트가 유지된다.
- **위치:** gitignored `cache/`(자동 쿡) 또는 커밋되는 `assets/cooked/`(배포용). 결정은 M1에서.

## 마일스톤 (각 게이트: build+fmt+clippy(-D warnings) + 동작 확인; 메시 경로는 양 백엔드 렌더 일치)

### M1 — 메시 직렬화 (.dcasset 골격) — Phase 10과 독립, 먼저 가능 — ✅ 완료
**수동 little-endian 청크 컨테이너**로 구현(`serde`/`bincode` 미사용 → 새 의존 0, 바이트
레이아웃 직접 통제 → 크로스백엔드 바이트 동일·버전 마이그레이션 용이). 모듈:
`crates/asset/src/dcasset.rs`(포맷), `crates/asset/src/cook.rs`(오케스트레이션).

- **M1.1 (`602a75b`)** — `.dcasset` reader/writer + 헤더 + **메시 청크**(머티리얼 팩터 +
  vtx/idx). FNV-1a 콘텐츠 해시(M4 셰이더 캐시 상수 미러; 빌드 스크립트는 lib 공유 불가).
  무효화 키 `{version, source_hash, cook_params_hash}` 단일 소스. 경계 검사 reader
  (손상/절단 → `Err`=미스, 패닉 없음). 테스트: round-trip·결정성(2회 쿡 바이트 동일)·
  bad-magic·truncation.
- **M1.2 (`12a3502`)** — **텍스처 청크**(슬롯 태그: base_color/MR/normal/emissive RGBA8)
  라운드트립. 없는 슬롯은 청크 미생성. `.dcasset`이 `MeshData` 완전 표현.
- **M1.3 (`2e7a843`)** — **cook 오케스트레이션** `load_cooked(source, cache_key, cache_dir)`
  (lazy: 히트→파싱 생략 / 미스→glTF 쿡+atomic write / 소스 부재+캐시 존재→직접 로드) +
  샌드박스 연결(`app::cooked_cache_dir()`, gitignored `/cache/`, unit_cube 폴백 유지).
  **캐시 파일명은 cwd-독립 논리 키(원본 ref)** 기준(해석된 경로 아님) — 안 그러면 쿡 런과
  소스-부재 런이 다른 파일명을 만들어 서로 미스.

검증(RTX 2070 SUPER, Avocado): run1 `Cooked` → run2 `CacheHit`(glTF 파싱 생략). DX/VK 쿡
vs 기준선 0.000/ch, **DX≡VK 0.000/ch**(max 5, 기준선과 동일). glTF 제거 시
`CacheHitNoSource`로 0.000/ch 동일 렌더. clippy/fmt 클린, asset 테스트 9 통과.
(.dcasset 크기: Avocado 2K 텍스처 raw RGBA8라 ~50MB → 텍스처 압축 KTX2/DDS는 M3.)

### M2 — SDF 베이크 청크 통합 (Phase 10 Stage B 이후) — ✅ 완료
대상은 실제 렌더가 소비하는 **scene GDF**(Stage C1 fused world-space SDF). 핵심 결정:
**GPU 베이크 → CPU 베이크**로 전환(방침: cook은 CPU·결정적). `sdf_bake.slang`을 Rust로 충실
포팅 → 결과를 GPU 볼륨에 업로드해 1회성 GPU 베이크 패스를 대체.

- **M2.1 (`6342851`)** — CPU SDF 베이크 커널 `crates/asset/src/sdf.rs`(closest-point-on-triangle
  Ericson + 부호=최근접 삼각형 평균법선). fused 32B 정점 레이아웃 직접 읽기(단일소스). Z-슬랩
  병렬(std::thread::scope, rayon 무의존). 테스트: 해석적 구 대비 + 결정성.
- **M2.2 (`b477532`)** — RHI **볼륨 업로드** `Device::create_volume_init(&VolumeDesc, bytes)`
  (vulkan: 스테이징→3D 이미지 copy→SHADER_READ / d3d12: UPLOAD heap→256B row pitch→
  NON_PIXEL_SHADER_RESOURCE / metal: Shared replaceRegion). 양 백엔드 동일 바이트 업로드.
- **M2.3 (`54fc2cf`)** — `.dcasset` **SDF 청크**(`CHUNK_SDF` {dim, aabb, dim³ R32F}) +
  `write_sdf`/`read_sdf` + `hash_begin`/`hash_update`(대용량 fused 버퍼 무복사 키). cook
  `load_or_bake_scene_sdf`(키=hash(fused_vtx,fused_idx,dim,aabb); 히트→로드, 미스→CPU베이크+
  저장). gdf.rs `build_scene_sdf(…, cooked_sdf)` → `create_volume_init` 업로드 +
  `scene_sdf_cooked` 플래그, main.rs는 `scene_gdf_baked=cooked`로 `record_scene_bake` 스킵.

검증(RTX 2070 SUPER): run1 `scene SDF 48^3 (Cooked)` → run2 `(CacheHit)`, GPU 베이크 패스 제거.
**CPU-베이크 필드 렌더 vs 변경 전 GPU-베이크 기준선: DX 0.000/ch, VK 0.000/ch(max 1),
DX≡VK 0.000/ch(max 5=기준선)** — GPU→CPU 전환 무회귀, SW-RT 결과 불변. CPU 쿡=결정적이라
재쿡 없이 로드 = 직접 베이크 바이트 동일 + 크로스백엔드 동일이 자명. 12 asset 테스트, clippy/fmt 클린.
- **M2 확장 — C8a albedo 볼륨 쿡 (`2496b44`)**: per-voxel albedo 3볼륨(R/G/B)도 동일하게
  CPU 쿡→캐시→업로드. `sdf.rs` `bake_albedo_from_fused`(SDF와 동일 최근접-삼각형 탐색, 승자
  삼각형 albedo를 3채널에), `.dcasset` `CHUNK_ALBEDO`({dim, 3×dim³ R32F}), cook
  `load_or_bake_scene_albedo`(키=hash(지오메트리+per-triangle albedo+그리드), SDF와 별도 캐시·가산),
  gdf.rs `build_scene_sdf(…, cooked_albedo)`→`create_volume_init` 3볼륨 업로드+`scene_albedo_cooked`,
  main은 `scene_albedo_baked=cooked`로 `record_scene_albedo_bake` 스킵. 검증: run1 SDF+albedo
  `Cooked`→run2 둘 다 `CacheHit`(양 GPU 베이크 패스 제거), CPU-베이크 vs GPU-베이크 baseline
  DX/VK 0.000/ch·DX≡VK 0.000=무회귀. albedo .dcasset=1.33MB(48³×4×3). 22 asset 테스트.
- 미대상(설계대로): 클립맵 동적 부분(런타임 갱신), 디버그용 per-mesh `volume`(렌더 비소비).

### M3 — 텍스처 블록 압축 (BCn) — ✅ 완료 (텍스처 트랙)
요구: 텍스처 압축하되 **런타임 압축 해제 비용 최소**. 답=**GPU 네이티브 BCn**(하드웨어가 4×4 블록
직접 샘플 → 해제 단계 0, 디스크+VRAM 동시 절감, KTX2/DDS가 예고한 경로). 손실이라 **옵트인**
(`P12_TEX_COMPRESS=1`, 기본 off=렌더 바이트 동일).

- **M3.1 (`0df7152`)** — CPU BC 인코더 `crates/asset/src/bc.rs`(무의존·결정적): **BC1**(컬러 8:1,
  bbox 엔드포인트+565+2bit) + **BC5**(노멀 2×BC4, 16B/블록). 디코더(블록 1개→평균색). 테스트:
  RMSE 한계·크기·비4배수 차원.
- **M3.2 (`e9e5898`)** — RHI `Format::{Bc1Srgb,Bc1Unorm,Bc5Unorm}` + VK/DX/Metal 매핑 +
  `Device::create_texture_compressed`(미리 압축된 BCn mip 업로드: VK 블록패킹 자동, DX
  GetCopyableFootprints 블록 row pitch, Metal 블록피치 replaceRegion). create_texture를 공유
  `upload` 헬퍼로 리팩터.
- **M3.3 (`2f77dd8`)** — `TexData{Rgba8|Bc}`가 Material의 `Option<ImageData>` 대체. dcasset
  텍스처 청크에 kind 태그+BC 페이로드. **per-slot 정책**: base_color/emissive→BC1, normal→BC5,
  **metallic_roughness+데이터 텍스처=무압축**(블록압축이 비지각/벡터-선형 값 손상 — 사용자 요구),
  알파 있는 base_color도 무손실 유지. cook 압축은 `generate_mip_chain`(rhi-types, 패리티 단일소스)
  으로 mip 생성 후 BC 인코딩. 플래그는 캐시 키에 포함(토글 재쿡). 샌드박스 upload_texture가
  Bc→create_texture_compressed 라우팅, GI 알베도는 average_linear(BC는 최소 mip 1블록만 디코드).

검증(RTX 2070 SUPER, Avocado): 기본(무압축) 렌더 = 기준선 **0.000/ch**(무회귀). 압축 시
`.dcasset` **50.4MB→25.2MB(-50%**; metallic_roughness 의도적 무압축이라 더 안 줄음), 렌더 델타 vs
기준선 0.591/ch(손실·옵트인), **DX≡VK 0.000/ch**(양 백엔드 동일 블록 업로드). 런타임 해제비용 0.
19 asset 테스트, clippy/fmt 클린.
- **M3 확장 — BC3/BC4/BC7 + 압축 티어 (`e83f307`/`226b033`)**: BC3(알파, BC4+BC1 조합), BC4(단채널),
  **BC7(고품질 RGBA, mode 6)** 인코더 추가 + RHI 포맷(VK/DX/Metal). cook에 **`TexCompress{Off,Fast,High}`
  티어**: Fast=BC1/BC3(크기), High=BC7(품질). 노멀=BC5 고정, 데이터 텍스처=무압축. `P12_TEX_COMPRESS=1|fast|high`.
  **측정(Avocado): Off 50.35MB/0.000, Fast 25.19MB/0.591, High 27.98MB/0.593, BC7 mode-6 GPU 샘플 정확
  (DX≡VK 0.000).** 정직한 발견: 이 매끈한 에셋은 BC7≈BC1(BC7-vs-BC1 렌더 0.008/ch)인데 BC7이 더 큼 → Fast가
  이 에셋엔 유리, BC7은 고주파 컬러용(유닛 테스트가 우위 입증). 측정-구동 선택지로 티어 제공.
- 후속: BC7 멀티모드/파티션(품질↑), 압축 기본화 RenderQuality 결속, 실제 게임 텍스처 VRAM 측정.

### 아이템 2 — `.dclevel` 씬/레벨 청크 ✅ (`8b7259b`, Phase 12 Stage E 기반)
`crates/asset/src/level.rs` `LevelData{entities, lights, camera, environment}` — 엔티티는 쿡된
에셋을 **논리 키로 참조** + 월드 트랜스폼 + 머티리얼 오버라이드, 라이트(directional/point), 카메라, 환경.
GPU 핸들 없음(런타임이 ref 해석). dcasset `CHUNK_LEVEL`(=5) + write_level/read_level(같은 컨테이너,
Writer/Reader에 길이접두 UTF-8 문자열 추가). round-trip+결정성 테스트. **라이브 렌더 구동(데이터-주도
씬 셋업)은 Phase 12** — 여기선 포맷만 확정해 레벨 저작/쿡 가능하게.

### 아이템 3 — 볼륨 readback (GPU→CPU) ✅ (`63855f7`)
M2가 업로드만 추가했던 볼륨 I/O 완성: `Device::read_volume`(create_volume_init의 역). w·h·d·bpp 타이트
바이트. vulkan(host buffer+copy_image_to_buffer, 타이트), d3d12(READBACK heap+256B footprint→행 de-pad),
metal(getBytes). 샌드박스 `P12_VERIFY_VOLUME=1`=업로드 SDF 리드백 후 바이트 비교. **검증: 442368B, 양
백엔드 0 mismatch**(DX row-pitch de-pad 검증). GPU 산출 볼륨을 데이터 레벨에서 쿡/검증 가능.

### M3+ — 컨테이너 확장 (후속, 선택)
- 추가 베이크 페이로드(BVH, 라이트맵, 프로브)를 청크로. Phase 14/11 산출물과 연계.
- 에셋 의존성 그래프 등은 필요 시 설계.
- **씬/레벨 청크 Phase 12 본구현:** 위 `.dclevel` 포맷을 실제 씬그래프에 결속 (Stage E) —
  엔티티(애셋 source-hash 참조)+트랜스폼+머티리얼 오버라이드+라이트/카메라/환경, 월드는 청크 그래프
  (인접·월드 배치·스트리밍 반경)를 직렬화. Phase 12 Stage E가 이 M3을 채운다. 자세히는
  [phase-12-scene-graph.md](phase-12-scene-graph.md) "직렬화 & 차후 애셋화".

## 설계 결정 (M1에서 확정)
- **직렬화:** 수동 little-endian 청크 컨테이너(§포맷). `bincode`/`rkyv` 미사용. zero-copy
  로드는 미도입(현재 디코드는 Vec 복사) — 필요 시 후속.
- **캐시 위치:** gitignored `/cache/dcasset/`(M4 정책과 동일, 로컬 전용·재쿡 가능). 커밋
  배포(`assets/cooked/`)는 M3에서 재논의.
- **cook 트리거:** lazy 첫 실행(`load_cooked`). 오프라인 `tools/` 쿡은 후속 옵션.
- **버전 마이그레이션:** 헤더 `version` 불일치 = 미스 → 재쿡(구 포맷 오독 불가). 청크
  디렉터리라 신규 청크 타입 추가는 포맷 호환(구 reader는 미지 타입 skip).

## 남은 설계 항목 (후속)
- 메시 외 자산(텍스처 압축 KTX2/DDS, 씬)으로의 확장 — M3.
- 다중 프리미티브/멀티 메시(현재 `load_gltf`는 첫 프리미티브만; 포맷은 메시 청크 복수화로
  확장 가능).
- 소스 신선도 빠른 경로(현재 히트도 소스 바이트 해시 위해 전체 read; mtime+size fast-path
  여지) — 측정 후 필요 시.
