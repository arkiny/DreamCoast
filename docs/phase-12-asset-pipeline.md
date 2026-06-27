# Phase 12 — 에셋 파이프라인 / 쿠킹된 에셋 (세부 계획 / 스텁)

상위: [ROADMAP.md](ROADMAP.md) Phase 12. **크로스컷팅 엔진 인프라** — 특정 렌더 기법이 아니라
`crates/asset`의 자산 직렬화 계층이다. Phase 11(Distance-Field GI)의 GDF 베이크 영속화가 직접적 동기였으나,
**메시 지오메트리 자체를 포함해** 가공된 자산을 하나의 바이너리로 저장/로드하는 실질적 에셋 개념이다.

> 목표: 매 실행마다 glTF를 재파싱하고 SDF를 재베이크하지 않는다. **가공된 메시 + 베이크 데이터(SDF,
> 향후 BVH/라이트맵 등)를 하나의 직렬화된 `.dcasset`으로 cook → 저장 → 런타임 직접 로드.**

## 동기 / 배경
- 현재 `crates/asset`은 런타임에 `gltf`+`image`로 glTF→`MeshData`를 파싱한다(즉석, 영속화 없음).
- Phase 11 Stage B의 per-mesh SDF 베이크는 비싸다(brute-force 점→삼각형). 한 번 베이크하고 캐시해야 한다.
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

### M1 — 메시 직렬화 (.dcasset 골격) — Phase 11과 독립, 먼저 가능 — ✅ 완료
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

### M2 — SDF 베이크 청크 통합 (Phase 11 Stage B 이후) — ✅ 완료
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
- 미대상(설계대로): 클립맵 동적 부분(런타임 갱신), per-voxel albedo 볼륨(C8a, 별도 GPU 베이크 유지 —
  후속 M2 확장 후보), 디버그용 per-mesh `volume`(렌더 비소비).

### M3 — 확장 (후속, 선택)
- 추가 베이크 페이로드(BVH, 라이트맵, 프로브)를 청크로. Phase 10/11 산출물과 연계.
- 텍스처(KTX2/DDS) 참조·임베드, 에셋 의존성 그래프 등은 필요 시 설계.
- **씬/레벨 청크 (Phase 13 결속):** `.dclevel`/`.dcworld`를 같은 컨테이너에 청크 타입으로 추가 —
  엔티티(애셋 source-hash 참조)+트랜스폼+머티리얼 오버라이드+라이트/카메라/환경, 월드는 청크 그래프
  (인접·월드 배치·스트리밍 반경)를 직렬화. Phase 13 Stage E가 이 M3을 채운다. 자세히는
  [phase-13-scene-graph.md](phase-13-scene-graph.md) "직렬화 & 차후 애셋화".

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
