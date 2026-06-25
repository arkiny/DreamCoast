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

### M1 — 메시 직렬화 (.dcasset 골격) — Phase 11과 독립, 먼저 가능
- `crates/asset`에 `.dcasset` reader/writer(헤더 + 메시 청크). `serde`+`bincode` 또는 수동 바이너리.
- **cook 경로:** glTF → `MeshData` → `.dcasset` 기록. 오프라인 툴(`tools/`) 또는 첫 실행 시 lazy 쿡.
- **런타임 로드:** `.dcasset` 직접 → `MeshData`(glTF 파싱 생략). 캐시 미스 시 쿡 후 저장.
- 검증: 샌드박스가 `.dcasset` 로드한 메시로 기존과 동일 렌더(양 백엔드 픽셀 일치), startup 가속 확인.

### M2 — SDF 베이크 청크 통합 (Phase 11 Stage B 이후)
- Phase 11 B2의 per-mesh SDF 베이크 결과를 SDF 볼륨 청크로 `.dcasset`에 영속화.
- **로드:** 캐시 히트 → GPU 볼륨 업로드(B1의 3D 텍스처 경로 재활용), 미스 → B2 베이크 후 저장.
- 쿡 대상은 **정적 메시 + SDF**(클립맵 동적 부분은 런타임 갱신이라 비대상).
- 검증: 재쿡 없이 로드한 SDF가 직접 베이크와 바이트 동일, Phase 11 B4 SW RT 결과 불변.

### M3 — 확장 (후속, 선택)
- 추가 베이크 페이로드(BVH, 라이트맵, 프로브)를 청크로. Phase 10/11 산출물과 연계.
- 텍스처(KTX2/DDS) 참조·임베드, 에셋 의존성 그래프 등은 필요 시 설계.

## 미결 / 설계 항목
- 직렬화 라이브러리(`bincode`/`rkyv`/수동) 및 zero-copy 로드 여부.
- 캐시 위치/배포 정책(gitignored vs 커밋), 버전 마이그레이션.
- cook 트리거(오프라인 툴 vs lazy 첫 실행 vs 빌드 스텝).
- 메시 외 자산(텍스처/머티리얼/씬)으로의 확장 범위.
