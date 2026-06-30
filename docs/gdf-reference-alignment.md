# GDF 아키텍처 — 레퍼런스 대조 + 코드 기반 격차/개선 로드맵

> 출처: 워크스페이스에 있는 상용 레퍼런스 엔진의 global distance field + 소프트웨어 레이트레이싱
> 소스를 직접 대조(2026-06-30). 이 문서는 그 대조를 **우리 코드의 구체 변경 지점**에 매핑하고 순서를
> 잡는다. 상위/형제: [per-mesh-distance-fields.md](per-mesh-distance-fields.md)(per-mesh DF 트랙,
> S0/S1 구현됨), [scalable-gi.md](scalable-gi.md),
> [commercial-engine-gap-analysis.md](commercial-engine-gap-analysis.md).
> 메모리: `dynamic-gdf-deferred`, `gdf-streaming-future`.

## 한 줄 요약
**소비측(거리장 march로 AO/GI/반사 + GTAO×DFAO 레이어링)은 레퍼런스와 정렬됐다. 격차는 표현/업데이트
측** — 레퍼런스의 *sparse · per-mesh-composite · static/dynamic 클립맵* vs 우리의 *dense · fused ·
정적 48³*.

## 레퍼런스 방식 (소스 대조, 확정 사실)
- **구조**: 4 클립맵 캐스케이드, 카메라 중심, 거리지수 2(각 2배).
- **해상도**: 클립맵당 128³이나 **sparse** — page table(uint 인디렉션) → page atlas, **8³ 페이지**,
  **~25% occupancy만 할당**.
- **합성**: per-object mesh SDF(메시별 거리장 아틀라스, 스트리밍)를 레벨 AABB로 컬 → composite. 작은
  메시는 반경 임계로 컬링.
- **업데이트**: partial(dirty 영역만) + staggered(프레임당 2 클립맵) + **static/dynamic 분리**(정적
  별도 레이어 캐시, 움직이는 것만 매 프레임 재합성) → 완전 동적.
- **소비**: cone-trace(far-field GI/reflection) + 거리장 AO + 거리장 소프트섀도우.

## 코드 기반 격차 표 (우리 코드의 구체 지점)

| 레퍼런스 기법 | 우리 현재 코드 | 격차 | 개선 지점(파일:심볼) |
|---|---|---|---|
| per-object mesh SDF composite | **이미 GPU 머지 머신 존재**: `gdf.rs` `bake_pipeline`(`sdf_bake.slang`)·`merge_pipeline`(`gdf_merge.slang`)·`instances` 테이블 = 갤러리 B2/B3. **콘텐츠는 이를 우회**하고 fused brute-force cook(`main.rs build_scene_gdf`). S1 CPU 합성(`compose.rs`)은 그 사이 정적 경로 | 콘텐츠가 per-object composite를 안 씀(통짜 fused) | `main.rs:1260` build_scene_gdf, `compose.rs`(정적), `gdf.rs:1578` record_bake/merge(동적 재활용 대상) |
| 작은 메시 반경 컬 | **없음** — 메시 전부 합성/베이크 | 작은 디테일까지 GDF에 → 베이크 비용 | `compose.rs compose_sdf_level` 진입 컬, `fuse.rs` |
| sparse page table | 클립맵 레벨 = **dense** `Volume` R32F dim³(`gdf.rs:126 ClipLevel{sdf:Volume}`), `clipmap.slang`이 dense 샘플 | dense라 128³×4 불가(메모리). 48³ 고정 | `gdf.rs ClipLevel`/볼륨 생성, `clipmap.slang cm_geo_*/cm_albedo`, RHI 3D-uint 볼륨+atlas |
| static/dynamic 분리 | `build_scene_gdf` **로드 시 1회**(`main.rs:1260`, cooked 가드). 동적 GDF 갱신 보류 | 무엇 하나 움직이면 전체 재베이크 → 정적 전용 | `main.rs` 빌드 분기, 신규 per-frame composite 패스, mobility 분류(`scene`/`registry`) |
| partial + staggered 업데이트 | 없음(정적). 대신 **라이팅**(surface cache relight)만 amortize(`cache_relight_period`) | SDF 자체는 정적 | static/dynamic과 함께 |
| 소비(cone-trace AO/GI/refl) | `gi.rs`/`reflect.rs`/`gdf_ao` SW march + `gdf_cone_k` LOD + GTAO×DFAO(`gtao.slang`) | **정렬됨**(철학 동일) | — (유지) |
| 색(표면 라디언스) | mesh-card surface cache(`cards`/`cache_*`) 존재하나 캡처가 **coarse albedo 볼륨** 읽음 | 얇은 천 색 소실(원 과제) | 카드 캡처(C): `sdf_cache_capture.slang` 메시 삼각형 albedo+opacity |

## 개선 로드맵 (우선순위 = 비용 대비 효과 + 의존성)

이미 구현: **S0** per-mesh DF 베이크+캐시(`crates/asset`), **S1** CPU 합성기(`compose.rs`, opt-in
`P11_PERMESH_GDF`), **G1** 작은-메시 반경 컬(`P11_GDF_MIN_RADIUS`), **clay GDF 디버그 뷰**
(`P11_SCENE_GDF`, `GDF_VIEW_GAIN`). 아래는 그 위에서 레퍼런스 모델로 수렴.

### per-mesh 트레이스 버그 (G2 선결) — **clay 뷰가 잡아냄**
per-mesh 합성(`P11_PERMESH_GDF`)은 카메라 primary 트레이스에서 **degenerate**(flat clay = 카메라가
어디서나 "내부"). GI 바운스(짧은 레이)는 견뎠지만 카메라 레이가 드러냄. 원인 후보 = 비일관 노멀
메시(doubleSided 커튼 등)의 per-mesh DF 음수 영역이 합성 `min`을 오염 → 열린 공간이 "내부".
**정합성 선결**: watertight 처리 / 부호 산정 견고화 / 합성 시 음수 영역 클램프. `compose.rs`에서 수정.

### G2 — composite를 GPU로 (기존 `gdf_merge.slang` 부활/확장) — **동적의 전제**
콘텐츠 클립맵 레벨을 **GPU 머지**(`merge_pipeline` + `instances`)로 합성: per-mesh SDF 아틀라스 →
레벨 AABB로 오브젝트 컬 → composite 컴퓨트. CPU `compose.rs`는 정적 cook 폴백으로 잔존. 이게
partial/staggered/동적의 토대.
검증: 합성 결과가 S1 CPU 합성과 일치(±FP), 갤러리 무회귀, **DX≡VK**(머지 셰이더 결정성).

### G3 — static/dynamic 분리 캐시 — **동적 씬 핵심**
레벨당 **2 레이어**: static(정적 오브젝트 1회 합성·캐시) + full(static 위에 동적만 매 프레임 dirty
영역 재합성). mobility는 `dreamcoast-scene`에 분류 추가. dirty = 움직인 오브젝트 영향 AABB. (현재
동적 오브젝트는 애니메이티드 메시 정도 — 우선 그것부터.)
검증: 정적 씬 = static 캐시만(추가비용 0), 오브젝트 1개 이동 시 그 주변만 갱신(비용 측정), GI 반응.

### G4 — sparse page table — **고해상 클립맵 메모리 효율**
dense `ClipLevel.sdf: Volume` → **page table(인디렉션 3D-uint 볼륨, 8³ 페이지) + page atlas(R32F)**.
near-surface 페이지만 할당(~25%). `clipmap.slang`의 `cm_geo_*/cm_albedo`를 page-table 룩업→atlas
샘플로 교체. RHI에 3D-uint 볼륨/atlas 지원 추가. 이걸로 128³×4를 dense 48³ 메모리에 수용.
검증: 동일 씬 page-atlas vs dense 거리장 일치(±FP), 메모리 사용량 측정(목표 ~25%), DX≡VK.

### C — 색: mesh-card 캡처 — **원 과제(커튼 색)**
`sdf_cache_capture.slang`가 coarse albedo 볼륨 대신 **드로어블 자기 메시 삼각형 albedo**(메시당 수백
tri 최근접 = 저렴) + **카드 opacity** 캡처. GI/반사는 기존대로 카드 샘플. 추가 볼륨 베이크 없음.
검증: `bleed.py` 커튼 색 번짐 복원(빨강 옆 바닥 R−B ↑), 얇은 천/잎/펜스 일반화, DX≡VK.

## 디버그 뷰 — clay GDF 시각화 (구현됨)
`gdf_trace.slang`에 **clay 뷰**(mode bit1): 카메라 레이로 씬 클립맵을 sphere-march → gradient-normal
N·L를 **중립 clay 단색**(가짜 표면색·하드섀도우 없음)으로 셰이딩 → 필드가 실제로 해상하는 지오메트리를
읽는다. `P11_SCENE_GDF=1`로 활성, `GDF_VIEW_GAIN`(기본 0.05)이 씬 HDR 태양을 공유 톤맵에서 mid-grey로.
**검증(Metal, sponza_intel)**: fused GDF가 깔끔한 clay(기둥·아치·blobby 커튼 덩어리·바닥; 열린
nave=검정)로 읽힘. **이 뷰가 per-mesh 트레이스 버그를 즉시 잡아냄**(위 G2 선결).

## 의존성 / 순서
```
S0,S1,G1,clay뷰(완료) → per-mesh 트레이스 버그 → G2(GPU composite) → G3(static/dynamic, G2 위)
                                                → G4(sparse, 독립적이나 G2 후 권장)
                        C(mesh-card 색, G1~G4와 병렬·독립)
```

## 잘 맞는 부분(유지) — 회귀 주의
- 소비측 SW march(AO/GI/반사) + `gdf_cone_k` cone-LOD + **GTAO(근거리)×DFAO(원거리) 레이어링**
  (`gtao.slang`)은 레퍼런스와 같은 구조 → 표현/업데이트 리팩터 중 **소비 인터페이스 불변** 유지.
- 정적 씬 바이트 동일(갤러리) + DX≡VK ≤0.001 게이트는 전 스테이지 공통.

## 한계 / 정직
- 이번 트랙은 **정적 레벨 우선**; 완전 동적(G3)·스트리밍은 메모리 `gdf-streaming-future` 트랙과 합류.
- 비인스턴스 에셋(448 유니크)에서 per-mesh는 첫 쿡이 fused보다 비싸다(캐시 후 무관). 이점은
  인스턴싱/동적/부분 업데이트에서 나옴 — G1(컬)·G2(GPU)·캐시로 완화.
- DX≡VK는 Windows(현재 동결) 게이트; macOS는 Metal 검증.
