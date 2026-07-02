# View Family (다중 뷰) — PR-9

상위: [render-pipeline-reference.md](render-pipeline-reference.md) §3 PR-9 · 관련: [depth-prepass.md](depth-prepass.md)
(view-종속 패스의 depth 생산자), [clustered-lighting.md](clustered-lighting.md) (froxel 리스트는 primary-frustum
전용 = view-종속).

이 문서는 렌더 파이프라인 재정합 트랙의 마지막 항목 **PR-9: View Family**를 기술한다. 목표는 "새 기능"이 아니라
**구조 정합** — 한 프레임에 N개의 뷰(스플릿스크린 / 스테레오 / 씬 캡처 / 에디터 다중 뷰포트)를 **공용 씬
리소스**로 렌더할 수 있도록 프레임 루프의 뷰-종속 상태를 **`SceneView` 디스크립터**로 분리한다. 레퍼런스
엔진의 canonical 구조와 동일하다: **view family**가 공용 씬 렌더를 소유하고, 각 **scene view**는 카메라 +
per-view 기능 집합을 자기 타깃(화면 영역 또는 오프스크린 캡처 텍스처)에 렌더한다.

opt-in: 디폴트는 단일 뷰 = 기존 경로/비용/출력 그대로(바이트 동일). `P_SECOND_VIEW=1`이 증명 데모다.

---

## 1. View-dependent vs. view-independent (핵심 분리)

한 프레임을 N뷰로 렌더할 때 **무엇이 뷰마다 반복되고 무엇이 한 번만 도는지**가 설계의 전부다. 아래 분류는
`apps/sandbox/src/main.rs`의 프레임 루프 실제 배선을 그대로 반영한다.

### 뷰-독립 (프레임당 1회 — 공용 씬 리소스)
카메라 뷰가 아니라 **씬 + 태양광**에만 의존하므로 모든 뷰가 같은 결과를 샘플한다:

| 리소스/패스 | 코드 | 왜 뷰-독립 |
|---|---|---|
| Shadow depth (단일 맵 / CSM 아틀라스) | `deferred.record_shadow` / `record_shadow_atlas` | 광원 시점 래스터 — 카메라 무관 |
| IBL 캡처 (env/irradiance/prefilter/BRDF) | `ibl.maybe_capture` | 씬 중심 프로브 (카메라 아님) |
| GDF / surface-cache 베이크·relight | `gdf.*` | 월드공간 필드 |
| Clustered-light 업로드 (라이트 버퍼) | `cluster.upload` | 라이트 리스트 자체는 뷰 무관 (froxel *컬링*만 뷰-종속) |
| Path-tracer accumulation prepare | `rt.prepare` | (primary 뷰 전용 도구) |

> 주의: **froxel 컬링 리스트**(`ClusterSystem::build` compute)는 primary-frustum 기준이므로 엄밀히는
> 뷰-종속이다. 라이트 *버퍼*(업로드)는 공용, *빌드*(컬링)는 primary 전용 — secondary 뷰는 브루트포스
> point-light 경로(`cluster = None`)로 라이팅한다(단순화, 디스크립터의 `screen_space_gi=false`와 일관).

### 뷰-종속 (뷰마다 반복 — `SceneView`로 파라미터화)
카메라 `view_proj`/`inv_view_proj`/`eye`/jitter/globals-slice에 의존:

- Depth pre-pass (`record_prepass`, `DEPTH_PREPASS=1`)
- G-buffer fill (`record_gbuffer`) + deferred decals (`record_decals`) + velocity (`velocity.record`)
- 화면공간 간접광: GDF AO / GTAO / screen-probe·ray-march GI / SSR·reflection composite
- PBR deferred lighting (`record_lighting`) + auto-exposure meter + lit-history 캡처
- 대기/포그 슬롯 (`atmosphere.record_fog`) + 투명 (`translucency.record`)
- Post 체인: motion-blur → TAAU → bloom → DoF → tonemap+grade
- **Per-view temporal 상태**: TAAU 히스토리(`TaauSystem`), velocity prev-매트릭스, SSR/GI 디노이저 히스토리

---

## 2. `SceneView` 디스크립터 (`apps/sandbox/src/view.rs`)

```
SceneView {
    index, eye, focus,
    view_proj, view_proj_stable, view, inv_view_proj, prev_view_proj,  // 카메라 수학
    jitter_uv,                 // per-view TAA 서브픽셀 지터 (secondary = 0)
    globals_offset,            // (fif * MAX_VIEWS + index) * GLOBALS_SLICE
    features: SceneViewFeatures { taau, velocity, post, screen_space_gi },
}
```

핵심은 **단순화가 뷰 구조로 표현된다**는 점이다. secondary 뷰가 TAAU/velocity/post/screen-GI를 끄는 것은
스캐터된 `if second_view`가 아니라 `SceneViewFeatures::secondary()`가 각 플래그를 클리어하기 때문이다. 세
번째 뷰(예: 실시간 env 캡처 프로브)를 추가하려면 그 뷰의 feature set을 고르면 된다.

- `SceneViewFeatures::full()` = 완전 기능 = 레거시 단일-뷰 경로(바이트 동일 앵커).
- `SceneViewFeatures::secondary()` = TAAU·velocity·post·screen-GI **off** — 값싼 인셋용이자, **primary
  뷰의 per-view temporal 상태와 절대 충돌하지 않음**을 구조로 보장(secondary는 자기 히스토리를 안 가지므로
  temporal 기능을 끈다 = 뷰 수 안전).

### Globals UBO의 per-view 슬라이스
globals 버퍼는 `GLOBALS_SLICE * FRAMES_IN_FLIGHT * MAX_VIEWS`로 확장(현 per-fif 슬라이스 위에 per-view
오프셋). 뷰의 globals 오프셋 = `(fif * MAX_VIEWS + view_index) * GLOBALS_SLICE`. secondary 슬라이스는
primary `Globals`를 복사한 뒤 **카메라 종속 필드만 오버라이드**(camera_pos / inv_view_proj / prev_view_proj
/ cluster_view_z_row) — 태양·shadow·IBL·CSM은 뷰-독립이므로 single source로 공유한다.

`MAX_VIEWS`(현재 2)를 늘리면(그리고 그것만) 더 많은 동시 뷰가 가능하다. 디폴트 단일-뷰 경로는 각 프레임의
슬라이스 0만 건드리므로 앵커가 바이트 동일이다.

---

## 3. 증명 데모 — `P_SECOND_VIEW=1`

메인 뷰(오빗) + 두 번째 뷰(상공 오버헤드 카메라, 씬 중심을 수직으로 내려다봄)를 **같은 프레임**에 렌더해
백버퍼 우상단에 **PiP(인셋 사분면)** 로 합성한다.

배선(`main.rs` 프레임 루프, 메인 tonemap 직후·UI 직전):
1. `view::SceneView::build(1, overhead_eye, center, …, secondary())` — 두 번째 뷰 디스크립터.
2. secondary globals 슬라이스 write(카메라 필드만 오버라이드).
3. 인셋 해상도(`min(sw,sh)/3` 정사각) 전용 G-buffer + depth + HDR 생성.
4. **뷰-종속 체인만** 재실행: `record_gbuffer`(pre-pass·jitter 없음) → `record_lighting`(공용 shadow
   아틀라스 + IBL 샘플, screen-GI 입력 없음, 브루트포스 point) .
5. `record_tonemap_inset` — 백버퍼를 **LOAD**(클리어 안 함)하고 `set_viewport_scissor_rect(inset)`으로
   풀스크린 삼각형을 인셋 사분면에만 그림(메인 뷰 위에 오버레이).

두 번째 뷰가 소비하는 shadow 아틀라스·IBL·GDF는 위에서 **한 번만** 렌더된 것을 그대로 샘플 — 이것이 §3
PR-9의 "공용 리소스로 N뷰 렌더"의 증명이다.

---

## 4. 검증

1. clippy `-D warnings` 클린 + fmt.
2. **디폴트 골든 앵커 바이트 동일**: `P_SECOND_VIEW` off → 두 번째 뷰 패스 자체가 그래프에 없음 +
   primary globals 슬라이스 내용 불변(오프셋 stride만 확장) → sha256 `af70c1a5…` 유지.
3. `P_SECOND_VIEW=1` 갤러리: 우상단 인셋에 상공 시점(다른 카메라 각도)이 합성됨.
4. sponza(`LEVEL=sponza_intel`)에서도 동작.
5. `PROFILE_GPU=1`: shadow/IBL/GDF 등 뷰-독립 패스가 **한 번만** 스케줄됨(패스 리스트에 `sv_*` 접두
   패스만 추가되고 `shadow`/`csm`은 1회) — 두 번째 뷰의 추가 비용은 인셋 해상도의 gbuffer+lighting+tonemap뿐.

(수치는 커밋 메시지 / 최종 보고 참조. Metal 검증; DX≡VK Windows 후속.)

---

## 5. 남은 것 (골격 밖 — 후속)

- **완전 파라미터화**: 현재 primary 뷰의 뷰-종속 패스들은 여전히 개별 로컬(`view_proj` 등)을 직접 읽는다
  (바이트 동일 baseline 보존을 위해). `SceneView`는 primary에도 구축되어 그 수학을 **문서화·미러링**하지만,
  primary 경로 전체를 디스크립터-구동 함수로 접는 것은 앵커 리스크가 커서 후속으로 남긴다.
- **secondary 뷰의 화면공간 GI / post / TAAU**: 구조상 `features` 플래그로 켤 자리는 있으나, 각 뷰가 자기
  히스토리 버퍼 세트를 소유해야 한다(현재 `TaauSystem`/디노이저는 1세트) → 뷰별 히스토리 링 확장이 선결.
- **스테레오 단일-패스 / multiview 인스턴싱**(하드웨어 뷰 인덱스로 1 draw = N 뷰)는 별개 최적화. 현 골격은
  뷰당 별도 패스(정직한 N× draw)로, 구조를 먼저 세우고 인스턴싱은 후속.
- **오프스크린 캡처 소비자**: 실시간 env 프로브(`ibl.rs`의 미래 소비자)·에디터 뷰포트는 인셋 대신 텍스처
  타깃으로 `SceneView`를 렌더하면 된다(합성 대신 캡처) — 동일 디스크립터, 다른 출력.
