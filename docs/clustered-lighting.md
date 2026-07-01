# Clustered Light Culling (PR-6)

상위: [render-pipeline-reference.md](render-pipeline-reference.md) §1.4 · §2 표 #8 · §3 PR-6.

디퍼드 라이팅의 다광원 확장을 위한 **클러스터드 라이트 컬링 인프라**. 뷰 절두체를 3D
froxel(클러스터) 그리드로 나눠 compute 패스로 per-cluster 라이트 리스트를 빌드하고,
`record_lighting`(PBR 풀스크린 패스)이 각 픽셀의 클러스터 리스트만 순회해 point 라이트를
셰이딩한다. 단일 디렉셔널(sun)은 기존 특수 경로를 그대로 유지한다.

- **opt-in seam:** `CLUSTERED_LIGHTS=1` (디폴트 off = 기존 브루트포스 `globals.point_pos[]`
  루프, 바이트 동일).
- **A/B baseline:** `CLUSTERED_BRUTE=1` (같은 라이트 버퍼를 올리되 셰이더가 전 라이트를 루프 —
  froxel 리스트 없이 — 클러스터드와 동일 라이트 셋에서 GPU 시간 비교용).
- **스케일 스포너:** `TEST_LIGHTS=N` (고정 그리드/고정 팔레트/무애니메이션 결정론 배치).
- **디버그 뷰:** `DEBUG_VIEW=11` (per-pixel 클러스터 라이트-카운트 히트맵, 파랑→초록→빨강).

---

## 1. 설계 리서치 — froxel 3D 클러스터 vs tiled + Z-binning

두 canonical 접근을 조사했다(출처 §5):

### (A) 3D froxel 클러스터 (aortiz / 3dgep, "Clustered Shading")
- 그리드 `X×Y×Z` (예: 16×9×24). 각 클러스터(froxel)가 자기 라이트 인덱스 리스트를 소유.
- **Z 슬라이싱은 exponential(log):** `Z(slice) = near·(far/near)^(slice/numZ)`. self-similar
  슬라이스가 원근 비선형성을 상쇄해 근거리(라이트가 중요한 곳) 클러스터가 얇다.
- 라이트 그리드(offset+count) + 글로벌 인덱스 리스트. 메모리 `O(X·Y·Z)`.

### (B) tiled + Z-binning (DOOM 2016 / Detroit / Granite)
- **XY와 Z를 분리:** XY 2D 타일당 라이트 비트마스크 `u32[ceil(N/32)]` + 뷰-Z로 정렬된
  1D Z-bin(각 bin이 min/max 라이트 인덱스). 셰이드 시 두 마스크를 AND. 메모리 `O(X·Y + Z)`.
- 최신 엔진이 채택하는 우월한 방식: 수천 라이트로 스케일, 메모리가 froxel보다 훨씬 작다.
- **약점:** 큰 라이트가 Z-range를 지배해 false-positive(over-shading) 유발 가능.

### 채택 결정: **(A) froxel 3D 클러스터, 라이트를 글로벌 인덱스 오름차순으로 binning**

이 엔진의 **하드 검증 게이트는 브루트포스와의 바이트 동일**(파이프라인 재정합 트랙의 무회귀
규칙)이다. (B) Z-binning은 라이트를 **뷰-Z로 정렬**하므로 per-pixel 라이트 누적 순서가
브루트포스 루프와 달라진다 → 바이트 동일 불가(부동소수 누적 순서 의존). (A) froxel은 라이트를
**원본 배열 인덱스 순서**로 리스트에 넣고 셰이더가 같은 순서로 읽으므로, 소수 라이트(전 라이트가
클러스터에 포함될 때) 누적이 브루트포스와 **정확히 일치**한다.

froxel의 `O(X·Y·Z)` 메모리는 현 스케일(16×9×24 = 3456 클러스터, MAX 128 lights/cluster →
인덱스 리스트 1.7 MB, 그리드 14 KB)에서 비이슈다. 따라서 파리티가 더 어려운 게이트인 지금은
froxel을 택하고, **Z-binning은 문서화된 스케일 진화 경로**로 남긴다(수천 라이트 요구 시
`light_cluster.slang`의 build 패스를 XY 비트마스크 + Z-bin으로 교체, 셰이더 read 경로만 수정,
파리티 게이트를 ≤0.001/ch로 완화). 클러스터 치수는 단일 소스 상수(`CLUSTER_X/Y/Z`)라 상위
RenderQuality 티어로 스왑 가능하다.

---

## 2. 구현

### 셰이더 (단일 소스 `crates/shader/shaders/`)
- `light_cluster_common.slang` — froxel 그리드 상수(`CLUSTER_X/Y/Z`, `MAX_LIGHTS_PER_CLUSTER`),
  packed `Light` 레이아웃(2×float4 = 32 B: pos+radius, color+intensity), `cluster_index_for()`
  (픽셀 UV + 양수 선형 뷰깊이 → 클러스터 인덱스, exponential Z의 역함수). **producer/consumer
  단일 소스** — build 패스와 pbr 패스가 같은 헤더를 include.
- `light_cluster.slang` `csBuildClusters` — 스레드 1개당 클러스터 1개: 스크린 타일 + exponential
  Z 슬라이스로 클러스터의 **월드공간 AABB**를 8코너로 구성 → 전 라이트를 sphere-vs-AABB로
  컬링 → 생존자를 **글로벌 인덱스 오름차순**으로 flat 인덱스 리스트의 클러스터 슬롯에 append
  (per-cluster count는 병렬 `grid` u32 배열; alloc-free, atomic 불필요).
- `pbr.slang` — point-light 루프를 분기: (a) 클러스터 버퍼 바인딩 시 픽셀 클러스터 리스트만
  순회(같은 순서 → 바이트 동일), (b) `cluster_index_buf==MAX`면 전 라이트 브루트포스(A/B),
  (c) 미바인딩이면 기존 `globals.point_pos[]` 루프(디폴트 앵커). `DEBUG_VIEW=11` 히트맵 추가.

### Rust
- `apps/sandbox/src/cluster.rs` `ClusterSystem` — build compute 파이프라인 + per-fif host-write
  라이트 버퍼(라이트 수 초과 시 2배 재할당) + device-local grid/index UAV 버퍼. `upload()`(프레임
  라이트 → 버퍼, 그래프 빌드 전 호출) + `record_build()`(compute 패스). `bindless-first`: 라이트
  데이터는 storage buffer.
- `main.rs` — `CLUSTER_*` 단일-소스 상수, 카메라 near/far(`CLUSTER_Z_NEAR/FAR`)를 perspective와
  froxel Z 슬라이싱에 공유, `Globals`에 `cluster_view_z_row`(월드→뷰 row2, 양수 선형 뷰깊이 복원)
  + `cluster_params`(near/far) 추가. `GLOBALS_SLICE` 512→768(256 정렬 유지). `TEST_LIGHTS`
  결정론 그리드 스포너 `test_light_grid()`.

### 파리티 설계 (바이트 동일의 핵심)
씬 authored 라이트(gallery 2개)에는 **사실상 무한 radius**(`CLUSTER_Z_FAR*4`)를 줘서 모든
클러스터가 그 라이트를 bin → 셰이더가 브루트포스와 **같은 라이트를 같은 순서**로 누적(브루트포스는
거리 컷오프 없음). `TEST_LIGHTS` 스트레스 그리드는 유한 radius(컬링이 실제로 payoff나는 곳).

---

## 3. 검증 수치 (Metal, macOS M3, 2560×1440)

| 게이트 | 결과 |
|---|---|
| clippy `-D warnings` + fmt | 클린 |
| 디폴트 OFF 골든 앵커 sha256 | `af70c1a5…8b2b74` == 기대값 (바이트 동일) |
| `CLUSTERED_LIGHTS=1` vs OFF (gallery) | **sha256 동일 = 바이트 동일** (목표 상회; ≤0.001/ch 불필요) |

### 스케일 (PROFILE_GPU, 라이팅 패스 GPU ms)

| 라이트 수 | 브루트포스(`CLUSTERED_BRUTE`) | 클러스터드 | 클러스터 빌드 | speedup |
|---|---|---|---|---|
| 256 | 37.4 ms | 3.67 ms | 0.06 ms | **~10×** |
| 1024 | 170.0 ms | 8.74 ms | 0.24 ms | **~19×** |

클러스터-빌드 compute는 무시 가능(0.06–0.24 ms). 라이트 수가 늘수록 speedup이 커진다(브루트포스는
픽셀×라이트 선형, 클러스터드는 픽셀×클러스터당-라이트).

**백엔드:** Metal만 검증(macOS M3). **DX/VK parity pending Windows verification** — 셰이더는
3 백엔드 컴파일(Metal `NonUniformResourceIndex` 미사용, storage buffer는 스칼라 인덱스), 크로스
백엔드 레이아웃은 기존 `Globals`/push 컨벤션(std140/cbuffer 정합, Y-flip은 월드공간 매칭으로 상쇄)을
따른다.

---

## 4. 남은 리스크 / 후속

- **DX≡VK 미검증** (Windows 박스 필요). froxel AABB를 월드공간에서 구성하고 셰이드도 월드
  포지션으로 매칭하므로 clip-Y flip은 상쇄되도록 설계했으나 Windows에서 0.000/ch 확인 필요.
- **spot/area 라이트 미지원** — 현재 point만. Light 레코드에 방향/cone 추가 + AABB 컬을 cone으로
  확장하면 됨(PR-7 그림자 아틀라스와 함께 Phase 21).
- **radius 유한 컷오프** — authored 라이트는 무한 radius(파리티). 실게임에선 authored range를
  Light.radius로 넘겨 실제 컬링 활성화(레벨 스키마에 range 필드 추가 시).
- **스케일 진화** — 수천 라이트 요구 시 §1 (B) tiled + Z-binning으로 교체(파리티 게이트 ≤0.001/ch
  완화). build 패스 + pbr read 경로만 수정, seam/버퍼/스포너는 재사용.

---

## 5. 출처

- A Primer On Efficient Rendering Algorithms & Clustered Shading — https://www.aortiz.me/2018/12/21/CG.html
- Clustered shading evolution in Granite — https://themaister.net/blog/2020/01/10/clustered-shading-evolution-in-granite/
- Volume Tiled Forward Shading (3dgep) — https://www.3dgep.com/volume-tiled-forward-shading/
- Thoughts on light culling for clustered shading (Sylvan) — https://www.sebastiansylvan.com/post/light_culling/
