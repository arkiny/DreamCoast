# Clustered Light Culling (PR-6)

상위: [render-pipeline-reference.md](render-pipeline-reference.md) §1.4 · §2 표 #8 · §3 PR-6.

디퍼드 라이팅의 다광원 확장을 위한 **클러스터드 라이트 컬링 인프라**. 뷰 절두체를 3D
froxel(클러스터) 그리드로 나눠 compute 패스로 per-cluster 라이트 리스트를 빌드하고,
`record_lighting`(PBR 풀스크린 패스)이 각 픽셀의 클러스터 리스트만 순회해 point 라이트를
셰이딩한다. 단일 디렉셔널(sun)은 기존 특수 경로를 그대로 유지한다.

- **seam (R1 이후 기본 자동):** `CLUSTERED_LIGHTS` 미설정 = **자동** — 프레임의 point 라이트가
  `Globals` UBO의 4슬롯을 넘을 때만 froxel 경로가 켜진다. 4개 이하 씬(골든 전 config)은 기존
  브루트포스 `globals.point_pos[]` 루프를 **코드 경로째 그대로** 타므로 바이트 동일이 정의상 보장.
  `=1` 강제 on(등가성 A/B 도구), `=0` 강제 off(폴백 seam — 4개 초과분은 드롭 + 경고).
- **A/B baseline:** `CLUSTERED_BRUTE=1` (같은 라이트 버퍼를 올리되 셰이더가 전 라이트를 루프 —
  froxel 리스트 없이 — 클러스터드와 동일 라이트 셋에서 GPU 시간 비교용). **자동 판정보다 우선**
  (안 그러면 측정 대상이 스스로 클러스터드로 바뀐다).
- **스케일 스포너:** `TEST_LIGHTS=N` (고정 그리드/고정 팔레트/무애니메이션 결정론 배치).
- **디버그 뷰:** `DEBUG_VIEW=11` (per-pixel 클러스터 라이트-카운트 히트맵, 파랑→초록→빨강).
- **하드 캡:** 프레임당 point 라이트 `MAX_SCENE_LIGHTS = 256`(main.rs). 초과 시 카메라에서 **먼
  것부터 드롭** + 1회 경고, 생존자는 **authored 순서로 재정렬**(누적 순서가 카메라 위치에
  의존하면 FP 비결합성 때문에 카메라가 움직일 때 깜빡인다).

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
세 소비자(UBO 브루트포스 / froxel / storage 브루트포스)가 **같은 감쇠 함수**
`point_attenuation(dist, radius)`(`light_cluster_common.slang`, 단일 소스)를 호출한다. 따라서
라이트를 경로 사이로 옮겨도 셰이딩이 불변이다.

- `radius <= 0` = **컷오프 없음** = `1/d²` 그대로 → 레인지 필드 이전 표현식과 비트 동일
  (authored 기본값이므로 골든 앵커가 곧 과거 이미지).
- `radius > 0` = 같은 역제곱에 `(1 - (d/r)^4)^2` 윈도우 → `d = r`에서 **정확히 0**에 매끄럽게
  도달(컷오프 엣지 없음).

froxel 컬링이 근사가 아닌 이유가 여기 있다: sphere-vs-AABB에 걸러진 클러스터의 모든 픽셀은
`d > r`이라 기여가 **정확히 +0.0**이다. 컷오프가 없는 라이트는 모든 클러스터에 bin되므로
브루트포스와 같은 라이트를 같은 순서로 누적한다. 어느 쪽이든 결과는 비트 동일 — 실측으로
확인(§3).

**레인지는 authored 데이터**(`asset::level::Light::range`, dcasset v11 트레일링 블록,
serde default = 0). 게임 라이트는 실제 range를 줘야 컬링이 payoff한다 — §3의 no-cutoff 행이
그 대가다.

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

### R1 (게임 라이트 편입) 실측 — 던전 횃불 스케일

24 횃불 테스트 씬(ground + 24 기둥 + 구 2개, `tmp/r1/torches24.level`), 2560×1440, PROFILE_GPU,
config를 인터리브해 5회 반복·프레임 955개의 **최소값**(측정 중 머신 부하가 있어 median은 드리프트;
min이 진짜 비용의 추정치로 안정적).

| config | lighting | cluster build | 합계 | Δ vs 4-light |
|---|---|---|---|---|
| 4 라이트 · 레거시 UBO (기준) | 1.2369 ms | — | 1.2369 ms | — |
| **24 횃불 · range 8 · 클러스터드** | 1.6841 ms | 0.0104 ms | **1.6945 ms** | **+0.458 ms** |
| 24 횃불 · range 8 · `CLUSTERED_BRUTE` | 3.6210 ms | — | 3.6210 ms | +2.384 ms |
| 24 횃불 · **range 0(컷오프 없음)** · 클러스터드 | 4.8549 ms | 0.0142 ms | 4.8691 ms | +3.632 ms |

- 24 횃불의 실비용 **+0.46 ms** — 예산(~1 ms) 이내.
- 컬링 payoff **2.14×**(1.69 vs 3.62).
- **range를 안 주면 브루트포스보다 나쁘다**(4.87 vs 3.62): 컷오프 없는 라이트는 모든 froxel에
  bin되므로 컬링 이득은 0인데 grid/index 간접 로드만 라이트당 추가된다. 게임 라이트에 range를
  authored로 주는 것이 성능 계약.

| 등가성 게이트 | 결과 |
|---|---|
| 4 라이트: 레거시 UBO vs `CLUSTERED_LIGHTS=1` vs `CLUSTERED_BRUTE=1` | **3자 sha256 동일** (rt-compare 0.000 avg/ch, max 0) |
| 24 횃불: 클러스터드 vs `CLUSTERED_BRUTE` (앵글 2종) | **sha256 동일** — froxel 경계 아티팩트 0 |
| gallery 앵커 (`2fb9c207…`) | 불변 |

**백엔드:** Metal 검증(macOS M3) 후 **Windows 배치(2026-08-01, RTX 2070 SUPER)에서 DX≡VK
검증 완료** — 단, 아래 §4 첫 항목의 우려가 실제 버그로 확인되어 수정 후 통과했다.

### Windows DX≡VK 배치 실측 (수정 후)

| 게이트 | 결과 |
|---|---|
| 24 테스트라이트: VK 클러스터드 vs 브루트 | avg 0.000000 / max 1 (= VK 런투런 플로어) |
| 24 테스트라이트: DX 클러스터드 vs 브루트 | avg 0.000156 / max 4 (= DX 런투런 플로어) |
| grid/index 버퍼 덤프 | **DX≡VK 바이트-동일** + CPU 기하 재구성과 일치 |
| 던전(17횃불) 클러스터드 vs 브루트 (VK) | 1.614 → 0.293 (씬 자체의 wall-clock 플로어 ~0.27) |
| `DEBUG_VIEW=11` 히트맵 DX≡VK (던전) | 18.59 → 0.29 |
| 갤러리 앵커 (양 백엔드) | 수정 전후 런투런 플로어 이내 = 불변 |

---

## 4. 남은 리스크 / 후속

- ~~**DX≡VK 미검증**~~ — **해소(Windows 배치 2026-08-01): 우려가 실제 버그였다.**
  "clip-Y flip은 상쇄" 설계는 빌드 패스가 **무플립 행렬**을 받을 때만 성립하는데, 호스트가
  VK Y-플립이 구워진 `inv_view_proj`를 push로 넘겨 **VK의 froxel AABB 전부가 세로 미러**
  — 각 froxel이 미러 타일의 라이트 리스트를 담았다(라이트장이 세로 비대칭인 던전에서
  클러스터드 vs 브루트 1.61 avg/ch, 準대칭 갤러리에선 0.004로만 발현해 잠복). 수정 =
  클러스터 빌드 전용 플립-프리 지터 트윈 `proj_cluster`(D3D 방향 지터 포함)를 전달;
  DX/Metal은 입력이 비트-동일이라 앵커 원천 중립. `light_cluster.slang`의
  `screen_to_world_dir`에 행렬 계약(플립-프리 필수)을 주석으로 명문화했다.
- **spot/area 라이트 미지원** — 현재 point만. Light 레코드에 방향/cone 추가 + AABB 컬을 cone으로
  확장하면 됨(PR-7 그림자 아틀라스와 함께 Phase 21).
- ~~**radius 유한 컷오프**~~ — **해소(R1)**: `asset::level::Light::range`가 authored 필드로
  들어왔고 `point_attenuation`이 윈도우드 역제곱으로 소비한다. `range = 0`은 컷오프 없음(레거시).
- **point 라이트 그림자 없음 (v1 수용)** — 횃불은 **그림자를 던지지 않는다**. 그림자 캐스터는
  여전히 태양(디렉셔널) 하나뿐이다. point 그림자는 큐브맵/아틀라스 + 라이트당 6면 렌더가 필요해
  별도 페이즈(PR-7 그림자 아틀라스 연장) 몫. 실제로 보이는 결과: 벽 뒤 횃불이 벽을 통과해 샌다.
  던전은 횃불이 벽에 붙어 있어 대체로 감춰지지만, 얇은 칸막이에서는 드러난다.
- **clustered 라이트가 GI/서피스 캐시에 기여하지 않음 (v1 수용)** — GDF GI·서피스 캐시·반사
  릴라이트는 태양 + 스카이만 본다. 횃불 빛은 **바운스하지 않는다**(직접광 전용). 프로듀서 측
  (`sdf_cache_light.slang` 등)에 라이트 버퍼를 물리는 것이 후속 작업이며, 그때는 릴라이트 비용이
  라이트 수에 비례한다는 점을 예산에 넣어야 한다.
- **`CLUSTER_Z_FAR = 100 m` 하드 far** — froxel Z 슬라이싱과 씬 프로젝션이 공유하는 상수라
  레벨의 `camera.zfar`가 무시된다. 100 m 밖 표면은 마지막 슬라이스로 클램프되므로 그 거리의
  라이트는 정확하지 않다. 던전(≤80 m) 범위 밖 이슈이나, 넓은 야외 레벨 전에 레벨-구동으로
  풀어야 한다(M1 리포트에도 기록됨).
- **스케일 진화** — 수천 라이트 요구 시 §1 (B) tiled + Z-binning으로 교체(파리티 게이트 ≤0.001/ch
  완화). build 패스 + pbr read 경로만 수정, seam/버퍼/스포너는 재사용. 현 하드 캡 256에서는
  froxel로 충분.
- **PT(패스 트레이서)에 point 라이트 없음** — `rt_path.slang`은 태양 + 스카이만 샘플한다. 따라서
  다광원 씬은 PT 레퍼런스가 존재하지 않으며, PT 게이트(sponza_*)는 point 라이트가 4개 이하라
  영향받지 않는다. 횃불 조명의 PT 검증이 필요해지면 `Light` 버퍼를 PT 셰이딩에 넣고 NEE로
  샘플링하는 별도 작업이 선행돼야 한다.

---

## 5. 출처

- A Primer On Efficient Rendering Algorithms & Clustered Shading — https://www.aortiz.me/2018/12/21/CG.html
- Clustered shading evolution in Granite — https://themaister.net/blog/2020/01/10/clustered-shading-evolution-in-granite/
- Volume Tiled Forward Shading (3dgep) — https://www.3dgep.com/volume-tiled-forward-shading/
- Thoughts on light culling for clustered shading (Sylvan) — https://www.sebastiansylvan.com/post/light_culling/
