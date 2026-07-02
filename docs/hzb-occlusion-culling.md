# PR-8 · HZB Occlusion 컬링

상위: [render-pipeline-reference.md](render-pipeline-reference.md) §1.0(V) / §2 표 V행 / §3 PR-8.
opt-in seam: **`HZB_CULL=1`** (전제 `P7_CULL=1` — GPU 컬 그리드가 컬링 대상. 전제가 없으면
`[hzb] HZB_CULL=1 ignored: …` 로그와 함께 무시). 디폴트 OFF = 바이트 동일.

## 1. 무엇을 만들었나

- **Hi-Z(HZB) 피라미드 빌드** (`crates/shader/shaders/hzb_build.slang` + `apps/sandbox/src/hzb.rs`):
  씬 depth(G-buffer depth)를 소스로 R32Float 렌더타깃 밉 체인(레벨 0 = 렌더 해상도 1/2, 1×1까지)을
  compute로 max-reduce. 이 엔진은 **standard-Z**(near=0, far=1, clear=1.0, `LessEqual`)이므로
  "영역 내 가장 먼 occluder depth"는 **max**다 — 배경(1.0)은 어떤 것도 가리지 않는 값으로 유지되어
  보수성이 구조적으로 보장된다. 홀수 크기 축은 리듀스 시 한 줄을 더 접어(3-tap) 어떤 소스 텍셀도
  누락하지 않는다(누락 = max 과소평가 = false-cull 위험이므로 필수).
- **HZB-aware 컬 패스** (`csCullHzb` in `cull.slang`): P7 GPU frustum 컬과 동일한 평면 테스트 후,
  인스턴스의 8개 월드 코너를 (unjittered, no-Y-flip) view_proj로 투영해 스크린 AABB + **최근접
  NDC depth**를 구하고, AABB를 덮는 가장 거친 밉에서 **AABB 코너 텍셀 4탭(2×2)** max와 비교한다
  (canonical Hi-Z 테스트). `z_near > hzb_max + ε`일 때만 컬. 코너가 near 평면 뒤에 있거나 AABB가
  화면 밖으로 나가면 **무조건 visible 유지**(보수적 폴백).
- **컬 그리드 드로우의 씬-depth 정합** (`cull_draw.slang` fragment): 그리드는 톤맵된 백버퍼 위에
  자체 depth로 그려지는 오버레이였다 — 벽 뒤 큐브가 벽을 뚫고 보였다. fragment에서 씬 depth를
  샘플해 뒤에 있으면 discard하는 manual depth test를 추가(디스플레이→렌더 해상도 스케일 지원).
  이로써 "occlusion-cull된 집합 == 어차피 안 보이는 집합"이 **구성적으로** 성립하고, 컬링 on/off
  이미지 동일성이 검증 가능한 명제가 된다. (백버퍼 패스는 depth를 무조건 clear하는 RHI 경로라
  attachment-load 방식은 불가 — manual test가 extent 분리(TAAU 업스케일)까지 커버하는 올바른 해법.)
- **컬링 통계**: host-visible 스토리지 버퍼에 (survived, occlusion-culled) 카운터를 GPU 아토믹으로
  기록. 전용 1-스레드 GPU clear 디스패치가 컬 앞에 barrier로 선행 — 호스트 zero는 frames-in-flight와
  경합해 2×256 같은 누적 오류를 냈던 것을 실측으로 확인 후 GPU clear로 교체. 60프레임마다 로그 +
  ImGui "Compute / GPGPU" 패널 표시. 읽기용 `StorageBuffer::read_into`를 RHI 3백엔드에 추가
  (Metal 검증; DX/VK parity pending Windows verification). frames-in-flight 특성상 리드백은
  근사(최근 프레임) 진단값이다.

## 2. 설계 리서치 — 프레임 내 의존(순서 모순) 해법 선택

같은 프레임의 depth로 같은 프레임을 컬링하면 "컬링이 그리기 전에 필요하고, depth는 그리기 결과"라는
순서 모순이 생긴다. canonical 해법 2가지를 조사했다:

- **(a) 이전 프레임 HZB (+ 옵션 리프로젝션)**: 지난 프레임 depth 피라미드로 이번 프레임을 컬.
  리프로젝션(카메라 이동을 반영해 depth를 워프)은 **비보수적** — 디스오클루전 영역의 depth가
  낙관적으로 채워져 false-cull(popping)을 낳는다는 것이 실무 공통 결론이다
  ([reprojection 경고](https://gist.github.com/devshgraphicsprogramming/faa1f98f65661c54a960b45ed1d450ea)).
- **(b) two-phase (2패스)**: 1차에 prev-HZB로 컬해 "지난 프레임에 보였던 것"을 먼저 그리고 그 depth로
  HZB를 갱신, 2차에 1차에서 컬린 것들을 새 HZB로 재테스트해 false-negative를 마저 그린다 — 대규모
  GPU-driven 렌더러의 표준
  ([Two-Pass Occlusion Culling](https://medium.com/@mil_kru/two-pass-occlusion-culling-4100edcad501),
  [Two-Pass HZB](https://medium.com/@Lucmomber/two-pass-hierarchical-z-buffer-occlusion-culling-93171c5a9808),
  [GPU-driven occlusion culling 실험](https://interplayoflight.wordpress.com/2017/11/15/experiments-in-gpu-based-occlusion-culling/)).

**선택: (a)의 리프로젝션-없는 형태 — "prev-frame HZB, 단일 페이즈, 보수적 4탭 테스트".** 근거:
1. **컬링 대상이 씬 depth에 기여하지 않는다.** P7 컬 그리드는 톤맵 후 합성되는 데모 지오메트리라
   two-phase의 "1차 서바이버로 HZB 갱신" 단계가 성립하지 않는다(그려도 HZB 소스가 안 바뀜).
   two-phase는 컬링 대상 == depth 기여자일 때 의미가 있고, 그 전제는 메인 씬 드로우가 GPU-driven이
   된 뒤(Phase 23 월드 렌더링)에 온다.
2. **고정 카메라에서 보수성이 증명 가능하다.** 카메라 정지 시 prev-HZB == cur-HZB로 수렴하고, 4탭
   코너-텍셀 테스트는 AABB 전 픽셀의 depth 상한(max)과 비교하므로 false-cull이 없다. 검증 게이트
   (고정 카메라 캡처 OFF≡ON 바이트 동일)가 이를 직접 측정하며, §4의 가림-양성 케이스(98/256 컬)에서
   실제로 통과했다.
3. **움직이는 카메라에서도 1프레임 지연 팝인만 가능**(전형적 트레이드오프). 첫 프레임은 `enabled=0`
   (피라미드 없음) 가드. 리프로젝션은 명시적으로 배제(위 비보수성).
4. **two-phase 업그레이드 경로 확보**: 빌드/테스트 셰이더와 `HzbSystem`은 컬링 대상과 무관한 재사용
   모듈이라, 메인 씬이 GPU-driven이 되면 페이즈 디스패치만 추가하면 된다(§5).

### depth 소스 (PR-1 전제 관련)

이 브랜치의 베이스(main `534d1df`)에는 PR-1 depth pre-pass가 없다(§2 표 #1이 여전히 🔴). 스펙이
허용하는 대체 소스인 **G-buffer depth**를 쓴다. **prev-frame HZB 방식은 "완성된 지난 프레임 depth"만
필요하므로 prepass 유무와 독립적**이다 — PR-1이 머지되면 `record_build`의 read 리소스만 prepass
depth로 바꾸면 되고(그래프 배선 1줄), 빌드/테스트 로직은 그대로다. 이 때문에 `DEPTH_PREPASS=1` 대신
`P7_CULL=1`을 전제 seam으로 삼았다(컬링 대상이 P7 그리드이므로 이쪽이 실질 전제).

## 3. 데이터 흐름 / 그래프 배선

```
frame N:   cull_reset → cull_hzb(read: HZB[N-1] via import_external) → indirect draw
           gbuffer(depth) → … → hzb_build(read: g_depth, write: HZB external)
frame N+1: cull_hzb가 HZB[N]을 읽음
```

- HZB 밉 체인은 **app-owned persistent** `RenderTarget`(R32Float, storage) 배열 — 렌더 그래프
  트랜지언트는 단일 밉이므로 레벨당 타깃 1개 + 연속 bindless 슬롯(`hzb_base + mip` 인덱싱; 연속성은
  매 프레임 `slots_are_consecutive()` 검증, 아니면 occlusion 테스트 자동 off).
- 같은 external 리소스에 컬 패스가 **read**, 빌드 패스가 **write** 선언 → 그래프 WAR 엣지가
  "컬(지난 프레임 읽기) → 빌드(덮어쓰기)" 순서를 강제. 빌드는 external-writer라 dead-pass culling에서
  살아남는다(소비자가 다음 프레임).
- 밉 간 리듀스는 패스 내부 `storage_to_sampled`/`rt_to_storage` 전이로 체이닝.
- 레벨 0 = 렌더 해상도 1/2 (`HZB_BASE_DIVISOR=2`): 메모리/대역 절반, 보수성 무손실(더 거친 max).

## 4. 검증 (Metal, M3 · 2560×1440 · RENDER_SCALE=1 native; DX/VK parity pending Windows verification)

| 게이트 | 결과 |
|---|---|
| `cargo clippy --all-targets -- -D warnings` + `cargo fmt` + `cargo test` | 클린 / 94 tests pass |
| 디폴트 OFF 골든 앵커 | `af70c1a5c8db49661d2c7926140c1309c28fda04c82cc1ab8aa6638d588b2b74` **일치 (바이트 동일)** |
| 갤러리 `P7_CULL=1` OFF vs ON | `2b3917e1…93d5` **동일**; 91 survived / 0 occluded (그리드가 씬 위 부유 — 가림 없음, 기대값) |
| sponza lion-view (`CAM_EYE=-14,2,0→14,2,0`) OFF vs ON | `bdd99529…faa8` **동일**; 그리드 전량 frustum-cull (0 survived) |
| **가림-양성 케이스** sponza 아트리움 룩업 (`CAM_EYE=0,2,0 CAM_TARGET=10,25,0`) OFF vs ON | `f46cca26…7298` **바이트 동일** · **256 중 98 occlusion-culled / 44 survived** (하늘 개구부로 보이는 큐브만 생존 — 시각 확인) |
| `HZB_CULL=1`만 (P7_CULL 없음) | ignore 로그 + 디폴트 경로 유지 |

`PROFILE_GPU=1` (가림-양성 뷰, 동일 프레임 조건):

| 패스 | OFF | ON |
|---|---|---|
| `gbuffer` | 8.21 ms | 6.97 ms (공유 GPU 박스 변동폭 내 — HZB와 무관) |
| `cull` / `cull_hzb` | 0.051 ms | 0.061 ms |
| `hzb_build` | — | 0.130 ms |
| `cull_draw` | **0.353 ms** | **0.108 ms** (−69%, 98개 culled) |

순효과: 이 데모 그리드에서 +0.14 ms(빌드) vs −0.245 ms(드로우) ≈ −0.1 ms. 절대치는 작지만
(그리드가 가볍다) 드로우 비용에 비례해 커지는 구조이고, 빌드 비용은 씬 복잡도와 무관(고정 depth
리듀스)이다.

### 검증이 잡아낸 버그 3건 (보수성 게이트의 가치)
1. **빌드 소스 extent 0×0**: compute 패스의 `ctx.extent()`가 external(0×0)을 보고해 전 레벨이
   depth 텍셀 (0,0)만 읽음 → 전량 false-cull. extent를 명시 전달로 수정.
2. **단일 center-탭**: AABB가 텍셀 경계에 걸치면 이웃 텍셀의 하늘 depth를 놓쳐 지붕/하늘 경계에서
   false-cull. **4탭 코너-텍셀** 테스트로 교체(+경계 정렬 시 밉 1단 상승).
3. **V 방향 뒤집힘**: NDC +Y(위) ↔ 텍스처 row 0(위) 매핑에서 V 미러링 — 화면 상단 인스턴스가
   바닥 depth를 읽어 false-cull. `uv.y = 0.5 − 0.5·ndc.y`로 수정(전 백엔드 공통 — 컬 매트릭스가
   no-flip이므로).

## 5. 남은 것 / 업그레이드 경로

- **two-phase**: 메인 씬 드로우가 GPU-driven(인스턴스 리스트 + indirect)이 되면 1차(prev-HZB 컬) →
  depth 갱신 → HZB 재빌드 → 2차(재테스트)를 이 모듈 위에 얹는다.
- **PR-1 연동**: prepass depth가 생기면 `record_build`의 read만 교체.
- **TAAU jitter**: 업스케일 경로에서는 씬 depth가 jitter되고 컬 매트릭스는 unjittered — 서브픽셀
  경계에서 이론상 false-cull 가능(coarse-밉 4탭이 실질 흡수). 검증 구성은 native(`RENDER_SCALE=1`,
  jitter off). jitter-aware 보정(AABB 1px 팽창)은 필요 시 추가.
- **storage_images 슬롯**: 밉당 1 UAV 슬롯(1440p 기준 ~11개) — 64-슬롯 한도 내지만 Phase 23
  규모에서는 단일 밉드 텍스처 + per-mip UAV 뷰 RHI 확장이 맞다.
- **DX≡VK**: Windows 박스에서 `HZB_CULL=1` 갤러리/sponza 캡처 비교 + `read_into` 확인.
