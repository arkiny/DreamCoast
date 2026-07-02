# Velocity (모션벡터) G-buffer 채널 — 파이프라인 재정합 PR-2

상위: [render-pipeline-reference.md](render-pipeline-reference.md) §2 표 #3·#13·#14, §3 PR-2.
opt-in seam: **`P_VELOCITY=1`** (디폴트 off = velocity RT 미생성·기존 TAAU 경로·골든 앵커 바이트 동일).

## 1. 무엇을/왜

base pass 시점에 픽셀별 **스크린 공간 모션벡터** `cur_ndc.xy − prev_ndc.xy` (NDC 기준,
perspective-divide 후)를 전용 `Rg16Float` RT에 기록한다. 종전 TAAU는 prev view-proj +
world-pos 재투영으로 **카메라 모션만** 리프로젝션했기 때문에 움직이는 오브젝트(Spin/노드
애니메이션/스키닝/모프)에서 히스토리가 어긋나 고스팅/스미어가 생겼다. velocity가 있으면 TAAU가
각 픽셀을 **그 표면 자신의 모션**으로 리프로젝션한다. 모션블러(#13)와 velocity-aware TAA(#14)의
공통 선결이며, 이 PR은 TAA 소비자까지 배선해 페이오프(고스팅 감소)를 검증했다.

## 2. 설계 결정 (canonical 리서치 근거)

리서치 소스: Intel TAA 레퍼런스 구현(GameTechDev/TAA), AMD FSR2 GDC 자료, Alex Tardif
"Temporal Antialiasing Starter Pack", GameDev.net velocity-jitter 스레드. 핵심 합의:

1. **전용 RT, RG16Float.** 부호 있는 서브픽셀 모션에 충분한 부동소수 정밀도의 표준 포맷.
   기존 4-MRT G-buffer에 5번째 MRT로 넣지 않고 **별도 지오메트리 패스**로 분리 — 디폴트(off)
   경로의 G-buffer 파이프라인/포맷/패스가 1바이트도 변하지 않는 가장 강한 seam. (트레이드오프:
   on일 때 불투명 지오메트리 2회 래스터. opt-in 비용이며, 추후 depth-prepass/base-pass 통합
   MRT로 옮길 수 있는 구조로 격리해 둠.)
2. **jitter 제외.** velocity에 TAA 서브픽셀 jitter가 들어가면 리프로젝션에 jitter가 새어들어
   shimmer가 재발한다(레퍼런스들 공통 경고). 그래서 velocity 패스는 씬 렌더가 쓰는 jitter된
   `view_proj`가 아니라 **unjittered** `view_proj_stable`(현재) + `prev_view_proj_taau`(이전)을
   사용한다. 두 unjittered clip 위치의 차이므로 구성상 jitter-free.
3. **Y-flip 상쇄.** Vulkan은 clip Y가 아래 방향이지만, 모션벡터는 **같은 컨벤션으로 계산한 두
   NDC 위치의 차**라서 양변에 걸린 flip이 뺄셈에서 상쇄된다. RT에는 raw NDC delta를 저장하고,
   소비자(TAAU)가 자기 UV 컨벤션(`sy`)으로 변환한다 — 3백엔드 동일 바이트.
4. **3×3 dilated velocity.** TAAU 리프로젝션은 3×3 이웃에서 **가장 긴** 모션을 채택(Intel TAA
   "edge following", FSR2 "reconstruct & dilate"). 실루엣 픽셀이 전경(움직이는 물체)의 모션을
   물려받아 이동 물체 가장자리의 스미어를 죽인다.

## 3. Prev-transform 단일 소스

“이전 프레임 포즈”는 애니메이션/트랜스폼 시스템 한 곳에서 공급한다 (5원칙 #4):

| 모션 종류 | prev 소스 | 위치 |
|---|---|---|
| 카메라 | `prev_view_proj_taau` (unjittered, 프레임 말 저장) | `main.rs` |
| 스태틱/Spin/노드 애니메이션 | `App::prev_transforms: Vec<Mat4>` — draw-list의 **결정론적 삽입 순서**(인덱스 안정)를 키로 프레임 말에 이번 프레임 world transform을 저장 | `main.rs` |
| 스키닝 | `SkinnedMesh::prev_palette_idx` — per-fif(2) 팔레트 링에서 **다른 슬롯**(= 정확히 1프레임 전 조인트 팔레트)의 bindless 인덱스를 `update_palettes`가 덮어쓰기 전에 기록 | `skin.rs` |
| 모프 | `GpuMorphMesh::prev_weight_idx` — 같은 방식의 per-fif 웨이트 링 이전 슬롯 | `morph.rs` |

프레임 초 `prev_scene: Vec<PrevPose>`(velocity.rs)를 scene 순서로 조립: 스태틱은
`prev_transforms[i]`(첫 등장/첫 프레임은 현재 transform = 모션 0), 스키닝은 identity transform +
prev 팔레트, 모프는 노드 transform + prev 웨이트. `P_VELOCITY` off면 빈 벡터(디폴트 경로 비용 0).
CPU-morph 폴백(`MORPH_CPU=1`)은 vbuf 스왑 방식이라 deform 모션은 0으로 처리(노드/카메라 모션만)
— GPU morph가 기본 경로.

## 4. 셰이더 구성

- `crates/shader/shaders/velocity.slang` — `vsMain`(스태틱) / `vsMainSkinned`(현재+prev 팔레트로
  LBS 2회) / `vsMainMorphed`(현재+prev 웨이트로 블렌드 2회) → 공용 `fsMain`이
  `motion_ndc = cur.xy/cur.w − prev.xy/prev.w` 출력. `csViz`는 DEBUG_VIEW=11 시각화(방향→RG,
  크기→B; `VELOCITY_VIZ_SCALE=40` 증폭).
- `taau.slang` — `velocity_reproject(uv, cuv, sy)`: velocity 인덱스가 있으면 3×3 최장 모션을
  **지오메트리 위치(cuv)**에서 샘플하고 **안정 그리드(uv)**에서 `prev_uv = uv − Δuv` 적용.
  (jitter된 cuv를 리프로젝션 기점으로 쓰면 jitter가 히스토리 정렬을 깨서 오히려 악화 — 구현 중
  실측으로 확인, §6.) 모션이 정확히 0인 픽셀은 기존 카메라-온리 경로로 폴백해 정지 씬 비트 동일.
- velocity 패스 depth는 G-buffer depth를 **test-only**(`depth_write:false`)로 공유 — depth 재기록이
  다운스트림 소비자를 교란하지 않는다(정지 씬 byte-identity의 필요조건이었음, §6).

## 5. 검증 (Metal, macOS M3 — DX/VK parity pending Windows verification)

> **Windows parity 시 확인 필수**: velocity 파이프라인은 G-buffer depth를 test-only로 재사용하는데,
> RHI의 `DepthCompare::Less`는 Metal에서 역사적으로 `LessEqual`로 매핑되지만 VK/DX는 strict `LESS`다.
> 동일/유사 depth 재래스터에서 VK/DX는 프래그먼트가 대량 reject되어 velocity가 0으로 남을 수 있다
> (이미지 깨짐이 아니라 카메라-온리 TAAU 폴백). Windows 검증에서 확인 후 필요하면 `LessEqual`
> variant를 RHI에 추가하거나 SV_Position을 jittered VP로 통일하는 쪽으로 정리한다.

- `cargo clippy --all-targets -- -D warnings` 클린, `cargo fmt` 적용.
- **디폴트 OFF 골든 앵커**: `--screenshot-clean` sha256
  `af70c1a5c8db49661d2c7926140c1309c28fda04c82cc1ab8aa6638d588b2b74` — **바이트 동일**.
- **정지 씬 ON==OFF**: `P_VELOCITY=1`(정지 갤러리, TAAU 비활성) → 같은 sha256(위) 바이트 동일.
  `P_VELOCITY=1 P_TAAU_FORCE=1`(TAAU 활성·카메라/씬 정지, 16프레임) → off와 **sha256 동일**
  (`78273955…`, zero-velocity 폴백 + depth test-only 덕).
- **고스팅 개선** (`P15_SPIN=8 P_TAAU_FORCE=1 CAPTURE_SEQ=20 CAPTURE_SEQ_STEP=0`, 카메라 고정·
  오브젝트 회전, no-TAA 동일 프레임을 ground truth로 mean|Δ| 잔차):
  - full frame: OFF **0.2688** → ON **0.2623**
  - 이동 오브젝트 crop: OFF **1.0680** → ON **1.0265** (고스팅 잔차 ~4% 감소; 회전은 실루엣
    이동이 작아 보수적인 케이스 — 병진 모션에서 이득이 더 큼)
  - jitter-off 대조군에서도 동일 방향(crop 0.7264 → 0.6934).
- **디버그 뷰**: `P_VELOCITY=1 DEBUG_VIEW=11` — 정지 배경 = 무모션 균일색, 회전 오브젝트에
  모션 하이라이트 (스크린샷 확인).

## 6. 구현 중 확인된 함정 (기록)

1. **jitter된 cuv 기점 리프로젝션은 악화시킨다.** 첫 구현은 `prev_uv = cuv − Δuv`였고 스핀
   잔차가 OFF보다 나빠졌다(1.32 vs 1.07). velocity는 unjittered 그리드 기준이므로 안정 `uv`
   기점으로 바꾸자 개선으로 반전. jitter-off 대조 실험으로 원인 분리.
2. **velocity 패스의 depth 재기록이 정지 씬 byte-identity를 깼다.** 같은 값을 다시 쓰는
   depth_write라도 이후 패스가 읽는 depth 라이프타임/스케줄에 영향 → `depth_write:false`로 해결.
3. **디버그 뷰 값은 톤맵을 통과한다**: viz의 0.5(무모션)가 화면에서 ~0.81로 보이는 건 ACES+sRGB
   인코딩 때문이지 velocity 오류가 아님 (분석 중 오판 주의).
