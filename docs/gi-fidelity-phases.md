# GI 충실도 페이즈 — 상세 구현 계획 (팀 실행용)

상위: [gi-fidelity-roadmap.md](gi-fidelity-roadmap.md). 이 문서는 각 F 페이즈의 **착수 계획 +
1차 증분(first increment)** 을 팀(병렬 작업자)이 바로 실행할 수 있게 구체화한다. 각 작업자는
**독립 worktree**에서 **1차 증분만** 검증까지 끝내 커밋(푸시 금지, 리뷰 대기)한다.

## 공통 규약 (전 페이즈 필수)

- **먼저 읽기:** `DreamCoast/CLAUDE.md`(엔지니어링 5원칙), 이 문서, [gi-fidelity-roadmap.md](gi-fidelity-roadmap.md).
- **불변 게이트(순서대로):** `cargo fmt` → `RUSTFLAGS="-D warnings" cargo clippy -p <크레이트> --all-targets`
  → **갤러리 바이트 동일** SHA `af70c1a5c8db49661d2c7926140c1309c28fda04c82cc1ab8aa6638d588b2b74`
  (`./target/release/sandbox --backend metal --screenshot-clean out.png` 후 `shasum -a 256`) →
  **결정론**(run-to-run 바이트 동일) → (품질 변경 시) **콘텐츠 시각/정성** 검증 → DX≡VK(Windows 동결→보류 명시).
- **콘텐츠 검증 씬:** `LEVEL=sponza_intel EV100=11 WARMUP_FRAMES=64 CAM_EYE=-14,2,0 CAM_TARGET=14,2,0`
  RELEASE 빌드. 뷰 회귀: `P_SC_VIZ=1`(표면캐시), `DEBUG_VIEW=9/10`(GDF AO/GI).
- **원칙:** 근본원인·단일소스·heavy=opt-in(RenderQuality 티어 `quality.rs`)·verify-then-claim. **상표명 금지**(일반 서술).
- **셰이더 캐시:** 공유 include 편집 시 `crates/shader/build.rs`의 `SHARED_INCLUDES`에 등록돼 있어야 재컴파일됨.
- **1차 증분 철학:** 페이즈 전체를 끝내려 하지 말고, **검증 가능한 최소 토대**를 (가능하면 유닛테스트 포함)
  깨끗이 랜딩. 후속은 별도 증분. (per-mesh 피처의 P0/P1 방식 참고: `sdf_atlas.rs`/`mesh_sdf.rs`.)

---

## F1 — 표면 캐시 가상화 (최우선; 관측된 최대 갭)

**문제:** `apps/sandbox/src/fuse.rs` `MAX_CARDS=1024`, draw당 6면 고정 카드 → sponza_intel에서 448 draw 중
**278이 카드 없이 드롭**(라이팅 누락). 큰 씬일수록 악화.
**현재 코드:** `fuse.rs`(카드 빌드 + budget cull), `gdf.rs`(`build_surface_cache`, `cache_pos/cache_albedo/
cache_radiance` 아틀라스 버퍼, `record_cache_capture/light/visibility`), `sdf_cache_{capture,light,visibility}.slang`.
**목표:** 하드 드롭 제거 — 모든 (가시) 지오가 캐시에 참여. 데맨드 기반 카드 레지던시.
**설계 (스테이지):**
- S1 **카드 요청/레지던시** — 카메라 프러스텀 + (가능하면) GI/reflect가 실제 샘플한 draw를 표시 → 필요한 카드만 상주.
- S2 **페이지 풀 + LRU 방출** — 고정 슬롯 대신 페이지 풀; 예산 초과 시 최근 미사용 카드 방출·재사용(드롭이 아니라 스트리밍).
- S3 **카드 mip** — 원거리 draw 저해상 카드(cone 게더 정합).
**1차 증분(이번):** S1의 **결정론적 카드 우선순위 + 페이지 레지던시 스켈레톤** — draw별 우선순위(카메라 거리/프러스텀)를
CPU에서 산출해 예산 내 카드를 **결정적으로** 선택하고, 남는 draw는 "coarse 폴백"으로 명시(드롭 로그 제거). 실제 LRU
스트리밍은 후속. **유닛테스트**: 우선순위 정렬 결정론 + 예산 경계.
**게이트:** 갤러리 바이트 동일(콘텐츠 전용 seam), `P_SC_VIZ` 채움 개선/무회귀, 결정론.
**리스크:** 재조명이 GI/반사 공유 → 회귀 검증 필수. **의존:** 없음(착수 가능). 파일: `fuse.rs`, `gdf.rs`, 카드 셰이더.

## F2 — 스파스 브릭 mesh SDF (per-mesh 아틀라스의 진화)

**문제:** `crates/asset/src/sdf_atlas.rs`는 dense `dim³` 타일(캡 32³)만 — sparse/brick/mip 없음. 확장성 상한.
**현재 코드:** `sdf_atlas.rs`(팩커·`tile_uvw` 계약), `apps/sandbox/src/mesh_sdf.rs`(인스턴스·셀그리드),
`crates/shader/shaders/mesh_sdf_sample.slang`(직접샘플), `clipmap.slang`(`count==0` 위임).
**목표:** occupied 밴드/브릭만 저장 → 메모리↓·해상도↑; cone LOD용 mip.
**설계 (스테이지):**
- S1 **브릭 점유 분석** — per-mesh SDF를 8³ 브릭으로 나눠 |dist|<밴드 인 브릭만 선별(빈 브릭 표시). CPU, 결정론.
- S2 **브릭 아틀라스 + 인디렉션** — 점유 브릭만 아틀라스에 팩 + 인디렉션 볼륨(브릭좌표→아틀라스 페이지); 빈 브릭은 coarse 폴백.
- S3 **mip 체인** — 브릭 mip으로 cone LOD 샘플.
**1차 증분(이번):** S1 **브릭 점유 분석 + 메모리 견적** — `sdf_atlas.rs`에 브릭 분할·점유 판정(밴드 폭 파라미터) +
"이 씬에서 dense 대비 몇 %로 줄어드는가" 계측 함수. **유닛테스트**: 점유 판정 정확·결정론. GPU 배선은 S2에서.
**게이트:** 유닛테스트, (S2 이후) 직접샘플 대비 PT/시각 중립+메모리 감소. **의존:** per-mesh 아틀라스(이미 있음). 파일: `sdf_atlas.rs`.

## F3 — HW-RT 충실도 경로 (High 티어; SW/HW 분리)

**문제:** HW-RT는 ground-truth 패스트레이서(`apps/sandbox/src/rt.rs`) 전용, 실시간 GI 미사용. SW march는 원거리 오차.
**현재 코드:** `rt.rs`(BLAS/TLAS + 패스트레이서, DXR/VK_KHR/Metal), `gi.rs`/`reflect.rs`(SW 소비자), `bindless.slang` `tlas`.
**목표:** 동일 게더 seam 뒤 **HW-RT 트레이스 백엔드**를 High 티어 opt-in으로. SW=디폴트(스케일), HW=충실도.
**설계 (스테이지):**
- S1 **게더 레이 HW-RT 프로토타입** — 1개 소비자(예: `gdf_gi`)의 게더 레이를 TLAS closest-hit으로 교체, hit에서
  표면캐시 라이팅 읽기. `RT_*`/High 티어 게이팅.
- S2 반사 게더도 HW-RT. S3 백엔드 파리티(DXR≡VK_KHR).
**1차 증분(이번):** S1 **프로토타입 + 게이팅 seam** — High 티어(`P_HWRT_GI=1`류) opt-in, 디폴트 off(=현 SW, 바이트 동일).
TLAS 재사용 확인, Metal에서 빌드·동작(정성). **PT 잔차가 SW보다 나아지는지** 정량은 S1 목표.
**게이트:** 디폴트 off=바이트 동일, opt-in 시 Metal 동작+PT 잔차 보고, DXR≡VK_KHR(Windows 보류). **의존:** 없음(rt.rs 기존). **주의:** 리스크 높음—별도 트랙 격리.

## F4 — 계층 월드 라디언스 캐시 + 중요도 파이널 게더 (wave 2)

**문제:** 원거리 GI가 코사인 게더(노이즈)·월드 캐시 1레벨. **현재:** `gi.rs`, `gdf_gi.slang`, `world-radiance-cache.md`.
**목표:** 다중 클립 월드 라디언스 캐시 + 중요도 샘플 게더. **1차 증분:** 중요도 샘플 게더(BRDF/조도 가중)로 코사인 대체
(같은 spp 노이즈↓) — 셰이더 국소 변경. **게이트:** 콘텐츠 PT 잔차·정적 셔머 무회귀. **의존:** F1(캐시)·F2(필드) 후 착수 권장.

## F5 — in-cache 재질 충실도 (wave 2)

**목표:** (a) per-mesh **albedo 아틀라스**(F2 SDF 아틀라스와 동형; 히트 색 정밀), (b) **emissive** 캐시 방출, (c) 양면 식생.
**1차 증분:** per-mesh albedo 아틀라스(현 dense albedo 대체) — `sdf_atlas.rs` 재사용, `mesh_sdf_sample.slang` `ms_albedo`를
아틀라스로. **게이트:** PT 중립+색 정밀, 갤러리 불변. **의존:** F2(아틀라스 인프라) 후.

## F6 — 검증·견고성 인프라 (독립, 즉시 착수)

**문제:** PT 잔차 자동화가 **갤러리 전용**; 콘텐츠·표면캐시·셔머 회귀를 자동으로 못 잡음.
**현재:** `tools/rt-compare.py`, `CAPTURE_SEQ`(정적 셔머), `--screenshot-clean`.
**목표:** 콘텐츠 PT 대조 + 골든이미지 회귀(결정적).
**1차 증분(이번):** **골든이미지 회귀 러너** — 지정 씬 집합(gallery 앵커 + sponza_intel `P_SC_VIZ`/`DEBUG_VIEW`)을
캡처해 저장된 골든과 SHA/픽셀 diff 비교하는 `tools/` 스크립트 + 골든 갱신 모드. Metal 우선. **결정론** 필수.
**게이트:** 스크립트 자체가 결정적, 앵커 정확 매칭. **의존:** 없음.

---

## 팀 실행 (dependency-aware)

- **Wave 1 (병렬, 독립/토대):** **F1**, **F2**, **F3**, **F6** — 서로 다른 파일군, worktree 격리로 충돌 없음.
- **Wave 2 (F1/F2 랜딩 후):** **F4**(F1·F2 위), **F5**(F2 위).
- 각 작업자: 자기 페이즈 **1차 증분만** 검증까지 커밋(푸시 금지). 통합/리뷰는 오케스트레이터가 순차 진행.
- 충돌 최소화: F1=캐시(`fuse.rs`/카드셰이더), F2=필드(`sdf_atlas.rs`), F3=RT(`rt.rs`+게이팅), F6=`tools/` — 교집합 최소.
