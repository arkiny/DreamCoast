# F6 Part B 시작 프롬프트 — 콘텐츠 PT 잔차 측정 자동화

새 Claude Code 세션(맥, Metal)에 아래를 그대로 붙여넣어 F6 Part B를 시작한다. 이건 로드맵의
**1순위 성공 척도(PT 잔차)를 콘텐츠에서 실제로 쓸 수 있게 만드는 선결 인프라**다 — 이후 F2/F4/F5의 GI
충실도를 "개선/중립"으로 게이트하려면 이 도구가 먼저 있어야 한다.

---

## 붙여넣을 프롬프트

DreamCoast 엔진(`/Users/arkiny/GitRepos/DreamCoast`, Rust, RHI over Vulkan/D3D12/Metal)에서
**GI 충실도 로드맵 F6 Part B — 콘텐츠 PT 잔차 측정 자동화**를 시작한다. 먼저 `git fetch origin &&
git checkout main && git pull`로 최신 main을 받고(현재 `a1da684` = F1 표면캐시 가상화 Stages 0–4 +
F6 Part A PT 자동노출 머지 완료), 새 `feature/f6b-content-pt-residual` 브랜치를 판다.

### 왜 (측정된 갭)
로드맵 §0의 1순위 성공 척도는 **"자체 path-tracer 잔차"** 인데, **콘텐츠 씬엔 그 잔차가 자동화·신뢰
측정되지 않는다.** 방금 끝낸 F1 표면캐시 가상화도 이득을 "갤러리 바이트 동일 + 정성"으로만 검증했다
(콘텐츠 PT 잔차를 정량화 못 함). 이 도구를 벼려야 F2~F5를 PT-잔차로 게이트할 수 있다.

### 진단 (이미 확인됨 — 반복 조사 금지)
- **F6 Part A(PT 자동노출)는 랜딩·동작한다** (`73f96d9`, main에 머지됨). `pt_active && auto_exposure`일 때
  tonemap(`post.slang`)이 `exposure_buf`를 읽어 **raster와 동일 적응노출**을 곱한다(단일 소스). 갤러리는
  고정노출 sentinel → 앵커 불변. **즉 raster↔PT 노출은 매칭된다.**
- **함정(중요):** 잔차를 잴 땐 반드시 `AUTO_EXPOSURE=1`로 렌더할 것. `EV100=<n>` 고정을 주면 Part A를 안
  타서 실내 PT가 검정으로 crush된다(직전 F1 측정이 이 실수를 범했다 — 버그 아님, 측정 실수).
- **실측(`sponza_intel`, `CAM_EYE=-14,2,0 CAM_TARGET=14,2,0`, `AUTO_EXPOSURE=1`):** raster mean 73.1 /
  PT mean 26.5. 노출 매칭 상태의 이 차이 = (a) **SW-RT GI vs 물리 GI 갭**(PT-lit 영역 = 측정하려는 신호) +
  (b) **PT 물리-검정 영역 ~38%**(커튼-차폐 등 바운스 예산 내 광경로 도달 불가 = 물리 GI-reach 한계,
  **F4 영역·버그 아님**). (b)가 전체 잔차를 오염시킨다.

### 목표
콘텐츠 raster-vs-PT 잔차를 **회귀 게이트로 자동화**하되, **actionable 신호(PT-lit 영역의 SW-RT-vs-PT)를
물리-검정 오염과 분리**해 측정한다.

### 프로세스 (프로젝트 규칙 — 반드시 준수)
1. **계획서 이미 있음**: [docs/phase-f6b-content-pt-residual-plan.md](phase-f6b-content-pt-residual-plan.md)
   를 정독하고, 코드 근거로 정련해 사용자 승인 → 그 다음 구현. 승인 전 코드 변경 금지.
2. 착수 전 읽기: `tools/rt-compare.py`(잔차 metric — lit-마스크 추가 대상), `tools/golden-image.py`
   (config 러너 — PT config 추가 대상), `crates/shader/shaders/post.slang`(tonemap `exposure_buf` 읽기,
   Part A), `apps/sandbox/src/deferred.rs`(`record_auto_exposure`)·`main.rs`(`auto_exposure`/`tm_exposure`/
   `pt_active` 소스 선택 L7551·L8097 부근), `crates/shader/shaders/rt_path.slang`(PT rng·spp·`pc.frame` —
   결정성), `docs/golden-image-regression.md`(매니페스트 스펙).
3. 검증된 단일 커밋으로 랜딩(부분 단계 가능, 각 게이트 통과).

### 접근 (계획서 §3)
- **B1 lit-마스크 metric**: `rt-compare.py`가 PT luma > ε(≈4/255) 픽셀만 잔차 측정(`masked_avg`) →
  물리-검정(F4) 제외. `pt_black_frac`(커버리지)는 리포트만·게이트 아님. 게이트 = `masked_avg ≤ budget`.
- **B2 카메라**: sunlit(하늘/햇빛 보여 `pt_black_frac` 낮음 = 깨끗) 주 게이트 + interior 보조. 2–3 후보
  렌더로 `pt_black_frac` 최저 각을 sunlit로 확정.
- **B3 golden-image PT config**: 매니페스트에 `pt: true` + `residual_budget`. 러너가 raster + `P8_PATHTRACE=1`
  두 번(둘 다 `AUTO_EXPOSURE=1`, 같은 고정 카메라·warmup) 렌더 → B1 metric → budget 게이트. budget 실측 시드.

### 불변 게이트 (전부 통과)
- **갤러리 바이트 동일** 앵커 `65d04ceca2c4dbff`. PT config는 콘텐츠 전용, 갤러리 SHA 경로 무변 —
  `python tools/golden-image.py --only gallery`로 확인.
- **PT 결정론**: 고정 spp·시드·warmup으로 run-to-run 재현(콘텐츠 GI 노이즈 floor는 tolerant, budget에
  노이즈 마진 포함). metric은 결정론적 파이썬.
- **DX≡VK ≤0.001**: Windows RTX2070S 별도 검증(**현재 동결·Metal 우선**). 자동노출/측정은 Windows 재검증
  배치에 추가. (F1 전 스테이지도 이 배치 대기 중.)
- **단일 소스**: `exposure_buf`·`rt-compare` 재사용, 중복 metric/노출상태 신설 금지. **상표명 금지**
  (문서/주석/커밋 "reference engine").

### 리스크
PT 캡처 결정성(spp/시드/warmup 고정)이 budget 신뢰의 핵심. 카메라 `pt_black_frac`가 높으면 lit-마스크
표본이 작아 노이즈↑ — sunlit 각 선정이 중요. budget은 개선 시에만 하향 재기준선(측정 없는 단정 금지).

### 검증 + 부수 효과
- sunlit config PASS(`masked_avg ≤ 시드 budget`), interior 측정치 기록, 갤러리 앵커 불변, PT run-to-run 재현.
- **회고 측정**: 완성 후 이 도구로 **F1 스트리밍 ON/OFF(`P11_CACHE_STREAM`)의 콘텐츠 PT 잔차를 처음으로
  정량화**한다(F1 이득의 사후 검증 + budget 기준선).

### 참고 문서
계획서 `docs/phase-f6b-content-pt-residual-plan.md`(§3 접근·§4 게이트), `docs/phase-f6-pt-reference-usability.md`
(Part A 진단·구현), `docs/gi-fidelity-roadmap.md` §F6, `docs/phase-f1-surface-cache-virtualization-plan.md`
(F1 후속 배치 — 이 잔차 자동화가 F1 회고 측정 기반).

먼저 위 계획서·코드를 읽고 **F6 Part B 계획 정련안**을 제시하라(lit-마스크 metric 시그니처, 카메라 후보,
golden-image config 스키마, budget 시드 절차 포함). 승인 후 구현한다.

---

## 참고: 직전 세션 상태 (2026-07-13)
- `origin/main` = `a1da684` (Merge: F1 표면캐시 가상화 Stages 0–4 + F6 Part A PT 자동노출). 갤러리 앵커
  `65d04ceca2c4dbff` 불변, Metal 검증. 브랜치 `feature/f1-surface-cache-virtualization`도 푸시됨.
- **미결(전역)**: DX≡VK Windows 재검증(F1 전 스테이지 + aniso 등 Metal-only 배치 — 동결 중, 사용자가 차후).
- **F1 후속 배치**(계획서에): card_touched 미러-가시 방출 / 우선순위 방출 / grid-over-slots(스트리밍 시
  반사 grid 가속 복원) / freeze-latch 재-arm 튜닝 / Stage 4(b) 원거리 저해상 승인(균일-슬롯 풀로 미착수).
- **측정 함정 재강조**: 콘텐츠 PT는 `AUTO_EXPOSURE=1`로만 노출 매칭. `EV100` 고정 금지. 커튼-차폐 실내는
  자동노출로도 물리적 near-black(F4 GI-reach 한계) — lit-마스크로 제외하고 측정.
