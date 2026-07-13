# F6 (Part B) 계획서 — 콘텐츠 PT 잔차 측정 자동화

> 상태: **승인·구현 (2026-07-13)**. 상위 [gi-fidelity-roadmap.md](gi-fidelity-roadmap.md) §F6,
> 선행 [phase-f6-pt-reference-usability.md](phase-f6-pt-reference-usability.md)(Part A).
> 구현: `tools/rt-compare.py`(lit-마스크 metric) + `tools/golden-image.py`(PT config 러너) —
> **엔진 코드 무변경**(Python 도구 + 매니페스트 + 문서만; 갤러리 앵커는 구조적으로 불변).

## 0. 왜 지금

F1 내내 콘텐츠 PT 잔차를 정량화할 수 없었다. 로드맵의 **1순위 성공 척도가 "PT 잔차"** 인데 콘텐츠에선
자동화·신뢰 측정이 없어서 F1~F5 이득이 "갤러리 바이트 동일 + 정성"으로만 검증됐다. 이 도구를 벼려야
이후 GI 작업(F2/F4/F5)을 **개선/중립으로 게이트**할 수 있다.

## 1. 진단 (코드 + 실측, 2026-07-13)

**Part A(PT 자동노출)는 랜딩·동작한다** — `73f96d9`: `pt_active && auto_exposure`일 때 tonemap
(`post.slang:134-137`)이 `exposure_buf`를 읽어 디퍼드 라이팅과 동일 적응노출을 곱한다(단일 소스).
`AUTO_EXPOSURE`는 콘텐츠 기본 ON(`main.rs:2628`), 갤러리는 항상 OFF(고정노출 앵커).

**남은 문제 = (1) 자동화 부재 + (2) PT-dim 오염.** 실측(콜로네이드, AUTO_EXPOSURE): raster mean 73.1 vs
PT mean 26.5. 노출 매칭 상태의 이 차이 = (a) **SW-RT GI vs 물리 GI 갭**(lit 영역 = 측정하려는 신호) +
(b) **PT-dim 영역 ~39%**(바운스 예산 내 광경로 도달 불가 — GI-reach 한계, F4 영역·버그 아님)이 전체
잔차를 오염.

### 착수 전 적대적 코드 검증에서 잡은 설계 결함 5건 (전부 반영됨)

1. **AE 수렴이 wall-clock 기반** (`main.rs:7404`, dt는 L4304 — FIXED_DT 아님): PT 캡처는 프레임이 느려
   즉시 수렴하지만 raster 캡처는 warmup 64에서 ~2–8% EMA 잔여 → 두 캡처의 노출이 어긋나 잔차에 전역
   밝기 오프셋 오염. → **양 캡처 `WARMUP_FRAMES=192`** (raster 잔여 <0.05%, PT는 (192+1)×8 ≈ 1544spp).
2. **Apple 티어 콘텐츠 기본 `render_scale=0.67` + TAAU**: 서브픽셀 지터가 `inv_view_proj`에 접혀 PT
   레이가 매 프레임 흔들리고, PT가 0.67× 내부 해상도로 누적된 뒤 업스케일됨. → **`RENDER_SCALE=1` 고정**
   (`main.rs:3387`) — 지터 없음·네이티브 누적·결정적 캡처.
3. **sub-ε 영역은 "물리-검정"이 아니라 노이즈 지배 저휘도 연속체** (하늘 도달 경로가 간헐적으로 존재):
   ε=4는 MC 노이즈 플로어에 걸려 문턱 근처 고분산 픽셀이 마스크를 오염. 인코딩 자체는 건전(0→8-bit 0,
   PT 경로에 bloom/grade/LUT 리프트 없음 — 검증). → **ε 기본 8** + 시드 시 ε∈{4,8,16} 민감도 실측·기록,
   사용 ε을 매니페스트 `lit_eps`에 명시. 서사도 "PT-dim 제외"로 정정.
4. **`golden-image.py --update`가 엔트리를 `{sha256,desc,env}`로 통째 교체** → 수기 매니페스트 키는
   증발(기존 gallery desc 수기 suffix가 실제로 이 상태였음). → **CONFIGS 레시피 = 단일 소스**: update가
   pt 필드(`pt/lit_eps/residual_budget/…`)를 레시피+측정에서 재방출; gallery desc를 CONFIGS에 흡수;
   "매니페스트 수기 편집 금지" 명문화.
5. **`rt-compare.py` stdout을 파싱하는 기존 소비자**: `tools/verify-rhi-thread.ps1:49-51`이
   `avg abs diff / channel:` 줄을 정규식 게이트로 사용. → 무플래그 출력 **바이트 동일 유지**(A/B 검증),
   신규는 전부 opt-in(`--lit-mask`,`--json`), 러너는 `RTCOMPARE_JSON` 한 줄 계약만 소비.

부수 확인: EV100은 AE를 끄지 않지만 raster 경로의 firefly 클램프가 AE 중에도 `self.exposure`(EV100 파생)
를 소비(`main.rs:5673-5679`, `3920-3928`) → pt config는 EV100 미설정 + 러너가 상속 env에서
`EV100`/`EXPOSURE` strip(셸 누수 차단), `AUTO_EXPOSURE=1` 명시.

## 2. 목표

콘텐츠 raster-vs-PT 잔차를 **회귀 게이트로 자동화**하되, actionable 신호(PT-lit 영역의 SW-RT-vs-PT)를
PT-dim 오염과 분리해 측정한다.

## 3. 구현 (랜딩 스펙)

### B1 — `rt-compare.py` lit-마스크 metric
```
python tools/rt-compare.py RASTER.png PT.png OUT.png [--amp N] [--lit-mask[=EPS]] [--json]
```
- `--lit-mask[=EPS]`(기본 8, 8-bit Rec.709 luma): **PT luma > EPS 픽셀만** 집계. 리포트:
  `masked_avg`(게이트 값), `pt_black_frac`(커버리지·게이트 아님), `masked_over8/32`,
  `lit_mean_raster/pt`(노출 오프셋 자가진단 — 곱셈성 어긋남이 여기서 바로 드러남).
- `--json`: `RTCOMPARE_JSON {...}` 한 줄(도구 소비 계약). lit 픽셀 0 → 명확한 에러 + exit 1
  (고정 EV100 함정 자가진단). 몽타주에서 제외 픽셀은 어두운 파랑.
- 무플래그 호출은 stdout·몽타주 **바이트 동일**(합성 이미지 A/B로 검증).

### B2 — 측정 카메라 (후보 스윕 실측으로 확정)

후보 3개를 warmup 64로 렌더, `pt_black_frac` 최저각 채택 (2026-07-13, Metal, 2560×1440):

| 후보 | eye → target | pt_black_frac | masked_avg | lit_mean r/pt |
|---|---|---|---|---|
| S1 나브 상향(볼트 천장 지배) | -14,2,0 → 8,10,0 | 0.519 | 38.3 | 67.0/36.0 |
| **S2 중정 아트리움 상향 (채택)** | **0,2,0 → -12,9,0** | **0.189** | 30.6 | 82.2/64.0 |
| S3 레벨 기본(나브 축) | 7,2.2,0 → -15.84,2.27,0 | 0.386 | 30.6 | 91.8/72.3 |

- **`sponza_pt_sunlit`(주 게이트)** = S2: 직사광이 아트리움 벽을 훑고 프레임 81%에 PT 라디언스.
- **`sponza_pt_interior`(보조 게이트, tolerant)** = 기존 콜로네이드 `-14,2,0 → 14,2,0`(F1 측정 연속성,
  pt_black_frac 높음 — 마스크 표본 적어 노이즈 마진 넉넉히).

### B3 — `golden-image.py` PT config 러너
- CONFIGS 레시피(발췌): `pt: True`, `lit_eps: 8`, env = `LEVEL=sponza_intel AUTO_EXPOSURE=1
  RENDER_SCALE=1 WARMUP_FRAMES=192 CAM_EYE/CAM_TARGET`(EV100 없음·strip).
- 러너: raster 렌더 → 같은 recipe + `P8_PATHTRACE=1` 렌더 → `rt-compare --lit-mask=<eps> --json` →
  **PASS ⇔ `masked_avg ≤ residual_budget`**. `pt_black_frac ≥ 0.9` → FAIL("check AUTO_EXPOSURE").
  SHA/PNG 골든 없음(잔차 전용), asset 부재 SKIP, PIL 부재는 명확한 FAIL. `--update` 중 실패는 exit 1.
- `--update`(pt config): `residual_budget = round(masked_avg + 0.3, 2)`(`PT_BUDGET_MARGIN`, 시드
  spread 실측으로 검증) + `residual_measured`/`pt_black_frac` 기록.

## 4. 게이트 (프로젝트 불변)
- **갤러리 바이트 동일** `65d04ceca2c4…`: 엔진 diff 0 + `--only gallery` 실행 확인.
- **결정론**: 시드 직후 재실행(run 2)이 PASS = run-to-run 재현 게이트. spread 실측 기록(§6).
- **DX≡VK**: 도구는 파이썬(무관). pt config 2종의 VK/D3D12 실행을 동결 중인 **Windows 재검증 배치에
  추가**(F1 전 스테이지와 함께).
- **단일 소스**: rt-compare 확장(중복 metric 금지), `RTCOMPARE_JSON` 계약, CONFIGS=매니페스트 소스.
- **측정 없는 단정 금지**: budget은 실측 시드, 하향 재기준선은 개선 실측 시에만.

## 5. 하지 말 것 / 비목표
- PT 실내 밝기를 인위로 올리기(가짜 ambient/바운스↑) — 물리 정확성 유지.
- PT-dim(F4) 영역을 게이트에 넣기 — `pt_black_frac`로 추적만. sub-ε 영역을 "물리-검정"으로 서사하기.
- 갤러리에 PT 게이트 적용(고정노출 앵커). 매니페스트 수기 편집(CONFIGS가 소스).

## 6. 시드 측정 기록 (2026-07-13, Metal, main `a1da684` 위, 2560×1440, warmup 192 ≈ 1544spp)

**Budget 시드 (`--update` run 1) + 재현 게이트 (run 2, 독립 재렌더):**

| config | masked_avg run1 → run2 | spread | budget | pt_black_frac | lit_mean r/pt |
|---|---|---|---|---|---|
| `sponza_pt_sunlit` | 30.601 → 30.606 | **0.005** | 30.9 | 0.171 | 82.3/64.5 |
| `sponza_pt_interior` | 34.350 → 34.350 | **<0.001** | 34.65 | 0.502 | 76.8/48.9 |

- spread ≤ 0.005/ch ≪ 마진 0.3 (60× 여유) — PT 누적+AE 고정점의 run-to-run 재현 확인.
  `PT_BUDGET_MARGIN=0.3`은 선례 콘텐츠 노이즈 플로어(0.035–0.047/ch)와 함께 넉넉히 검증됨.
- warmup 수렴 확인: sunlit masked_avg가 warmup 64에서 30.597, 192에서 30.601 — Δ0.004(수렴 완료).
- 갤러리 앵커 `65d04ceca2c4dbff` **PASS**(같은 run 2에서 확인, 엔진 diff 0).

**ε 민감도 (seed1 캡처 재계산):**

| config | ε=4 | ε=8 (채택) | ε=16 |
|---|---|---|---|
| sunlit: masked_avg (pt_black) | 31.91 (0.090) | 30.60 (0.171) | 29.46 (0.319) |
| interior: masked_avg (pt_black) | 38.33 (0.400) | 34.35 (0.502) | 28.99 (0.637) |

sunlit은 ε 4→16에서 ~8% 변화(강건); interior는 ~24%(저휘도 연속체 — 예상대로). ε=8이 노이즈 토와
표본 크기의 균형점. 사용 ε은 매니페스트 `lit_eps`에 기록되므로 수치 인용 시 항상 함께 인용.

**해석(정직):** masked_avg ~30은 "lit 영역에서도 raster가 PT보다 평균 ~18–19/255 밝음"(lit_mean 갭)을
포함한 SW-RT GI vs 물리 GI의 실제 구조 갭이다 — 이것이 F2/F4/F5가 낮춰야 할 기준선이며, 게이트는
"이보다 나빠지지 않음(개선/중립)"을 강제한다.

**F1 회고 (스트리밍 ON vs OFF) — 콘텐츠 PT 잔차 최초 정량화:**

| config | OFF(기본, =budget 시드) | `P11_CACHE_STREAM=1` | Δ |
|---|---|---|---|
| sunlit | 30.601 | 30.583 | **−0.02** (중립) |
| interior | 34.350 | 34.471 | **+0.12** (중립, 마진 0.3 이내) |

interior의 +0.12는 스트리밍 ON이 적응형 가변해상 카드를 균일 슬롯으로 바꾸는 설계 결과
(F1 계획서 Stage 1)로 예상 범위. 결론(정직): **정적 카메라에서 F1 스트리밍은 PT 잔차 중립** —
F1의 실질 가치가 정적 화질이 아니라 라이브 카메라 추종+메모리 상한이라는 기존 정성 평가를
이 도구로 처음 정량 뒷받침. 이후 F2/F4/F5는 이 기준선(30.9/34.65) 대비 개선/중립으로 게이트.

## 7. 검증 + 부수 효과
- sunlit/interior config PASS(재실행 재현), 갤러리 앵커 불변, ε 민감도 기록.
- **F1 회고 측정**: 이 도구로 `P11_CACHE_STREAM` OFF(기본) vs ON의 콘텐츠 PT 잔차를 최초 정량화 —
  F1 이득의 사후 검증 + budget 기준선(§6에 기록).
