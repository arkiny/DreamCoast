# F6 (Part C) — 콘텐츠 PT 레퍼런스 감사 + skylight tint 캘리브레이션

## 후속 랜딩 (2026-07-14): `P_SKYVIS_TINT` 기본 0.5 → 0.3 (PT-캘리브레이션)

감사 §3의 용의자 1을 스윕으로 판정 — interior masked_avg / lit_mean 비율(raster/PT) / lit-크롭 색 캐스트:

| TINT | masked_avg | 비율 | mean(B−R) — PT 기준 ≈ −1.0 |
|---|---|---|---|
| 0.5(구 기본) | 34.15 | 1.58 | −2.91 |
| **0.3(신 기본)** | **32.58 (−1.57)** | 1.41 | **−1.02 (PT 일치)** |
| 0.25 | 32.36 | 1.34 | −0.25 |
| 0.2 | 32.31(최소) | 1.26 | +0.75 (PT보다 차가움) |
| 0.0 | 70.65 | **0.68(역전)** | — |

- **선정 논리**: 게이트는 0.15–0.3이 평평한 분지(32.3–32.6). 그 안에서 **색 캐스트가 PT와 일치하는
  0.3**을 채택 — 0.2는 게이트 +0.27 이득뿐인데 캐스트가 PT보다 차가워짐(중성 바닥값이 캐스트를 PT의
  따뜻함에 맞춰주고 있었음). sunlit도 0.3에서 개선(§표 아래 재시드 값).
- **TINT=0 역전의 의미(정직)**: 바닥값을 다 걷으면 raster가 PT보다 32% 어두움 — 0.5는 GI 볼륨의
  멀티바운스 부족을 과보상하던 것. 남은 갭 축소는 이 상수가 아니라 **sky-vis 필드/GI 전송 개선**이
  다음 레버(깊은 차폐부 raster 86 vs PT 22는 tint와 무관 — GI 볼륨 누설/sky_fill 몫).
- **측정 함정(중요, F6B 규약에 추가)**: knob 스윕 시 **AE 커플링** — raster hdr이 어두워지면 자동노출이
  올라가 PT 표시값·masked_avg 절대치·pt_black_frac(ε=8은 디스플레이 공간)이 함께 움직인다. 행 내부
  비교는 유효하나 **행 간은 lit_mean 비율·B−R 같은 노출-불변 진단을 병용**할 것.
- budget 재기준선(하향): 커밋 메시지·매니페스트 참조.

# (원) 감사 — 판정: 신뢰 확보, 무수정

> 상태: **완료 (2026-07-14, 측정 전용 — 엔진 무변경)**. 상위 [gi-fidelity-roadmap.md](gi-fidelity-roadmap.md)
> §F6, 선행 [phase-f6b-content-pt-residual-plan.md](phase-f6b-content-pt-residual-plan.md).

## 0. 왜

F6B가 정량화한 lit-영역 잔차(sunlit ~30.6 / interior ~34.2, lit_mean 갭 18–28/255)를 SWRT 캘리브레이션의
절대 나침반으로 쓰려면, 레퍼런스(PT) 자체의 어두움-편향 용의자 둘을 먼저 기각/정량해야 한다:
(a) 바운스 예산 절단(`MAX_BOUNCES=8`), (b) PT↔IBL 하늘 단일소스 불일치. 이번 세션의 SWRT 레버 실측이
전부 ±0.2 이하(F2 −0.20, F4 fine −0.08, sc-feedback ±0.02)인 상황에서, 자를 검증하는 것이 선행 과제였다.

## 1. 하늘 단일소스 — 무혐의 (코드 검증)

- PT의 `sky()`는 `procedural_sky(dir, sun, sun_i, pc.sky.x, …) * pc.sky.yzw`로 **sky_gain + SKY_WB를
  이미 적용**(`rt_common.slang:37,78-82`).
- 호스트는 `record_path(…, self.sky_gain, self.sky_wb)`(main.rs:7712 부근)로 **IBL 캡처(`ibl.rs:358-381`)와
  동일한 해석 값**을 전달 — sun_dir/sun_intensity도 동일 소스. 레퍼런스는 "물리 하늘"이 아니라 **저작된
  씬 하늘**을 적분하며, 이는 의도된 계약(공정 비교)이다.
- sun disk는 primary ray 전용 + 바운스는 NEE(디스크 이중계상 없음, `rt_path.slang:45-47,127-147`).

## 2. 바운스 절단 — 무혐의 (실측, 2026-07-14)

방법: `MAX_BOUNCES` 8→32 임시 프로브(RR은 bounce 3부터 unbiased 연장이므로 하드 캡만이 편향원),
F6B 레시피(AUTO_EXPOSURE=1·RENDER_SCALE=1·WARMUP_FRAMES=192) 캡처, 8-바운스 기준과 직접 diff.

| 측정 | interior | sunlit |
|---|---|---|
| PT8 vs PT32 masked diff | **0.19/ch** | **0.25/ch** |
| lit_mean_pt | 48.61 → 48.81 (+0.20/255) | 64.39 → 64.65 (+0.26) |
| pt_black_frac | 0.502 → 0.501 | 0.171 → 0.169 |
| raster vs PT32 masked_avg | 34.21 (PT8 기준 34.15) | 30.59 (30.63) |

**절단 편향 ≤ 0.25/255 = lit_mean 갭(18–28/255)의 ~1%.** 커튼-차폐 pt_black 영역도 32바운스로 사실상
열리지 않음(0.502→0.501) — GI-reach 한계는 바운스 수가 아니라 광경로 기하의 문제(기존 F4-검정 서사 유지).

## 3. 판정과 후속 (반복 금지)

- **PT 레퍼런스는 이 콘텐츠에서 에너지-충분·하늘-정합 — 무수정 종결.** `MAX_BOUNCES` 상향, PT 하늘
  게인 조정, budget 재시드 전부 불필요(측정으로 기각).
- **lit_mean 갭의 실체 = raster(SWRT) 앰비언트 과밝음.** 다음 페이즈 = SWRT skylight 캘리브레이션.
  용의자 우선순위(코드 근거): (1) IBL diffuse × sky-vis 오클루전 경로의 min-occlusion/tint 누출
  (`skyvis_tint`/min_occ — 실내에 남는 하늘광 바닥값), (2) `SKY_GAIN=6.0` look-튜닝이 raster 앰비언트에만
  주는 실효 이득의 재검(레퍼런스 §1과 정합인지 — PT도 같은 게인을 쓰므로 *하늘 자체*는 정합; 문제는
  raster의 **unoccluded IBL 평가 + 누출 바닥값**), (3) `gi_volume` 히트 `sky_fill`(절차 하늘 × 반구 가중)
  과대 여부. 전부 F6B 게이트(개선 = masked_avg·lit_mean 갭 하락)로 판정.
