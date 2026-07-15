# F6D 계획서 — PT 잔차 메트릭의 바이어스/산포 분해 (도구 전용)

> 상태: **진행 중** (2026-07-15). F6 검증 인프라 트랙. 동기 =
> [phase-f4b-hierarchical-cache-plan.md](phase-f4b-hierarchical-cache-plan.md) §5 +
> [phase-f4b2](phase-f4b2-mb-boost-probe-plan.md)·[phase-f4b3](phase-f4b3-box-half-probe-plan.md)
> 기각 판정 — masked_avg가 서로 다른 두 오차 유형(균일 오프셋 vs per-pixel 불일치)을 합산해
> 물리 지표(lit 비율)와 정반대 판정을 내는 실측 사례가 확보됨. 엔진 무변경, rt-compare.py /
> golden-image.py 확장만.

## 0. 동기 편향 가드 (사전 등록 — fine 재판정 이전에 확정)

이 페이즈는 "fine이 통과하도록 메트릭을 바꾸는" 작업이 아니다. 가드 3중:
1. **분해 정의를 아래 §1에 먼저 확정**하고, fine 재판정(§4)은 그 뒤에 돌린다. 정의는 fine
   수치를 보기 전에 커밋된다(이 문서의 커밋 순서가 증빙).
2. **기존 masked_avg는 계속 보고·게이트**한다 — 분해는 병기(shadow) 지표로 추가되고, 게이트
   정책 변경(있다면)은 분해 데이터를 본 뒤 별도 결정·별도 커밋으로 제안만 한다(이 페이즈가
   budget을 바꾸지 않는다).
3. 재판정의 결과가 fine에 불리해도 그대로 기록한다(F4B2·F4B3과 동일한 정직성 규약). 특히
   주의: fine의 산포 증가가 실재라면 분해 게이트에서 fine은 **더 명확히** 차단될 수 있다 —
   그것도 유효한 성과다(필드-품질 페이즈의 목표 수치를 제공).

## 1. 분해 정의 (사전 등록)

lit-마스크(기존 PT luma > EPS=8, 불변) 픽셀 집합에서, 채널별 **부호 있는** 델타
s_c(i) = raster_c(i) − pt_c(i):

- **masked_bias_c** = mean_i(s_c(i)) — 채널별 부호 있는 평균 오프셋.
- **masked_scatter_c** = mean_i(|s_c(i) − masked_bias_c|) — 채널별, 자기 평균 둘레의 평균
  절대 편차(MAD).
- 보고값: `masked_bias` = (bias_R+bias_G+bias_B)/3 (부호 유지 — 전 채널 공통 오프셋 =
  노출/에너지 항), `masked_scatter` = (scatter_R+scatter_G+scatter_B)/3.

물리적 정당화(정의의 근거, fine과 무관하게 성립):
- **bias**: 화면 전역 공통의 밝기 오프셋 — 지각적으로 균질(장면 구조 보존), 노출·캘리브
  스칼라로 소거 가능한 성분. F6C가 확인한 "lit 갭"이 이 성분.
- **scatter**: 평균을 맞춰도 남는 per-pixel 구조 불일치 — 그림자 위치·그라디언트·블록화 등
  실제로 "달라 보이는" 성분. 스칼라 레버로 소거 불가(F4B2·F4B3 실증).
- 관계: |bias| ≤ masked_avg ≤ |bias| + scatter (삼각부등식). 순수 오프셋 이미지 쌍은
  masked_avg = |bias|, scatter = 0; 제로-평균 노이즈 쌍은 bias ≈ 0, masked_avg ≈ scatter.

구현: 채널별 부호 있는 델타 히스토그램(−255..255) 1-패스 누적 → 평균·MAD를 히스토그램에서
계산(2-패스/메모리 증가 없음).

## 2. 도구 계약 (변경 금지 항목의 재확인)

- rt-compare.py 무플래그 stdout 라인(`avg abs diff / channel: …`)과 `RTCOMPARE_JSON` 라인
  프리픽스는 정규식/프리픽스 파싱 계약(verify-rhi-thread.ps1, golden-image.py) — **기존 라인
  불변, 신규 라인·신규 JSON 키만 추가**(json.loads 소비자는 키 추가에 안전).
- golden-image.py: PT 행 detail 문자열에 bias/scatter 병기, `--update` 시 매니페스트에
  `residual_bias`/`residual_scatter` **기록만**(게이트는 여전히 masked_avg ≤ budget).

## 3. 검증

1. **합성 자기검증**: (a) 동일 이미지 쌍 → 전부 0; (b) +k 균일 오프셋 쌍 → bias=k,
   scatter≈0, masked_avg≈k; (c) ±k 체커보드 쌍 → bias≈0, scatter≈k, masked_avg≈k.
2. **실측 회귀 불변**: 기존 키(masked_avg 등)가 신규 코드 경로에서 종전 값과 동일(소수 4자리)
   — dc-golden의 기존 캡처 쌍으로 전/후 비교.

## 4. fine 재판정 (분해 지표로 — 정의 확정 후 실행)

4 구성 분해 테이블: {coarse, fine(CLIP=1, t0.15)} × {sunlit, interior}. coarse 쌍은 최신
공식 런의 캡처 재사용, fine 쌍은 같은-설정 신선 쌍 캡처(F4B 함정 규약). 기대 가설(사전
기록): coarse는 bias-우세(+5.8대), fine은 bias≈0·scatter 증가. 판정 산출물:
- 두 오차 유형의 실측 크기 — fine의 "바이어스→산포 전환"의 정량화.
- 게이트 정책 **제안**(적용은 별도 결정): 예) masked_avg 유지 + scatter 상한 병기, 또는
  bias/scatter 개별 budget. 제안 기준: 갤러리·기존 그린 구성이 전부 그린으로 남을 것.
- 필드-품질 페이즈(후속)의 목표 수치: fine이 통과하려면 scatter를 얼마나 줄여야 하는가.

## 5. 비목표

budget/게이트 정책의 이 페이즈 내 변경 / 엔진·셰이더 변경 / lit-마스크 EPS 변경 / fine
기본 ON 재시도(분해 데이터 없이) / 기존 stdout·JSON 계약 파괴.
