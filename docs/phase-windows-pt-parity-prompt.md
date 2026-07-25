# Phase 프롬프트 — Windows PT↔래스터 발산 규명 (G2 잔여)

> 라이팅 종결 배치(docs/lighting-ao-shadow-closure.md, 2026-07-25)가 남긴 **단 하나의
> 미해결 종료 조건**. Windows(RTX 2070 SUPER)에서 PT 잔차 게이트가 Mac 봉인값과
> 전혀 다른 지점에 있다 — 이것은 튜닝 문제가 아니라 **발산 규명** 문제다.

## 증상 (전부 2026-07-25 실측, HEAD 5eb3691 + 종결 배치 수정 위)

- `sponza_pt_interior`: block64 **43.66** (예산 27.21, Mac 측정 26.91), lit-mask
  bias **+32.2** (raster lit_mean 114.7 vs PT 82.4). `sponza_pt_sunlit`: 22.70
  (예산 20.74), bias −5.3.
- Mac(F6N 랜딩)의 동일 게이트: interior bias **+8.9**. 즉 Windows에서 래스터↔PT
  간극이 Mac 대비 ~4배.

## 배제된 가설 (게이트 수치 A/B, PT 캐시로 각 ~2분)

| 토글 | interior block64 | 판정 |
|---|---:|---|
| 기준 | 43.66 | |
| `P_CACHE_SKY_OCCLUDE=1` (Apple 노브) | 43.56 | 무효 |
| `P11_REFLECT_MAX_ROUGHNESS=0.4` (Apple) | 43.56 | 무효 |
| `P_REFLECT_SKYFILL=0` | 40.98 | 미미 |
| `P11_LEGACY_IBL=1` (SW-RT 앰비언트 전체 OFF) | 53.13 | 악화 — 반사/GI 스택이 주범 아님 |
| `P_GI_VOL_CLIP=0` (fine 레벨 OFF) | 45.97 | 무효 |
| `SKY_GAIN=0.25` (스카이 라디언스 1/4) | 40.47 | **미미 — AE가 래스터 밝기를 재고정** |
| `P_GI_MULTIBOUNCE=0` | 43.57 | 무효 |

핵심 관찰: **AUTO_EXPOSURE가 래스터 lit_mean을 ~114로 고정**하므로(전 토글에서 불변)
bias는 개별 앰비언트 노브가 아니라 래스터↔PT의 구조적 배분 차이다.

## 시각 증거 (스크래치패드/dc-golden 캡처, 재현 레시피 포함)

- 래스터(interior 게이트 레시피): 실내 전면에 **차가운 청색 스카이라이트 캐스트**,
  로우 컨트라스트. PT: 물리적으로 그럴듯(따뜻한 석재, 깊은 그림자).
- `DEBUG_VIEW=10`(GDF GI): 천장/아치로 **하늘 라디언스가 새는 과밝은 GI**(볼륨-릭
  클래스 블롯치). `DEBUG_VIEW=7`(직접광): PT의 햇빛 패치와 일치(정상).
- `DEBUG_VIEW=8`(앰비언트): 실내 앰비언트가 거의 무차폐 수준으로 밝음.
- pt_black_frac: Windows 24.1% vs Mac 매니페스트 30.88%.
- VK-PT vs DX-PT 교차 diff: 종결 배치 최종 스위트 산출물
  (`m_ptx_interior.png`/`m_ptx_sunlit.png`) 참고 — 두 Windows RT 백엔드가 일치하면
  공통-코드 경로, 불일치하면 백엔드별 RT 버그.

## 다음 페이즈가 판별할 가설

- **H-A: Windows PT가 어둡다** — DXR/VK-RT 히트 셰이더의 MASK 알파(커튼 투과)
  경로가 Metal RT와 다르게 동작(F6M `e4cf267`은 Metal에서 검증). 커튼이 불투명하면
  콜로네이드 유입광 급감. 판별: 커튼 없는 레벨 변형으로 PT 재렌더, 또는 PT 히트
  카운터 계측.
- **H-B: Windows 래스터 GI/skyvis 체인이 Metal과 발산** — slang→DXIL/SPIR-V 공통
  경로의 의미 차(F1 배치의 push-크기류 같은 "Mac에서 원리적 검출 불가" 클래스가
  DX·VK에 **공통**이면 DX≡VK 게이트로는 안 잡힘). R32F 볼륨 trilinear, 샘플러 좌표,
  bent-normal validity 등. 판별: **Mac 박스에서 동일 HEAD로 게이트/디버그 뷰 재캡처**
  후 컴포넌트별 교차 비교(가장 결정적).
- **H-C: 예산이 Apple-스택 결합** — 미검증 v2 조합(reflect_stochastic+screen_hit+
  compact 동시 등). 위 표의 단일 토글은 배제됐으나 조합은 미검증.

## 제약

- F6I do-not-retry 원장 준수: 봉인된 sky-chain 재보정 금지 — 이 페이즈의 목표는
  **발산 지점 규명**이지 재튜닝이 아니다.
- 게이트/골든 레시피 변경 금지(예산은 하향 래칫 전용).
- 모든 판정은 `tools/golden-image.py` 수치 + 컴포넌트 디버그 뷰 이미지로.

## 도구 (종결 배치가 정식화)

`tools/perf-profile.py`(패스별 GPU 프로파일), `tools/seq-stability.py`(시퀀스 플리커),
PT 캐시(백엔드별 키; 래스터만 재렌더 ~2분), `DEBUG_VIEW=7/8/9/10` 컴포넌트 분해.
