# Shadow + Reflection Quality (PT-parity 후속)

> G-buffer world_pos 수정 후 재기준선(기본 씬·풀 PT) = **~6.9/ch**. 몽타주 diff 지배 요인:
> ① 크롬/글로시 근거울 반사 ② PCF(하드) vs 레이트레이스 소프트 그림자. 이 둘을 줄인다.
> **설계 원칙: 최적화 항상 고려 + 품질을 "티어"로 분리 가능하게(scalability seam) 둔다.**

## Scalability seam (공통)
- 각 기능은 **기본 on + env/플래그로 off→저비용 폴백**. 나중에 `RenderQuality{low,med,high}` 한 곳으로
  묶어 분리할 수 있도록, 품질 파라미터(샘플 수·반경·토글)는 **globals의 여유 슬롯 + 상수**로 격리한다.
- 셰이더는 컴파일타임 상수(SAMPLE_COUNT 등)를 한 곳에 모아 두어 티어 교체가 1줄이 되게.

## Phase 1 — Soft directional shadows (PCSS-lite)  ✅ 구현(옵트인 티어)
PCSS-lite 구현: blocker search(16 Poisson) → 평균 차폐 깊이 → **거리비례 페넘브라**(directional이라
선형 깊이차 × 캘리브레이션 팩터) → 16 Poisson PCF, per-pixel IGN 회전. `globals.shadow.z`=penumbra
factor(0=하드 3×3 폴백, scalability seam), 품질 상수(샘플수/search/max)는 셰이더 한 곳에.
- **핵심 발견(측정으로 확정)**: PT 태양 디스크 `SUN_COS_MAX=0.9998`(각반경 ~1.15°)는 **거의 샤프** →
  캘리브레이션 페넘브라가 1~6 텍셀로 매우 작음. 그림자 영역 비교: **hard 24.378 < soft 24.5~24.9 (vs PT)** =
  소프트가 오히려 PT에서 멀어짐. 그림자 영역 24/ch 차이는 부드러움이 아니라 **간접광/바운스·위치**가 지배.
- **결정**: 기본 = 하드 3×3 PCF(가장 저렴 9탭 + PT 최근접). PCSS-lite는 **옵트인 품질 티어**
  (`SHADOW_SOFTNESS=<f>` / UI 슬라이더; PT-캘리브 ~0.0375, 클수록 미적 소프트). 0.30에서 자연스러운
  contact-hardening 소프트섀도우 확인. 소프트 경로 DX≡VK=0.0165/ch(페넘브라 가장자리 V-flip×회전,
  옵트인이라 허용; 기본 하드 경로는 0.0013 유지).
- **결론**: 그림자는 PT-패리티 레버가 아님(태양이 샤프). 패리티 개선은 Phase 2(반사)가 본진.

## Phase 2 — Glossy/chrome reflection 정확도
현재: C8j 확률적 GGX GDF 레이 + temporal resolve. 크롬(rough 0.08)·글로시 잔차 큼(저해상 SDF blob + SSR 미스).
후보(저비용 우선):
- **à-trous 공간 디노이즈 강화**: roughness 스케일 반경·깊이/법선 가중 edge-stop 강화 → 노이즈↓(레이 수 유지).
- 또는 **cone→GDF-mip**: 거친 로브를 GDF 반사 mip 피라미드의 cone LOD로 싸게 근사(레이 1개 + 와이드 콘).
- 파라미터/토글을 seam 규칙대로. 본질적 한계(48³ SDF blob)는 Phase 3(B3 클립맵/고해상)로 분리.
- 검증: rt-compare 잔차(금속 영역) ↓, 코스트 측정(P11 타이밍), DX≡VK 유지.

## 진행
Phase 1 → 측정/승인 → Phase 2 → 측정. 각 Phase는 기본 on·env 폴백으로 commit.
