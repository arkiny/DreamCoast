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

## Phase 2 — GDF GI 과밝음 수정 ✅ (데이터가 재정의)
조사 중 발견: 크롬 뷰 PT 잔차의 최대 원인은 반사가 아니라 **디퓨즈 GDF GI 과밝음**(아보카도 +60/ch).
원인: `gdf_gi.slang` `trace_bounce`의 바운스 표면 재조명이 Lambertian **`/π` 누락** — pbr 직접광은
`albedo/PI*radiance*ndl`인데 바운스는 `albedo*(...)`로 ~π(3.14×) 과밝음. (1차 표면은 코사인 샘플링으로
π가 상쇄되어 정상.) 수정: 재조명에 `* (1/PI)`.
- 결과: 아보카도 diff (60,22,0)→(7,-10,-16). 크롬 뷰 9.14→**6.77/ch**(off>32 9.24%→5.13%),
  기본 씬 6.90→**6.56/ch**, DX≡VK **0.0010**(게이트 통과). 시각: diff에서 아보카도 글로우 소거.
- 남은 잔차 지배 = 크롬/글로시 **반사 지오메트리**(저해상 48³ SDF blob) → Phase 3.

## Phase 3 — 반사 재조명 1/π (저비용 완화) ✅
사용자: "저비용 완화 정도로". GI와 같은 `/π` 누락이 반사 경로에도 있었음:
- `sdf_cache_light.slang`: `albedo*(direct_irr + indirect)` → direct 항만 `/π`(indirect는 코사인샘플 π 상쇄).
  reflect_cache(기본 on)가 이 캐시를 반사에 먹이므로 메시 객체 반사가 π× 과밝았음.
- `gdf_reflect.slang` 미스 폴백(지면 반사): `albedo*(sun*ndl*sh+fill)` → `*1/GGX_PI`.
- 결과: 반사된 디퓨즈 표면이 직접 본 모습과 일치. **기본 씬 6.56→6.26/ch**(off>32 2.31→1.68%),
  DX≡VK 0.0003. 크롬 뷰는 하늘 반사 지배라 변화 작음(혼합). 시각 회귀 없음.

## 남음 (큰 작업, 보류) — Glossy/chrome 반사 지오메트리
크롬(rough 0.08)·글로시가 저해상 48³ SDF blob이라 PT의 정확한 반사 형상과 차이(copper 뷰 잔차 지배).
근본은 B3 고해상/클립맵 SDF(비용 큼) — 게임 최적화 우선이라 **보류**. à-trous/cone-mip은 노이즈용인데
현 GGX+temporal이 이미 글로시 노이즈를 해소해 효과 작음 → 현재로선 비용 대비 이득 낮음.

## 진행
Phase 1 → 측정/승인 → Phase 2 → 측정. 각 Phase는 기본 on·env 폴백으로 commit.
