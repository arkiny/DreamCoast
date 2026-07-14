# GI 볼륨 누설/sky-vis 필드 조사 시작 프롬프트 — 깊은 차폐부 잔여 갭

새 Claude Code 세션(맥, Metal)에 아래를 그대로 붙여넣는다. 이건 skylight tint 캘리브레이션(`04b26af`)이
남긴 **깊은 차폐 영역 잔여 갭(raster ~86 vs PT ~22)**의 근본 원인 작업이며, tint 추가 하향(0.3→0.2대)의
선행 조건이다.

---

## 붙여넣을 프롬프트

DreamCoast 엔진(`/Users/arkiny/GitRepos/DreamCoast`, Rust, RHI over Vulkan/D3D12/Metal)에서 **GI 볼륨
누설·sky-vis 필드 품질 조사**를 시작한다. 먼저 `git fetch origin && git checkout main && git pull`로
최신 main을 받고(기준 `04b26af` = skylight tint 0.5→0.3 PT-캘리브레이션), 새
`feature/gi-volume-leak` 브랜치를 판다.

### 왜 (측정된 갭)
skylight tint 캘리브레이션이 lit-영역 게이트를 크게 낮췄지만(sunlit 28.85/interior 32.53), **깊은 차폐
영역은 tint와 무관하게 raster가 PT의 ~4배 밝다**: interior 카메라(`CAM_EYE=-14,2,0 CAM_TARGET=14,2,0`)
왼쪽 기둥 크롭(x 0–900) 평균 luma **raster ~86 vs PT ~22** — tint 0.5→0.2에서도 86.3(불변 실측).
이 잔여가 lit-마스크 경계 안쪽까지 오염시키는 것이 interior 게이트가 sunlit보다 높은 이유다.

### 진단 (이미 확인됨 — 반복 조사 금지)
- **PT 레퍼런스는 무혐의**(F6C 감사, `92242a1`): 바운스 절단 ≤0.25/255(8→32 스윕), 하늘 단일소스 정합.
  깊은 차폐부의 PT ~22가 물리 정답이다.
- **tint 무관**: `P_SKYVIS_TINT` 0.5→0.2에서 해당 크롭 86.5→86.3(AE가 전역 보상). 기본은 이제 **0.3**
  (PT 색-캐스트 B−R≈−1.0 정합점 — 추가 하향은 이 작업이 선행 조건).
- **기각 실험(반복 금지)**: bent-normal off(`P_BENT_NORMAL=0`) 무효 / F4 fine 볼륨(`P_GI_VOL_CLIP=1`)
  게이트 동률·시각 무효 / surface-cache GI feedback(`P11_SURFACE_CACHE=1`) 중립 ±0.02 / WRC
  escaped-ray 부활 금지(main.rs:3103 측정 판정) / `gi_importance`는 볼륨 경로에 inert.
- **콘텐츠 기본 GI = `gi_volume`(DDGI-lite)**: 32³ 씬-고정 SH-L1 + sky-vis SH 4종, EMA α=0.1, period 4,
  probe당 `gi_spp`(Apple 4) uniform-sphere 레이. 소비 `gdf_gi.slang:151-218`(vol 분기).

### 용의자 (코드 근거 — 이 순서로 검증)
1. **프로브 벽 관통 누설(최우선)**: `gdf_gi.slang` vol 분기와 `gi_volume.slang` `read_coeffs`는
   **가시성/점유 가중 없는 하드웨어 트라이리니어** — 벽 안/뒤 프로브의 SH가 실내로 스민다. **선례가
   이미 리포에 있다**: `gdf_reflect.slang:294-339` `sample_gi_irradiance_valid`가 occupancy-가중
   수동 트라이리니어(프로브 중심 `scene_occ > 0`만 혼합)를 반사 폴백에 구현해놨다(cache_tile bit13
   opt-in). 이 패턴을 GI 소비(vol 분기)와 볼륨 자체 멀티바운스 read에 포팅하는 것이 1차 증분 후보.
   비용 주의: 수동 8-corner 태핑 ×12 SH — Load 기반이라 태핑당 1페치, PROFILE_GPU로 측정.
2. **`sky_fill` 항 과대**: `gi_volume.slang:209-210` 히트 셰이딩의 `procedural_sky(hn,…)×(0.5+0.5·hn.y)`
   — 차폐 깊숙한 히트에도 하늘이 절반 이상 가중된다(히트점의 실제 sky-vis 미반영). 히트점의 이전-프레임
   sky-vis SH로 이 항을 오클루전하는 것이 후보(단일소스 — 새 상태 신설 금지).
3. **sky-vis V(n) 필드 품질**: tint 하향을 막았던 blue-cast의 근원. 1·2가 정리된 후 재평가.

### 프로세스 (프로젝트 규칙 — 반드시 준수)
1. **계획서 먼저**: `docs/phase-gi-volume-leak-plan.md` 작성(용의자별 접근·게이트·비용) → 사용자 승인 →
   구현. 승인 전 코드 변경 금지(측정 프로브는 예외, 커밋 금지).
2. 착수 전 읽기: `gi_volume.slang`(전체), `gdf_gi.slang:151-218`, `gdf_reflect.slang:255-345`,
   `gi.rs record_gi_volume/record_gi`, `pbr.slang:125-200`(skylight 오클루전 소비), CLAUDE.md.
3. 검증된 단일 커밋(부분 단계 가능), 승인된 계획서 커밋 포함.

### 측정 규약 (F6B/F6C에서 확립 — 어기면 측정 무효)
- 캡처 레시피: `LEVEL=sponza_intel AUTO_EXPOSURE=1 RENDER_SCALE=1 WARMUP_FRAMES=192` + 고정 카메라.
  **EV100 설정 금지**(고정노출 함정). 게이트: `python tools/golden-image.py --only sponza_pt_sunlit
  --only sponza_pt_interior` — budget **sunlit 29.15 / interior 32.83**(개선 시 하향 재기준선).
- **AE 커플링 함정**: knob 스윕의 행간 비교는 masked_avg 절대치만 보지 말 것 — 자동노출이 디스플레이
  공간 지표를 함께 움직인다. **lit_mean 비율(raster/PT, 목표 1.0)·크롭 B−R(PT ≈ −1.0)** 병용.
- **깊은 차폐부 전용 진단**: interior 캡처의 x 0–900 크롭 평균 luma — raster 86 → PT 22 방향으로
  내려가는지가 이 작업의 성공 지표(게이트와 별도로 추적).
- 갤러리 앵커 `65d04ceca2c4dbff` 불변(콘텐츠 전용 seam 필수), run-to-run 재현, DX≡VK Windows 배치
  추가(동결 중), 상표명 금지, PROFILE_GPU 비용 보고.

### 리스크
occupancy-가중 혼합은 프로브가 전부 벽 안일 때 0으로 크러시(gdf_reflect 선례는 "genuinely dark" 수용)
— 실내가 과도하게 어두워지면 시각 회귀. sky_fill 오클루전은 볼륨 수렴(EMA 부트스트랩)과 상호작용 —
첫 프레임 reset 경로 확인. 둘 다 콘텐츠 opt-in seam(`P_GI_VOL_VALID=1`류) 뒤에서 측정 후 기본 편입.

### 참고
`docs/phase-f6c-pt-reference-audit.md`(감사+캘리브레이션 전체 표), `docs/phase-f6b-content-pt-residual-plan.md`
(게이트 규약), `docs/phase-f4-hierarchical-radiance-cache-plan.md`(볼륨 구조·선행 판정), 메모리
`dreamcoast-f6c-pt-audit`·`dreamcoast-f4-fine-volume-level`.

---

## 참고: 직전 세션 상태 (2026-07-14)
- `origin/main` = `04b26af`. 이번 이틀 랜딩 체인: `6b6b6bf`(F6B 잔차 자동화) → `52423b2`+`41d1923`
  (F2 f16+비등방, 아틀라스 71→28.6MB) → `fbfd4e6`(F4 fine 볼륨 opt-in) → `92242a1`(F6C 감사) →
  `04b26af`(tint 0.3 캘리브레이션, budget 29.15/32.83).
- 전역 미결: DX≡VK Windows 재검증 배치(동결) — F1 전 스테이지, aniso, R16Float, P_GI_VOL_CLIP,
  pt config 2종, tint 상수. F4 후속(반사 fine 폴스루·edge-fade·재중심)은 이 작업 뒤 재평가.
