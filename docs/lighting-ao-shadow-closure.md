# 라이팅·AO·그림자 트랙 종결 — 계획 + 총체 검증 (2026-07-25)

3주간의 GI-fidelity/재검증 웨이브를 **명시적 종료 조건 3개**로 수렴시켜 닫는다.
HEAD `5eb3691` (F1/F6 Windows 재검증 배치 직후), RTX 2070 SUPER, release.

## 0. 종료 조건 → 정량 게이트

| # | 종료 조건 (사용자) | 정량 게이트 | 판정 도구 |
|---|---|---|---|
| G1 | 리얼타임 60fps (Lumen 참조) | `sponza_intel` 1080p Med, 3앵글(door/nave/atrium) × DX·VK 전부 **gpu_total ≤ 16.0 ms** (60fps + 여유) | `tools/perf-profile.py` (신규, 구 scratchpad measure.py의 정식화) |
| G2 | Lumen급 품질 | PT block64 잔차 래칫 예산 유지: **sunlit ≤ 20.74 / interior ≤ 27.21** + 갤러리 앵커 불변 + 콘텐츠 DX≡VK 기지 클래스 내 | `tools/golden-image.py` |
| G3 | 이동시 깜빡임/픽셀/파이어플라이 없음 | **신규 셔머 게이트**: 문 뷰 CAPTURE_SEQ 시퀀스의 인접 프레임 diff — 정지 ROI flicker ≈ 0 수렴, 이동(돌리) ROI flicker가 수정 후 기준선 대비 대폭 감소 + 밝은 픽셀 개수 안정 (수치는 측정 후 이 문서에 래칫으로 고정) | `tools/seq-stability.py` (신규) |

G3의 게이트 신설은 F6O가 명시한 후속("티어-res 셔머/HF 품질 게이트 신설 → `P_TAAU_ANTIFLICKER` 기본 ON 재심")을 그대로 수행하는 것.

## 1. 리콘 요약 (2026-07-25, 5-리더 병렬 조사)

- **퍼포먼스**: 1080p 60fps 실측은 Crytek Sponza 기준(79282e9, DX~15.8/VK~16.5 ms).
  **`sponza_intel`은 GPU 프로파일 이력이 전혀 없음** → 본 배치에서 최초 측정.
  Windows Med = native 1.0 스케일(TAAU/FXAA **비활성**), ssao off, ao_res_div 2,
  gi_res_div 3, gi_volume_period 4. 최대 미사용 레버 = **render_scale<1 + TAAU**
  (Apple 티어는 이미 0.67×+TAAU로 출시 중 — Windows Med만 네이티브).
- **문 깜빡임**: F6O 계획 문서가 동일 증상을 이미 규명 — "문 스페클의 정체 =
  서브픽셀 하늘-틈의 프라이머리 비저빌리티(TAAU 지터/무-AA에서 커버리지 플립)".
  대응 노브 `P_TAAU_ANTIFLICKER`(luma 안티플리커, bit2)는 랜딩됐으나 게이트 부재로
  기본 OFF. 추가 용의: SSR 미러 lit-history 피드백 셔머(`ssr_history_clamp=0`,
  코드 주석에 기지 이슈로 명기), AE wall-clock dt 펌핑, 확률적 GDF 반사 AR(1) wiggle.
- **품질**: F6 아크 봉인 상태로 PT 게이트 PASS (sunlit 20.44 / interior 26.91).
- **잔존(범위 밖 선존)**: 갤러리 DX≡VK 0.004/ch(선존, 별도 추적), 백엔드별 수렴
  스케줄 차이(인터랙티브 첫 수 초), aniso16 콘텐츠 발산(문서화된 예외).

## 2. 측정 매트릭스

### 2a. 퍼포먼스 (G1)
`sponza_intel` 1080p, 앵글 3종 — door(7,2.2,0→20,2.2,0)/nave(-14,2,0→14,2,0)/
atrium(0,2,0→-12,9,0), DX+VK, native 및 0.6667×TAAU 레버 A/B.
62프레임 정상상태 평균(settle 60), 패스별 상위 비용 포함.

### 2b. 문 깜빡임 판별 (G3) — 가설 H1~H6 A/B
문 뷰 고정, `CAPTURE_SEQ=48`, 정지(STEP=0) + 돌리(CAM_EYE 7→9.5) 변형:

| 런 | 토글 | 판별 대상 |
|---|---|---|
| f1 기본 | — | 기준선 |
| f2 | `AUTO_EXPOSURE=0` | H1 AE dt 펌핑 (전화면 diff 급감 여부) |
| f3 | `DEBUG_VIEW=1` | H2 프라이머리 비저빌리티 (알베도에서도 잔존?) |
| f4 | `P11_LEGACY_IBL=1` | H3+H4 SW-RT 반사 체인 일괄 |
| f5 | `P_SSR_HISTORY_CLAMP=1` | H4 SSR lit-history 피드백 |
| f6 | `P_REFL_CLAMP=0` | H3 확률적 GDF 반사 클램프 |
| f7/f8 | `RENDER_SCALE=0.6667` (±`P_TAAU_ANTIFLICKER=1`) | H2 처방 검증 |
| f10/f11 | VK 교차 | 백엔드 공통성 |

지표: 인접 프레임 avg/ch(전화면 + 문 ROI 0.44,0.38,0.56,0.72) + ROI 밝은픽셀
개수 표준편차(파이어플라이) + 템포럴 stddev 히트맵.

### 2c. 품질 (G2)
`golden-image.py --backend d3d12` / `--backend vulkan` 전 구성(PT 캐시 활용) +
갤러리 DX≡VK. 수정 랜딩 후 재실행으로 회귀 부재 확인.

## 3. 갭 수정 전략 — 측정으로 확정된 수정 웨이브

원칙: 근본 원인만, 씬-패치 금지, 노브는 티어 프리셋으로 (CLAUDE.md 엔지니어링 룰).

### 3a. G3 판별 측정 결과 (2026-07-25, tools/seq-stability.py, 문 ROI 0.44,0.38,0.56,0.72)

돌리(7→9.5m, 48프레임) 인접-프레임 diff avg/ch (전화면 / 문 ROI / ROI 밝은픽셀 std):

| 런 | 전화면 | 문 ROI | 밝은픽셀 std | 판정 |
|---|---:|---:|---:|---|
| f0 정지 기본 | 0.038 | 0.026 | 2.2 | 정지 수렴 상태는 이미 안정 |
| f1 돌리 기본 | 13.25 | 5.88 | 124.6 | 기준선 |
| f2 + AUTO_EXPOSURE=0 | **1.20** | 1.03 | — | **H1 AE 펌핑이 전화면 지배** (8fps에서 dt=130ms → 프레임당 적응 0.28 — 저fps가 증폭기) |
| f3 + DEBUG_VIEW=1(알베도) | 14.03 | **4.12** | — | **H2 프라이머리 비저빌리티 확정** (라이팅 없이도 잔존) |
| f4 + P11_LEGACY_IBL=1 | 13.72 | 7.57 | 42.0 | 반사 체인은 문 아티팩트의 주범 아님 |
| f5 + P_SSR_HISTORY_CLAMP=1 | 13.25 | 5.88 | 124.5 | H4 기각 (기준선과 동일) |
| f6 + P_REFL_CLAMP=0 | 13.02 | 6.52 | 137.4 | H3 부차적 |
| f7 + RS=0.6667(TAAU) | 7.47 | 4.19 | 186.6 | TAAU만으로 −29% |
| **f8 + RS=0.6667+ANTIFLICKER** | 7.25 | **1.96** | **53.9** | **처방 확정 (−67%)** |
| f9 정지 TAAU+AF | 0.079 | 0.168 | 13.0 | 정지 잔존 지터 미미 |
| f10/f11 VK 교차 | 12.72/7.31 | 5.72/**1.91** | 118.3/54.2 | 백엔드 무관 재현·처방 동일 |

### 3b. 퍼포먼스 근본 원인 (코드 판독으로 확정)

1. **reflect 히트-평가 = 히트당 O(num_cards) 서피스캐시 선형 스캔** — Windows 티어는
   `cache_grid=false`. gdf.rs:90 주석에 "30+ ms at 449 drawables / 2694 cards" 실측 명기,
   그리드는 "동일 결과 superset"으로 문서화(= vol-off 바닥 34.7ms의 본체). vol-on일 때는
   그리드 부재로 acceptance tolerance가 `t`에 비례해 무한 성장(gdf_reflect.slang:1288)
   → 원거리 히트에서 카드 수용 폭증.
2. **모든 GDF 마치 스텝이 128B 메시-SDF 헤더를 재로드** (8×Load4, RWByteAddressBuffer라
   컴파일러 호이스트 불가; mesh_sdf_sample.slang:57-89) — 레이당 96스텝 + 히트측
   normal 4회·shadow ≤48회 전부에 곱해짐.
3. **gi_volume**: 슬랩 디스패치는 정상이나 ① z-그룹 하한(4슬라이스)으로 period>8 무의미
   ② 셰이더에 슬랩 상한 가드 부재(gi_volume.slang:372) — 반올림 꼬리 스레드가 풀 트레이스
   ③ 언배치 배리어 32개/프레임이 이 타이머에 귀속 ④ 레이당 히트 셰이딩(normal 4×ms_geo +
   gv_shadow ≤32×ms_geo + SH 16탭)이 지배. spp에 완벽 비례(1→3.5ms, 4→9.4, 16→27),
   period 1→53.2 / 32→21.6(서브리니어 = 레이턴시 바운드).
4. **SH 볼륨이 12×R32F(+skyvis 4×R32F)로 분산** — 소비측 16탭이 4탭(RGBA 패킹, 비트-동일)
   으로 줄 수 있음. 인텔 스폰자가 Crytek보다 극단적으로 느린 건 셀당 인스턴스 후보 수(C)
   증가로 ms_geo 스텝 비용 자체가 커진 것.

### 3c. 수정 스테이지

- **P1 (티어 승격 + 안정성, 설정/소규모 코드)**: Windows Med를
  `render_scale 0.6667`(TAAU 활성) + `reflect_res_div 4` + `cache_grid true`(동일-결과
  문서화, DX≡VK는 본 배치에서 검증) + 신설 프리셋 필드 `taau_antiflicker=true`(F6O가
  요구한 셔머 게이트를 §3a로 신설 완료 → 기본 ON 재심 통과)로 승격. Apple 티어와 정렬.
  + AE 적응 dt를 1/30s로 클램프(히치/저fps 폭주 가드, ≥30fps 무변화; 스크린샷 모드는
  기존 FIXED_DT 유지). 갤러리는 hard-coded preset이라 전부 무영향(바이트 앵커 검증).
  quality.rs의 med 잠금 테스트는 본 문서를 근거로 의식적으로 갱신.
- **P2 (셰이더/RHI 무손실 최적화, 항목별 비트-동일 검증)**: ① 메시-SDF 헤더 호이스트
  ② gi_volume 슬랩 상한 가드 ③ (측정 후 필요시) SH 12×R32F→3×RGBA32F 패킹,
  d3d12 배리어 배칭.
- **P3 (P1+P2 재측정 후 잔여 갭 시)**: reflect_res_div 6(Apple 값), ssr_stochastic 등
  v2 스택 승격 — 각각 DX≡VK 게이트 후.

**do-not-retry 제약**: TAAU 내부 이웃-클램프 변형(neighbor-mean pull, flicker-adaptive
box) 재시도 금지(F6N/F6O 기각), 봉인된 sky-chain 재개 금지(F6I 원장), 바이너리-사인
전략군/해상도 증가 금지, `gi_res_div 4` 비-Apple 승격 금지(0.117/ch 파리티 기각 이력).

## 4. G1 측정 결과 (2026-07-25, 수정 전 기준선) — **불합격, 갭 확정**

`sponza_intel` 1080p native Med, 62프레임 평균 (gpu_total ms / fps):

| 앵글 | DX | VK |
|---|---:|---:|
| door | **132.6 / 7.5** | **125.1 / 8.0** |
| nave | 167.2 / 6.0 | 124.5 / 8.0 |
| atrium | 192.0 / 5.2 | 145.2 / 6.9 |

- 상위 비용: **gdf_reflect 89~148 ms** + **gi_volume 19~30 ms**. `P11_LEGACY_IBL=1`이면
  총 3.3 ms — 초과분 전부가 SW-RT 스택. 이 씬은 GPU 프로파일 이력이 없었고(문서 공백),
  GI-fidelity/F-웨이브의 비용이 Windows에서 한 번도 재측정되지 않은 것이 근본 배경.
  Mac은 phase-macos-perf 트랙이 같은 전투를 이미 치러 Apple 티어에 회복 스택을 실었으나
  (reflect_res_div 6·ssr_stochastic·ao_res_div 4·rs 0.5~0.67), **Windows Med 승격은
  "parity run 대기"로 미착수 상태였다.**

### 4a. door 뷰 분해 (DX native, gdf_reflect 93.3 / gi_volume 30.4 기준)

| 토글 | gdf_reflect | gi_volume | 총 | 해석 |
|---|---:|---:|---:|---|
| `P11_REFLECT_MAX_STEPS=32` | 49.4 | 27.5 | 85.7 | 마치가 스텝-바운드 (Mac과 달리 Windows는 반응) |
| `P_REFLECT_RES_DIV=4` | **22.9** | 27.5 | 58.1 | 스레드 수에 완벽 비례 — 최대 레버 |
| `P_GI_VOLUME=0` | **34.7** | — | 68.8 | **히트측 GI볼륨 평가가 ~58 ms** (gdf_gi는 27.9로 폭증 = 볼륨은 유지해야) |
| `P_GI_VOL_CLIP=0` (fine off) | 86.8 | 26.8 | 122.2 | fine 레벨 기여는 ~6.5 ms뿐 |
| `P_GI_VOLUME_PERIOD=16` | 87.5 | **24.2** | 120.5 | **gi_volume은 슬랩 수에 비례 안 함** — 고정 오버헤드(배리어/전환 의심, RenderDoc 필요) |
| `RENDER_SCALE=0.6667`(TAAU) | 43.7 | 26.6 | 77.0 | 내부해상도 레버 확인 |
| VK 교차 | door 94.5 / nave 89.2 | 19~20 | — | gdf_reflect는 **DX가 VK보다 느림**(역전, DX 병리 의심) |

## 4b. 수정 진행 로그

- **P1 랜딩** (Med 승격: rs 0.6667+TAAU / reflect_res_div 4 / cache_grid on /
  taau_antiflicker on + AE dt 1/30s 클램프): door 1080p **DX 132.6→41.9 ms,
  VK 125.1→36.8 ms** (3.2×). 갤러리 앵커 픽셀-불변 증명(P1 전후 diff 0.000 max4 =
  run-to-run 노이즈와 동일; DX SHA는 기지의 1-LSB 비결정론이라 계측기로 부적합).
  quality.rs med 잠금 테스트/그룹 스냅샷 의식적 갱신, 22/22 통과.
- **P2 랜딩** (① 메시-SDF 헤더 per-invocation static 캐시 — 마치 스텝당 8×Load4 제거,
  비트-동일 ② gi_volume 슬랩 상한 가드 — push clip.z(기존 여유 슬롯, 크기 불변)로 슬랩
  길이 전달, 반올림 꼬리 스레드 제거): door DX **gdf_reflect 10.3 ms** (P1 직후 대비
  추가 하락), gi_volume 26.6→23.1 ms. gi_volume이 잔여 벽(프레임의 ~60%).
- **P2b (측정된 부정 결과)**: 메시-SDF 헤더 static 캐시는 DX −3ms였으나 **VK +10ms
  회귀**(SPIR-V Private 스토리지 미승격/스필) → 리버트. 슬랩 가드+리셰이프는 유지
  (출력 불변: door VK 전후 0.006, 해당 레시피 run-to-run 노이즈 0.126 이내).
- **N-래더 (gi_volume 추정기 재배치)**: spp16×스레드1/probe가 레이턴시-바운드임을
  확정 — **동일 레이 예산**을 spp4·period2·dir_sets4(F6K 로테이션으로 각도 커버리지
  보존)로 재배치 시 gi_volume DX 22.5→**11.1** / VK →**8.5 ms**. 문 뷰 총합
  **DX 19.8 ms(50.6fps) / VK 20.5 ms(48.9fps)**. → Med 프리셋으로 승격
  (`gi_volume_spp`·`gi_dir_sets` 신설 + Apple 값 reflect_res_div 6/steps 56/cone_k
  0.06). 60fps 잔여 갭(~3ms)의 다음 레버: gi_volume 비동기 겹침(문서화된 Phase-E
  후속) + d3d12 배리어 배칭 + spp-병렬 리덕션.
- **Windows PT 게이트 첫 실측 (수정 전 HEAD): FAIL** — sunlit 22.70/예산 20.74,
  interior **43.66**/27.21 (bias +32.2). 단일-노브 A/B 전멸(LEGACY_IBL은 악화,
  SKY_GAIN 1/4에도 AE가 래스터 밝기 재고정) → 래스터↔PT **구조적 발산**으로 판정,
  Mac 교차 검증이 필요한 별도 조사 페이즈로 이관:
  **docs/phase-windows-pt-parity-prompt.md** (증거·배제표·판별 계획 완비).
  G2는 이 페이즈 종결 전까지 조건부-미충족으로 정직하게 보고.

## 5. 최종 판정표 (2026-07-25 최종 스위트: 빌드+22테스트+clippy 클린)

| 게이트 | 수정 전 (이 배치 첫 실측) | **수정 후 (Med 신규 기본값)** | 판정 |
|---|---|---|---|
| G1 DX door/nave/atrium | 132.6 / 167.2 / 192.0 ms | **19.6 / 25.9 / 26.4 ms** (51/38.6/37.8fps) | **6.8~7.3×, 부분 통과** |
| G1 VK door/nave/atrium | 125.1 / 124.5 / 145.2 ms | **18.5 / 24.5 / 24.6 ms** (54/40.8/40.6fps) | 〃 |
| G3 정지 셔머 (door ROI avg/ch) | 0.026 (native, 수렴 시) | **0.17 DX / 0.17 VK** (지터 잔존, 육안 무해) | **통과** |
| G3 돌리 플리커 (door ROI avg/ch) | **5.88** (bright-std 124.6) | **2.06 DX / 2.01 VK** (bright-std ~54) | **통과 (−65%), 래칫 확정: ROI ≤ 2.2 / bright-std ≤ 60** |
| G2 갤러리 앵커 | 0.004/ch 잔차(선존) | **전 변경 픽셀-불변** (0.001 max4 = run-noise) | 통과 (선존 잔차 별도) |
| G2 DX≡VK 콘텐츠 (door w256) | — | 2.47/ch (>8: 5.2%) — 기지 클래스(aniso16+수렴 과도) 이내 | 통과 (문서화된 예외 클래스) |
| G2 PT sunlit (DX) | 22.70 (예산 20.74) | 23.65 (티어 변경 +0.95; `P_REFLECT_RES_DIV=4`로 −복구 가능) | **조건부 미충족 → 발산 페이즈** |
| G2 PT interior (DX/VK) | 43.66 (예산 27.21) | 43.77 / 47.56 | **조건부 미충족 → 발산 페이즈** |

**G2 발산 페이즈의 결정적 신규 데이터**: VK-PT vs DX-PT 교차 diff가 interior에서
**5.20/ch (>8: 25.7%)**, sunlit 1.17/ch — 실내에서 Windows의 두 RT 백엔드 PT 레퍼런스
자체가 서로 발산(pt_black 24.1/25.8%). PT 레퍼런스의 실내 신뢰성부터 의심하라
(H-A 강화). 상세: docs/phase-windows-pt-parity-prompt.md.

## 6. 남긴 것 (정직한 잔여 목록)

1. **60fps 잔여 갭**: nave/atrium 38~41fps (door는 51~54). 다음 레버(문서화):
   gi_volume 비동기 겹침(뷰-독립·핑퐁 구조상 자연스러운 async 후보, Phase-E 후속),
   d3d12 배리어 배칭(32개/프레임 언배치), gi_volume spp-병렬 리덕션, vgeo 각도별 비용.
2. **G2 PT 발산 페이즈** (docs/phase-windows-pt-parity-prompt.md) — Mac 교차 검증 필수.
3. 선존 갤러리 DX≡VK 0.004 잔차(F1 배치 §5), 백엔드별 수렴 스케줄 차 — 종전 문서 그대로.
4. Metal에서 이번 배치 재검증 (Med는 non-Apple 티어라 Apple 출력 불변이어야 하나,
   gi_volume 리셰이프/가드는 공용 셰이더 — Mac 골든 러너로 확인 필요).
5. 헤더 호이스트의 교차백엔드-안전판(값 전달식) — 측정된 부정 결과 기록 있음(§4b).

## 7. 설계 결정 — 프리즈 금지 (2026-07-25 사용자 지시, §5b 리버트)

웨이브 2의 gi_volume 수렴-프리즈는 랜딩 직후 **설계 지시로 리버트**(`ea22a93`):
동적 요소가 많은 게임 특성상 "무변화 감지" 기반 래치는 스테일-GI 오류 위험이 있어
**앞으로 프리즈 계열 접근을 금지**한다. 레퍼런스 엔진과 동일하게 **고정 예산 상각 +
비동기 겹침**이 지속 60fps의 정공법이며(§6-1), 별도 페이즈로 진행한다. 참고: 기존
서피스 캐시 릴라이트 프리즈(`cache_frozen`, 선행 출시)도 같은 지시의 재검토 대상 —
제거 시 릴라이트+밉젠 상시 비용이 돌아오므로 상각/async 페이즈와 함께 다룰 것.
§5b의 프리즈 수치는 기록으로만 남긴다(현행 아님). 현행 확정 수치 = §5 판정표
(door 51~54fps, nave/atrium 38~41fps; settle 260 기준 door DX 60.1fps).

## 8. 신규 개방 항목 — TAAU 이동 블러 → **종결 (2026-07-25)**

사용자 보고: 이동 중 TAA 블러 과다. 진단·계획: **docs/phase-taau-motion-sharpness-prompt.md**.
**결과: docs/taau-motion-sharpness.md** — 빌리니어 히스토리 리샘플이 주범이라는 진단은
확정, Catmull-Rom(12탭 코너컷) 랜딩. 이동 그래디언트 에너지 **+43% DX / +40% VK**,
지연 잔차 −5.7%, 정지 이미지도 개선.

**본 문서 §3a 래칫에 대한 영향**: 돌리 door-ROI flicker **1.92 → 2.18 (DX) / 2.08 (VK)**
로 래칫 **≤ 2.2 유지**, bright-std **47 (≤ 60 유지)**. ROI flicker 상승분은 셔머가 아니라
선명도 오염(같은 시차라도 선명하면 인접 diff가 커진다) — bright-std 하락 + 정지 flicker
개선(0.218→0.167) + 디테일당 flicker 11% 개선으로 뒷받침. 향후 래칫은 선명도 불변인
**디테일당 flicker**로 승계 권고(§7 of the result doc).

**계획의 처방 B(속도-게이트 블렌드 정책)는 측정으로 기각** — TAA-**U**에서는 히스토리가
재구성 그 자체라 이동 중 히스토리를 자르면 더 흐려진다(grad 2.88→2.74, 지연 잔차
16.1→20.9). 안티플리커/clamp-expand의 이동 중 해제도 역효과: **본 문서 §3a의 문-ROI
수치는 전부 돌리 측정**이므로 그 두 안정화기의 실측 효용은 정지가 아니라 **이동 중**에
있다. 재시도 금지 목록에 추가.
