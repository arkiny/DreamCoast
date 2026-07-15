# Interior-bias 근본 조사 계획서 — lit 갭(+18.13/255)의 항별 귀속과 근본 수리

> 상태: **진행 중** (2026-07-15). 동기 =
> [phase-f6d-residual-decomposition-plan.md](phase-f6d-residual-decomposition-plan.md) §4 —
> coarse interior 잔차의 57%가 **bias +18.13**(균일 과밝음)로 고립됨. 이력: F6B가 최초 정량
> ("lit 영역에서도 ~18/255 밝음"), F6C가 raster-측 과밝음으로 확정(PT 무죄), GI-볼륨-누설
> 페이즈가 색은 정합시켰으나(B−R = PT) 휘도는 EV11 깊은 크롭 기준 ~2× 잔존.

## 0. 기지 사실 (leak §1b·§13b·F6D — 재측정 금지, 단 §2의 재계측 예외 명시)

- EV11 항별 크롭 분해(leak §1b, tint 0.3 시대): **tint 플로어 ~33** ≫ 앰비언트 스페큘러
  ~5–7(청색) ≫ 잔여 13.9 vs PT 18.24 — tint=0이면 휘도는 PT와 거의 일치하나 색이 깨짐
  (청색 스페큘러 초과 + 따뜻한 E 부족의 상쇄 노출).
- 현 coarse 기본(tint 0.2): bias **+18.13**, B−R −0.68(앵커 −0.58 근방). tint 0.1/0.15 스윕은
  leak §13b에서 0.2 대비 열세(색 앵커) — **새 물리 없이 tint 재스윕만 반복하는 것은 금지 유지**.
- fine 스택은 tint 0.15로 bias +10.67 — 따뜻한 근거리 E 공급이 tint 하향을 연 선례(=이 페이즈의
  가설을 지지: 색 균형의 청색 항을 소스에서 수리하면 tint가 더 내려간다).
- 스페큘러 관련 기각 이력(leak §13b): `P_CACHE_OCCL_ROUTE`(+0.02)/`P_REFLECT_SKYFILL=0`(+0.41)/
  `P_SPEC_OCCLUSION=1`(+0.05) — 전부 **AE 비결정 시대의 masked_avg 델타**. §2의 bias-지표
  재계측은 새 계측기(결정론 AE + 부호 있는 bias)로 **다른 질문**(항의 bias 몫)을 재는 것이므로
  기각 반복이 아니다 — 단, 재계측 결과가 다시 무효면 그대로 기각 확정.

## 1. 가설 (사전 기록)

interior bias의 구성: **(a) tint 플로어**(의도된 크러치 — 지배 몫 예상), **(b) 무차폐 앰비언트
스페큘러**(prefilter 큐브 경로 — 반사 V-게이팅(leak §14a)은 SW-RT 폴백만 다뤘고 pbr의 큐브
스페큘러는 `P_SPEC_OCCLUSION`(기본 off) 뒤에 잔존), **(c) E/직사 잔여**. 근본 수리 경로 =
(b)를 소스에서 차폐 → 색 균형의 청색 항 축소 → tint 하향 여지(=bias 제거)가 색 앵커를 지키며
열림. (a)를 직접 깎는 것(tint 하향 단독)은 색 붕괴로 기각 확정 상태.

## 2. R1 — bias 귀속 행렬 (게이트 레시피 같은-설정 쌍, env 전용)

각 행 = interior raster+PT 쌍(같은 env, F4B 함정 규약) → bias/scatter/B−R. coarse 스택:

| 행 | env | 재는 것 |
|---|---|---|
| A (기지) | 기본 | +18.13 / 28.19 |
| B | `P_SKYVIS_TINT=0` | tint의 bias 몫(= A−B) + 색 붕괴의 현재 크기(V-게이팅 랜딩 후) |
| C | `P_SPEC_OCCLUSION=1` | 큐브 스페큘러 차폐의 bias 몫(새 계측기 재계측) |
| D | `P_SKYVIS_TINT=0 P_SPEC_OCCLUSION=1` | (b) 수리 가정 하 tint=0의 색 잔차(가설의 직접 검증) |

판정 분기: D의 B−R이 앵커(−0.58±0.3)에 들고 bias가 크게 죽으면 → §3의 소스 수리 설계로.
D가 여전히 청색(+1 이상)이면 (b) 외의 청색 운반자 수색(코드 조사 확대). C가 재차 무효(±0.1)면
`P_SPEC_OCCLUSION` 메커니즘 자체의 커버리지 조사(적용 픽셀 모집단 확인 — DEBUG 프로브).

## 3. 근본 수리 (R1 판정 후 설계·별도 커밋)

후보(코드 조사로 확정): pbr.slang 프리필터-큐브 앰비언트 스페큘러의 sky-vis 차폐(기존
`spec_occlusion` 경로의 커버리지/강도 수리 또는 V-게이팅 일반화 — 반사 트랙과 단일소스).
게이트: budget 쌍(sunlit 28.25/interior 31.91, coarse 기본 개선 기대) + **bias 하향 추적**
(섀도 지표) + B−R 앵커 + 셔머 + 갤러리 바이트 + 풀 매니페스트. 개선 랜딩 시 tint 재캘리브는
**후속 커밋**(F6C 방법론 — 단, R1-D가 근거를 제공할 때만; 무근거 재스윕 금지 유지).

## 4. 비목표

PT 수정 / tint 단독 재스윕(색 근거 없이) / fine 기본 ON 재시도 / masked_avg 게이트 정책 변경
(F6D 섀도 유지) / budget 상향.
