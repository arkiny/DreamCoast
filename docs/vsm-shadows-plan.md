# Virtual Shadow Maps (VSM) — UE 차용 계획

상위: [render-pipeline-reference.md](render-pipeline-reference.md) · PR-7 CSM([shadow-reflection-quality.md](shadow-reflection-quality.md))의 후속.
레퍼런스: UE5 소스 `D:/Repositories/UnrealEngine-1` —
`Engine/Source/Runtime/Renderer/Private/VirtualShadowMaps/*`,
`Engine/Shaders/Private/VirtualShadowMaps/*` (아래 인용은 전부 이 루트 기준).

## 0. 왜 (2026-08-01 던전 플레이테스트 배치의 결론)

M3 던전 웨이브 검증에서 그림자 트랙이 드러낸 문제들이 전부 **"한 장의 맵을 통째로
다루는" 구조의 한계**로 수렴한다:

| 증상 | 근본 원인 | 세션에서의 임시 처방 |
|---|---|---|
| 동적 오브젝트 그림자 동결 | S1 섀도 캐시가 맵 전체 단위로 freeze/재렌더 (`skin\|\|morph`만 동적 판정) | 에폭에 전 캐스터 transform 해시 → **움직임 = 맵 전체 재렌더** (비쌈) |
| 벽 그림자 부유(25→70cm) | 단일맵/CSM 모두 커버 볼륨에 비례하는 NDC bias | 텍셀-비례 bias + CSM_NEAR/FAR 트림 (커버리지 계약이라는 새 함정 생성) |
| 해상도 12cm/텍셀 | CSM 분할이 탑다운 카메라와 부정합 | 분할 범위 수동 튜닝 (씬마다 재튜닝 필요) |
| 벽-바닥 라이트리킹 | 섀도맵 텍셀이 실루엣 경계에 걸침(스트래들) | 미해결 (bias·normal-offset 무효) |

UE VSM은 이 네 가지를 구조로 해결한다: **화면이 실제 참조하는 128²-텍셀 페이지에만**
고해상 텍셀을 배분하고(분할 튜닝 소멸), 페이지를 **정적/동적 2레이어로 캐시**하며
(freeze 아님 — no-freeze 지시 준수: 모든 상태 변화가 명시적 무효화를 생성),
**움직인 오브젝트가 겹치는 페이지만 무효화**하고(맵 전체 재렌더 소멸), 바이어스를
전부 **리시버 유도**로 대체한다(월드 상수 소멸, 스트래들 보정 내장).

## 1. UE 구조 요약 (차용할 것)

수치·메커니즘의 1차 출처는 세션 리서치 노트(`docs/research/vsm_{arch,cache,clip}.txt`,
UE 파일:라인 인용 포함). 요지:

- **가상 주소 공간**: VSM 1장 = 16k² 가상 = 128×128 페이지 × 128² 텍셀, 뮵 8레벨.
  방향광은 클립맵 — **레벨 L이 반경 2^(L+1)을 커버하는 독립 VSM** (UE 기본 L6..22),
  리시버의 레벨 선택은 `floor(log2(거리) + 해상도 LOD bias)` — 화면 풋프린트가
  자동으로 텍셀 밀도를 결정한다 (CSM 분할식의 상위호환).
- **페이지 테이블**: R32_UINT 텍스처, 엔트리 32비트 = 물리주소 10+10b + LODOffset 6b
  + 플래그 3b. `PropagateMappedMips`가 미매핑 엔트리를 상위(클립맵은 더 거친 레벨)
  물리 페이지 포인터로 채워 **샘플링은 항상 PT 1회 페치** — 페이지 폴트 루프 없음.
- **물리 풀**: R32_UINT Texture2DArray, depth를 `asuint`로 `InterlockedMax` (reverse-Z).
  슬라이스 0 = 동적(=최종), 슬라이스 1 = 정적 캐시. UE 기본 2048페이지 = 슬라이스당 128MB.
- **페이지 마킹**: G-버퍼 깊이에서 픽셀당 1스레드로 "필요 페이지" 플래그 store
  (아이덤포턴트). 페이지 경계 5% 안이면 대각 이웃도 마킹(딜레이션).
- **수명주기**: 물리 페이지별 메타데이터 + 4리스트(LRU/AVAILABLE/EMPTY/REQUESTED)로
  캐시-aware 할당; 재요청된 캐시 페이지는 PT 엔트리만 다시 씀(재렌더 0); 미요청은
  1000프레임까지 LRU 생존. 풀 고갈 = 조용한 coarse 폴백(오버서브스크립션).
- **무효화(이번 도입의 핵심)**:
  - CPU: 프리미티브 add/remove/transform 변경 시 **구/신 풋프린트 둘 다** 인스턴스
    레인지로 수집(GPUScene 업데이트 전/후 2회) → GPU 디스패치.
  - GPU(`InvalidateInstancePages`): 인스턴스 AABB → 섀도 뷰 절두체 컬 → 페이지 렉트
    → 계층 플래그 조기 탈출 → (옵션) 섀도 HZB 오클루전 → 겹친 페이지에
    `INVALIDATE_STATIC/DYNAMIC` `InterlockedOr` (**전 프레임** request-flags 텍스처에 —
    PT에 없는 캐시 페이지도 맞도록). 다음 프레임 수명주기가 이를 UNCACHED로 변환.
  - **정적/동적 분류 이력**: 무효화당한 프리미티브는 동적으로 승격(동적 레이어에만
    매 프레임 그림, 밑의 정적 레이어는 캐시 유지 — `InitializePhysicalPages`가 정적
    슬라이스를 복사해 시드, 렌더 후 `max()` 머지), **100프레임 조용하면 정적으로
    강등**(FORCE_STATIC 무효화 1회 동반). 래치가 아니라 명시적 무효화 프로토콜.
  - **클립맵 스크롤**: 레벨 원점을 레벨 반경(=32페이지) 단위로 스냅, 캐시 페이지는
    페이지-공간 오프셋으로 슬라이드(카메라 이동 ≠ 무효화). Z는 ±R×1000 가드밴드에
    핀 + 캐시 센터 기준 ZOffset 재보정. 태양 방향 변경 = 라이트 통짜 무효화.
- **바이어스 스택(전부 리시버 유도, 월드 상수 0)**: ① 노멀 바이어스
  `max(0.02, 0.5·거리/cotHalfFOV)`, ② 스크린-레이 프리마치(0.015·depth, 4탭 — 자체
  깊이버퍼로 "스마트 바이어스" 시작점), ③ **optimal slope bias** — 리시버 평면
  노멀을 UV-노멀 행렬로 변환해 `2·max(0, dot(DepthSlopeUV, 페치된 텍셀 중심까지의
  오프셋))` — **텍셀-스트래들을 수학으로 보정**(우리 리킹의 직접 처방), ④ 텍셀 디더.
- **SMRT 필터**: PCF 대신 태양 디스크로 7레이 × 8샘플 섀도맵 레이마치(리시버 쪽
  조밀 2차 간격 = 컨택트 하드닝), 실루엣에서 DepthHistory 기울기 외삽(클램프 5.0)이
  리크 가드. 웨이브 투표로 umbra/lit 조기 탈출.

## 2. DreamCoast 설계

### 스코프/치수 (UE 대비 축소)

- **M-스코프: 태양 방향광 1개만.** 포인트(횃불)는 클러스터드 직접광 무그림자
  유지(v1 수용 사항), 큐브면 VSM은 후속.
- 단위: UE는 cm, 우리는 m. 클립맵 레벨 반경 2^(L+1) m — 던전(카메라고 14m, 시야
  ~60m)엔 **L=1..6 (반경 4..128m) 6레벨**이면 충분. 레벨당 가상 8k²(=64페이지 그리드,
  UE의 절반)로 시작 — L1 텍셀 1mm, L5(반경 64m) 텍셀 16mm. 페이지 테이블/플래그
  레이아웃은 UE 인코딩 그대로(10+10+6+플래그), 규모만 축소.
- 물리 풀: **512페이지 × 2슬라이스 = 64MB** 시작(오버서브 통계 로그로 조정).
  R32_UINT storage image + `InterlockedMax`; **Metal은 텍스처 아토믹 제약 시
  버퍼-백업 풀로 대체**(주소 계산 동일, 리스크 §4).

### 패스 배치 (기존 렌더그래프 위, 전부 compute + 1 raster)

```
G-buffer 후:
  vsm_mark        픽셀→클립맵 레벨 선택→request-flags store (+ 5% 딜레이션)
  vsm_invalidate  (프레임 초두, CPU 수집 인스턴스 레인지) AABB→페이지 렉트→InterlockedOr
  vsm_update      물리 페이지 수명주기 (재사용/무효화 반영/에이징, 4리스트)
  vsm_alloc       신규 요청 페이지 할당 + PT 기록
  vsm_hier        계층 페이지 플래그 OR-업 + uncached 페이지 렉트
  vsm_propagate   미매핑 PT → 거친 클립맵 레벨 포인터
  vsm_init        렌더 대상 페이지 클리어/정적 슬라이스 복사 (indirect)
  vsm_raster      캐스터를 뮵-뷰당 1회 래스터, PS가 가상→물리 변환 + InterlockedMax
                  (컬링: cull.rs 확장 — 인스턴스 렉트 vs 계층 플래그 2×2 개더)
  vsm_merge       STATIC_DIRTY 페이지: slice0 = max(slice0, static) (indirect)
라이팅(pbr.slang):
  sun_shadow → VSM 경로: PT 1페치 + 물리 샘플 + 리시버 바이어스 스택 (V1은 PCF 유지)
```

### 무효화 배선 (사용자 지시의 핵심, 우리 지형에 맞춤)

- 우리 무버는 **리지드 노드 애니**(스킨 없음) — UE의 GPU deforming-bit 기계 전체를
  건너뛰고, **S1 에폭 수정에서 이미 깔린 CPU transform 추적**을 인스턴스 레인지
  수집기로 승격한다: 프레임당 `(SceneObject 인덱스, 구 transform, 신 transform)`
  변경 리스트 → 구/신 AABB 각각 GPU 무효화 디스패치(UE의 pre/post 2회와 등가).
- `casts_shadow` 오브젝트별 `cache_as_dynamic` 비트 + `last_invalidated_frame` —
  100프레임 강등 프로토콜 그대로(FORCE_STATIC 1회 동반). vcache/스킨 캐스터는
  UE `HasDeformableMesh()`처럼 매 프레임 무효화 리스트에 편입.
- 던전 등장인물(전사+그런트 7)의 정상 상태: **레벨 지오메트리 전부 정적 캐시**,
  캐릭터가 밟는 3~8페이지만 매 프레임 동적 레이어 재렌더 — 현 에폭 방식(전체 맵
  재렌더)의 페이지-국소화 버전.

### 게이트 (Engineering Rules §5)

- 기본 **OFF** seam: `VSM=1` 옵트인, 던전 앱만 기본 온(CSM 기본과 교체). 골든
  앵커/스톡 씬은 레거시 경로 바이트-불변.
- DX≡VK ≤0.001 (아토믹 max는 커뮤터티브 → 결정론 유지 전망; 클러스터 미러 교훈:
  **빌드/샘플 양쪽에 무플립 행렬 계약을 주석으로 명문화**하고 히트맵급 디버그 뷰
  (`DEBUG_VIEW`: 페이지 레벨/캐시상태 시각화)를 처음부터 함께 구현).
- `PROFILE_GPU` 예산: 정지 프레임 vsm_* 합계 ≤0.3ms, 이동 프레임 ≤1.2ms(Med, 720p).
- 캐시 정합성 게이트: (a) 정지 씬 N프레임 후 vsm_raster 인스턴스 수 == 0,
  (b) `무효화 OFF 강제` A/B에서 이동 후 잔상 재현 == 캐시가 실제로 일하는 증거,
  (c) 전 페이지 매 프레임 uncached 강제(`VSM_NOCACHE=1`) vs 캐시 경로 이미지 동일.

## 3. 마일스톤

- **V0 (선행 퀵윈, VSM과 독립)**: UE 바이어스 스택 중 ③ optimal slope bias + ① 거리
  비례 노멀 바이어스를 **현 CSM 샘플링에 포팅** (`ProjectionCommon.ush:329-360` →
  `pbr.slang`). 벽-바닥 리킹의 직접 처방 후보 — 반나절, 실패해도 VSM에서 재사용.
- **V1 — 정적 VSM 코어**: 마킹→할당→래스터→샘플(캐시 없음, 매 프레임 전체 재렌더).
  게이트: CSM 대비 이미지 A/B(리킹·부유·해상도), DX≡VK, 던전 프레임 예산.
- **V2 — 페이지 캐싱 + 클립맵 스크롤**: 2슬라이스 풀, 수명주기 4리스트(첫 컷은
  age-기반 eviction, LRU 리스트-순서는 후속), 페이지-오프셋 스크롤 + Z 가드밴드.
  게이트: 정지 씬 재렌더 0, 카메라 이동 중 캐시 생존율 로그.
- **V3 — 동적 무효화** (지시의 본체): CPU 수집기 + GPU `InvalidateInstancePages` +
  정적/동적 2레이어 + 100프레임 강등. 게이트: 캐릭터 이동 시 재렌더 페이지 수
  로그(≤8/프레임 목표), 잔상 0, S1 에폭 경로 은퇴(레거시 단일맵은 콘텐츠 씬 폴백용
  유지).
- **V4 — 필터/품질**: SMRT(7×8, 외삽 가드) 또는 우선 PCF+바이어스 스택 확정,
  quality.rs 티어 노브(풀 크기/레벨 수/SMRT 레이 수), `DIAG` 통계(오버서브,
  무효화 페이지/프레임). 포인트 라이트 VSM은 별도 계획.

## 4. 리스크

- **Metal 텍스처 아토믹**: R32Uint image atomic 미보장 기기 → 버퍼-백업 풀(주소
  산술 동일)로 우회. V1에서 3백엔드 브링업 게이트에 포함.
- **vgeo(메시셰이더) 캐스터**: 현 섀도 패스는 클래식 래스터 경로만 그림 — vgeo
  지오메트리의 VSM 편입은 vgeo 프로듀서에 페이지-렉트 컬을 붙여야 하며 V1에선
  클래식 경로 캐스터만(던전은 전부 클래식) 지원, vgeo는 V4 이후.
- **풀 오버서브스크립션**: 조용한 coarse 폴백이라 화질 저하가 은닉됨 —
  `VSM_STAT_OVERFLOW` 상당의 카운터를 DIAG_SLOTS처럼 1회 경고로 노출.
- **PS discard 오버드로**: 뮵-뷰당 1회 래스터에서 미매핑/캐시 페이지 픽셀은 PS
  discard — 캐스터가 큰 씬에서 낭비. UE도 동일 설계(수용); 페이지 렉트 뷰포트
  클립으로 1차 완화.
- **레벨당 8k 가상이 부족한 경우**(초근접 컷씬 등): 레벨 추가가 아니라 가상 해상도
  16k 승격으로 대응 가능하게 상수를 단일 소스로.

## 5. 출처

- UE5 VirtualShadowMaps 소스 (위 경로). 세부 파일:라인 인용은
  `docs/research/vsm_arch.txt` / `vsm_cache.txt` / `vsm_clip.txt`.
- SIGGRAPH 2022, "Virtual Shadow Maps in Fortnite Battle Royale" (설계 배경).
