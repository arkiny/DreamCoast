# Phase 11 — 소프트웨어 레이트레이싱 + Distance-Field GI (세부 계획 / 스텁)

상위: [ROADMAP.md](ROADMAP.md) Phase 11. **전제: Phase 7(컴퓨트)**. Phase 8 HW RT와 **별개 경로** —
하드웨어 RT 없이 컴퓨트만으로 동적 GI/반사/AO를 근사한다(넓은 씬·저사양 타깃). 무편향 패스트레이서
([rt-pbr-parity.md](rt-pbr-parity.md))가 정답 레퍼런스.

> 목표: **씬의 전역 거리장(Global Distance Field)을 생성**하고, 그 거리장을 **컴퓨트 셰이더로
> ray-march(소프트웨어 레이트레이싱)** 하여 **stochastic(몬테카를로) 라이팅**으로 동적 GI를 구한다.
> 단, 먼저 **컴퓨트 셰이더로 레이트레이싱을 구현하는 기반**(Stage A)을 만든 뒤 거리장 GI로 확장한다.

순서: **A(컴퓨트 SW RT) → B(GDF) → C(Stochastic Lighting)**. 각 스테이지 양 백엔드 + 검증 클린 게이트.

## Stage A — 컴퓨트 소프트웨어 레이트레이싱
HW RT 파이프라인(Phase 8) 없이 컴퓨트 셰이더로 레이를 추적하는 기반.
- 1차 접근: **거리장 ray-marching**(sphere tracing) — Stage B의 SDF를 그대로 쓰는 자연스러운 경로.
- 대안/병행: 컴퓨트 BVH 트래버설(삼각형 정확). 우선순위는 SDF 마칭(거리장 GI와 자연스럽게 정합).
- 검증: Phase 8과 동일 씬에서 1차 가시성/그림자 결과 대조.

## Stage B — Global Distance Field
- **per-mesh SDF 베이크:** 각 메시를 3D 거리장 텍스처로(컴퓨트, 점→삼각형 거리 / 또는 voxelize+JFA).
- **전역 머지:** 카메라 주변을 덮는 **GDF 클립맵**(여러 해상도 레벨의 3D 볼륨)으로 per-mesh SDF를
  합성. 정적은 베이크, 동적 오브젝트는 매 프레임/저빈도 갱신(영향 영역만).
- 신규 RHI 가능성: 3D(볼륨) 텍스처 + UAV, 3D 디스패치. (Phase 7 storage image의 3D 확장.)

## Stage C — Stochastic Lighting
- GDF를 ray-march해 **디퓨즈 GI(1+ 바운스)·AO·러프 반사**를 stochastic(몬테카를로) 샘플.
- **시공간 디노이즈:** temporal accumulation(재투영) + 공간 필터. 스크린-스페이스 프로브 /
  래디언스 캐시 / per-pixel 중 구조는 이 스테이지에서 확정.
- 머티리얼 히트 셰이딩: GDF는 거리만 → 표면 머티리얼/라이팅은 surface cache 또는 근사 필요(설계 항목).
- 결과를 디퓨드 라이팅(Phase 6)의 ambient/GI 항으로 합성.

## 미결 / 설계 항목
- GDF 표현(클립맵 레벨 수/해상도), 메모리 예산.
- SW RT 정확도 vs 비용(마칭 스텝/원뿔 추적).
- 디노이저 구조, 동적 오브젝트 GDF 갱신 빈도.
- HW RT(Phase 8)와의 선택/하이브리드 관계.
