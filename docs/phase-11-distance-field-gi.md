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

**부트스트랩 결정:** SDF 마칭은 거리장이 필요한데 거리장(GDF)은 Stage B에서 만든다. 그래서 Stage A는
**해석적 SDF 프리미티브**(구/박스/평면 = 샘플 씬 레이아웃을 미러링; model_radius=1.0 정규화라 좌표가 상수)로
부트스트랩한다 — GDF 베이크 없이 컴퓨트 RT 기계(카메라 레이 생성, march 루프, gradient 노멀, storage-image
출력)를 세우고 검증한 뒤, Stage B가 해석적 프리미티브를 베이크된 per-mesh SDF + 클립맵으로 교체한다.

마일스톤: **A1 1차 가시성 → A2 소프트 섀도우 + AO → A3(선택) 컴퓨트 BVH 삼각형 트래버설**.

### A1 — 1차 가시성 ✅ (양 백엔드 검증, `925c266` 다음 커밋)
- 신규 `crates/shader/shaders/sdf_trace.slang` (`sdf_trace_cs`): 픽셀당 1차 카메라 레이를 해석적 SDF 씬에
  **sphere-trace**(최대 192 스텝). 히트 시 중앙차분 gradient = 노멀, miss 시 패스트레이서와 동일한 `sky`.
  셰이딩 = Lambert 태양 + 단순 반구 스카이 앰비언트(섀도우/AO는 A2). 카메라 레이 재구성은 `rt_common`의
  `primary_ray_dir`과 동일(inv_view_proj·z=1). raw radiance 출력 → 기존 tonemap이 노출+ACES 적용.
- TLAS/머티리얼 테이블 plumbing 없음 — `bindless.slang`의 `storage_images[]`만 사용, `compute_supported`
  게이트(순수 컴퓨트). push 112B(`sdf_trace_push`, rt_trace와 동일 레이아웃 + sun.w=강도). 통합은 M3
  `rt_trace` viz와 동형: 컴퓨트 패스가 `sdf_out` storage image에 쓰고 tonemap이 HDR 대신 표시
  (`rt_out.or(sdf_out).or(hdr_post)`). 토글: env `P11_SDF` + UI "Software ray tracing (Phase 11)".
- **검증(RTX 2070 SUPER):** build+fmt+clippy(-D warnings) 클린. VK·DX 렌더 정상(구3+박스+그라운드+스카이,
  매끄러운 gradient 노멀). **VK≡DX: 920k 픽셀 중 1픽셀만 >2 차이**(실루엣 엣지 1px — 반복 march의
  SPIR-V/DXIL fp contraction 차이; mean 0.0002/ch). Vulkan VUID 0. SDF-off 기본 래스터 씬 byte-identical
  (회귀 없음). 한계: 아보카도 메시는 구 프록시, 위치는 근사 — 메시 픽셀 매치는 Stage B(GDF 베이크)에서.

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
