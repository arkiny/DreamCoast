# 커스텀 그래픽스 엔진 + 렌더러 — 최상위 로드맵

## Context (왜 이 프로젝트인가)

그래픽스 구현 기법(PBR/디퍼드, 레이트레이싱, 컴퓨트, 렌더그래프)을 직접 실험·학습하기 위한
**커스텀 렌더러 + 엔진**을 처음부터 구축한다. 기성 추상화(wgpu)에 의존하지 않고 D3D12·Vulkan
두 네이티브 API를 raw로 다루며, 그 위에 본인만의 RHI(Render Hardware Interface)를 설계하는 것이
핵심 목적이다. 이를 통해 두 API의 명시적(explicit) 모델, 동기화, 디스크립터/바인드리스, 레이트레이싱
파이프라인을 깊이 이해하는 것을 목표로 한다.

이 문서는 **최상위 로드맵**이다. 각 Phase의 세부 계획은 `docs/phase-*.md`로 따로 둔다.

## 확정된 기술 스택

| 항목 | 선택 |
|---|---|
| 언어/빌드 | Rust (workspace, cargo) |
| Vulkan 백엔드 | `ash` (raw Vulkan) |
| D3D12 백엔드 | `windows` crate (windows-rs, raw D3D12) |
| Metal 백엔드 | `objc2` / `objc2-metal` (raw Metal, macOS) |
| RHI | 직접 설계 (모든 백엔드를 추상화하는 자체 계층) |
| 플랫폼 | Windows (Vulkan/D3D12) · macOS (Metal) |
| 셰이더 | Slang → DXIL(D3D12) + SPIR-V(Vulkan) + metallib(Metal) 동시 컴파일 |
| UI | Dear ImGui (`imgui` crate + 커스텀 RHI 렌더 백엔드) |
| 수학 | `glam` (SIMD, de-facto 표준) |
| 목표 기법 | PBR/디퍼드 · 레이트레이싱(DXR/VK_KHR) · 컴퓨트/GPGPU · 렌더그래프 |

## 워크스페이스 구조 (제안)

```
engine/                 # cargo workspace root
├── crates/
│   ├── platform/       # Win32 윈도잉 + 입력 (자체 계층)
│   ├── rhi/            # RHI 추상화 인터페이스 (trait/타입 정의, 백엔드 무관)
│   ├── rhi-vulkan/     # ash 기반 Vulkan 백엔드
│   ├── rhi-d3d12/      # windows-rs 기반 D3D12 백엔드
│   ├── shader/         # Slang 컴파일 파이프라인 + 리플렉션 + 핫리로드
│   ├── render/         # 렌더그래프 + PBR/디퍼드/RT/컴퓨트 패스
│   ├── scene/          # 자체 ECS + 씬 그래프(변환 계층 컴포넌트) + 레벨/스트리밍 — RHI 비의존 (Phase 13)
│   ├── asset/          # glTF 모델, 텍스처(KTX2/DDS) 로딩
│   ├── gui/            # imgui-rs + 커스텀 RHI 렌더 백엔드
│   └── core/           # 공통 유틸(로깅, 에러, 핸들/풀, 수학 재노출)
└── apps/
    └── sandbox/        # 기법 전환용 플레이그라운드 실행 파일
```

## 핵심 설계 원칙 (Phase 전반에 적용)

1. **RHI는 modern/explicit 모델 기준** — 커맨드 리스트, 명시적 동기화(fence/semaphore/barrier),
   PSO(Pipeline State Object), 디스크립터 관리를 두 API의 공통 분모로 추상화.
2. **바인드리스 우선(bindless-first)** — D3D12 디스크립터 힙 / Vulkan descriptor indexing 기반.
   레이트레이싱과 현대 렌더링을 깔끔하게 만드는 핵심. 초기 RHI 설계부터 반영.
3. **백엔드 디스패치** — RHI는 trait로 인터페이스 정의, 디바이스/리소스는 런타임 백엔드 선택.
   (trait object vs enum-dispatch는 Phase 1 세부 설계에서 확정)
4. **렌더그래프가 모든 렌더 기법의 척추** — 패스 선언, transient 리소스 할당/aliasing,
   자동 배리어/상태 전이. PBR·컴퓨트·RT 모두 이 위에 얹는다.
5. **Slang 단일 소스** — 셰이더는 Slang로 한 번 작성, 빌드 시 DXIL과 SPIR-V로 동시 컴파일.

## 단계별 로드맵 (Milestones)

각 Phase의 세부 계획 문서 링크는 진행하며 채운다.

### Phase 0 — 기반 다지기 (Foundations) — ✅ 완료
세부: [phase-0-foundations.md](phase-0-foundations.md)
- cargo workspace 골격, 의존성 확정
- Slang 컴파일 통합: `build.rs`가 `.slang` → SPIR-V + DXIL (slangc 호출, SDK 비의존)
- 윈도잉/입력: 자체 Win32 계층
- 로깅·에러·핸들/풀 등 공통 컨벤션
- **완료 기준**: 빈 창이 뜨고, Slang 셰이더가 두 포맷으로 컴파일됨
  (셰이더 컴파일은 slangc 확보 시 활성화)
- **후속(계획)**: 빌드 시 매번 전 셰이더 재컴파일 → **per-OS 바이트코드 쿡 캐시 + 콘텐츠 해시 기반
  증분 재컴파일**로 개선 (Phase 12 M4, [shader-asset-cache.md](shader-asset-cache.md))

### Phase 1 — RHI 코어 + Vulkan 백엔드 (Hello Triangle) — ✅ 완료
세부: [phase-1-rhi-vulkan.md](phase-1-rhi-vulkan.md)
- enum-dispatch RHI 골격(`rhi-types` / `rhi-vulkan` / `rhi` 파사드), 최소 수직 슬라이스
- `ash` 백엔드로 삼각형 렌더(dynamic rendering) → 프레임 루프, 스왑체인 present, 동기화
- **완료 기준**: Vulkan으로 삼각형(RTX 2070 SUPER 검증). 검증 레이어는 "있으면 켜기"

### Phase 2 — D3D12 백엔드 패리티 — ✅ 완료
세부: [phase-2-d3d12.md](phase-2-d3d12.md)
- `windows-rs` D3D12 백엔드(`rhi-d3d12`) + 파사드 `D3d12` 변형, 같은 삼각형 렌더
- 동기화 임피던스(세마포어 no-op, 펜스 에뮬레이션) 흡수로 파사드 무변경 검증
- **완료 기준**: `--backend d3d12|vulkan` 런타임 전환, 양쪽 동일 결과 (RTX 2070 SUPER 검증)

### Phase 3 — ImGui 통합 + 바인드리스 기반 — ✅ 완료
세부: [phase-3-imgui.md](phase-3-imgui.md)
- RHI에 버퍼/텍스처/**바인드리스 디스크립터**(Vulkan descriptor-indexing / D3D12 unbounded SRV table) 도입
- `crates/gui`: imgui-rs + 커스텀 RHI 렌더러(폰트 atlas=바인드리스 텍스처) + Win32 입력 브리지
- **완료 기준**: 두 백엔드 모두 삼각형 위 ImGui 창(데모/통계/클리어색) 동작 (도킹은 이후로 연기)

### Phase 4 — 에셋 파이프라인 + 메시 렌더링 — ✅ 완료
세부: [phase-4-assets.md](phase-4-assets.md)
- RHI 깊이 버퍼 + 메시 정점 레이아웃, glam 카메라(궤도, 백엔드별 y-flip)
- `crates/asset`: gltf+image로 glTF 로딩(+큐브 폴백), mesh.slang(lambert+바인드리스)
- **완료 기준**: glTF 모델이 텍스처와 함께 깊이 정확히 표시(두 백엔드). 리플렉션/핫리로드는 분리(이후)

### Phase 5 — 렌더그래프 / 프레임그래프 — ✅ 완료
세부: [phase-5-render-graph.md](phase-5-render-graph.md)
- 패스 선언 API, 의존성 DAG + 위상정렬 + dead-pass culling, transient 리소스 lifetime·aliasing, 자동 배리어/상태 전이
- RHI 오프스크린 렌더 타깃(어태치먼트+바인드리스) + `TransientHeap`(placed/aliased) 도입, `crates/render` 추가
- 데모: 블룸 체인(scene→blur×3→composite) + ImGui 포스트 토글/aliasing 토글
- **완료 기준**: 멀티 패스(오프스크린 → 포스트) 그래프가 두 백엔드에서 동작 (RTX 2070 SUPER)

### Phase 6 — PBR 렌더러 (Deferred) — ✅ 완료
세부: [phase-6-pbr.md](phase-6-pbr.md)
- G-buffer, PBR BRDF, 라이팅, 섀도우 맵(PCF), IBL(irradiance/prefilter/BRDF LUT), 톤매핑/포스트 (디퍼드 경로)
- 전제 도입 완료: 렌더그래프/RHI에 MRT, HDR 포맷, per-frame uniform buffer, 샘플 가능 depth, cubemap
- 추가: 스왑체인 readback → PNG 스크린샷 툴, **카메라 기준 실시간 환경 캡처**([realtime-env-capture.md](realtime-env-capture.md))
- **완료 기준 달성**: 디퍼드 PBR 씬 렌더(직접광+PCF 섀도우+IBL+톤매핑), 두 백엔드 픽셀 일치, Vulkan 검증 클린

### Phase 7 — 컴퓨트 / GPGPU — ✅ 완료
세부: [phase-7-compute.md](phase-7-compute.md)
- 렌더그래프 위 **1급 컴퓨트 패스** + read-write(UAV/storage) 리소스, 예제 셋(컴퓨트 포스트프로세싱, GPU 파티클 시뮬레이션, GPU 컬링+indirect draw)
- 신규 RHI: 컴퓨트 파이프라인/dispatch, storage image·storage buffer(바인드리스 UAV), indirect draw, 컴퓨트 가시 바인드리스
- **완료 기준 달성**: 세 컴퓨트 기법이 렌더 패스와 연동, 두 백엔드 픽셀 일치, Vulkan 검증 클린

### Phase 8 — 레이트레이싱 — ✅ 완료
세부: [phase-8-raytracing.md](phase-8-raytracing.md)
- RHI에 가속 구조(BLAS/TLAS), RT 파이프라인, Shader Binding Table 추상화 추가
- DXR + VK_KHR_ray_tracing 양 백엔드 구현
- **2단계**: 인라인 ray query(RayQuery / VK_KHR_ray_query) 먼저 → 풀 RT 파이프라인 + SBT
- 예제: **간단 패스트레이서** (디퓨즈 GI 누적; 인라인·파이프라인 두 경로로 검증)
- **완료 기준 달성**: 두 백엔드에서 하드웨어 RT 결과 일치 — 인라인 ≈ 파이프라인(픽셀 근사),
  VK ≡ DX(파이프라인 Cornell avg 0.0000/max 1), Vulkan 검증 클린 / D3D12 디버그 클린
- **후속(완료)**: 패스트레이서를 무편향 **Ground-Truth PBR** 레퍼런스로 확장 —
  [rt-pbr-parity.md](rt-pbr-parity.md) (full metallic-roughness BSDF·VNDF IS·NEE·러시안 룰렛·디스크 태양광·
  텍스처/노멀맵; 래스터 vs PT 비교 하니스 `tools/rt-compare.py`). 래스터가 수렴해야 할 정답.

### Phase 9 — 툴링 & 마무리 — ✅ 완료
세부: [phase-9-tooling.md](phase-9-tooling.md)
- GPU 프로파일링(패스별 타임스탬프 쿼리) ✅, 디버그 마커 + 오브젝트 네이밍(RenderDoc/PIX/NSight) ✅,
  검증 레이어 토글(`--no-validation` 런치 플래그) ✅
- 기법 전환용 샘플 브라우저(샌드박스 collapsing 섹션) ✅
- **async compute** (전용 컴퓨트 큐로 파티클 sim 오버랩) ✅ 선행 완료: [async-compute.md](async-compute.md)
- **완료 기준 달성**: 패스별 GPU ms 프로파일러 + 디버그 마커가 두 백엔드에서 동작(검증 클린),
  샌드박스에서 기법 자유 전환. M1·M2·M3 모두 완료.

### Phase 10 — Virtual Geometry — 🧪 실험적 / 계획
세부: [phase-10-virtual-geometry.md](phase-10-virtual-geometry.md)
- 클러스터 LOD DAG(meshlet 그룹 단순화) + 뷰 종속 컷 선택 + GPU 컬링/HZB 2-pass 오클루전 +
  컴퓨트 SW 래스터 + 비저빌리티 버퍼 → 머티리얼 해석으로 **Phase 6 디퍼드 G-buffer** 재사용
- 전제: **Phase 7(컴퓨트/GPU-driven)**. 신규 RHI: 메시 셰이더, 64-bit 아토믹, 인다이렉트 카운트,
  BDA. 외부 의존 `meshopt`(+선택 `metis`) 사용자 승인 필요
- **완료 기준**: 고밀도 메시가 화면 적응 LOD로 크랙/팝핑 없이 렌더, SW/HW 경로 seam 없음, 두 백엔드 일치

### Phase 11 — 소프트웨어 레이트레이싱 + Distance-Field GI — 🧪 실험적 / 계획
세부: [phase-11-distance-field-gi.md](phase-11-distance-field-gi.md)
하드웨어 RT(Phase 8) 없이도 동작하는, **컴퓨트 기반 소프트웨어 레이트레이싱 → 전역 거리장(Global
Distance Field) → 그에 대한 stochastic lighting**으로 동적 GI/반사/AO를 구현한다. 전제: **Phase 7
(컴퓨트/GPU-driven)**. Phase 8 HW RT와는 별개 경로(저사양/넓은 씬용 근사 GI).
- **Stage A — 컴퓨트 소프트웨어 레이트레이싱:** ✅ (A1 1차 가시성 + A2 소프트 섀도우/AO, 양 백엔드 검증).
  HW RT 없이 컴퓨트로 해석적 SDF 씬을 sphere-trace(`sdf_trace.slang`, env `P11_SDF`). Stage B가 해석적
  프리미티브를 베이크된 GDF로 교체. (A3 컴퓨트 BVH는 선택.)
- **Stage B — Global Distance Field:** per-mesh SDF(메시 거리장) 베이크 → 카메라 주변을 덮는 **전역
  거리장 볼륨**(클립맵/스파스 볼륨 텍스처)으로 머지. 동적 오브젝트는 매 프레임/저빈도 갱신.
- **Stage C — Stochastic Lighting:** GDF에 대해 stochastic(몬테카를로) 샘플링으로 GI(디퓨즈
  바운스)·AO·반사를 ray-march + **시공간 디노이즈**(temporal accumulation + 공간 필터). **최종 목표:
  캡처 기반 IBL을 SW-RT로 대체** — 디퓨즈 IBL→GDF GI, 스페큘러 IBL→**SSR(온스크린)+GDF 반사(오프스크린)+
  스카이(miss) 하이브리드**(C5 SSR·C6 GDF 반사·C7 합성+IBL 대체). 캡처 env 큐브는 스카이 전용으로 격하.
  스크린-스페이스 프로브/래디언스 캐시 구조는 Stage C 세부에서 확정.
  - **진행:** C1–C7 + C8a ✅ 양 백엔드 검증·푸시 (C1 씬 GDF, C2 AO, C3 GI, C4 디노이즈, C5 SSR, C6 GDF
    반사, C7 하이브리드 합성→라이팅 specular 대체, C8a per-voxel 알베도→컬러 GI·반사). **반사 트랙 성공
    지표 달성: 하이브리드-vs-PT 잔차 4.18→2.58/ch(−38%)**(`P11_SWRT_REFLECT`); C8a로 GDF 히트가 실제 표면색
    재조명(`P11_GDF_COLOR`). NEXT: C8b 서피스 캐시 라디언스(멀티바운스·잔차 추가 감소).
- **완료 기준**: 동적 씬에서 HW RT 없이 GDF 기반 GI/AO가 두 백엔드에서 동작, 패스트레이서(Phase 8)
  레퍼런스 대비 그럴듯하게 수렴, 검증 클린.

### Phase 12 — 에셋 파이프라인 / 쿠킹된 에셋 — 🧪 실험적 / 계획
세부: [phase-12-asset-pipeline.md](phase-12-asset-pipeline.md)
**크로스컷팅 엔진 인프라** — `crates/asset`의 자산 직렬화 계층. 가공된 **메시 지오메트리 + 베이크 데이터
(SDF, 향후 BVH/라이트맵)를 하나의 `.dcasset` 바이너리로 cook → 저장 → 런타임 직접 로드**. 매 실행
glTF 재파싱 + SDF 재베이크를 없앤다. Phase 11 Stage B의 GDF 베이크 영속화가 직접 동기.
- **M1 — 메시 직렬화**(.dcasset 골격: 헤더 + 메시 청크; cook glTF→.dcasset, 런타임 직접 로드). Phase 11과 독립.
- **M2 — SDF 베이크 청크 통합**(Phase 11 B2 결과 영속화 + 로드). Stage B 이후.
- **M3 — 확장**(후속·선택: BVH/라이트맵/프로브, 텍스처 참조).
- **M4 — 셰이더 바이트코드 쿡 캐시 (per-OS, 콘텐츠 해시)**: 세부 [shader-asset-cache.md](shader-asset-cache.md).
  `.slang`을 **OS별 바이트코드 에셋**(Windows SPIR-V+DXIL / macOS metallib)으로 쿡 + **소스 내용 해시
  + mtime**으로 변경 감지 → 바뀐 셰이더만 자동 재컴파일(현재는 mtime만 보고 매 빌드 전 잡 재컴파일).
  매니페스트 + 무의존 해시. **P11/P12 본선과 독립**(셰이더 캐시는 별도 산출물, 단독 진행 가능).
- **완료 기준**: glTF 재파싱·재베이크 없이 `.dcasset` 로드로 동일 렌더(양 백엔드 일치), startup 가속,
  베이크 결과 바이트 동일 캐시. **M4**: 무변경 빌드 시 slangc 호출 0회, 단일 셰이더 변경 시 해당 잡만
  재컴파일, 최종 바이트코드 무회귀(양 백엔드 픽셀 일치).
- **교차 (Phase 13)**: 차후 **씬/레벨 청크**(`.dclevel`/`.dcworld`)를 같은 `.dcasset` 컨테이너에 추가 —
  Phase 13 Stage E 또는 본 Phase의 M3. 엔티티(애셋 source-hash 참조)+트랜스폼+라이트/카메라/환경, 월드는
  청크 그래프(인접·배치·스트리밍 반경)를 직렬화. [phase-13-scene-graph.md](phase-13-scene-graph.md) 참조.

### Phase 13 — 씬 그래프 + 레벨 스트리밍 — 🧪 실험적 / 계획
세부: [phase-13-scene-graph.md](phase-13-scene-graph.md)
렌더 그래프(Phase 5)가 "어떻게 그리는가"라면, 씬 그래프는 "무엇이 어디에 존재하는가"의 공간·논리 표현이다.
**DreamCoast는 학습용을 넘어 장기적으로 직접 게임 개발에 쓸 엔진**이므로, 씬 표현은 게임 런타임에 맞는
**자체 제작 ECS**를 1급 코어로 둔다(RHI·렌더그래프를 from-scratch로 만든 철학과 일관, 외부 ECS 의존 없음).
현재 샌드박스는 평면 `Vec<SceneObject>`로 씬을 하드코딩하고 `load_gltf`는 첫 프리미티브만 읽어 계층을
버린다(예: `Lantern.glb` 4노드/3메시 중 1메시만 렌더). 신규 **RHI 비의존 크레이트 `crates/scene`**에
**최소 ECS**(generational `Entity` + `World` + 컴포넌트 스토리지/쿼리)와, 그 위의 변환 계층 씬 그래프
(`Parent`/`Children`/`LocalTransform`/`WorldTransform` 컴포넌트 + `propagate_transforms` 시스템), 선언적
레벨/스트리밍을 구축한다. `(WorldTransform, MeshInstance)` 쿼리가 래스터·HW RT(TLAS)·GPU 컬링에
per-instance 트랜스폼을 공급하는 단일 소스가 된다. 테스트는 보유 애셋(Avocado/BoomBox/Lantern)으로 구동.
전제: Phase 4(애셋)·5(렌더그래프)·8(RT).
- **선행조건**: P1 자유 비행 카메라(현 궤도 전용 → Stage D 주행용), P2 레지스트리 기반 다중 머티리얼
  업로드, P3 Phase 12 `.dcasset` 골격(레벨/월드 *바이너리* 쿡만 차단; RON 텍스트는 무관).
- **직렬화**: 레벨/월드 데이터 모델은 처음부터 serde-ready(RON 텍스트 로드·저장) → 차후 Phase 12
  컨테이너로 쿡(위 교차 참조).
- **Stage 0** 자유 비행 카메라 → **A** 자체 ECS 코어 + 컴포넌트 트랜스폼 계층 + 기존 씬 마이그레이션(픽셀
  회귀) → **B** 전체 glTF 계층 임포트(Lantern 3메시, `Parent`/`Children`) → **C** 선언적 `.level` 포맷 +
  로더/세이브 + 런타임 전환 → **D** 레벨 그래프(`LevelGraph`) + 카메라 기반 청크 스트리밍(엔티티
  스폰/디스폰) → **E**(선택, Phase 12 이후) 쿡된 바이너리 레벨/월드.
- **완료 기준**: ECS 씬이 단일 드로우 리스트로 세 소비처(래스터/RT/컬)를 공급, glTF 계층이 올바른 상대
  트랜스폼으로 렌더, 선언적 레벨 핫스왑, 카메라 주행 시 청크 스트림 인/아웃(엔티티 디스폰, 누수 없음), 두
  백엔드 픽셀 일치, 검증 클린.
- **범위 외**: glTF 애니메이션/스키닝, ECS 멀티스레드 스케줄링·변경 감지(계층/ECS가 잠금 해제하는 후속이나
  본 Phase 제외).

### macOS / Metal 백엔드 — 🚧 진행 중
세부: [metal-backend.md](metal-backend.md)
- 네이티브 Metal 백엔드(`crates/rhi-metal`, `objc2`)를 동일한 enum-dispatch RHI 뒤에 추가.
  플랫폼 레이어는 macOS에서 손수 작성한 Cocoa/AppKit 창 + `CAMetalLayer`. 현재 목표는
  **Phase 7 + Phase 8 inline RT**까지의 macOS 실행 패리티와, Metal Shader Converter를 통한
  DXR-style RT pipeline 경로의 실험적 연결.
- 백엔드는 OS별 `#[cfg]` 게이팅: Windows=Vulkan+D3D12, macOS=Metal. `rhi-vulkan`/`rhi-d3d12`는
  `#![cfg(windows)]`, `rhi-metal`은 macOS 전용.
- 마일스톤: **M0** 골격+클리어 ✅ · **M1** Slang→metallib ✅ · **M2** 삼각형 ✅ · **M3** 바인드리스
  (argument buffer)+텍스처+ImGui ✅ · **M4** 렌더타깃+PBR ✅ · **M5** 컴퓨트/async/인다이렉트 ✅ ·
  **M6** inline ray tracing ✅ · **M7** Metal Shader Converter RT pipeline plumbing ✅
- 최근 수정: 일반 Slang→`metallib` 경로가 Apple `metal`의 clang module cache를 `~/.cache` 아래에
  쓰려다 sandbox에서 실패하던 문제는 shader build가 `HOME`/`XDG_CACHE_HOME`을 `OUT_DIR`로
  고정하도록 해결. M7 runtime은 SBT stride artifact를 고친 뒤 inline/pipeline screenshot 및
  Metal API+GPU validation layer까지 통과. RT 텍스처 머티리얼은 hit UV 보간 + mip0 `Load`
  기반 bilinear 샘플링으로 base/mr/normal/emissive를 inline과 M7 양쪽에 적용했고, M7
  converter descriptor table도 sampled texture/cube/storage/TLAS 범위를 채우도록 갱신.

## 의존성 위험 / 미결 사항 (세부 계획에서 해소)

- **백엔드 디스패치**: trait object vs enum-dispatch. Phase 1.
- **디스크립터/바인드리스 모델 통일**: 두 API의 디스크립터 모델 차이 흡수 설계. Phase 1.
- **windows-rs D3D12 ergonomics**: raw COM 인터페이스라 RAII 래퍼 설계 필요. Phase 2.
- **RT 추상화**: SBT 레이아웃·AS 빌드 인터페이스의 두 API 공통화 난이도 높음. Phase 8.

## 검증 전략 (전 단계 공통)

- 각 백엔드는 **검증 레이어 클린**(Vulkan validation layers / D3D12 debug layer + GPU-based validation)을
  기본 게이트로 둔다.
- 모든 핵심 마일스톤은 **D3D12·Vulkan 양쪽에서 동일 결과**를 내는 것을 합격 기준으로 한다.
- RenderDoc/PIX 캡처로 시각적·리소스 상태 검증.
- 샌드박스 앱에 각 기법별 씬을 추가해 회귀 확인.
