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
| RHI | 직접 설계 (두 백엔드를 추상화하는 자체 계층) |
| 플랫폼 | Windows 전용 |
| 셰이더 | Slang → DXIL(D3D12) + SPIR-V(Vulkan) 동시 컴파일 |
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

### Phase 6 — PBR 렌더러 (Deferred) — 🚧 진행 중
세부: [phase-6-pbr.md](phase-6-pbr.md)
- G-buffer, PBR BRDF, 라이팅, 섀도우 맵, IBL, 톤매핑/포스트 (디퍼드 경로)
- 전제: 렌더그래프/RHI에 MRT, HDR 포맷, per-frame uniform buffer, 샘플 가능 depth, cubemap 도입
- **완료 기준**: 디퍼드 PBR 씬 렌더(직접광+섀도우+IBL+톤매핑), 두 백엔드 동일 결과

### Phase 7 — 컴퓨트 / GPGPU
- 렌더그래프 위 컴퓨트 패스, 예제(GPU 컬링, 파티클 시뮬레이션, 포스트프로세싱)
- **완료 기준**: 컴퓨트 기반 효과가 렌더 패스와 연동

### Phase 8 — 레이트레이싱
- RHI에 가속 구조(BLAS/TLAS), RT 파이프라인, Shader Binding Table 추상화 추가
- DXR + VK_KHR_ray_tracing 양 백엔드 구현
- 예제: RT 섀도우/AO/반사, 간단한 패스트레이서
- **완료 기준**: 두 백엔드에서 하드웨어 RT 결과 일치

### Phase 9 — 툴링 & 마무리
- GPU 프로파일링(타임스탬프 쿼리), 디버그 마커(PIX/RenderDoc/NSight), 검증 레이어 토글
- 기법 전환용 샘플 브라우저(샌드박스) 완성
- **완료 기준**: 프로파일러 + 캡처 툴 연동, 샌드박스에서 기법 자유 전환

### Phase 10 — Virtual Geometry — 🧪 실험적 / 계획
세부: [phase-10-virtual-geometry.md](phase-10-virtual-geometry.md)
- 클러스터 LOD DAG(meshlet 그룹 단순화) + 뷰 종속 컷 선택 + GPU 컬링/HZB 2-pass 오클루전 +
  컴퓨트 SW 래스터 + 비저빌리티 버퍼 → 머티리얼 해석으로 **Phase 6 디퍼드 G-buffer** 재사용
- 전제: **Phase 7(컴퓨트/GPU-driven)**. 신규 RHI: 메시 셰이더, 64-bit 아토믹, 인다이렉트 카운트,
  BDA. 외부 의존 `meshopt`(+선택 `metis`) 사용자 승인 필요
- **완료 기준**: 고밀도 메시가 화면 적응 LOD로 크랙/팝핑 없이 렌더, SW/HW 경로 seam 없음, 두 백엔드 일치

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
