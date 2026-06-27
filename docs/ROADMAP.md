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
| 에셋 임포트 | glTF (`gltf` crate) · **FBX**(ufbx 기본 / Autodesk FBX SDK 옵션 — `tools/` fetch 스크립트, Phase 14) |
| 목표 기법 | PBR/디퍼드 · 레이트레이싱(DXR/VK_KHR) · 컴퓨트/GPGPU · 렌더그래프 · **스켈레탈 애니메이션/GPU 스키닝** |

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
│   ├── anim/           # 스켈레톤·클립·포즈 샘플링/블렌딩 + 본 팔레트 — RHI 비의존 (Phase 14)
│   ├── asset/          # glTF + FBX(ufbx/SDK) 모델, 스킨/애니메이션, 텍스처(KTX2/DDS) 로딩
│   ├── gui/            # imgui-rs + 커스텀 RHI 렌더 백엔드
│   ├── core/           # 공통 유틸(로깅, 에러, 핸들/풀, 수학 재노출)
│   │                   # ── 상용 확장(Phase 15+): 비그래픽스 시스템은 facade+백엔드 분리 ──
│   ├── jobs/           # work-stealing 잡 시스템 / 태스크 그래프 (P15, S)
│   ├── physics/        # 피직스 facade(중립 타입) — physics-rapier/jolt 백엔드 (P16, S/B)
│   ├── audio/          # 오디오 facade — audio-kira/miniaudio 백엔드 (P17, S/B)
│   ├── script/         # 스크립트 facade — script-luau(기본, mlua) + script-wasm(샌드박스, wasmtime) (P25, S/B)
│   ├── net/            # 네트워킹 facade — 리플리케이션/트랜스포트 백엔드 (P30, S/B)
│   ├── ui/             # 게임 리테인드 UI + 텍스트 셰이핑 + i18n (P26, S)
│   ├── vfx/            # Niagara式 파티클/VFX 그래프 (P28, S)
│   └── ai/             # navmesh + 패스파인딩 + 비헤이비어 트리 (P29, S/B)
└── apps/
    ├── sandbox/        # 기법 전환용 플레이그라운드 실행 파일
    └── editor/         # 독립 standalone 에디터 앱 (에디터 트랙, S)
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
6. **비그래픽스 대형 시스템은 "RHI 패턴"으로 통합** (Phase 15+) — 피직스/오디오/스크립트/네트워킹은
   from-scratch facade 크레이트(중립 타입+trait) + 백엔드 크레이트(서드파티 어댑터)로 분리, 엔진은
   facade에만 의존 → 차후 자체 구현으로 교체 가능. `rhi`/`rhi-vulkan` 구조 미러링.
7. **잡 시스템이 멀티스레드의 척추** (Phase 15+) — work-stealing 스케줄러 위에 ECS 병렬·병렬 RHI 기록·
   비동기 스트리밍·피직스 스텝을 얹는다(렌더그래프가 GPU 패스의 척추이듯).
8. **결정성·재현성 우선** — 멀티스레드/피직스/네트워크는 고정 타임스텝 + 결정적 실행 전제(헤드리스
   골든이미지·양 백엔드 회귀를 깨지 않게).

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

### Phase 11 — 소프트웨어 레이트레이싱 + Distance-Field GI — ✅ 완료 (Stage A–D, 양 백엔드)
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
  - **진행:** C1–C7 + C8a–C8j ✅ + 레거시 IBL deprecated ✅ 양 백엔드 검증·푸시. C1 씬 GDF, C2 AO,
    C3 GI, C4 디노이즈, C5 SSR, C6 GDF 반사, C7 하이브리드 합성→라이팅 specular 대체(잔차 4.18→2.58/ch
    −38%), C8a per-voxel 알베도→컬러, **C8b 메시-카드 서피스 캐시(캡처→라이팅→멀티바운스→컨슈머
    룩업, `P11_SURFACE_CACHE` opt-in)**. **레거시 캡처-큐브 IBL → 기본값을 SW-RT 반사+GDF GI로 전환,
    `P11_LEGACY_IBL` 플래그로 격하**(씬 캡처 sky-only).
  - **반사 트랙 마무리 C8c–C8j ✅** (세부 [phase-11-distance-field-gi.md](phase-11-distance-field-gi.md)):
    러프니스-aware 컴포짓 + GDF 러프니스 프리필터(C8c/C8c2) → **풀-res 미러 SSR을 정확 소스로 복원 +
    reflection max-roughness 임계**(`P11_REFLECT_MAX_ROUGHNESS`, C8d) → 구리 하이라이트 blow-out·이중
    디렉셔널 스펙큘러 수정(반사된 태양 디스크 제거, C8e/C8f) → 서피스 캐시 히트 라디언스 반사(C8g) →
    밉-피라미드 프리필터(C8h) → **스토캐스틱 GGX GDF 반사 + 시공간 디노이즈**(C8j). 현재
    하이브리드-vs-PT **≈3.45/ch**(컬러·러프니스·이중스펙·blow-out 모두 처리; 풀-res 미러 2.58이 best-known).
  - **남은 본질 한계 = GDF 저해상 48³ SDF 블롭 형상** — 해상도/클립맵 레버는 측정으로 기각·종료
    (이 작은 테스트 씬 한정, [reflection-sdf-resolution.md](reflection-sdf-resolution.md)). 이 PT 잔차
    (≈3.45/ch)는 **불가피한 한계로 수용하고 Phase 11 완료 처리**. 추가 반사 작업은 *실제 게임 씬*에서
    측정-구동으로 재개. NEXT 후보: 동적 오브젝트 GDF 갱신.
- **Stage D — RenderQuality 티어 (확장성, ✅ 구현, 가로지르는 항목):** 이 씬/엔진은 차후 범용 게임용이라,
  트랙 전반에 흩어진 품질 노브(GI spp, 반사 GGX 샘플/디노이즈 반경, 소프트 그림자 PCSS 샘플 수,
  서피스-캐시 해상도)를 **단일 `RenderQuality{low,med,high}` enum 한 곳으로** 묶어 런타임/플랫폼별로
  분리한다(저사양 = 저티어 폴백). 각 기능은 이미 "기본 off/저비용 + env·플래그 seam"으로 설계돼 있어
  (`SHADOW_SOFTNESS`/`SOFT_SHADOWS`, `P11_*`, 셰이더 상수 블록) **티어가 그 seam을 선택만** 하면 된다.
  Phase 6 그림자도 포함. 게임용 스케일러빌리티의 토대. 세부: [render-quality-tiers.md](render-quality-tiers.md).
  - **진행 ✅:** P1 스캐폴드(`4df8e63`) + P2 특성화 + P3 런타임 UI 전환·플랫폼 기본 seam(`5f11f69`).
    `apps/sandbox/src/quality.rs` = 단일 티어→노브 테이블, `RENDER_QUALITY=low|med|high`(미설정=Med=무회귀),
    개별 `P11_*`/`SHADOW_*` env 오버라이드 우선. 측정(d3d12): Low 6.019/ch·3.96ms, Med 6.039·5.58, High
    6.722·11.74(미적 소프트섀도우라 PT 잔차↑·DX≡VK 0.009 옵트인). 측정 기각 레버(SDF 해상도/CARD_TILE) 제외.
- **완료 기준**: 동적 씬에서 HW RT 없이 GDF 기반 GI/AO가 두 백엔드에서 동작, 패스트레이서(Phase 8)
  레퍼런스 대비 그럴듯하게 수렴, 검증 클린. → **충족·완료**(잔차는 48³ GDF 해상도가 지배하는 본질
  한계로 수용; 정밀화는 실제 게임 씬에서 측정-구동 재개).

### Phase 12 — 에셋 파이프라인 / 쿠킹된 에셋 — ✅ 완료 (양 백엔드)
세부: [phase-12-asset-pipeline.md](phase-12-asset-pipeline.md)
**크로스컷팅 엔진 인프라** — `crates/asset`의 자산 직렬화 계층. 가공된 **메시 지오메트리 + 베이크 데이터
(SDF/albedo) + 압축 텍스처를 하나의 `.dcasset` 바이너리로 cook → 저장 → 런타임 직접 로드**. 매 실행
glTF 재파싱 + 텍스처 디코드 + SDF/albedo 재베이크를 없앤다. 수동 little-endian 청크 컨테이너
(헤더 + 타입/오프셋/크기 디렉터리), 무효화 키 `{version, source_hash, cook_params_hash}`, 결정적 CPU
쿡(크로스백엔드 바이트 동일), gitignored `/cache/`.
- **M1 ✅ — 메시 직렬화**(헤더 + 메시 청크 + 텍스처 청크; cook glTF→.dcasset, 런타임 직접 로드, glTF
  부재 시 shipped 로드). Phase 11과 독립.
- **M2 ✅ — SDF + albedo 베이크 청크**: scene GDF 베이크를 **GPU→CPU 베이크로 전환**해 영속화
  (`sdf_bake.slang` Rust 포팅) + RHI 볼륨 업로드(`create_volume_init`)로 GPU 베이크 패스 대체.
- **M3 ✅ — 텍스처 BCn 블록 압축**(GPU 네이티브, 런타임 해제비용 0): BC1/BC3/BC4/BC5/**BC7** 인코더 +
  RHI 포맷(VK/DX/Metal). cook `TexCompress{Off,Fast,High}` 티어(`P12_TEX_COMPRESS`), per-slot 정책
  (컬러→BC1/BC7, 노멀→BC5, **데이터 텍스처 무압축**), 옵트인(기본 off=무회귀). Avocado: Off 50.4MB /
  Fast 25.2MB / High 28.0MB.
- **M4 ✅ — 셰이더 바이트코드 쿡 캐시 (per-OS, 콘텐츠 해시)**: 세부 [shader-asset-cache.md](shader-asset-cache.md).
  `.slang`을 OS별 바이트코드 에셋으로 쿡 + 콘텐츠 해시 → 바뀐 셰이더만 재컴파일. 무변경 빌드 0 slangc.
- **아이템 — 추가 산출**: `.dclevel` 씬/레벨 청크(엔티티+트랜스폼+머티리얼 오버라이드+라이트/카메라/환경,
  Phase 13 Stage E 기반), 볼륨 readback `Device::read_volume`(GPU 산출 볼륨의 데이터-레벨 쿡/검증).
- **완료 기준 달성**: glTF 재파싱·재베이크 없이 `.dcasset` 로드로 동일 렌더(DX≡VK 0.000/ch, 기준선
  무회귀), run2=CacheHit startup 가속, 베이크 바이트 동일 캐시. clippy/fmt 클린.
- **남은 후속(선택)**: BC7 멀티모드(품질), 추가 베이크 페이로드(BVH/라이트맵/프로브), 텍스처 압축 기본화
  RenderQuality 결속. **씬/레벨 렌더 구동은 Phase 13 Stage E**가 `.dclevel` 포맷에 결속.

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
- **범위 외**: glTF 애니메이션/스키닝(→ **Phase 14**), ECS 멀티스레드 스케줄링·변경 감지(계층/ECS가
  잠금 해제하는 후속이나 본 Phase 제외).

### Phase 14 — 스켈레탈 애니메이션 + GPU 스키닝/스킨 캐시 — 🧪 실험적 / 계획
세부: [phase-14-animation-skinning.md](phase-14-animation-skinning.md)
범용 게임용 엔진(Phase 13)에 거의 모든 동적 콘텐츠의 기반인 **스켈레탈 애니메이션 + GPU 스키닝**을
추가한다. 핵심은 **GPU 스킨 캐시** — 스킨된 정점을 정점 셰이더가 아니라 **프레임당 한 번 컴퓨트로
계산해 버퍼에 캐시**하고, G-buffer·섀도우·HW 패스트레이서(BLAS)·SW-RT GDF 등 **모든 지오메트리
소비처가 그 단일 버퍼를 공유**한다(*skin once, consume many*; 멀티 소비처/RT 엔진의 GPU 스킨 캐시 패턴과 동일 동기). 정점 셰이더
스키닝은 패스마다 재계산하고 레이트레이싱에 줄 정점 버퍼가 없어, 멀티 소비처/RT 엔진엔 스킨 캐시가 정답.
업계 표준 **FBX 임포터**도 포함(서드파티 백엔드 + 별도 설치 스크립트).
- **신규 크레이트 `crates/anim`** (RHI 비의존): 스켈레톤·클립·포즈 샘플링/블렌딩 → **본 팔레트**(단일 소스).
- **스킨 영향치는 별도 스트림**(`SkinInfluence`, 12B): `VertexLayout::Mesh`(32B) 무변경 → 기존 PSO 처닝 0,
  스킨 캐시 출력도 32B Mesh 레이아웃 → 다운스트림 래스터 무구별.
- **GPU 스키닝 컴퓨트**(`skinning.slang`) + **프레임 그래프 `skin_cache` 패스**(모든 소비처 앞 1회) +
  더블 버퍼(모션 벡터) + dirty 스킵. 결정적 컴퓨트 → **VK ≡ DX 비트 동일**. 신규 RHI: 버퍼 다용도
  `Vertex|Storage|AccelInput` + 상태 전이.
- **RT 통합**(Phase 8 게이트): 스킨 캐시 출력으로 매 프레임 **스킨된 BLAS 리핏 + TLAS 갱신** → 패스트레이서/
  GDF가 애니메이션 반영.
- **FBX 임포터 (둘 다 seam)**: **ufbx**(MIT 단일 .c, 기본, `cc` 빌드) + **Autodesk FBX SDK**(옵션 feature
  `fbx-sdk`). 서드파티 → `tools/fetch-ufbx.{ps1,sh}` / `tools/fetch-fbxsdk.ps1`로 확보(gitignored, 게이트
  다운로드는 안내·배치). glTF와 **동일 중립 타입**(`SkinnedModel`)으로 매핑.
- **확장성**: 스키닝 품질 노브(갱신 레이트/최대 영향치/본 LOD)를 `RenderQuality` 티어 seam에 접속(`P14_*`).
- **전제**: Phase 8(RT, Stage D만) — 완료. Phase 13(씬 그래프)은 **시너지·하드 의존 아님**(단일 스킨
  인스턴스로 독립 출하, 재생 상태/다중 인스턴스는 Phase 13 ECS `AnimationPlayer`/`SkinnedMesh`로 승격).
- **완료 기준**: 스킨 메시가 GPU 스키닝/스킨 캐시로 애니메이션, CPU 스킨 ≡ GPU 스킨 픽셀 일치, 스킨된
  BLAS로 패스트레이서 잔차가 정적 메시 수준 수렴, glTF/FBX가 같은 스켈레톤·클립으로 같은 렌더, 두 백엔드
  픽셀 일치, 검증 클린.

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
- **Phase 11 SW-RT 패리티: Stage B(3D 볼륨/GDF) + Stage C(GDF AO·GI·SSR·GDF 반사·Lumen
  surface cache) 모두 Metal에서 완료·검증**(M3 box). Stage C의 유일한 Metal 갭은 컴퓨트
  파이프라인의 `uniform_buffer`(SSR의 per-frame globals UBO 바인딩)였고 `rhi-metal`에만
  반영해 해결 — 공유 셰이더/`rhi-types`/렌더 그래프 무변경이라 **Vulkan/D3D12 무회귀**.
  세부: [metal-backend.md](metal-backend.md) "Phase 11 Stage B/C".

## 상용 엔진 확장 — 런타임/툴링 계층 (Phase 15+) — 🧭 전략 계획

세부 검토·갭 분석: [commercial-engine-gap-analysis.md](commercial-engine-gap-analysis.md).
렌더링 코어(P0–P14)는 상용 R&D급이나, **게임 런타임 서비스 + 저작 툴 계층**이 비어 있다. 목표는
**범용 엔진 breadth**. 비그래픽스 대형 시스템은 **하이브리드 + 인터페이스 분리**(RHI처럼 from-scratch
facade trait 뒤에 성숙 라이브러리 백엔드를 격리 → 차후 자체 구현 교체 가능). 에디터는 **독립 standalone
앱**. `S`=from-scratch, `S/B`=facade 자체+백엔드 통합.

- **T0 토대 — Phase 15 잡 시스템/멀티스레드 코어 + 병렬 렌더 [S].** work-stealing 스케줄러, 태스크
  그래프, 병렬 RHI 커맨드 기록, 고정 타임스텝 sim, 결정적 스케줄. *멀티스레드 전 시스템의 전제.*
- **T1 런타임 코어** — **16 피직스 [S/B]**(facade + Rapier/Jolt) · **17 오디오 [S/B]**(facade + kira/
  miniaudio) · **18 입력/플랫폼 서비스 [S]**(액션 매핑·게임패드·설정 영속화) · **19 ECS 성숙 + 프리팹/
  세이브 [S]**(시스템 스케줄링·이벤트·직렬화).
- **T1→T2 렌더 완성도** — **20 AA(TAA)+포스트 스택+투명/OIT [S]**(P14 모션벡터) · **21 다광원
  클러스터드+CSM+데칼+프로브 [S]** · **22 대기/볼류메트릭 [S]** · **23 월드(지형/식생/물/버추얼 텍스처)
  [S]** · **24 머티리얼 그래프+고급 셰이딩(SSS/헤어/클로스) [S]**.
- **T2 게임플레이 breadth** — **25 스크립팅 [S/B]**(facade + **Luau 기본**(mlua luau feature) + WASM 샌드박스) · **26 게임 UI [S/B]**(리테인드+
  텍스트 셰이핑+i18n) · **27 고급 애니메이션 [S]**(스테이트머신/블렌드트리·IK·리타깃, P14 확장) ·
  **28 VFX 저작 [S]**(Niagara式, P7 위) · **29 AI/내비 [S/B]**(navmesh+BT) · **30 네트워킹/리플리케이션
  [S/B]**.
- **에디터 트랙 `apps/editor`(독립 앱, 연속)** — E1 셸/도킹/undo → E2 씬 편집(기즈모·인스펙터·`.level`) →
  E3 콘텐츠 브라우저/임포트 → E4 서브에디터(머티리얼/애님/VFX/지형) → E5 PIE/라이브링크/패키지.
- **인프라(가로지름, 연속)** — 핫리로드(셰이더→에셋→스크립트), CPU 프로파일러/메모리 추적, 크래시/
  텔레메트리, 골든이미지 회귀 CI, 패키징/배포(P12 확장), 로컬라이제이션.
- **권장 1차 수직 슬라이스**(플레이 가능 최소 루프): `15 잡 → 16 피직스 → 18 입력 → 19 ECS/세이브 →
  20 AA/포스트 → 17 오디오 → 25 스크립팅 → 에디터 E1–E2`. 이후 21–24(렌더)·26–30(게임플레이)·E3–E5로 확장.
- **정직한 점검**: UE/Unity式 breadth + 독립 에디터는 **수년 규모**다. 본 계획의 가치는 완성이 아니라
  **올바른 의존 순서 · 백엔드 추상화 seam · 언제든 수직 슬라이스로 절단 가능한 구조**를 박아두는 것.

## 의존성 위험 / 미결 사항 (세부 계획에서 해소)

- **백엔드 디스패치**: trait object vs enum-dispatch. Phase 1.
- **디스크립터/바인드리스 모델 통일**: 두 API의 디스크립터 모델 차이 흡수 설계. Phase 1.
- **windows-rs D3D12 ergonomics**: raw COM 인터페이스라 RAII 래퍼 설계 필요. Phase 2.
- **RT 추상화**: SBT 레이아웃·AS 빌드 인터페이스의 두 API 공통화 난이도 높음. Phase 8.
- **FBX 외부 의존 + 버퍼 다용도**: `cc`+ufbx 벤더링/FBX SDK 게이트 다운로드(승인 대상), 스킨 캐시 버퍼의
  `Vertex|Storage|AccelInput` 상태 전이 두 API 통일, 비균등 스케일 노멀 매트릭스. Phase 14.
- **상용 확장(Phase 15+)**: 서드파티 의존 승인(Rapier/Jolt·kira·Luau(mlua)/wasmtime·Recast 등, facade 뒤 격리), 결정성
  vs 멀티스레드/피직스(고정 타임스텝), 에디터↔런타임 결합(리플렉션/직렬화 선결), 범위 폭주(수직 슬라이스
  절단 유지), 백엔드 추상화 누수. 세부: [commercial-engine-gap-analysis.md](commercial-engine-gap-analysis.md).

## 검증 전략 (전 단계 공통)

- 각 백엔드는 **검증 레이어 클린**(Vulkan validation layers / D3D12 debug layer + GPU-based validation)을
  기본 게이트로 둔다.
- 모든 핵심 마일스톤은 **D3D12·Vulkan 양쪽에서 동일 결과**를 내는 것을 합격 기준으로 한다.
- RenderDoc/PIX 캡처로 시각적·리소스 상태 검증.
- 샌드박스 앱에 각 기법별 씬을 추가해 회귀 확인.
