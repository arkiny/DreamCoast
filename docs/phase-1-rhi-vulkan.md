# Phase 1 — RHI 코어 + Vulkan 백엔드 (Hello Triangle) 세부 계획

> 상위 로드맵: [ROADMAP.md](ROADMAP.md). **상태: ✅ 완료** (RTX 2070 SUPER / Vulkan 1.4 검증)

## Context (이 단계의 목표)

Phase 0의 빈 창 위에 **첫 GPU 렌더링**을 올린다. enum-dispatch 기반 RHI 골격을 세우고,
`ash`로 Vulkan 백엔드를 구현해 `triangle.slang`(SPIR-V)을 화면에 그린다. 이 단계에서 정한
RHI 표면·동기화·스왑체인·파이프라인 패턴이 이후 D3D12(Phase 2)와 모든 렌더 기법의 기준이 된다.

**완료 기준**: `cargo run -p sandbox` → 창에 RGB 삼각형 표시, 리사이즈 동작, (검증 레이어가
설치돼 있으면) 검증 에러 0건으로 정상 구동/종료.

## 확정 사항 (사용자 결정)

| 항목 | 결정 |
|---|---|
| 백엔드 디스패치 | **enum-dispatch** (`enum Device { Vulkan(..) }`, 런타임 선택, vtable 없음) |
| Phase 1 RHI 범위 | **최소 수직 슬라이스** (삼각형에 필요한 타입만) |
| 검증 레이어 | **있으면 켜기** (`VK_LAYER_KHRONOS_validation` 탐지 시 활성화) |

## 선행 조건

- **slangc 필수**: 삼각형 SPIR-V 필요. SLANGC env / `tools/slang/bin/` / PATH 중 하나로 확보.
- **Vulkan 1.3 GPU/드라이버**: dynamic rendering + synchronization2 코어 사용. (확인된 환경: RTX 2070 SUPER)

## 크레이트 구조

```
crates/
├── rhi-types/     # 백엔드 무관 plain 타입 (deps 없음)
├── rhi-vulkan/    # ash 기반 Vulkan 구현. deps: rhi-types, engine-core, ash, windows, engine-shader
└── rhi/           # enum-dispatch 파사드 + rhi-types 재노출. deps: rhi-types, rhi-vulkan
```

- `sandbox`는 **오직 `rhi` 파사드**에만 의존 → Phase 2(d3d12) 전환 시 무수정.

## RHI 표면 (최소 슬라이스)

- `Instance::new(window, InstanceDesc{validation})` → 인스턴스+서피스+물리디바이스
- `Instance::create_device()` → `Device`(+큐)
- `Device::create_swapchain(SwapchainDesc)` / `create_graphics_pipeline(GraphicsPipelineDesc)`
  / 커맨드버퍼·동기화 객체 생성 / `wait_idle()`
- `Swapchain::acquire_next_image` / `present` / `recreate`
- `CommandBuffer`: `begin/end`, `image_barrier`, `begin_rendering`/`end_rendering`,
  `set_viewport_scissor`, `bind_pipeline`, `draw`
- `Queue::submit(cmd, wait_sem, signal_sem, fence)`

## Vulkan 구현 단계 (`rhi-vulkan`) — dynamic rendering + synchronization2

1. Entry/Instance (`VK_KHR_surface`+`VK_KHR_win32_surface`, 검증 레이어 탐지 시 +`VK_EXT_debug_utils`+메신저, API 1.3)
2. Surface (Win32, HWND/HINSTANCE)
3. Physical device/Queue family (graphics+present+swapchain+Vulkan13 features)
4. Device/Queue (`VK_KHR_swapchain`, Vulkan13Features{dynamic_rendering, synchronization2})
5. Swapchain (B8G8R8A8_SRGB 우선, FIFO, min+1, 이미지 뷰; 재생성 헬퍼)
6. Pipeline (engine-shader SPIR-V, PipelineRenderingCreateInfo, 정점입력 없음, 동적 viewport/scissor)
7. Commands/Sync (커맨드 풀+버퍼, 프레임 인플라이트 2)
8. Frame loop (acquire → 배리어 UNDEFINED→COLOR → begin_rendering(clear) → draw(3) → end →
   배리어 COLOR→PRESENT → submit → present; OUT_OF_DATE/resize 시 재생성)
9. Drop: device_wait_idle 후 역순 파괴

## 검증 결과

| 항목 | 결과 |
|---|---|
| `cargo build` (워크스페이스 전체) | ✅ |
| `cargo clippy --all-targets -- -D warnings` | ✅ |
| `cargo fmt --check` | ✅ |
| `cargo run -p sandbox` | ✅ instance→device→swapchain→pipeline 생성, 렌더 루프 4s+ 무오류 구동, 정상 종료 |
| 셰이더 컴파일 (SPIR-V) | ✅ slangc `tools/slang/bin/`, `-fvk-use-entrypoint-name`로 엔트리명 보존 |

- 검증 레이어(`VK_LAYER_KHRONOS_validation`)는 미설치라 "있으면 켜기" 설계대로 경고만 출력하고 진행.
  엄격 검증은 SDK/standalone 레이어를 `VK_LAYER_PATH`로 제공 시 활성화됨.
- 창 픽셀 캡처는 Windows의 foreground 제약으로 자동화 불가 → `cargo run -p sandbox`로 육안 확인.

## 구현 중 해결한 이슈 (메모)

- **디바이스 생성 `ERROR_UNKNOWN`**: 파이프라인 단계에서 발생. 원인은 Slang이 SPIR-V 엔트리포인트를
  기본적으로 `main`으로 출력 → 파이프라인이 `vsMain`/`fsMain`을 못 찾음. `build.rs`에 SPIR-V 타깃 한정
  `-fvk-use-entrypoint-name` 추가로 해결.
- **1.3 features 체인**: `PhysicalDeviceFeatures2`로 `Vulkan13Features`(dynamic_rendering,
  synchronization2) 체인.
- **WSI 세마포어 재사용 위험**: render-finished 세마포어를 스왑체인 이미지당 1개로, acquire는
  `Option<u32>`(OUT_OF_DATE 시 세마포어 미신호 → 안전 재사용)로 처리.

## 다음 단계

- **DXC (Phase 2 선행)**: DXIL 출력에는 `dxcompiler.dll`/`dxil.dll`이 필요(현재 Slang 독립본에 미포함).
  Phase 2 D3D12에서 DXC를 `tools/`에 확보하면 `build.rs`의 DXIL 타깃이 자동 활성화됨(현재는 경고+`None`).
- Phase 2: `rhi-d3d12` + 파사드 `D3d12` 변형 + 런타임 `--backend` 전환, 동일 삼각형 패리티.
- 바인드리스/디스크립터/Buffer/Texture는 Phase 4(텍스처)에서 도입.
