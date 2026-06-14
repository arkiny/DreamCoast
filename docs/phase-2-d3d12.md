# Phase 2 — D3D12 백엔드 패리티 세부 계획

> 상위 로드맵: [ROADMAP.md](ROADMAP.md). **상태: ✅ 완료** (RTX 2070 SUPER에서 두 백엔드 검증)

## Context

Phase 1의 enum-dispatch RHI에 **D3D12 백엔드**를 추가해, 동일 삼각형을 Vulkan/D3D12 양쪽에서
런타임 전환 렌더. RHI 추상화가 두 네이티브 API에서 성립하는지 검증.

**완료 기준**: `cargo run -p sandbox -- --backend d3d12` / `--backend vulkan` 모두 동일 RGB 삼각형,
리사이즈/종료 정상, clippy/fmt 통과.

## 확정 사항

| 항목 | 결정 |
|---|---|
| DXC | 자동 다운로드 → `tools/slang/bin/`에 `dxcompiler.dll`+`dxil.dll` (✅ 완료, DXIL 컴파일됨) |
| 기본 백엔드 | OS별: Windows→D3D12, 그 외→Vulkan. `--backend`로 override |
| 동기화 매핑 | D3D12 에뮬레이션 (세마포어 no-op, 펜스=모노토닉+이벤트), 파사드/sandbox 무변경 |

## 동기화 매핑 (Vulkan 표면 → D3D12)

- `D3d12Semaphore` = no-op (빈 타입). 단일 DIRECT 큐 내 순서는 큐가 보장.
- `D3d12Fence` = `ID3D12Fence` + Win32 이벤트 + 모노토닉 카운터(`Cell<u64>`). `wait()`=목표값 대기,
  `reset()`=no-op, signaled 생성=목표 0. 큐 `submit`이 ExecuteCommandLists 후 `Signal(++value)`.
- `acquire_next_image` = `GetCurrentBackBufferIndex()`. frames-in-flight(2) 펜스 게이팅 + 버퍼 3으로 안전.

## 크레이트 `crates/rhi-d3d12` (rhi-vulkan 미러, windows-rs)

타입: `D3d12Instance/Device/Queue/Swapchain/GraphicsPipeline/CommandBuffer/Fence/Semaphore`.
COM은 Drop 시 자동 Release. `Arc<DeviceShared>`로 device/queue 공유.
features: `Win32_Foundation, Win32_Graphics_Direct3D12, Win32_Graphics_Direct3D,
Win32_Graphics_Dxgi, Win32_Graphics_Dxgi_Common, Win32_System_Threading`.

단계: instance(디버그레이어/factory/어댑터) → device/queue → swapchain(FLIP_DISCARD,count=3,RTV힙) →
pipeline(빈 루트시그+PSO/DXIL) → command(allocator+list, barrier, OMSetRT+Clear, draw) →
sync(펜스 에뮬레이션). Queue submit=Execute+Signal, present=Present(1,0).

## 파사드 + sandbox

- `rhi`: 각 enum에 `D3d12` 변형 + match 암, `Instance::new` D3D12 분기 실구현, 혼합백엔드는 unreachable.
- `sandbox`: `default_backend()`=cfg(windows)?D3d12:Vulkan, `--backend` 인자 override.

## 검증 결과

| 항목 | 결과 |
|---|---|
| DXC 배치 후 DXIL 컴파일 | ✅ `tools/slang/bin/`에 dxcompiler/dxil 배치, 경고 사라짐 |
| `cargo build` / `clippy -D warnings` / `fmt --check` | ✅ |
| `--backend d3d12` 실행 | ✅ adapter 선택→device→swapchain→PSO(DXIL)→렌더 루프 4s+ 무오류 |
| `--backend vulkan` 실행 | ✅ 회귀 없음 |

- 창 픽셀 자동 캡처는 Windows foreground 제약으로 불가 → `cargo run -p sandbox -- --backend d3d12` 육안 확인.
- D3D12 디버그 레이어는 "Graphics Tools" 설치 시 활성(메시지는 OS 디버그 출력). InfoQueue1 콜백→tracing 연동은 향후 개선.

## 구현 중 해결한 이슈 (메모)

- **PSO 생성 `E_FAIL`**: sandbox가 두 백엔드 모두에 SPIR-V를 넘기고 있었음. 백엔드별로
  SPIR-V(Vulkan)/DXIL(D3D12)을 선택하도록 수정.
- **`Arc` → `Rc`**: D3D12 `DeviceShared`는 `Cell`+COM이라 Send/Sync 아님 → 단일 스레드 엔진이므로 `Rc` 사용
  (clippy `arc_with_non_send_sync` 해소).
- **DXGI 플립 모델 sRGB**: 플립 스왑체인은 `_SRGB` 포맷 불가 → 버퍼는 UNORM, RTV/PSO는 `_UNORM_SRGB`로 sRGB 적용.
- **배리어 refcount**: `transmute_copy`로 리소스 포인터를 AddRef 없이 차용(ManuallyDrop가 Release 막음).

## 다음 단계
- Phase 3: ImGui 통합 (두 백엔드 렌더 백엔드).
