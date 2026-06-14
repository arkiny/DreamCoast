# Phase 0 — 기반 다지기 (Foundations) 세부 계획

> 상위 로드맵: [ROADMAP.md](ROADMAP.md). 이 문서는 그 중 **Phase 0**의 실행 계획이다.
> **상태: ✅ 완료** (셰이더 컴파일 절반은 slangc 확보 시 자동 활성화 — 아래 "남은 작업" 참조)

## Context (이 단계의 목표)

커스텀 그래픽스 엔진의 토대를 만든다. 아직 렌더링은 없다. Phase 0의 목적은
**(1) 빌드 가능한 cargo workspace 골격**, **(2) Win32 빈 창 + 메시지 루프**,
**(3) Slang → SPIR-V + DXIL 자동 컴파일 빌드 파이프라인**, **(4) 공통 인프라(로깅/에러)**
를 세워, 이후 모든 Phase가 얹힐 발판을 확보하는 것이다.

**완료 기준**: `cargo run -p sandbox` 시 빈 Win32 창이 뜨고(ESC/닫기로 정상 종료),
빌드 과정에서 `triangle.slang`이 `.spv`와 `.dxil` 두 포맷으로 컴파일된다.

## 확정 사항 (Phase 0)

| 항목 | 결정 |
|---|---|
| 윈도잉 | 자체 Win32 (`windows` crate, WNDPROC + 메시지 펌프 직접 구현) |
| Slang 통합 | `build.rs`에서 `slangc` 호출 → `.spv` + `.dxil` 생성 |
| Rust edition | 2024 (rustc 1.94) |
| 로깅 | `tracing` + `tracing-subscriber` |
| 에러 | 라이브러리 crate은 `thiserror`, 앱은 `anyhow` |
| 수학 | `glam` (core에서 재노출) |

## 외부 도구 의존성 (시스템 설치 최소화)

**원칙: 무거운 Vulkan SDK 전체 설치를 피하고, Phase 0에 꼭 필요한 `slangc`만 경량으로 확보한다.**

- **Vulkan 런타임**: 별도 설치 불필요. `ash`가 GPU 드라이버 제공 `vulkan-1.dll`을 런타임 동적 로드(Phase 1).
- **slangc (SDK 불필요)**: [shader-slang 릴리스](https://github.com/shader-slang/slang/releases)의
  독립 실행 zip(slangc.exe + `slang.dll`, DXIL 출력용 `dxcompiler.dll`/`dxil.dll` 포함)을 받아
  SPIR-V·DXIL 둘 다 생성. 아래 셋 중 하나로 노출(`build.rs`가 모두 지원):
  1. `SLANGC` 환경변수에 `slangc.exe` 전체 경로 지정, 또는
  2. 저장소 내 `tools/slang/bin/`에 배치(.gitignore로 바이너리는 커밋 제외), 또는
  3. `slangc.exe`를 PATH에 추가.
- **검증 레이어 / dxc / D3D12**: Phase 0 불필요. Vulkan 검증 레이어는 Phase 1에서 standalone
  레이어를 `VK_LAYER_PATH`로 지정하는 경량 방식 사용. DXIL은 slangc 번들 DXC로 충분.
  D3D12는 시스템 `d3d12.dll`로 충분(Agility SDK는 Phase 8 RT에서 도입).

## 실제 구현 결과 (워크스페이스)

```
D:\Playground\
├── Cargo.toml              # [workspace] + [workspace.dependencies] + 프로필
├── rust-toolchain.toml     # stable + rustfmt/clippy
├── rustfmt.toml            # edition 2024, max_width 100
├── .gitignore              # /target, /tools/slang/, /shaders/compiled
├── docs/                   # 계획 문서 (이 폴더)
│   ├── ROADMAP.md
│   └── phase-0-foundations.md
├── crates/
│   ├── core/               # init_logging(tracing), EngineError(thiserror),
│   │                       #   Handle/Pool(제너레이셔널 인덱스), glam 재노출
│   ├── platform/           # Win32 Window(클래스 등록·WNDPROC·PeekMessage 펌프·
│   │                       #   HWND/HINSTANCE 노출) + Input(키/마우스)
│   └── shader/             # build.rs(slangc 해석·컴파일) + shaders/triangle.slang
│       │                   #   + 생성된 접근자(<key>_spirv()/_dxil())
│       └── ...
└── apps/
    └── sandbox/            # init_logging → Window → 프레임 루프 → ESC/닫기 종료
```

### 핵심 구현 메모

- **core::Pool / Handle** — 제너레이셔널 인덱스 아레나. 이후 RHI 리소스 핸들(Buffer/Texture/
  Pipeline 등)에 재사용. stale 핸들은 generation 불일치로 `None` 반환(테스트 포함).
- **platform::Window** — 윈도우별 가변 상태(`WindowState`)를 힙에 두고 포인터를
  `GWLP_USERDATA`에 보관하는 표준 Win32 패턴. `WM_NCCREATE`에서 `CREATESTRUCTW`로 포인터 수신.
  ESC/닫기 → `should_close`. `WM_SIZE` → 크기/리사이즈 플래그. `pump_events()`는 `PeekMessageW`
  논블로킹 루프(프레임 루프와 맞물림).
- **shader::build.rs** — slangc 해석 순서: ① `SLANGC` env → ② `tools/slang/bin/slangc.exe`
  → ③ PATH → ④ `%VULKAN_SDK%\Bin`(호환). 각 엔트리×타깃(spirv/dxil)을 `sm_6_5` 프로파일로 컴파일,
  산출물은 `OUT_DIR`. slangc 부재 시 **빌드 비차단**(경고 + 접근자 `None` 생성). slangc 컴파일
  실패(비제로 종료)는 stderr 출력 후 panic. `OUT_DIR/shaders.rs`를 생성해 `lib.rs`가 `include!`.

## 검증 결과

| 항목 | 결과 |
|---|---|
| `cargo build` (워크스페이스 전체) | ✅ |
| `cargo fmt --check` | ✅ |
| `cargo clippy --all-targets -- -D warnings` | ✅ |
| `cargo test -p engine-core` (Pool 2개) | ✅ |
| `cargo run -p sandbox` | ✅ 1280×720 창 생성·리사이즈 로그·정상 종료, 패닉 없음 |
| slangc 미설치 시 빌드 비차단(경고+스킵) | ✅ |

## 남은 작업 (셰이더 컴파일 절반 활성화)

`slangc` 확보 후 `cargo build` 재실행 → `cargo:warning=using slangc at ...` 출력 +
`OUT_DIR`에 `triangle_vs.spv/.dxil`, `triangle_fs.spv/.dxil` 생성 + `triangle_vs_spirv()` 등이
실제 바이트코드 반환. 이로써 Phase 0 완료 기준의 셰이더 컴파일 절반이 검증된다.

## 다음 단계 (Phase 1 예고)

RHI trait 시그니처 초안(Device/Queue/Swapchain/CommandBuffer/Buffer/Texture/Pipeline/
Descriptor/Sync), `ash` Vulkan 백엔드로 `triangle.slang`(SPIR-V) 기반 hello-triangle,
검증 레이어 클린 달성.
