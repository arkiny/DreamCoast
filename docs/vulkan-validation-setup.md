# Vulkan 검증 레이어 세팅 (SDK 비설치)

전체 Vulkan SDK 설치 없이 **standalone 검증 레이어**만 로컬에 두고 엔진이 자동으로 찾게 한다.
slangc를 `tools/slang/`에 두는 것과 같은 철학 — 머신을 깨끗하게, git에는 안 올린다.

## 받기

```bash
python tools/fetch-vulkan-layers.py        # 최신 핀 버전
python tools/fetch-vulkan-layers.py 1.4.341.0   # 버전 지정
```

- 소스: conda-forge의 prebuilt **MSVC** `vulkan-validation-layers` 패키지(Apache-2.0).
  `.conda`(zip+zstd)에서 `VkLayer_khronos_validation.dll` + 매니페스트만 추출 →
  `tools/vulkan-layers/`에 저장하고 매니페스트 `library_path`를 동일 폴더 DLL로 재작성.
- 의존: Python 3 + `zstandard`(`pip install --user zstandard`).
- `tools/vulkan-layers/`는 `.gitignore`됨 — 바이너리는 커밋/재배포하지 않는다(라이선스 클린).

## 엔진 자동 발견

`crates/rhi-vulkan/src/instance.rs`의 `add_local_layer_path()`가 `InstanceDesc.validation`이 켜져 있을 때
인스턴스 생성 전에 다음을 순서대로 찾아 `VK_ADD_LAYER_PATH`에 추가한다:

1. `$ENGINE_VK_LAYER_DIR` (수동 오버라이드)
2. 빌드타임 워크스페이스 기준 `tools/vulkan-layers/`
3. 실행 디렉터리 기준 `tools/vulkan-layers/`

따라서 받은 뒤에는 그냥 실행하면 검증이 켜진다:

```bash
cargo run -p sandbox -- --backend vulkan
```

검증 메시지는 `debug_callback`을 통해 `tracing`(target `vulkan`)으로 나온다.
시스템에 Vulkan SDK가 이미 설치돼 있으면 그 레이어가 그대로 쓰이며, 위 폴더가 비어 있어도 무해하다.

## Shipping(release) 빌드 — 검증 제외

검증은 개발 전용이다. `instance.rs`의 게이트가 `cfg!(debug_assertions)`로 묶여 있어
**release 빌드에서는 const-false가 되어 레이어 활성화·매니페스트 탐색·디버그 메신저가 정적으로
제거**된다. 즉 `cargo build --release`(= shipping)는 `InstanceDesc.validation`이 `true`라도,
`tools/vulkan-layers/`에 레이어가 있어도 검증을 전혀 로드하지 않는다.

- 개발: `cargo run -p sandbox -- --backend vulkan` → 검증 ON(레이어 있으면).
- 출시: `cargo build --release` → 검증 코드 자체가 빌드에서 빠짐.
- (release에서 프로파일링용으로 강제로 켜야 한다면 `[profile.release] debug-assertions = true`로 따로 빌드.)

## 노이즈 / 트러블슈팅

- 화면 캡처/오버레이(XSplit·OBS·Discord 등)는 **implicit 레이어**로 끼어들어 스왑체인에 STORAGE usage나
  외부 메모리 probe를 주입 → 우리 코드와 무관한 VUID가 뜰 수 있다. 우리 코드만 보려면:
  ```bash
  VK_LOADER_LAYERS_DISABLE="~implicit~" cargo run -p sandbox -- --backend vulkan
  ```
- 레이어 로딩 자체를 추적하려면 `VK_LOADER_DEBUG=error,warn`.
