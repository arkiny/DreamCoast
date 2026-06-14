# Phase 4 — 에셋 파이프라인 + 메시 렌더링 세부 계획

> 상위 로드맵: [ROADMAP.md](ROADMAP.md). **상태: ✅ 완료** (RTX 2070 SUPER, 두 백엔드 / BoxTextured.glb 검증)

## Context
2D(삼각형/ImGui)에서 나아가 **glTF 메시를 텍스처와 함께 3D로 표시**한다. RHI에 깊이 버퍼·메시 정점
레이아웃이 처음 들어오고, glTF/이미지 로딩용 `asset` 크레이트 추가. 라이팅은 디퓨즈 lambert.

**완료 기준**: 두 백엔드 모두 텍스처드 3D 모델이 카메라 궤도 회전 + 깊이 정확히 렌더, ImGui 오버레이.

## 확정 사항
- 범위: 에셋 우선(glTF+텍스처+깊이+카메라). 리플렉션/핫리로드는 다음 단계.
- 렌더링: 디퓨즈 lambert + 베이스컬러. 라이브러리 gltf/image/glam.

## A. RHI 확장
- rhi-types: `Format::Depth32Float`, `VertexLayout::Mesh`(pos3/normal3/uv2=32B), pipeline `depth_test`+`depth_format`.
- 파사드: `DepthBuffer` + `Device::create_depth_buffer(Extent2D)`; `begin_rendering(sc, idx, color_clear: Option<ClearColor>, depth: Option<&DepthBuffer>)`.
- 프레임 2패스: ① 메시(clear+depth) ② ImGui(load, depth 없음) → UI 파이프라인 깊이 무관 유지.
- Vulkan: D32_SFLOAT 이미지/뷰, depth attachment, depth-stencil state(LESS), PipelineRenderingCreateInfo depth format.
- D3D12: D32_FLOAT 리소스+DSV, OMSetRenderTargets(rtv,dsv), PSO depth+DSVFormat.

## B. mesh.slang
- 입력 pos3/normal3/uv2. push `{ float4x4 mvp; uint tex_index; }`. 모델=단위(노말=월드).
- fs: lambert(고정 광원) * 베이스컬러(바인드리스). register t0/s0/b0 + vk::binding. build.rs JOBS 추가.

## C. crates/asset (RHI 비의존)
- `MeshVertex{pos,normal,uv}`, `MeshData{vertices,indices(u32),base_color:Option<ImageData>}`, `ImageData{w,h,rgba8}`.
- `load_gltf(path)`(gltf+image), `unit_cube()` 폴백.

## D. sandbox
- 삼각형 제거. glam 카메라(perspective_rh 0..1 + look_at_rh 궤도, Vulkan y-flip).
- 모델 로드(assets/*.glb 또는 unit_cube + 체커). 정점/인덱스/텍스처 업로드. 깊이 버퍼(리사이즈 재생성).
- 메시 패스 → ImGui 패스.

## 검증 결과
| 항목 | 결과 |
|---|---|
| build / clippy -D warnings / fmt | ✅ |
| `--backend d3d12` / `--backend vulkan` | ✅ glTF(BoxTextured) 24v/36i 로드, 깊이+lambert+텍스처, 카메라 궤도, ImGui 오버레이, 레이어 오류 0 |
| glTF 폴백 | `assets/model.glb` 없으면 `unit_cube`+체커로 자동 폴백 |

- 자동 픽셀 캡처는 OS foreground 제약 → `cargo run -p sandbox -- --backend d3d12` 육안 확인.
- 메모: push 상수 68B(mat4+u32). 컬럼메이저 mvp(`to_cols_array`) + HLSL `mul(mvp,pos)`. Vulkan은 proj.y 반전.
  glTF 샘플은 `assets/`(gitignore)에 런타임 확보.

## 샘플 에셋 / 라이선스
- `tools/fetch-assets.ps1`이 [Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/Models.md)에서
  **CC0 1.0(퍼블릭 도메인)** 모델만 받아 `assets/`(gitignore)에 배치하고 `assets/CREDITS.md`(출처/저자) 생성.
  - 기본: Avocado(default `model.glb`), BoomBox, Lantern — 모두 CC0, Microsoft/sbtron.
  - CC-BY인 BoxTextured 등은 출처표기 의무가 있어 기본에서 제외(필요 시 `--model`로 직접 지정).
- 실행: `pwsh tools/fetch-assets.ps1` → `cargo run -p sandbox`(기본 model.glb) 또는 `-- --model assets/Lantern.glb`.
- 에셋 미확보 시 sandbox는 `unit_cube` + 체커로 자동 폴백.

## 다음 단계
- Phase 5(렌더그래프) 또는 Slang 리플렉션/핫리로드.
