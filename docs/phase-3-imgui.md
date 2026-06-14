# Phase 3 — ImGui 통합 + 바인드리스 기반 세부 계획

> 상위 로드맵: [ROADMAP.md](ROADMAP.md). **상태: 🚧 진행 중**
> ⚠️ 가장 큰 Phase. ImGui가 RHI에 버퍼/텍스처/바인드리스/정점입력/블렌딩/푸시상수를 처음 도입.
> 구현 순서: (A) RHI 확장 → 텍스처 1장 바인드리스 샘플링 검증 → (B) gui 크레이트.

## Context

Phase 0~2는 리소스 없는 삼각형이었다. Phase 3은 Dear ImGui 디버그 UI를 두 백엔드에 올린다.
ImGui는 동적 정점/인덱스 버퍼 + 폰트 텍스처(샘플링) + 알파 블렌딩 + 스시저를 요구 → 이를 계기로
RHI의 **버퍼·텍스처·바인드리스 디스크립터 모델**("bindless-first")을 세운다.

**완료 기준**: 두 백엔드 모두 삼각형 위 ImGui 창(데모/통계/컨트롤) + 마우스/키보드 동작, clippy/fmt 통과.

## 확정 사항

| 항목 | 결정 |
|---|---|
| 디스크립터 | 바인드리스 지금 도입 (Vulkan descriptor indexing / D3D12 unbounded SRV table) |
| ImGui 범위 | 기본 UI (단일 뷰포트, 도킹 없음) |
| 크레이트 | `imgui`(imgui-rs) + 커스텀 RHI 렌더러 |

## A. RHI 확장
- rhi-types: `BufferDesc/BufferUsage`, `TextureDesc`, `VertexLayout`(pos2/uv2/unorm8x4), `BlendMode`,
  `Rect2D`, `GraphicsPipelineDesc` 확장(vertex_layout/blend/push_constant_size/bindless).
- 파사드/백엔드: `Buffer`(create/write), `Texture`(create+업로드+바인드리스 등록, bindless_index),
  커맨드 `set_scissor/bind_vertex_buffer/bind_index_buffer/push_constants/draw_indexed`.
- Vulkan: Vulkan12 descriptor-indexing features, 바인드리스 셋(SAMPLED_IMAGE[N] + immutable sampler),
  host-visible 버퍼, device-local 텍스처+staging.
- D3D12: 루트시그(unbounded SRV table + root constants + static sampler), shader-visible 힙,
  UPLOAD 버퍼, DEFAULT 텍스처+업로드 복사.

## B. 셰이더 `imgui.slang`
- 정점 pos2/uv2/unorm8x4, push 상수(scale/translate/tex_index), `Texture2D g_textures[]` + `SamplerState`.
- fs: `col * g_textures[tex_index].Sample(g_sampler, uv)`. build.rs JOBS에 추가.

## C. `crates/gui`
- imgui Context + 폰트 atlas→Texture 업로드 + 파이프라인/버퍼. `new_frame`/`render(draw_data)`.
- Win32 입력 브리지(마우스/휠/문자/키/크기/dt). platform `Input`에 휠+문자 큐 보강.

## D. sandbox
- 삼각형 draw 후 같은 패스에 ImGui 렌더. 데모 창 + FPS/백엔드 + 클리어색 슬라이더.

## 검증
1. build/clippy/fmt 두 백엔드 통과.
2. `--backend vulkan|d3d12` ImGui 창 + 입력 + 폰트(바인드리스 샘플링) 정상, 레이어 오류 0.
3. 육안 확인은 직접 실행.

## 리스크/메모
- A부터 텍스처 1장 샘플링으로 작게 검증 후 gui 연결.
- Slang push_constant→D3D12 root constants, 바인드리스 register 매핑은 attribute로 조정.
- 메모리 할당자/유저 ImTextureID/도킹은 이후 Phase.
