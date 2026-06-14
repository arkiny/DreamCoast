# Phase 3 — ImGui 통합 + 바인드리스 기반 세부 계획

> 상위 로드맵: [ROADMAP.md](ROADMAP.md). **상태: ✅ 완료** (RTX 2070 SUPER, 두 백엔드 검증)
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

## 검증 결과

| 항목 | 결과 |
|---|---|
| `cargo build` / `clippy -D warnings` / `fmt --check` | ✅ |
| 텍스처드 쿼드 스모크 (바인드리스 샘플링) | ✅ 두 백엔드 |
| `--backend d3d12` / `--backend vulkan` ImGui | ✅ 폰트 atlas(바인드리스) + 데모 창 + 클리어색/FPS, 렌더 루프 무오류 |

- 창 픽셀 자동 캡처는 Windows foreground 제약으로 불가 → `cargo run -p sandbox -- --backend d3d12` 육안 확인.
- ImGui 창에서 클리어색 슬라이더가 배경색을 실시간 변경(입력 브리지 동작 확인).

## 구현 메모 / 해결한 이슈
- **D3D12 PSO `E_INVALIDARG`**: 바인드리스 SRV 범위는 unbounded(`NumDescriptors=u32::MAX`)여야 하고,
  `imgui.slang`에 명시적 `register(t0/s0/b0)`가 있어야 함(없으면 PSO 생성 실패).
- **NDC y-flip**: ortho 투영의 scale/translate.y가 Vulkan(아래로 +)과 D3D12(위로 +)에서 반대.
- **동적 버퍼 수명**: 정점/인덱스 버퍼를 frame-in-flight별로 보관하고 필요 시 grow(해당 프레임 펜스가
  재사용 전 대기를 보장).
- 메모리 할당자/유저 ImTextureID/도킹/키보드 전체 매핑은 이후 Phase.

## 다음 단계
- Phase 4: 셰이더 시스템(리플렉션/핫리로드) + glTF/텍스처 에셋 파이프라인.
