# Phase 6 — Deferred PBR 렌더러 세부 계획

> 상위 로드맵: [ROADMAP.md](ROADMAP.md). **상태: 진행 중.** 검증 게이트(각 마일스톤): 빌드 +
> `clippy -D warnings` + 두 백엔드 런타임 + **Vulkan 검증 레이어 클린**.

## Context

Phase 0–5 완료(렌더그래프 + transient aliasing). 이 Phase는 **디퍼드(G-buffer) 경로의 물리 기반
렌더링**을 두 백엔드에 구현한다: G-buffer → Cook-Torrance PBR 직접광(directional + point) → 섀도우
맵 → IBL(diffuse irradiance + specular prefilter + BRDF LUT) → HDR 톤매핑 → 백버퍼.

전제로 RHI에 처음 들어오는 것들:
- **MRT**(패스당 다중 컬러 어태치먼트) — G-buffer의 블로커. 현재는 패스당 단일 컬러만 지원.
- **부동소수 포맷** `Rgba16Float`(HDR/노멀), `Rg16Float`(BRDF LUT).
- **per-frame uniform buffer** — 카메라/라이트/섀도우/IBL 글로벌. 256B push-constant 예산 초과 해소.
- **샘플 가능한 depth**(섀도우 맵)와 **cubemap**(IBL).

## 확정 사항 (사용자)

- 범위: **Phase 6 전체를 한 번에**. 단 각 마일스톤(M1–M7)을 독립 검증 게이트로 둬 점진 확인.
- 첫(유일) 경로: **Deferred** — MRT로 G-buffer를 채우고 풀스크린 라이팅. Forward+는 채택하지 않음.

## 마일스톤

### M1 — RHI: MRT + 신규 포맷 (블로커)
- `rhi-types`: `Format::{Rgba16Float, Rg16Float}`; `GraphicsPipelineDesc.color_format` → `color_formats: &[Format]`
  (빈 슬라이스 = depth-only). 파사드 `begin_rendering_targets(&[(&RenderTarget, Option<ClearColor>)], depth)`.
- Vulkan: `to_vk_format` 신규 매핑, `pipeline.rs` 다중 color 포맷 + blend attachment N개, `command.rs`
  `begin_rendering_targets`(N개 `RenderingAttachmentInfo`).
- D3D12: `to_dxgi_format` 신규 매핑, `pipeline.rs` `NumRenderTargets`/`RTVFormats[i]`, `command.rs`
  N개 RTV 핸들 + 핸들별 clear.
- `crates/gui`·`apps/sandbox` 호출부 `color_formats`로 마이그레이션.
- **게이트**: 기존 블룸 데모 단일-원소 colors로 동작 그대로.

### M2 — RHI: per-frame uniform buffer (globals)
- `BufferUsage::Uniform`; `GraphicsPipelineDesc.uniform_buffer`(opt-in).
- 단일 host-visible 버퍼를 frame-in-flight 슬라이스로. Vulkan `UNIFORM_BUFFER_DYNAMIC`(binding 2,
  dynamic offset), D3D12 root CBV(param 2, `va + offset`). `cmd.set_globals(&Buffer, offset)`.
- 셰이더 규약: `ConstantBuffer<Globals> g : register(b1, space0)` + `[[vk::binding(2,0)]]`.

### M3 — Asset: PBR 머티리얼
- `load_gltf`/`MeshData`/`Material`에 metallic-roughness·normal·emissive 텍스처+팩터, base-color 팩터.
- base-color는 `Rgba8Srgb`(샘플 시 선형화). m-r·normal은 `Rgba8Unorm`(선형).

### M4 — Deferred: G-buffer + 직접광 PBR
- G-buffer: RT0 Albedo `Rgba8Unorm`(+ao), RT1 Normal `Rgba16Float`, RT2 Material `Rgba8Unorm`
  (metallic/roughness/ao). Depth `Depth32Float`. world pos는 depth+invViewProj 재구성.
- 렌더그래프: `PassInfo.colors: Vec<…>`(MRT), execute 배리어/begin 루프 갱신. 백버퍼는 단일.
- 셰이더: `gbuffer.slang`(3 MRT, normal map/TBN), `pbr.slang`(풀스크린, GGX/Smith/Fresnel,
  dir+point). HDR `Rgba16Float` scene → `post.slang` 톤매핑(ACES/Reinhard + 노출) → 백버퍼.
- sandbox 그래프: `gbuffer → lighting(HDR) → [bloom] → tonemap → ui`.

### M5 — 섀도우 맵 (directional)
- 샘플 가능 depth 타깃(SAMPLED + 바인드리스). depth-only 파이프라인(`color_formats: &[]`).
- 렌더그래프: 읽기 가능 depth를 1급 transient(쓰기=depth, 읽기=bindless). aliasing 제외, store 경로.
- `pbr.slang`: PCF. globals에 `lightViewProj` + 섀도우 맵 인덱스.

### M6 — IBL (확정: **Cubemap** 표현 + **절차적 스카이**)
- Cubemap RHI(6 레이어/CUBE view/mip/면·밉별 RTV) + 바인드리스 `TextureCube`(별도 0-base 인덱스 공간).
- 환경 소스: **절차적 스카이**(`sky.slang`) — 추후 **물리 기반 대기(atmospheric scattering) 모델**로 교체 가능하도록 sky 셰이더만 분리.
- 런타임 생성(graphics 패스): sky → env cubemap → irradiance → prefilter(roughness mip) → BRDF LUT(`Rg16Float`).
- `pbr.slang`: 배경=env 스카이박스, diffuse=irradiance·albedo, specular=prefilter·(F·brdf.x+brdf.y).
- **하위 단계**: M6a 큐브맵 RHI / M6b 스카이→env 큐브+스카이박스 / M6c irradiance+prefilter+BRDF LUT / M6d pbr IBL 항.
- **차후(설계 반영)**: 환경맵 **실시간 캡처** — 현재는 시작 시 1회 생성이지만, env 생성을 재호출 가능한 함수로 두어
  (1) 태양/시간 변경 시 재생성, (2) 씬 지오메트리를 큐브에 캡처하는 동적 환경 프로브로 확장 가능하게 함.
  실시간화 시 생성 비용(6면×4단계)을 프레임 분할/저해상도/온디맨드로 관리.

### M7 — 마무리 ✅
- 디버그 뷰: Lit/Albedo/Normal/Metallic/Roughness/Position/AO + **Direct(직접광만)/IBL(환경광만)** 분리 뷰.
- UI 토글: 점광/섀도우/실시간 env 캡처, sun dir·intensity, ambient, exposure, **Override material(metallic/roughness)**.
- 본 문서 검증/한계 채움, ROADMAP Phase 6 ✅.

## 검증 (완료)
- `cargo fmt --all` + `cargo clippy --workspace --all-targets -- -D warnings` **clean**.
- D3D12·Vulkan 양쪽 안정 렌더, **스크린샷 픽셀 일치**(스크린샷 툴 `--screenshot[-clean]`로 확인).
- **Vulkan 검증 레이어 클린**(`VK_LOADER_LAYERS_DISABLE=~implicit~`, 다중 프레임).
- 시각 확인: 디퍼드 직접광+PCF 섀도우(지면에 그림자), IBL 디퓨즈(IBL 뷰에서 지면이 하늘색 irradiance 수광),
  IBL 스페큘러(metallic override 시 환경 반사+태양 하이라이트), 스카이박스. 실시간 env 캡처 ~1.17ms/frame.

## 알려진 한계 / 이후 작업
- **그림자**: 단일 캐스케이드 directional만. CSM/포인트·스팟 섀도우 없음. 섀도우 캐스터=모델만(지면 제외).
- **IBL**: env 큐브는 카메라 위치 1프로브(시차 오차). 디퓨즈 irradiance = 저해상도 큐브(SH 아님 — 컴퓨트 도입 시).
  실시간 캡처 씬 = 지면 평면만(모델 등 볼록 지오메트리는 깊이 버퍼/백페이스 컬링 캡처 필요). → `realtime-env-capture.md`.
- **G-buffer**: 4번째 RT에 world position 저장(깊이 재구성 대신). 추후 최적화 여지.
- **머티리얼**: 첫 glTF primitive 1개 + 단일 머티리얼. emissive/AO 텍스처 미사용. 멀티 머티리얼/메시 없음.
- **포스트**: ACES 톤맵 + 간단 효과(grayscale/vignette). 블룸 체인은 M4에서 제거(차후 HDR 블룸 재추가 가능).
- **바인드리스 슬롯**: free-list 없음(증가만). cubemap 인덱스 공간 64개 상한.
