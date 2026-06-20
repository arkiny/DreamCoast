# Phase 5 — 렌더그래프 / 프레임그래프 세부 계획

> 상위 로드맵: [ROADMAP.md](ROADMAP.md). **상태: ✅ 완료** (RTX 2070 SUPER, 두 백엔드 / 빌드·clippy·런타임 + **Vulkan 검증 레이어 클린** 검증)

## Context
기성 추상화 없이 **렌더그래프**를 직접 구축한다: 패스 선언 API, 의존성 DAG + 위상정렬,
transient 리소스 lifetime 분석, 자동 배리어/상태 전이, **transient 메모리 aliasing**.
PBR·컴퓨트·RT 등 이후 모든 기법이 이 위에 얹힌다.

전제로 RHI에 **오프스크린 렌더 타깃**(어태치먼트+바인드리스 샘플 겸용)과 **범용 배리어**가 처음 들어온다.
**완료 기준**: 멀티 패스(오프스크린→포스트) 그래프가 두 백엔드에서 동작.

## 확정 사항 (사용자)
- 범위: **풀 그래프** — DAG + 위상정렬 + dead-pass culling + lifetime + transient aliasing.
- 데모: **토글식 포스트** — ImGui 콤보(None/Grayscale/Vignette) + aliasing on/off 체크박스.
- aliasing을 관찰 가능하게 하려고 **블룸 체인**(scene→blur×3→composite) 추가 → disjoint lifetime 발생.

## A. RHI 오프스크린 프리미티브
- rhi-types: `RenderTargetDesc{w,h,format}`, `MemoryRequirements{size,alignment}`.
- 파사드: `RenderTarget`(`bindless_index`), `Device::create_render_target`;
  `CommandBuffer::{begin_rendering_target, set_viewport_scissor_extent, rt_to_render_target, rt_to_sampled}`.
- Vulkan(`render_target.rs`): `COLOR_ATTACHMENT|SAMPLED` 이미지 + 뷰 + `register_sampled_image`,
  추적 레이아웃 `Cell<ImageLayout>`. 배리어는 `COLOR_ATTACHMENT_OPTIMAL ↔ SHADER_READ_ONLY_OPTIMAL`.
- D3D12(`render_target.rs`): `ALLOW_RENDER_TARGET` 리소스 + 1-슬롯 RTV 힙 + 바인드리스 SRV,
  추적 상태 `Cell<RESOURCE_STATES>`. 배리어는 `RENDER_TARGET ↔ PIXEL_SHADER_RESOURCE`(중복 시 no-op).

## B. crates/render (`dreamcoast-render`, `rhi`에만 의존)
- `RenderGraph<'a>`: `import_backbuffer` / `create_color` / `create_depth` / `add_pass(PassInfo, record)`.
  record 클로저 = `FnMut(&mut PassContext) -> Result<()>`; `PassContext`는 `cmd()` + `sampled_index(id)`(read 리소스 바인드리스 인덱스).
- `compile()`: RAW/WAW/WAR 엣지 DAG(선언 순서상 acyclic) → Kahn 위상정렬 → 백버퍼 도달 불가 패스 culling →
  스케줄 위치 기준 리소스 lifetime(first/last).
- `execute(device, pool, cmd, swapchain, image_index, aliasing)`: 2-페이즈(① 물리 리소스 실현 ② 기록).
  자동 배리어 — read 리소스 `rt_to_sampled`, write 타깃 `rt_to_render_target`(또는 aliasing 배리어),
  백버퍼는 기존 `transition_to_render_target/present`.
- `ResourcePool`(프레임당 1개): 커밋티드 타깃/깊이 캐시 + aliased 셋. 깊이는 aliasing 대상 아님.

## C. Transient 메모리 aliasing
- RHI: `TransientHeap`(Vulkan `VkDeviceMemory` + `VK_IMAGE_CREATE_ALIAS_BIT` placed / D3D12 `ID3D12Heap` + `CreatePlacedResource`),
  `Device::{render_target_memory, create_transient_heap, create_aliased_target}`, `CommandBuffer::aliasing_barrier`.
  타깃 메모리 소유는 owned/aliased 양분(Vulkan `Arc<HeapMemory>` 공유, D3D12 placed-heap COM 보존).
- 플래너(`plan_aliasing`): 컬러 transient를 first-use 순 greedy first-fit로 슬롯 배정
  (lifetime 비겹침 시 슬롯 공유) → 정렬된 offset + 힙 크기. 멀티-테넌트 슬롯 멤버는 매 프레임 첫 write 전
  aliasing 배리어로 직전 테넌트 콘텐츠 discard + 동기화. 플랜은 프레임 간 동일 → 변경 시에만(`wait_idle` 후) 재구축.
- aliasing 배리어 = Vulkan `UNDEFINED→COLOR`(prior read/write 대기 포함) / D3D12 `ALIASING` 배리어 + RT 전이.

## D. sandbox (블룸 체인)
- post.slang = **composite**(scene+bloom 가산 + mode), blur.slang = 분리형 가우시안(방향/threshold 푸시).
  풀스크린 삼각형 VS는 `flip_y`로 백엔드별 클립공간 Y 정렬(메시 패스의 proj.y flip과 동일 규약).
- 그래프: `scene → bloom_h0(bright+H) → bloom_v → bloom_h1 → composite → ui`.
  컬러 transient 4개(scene, bloom×3) 중 bloom_a·bloom_c가 lifetime 비겹침 → 슬롯 공유.
- 프레임-인-플라이트 hazard 회피: `ResourcePool`을 **프레임 슬롯당 1개** 운용(슬롯 fence 대기 후에만 재사용).
- ImGui: Post effect 콤보 + Transient aliasing 체크박스.

## 검증
- `cargo fmt --all`, `cargo clippy --workspace --all-targets`(`-D warnings`) clean.
- 두 백엔드 `--backend vulkan|d3d12` 모두 mesh(오프스크린)→블룸→composite→ImGui 안정 렌더, 크래시/패닉 없음.
- aliasing on: 로그 `4 color targets → 3 heap slots, 11520 KiB (dedicated 15360 KiB)` — 풀-res 1개분 절감.
  off(커밋티드)도 동일 출력. ImGui 토글로 즉시 전환.
- **Vulkan 검증 레이어 클린**(standalone 레이어 세팅 후, [vulkan-validation-setup.md](vulkan-validation-setup.md) 참고).
  오프스크린 타깃·자동 배리어·placed aliasing 배리어 모두 VUID 없음. 레이어 도입 시 잡힌 `shaderDrawParameters`
  미활성(Phase 1부터 잠복, SV_VertexID 셰이더) 버그를 device 생성에서 활성화하여 수정.
- ⚠️ D3D12 디버그 레이어 메시지는 stderr로 안 와 별도 캡처 필요(하드웨어 PIX/디버그 출력 권장).
  사용자 환경의 XSplit 오버레이(implicit 레이어)가 스왑체인에 STORAGE usage를 주입해 외부 VUID 노이즈를
  유발 — 우리 코드 아님(`VK_LOADER_LAYERS_DISABLE=~implicit~`로 확인).

## 알려진 한계 / 이후 작업
- 깊이 타깃은 aliasing 미대상(별도 커밋티드). 컬러만 aliasing.
- 바인드리스 슬롯은 monotonic — 리사이즈 반복 시 슬롯 누수(상한 1024). 슬롯 free-list는 이후.
- `plan_aliasing`이 매 프레임 메모리 요구치 질의(임시 리소스 생성/파괴) — 플랜 캐시는 이후 최적화.
- 패스당 단일 컬러 어태치먼트(MRT 미지원) — G-buffer(Phase 6)에서 확장.
