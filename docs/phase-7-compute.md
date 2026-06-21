# Phase 7 — 컴퓨트 / GPGPU 세부 계획

> 상위 로드맵: [ROADMAP.md](ROADMAP.md). **상태: 🚧 계획 (승인 대기)** — 구현 전 사인오프 단계.
> 전제: Phase 6(디퍼드 PBR + 렌더그래프) ✅ 완료. 이 Phase는 Phase 8(RT)·Phase 10(Virtual Geometry)의 전제다.

## Context

지금까지 렌더그래프는 **그래픽스 패스만** 안다: 컬러/깊이 어태치먼트 + 바인드리스 *샘플* 읽기.
Phase 7은 여기에 **컴퓨트 패스**와 **read-write(UAV/storage) 리소스**를 처음 도입한다.
바인드리스 디스크립터는 현재 FRAGMENT 스테이지 전용이며 sampled image(binding 0) + cube(binding 2)만 있다 —
컴퓨트가 쓰려면 COMPUTE 스테이지 가시성과 storage(UAV) 테이블이 새로 필요하다.

**완료 기준 (ROADMAP)**: 컴퓨트 기반 효과가 렌더 패스와 연동.

## 확정 사항 (사용자)

- **예제 범위: 세 가지 모두** — (1) 컴퓨트 포스트프로세싱, (2) GPU 파티클 시뮬레이션, (3) GPU 컬링 + indirect draw.
  GPU 컬링은 Phase 10(Virtual Geometry)의 전제를 바로 깔아준다.
- **그래프 통합: 렌더그래프 1급 컴퓨트 패스** — 그래프에 compute pass + storage 리소스 + 자동 storage 배리어 도입.
  엔진 원칙 #4(렌더그래프 = 모든 기법의 척추)에 부합.

## 핵심 설계 결정 (제안 — 사인오프 대상)

1. **큐 모델: 그래픽스 큐 단일 사용 (async compute 미도입).**
   컴퓨트는 메인 그래픽스 큐에 같이 기록·제출한다. 별도 컴퓨트 큐(async compute)는 큐 간 세마포어
   동기화·리소스 소유 분할이라는 큰 복잡도를 더하며 정확성에는 불필요 → **이후(Phase 9 폴리시)로 연기**.
   D3D12의 펜스-에뮬레이트 동기 모델과 Vulkan FIFO 큐 모두 단일 큐로 유지.

2. **바인드리스 스테이지 가시성 확장.**
   바인드리스 디스크립터(sampled image / sampler / cube)의 `stage_flags`를 `FRAGMENT` →
   `FRAGMENT | COMPUTE`로 넓힌다(컴퓨트가 같은 테이블을 읽기 위함). D3D12 루트 시그니처는 SRV 테이블
   가시성을 `ALL`로 둔다(그래픽스+컴퓨트 공용 루트 시그니처가 필요 — C 항목 참조).

3. **Storage 이미지 = `RenderTarget` 확장 (신규 타입 대신).**
   컴퓨트가 쓰고 그래픽스가 샘플하는 출력 이미지는 기존 `RenderTarget`에 **STORAGE(UAV) usage + UAV 뷰 +
   storage-image 바인드리스 인덱스**를 추가해 표현한다. 이러면 그래프의 transient/aliasing/풀 인프라를
   그대로 재사용하고, 한 리소스가 컴퓨트 UAV write → 그래픽스 SRV read로 자연스럽게 흐른다.
   (대안인 별도 `StorageImage` 타입은 풀/aliasing을 중복 구현해야 해 기각.)

4. **Storage 버퍼 = 바인드리스 storage-buffer 테이블.**
   파티클·컬링 버퍼는 새 바인드리스 storage-buffer 테이블(Vulkan binding 3 `STORAGE_BUFFER`,
   D3D12 UAV range)에 등록하고 push-constant 인덱스로 접근한다 — 엔진의 bindless-first 원칙과 일치.
   (루트 UAV 직접 바인딩보다 셰이더 작성이 균일하고, 여러 버퍼를 한 패스에서 다루기 쉽다.)

5. **Indirect draw.**
   `BufferUsage::Indirect` 도입(Vulkan `INDIRECT_BUFFER` usage; D3D12는 **command signature** 필요).
   `CommandBuffer::draw_indexed_indirect(args, offset, draw_count)`. 컴퓨트가 atomic으로 채운 args 버퍼를
   그래픽스가 그대로 소비.

6. **영속(persistent) vs transient 그래프 리소스.**
   파티클 상태 버퍼는 **프레임 간 영속**이라 transient(매 프레임 재할당)일 수 없다 → 그래프에 외부 리소스
   **import 경로**(`import_buffer`/`import_storage_image`, 백버퍼 import와 동형)를 추가한다. 컬링용 args/가시
   리스트는 매 프레임 새로 쓰므로 transient로 둔다.

---

## 마일스톤 (M1–M7, 각 게이트: build + fmt + clippy `-D warnings` + 두 백엔드 + Vulkan 검증 클린 + 스크린샷)

### M1 — 컴퓨트 파이프라인 + dispatch (RHI 코어)

- **rhi-types**: `ComputePipelineDesc { compute_bytes, compute_entry, push_constant_size, bindless }`.
- **파사드**: `ComputePipeline` enum, `Device::create_compute_pipeline`,
  `CommandBuffer::{bind_compute_pipeline, dispatch(x,y,z)}` (push constants는 기존 `push_constants` 재사용).
- **Vulkan**(`pipeline.rs`): `vk::ComputePipelineCreateInfo` + 컴퓨트 파이프라인 레이아웃(바인드리스 set 0 포함).
  디스크립터 set·push 는 `BIND_POINT::COMPUTE`로 바인드. 바인드리스 `stage_flags`에 COMPUTE 추가(결정 #2).
- **D3D12**(`pipeline.rs`): `CreateComputePipelineState` + 컴퓨트/공용 루트 시그니처, `Dispatch`.
- **셰이더 빌드**(`build.rs`): `stage:"compute"` 잡 지원(현재 vertex/fragment만). `sm_6_5` 유지.
- **검증용 최소 셰이더**: `compute_gradient.slang`이 storage image에 그라디언트 write → 화면 표시(M2 의존).
  M1 단독 검증은 "컴퓨트 PSO 생성 + dispatch가 검증 레이어 클린"까지.

### M2 — Storage 리소스 (UAV): storage image + storage buffer + 바인드리스 UAV 테이블 + 배리어

- **rhi-types**: `BufferUsage::{Storage, Indirect}` 추가. `RenderTargetDesc`에 `storage: bool`(UAV 겸용 플래그).
- **Storage image (결정 #3)**: `RenderTarget`에 STORAGE usage·UAV 뷰·storage 바인드리스 인덱스 추가.
  - Vulkan: 이미지 usage에 `STORAGE` 추가, 새 바인드리스 binding 3(`STORAGE_IMAGE`) 테이블 + `register_storage_image`.
  - D3D12: 리소스 `ALLOW_UNORDERED_ACCESS`, 힙에 UAV 디스크립터 + 바인드리스 UAV range.
  - 레이아웃/상태 머신 확장: Vulkan `GENERAL`(storage) ↔ `SHADER_READ_ONLY`/`COLOR_ATTACHMENT`,
    D3D12 `UNORDERED_ACCESS` ↔ `PIXEL_SHADER_RESOURCE`/`RENDER_TARGET`.
- **Storage buffer (결정 #4)**: `Device::create_storage_buffer`(GPU-local, UAV) + 바인드리스 storage-buffer 테이블.
  - Vulkan: `STORAGE_BUFFER` usage + binding 4 `STORAGE_BUFFER` 바인드리스, `register_storage_buffer`.
  - D3D12: UAV 버퍼(structured/raw) + UAV range. (host upload용 별도 staging은 기존 버퍼 경로 재사용.)
- **배리어 파사드**: `CommandBuffer::{buffer_uav_barrier, storage_to_sampled, sampled_to_storage,
  storage_image_to_*}` — 컴퓨트 write↔read, 컴퓨트↔그래픽스 전이.
- **검증**: `compute_gradient`가 storage image에 write → fullscreen 패스가 sampled로 읽어 표시. 두 백엔드 픽셀 일치.

### M3 — 렌더그래프 1급 컴퓨트 패스

- **render**: `PassKind { Graphics, Compute }`(또는 `add_compute_pass`). 컴퓨트 패스는 `begin_rendering` 없이
  `bind_compute_pipeline`/`dispatch` 기록.
- **그래프 리소스 확장**:
  - `create_storage_image` / `create_storage_buffer` (transient), `import_buffer` / `import_storage_image` (영속, 결정 #6).
  - `PassInfo`에 `storage_reads` / `storage_writes` 추가. DAG 엣지(RAW/WAW/WAR)가 storage 리소스도 포함.
- **자동 배리어**: 스케줄상 write 패스 뒤 read 패스 사이에 storage→read(또는 UAV) 배리어 자동 삽입.
  컴퓨트→그래픽스(샘플), 그래픽스→컴퓨트(샘플드→storage) 전이를 execute 루프에서 처리.
- **PassContext**: 컴퓨트용 `dispatch`, `storage_index(id)`(바인드리스 인덱스) 노출.
- **검증**: M2의 그라디언트 데모를 그래프 컴퓨트 패스로 재배선 → 동일 결과.

### M4 — 예제 1: 컴퓨트 포스트프로세싱

- HDR 라이팅 결과(`Rgba16Float`)를 입력으로 컴퓨트 셰이더가 효과(예: 분리형 블러 또는 톤매핑 전처리)를
  **storage image에 write** → 톤매핑/표시 패스가 샘플. 8×8 워크그룹.
- 기존 풀스크린 포스트 경로와 **나란히** 두고 ImGui로 "컴퓨트 포스트" 토글(그래픽스 vs 컴퓨트 경로 비교).
- **검증**: 컴퓨트 경로 on/off 결과 비교, 두 백엔드 일치.

### M5 — 예제 2: GPU 파티클 시뮬레이션

- 파티클 storage 버퍼(영속, `import_buffer`) — pos/vel. 컴퓨트가 매 프레임 적분(중력/수명/리스폰), in-place 갱신.
- 갱신된 버퍼를 **인스턴스 드로우 또는 vertex-pull**로 점/쿼드 렌더(바인드리스 storage-buffer 인덱스로 VS가 읽기).
- 컴퓨트 write → 그래픽스 read 배리어는 그래프가 자동 삽입.
- 초기화: 첫 프레임 시드(또는 일회성 컴퓨트). ImGui로 파티클 수/방출 토글.
- **검증**: 파티클이 시간에 따라 움직이고 두 백엔드에서 동일 거동(고정 시드/고정 dt 스크린샷).

### M6 — 예제 3: GPU 컬링 + indirect draw

- 인스턴스 데이터 버퍼(transform/AABB). 컴퓨트가 프러스텀 컬링 → **가시 인스턴스 리스트 + indirect args**를
  atomic 카운터로 기록.
- `draw_indexed_indirect`로 가시 인스턴스만 그리기.
  - Vulkan: `vkCmdDrawIndexedIndirectCount` 또는 `vkCmdDrawIndexedIndirect`(args 버퍼).
  - D3D12: `ID3D12CommandSignature` + `ExecuteIndirect`(args 버퍼 + 카운트 버퍼).
- 셰이더 atomic(`InterlockedAdd`)로 가시 카운트 누적.
- ImGui로 인스턴스 그리드 크기/컬링 on-off, 컬링 통계(가시/전체) 표시.
- **검증**: 카메라 밖 인스턴스가 args에서 빠짐(통계로 확인), 컬링 on/off 화면 일치(보이는 것만), 두 백엔드 일치.
- Phase 10 전제(인다이렉트·atomic·GPU-driven) 확보.

### M7 — 마무리

- ImGui 폴리시(세 데모 토글/통계 정리), `docs/phase-7-compute.md` 검증/한계 채우기,
  `docs/ROADMAP.md` Phase 7 → ✅ 완료, 백로그 메모리 갱신.
- 전 백엔드 build + fmt + clippy(`-D warnings`) + Vulkan 검증 클린, 스크린샷 두 백엔드 일치.
- Phase 7 커밋.

---

## 검증 전략 (전 마일스톤 공통)

- `cargo fmt --all`, `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets` clean.
- 두 백엔드 `--backend vulkan|d3d12` 실행 + `--screenshot[-clean]`로 PNG 캡처 후 Read로 시각 확인, 픽셀 일치.
- `VK_LOADER_LAYERS_DISABLE="~implicit~"` Vulkan 검증 레이어 클린(컴퓨트 PSO·UAV 배리어·indirect 모두 VUID 없음).
- 리포지토리 주석/문서에 **타사 엔진/상표명 미사용**(사용자 라이선스 요구) — 표준 기법명으로 기술.

## 알려진 한계 / 이후 작업 (예정)

- **async compute 미도입**(단일 그래픽스 큐) — 별도 컴퓨트 큐는 Phase 9 폴리시로 연기.
- 바인드리스 슬롯 monotonic(상한) — storage 테이블도 free-list 없음(Phase 5에서 이월된 과제).
- 파티클은 단일 버퍼 in-place(더블 버퍼/정렬/충돌 없음) — 데모 수준.
- 컬링은 프러스텀만(오클루전/HZB는 Phase 10).
- (이월) 런타임 Slang 리플렉션/핫리로드 — 빌드타임 임베드 유지.
