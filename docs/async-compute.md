# Async Compute (비동기 컴퓨트 큐) 세부 계획

> 상위 로드맵: [ROADMAP.md](ROADMAP.md) Phase 9(툴링 & 마무리)의 폴리시 항목. 전제: [Phase 7](phase-7-compute.md) ✅ 완료.
> **상태: ✅ 완료** (RTX 2070 SUPER, 두 백엔드 / 빌드·fmt·clippy(-D warnings)·런타임 + Vulkan 검증 레이어 클린). 파티클 sim이 전용 컴퓨트 큐에서 그래픽스와 오버랩 실행되고, 단일 큐 폴백 경로가 보존된다.

## Context

Phase 7은 **단일 그래픽스 큐**로 컴퓨트를 처리했다(컴퓨트 패스를 메인 그래픽스 커맨드에 같이 기록·제출).
정확성에는 충분하지만, 컴퓨트 작업이 그래픽스와 **시간상 직렬**로 실행돼 GPU의 비동기 컴퓨트 유닛이 놀게 된다.

이 작업은 **전용 컴퓨트 큐**를 도입해 컴퓨트(예: 파티클 시뮬)를 그래픽스(G-buffer/라이팅/톤매핑)와
**오버랩**시킨다. 정확성 기능이 아니라 **성능 폴리시**다 — 단일 큐 경로는 폴백으로 그대로 유지한다.

**완료 기준**: 파티클 시뮬이 별도 컴퓨트 큐에서 돌고, 그래픽스가 그 출력을 GPU-동기로 안전하게 소비하며,
두 백엔드에서 거동·픽셀이 단일 큐 경로와 동일하다(검증 레이어 클린).

## 확정 사항 (사용자)

- 진행 방식: **먼저 plan 문서 작성**(이 문서) → 사인오프 후 sandbox 통합.

## 완료된 부분 (RHI — 커밋 대기 중)

### 큐/펜스 인프라

- **Vulkan**(`device.rs`):
  - 디바이스 생성 시 **전용 컴퓨트 패밀리**(COMPUTE & !GRAPHICS, 그래픽스와 다른 인덱스) 탐색.
    있으면 그 패밀리에서 큐 1개 추가 + 전용 커맨드 풀; 없으면 `compute_family == graphics_family`로
    폴백(`has_dedicated_compute = false`, 실제 오버랩 없음 — 경로만 동작).
  - `DeviceShared`에 `graphics_family / compute_family / compute_queue / compute_command_pool /
    has_dedicated_compute` 추가. 폴백이 아닐 때만 컴퓨트 풀 파괴.
- **D3D12**(`device.rs`):
  - **COMPUTE 타입 큐**(`D3D12_COMMAND_LIST_TYPE_COMPUTE`) + 크로스-큐 동기용 `async_fence`(+ `async_value` Cell).
    D3D12는 항상 COMPUTE 큐를 노출하므로 `has_async_compute()` = 항상 true.

### 커맨드 버퍼

- **Vulkan**(`command.rs`): `from_pool` 헬퍼로 일반화 — `new`(그래픽스 풀) / `new_compute`(컴퓨트 풀).
  버퍼가 자기 풀(`pool` 필드)로 free 되도록 수정(이전엔 항상 그래픽스 풀로 free → 버그 소지).
- **D3D12**(`command.rs`): `with_type` 헬퍼 — `new`(DIRECT) / `new_compute`(COMPUTE) 커맨드 리스트.

### 큐 제출 (동기화 모델)

- **`ComputeQueue::submit(cmd, signal)`** — 컴퓨트 큐에 제출, 완료 시 `signal`을 알림. wait·fence 없음
  (프레임 페이싱은 그래픽스 제출의 펜스가 전이적으로 담당).
  - Vulkan: 바이너리 세마포어 시그널.
  - D3D12: 컴퓨트 큐가 `async_fence`를 ++값으로 Signal(세마포어는 no-op).
- **`Queue::submit_async(cmd, wait, compute_wait, signal, fence)`** — 그래픽스 큐가 컴퓨트 출력을 소비.
  - Vulkan: `wait`(이미지 획득, COLOR_OUTPUT 단계) + `compute_wait`(컴퓨트 완료, **VERTEX 단계** — 파티클
    드로우가 정점 단계에서 storage 버퍼를 읽으므로) 대기 후 `signal`/`fence` 시그널.
  - D3D12: 그래픽스 큐가 `async_fence`의 최신 값에 **GPU-side `Wait`** 후 실행, `fence` 시그널
    (세마포어는 no-op — 펜스가 크로스-큐 동기를 담당).

### 리소스 공유

- **Vulkan storage 버퍼**(`buffer.rs`): 전용 컴퓨트 패밀리가 있으면 `CONCURRENT` 공유 모드 +
  `[graphics_family, compute_family]` 인덱스(큐 간 소유권 이전 회피). 없으면 `EXCLUSIVE`.
  - D3D12는 리소스 소유권 개념이 없어 별도 처리 불필요.

### 파사드(`rhi/src/lib.rs`)

- `ComputeQueue` enum(backend_enum), `Device::{create_compute_command_buffer, compute_queue,
  has_async_compute}`, `Queue::submit_async`(상위집합 시그니처 — D3D12 arm은 `compute_wait` 무시),
  `ComputeQueue::submit`. 빌드·clippy `-D warnings` 클린.

---

## 핵심 설계 결정 (남은 sandbox 통합 — 사인오프 대상)

### 1. 파티클 상태 버퍼 **더블 버퍼링(ping-pong)** — 정확성 필수

현재 파티클은 **단일 영속 버퍼 in-place** 갱신이다. 단일 큐에서는 컴퓨트 write → 드로우 read가 같은 큐에
직렬이라 안전했다. 그러나 async로 가면 **WAR 해저드**가 생긴다:

- 프레임 N 컴퓨트가 버퍼에 write를 시작.
- 프레임 N-1 그래픽스 드로우가 **같은 버퍼**를 아직 read 중일 수 있음(2-프레임 인플라이트라 N-1 그래픽스가
  N 시작 시점에 끝나지 않았을 수 있음).

해결: **파티클 버퍼를 2개로 ping-pong**(이미 적용한 멀티바운스 큐브 ping-pong과 동형). 프레임 N 컴퓨트는
`buf[(N-1)%2]`를 읽어 `buf[N%2]`에 write, 같은 프레임 드로우는 `buf[N%2]`를 read. 이러면 프레임 N 컴퓨트가
쓰는 버퍼와 프레임 N-1 드로우가 읽는 버퍼가 달라 WAR 해저드가 사라지고 진짜 오버랩이 된다.
- 초기 시드는 두 버퍼 모두 채워 프레임 0 read가 미초기화 메모리를 보지 않게 한다(멀티바운스 부트스트랩과 동일).
- 적분이 in-place 가정이므로 read/write 인덱스를 셰이더 push에 분리해 넘긴다(현재는 단일 인덱스).

### 2. 프레임 루프 구조

프레임당:
1. (그래픽스와 무관하게) **컴퓨트 커맨드** 기록: `bind_compute_pipeline(particle_sim)` →
   push(read idx, write idx, dt, time) → `dispatch`. 컴퓨트 큐에 `ComputeQueue::submit(cmd, compute_done[frame])`.
2. **그래픽스 커맨드** 기록: 기존 그래프(G-buffer→라이팅→톤매핑→**파티클 드로우(buf[N%2] read)**→UI).
   `Queue::submit_async(cmd, image_available[frame], compute_done[frame], render_finished[frame], fence[frame])`.
3. present.

세마포어 `compute_done[frame]`을 `FRAMES_IN_FLIGHT`개 추가. 단일 큐 경로(아래 4번 폴백)에서는 파티클
sim을 기존처럼 그래픽스 그래프 컴퓨트 패스로 두고 `submit_async` 대신 `submit` 사용.

### 3. 토글 & 폴백 — `has_async_compute()` 가드

- Vulkan에서 전용 컴퓨트 패밀리가 없으면(`has_async_compute()==false`) async 경로는 오버랩 이득이 없고
  ping-pong만 오버헤드 → **단일 큐 경로로 폴백**.
- ImGui **"Async compute"** 토글(파티클이 켜져 있고 `has_async_compute()`일 때만 활성). off거나 미지원이면
  Phase 7의 단일 큐 컴퓨트 패스 경로 그대로.
- 헤드리스 토글: `ASYNC_COMPUTE` 환경변수(초기 on).

### 4. 단일 큐 경로 보존 (회귀 안전)

Phase 7의 그래프-내 컴퓨트 패스 경로는 **삭제하지 않는다**. async 토글 off / 미지원이면 그 경로가 돌아
거동·픽셀이 Phase 7과 동일해야 한다(회귀 게이트). ping-pong 버퍼는 두 경로 공용으로 둔다.

---

## 마일스톤 (각 게이트: build + fmt + clippy `-D warnings` + 두 백엔드 + Vulkan 검증 클린 + 스크린샷)

### A1 — RHI 백엔드 + 파사드 ✅ 완료(커밋 대기)

위 "완료된 부분" 참조. 빌드·clippy 클린. 아직 호출하는 곳이 없어 런타임 미검증.

### A2 — 파티클 ping-pong 버퍼 (정확성 선행) ✅

- 파티클 storage 버퍼 **2개**(`particle_buffers[2]`) + `particle_sim.slang` PushConstants에 `read_index`/
  `write_index` 분리(20→24B; `src.Load`→`dst.Store`). `particle_sim_push`도 read/write 분리.
- 프레임마다 `particle_parity` 토글 → write=`[parity]`, read=`[parity^1]`. draw는 write 버퍼를 읽음.
- 두 버퍼 모두 startup init dispatch로 시드(프레임 0 read 안전). 단일 큐 경로에서 먼저 적용해 분수 거동이
  Phase 7과 동일함 확인(두 백엔드).

### A3 — sandbox async 통합 ✅

- per-frame `compute_command_buffers`(`create_compute_command_buffer`) + `compute_done` 세마포어 +
  `compute_queue = device.compute_queue()`.
- `async_sim = particles_on && has_async_compute() && async_compute_on`일 때: 사임을 그래프 패스 대신
  컴퓨트 cmd로 기록(execute **이후** — 그래픽스 그래프와 독립이라 그래프 lifetime borrow와 안 엉킴) →
  `compute_queue.submit(ccmd, compute_done)` → `queue.submit_async(cmd, image_available, compute_done, …)`.
  아니면 Phase 7대로 그래프 컴퓨트 패스 + `queue.submit`.
- ImGui "  - async compute queue" 토글(미지원 GPU는 `text_disabled`로 비활성), 헤드리스 `ASYNC_COMPUTE` 환경변수.
  스크린샷 모드 기본 off(단일 큐 베이스라인), `ASYNC_COMPUTE=1`로 on.

### A4 — 마무리 ✅

- 문서/메모리 갱신, 커밋.

---

## 검증 전략

- `cargo fmt --all`, `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets` clean.
- 두 백엔드 `--backend vulkan|d3d12` + `--screenshot[-clean]` → Read로 시각 확인. **async on/off 픽셀 일치**,
  **두 백엔드 일치**.
- `VK_LOADER_LAYERS_DISABLE="~implicit~"` Vulkan 검증 클린: 크로스-큐 바이너리 세마포어, CONCURRENT storage
  버퍼, 컴퓨트 패밀리 큐/풀에서 VUID 없음.
- 리포지토리 주석/문서에 **타사 엔진/상표명 미사용**.

## 검증 결과 (✅ 완료)

- **async ON 양 백엔드:** `P7_PARTICLES=1 ASYNC_COMPUTE=1` 헤드리스 스크린샷 — 분수가 단일 큐 경로와 동일하게
  정상 렌더(손상/누락 없음) → 크로스-큐 동기(컴퓨트 write → 그래픽스 vertex read)가 정확. D3D12≡Vulkan.
- **Vulkan 검증 클린:** 크로스-큐 바이너리 세마포어(`compute_done`)·CONCURRENT 파티클 버퍼·전용 컴퓨트 패밀리
  큐/풀에서 VUID 없음. D3D12 디버그 클린(디바이스 제거/폴트 없음).
- **단일 큐 폴백:** `P7_PARTICLES=1`(ASYNC_COMPUTE 미설정) → 그래프 컴퓨트 패스 + `queue.submit` 경로,
  분수 정상(두 백엔드). 회귀(particles off) = Phase 7 씬과 동일.
- **WAR 해저드 논증(결정 #1):** 2 프레임 인플라이트 + 2 ping-pong 버퍼에서, 프레임 N이 쓰는 `buf[parity]`를
  마지막으로 읽은 건 프레임 N-2의 draw인데 이는 프레임 시작의 `in_flight[N%2]` 펜스 대기로 이미 완료 → 안전.
- 정확성·구조가 1차 목표였고 확보됨. 픽셀은 wall-clock dt라 프레임마다 달라 정밀 픽셀 일치는 비교 대상 아님
  (분수 형상 일치로 확인). 오버랩 이득의 정량 측정(GPU 타임스탬프)은 미구현 — 한계 참조.

## 위험 / 주의

- **WAR 해저드(결정 #1)** 가 이 작업의 핵심 정확성 리스크 — ping-pong을 A2에서 단독 검증한 뒤 async를 얹는다.
- Vulkan: 전용 컴퓨트 패밀리가 없는 GPU(통합 그래픽 등)에서는 폴백 경로가 정확히 도는지 별도 확인
  (개발기 RTX 2070 SUPER는 전용 컴퓨트 패밀리 있음 — 폴백은 코드 검토로 보강).
- D3D12 펜스-에뮬레이트 vs Vulkan 세마포어 모델 차이로 `submit_async` 시그니처가 비대칭(파사드가 흡수) —
  세마포어가 no-op인 D3D12에서 크로스-큐 펜스 값 단조 증가가 프레임 페이싱과 안 엉키는지 확인.

## 알려진 한계 (예정에 포함)

- 오버랩 이득은 파티클 sim이 가벼워 측정상 작을 수 있음(데모 규모) — 구조/정확성 확보가 1차 목표.
- ping-pong은 파티클 버퍼 메모리 2배.
