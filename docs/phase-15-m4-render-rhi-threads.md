# Phase 15 M4 — 렌더그래프 스레드 ↔ RHI 스레드 분리 + 병렬 패스 기록 (세부 계획)

상위: [phase-15-job-system.md](phase-15-job-system.md) M4 · [ROADMAP.md](ROADMAP.md) Phase 15.

## 동기 / 사용자 결정

M1–M3(잡 시스템·고정 타임스텝·ECS 병렬)은 CPU 시뮬레이션 측을 병렬화했다. M4는 **렌더 제출 경로**를
멀티스레드로 가른다. **사용자 결정: 렌더그래프 스레드와 RHI 스레드를 분리한다** (UE식 Render thread ↔
RHI thread 모델):

- **렌더그래프(record) 스레드** — 렌더 그래프를 빌드하고 패스 로직을 **커맨드 버퍼로 기록**한다(CPU 작업,
  드라이버 호출 다수). 잡 시스템 위에서 독립 패스를 **병렬 기록**(M4b)할 수 있다.
- **RHI(submit) 스레드** — 큐 `submit` + 스왑체인 `acquire`/`present` + 펜스/세마포어 관리를 전담한다.
  기록과 제출/드라이버 동기화를 분리해 **CPU 기록과 GPU 제출을 프레임 단위로 오버랩**(파이프라인)한다.

## 현 상태 (왜 큰 작업인가)

- RHI는 **완전 단일 스레드 설계**다. Metal 백엔드 타입(`MetalDevice/Queue/CommandBuffer/Swapchain/Fence`)이
  전부 `RefCell`/`Cell`(내부 가변성, `!Sync`) + `Retained<ProtocolObject<dyn MTL…>>`(objc2 기본 `!Send`/
  `!Sync`). 어떤 백엔드/파사드 타입에도 `unsafe impl Send`가 없다.
- 프레임 루프(`apps/sandbox/src/main.rs::frame`)는 한 스레드에서
  `acquire → cmd.begin → graph.execute(→cmd) → cmd.end → queue.submit(±async) → readback → queue.present`
  를 순차 수행한다. 헤드리스 캡처는 `in_flight.wait()` 후 동기 readback→PNG.
- **검증 제약:** Metal만 이 박스에서 런타임 검증 가능. VK/DX는 Windows(또한 D3D12 COM·ash 핸들의 Send/Sync
  특성도 그쪽에서 확인). 따라서 M4는 [[dreamcoast-verification-split]] 규칙이 강하게 적용된다.

## ⚠️ 착수 중 발견한 아키텍처 블로커 (2026-06-29)

단순 `Send` 부착이 **불가능**함을 코드 조사로 확인했다. `MetalQueue`/`MetalComputeQueue`/`MetalDevice`/
`MetalCommandBuffer`/`MetalSwapchain`/파이프라인이 전부 **`Rc<DeviceShared>`**(비원자 refcount)를 공유하고,
`DeviceShared` 내부는 `RefCell`/`Cell`(단일 스레드 내부 가변성)로 가득하다. Queue를 RHI 스레드로 *이동*하면
record 스레드의 Device가 같은 `Rc`를 들고 있어 **refcount가 레이스 → UB**. 즉 스레드 분리는 백엔드의
공유 모델 자체를 건드려야 한다. 두 가지 방향이 있고 비용/리스크가 크게 다르다:

- **옵션 A — 백엔드 디바이스 상태를 스레드-세이프화.** `Rc<DeviceShared>` → `Arc`, 내부 `RefCell`/`Cell`
  → `Mutex`/atomics(`DeviceShared: Sync`). 그러면 Queue/Swapchain을 RHI 스레드로 이동 가능. **검증된
  Metal 백엔드의 공유 모델을 침습적으로 재작성**(+ VK/DX 동일 작업, Windows에서만 검증). 락 오버헤드/
  교착 리스크. 변경 표면이 넓고 회귀 위험 큼.
- **옵션 B — RHI 커맨드-리스트 IR (UE RHICommandList식, 권장).** record(렌더그래프) 스레드는 **백엔드
  무관 커맨드 리스트**(순수 CPU 데이터 = 자명하게 `Send`)에 기록하고, **단일 RHI 스레드가 모든 백엔드
  객체(Device/Queue/Swapchain)를 소유**해 그 IR을 실제 백엔드 호출 + submit + present로 번역한다.
  스레드 간 백엔드 객체 공유가 없어 **`Rc`/`Send` 수술 불필요**(백엔드 코드 무변경). "RHI 스레드"
  명칭과도 정확히 일치하고 M4b(병렬 기록)도 IR 버킷을 워커가 병렬 생성하는 형태로 자연스럽다. 대신
  **커맨드 IR + 기록 API + 백엔드별 번역 계층**을 신규로 만들어야 함(기존 record 호출부를 IR로 라우팅).

> **결정 (2026-06-29): 옵션 B — RHI 커맨드-리스트 IR.** 검증된 백엔드의 공유 모델을 침습적으로
> 재작성하는 대신, record 스레드가 순수 데이터 IR을 만들고 단일 RHI 스레드가 소유·번역·제출한다.

## 아키텍처 (옵션 B — RHI 커맨드-리스트 IR)

- **`CommandList`(IR, `Send`)** — 백엔드 무관 커맨드의 평면 데이터 열거형 `Vec<RhiCommand>`. 바운드
  객체는 `Send`-안전 핸들로 참조한다(바인들리스 인덱스는 이미 `u32`; 파이프라인/타깃 등은 record 스레드가
  미리 구성한 **테이블 인덱스** 또는 RHI 스레드가 소유한 핸들). 리소스 *생성*은 IR에 안 들어간다(여전히
  record 스레드가 Device로 미리 생성; 핸들만 IR에 실린다).
- **record(렌더그래프) 스레드** — 패스가 `cmd.xxx()` 대신 **`Recorder`에 커맨드를 append**. `graph.execute`는
  `cmd` 대신 `Recorder`로 IR을 산출한다.
- **RHI 스레드** — Device/Queue/Swapchain/커맨드버퍼를 **단독 소유**. `acquire → translate(IR→실제 cmd) →
  submit(±async) → present`. record 스레드는 `CommandList`(+제출 메타)만 채널로 넘긴다 → 백엔드 객체
  스레드 공유 0, `Rc` 무변경.

## 마일스톤 (단계별 — 검증 게이트마다 캡처 바이트 동일)

### B1 — 커맨드 IR + 번역기 (foundation, 가산적·렌더러 무변경) ✅ (commit f0646e7)
- `crates/rhi/src/command_list.rs`: `RhiCommand` 열거형(전 `CommandBuffer` 표면 커버) + `CommandList`
  (레코더 = `CommandBuffer`와 동일 시그니처 메서드) + `translate(&CommandList, &CommandBuffer)`(실제 백엔드
  cmd로 재생). 리소스 참조는 인라인 `ResPtr<T>`(Send 래핑 raw ptr, 핸드오프 계약하에 프레임 동안 유효).
  push-constant/MRT/디버그라벨은 사이드 아레나.
- 검증 ✅: rhi 단위 테스트 3개(기록 + 아레나 + **`CommandList`/`RhiCommand` Send 단언**), 가산적이라
  캡처 baseline 바이트 동일, clippy/fmt 클린.

### B2 — 기록을 IR로 라우팅 (캡처 바이트 동일) — ✅ 완료
**구현:** `Recorder` trait(rhi, `&self`) — `CommandBuffer`(즉시, inherent로 포워드) + `CommandList`
(지연, `RefCell` 내부가변성)가 구현. `RenderGraph::execute`는 프레임 전체를 `CommandList`에 기록 후
마지막에 `list.translate(cmd)?`(같은 스레드, 동작 동일). `PassContext::cmd()` → `&dyn Recorder`라
~42개 `ctx.cmd()` 호출부는 무변경. 시그니처 변경은 `gui::render`(`&dyn Recorder`) 1곳뿐 — IBL capture
/compute(ccmd) 직접 경로 헬퍼는 concrete `&CommandBuffer` 유지(coerce). begin/end는 trait 제외(프레임
루프가 real cmd에 직접 호출).
- **검증 ✅ (Metal):** 기본 캡처 baseline 바이트 동일(`b9778dcc`); `P15_SPIN=8` 모션 시퀀스 4프레임이
  **pre-B2와 프레임별 동일**(정적+동적 모두 IR 경유 무회귀); clippy `-D warnings`/fmt 클린, rhi 3 테스트.

### B3 — RHI 스레드 + 핸드오프 (`P15_RHI_THREAD`, 기본 off) — ✅ 완료 (1프레임 오버랩)

**구현 (B3-a, 사용자 결정 = 오버랩 바로):**
1. **IR에서 backbuffer 프레임-글로벌 분리** (`crates/rhi/src/command_list.rs`): swapchain/image_index를
   per-command에서 빼고 `translate(cmd, swapchain, image_index)` 컨텍스트로 해소. 인히런트
   `backbuffer_to_render_target/_present`, `begin_backbuffer_rendering`, `set_backbuffer_viewport`로
   record 스레드가 swapchain/index 없이 IR을 빌드 → acquire를 RHI 스레드로 이동 가능.
2. **그래프 record/execute 분리** (`crates/render/src/lib.rs`): `record_into(&CommandList, …)`(translate 없이
   IR만) + `record() -> CommandList` + 기존 `execute()` = record_into + translate. 인라인 경로 무변경.
3. **경계 타입 `unsafe impl Send`** (`crates/rhi/src/lib.rs`): Queue/ComputeQueue/Swapchain/CommandBuffer/
   Fence/Semaphore. SAFETY = 단일-소유 핸드오프(부팅 시 1회 move, 프레임 중 borrow만 → refcount 미변경,
   teardown은 join 후 단일 스레드).
4. **`apps/sandbox/src/rhi_thread.rs`**: `RhiThread`가 queue/swapchain/per-fif command buffer/in_flight/
   image_available/render_finished + **영속 per-fif readback 버퍼**를 move-소유. rendezvous 채널(cap 0)로
   record를 ≤1프레임 ahead로 바운드. 워커: `in_flight[fif].wait/reset → acquire → begin → translate →
   (capture copy) → end → submit → (capture wait+read+save) → present`. recreate는 `AtomicBool`로 record
   쪽에 알림(워커 join 후 record가 리사이즈).
5. **`main.rs` 라우팅**: `queue/swapchain`을 `Option`(워커가 소유 시 None), 4개 Vec은 `mem::take`로 이동.
   record 반부는 IBL `maybe_capture`(→`&dyn Recorder`)를 frame IR에 prepend 후 `record_into`로 그래프
   append → `submit(list, fif, capture)`. acquire/submit/present/capture-readback는 워커. async(particle/
   cache) 활성 시 워커 미spawn = 인라인 fallback(문서화).

**소유권/Send 난점 해소 요약:**
- record 스레드가 `&mut self`에서 파생한 raw-ptr로 워커가 객체를 만지면 Stacked-Borrows UB → **객체를
  워커로 move**(파생 아님)로 회피.
- 영속 readback 버퍼는 record 스레드 생성·join 후 record 스레드 drop → **워커에서 백엔드 객체 drop 0**
  (오버랩 중 record 스레드의 per-frame 리소스 create/drop과 `Rc<DeviceShared>` refcount 경합 없음).
- 워커는 acquire/translate/submit/present에서 **borrow만**(clone/drop 없음) → refcount 단일 writer(record).
- FRAMES_IN_FLIGHT=2 + rendezvous로 record N+1이 쓰는 fif가 워커가 translate 중인 fif와 안 겹침.

**검증 ✅ (Metal):** 기본 `--screenshot-clean` 인라인=스레드 **둘 다 byte-identical `b9778dcc`**; 스레드
3회 반복 모두 동일(결정적, 무경합); `P15_SPIN=8 CAPTURE_SEQ=4` 모션 4프레임이 인라인=스레드 **프레임별
동일**(동적 오브젝트 + 오버랩 결정성). 기본 off 무회귀, `cargo clippy --all-targets -D warnings`/`fmt`
클린, rhi 3 + render 테스트.

**검증 ✅ (Windows VK/DX, RTX 2070 SUPER, 2026-06-29):** `tools/verify-rhi-thread.ps1`.
- **Vulkan:** flag-off == flag-on(`P15_RHI_THREAD=1`) **바이트 동일**(SHA `06BDD797…`), `P15_SPIN`
  모션 4프레임도 프레임별 동일. VK 검증 메신저 클린(신규 에러 없음 — `g.storage_buffers` fragment
  `fragmentStoresAndAtomics` 경고는 auto-exposure 유래의 기존 이슈로 flag 무관, 별건).
- **D3D12:** flag-on이 flag-off와 **D3D12 자체의 런-투-런 비결정성(1-LSB, max 1) 범위 내에서 동일** —
  flag-off↔flag-off 두 번도 max 1로 다름(VK는 런-투-런 바이트 동일, 대조군 확인). 즉 **RHI 스레드는
  그 기존 노이즈 이상으로 차이를 만들지 않음**. SHA bit-exact 게이트는 D3D12가 런-투-런 바이트 결정적이
  아니라 통과 못 함(B3 버그 아님, 별건의 D3D12 비결정성).
- **D3D12 디버그 레이어:** 신규 `Device::drain_debug_messages()`(ID3D12InfoQueue→tracing 브리지,
  VK 메신저와 대등)로 캡처 — flag-on + `P15_SPIN` 오버랩에서 **ERROR/CORRUPTION 0건**. 유일한 메시지는
  무해한 기존 WARN(`CreateCommittedResource: Ignoring InitialState … UNORDERED_ACCESS`, 리소스
  *생성* 시점, 스레딩 무관). → 워커 스레드의 COM(`ID3D12CommandQueue`/`IDXGISwapChain`) submit/present +
  경계 6타입 `unsafe impl Send`가 D3D12에서 안전.
- **DX≡VK(flag-on):** rt-compare avg **0.001**/ch(하드룰 ≤0.001, max 5는 위 D3D12 1-LSB + 통상 cross-backend).

**결론: B3는 VK/DX 양쪽에서 정확.** 스레딩 버그·검증 에러 없음. (별건 후속: D3D12 1-LSB 런-투-런 비결정성
조사 — 게이트를 bit-exact로 쓰려면 그 원인 제거 필요. VK는 결정적.)

#### (참고) 원 설계 메모
RHI 스레드가 `CommandList`를 받아 translate + submit + present한다. **핵심 난점 = 소유권/Send.**
`translate`/`submit`/`present`는 `CommandBuffer`/`Queue`/`Swapchain`(전부 `Rc<DeviceShared>` 보유,
`RefCell`로 `!Send`/`!Sync`)을 만진다. 두 방안:
- **B3-a (권장 1차): per-fif `CommandBuffer` + `Queue`/`Swapchain`을 RHI 스레드가 단독 소유.** 부팅 시
  생성/이동 1회, 이후 record 스레드는 **절대 만지지 않음**(IR `CommandList`만 채널로 전달; 제출 메타는
  fif 인덱스 등 평면 값). Rc<DeviceShared> refcount는 clone/drop이 동시 발생하지 않으면 안전하나(프레임
  중엔 borrow만, 종료는 join 후 단일 스레드), 타입상 `!Send`라 **경계 객체에 정당화된 `unsafe impl Send`**
  이 필요(refcount 비동시 변경 근거). `Device`(리소스 생성/pool)는 record 스레드 잔류 — 단, Device와
  Queue가 같은 `Rc<DeviceShared>`를 공유하므로 **둘이 다른 스레드면 그 Rc의 clone/drop이 없도록** 보장해야
  함(핸드오프 계약).
- **B3-b (정석, 더 큼): Device까지 RHI 스레드 소유.** 리소스 생성/pool도 RHI 스레드로 → record 스레드는
  순수 IR 생성기. App 구조 대수술. 후순위.
- 파이프라인 1프레임(record N+1 ∥ submit N). 캡처는 RHI 완료 동기 배리어 후 readback → 결정성. 기본
  off=단일 스레드 fallback=무회귀.
- 완료: `P15_RHI_THREAD=1` Metal 캡처 바이트 동일 + 인터랙티브 동작 + 검증 클린. **Windows VK/DX 패리티
  검증 완료**(VK 바이트 동일, DX는 자체 1-LSB 런-투-런 노이즈 내 동일 + 디버그 레이어 0 에러 — 위 검증절).

### B4 — 병렬 IR 생성 (잡 시스템 위, `P15_PARALLEL_RECORD`) — ✅ 완료

**핵심 통찰:** 패스의 GPU 의존성은 concat 후 **스케줄 순서의 barrier 커맨드**에 인코딩되므로, *기록* 순서는
결과에 무관 → **모든 스케줄 패스를 동시에 기록 가능**(의존 체인이 병렬도를 제한하지 않음). 워커는 패스
클로저를 실행해 자기 버킷에 IR을 append하고, 끝나면 스케줄 순서로 concat.

**구현 (2단계, 각 byte-identical 게이트):**
- **B4 step 1 (foundation):** `CommandList::append(other)` — 버킷의 커맨드를 이 리스트에 이어붙이며
  push/labels/targets 아레나 오프셋을 rebase(= 순차 기록과 동일). + 단위 테스트. `record_pass()` 추출 —
  한 패스의 전체 IR 서브시퀀스(label·read→sampled barrier·attachment transition·begin_rendering·클로저·
  end)를 주어진 버킷에 기록. `&self`/패스간 순차 상태 없음(유일한 backbuffer first-use transition은
  `first_backbuffer` 사전계산 후 플래그로 전달). 프로파일러 timestamp만 호출부 잔류(인라인 전용·순차).
- **B4 step 2 (parallel):** `record_into`에 `jobs: Option<&JobSystem>` 추가. `Some` + 프로파일러 off일 때
  스케줄 패스마다 `PassJob`(disjoint `&mut PassNode` + read-only 그래프 컨텍스트 + 자기 `CommandList`
  버킷)을 만들어 `jobs.parallel_for`로 병렬 기록 → 스케줄 순서로 `append`. `record_parallel()` 공개 API
  추가. 샌드박스: `P15_PARALLEL_RECORD`(스레드 경로 전용) → `record_into(.., Some(global()))`.

**Send/Sync 해소:** `PassJob`에 read-only 컨텍스트(resources/pool/맵 refs)를 **인라인으로 보유**해 워커
클로저가 외부를 **아무것도 캡처 안 함**(Rust 2021 disjoint capture가 래퍼를 뚫고 개별 `!Sync` 필드 ref를
잡는 문제 회피) → 클로저는 자명히 `Send+Sync`, `PassJob`만 `unsafe impl Send`. **SAFETY:** 각 `PassJob`은
워커 1개 전담(disjoint `&mut`·자기 버킷만 기록), 패스 클로저는 **기록 전용**(클로저 내 리소스 생성/업로드
없음 — 감사 완료; per-frame 업로드·생성은 `execute` 전에 record 스레드에서) → 병렬 구간에 `Rc<DeviceShared>`
refcount·공유 상태 변형 0.

**검증 ✅ (Metal):** 기본 / 스레드-순차 / **스레드-병렬 ×3 모두 byte-identical `b9778dcc`**(병렬 결정적·
무경합); `P15_SPIN=8 CAPTURE_SEQ=4` 모션 4프레임이 병렬=순차 프레임별 동일; clippy `-D warnings`/fmt 클린,
rhi 4(append 포함) + render 테스트. 기본 off 무회귀.

**검증 ✅ (Windows VK/DX, RTX 2070 SUPER, 2026-06-29):** `tools/verify-rhi-thread.ps1` 게이트 2b
(`P15_RHI_THREAD=1 P15_PARALLEL_RECORD=1` == flag-off).
- **Vulkan:** 병렬 기록 캡처가 flag-off와 **바이트 동일**(SHA `06BDD797…`, B3 flag-on과도 동일 해시) —
  잡 워커들의 병렬 패스-IR 생성 + 스케줄-순 concat이 결정적으로 동일한 IR을 만든다(data race 없음).
- **D3D12:** 병렬 기록이 flag-off와 **D3D12 자체 1-LSB 런-투-런 노이즈 내에서 동일**(on-vs-off 0.0000 ≤
  off-vs-off 0.0000) — 병렬화가 그 기존 노이즈 이상의 차이를 만들지 않음.
- **검증 레이어:** flag-on+parallel 전 구간에서 **VK validation / D3D12 디버그 레이어(InfoQueue 브리지)
  ERROR·CORRUPTION 0건** (기존 별건 경고 2종 제외 — `g.storage_buffers` fragment NonWritable[auto-exposure
  유래], NV-external 로더 쿼리). → `PassJob`/`Send` 가정·잡 워커의 리소스 접근 안전.
- 다른 게이트(default·P15_SPIN)도 전부 PASS, DX≡VK 0.001/ch.

**결론: B4 병렬 렌더그래프 IR 기록은 VK/DX 양쪽에서 정확** — 결정적 동일 IR, data race·검증 에러 없음.

## 리스크 / 미결
- **IR 커버리지 = 큰 표면** — 현 cmd 메서드 전부를 IR 커맨드로. 단계적(B1 서브셋→B2 전수)으로, 미구현
  커맨드는 translate에서 `unimplemented!` 가드.
- **핸들/소유권** — IR이 `Send`이려면 바운드 객체를 인덱스/핸들로. 파이프라인·타깃 테이블 indirection 설계.
- **파이프라인 vs 결정성** — 캡처 readback은 RHI 완료 동기 배리어.
- **범위/리스크** — 제출+기록 코어를 건드리므로 단일 스레드 경로를 기본 fallback 유지(opt-in). 다중 세션
  규모 — 각 B단계가 캡처 바이트 동일 게이트를 통과한 검증 커밋으로 랜딩.
