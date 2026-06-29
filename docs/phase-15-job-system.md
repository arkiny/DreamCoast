# Phase 15 — 잡 시스템 / 멀티스레드 코어 + 병렬 렌더 (세부 계획)

상위: [ROADMAP.md](ROADMAP.md) Phase 15 · 전략: [commercial-engine-gap-analysis.md](commercial-engine-gap-analysis.md) §토대(T0).

## 동기 / 배경

DreamCoast의 렌더링/GPU 코어(Phase 0–14 + Metal)는 상용 R&D급이지만, **CPU는 전부 단일 스레드**다.
프레임 루프(`apps/sandbox/src/main.rs:1872` `fn frame`)는 입력 → 카메라 → `propagate_transforms` →
드로우 리스트 빌드 → 렌더 그래프 기록을 한 스레드에서 순차 수행하고, ECS(`crates/scene/src/ecs.rs:10`)는
**의도적으로 single-thread / `!Send`** 이며 시스템 스케줄러가 없다. 시뮬레이션은 **가변 dt**
(`main.rs:1996` `sim_dt = dt.clamp(0.0, 1.0/30.0)`)로 돌아 고정 타임스텝 누적기가 없다.

렌더그래프가 **GPU 패스의 척추**이듯, **work-stealing 잡 시스템이 CPU 병렬의 척추**다(설계 원칙 7).
이후의 거의 모든 런타임 시스템 — ECS 시스템 병렬화(P19), 병렬 RHI 커맨드 기록, 비동기 에셋 스트리밍
(P12 확장), 피직스 스텝(P16) — 이 그 위에 얹힌다. 따라서 **다른 그래픽스 breadth 작업(P20–24)보다
먼저** 박아두는 **T0 토대**다.

> **엔진 위상:** 이 잡 시스템은 데모가 아니라 **재사용 가능한 프로덕션 인프라**로 설계한다
> (CLAUDE.md Engineering Rules). 멀티스레드 도입은 **고정 타임스텝 + 결정적 스케줄**을 전제로 해
> 헤드리스 골든이미지 회귀와 양 백엔드(DX≡VK ≤0.001) 검증 정책을 깨지 않는다(설계 원칙 8).

### 사용자 결정 (이번 세션)
- **work-stealing deque = from-scratch.** Chase-Lev deque를 직접 구현한다. RHI·ECS를 from-scratch로
  만든 철학과 일관, **외부 의존 0**. 동시성 코드라 검증 부담이 큰 만큼 단위 테스트 + (가능하면) `loom`
  모델 체크로 보강한다.
- **범위 = 토대 우선, 병렬 렌더는 후속.** 이번 Phase는 **M1 잡 시스템 → M2 고정 타임스텝 sim 루프 →
  M3 ECS 시스템 병렬 스케줄**까지. **병렬 RHI 커맨드 기록(M4)은 별도 후속 마일스톤으로 격리**해 양
  백엔드 병렬 기록 검증 부담과 회귀 위험을 분리한다.

## 아키텍처

### 신규 크레이트 `crates/jobs` (`dreamcoast-jobs`)
RHI 비의존 — `dreamcoast-core`(+ `std`)에만 의존. 순수 CPU 병렬 *메커니즘*을 소유하고, 그래픽스/ECS는
이 facade 위에 *정책*을 얹는다(`rhi`/`crates/render`가 백엔드/로직을 분리하는 구조 미러링). 모듈:

- `deque.rs` — **Chase-Lev work-stealing deque** (lock-free). 소유 워커만 호출하는 `push`/`pop`(LIFO,
  bottom) + 타 워커의 `steal`(FIFO, top). `AtomicUsize` top/bottom + 원자적 버퍼 교체로 성장. **이
  크레이트의 유일한 unsafe 집약점** — 별도 테스트 + 문서화 + loom 게이트.
- `worker.rs` — 워커 스레드 풀(기본 `available_parallelism()-1`, 메인이 1). 각 워커는 자기 deque에서
  `pop`, 비면 라운드로빈 `steal`, 전부 비면 백오프/파킹. 종료 신호로 graceful join.
- `scheduler.rs` — `JobSystem`(전역 또는 명시 핸들). `spawn(closure) -> JobHandle`, 메인이 결과를
  기다리는 `wait(handle)`(워커는 대기 중에도 다른 잡을 처리 = 협력적 블로킹). **결정성 시드**(워커 수·
  steal 순서 고정 옵션)로 헤드리스 재현성 확보.
- `scope.rs` — `scope(|s| { s.spawn(..) })` 구조적 병렬(스코프 종료 시 자식 잡 join 보장, 빌림 안전).
  병렬 for(`parallel_for(slice, grain, fn)`)는 청크 분할 → scope spawn 위에 구현.
- `graph.rs` — **태스크 그래프**: 노드(잡) + 의존 엣지. 위상 정렬 후 의존 충족된 노드부터 스케줄
  (렌더그래프 DAG의 CPU 버전). per-frame 시스템 스케줄(M3)이 소비.
- `lib.rs` — 재노출 + `JobSystem` 라이프사이클(`init`/`shutdown`).

> **`!Send` 흡수:** ECS 코어(`crates/scene`)는 현재 `!Send`다. M3에서 **`World`를 통째로 스레드 간
> 이동하지 않고**, 시스템이 읽고 쓰는 컴포넌트 스토리지에 대한 **분리된 가변 슬라이스(disjoint borrow)**
> 만 워커에 넘긴다 → `World`의 `!Send` 불변을 유지한 채 컴포넌트 단위 병렬. (아키텍처 상세는 M3 참조.)

## 마일스톤

> **상태 (2026-06-29): M1–M3 + M-verify(움직이는 오브젝트) ✅ 구현·검증 (macOS/Metal 박스). Windows
> VK/DX 패리티는 대기.** M4(병렬 RHI 기록)는 후속 격리로 미착수. 검증: `cargo test`(jobs 12 + scene 15
> 통과), 헤드리스 캡처가 pre-M2 baseline과 **바이트 동일**(shasum 일치, git stash로 대조), 움직이는
> 오브젝트가 결정적으로 그려짐(M-verify), 워크스페이스 clippy `-D warnings`/fmt 클린. loom 모델 체크 +
> Miri(WorldCell unsafe)는 외부/nightly 도구라 후속 권장(현재는 deque 동시성 스트레스 + 병렬≡순차
> bit-identical 테스트로 커버).

### M1 — 잡 시스템 코어 (deque + 풀 + scope + 태스크 그래프) ✅
- `crates/jobs` 신설, 위 모듈. Chase-Lev deque → 워커 풀 → `JobSystem` → `scope`/`parallel_for` →
  `graph`.
- 데모/검증 하니스: 병렬 `parallel_for`로 큰 슬라이스 맵·리듀스, 태스크 그래프로 의존 잡 DAG 실행 →
  단일 스레드 결과와 **결정적 일치**.
- **완료 기준**: `cargo test -p dreamcoast-jobs` 통과(deque 단위 테스트 + scope join + graph 위상),
  N코어 스루풋 스케일 확인, clippy/fmt 클린. 샌드박스 렌더 경로는 **무변경 = 무회귀**.

### M2 — 고정 타임스텝 시뮬레이션 루프 ✅
- `main.rs` 프레임 루프를 **고정 dt 누적기**로 전환: `accumulator += frame_dt;
  while accumulator >= FIXED_DT { sim_step(FIXED_DT); accumulator -= FIXED_DT; }` + 렌더용 **보간
  알파**(`accumulator / FIXED_DT`). 가변 `sim_dt`(`main.rs:1996`)를 대체하되 카메라/스피너 등 sim 로직을
  고정 스텝으로 이동.
- **헤드리스 무회귀가 절대 조건**: 스크린샷/캡처 경로(`--screenshot-clean`, `P8_PATHTRACE`,
  `CAPTURE_SEQ`)는 이미 결정적 고정 포즈다 — 고정 타임스텝이 **이 픽셀을 바꾸지 않도록** 캡처 모드는
  결정적 단일 스텝 경로로 우회(현 동작 보존).
- **완료 기준**: 인터랙티브는 프레임레이트 독립적 sim, 헤드리스 캡처는 **바이트 동일**(갤러리 회귀 0),
  DX≡VK ≤0.001 유지.

### M3 — ECS 시스템 병렬 스케줄 ✅
- `crates/scene`에 **시스템 디스크립터**(읽기/쓰기 컴포넌트 집합 선언) + 스케줄러: 컴포넌트 접근이
  겹치지 않는 시스템을 M1 태스크 그래프로 **병렬 디스패치**, 충돌(쓰기-쓰기/읽기-쓰기)은 자동 직렬.
  `propagate_transforms`(`main.rs:1851/966`) 같은 기존 패스를 시스템으로 등록.
- **결정성**: 같은 프레임에서 병렬 실행해도 산출(드로우 리스트 순서·트랜스폼)이 **단일 스레드와 동일**
  하도록 — 병렬은 *충돌 없는* 시스템에만, 산출 버퍼는 엔티티 인덱스 안정 순서 유지(현 sparse-set
  insertion order 불변).
- **완료 기준**: ECS 시스템이 N코어로 스케일하면서 드로우 리스트가 **단일 스레드와 비트 동일**, 양
  백엔드 픽셀 일치, 검증 클린, 골든이미지 무회귀.

### M-verify — 움직이는 오브젝트 그리기 (파이프라인 마무리) ✅
M1–M3가 **실제 동적 콘텐츠**로 맞물려 도는지 끝까지 증명한다(고정 타임스텝 sim → ECS 쓰기 → 병렬
propagate → 드로우).
- `crates/scene`: `Spin{axis,speed}` 컴포넌트 + `advance_spin(world, dt)` 시스템(Spin 읽기 → LocalTransform
  쓰기, 결정적). 샌드박스 프레임 루프: 고정 스텝마다 `advance_spin(FIXED_DT)`(스크린샷 CAPTURE_SEQ는
  프레임당 결정적 1스텝) → **매 프레임 `propagate_transforms_parallel`** → `build_scene` 드로우. 정적
  씬에선 매 프레임 propagate가 동일 행렬을 재계산하므로 **갤러리 바이트 동일 유지**.
- 데모/검증: `P15_SPIN[=<rad/s>]`로 모델·큐브(비대칭이라 회전이 눈에 보임)에 Spin 부착(기본 off=무회귀).
- **검증 결과 (macOS/Metal)**: ① 기본(off) 캡처가 pre-change baseline과 **바이트 동일**(shasum 일치) —
  매 프레임 propagate가 정적 씬 무영향. ② `P15_SPIN=8 CAPTURE_SEQ=4 CAPTURE_SEQ_STEP=0`(카메라 고정,
  오브젝트만 모션): 4프레임이 **서로 다른 shasum**(frame0 vs frame3 변경 픽셀 2.66%·국소 maxΔ203 =
  배경 정적·모델/큐브만 이동 → 실제 회전 렌더) ③ **런 A ≡ 런 B 프레임별 동일**(결정성). 즉 움직이는
  오브젝트가 결정적으로 그려짐 → 파이프라인 검증 종료.

### M4 — 렌더그래프 ↔ RHI 스레드 분리 + 병렬 기록 — 🚧 설계 확정·구현 진행 중
세부: [phase-15-m4-render-rhi-threads.md](phase-15-m4-render-rhi-threads.md).
- **사용자 결정: 렌더그래프 스레드와 RHI 스레드를 분리.** 착수 중 `Rc<DeviceShared>`가 백엔드 전반 공유라
  단순 `Send` 부착 불가(비원자 refcount 레이스)임을 확인 → **옵션 B(RHI 커맨드-리스트 IR)** 채택: record
  스레드가 순수 데이터 IR 생성, 단일 RHI 스레드가 백엔드 객체 단독 소유·번역·제출.
- 단계: **B1** 커맨드 IR+번역기(가산적) → **B2** 기록을 IR로 라우팅(캡처 바이트 동일) → **B3** RHI 스레드
  +핸드오프(`P15_RHI_THREAD`, 기본 off) → **B4** 병렬 IR 생성(잡 워커). 다중 세션 규모, 각 단계 캡처
  바이트 동일 게이트 검증 커밋.

## 구현 노트 (M3 소운드니스 모델 — 확정)
`World`는 `!Send`/`!Sync`라 병렬 영역에서 `&World`/`&mut World`를 절대 만들지 않는다. 대신 `SystemSchedule::run`
이 **단일 스레드에서 `&mut World`를 쥔 채** 각 시스템의 컴포넌트별 **타입소거 스토리지 포인터 테이블**을
미리 해석한다(`World::storage_ptr::<T>` → `(TypeId, *mut ())`). 쓰기 포인터는 `&mut World`에서 파생되므로
정당하게 가변(공유→가변 캐스트 아님; `invalid_reference_casting` 회피). 배치 불변식(쓰기 disjoint·읽기-쓰기
무중첩) 덕에 동시 실행 시스템의 포인터 테이블은 서로소 스토리지를 가리킨다. 각 시스템은 `WorldCell`(Send,
자기 포인터 테이블만 보유)로 `collect_read`/`get_copy`/`insert`만 수행 — 구조적 변경(spawn/despawn)은
병렬 영역에서 미제공. Box-backed 스토리지 주소는 HashMap 성장에도 안정이라 선해석 포인터가 끝까지 유효.
병렬화 자체(N코어 스케일)는 시스템-간(배치) + 시스템-내(`propagate_transforms_parallel`의 `parallel_for`)
양쪽으로 제공.

## 결정성 / 재현성 전략 (설계 원칙 8)
- **워커 수·steal 순서 시드 고정 옵션** — 헤드리스/CI는 결정적 모드로 실행(또는 캡처 경로는 sim을 메인
  스레드 단일 스텝으로 우회).
- **병렬은 산출이 순서 독립인 곳에만** — 병렬 for/시스템의 결과는 결합법칙/안정 정렬로 단일 스레드와
  비트 동일. 부동소수 누적 순서가 결과를 바꾸는 리듀스는 결정적 트리 리듀스 사용.
- **고정 타임스텝**이 sim 재현성의 기반(M2). 캡처 픽셀 회귀 0이 합격선.

## 검증 전략
- `cargo test -p dreamcoast-jobs`: deque(push/pop/steal 동시성), scope join, parallel_for 정확성,
  graph 위상 — 가능하면 deque에 `loom` 모델 체크 게이트(dev-only).
- 골든이미지/캡처: `--screenshot-clean` + `P8_PATHTRACE` 패리티(`tools/rt-compare.py`)로 M2/M3
  전후 **잔차 무변화** 확인, DX≡VK ≤0.001.
- clippy `-D warnings` + fmt 클린(CI 게이트). `tracing` 스팬으로 워커 점유 가시화(후속 CPU 프로파일러
  토대).

## 리스크 / 미결
- **lock-free deque 정확성** — 가장 까다로운 unsafe. `deque.rs`에 격리 + 집중 테스트 + loom. 초기엔
  보수적(Mutex 보호 deque) 폴백을 두고 Chase-Lev로 교체하며 동일성 검증하는 안도 가능.
- **ECS `!Send` 누수(M3)** — `World` 전체 이동 금지, disjoint 컴포넌트 borrow만 워커로. 안전 추상화가
  새면 컴파일 단계에서 잡히도록 타입 설계.
- **결정성 vs 병렬** — 부동소수 누적/실행 순서가 골든이미지를 깨지 않게 §결정성 규칙 준수.
- **범위 폭주** — 병렬 RHI 기록(M4)은 단호히 후속으로 격리(사용자 결정).

## ROADMAP 반영
- `Cargo.toml` workspace에 `crates/jobs` 추가, `dreamcoast-jobs` 경로 의존 등록.
- ROADMAP Phase 15 스텁에 본 문서 링크 + M1–M3 진행 마킹(M4는 후속 명시).
