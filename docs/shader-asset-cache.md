# 셰이더 바이트코드 쿡 캐시 (Shader Asset Cache)

> **상태:** 🧪 계획 (미구현). 빌드 인프라 / 크로스컷팅 — [ROADMAP](ROADMAP.md) **Phase 12 M4**.
> **연관:** [phase-12-asset-pipeline.md](phase-12-asset-pipeline.md)(쿡된 에셋 트랙), [phase-0-foundations.md](phase-0-foundations.md)(Slang 빌드 통합).
> **무관(별개 트랙):** 런타임 Slang 리플렉션/핫리로드는 *런타임* 문제로 이 문서 범위 밖.

## 1. 문제 (Why)

현재 셰이더 컴파일 경로(`crates/shader/build.rs`)는 다음 특성을 가진다.

1. **빌드 스크립트가 매번 전부 재컴파일.** `build.rs`는 `JOBS`(엔트리포인트 ~45개)를
   순회하며 각각 `slangc`를 서브프로세스로 호출한다. `cargo:rerun-if-changed`가 **하나라도**
   걸리면 `build.rs` 전체가 다시 돌고, **모든 잡**이 무조건 재컴파일된다. 셰이더 하나만 고쳐도
   전부 다시 컴파일된다.
2. **변경 감지가 mtime 단독.** `rerun-if-changed`는 **수정 시각(mtime)** 기준이다. `git checkout`,
   `git stash pop`, `touch`, 포맷터의 줄바꿈 정규화 등 **내용이 그대로**여도 mtime만 바뀌면
   전체 재컴파일이 유발된다.
3. **산출물이 휘발성.** 컴파일 결과는 `OUT_DIR`(= `target/.../build/...`)에만 존재하고
   `include_bytes!`로 실행파일에 임베드된다. `cargo clean`이면 사라지고, 다른 체크아웃/머신과
   공유되지 않는다. **신규 체크아웃은 항상 slangc가 있어야 하고 전 셰이더를 풀 컴파일**한다
   (slangc는 gitignored `tools/slang/`이라 없으면 셰이더 `None` 폴백).
4. **OS별 산출물 영속 저장소 부재.** Windows(SPIR-V+DXIL)·macOS(metallib) 모두 그때그때 다시 굽는다.

목표: **각 OS에 맞게 바이트코드를 "에셋"으로 굽고**(per-OS), **소스 내용 해시 + mtime으로 변경을 감지해
바뀐 셰이더만 자동 재컴파일**한다. 바뀌지 않았으면 slangc를 아예 호출하지 않는다.

## 2. 설계 개요 (What)

소스 `.slang` → **per-OS 바이트코드 에셋** + **콘텐츠 해시 매니페스트**로 쿡한다.

```
crates/shader/compiled/                 # 쿡된 셰이더 에셋 루트 (per-OS)
├── manifest.json                       # key+target → {src_hash, dep_hashes, params_hash, artifact_hash, slangc_ver}
├── windows/
│   ├── gbuffer_vs.spv
│   ├── gbuffer_vs.dxil
│   └── … (spirv + dxil)
└── macos/
    └── … (metallib)
```

빌드 시 각 (잡 × 타깃)에 대해:

1. **키 계산** = `hash(소스 파일 바이트 + 전이 include 파일들 바이트 + 컴파일 파라미터 + slangc 버전)`.
   - 파라미터 = `{target, entry, stage, profile, defines}` (예: metallib의 `RT_METAL_TARGET=1`,
     dxil RT의 `lib_6_5` 등).
   - slangc 버전 = `slangc -v` 출력 — **컴파일러 업그레이드 시 캐시 무효화**.
2. **mtime 빠른 경로** — 산출물 mtime ≥ 모든 입력 mtime이면 해시 비교 없이 히트로 간주(선택적 최적화).
   mtime이 더 최신이면 해시까지 확인(아래). **해시가 최종 권위**.
3. **매니페스트 조회** — 키가 일치하고 산출물 파일이 실재하면 **slangc 호출 생략**, 캐시 산출물 재사용.
4. **미스 시** — `slangc` 호출 → 산출물을 `compiled/<os>/`에 기록 → 매니페스트 갱신.
5. **임베드는 캐시 경로에서** — `include_bytes!`가 `OUT_DIR` 대신 `compiled/<os>/<key>.<ext>`를 가리키도록
   생성. (런타임 로드는 §6 옵션 참고.) 실행파일은 여전히 자기완결적.

핵심 효과: `build.rs`는 **항상** 다시 돌지만, 변경이 없으면 **해시 계산(수 ms)만 하고 slangc는 0회 호출**한다.
바뀐 셰이더 1개만 다시 굽는다(전체 ~45개가 아니라).

## 3. include 의존성 추적

해시가 `#include`/`import` 대상까지 덮어야 정확하다. 두 단계로 간다.

- **1단계(보수적, 먼저):** 공유 include 집합(`bindless.slang`, `rt_common.slang`,
  `rt_pipeline_metal_rootsig.json`)을 **모든 잡의 해시에 포함**. 현재 `rerun-if-changed`가 이미 이들을
  감시하므로 동작 동등 + 정확. 단점: 공유 include 1개만 바뀌어도 이를 쓰는 잡 전부 재컴파일(올바른 보수).
- **2단계(정밀, 후속):** slangc depfile(`-depfile` / `-M` 류)로 잡별 실제 의존 파일 목록을 받아 그 집합만
  해시. 공유 include를 안 쓰는 잡은 불필요한 재컴파일 회피. 1단계로 충분하면 생략 가능.

## 4. 해싱 구현 (의존성 정책)

`crates/shader/Cargo.toml`은 현재 **build-deps 0개**다([최소 설치 원칙](#)). 두 선택지:

| 옵션 | 장점 | 단점 |
|---|---|---|
| **A. 무의존 인라인 해시** (FNV-1a 64 또는 소형 SHA-256을 `build.rs`에 직접) | 의존성 0 유지, 빌드 가벼움 | 직접 작성/검증 |
| B. `blake3`/`sha2` build-dependency | 검증된 강한 해시, 빠름 | build-dep 1개 추가 |

**권장 = A** (충돌 위험이 무의미한 캐시 키 용도라 64-bit FNV-1a로도 충분; 보수적으로 128-bit 합성 가능).
최소 설치 선호와 일치. 추후 필요하면 B로 승격.

## 5. 캐시를 커밋할 것인가 (열린 결정 — 사용자 승인 필요)

가장 큰 설계 분기. **"어셋화"의 강한 버전은 쿡된 바이트코드를 레포에 커밋**하는 것이다.

| | **커밋(권장)** | **gitignore(로컬 캐시만)** |
|---|---|---|
| 신규 체크아웃 | **slangc 없이 즉시 빌드·실행** (셰이더 변경 시에만 slangc 필요) | 항상 풀 컴파일 + slangc 필수 |
| 결정성/공유 | CI·타 머신이 동일 바이트코드 재사용 | 머신별 재생성 |
| 레포 비용 | 바이너리 blob 증가(SPIR-V/DXIL/metallib) | 0 |
| 디프 노이즈 | 셰이더 변경 시 blob 디프 | 없음 |

현재 레포 관행: slangc/dxc/validation layer는 전부 **gitignored** `tools/`. 셰이더 *소스*는 커밋,
*산출물*은 미커밋(`.gitignore`에 `/shaders/compiled/` 이미 예약됨).

- (참고) 커밋 버전은 "미리 구운 에셋 배포"의 강한 형태로 fresh checkout이 slangc 없이 동작하지만,
  바이너리 blob이 레포에 들어가고 디프가 지저분해진다.

> **✅ 결정(2026-06-26, 사용자):** **로컬 전용(gitignore)** — 캐시 디렉터리와 매니페스트는 커밋하지
> 않는다. `.gitignore`의 `/shaders/compiled/` 예약 항목을 캐시 실제 경로에 맞춰 사용/조정.
> 따라서 fresh checkout은 여전히 slangc로 1회 풀 컴파일하지만(현행과 동일), 이후 무변경 빌드에서는
> 해시 캐시로 재컴파일이 생략된다. 커밋 버전은 추후 필요 시 재논의.

## 6. 임베드 vs 런타임 로드 (열린 결정)

- **6a. 캐시-백 임베드(권장, 1차):** 지금처럼 `include_bytes!`로 임베드하되 **소스를 캐시 경로로** 바꾼다.
  변경 최소, 실행파일 자기완결 유지. "매번 컴파일" 문제를 곧장 해결.
- **6b. 런타임 에셋 로드(후속, 선택):** 바이트코드를 실행파일 옆 `assets/shaders/<os>/`에서 런타임 로드.
  진짜 "에셋"이지만 런타임 셰이더 로더 + 배포 디렉터리 + 누락 처리 필요. 핫리로드
  ([shader-system-todo])와 자연스럽게 합류. **1차 범위 밖**, 가치 생기면 별도 마일스톤.

사용자 표현("어셋화해서 사용")은 6a로 충분히 충족된다(빌드타임 캐시 = 더 이상 매번 컴파일 안 함).
6b는 런타임 교체가 필요해질 때.

## 7. 마일스톤 (Phase 12 M4)

각 단계 게이트 = `cargo fmt --all` + `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets`
+ **양 백엔드(VK/DX) 픽셀 동일** + Vulkan 검증 클린(셰이더 바이트코드가 동일하므로 렌더 무회귀가 기준).

- **M4.1 — 콘텐츠 해시 + 매니페스트.** `build.rs`에 무의존 해시 + `manifest.json` 읽기/쓰기.
  잡별 키 계산(소스+공유 include+파라미터+slangc 버전). 아직 캐시 디렉터리 없이 OUT_DIR 유지하되
  **해시 히트면 slangc 생략** 로직만 먼저. (이미 측정 가능한 빌드 단축.)
- **M4.2 — per-OS 캐시 디렉터리.** 산출물을 `crates/shader/compiled/<os>/`로 이동, `include_bytes!`가
  거기서 임베드(6a). `target_selected`와 일치하는 OS 네임스페이싱. cache miss/hit 로그.
- **M4.3 — gitignore 반영(로컬 캐시).** 캐시 디렉터리/매니페스트를 `.gitignore`에 반영(§5 결정 =
  로컬 전용). fresh checkout은 1회 풀 컴파일, 이후 무변경 빌드는 캐시 히트로 재컴파일 생략을 검증.
- **M4.4(선택) — 정밀 의존성.** slangc depfile로 잡별 include 집합 좁히기(§3 2단계).
- **M4.5(선택) — 런타임 에셋 로드.** 6b. 핫리로드 합류 지점.

## 8. 검증

- **무변경 빌드:** 깨끗한 트리에서 `cargo build -p sandbox` 두 번 → 2회차는 **slangc 호출 0회**
  (빌드 로그 `cargo:warning` 또는 자체 로그로 확인), 산출물 바이트 동일.
- **단일 셰이더 변경:** `gbuffer.slang` 1줄 수정 → **gbuffer 잡만** 재컴파일, 나머지 캐시 히트.
- **공유 include 변경:** `bindless.slang` 수정 → 이를 포함하는 잡들만 재컴파일.
- **mtime-only 변경:** 내용 동일하게 `touch` → 해시 히트로 재컴파일 0회(현행 대비 개선 핵심).
- **slangc 버전 변경:** slangc 교체 → 전 캐시 무효화·재컴파일.
- **무회귀:** 위 모든 경우 최종 바이트코드/렌더가 기존과 동일(양 백엔드 픽셀 일치, 검증 클린).

## 9. 비범위 (Out of scope)

- 런타임 Slang 인프로세스 컴파일·리플렉션·핫리로드([shader-system-todo]) — 별개 *런타임* 트랙.
- `.dcasset` 바이너리 컨테이너(메시/SDF) — Phase 12 M1–M3. 셰이더 캐시는 **독립 산출물**
  (별도 디렉터리 + 매니페스트)로, P11/P12 본선 의존 없이 단독 진행 가능.
- 셰이더 변형(permutation)/specialization 관리 — 현재 잡 모델은 고정 엔트리포인트 집합이라 불필요.
