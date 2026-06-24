# Phase 9 — 툴링 & 마무리 (세부 계획)

상위: [ROADMAP.md](ROADMAP.md) Phase 9. **async compute는 선행 완료**([async-compute.md](async-compute.md)).
남은 범위: **GPU 프로파일링(타임스탬프) · 디버그 마커/오브젝트 네이밍 · 샘플 브라우저 · 검증 토글**.

워킹 스타일: 마일스톤 단위로 진행, 각 마일스톤은 **build + `cargo fmt` + `clippy -D warnings` + 양 백엔드 실행 +
Vulkan 검증 클린 / D3D12 디버그 클린 + 회귀 클린** 게이트. 리포 주석/문서에 외부 엔진·상표명 금지.

렌더 그래프가 척추이므로, 타이밍·마커는 `RenderGraph::execute`의 **패스 경계**(이미 `PassInfo.name` 보유)에
한 번 끼우면 모든 기법(PBR/컴퓨트/RT/PT)에 자동 적용된다.

---

## M1 — GPU 타임스탬프 프로파일링 (패스별 GPU ms)  ✅ 완료

GPU에서 실제로 각 패스가 얼마나 걸리는지 측정해 ImGui에 표로 표시.

**구현 완료 (양 백엔드 + facade + 렌더그래프 + 샌드박스):**
- RHI: `QueryHeap`(Vulkan `VkQueryPool(TIMESTAMP)` + `timestampPeriod`; D3D12 `ID3D12QueryHeap`
  + READBACK 버퍼 + `GetTimestampFrequency`; Metal 스텁=0). `CommandBuffer::{reset_queries,
  write_timestamp, resolve_queries}` (Vulkan reset/write; D3D12 EndQuery/ResolveQueryData;
  reset는 D3D12 no-op, resolve는 Vulkan no-op). `Device::create_query_heap(count)`,
  `QueryHeap::{count, period_ns, read}`.
- 렌더그래프: `GraphProfiler{heap, names}` → `execute(..., Option<&mut GraphProfiler>)`. 스케줄의
  각 패스 시작 경계 + 마지막 경계에 timestamp → 패스 i GPU시간 = `ticks[i+1]-ticks[i]`. Vulkan은
  전체 풀 reset(읽기 시 전 슬롯 reset 요구), D3D12는 끝에서 resolve.
- 샌드박스: 프레임당 `QueryHeap`(FRAMES_IN_FLIGHT개, fence 후 읽어 스톨 없음), `slot_pass_names`로
  다음 리드백 해석, ImGui "GPU profiler" 토글 + `패스 | ms` 표(+total). env `PROFILE_GPU`로 기본 on.
- **검증:** build+fmt+clippy(-D warnings) clean. 양 백엔드 합리적 수치(VK total ~0.25ms /
  DX ~0.28ms; gbuffer/lighting/tonemap 분해). Vulkan VUID 0(풀 reset 후), D3D12 클린. 프로파일러
  off(기본) 시 VK≡DX 0.0001/ch — 렌더 무변경.

- **RHI (양 백엔드 + 파사드):**
  - `QueryHeap` 추상 (타임스탬프 N개).
    - Vulkan: `VkQueryPool(QUERY_TYPE_TIMESTAMP)`, `vkCmdResetQueryPool`(프레임 시작),
      `vkCmdWriteTimestamp`, `vkGetQueryPoolResults`. `timestampPeriod`(ns/tick)는
      `VkPhysicalDeviceLimits`에서. 큐의 `timestampValidBits` 확인.
    - D3D12: `ID3D12QueryHeap(TIMESTAMP)`, `EndQuery`, `ResolveQueryData` → READBACK 버퍼,
      `Queue::GetTimestampFrequency`(ticks/sec).
  - 파사드: `Device::create_query_heap(count)`, `CommandBuffer::{reset_queries, write_timestamp(i)}`,
    `QueryHeap::read(device) -> Vec<u64>` (또는 ns 변환 helper).
  - **레이턴시 회피:** 쿼리 결과는 `FRAMES_IN_FLIGHT`만큼 더블/트리플 버퍼 → 프레임 N에서 N-2 결과를
    읽어 GPU 스톨 없이. 패리티는 기존 프레임 인덱스 재사용.
- **렌더 그래프(`render/lib.rs` execute):**
  - 스케줄의 각 패스 record 직전/직후 `write_timestamp`. (begin_idx, end_idx, name) 수집 →
    `execute`가 패스별 (name, start_tick, end_tick) 리스트를 반환하거나, 호출측이 조회할 핸들 제공.
  - 컴퓨트 패스도 동일. 전체 프레임 begin/end도 1쌍.
- **샌드박스:** ImGui "GPU Profiler" 패널 — `패스 | ms` 표 + 합계 + (옵션) 막대. 양 백엔드.
- **검증:** 숫자 합리성(합계 ≈ 프레임시간 일부), 양 백엔드, Vulkan VUID 0.

## M2 — 디버그 마커 + 오브젝트 네이밍  ✅ 완료

RenderDoc/PIX/NSight 캡처에서 패스/리소스가 이름으로 보이게.

**구현 완료 (양 백엔드 + facade + 렌더그래프 + 샌드박스):**
- `CommandBuffer::{begin_debug_label, end_debug_label}` — Vulkan `VK_EXT_debug_utils`
  (`vkCmdBegin/EndDebugUtilsLabelEXT`, DeviceShared에 device-level loader 추가, debug+validation
  게이트), D3D12 `BeginEvent/EndEvent`(ANSI metadata=1, `cfg!(debug_assertions)` 게이트), Metal no-op.
- 렌더그래프 `execute`가 스케줄의 각 패스 record를 `pass.name`으로 감쌈(compute/graphics 양 경로
  end 밸런스). 타임스탬프 마커와 같은 경계 공유.
- 오브젝트 네이밍: `RenderTarget/DepthBuffer/Cubemap::set_name` — Vulkan
  `vkSetDebugUtilsObjectNameEXT`(이미지 핸들), D3D12 `ID3D12Object::SetName`, Metal no-op. 샌드박스가
  IBL 리소스(env/irradiance/prefilter 큐브 ×2, BRDF LUT, capture depth) 명명.
- **검증:** build+fmt+clippy(-D warnings) clean. **Vulkan 검증 클린**(레이블 begin/end 밸런스 + 오브젝트
  네임 핸들/타입을 검증 레이어가 확인 → VUID 0), D3D12 디버그 클린. VK≡DX 0.0001/ch(주석은 픽셀 무관).
  ⚠️ 캡처 툴에서 실제 이름 표시는 **사용자 수동 확인**(RenderDoc/PIX 이 환경에서 불가). 미적용: 그래프
  transient(G-buffer/shadow)는 풀 소유라 미명명(후속).

- **RHI (양 백엔드 + 파사드):**
  - `CommandBuffer::{begin_debug_label(name, [rgba]), end_debug_label()}`.
    - Vulkan: `VK_EXT_debug_utils`의 `vkCmdBeginDebugUtilsLabelEXT`/`...End...`. (인스턴스 확장,
      디버그 빌드에서만 로드 — 기존 검증 게이팅과 동일 패턴.)
    - D3D12: `ID3D12GraphicsCommandList::BeginEvent/EndEvent` (PIX 포맷 문자열, 또는 단순 UTF-16).
  - `Device::set_object_name(handle, name)`.
    - Vulkan: `vkSetDebugUtilsObjectNameEXT`(이미지/버퍼/파이프라인 핸들 + objectType).
    - D3D12: `ID3D12Object::SetName`.
  - 릴리스 빌드에서는 no-op로 컴파일 아웃(검증과 동일하게 `cfg!(debug_assertions)` 게이팅).
- **렌더 그래프:** 각 패스 record를 `begin_debug_label(pass.name)`…`end_debug_label()`로 감쌈.
- **부트스트랩:** 주요 리소스(G-buffer 타깃, 섀도우/큐브맵, 파이프라인)에 이름 부여.
- **검증:** Vulkan 검증 잡음 0, D3D12 디버그 클린. RenderDoc/PIX 캡처에서 이름 확인은 **사용자 수동**
  (캡처 툴은 이 환경에서 스크립트 불가) — 마커 코드는 검증 레이어가 라벨 밸런스/objectType를 잡아준다.

## M3 — 샘플 브라우저 + 검증 토글

흩어진 기법 토글을 일관된 "샘플 브라우저"로 정리.

- **샌드박스 UI:** 현재 산재한 체크박스/콤보(PBR·IBL·섀도우·머티리얼 오버라이드, 컴퓨트 3종,
  RT/PT/Cornell, async)를 **씬/기법 셀렉터**(좌측 리스트 + 설명 + 해당 토글만 노출)로 재구성.
  헤드리스 env 토글(`P7_*`/`P8_*`/`NO_POINT_LIGHTS` 등)은 그대로 유지.
- **검증 토글:** 현재 인스턴스 생성 시 `cfg!(debug_assertions)` 게이팅이라 **런타임 토글 불가**(인스턴스
  재생성 필요). 정직하게: 런치 플래그(`--validation on/off`)로 노출 + 디버그 메신저 verbosity만 런타임
  조절. 문서화.
- **검증:** 양 백엔드 실행, UI 사용성, 회귀 클린.

---

## 마일스톤 순서 / 가치

1. **M1 (타임스탬프)** — 실질 프로파일링 데이터. 가장 높은 가치, 독립적.
2. **M2 (마커/네이밍)** — 캡처 가독성. M1과 같은 패스-경계 훅을 공유.
3. **M3 (브라우저/토글)** — UX 정리. 순수 샌드박스, RHI 무변경.

## 완료 기준 (ROADMAP Phase 9)

- 프로파일러(패스별 GPU ms) + 디버그 마커가 양 백엔드에서 동작, 캡처 툴에서 패스/리소스 이름 표시.
- 샌드박스에서 기법을 깔끔히 전환.
- 전 구간 검증 클린, 양 백엔드 일치.
