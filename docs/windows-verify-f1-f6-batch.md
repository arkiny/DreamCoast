# Windows(D3D12·Vulkan) 재검증 — F1 가상화 + F6 GI 웨이브 (2026-07-25)

메인 `9b0dacd..639f4ff`(98커밋, Mac/Metal 작성)의 **DX≡VK Windows 재검증 배치**
([phase-f1-surface-cache-virtualization-plan.md](phase-f1-surface-cache-virtualization-plan.md)
후속 1순위 + [windows-verify-anisotropy-default.md](windows-verify-anisotropy-default.md)).
RTX 2070 SUPER, release, `tools/rt-compare.py`(avg/ch · >8 · >32).

## TL;DR

Windows 전용 버그 **3건 발견·수정** (전부 Mac에서 원리적으로 검출 불가). 수정 후
**갤러리 DX≡VK 0.004/ch = 직전 검증 지점(9b0dacd)과 동일**(그 0.004는 선존 잔차, §5).
콘텐츠 씬의 겉보기 대발산(≈18/ch)은 회귀가 아니라 **측정 함정 2가지**로 분해됨(§4).

## 1. 수정한 버그

| # | 파일 | 원인 | 증상(수정 전) |
|---|---|---|---|
| 1 | `crates/rhi-d3d12/src/lib.rs` | F2-S2b(`52423b2`)가 `Format::R16Float` 매핑 추가하며 `DXGI_FORMAT_R16_FLOAT` import 누락 — Mac은 rhi-d3d12를 컴파일하지 않음 | **rhi-d3d12 컴파일 실패** (Windows 릴리즈 빌드 전체 불능) |
| 2 | `crates/shader/shaders/vgeo_scene_hwvis.slang` | `3f20aa5`의 센티넬 조기-리턴이 `SetMeshOutputCounts`를 2회 호출 — DXIL 검증기는 단일 정적 호출만 허용(SPIR-V/Metal은 분기 호출 허용) | **D3D12 vgeo HW 메시 경로 전체 사멸**(mesh-DXIL 컴파일 거부 → SW 폴백) |
| 3 | `apps/sandbox/src/gdf.rs` | F1 Stage 3(`d76f861`)가 `cache_capture_push`를 96→112B로 늘리며 파이프라인 선언은 96 유지. D3D12 루트 시그니처는 선언/4 DWORD만 예약 → 초과 push는 미정의(꼬리 DWORD 소실) → `slot_dirty` 인덱스가 셰이더에 도달 못 함. VK/Metal은 관대해 Mac에서 비가시 | **DX 서피스 캐시 전면 오염**: `P_SC_VIZ` 4.542/ch(흑백 체커 카드), 최종 갤러리 0.030/ch (기준 ≤0.001의 30×) |
| 3b | 〃 | 동일 클래스: F1 Stage 2가 `cache_vis_push`를 224→232B로 늘림, vis/vis_calib 파이프라인 선언 224 유지 (VK `VUID-…-10069` 2건으로 검출) | 갤러리(pool off)에선 무해하나 **F1 pool/stream 활성 시 DX에서 LRU/frame 필드 소실** |
| 4 | `apps/sandbox/src/gi.rs` | F4B(`fbfd4e6`) 파인-박스 링 버퍼를 `create_storage_buffer_init`(VK=DEVICE_LOCAL)으로 생성 — 주석은 "host-visible" 의도 명시. `flip_fine_buf`가 카메라 재중심 시 host-write. DX(upload heap)·Metal(`contents`)은 우연히 허용, **고정 카메라 캡처는 재중심이 없어 전 게이트 통과** | **VK에서 카메라가 움직이면 프레임 즉사**(`host-write of a DEVICE_LOCAL …`) — 인터랙티브 VK 콘텐츠 세션 전멸급. 2026-07-02 clustered-lights와 동일 클래스. `P11_CACHE_STREAM` 동적 테스트로 발견(스트림 자체는 무죄) |

**재발 방지 (신규):** `crates/rhi-d3d12`의 모든 push 지점(compute/graphics/mesh)에
「push 바이트 > 파이프라인 선언 크기」 **debug_assert** 추가(파이프라인이 `push_size` 보유).
VK는 dev 빌드의 validation layer가 같은 클래스를 생성 시점에 잡음(3b가 그 경로로 검출됨).
근본 원인 #3의 발견 경로: env-토글 격리(`P11_LEGACY_IBL`→0.000, `P_SC_VIZ`로 40× 증폭) →
worktree `git bisect run`(98커밋, 핫픽스 캐리 스크립트) → first-bad `d76f861`.

## 2. 갤러리 수치 (수정 후, 1280×720)

| 구성 | DX≡VK avg/ch | 비고 |
|---|---:|---|
| 기본(aniso16) | **0.006** (max 7, >8 0.00%) | |
| `P_ANISO=1` | **0.004** (max 5, >8 0.00%) | **9b0dacd 베이스라인도 정확히 0.004(max 6)** — 회귀 완전 복구 |
| `P_SC_VIZ=1` | 0.107 (>8 0.00%) | 수정 전 4.542; 베이스라인 0.056 |
| run-to-run | DX 0.000(max1) / VK 0.000(max2) | 결정론 |
| `P11_CACHE_POOL=1` vs off | DX 0.000(max4) / VK 0.000(max1) | F1 Stage1 투명성 계약, Windows 성립 |
| pool ON DX≡VK | 0.004 | pool off와 동일 |

갤러리 aniso16의 DX≡VK 기여는 +0.002/ch 수준(0.006 vs 0.004) — 콘텐츠 예상치(~0.43)보다 훨씬 작음.

## 3. 검증 함정 2가지 (도구/절차 교훈)

1. **rtk가 실패한 `cargo build`를 exit 0으로 보고** — 버그 #1이 있는데도 "성공"으로 보여
   낡은 exe로 캡처를 진행할 뻔함. 빌드는 `rtk proxy cargo build`로 원본 출력 확인 필수.
2. **기본 헤드리스 워밍업은 3프레임**(`SCREENSHOT_WARMUP`). 콘텐츠 씬의 서피스 캐시·GI는
   수백 프레임 상각 수렴하고 **백엔드별 상각 스케줄이 달라**, 짧은 워밍업 비교는 수렴
   과도상태의 차이를 측정하게 됨(아래 §4). 콘텐츠 DX≡VK 비교는 `WARMUP_FRAMES=256+` 필수.
   (골든 레시피들이 64/192를 고정하는 이유.)

## 4. 콘텐츠 씬 (sponza_intel_chromeball, EV100=11, AUTO_EXPOSURE=0)

워밍업 수렴 궤적 (DX≡VK avg/ch, aniso16 기본):

| WARMUP_FRAMES | avg/ch | >8 |
|---:|---:|---:|
| 3(기본) | 18.11 | 48.1% |
| 256 | 7.35 | 18.6% |
| 1024 | 2.73 | 6.3% |

- **9b0dacd 베이스라인도 w3에서 16.87/ch** — 98커밋 회귀 아님(선존 측정 함정).
- 크롬볼 "DX 검정" 증상도 w3 과도상태였고 w256부터 양 백엔드 모두 정상 미러.
- 격리: `P11_LEGACY_IBL=1` → 2.45(aniso 스펙클 바닥) / `P_SC_VIZ` 2.59(직물 aniso) /
  G-buffer V1/V2/V7 ≈1.0 → `P_ANISO=1`시 V2 0.947→0.160. reflect=sw·f16·GI_VOL_OCC·
  BENT_NORMAL·P_GI_VOLUME 토글은 모두 비관여.
- 수렴 후 잔차의 구성: **aniso16 텍스처 샘플링(드라이버 의존, 직물 전면 스펙클) +
  확률적 미러볼 GGX 축적 잔차**(기존 문서화된 항목).
- `P_ANISO=1` @ w1024: **1.083/ch (>8 1.35%)** → **aniso16 기여 ≈ +1.65/ch** (이 직물-위주
  씬 기준; 구 측정 0.427은 다른 씬/코드 상태). iso 잔차 1.08은 확률적 미러볼 + 잔여 수렴 +
  선존 콘텐츠 발산(~0.03급)의 합. 바닥 타일은 양 백엔드 모두 정상 해상(스트라이프 없음).

## 5. 남은 문제 (선존, 이번 배치 범위 밖)

- **갤러리 DX≡VK 0.004/ch(max~6)**: 9b0dacd에서도 동일 — 어느 시점엔가 ≤0.001에서 0.004로
  이동(2026-07-02 "full bundle 0.001" 이후). SW-RT 반사 체인 잔차로 추정. 별도 추적 필요.
- 콘텐츠 수렴-과도 자체의 백엔드 간 속도 차(상각 스케줄) — 정지 화면 수렴값은 일치 방향이나,
  인터랙티브 첫 수 초의 모습이 백엔드마다 다름. 개선하려면 상각 스케줄의 백엔드 불변화 필요.

## 6. F1 검증 결과 (P11_CACHE_POOL / P11_CACHE_STREAM / Stage 0·4)

| 체크 | 결과 |
|---|---|
| 갤러리 pool ON vs OFF (백엔드별) | DX 0.000(max4) / VK 0.000(max1) — Stage 1 투명성 계약 Windows 성립 |
| 갤러리 pool ON DX≡VK | 0.004 (= pool OFF와 동일) |
| 콘텐츠 pool ON @w256 DX≡VK | 7.344 (= pool OFF 7.354와 동일 — pool 비관여) |
| 스트림 동적 카메라 (CAM_EYE→END, CAPTURE_SEQ=16) | **버그 #4 수정 후** 양 백엔드 크래시 없음·슬롯 재소유 정상 |
| 스트림 run-to-run (마지막 프레임) | **VK 0.003(max4)** / DX 0.077–0.099 (기지의 DX 1-LSB GI 비결정론 클래스, tolerant 게이트 내) |
| 스트림 중 DX≡VK | 8.95 @w64+이동 — 수렴-과도 클래스(§3·§4), 스트림 고유 발산 아님 |
| Stage 0(`P11_GATHER_FALLBACK`)·Stage 4(`P11_GI_MIP`) | 콘텐츠 기본 ON으로 전 캡처에 관여 — 백엔드 고유 파손 없음(수렴 후 iso 잔차 1.08에 포함). 개별 ON/OFF A/B는 미실시(후속) |

**주의(수정 전 상태 기록):** capture/vis push-크기 버그(#3/#3b) 미수정 상태에서는 pool/stream이
DX에서 dirty/LRU 필드를 잃고, fine-buf 버그(#4) 미수정 상태에서는 카메라가 움직이는 VK 세션이
즉사한다 — 즉 **F1 스트리밍은 이번 수정 4종이 전제**다.
