# F1 계획서 — 표면 캐시 가상화 (Surface Cache Virtualization)

> 상태: **승인됨 (2026-07-13) — 전 범위 Stage 0–4, Stage 0 포함**. 브랜치
> `feature/f1-surface-cache-virtualization`(HEAD `feature/pt-auto-exposure`에서 분기 — F6 자동노출로
> 실내 PT 잔차 측정 가능해야 하므로). 각 Stage 자체 커밋, 순서대로 검증 후 랜딩. 착수점
> [phase-f1-surface-cache-virtualization-prompt.md](phase-f1-surface-cache-virtualization-prompt.md),
> 상위 [gi-fidelity-roadmap.md](gi-fidelity-roadmap.md) §F1. 이 계획서는 8,600여 줄
> (`fuse.rs`/`gdf.rs`/`gi.rs`/`reflect.rs`/`sdf_cache_*.slang`/`sdf_atlas.rs`) 정독 위에 작성됐고,
> 모든 주장은 `file:line` 앵커로 근거를 단다.

---

## 0. TL;DR — 정직한 재정의 (지도 조사 결과)

로드맵은 F1을 "279 draw 드롭 → 라이팅 구멍"으로 요약한다. 코드를 읽어보니 **그림이 더 정확하고,
일부는 이미 shipped**다. F1을 실제 코드 상태에 맞춰 재정의한다.

**이미 존재하는 것 (F1이 "새로 만들" 필요 없음):**
- **(c) 카드 mip / 원거리 저해상**: `sdf_cache_mipgen.slang` + `card_mip_sample`(cone-LOD 선택,
  `surface_cache.slang:87-161`)로 **완전 구현**됨(커밋 `17d7666`). 거리기반 가변 per-card 해상도도
  `assign_card_res`/`normalize_card_res`(`fuse.rs:358-429`, res∈[8,64], 텍셀 예산 정규화)로 wired.
  단, **반사 경로 전용**(`extra_tol>0 && mip bound`) — GI 게더는 센티넬로 mip0 nearest.
- **하드 드롭 회피(무차별)**: `P11_REFLECT_HQ`가 `card_budget=1<<20`으로 **모든 draw를 카드화**
  (`main.rs:2402-2406`). 단 메모리·워밍업 비용이 씬 규모에 선형(가상화 아님).
- **데맨드 신호 substrate**: `card_vis`(per-frame 프러스텀, Y-flip-free, DX≡VK; `sdf_cache_visibility.slang`),
  `card_marks`(반사 샘플러의 `InterlockedOr`), `card_res_feedback`(`InterlockedMax`) — 전부 교환법칙 atomic.
- **재할당(re-pack) 선례**: `relayout_from_feedback`(`gdf.rs:1416-1454`) — 피드백 readback → 예산 재정규화
  → `cache_layout` 재구성 → 전 아틀라스 버퍼 realloc(`wait_idle` + 강제 재캡처 + relight 리셋). **스트리밍
  re-pack의 기성 메커니즘.**

**진짜로 없는 것 (= F1의 실질 과업):**
1. **동적 데맨드 레지던시.** 레지던시는 **정적**이다 — 씬 로드시 `App::new`에서 **단 한 번**,
   고정 `CardCamera` 포즈로 `select_card_residency`가 top-`MAX_CARDS/6`을 고르고 끝(`main.rs:2383-2408`,
   `fuse.rs:140-166`). 라이브 카메라가 대형 씬을 날아다녀도 **레지던트 집합은 절대 갱신 안 됨** →
   로드 시점에 멀었던 draw는 코앞에 가도 영원히 coarse-fallback.
2. **스트리밍 페이지 풀 + LRU.** 레지던시는 **1회성 하드 컷**이다 — 시간적 지속성 없음, 방출/재승인 없음,
   부분 재캡처 없음. 고정 풀에 카드를 데맨드로 스트림-인하고 LRU로 방출하는 구조가 없다. 아틀라스는
   `num_cards`에 비례해 up-front 할당(`alloc_cache_texel_buffers`, `gdf.rs:1387-1414`).

**부수로 발견한 잠복 버그 (F1 충실도 게이트의 핵심):**
- **relight 게더가 coarse-fallback 히트에 검정을 더한다.** `sdf_cache_light.slang:314-328`: 게더 레이가
  표면에 hit하면 `cache_hit`를 **선언만 하고 무시**한 채 `indirect += sample(...)`. 캐시 미스 시 샘플러는
  `float3(0)` 반환 → 비레지던트 표면에서 오는 **다중바운스 기여가 0**. 1차 GI 소비자(`bs_shade_hit`,
  `gdf_bounce.slang:247-252`)는 dense-field analytic 폴백이 있지만, **relight 내부 게더에는 없다.** 이게
  실제 "실내 라이팅 구멍"이고, **동시에 LRU 방출 카드의 올바른 폴백 시맨틱**(방출됨 → 검정 아님 →
  dense-field-lit)이다. → F1의 **Stage 0** 선결.

> 정직성 노트: GI **디퓨즈** 미스는 검정이 아니다(analytic sun+sky, dense albedo). "구멍" 서사는
> 정확히 (i) 위 다중바운스 검정-add 버그와 (ii) 반사의 analytic-tone 다운그레이드에 해당한다.
> F1의 GI 이득은 "검정 채우기"가 아니라 **다중바운스 색 정확도 + 스캔 미스 감소 + 라이브 카메라 추종**이다.

---

## 1. 현재 아키텍처 (근거 요약)

| 요소 | 사실 | 앵커 |
|---|---|---|
| 카드 | draw당 6면 축정렬 카드, 64B. `MAX_CARDS=1024`, `CARDS_PER_DRAWABLE=6` → 170 draw 수용. `CARD_TILE=32` | `fuse.rs:19,30`, `gdf.rs:240` |
| 레지던시 | 정적 1회성. `card_priority=proximity·4+relevance·2+size·0.1`(pure f64), top-N + draw-index tie-break | `fuse.rs:91-166`, `main.rs:2383-2408` |
| 아틀라스 | 평면 버퍼 `num_cards·tile²·16B` ×4(pos/albedo/radiance ring/mip). ~67MB@1024. up-front alloc | `gdf.rs:1387-1414` |
| 인디렉션 | 적응형 `cache_layout` 16B/카드 `(mip0_base,res,mip_base,pad)` — **base가 곧 페이지 테이블 후보** | `gdf.rs:1347-1383`, `surface_cache.slang:46-77` |
| 캡처 | **1회**(`scene_cache_captured` 래치). 텍셀당 GDF sphere-trace inward. 부분 재캡처 없음 | `main.rs:6101-6109`, `sdf_cache_capture.slang` |
| relight | **매 프레임**(period + `card_vis` 상각; off-screen ×8). **하나의 relight가 GI+반사 공유** | `sdf_cache_light.slang`, `gdf.rs:1789-1936` |
| 게더 seam | 카드-vs-coarse는 **암묵적**: 비레지던트 draw는 카드 부재 → 수락 실패 → `found=false` → 소비자 analytic 폴백 | `surface_cache.slang:242-341` |
| 소비자 | GI `bs_shade_hit`(legacy wrapper, 센티넬, mip0 nearest, O(num_cards) 스캔), 반사 `gdf_reflect`(cone+mip+grid) | `gdf_bounce.slang:236`, `gdf_reflect.slang:1193` |
| 재할당 선례 | `relayout_from_feedback` — readback→재정규화→realloc→강제 재캡처 | `gdf.rs:1416-1454`, `main.rs:5724` |

**결정론 불변식(반드시 보존):**
- within-budget(draw ≤ `MAX_CARDS/6`) → `all_resident` + 균일 tile 산술 → **갤러리 바이트 동일 앵커
  `65d04ceca2c4…`**(현행 기본 aniso16; `af70c1a5…`는 구 앵커. 착수 전 `golden-image.py`로 라이브 재확인).
- 데맨드 카운터는 **교환법칙 atomic(Or/Max)만** — `InterlockedAdd`/LRU-clock 가산은 순서의존 → DX≡VK 깨짐.
- 카드 그리드는 **오름차순 카드 인덱스**로 방출(FP 합 bit-identical) — 페이지 재정렬이 이를 깨면 안 됨.
- freeze는 **프레임-카운트 지평**(측정 EMA 아님) — DX/VK 동일 프레임 arm.
- readback 패턴: **N프레임 늦게 읽고 결정론적으로 act**(fence-guarded 고정 프레임; `cache_conv_probe`/
  `relayout_from_feedback` 방식).
- host-visible seed write(Metal device-local `contents`가 NULL).

---

## 2. 설계 — 고정 페이지 풀 + 데맨드 스트리밍 LRU

핵심 아이디어: **레지던시를 정적 1회성 하드 컷에서, 라이브 데맨드가 채우고 LRU가 방출하는 고정-크기
페이지 풀로 바꾼다.** 소비자는 이미 `found=false`를 처리하므로(카드 부재 == 방출), **방출은 소비자에
투명**하다. 인프라 대부분은 재사용.

```
[라이브 카메라 프러스텀 card_vis]  ┐
[GI+반사 데맨드 마크(=이번 프레임 실제 샘플된 카드)]  ┼─▶ 레지던시 요청 집합(프레임 R 늦게 readback)
[LRU last-touched 프레임 스탬프]  ┘         │
                                            ▼
   고정 페이지 풀(P 슬롯) ── 요청∖레지던트 = 승인(빈/LRU-방출 슬롯 할당) ──▶ 부분 재캡처(K/frame 예산)
                                            │
                          방출된 카드 → drawable coarse-fallback(= dense-field-lit, Stage 0 이후 검정 아님)
```

- **페이지 = 카드**(6면 중 1면 단위 슬롯). 풀 크기 P는 현행 메모리 예산(≈67MB, ~1024 카드)과 동급으로
  고정 → **레지던트 draw 수와 무관하게 메모리 상한 유지**(가상화의 핵심; `P11_REFLECT_HQ`의 1M-카드
  블로업 없이 동일 커버리지).
- **인디렉션**: 기존 `cache_layout.base`를 풀 슬롯 포인터로 재해석(또는 병렬 `card_page[]` 테이블).
  소비자 셰이더 변경은 `card_layout()`/`layout_find_card()`(`surface_cache.slang:46-77`) 한 곳.
- **LRU 키**: per-카드 `last_touched = InterlockedMax(frame)`(교환법칙 → 결정론). tie-break은 draw-index.
- **부분 재캡처**: 승인된 카드만 재캡처(현재 캡처는 전-아틀라스 1회; `relayout_from_feedback`가
  이미 `wait_idle`+강제 재캡처 경로를 가짐 — 이를 **per-슬롯 dispatch 범위**로 좁힌다). 비용 상한을
  위해 프레임당 K카드 예산.

---

## 3. 단계별 계획 (각 단계 독립 게이트 통과 · 자체 커밋)

> 규칙: 각 단계 = `cargo fmt` → `clippy -D warnings` → **PT 잔차 보고**(콘텐츠 raster vs `P8_PATHTRACE=1`,
> `tools/rt-compare.py`; 실내는 F6 자동노출로 측정 가능) → **갤러리 바이트 동일 `65d04ceca2c4…`** →
> **결정론(run-to-run)** → **DX≡VK**(Windows 동결 → Metal 검증 + 명시 보류) → `PROFILE_GPU` 비용.
> heavy=opt-in seam, 단일소스, 상표명 금지.

### Stage 0 — coarse-fallback 다중바운스 검정-add 폴백 수정 (선결·저위험)
**왜:** F1 전제("구멍 없음")의 실제 최대 구멍이자, LRU 방출의 올바른 시맨틱(방출→dense-field-lit).
**변경:** `sdf_cache_light.slang:314-328` — `cache_hit==false`일 때 `bs_shade_hit`와 동형의 dense-field
analytic 재조명(sun·shadow + sky-fill, `bs_albedo_at` voxel albedo) 값을 `indirect`에 더한다(0 대신).
단일소스: 폴백 재조명을 공유 헬퍼로 뽑아 게더/`bs_shade_hit`가 공유.
**게이트:** 콘텐츠 PT 잔차 **개선**(실내 다중바운스 밝기↑, 정확도↑) / 갤러리 바이트 동일(within-budget →
coarse-fallback 없음 → 무변). opt-out `P11_GATHER_FALLBACK=0`.
**측정:** `LEVEL=sponza_intel_chromeball` 실내 각도 raster vs PT 잔차, before/after; 갤러리 SHA.

### Stage 1 — 고정 페이지 풀 + 카드→페이지 인디렉션 (기능 무변 리팩터)
**왜:** 아틀라스 크기를 레지던트 카드 수에서 분리하는 메커니즘 도입.
**변경:** `gdf.rs alloc_cache_texel_buffers`/`upload_card_layout` — 아틀라스를 **고정 P 슬롯 풀**로 할당,
`cache_layout.base`를 슬롯 인디렉션으로. 초기엔 **레지던트 카드 = 슬롯 1:1 identity** → 출력 무변.
소비자 `card_layout()` 인디렉션 경유(1곳). 그리드는 오름차순 슬롯 순서 유지.
**게이트:** **전 씬 바이트 동일**(레지던트 ≤ 풀이면 no-op 리팩터). 결정론/DX≡VK 자명.
**측정:** 갤러리 + sponza 모두 SHA 불변, 메모리 리포트(풀 상한 고정 확인).

### Stage 2 — 데맨드 피드백: GI+반사가 이번 프레임 실제 샘플한 카드 마킹
**왜:** LRU 시계 + 레지던시 요청 집합의 입력.
**변경:** (a) GI `bs_shade_hit`가 legacy wrapper 대신 mark 인자를 가진 게더 호출 →
`card_touched`에 `InterlockedMax(frame)`(교환법칙). (b) 반사는 기존 `card_marks` 재사용하되 **full-frame
비용 블로업 방지**(측정치 5.0→13.2ms) 위해 best-card만 1회 마크(후보 전수 아님). gi_cache_arg에 마크
버퍼 인덱스 pack(반사 경로와 동형, `main.rs:6358` 참고 — 단 GI tuple 여유비트/신규 push 필드 확인).
**게이트:** **바이트 동일**(마크는 side-buffer, 출력 무영향) / 결정론(Max 교환법칙) / `PROFILE_GPU`
relight+게더 비용 회귀 없음(마킹 저비용 확인).
**측정:** 마크 히트맵(디버그 뷰) + relight ms before/after.

### Stage 3 — LRU 방출 + 데맨드 승인 + 부분 재캡처 (스트리밍 코어)
**왜:** F1의 본체 — 레지던트 집합이 라이브 데맨드를 추종.
**변경:** 프레임당(상각): (1) `card_touched` + `card_vis` + 라이브-카메라 프러스텀 요청을 **R프레임 늦게
fence-guarded readback**(결정론). (2) 요청∖레지던트 draw를 승인: 빈 슬롯 또는 LRU(min `card_touched`,
tie=draw-index) 방출 슬롯 할당. (3) 승인 카드 **부분 재캡처**(프레임당 K예산; `relayout_from_feedback`의
`wait_idle`+재캡처 경로를 per-슬롯 dispatch로 축소, relight period 리셋). (4) 방출 카드 → coarse-fallback
(Stage 0으로 dense-field-lit). **within-budget/갤러리는 정적 all_resident 경로 유지**(풀 방출 안 일어남).
**게이트:** 콘텐츠 PT 잔차 **개선/중립**(로드시 멀던 draw에 접근하면 카드 획득 → 향상) / 갤러리 바이트
동일(within-budget → 스트리밍 비활성) / 결정론(LRU 키가 결정론적 프레임스탬프+인덱스, 시드 고정) / DX≡VK.
opt-in seam `P11_CACHE_STREAM`(콘텐츠 기본; 갤러리/within-budget 무영향).
**측정:** 카메라 flythrough 경로에서 coarse-fallback 카운트 시계열(279→가변, 근접 draw 0으로 수렴),
before/after PT 잔차, 스트림-인 팝 없음(temporal 안정), 재캡처 K-예산 비용.
**리스크:** async relight 3슬롯 ring과 신규 레지던트 카드의 첫 relight 타이밍(읽기 전 relight 보장);
카드 그리드 재빌드(페이지 in/out 시) — 증분 or 관용. freeze 래치가 정적 카드 집합 가정 → 스트리밍 중
epoch/재캡처 재-arm 경로 필요.

### Stage 4 — GI 카드-mip 패리티 + 원거리 저해상 승인 (기존 인프라 재사용·선택)
**왜:** 풀 예산으로 더 많은 draw 수용 + GI cone 정합.
**변경:** (a) GI 게더에 기존 cone-LOD mip 파이프(`card_mip_sample`) 배선(현재 센티넬 →
sample_radius 계산 + pyramid 인덱스 plumb). (b) 원거리 승인 카드는 `assign_card_res`(∝ext/dist)로 저해상
슬롯 할당 → 동일 메모리에 더 많은 draw. mip 인프라(`record_cache_mipgen`)는 무변.
**게이트:** PT 잔차 중립/개선, 비용↓/중립, 갤러리 바이트 동일.
**측정:** 동일 풀 예산에서 레지던트 draw 수↑, GI 원거리 스펙클↓.

---

## 4. 재사용 매핑 (기성 인프라 → F1 역할)

| 기성 | 위치 | F1 역할 |
|---|---|---|
| `cache_layout.base` 인디렉션 | `gdf.rs:1347-1383`, `surface_cache.slang:46-77` | 페이지 테이블 |
| `relayout_from_feedback` | `gdf.rs:1416-1454` | 부분 재캡처/재-pack의 wait_idle 선례 |
| `card_vis`(프러스텀) | `sdf_cache_visibility.slang`, `gdf.rs:1698-1776` | 라이브 레지던시 요청(Y-flip-free, DX≡VK) |
| `card_marks`(InterlockedOr) | `surface_cache.slang:306-335` | 반사 데맨드 마크(재사용) |
| `card_res_feedback`(InterlockedMax) | `surface_cache.slang:297-303` | 원거리 저해상 승인(Stage 4) |
| mip 피라미드 + cone-LOD | `sdf_cache_mipgen.slang`, `surface_cache.slang:87-161` | (c) — GI로 확장만 |
| `assign_card_res`/`normalize_card_res` | `fuse.rs:358-429` | 거리기반 슬롯 해상도 |
| `sdf_atlas.rs` 결정론적 pack | `crates/asset/src/sdf_atlas.rs` | **주의: 3D SDF-voxel 아틀라스로 2D 카드 캐시와 별개.** 재사용은 pack 알고리즘/gutter 계약 패턴에 한정(풀 lifecycle 없음) |

---

## 5. 하지 말 것 / 비목표
- (c) 카드-mip을 **재구현**(이미 존재 — GI 확장만).
- `MAX_CARDS`만 올려 때우기(메모리 선형 폭주 — 풀+LRU가 근본).
- `CARDS_PER_DRAWABLE=6` 변경(셰이더 `card/6` 매핑에 하드코딩, `sdf_cache_capture.slang:189`).
- 데맨드 카운터에 `InterlockedAdd`/비교환 LRU-clock(결정론 파괴).
- 갤러리/within-budget 경로에 스트리밍 활성(바이트 앵커 파괴 — content-only seam).
- 반사 full-frame 마킹 전수(측정 5.0→13.2ms 블로업 — best-card 1회만).
- 상표명(문서/주석/커밋 "reference engine").

---

## 6. 승인 요청 — 결정 필요 사항

1. **범위/순서.** 제안: Stage 0(검정-add 폴백) 선결 → 1(풀/인디렉션) → 2(데맨드 마크) → 3(스트리밍 LRU)
   → 4(선택, GI mip 패리티). 각 단계 자체 커밋. Stage 3까지가 F1 코어, 4는 폴리시.
2. **Stage 0을 F1에 포함 vs 별건 픽스.** 포함 권장(방출 시맨틱 선결이자 최대 실측 구멍).
3. **정직 체크.** F1의 실질은 "검정-add 수정 + 라이브 카메라 추종 스트리밍 풀"이고, (c)·무차별 no-drop은
   이미 shipped. 이 재정의가 로드맵 의도와 합치하는지 확인 요청.

**승인되면** Stage 0부터 검증 단일 커밋으로 착수한다. 착수 전 `golden-image.py`로 라이브 갤러리 앵커
재확인 + 새 `feature/f1-surface-cache-virtualization` 브랜치 분기.
