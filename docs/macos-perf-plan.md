# macOS (Apple Silicon) 성능 최적화 계획 — Sponza Med 60fps @ M3

> 다음 세션 콜드스타트 작업 계획. 목표: **MacBook Air M3에서 Sponza Med 60fps(≤16.6ms)**.
> 자기완결적으로 작성. 먼저 `DreamCoast/CLAUDE.md`, 이 문서, 아래 참고 문서.

## 컨텍스트 / 왜 새 트랙인가

- 기존 성능 트랙([swrt-gi-perf-track.md](swrt-gi-perf-track.md), [qhd-perf.md](qhd-perf.md))은
  **RTX 2070 SUPER(Windows)** 튜닝이다. **M3 MacBook Air**는 완전히 다른 타깃: 약한 **통합 GPU**,
  **TBDR(타일 기반 지연 렌더)**, **통합 메모리**, 열 제한. 같은 Med 설정으론 60fps 미달(사용자 확인).
- 프레임의 ~80%는 **SW-RT GI 스택**(`gdf_gi`/`gdf_reflect`/`sdf_cache_light`)이 차지(perf 트랙 확인).
  M3에선 여기에 **TBDR 대역폭**(G-buffer/라이팅 풀스크린 패스)이 더해진다.

## ★ 1원칙: 측정 먼저 (perf 트랙과 동일)

M3에서 `PROFILE_GPU=1 LEVEL=sponza RENDER_QUALITY=med … --screenshot-clean`로 **패스별 ms**를 먼저 분해.
모든 before/after를 ms로 보고. (Metal GPU 타이머 동작함 — `dreamcoast-metal-milestones`.) 상위 3패스가 타깃.

## 현재 스케일러빌리티 노브 (이미 존재 — 재사용)

`apps/sandbox/src/quality.rs` `RenderQuality{Low,Med,High}` (`RENDER_QUALITY` env; unset ⇒ **platform default**
— Apple 자동 티어를 끼울 훅). 노브(단일소스):
- `render_scale` — 내부 렌더 해상도 비율 + TAAU 업스케일. **Low=0.6667, Med/High=1.0(네이티브)**. `RENDER_SCALE`.
- `gdf_cone_k` — cone-trace LOD(원거리 step↓). `gi_res_div` — GI 추적 sparse(Low/Med 3=third-res).
- `P11_CACHE_RELIGHT_PERIOD` — 서페이스 캐시 상각 재조명 주기. `card_tile` — 캐시 아틀라스 타일 크기.
- 반사 clamp/roughness 게이트. **갤러리는 전 노브 레거시 강제(바이트 동일 앵커).**

## 축 A — Scalability 티어로 60fps 우선 달성 (가장 빠른 경로, 셰이더 무변)

**M3 Med가 네이티브 1440p(`render_scale=1.0`)라 비쌈** — 이게 첫 레버.
- **A1 Apple 플랫폼 기본 티어**: `RenderQuality::from_env()`의 platform-default를 **Apple GPU 감지 시 공격적**으로
  (device 이름/vendor로 Apple 판별 → 내부 해상도↓ + GI/반사/relight 노브↓). 새 티어 `MacMed` 또는 Apple에서 Low-Med 블렌드.
- **A2 내부 해상도 다운 + TAAU**: Med(Apple)에 `render_scale≈0.67`(1440p 출력 → ~960p 내부) 적용 —
  SW-RT는 픽셀당이라 **~2.2× 픽셀 감소 = GI/반사 스택 직접 절감**. TAAU가 재구성(qhd 트랙 검증).
- **A3 GI/반사/relight 공격적**: `gi_res_div=4`(quarter, 단 DX≡VK 회귀 유의 — perf 트랙 정정 참고, macOS는 Metal 단독이라 여유), `cone_k`↑, `relight_period`↑, `card_tile`↓, 반사 저해상.
- **게이트**: M3 60fps 달성 + 시각 정성 수용 + **갤러리 앵커 바이트 동일**(노브 레거시 강제) + 결정론.

## 축 B — Apple GPU 아키텍처 최적화 (macOS 전용, 중기)

- **B1 Memoryless transient 타깃 (대역폭 큰 이득)**: `crates/rhi-metal`에 **없음**(grep 확인) → TBDR에서
  G-buffer/depth/transient을 **`MTLStorageModeMemoryless`**로 할당하면 타일 메모리에만 상주, 시스템 메모리
  왕복 제거. 렌더그래프 transient가 후속에서 안 읽히면 memoryless 후보. RHI-metal 렌더타깃 할당 + 렌더그래프 aliasing 연동.
- **B2 SIMD-group 32 튜닝**: 컴퓨트 threadgroup이 `[8,8]=64`; Apple SIMD 32-wide occupancy를 `PROFILE_GPU`로
  측정해 조정(perf 트랙 P5는 Turing에서 음의 결과였음 — Apple은 별개, 재측정).
- **B3 (후속) 타일 셰이딩 딥퍼드**: 풀스크린 G-buffer read 대신 타일 메모리에서 라이팅(programmable blending) —
  Apple TBDR 강점, 큰 리팩터.

## 축 C — SW-RT GI 비용 절감 (레퍼런스 상용 엔진 SW GI 소스 참고)

레퍼런스 SW GI의 **상각 + 피드백** 기법을 이식(일반 기법, 상표명 금지):
- **C1 서페이스 캐시 상각 재조명**: 레퍼런스는 프레임당 텍셀 **일부만** 갱신(직접·간접 각각 update-factor로 N/M
  프레임 분산). 우리 `P11_CACHE_RELIGHT_PERIOD` 상향으로 노출됨 — M3 Med는 공격적 주기(품질=이동카메라 lag 트레이드).
- **C2 가시성/우선순위 피드백**: 실제 GI/반사에 **샘플된 카드만** 고빈도 재조명(`card_vis` 버퍼 존재). perf 트랙은
  Windows 데모(온스크린 카드 대부분 샘플)에서 ROI 낮아 **보류**했으나, M3 + off-screen 카드 많은 씬에선 재측정 권장.
  **F1 카메라-우선순위 레지던시와 결합** = 강한 시너지.
- **C3 스크린 프로브 중요도 게더**: F4(중요도 샘플, 현재 defer — 바이트-정확 재작업 후) 완료 시 같은 spp에 노이즈↓ → spp 자체를 낮출 여지.
- **C4 반사**: GGX spp/해상도↓, roughness 게이트(거친 표면 저해상).

## 스테이지 (권장 순서 — 빠른 승리 먼저)

1. **M0 측정** — M3 `PROFILE_GPU` Sponza Med 패스별 ms 베이스라인.
2. **M1 Apple 티어(축 A)** — 내부해상도↓ + TAAU + GI/반사/relight 노브로 **60fps 우선 달성**(셰이더 무변, 최소 리스크).
3. **M2 Memoryless(축 B1)** — RHI-metal transient을 memoryless로(대역폭).
4. **M3 캐시 상각/피드백(축 C1·C2)** — relight period + `card_vis` 피드백 M3 재측정.
5. **M4 (후속)** — 타일 셰이딩 딥퍼드(B3), SIMD 튜닝(B2), F4 중요도 게더 결합(C3).

## 게이트 (전 스테이지)

`PROFILE_GPU` before/after(M3) → **60fps(≤16.6ms) 목표** → **갤러리 앵커 바이트 동일**
(`af70c1a5…`, 노브는 갤러리 레거시 강제) → 결정론(run-to-run) → 콘텐츠 시각 정성 → `cargo fmt` +
`clippy -D warnings` → **골든이미지 러너**(`tools/golden-image.py`) 무회귀. DX≡VK는 Windows 동결이나
크로스백엔드 노브(gi_res_div 등)는 파리티 유의(perf 트랙 P1 정정 교훈).

## 하지 말 것
- 측정 없는 최적화. 갤러리 앵커 무단 변경(라이팅 개선이면 PT 검증 후 재기준선). 상표명 산출물.
- Apple 전용 경로가 다른 백엔드를 깨기(memoryless/티어는 백엔드 seam 뒤로). heavy 기본 ON 강제.

## 참고 문서 / 소스
- [swrt-gi-perf-track.md](swrt-gi-perf-track.md)(cone/sparse/relight 노브·측정법), [qhd-perf.md](qhd-perf.md)(TAAU),
  [render-quality-tiers.md](render-quality-tiers.md)(티어 구조), [metal-backend.md](metal-backend.md),
  [gi-fidelity-roadmap.md](gi-fidelity-roadmap.md)/[gi-fidelity-phases.md](gi-fidelity-phases.md)(충실도 트랙과의 관계).
- 레퍼런스 상용 엔진 SW GI 소스: scalability cvar 계층(스크린 퍼센티지·GI/반사/그림자 품질 버킷) → 우리 티어로 매핑;
  서페이스 캐시 update-factor + 가시성 feedback(일반 기법으로 이식, 상표명 미표기).
- 코드: `apps/sandbox/src/quality.rs`(티어·노브), `main.rs`(배선), `gi.rs`/`reflect.rs`/`gdf.rs`(SW-RT 소비자),
  `crates/rhi-metal`(memoryless/타일 — B축).
