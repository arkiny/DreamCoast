# 그림자 아틀라스 + CSM 골격 (파이프라인 정합 PR-7)

상위: [render-pipeline-reference.md](render-pipeline-reference.md) §3 PR-7 · 관련:
[shadow-reflection-quality.md](shadow-reflection-quality.md) (단일 맵 PCF/PCSS-lite 품질 트랙).

단일 디렉셔널 shadow map(`SHADOW_SIZE=2048²` + 씬 전체 ortho box)을 **그림자 depth 아틀라스 +
디렉셔널 CSM(cascaded shadow maps)** 으로 일반화한다. 디폴트 OFF(레거시 단일 맵, 골든 앵커
바이트 동일), opt-in `CSM=<N>`.

## 사용법 (opt-in seam)

| env | 의미 | 디폴트 |
|---|---|---|
| `CSM=<N>` | N-cascade CSM 활성 (1..4). `CSM=0`/미설정 = 레거시 단일 맵 | off |
| `CSM_ATLAS=<px>` | 아틀라스 한 변 texel (1024..8192) | 4096 |
| `CSM_LAMBDA=<f>` | practical-split 블렌드 (0=uniform, 1=log) | 0.75 |
| `CSM_BLEND=<f>` | cascade 경계 블렌드 밴드 (cascade 깊이 구간 비율, 0..0.5) | 0.1 |
| `CSM_DEBUG=1` | cascade 인덱스 컬러 오버레이 (0 빨강 → 1 초록 → 2 파랑 → 3 노랑) | off |

## 설계

### 아틀라스 인프라 (`apps/sandbox/src/csm.rs`)
- **하나의 큰 depth 텍스처**(`shadow_atlas`, 렌더 그래프 transient)를 고정 타일 그리드로 분할:
  1 cascade → 1×1, 2 → 2×1, 3/4 → 2×2. 타일 변 = `atlas / max(cols, rows)` (디폴트 4-cascade
  4096² 아틀라스 = 2048² 타일 → 타일당 해상도는 레거시 단일 맵과 동일하나 **커버 범위가 훨씬
  좁아** 유효 texel 밀도가 올라간다).
- **슬롯 테이블이 타입드**: `ShadowSlot { kind: Cascade|Spot|PointFace, rect, view_proj,
  split_far }`. 이번 PR은 `Cascade`만 채우고, 스팟/포인트는 (a) 슬롯 kind, (b) 슬롯별
  view-projection 배열, (c) 아틀라스 UV sub-rect 배열이 이미 per-slot이므로 **재배선 없이**
  fill 루프만 추가하면 된다 (`record_shadow_atlas`가 kind로 분기).
- **fill**: `deferred.rs::record_shadow_atlas` — 한 depth 패스가 아틀라스 전체를 clear한 뒤
  슬롯마다 `set_viewport_scissor_rect(tile)`(신규 RHI 프리미티브, 3 백엔드 구현)로 타일에
  제한하고 캐스터를 그 cascade의 view-proj로 래스터. 캐스터 루프는 단일 맵 `record_shadow`와
  동일(스태틱/스킨/모프 파이프라인 + masked alpha-test discard) → cascade 그림자의 컷아웃이
  lit 메시와 일치.
- **샘플링**: 라이팅(`pbr.slang`)은 같은 `shadow_map` 리소스 id를 그대로 읽는다(재배선 zero).
  cascade의 clip → [0,1] 타일 UV → `csm_atlas_uv[i]` sub-rect로 리맵.

### CSM split scheme — practical split (canonical)
GPU Gems 3 ch.10 (parallel-split shadow maps)의 **log/uniform 혼합**:

```
d_log_i  = near · (far/near)^(i/N)
d_uni_i  = near + (far−near)·(i/N)
split_i  = lerp(d_uni_i, d_log_i, λ)        λ = 0.75 (CSM_LAMBDA)
```

log만 쓰면 근경에 몰리고 uniform만 쓰면 원경이 뭉개진다; λ≈0.75 혼합이 전 깊이 구간에 균형
잡힌 샘플 밀도를 주는 표준 선택이다.

### 뷰-안정성 (shimmer 방지, Valient stable CSM)
1. **슬라이스 bounding-sphere fit**: cascade의 절두체 슬라이스 8코너의 bounding sphere로 ortho
   박스를 잡는다. 구는 회전 불변이라 **카메라 회전에도 투영 크기가 불변** (tight AABB는 회전마다
   리사이즈 → 가장자리 crawling).
2. **texel snapping**: 구 중심을 라이트 공간에서 `tile / (2·radius)` texel 그리드에 스냅하고,
   스냅 오프셋을 투영의 NDC 평행이동(`proj.w_axis.xy += offset/half`)으로 반영 → 카메라 이동 시
   그림자 texel이 통짜 texel 단위로만 이동 (sub-texel jitter 제거).
3. radius를 1/16 단위로 올림해 float 미동에 의한 리스케일도 차단.

### Cascade 선택 + 블렌드 (pbr.slang)
- **containment-first select**: near→far 순으로 픽셀의 월드 좌표를 각 cascade에 투영해 **처음
  들어맞는(타일 UV ∈ [0,1]) cascade**를 쓴다. 순수 depth-split 선택은 (радial view-depth ≠
  perspective z) + (구가 슬라이스보다 큼) 때문에 포함 안 되는 cascade로 라우팅될 수 있다 —
  containment는 항상 커버하는 것 중 최고 해상도 cascade를 고른다.
- **경계 블렌드**: cascade 깊이 구간의 마지막 `blend_frac`(디폴트 10%)에서 다음 cascade와
  cross-fade → 해상도 스텝이 하드 seam으로 안 보인다.
- **필터 품질 유지**: 기존 3×3 PCF(하드) / PCSS-lite(블로커 서치 + Poisson PCF, `SHADOW_SOFTNESS`)
  로직을 `shadow_filter()`로 추출해 단일 맵과 cascade가 **같은 코드**를 쓴다. cascade 탭은 타일
  경계 1-texel 인셋으로 clamp해 이웃 타일로의 bleed를 차단.

### 단일 소스
- 모든 매트릭스/split/타일 rect는 Rust(`csm.rs::compute_cascades`) 한 곳에서 계산, globals
  (`csm_params/csm_split/csm_opts/csm_view_proj[4]/csm_atlas_uv[4]`)로 셰이더에 공급.
- `GLOBALS_SLICE` 512 → 1024 (256-정렬 유지; D3D12 root CBV OK). OFF 경로는 새 필드가 전부 0
  → 셰이더가 `csm_params.x == 0` 분기에서 레거시 경로만 타므로 **바이트 동일**.
- 신규 RHI 프리미티브 `set_viewport_scissor_rect(Rect2D)`: Metal/Vulkan/D3D12 3 백엔드 +
  `Recorder` trait + command-list IR(`RhiCommand::SetViewportScissorRect`) 구현.

## 검증 (Metal, macOS M3 — DX/VK parity pending Windows verification)

| 게이트 | 결과 |
|---|---|
| `cargo clippy --all-targets -- -D warnings` + `cargo test` | clean / 전체 통과 |
| 디폴트 OFF 골든 앵커 | `af70c1a5…` **바이트 동일** (sha256 일치, GLOBALS_SLICE 1024 확장 후에도) |
| 갤러리 `CSM=4` vs 단일 맵 | avg **0.016/ch** (max 90, >8 diff 0.06%) — 근경 그림자 동등 (cascade가 taut한 투영이라 가장자리 소폭 sharpen). `CSM=1`도 0.174/ch |
| 갤러리 PT parity (canonical) | 단일 맵 **6.164/ch** vs `CSM=4` **6.177/ch** (같은 PT 레퍼런스) — 경로추적 대비 오차 동등 (+0.2% 상대) |
| sponza 원경 그림자 | 단일 맵은 씬 전체를 한 ortho box로 커버 → 바닥 광패치 경계가 톱니 aliasing; `CSM=4`는 같은 경계가 **매끈** (crop 비교 스크린샷) — 원경 해상도 개선 입증 |
| cascade 디버그 뷰 | `CSM_DEBUG=1` sponza — cascade 0(빨강) 근경 / 1(초록) 중경 / 2(파랑) 원경 문, split 경계가 nave 깊이를 따라 타당 |
| GPU 시간 | 갤러리 `CSM=4` ≈ **+3 ms/frame** (min-frame 112.0→114.8 ms + 프레임 램프 증분 246.9→250.3 ms 두 측정 일치; 4×2048² 타일 캐스터 재래스터 + 4096² 아틀라스 clear 비용). sponza는 공유 GPU 경합으로 헤드리스 수치 불확정 — Windows parity 검증 시 재측정 권장. 헤드리스 PROFILE_GPU의 첫 패스 행은 프레임 간 idle을 흡수하는 기존 아티팩트가 있어 절대값은 신뢰 불가 |

### 알려진 한계 / 후속
- 스팟/포인트 슬롯은 **골격만** (타입/배열/fill-분기): Phase 21 다광원에서 캐스터 루프 추가.
- cascade별 depth bias는 단일 `shadow.x`를 공유 — cascade ortho 스케일별 bias 보정(slope-scaled
  bias 또는 per-cascade bias 배열)은 아티팩트가 관찰되면 후속.
- `SHADOW_SEARCH_UV`/`SHADOW_MAX_PENUMBRA`는 아틀라스 UV 기준이라 타일이 작아지면 월드 기준
  penumbra가 cascade마다 달라진다 — PCSS 파라미터의 per-cascade 정규화는 후속(현재는 육안 동등).
- 갤러리처럼 far(100m) ≫ 씬 반경인 경우 원경 cascade가 낭비된다 — 씬 AABB 기반 far-clamp 후속.
- 캐스터를 cascade마다 전부 재래스터한다(N× draw) — cascade 절두체별 캐스터 컬링(기존 P7 GPU
  컬링 재사용)으로 절감 가능, 다광원(Phase 21)과 함께 후속.
