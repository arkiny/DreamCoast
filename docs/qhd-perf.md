# QHD 90fps + UHD 확장 성능 트랙 (권위 계획)

상위: [sponza-perf.md](sponza-perf.md)(Sponza HD 60fps 완료 — DX 12.6 / VK 16.6ms). 이 트랙은
**QHD(2560×1440) 90fps(≤11.1ms/frame)** 를 목표하고, **UHD(3840×2160) 복잡 씬**으로 확장 가능한
구조를 만든다. 양 백엔드(D3D12·Vulkan), 데모 앵글, 갤러리 무회귀 유지.

## ★ 1원칙: 측정 먼저 (Sponza 트랙과 동일)

## Stage 0 — 해상도 스케일링 측정 (2026-06-28, RTX 2070 SUPER, Med, Sponza 데모)

`RENDER_RES=WxH`(신규 임시 헤드리스 오버라이드 — **현재는 윈도 크기라 디스플레이가 2052×1133로 클램프**).
HD(0.92MP)와 클램프된 ~2.3MP 두 점으로 픽셀당 비용 모델링.

| 해상도 | MP | DX 총합 | VK 총합 |
|---|---:|---:|---:|
| HD 1280×720 | 0.92 | 12.6ms | 16.6ms |
| 2052×1133 (clamp) | 2.32 | 24.0ms | 34.8ms |

**선형 모델** `frame = fixed + slope×MP`:
- DX: slope **8.14 ms/MP**, fixed **5.1ms**.  VK: slope **13.0 ms/MP**, fixed **4.6ms**.
- fixed(해상도 독립) = sdf_cache_light(~3.2 DX/8.6 VK) + shadow(0.8) + 카드 가시성 + 오버헤드.
- per-pixel(해상도 비례) = gdf_gi, gdf_reflect, reflect_temporal, GI 디노이저(atrous×2), ssr, gbuffer,
  lighting, tonemap, upsample. (측정: 2.32MP에서 gdf_gi 2.6→6.3, gdf_reflect 3.0→7.1 ≈ 픽셀비 2.5×)

### 실측 (내부 렌더 extent 디커플링 후, 2026-06-28)
디스플레이가 윈도를 클램프해도 **씬 패스는 오프스크린 타깃이라 무제한** — `RENDER_RES`를 윈도 크기가 아니라
**내부 렌더 extent**로 승격(tonemap이 render-extent HDR을 UV 샘플→스왑체인 다운스케일). 기본(RENDER_RES 미설정)은
render=swap=윈도라 **바이트 동일**. 이제 진짜 QHD/UHD를 헤드리스 측정 가능.
| | MP | DX | VK | 90fps(11.1ms)? |
|---|---:|---:|---:|:--:|
| QHD 2560×1440 | 3.69 | **36.5ms (27fps)** | **45.3ms (22fps)** | ❌ 3.3×/4.1× |
| UHD 3840×2160 | 8.29 | 77.8ms (13fps) | 97.1ms (10fps) | ❌ |

**핵심**: QHD를 render_scale 0.5로 렌더 = 내부 1280×720 = 이미 최적화한 HD 비용(DX 12.6/VK 16.6ms)을 QHD로
업스케일. 즉 **internal render scale + 업스케일**이 Sponza HD 작업을 그대로 재활용해 QHD 90fps에 도달하는 길.

**결론**: 고해상에서 per-pixel 비용이 지배. 픽셀당 비용을 ~3–5× 줄여야 QHD 90fps. 절대 픽셀 수를
줄이는 **내부 렌더 해상도 디커플링(internal render scale + 업스케일)** 이 핵심 레버이자 UHD 확장의 정석
(콘솔/DLSS/FSR/TAAU = 내부 저해상 렌더 → 디스플레이로 업스케일).

## 제안 아키텍처 (검토 필요)

1. **내부 렌더 해상도 디커플링 (기반)**: 모든 무거운 패스(g-buffer + GDF + 디노이저 + 라이팅)를 **내부
   extent**(스왑체인과 분리)로 렌더 → 오프스크린 타깃. 현재 `cw,ch=swapchain.extent`를 `render_extent`로
   분리. 최종 present/스크린샷은 내부 LDR을 디스플레이로 **업스케일 블릿**. (이게 디스플레이 클램프 회피 +
   QHD/UHD 측정 가능 + 최적화 동시 달성.) `RENDER_RES`를 내부 extent로 승격.
2. **렌더 스케일 노브**: `quality.rs`에 `render_scale`(예 Low 0.5 / Med 0.67 / High 0.8, 또는 동적). QHD
   출력에서 내부 0.44~0.5 스케일이면 per-pixel ~4× 절감.
3. **업스케일러**:
   - **A안(빠름)**: 공간 업스케일(FSR1-lite: Lanczos/EASU 류) — 단순, temporal 의존 없음.
   - **B안(고품질)**: 시간적 업스케일(TAAU) — jitter + 히스토리 재투영(엔진에 이미 temporal 인프라 多).
   - 점진: A 먼저(측정/기반), 필요 시 B.
4. **fixed 비용 추가 절감**: 내부 스케일이 낮아지면 fixed(cache relight)가 상대적으로 커짐 → 캐시 추가
   상각/async-compute로 raster와 오버랩(Sponza 트랙의 VK 후속 후보와 동일).
5. **복잡 씬 대비(UHD)**: 지오메트리 컬링(Sponza 트랙서 보류한 Stage A/B/C)이 복잡 씬에선 유효 →
   프러스텀/오클루전 컬링으로 g-buffer/shadow 제출 절감. 측정 후 우선순위.

## 게이트 (Sponza 트랙과 동일)
- PROFILE_GPU before/after, 양 백엔드, **갤러리 무회귀(render_scale=1=바이트 동일 앵커)**, DX≡VK,
  Vulkan 검증 클린, fmt+clippy. 업스케일 품질은 PT 잔차 + 육안 + (가능하면) 풀해상 레퍼런스 대비.

## 하지 말 것
- 측정 없이 추측. 갤러리 무회귀 위반(render_scale=1 경로는 바이트 동일). DX≡VK 깨기. 새 무거운 의존
  (FSR/DLSS SDK 등 — 자체 구현 또는 승인). 한 해상도만 최적화.

## 파일 (예상)
- `main.rs`(render_extent 분리 + present 업스케일 블릿 + RENDER_RES 승격), 신규
  `crates/shader/shaders/upscale.slang`, `quality.rs`(render_scale), `deferred.rs`/그래프(오프스크린 타깃).
- 본 문서(측정 표 갱신), ROADMAP.
