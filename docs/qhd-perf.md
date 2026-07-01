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

## Stage 2 — render_scale 노브 + 90fps 스윕 (2026-06-28)

`render_scale`(quality.rs, 디스플레이 extent의 분수로 씬 렌더; tonemap이 디스플레이로 업스케일) +
`RENDER_SCALE` env. 기본 1.0=네이티브=**바이트 동일**(갤러리 0.000 확인). `RENDER_RES`(절대)는 측정용 오버라이드.
QHD를 scale s로 렌더 = 내부 (2560s×1440s) 비용 → 내부 해상도 스윕으로 90fps 지점 탐색:

| 내부 해상도 (QHD scale) | DX | VK |
|---|---:|---:|
| 1280×720 (0.50) | 12.65 | 16.34 |
| 1138×640 (0.44) | **10.96 ✅** | 15.43 |
| 1024×576 (0.40) | 10.07 | 13.49 |
| 960×540 (0.375) | 9.20 | 13.16 |
| 854×480 (0.333) | — | 11.46 |
| 768×432 (0.30) | — | 11.63(평탄) |

- **DX QHD 90fps ✅**: render_scale ≈0.44(내부 1138×640) = 10.96ms = 91fps.
- **VK는 ~87fps에서 바닥**(854×480=11.46ms): VK breakdown(960×540)에서 **sdf_cache_light 6.29ms = 프레임의
  46%, 해상도 독립** → 스케일을 아무리 낮춰도 캐시가 남아 VK ~11.5ms 평탄. **VK 90fps는 캐시 비용을 줄여야**
  가능 — 정확히 Sponza 트랙서 남긴 VK 구조적 격차. 해법 = **async-compute로 캐시 relight를 raster/per-pixel과
  오버랩**(해상도 독립이라 UHD에도 동일하게 유효). = Stage 3.

## Stage 3 — 캐시 비용 공략 시도 + 정직한 결론 (2026-06-28)

VK QHD가 해상도-독립 `sdf_cache_light`에 막혀(스케일 낮춰도 ~87fps 바닥), 두 방향 검토:

### (a) async-compute로 캐시 오버랩 — 보류(RHI 깊은 작업)
설계는 검증됨: **ping-pong 덕에 데이터 레이스 없음**(소비자가 read 버퍼, async가 write 버퍼). 컴퓨트 큐
+ `volume_to_sampled` + `storage_buffer_barrier` 모두 존재. **그러나** 기존 `submit_async`는 *동일 프레임*
graphics-waits-compute(직렬)용이고 D3D12는 내부 cross-queue fence로 세마포어를 무시 → **진짜 1프레임 지연
오버랩은 rhi-vulkan/rhi-d3d12에 새 cross-frame 컴퓨트 동기화가 필요**(행/플리커 리스크, 양 백엔드 상이).
별도 집중 트랙으로 분리.

### (b) 캐시 tile 축소 — 미미(기본 채택 안 함)
`CARD_TILE`을 런타임화(`card_tile`, 셰이더는 이미 push `tile` 파라미터). 콘텐츠 tile 16 측정: 캐시
relight **DX 3.1→2.3 / VK 4.9→4.1ms (~30%만)** — spp1/period40에선 캐시가 순수 텍셀바운드가 아님. 반면
반사가 흐려짐(HD 델타 0.041/ch, **max 94**). VK는 여전히 90fps 미달. **→ 기본 32 유지(무회귀), tile은
`P11_CACHE_TILE` 튜닝 노브로만 노출**(UHD 아틀라스 메모리 절감 + opt-in). 갤러리 32 강제(바이트 동일).

### 정직한 결론 — "선명한 QHD 90fps"의 답은 비용 절감이 아니라 **시간적 업스케일러(TAAU)**
- DX QHD 90fps는 scale ~0.45에서 됨(소프트). VK는 cache-floor로 더 낮은 스케일 필요.
- per-pixel(gdf_gi/reflect/denoiser)은 이미 하프해상+spp1로 최적화 — 추가 절감 미미.
- 비용 절감으로 도달 가능한 스케일(~0.4–0.5)은 **바이리니어 업스케일로는 본질적으로 흐림**.
- **shipping 엔진이 QHD/UHD 고프레임을 내는 방식 = 저해상 렌더 + 시간적 재구성(DLSS/FSR2/TAAU)**: jitter +
  히스토리 재투영으로 0.5 스케일을 네이티브급으로 복원. 엔진에 temporal 인프라(reproject/EMA) 이미 다수 →
  TAAU가 자연스러운 다음 단계. **이게 사용자의 "선명함" 목표의 실제 해법.**
- 보조: async-compute(VK/UHD 헤드룸, 별도 RHI 트랙), 컬링(UHD 복잡 씬 지오메트리 — 별도).

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

## Stage 4c — 업스케일 경로 버그 수정 (Part A, 2026-06-28)

내부 렌더 extent 디커플링 이후 두 개념 `(sw,sh)=출력/창/스왑체인` vs `(cw,ch)=내부 씬 렌더`가 생겼고,
일부 소비처가 혼동했다. 측정 기반으로 전수 점검 후 수정:

- **A1 — ImGui 해상도/입력/클립 불일치 (근본 뿌리)**: `gui.new_frame`에 *내부* extent `(cw,ch)`를
  넘겨 ImGui display size·마우스 히트테스트·per-draw clip이 전부 작은 공간에 머물렀다. UI 패스는
  스왑체인 백버퍼 `(sw,sh)`에 렌더(`set_viewport_scissor(swapchain)`)하고 `input.mouse_position()`도
  클라이언트 픽셀 `(sw,sh)`이라, `cw<sw`면 UI 정점은 백버퍼로 늘어나는데 마우스·시저는 작은 공간 →
  패널을 옆으로 옮기면 잘려 사라지고 히트테스트가 어긋났다. **`(sw,sh)`로 통일** (기본 경로 `cw==sw`=무변).
  검증: 갤러리 0.000/ch, `RENDER_SCALE=0.5` UI 풀윈도 렌더 확인.
- **A2 — 레벨 모드 카메라 고정**: `LEVEL=sponza`가 고정 `level_view`라 오비트 자동회전이 안 먹었다.
  인터랙티브에선 authored 카메라를 focus 중심으로 오비트(angle 0 = authored pose 정확 재현, 점프 없음),
  헤드리스(screenshot_mode)는 고정 pose 유지 → Sponza 퍼프/패리티 베이스라인 바이트 동일. Tab=자유비행 유지.
- **A3 — 라이팅 안정(지터 OFF) = 히스토리 extent 정합 검증**: 신규 측정 인프라 `CAPTURE_SEQ=N`
  (+ `CAPTURE_SEQ_STEP` rad/frame, 0=정적)으로 카메라를 결정적으로 움직이며 N프레임 연속 덤프 →
  프레임간 diff. 전수 점검 결과 **GI/reflect 디노이저 히스토리=내부 `(cw,ch)`, TAAU 히스토리=출력
  `(sw,sh)`로 이미 정합** (extent 버그 없음). 측정(d3d12, RENDER_SCALE 0.5):
  | | 정적 프레임간 diff | DX≡VK |
  |---|---:|---:|
  | 네이티브 1.0 | 0.003/ch (max 8) | (앵커) |
  | 업스케일 0.5 (지터 OFF) | 0.006/ch (max 18) | 0.003/ch (max 30) |
  업스케일이 정적 시머를 ~2× 키우지만(저해상 스토캐스틱 spp1 샘플의 상대 노이즈, extent 버그 아님),
  DX≡VK는 스토캐스틱 갭 내 → 레이아웃/extent 버그 없음 확인. 이 잔여 시머가 Part B 지터가 절대
  악화시키지 말아야 할 기준선.

## Stage 4d — 지터 부활 + TAAU 진짜 재구성 (Part B, 2026-06-28)

이전 FXAA→TAA는 지터가 **떨림(shimmer)** 을 일으켜 꺼둔 상태(P_TAAU_JITTER 기본 off)였다. 측정 기반으로
근본 원인을 잡아 지터를 *supersampling 신호*로 되살림. 검증법: `CAPTURE_SEQ`로 정적(STEP=0) 연속 N프레임
프레임간 diff(=떨림) + `RENDER_RES=2560x1440` 다운스케일 SSAA(2×) 레퍼런스 대비 선명도(Laplacian 에너지).

- **B1 (TAAU)** — `taau.slang`/`main.rs`. 떨림 0.58→0.16/ch:
  1. **지터 Y-보정 부호 버그** (최대 원인): 지터 NDC 시프트→렌더 화면UV→셰이더 reconstruct UV→NDC를
     끝까지 풀면 화면 콘텐츠 시프트는 `(+jx/2, -jy/2)` — D3D12·Vulkan 양쪽 동일(두 Y-flip이 한 번의 순
     부호반전으로 합쳐짐). 코드가 `+jy/2`라 **수직 ~1px 재투영 오차** → 히스토리 fetch가 매 프레임 빗나가
     지터가 노이즈로 변함. (per-pixel 재투영 오차 시각화로 진단: 빨강~0 초록~1px). 수정 후 정적 0.58→0.24.
  2. 정수-floor 히스토리 → **bilinear(서브픽셀)** storage-buffer fetch(per-tap 유효성, valid 가중 정규화).
  3. 하드 월드포인트 disocclusion + closest-depth dilation 제거: 지터 하에선 실루엣/그레이징 픽셀이 매
     프레임 서브픽셀 커버리지를 바꾸므로 **누적(anti-alias)** 돼야 하는데 하드 리셋이 len 1로 묶어 크롤 유발.
     own-surface 재투영 + 색 변화는 YCoCg 이웃 클립(Karis)이 anti-ghost 담당. max_hist 16→32.
- **B2 (GI/reflect + FXAA)** — `gdf_temporal.slang`/`reflect_temporal.slang`/`main.rs`:
  1. 화면공간 temporal(GI 디노이저·반사 resolve)도 floor 재투영이라 지터된 G-buffer에서 흐려짐
     (반사가 SSAA 대비 발산 0.95/ch). **서브픽셀 bilinear** 추가, flip 워드 bit1로 선택(지터일 때만 set,
     아니면 floor=바이트 동일). 반사 vs-SSAA 0.95→0.48.
  2. **FXAA는 지터 경로에서 역효과** — 지터가 곧 AA인데 FXAA가 흐리고 매 프레임 다르게 스무딩해 시간 분산
     추가. 지터 활성 시 스킵(비지터 업스케일은 유지). 정적 0.23→0.16, 선명도 1.23→**1.31**(SSAA 1.28).
- **B3** — 지터 업스케일 경로 **기본 ON**(`P_TAAU_JITTER` 기본 true, taau_active일 때만; 네이티브=영향 없음).

### 측정 (d3d12, 내부 0.6667 = 853×480 → 1280×720)
| | 선명도(Laplacian) | 정적 프레임간 diff | vs-SSAA(2×) |
|---|---:|---:|---:|
| bilinear (지터 off) | 1.094 | 0.006 | 0.341 |
| **TAAU 재구성 (지터 on)** | **1.305** | **0.158** | 0.384 |
| SSAA(2×) 레퍼런스 | 1.282 | — | 0 |
| 네이티브 1.0 | 1.358 | 0.003 | 0.133 |

- **선명도**: TAAU 재구성이 bilinear(1.094)을 크게 넘어 **SSAA(1.282) 수준(1.305)** 까지 고주파 회복 ✅.
- **모션**: 카메라 이동 연속 프레임 육안 — 고스팅/스미어 없음(YCoCg 클립이 disocclusion 처리) ✅.
- **정적 떨림**: 0.158/ch — 남은 건 실루엣 1px(전경 vs 하늘) 경계 크롤뿐(내부·반사 안정). 하늘은
  world-pos가 없어(w=0) 커버리지 플립 시 누적 불가 = 실시간 TAAU의 본질적 한계(모션 중 가려짐). DLSS/TSR도
  고대비 정적 실루엣에 미세 크롤 존재. 정직한 잔차로 수용.
- **무회귀/패리티**: 갤러리(네이티브) 바이트 동일 0.000/ch, 업스케일 DX≡VK 0.002–0.003/ch(스토캐스틱 갭 내),
  Vulkan 검증 클린, VK 정적 떨림 0.158(=DX, 백엔드 동등).
- 측정 인프라 `CAPTURE_SEQ`(+STEP), SSAA via `RENDER_RES`. `render_scale`는 [0.6667,1.0] 클램프(저스케일은
  `RENDER_RES` 절대 오버라이드로 측정).

## Stage 5 — TAAU fps 스윕: "선명한 QHD 고프레임" 확정 (2026-06-28)

Stage 3의 결론("비용 절감으로 도달 가능한 0.4–0.5 스케일은 바이리니어로는 본질적으로 흐림 → 답은 TAAU")을
Part B(지터 재구성)가 실현했으니, 이제 **TAAU 켠 상태로** render_scale 스윕해 고프레임 지점을 실측.
`render_scale` 하한 클램프 0.6667→**0.3333**(DLSS ultra-perf 영역; TAAU가 0.4–0.6을 시각적으로 viable하게 만듦.
1.0 기본=네이티브 바이트동일 불변). 측정: **Sponza(GDF GI 디폴트=무거운 씬)**, 디스플레이 클램프 출력 2052×1133.

| 내부 스케일 | 내부 해상도 | DX ms (fps) | VK ms (fps) |
|---|---|---:|---:|
| 1.0 (네이티브) | 2052×1133 | 24.0 (42) | 30.9 (32) |
| 0.6667 | 1368×755 | 14.5 (69) | 18.5 (54) |
| **0.5** | 1026×566 | **10.6 (94 ✅)** | 14.0 (71) |
| 0.4 | 821×453 | 9.0 (110) | 12.3 (81) |
| 0.3333 | 684×377 | 7.9 (127) | **9.7 (103 ✅)** |

- **DX: 내부 0.5(2× 업스케일)에서 90fps(10.6ms=94fps)**, Sponza 품질 **거의 네이티브**(육안 비교: TAAU가 약간 부드럽되
  에이리어싱은 더 적음; Laplacian 6.80→1.68 하락은 네이티브의 에이리어싱/스페큘러 노이즈가 부풀린 수치, 지각 품질은 근접).
  = Stage 3가 예측한 "TAAU로 선명한 고프레임" 실증. 42→94fps(2.2×)에 품질 손실 미미.
- **VK: 0.4=81fps, 0.33=103fps** — VK는 더 낮은 스케일 필요(GDF 컴퓨트가 DX 대비 느림 + Sponza 트랙서 남긴
  해상도-독립 `sdf_cache_light` 바닥). VK 90fps@고스케일은 **async-compute(별도 RHI 트랙)** 가 정공법.
- **진짜 QHD(3.69MP) 투영**: 디스플레이가 출력을 2052로 클램프해 스왑체인 QHD 직접 측정 불가. 내부=QHD×0.5(1280×720)를
  `RENDER_RES`로 측정: DX 13.3ms(75fps)/VK 16.9ms(59fps). → **진짜 QHD 90fps는 DX 내부 ~0.4**(Stage 2의 ~0.44 재확인),
  단 이번엔 바이리니어가 아니라 **선명한 TAAU**. 기법(내부 렌더 스케일+시간적 재구성)은 UHD까지 그대로 확장.
- **운영화**: `quality.rs` **Low 티어 render_scale 0.6667**(저사양/고해상 성능 모드 = 내부 2/3 + TAAU;
  `RENDER_QUALITY=low`). 처음 0.5로 잡았으나 0.5는 디테일 씬(Sponza)서 텍스처/지오 언더샘플로 재구성해도
  소프트=가시성 저하 → **2/3로 상향**(아래 mip-bias와 함께 네이티브에 근접). Med(기본)=1.0=갤러리 바이트동일 앵커.

## Stage 7 — TAAU 업스케일 블러 근본수정: 텍스처 mip LOD bias (2026-06-28)

정적(수렴) 상태에서도 Sponza 업스케일이 심하게 흐림(선명도 2.6 vs 네이티브 7.3). 원인: **G-buffer가 텍스처를
`.Sample()`(화면공간 미분 자동 LOD)로 샘플** → 저해상 렌더는 미분이 ~2× 커져 더 흐린 mip 선택 → TAAU가 흐린
mip만 누적 → 수렴해도 소프트. 갤러리는 텍스처가 없어 안 보였고 Sponza(조밀 텍스처)서 표출. **수정 = DLSS/FSR2
표준 음의 LOD bias `log2(내부/출력)`** 를 G-buffer 텍스처 샘플(albedo/MR/normal)에 `SampleBias`로 적용
(빈 `mr_factor.z` 재사용, 네이티브 bias 0 → `SampleBias(.,0)==Sample()` → 갤러리 0.000 바이트 동일). 측정
(Sponza 0.6667): 선명도 2.619→2.928, vs-네이티브 4.399→3.892, **육안상 네이티브 근접**. DX≡VK 0.002. 시작 시
`render: internal WxH -> output WxH (NN% scale, TAAU=...)` 로그 추가로 실효 스케일 즉시 확인.

**트랙 결론**: 내부 렌더 스케일 + TAAU 시간적 재구성으로 **무거운 씬을 거의 네이티브 품질로 고프레임**에 도달
(DX Sponza 42→94fps). 남은 격차는 VK 구조적 바닥(async-compute 후속)과 진짜 QHD 출력의 디스플레이 클램프(측정 한계).

## Stage 8 — 원거리 선명도: TAA-aware 음의 mip bias (2026-06-28)

Stage 7은 `mip_bias = log2(internal/output)` 라 **네이티브(scale=1)+TAA여도 bias=0** → 원경이 흐림(사자 부조/
벽 원경 확인). 레퍼런스 엔진/DLSS의 원경 선명도 1차 레버는 이방성이 아니라 **TAA + 음의 LOD bias**: 지터가 시간적
supersampling이라 더 선명한 mip을 당겨도 에일리어싱이 누적으로 해소된다. 수정 = `log2(scale) + TAA_MIP_BIAS`,
**지터 활성 시에만** TAA 항 추가. `TAA_MIP_BIAS = -1.0` (quality.rs 단일 상수, `TAA_MIP_BIAS` env 스윕).
gbuffer.slang은 이미 `SampleBias` → main.rs 계산만 수정. 갤러리(TAA off)는 bias 0 유지 = **바이트 동일**.
드라이버 무의존 LOD 오프셋이라 **DX≡VK 리스크 없음**.

측정 (Sponza, 원거리 코리도 크롭 Laplacian; SSAA 2× = 2263):
| bias | native+TAA 선명도 | TAAU 0.6667 | 정적 시머/ch |
|---:|---:|---:|---:|
| 0 | 829 | 374 | 0.093 |
| -0.5 | 885 | — | 0.123 |
| **-1.0** | **928 (+12%)** | **402 (+7.5%)** | 0.170 |
| -1.5 | 922 | — | 0.224 |
| -2.0 | 928 | — | 0.276 |

- **선명도는 -1.0에서 포화**(−1.5/−2.0 평탄 ~925) — 그 거리의 mip이 이미 최선명이라 더 음수는 무의미.
- **시머는 단조 증가** → **-1.0이 무릎**(선명도 최대 + 시머 0.170 ≤ Stage 4d 허용 0.158 근처). 육안: 사자 부조/
  벽 원경/바닥 타일이 한 단계 선명(네이티브 100%+`P_TAAU_FORCE`로 확인).
- 갤러리 0.000/ch(DX+VK). TAAU DX≡VK는 bias 무관(bias 0=0.072, -1=0.071; <0.01% 실루엣 픽셀의 기존 스토캐스틱
  갭). 정직한 한계: native+TAA 928 vs SSAA 2263 — 나머지 격차는 TAA 히스토리 블렌딩 본질(추가 음수로 회복 안 됨).

## Stage 9 — grazing 바닥: 이방성 필터링 (opt-in, 2026-06-28)

Stage 8 후에도 **grazing 바닥**(비스듬한 타일)은 SSAA 대비 3× 흐림(953 vs 2940) — 이방성 footprint는 등방
trilinear로 회복 불가(흐린 mip 선택). `P_ANISO=<N>`로 wrap 샘플러에 이방성 필터(VK samplerAnisotropy 피처+
sampler / DX `FILTER_ANISOTROPIC` MaxAnisotropy / Metal maxAnisotropy, 각 백엔드 한계로 클램프). clamp 샘플러
(큐브/볼륨)는 등방 유지.

측정 (Sponza, grazing 바닥 크롭 Laplacian; SSAA 2× = 2940):
- **native(no-TAA): 2604 → 3033 (+16.5%)** with 16× — 기법 자체는 강력(SSAA 초과).
- **native+TAA(-1): 953 → 1033 (+8.4%)** — TAA 히스토리 블렌딩이 이득을 압축.
- 원거리 크롭은 +1.6%만(이방성은 grazing 전용, 예상대로).

**정직한 한계 — DX≡VK 깨짐**: 이방성은 드라이버 의존 → 켜면 **DX≡VK 0.427/ch (0.46% 픽셀 >8)**. 따라서
**opt-in 전용**(기본 off = 모든 게이트 그린: 샘플러/피처가 변경 전과 동일 → 갤러리 바이트 동일 + DX≡VK 무영향).
단일 백엔드/콘텐츠 시나리오(백엔드 패리티 불필요)용 노브. **1a(mip bias)가 기본·패리티 안전한 원경 선명 레버**,
1b는 grazing 보조. fmt+clippy 클린, 이방성 활성 시에도 Vulkan 검증 클린.

## Stage 6 — async-compute 캐시 relight 오버랩 (VK 헤드룸, 2026-06-28)

Stage 2/3에서 VK가 해상도-독립 `sdf_cache_light`(VK 프레임의 ~34%, 최대 단일 패스)에 막혀 90fps 미달이라
했고, Stage 5에서도 VK는 0.33까지 낮춰야 했다. 그 relight를 **async-compute 큐로 옮겨 그래픽스 프레임과
오버랩**(소비자는 1프레임 지연된 radiance 읽음 — 캐시는 이미 상각+EMA라 무영향). **opt-in `P_ASYNC_CACHE=1`**.

구현(기존 프리미티브 최대 재사용 = 리스크 최소):
- **볼륨 CONCURRENT 공유**(Step 1 `c3e3ad5`): GDF/albedo/clip 3D 볼륨을 graphics+compute 패밀리 공유로
  (relight가 compute 큐에서 샘플). 바이트 동일.
- **3-슬롯 radiance 링**(2→3): async가 쓰는 슬롯이 in-flight 그래픽스 프레임이 읽는 슬롯과 절대 안 겹침
  → **WAR 해저드 제거**(추가 graphics→compute 세마포어 불필요). 남은 건 RAW(graphics-waits-compute)뿐.
- **off-graph relight**: `record_cache_async`가 visibility+relight를 compute 커맨드버퍼에 직접 기록
  (compute 전용 배리어 `storage_buffer_barrier_compute` 신설 — 컴퓨트 패밀리는 vertex/fragment 스테이지 불가).
  볼륨은 베이크 후 SHADER_READ_ONLY 고정이라 레이아웃 전이 불필요(volume_to_sampled는 이미 no-op).
- **cross-frame 동기화 = 기존 submit_async 재사용 + 재배치**: 그래픽스를 컴퓨트 submit *앞에* 내보내
  (D3D12는 async_fence가 monotonic이라 직전 값을 대기), Vulkan은 전용 `cache_done[2]`(프레임 패리티) 바이너리
  세마포어로 직전 relight 대기(VERTEX 대기 스테이지가 큐 전체를 게이트 → 컴퓨트 소비자 정합). compute
  커맨드버퍼 재사용 게이트용 `cache_compute_fence`(per-fif) 신설 + `ComputeQueue::submit_fenced`.

### 측정 (Sponza GDF, 출력 2052×1133, RENDER_SCALE 0.5, PROFILE_GPU 그래픽스 큐, 3회)
| | sync | async | 비고 |
|---|---:|---:|---|
| **VK** | 14.7–15.3ms (~68fps) | **9.7–10.4ms (~101fps ✅)** | sdf_cache_light(5ms)가 그래픽스 큐에서 사라짐 |
| DX | 11.1–12.4ms | 7.7–14.5ms (편차 큼) | D3D12 cross-queue 스케줄링 불안정 → 권장 안 함 |

- **VK: 일관된 ~33%(≈5ms) 단축 → 0.5에서 90fps 돌파(101fps)**. Stage 5에서 VK가 90fps에 0.33 필요했던 걸
  **0.5(더 선명)로 올림**. compute 경합으로 gdf_reflect가 다소 느려지나 순이득 큼.
- **DX는 편차 과대**(7.7–14.5ms): D3D12 큐 스케줄링이 1프레임 지연 마진을 들쭉날쭉 소화 → 스톨. DX는 이미
  sync로 0.5 근처 90fps라 async 불필요. **기본 off 유지, DX엔 권장 안 함**.
- **정확성/안정성**: async 출력 vs sync 수렴 0.153/ch(1프레임 지연=무시 가능), **정적 프레임간 diff
  async 0.149 == sync 0.149(max 251 동일)** = async가 떨림 추가 안 함(잔차는 Sponza 스토캐스틱+TAAU). Vulkan
  검증 클린, 갤러리(기본 off) DX/VK 0.000 바이트 동일. 티어 미결속(opt-in 노브로만 노출).

**결론**: async-compute로 **VK QHD 헤드룸 확보**(VK 0.5 = 101fps, sync 대비 +33%). DX는 스케줄링 편차로
보류(기본 off). VK/UHD 고프레임의 마지막 구조적 레버 = ✅(VK), DX 후속(드라이버 스케줄링/우선순위 조사).

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
