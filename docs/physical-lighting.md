# 물리기반 라이팅 (photometric units) — 정리

엔진의 라이팅을 실제 광도측정(photometric) 단위로 정리한다. 콘텐츠 씬은 물리 단위·물리 카메라 노출을 쓰고,
**갤러리(합성 회귀 씬)는 레거시 임의값 그대로** 유지해 바이트 동일 앵커를 깨지 않는다.

## 단위 (단일 노출이 전부를 매핑)

라이팅 패스는 마지막에 **한 번** `color = (ambient + lo) · exposure` 로 노출을 적용한다(env/GI/반사는 raw
radiance로 저장 후 끝에서 1회 노출). 따라서 모든 광원이 같은 광도측정 단위면 노출 하나로 전부 매핑된다.

| 항목 | 단위 | 코드 | 기본(콘텐츠) | 갤러리(앵커) |
|---|---|---|---|---|
| 디렉셔널 sun | **illuminance, lux** | `sun_color.a` | 100,000 (`SUN_LUX`) | 3.0 |
| 점광원 | **luminous intensity, candela** (`E=I/d²`) | `point_color.a` | glTF candela | 임의 |
| 노출 | **EV100** → `exposure=1/(1.2·2^EV100)` | `ambient.a` | EV100 14 (`EV100`) | 0.6 raw |
| sky 게인 | inscatter→radiance, sun:sky 비율 | `sky.slang` `sky_gain` | 6.0 (`SKY_GAIN`) | 6.0 |

- 맑은 정오 직사광 ≈ 100,000 lx, sunny-16 ≈ EV15. 광량계가 읽을 값이 sun lux, 카메라가 돌릴 값이 EV100.
- 점광원: glTF KHR_lights_punctual은 point/spot를 candela로 저작 → 그대로 사용. 루멘 Φ는 `I=Φ/4π`.

## 실내 조명 — sun 각도 · AO · 멀티바운스 (`a53d0d8` 외)

Sponza 실내가 검게 타던 문제. 셋 다 콘텐츠 한정(갤러리 앵커 0.000 불변):

- **sun 각도**: 좁은 nave(~12m 벽)는 저각 sun이 벽 그림자에 완전히 가려 바닥에 직사광이 **전혀** 안 들어온다
  (직사광-only 디버그뷰 ~2 luma; overhead 55°도 동일). 콘텐츠 기본을 **고각 `[0.3,0.9,0.2]`(~68°)** 로 — 벽을
  넘어 바닥에 직사광(22.6)·기둥 raking 그림자. 지붕 덮인 측랑은 실제처럼 간접광으로 남는다. `SUN_DIR` override.
- **AO exp falloff**: `gdf_ao`가 `saturate(1-k·occ)`라 오목 코너서 **정확히 0**(가시 recess도 하늘 대부분이
  보이는데 순검정). **`exp(-k·occ)`**(Beer-Lambert) 로 — 0에 점근하되 도달 안 함 → 코너가 soft AO.
- **GI 멀티바운스 π**: `gi_volume`이 저장된 평균-radiance `indirect`를 (irradiance인) direct 항과 같은
  `albedo/PI` BRDF에 묶어 매 바운스 π배 약했다. `albedo*indirect`로(저장값이 radiance라 ×π 후 BRDF의 /π와 상쇄)
  — surface-cache 관례와 일치. 깊은 nave처럼 멀티바운스 의존 영역이 ~10× 어둡던 원인.

측정: 실내 전체 lit 27→44, recess 검정 해소. DX≡VK 0.005, clippy 클린.

## env 큐브 clamp (의도된 설계, UE식)

env 캡처 큐브는 Rgba16Float(max ~65504). lux sun이 `sky.slang`의 baked sun 디스크를 +inf로 overflow시켜
IBL convolution을 오염시키므로 env sky를 60000으로 clamp한다. **디스크는 반사/irradiance용이고, 실제 sun
에너지는 analytic 디렉셔널 라이트가 운반**(UE도 동일). prefilter(저-roughness)가 디스크를 직접 샘플해도
fp16을 안 넘기게 하는 안전장치 — fp32 env로 바꿔도 prefilter가 다시 넘치므로 clamp가 올바른 해법.

## sky:sun 비율 — 측정 기록

`sky_gain`을 6→3으로 낮추면(직사광이 sky를 더 지배 = 더 "물리적") **개방형 실내가 어두워진다**(Sponza nave
mean luma 26.9→16.0, 외부 124→102). Sponza 아트리움은 하늘에 열려 있어 강한 skylight를 실제로 받으므로 sky를
줄이는 게 이 씬엔 부적절. 그래서 기본 6.0 유지 + `SKY_GAIN` 노브로 노출 — 폐쇄형 씬은 낮춰서 sun-dominant로
가고 실내는 멀티바운스 GI(DDGI [[gi-radiance-cache]])로 채운다.

## 노브 (env / 기본=콘텐츠)

- `SUN_LUX` (또는 `SUN_INTENSITY`) — 디렉셔널 sun illuminance, lux. 기본 100,000.
- `EV100` — 노출 stop. 기본 14. `EXPOSURE`는 raw 노출 배수 직접 override.
- `SKY_GAIN` — sun:sky 비율. 기본 6.0.
- 갤러리는 위 모두 미설정 시 레거시(3.0 / 0.6 / 6.0)로 바이트 동일.

## Auto-exposure (구현됨, `AUTO_EXPOSURE=1`)

물리 카메라 적응 노출 — 고정 EV100 대신 매 프레임 lit HDR을 측광해 노출을 적응시킨다. **GPU-only**(CPU 리드백/
RHI 변경 없음): `auto_exposure.slang`가 단일 16×16 그룹으로 HDR을 stratified 샘플 → log-평균(기하평균) 휘도를
groupshared 리덕션 → thread0이 시간평활(`1-exp(-dt·speed)`) 노출을 1-요소 storage 버퍼에 기록. 다음 프레임
`pbr.slang scene_exposure()`가 그 버퍼를 읽는다(sentinel 인덱스면 정적 `globals.ambient.a` = **auto off 시
바이트 동일**). 프레임 간 캐리는 in-order 큐 제출 순서로 정렬(기존 temporal 자원과 동일). HDR은 직전 노출이 baked
되어 있어 측광값을 de-expose(÷prev) 후 `target=key/L_scene`(key=0.18 middle-grey)로 계산.
- 게이트: `AUTO_EXPOSURE` opt-in, 갤러리 강제 off(앵커). ImGui "Auto exposure" 토글.
- 검증: 갤러리 0.000·콘텐츠 non-auto 0.000(sentinel 경로 bit-identical). EV100=20(거의 검정 luma 0.3)+auto →
  적정 노출 luma 125.7로 적응. DX≡VK auto-on 0.048byte(≈0.00019/ch, 게이트 통과 — 노출 피드백이 기존 stochastic
  GI gap을 증폭).
- **v1 한계**: 반사 history de-exposure(`lit_history` inv_exposure)·일부 trace 톤맵은 정적 `self.exposure`를
  계속 사용 → auto가 EV100 기본에서 크게 벗어나면 반사 radiance 스케일이 약간 어긋남(2차 효과). 완전 일관화는
  그 경로들에도 노출 버퍼를 배선하는 후속.
- 측광 개선 후속: 중앙가중/히스토그램(현재 256-tap 평균은 고DR 씬에서 밝은 부분 clip), key/speed UI 노출.

## 차후

- env fp32/pre-exposure로 디스크 clamp 제거(시각 차 거의 없음, 우선순위 낮음).
- 점광원 IES 프로파일·면광원(area light), 물리 카메라(조리개/셔터/ISO) UI.
- auto-exposure 완전 일관화(반사 history·trace 톤맵에 노출 버퍼 배선) + 중앙가중/히스토그램 측광.
