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

## 차후 (auto-exposure 등)

- **Auto-exposure** (히스토그램 적응): 현재 GPU reduction/리드백 인프라 없음 → 별도 트랙. HDR 평균 log-luminance
  compute reduction → 적응 EV(시간 평활) → 노출 피드. opt-in, 갤러리 강제 고정 EV. pbr 노출 적용 지점을
  버퍼-구동으로 바꾸거나 CPU 리드백(1~2프레임 지연) 중 택.
- env fp32/pre-exposure로 디스크 clamp 제거(시각 차 거의 없음, 우선순위 낮음).
- 점광원 IES 프로파일·면광원(area light), 물리 카메라(조리개/셔터/ISO) UI.
