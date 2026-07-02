# 대기/포그 합성 슬롯 (PR-4)

상위: [render-pipeline-reference.md](render-pipeline-reference.md) §1.6 · §2 표 #10·#11 · §3 PR-4.

## 무엇을 넣었나

`docs/render-pipeline-reference.md`가 지적한 구조적 공백 — **불투명 라이팅+반사가 끝난 뒤, (미래)
투명 패스와 post 체인 앞에 sky/fog를 합성할 자리가 아예 없었던 문제** — 를 렌더 그래프에 정식
슬롯으로 확보했다. 슬롯 자체는 그래프 배선(패스 지점)이고, 오늘은 그 슬롯에 하나의 opt-in 기능
(analytic exponential height fog)을 채워 슬롯이 실제로 동작함을 증명한다.

- **슬롯 위치**: `apps/sandbox/src/main.rs`의 프레임 그래프에서 `record_compute_post`(있으면 그
  뒤, 없으면 `record_lighting`이 쓴 `hdr` 바로 뒤) — 즉 불투명 씬 컬러가 최종 완성된 시점 — 와
  TAAU/tonemap 사이. `let fog_src = hdr_post.unwrap_or(hdr);` 로 "지금까지 완성된 불투명 HDR"을
  잡고, `self.height_fog`가 켜져 있을 때만 `hdr_fog`라는 새 컬러 타겟을 만들어
  `AtmosphereSystem::record_fog`를 그래프에 추가한다. 이후 TAAU의 `main_lit`과 `tonemap_src`
  `.or()` 체인이 `fog_out`을 우선 소비하도록 재배선했다 — 켜져 있으면 포그가 낀 컬러가, 꺼져 있으면
  기존 `hdr`/`hdr_post`가 그대로 흘러간다.
- **디폴트 = OFF, 비용 0**: `P_HEIGHT_FOG` 미설정(또는 `0`/`false`/`off`)이면 `self.height_fog`가
  `false`이고, `if self.height_fog { graph.add_pass(...) }` 가드 자체가 패스를 그래프에 추가하지
  않는다 — `crates/render`의 데드패스 컬링과 무관하게, 애초에 스케줄되지 않는다. 골든 이미지
  (`--screenshot-clean`) sha256이 이 작업 전후로 바이트 동일함을 검증했다(아래 §검증).
- **구조 재사용**: 새 코드는 없고, 기존 `record_compute_post`(hdr → hdr_post)/`record_tonemap`
  (src → backbuffer) 이 쓰는 "read old, write new, rethread the ResourceId forward" 컨벤션을
  그대로 따른다. 슬롯 자체는 기능 불특정이므로, 다음 단계(에어리얼 퍼스펙티브/light shaft/
  volumetric fog)는 같은 호출부에서 `PushConstants`+`fsMain`만 확장하면 된다 — 그래프 재배선 불필요.

## Height fog: 왜 이 수식인가 (analytic exponential height fog)

높이에 따라 지수적으로 옅어지는 참여매질(participating medium)의 밀도를:

```
d(y) = a * exp(-b * y)      // a = 지표(y=0) 밀도, b = 1/특성감쇠고도
```

로 모델링하면, 카메라 레이 `o + t*rd` 를 따라가는 광학적 깊이(optical depth) 적분은 **닫힌 형태
(closed form)로 해석적으로 풀린다** — 레이마칭(수십 스텝) 없이 픽셀당 한 번의 `exp` 계산으로 정확한
결과를 얻는 실시간 그래픽스의 표준 기법이다 (I. Quilez, "fog", 2010,
https://iquilezles.org/articles/fog/; 상용 엔진들의 "exponential height fog"가 쓰는 것과 동일한
닫힌 형태). 직접 적분으로 도출/검증한 결과(`sympy`로 재확인):

```
b*rd.y != 0:  depth(T) = (a/b) * exp(-b*o.y) * (1 - exp(-b*rd.y*T)) / rd.y
b*rd.y  = 0:  depth(T) = a * exp(-b*o.y) * T        // 수평 레이의 극한(제거 가능한 특이점) —
                                                     // rd.y -> 0이면 레이를 따라 밀도가 상수이므로
                                                     // depth = 밀도 * 거리, 직접 검증됨.
```

`b == 0`(균일 밀도, 고도 무관)도 두 번째 분기가 자연스럽게 커버한다(`by == 0`이면 `rd.y`에 상관없이
레이 전체에서 밀도가 상수). 셰이더는 `abs(b*rd.y) < 1e-5`로 두 분기를 나눈다
(`atmosphere.slang::height_fog_optical_depth`).

투과율(transmittance) = `exp(-depth)`, 포그 팩터 = `1 - transmittance`(Beer-Lambert)로 안개색과
표면색을 선형 보간한다.

## Inscatter 색: 단일 소스 (중복 정의 금지) — 그리고 첫 시도의 실패

포그의 inscatter 색은 **새 상수가 아니라 기존 `sky_common.slang`의 `procedural_sky()`를 그대로
재사용**한다 — env-cube 캡처(`sky.slang`)와 경로 추적기(`rt_common.slang`)가 이미 쓰는 동일한
Rayleigh+Mie+ozone 단일 산란 대기 모델이다.

**첫 시도(실패)**: 포그 픽셀마다 카메라→월드픽셀의 실제 레이 방향(`rd`)으로 `procedural_sky(rd, ...)`
를 호출했다 — "안개가 태양/하늘과 같은 방향으로 물든다"는 발상 자체는 그럴듯했지만, 두 가지로
깨졌다: (1) `sun_intensity`가 물리 단위(예: `sponza_intel.level`의 100000 lux)라 `procedural_sky`가
반환하는 raw radiance가 카메라 노출 전이라는 걸 놓쳐 합성 결과가 전부 흰색으로 날아갔다(§노출 버그
참고), (2) 노출을 고치고 나서도 실내의 짧고 가파른 레이를 대기 적분에 그대로 태우니 **대기 모델이
"진짜 하늘까지 닿는 바깥쪽 레이"를 위해 갖고 있는 지평선 부근의 급격한 밝기 변화(bright horizon
band)가 실내 지오메트리 위에 뚜렷한 수평 밴딩 아티팩트로 그대로 드러났다** — 물리적으로는
"정확"하지만 짧은 실내 레이에 대기 전체 적분을 재적용하는 것 자체가 부적절한 근사였다.

**최종 구현**: WebSearch로 상용 엔진들의 exponential height fog 문서를 확인한 결과, inscatter
색은 레이 방향에 따라 매 픽셀 다시 계산하는 게 아니라 **프레임당 하나의 고정된(fixed) ambient/
directional 색**으로 쓰는 것이 표준이었다("Fog Inscattering Color" + 별도 태양 방향 항, 소스: 하단
Sources). 이를 반영해 `procedural_sky`를 **레이 방향(`rd`) 대신 고정된 zenith 방향
`(0,1,0)`**으로 한 번만 평가하도록 바꿨다 — 여전히 단일 소스(같은 함수, 같은 태양 파라미터)이지만,
매 픽셀 레이 방향에 따라 급변하지 않는 안정적인 안개색이 된다. 지평선 밴딩이 사라지고, 일몰/일출에
따라 안개색이 여전히 변한다(zenith 자체가 태양 고도에 반응하므로).

Rust 쪽 값도 새로 만들지 않고 기존 필드를 그대로 넘긴다: `sun_dir`/`sun_intensity`(프레임별 해석된
가리키는 방향/조도, `Globals` 조립과 동일 소스), `self.sky_wb`(`sky.slang`이 쓰는 화이트밸런스),
`fog_inscatter_gain`은 기본값이 `self.sky_gain`(env-cube 캡처의 sun:sky 비율)과 동일 — 즉
`P_FOG_INSCATTER_GAIN`을 안 주면 하늘과 정확히 같은 gain을 쓴다.

## 노출(exposure) 버그와 수정

`procedural_sky`는 **raw(노출 전) radiance**를 반환한다 — `pbr.slang`의 하늘 배경 miss 경로가
`sky * exposure`로 처리하는 것과 동일한 컨벤션이다. 반면 `hdr`(포그가 읽는 입력)은 `record_lighting`
단계에서 **이미 노출이 적용된** 값이다. 이 사실을 놓치고 `inscatter`를 노출 없이 그대로
`hdr.rgb`와 lerp했더니, `sun_intensity=100000`(sponza_intel) 같은 물리 스케일 값에서 카메라 노출
이전(수만~수십만 단위) radiance가 노출된 씬 컬러(0~1 부근)를 완전히 압도해 화면이 통짜 흰색으로
날아갔다(실측: 평균 채널값 78→255, 최솟값 254). 고쳐서 `inscatter *= exposure`(포그 호출부가
`self.exposure`, 즉 `record_lighting`이 쓰는 것과 같은 EV100 유도 스칼라를 그대로 넘김)를 추가하니
정상 범위로 돌아왔다. **자동노출(auto-exposure) 사용 시엔 적응된 값이 아니라 정적 EV100 노출을
쓴다** — 포그는 저주파 ambient 항이라 노출 한 프레임 지연이 시각적으로 문제되지 않는다는 판단의
의도적 단순화(문서화된 근사).

배경(하늘) 픽셀은 건드리지 않는다: `pbr.slang`의 miss 경로에서 이미 전체 절차적 하늘을 보여주므로,
그 위에 포그를 또 합성하면 대기가 이중으로 낀다. G-buffer `position.w < 0.5`(지오메트리 없음)일 때
`fsMain`이 조기 반환한다.

## 파라미터 (opt-in, 확장 가능한 단일 상수 블록)

| Env | 기본값 | 의미 |
|---|---|---|
| `P_HEIGHT_FOG` | off | 슬롯 자체를 켠다(`quality::env_bool`) |
| `P_FOG_DENSITY` | `0.15 / scene_radius` | 지표 밀도 `a` (1/world-unit) — 씬 스케일에 비례한 디폴트라 갤러리(단위 반경)와 Sponza(훨씬 큰 씬) 양쪽에서 합리적으로 보이도록 정규화 |
| `P_FOG_HEIGHT_FALLOFF` | `1.0 / scene_radius` | 감쇠율 `b` (1/world-unit) |
| `P_FOG_INSCATTER_GAIN` | `self.sky_gain` | `procedural_sky`에 넘기는 sun:sky 이득(하늘과 단일 소스) |

모든 파라미터는 `push.rs::atmosphere_push`가 채우는 **하나의 push-constant 블록**에 모여 있다
(CLAUDE.md 엔지니어링 원칙 3: "품질 파라미터는 한 곳에" — 추후 `RenderQuality{low,med,high}` 티어로
쪼갤 때 이 블록 하나만 손대면 된다). `Globals` UBO는 건드리지 않았다(464/512바이트 사용 중이라 여유는
있었지만, 포그 전용 값을 push constant로 격리하는 편이 이 패스의 opt-in 특성과 더 잘 맞는다).

## Push-constant 레이아웃 (Metal/HLSL/SPIR-V 안전)

`atmosphere.slang`의 `PushConstants`는 **모든 행을 완전한 float4로** 선언한다 — `float3` 뒤에
스칼라를 바로 붙이면 HLSL/SPIR-V에서는 +12 오프셋에 오지만 MSL(Metal)에서는 +16으로 패딩되는,
이 저장소에 이미 문서화된 함정(`push.rs`의 `gdf_gi_push`/`ground_albedo` 참고)을 피하기 위해서다.
`camera_pos_density`(xyz+w), `sun_dir_intensity`(xyz+w), `sky_wb_gain`(xyz+w),
`falloff_exposure_pad`(x=height_falloff, y=exposure, zw 미사용) — 4개의 uint(hdr/position 인덱스 +
예약 2개) + 4개의 float4 = 80바이트.

## 검증

- `cargo clippy --all-targets -- -D warnings` 클린, `cargo fmt` 적용.
- 디폴트(OFF) 골든 앵커 바이트 동일: `--screenshot-clean` sha256 = `af70c1a5c8db49661d2c7926140c
  1309c28fda04c82cc1ab8aa6638d588b2b74` — 노출 버그 수정 전후 모두 재확인, 불변.
- `P_HEIGHT_FOG=1`로 갤러리 + `sponza_intel`(`LEVEL=sponza_intel EV100=11 WARMUP_FRAMES=64
  CAM_EYE=-14,2,0 CAM_TARGET=14,2,0`) 스크린샷을 찍어 거리에 따라 포그가 짙어지고 색이 하늘과
  맞는지 육안 확인 — 복도 안쪽(카메라에서 먼 아치/문)이 가까운 기둥보다 뚜렷하게 옅어지고 푸르스름한
  톤으로 물듦, 노출 수정 후 밴딩 없이 매끈한 거리 그라디언트. 갤러리는 씬이 작아 효과가 은은하지만
  (평균 채널 diff 3.75/255, 변경 픽셀 52%, 하늘 영역은 불변) 존재는 확인됨.
- **Metal(macOS M3)만 검증**. SPIR-V/DXIL은 이 macOS 박스에 Vulkan SDK/DXC가 없어 기존 셰이더들과
  동일하게 `None`(빌드는 성공, 바이트코드만 비어 있음) — 다른 모든 기존 셰이더와 같은 상태이지 이
  변경의 회귀가 아니다. **DX≡VK parity pending Windows verification.**

## Sources

- [I. Quilez, "fog" (2010)](https://iquilezles.org/articles/fog/) — analytic height-fog closed-form derivation
- [오픈소스 엔진 exponential height fog 문서](https://docs.flaxengine.com/manual/graphics/fog-effects/exponential-height-fog.html) — 고정(fixed) inscattering color + 별도 방향성(directional) inscattering color 컨벤션 확인(참고로 검토한 다른 상용 레퍼런스 엔진 문서의 URL은 DreamCoast 상표명 금지 규칙에 따라 생략)
