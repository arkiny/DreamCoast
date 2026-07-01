# Depth Pre-pass (pipeline rebaseline PR-1)

상위: [render-pipeline-reference.md](render-pipeline-reference.md) §2 표 #1 · §3 PR-1.

불투명 **depth-only pre-pass**를 G-buffer fill **앞**에 추가한다. 레퍼런스 디퍼드
파이프라인의 canonical 첫 지오메트리 스테이지(§1.1-1)로, 이후 velocity·HZB occlusion·
hi-Z SSR·화면공간 트레이싱 정확도의 **공통 토대**다. Opt-in seam `DEPTH_PREPASS=1`
(디폴트 off = 기존 경로, 바이트 동일).

## 무엇이 바뀌나

프레임 그래프 (pre-pass ON):

```
shadow → PREPASS(depth-only) → G-buffer(Equal-test, write-off) → decals → AO/GI/SSR/reflect → lighting → …
```

- **PREPASS**: 불투명 씬(데칼 제외) + 그라운드를 depth-only로 래스터해 `g_depth`를 채운다.
  이것이 이 프레임의 **첫 depth writer**이므로 그래프의 per-depth first-writer 규칙에 따라
  depth를 CLEAR한다.
- **G-buffer**: pre-pass가 이미 depth를 만들었으므로 depth-test `Equal` + depth-write **off**로
  전환 → 각 픽셀은 pre-pass 깊이와 일치하는 프래그먼트만 셰이딩(Early-Z 오버드로 제거).
  그래프가 같은 `g_depth`를 두 번째로 attach하므로 자동으로 LOAD(clear 아님).
- **화면공간 패스(AO/GI/SSR/reflect)**: 이미 `g_depth`를 read로 선언하고 있다. pre-pass가
  `g_depth`의 producer가 되면서(그래프 WAW: pre-pass → G-buffer, RAW: G-buffer/pre-pass →
  화면공간) 이들이 **명시적으로 pre-pass가 만든 완성 depth를 소비**하는 구조가 된다.

## 핵심: position invariance (Equal-test의 전제)

Equal depth-test는 pre-pass와 base pass가 클립공간 위치를 **비트 단위로 동일**하게 계산할
때만 성립한다. 부동소수 정밀도가 조금이라도 어긋나면 실루엣에서 z-fighting이 난다
(Interplay of Light, "To z-prepass or not to z-prepass"; MJP, "To Early-Z, or Not To
Early-Z").

**채택한 방식 — VS 단일 소스 재사용:** pre-pass 파이프라인은 `gbuffer.slang`의 기존 정점
셰이더(`vsMain` / `vsMainSkinned` / `vsMainMorphed`)를 **그대로** 쓴다. 프래그먼트만
depth-only `fsDepth`로 교체한다. 즉 클립 위치는 base pass와 *같은 셰이더의 같은 명령
시퀀스*(`o.clip = mul(pc.mvp, …)`)로, *같은 `mvp` push*를 받아 계산된다 → 컴파일된 VS가
동일하므로 깊이가 비트 단위로 같다. 별도의 pre-pass VS를 새로 쓰면 (같은 수식이어도)
컴파일러 스케줄링 차이로 미세하게 어긋날 수 있으므로, 새 VS를 만들지 않고 **재사용**하는
것이 가장 안전한 단일화다.

- **push 동일성**: `record_prepass`는 `record_gbuffer`와 동일한 `gbuffer_push`(mvp/model/
  skin/morph/텍스처 인덱스/cutoff)를 pack한다. 스태틱·스키닝(`vsMainSkinned`)·모프
  (`vsMainMorphed`) 전부 같은 push로 같은 정점 변형을 재현한다.
- **alpha-cutout**: `fsDepth`는 base pass(`fsMain`)와 동일한 `SampleBias`+`cutoff` 하드
  discard를 수행해 masked(foliage cutout) 지오메트리의 구멍을 pre-pass depth에도 똑같이
  뚫는다(안 그러면 base pass가 구멍에서 Equal-test에 실패해 셰이딩되지 않음). hashed
  alpha(cutoff<0)는 pre-pass에선 `|cutoff|` 하드 테스트로 처리한다 — pre-pass는 TAA와
  무관한 결정적 depth를 써야 하고, 여기서 확률적 discard를 하면 base pass의 hashed discard와
  싸워 다시 z-fighting이 나기 때문(그림자 패스가 같은 모호성을 해소하는 방식과 동일).

## 깊이 비교(depth compare) 백엔드 정합

새 필드 `GraphicsPipelineDesc::depth_compare: DepthCompare { Less, Equal }`를 추가:

| | Vulkan | D3D12 | Metal |
|---|---|---|---|
| `Less` (디폴트, 기존 전 파이프라인) | `LESS` | `LESS` | `LessEqual` (기존 매핑 유지 → 바이트 동일) |
| `Equal` (pre-pass base pass) | `EQUAL` | `EQUAL` | `Equal` |

`Less`의 백엔드별 매핑은 기존 코드 그대로 유지(Metal은 예전부터 LessEqual)해 디폴트 골든
앵커가 바뀌지 않게 했다. `Equal`은 세 백엔드 모두 EQUAL로 정합.

## 파이프라인 구성

`DeferredRenderer`가 불변으로 6개 파이프라인을 추가 생성(값싼 객체, pre-pass off여도 무해):

- `prepass_{,,skinned,morphed}_pipeline` — depth-only, VS = 기존 gbuffer VS, FS = `fsDepth`,
  `Less` + write-on, `Mesh` 레이아웃, 208B push.
- `gbuffer_equal_{,,skinned,morphed}_pipeline` — 기존 gbuffer VS+FS 그대로, `Equal` +
  write-off. pre-pass on일 때 base pass가 이 세트를 쓴다.

`record_gbuffer(…, prepass: bool)`가 `prepass`면 Equal 세트를, 아니면 기존 `Less` 세트를
바인딩한다.

## 검증 (Metal, macOS M3)

- **디폴트(OFF) 골든**: `--screenshot-clean` sha256 =
  `af70c1a5c8db49661d2c7926140c1309c28fda04c82cc1ab8aa6638d588b2b74` — 기존 앵커와 **바이트 동일**.
- **`DEPTH_PREPASS=1` 골든**: 같은 캡처가 **역시 바이트 동일** (동일 sha256). Equal-test가
  셰이딩을 바꾸지 않음을 증명 — position invariance가 비트 단위로 성립.
- **sponza_intel sanity**: OFF/ON 캡처 시각적 동일(수치는 커밋/보고 참조).
- clippy `-D warnings` 클린, `cargo fmt` 적용.

DX≡VK parity는 Windows 후속 검증(이 머신은 Metal 전용).

## 남은 것 / unblocks

pre-pass depth가 이제 명시적 producer이므로 후속 트랙이 이를 토대로 쌓인다: PR-8 HZB
occlusion 컬링(hi-Z 피라미드), hi-Z SSR 트레이스, 화면공간 트레이싱 정확도, GDF AO
depth-lifetime 근본 정리. Velocity(PR-2)와 함께 Phase 20(AA/포스트/투명)의 선결 인프라.

## 참고

- [To z-prepass or not to z-prepass — Interplay of Light](https://interplayoflight.wordpress.com/2020/12/21/to-z-prepass-or-not-to-z-prepass/)
- [To Early-Z, or Not To Early-Z — MJP](https://therealmjp.github.io/posts/to-earlyz-or-not-to-earlyz/)
