# Post-process 시퀀스 — 파이프라인 재정합 PR-5

상위: [render-pipeline-reference.md](render-pipeline-reference.md) (§1.8, §2 #13–#18, §3 PR-5).

이 문서는 DreamCoast의 **순서 있는 post-process 노드 시퀀스**(PR-5)를 설명한다. 재정합 이전의
파편적 경로(`hdr → (compute box-blur 데모) → atmosphere_fog → taau → tonemap`)를 레퍼런스
디퍼드 렌더러의 canonical post 순서에 맞춰 정리했다.

> 규칙(메모리): 서드파티 엔진/제품명은 쓰지 않는다. 기법은 일반명 또는 원저자(공개 논문/발표)로만
> 지칭한다.

---

## 1. 노드 순서 (레퍼런스 §1.8)

불투명 씬 컬러가 **라이팅 + 반사 + 대기/포그 슬롯(PR-4)** 까지 완성된 뒤:

```text
(opaque HDR)
  → [#13] motion-blur      (P_MOTION_BLUR, velocity 소비)   ── linear-HDR
  → [#14] TAA / upscale    (기존 taau.rs, jitter 리프로젝션) ── linear-HDR
  → [#15] auto-exposure    (기존 deferred.rs, 미터링)        ── (앞 프레임 값)
  → [#16] bloom            (P_BLOOM, dual-filter 피라미드)   ── linear-HDR
  → [#17] DoF (stub)       (P_DOF, 자리만)                   ── linear-HDR
  → [#18] tonemap + grading(ACES + ASC-CDL)                 ── HDR → display
```

- **모션 블러가 TAA 앞**: 모션블러된 컬러를 시간적으로 누적해야 스트릭이 프레임 간 안정된다.
- **블룸/DoF/그레이딩이 톤맵 앞**: 전부 linear-HDR에서 동작(톤맵 후 밴딩/고스팅 회피). 블룸은
  밝은 영역을 추출해 톤맵 입력에 additive로 합성하고, 그레이딩은 노출 후·필믹 커브 전의 linear HDR에
  적용한다.

각 노드는 렌더 그래프에 **"read old, write new, rethread"** 컨벤션으로 삽입된다(atmosphere.rs와
동일). 즉 현재 HDR 타깃을 읽어 **새 타깃**을 쓰고, 호출부가 그 새 타깃을 다음 노드로 넘긴다 —
씬 HDR 자체는 절대 in-place로 바뀌지 않는다.

**모든 노드는 opt-in이며 기본 OFF = 바이트 동일**(골든 앵커 `af70c1a5…`). 파이프라인은 항상
빌드되므로(3 백엔드가 셰이더를 컴파일) OFF여도 컴파일은 되지만, `record_*` 호출을 안 하면 그래프에
패스가 추가되지 않아 비용 0.

---

## 2. 노드별 상세

### #13 Motion blur — `crates/shader/shaders/motion_blur.slang`, `postfx::MotionBlurSystem`

per-pixel velocity-along 블러. PR-2 velocity 타깃(RG16F, NDC 모션)을 읽어 픽셀 중심 기준
`[-0.5, +0.5]·velocity` 구간을 N탭 평균한다. 카메라·오브젝트 모션 모두 화면 이동 방향으로 스트릭.

- **opt-in `P_MOTION_BLUR=1`** (전제: `P_VELOCITY=1`). velocity 타깃이 없으면 1회 로그 후 무시
  (하드 에러 아님 — 문서화된 degrade).
- **TAA 앞**, linear-HDR.
- 파라미터(단일 소스, `main.rs`의 `MOTION_BLUR_*` 상수): `SAMPLES=8`, `INTENSITY=1.0`
  (셔터 1프레임), `MAX_UV=0.05` (한 픽셀 스트릭 UV 길이 상한 — 초고속 오브젝트가 화면 전체로
  번지는 것을 방지).
- **후속(Phase 20)**: tile-max / neighbor-max 지배 속도 dilation (McGuire 2012) — 빠른
  전경 오브젝트가 느린 배경 위로 블러를 번지게 하는 확장. 현재는 단일-velocity "reconstruction
  lite"만 구현.

### #16 Bloom — `crates/shader/shaders/bloom.slang`, `postfx::BloomSystem`

progressive **dual-filter** 블룸. canonical 실시간 기법(Jimenez, "Next Generation Post
Processing in Call of Duty: Advanced Warfare", SIGGRAPH 2014)을 채택 —
[iryoku.com](https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/).

채택 근거: **13-tap 다운샘플**(내부 2×2 4탭 + 외부 3×3 9탭, 5개 박스 그룹) + **3×3 텐트
업샘플**은 분리형 가우시안 체인보다 훨씬 적은 패스로 둥글고 아티팩트 없는 넓은 글로우를 만든다.
mip0(가장 밝은 레벨) 빌드에 **부분 Karis 평균**(`1/(1+luma)` 가중)을 적용해 단일 픽셀
파이어플라이가 다운샘플을 지배하지 못하게 한다(고전적 블룸 파이어플라이 방지).

파이프라인:

```text
prefilter(bright-pass + Karis 13-tap ↓) → mip0
mip[k] = 13-tap box ↓ (mip[k-1])       (k=1..N-1)   ── 다운샘플 체인
up[k]  = 3×3 tent ↑ (up[k+1]) + mip[k]              ── 업샘플(additive, 새 타깃에)
반환 up[0] (mip0 해상도) → tonemap 입력에 intensity로 additive 합성
```

additive는 별도 blend 모드 없이 **셰이더 내에서** 처리(coarse 텐트 결과 + fine mip을 읽어 더한
뒤 새 타깃에 write). 3 백엔드 blend 상태를 안 건드려 parity 리스크 최소화.

- **opt-in `P_BLOOM=1`**.
- 파라미터(단일 소스, `postfx::BloomParams`): `THRESHOLD=1.0`(linear-HDR bright-pass knee,
  물리적으로 diffuse mid-grey 위 하이라이트만 블룸), `INTENSITY=0.06`(합성 스케일),
  `MIPS=5`(mip0=½ 렌더해상도 → ~1/32). `RenderQuality {low:4, high:6}` 티어로 분할 준비됨.

### #17 Depth of field — `crates/shader/shaders/dof.slang`, `postfx::DofSystem` (STUB)

자리 확보용 passthrough 노드(현재 정확한 identity 복사). 순서·시임을 미리 박아둬 실제 구현이
재배선 없이 들어간다.

- **opt-in `P_DOF=1`** (켜도 no-op).
- **후속(Phase 20)**: depth G-buffer + 카메라 포커스 거리/조리개로 CoC 계산 → near/far
  separable 또는 scatter-as-gather 보케 블러 → 합성. 같은 read-old/write-new 시임에서 이
  fragment를 CoC+게더 체인으로 교체(호출부는 depth 타깃을 추가로 스레드)하면 된다.

### #18 Tonemap + color grading — `crates/shader/shaders/post.slang`, `record_tonemap`

기존 Narkowicz ACES 필믹 톤맵 + sRGB 인코드에 **컬러 그레이딩 훅** 추가.

canonical 방식으로 **ASC CDL**(American Society of Cinematographers Color Decision List)
채택 — `out = (slope · in + offset)^power`, per-channel. Slope=Gain(하이라이트), Offset=Lift
(섀도), Power=Gamma(미드톤). 근거: 업계 표준·이식 가능한 최소 그레이딩 프리미티브(3D LUT보다
가벼우면서 lift/gamma/gain을 정확히 표현), 참고
[Pomfort ASC-CDL](https://pomfort.com/article/an-in-depth-look-at-asc-cdl-based-color-controls/).

노출 후·필믹 커브 전의 linear HDR에 적용. **블룸 additive 합성도 이 패스에서** 수행(bloom mip0을
읽어 `intensity`로 더한 뒤 노출·그레이딩·커브).

- **opt-in `P_GRADE=1`**. per-channel 파라미터: `P_CDL_SLOPE`/`P_CDL_OFFSET`/`P_CDL_POWER`
  (`"r,g,b"` 또는 단일 `"v"`), 기본 = 중립(slope 1, offset 0, power 1).
- **중립 = 바이트 동일**: `grade_on == 0`이면 그레이딩을 평가조차 안 하고, 중립 CDL을 적용해도
  `(1·x+0)^1 = x` 항등이라 결과 동일. 검증: `P_GRADE=1`(중립) 캡처 sha256 == 앵커.

---

## 3. 데이터 시임 요약

| 시임(env) | 노드 | 소비 리소스 | 생성 리소스 | OFF 동작 |
|---|---|---|---|---|
| `P_MOTION_BLUR` (+`P_VELOCITY`) | #13 motion-blur | opaque HDR + velocity | `motion_blur` HDR | opaque HDR 그대로 |
| — (기존) | #14 TAAU | post HDR + depth (+velocity) | `taau_out` | native, no upscale |
| `AUTO_EXPOSURE` (기존) | #15 exposure | lit HDR | exposure buf | 정적 EV100 |
| `P_BLOOM` | #16 bloom | main-lit HDR | `bloom_u0` (mip0) | 합성 안 함 |
| `P_DOF` | #17 DoF(stub) | main-lit HDR | `dof` HDR | main-lit 그대로 |
| `P_GRADE`, `P_CDL_*` | #18 grading | (톤맵 내부) | backbuffer | 항등 |

블룸/그레이딩은 **메인 라이트 경로에만** 적용(RT/디버그 viz 출력은 `.or()` 체인 우선순위로
tonemap_src를 차지하면 bloom=off·grade=off로 톤맵되어 캡처 불변).

---

## 4. 검증 (Metal, 이 M3 박스)

- clippy `-D warnings` 클린 + fmt.
- **골든 앵커 바이트 동일**(전부 OFF): sha256 `af70c1a5c8db49661d2c7926140c1309c28fda04c82cc1ab8aa6638d588b2b74` — 매치.
- **중립 그레이딩 바이트 동일**: `P_GRADE=1`(중립 CDL) 캡처 == 앵커.
- **블룸 시각 타당성**: `P_BLOOM=1` 갤러리 — 크롬/펄 스페큘러 하이라이트 + 하늘에 부드러운 글로우
  (앵커 대비 mean 0.021/ch, 2.1% 픽셀 영향 — 국소 집중 글로우, 전역 워시 아님).
- **모션 블러 시각 타당성**: 카메라 궤도(`CAPTURE_SEQ` step) + `P_VELOCITY=1 P_MOTION_BLUR=1` —
  큐브/구 엣지가 모션 방향으로 방향성 스트릭(ON vs OFF mean 0.83/ch, 6.8% 픽셀).
- **비중립 그레이딩**: warm teal-orange (`P_CDL_SLOPE=1.15,1.0,0.85` 등) — 하늘 앰버, 그림자
  쿨 — 명확한 톤 시프트.
- **PROFILE_GPU (2560×1440, M3)**: `motion_blur` ~0.44 ms · `bloom` 총 ~0.5 ms
  (mip0 prefilter ~0.28 ms 지배) · `tonemap`(블룸 합성+그레이딩 포함) ~0.29 ms.
- 콘텐츠 씬: `sponza_gdf_ao` SHA는 PR-5 적용 전/후 동일 → post 체인이 콘텐츠 씬에 무영향 확인.

**DX/VK parity pending Windows verification.** (푸시 상수 각 float 벡터는 float4 행으로 패킹해
HLSL/SPIR-V vs MSL 정렬 발산을 회피 — `push.rs`/셰이더 주석 참조.)

---

## 5. 제거된 데모

- `blur.slang` (Phase 5 분리형 가우시안 블룸 데모, Rust 소비자 없음) — 삭제, bloom.slang이 대체.
- `post_compute.slang` + `record_compute_post` + `P7_COMPUTE_POST` (Phase 7 3×3 박스 블러
  compute 데모, 블룸 아님) — 삭제. UI "Compute post (blur)" 체크박스도 제거, PR-5 노드 토글로 교체.
