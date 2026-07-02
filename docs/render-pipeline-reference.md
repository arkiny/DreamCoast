# 렌더 파이프라인 재정합 — 레퍼런스 디퍼드 렌더러 대비

상위: [ROADMAP.md](ROADMAP.md) · 관련: [commercial-engine-gap-analysis.md](commercial-engine-gap-analysis.md).

이 문서는 DreamCoast의 프레임 패스 구조를 **상용 디퍼드 렌더러의 정식 패스 순서**에 맞춰 재정합
(re-baseline)한 것이다. 레퍼런스 상용 엔진의 디퍼드 씬 렌더러 소스를 읽어 **범용 디퍼드 파이프라인의
canonical 패스 순서**를 추출하고(§1), DreamCoast 현 프레임 흐름을 그에 매핑해 정직한 스테이지별 갭을
표로 정리하며(§2), 마지막으로 의존순으로 정렬된 **"파이프라인 정합" 작업 트랙**(§3)을 제시한다.
개별 기능(TAA·다광원·볼류메트릭 등)의 *구현* 계획은 Phase 20+에서 다루고, 이 문서는 그 기능들이
올바른 자리에 들어가도록 **패스 구조(ordering/데이터 의존)** 자체를 바로잡는 데 집중한다.

> 규칙: 레퍼런스 엔진은 항상 "레퍼런스 엔진"으로만 지칭하고, 기법은 일반명으로 기술한다
> (예: "software-traced GI", "virtualized geometry", "cascaded shadow maps").

---

## 1. Canonical 디퍼드 파이프라인 패스 순서 (레퍼런스 엔진에서 추출)

레퍼런스 엔진의 디퍼드 씬 렌더 함수(메인 `Render()`)를 따라간 결과다. **핵심 관찰: 전 과정이 하나의
render dependency graph(RDG) 위에 선언되고**, 패스는 리소스 read/write 의존으로 순서가 결정된다
(우리 `crates/render` 그래프와 동일 철학). 아래 번호는 데이터 의존이 강제하는 논리적 순서이며, 각 줄의
"왜"는 그 스테이지가 **왜 그 자리에 있어야 하는지**(직전 패스가 무엇을 만들어줘야 하는지)를 설명한다.

### 1.0 프레임 셋업 / 가시성
- **(V) 씬 가시성 & 컬링** — view family/뷰별 frustum + occlusion(HZB) 컬링, GPU-scene 갱신, 라이트
  그리드/그림자 셋업. *왜 최상단:* 이후 모든 draw/dispatch의 워크리스트를 좁힌다. HW-occlusion용 HZB는
  이전 프레임 depth를 쓰거나 prepass 뒤 생성된다.

### 1.1 지오메트리 → G-buffer
1. **Depth Pre-pass (Early-Z)** — 불투명 지오메트리를 depth만 먼저 그린다(옵션으로 여기서 opaque
   velocity도 함께). *왜 먼저:* base pass의 픽셀 오버드로를 제거하고(Early-Z reject), HZB·SSAO·SSR·
   화면공간 GI가 **완성된 depth**를 전제로 하기 때문. prepass가 없으면 이 화면공간 패스들이 base pass와
   순서 경쟁을 하게 된다.
2. **DBuffer 데칼 (base pass 이전)** — G-buffer *기록 전에* albedo/normal/roughness를 수정하는 데칼을
   depth 위에 합성. *왜 여기:* base pass가 이 데칼 결과를 읽어 G-buffer에 섞어 넣어야 하므로 base pass보다
   앞서야 한다(prepass depth 필요).
3. **Base Pass (G-buffer fill)** — 불투명 머티리얼을 MRT로 래스터: base color·metallic/roughness·world
   normal·shading model 등 + velocity(옵션). *왜 여기:* 이후 라이팅/GI/반사가 전부 이 G-buffer를 소비한다.
4. **Custom Depth / Custom Stencil** — 아웃라인·특수 마스크용 별도 depth. *왜 여기:* base pass 뒤에
   특정 오브젝트만 다시 그려 마스크를 만들고 post에서 소비.
5. **GBuffer 데칼 (base pass 이후)** — G-buffer를 덮어쓰는 일반 디퍼드 데칼. *왜 여기:* 완성된 G-buffer
   위에 합성해야 라이팅이 데칼 반영된 표면을 조명한다.

### 1.2 그림자 · 그림자 전 라이트 준비
6. **Shadow Depth Maps** — 디렉셔널 CSM(cascade)·스팟/포인트 큐브·(가상 그림자 맵) depth 래스터.
   *왜 이 자리:* 라이팅 이전에 그림자 depth가 준비되어야 하고, base pass와 async로 겹칠 수 있어 대개
   G-buffer 전후로 스케줄된다. 라이트 그리드/컬링 결과에 의존.

### 1.3 간접광(GI) + AO — **직접광 이전**
7. **Diffuse Indirect + Ambient Occlusion** — 화면공간 AO/GI, software-traced GI(distance-field/surface
   cache), hardware-traced GI, 그리고 각 경로의 **디노이저**. *왜 직접광보다 먼저:* 결과가 씬 컬러의
   ambient/indirect 항으로 라이팅에 합쳐지고, 반사·스카이라이트 합성이 이 GI 텍스처를 입력으로 쓴다.
   traced 경로는 완성된 depth+G-buffer+shadow를 전제한다.

### 1.4 직접광 (라이팅 컴포지션)
8. **Direct Lighting (clustered/tiled)** — 라이트 그리드로 타일/클러스터별 라이트 리스트를 만들고, 그림자
   있는/없는 광원을 순회하며 G-buffer를 셰이딩해 씬 컬러에 누적. *왜 GI 뒤:* ambient/indirect가 이미 준비된
   상태에서 direct를 더한다. clustered 리스트는 depth 범위(prepass)에 의존.

### 1.5 반사 + 스카이라이트 — **직접광 이후**
9. **Reflections + Sky Lighting** — screen-space reflection(SSR) → software-traced → hardware-traced의
   하이브리드 합성 + 스카이라이트 스페큘러, 각 경로 디노이저. *왜 직접광 뒤:* SSR/traced 반사가 **직접광까지
   반영된 씬 컬러(라이팅된 히스토리)**를 샘플해야 정확하다(라이팅 피드백). 여기까지가 불투명 씬 컬러 완성.

### 1.6 하늘 · 대기 · 포그 · 볼류메트릭
10. **Sky / Atmosphere** — 물리기반 sky(에어리얼 퍼스펙티브 포함)를 불투명 씬 뒤 배경에 합성.
    *왜 여기:* 불투명 라이팅이 끝난 뒤 빈 depth 영역을 채운다.
11. **Height Fog / Volumetric Fog / Light Shafts / Volumetric Cloud** — depth 기반 포그와 볼류메트릭.
    *왜 여기:* 완성된 불투명 씬 컬러+depth 위에 거리별로 합성하며, 투명 이전에 적용돼야 투명이 포그 낀
    배경 위에 올라간다. 클라우드/볼류메트릭 라이팅은 종종 async compute로 앞 프레임과 겹쳐 준비된다.

### 1.7 투명 (Translucency)
12. **Translucency** — 정렬된 반투명/굴절/single-layer water를, 별도의 translucency lighting volume을
    써서 불투명 씬 컬러 위에 그린다(+투명 velocity). *왜 post 직전:* 불투명·포그가 완성된 뒤 순서대로 위에
    올려야 하고, depth 테스트는 하되 G-buffer 기록은 안 하므로 디퍼드 라이팅 밖에 있다.

### 1.8 Post-process 체인 (전형적 순서)
13. **Motion Blur** — velocity buffer 소비. *왜 먼저:* 이후 블룸/톤맵이 모션블러된 컬러를 받아야 함.
14. **Temporal Upscale / Temporal AA** — velocity로 히스토리 리프로젝션+누적, 내부해상도→출력해상도
    업스케일. *왜 이 지점:* **jitter된 씬 컬러 + velocity + depth**가 필요하고, 톤맵/블룸 **이전**의
    linear-HDR에서 동작해야 히스토리 누적이 안정적이다(톤맵 후 AA는 밴딩/고스팅).
15. **Auto Exposure (eye adaptation)** — 히스토그램/평균 루미넌스로 노출 산출. *왜 톤맵 앞:* 톤맵이 이
    노출을 소비. 실제로는 앞 프레임 값을 쓰도록 파이프라인 초반에 계산을 걸어두는 경우가 많다.
16. **Bloom (+ Lens Flare)** — 밝은 영역 다운샘플 블러 체인. *왜 톤맵 앞:* linear-HDR에서 추출해 톤맵
    입력에 더한다.
17. **Depth of Field** — CoC 기반 보케. *왜 톤맵 앞·모션블러 부근:* linear-HDR + depth 소비.
18. **Tonemap + Color Grading (LUT)** — 노출·블룸·그레이딩을 적용해 HDR→디스플레이. *왜 여기:* 위
    HDR 효과가 전부 끝난 뒤 단 한 번 디스플레이 인코딩.
19. **FXAA / SMAA (옵션)** — 톤맵 **후** 공간 AA(TAA를 쓰지 않을 때). *왜 톤맵 뒤:* LDR 엣지 기반.
20. **Primary/Secondary Upscale** — 최종 출력 해상도로 스페이셜 업스케일 + 샤픈.
21. **Editor/Debug Overlay** — 셀렉션 아웃라인·기즈모·시각화·HUD. *왜 최후:* 씬 파이프라인 밖.

### 1.9 구조적 기계장치 (순서 아님, 전 과정에 관통)
- **Render Dependency Graph(RDG):** 모든 패스를 read/write로 선언 → 배리어·트랜지언트 aliasing·async
  스케줄을 자동화. DreamCoast `crates/render`가 같은 역할.
- **Async Compute 오버랩:** 그림자 depth·볼류메트릭 클라우드·GI 디노이즈·라이트 컬링 등 무거운 compute를
  그래픽스 큐와 겹쳐 실행(그래프가 프리레퀴짓으로 관리).
- **Scalability 설정 그룹:** 기능별 품질(샘플 수/해상도 divisor/토글)을 티어(low/med/high/ultra)로 묶어
  런타임 조정. DreamCoast `RenderQuality`가 대응.
- **View Family:** 한 프레임에 여러 뷰(스플릿스크린/스테레오/씬 캡처)를 공용 리소스로 렌더.

---

## 2. DreamCoast ↔ 레퍼런스 매핑 (정직한 스테이지별 진단)

현 프레임 흐름(코드 기준): `apps/sandbox/src/main.rs`의 프레임 루프가
**shadow depth → G-buffer(4 MRT) → deferred decals(A3) → GDF/SW-RT compute(AO / screen-probe 또는
ray-march GI / SSR+반사 composite) → PBR deferred lighting → (auto-exposure) → (compute post box-blur) →
TAAU(내부해상도 업스케일) → tonemap → (particle/cull draw/ImGui)`.
근거 코드: `deferred.rs`(`record_shadow`/`record_gbuffer`/`record_decals`/`record_lighting`/
`record_tonemap`), `gi.rs`·`reflect.rs`·`gdf.rs`·`gtao.rs`·`taau.rs`, `ibl.rs`.

범례: ✅ 존재·정합 · 🟡 존재하나 구조/순서 상이 · 🔴 부재.

| # | 레퍼런스 스테이지 | DreamCoast 현황 | 판정 | 비고 (근거) |
|---|---|---|---|---|
| V | 씬 가시성/컬링 | GPU frustum 컬링(P7 `cull.rs`), HZB occlusion 없음 | 🟡 | 컬링은 있으나 occlusion/HZB 부재; 라이트 컬링(클러스터) 없음(단일 디렉셔널) |
| 1 | Depth Pre-pass | **없음** — G-buffer가 첫 depth writer | 🔴 | `record_gbuffer`가 depth를 처음 생성. 화면공간 패스(SSR/AO)가 G-buffer depth에 직접 의존 → prepass 없이 순서 강제됨. GDF AO flicker 근원도 depth 라이프타임(메모리 참조) |
| 2 | DBuffer 데칼(전) | **없음** | 🔴 | 데칼은 base pass **후** 경로만 존재(아래 #5) |
| 3 | Base Pass(G-buffer) | ✅ 4 MRT + **velocity RT(RG16F, PR-2, opt-in `P_VELOCITY=1`)** | 🟡 | velocity 채널 도입(전용 RT + 별도 불투명 패스, unjittered `clip−prevClip`, 스태틱·Spin·스키닝·모프 prev-pose 단일 소스 — [velocity-motion-vectors.md](velocity-motion-vectors.md)); world-position 명시 저장은 여전(레퍼런스는 depth에서 재구성) |
| 4 | Custom Depth/Stencil | **없음** | 🔴 | 아웃라인/마스크 파이프라인 부재 |
| 5 | GBuffer 데칼(후) | ✅ `record_decals`(A3, deferred decal) | ✅ | base pass 후·라이팅 전 — 정합. Intel Sponza `dirt_decal`에서 검증 |
| 6 | Shadow Depth | ✅ `record_shadow`(단일 디렉셔널 shadow map, PCF/PCSS-lite) | 🟡 | 단일 맵만 — **CSM/cascade·스팟/포인트 큐브·아틀라스 없음** |
| 7 | Diffuse Indirect + AO | ✅ GDF AO + GTAO + software-traced GI(screen-probe / ray-march) + 디노이저 | ✅ | **직접광 이전**에 배치 — 정합. 이 트랙은 오히려 레퍼런스급으로 깊다(P10/P11) |
| 8 | Direct Lighting | 🟡 `record_lighting`(풀스크린 PBR, 단일 디렉셔널 + point) | 🟡 | **clustered/tiled 라이트 리스트 없음** — 다광원 확장 시 병목. 순서(GI 뒤)는 정합 |
| 9 | Reflections + Sky | ✅ SSR→GDF→sky 하이브리드 composite, lit-history 피드백 | 🟡 | 순서(직접광 뒤 lit-history 샘플) 정합. 단 lit-history 피드백이 flicker 유발(메모리: swrt_reflect 루프) |
| 10 | Sky/Atmosphere | 🟡 절차 sky → env cube(`ibl.rs`), 씬 배경 합성은 IBL로 | 🟡 | 물리기반 sky/에어리얼 퍼스펙티브/time-of-day 없음; 별도 sky 합성 패스 아닌 IBL 캡처 |
| 11 | Fog/Volumetric | **없음** | 🔴 | height fog·volumetric·light shaft·cloud 전무 |
| 12 | Translucency | **없음**(불투명만) | 🔴 | 정렬 투명/OIT/굴절 없음 — foliage는 alpha-cutout(불투명 경로)로 우회 |
| 13 | Motion Blur | **없음** (선결이던 velocity는 PR-2로 확보) | 🔴 | velocity RT가 생겨 이제 구현 가능 — PR-5 post 시퀀스의 스텁 슬롯에 삽입 예정 |
| 14 | Temporal AA/Upscale | ✅ `taau.rs`(TAAU: jitter + 리프로젝션 누적, 톤맵 **전** linear-HDR) + **velocity-aware 리프로젝션(PR-2)** | 🟡 | 위치(톤맵 전)·jitter 정합. `P_VELOCITY=1`이면 3×3 dilated per-pixel velocity로 리프로젝션(움직이는 오브젝트 고스팅 감소, 검증 수치는 [velocity-motion-vectors.md](velocity-motion-vectors.md)); off면 종전 카메라-온리(바이트 동일) |
| 15 | Auto Exposure | ✅ `record_auto_exposure`(히스토그램/평균, 적응) | ✅ | 톤맵 전 배치 정합(opt-in) |
| 16 | Bloom | 🟡 P5 데모 블룸 체인 존재하나 현 프레임 경로엔 미배선 | 🟡 | `record_compute_post`는 3×3 박스 블러(데모)이며 블룸 아님. 실 블룸은 tonemap에 미통합 |
| 17 | Depth of Field | **없음** | 🔴 | — |
| 18 | Tonemap + Grading | 🟡 `record_tonemap`(ACES + sRGB 인코드) | 🟡 | 톤맵은 있으나 **컬러 그레이딩/LUT 없음**, 노출은 라이팅 패스에서 선적용 |
| 19 | FXAA/SMAA | 🟡 `record_fxaa`(TAA 미사용·비jitter 시 FXAA pre-pass) | 🟡 | 존재하나 TAA **pre-pass**로 배치(레퍼런스는 톤맵 후 대체 AA). 역할 상이 |
| 20 | Primary/Secondary Upscale | 🟡 TAAU가 내부→출력 업스케일 겸함 + 톤맵 샤픈 | 🟡 | 전용 스페이셜 업스케일 단계 없음(TAAU에 융합) |
| 21 | Editor/Debug Overlay | ✅ ImGui 오버레이 + 디버그 뷰(`DEBUG_VIEW`) + particle/cull draw | 🟡 | 톤맵 후 draw. 단 particle/cull이 **톤맵 후 LDR**에 그려짐(HDR 합성 아님) |
| — | Render Graph | ✅ `crates/render`(트랜지언트 aliasing, read/write 선언) | ✅ | 레퍼런스 RDG와 동일 철학 |
| — | Async Compute | ✅ P7 async(메모리: async-compute.md) | 🟡 | 인프라는 있으나 현 프레임 패스 오버랩은 제한적 |
| — | Scalability 티어 | ✅ `RenderQuality{low/med/high}` | ✅ | cvar 그룹 대응 |
| — | View Family | 🟡 단일 뷰 | 🟡 | 스플릿/스테레오/씬캡처 다중 뷰 없음 |

### 2.1 순서/구조 상이가 **미래 기능을 막는** 지점 (핵심)
1. **Depth pre-pass 부재 (#1).** G-buffer가 최초 depth writer라 SSR/AO/GI가 완성 depth를 전제하는 순서를
   *암묵적으로만* 만족한다. Early-Z 오버드로 제거, HZB occlusion 컬링, hi-Z 기반 SSR 트레이스, 정확한
   화면공간 트레이싱 모두 prepass depth를 요구한다. 부재가 GDF AO depth-lifetime 버그의 배경이기도 했다.
2. **~~Velocity / 모션벡터 부재 (#3·#13·#14)~~ — PR-2로 해소.** velocity RT(RG16F, opt-in
   `P_VELOCITY=1`) + velocity-aware TAAU 리프로젝션이 들어가 움직이는 오브젝트(스키닝/스핀/모프)의
   고스팅이 감소. 모션블러(13)의 선결도 확보 — [velocity-motion-vectors.md](velocity-motion-vectors.md).
3. **투명 패스 부재 (#12).** 디퍼드 라이팅 뒤·post 앞의 정렬 투명 슬롯 자체가 없어 유리/굴절/파티클
   HDR 합성/포그 상호작용을 넣을 자리가 없다. 현재 particle/cull이 **톤맵 후 LDR**에 그려지는 것도 이
   슬롯 부재의 증상이다.
4. **포그/대기/볼류메트릭 슬롯 부재 (#10·#11).** 불투명 완성 후·투명 전의 합성 지점이 비어 있어, sky를
   IBL 캡처로만 처리하고 aerial perspective·height fog·light shaft를 꽂을 자리가 없다.
5. **클러스터드 라이트 리스트 부재 (#8).** 라이팅이 풀스크린 단일-패스라 다광원(point/spot/area 다수 +
   다수 그림자)으로 확장하려면 라이트 컬링/그리드 인프라를 먼저 깔아야 한다.
6. **Post 체인이 톤맵 위주 단일 노드 (#16·#17·#18).** 블룸(데모만)·DoF·그레이딩/LUT을 꽂을 **순서 있는
   post 시퀀스**가 없다. 현재는 `hdr → (compute box-blur 데모) → taau → tonemap`으로 파편적.

---

## 3. "파이프라인 정합" 작업 트랙 (의존순 · 우선순위)

§2.1의 구조 블로커를 **의존 순서**로 정렬한 작업 항목. 각 항목은 기존 5원칙(root-cause·최적화·스케일러
빌리티·단일 소스·검증)과 양 백엔드(DX≡VK ≤0.001/ch)·골든이미지 무회귀·opt-in seam을 준수한다.
크기: **S**(수일)·**M**(1–2주)·**L**(2주+). 이 트랙은 Phase 20+ 기능 구현의 **선결 인프라**다.

### 정합 P0 — 프레임 골격 (기능보다 먼저)
- **PR-1 · Depth Pre-pass 도입 [M].** 불투명 depth-only 패스를 G-buffer 앞에 추가, G-buffer는 `Equal`
  depth-test로 오버드로 제거. 화면공간 패스가 **명시적으로** prepass depth를 읽도록 그래프 배선.
  *왜 먼저:* 아래 velocity·HZB·SSR 정확도의 공통 토대. *unblocks:* HZB occlusion 컬링, hi-Z SSR,
  화면공간 트레이싱 정확도, GDF AO depth-lifetime 근본 정리. *리스크:* 양 백엔드 depth load/store +
  골든이미지 anchor 바이트 동일 유지(기존 depth graph-driven load/store 재사용).
- **PR-2 · Velocity(모션벡터) G-buffer 채널 [M] — ✅ DONE.** 전용 RG16F RT + 별도 불투명 velocity
  패스로 `clip − prevClip`(unjittered NDC) 출력 — 스태틱·Spin·스키닝(prev 팔레트)·모프(prev 웨이트)
  전부 prev-pose 단일 소스에서 공급. velocity-aware TAAU(3×3 dilated) 배선까지 포함, opt-in
  `P_VELOCITY=1`(off = 골든 앵커 바이트 동일) + `DEBUG_VIEW=11` 시각화. 설계·검증 수치:
  [velocity-motion-vectors.md](velocity-motion-vectors.md). DX≡VK parity pending Windows.

### 정합 P1 — 합성 슬롯 열기
- **PR-3 · 투명 패스 슬롯 [M].** 디퍼드 라이팅+반사 뒤, post 앞에 **정렬 불투명-후 투명 패스**를 그래프에
  삽입(깊이 테스트, G-buffer 미기록). 우선 단순 정렬 알파 블렌드; OIT/굴절은 Phase 20. *unblocks:*
  유리/파티클 HDR 합성·포그 상호작용. **부수효과:** particle/cull draw를 이 슬롯(HDR)으로 이동 →
  톤맵-후 LDR 드로 제거.
- **PR-4 · 대기/포그 합성 슬롯 [S].** 불투명 완성 후·투명 전에 **sky+fog 합성 지점**을 그래프에 확보
  (지금은 no-op 통과, Phase 22에서 채움). *왜 지금:* 순서 자리를 박아두면 Phase 22가 재배선 없이 삽입.
- **PR-5 · Post-process 시퀀스 정식화 [M].** 파편적 `compute_post(box-blur 데모)`를 제거하고, 순서 있는
  post 노드 시퀀스로 정리: `motion-blur(스텁) → TAA/upscale → exposure(기존) → bloom(P5 체인 재배선) →
  DoF(스텁) → tonemap+grading`. 각 노드 opt-in. *unblocks:* Phase 20 포스트 스택이 자리에 꽂힘.

### 정합 P2 — 라이팅 확장 토대
- **PR-6 · 클러스터드 라이트 컬링 인프라 [L].** 뷰 절두체를 클러스터로 나눠 라이트 리스트 빌드(compute),
  `record_lighting`이 타일/클러스터 리스트를 소비하도록 변경(단일 디렉셔널은 특수 케이스로 유지, 바이트
  동일). *unblocks:* Phase 21 다광원(point/spot/area)·다수 그림자. *의존:* P7 compute.
- **PR-7 · 그림자 아틀라스/CSM 골격 [L].** 단일 shadow map을 cascade/아틀라스로 일반화(디렉셔널 CSM +
  스팟/포인트 슬롯). *의존:* PR-6 라이트 리스트. *unblocks:* Phase 21 다수 그림자.

### 정합 P3 — 가시성/멀티뷰 (후속)
- **PR-8 · HZB Occlusion 컬링 [M].** PR-1 prepass depth로 Hi-Z 피라미드 빌드 → GPU occlusion 컬링.
  *의존:* PR-1. *unblocks:* 대규모 씬(월드 렌더링 Phase 23) 스케일.
- **PR-9 · View Family(다중 뷰) [M].** 공용 리소스로 N뷰 렌더(씬 캡처/스플릿/스테레오). *unblocks:*
  실시간 env 캡처·에디터 다중 뷰포트.

### 권장 착수 순서
`PR-1(prepass) → PR-2(velocity) → PR-5(post 시퀀스) → PR-3(투명 슬롯) → PR-4(대기 슬롯) →
PR-6(클러스터드) → PR-7(CSM) → PR-8(HZB) → PR-9(멀티뷰)`.
**PR-1·PR-2·PR-5 세 개가 Phase 20(AA/포스트/투명)의 실질 선결**이며, 프레임 골격을 레퍼런스 정합으로
끌어올리는 최소 집합이다.

---

## 4. ROADMAP 반영
- Phase 20 이전에 **"파이프라인 정합(PR-1..9)" 트랙**을 신설 — 기능 구현(20+)의 선결 인프라로 명시.
- Phase 20(AA/포스트/투명)·21(다광원/그림자)의 완료 기준을 이 문서의 PR 항목에 연결.
- 상세 §A(렌더 완성도) 표는 [commercial-engine-gap-analysis.md](commercial-engine-gap-analysis.md)에서
  소스레벨 발견(velocity·prepass·투명 슬롯·클러스터드)을 반영해 sharpen.
