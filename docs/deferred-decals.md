# Deferred decals — 표면 데칼(트랙 A) 계획

상위: [ROADMAP.md](ROADMAP.md) · 관련: [phase-6-pbr.md](phase-6-pbr.md)(deferred G-buffer),
[phase-11-asset-pipeline.md](phase-11-asset-pipeline.md)(glTF 머티리얼 임포트),
[sponza-perf.md](sponza-perf.md)(Sponza 컨텍스트).

상태: **계획 / 미구현 (리뷰 대기)**. 후속 트랙 B(포워드 투명)는 [§ 후속](#후속-트랙-b--포워드-투명) 참고.

## 문제 (RenderDoc로 근본 원인 확정, 2026-06-30)

Intel Sponza 내부에서 기둥/벽/천장의 큰 영역(데모 앵글 기준 화면의 ~27%)이 **검은 금속**으로
보였다. RenderDoc 드로우별 캡처(`RDOC_CAPTURE=1`, D3D12)로 추적한 결과 **metallic 파이프라인은
정상**이고 — 모든 드로우의 push constant `tex.y`가 valid — 원인은 **`dirt_decal` 머티리얼**이었다:

| dirt_decal (Sponza material 21) | 값 |
|---|---|
| `alphaMode` | **BLEND** (35% 불투명: `baseColorFactor.a = 0.35`) |
| `metallicRoughnessTexture` | **없음** → `metallicFactor` = glTF 기본 **1.0** |
| `baseColorTexture` | 있음 (먼지 텍스처) |

의도는 석재 위에 **35% 투명하게 블렌딩되는 먼지 오버레이**다. 그러나 현재 엔진은:

1. **`alphaMode=BLEND`를 opaque로 처리**한다(문서화된 한계 — `gltf_scene.rs`/`shadow.slang` 주석:
   "BLEND is treated as opaque until true alpha blending lands"). → 먼지가 섞이는 대신 데칼이 석재를
   **완전 불투명**하게 덮어쓴다.
2. **MR 텍스처가 없어 metallic = 기본값 1.0** → 디퓨즈 없는 금속 → 어두운 GI 실내에서 **검정**.

RenderDoc 픽셀 히스토리(예: `(100,360)`): 석재(metallic 0.012)를 그린 뒤 dirt_decal(`tex=[57,-1,-1,-1]`,
metallic 1.0)이 **약간 앞 깊이**로 depth-test를 통과해 석재를 덮어씀 → 검은 금속 패치. 프레임당 데칼
드로우 4개가 넓은 면적을 덮는다.

**핵심:** 표면 데칼을 "자기 자신을 라이팅하는 불투명 표면"으로 그리면 안 된다. 먼지는 **밑면(석재)의
라이팅을 그대로 받으며 albedo만 틴트**해야 한다. 단순 포워드 알파블렌딩으로도 데칼이 자기 metallic(1.0)로
라이팅돼 석재를 검게 darkening한다 → **deferred(G-buffer) 데칼**이 정공법.

## 설계 — Deferred(G-buffer) 데칼

UE/Frostbite 방식: 메인 G-buffer fill **후**, 라이팅 **전**에 데칼이 **이미 기록된 G-buffer를 수정**한다.
라이팅은 수정된 G-buffer에 **한 번만** 실행 → 데칼이 밑면의 올바른 라이팅을 받고, metallic 아티팩트·더블
셰이딩이 없다.

### 프레임 흐름
```
shadow → G-buffer fill(불투명) → [NEW] 디퍼드 데칼 패스 → 컴퓨트 GDF → PBR 라이팅 → tonemap
```

### 데칼 패스가 G-buffer에 쓰는 것
G-buffer MRT(현재): `RT0 albedo(rgb)+AO(a)`, `RT1 normal`, `RT2 material(r=metallic,g=rough,b=AO)`,
`RT3 worldpos`. 데칼은:

- **RT0 albedo**: 데칼 base-color를 **알파 블렌드**(`src = decal.rgb`, `α = decalTex.a × baseColorFactor.a`).
- **RT2.g roughness**(선택): 데칼이 roughness를 가지면 같은 α로 블렌드.
- **metallic(RT2.r) / normal(RT1) / worldpos(RT3)는 건드리지 않음** — 밑면 값 유지. ← dirt_decal의
  metallic=1.0 문제가 원천 차단된다.
- **깊이**: depth-test `LEqual`(밑면에만 부착), **depth-write OFF**(데칼은 가림 대상 아님).

### 메시 데칼 vs 박스 데칼
Sponza dirt_decal은 표면에 동일평면으로 놓인 **메시 데칼**(glTF 지오메트리). 이번 트랙은 **메시 데칼**을
대상으로 한다(데칼 메시를 위 블렌드 상태로 그대로 래스터). 투영 **박스 데칼**(데칼 볼륨 → 스크린/박스 투영)은
범용 게임용 후속 확장으로 분리(§ 후속).

## 필요한 변경

### 1. 임포트 — alphaMode 보존 + 데칼 분류 (`crates/asset`, `apps/sandbox`)
- `GltfMaterial`에 `alpha_mode: AlphaMode {Opaque, Mask, Blend}` 추가(현재 cutoff만 캡처). 단일 소스.
- **데칼 분류**: glTF엔 decal 플래그가 없다(데칼·투명 모두 BLEND). 머티리얼에 `kind: MaterialKind
  {Opaque, Masked, Decal, Transparent}`를 임포트 시 결정:
  - 단기: **이름 휴리스틱**(`name.contains("decal")`) — Sponza가 이 컨벤션을 씀.
  - 장기: 저작 컨벤션 / `KHR_*` ext / 씬 메타데이터로 명시(휴리스틱 대체). 분류 로직은 한 함수에 격리.
  - BLEND ∧ ¬decal → `Transparent`(트랙 B로 보류, 현행처럼 opaque 폴백 유지).
- `MaterialDesc`/`SceneObject`에 `kind` 전파. 데칼은 별도 draw 리스트로 분리.

### 2. RHI — per-RT 블렌드 + write-mask (`rhi-types` + 3 백엔드)
현재 `BlendMode {Opaque, AlphaBlend}`는 **전역**이라 "RT0만 알파블렌드, RT1/RT3 비기록, RT2는 g만"을
표현 못 한다. 데칼 파이프라인엔 **per-attachment 블렌드 + color write mask**가 필요:
- `GraphicsPipelineDesc`에 per-RT 블렌드/write-mask를 추가하거나, 데칼 전용 프리셋
  `BlendMode::DecalAlbedo`(RT0 알파블렌드 + 쓰기, 나머지 RT write-mask=0; RT2는 후속에 g만) 도입.
- VK: `vk::PipelineColorBlendAttachmentState` 배열을 RT별로, `color_write_mask` 지정.
- D3D12: `D3D12_BLEND_DESC.RenderTarget[i]` per-RT + `RenderTargetWriteMask`.
- Metal: `MTLRenderPipelineColorAttachmentDescriptor[i]`의 blend + `writeMask`.
- **DX≡VK 게이트**(per-RT 블렌드 상태 = 크로스백엔드 표면). Metal은 macOS에서 검증.

### 3. 셰이더 (`crates/shader`)
- `decal.slang`(또는 `gbuffer.slang`의 `fsDecal` 엔트리): 데칼 base-color 샘플 → `RT0 = float4(rgb, α)`
  (블렌드 상태가 α로 over). roughness 후속이면 `RT2.g`도. metallic/normal/pos는 write-mask로 차단.
- VS는 기존 `vsMain`(Mesh 레이아웃) 재사용(메시 데칼은 일반 지오메트리).

### 4. sandbox 배선 (`apps/sandbox/deferred.rs`, `main.rs`)
- `gbuffer_decal_pipeline`(데칼 셰이더 + `DecalAlbedo` 블렌드 + depth-test/no-write).
- G-buffer fill 직후, 라이팅 전에 **데칼 패스**: `scene`의 `kind==Decal` 드로우를 위 파이프라인으로 그림.
- 데칼 없으면 패스 스킵 → **갤러리/비데칼 씬 바이트 동일**(no-op).

## 검증 (프로젝트 규칙: verify, then claim)
- **Sponza 데모 앵글**: 검은 금속 패치 → **은은한 먼지 틴트**(밑 석재 보임). RenderDoc 재캡처로 데칼이
  RT0 albedo만 블렌드하고 RT2 metallic은 불변임을 확인.
- **DX≡VK ≤ 0.001/ch**(데칼 패스 + per-RT 블렌드). VK 검증 클린.
- **갤러리/비데칼 씬 바이트 동일**(데칼 opt-in/스킵 = no-op).
- **PT 잔차**: Sponza는 PT 레퍼런스가 없으나(콘텐츠), 데칼 적용 전후 비-데칼 영역 불변 확인.
- `clippy -D warnings` / `fmt` 클린, VK 검증·D3D12 디버그레이어 무에러.

## 스테이징 (각 단계 = 검증된 단일 커밋)
- **A1 — 임포트/분류 (CPU, 렌더 무변경)**: `AlphaMode`/`MaterialKind` 파싱 + 전파 + 데칼 draw 분리.
  데칼 패스 미연결 → **바이트 동일**. 유닛 테스트(분류).
- **A2 — RHI per-RT 블렌드/write-mask (크로스백엔드)**: `GraphicsPipelineDesc` 확장 + 3 백엔드.
  기존 파이프라인 영향 0(기본 = 현행). DX≡VK 게이트.
- **A3 — 데칼 패스 + 셰이더 (albedo 블렌드)**: `decal.slang` + `gbuffer_decal_pipeline` + 배선.
  Sponza 먼지 패치 정상화. DX≡VK + 갤러리 무회귀.
- **A4 — roughness 블렌드(선택) + 마무리**: RT2.g 블렌드, 검증·문서·메모리.

## 스케일링 / 품질 노브 (한 곳)
- 데칼은 **정확성**(기본 on). 토글/캡은 `quality.rs` 한 곳: `decals_on`, `max_decals`(드로우 캡), 추후
  `RenderQuality` 티어 연동(Low에서 데칼 스킵 등). 무거운 게 아니라 기본 on이 적절.

## 후속 — 트랙 B / 박스 데칼
- **트랙 B — 포워드 투명**: 진짜 투명(glass, 잎)용. 라이팅 후 정렬된 OVER 블렌드 패스. 데칼과 별개 기능
  (BLEND ∧ ¬decal). 별도 계획.
- **박스(투영) 데칼**: 데칼 볼륨 → G-buffer 박스 투영. 메시 데칼이 자리잡은 뒤 범용 게임용으로 확장.
- **데칼 normal 블렌드**: 노멀까지 섞는 고급 데칼(범프 데칼). RT1 부분 블렌드 — 정밀도 주의.
- **다중 데칼 정렬**: 겹치는 데칼의 블렌드 순서(현재는 draw 순서). 필요 시 정렬/우선순위.

## 임시 폴백 (계획 구현 전, 선택)
정공법 전까지 임시로: **MR 텍스처 없는 BLEND 머티리얼의 metallic을 0으로 폴백**하면 검은 금속은
사라진다(먼지가 회색 디퓨즈로). 정확한 틴트는 아니지만 "검정"보다 낫다. 단, 이는 데칼이 여전히
opaque로 석재를 덮으므로 임시방편 — A3가 정답.
