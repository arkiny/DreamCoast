# 다음 세션 프롬프트 — 실내 GI: 화면-공간 라디언스 프로브 게더 (레퍼런스 품질 구현)

> 이 문서 = 다음 세션 콜드스타트 작업 지시. 그대로 붙여 시작할 수 있게 자기완결적으로 작성.

## 빌드 디렉티브 (최우선, 메모리 `dreamcoast-build-to-quality`)
**한계효용으로 기능을 제약하지 말 것.** 특정 씬에서 가시 변화가 작아도 되돌리지 말 것. GI는 **레퍼런스
충실도 + path-tracer 패리티**로 측정하고, **최적화된 고품질 재사용 라이브러리**로 구현한다. 기존 하드
게이트(갤러리 바이트 동일·DX≡VK·결정론·heavy=opt-in)는 *품질* 게이트이므로 유지.

## 작업
DreamCoast 실내 GI를 **화면-공간 라디언스 프로브 게더**로 구현한다(레퍼런스의 주 diffuse GI 방식).
기존 월드-볼륨 GI(32³ SH-L1)는 단일 그리드 배치 한계로 닫힌/열린 실내를 못 채운다(아래 "확정된 배경").
화면 프로브는 **보이는 표면에 배치**되어 그 한계를 구조적으로 없앤다.

## 프로젝트 (콜드스타트)
DreamCoast — 순수 Vulkan(ash)/D3D12(windows-rs)/Metal(objc2)를 직접 깐 Rust 그래픽 엔진(wgpu/프레임워크
없음). 하나의 hand-rolled 바인드리스 RHI 뒤 3 백엔드. 딥퍼드(G-buffer) PBR + SW-RT GI/AO/reflect(베이크된
글로벌 거리장 GDF + 클립맵). 게임용 프로덕션 코드. 검증 분담: macOS=Metal(이 머신), Windows=Vulkan/D3D12
DX≡VK 게이트(RTX 2070 SUPER, **현재 동결**). 먼저 `DreamCoast/CLAUDE.md` 읽기. 작업 루트
`/Users/arkiny/GitRepos`, 엔진은 `DreamCoast/`. 브랜치 `feature/per-mesh-distance-fields` 이어서.
레퍼런스 상용 엔진 소스 `/Users/arkiny/GitRepos/UnrealEngine-1` (추측은 이 소스 대조; 산출물엔 상표명 금지).

## 하드 룰
- **상표명 금지**: 제3자 제품명·소스 식별자(Unreal/UE/Lumen/Nanite/Epic 등)를 문서/주석/커밋에 쓰지 말 것.
  기법을 일반어로 기술. (메모리 `dreamcoast-no-trademark-names`)
- **갤러리 바이트 동일**이 매 변경 1순위 무회귀 게이트(SHA-256). 신규 GI는 콘텐츠 전용 게이트.
- **DX≡VK ≤0.001 avg/ch** (Windows 동결 시 Metal 구현·검증 + 보류 명시). 푸시 레이아웃은 후행 스칼라/스페어로
  안전하게(256B Vulkan 한계 유의). (메모리 `dreamcoast-verification-split`)
- 근본원인·단일소스·heavy opt-in·**verify-then-claim**. **정확도 1순위 = path-tracer 패리티**.

## 확정된 배경 (이미 출하/규명 — 반복·재진단 금지)
- **출하됨**(브랜치, 커밋 `d951b00`+): SH-L1 **방향성 라디언스 프로브**(probe당 밴드0/1 SH = 12 R32F 볼륨,
  contiguous-base = 푸시 무증가) + **실내 스카이라이트 차폐**(probe sky-vis SH-L1 → IBL diffuse를 V로 차폐
  + 중립 OcclusionTint/MinOcclusion leak; `P_SKYVIS_TINT`/`P_SKYVIS_MIN_OCC`). 갤러리 바이트 동일. README
  히어로(`docs/media/sponza.png`)도 디블루 반영. 문서 `docs/gi-radiance-cache.md`(v1~v3).
- **월드-볼륨 bounce-fill 트랙은 막힘**(문서 `docs/radiance-cache-fill.md`에 소진적 기록): 단일 32³ 그리드가
  (a) 닫힌 중앙까지 전파 못 함 + (b) 열린 홀 중앙은 하늘을 봐서 E≈0. S0a(probe 거리모멘트 + 8-probe 체비셰프
  가시성 보간)까지 **구현·측정했으나 이 씬에선 inert**(cause B는 probe 선택으로 못 고침) → 되돌림. **결론:
  월드-볼륨으로는 한계. 화면-공간 프로브가 정답.** (이 결론은 확정 — 다시 월드-볼륨 튜닝으로 회귀 말 것.)

## 이번 작업 = 화면-공간 라디언스 프로브 게더 (스테이지)
레퍼런스 대조: `UnrealEngine-1/Engine/Shaders/Private/Lumen/LumenScreenProbe*.usf/.ush`
(`LumenScreenProbeGather`, `…ImportanceSampling`, `…TileClassication`, `…Filtering`, `…Common`),
월드 캐시 `LumenRadianceCache*` / `LumenIrradianceFieldGather.cpp`(기본값 추출처). 기본값·동작방식을
이 소스에서 뽑아 우리 GDF SW-RT 위에 구현(상표명 미표기).

- **P1 — 프로브 배치 + 트레이스 + octahedral 저장.** 화면을 타일(레퍼런스 ~16×16px)로 나눠 타일당 프로브 1개를
  대표 표면(가장 가까운/대표 depth·normal)에 배치. 프로브당 **octahedral radiance**(시작 8², `MinRadianceProbeResolution`
  참고) 아틀라스. 각 텍셀 방향으로 **기존 `gdf_gi.slang trace_bounce`를 재사용**해 GDF 트레이스(hit 재라이트 =
  지금 그대로; miss = 0/sky 정책 일관). 프로브가 표면에 있으니 지하/배치 문제 없음. half-res 트레이스 OK.
- **P2 — importance sampling + 필터.** BRDF + 이전 프레임 라디언스로 중요도 샘플(`…ImportanceSampling`),
  화면-공간 spatial + temporal 필터(`…Filtering`, octahedral 보더 처리). 노이즈/반딧불 억제.
- **P3 — 픽셀 통합(현 GI 소비 대체/병행).** 픽셀이 주변 프로브들을 depth/normal 가이드로 보간해 반구 적분 →
  indirect irradiance E. 현 `gdf_gi` 볼륨-샘플 분기를 화면-프로브 통합으로 교체(콘텐츠), 갤러리는 현 경로 유지.
  스카이라이트 차폐(출하된 sky-vis)와 정합: 프로브 게더가 sky-visible/occluded를 자연 포함하면 차폐 항 재정리.
- **P4 — 월드 라디언스 캐시(클립맵) 폴백.** 화면 프로브의 긴-거리/off-screen/무한바운스 폴백 = 월드 캐시
  (카메라-추종 클립맵, 레퍼런스 기본 4레벨 ×2.0, 64³/레벨, octahedral + Chebyshev occlusion + 프로브 마킹 +
  아틀라스/인디렉션). 64-볼륨 슬롯 제약 → 프로브 아틀라스(2D)+인디렉션로 저장 재설계(현 per-coeff 3D 볼륨 탈피).
- **P5 — 최적화.** 타일 분류(`…TileClassication`: 평면/복잡 분기), ray budget, half/quarter-res, 시간 상각,
  `PROFILE_GPU` 비용 측정·보고.

## 기본값 (레퍼런스 → 우리 ~37 m 씬)
| 파라미터 | 레퍼런스 기본 | 우리 시작값 |
|---|---|---|
| 화면 프로브 타일 | 16×16 px | 16 (조정 가능) |
| probe octahedral radiance | 8²~16² | 8² 시작 |
| ray budget / probe | 적응 | spp 타일분류 연동 |
| 월드 캐시 클립맵 | 4 레벨 ×2.0, 64³ | P4에서, 메모리 맞춰 축소 |
| occlusion bias (Chebyshev) | 0.8 | 0.8 |

## 측정 (verify-then-claim)
- **1순위 = path-tracer 패리티**: `P8_PATHTRACE=1` 캡처 vs raster를 동일 카메라에서 `tools/rt-compare.py`로
  잔차 비교(정확도). 정확도가 목표지 "이 씬 가시변화"가 아님.
- 갤러리 바이트 동일(SHA-256), 결정론(run-to-run 바이트 동일), 비용(`PROFILE_GPU`).
- 씬: `gallery`(앵커), `LEVEL=sponza_intel`(EV100=11), `LEVEL=sponza_hero`(EV100=12, 히어로).
- 스크린샷: `EV100=11 LEVEL=sponza_intel ./target/release/sandbox --backend metal --screenshot-clean out.png`
  (RELEASE, 64프레임 warmup). `bleed.py`(scratchpad) 바닥 메트릭은 보조 지표일 뿐.

## 게이트 (스테이지마다)
`cargo fmt` → `RUSTFLAGS="-D warnings" cargo clippy -p sandbox -p dreamcoast-asset --all-targets`
→ **path-tracer 잔차 보고** → 갤러리 바이트 동일 → 결정론 → DX≡VK(Windows 동결 보류 명시) → `PROFILE_GPU`.

## 관련 코드/문서/메모리 (먼저 읽기)
- 코드: `crates/shader/shaders/{gdf_gi,gi_volume,pbr,gdf_reflect,clipmap}.slang`, `apps/sandbox/src/{gi,deferred,
  push,main,reflect}.rs`. `gdf_gi.slang trace_bounce`가 프로브 트레이서의 토대.
- 문서: `docs/radiance-cache-fill.md`(월드-볼륨 한계·S0 진단), `docs/gi-radiance-cache.md`(출하된 SH-L1+차폐),
  `docs/scalable-gi.md`, `docs/gdf-reference-alignment.md`, **이 문서**.
- 레퍼런스: `UnrealEngine-1/Engine/Shaders/Private/Lumen/LumenScreenProbe*`, `LumenRadianceCache*`,
  `Engine/Source/Runtime/Renderer/Private/Lumen/LumenIrradianceFieldGather.cpp`(기본값).
- 메모리: `dreamcoast-build-to-quality`(디렉티브), `dreamcoast-permesh-df-plan`(GI 트랙 전체사),
  `dreamcoast-no-trademark-names`, `dreamcoast-verification-split`, `dreamcoast-metal-milestones`.

## 하지 말 것
- 월드-볼륨 단일 그리드 튜닝으로 회귀(한계 확정). 한계효용으로 기능 되돌리기. path-tracer 패리티 없이 단정.
  상표명을 산출물에 쓰기. 갤러리 앵커 깨기.
