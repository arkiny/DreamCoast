# 투명(Translucency) 패스 슬롯 — PR-3

상위: [render-pipeline-reference.md](render-pipeline-reference.md) §1.7 #12 · §2 표 #12/#21 · §3 PR-3.
관련: [atmosphere-fog-slot.md](atmosphere-fog-slot.md)(PR-4, 바로 앞 슬롯) · [foliage.md](foliage.md)(alpha-cutout 경로, 이 슬롯과 별개).

디퍼드 셰이딩은 G-buffer 텍셀당 하나의 불투명 표면만 저장하므로 반투명 지오메트리를 담을 수 없다.
PR-3은 레퍼런스 디퍼드 렌더러의 canonical 하이브리드 해법 — **불투명 씬 컬러가 완성된 뒤(디퍼드
라이팅+반사+포그) post(TAAU/톤맵) 앞에 별도의 포워드 투명 패스**를 그래프에 삽입한다.

## 1. 설계 결정과 근거

리서치(3dgep Forward+, KTH "Transparency with Deferred Shading", Vulkan Docs "Forward vs Deferred")로
확인한 canonical 접근: **디퍼드는 투명을 네이티브로 못 다루므로 투명 지오메트리는 포워드 방식으로
별도 패스에서 그린다.** 정렬 알파 블렌드는 프래그먼트가 back-to-front 순서일 때만 올바르다.

- **위치:** 불투명 완성(라이팅→반사→포그) **뒤**, post **앞**. HDR 씬 컬러에 in-place 블렌드.
  근거: 투명은 포그 낀 배경 **위에** 올라가야 하고(순서 정합), TAAU/톤맵 이전의 linear-HDR에서
  합성돼야 노출/블룸이 투명 표면에도 일관되게 적용된다.
- **래스터 상태:** `depth-test ON` + `depth-write OFF`. depth-test는 공유 불투명 depth(`g_depth`)에
  대해 수행 → 투명이 불투명 뒤에 있으면 가려진다. depth-write는 끔 → 겹친 투명 레이어가 서로
  Z-reject 없이 전부 블렌드된다. G-buffer는 기록하지 않는다(디퍼드 라이팅 밖).
- **블렌드:** `BlendMode::AlphaBlend` = color `(SRC_ALPHA, ONE_MINUS_SRC_ALPHA)`, alpha
  `(ONE, ONE_MINUS_SRC_ALPHA)`. 셰이더는 straight(비-premultiplied) 컬러 + 커버리지 알파를 출력.
- **정렬:** CPU에서 카메라까지의 거리제곱 기준 **back-to-front**(먼 것 먼저). tie-break는 오브젝트
  원본 인덱스 → run-to-run·백엔드 간 **결정론적**(coincident 평면도 안정 순서). `translucent.rs::record`.
- **라이팅:** 단순 포워드 PBR. 디렉셔널 직접광(공유 shadow map 하드 3×3 PCF) + IBL ambient
  (irradiance/prefilter 큐브). **직접광 BRDF는 디퍼드 패스와 동일 소스**(`pbr_brdf.slang`) —
  `distribution_ggx`/`geometry_smith`/`fresnel_schlick`/`shade`를 헤더로 추출해 pbr.slang과 공유
  (드리프트 방지, 단일 소스). 노출은 디퍼드가 라이팅에 baked-in한 값과 동일하게 `globals.ambient.a`.

## 2. 구현 맵

| 요소 | 위치 |
|---|---|
| 포워드 셰이더(VS/FS) | `crates/shader/shaders/translucent.slang` |
| 공유 BRDF 헤더(단일 소스) | `crates/shader/shaders/pbr_brdf.slang` (pbr.slang·translucent.slang 공용) |
| 파이프라인 + `record` + 정렬 + 테스트 평면 | `apps/sandbox/src/translucent.rs` |
| push 패커(176B) | `apps/sandbox/src/push.rs::translucent_push` |
| 프레임 배선(포그 뒤·TAAU 앞) | `apps/sandbox/src/main.rs` (`self.translucency.record(...)`) |
| 셰이더 등록 | `crates/shader/build.rs` (`translucent_vs`/`translucent_fs` + `SHARED_INCLUDES`에 `pbr_brdf.slang`) |

## 3. opt-in 시임 / 비용

- 투명 오브젝트가 **0개면 `record`가 패스를 추가하지 않는다** → 비용 0, 디폴트 출력 **바이트 동일**
  (갤러리 앵커 `af70c1a5…` 유지).
- **`P_TRANSLUCENT_TEST=1`** — 반투명 예시 애셋이 아직 없으므로 갤러리에 유리 같은 반투명 평면 2장을
  스폰(살짝 기울여 겹치게 배치, alpha 0.4). 겹침 정렬·depth 가림·포그 상호작용 검증용.
- **부수효과(§2 표 #21):** Phase-7 파티클/GPU-컬링 draw가 종전 **톤맵 후 LDR**에 그려지던 문제를,
  이 투명 슬롯(HDR, 톤맵 전)으로 이동해 바로잡았다. 둘 다 **디폴트-off**(env `P7_PARTICLES`/`P7_CULL`)
  이므로 디폴트 출력은 불변 → opt-in 시임 불필요(그대로 이동, HDR_FORMAT로 파이프라인 재구성).

## 4. glTF `Transparent`(BLEND) 라우팅 골격

머티리얼 kind 분기 골격은 마련돼 있다: `TranslucentObject::from_scene(&SceneObject)`가 불투명 draw
리스트의 BLEND 드로어블을 포워드 머티리얼로 변환한다. **디폴트로 실행되지 않는다** — 활성화하려면
(a) 그 오브젝트의 **불투명 G-buffer draw를 억제**해야 하고(안 그러면 이중 렌더), (b) foliage 경로가
`Transparent` 드로어블을 양의 `alpha_cutoff`로 **불투명 G-buffer에 남겨두는 것에 의존**한다(크리스프
컷아웃/해시드 소프트 엣지, [foliage.md](foliage.md)). 따라서 gltf 라우팅 실배선은 G-buffer 억제까지
포함하는 Phase 20 작업으로 미룬다. 현재 PR-3은 foliage 동작을 전혀 바꾸지 않는다.

## 5. TAAU 상호작용 (알려진 한계)

투명 표면은 **velocity를 기록하지 않는다**(PR-2 velocity RT는 불투명 전용). TAAU가 켜진 상태
(`P_TAAU=1` 또는 업스케일 활성)에서 투명 표면이 화면상 이동하면 velocity가 0(=카메라 리프로젝션만)
이라 히스토리 리프로젝션이 어긋나 **고스팅**이 생길 수 있다. 정적 카메라/정적 투명(테스트 평면)에서는
문제 없음. 투명 velocity는 Phase 20 확장 지점(투명 velocity RT + TAAU dilate 편입).

## 6. Phase 20 확장 지점

- OIT(순서 독립 투명): weighted-blended 또는 per-pixel linked-list — CPU 정렬 제거.
- 굴절(refraction): 씬 컬러 back-buffer 샘플 + 노멀 왜곡(single-layer water 포함).
- 전용 translucency lighting volume(froxel irradiance) — 다광원 투명 라이팅.
- 투명 velocity RT + TAAU 편입(§5 고스팅 해소).
- glTF BLEND 실라우팅(§4) + 불투명 G-buffer 억제.

## 7. 검증 (Metal)

- clippy `-D warnings` 클린 + fmt.
- 디폴트 갤러리 골든 앵커 sha256 == `af70c1a5c8db49661d2c7926140c1309c28fda04c82cc1ab8aa6638d588b2b74`
  바이트 동일(투명 0개 → 패스 미스케줄).
- `P_TRANSLUCENT_TEST=1`: 반투명 평면 너머 뒤 오브젝트가 알파 블렌드로 비침, 겹친 두 장의 근접 장이
  위, 불투명 뒤 가려짐(depth-test).
- `P_TRANSLUCENT_TEST=1 P_HEIGHT_FOG=1`: 투명이 포그 낀 배경 위에 올라감(순서 정합).

**DX/VK parity pending Windows verification.**
