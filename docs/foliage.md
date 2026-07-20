# Foliage — alpha-tested cutout + hashed-alpha soft edges

상위: [ROADMAP.md](ROADMAP.md) · 관련: [deferred-decals.md](deferred-decals.md)(BLEND 분류·트랙 B 맥락),
[phase-6-pbr.md](phase-6-pbr.md)(deferred G-buffer / masked alpha test),
[phase-11-asset-pipeline.md](phase-11-asset-pipeline.md)(glTF 머티리얼 임포트).

상태: **A(컷아웃) + C(해시드 알파 소프트엣지) 구현·Metal 검증 완료. Windows DX≡VK 게이트 대기.**
브랜치 `feature/foliage-cypress-trees` (커밋 `c2f25dc` 컷아웃 / `c6e94dd` 해시드).

## 문제

Intel Sponza 사이프러스 나무 팩(`pkg_c_trees`)의 `LeafSpring` 머티리얼은 `alphaMode=BLEND` +
`doubleSided=true`, 잎 실루엣은 **baseColorTexture의 알파 채널**에 있음(baseColorFactor α=1.0, 자체
`alphaCutoff` 없음). 데칼 작업이 BLEND∧¬decal을 `MaterialKind::Transparent`로 분류하되 렌더는
**opaque 폴백**이었으므로, 잎 알파가 무시돼 **불투명한 사각 잎 카드**(블로키 캐노피 + 솔리드 잎 카펫)로
깨져 보였다. 딥퍼드(G-buffer) 파이프라인이라 진짜 OIT는 비용이 큼 → 폴리지는 알파-테스트 컷아웃이 정공법.

나무 월드 AABB: X∈[-3.05,3.13], Y∈[0,14.83], Z∈[-3.0,3.17], 중심≈원점, 높이 ~14.8 m. 커튼처럼 New
Sponza 공유 좌표계에 원점 저작 → **identity 배치 = 신랑(nave) 중앙**.

## 설계

### A — 알파-테스트 컷아웃 (기본 on, 정확성)
**기존 Masked 경로 전체 재사용.** `crates/asset/gltf_scene.rs`에서 `Transparent`(비-데칼 BLEND)에
단일소스 상수 `FOLIAGE_ALPHA_CUTOFF`(0.5, glTF MASK 기본과 동일)를 임포트 시 파생. 그 cutoff가
`MaterialDesc→SceneObject.alpha_cutoff`로 흘러 **이미 존재하던** `gbuffer.slang fsMain` /
`shadow.slang fsMain`의 `cutoff>0` discard를 그대로 탄다 → 잎 실루엣 + 잎 모양 그림자 복원. 셰이더·
파이프라인·RHI 무변경. cull은 3 백엔드 모두 NONE(양면 무료). 갤러리/sponza_intel은 `Transparent`가
0개라 **바이트 동일**.

### C — 해시드/스토캐스틱 알파 (opt-in, 소프트 엣지)
A는 가장자리가 crisp(하드/에일리어싱). C는 **부드러운 반투명 가장자리**를 정렬 없이 얻는다(Wyman &
McGuire 2017, MSAA 대신 TAA로 resolve). 기본 경로 바이트 동일 + 크로스백엔드 결정성을 위해:

- `gbuffer.slang`: cutoff를 **부호로 분기**(푸시 필드 그대로, 레이아웃 무변경) — `0` opaque /
  `>0` 하드 컷아웃 / `<0` 해시드. 해시드 분기는 `alpha < hashed_alpha_threshold(world_pos)`면 discard.
  **월드 좌표 PCG 해시**(스크린 SV_Position 아님 → VK Y-flip 모호성 없음, 순수 정수연산 → SPIR-V/DXIL
  비트 동일). ~0.25 mm 양자화로 이웃 픽셀 디코릴레이션 + 카메라 서브픽셀 **TAA 지터**가 매 프레임 임계를
  리샘플 → 누적이 이진 keep/discard를 분수 커버리지(=소프트 엣지)로 수렴.
- `shadow.slang`: **`|cutoff|` 하드 테스트** — 섀도맵은 TAA resolve 대상이 아니므로 해시드 폴리지도
  crisp·노이즈 없는 컷아웃 그림자.
- sandbox: `FOLIAGE_HASHED=1`이 씬 빌드 시 폴리지 cutoff를 음수로 플립(한 곳). 디더 resolve엔
  **`P_TAAU_FORCE=1`(네이티브 해상도 TAA)** 페어링 필요. 기본 off = crisp 컷아웃 + 무회귀 베이스라인.

## 씬
`sponza_trees` 레벨(= main + 커튼 + 사이프러스, 전부 identity). `sponza_intel`(데칼/AO 검증 베이스라인)을
건드리지 않으려고 **분리**. 카메라는 sponza_intel 복도뷰(eye `7,2.2,0`→`-15.84,2.27,0`)로 나무가 ~7 m
앞 중앙에 오게 함. 실행: `EV100=11 LEVEL=sponza_trees [FOLIAGE_HASHED=1 P_TAAU_FORCE=1] sandbox
--backend metal --screenshot-clean out.png` (RELEASE 필수 — 64프레임 GI/TAA 워밍업; 첫 실행 SDF/albedo
cook으로 느림).

## 검증 (Metal, macOS M3)
- **A**: 캐노피·바닥 리터가 불투명 카드 → 잎 모양 컷아웃(가장자리 사이 투과). 잎 모양 그림자는
  마스크-섀도 경로(`|cutoff|`) 재사용으로 동작(잎과 동일 알파텍스처·샘플러).
- **C**: `FOLIAGE_HASHED=1 P_TAAU_FORCE=1`로 사이프러스 프론드가 crisp/에일리어싱 → **소프트 페더리
  반투명 가장자리**(캐노피-엣지 크롭 비교), 리터가 석재로 페이드. **런-투-런 비트 동일**(해시 결정적).
- **무회귀**: 공유 셰이더(gbuffer/shadow) 편집 후에도 **갤러리 + sponza_intel = main과 바이트 동일
  (SHA-256 일치)** — 두 씬은 `Transparent` 0개, 분기는 음수 cutoff에서만 실행.
- `clippy --all-targets -D warnings` / `fmt` 클린, asset 유닛테스트 통과.

## Windows DX≡VK 게이트 (대기)
- VK/D3D12 빌드 + `clippy --all-targets`(과거 stale 테스트 적발 지점) + `sponza_trees` DX≡VK ≤0.001/ch
  (해시드 on/off 둘 다) + 갤러리/sponza_intel 무회귀 바이트 동일 + VK검증/DX 디버그레이어 무에러.
- **해시 결정성 근거**: 월드좌표 + 순수 정수 PCG(스크린좌표/Y-flip/프레임카운터 항 없음)라 설계상
  SPIR-V≡DXIL. TAA(TAAU)는 기존 DX≡VK 검증 기능. 지터 시퀀스(Halton 8)도 백엔드 공유.

## 스케일링 / 노브 (한 곳)
- `FOLIAGE_ALPHA_CUTOFF`(asset, 단일소스) — 컷아웃 임계.
- `FOLIAGE_HASHED`(sandbox env, 기본 off) — 소프트 엣지 opt-in. `P_TAAU_FORCE`와 페어링.
- 추후 `RenderQuality` 티어 연동(Low에서 해시드 off 등) 가능 — 모두 폴리지 일반(씬별 패치 없음).

## Ivy + 히어로 씬 (`sponza_hero`)
New Sponza **ivy growth** 팩(`pkg_b_ivy`) 추가. **사이프러스와 다름**: `IvyLeaf` 머티리얼은
`alphaMode=OPAQUE` + **4.9 M-tri 모델링된 잎 지오메트리**(잎 실루엣이 실제 메시, UV가 아틀라스의 잎
영역만 샘플 — baseColor는 RGB, 알파 없음). 따라서 **알파테스트 불필요** = 일반 OPAQUE glTF로 그대로
들어오고 폴리지 컷아웃이 안 건드림(정상). 라이아나/잎 in-place 저작(X≈-7, Y≈4–18, 사자 끝 아케이드를
타고 오름). 무거움(`.bin` ~270 MB, 첫 로드 SDF/albedo cook). `assets/`는 gitignore — ivy는
`~/assets/pkg_b_ivy`에서 심링크.

`sponza_hero` 레벨 = main + 커튼 + 사이프러스 + ivy(전부 identity). **README 배너**
(`docs/media/sponza.png`) 교체용 히어로: 역방향 콜로네이드(사자 끝→입구 방향, ivy가 전경 아치를 덮고
사이프러스가 신랑 중앙, 색 커튼이 양옆)를 레벨 카메라에 베이크. 재현(F6L 레시피 — level.rs 주석이
단일 소스): `AUTO_EXPOSURE=1 AO_STRENGTH=1.0 AO_FLOOR=0.6 P_SKYVIS_BENT_FLOOR=0.25
RENDER_SCALE=1 WARMUP_FRAMES=192 LEVEL=sponza_hero … --screenshot-clean hero.png` (2560×1440 렌더 → 1920×1080
다운스케일). 구 레시피(EV100=12, 커밋 `895b4e0`)는 F6 재캘리브 이후 저노출.

## 후속
- 진짜 유리/투명(트랙 B 포워드 OIT)은 별개 — `Transparent` 중 폴리지가 아닌 자산이 생기면 분류를
  세분화(현재는 모든 비-데칼 BLEND를 컷아웃/해시드로 처리, 딥퍼드에서 안전한 일반 폴백).
- 바람에 흔들리는 잎(버텍스 애니메이션)·LOD·임포스터는 게임플레이 단계 확장.
