# Phase 11 — 소프트웨어 레이트레이싱 + Distance-Field GI (세부 계획 / 스텁)

상위: [ROADMAP.md](ROADMAP.md) Phase 11. **전제: Phase 7(컴퓨트)**. Phase 8 HW RT와 **별개 경로** —
하드웨어 RT 없이 컴퓨트만으로 동적 GI/반사/AO를 근사한다(넓은 씬·저사양 타깃). 무편향 패스트레이서
([rt-pbr-parity.md](rt-pbr-parity.md))가 정답 레퍼런스.

> 목표: **씬의 전역 거리장(Global Distance Field)을 생성**하고, 그 거리장을 **컴퓨트 셰이더로
> ray-march(소프트웨어 레이트레이싱)** 하여 **stochastic(몬테카를로) 라이팅**으로 동적 GI를 구한다.
> 단, 먼저 **컴퓨트 셰이더로 레이트레이싱을 구현하는 기반**(Stage A)을 만든 뒤 거리장 GI로 확장한다.

순서: **A(컴퓨트 SW RT) → B(GDF) → C(Stochastic Lighting)**. 각 스테이지 양 백엔드 + 검증 클린 게이트.

## Stage A — 컴퓨트 소프트웨어 레이트레이싱
HW RT 파이프라인(Phase 8) 없이 컴퓨트 셰이더로 레이를 추적하는 기반.
- 1차 접근: **거리장 ray-marching**(sphere tracing) — Stage B의 SDF를 그대로 쓰는 자연스러운 경로.
- 대안/병행: 컴퓨트 BVH 트래버설(삼각형 정확). 우선순위는 SDF 마칭(거리장 GI와 자연스럽게 정합).
- 검증: Phase 8과 동일 씬에서 1차 가시성/그림자 결과 대조.

**부트스트랩 결정:** SDF 마칭은 거리장이 필요한데 거리장(GDF)은 Stage B에서 만든다. 그래서 Stage A는
**해석적 SDF 프리미티브**(구/박스/평면 = 샘플 씬 레이아웃을 미러링; model_radius=1.0 정규화라 좌표가 상수)로
부트스트랩한다 — GDF 베이크 없이 컴퓨트 RT 기계(카메라 레이 생성, march 루프, gradient 노멀, storage-image
출력)를 세우고 검증한 뒤, Stage B가 해석적 프리미티브를 베이크된 per-mesh SDF + 클립맵으로 교체한다.

마일스톤: **A1 1차 가시성 → A2 소프트 섀도우 + AO → A3(선택) 컴퓨트 BVH 삼각형 트래버설**.

### A1 — 1차 가시성 ✅ (양 백엔드 검증, `925c266` 다음 커밋)
- 신규 `crates/shader/shaders/sdf_trace.slang` (`sdf_trace_cs`): 픽셀당 1차 카메라 레이를 해석적 SDF 씬에
  **sphere-trace**(최대 192 스텝). 히트 시 중앙차분 gradient = 노멀, miss 시 패스트레이서와 동일한 `sky`.
  셰이딩 = Lambert 태양 + 단순 반구 스카이 앰비언트(섀도우/AO는 A2). 카메라 레이 재구성은 `rt_common`의
  `primary_ray_dir`과 동일(inv_view_proj·z=1). raw radiance 출력 → 기존 tonemap이 노출+ACES 적용.
- TLAS/머티리얼 테이블 plumbing 없음 — `bindless.slang`의 `storage_images[]`만 사용, `compute_supported`
  게이트(순수 컴퓨트). push 112B(`sdf_trace_push`, rt_trace와 동일 레이아웃 + sun.w=강도). 통합은 M3
  `rt_trace` viz와 동형: 컴퓨트 패스가 `sdf_out` storage image에 쓰고 tonemap이 HDR 대신 표시
  (`rt_out.or(sdf_out).or(hdr_post)`). 토글: env `P11_SDF` + UI "Software ray tracing (Phase 11)".
- **검증(RTX 2070 SUPER):** build+fmt+clippy(-D warnings) 클린. VK·DX 렌더 정상(구3+박스+그라운드+스카이,
  매끄러운 gradient 노멀). **VK≡DX: 920k 픽셀 중 1픽셀만 >2 차이**(실루엣 엣지 1px — 반복 march의
  SPIR-V/DXIL fp contraction 차이; mean 0.0002/ch). Vulkan VUID 0. SDF-off 기본 래스터 씬 byte-identical
  (회귀 없음). 한계: 아보카도 메시는 구 프록시, 위치는 근사 — 메시 픽셀 매치는 Stage B(GDF 베이크)에서.

### A2 — 소프트 섀도우 + AO ✅ (양 백엔드 검증)
- `sdf_trace.slang`에 두 SDF-march 보조 함수 추가:
  - `soft_shadow(origin, dir, k)` — 태양 방향으로 shadow ray를 sphere-trace하며 표면 최근접 거리를
    `k*h/t`로 추적(Inigo Quilez penumbra). [0,1] 가시성 = 부드러운 그림자 가장자리. `k=24`로 PT의 ~1.1°
    디스크 태양 penumbra에 근사. (히트가 나오면 0 = 완전 차폐.)
  - `calc_ao(p, n)` — 노멀 방향 5탭 마칭으로 기대 자유거리 vs 실제 필드값 비교(IQ AO). 스카이 앰비언트를
    변조해 접촉부/주름이 어두워짐.
- 셰이딩: 태양 항 `*= shadow`(ndl>0일 때만 trace), 스카이 앰비언트 `*= ao`. A1 대비 추가만 — 1차 가시성/
  노멀/스카이는 동일.
- **검증(RTX 2070 SUPER):** build+fmt+clippy(-D warnings) 클린. VK·DX 각 구·박스 아래 **소프트 컨택트
  섀도우 + AO 어두워짐** 정상(태양 좌상단 방향과 일치). **VK≡DX: 920k 중 1픽셀만 >2**(엣지; mean
  0.0003/ch, max 61). Vulkan VUID 0. A1→A2 차이는 섀도우/AO 영역에 국한(mean 1.24/ch). 스크린샷
  tmp/sdf2-{vk,dx}.png. **Stage A 기계(컴퓨트 1차+2차 SDF march) 완성** — A3(컴퓨트 BVH 삼각형)은 선택,
  거리장 GI 정합을 위해 Stage B(GDF 베이크)로 바로 진행 가능.

## Stage B — Global Distance Field
해석적 프리미티브(Stage A)를 **실제 메시 거리장**으로 교체한다. per-mesh SDF를 베이크하고, 카메라 주변을
덮는 클립맵 3D 볼륨으로 합성한 뒤, Stage A의 sphere-trace를 그 볼륨 샘플로 바꾼다.

마일스톤(각 양 백엔드 + 검증 클린 게이트, phase-by-phase 승인):

### B1 — 3D 볼륨 텍스처 RHI ✅ (양 백엔드 검증)
- `bindless.slang` 블록에 `Texture3D<float> volumes[64]`(binding 6) + `RWTexture3D<float>
  storage_volumes[64]`(binding 7) 추가. **slangc 리플렉션으로 register 매핑 검증**: SPIR-V binding 6/7,
  DXIL `volumes`=`t1089,space1`(TLAS t1088 다음), `storage_volumes`=`u128,space1`(storage_buffers u64–127
  다음). 신규 포맷 `Format::R32Float`(단일채널 거리; half R16F는 후속 최적화).
- 양 백엔드 리소스: Vulkan `VK_IMAGE_TYPE_3D` 이미지+3D 뷰(`volume.rs`), bindings 6/7 디스크립터
  레이아웃+풀+`register_volume`/`register_storage_volume`. D3D12 `Texture3D` 리소스+SRV(`TEX3D_SRV`)
  +UAV(`TEX3D_UAV`, WSize=depth), 힙 영역 `VOLUME_BASE`/`STORAGE_VOLUME_BASE`, **루트시그니처
  bindless_ranges 5→7**(volumes SRV t1089 + storage_volumes UAV u128). 파사드 `Volume` enum +
  `create_volume` + `volume_to_storage`/`volume_to_sampled` 배리어(상태 추적 Cell, 2D storage RT 미러링).
  Metal은 스텁(`unimplemented!` — argument buffer 볼륨 슬롯은 메탈 세션이 구현). 3D 디스패치는 기존
  `dispatch(x,y,z)` 재사용. **→ Metal 구현 완료** (아래 "Metal 백엔드" 참조; M3 box 검증).
- 스모크 테스트 `volume_test.slang`(`fillMain`/`viewMain`): 컴퓨트가 storage_volume에 중심 구 부호거리
  기록 → `volume_to_sampled` 배리어 → `volumes[]` SRV를 Z=0.5 슬라이스 트라이리니어 샘플 → 화면.
  그래프 통합은 `import_external`로 fill→view 순서 보장, tonemap `rt_out.or(sdf_out).or(vol_out)`. 토글
  env `P11_VOLUME_TEST` + UI.
- **검증(RTX 2070 SUPER):** build+fmt+clippy(-D warnings) 클린. VK·DX 슬라이스 정상(중심 구 부호거리
  그라데이션 + zero 등위면 녹색 링). **VK≡DX 픽셀 동일(mean 0.0000, max 0)** — 결정적 fill+트라이리니어
  샘플. Vulkan VUID 0. **bindless 블록에 멤버 추가했지만 기존 래스터 씬 byte-identical(회귀 없음)** —
  Slang이 미사용 binding을 drop. tmp/vol-{vk,dx}.png.

### B2 — per-mesh SDF 베이크 ✅ (양 백엔드 검증)
- `sdf_bake.slang`(`bakeMain`, `[numthreads(4,4,4)]`): voxel당 1스레드, voxel center를 볼륨
  AABB로 매핑 → **brute-force 점→삼각형 최소 거리**(closest-point-on-triangle, Ericson). 부호는
  최근접 삼각형의 **저장된 정점 노멀**(면-평균, outward)로 결정: `dot(p-q, n) < 0 ⇒ 내부(음수)`.
  cross(b-a,c-a) 면법선 대신 정점 노멀을 쓴 이유 — uv_sphere 와인딩이 내부 방향이라 부호가 뒤집히고,
  공유 엣지/정점에서 면법선 방향이 미정의이기 때문(와인딩-독립, 첫 검증에서 부호 반전 발견 후 수정).
  O(voxels·tris) 1회 베이크(JFA는 후속 최적화).
- 메시 정점/인덱스는 bindless storage buffer로 업로드 — 래스터/HW RT와 **동일 32B 정점 스트라이드**
  (pos@0, normal@12)라 지오메트리 1회 업로드로 모든 경로 공유. 베이크 메시는 unit uv-sphere(48×32)를
  반경 0.3·중심 (0.5,0.5,0.5)로 스케일 → B1의 해석적 중심 구와 동일 필드(직접 대조용). 볼륨 AABB는 단위 큐브.
- 그래프 통합: 베이크는 비싸므로 1회만(`sdf_bake_done`), 이후 프레임은 B1 `volume_view` 패스 재사용으로
  슬라이스 표시(베이크↔해석적 픽셀 비교 + VK≡DX). 토글 env `P11_SDF_BAKE` + UI. tonemap 체인에 `bake_out` 추가.
- **검증(RTX 2070 SUPER):** build 클린. **VK≡DX 픽셀 동일(max 0)** — 결정적 베이크. 베이크 SDF는
  해석적 구와 **영점 등고선의 얇은(1–2px) 링을 제외하면 완전 일치**(내부/외부 그라데이션 black diff);
  남은 차이는 면분할 메시 등고선이 이상적 구와 서브픽셀 어긋난 것(기대된 테셀레이션 오차). max shade diff 58/255.

### B3 — 전역 거리장 머지 ✅ (양 백엔드 검증)
- `gdf_merge.slang`(`mergeMain`, `[numthreads(4,4,4)]`): GDF voxel당 1스레드, world-space voxel
  center를 각 인스턴스의 로컬 bake 박스로 변환(`uvw = (p - origin)·inv_extent`) → 인스턴스의
  per-mesh SDF 볼륨을 트라이리니어 샘플 → bake 거리에 `dist_scale`을 곱해 월드 단위로 환산 →
  **모든 인스턴스에 대해 min-결합**. 인스턴스는 voxel이 자기 bake 박스 내부(uvw∈[0,1])일 때만 기여
  (박스 밖은 트라이리니어 clamp가 거리를 과소평가하므로 skip=+inf). 유효 GDF 영역 = 인스턴스 박스들의
  합집합 — 실제 클립맵이 카메라 주변을 타일링하는 방식 그대로.
- 인스턴스 테이블은 32B 레코드 storage buffer: `origin`(bake 박스 min 코너의 월드 위치) + `dist_scale`
  + `inv_extent`(월드 delta→uvw) + per-mesh SDF의 sampled 인덱스. B2 베이크(`volume_res`)를 소스로,
  머지 타깃은 별도 `gdf_res` 볼륨. 그래프는 bake→merge→view를 `import_external("volume"/"gdf")`로 순서화.
  bake+merge는 1회(`gdf_merge_done`), 이후 프레임은 영속 GDF를 B1 `volume_view`로 재표시. 토글 env
  `P11_GDF_MERGE` + UI. 인스턴스 셋은 `P11_GDF_INSTANCES=1`(전체-큐브 단일 = 회귀 앵커) / 기본(반-크기 구 3개).
- **첫 구현 범위:** 단일 글로벌 레벨 + 정적 인스턴스 1회 머지. **멀티 해상도 클립맵 + 동적 인스턴스
  per-frame 갱신은 후속 정제**(미결 항목 참조).
- **검증(RTX 2070 SUPER):** build+fmt+clippy(-D warnings) 클린. **단일 전체-큐브 인스턴스 = B2 베이크
  볼륨과 픽셀 완전 동일(max 0)** — 머지 변환/샘플/스케일 수학이 정확함을 증명하는 회귀 앵커.
  3-인스턴스 머지는 각 인스턴스 변환 위치에 구가 배치되고 겹치는 박스에서 min-union 정상. **VK≡DX:
  녹색 링 분류 밴드(`abs(d)<0.01`) 제외 시 픽셀 완전 동일(max 0)**; 전체 프레임 기준 단 1픽셀만 차이
  (그 한 픽셀이 링 임계값의 knife-edge에 놓여 백엔드 간 1-ULP로 분류가 뒤집힘 — 거리장 자체는 동일).

### B4 — GDF SW RT ✅ (양 백엔드 검증)
- `gdf_trace.slang`(`csMain`, `[numthreads(8,8,1)]`): Stage A SW-RT 머신(카메라 레이 생성, sphere-trace
  march, gradient 노멀, 페넘브라 소프트 섀도우, 마치 AO)을 **그대로** 쓰되 `scene_dist`가 분석적
  프리미티브 대신 **머지된 GDF 볼륨을 트라이리니어 샘플**(world→uvw, B3 AABB) → 비로소 베이크된 실제
  지오메트리(B2 베이크→B3 머지→여기)를 SW 레이트레이스. 지면 평면과 min-union해 그림자 리시버 제공.
- **march용 거리와 occlusion용 거리 분리**가 핵심: 1차 march는 GDF 박스 밖에서 박스 거리로 전진하되
  박스 경계가 hit이 되지 않게 hit-epsilon 위로 클램프(`geo_march`); AO/섀도우/노멀은 박스를 솔리드로
  보지 않도록 박스 밖을 +large로 반환(`geo_inside`/`scene_occ`) — 안 그러면 지면에 박스의 정사각
  등거리선이 찍힌다(개발 중 실제로 관측·수정). 노멀 epsilon은 복셀 크기(~1/64)에 맞춰 셀 간 평균
  (sub-voxel이면 단일 trilinear 셀만 읽어 표면이 계단짐).
- 푸시(128B)는 `sdf_trace_push` 헤드 레이아웃 + `gdf_sampled` + `mode`. `mode` bit0은 GDF 샘플을
  **베이크 원본인 분석적 구로 스왑**(B4 정합 레퍼런스). 카메라는 단위 큐브 씬을 프레이밍하는 고정
  카메라(궤도 카메라와 동일 Y-flip 규약으로 VK/DX 동일 월드 레이). 토글 env `P11_GDF_TRACE`
  (+ `P11_GDF_ANALYTIC`, `P11_GDF_INSTANCES=1`로 전체-커버리지 GDF) + UI. 영속 GDF는 1회만 빌드.
- **검증 범위:** 전체-커버리지(단일 인스턴스) GDF로 검증 — 모든 voxel이 유효 거리라 march가 overshoot
  없이 견고. **희소 멀티-박스 클립맵의 견고한 트레이싱(빈 영역 거리 채움)은 후속 정제.**
- **검증(RTX 2070 SUPER):** build+fmt+clippy(-D warnings) 클린. GDF 트레이스가 깔끔한 구(소프트
  섀도우 + 접촉 AO)를 렌더, **같은 카메라의 분석적 레퍼런스(`mode=1`)와 근접 일치** — 차이는 64³
  해상도의 표면 셰이딩 미세 밴딩 + 실루엣(>8 차이 1279px), 즉 의도된 복셀화 오차(실루엣/셰이딩 모델은
  동일). **VK≡DX: 921,600px 중 >8 차이 14px**(모두 구 실루엣/터미네이터의 가장 가파른 노멀 지점 —
  iterative march의 step 수가 SPIR-V/DXIL 빌드 간 sub-ULP로 갈린 것이지 거리장 차이가 아님; B1–B3는
  분기 없는 trilinear라 bit-identical, march는 데이터 의존 반복이라 소수 엣지 픽셀 발생).

신규 RHI: 3D(볼륨) 텍스처 + UAV, 3D 디스패치. (Phase 7 storage image의 3D 확장.)

> **GDF 베이크 영속화는 별도 워크스트림으로 승격됨 → [Phase 12 — 에셋 파이프라인](phase-12-asset-pipeline.md).**
> 사용자 요청대로 SDF 베이크만이 아니라 **메시까지 함께 직렬화하는 쿠킹된 에셋(`.dcasset`)** 개념이라
> 규모가 커서 `crates/asset`의 크로스컷팅 인프라(독립 Phase)로 분리했다. Stage B의 per-mesh SDF 베이크
> 결과가 Phase 12 M2의 SDF 청크로 영속화된다. (메시 직렬화 M1은 Phase 11과 독립적으로 먼저 가능.)

## Stage C — Stochastic Lighting
GDF를 ray-march해 **디퓨즈 GI(1+ 바운스)·AO·반사**를 stochastic 샘플하고, 결과를 디퍼드
라이팅(Phase 6)의 ambient/GI 항으로 합성한다. Stage B의 GDF는 *단위 큐브 데모*(고정 카메라)였으므로
Stage C는 먼저 **실제 씬을 월드 공간 GDF로 굽고**, 그것을 **실제 디퍼드 G-buffer**에서 march한다.

### Stage C의 최종 목표 — 캡처 기반 IBL을 SW-RT로 대체 (사용자 확정 2026-06-26)
지금까지 [rt-pbr-parity.md](rt-pbr-parity.md)(PT가 정답)와 [realtime-env-capture.md](realtime-env-capture.md)
(캡처 기반 IBL)가 거듭 확인한 결론: **split-sum 큐브 IBL은 PT 반사를 근본적으로 못 따라간다** — 단일 프로브
시차 오차, 이웃 오브젝트 미반영, 프리필터 블러, 구/박스 프록시 부정합. 컴퓨트 SW RT(Stage A)가 잘
동작하므로(A1/A2 ✅), Stage C는 **캡처 기반 IBL의 디퓨즈·스페큘러 항을 SW-RT 결과로 교체**하는 것을 최종
목표로 한다:
- **디퓨즈 간접광(IBL irradiance) → GDF 디퓨즈 GI**(C3/C4). 캡처 큐브의 씬 캡처가 불필요해진다.
- **스페큘러 반사(IBL prefilter) → 하이브리드 반사**: **(1) SSR**(스크린-스페이스, 온스크린 정확
  반사 — 캡처 시차/이웃 오브젝트 오류를 직접 해소) **+ (2) GDF 반사**(오프스크린/디스오클루전 폴백, SW-RT)
  **+ (3) 절차적 스카이**(레이 miss 폴백). 캡처 env 큐브는 **스카이 전용 폴백/디퓨즈 스카이 irradiance**로
  격하(시차·프록시·이웃반사 한계 모두 은퇴).
- 절차적 스카이는 레이 miss 라디언스로 그대로 유지 — "IBL을 없앤다"가 아니라 **씬 캡처를 SW-RT로
  대체하고, 큐브는 하늘만 담당**하게 한다.

마일스톤(각 양 백엔드 + 검증 클린 게이트, phase-by-phase 승인):

### C1 — 월드 공간 씬 GDF ✅ (양 백엔드 검증)
- 샘플 씬의 불투명 오브젝트(아보카도/구×2/큐브)를 **하나의 월드 공간 삼각형 수프로 융합** → 씬 AABB
  위에 brute-force 베이크(`sdf_bake.slang` 재사용, AABB를 push로 일반화)하여 단일 월드 GDF 볼륨 생성.
  per-mesh SDF + 클립맵 머지(동적 오브젝트)는 후속 정제; 지면은 트레이스 시 해석적 평면(y=0)으로 min-union.
  TDR 회피로 `SCENE_DIM=48`(융합 ~6.8k tris × 48³ ≈ B2 베이크와 동급). 융합은 App에서 정점/노멀을 월드로
  변환(이동+균등스케일이라 노멀 통과), 디스조인트 오브젝트라 closest-triangle 부호 = union SDF.
- 검증은 `gdf_trace.slang`을 **라이브 궤도 카메라**로 트레이스(AABB/지면/거리클램프를 push로 이동,
  B4 단위큐브 경로는 동일 값 전달로 무회귀). 토글 env `P11_SCENE_GDF` + UI "Scene GDF (world, live camera)".
- **검증(RTX 2070 SUPER):** build+clippy(-D warnings) 클린. 라이브 카메라 트레이스가 래스터 씬 레이아웃과
  일치 — **아보카도 실루엣이 실제 메시 베이크**(A1 구 프록시 한계 해소), 구·큐브가 정확한 월드 위치에
  지면 위 소프트 컨택트 섀도우+AO와 함께 렌더. **VK≡DX: 921,600px 중 >8 차이 16px**(B4와 동일 — iterative
  march의 SPIR-V/DXIL sub-ULP step 차이, 거리장 자체는 동일). 기본 래스터/B4 무회귀(default 1px, B4 17px).
  한계: 48³ 복셀화 패싯 + 단위큐브 튜닝된 march/AO 상수(C2에서 G-buffer march로 정밀화).

### C2 — GDF AO → 디퍼드 ambient ✅ (양 백엔드 검증)
- 풀스크린 컴퓨트 패스(`gdf_ao.slang`, `gdf_ao_cs`)가 픽셀별로 씬 GDF를 march해 AO[0,1]를 storage
  image에 출력 → 디퍼드 라이팅(`pbr.slang`)이 ambient/IBL 항에만 곱셈 합성(직접광은 기존 섀도우맵 가시성
  유지). **GDF가 실제 렌더에 처음 영향.** 토글 env `P11_GDF_AO` + UI "GDF ambient occlusion (deferred)",
  디버그뷰 9 "GDF AO".
- **월드 좌표는 depth G-buffer에서 재구성**(`inv_view_proj·(ndc, depth)`), G-buffer position MRT가
  **아니라**. 래스터라이저가 모델 행렬을 클립 위치에만 접어 넣어 position MRT는 오브젝트 공간 →
  변환된 오브젝트(구·큐브)가 월드 GDF와 어긋남. depth 재구성은 모든 오브젝트에 균일하게 참 월드점을 주며
  C1 primary-ray 경로와 동일. 월드 노멀은 G-buffer 그대로(샘플 씬 변환=이동+균등스케일이라 방향 보존).
  AO march: IQ 5-tap, reach=AABB대각×0.07, bias=대각×0.004, strength=1.6(월드 스케일 튜닝, 호스트 상수).
  bake-once 래치는 C1 트레이스와 공유(`scene_gdf_baked`).
- 라이팅 push 24→28B(+`gdf_ao_index`, 부재 시 `0xFFFFFFFF`→곱 1.0=무회귀). AO 패스는 graph에서
  gbuffer→AO→lighting 순서(depth+normal sampled read, AO storage write, scene GDF 1회 bake).
- **검증(RTX 2070 SUPER):** build+fmt+clippy(-D warnings) 클린. 오브젝트 접촉부에 소프트 컨택트 AO가
  ground ambient를 어둡게(아보카도·구·큐브 밑면). **VK≡DX 픽셀 동일**(AO on/off 둘 다 mean 0.0001/ch,
  max 1, >2px 0/921600 — 결정적 컴퓨트). AO on vs off는 59,271px 국소 변경(접촉부, mean 1.50/ch max240).
  AO-off는 pre-C2와 동일(곱 1.0). Vulkan VUID 0, D3D12 클린.
  한계: 48³ 복셀화로 AO 저주파(soft); per-pixel cone-trace나 클립맵 고해상은 후속. 패스트레이서 AO 정량
  대조는 미수행(시각적 타당성으로 확인).

### C3 — 디퓨즈 GI 1바운스 (stochastic) ✅ (양 백엔드 검증)
- 풀스크린 컴퓨트 `gdf_gi.slang`(`gdf_gi_cs`, push **176B**): 픽셀별로 depth에서 월드 표면 재구성(C2와 동일)
  → `spp`개 **코사인-반구 레이**를 씬 GDF에 sphere-trace → 히트 셰이딩 → 평균 incoming radiance(간접
  irradiance E)를 storage image에 출력. `pbr.slang`이 `(1-metallic)·albedo·E`를 ambient에 가산.
- **설계 포크 확정 = (A) 상수 알베도 재조명.** 히트에서 gradient 노멀 + 태양(짧은 penumbra 소프트섀도우,
  48-step) + 소형 스카이 fill로 재조명, 단일 상수 알베도(0.7). GDF에 머티리얼이 없어 **색 bleeding 없음(무채색
  fill)** — 컬러 바운스는 (B) surface cache 후속. RNG/코사인 샘플은 `rt_common.slang`과 동일(pcg/`cosine_hemisphere`)
  → 패스트레이서와 일관 + 백엔드 결정적.
- **합성 분해(이중계산 없음):** 스카이로 탈출한 레이는 0 반환(IBL 디퓨즈가 이미 미차폐 스카이 공급, C2 AO가
  차폐분 제거, C3가 차폐물의 바운스광 가산) → ambient = IBL·AO(스카이) + albedo·E_gi(씬 바운스). AO·GI 동시 가능.
- 토글 env `P11_GDF_GI` + UI "GDF diffuse GI (deferred)" + `P11_GI_SPP`(기본 8, 1–256) + 디버그뷰 10 "GDF GI".
  RNG 시드에 frame 포함(C4 temporal 대비). 라이팅 push **28→32B**(+`gdf_gi_index`, 부재 시 0xFFFFFFFF→가산 0=무회귀).
- **검증(RTX 2070 SUPER):** build+fmt+clippy(-D warnings) 클린. 태양이 밝은 지면에서 바운스되어 그림자부·오브젝트
  밑면을 채우는 간접 fill 확인. **VK≡DX 픽셀 동일**(GI on spp8 mean 0.0001/ch max3 >2px 1/921600 — RNG 정수
  결정적이라 노이즈 패턴까지 일치; GI off는 max1=무회귀). GI on-vs-off 100,147px(mean1.20/ch). spp8 vs spp64
  레퍼런스 mean0.70/ch(=노이즈, C4 디노이즈 대상; GDF 저주파라 과하지 않음). VUID 0, DX 클린, TDR 없음.
  한계: 무채색(상수 알베도) + spp8 노이즈 + 48³ 저주파. NEXT C4가 temporal+공간 필터로 정리.

### C4 — 시공간 디노이즈 ✅ (양 백엔드 검증)
- **temporal 재투영 + 누적**(`gdf_temporal.slang`, push **192B**): 픽셀별로 depth에서 월드점 P 재구성 →
  **이전 프레임 view-proj로 reproject** → 저장된 월드점으로 히스토리 검증(disocclusion: reproject 텍셀의 이전
  전면 표면이 다른 점이면 리젝트) → EMA 누적(alpha=1/len, len≤64). 히스토리는 **ping-pong byte-address
  storage buffer 2쌍**(`gi_hist` rgb+len, `gi_pos` xyz+valid), 렌더 extent로 (재)할당. 씬 지오메트리가 정적이라
  월드점이 프레임 불변 → 이동 카메라에서도 reproject 정확. `reset`(첫 프레임/조명·해상도 변경)은 히스토리 무시.
- **공간 à-trous**(`gdf_atrous.slang`, push **112B** — `float4 params`가 offset 96 정렬이라 96이 아닌 112!):
  5×5 B3-스플라인 커널 2패스(step 1,2), edge-stopping = world-pos(depth 재구성) + normal 가중 → 저주파 간접
  irradiance를 표면 경계 안 넘게 평활. disoccluded/짧은 히스토리 픽셀의 잔여 노이즈 정리.
- 영속 상태 + 리셋은 `GdfSystem::prepare_denoise`(그래프 전, rt accum 미러) + end-of-frame `advance_denoise`
  (ping-pong swap) + `App::prev_view_proj` 저장. 스크린샷은 카메라 고정이라 reproject=identity로 환원,
  64프레임 warmup(`GI_DENOISE_WARMUP`)으로 progressive 수렴. 토글 `P11_GI_DENOISE`(기본 on) + UI.
- **검증(RTX 2070 SUPER):** build+fmt+clippy(-D warnings) 클린. 디노이즈된 GI가 깔끔(노이즈 제거된 간접 fill).
  **VK≡DX 픽셀 동일**(spp8 디노이즈 mean 0.0026/ch, max 1, >2px 0/921600 — 정수 RNG+결정적 reproject).
  **디노이즈가 ground truth(spp64) 대비 오차 절반↓**: raw spp8 0.70/ch → 디노이즈 0.32/ch, >8px 13,591→1,639(≈8×↓).
  GI-off는 C3 베이스라인과 **byte-identical(max 0)** = 무회귀. VUID 0(à-trous push 112B 수정 후), DX 클린, TDR 없음.
  한계: storage-buffer 히스토리라 reproject는 nearest(정적 스크린샷엔 정확, 이동 시 bilinear는 후속);
  denoise 토글 재활성 시 1–2프레임 stale(self-heal). NEXT C5(선택) 러프 반사.

---
**C5–C7 = 반사 트랙(캡처 기반 IBL 스페큘러 대체) ✅ (C5/C6/C7 모두 양 백엔드 검증, 푸시됨).** SSR(온스크린)
→ GDF SW-RT(오프스크린) → 스카이(miss) 3단 폴백을 세운 뒤(C7c) 하나로 합성해 `pbr.slang`의 prefilter-큐브
스페큘러 lookup을 교체 — **하이브리드-vs-PT 잔차 4.18→2.58/ch(−38%)로 트랙 성공 지표 달성(C7d)**.
**C8 = GDF 컬러(서피스 캐시).** C3 GI·C6/C7 반사의 무채색 상수 알베도(0.7)를 실제 표면 컬러/라디언스로
교체 — C3 컬러 bleed + 컬러 반사. C7과 독립이나 C7 전에 C8a를 하면 반사가 곧바로 컬러가 된다.

### C5 — 스크린-스페이스 반사 (SSR) ✅ (양 백엔드 검증)
- 풀스크린 컴퓨트 `ssr.slang`(`ssr_cs`, push **192B**): 픽셀별로 depth에서 월드점 P 재구성 → 반사 레이
  `R = reflect(-V, N)`를 **월드 공간 선형 march**(96스텝), 각 샘플을 view-proj로 화면 투영해 깊이 버퍼와
  비교(ray ndc.z > scene depth + thickness 내면 히트) → 6회 binary-refine → **셰이딩된 HDR을 히트 픽셀에서
  샘플** → 이웃 오브젝트가 실제로 비침(크롬/구리 구가 녹색 아보카도·이웃을 반사 = 캡처 큐브의 시차/이웃
  미반영 한계 해소). 출력 = `float4(반사색, confidence)`, confidence = 화면 가장자리 페이드(C7 폴백 블렌드용).
- **컬러 소스 결정:** C5는 **라이팅 직후 실행 → 현재 프레임 HDR 샘플**(노출 베이크된 래스터 경로와 동일,
  tonemap 노출 1.0). C7에서 SSR이 라이팅 스페큘러로 **피드백**할 때는 read-before-write라 직전 프레임 컬러
  history(재투영)로 전환 필요 — C7 범위.
- **미스 = 0**(화면 밖/디스오클루전/가장자리) → C6 GDF 폴백이 메우고 C7이 confidence로 가중 합성.
  러프니스 블러·GGX 지터는 미구현(1차는 미러 레이) — C7/후속.
- 토글 env `P11_SSR` + UI "Screen-space reflections (viz)"(다른 전체화면 viz와 배타, tonemap 소스 교체).
- **검증(RTX 2070 SUPER):** build+fmt+clippy(-D warnings) 클린. SSR 버퍼가 각 표면의 반사를 보여줌(구의
  하반부=지면/하늘, 상부=녹색 아보카도; 큐브·구리 구도 이웃 반사; 하늘=검정 miss). **VK≡DX 33px만 차이**
  (mean 0.0026/ch — iterative march의 SPIR-V/DXIL fp-contraction로 hit이 뒤집히는 knife-edge 픽셀, B4/C1급).
  **SSR-off 래스터는 베이스라인과 byte-identical(max 0)=무회귀.** VUID 0, DX 클린, TDR 없음.
  한계: 월드공간 선형 march라 grazing 각에서 동심 스트라이프(스텝의 화면투영 불균일) — Hi-Z/스크린공간 DDA는
  후속; C7의 confidence+러프니스+GDF 폴백이 가린다.

### C6 — GDF 반사 (오프스크린 폴백, SW-RT) ✅ (양 백엔드 검증)
- 풀스크린 컴퓨트 `gdf_reflect.slang`(`gdf_reflect_cs`, push **176B**): 픽셀별로 depth→월드점 P 재구성 →
  반사 레이 `R=reflect(-V,N)`를 **씬 GDF에 sphere-trace**(96스텝; B4/C3와 동일한 geo_inside/geo_march/
  scene_normal 헬퍼) → 히트 재조명(gradient 노멀 + 짧은 penumbra 소프트섀도우 태양 + 소형 스카이 fill,
  상수 알베도 0.7 — C3와 동일, 차후 surface cache로 컬러). 화면 밖 지오메트리·디스오클루전도 그럴듯한 반사 제공.
- **이중계산 없음 / KEY vs C3:** GDF 레이가 스카이로 탈출하면 0이 아니라 **절차적 스카이(`sky(R)`) 반환**
  (스페큘러 miss = 거울이 하늘을 비춤) — C3 디퓨즈 GI의 "스카이 탈출=0"과 역할 반대(거긴 IBL 디퓨즈가
  스카이 공급). 출력=raw radiance(C1 트레이스처럼 tonemap이 노출 적용). 러프 반사(GGX-VNDF+디노이저)는 후속.
- `ReflectSystem::record_gdf_reflect`(reflect.rs): 씬 GDF 볼륨+ext+AABB를 인자로 받음(AO/GI와 동일 패턴,
  bake는 App가 1회). 토글 env `P11_GDF_REFLECT` + UI "GDF reflections (viz)"(다른 전체화면 viz와 배타,
  tonemap 소스 교체 + 노출 적용). bake 게이트 `(gdf_ao||gdf_gi||gdf_reflect)`에 포함.
- **검증(RTX 2070 SUPER):** build+fmt+clippy(-D warnings) 클린. GDF 반사 viz가 정상(지면이 오브젝트를 반사,
  구가 하늘+지면 반사, 스카이 탈출=파란 하늘). **VK≡DX 4px만 차이**(mean 0.0003/ch — sphere-trace
  fp-contraction knife-edge, B4/C1급). **reflect-off 래스터는 베이스라인과 byte-identical(max 0)=무회귀.**
  VUID 0, DX 클린, TDR 없음. PT 정량 대조는 C7 합성 후(rt-compare.py). 한계: 무채색 상수 알베도 + 월드
  march 스트라이프(B4와 동일).

### C7 — 하이브리드 반사 합성 + IBL 스페큘러/디퓨즈 대체 ✅ (C7a–C7d 양 백엔드 검증, 푸시됨)
- **합성:** 픽셀별로 **SSR(C5, 신뢰도 높음) → GDF SW-RT(C6, 오프스크린) → 절차적 스카이(miss)** 순으로
  폴백 선택/블렌드(SSR confidence + 화면 가장자리 페이드). Fresnel·러프니스로 가중해 `pbr.slang`의
  **prefilter-큐브 스페큘러 lookup을 이 하이브리드 결과로 교체**.
- **디퓨즈 대체:** ambient irradiance 항을 **C3/C4 GDF 디퓨즈 GI**(씬 바운스) + **스카이 irradiance**
  (절차적 스카이 적분; 캡처 큐브 대신 SH 또는 저해상 스카이-only 큐브 — 씬 캡처 불필요)로 구성.
- **캡처 파이프라인 격하:** [realtime-env-capture.md](realtime-env-capture.md)의 **씬 캡처(RT3/RT5
  멀티바운스)는 불필요**해짐 — 반사는 SSR+GDF가 실제 이웃 오브젝트를 보고 공급. env 큐브는 **스카이 전용**
  소스로 축소(시차·단일프로브·이웃반사·프록시 한계 모두 은퇴). 토글로 레거시 IBL ↔ 신 SW-RT 경로 비교 유지.
- **검증:** `tools/rt-compare.py`로 **신 하이브리드 반사 vs PT** 평균차가 캡처 IBL(잔차 ~4.0/ch) 대비
  유의미하게 감소함을 정량 확인(= 이 트랙의 성공 지표). 양 백엔드 픽셀 일치, 검증 클린.

**C7a/C7b ✅ (`fb0dd0a`/`9e91370`):** 합성 viz(`reflect_composite.slang`) + 컴퓨트-UBO 인프라 +
재투영 raw-radiance SSR history(`lit_history.slang`, `ssr.slang` history 모드). 자세한 내역은
engine-backlog 메모리.

**C7c ✅ (`0f61395`) — 하이브리드 반사를 라이팅에 연결:** 합성 이미지를 **`pbr.slang` `ambient_ibl`의
prefilter-큐브 스페큘러(`prefiltered`) 대신** 사용. env-BRDF(`specular = refl*(f*brdf.x+brdf.y)`)·
디퓨즈 irradiance는 유지; 합성은 raw radiance라 큐브 샘플의 drop-in(pbr가 마지막 `*ambient.a`로 노출
1회 적용). `PushConstants += reflect_index`(32→36B); `has_swrt`면 box-프록시 시차 보정 생략(합성이 이미
실제 지오메트리를 봄), 없으면(0xFFFFFFFF) 레거시 box-시차 prefilter 큐브 경로 = **반사-off 무회귀**.
**그래프 순서 이동:** 합성이 라이팅 스페큘러에 피드백되므로 **라이팅 이전**에 실행 —
gbuffer→AO/GI→**SSR(history)+GDF reflect+composite**→lighting→lit_history(다음 프레임 history 캡처).
SSR은 history 모드(직전 프레임 raw-radiance 재투영)라 이번 프레임 미작성 HDR을 읽지 않음(read-before-write
회피). **토글 `P11_SWRT_REFLECT`**(+UI)로 레거시 IBL ↔ SW-RT 병존 비교. 활성 시 standalone SSR/GDF/
hybrid viz는 억제(history 인프라 공유). **검증 RTX2070S:** fmt+clippy(-D) 클린; 크롬 구가 실제 녹색
아보카도 이웃을 반사(레거시는 블러 큐브뿐). **VK≡DX** — 기본 래스터 max1, SW-RT max26/4px(C5/C6 march
fp-contraction 상속). **레거시-IBL(반사-off) pre-C7c HEAD와 바이트 동일(max0)=무회귀**. VUID 0, DX 클린.

**C7d ✅ — 정량 성공 지표 달성:** `tools/rt-compare.py`로 `NO_POINT_LIGHTS=1` 공정 비교(기존 ~4.0/ch
방법론), Vulkan 고정 카메라:
- **레거시 캡처-IBL vs PT: 4.178/ch** (>8: 6.55%, >32: 3.02%) — 문서화된 ~4.0/ch 잔차 재현.
- **SW-RT 하이브리드 vs PT: 2.580/ch** (>8: 4.00%, >32: 1.27%) — **잔차 −38%**, 큰 오차(>32) 절반 이하.
- 몽타주 `docs/images/hybrid-vs-pt.png`(SW-RT 래스터 | PT | diff×4): 잔차가 금속 구의 스페큘러
  하이라이트/에지에 집중 — **남은 차이 = SSR 미러-only(러프니스 GGX 블러 미구현) + GDF 히트 상수
  알베도(0.7)**. → **C8(GDF 컬러/서피스 캐시) + 러프 반사가 다음 잔차 감소 트랙**.
- **= 반사 트랙 성공 지표 충족: 캡처-IBL이 원리적으로 못 닫던 이웃 반사 잔차를 SW-RT가 유의미하게 축소.**

**씬 캡처 스카이-only 격하 — 후속(미착수):** C7c 경로(`P11_SWRT_REFLECT`)에서는 specular가 SW-RT라 캡처
큐브 specular가 불필요하나, env 큐브의 **디퓨즈 irradiance(mip2 = 이미 스카이-only)** 와 **레거시 비교
경로**가 여전히 큐브에 의존. 캡처 파이프라인(RT3/RT5 멀티바운스) 실제 제거/스카이-only 재구성은
`P11_SWRT_REFLECT`가 기본값이 된 뒤의 정리 작업 → [realtime-env-capture.md](realtime-env-capture.md)
RT3/RT5 격하와 함께 후속 단계로 문서화(현재는 토글 병존 유지).

### C8 — GDF 컬러 / 서피스 캐시 (무채색 상수 알베도 → 실제 표면 컬러·라디언스)
현재 C3 GI·C6/C7 반사는 GDF 히트를 **상수 알베도 0.7로 재조명** → 무채색(녹색 아보카도·빨간 큐브의 색이
반사/바운스에 안 묻음). GDF 볼륨은 **거리(R32Float)만** 보유하므로 표면 머티리얼/라디언스를 별도 저장해야
컬러가 된다. 점증 2단계.

**베이크 정리(자주 묻는 점):**
- **알베도만 저장 = 정적 1회 베이크.** 지오메트리가 정적이므로 base_color를 복셀/아틀라스에 한 번 굽고
  (sdf 베이크처럼 1회), 히트에서 그 알베도로 *재조명*. 컬러 GI/반사 OK, 단 조명은 레이당 재계산, 멀티바운스 없음.
- **풀 서피스 캐시(라디언스) = 연속 갱신.** 파라미터화(카드 배치/아틀라스 UV)는 1회 베이크지만, 저장값이
  *라디언스*(태양+GI 결과)라 조명/시간 변화에 따라 **매 프레임(또는 분산) 재조명 갱신** 필요. 직전 캐시를 읽어
  멀티바운스 자연 누적. 쿼리=직접 룩업(레이당 셰이딩/섀도우 레이 불필요 → 정확+저렴).

#### C8a — per-voxel 알베도 (라이트, 1차 권장) ✅ (양 백엔드 검증, 푸시됨)
- 거리 볼륨과 나란히 **알베도 볼륨**. 저장 선택: **3×R32Float 재사용**(기존 `volumes[]` 슬롯 3개 = RHI 무변경,
  채널당 트라이리니어) **vs** RGBA8/RGBA16F 색-볼륨(`Texture3D<float4> color_volumes[]` bindless 신설 = B1급 RHI 작업).
  **1차는 3×R32 권장**(무위험).
- fuse 시 삼각형마다 소속 오브젝트 base_color를 태깅(병렬 per-tri 알베도 버퍼) → 별도 알베도 베이크 패스가
  voxel별 최근접 삼각형의 알베도를 컬러 볼륨에 기록(`sdf_bake`의 최근접-삼각형 탐색 로직 재사용; 거리 베이크는
  무변경 유지 위해 별도 패스). 1회 정적 베이크.
- C3 `gdf_gi`·C6 `gdf_reflect` 히트에서 상수 0.7 대신 **컬러 볼륨 트라이리니어 샘플** → 실제 알베도로 재조명.
  → 컬러 GI 바운스 + 컬러 반사. 조명은 여전히 레이당 재계산(태양 NdotL+섀도우+스카이), 멀티바운스는 캐시에 없음.
- 검증: C3/C6 컬러가 PT의 이웃반사/색 bleed와 정성 일치, 양 백엔드 픽셀 일치, 무회귀.

**C8a ✅ 구현 (`952cdec`):** 3×R32Float 알베도 볼륨(scene GDF와 동일 그리드/AABB) + per-triangle 선형
알베도 버퍼(12B/tri). fuse 시 오브젝트별 대표 알베도 태깅 — **텍스처 오브젝트(아보카도)는 base-color
이미지 평균(sRGB→선형)×factor**, 절차 오브젝트는 선형 base_color 직접. 신규 `sdf_albedo_bake.slang`
(`albedoBakeMain`, push 64B): voxel별 최근접 삼각형 탐색(거리 베이크와 동일 closest-point) → 그 삼각형의
알베도를 3볼륨에 기록(거리 베이크는 **무수정** = 별도 패스). `gdf_gi.slang`(pad 슬롯 재사용, 176B 유지)·
`gdf_reflect.slang`(176→192B)이 히트에서 `albedo_at(p)` 트라이리니어 샘플로 상수 0.7 대체; 센티넬
(0xFFFFFFFF)이면 상수 폴백. `GdfSystem`이 볼륨+버퍼+`record_scene_albedo_bake` 소유, `GiSystem`/
`ReflectSystem`의 `record_gi`/`record_gdf_reflect`가 `Option<(&[Volume;3], ResourceId)>` 인자로 수령.
별도 `scene_albedo` external + bake-once 래치. **토글 `P11_GDF_COLOR`**(기본 ON, =0이면 무채색 폴백).
**검증 RTX2070S:** fmt+clippy(-D) 클린; 컬러 GI 바운스(아보카도 녹색·큐브 적색 bleed, 22,963px) + 컬러
반사(GDF 폴백, 9,169px), 몽타주 `docs/images/c8a-colored-gi.png`. **VK≡DX** GI max1, SW-RT max31/6px(C5/C6
march fp-contraction 상속). **`P11_GDF_COLOR=0` 폴백 + 기본 래스터 pre-C8a와 바이트 동일(max0)=무회귀.**
**rt-compare SW-RT-vs-PT 2.58→2.60/ch(평탄)** — C8a는 GDF 폴백 소수 영역만 컬러화(온스크린 반사는 이미
SSR lit-HDR history로 컬러), 잔차 추가 감소는 C8b(서피스 캐시·멀티바운스) 트랙. C8a 게이트(정성 컬러 일치+
양 백엔드+무회귀) 충족.

#### C8b — 서피스 캐시 라디언스 (풀, 정확 = 설계 포크 B)
- 표면 파라미터화: **per-mesh 라디언스 아틀라스(UV)** 또는 **카드(오리엔티드 쿼드 그리드)** — 정적 1회.
- **연속 갱신 패스**: 캐시 텍셀마다 태양(소프트섀도우)+스카이+**직전 캐시에서 GI 1+바운스** 적분 → 라디언스
  기록(시공간 누적/분산 갱신; C4 디노이저 인프라 공유 가능). 멀티바운스 자연 누적.
- C3/C6 히트·C7 합성에서 **캐시 라디언스 직접 룩업**(재조명·섀도우 레이 불필요 → 쿼리 저렴+정확).
- 검증: `rt-compare.py`로 PT 잔차 추가 감소(컬러 멀티바운스), 양 백엔드.

**권장 순서:** C8a(저위험 컬러) 먼저 → 필요 시 C8b. C7과 독립이지만 **C8a를 C7 앞에 두면 C7 반사가 바로
컬러**가 된다(사용자 선택: C7 → C8a, 또는 C8a → C7). C8b는 동적 오브젝트/시간변화 조명까지 정확히 가는
장기 트랙이라 C7/C8a 검증 후 별도 착수.

## 미결 / 설계 항목
- **GDF 컬러 저장(C8):** per-voxel 알베도(3×R32 무RHI변경 vs RGBA8/16F 색-볼륨 bindless 신설) vs 서피스 캐시
  (아틀라스 UV vs 카드 그리드; 라디언스 연속 갱신·멀티바운스). 1차는 알베도 라이트(C8a).
- GDF 표현(클립맵 레벨 수/해상도), 메모리 예산.
- SW RT 정확도 vs 비용(마칭 스텝/원뿔 추적).
- 디노이저 구조(GI + 러프 반사 공유), 동적 오브젝트 GDF 갱신 빈도.
- **SSR 컬러 소스:** 직전 프레임 라이팅 history(재투영) vs 현재 프레임 부분 결과 — 재투영 필요(C4 인프라 공유).
- **반사 러프니스 모델:** 미러+mip 블러(저비용) vs GGX-VNDF 레이(정확, 디노이즈 필요). 1차는 전자.
- **IBL 디퓨즈 스카이 irradiance:** SH 9계수(컴퓨트) vs 저해상 스카이-only 큐브 유지. 씬 캡처는 제거.
- **레거시 IBL 경로 유지 기간:** SW-RT 반사가 PT 대비 확실히 우월함을 정량 확인할 때까지 토글로 병존.
- HW RT(Phase 8)와의 선택/하이브리드 관계(GDF 폴백 대신 TLAS 반사 레이 옵션).
