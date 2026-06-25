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
GDF를 ray-march해 **디퓨즈 GI(1+ 바운스)·AO·러프 반사**를 stochastic 샘플하고, 결과를 디퍼드
라이팅(Phase 6)의 ambient/GI 항으로 합성한다. Stage B의 GDF는 *단위 큐브 데모*(고정 카메라)였으므로
Stage C는 먼저 **실제 씬을 월드 공간 GDF로 굽고**, 그것을 **실제 디퍼드 G-buffer**에서 march한다.

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

### C2 — GDF AO → 디퍼드 ambient (다음)
- G-buffer(월드 pos+노멀)에서 씬 GDF를 ray-march해 AO 산출 → 라이팅 패스의 ambient/IBL 항에 곱셈 합성.
  GDF가 실제 렌더에 처음 영향. 패스트레이서 AO 레퍼런스 대조.

### C3 — 디퓨즈 GI 1바운스 (stochastic)
- G-buffer 표면에서 코사인-반구 레이를 GDF에 sphere-trace → 히트 셰이딩 → 간접 디퓨즈 누적(노이지).
- **설계 포크 — GDF 히트 셰이딩:** GDF는 거리만 보유. (A) 히트에서 gradient 노멀+태양(소프트섀도우)+스카이로
  재조명, 상수 알베도(최소 인프라, B4 히트 셰이딩 재사용) / (B) surface cache(라디언스 아틀라스/카드, 정확) /
  (C) 스크린스페이스 샘플+폴백. C3 도달 시 확정(권장 시작점 A).

### C4 — 시공간 디노이즈
- temporal 재투영(히스토리 누적) + 공간 필터(à-trous/bilateral)로 noisy GI 정리.

### C5 (선택) — 러프 반사
- GGX 샘플 GDF 레이로 광택 반사.

## 미결 / 설계 항목
- GDF 표현(클립맵 레벨 수/해상도), 메모리 예산.
- SW RT 정확도 vs 비용(마칭 스텝/원뿔 추적).
- 디노이저 구조, 동적 오브젝트 GDF 갱신 빈도.
- HW RT(Phase 8)와의 선택/하이브리드 관계.
