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
  `dispatch(x,y,z)` 재사용.
- 스모크 테스트 `volume_test.slang`(`fillMain`/`viewMain`): 컴퓨트가 storage_volume에 중심 구 부호거리
  기록 → `volume_to_sampled` 배리어 → `volumes[]` SRV를 Z=0.5 슬라이스 트라이리니어 샘플 → 화면.
  그래프 통합은 `import_external`로 fill→view 순서 보장, tonemap `rt_out.or(sdf_out).or(vol_out)`. 토글
  env `P11_VOLUME_TEST` + UI.
- **검증(RTX 2070 SUPER):** build+fmt+clippy(-D warnings) 클린. VK·DX 슬라이스 정상(중심 구 부호거리
  그라데이션 + zero 등위면 녹색 링). **VK≡DX 픽셀 동일(mean 0.0000, max 0)** — 결정적 fill+트라이리니어
  샘플. Vulkan VUID 0. **bindless 블록에 멤버 추가했지만 기존 래스터 씬 byte-identical(회귀 없음)** —
  Slang이 미사용 binding을 drop. tmp/vol-{vk,dx}.png.

### B2 — per-mesh SDF 베이크
- 컴퓨트로 각 메시의 로컬 AABB를 N³ 그리드(예 32³~64³)로 보로 voxel당 **부호 있는 거리** 계산.
  1차 구현: **brute-force 점→삼각형 거리**(voxel center vs 모든 삼각형의 최소 거리). 부호는
  angle-weighted pseudonormal 또는 ray-stabbing parity(내부/외부). 베이크-타임이므로 단순/정확 우선.
  최적화(후속): voxelize + **JFA**(jump flooding)로 O(N³ log N).
- 메시 정점/인덱스는 이미 storage buffer로 GPU 상주(Phase 8 RT geometry 경로 재활용 가능).
- 검증: 구/박스 메시의 베이크 SDF를 Stage A 해석적 SDF와 대조(거리 오차 ≤ voxel 대각선).

### B3 — 전역 클립맵 머지
- 카메라 주변을 덮는 **GDF 클립맵**(여러 해상도 레벨의 3D 볼륨)으로 per-mesh SDF를 인스턴스 변환과
  함께 합성(min 결합). 정적은 한 번 베이크, 동적 오브젝트는 매 프레임/저빈도로 영향 영역만 갱신.
  클립맵 레벨/해상도/메모리 예산은 이 단계에서 확정(미결 항목 참조).

### B4 — GDF SW RT
- `sdf_trace.slang`의 `scene_dist`/`scene_normal`을 **클립맵 볼륨 트라이리니어 샘플**로 교체
  (적절 레벨 선택 + 경계 폴백). 노멀은 볼륨 gradient. Stage A의 march/soft_shadow/AO 로직은 그대로
  재사용 → 비로소 **실제 메시 지오메트리**를 SW RT.
- 검증: Phase 8 HW RT / 패스트레이서와 동일 씬 1차 가시성 + 소프트 섀도우 대조.

신규 RHI: 3D(볼륨) 텍스처 + UAV, 3D 디스패치. (Phase 7 storage image의 3D 확장.)

> **GDF 베이크 영속화는 별도 워크스트림으로 승격됨 → [Phase 12 — 에셋 파이프라인](phase-12-asset-pipeline.md).**
> 사용자 요청대로 SDF 베이크만이 아니라 **메시까지 함께 직렬화하는 쿠킹된 에셋(`.dcasset`)** 개념이라
> 규모가 커서 `crates/asset`의 크로스컷팅 인프라(독립 Phase)로 분리했다. Stage B의 per-mesh SDF 베이크
> 결과가 Phase 12 M2의 SDF 청크로 영속화된다. (메시 직렬화 M1은 Phase 11과 독립적으로 먼저 가능.)

## Stage C — Stochastic Lighting
- GDF를 ray-march해 **디퓨즈 GI(1+ 바운스)·AO·러프 반사**를 stochastic(몬테카를로) 샘플.
- **시공간 디노이즈:** temporal accumulation(재투영) + 공간 필터. 스크린-스페이스 프로브 /
  래디언스 캐시 / per-pixel 중 구조는 이 스테이지에서 확정.
- 머티리얼 히트 셰이딩: GDF는 거리만 → 표면 머티리얼/라이팅은 surface cache 또는 근사 필요(설계 항목).
- 결과를 디퓨드 라이팅(Phase 6)의 ambient/GI 항으로 합성.

## 미결 / 설계 항목
- GDF 표현(클립맵 레벨 수/해상도), 메모리 예산.
- SW RT 정확도 vs 비용(마칭 스텝/원뿔 추적).
- 디노이저 구조, 동적 오브젝트 GDF 갱신 빈도.
- HW RT(Phase 8)와의 선택/하이브리드 관계.
