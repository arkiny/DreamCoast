# Per-mesh distance fields — 베이크 구조 전환 (per-mesh DF + 카드 색 정렬)

> 상위: [scalable-gi.md](scalable-gi.md) (Phase 10 GI 트랙). 이 문서는 그 트랙의 **베이크 아키텍처를
> "씬 전체 fused 재베이크" → "per-mesh DF + 인스턴스 공유 + 클립맵 합성"** 으로 바꾸는 권위 계획이다.
> 선행 측정/근거: [reflection-sdf-resolution.md](reflection-sdf-resolution.md)(해상도는 레버 아님),
> [lumen-parity-swrt.md](lumen-parity-swrt.md), [foliage.md](foliage.md)(커튼 색 번짐 부재).
> **상태: 구현 완료 + 콘텐츠 기본값 승격.** per-mesh DF가 이제 **콘텐츠 씬의 기본 경로**이고, fused
> 전체-씬 베이크는 **DEPRECATED**(거친 복셀 ~0.76 m @ 37 m 씬 → 얇은 부조/벽/트레이서리를 잃어 DF 기반
> 패스가 관통). 갤러리만 fused 유지(바이트 동일 앵커·per-mesh 이득 없음). `P11_PERMESH_GDF=0`으로만 폐기된
> fused 경로 강제(fallback/A-B; 콘텐츠에서 켜면 WARN 로그). 비-인스턴스 씬(Intel Sponza ~426 유니크 메시)은
> 첫 쿡이 느리지만 캐시됨.

## 동기 / 배경 (2026-06-30, 측정·레퍼런스 확인으로 확정)

두 가지 문제가 **하나의 근본 원인**을 공유한다는 것이 이번 세션에서 확정됐다:

1. **커튼 GI 색 번짐 부재** ([foliage.md](foliage.md)에 기록): `sponza_intel`/`sponza_hero`에서 빨강/파랑
   커튼이 인접 석재에 간접광 색을 전혀 안 번지게 한다. 측정(이번 세션, 평평한 대리석 바닥 메트릭):
   빨간 커튼 베이스 옆 바닥 R−B = **−71**, 중립 중앙 바닥 R−B = **−70** → **번짐 ≈ 0**(오히려 커튼이
   하늘을 가려 살짝 더 차가움 = AO이지 색 바운스 아님). 빨간 커튼 자체는 R−B +110(제대로 빨강).
2. **베이크 폭발**: `P11_GDF_DIM=128` 측정 시 `sponza_intel` 쿡이 **15분+**. 특히 finer 클립맵 레벨이
   씬 중심(공중)에 놓여 **빈 복셀이 대부분** → 최근접 삼각형 ring 검색이 그리드 전체로 확장 → 복셀당
   비용 폭발.

### 근본 원인 (공통)
현재 GDF 파이프라인은 **씬 전체를 월드로 fuse한 dense 거리장 + per-voxel albedo 볼륨**이다
([fuse.rs](../apps/sandbox/src/fuse.rs), [sdf.rs](../crates/asset/src/sdf.rs),
[gdf.rs](../apps/sandbox/src/gdf.rs)). 그래서:

- **색**: GI 바운스 색은 `gdf_gi.slang`의 `albedo_at()`(= `clipmap.slang cm_albedo`, per-voxel
  albedo 볼륨)에서 나온다. surface cache 캡처([sdf_cache_capture.slang](../crates/shader/shaders/sdf_cache_capture.slang))
  **조차 같은 coarse albedo 볼륨을 읽는다** → Med(직접 albedo_at)·High(서피스캐시) 둘 다 같은 병목.
  36 m 씬을 48³(0.75 m/voxel)~128³(0.28 m/voxel)로 담으니 **얇은 커튼이 인접 석재와 한 복셀에
  섞여 색이 중립으로 희석**된다.
- **베이크**: 인스턴스/레벨이 바뀔 때마다 fused 삼각형 수프를 클립맵 레벨마다 통째로 재베이크.
  유니크 메시가 적어도(Sponza 기둥/아치는 인스턴스) 매번 전체를 다시 굽는다.

### 레퍼런스는 어떻게 하나 (2026-06-30 소스 대조)
- **Mesh distance field**: **메시(에셋) 단위**로 오프라인 1회 베이크 → 메시당 dense 볼륨 텍스처(최대
  128³, 8 MB; 거리장 해상도 스케일로 메시 크기에 비례). **인스턴스가 공유.** 런타임의
  **글로벌 distance field는 per-object 필드를 카메라 중심 클립맵으로 합성**하되, 새로 보이거나 바뀐
  영역만 갱신. → 씬 전체 재베이크가 없다.
- **mesh-card surface cache**: 메시당 오프라인 생성(~12 카드), 캡처 = **실제 메시를 카드 시점에서
  래스터화** → per-texel 진짜 머티리얼 albedo. 카드는 **opacity 저장 → 구멍**(체인링크 펜스/천/잎).
  색을 coarse voxel이 아니라 **메시 표면**에서 얻기 때문에 얇은 천이 살아난다.

**핵심 통찰**: per-mesh DF는 커튼 메시가 **자기 타이트 AABB에 거리장을 굽는다.** 커튼(4×3×0.1 m)을
자기 AABB 128³에 구우면 얇은 축도 충분히 해상 → **색 번짐 문제도 자연 해소**(별도 해상도 상향 불요),
**동시에 베이크가 per-mesh·인스턴스 공유·증분 합성으로 가벼워진다.** 두 문제, 한 수정.

## 설계

### A. Per-mesh distance field (로컬 공간, 캐시·공유)
- `MeshRegistry`는 이미 [`MeshCpu{vertices,indices}`](../apps/sandbox/src/registry.rs)를 `MeshHandle`
  별로 보관(fuse의 단일 지오 소스). 여기에 **메시별 로컬-공간 SDF**를 추가한다.
- 베이크: 메시의 **로컬 AABB**에 `bake_sdf_from_fused`(Stage A 그리드 가속, 결정론)로 굽되, 해상도는
  **메시 크기 비례**(거리장 해상도 스케일 모사): `dim ≈ clamp(round(longest_extent /
  target_voxel), MIN_DIM, MAX_DIM)`, 비입방 메시는 축별 dim(셀 ~입방 유지). 기본 `target_voxel`은
  메시-국소이므로 **세밀**(예 ≈ 메시 최장축/64, MAX_DIM=128). 커튼은 자기 스케일에서 완전 해상.
- 캐시 키 = **(로컬 vtx, 로컬 idx, dim)** — **월드 AABB 없음** → 동일 메시는 인스턴스/레벨/씬에 무관히
  **한 번만 굽고 공유**(cook 콘텐츠 해시 그대로 사용, [cook/scene.rs](../crates/asset/src/cook/scene.rs)).
- 결정론·DX≡VK: min-거리 연산이라 순서 무관 비트 동일(기존 `grid_matches_brute` 패턴 재사용).

### B. 런타임 합성 (Global DF 클립맵, **거리만**) — SDF 트레이스 경로 셰이더 불변
- 정적 레벨은 **로드 시 1회**, per-mesh DF들을 **기존 클립맵 볼륨으로 합성**한다(글로벌 DF 합성의 정적
  특수화). 클립맵 voxel `p`마다: 근방 오브젝트들에 대해 `min_i sdf_i(inv_xf_i · p)`(오브젝트 로컬로
  역변환 후 자기 DF 트라이리니어 샘플). **균등/유니폼 스케일만**(노멀 보존) 우선 지원.
- 가속: 오브젝트 월드 AABB 그리드(Stage A 그리드 재사용)로 voxel당 근방 오브젝트만 → O(clip_voxels ×
  근방 오브젝트). finer 레벨(공중·빈 영역)은 근방 오브젝트 0~소수 → **빈-복셀 폭발 제거**(베이크 병목 해소).
- 출력은 현재와 동일한 R32F 클립맵 거리 볼륨 → **거리 march 셰이더(`clipmap.slang cm_geo_*`) 무변경.**
- **갤러리 = fused 경로 유지(무변경) → 바이트 동일.** 합성은 per-object 필드의 trilinear 리샘플 + min이라
  fused 정확-최근접-삼각형 베이크와 **비트 동일하지 않다**(근사). 그래서 합성은 **콘텐츠 씬 전용**
  (`!gallery_scene`)이고, 갤러리는 코드 경로 자체를 안 건드려 앵커를 보존한다(기존 gallery/content 분기 관례 그대로).
- **⚠️ 합성은 거리만**: 합성된 클립맵은 coarse(0.1~0.75 m)라 **색을 여기서 읽으면 다시 coarsening**되어
  커튼 색이 안 산다. 그래서 **색은 합성 볼륨이 아니라 per-mesh 파인 데이터에서 직접 샘플**(C 참조).
- 비입방/비균등 스케일·동적 오브젝트·카메라 추종 증분 갱신은 **후속**(스트리밍 트랙).

### C. 색 경로 — hit에서 **per-mesh 파인 albedo** 직접 샘플 (id-indirect)
나이브하게 per-mesh albedo를 클립맵으로 합성하면 **B의 coarsening 함정에 빠진다**(클립맵 voxel 간격으로
재희석 → 색 안 삼). 그래서 색은 트레이스(coarse)와 분리해 **hit에서 파인 해상도로** 가져온다:
- 합성 시 **nearest-object-id 볼륨**(클립맵당 uint, 그 voxel의 최근접 오브젝트)도 함께 만든다.
- **오브젝트 디스크립터 버퍼**: 오브젝트별 `{inv_world(3×4), local_aabb_min/max, sdf_idx, albedo_idx}`.
  유니크 메시의 per-mesh albedo 볼륨은 GPU에 상주(인스턴스 공유).
- 셰이더(`gdf_gi`/`gdf_reflect`/캡처)의 `albedo_at(p)`를 교체: id 볼륨에서 오브젝트 → `inv_world·p`로
  로컬 좌표 → **그 메시의 파인 local albedo 볼륨 트라이리니어 샘플**. hit당 변환 1 + 샘플 1(바운드).
- **한계(정직)**: 트레이스가 coarse라 hit 위치 자체가 ±voxel 오차. 커튼처럼 **열린 아케이드에 매달린**
  지오는 최근접=커튼이라 색이 제대로 산다(측정으로 확인). 벽에 바짝 붙은 얇은 면은 coarse 트레이스가
  커튼/벽을 못 가르므로 여전히 한계 → **per-object DF 직접 march**가 필요(S4, per-object mesh-DF trace).
- **C2(후속, 카드 정렬, S4)**: 트레이스도 per-object 파인 DF로(thin-against-thick 해결) + surface
  cache 카드 **메시 래스터 캡처 + opacity**(천/잎/펜스 일반화), voxel albedo 볼륨 제거.

## 스테이지 (각 = 독립 커밋, 게이트 아래)

- **S0 — per-mesh SDF 베이크 + 캐시** (`crates/asset`): 메시 크기 비례 dim 산정 + 로컬 베이크
  + 콘텐츠-해시 캐시. 단위 테스트(결정론·brute 동치·dim 산정). 런타임 미배선 → **무회귀(런타임 불변)**.
- **S1 — 합성기(거리)**(`apps/sandbox`, 신규 `compose.rs`): per-mesh DF들 → 클립맵 **거리** 볼륨 +
  **nearest-object-id 볼륨** 합성(오브젝트 AABB 그리드 가속). **콘텐츠 씬 전용**(`!gallery_scene`);
  갤러리는 fused 경로 유지(바이트 동일). 검증: `sponza_intel`에서 합성 거리장으로 GI/AO 정상 동작
  (시각), **베이크 시간 fused 대비 측정**(분→초 기대), 갤러리 무회귀(SHA-256).
- **S2 — 색 복원**(C, id-indirect): 오브젝트 디스크립터 + per-mesh albedo GPU 상주 + `albedo_at`를 id
  볼륨 경유 파인 샘플로 교체(`gdf_gi`/`gdf_reflect`/캡처). `sponza_intel`에서 **커튼 색 번짐 메트릭**
  (bleed.py) 전/후 수치 보고(빨강 옆 바닥 R−B 상승). DX≡VK 재검증(셰이더 변경). 베이크 시간 전/후 보고.
- **S3 — fused 경로 교체**: 기본을 per-mesh+합성으로, 구 fused 삼각형 베이크 제거(또는 폴백 플래그).
  finer 클립맵 중심을 씬 중심 → **카메라/관심영역**으로(빈-레벨 낭비 제거, [main.rs:1281] 수정).
- **S4(후속, 분리 가능)** — C2 카드 래스터 캡처 + opacity (카드 색 정렬). 동적/스트리밍은 범위 외.

## 파일 (생성 / 수정)
- **신규**: 본 문서, `apps/sandbox/src/compose.rs`(합성기), 메시별 DF 베이크/캐시 헬퍼.
- **수정**: `crates/asset/src/sdf.rs`(메시 크기 비례 dim·로컬 베이크 진입), `crates/asset/src/cook/scene.rs`
  (per-mesh 캐시 키), `apps/sandbox/src/registry.rs`(per-mesh DF 보관), `apps/sandbox/src/gdf.rs`
  (합성 결과 업로드/클립 레벨), `apps/sandbox/src/main.rs`(베이크 배선·클립 중심), `docs/scalable-gi.md`
  ·`docs/ROADMAP.md`(트랙 링크). **셰이더는 S0–S3에서 무변경**(S4에서만 카드 캡처 변경).

## 리스크 / 미결
- **합성 = fused 동치?** 균등 스케일·아이덴티티에선 동일 필드(min over objects). 비균등 스케일에서 SDF가
  비균등 변환을 정확히 못 담음 → 우선 균등만, 비균등은 후속(또는 보수적 폴백). 갤러리 바이트 동일이 1차 게이트.
- **메시 경계 이음새**: 인접 오브젝트 DF의 min 합성이 표면 근처에서 정확(거리장 union의 성질). 멀리서의
  근사 오차는 GI 저주파라 허용(PT 정직 보고).
- **메모리**: per-mesh DF 다수(유니크 메시 수 × dim³). 인스턴스 공유라 총량은 fused보다 작을 수 있으나
  유니크가 많은 씬은 측정 필요. MAX_DIM/예산 노브.
- **DX≡VK / 결정론**: 베이크 min-연산 비트 동일, 합성도 min-연산이라 백엔드 무관(업로드 바이트 동일).
  cook 결정성 유지(콘텐츠 해시).

## 검증 (스테이지 공통)
`cargo fmt --all` → `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets` → 각 백엔드
헤드리스 → **VK≡DX ≤0.001/ch** + Vulkan 검증 클린 + `tools/rt-compare.py` PT 잔차. **갤러리 바이트
동일이 매 스테이지 1순위 게이트.** S2는 `bleed.py`(이 세션 작성, scratchpad) 색-번짐 전/후 수치 +
베이크 시간 전/후 수치를 정직 보고.
