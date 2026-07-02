# 실제 씬 컬링 + 거리 기반 LOD 설계 (Cull/LOD Design)

> 상태: 계획 (설계 문서, 2026-07-03). 리드 워크스트림: Sponza 1080p 60fps 퍼프 트랙.
> 전제: PR-7/PR-8 GPU 컬링 인프라(cull.rs/hzb.rs/cull.slang), Phase 12 씬 그래프
> (registry.rs/world.rs), Scalable-GI 퓨즈(fuse.rs)가 이미 존재. 이 문서는 그 인프라를
> 데모 그리드에서 "실제 씬 드로우"로 옮기는 설계이며, 아직 코드 변경 없음(read-only).
>
> 상위: render-pipeline-reference.md, hzb-occlusion-culling.md, phase-14-virtual-geometry.md.
> 검증 게이트: DX≡VK ≤ 0.001 avg/ch, 갤러리 바이트-앵커 sha256 유지, PROFILE_GPU before/after.

---

## 0. 요약 (TL;DR — 결론 먼저)

1. 현재 GPU 컬링은 실제 씬을 전혀 건드리지 않는다. cull.rs/hzb.rs가 컬링하는 것은
   instance_center(i)로 파라메트릭하게 생성되는 합성 큐브 그리드(GRID_COUNT개)뿐이다
   (cull.slang:31-37). 실제 G-buffer/섀도우 드로우는 deferred.rs에서 CPU 루프 + per-object
   draw_indexed로 나가며(§1) 바운드도 컬링도 없다.
2. AABB 소스는 이미 존재한다. fuse::fuse_scene가 draw-list 순서로 per-drawable 월드
   AABB(FusedScene::drawable_aabb, fuse.rs:181-183, 246)를 이미 계산한다. 이것이
   single-source-of-truth이며, build_scene의 SceneObject 순서(둘 다 world.draw_list()를 순회)와
   정확히 정렬된다. 새 AABB 계산 경로를 만들 필요가 없다.
3. Sponza 1080p에서 컬링은 60fps 레버가 아니다. 실측상 지오메트리 컬링은 27ms 프레임 중
   ~1.5ms 수준, 병목은 GDF SW-RT(캐시가 해상도 독립적으로 남아 VK ~11.5ms 평탄, qhd-perf.md:58).
   컬링/LOD의 진짜 페이백은 IntelSponza 규모(155+ 노드, ~8.6GB 에셋, 커튼/나무/아이비 4.9M-tri)
   에서 온다(§7). 따라서 이 작업은 Sponza를 60fps로 만드는 작업이 아니라, IntelSponza를 60fps로
   만들기 위한 스케일러빌리티 투자로 정직하게 프레이밍한다.
4. 단계적·바이트-중립 착지. CPU 프러스텀 컬(S1)은 순수 가시 집합 축소 → 이미지 동일. GPU-driven
   indirect(S2)와 HZB 오클루전(S3)은 HZB 문서의 검증된 2-패스/보수성 방법론을 실제 드로우에
   그대로 적용. LOD(S4)는 스크린-스페이스 에러 임계값으로 팝핑 없이. 각 단계 독립 검증(§6).

---

## 1. 실제 씬 드로우 루프 — 정확한 위치와 현재 발행 방식

실제 씬은 SceneObject 슬라이스(main.rs:194-221)에 대한 CPU 루프로 그려진다. 매 프레임
build_scene(&world, &mesh_registry, &material_registry)(main.rs:3367, registry.rs:173-198)가
world.draw_list()를 순회해 Vec<SceneObject>를 만들고, 각 패스가 이 슬라이스를 빌린다.

| 패스 | 파일:라인 | 루프 | 드로우 발행 |
|---|---|---|---|
| 섀도우(레거시 단일맵) | deferred.rs:701 record_shadow | for obj in scene (:718) | per-obj bind_vertex_buffer + bind_index_buffer + draw_indexed(obj.mesh.index_count,0,0) (:741-743) |
| 섀도우(CSM 아틀라스) | deferred.rs:858 record_shadow_atlas | for slot … for obj in scene (:874,886) | per-slot 뷰포트 + per-obj draw_indexed (:905) |
| 깊이 pre-pass | deferred.rs:763 record_prepass | for obj in scene (:787) | per-obj draw_indexed (:824) |
| G-buffer fill | deferred.rs:920 record_gbuffer | for obj in scene (:980) | per-obj draw_indexed (:1018) + ground (:1039) |
| 데칼 | deferred.rs:1054 record_decals | for obj in scene (:1085) | per-obj draw_indexed (:1104) |
| 속도(모션 벡터) | velocity.rs (record) | for obj in scene | per-obj draw_indexed |

호출 사이트: 메인 프레임은 main.rs:4553-4623(섀도우 → prepass → gbuffer → decals), 스크린샷/inset
세컨더리 뷰는 main.rs:6314(두 번째 record_gbuffer). 두 사이트 모두 같은 &scene 슬라이스를 공유한다.

현재 발행 방식의 요약:
- Indirect 아님. 실제 드로우는 전부 direct draw_indexed. Indirect는 오직 데모 컬 그리드
  (cull.rs:337 draw_indexed_indirect)와 파티클만 사용.
- Per-mesh 바인딩. 오브젝트마다 vbuf/ibuf를 리바인드(:1016-1017). 인스턴싱 없음(공유 지오메트리는
  Rc<GpuMesh>로 CPU 측 공유만, GPU 드로우는 개별).
- Bindless-first이지만 draw는 아님. 머티리얼/텍스처는 bindless 인덱스로 푸시 상수에 담기지만
  (gbuffer_push), 드로우 자체는 CPU가 오브젝트마다 발행. 즉 파이프라인은 bindless인데 제출은
  CPU-driven이다 — 이것이 컬링을 어렵게 만드는 핵심(§3).
- 파이프라인 스위칭. static/skinned/morphed 파이프라인을 오브젝트별로 전환(:999-1003).
  static이 지배적(Sponza/IntelSponza는 전부 static).
- AABB/바운드 없음. SceneObject에는 mesh/transform만 있고 바운딩 볼륨 필드가 없다.

이 CPU 루프 구조가 §2의 인프라와 만나는 지점이 이 설계의 전부다.

---

## 2. 기존 cull.rs / hzb.rs 기계 — 어떻게 동작하고, 왜 실제 드로우에 바로 못 쓰나

### 2.1 데이터 흐름 (현재, 데모 그리드)

```
[reset]  csReset  : indirect args 헤더 클리어 (index_count, instance_count=0, …)
[cull ]  csCull   : i in [0,GRID_COUNT) 각각 instance_center(i) 를 파라메트릭 계산 →
                    6-평면 프러스텀 테스트 → 통과분을 visible[]에 append + InterlockedAdd(args+4)
[draw ]  draw_indexed_indirect(args) : VS가 SV_InstanceID → visible[id] → 그리드 셀 → 큐브 오프셋
```

HZB 경로(HZB_CULL=1)는 csCull을 csCullHzb로 교체: 같은 프러스텀 테스트 + 지난 프레임 HZB
피라미드에 대한 오클루전 테스트(cull.slang:155-259). 피라미드는 hzb.rs가 씬 depth를 max-reduce로
빌드(record_build), app-owned persistent RT 밉 체인(HZB_BASE_DIVISOR=2, hzb.rs:28).

### 2.2 핵심 제약 — 인스턴스가 파라메트릭이다

컬 셰이더는 인스턴스 위치를 버퍼에서 읽지 않고 산술로 생성한다(cull.slang:31-37):
gx=i%grid_dim; gz=i/grid_dim; center=((gx-half)*spacing, y_height, (gz-half)*spacing).
바운딩 반경도 스칼라 하나(pc.cube_radius)로 모든 인스턴스 공유. 즉 이 기계는 "동일 크기 큐브의
규칙 격자" 전용이다. 실제 씬은 (a) 임의 월드 AABB, (b) 오브젝트마다 다른 지오메트리(다른 vbuf/ibuf/
index_count), (c) 다른 머티리얼/파이프라인을 가진다. csCull/csCullHzb/cull_draw.slang는 전부
재작성이 필요하고, cull.rs/hzb.rs의 스칼라 그리드 파라미터(CullGrid)는 per-draw 인스턴스 테이블로
대체된다.

### 2.3 재사용 가능한 것 (버리지 않는 것)

- HzbSystem 전체 — 피라미드 빌드/리듀스/슬롯 연속성 검증/통계는 컬링 대상과 무관한 순수 모듈
  (hzb-occlusion-culling.md:60이 명시). depth만 소비하므로 그대로 재사용.
- 프러스텀 평면 추출 frustum_planes(cull_view_proj) (push.rs:371-384).
- cull_view_proj = proj_noflip * view (main.rs:3535) — Y-flip 없는 매트릭스. 이것이 DX≡VK 컬
  병렬성의 뿌리(§2.4).
- 2-패스 오클루전 방법론 — HZB 문서 §2/§5가 "메인 씬이 GPU-driven이 되면 페이즈 디스패치만 추가"
  라고 미리 설계해 둠. 우리가 바로 그 시점이다.
- HZB 검증이 잡아낸 3버그의 교훈(문서 §4.3): (1) 빌드 소스 extent 명시 전달, (2) 4-탭 코너 텍셀,
  (3) uv.y = 0.5 - 0.5*ndc.y V-플립. 실제 드로우 셰이더에 그대로 이식.

### 2.4 DX≡VK 병렬성 — Y-flip 정확히 어디서 처리하나

- 씬 렌더 프로젝션은 Vulkan에서 proj.y_axis.y *= -1.0로 클립공간 Y를 뒤집는다(main.rs:3500-3502).
  따라서 Vulkan의 depth 이미지와 D3D12의 depth 이미지는 같은 top-down 방향을 가진다(플립이 씬
  래스터에 baked-in).
- 컬 매트릭스는 절대 뒤집지 않는다(cull_view_proj, main.rs:3533-3535). 그래야 가시 집합과 indirect
  카운트가 두 백엔드에서 동일. HZB 투영 UV도 no-flip 기준이라 uv.y = 0.5 - 0.5*ndc.y가 전 백엔드
  공통(cull.slang:144-151).
- 설계 규칙(불변): per-draw 컬 컴퓨트는 반드시 cull_view_proj(no-flip)로 AABB를 투영한다. 프러스텀
  평면도 no-flip vp에서 추출. 이렇게 하면 컬 결과가 백엔드 독립이고, 실제 드로우는 각자 플립된
  view_proj로 그리므로(변경 없음) 이미지가 일치한다. 컬링 온/오프도 이미지 동일해야 하며(가시 집합만
  축소, 색을 안 바꿈), 이것이 S1의 검증 명제다.

---

## 3. 실제 드로우를 컬링에 먹이기 — 데이터 구조 설계

### 3.1 AABB 소스 = fuse_scene의 drawable_aabb (single source of truth)

fuse::fuse_scene(fuse.rs:197-268)는 이미 draw-list 순서로 per-drawable 월드 AABB를 계산한다
(fuse.rs:214-246: xf.transform_point3(v.pos)로 CPU 정점을 월드로 변환, per-object min/max).
build_scene(registry.rs:173-198)과 fuse_scene은 둘 다 world.draw_list()를 같은 순서로 순회하므로
drawable_aabb[i]는 scene[i]와 1:1 정렬된다.

> 단일 소스 규칙(CLAUDE.md #4): AABB를 두 번째로 계산하지 않는다. 대신 fuse_scene의 AABB 계산을
> 얇은 공용 함수로 추출해(예: mesh::world_aabb(cpu, xf) -> (Vec3,Vec3)) fuse_scene과 컬링이 같은
> 함수를 쓴다. 그러면 GDF 퓨즈가 없는 씬(예: 컬링만 켠 순수 raster 벤치)에서도 AABB가 동일 로직으로
> 나온다.

per-mesh 로컬 AABB 캐싱(권장): 정점 순회는 업로드 시 1회면 충분하다. MeshRegistry::push
(registry.rs:81-98)에서 로컬 AABB를 계산해 GpuMesh에 저장(aabb_min/max: [f32;3])하고, per-frame에는
transform으로 8코너를 변환해 월드 AABB를 얻는다(O(1)/draw, 정점 재순회 없음). 스키닝/모핑 오브젝트에
보수적 확장이 필요하지만(§3.4), static 지배 씬에서 정확.

### 3.2 인스턴스 테이블 (GPU 컬링 입력, S2+)

- DrawBounds (per-draw, draw-list 순서): aabb_min/max(월드), draw_index(scene[] 인덱스, visible
  리스트가 되돌아 참조), flags(bit0 casts_shadow, bit1 decal …).
- DrawArgs (per-draw indirect 인자 = VkDrawIndexedIndirectCommand):
  index_count, instance_count, first_index, vertex_offset, first_instance.
- 바운드 버퍼는 CPU가 프레임마다 채운다(static 씬은 사실상 불변 → dirty-skip 가능, §7).
- args 버퍼는 per-draw 하나씩(멀티-draw-indirect) 또는 카운트-버퍼 + ExecuteIndirect/
  drawIndexedIndirectCount. RHI 확장 필요(§5): 현재 draw_indexed_indirect(buf, offset, 1)만 존재
  (cull.rs:337) — 실제 씬은 N draws라 indirect-count 또는 per-draw offset 루프가 필요.

### 3.3 vbuf/ibuf 다양성 문제 (실제 씬 ≠ 단일 큐브)

데모 그리드는 vbuf/ibuf가 하나(공유 큐브)라 single draw_indexed_indirect로 끝난다. 실제 씬은
오브젝트마다 다른 버퍼다. 두 가지 착지 경로:

- S2a (현실적 1차): CPU가 컬링 결과(visible 리스트)를 리드백해 그 부분집합만 direct draw_indexed로
  루프. GPU 컬 + CPU 제출. 1프레임 지연(리드백) 허용. RHI 확장 최소. IntelSponza 155노드에서 CPU
  루프 오버헤드는 무시가능.
- S2b (완전 GPU-driven, 스트레치): bindless 정점/인덱스 풀(하나의 큰 vbuf/ibuf에 모든 지오메트리,
  per-draw first_index/vertex_offset) + multi-draw-indirect-count. cull_draw.slang처럼 VS가
  visible[iid]로 draw 레코드를 페치. 이것이 Phase 14 virtual-geometry의 전신이며, 그 문서가 요구하는
  RHI(§5)와 겹친다.

> 권장: S1(CPU 프러스텀) → S2a(GPU 컬 + CPU 리드백 제출) → S3(HZB) → S2b(풀 GPU-driven, Phase 14와
> 합류). S2a에서 이미 대부분의 페이백을 얻고, S2b는 Phase 14로 접는다.

### 3.4 섀도우 패스 컬링 (별도 프러스텀)

섀도우는 광원 프러스텀이 다르다(light_vp, CSM는 per-cascade slot.view_proj). per-cascade 프러스텀
평면으로 별도 컬. 주의: 카메라에 안 보여도 섀도우를 드리우는 오브젝트는 살아야 하므로 카메라 프러스텀
컬을 섀도우에 재사용하면 안 된다. CSM 캐스케이드별로 frustum_planes(slot.view_proj)(no-flip 광원 vp)로
컬. 오클루전 컬은 섀도우에 부적합(광원 시점 HZB가 없음) — 프러스텀만.

---

## 4. LOD 설계 (거리 / 스크린-스페이스 에러 기반)

### 4.1 LOD 메시는 어디서 오나 (쿡 경로)

현재 쿡 경로: level.rs가 load_or_cook_gltf_scene(level.rs:14,167)으로 glTF를 BC-압축 .dcasset
컨테이너로 굽는다. LOD 메시는 아직 없다. 두 옵션:

- 오프라인 데시메이션(권장): 쿡 시점(cook::load_or_cook_gltf_scene)에서 각 프리미티브를
  meshopt::simplify(Phase 14가 이미 승인 대상으로 지정한 크레이트, phase-14-virtual-geometry.md:70)로
  N단계 데시메이트해 .dcasset에 LOD 체인(Vec<MeshData> per primitive) + 각 LOD의 스크린 에러 경계를
  저장. MeshRegistry가 LOD별 GpuMesh를 업로드하고 핸들이 LOD 배열을 가리키게 확장.
- 런타임 데시메이션: 첫 로드 시 CPU에서 생성. 콜드스타트 비용 큼 → 쿡 캐시에 넣는 게 맞다
  (single-source-of-truth: 쿡이 LOD를 소유).

> 정합: LOD 체인은 .dcasset가 소유하고 MeshRegistry가 로드. fuse_scene의 AABB는 LOD0(풀디테일)
> 기준으로 계산(바운드는 LOD 무관하게 보수적이어야 컬링이 안전).

### 4.2 LOD 선택 (스크린-스페이스 에러)

per-draw로 매 프레임: d = distance(camera_eye, aabb_center); proj_px = aabb_radius/d *
(screen_h/(2*tan(fov/2))); lod = 첫 i where lod[i].screen_error(proj_px) <= tau_px (tau≈1px).
tau는 RenderQuality{low,med,high} 틴어의 파라미터로(CLAUDE.md #3) 단일 상수 블록에 둔다. 팝핑 방지:
이력 히스테리시스(LOD 전환 거리에 ±마진) 또는 두 LOD 크로스-디졸브(초기엔 히스테리시스로 충분).

### 4.3 LOD × 컬링 통합

LOD 선택은 컬 결과(visible 리스트) 이후에, 살아남은 draw만 대상으로 수행. S2a에서는 CPU가 리드백한
visible 집합에 대해 LOD를 골라 draw_indexed의 index_count/vertex_offset을 LOD 메시로 스왑. S2b에서는
컬 컴퓨트가 draw 레코드를 만들 때 LOD를 골라 args를 채운다.

---

## 5. RHI / 셰이더 확장 요구 (블로커 후보)

| 기능 | 현재 상태 | 필요 단계 | 비고 |
|---|---|---|---|
| per-draw 바운드 SoA storage 버퍼 | storage 버퍼 존재 | S2 | create_storage_buffer_init 재사용 |
| multi-draw-indexed-indirect + count | draw_indexed_indirect(buf,off,1)만 | S2b | VK drawIndexedIndirectCount / DX ExecuteIndirect+count. 없으면 S2a(CPU 제출)로 우회 |
| bindless 정점/인덱스 풀 | per-mesh 개별 버퍼 | S2b | Phase 14와 공유 |
| meshopt FFI | 없음 | S4 | Phase 14 승인 대상 (phase-14-virtual-geometry.md:70). 사용자 승인 필요 |
| .dcasset LOD 체인 스키마 | 단일 메시 | S4 | 쿡 컨테이너 버전 범프 |
| StorageBuffer 리드백 | read_into 존재 (hzb.rs:141) | S2a | visible 리스트 리드백에 재사용 |

S1(CPU 프러스텀 컬)은 RHI 확장 0 — 순수 CPU 필터. 이것이 첫 착지가 안전하고 즉시 검증 가능한 이유다.

---

## 6. 단계적 구현 계획 (각 단계 독립 검증)

각 단계는 디폴트 OFF + env seam(CLAUDE.md #3), DX≡VK ≤0.001, 갤러리 바이트-앵커 유지,
PROFILE_GPU before/after를 게이트로 한다.

### S0 — AABB 단일 소스 추출 (무기능, 리팩터)
- fuse_scene의 per-object AABB 계산을 mesh::world_aabb(또는 GpuMesh::local_aabb + 8코너 변환)로 추출.
  fuse_scene이 이 함수를 호출하도록 변경.
- 게이트: 갤러리/Sponza 바이트-앵커 불변(퓨즈 출력 동일). cargo test(퓨즈 라운드트립).

### S1 — CPU 프러스텀 컬 (실제 드로우, env SCENE_CULL=1)
- 프레임당 frustum_planes(cull_view_proj)로 scene[]를 필터해 가시 부분집합 슬라이스를 만들고
  record_gbuffer/record_shadow*/record_prepass/record_decals에 그것을 넘긴다. 섀도우는 광원 프러스텀으로
  별도 필터(§3.4).
- 게이트(핵심): 고정 카메라에서 OFF≡ON 바이트 동일(가시 집합만 축소, 화면 밖 오브젝트는 어차피
  클립됨 → 색 불변). HZB 문서 §4의 "sponza lion-view 0 survived" 방식으로 명제 검증. PROFILE_GPU로
  gbuffer/shadow ms 감소 측정(카메라가 씬 일부만 볼 때).
- 정직한 기대치: Sponza는 대부분 프러스텀 안이라 이득 작음. IntelSponza에서 콜로네이드를 보면 절반
  이상 컬 가능(§7).

### S2a — GPU 프러스텀 컬 + CPU 리드백 제출 (env SCENE_GPU_CULL=1)
- per-draw 바운드 버퍼 업로드 → 새 scene_cull.slang(AABB-vs-6평면, no-flip vp) → visible 리스트 +
  카운트. CPU가 리드백(read_into)해 그 draw만 draw_indexed 루프.
- 게이트: S1과 가시 집합 동일(GPU 컬 == CPU 컬, 결정론적). 바이트 동일. 리드백 1프레임 지연의 팝인
  없음 확인(고정 카메라). PROFILE_GPU: 컬 컴퓨트 ms + 제출 ms.

### S3 — HZB 오클루전 컬 (env SCENE_HZB_CULL=1, 전제 SCENE_GPU_CULL=1)
- HzbSystem 재사용(변경 최소). scene_cull.slang에 csCullHzb의 오클루전 블록 이식(8코너 투영 → 스크린
  AABB → 4-탭 코너 텍셀 max → z_near > hzb_max+eps). HZB 문서의 3버그 교훈 그대로(extent 명시, 4-탭,
  V-플립).
- 2-패스(권장): 실제 씬은 컬 대상 == depth 기여자이므로 HZB 문서 §5가 예고한 two-phase가 성립:
  (1) prev-HZB로 컬 → 렌더 → depth 갱신 → HZB 재빌드 → (2) 1차 탈락분 재검사. false-negative 회수.
  단, S3 첫 착지는 단일-패스 prev-frame HZB(보수적)로 시작해 리스크를 낮추고, 2-패스는 S3.1로.
- 게이트: 가림-헤비 뷰(IntelSponza 벽 뒤)에서 OFF≡ON 바이트 동일 + occlusion-culled 카운트 > 0.
  HZB 문서 §4의 "가림-양성 케이스" 방법론 재사용. PROFILE_GPU: hzb_build + cull ms vs gbuffer 절감.

### S4 — 거리 기반 LOD (env SCENE_LOD=1)
- 쿡 경로에 meshopt 데시메이션 LOD 체인 추가(.dcasset 스키마 범프). MeshRegistry가 LOD 배열 로드.
  visible draw별 스크린-스페이스 에러로 LOD 선택(§4.2), 히스테리시스로 팝핑 방지.
- 게이트: LOD0-only(OFF)와 원경 시각 거의 동일(tau=1px면 원경 차이 서브픽셀). 근경은 바이트 동일(LOD0
  선택). tools/rt-compare.py 잔차로 원경 품질 정량화. PROFILE_GPU: 삼각형 수 감소 → gbuffer/shadow ms
  감소. DX≡VK: LOD 선택은 결정론적 스칼라라 백엔드 독립.

### S5 — 스트리밍 (IntelSponza, env SCENE_STREAM=1, 스트레치)
- §7. world.rs의 청크 스트리밍(Streaming)을 IntelSponza 서브-에셋(main/curtains/trees/ivy)에
  적용하거나, 텍스처 레지던시(비트-압축 .dcasset는 이미 존재)를 카메라 거리로 관리.

### S6 — 풀 GPU-driven (S2b, Phase 14 합류)
- multi-draw-indirect-count + bindless 지오메트리 풀. Phase 14 M2-M4의 전제와 동일 → 별도 착지가
  아니라 Phase 14로 접는다.

---

## 7. 정직한 비용/편익 — 컬링·LOD는 어디서 페이백하나

### 7.1 Sponza 1080p: 컬링은 60fps 레버가 아니다
- 실측 병목은 GDF SW-RT: 서피스 캐시가 해상도 독립적으로 남아 VK ~11.5ms 평탄, 스케일을 낮춰도
  캐시가 안 줄어듦(qhd-perf.md:58). GDF 컴퓨트는 max 품질에서 SM-포화(async-compute 문서의 정직한
  ~1.5ms 오버랩 실링과 부합).
- 지오메트리 컬링 이득은 27ms 중 ~1.5ms 규모 — 60fps(16.6ms) 갭을 메우지 못한다. Sponza는 노드 수가
  적고(단일 에셋) 대부분 프러스텀 안이라 프러스텀 컬 이득도 작다.
- 결론: Sponza 60fps의 레버는 GI/reflect 최적화(다른 워크스트림)와 TAAU 업스케일이지, 컬링이 아니다.
  이 작업을 "Sponza 60fps 작업"으로 팔면 안 된다.

### 7.2 IntelSponza: 여기가 페이백 지점
IntelSponza(level.rs:403 sponza_intel_level, main+curtains; :512 hero는 +tree+ivy 4.9M-tri,
level.rs:507-512)는 155+ 노드, ~8.6GB 에셋의 훨씬 큰 씬이다. 여기서:
- 프러스텀 컬: 콜로네이드/나브 시점은 씬의 상당 부분이 뒤/옆으로 빠진다 → CPU 제출량과 정점 처리량
  모두 큰 절감. S1만으로도 유의미.
- 오클루전 컬(HZB): 벽/기둥이 겹겹인 실내라 가림이 지배적 — HZB의 진가. 벽 뒤 나무/커튼/아이비
  (4.9M-tri!)가 컬되면 gbuffer/shadow 정점·픽셀 비용 급감.
- LOD: 원경 아이비/나무의 4.9M-tri는 스크린-스페이스 에러가 서브픽셀 → 공격적 데시메이트 가능.
  삼각형 수가 곧 gbuffer/shadow ms이므로 직접 페이백.
- 스트리밍: 8.6GB는 VRAM(≈8GB @ RTX 2070S)을 초과 → 레지던시 관리가 필수. 카메라에서 먼 서브-에셋/
  텍스처를 내려야 애초에 프레임이 돈다. 이건 60fps 이전에 OOM 회피 문제.

### 7.3 우선순위 권고
1. S0+S1(CPU 프러스텀) — 싸고 안전, IntelSponza에서 즉시 이득, RHI 확장 0.
2. S3(HZB 오클루전) — IntelSponza 실내 가림에서 최대 페이백. HzbSystem 재사용으로 저비용.
3. S4(LOD) — 4.9M-tri 원경 지오메트리에서 페이백. meshopt 승인 필요.
4. S5(스트리밍) — 8.6GB OOM 회피. 60fps 이전의 실행 가능성 문제.
5. S2b/S6(풀 GPU-driven) — Phase 14로 접어 중복 투자 방지.

---

## 8. 리스크 / 미결
- 바이트 동일성 명제(S1). 프러스텀 컬이 클립 경계 오브젝트를 잘못 떨구면 색이 바뀐다. 보수적 평면
  (정규화된 no-flip vp) + AABB(코너 확장)로 false-cull 0 보장. 게이트가 직접 측정.
- 스키닝/모핑 AABB. GPU 디폼 오브젝트는 로컬 AABB가 프레임마다 변함 → 바인드포즈 AABB를 디폼 한계로
  보수 확장(§3.4). Sponza/IntelSponza는 전부 static이라 초기엔 무관.
- multi-draw-indirect-count DX≡VK. VK drawIndexedIndirectCount ↔ DX ExecuteIndirect+count 시맨틱
  차이. S2b에서 검증. S2a(CPU 제출)가 이 리스크를 우회하는 안전한 1차 경로.
- meshopt 결정론. LOD 데시메이션이 백엔드 독립(CPU 오프라인)이라 DX≡VK 무관하지만, 쿡 결정론(같은
  입력 → 같은 LOD)은 캐시 무결성에 필요.
- HZB TAAU jitter. 업스케일 경로에서 씬 depth는 jitter, 컬 매트릭스는 unjitter → 서브픽셀 false-cull
  이론상 가능(coarse-밉 4탭이 흡수). HZB 문서 §5와 동일 — 필요 시 AABB 1px 팽창.

---

## 9. 검증 전략 (요약)
- cargo fmt + cargo clippy --all-targets -- -D warnings clean.
- 갤러리 바이트-앵커 sha256 (--screenshot-clean) 각 단계 후 불변 확인 (디폴트 OFF seam).
- DX≡VK ≤ 0.001 avg/ch: --backend vulkan vs d3d12 동일 카메라 캡처 diff.
- 컬링 OFF≡ON 바이트 동일: 고정 카메라, S1/S2a/S3 각각 (가시 집합만 축소 명제).
- PROFILE_GPU=1 before/after: gbuffer/shadow/cull/hzb_build 패스별 ms — IntelSponza 콜로네이드 뷰 +
  가림-헤비 뷰 두 구성.
- tools/rt-compare.py: LOD(S4) 원경 잔차 정량화.

---

## 부록 A — 파일:라인 인덱스
- 실제 드로우 루프: deferred.rs:701(shadow), :858(shadow_atlas), :763(prepass), :920(gbuffer),
  :1054(decals). 드로우: 각 draw_indexed(obj.mesh.index_count,0,0).
- 씬 재빌드: main.rs:3360-3367 (build_scene), 호출 registry.rs:173.
- SceneObject: main.rs:194-221. GpuMesh: registry.rs:24-30.
- per-draw AABB 소스: fuse.rs:197-268 (fuse_scene), 필드 fuse.rs:181-183.
- 컬 기계: cull.rs(그리드), hzb.rs(피라미드+통계), cull.slang(csCull/csCullHzb),
  cull_draw.slang(indirect 드로우), hzb_build.slang.
- Y-flip: 씬 main.rs:3500-3502; 컬 no-flip main.rs:3533-3535; HZB V-플립 cull.slang:144-151.
- 프러스텀 평면: push.rs:371-384.
- 레벨/Sponza: level.rs:342(sponza), :403(intel), :458(trees), :512(hero).
- 스트리밍: world.rs (Streaming).
- 쿡: level.rs:14,76,167 (load_or_cook_*).
