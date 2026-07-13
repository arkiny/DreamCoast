# F2 계획서 — mesh-SDF 필드 확장성 (실측 재스코프: 브릭 → 비등방 타일 + f16)

> 상태: **승인·S2b 랜딩** (2026-07-13). 상위 [gi-fidelity-roadmap.md](gi-fidelity-roadmap.md) §F2,
> [gi-fidelity-phases.md](gi-fidelity-phases.md) F2. 실측 프로브는 S1 분석 프리미티브의 의도된
> 사용(수치는 §2에 보존, 프로브 코드는 미커밋).

## S2a 검증 결과 (랜딩 커밋)

- **아틀라스 217×306×215: f16 28.6 MB / f32 57.1 MB** — 원점(cubic cap32 f32) 71.0 → S2b 35.5 →
  **S2a+S2b 28.6 MB (×0.40)**, 동시에 per-axis 캡 48로 롱축 해상도 +50%(32→48), 얇은 축은 목표
  0.05 m 유지(과해상 제거가 재원). `P11_ATLAS_F16=0` A/B 확인.
- **갤러리 앵커 PASS. 콘텐츠 PT budget PASS**: sunlit 30.629 ≤ 30.9(중립, +0.03),
  **interior 34.155 ≤ 34.65 — 기준선 34.350 대비 −0.20 실측 개선**(커튼·근접 지오가 per-axis
  해상도 수혜 — 예상 방향 그대로).
- `P_SC_VIZ` 시각: 카드 등록 건강(구멍/체커보드 없음). 워크스페이스 테스트 전부 PASS(asset 82종
  포함, per-axis 클램프 유닛테스트 추가). clippy/fmt clean.
- 쿡: dims가 캐시 키에 해시되므로 **1회 자동 재쿡**(per-mesh ~30 s 병렬 + 씬 SDF/albedo 레벨 수 분,
  이후 캐시 히트). dcasset SDF/albedo 청크는 dims[3] 인코딩(구 파일은 키 불일치로 자연 무효).
- 샘플러/셰이더 **무변경**(uvw bias/scale는 원래 축별) — 배선은 CPU 측 뿐.

## S2b 검증 결과 (랜딩 커밋)

- **메모리: 71.0 → 35.5 MB (×0.50)** — `per-mesh SDF direct sample: atlas 274x238x272 (35.5 MB,
  f16)` 로그. `P11_ATLAS_F16=0` → f32 71.0 MB 복원(A/B seam).
- **갤러리 앵커 `65d04ceca2c4…` PASS** (per-mesh 경로 콘텐츠 전용 + 실행 확인).
- **콘텐츠 PT budget PASS(F6B 게이트 첫 실전)**: sunlit masked_avg **30.605** ≤ 30.9(f32 기준선
  30.601/30.606과 동일 수준 — f16 영향 = 노이즈 이하), interior **34.372** ≤ 34.65(기준선 34.350,
  Δ+0.02 마진 내).
- 유닛테스트: `f16_conversion_roundtrip`(RNE·서브노멀·클램프) + `f16_bytes_match_voxels`(결정론).
  clippy/fmt clean. DX≡VK: R16Float 매핑 3백엔드 추가(컴파일 검증) — Windows 배치에 편입.

## 0. 왜

로드맵 F2: mesh-SDF 저장이 "dense `dim³` 타일, 캡 32³"라 **메모리·해상도·메시 수 상한 = 확장성 제약**.
F2 원안(S2)은 "occupied 브릭만 저장 + 인디렉션"이었고, S1(브릭 점유 분석 프리미티브,
`sdf_atlas.rs:375-551` `classify_bricks`/`analyze_bricks`)이 1차 증분으로 랜딩돼 있다 — **S2를 실측
위에서 설계하라는 목적**. 이 문서가 그 실측이고, 결과가 원안을 반박해 재스코프한다.

## 1. 현행 파이프라인 (조사 완료 — 반복 조사 금지)

- **베이크**: per-mesh `SdfVolume` — 정육면체 그리드 `dim³`(dim = `clamp(longest/0.05m, 8, 48)`,
  `sdf.rs:716-733`)를 **비등방 AABB 위에** 깐다 → 복셀이 이미 축별 비등방(얇은 축은 과해상: 10×3×0.2m
  벽 at dim48 → 복셀 0.21/0.0625/**0.004m**). 쿡 캐시(`load_or_bake_mesh_sdf`), 결정적.
- **팩**: `SdfAtlas::pack_capped`(런타임, `main.rs:2328`) — 캡 32(`P11_ATLAS_MAX_DIM`≤48) 초과 타일을
  trilinear 다운샘플 후 결정적 3D 셸프 팩(GUTTER=1). R32Float 단일 볼륨.
- **샘플 seam**: `mesh_sdf_sample.slang` `ms_geo`/`ms_albedo` ← `clipmap.slang` `count==0` 위임(단일
  스위치) ← 모든 SW-RT 소비자. **`uvw_bias/uvw_scale`은 이미 축별** — 비등방 타일에 샘플러 무변경.
- **인스턴스**: 112B 레코드(inv_world 3행, aabb+dist_scale, uvw bias/scale) + res³ 셀그리드. 콘텐츠
  전용(`use_permesh = !gallery_scene`) — **갤러리는 이 경로를 아예 안 탄다**(앵커 구조 안전).
- **실측 베이스라인** (sponza_intel, 2026-07-13): 426 유니크 메시(**276개가 베이크 캡 48에 걸림**),
  아틀라스 274×238×272 = **71.0 MB**(payload 47.1 MB + 셸프/거터 ~50%), coarse 씬 GDF 48³ 별도.
- 헤더/플래그 함정: `MeshSdfHeader`는 CPU/GPU 바이트 계약(`gdf.rs:934-1002` ↔ slang Load4), spare는
  w7.y/z/w 3개뿐. `flags` bit0(detail-replace)은 `gdf_reflect.slang:761`도 **독립 디코드**. 112 스트라이드는
  `ms_geo`/`ms_albedo` 두 루프에 하드코딩(수정 시 양쪽).

## 2. S1 프리미티브 실측 — 브릭 원안 반박 (sponza_intel, 캡 전 native 필드)

| 프로브 | 조건 | 결과 |
|---|---|---|
| A: iso-해상도 브릭 | margin 2/3/4 복셀 밴드 | 점유 **74.4/80.7/85.4%** — dense 138.8 MB → 브릭 **209.7/227.3/240.6 MB (×1.51/1.64/1.73 손해)** |
| B: 블랭킷 해상 상승 | 캡 메시(276개) 48→96/128 업샘플 후 브릭 | 점유 61.6/46.4% — 브릭 **1.18/2.10 GB** (불가) |
| C: 비등방 dims | 축별 `clamp(ext/0.05, 4, cap)` | cap48: **46.4 MB**(f16 **23.2**) ≈ 현행 cubic-cap32 payload 47.1 MB / cap96: 172.2(86.1) / cap128: 281.4(140.7) |

**결론(정직):**
1. **브릭 전제("dense 대부분이 far-from-surface 낭비")는 이 콘텐츠에서 성립하지 않는다.** 현행
   해상도에서 점유 74%+ — (8+2)³/8³≈1.95 apron 오버헤드 손익분기(점유 <~51%)를 크게 상회. 구겨진
   커튼·트림 등 표면 밀도가 높은 지오가 지배적. 브릭은 **점유 <40%가 확인되는 씬/해상도에서만** 유효
   (원안 S2 보류, S1 프리미티브·이 실측을 재평가 조건으로 문서화).
2. **실측이 가리키는 확장성 레버 = (a) 얇은 축 과해상 제거(비등방 dims) + (b) f16 스토리지.**
   합산 시 payload 47.1→23.2 MB(×0.49), 셸프 포함 추정 71→~35 MB — **동일 품질에서 메모리 반감**
   = 같은 예산으로 유니크 메시 ~2×(로드맵 F2의 확장성 목표 그 자체).

## 3. 재스코프 스테이지 (각각 검증된 커밋, opt-in seam)

### S2b — f16 아틀라스 스토리지 (먼저: 업로드 전용, 베이크/쿡 무변경)
- rhi `Format::R16Float` 추가(3백엔드 매핑 — VK `R16_SFLOAT`/DX `R16_FLOAT`/Metal `R16Float`,
  `Rg16Float` 전례). **per-mesh 아틀라스 볼륨만** f16(+ F5 albedo 아틀라스 동반) — 갤러리가 쓰는
  dense/scene 볼륨은 R32F 유지.
- 팩 출력에 f16 변환(`to_le_bytes_f16`). 정밀: 거리 ≤ ~30m에서 상대 2^-11(≈1.5cm@30m, <0.5mm@1m),
  march epsilon(0.003)·PT 게이트로 검증. `P11_ATLAS_F16=1` 기본 ON(콘텐츠), `=0` seam.
- **게이트**: `sponza_pt_sunlit ≤ 30.9` / `interior ≤ 34.65`(F6B budget — 첫 실전 사용), 갤러리 앵커
  불변, 메모리 로그(71→~35.5 MB), 결정론, clippy/fmt. DX≡VK 배치에 R16Float 매핑 추가.

### S2a — 비등방 per-mesh dims (베이크+팩; 샘플러 무변경)
- `SdfVolume.dim: u32` → `dims: [u32;3]`(축별 `clamp(ext/0.05, 4, 48)`), `bake_sdf_from_fused`
  인덱싱/셸프 footprint per-axis(셸프는 이미 가변 크기), `tile_uvw`는 이미 축별. albedo 볼륨 동반.
  쿡 캐시 **버전 범프**(1회 재쿡, 병렬 ~수초). 총 복셀 수 감소 → 베이크도 빨라짐.
- 캡은 **축별 48 유지**(iso-예산 순수 절약). 캡 상향(64/96) 재투자는 랜딩 후 PT 잔차로 별도 측정·결정
  (측정 없는 단정 금지).
- **게이트**: S2b와 동일 + 콘텐츠 tolerant 골든(sc_viz/gdf_ao) 시각 검토, payload 로그(47.1→~46.4
  MB@f32 기준 — f16 합산 23.2).

### S3 — 기각/보류 기록 + 후속 분리
- **브릭(원안 S2): 보류** — §2 실측 명시, 재평가 조건(점유 <40%: 도시급 씬·훨씬 높은 해상도)과 함께
  `gi-fidelity-phases.md` F2 갱신. S1 프리미티브는 측정 도구로 유지.
- **mip 체인(cone LOD)**: 유효하나 소비자 시그니처 대공사(`cm_geo_*`에 cone 파라미터 → 전 소비자) —
  **별도 페이즈로 분리**해 F4(계층 라디언스 캐시+중요도 게더)와 우선순위 비교 후 착수.

## 4. 게이트 (프로젝트 불변)
- **갤러리 바이트 동일** `65d04ceca2c4…`(per-mesh 경로는 콘텐츠 전용 — 구조 안전 + 실행 확인).
- **콘텐츠 PT 잔차 budget**(F6B): sunlit ≤30.9, interior ≤34.65 — 개선/중립 강제.
- **결정론**: 베이크·팩 결정적 유지(현행 셸프 순서 보존), run-to-run.
- **DX≡VK**: R16Float 매핑 + f16/비등방 콘텐츠 캡처를 동결 Windows 배치에 추가. Metal 우선 검증.
- **단일 소스**: seam(`clipmap.slang` count==0) 무변경, 헤더 계약 양측 동기, 캡/포맷 상수 1곳.
- 성능: `PROFILE_GPU`로 march 소비자 비용 중립 확인(탭 수 불변 — f16은 대역폭↓ 기대).

## 5. 하지 말 것
- 측정 없이 캡 상향/브릭 재시도. dense coarse 필드 해상 상향(하이브리드 원칙 — 정밀은 atlas 몫).
- 갤러리 경로(fused dense)·`sample_surface_cache_cone`(FP-민감 앵커) 접촉.
- 프로브 코드 커밋(이 문서에 수치 보존 후 revert).

## 6. 검증 계획
S2b/S2a 각각: 갤러리 앵커 → `golden-image.py --only sponza_pt_sunlit --only sponza_pt_interior` PASS →
메모리 로그 before/after → sc_viz/gdf_ao 시각 → clippy/fmt → 커밋. 완료 후 `gi-fidelity-roadmap.md`
§F2·`gi-fidelity-phases.md` 갱신(브릭 기각 실측 포함).
