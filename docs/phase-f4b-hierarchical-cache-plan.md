# F4B 계획서 — 계층 라디언스 캐시 본편: fine 레벨 기본 편입·재중심·소비자 통합

> 상태: **계획 — 승인 대기** (2026-07-14). 상위 [gi-fidelity-roadmap.md](gi-fidelity-roadmap.md) §F4,
> 선행 [phase-f4-hierarchical-radiance-cache-plan.md](phase-f4-hierarchical-radiance-cache-plan.md)
> (1차 증분: 카메라-앵커 fine 레벨, `P_GI_VOL_CLIP` opt-in) ·
> [phase-gi-volume-leak-plan.md](phase-gi-volume-leak-plan.md) §10–14(새 그라운드 트루스:
> 디스패치 버그 수정·슬랩 상각·고정 방향 세트·B2·반사 V-게이팅·재기준선). 승인 전 코드 변경 없음.

## 0. 왜 (측정으로 확정된 한계)

GI-볼륨-누설 페이즈가 Metal 디스패치 버그(볼륨 3/4 제로-init)를 고치고 필드를 정상화한 뒤,
남은 잔차의 바닥이 **32³ SH-L1 단일(coarse) 필드의 표현 한계**로 확정됐다:

- 차폐 석벽/아케이드 하부의 앰비언트 잔차가 diff 몽타주에서 균일 과충전으로 확인(색은 PT 정합:
  크롭 B−R −0.62 vs PT −0.58). 프로브 간격 sponza ~1.36 m라 V/E가 벽을 가로질러 스미고, 깊은
  크롭은 고정 EV11에서 여전히 PT의 ~2배대.
- **F4 1차 증분 재시험(정상 필드)**: `P_GI_VOL_CLIP=1`이 interior 게이트 **−0.35 유효**
  (과거 "동률·무효" 판정은 죽은 필드 위 측정 — 기각 기록 갱신됨, gi-volume-leak §14b).
  기본 편입의 유일한 차단자 = **업데이트 비용 2×**(fine이 디스패치 높이 2배 → 매 슬랩에 얹힘).
- **이번 조사에서 확인한 측정 교란(순서 재구성의 근거)**: gdf_gi.slang의 fine 소비 블록(:167)은
  fine/coarse **양쪽** 소비를 모두 처리하고 조기 `return`하므로, `P_GI_VOL_CLIP=1`이면 occ-가중
  소비(`P_GI_VOL_OCC`, 정상 필드 실측 **−0.77**)가 **화면 전체에서 무효**가 된다. 즉 기존
  "fine −0.35"는 occ 없이 잰 값 — 두 효과는 비가산이고, occ 포팅을 기본 편입 **이전에** 랜딩해야
  기본 ON 측정이 오염되지 않는다.

성공 지표 = interior budget 하향 + 깊은 크롭 → PT 방향 + 셔머·비용 무회귀.

## 1. 현재 구조 (2026-07-14 코드 조사 — 반복 조사 금지)

### 1a. 슬랩·핑퐁 프로토콜 (coarse 단독, 현행)

```
frame_no    : 0     1     2     3   | 4     5     ...        (period P = 4, Low/Med/Apple —
z-슬랩      : 0-7   8-15  16-23 24-31| 0-7   8-15             High는 1; scalability.ron:53/100/144/191)
write slot  : [------ gi_vol_frame%2 ------]| flip            slab = ceil(32/P) slices
advance     :                   ^ frame_no%P==P-1 (main.rs:8748-8753)
reset       : gi_vol_frame==0 (첫 사이클 전체 — EMA 스킵, 신선 기록; gi.rs:437)
```

P∤32이면 div_ceil 겹침으로 일부 슬라이스가 사이클 내 2회 기록("정확히 1회"가 아니라 "최소
1회·중복 멱등" — read 슬롯이 사이클 중 동결이고 시드가 사이클 내 상수라 2회째 기록이 비트
동일; 실제 티어는 4|32·1|32). 현행 코드의 잠재 불일치: 디스패치는 클램프된 period
(main.rs:6785), advance(:8749)와 지터 시드(:6812)는 언클램프 — 새 스케줄은 **클램프 P를 한 번
유도해 슬랩·advance·지터 모두에 단일 사용**(승계 금지).

- 디스패치: `z_offset=(frame_no%P)*slab`, `z_count=slab`(main.rs:6816-6817); 셰이더가
  `tid.z += pc.read_rgb.z`(gi_volume.slang:233). fine ON이면 **gy 2×**(gi.rs:521-525) — 매 슬랩에
  fine이 2× 비용으로 얹히는 것이 현행 구조.
- 소비자는 **이번 프레임 기록 중인 슬롯**을 읽고(신선 슬랩 + 2사이클 전 슬랩 혼합,
  gi.rs:332-341), 업데이트의 히스토리/멀티바운스 읽기는 반대 슬롯(직전 완결 사이클, gi.rs:424).
- `P_GI_STABLE=1` 기본: 지터 인덱스 0 고정 → 프로브별 결정적 방향 세트 → 정적 씬에서 업데이트가
  **멱등 덮어쓰기**(main.rs:6802-6813). EMA(α=0.1, main.rs:6815)는 노이즈 평균이 아니라
  멀티바운스 피드백의 감쇠 반복(damped iteration) 역할만 남음 — §4의 α 레버 근거.

### 1b. fine 패킹·소비 (F4 1차)

- 스토리지: 같은 볼륨의 **높이 2×**(coarse y∈[0,32), fine y∈[32,64)) — 신규 bindless 슬롯 0
  (gi.rs:97-107,263-267). 볼륨 소비자 uvw 리맵 `gv_level_uvw`(gi_volume.slang:141-159,
  finest-first containment + half-height 복셀센터 클램프).
- fine box: 시동 시 1회 고정(main.rs:2918-2945, half=(ext/6).clamp(4,12)=sponza 7.25 m, 간격
  0.45 m), `set_gi_fine_box`(gi.rs:352-379)가 32 B 스토리지 버퍼 업로드. **정적 캡처 전용.**
- 소비자 현황: ① gdf_gi.slang fine 블록(:167-233, **occ 미적용 — 조기 return이 :250 occ 블록을
  가림**) ② gdf_reflect.slang 4함수 전부 **coarse 고정 리맵**(sample_gi_irradiance:262 /
  sample_sky_vis:299 / sample_sky_vis_dir:325 / sample_gi_irradiance_valid:355 — `dh==dw*2`이면
  y를 coarse 반부로) ③ sdf_cache_light.slang sky-vis도 coarse 리맵(F4 1차에서 추가).

### 1c. 이번 본편이 쓸 스페어 seam (확인 완료)

- `GiVolumePush` 스페어 6슬롯(push.rs:1235-1298 전수): `write_rgb.z`(88..92, 호스트가 항상 0
  패킹 — gi.rs:498) → **fine 전용 슬랩 디스패치의 tid.y 오프셋**(§4); `fine_max.w`(188..192
  제로) → **fine-reset 플래그**(§8); 그 외 write_rgb.w/albedo_rgb.w/clip.z/clip.w 예비.
- `GiPush`(gdf_gi)는 256 B **캡 도달**(gi.rs:350) — 소비측 신규 fine 파라미터(edge-fade 마진 등)는
  `gi_fine_buf` 32 B 스토리지 버퍼의 **행 확장**으로(푸시 무성장).
- `ReflectPush`는 **240 B가 실질 캡**(256 B로 키우면 D3D12 root budget 초과 → CBV 스필 ~4 ms,
  push.rs:1769-1773). 대신 `flip_y` **bits 15..31이 미사용**(셰이더는 bits 0..14만 디코드,
  gdf_reflect.slang:565-573) — fine-buf 스토리지 인덱스를 bits 15..26에 12비트로 실음(§6 —
  기존 7비트 관례가 왜 부족한지 포함).
- `gi_fine_buf`는 `create_storage_buffer_init`(host-visible) — `StorageBuffer::write`
  (rhi/src/lib.rs:203-215)로 갱신 가능. 단 in-flight 프레임이 읽는 중 host-write는 하자드 →
  재중심은 **2-버퍼 링**(푸시가 매 프레임 인덱스를 나르므로 스왑 무비용)으로.
- sdf_cache_light.slang(sky-vis 4번째 소비자)은 **self-describing coarse 리맵**(:382-385) —
  fine 기본 ON에도 호스트 변경 0으로 안전(§6에서 의도적 coarse 유지 근거).
- `threads_per_group`은 빌드-생성 테이블 `COMPUTE_GROUP_SIZES` 단일 소스(gi.rs:139-144,
  build.rs:689-792 — numthreads 파싱, 미해석 시 빌드 패닉) — 신규 컴퓨트 파이프라인 추가 시에도
  리터럴 금지(이번 본편은 신규 엔트리포인트 없음 예정).

## 2. 증분 구성 (지시 순서에서 재구성 — 근거 포함)

| # | 증분 | 계획서 절 | 왜 이 순서 |
|---|---|---|---|
| 1 | fine 소비 occ-가중 포팅 | §3 | §0의 측정 교란 제거 — 기본 ON 측정(3)의 선행 조건 |
| 2 | 슬랩 레벨-인터리브 스케줄 | §4 | 비용 2×→1×(평탄) — 기본 ON의 유일 차단자 해소 |
| 3 | fine 콘텐츠 기본 ON + 재기준선 | §5 | 1+2 랜딩 후 공식 러너로 판정·`--update` |
| 4 | 반사 소비자 fine 폴스루 | §6 | cache-tone parity 영역 — 회귀 격리 별도 커밋 |
| 5 | fine/coarse edge-fade | §7 | 기본 ON이 노출하는 하드 seam의 소비측 완화(6의 전환 pop도 완화) |
| 6 | 카메라 재중심(EMA 재수렴) | §8 | 프로토콜 위험 최대 — 정적 게이트 무영향을 확인하며 마지막 |
| 7 | (조건부) 레벨 3 캐스케이드 | §9 | 6까지의 측정이 요구할 때만 |

지시 원안(1 비용→2 occ→3 반사→4 재중심→5 fade)과의 차이: occ 포팅을 최선두로(측정 교란),
edge-fade를 재중심 앞으로(기본 ON 직후부터 모든 콘텐츠 프레임에 보이는 seam을 먼저 처리, 재중심
전환 pop 완화 겸용).

## 3. 증분 1 — fine 소비의 occupancy-가중 포팅 (gdf_gi.slang)

**변경점**: fine 블록(:167-233) 내부에 `pc.vol_occ != 0` 분기 추가 — coarse occ 블록(:250-340)과
같은 수동 2×2×2 트라이리니어를 **선택된 레벨의 논리 격자**에서 수행:

- containment으로 레벨 선택(fine box 우선 — 기존과 동일), 레벨 박스 기준 격자 좌표 계산.
- 코너 프로브 월드 위치 = 레벨 박스 매핑, 제외 판정 = `min(cm_geo_inside, y−GROUND_Y) ≤ 0`
  (재료는 push에 기보유: clip_desc/clip_count/aabb_max.w/aabb_min.w).
- Load 좌표 = `int3(ip.x, ip.y + (fine ? ldim : 0), ip.z)`(물리 y 오프셋), radiance 12권·sky-vis
  4권·bent에 **같은 가중 일관 적용**(coarse occ 블록과 동일 규약).
- 전 코너 제외 → E=0·V=0(반사 선례의 "genuinely dark" 수용 — fine 간격 0.45 m라 coarse보다 유효
  코너 확보가 오히려 쉬움; 몽타주에서 검은 슬리버가 보이면 coarse 폴백을 후속으로).
- `vol_occ==0`이면 기존 하드웨어 트라이리니어 fine 경로 그대로(명령열 보존 규약 — 분리 블록).

**게이트**: opt-in 상태 측정(`P_GI_VOL_CLIP=1` 수동 행간 비교 — knob 비교는 수동 유효 규약):
CLIP=1+occ vs CLIP=1 구(occ 무효) vs CLIP=0+occ. 기대 = CLIP=1+occ가 양쪽 단독보다 개선.
**어두운 슬리버 크롭 비교 필수**: CLIP=1+occ vs CLIP=0+occ(현 출고 기본) — fine 0.45 m 8코너가
전멸하는 밀집 지오메트리에서 coarse 1.36 m 코너는 유효할 수 있어 행 평균이 가리는 국소 암화
방향이 존재(코스 폴백 보류 판단의 근거 데이터). 기본 경로(CLIP=0)는 무변경이므로 공식 게이트는
불변 확인만(풀 매니페스트 — §10). 갤러리 앵커(볼륨 분기 자체가 콘텐츠 전용) + clippy/fmt.
비용: coarse occ 블록과 동일 구조(+8 GDF 탭, half-res) — 이미 기본 ON인 비용 계열, PROFILE_GPU
벽시계 스팟체크.

## 4. 증분 2 — 슬랩 레벨-인터리브 스케줄 (비용 2×→1× 평탄)

**설계**: fine ON일 때 매 프레임 **한 레벨의 한 슬랩만** 디스패치(gy는 항상 8그룹=32행):

```
frame_no    : 0      1      2      3      4      5      6      7    | 8 ...
level       : C      F      C      F      C      F      C      F    | C        level = frame_no & 1
slab z      : 0-7    0-7    8-15   8-15   16-23  16-23  24-31  24-31|          slab_idx = (frame_no>>1) % P
advance     :                                                  ^ frame_no % 2P == 2P-1
reset       : gi_vol_frame==0 (첫 슈퍼사이클 2P프레임 전체)
```

- 셰이더: `tid.y += pc.write_rgb.z`를 슬랩 z 오프셋(:233) **바로 옆(시드 계산 :252 이전)**에 추가
  — C 디스패치는 0, F 디스패치는 `GI_VOL_DIM`. 기존 `level = tid.y / dims.y`·EMA 자기읽기 puvw·
  가드·**RNG 시드 해시(물리 tid 기준)**가 전부 그대로 성립 → fine 프로브의 방향 세트가 현행
  2×높이 디스패치와 비트 동일(F4 1차 측정과의 비교 가능성 보존). `write_rgb.z==0`이면 결과
  동일(레거시).
- 호스트(main.rs 슬랩 배선 + gi.rs record 시그니처에 y_offset/gy 인자): fine OFF는 현행 스케줄
  그대로(문자 그대로 무변경 경로). fine ON은 위 표 — 슈퍼사이클 2P(=8프레임)당 양 레벨 전 텍셀
  1회, advance는 슈퍼사이클 말로 이동. 스케줄은 `frame_no`의 순수 함수(결정론). High 티어
  P=1(scalability.ron:144)에서도 성립(슈퍼사이클 2프레임 — 짝수 C 전그리드/홀수 F 전그리드).
  `P_GI_STABLE=0` A/B 경로의 지터 시드는 사이클→슈퍼사이클 인덱스(`frame_no/(2P)`)로 대응
  (main.rs:6809-6813 — 슬랩이 사이클마다 같은 방향을 재추적하는 기존 의미 보존).
- **양 디스패치의 푸시는 write_rgb.z(y 오프셋)와 슬랩 z 필드만 다르고 나머지 동일 — 특히
  `fine_min.w`(fine_active)는 C 디스패치에도 1.0 유지**: levels/가드/EMA 자기읽기 분기
  (gi_volume.slang:238-243, :389)가 디스패치 모양이 아니라 fine_active에 키잉되므로, C 디스패치에
  0을 넣으면 coarse 프로브의 EMA 자기읽기가 레거시 `read_coeffs`(전체 64텍셀 높이를 32 수학으로)
  경로로 빠져 오염된다.
- **비용**: 프레임당 프로브 수 = coarse 단독과 동일(32×32×8×spp16) → **~2 ms/frame 평탄 유지**
  (레벨 교대는 슬랩 크기가 같아 프레임간 비용 균일 — AE 펌핑 없음). 목표: fine ON 벽시계 기여가
  coarse 단독 대비 +10% 이내.
- **트레이드오프(정직)**: 레벨당 갱신 주기 P→2P — EMA 수렴 벽시계 2× 감속. WARMUP_FRAMES=192
  기준 사이클 48→24회: 잔차 (0.9)^24≈8% vs 현 0.6%(멀티바운스 증분에만 작용하므로 절대 영향은
  그 일부). **보상 레버 준비**: `P_GI_VOLUME_ALPHA`(신규 env, 기본 0.1 유지) — P_GI_STABLE 고정
  방향 세트에서 EMA는 노이즈 평균이 아니라 피드백 감쇠라(§1a) α 상향(0.15/0.2)이 셔머 중립일
  가설, 게이트가 수렴 결손을 보이면 스윕으로 판정. 선제 변경은 하지 않음.
- **게이트**(아직 opt-in): CLIP=1 수동 행간 — 인터리브 전후 동률 기대(수렴 잔차만 차이),
  CAPTURE_SEQ=8 셔머(기준 0.049 무회귀 — 멱등 덮어쓰기라 cadence는 셔머 무관 가설의 검증),
  192프레임 벽시계 A/B(fine OFF vs ON — ≤+10%), CLIP=0 공식 러너 불변(풀 매니페스트 — §10),
  갤러리 앵커.

## 5. 증분 3 — fine 콘텐츠 기본 ON + 공식 재기준선

- `P_GI_VOL_CLIP` 기본 = **콘텐츠 ON / 갤러리 OFF**. 메커니즘(정찰 확정): `gallery_scene`
  (main.rs:1495)은 env 전용 계산(`WORLD`/`LEVEL`/`SCENE_GLTF`)이라 GiSystem::new 호출부
  (main.rs:1390) **위로 호이스트 가능** — 기본값을 `!gallery_scene`으로 두 읽기 지점
  (gi.rs:255 할당 + main.rs:2918 박스 설치)에 **락스텝** 적용. 갤러리는 단일 높이 할당 유지 —
  앵커 위험 원천 차단. `P_GI_VOL_CLIP=0` 킬스위치 유지.
- fine box 파라미터는 1차 값 유지(half=(ext/6).clamp(4,12)): 게이트가 요구할 때만 half 스윕.
- pbr.slang 멀티바운스 부스트는 볼륨 경로 조건부 0(main.rs:3704-3712) — fine도 같은 무한
  피드백 구조라 로직 그대로 유효(변경 없음).
- **게이트(공식 러너만으로 판정 — 수동 앵커 함정 규약)**: `python tools/golden-image.py --only
  sponza_pt_sunlit --only sponza_pt_interior` budget 28.58/32.87 — interior 개선 기대(1차 재시험
  −0.35 + occ 결합), 개선 시 `--update` 하향 재기준선(매니페스트 수기편집 금지). **sunlit도 자체
  fine box 완전 활성으로 판정**(fine box는 캡처별 CAM_EYE 중심 — sunlit CAM_EYE=0,2,0은 아트리움
  중심이라 fine box가 게이트 화면을 정통으로 덮음): budget 28.58에 개선-또는-중립, 회귀는 명시적
  차단자(sunlit 이동은 잡음이 아니라 신호). EV11 크롭 분해(interior x 0–900) → PT 방향·
  B−R ≈ −0.6 유지. 셔머 0.049 무회귀. 갤러리 앵커 65d04ceca2c4dbff 바이트 불변. run-to-run
  ≤0.005. DX≡VK Windows 배치에 "fine 기본 ON 콘텐츠 캡처" 추가(동결 중).
- **SHA-정확 콘텐츠 골든 2종 사전 선언**(풀 매니페스트 러너 — §10): `sponza_gdf_ao`는 볼륨
  비소비라 바이트 불변 요구. `sponza_sc_viz`는 표면 캐시 relight가 sky-vis SH를 소비
  (sdf_cache_light.slang:382-390)하므로 기본 ON(더블-하이트 + 인터리브 cadence)에서 WARMUP=64
  시점 필드 상태가 수치적으로 달라져 **바이트 변경이 예상됨** — 변경 시 시각 검토 후 같은 커밋에서
  `--update --only sponza_sc_viz` 재기준선(불변이면 그대로). 변경을 사후 발견이 아니라 사전
  선언으로.

## 6. 증분 4 — 반사 소비자 fine 폴스루 (별도 커밋)

- gdf_reflect.slang 4함수(:262/:299/:325/:355)의 coarse 고정 리맵을 finest-first containment로.
  fine AABB 전달: `ReflectPush`는 240 B 실질 캡(§1c)이라 **같은 `gi_fine_buf` 32 B 버퍼의
  스토리지 인덱스를 `flip_y` bits 15..26에**(base+1, **12비트** — 0 = off = 레거시 coarse 리맵
  명령열 보존) 인코딩. ⚠️ 기존 7비트 enc 관례(push.rs:1800-1805)는 64-엔트리 `volumes[]` 전용 —
  `gi_fine_buf`는 **2048-엔트리 스토리지 테이블 인덱스**(rhi-metal device.rs:55)이고 씬 빌드
  뒤에 할당되므로(메시당 2버퍼 + per-mesh SDF 뒤) Sponza에서 127을 확실히 초과한다(검증 확정).
  패킹부에 `debug_assert!(idx + 1 < (1 << 12))` + push.rs:1790의 스테일 "<64" 주석 정정 동반.
  gdf_gi와 버퍼 단일 소스 — 재중심(§8)의 소비-비활성 윈도(무효 박스)가 반사에도 자동 적용된다.
- 게이트-활성 호출부(전수, 정찰 A): SW 폴백 E(:1319-1321)·폴백 스카이라이트 필(:1334-1336)·
  스페큘러 miss V-게이팅(:1363-1365)·컴팩트 HWRT 히트라이팅(:1168-1176) — Apple 하이브리드
  기본 경로 전부. `sample_gi_irradiance_valid`(bit13)는 Apple 기본 OFF지만 같은 규약으로 포팅
  (논리 격자 기반이라 y 물리 오프셋만 — 증분 1과 동일).
- 히트점은 씬 전역이므로 fine 폴스루는 근거리 반사에만 유효 — 밖은 기존 coarse(무회귀 경로).
- sdf_cache_light.slang의 sky-vis coarse 리맵은 **의도적으로 유지**(캐시 카드 relight는 씬
  전역·상각 갱신이라 카메라-앵커 레벨 결합이 부적절 — 상각 주기와 재중심이 얽히면 카드별 시점
  불일치; self-describing 리맵이라 fine 기본 ON에도 무변경 안전, §1c).
- **게이트**: cache-tone parity 영역 규약 — PT budget 쌍 + 크롭 B−R 유지 + 반사 전용 스팟
  (크롬볼/그림자 반사 값 행간 비교) + 갤러리 앵커, 단독 커밋으로 회귀 격리.

## 7. 증분 5 — fine/coarse edge-fade

- gdf_gi fine 블록의 하드 containment(gdf_gi.slang:176)을 경계 밴드에서 블렌드로: fine box
  경계까지의 정규화 거리로 `w = smoothstep(0, m, d_edge)`(m = half의 ~15%), 밴드 안에서
  fine·coarse **양쪽 샘플 후 lerp** — E·V는 SH 계수에 선형이라 재구성 후 lerp 동치; **bent는
  lerp 후 재정규화**(정규화 벡터 lerp는 길이<1 — 기본 소비자는 재정규화하나 opt-in
  `spec_occlusion`의 bent_unit 계약(pbr.slang:161-169)은 raw dot이라 위반; 반대 방향 lerp가
  ~0이 되면 기존 길이<1e-4 제로-벡터 폴백에 정확히 안착). 밴드 밖은 기존 단일 샘플(비용 불변).
  마진 m 파라미터는 GiPush가 캡이므로 `gi_fine_buf` 버퍼 행 확장으로 전달(§1c).
- 소비자 적용 순서: gdf_gi(+ occ 변형) 먼저; 반사(§6)는 랜딩되어 있으면 같은 규약 포팅(별도
  커밋, cache-tone 게이트 재실행). **업데이트측(gi_volume.slang `gv_level_uvw`:141의 히트점
  읽기)은 이 증분 스코프 밖 — 후속 측정으로**(히트점 읽기는 EMA로 평활되어 seam 민감도 낮음 —
  선제 변경 금지 원칙).
- **게이트**: 공식 러너 쌍(중립~개선), interior 몽타주에서 box 경계 seam 소거 확인(시각),
  셔머 무회귀, 밴드 픽셀의 2× 샘플 비용 벽시계 스팟체크.

## 8. 증분 6 — 카메라 재중심: EMA 재수렴 + 소비 coarse 폴백 윈도

토로이달 대신 **EMA 재수렴**을 선택(1차 계획서의 후속 옵션 중): y-팩 레이아웃이 y 랩을 배제하고
x/z 랩만의 부분 토로이달은 복잡도 대비 이득이 불명, 재중심은 데드존으로 희소 이벤트가 되므로
재수렴+안정화 윈도(2 슈퍼사이클 ≈ 16프레임)의 비용이 작다. 단 box 중심을 **fine 복셀 격자
(0.45 m)에 스냅**해 두어 향후 토로이달 업그레이드가 히스토리를 재사용할 수 있게 한다.

**상태기계(호스트, main.rs/gi.rs)**:

```
        카메라가 box 중심에서 > half*0.5 이탈
Steady(B) ────────────────────────────────→ Reconverging(B′)     B′ = 카메라 중심, 복셀 스냅
   ↑                                             │                (전환은 슈퍼사이클 경계에서)
   │        advance 1회(업데이트 완주)              ↓
   └──── Settling(B′) ←──────────────────────────┘
         (advance 1회 더 — 소비 재활성)
Steady:       update push box=B,  fine_reset=0; 소비 gi_fine_buf=B
Reconverging: update push box=B′, fine_reset=1; 소비 gi_fine_buf=무효 박스(min>max → containment 항상 false)
Settling:     update push box=B′, fine_reset=0; 소비 gi_fine_buf=무효 박스 유지
```

- **소비 재활성은 두 번째 advance에서**(검증 확정 오프바이원): 소비자는 이번 슈퍼사이클에 기록
  중인 슬롯을 읽는다(gi.rs:335-341). Reconverging 사이클 N이 슬롯 A의 fine 반부를 B′로 다시
  쓰고 advance하면, 다음 사이클의 write 슬롯 = **슬롯 B — fine 반부가 여전히 구 박스 데이터**
  (사이클 N−1 기록)이고 2P프레임에 걸쳐 슬랩 단위로 재기록된다. 첫 advance에서 소비를 켜면 최대
  1 슈퍼사이클 동안 구-박스 텍셀을 신-박스 좌표로 샘플 — Settling 상태가 그 윈도를 막는다.
  업데이트측 fine_reset은 첫 사이클만 필요(Settling의 히스토리 read 슬롯은 이미 완전 B′).
- `fine_reset`(=`fine_max.w`, §1c): 셰이더에서 level==1 프로브의 **EMA 블렌드 양쪽 스킵 —
  radiance(gi_volume.slang:388-395)와 sky-vis(:396-406) 모두**(radiance만 스킵하면 B2 게이팅과
  디퍼드 스카이라이트가 소비하는 sky-vis 필드가 구-박스 데이터로 오염) + 히트점 읽기
  (`gv_level_uvw` 멀티바운스 :315·B2 sky-vis :335)의 fine containment 비활성. `read_rgb.w`
  (전역 reset)는 비트필드화하지 않고 그대로 둠(B2 게이트 `read_rgb.w==0` 비교 불변).
- 소비측 무효 박스: 재수렴+안정화 윈도(2 슈퍼사이클 ≈ 16프레임) 동안 소비자는 자연히 coarse
  폴백(fine은 근거리 디테일 레이어라 seamless — §7의 edge-fade가 전환 pop 추가 완화).
  gi_fine_buf는 2-버퍼 링(§1c)으로 in-flight 하자드 회피.
- 연속 재중심(윈도 중 재이탈): Reconverging의 목표 박스만 다음 경계에서 갱신(윈도 재시작).
  데드존(half*0.5)이 스래싱 차단.
- **정적 캡처 불변**: 고정 카메라는 초기 box 중심에 있으므로 데드존 이탈 없음 → 재중심 미발동 →
  결정론·게이트 경로 무변경. 이것이 이 증분의 1차 게이트.
- **모션 계측 인프라(이 증분에 포함 — 검증에서 부재 확인)**: 헤드리스 캡처는 카메라가 고정
  (level view 핀 main.rs:4651-4653, CAM_EYE 정적 오버라이드 :4707-4712, fly는 screenshot 모드
  비활성 :4679-4680)이라 재중심을 트리거할 수단이 없다. **`CAM_EYE_END`**(opt-in env): 설정 시
  CAPTURE_SEQ 윈도에 걸쳐 CAM_EYE→CAM_EYE_END 결정적 lerp — 미설정이면 코드 경로 무접촉(기존
  레시피 바이트 불변 확인 포함).
- **게이트**: 공식 러너 불변(정적 미발동 확인) + 풀 매니페스트(§10) + 모션 스팟체크(CAM_EYE_END
  경로로 데드존 이탈을 강제한 CAPTURE_SEQ — 재중심 프레임 전후 f2f diff에 스파이크/펌핑 없음,
  벽시계 평탄 유지) + 갤러리 앵커.

## 9. 증분 7 — (조건부) 레벨 확장/캐스케이드

착수 조건: 증분 3~6 랜딩 후에도 EV11 깊은 크롭이 PT 대비 유의 잔차를 유지하고, 그 잔차가
프로브 간격(coarse 1.36 m / fine 0.45 m 사이 갭 또는 fine box 밖 중거리)으로 소급될 때만.
접근: Y-팩 연장(높이 3× — 신규 bindless 슬롯 0 유지) 우선. WRC 옥타 아틀라스 인프라
(gi.rs:570-720 — 아틀라스 할당 :665, record_wrc_update :683)는 SH-L1 각도 표현 자체가 병목으로
실측될 때의 저장 후보로만(escaped-ray 부활 아님 — 비목표 유지).

## 10. 게이트·측정 프로토콜 (모든 증분 공통 — 어기면 측정 무효)

- **절대 게이트 판정은 공식 러너만**: `python tools/golden-image.py --only sponza_pt_sunlit
  --only sponza_pt_interior`(budget 28.58/32.87, 측정 28.28/32.57 — manifest.json:21-52 확인).
  `--update`는 budget = 신선 측정 +0.3 마진으로 전량 재기록(수기편집 금지, budget은 하향만).
  수동 rt-compare를 세션 초기 PT 캡처에 앵커 금지(AE 궤적 차이로 +1~+4 비관 — 실증). knob
  행간 비교는 수동 유효.
- **매 증분 풀 매니페스트 러너**(`python tools/golden-image.py` 무 `--only`): 매니페스트에는
  PT budget 쌍 외에 **SHA-정확 콘텐츠 골든 `sponza_sc_viz`/`sponza_gdf_ao`**(WARMUP=64·EV11,
  gi_volume 기본 ON 경로)가 있다 — PT budget의 0.3 톨러런스가 못 잡는 CLIP=0 드리프트의 유일한
  바이트 검출기. 증분 1·2(opt-in)는 바이트 불변 요구; 증분 3의 sc_viz 예상 변경은 §5에 사전
  선언.
- 분해 진단은 고정 EV100=11(AE 로그계측 — 단항 절대 비교 무효), 크롭
  `python3 tools/crop-luma.py img.png --x0 0 --x1 900`(interior 좌 주랑 밴드).
- 셔머: **`CAPTURE_SEQ=8 CAPTURE_SEQ_STEP=0`** interior 카메라, 인접쌍 `rt-compare.py` 무플래그
  avg의 평균(스텝 0은 방어적 명시 — CAM_EYE 캡처는 카메라가 어차피 고정이고 스텝은 갤러리/오빗
  캡처에서만 회전). 기준 0.0486 무회귀 — **판정은 반드시 동일 커맨드 A/B 쌍**(기준 재캡처)으로.
  트레이스 지터 재도입 금지(P_GI_STABLE 기본 ON=계약), 전그리드 버스트 금지 — 슬랩 평탄 비용
  유지. run-to-run spread ≤0.005 병기(결정론).
- GPU 비용은 **벽시계 델타**: 게이트 레시피(WARMUP_FRAMES=192) + `PROFILE_GPU=1` RELEASE로 돌려
  per-frame 로그의 `fence-wait` 필드 평균을 외부 집계(main.rs:5483-5486 라인). Metal per-pass
  타이머 불신(54 ms 오보 실증) — fence-wait/벽시계만.
- `rt-compare.py` 무플래그 stdout 라인(`avg abs diff / channel: …`)과 `RTCOMPARE_JSON` 라인은
  외부 정규식 소비자(verify-rhi-thread.ps1:48-52, golden-image.py:207-226)의 계약 — 변경 금지.
- 갤러리 앵커 `65d04ceca2c4dbff` 바이트 불변, run-to-run ≤0.005/ch, `cargo clippy -D warnings`/
  fmt, DX≡VK Windows 배치 항목 추가(동결 중), 상표명 금지, 계획서 커밋 포함, 증분별 검증된
  단일 커밋, 셰이더 변경 빌드마다 "Compiling dreamcoast-shader" 라인 확인, 백그라운드 명령 cd 명시.

## 11. 리스크

- **EMA 수렴 반감**(증분 2): 게이트가 결손을 보이면 α 스윕으로 보상(§4 레버) — 스윕도 셔머
  게이트 동반.
- **레벨 교대의 프레임간 비용 편차**: fine 슬랩(지오메트리 밀집 근거리)과 coarse 슬랩의 마치
  비용이 다를 수 있음 → 벽시계-AE 펌핑 감시(셔머 지표가 잡음; 편차가 크면 슬랩 두께 미세 조정).
- **재중심 프로토콜**(증분 6): 상태 전환을 슈퍼사이클 경계에 정렬해 mid-cycle 반쪽 갱신 배제;
  소비 재활성은 두 번째 advance(§8 — 첫 advance 재활성은 구-박스 데이터 오독); 정적 캡처
  미발동이 1차 게이트. fine_reset이 B2 게이트(`read_rgb.w==0`)와 독립 플래그임을 유지.
- **반사 fine 폴스루**(증분 4): cache-tone parity 재캘리브 상호작용 — 단독 커밋 + 전용 게이트.
- **기본 ON의 sunlit 영향**: fine box는 캡처별 CAM_EYE 중심이라 sunlit 게이트도 아트리움 중심의
  완전 활성 fine box로 돈다(§5) — sunlit 이동은 잡음이 아니라 신호로 취급, 회귀는 차단자.
  두 게이트의 fine box가 서로 다른 박스임을 인지하고 판정(둘 다 공식 러너 기준).
- 갤러리: GiSystem 생성자 갤러리 인자화로 단일 높이 유지 — 앵커 원천 차단.

## 12. 비목표·반복 금지 (기존 + 유지)

WRC escaped-ray 부활(main.rs 측정 판정) / gi_importance 볼륨 경로(inert) / 트레이스 지터
재도입 / 전그리드 버스트 / spacing-비례 bias(절대 0.05 대비 −0.64 열세) / gv_shadow 완화
(sh≡1 → 57 폭발) / P_CACHE_OCCL_ROUTE·P_REFLECT_SKYFILL=0·P_SPEC_OCCLUSION 레버(무효 실측) /
PT 레퍼런스 수정 / 매니페스트 수기편집 / 컴퓨트 그룹 크기 리터럴(COMPUTE_GROUP_SIZES 단일 소스) /
볼륨 update spp 상향으로 위장한 비용 증가.
