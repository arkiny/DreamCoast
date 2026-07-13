# F4 계획서 — 계층 월드 라디언스 캐시 (1차 증분: 카메라-추종 fine SH 볼륨 레벨)

> 상태: **승인·1차 증분 랜딩** (2026-07-13). 상위 [gi-fidelity-roadmap.md](gi-fidelity-roadmap.md) §F4,
> [gi-fidelity-phases.md](gi-fidelity-phases.md) F4. 승인 전 코드 변경 없음.

## 0. 왜

로드맵 F4: "원거리 GI가 코사인 게더라 노이즈·수렴 느림; **월드 캐시 1레벨**". F1(캐시)·F2(필드) 랜딩으로
착수 조건 충족. F6B 도구가 게이트 기준선을 제공: sunlit 30.9 / interior 34.65 (lit_mean 갭 — raster가
PT보다 lit 영역에서도 ~18/255 밝음 — 이 페이즈가 겨냥할 구조 신호).

## 1. 진단 (2026-07-13 코드 조사 — 반복 조사 금지)

**콘텐츠 기본 확산 GI = `gi_volume`(DDGI-lite) 단일 레벨 씬-고정 SH-L1 볼륨.**
- 32³ 프로브(`GI_VOL_DIM=32`, gi.rs:37) × 12 SH R32F × 핑퐁 + sky-vis SH 4종. **씬 AABB 전체에 고정**
  (sponza 36 m → 프로브 간격 ~1.1 m), 카메라 무관(gi_volume.slang:157-159).
- 업데이트: 프로브당 uniform-sphere `gi_spp`(Apple 4)레이, GDF 마치 48스텝, 히트에 sun+sky+이전볼륨
  (멀티바운스), EMA α=0.1, **period 4 amortized**(scalability.ron:191). 소비: `gdf_gi.slang:151-218`이
  SH에서 E(n)·sky-vis V(n)·bent normal 재구성 → denoise → upsample → PBR ambient.
- **볼륨 박스 밖 하드 seam(e=0)** — 클립맵식 fall-through 부재(gdf_gi.slang:155 `if inside`).

**중요 선행 판정(반복 금지):**
- 구세대 F4(중요도 게더 `gi_importance`/`bounce_importance_dir`)는 랜딩·측정 완료(Apple 0.5, firefly
  p99 233→75)이나 **march 경로 전용 — 콘텐츠 디폴트(볼륨 경로)에는 inert**(gi.rs:818-833 명시).
- **WRC(escaped-ray 폴백)는 측정상 무효~미미한 손해로 기본 OFF**(main.rs:3103-3108): 전장면 GDF
  클립맵이 오프스크린을 이미 커버. 이 역할의 부활은 비목표. 단 WRC 인프라(멀티레벨 아틀라스·핑퐁·
  옥타 타일, gi.rs:541-668)는 향후 계층 캐시의 저장 기반으로 유효.
- screen probe(`SCREEN_PROBE`)·surface-cache GI feedback(`P11_SURFACE_CACHE`, Apple/Med OFF —
  캐시 라디언스는 반사 전용) 모두 기본 OFF.

## 2. 1차 증분 — 카메라-추종 fine SH 볼륨 레벨 (계층화의 최소 검증 단위)

**가설(게이트로 검증):** 32³/36 m(1.1 m 간격)의 공간 blur·벽 관통 누설이 실내 GI를 과평탄·과밝게
만든다(→ interior lit_mean 갭 76.8/48.9의 일부). 카메라 주변 fine 레벨(같은 32³을 ~12 m 박스에 —
간격 0.375 m)로 근거리 디테일을 올리면 PT 잔차 개선/중립 + 시각 디테일 향상.

**설계 (조사에서 식별된 seam 그대로):**
1. **스토리지**(gi.rs:228-261): SH/sky-vis 볼륨 세트에 레벨 축 추가 — fine 레벨용 12+4 볼륨 × 핑퐁
   1세트 증설(32³ R32F ×32개 ≈ +4.2 MB). 할당 루프·연속성 debug_assert는 세트 반복이라 거의 그대로.
2. **업데이트**(record_gi_volume, gi.rs:329-439): fine AABB(카메라 중심 큐브, 반경 = 씬 최소축/6쯤 —
   시드 측정으로 확정) 인자로 **두 번째 디스패치**(8³). AABB는 호스트 인자(main.rs:6708-6709)라
   재중심화는 호스트 수정. 정적 캡처(고정 카메라)에서는 고정 박스 → 결정론 유지. period·EMA 동일.
   재중심 시 히스토리 무효화는 1차에선 "EMA 재수렴"으로 수용(정적 파리티 타깃), 토로이달은 후속.
3. **소비**(gdf_gi.slang vol 분기): `clipmap.slang:42-68`의 finest-first containment fall-through
   패턴을 SH 샘플에 포팅 — fine 박스 안이면 fine 세트, 아니면 coarse 세트(기존). 경계 하드 seam은
   1차 수용(클립맵과 동일 규약), edge-fade는 후속. `GiPush` 스페어(read_rgb.z/write_rgb.z, gi.rs:413/
   415)에 fine base 전달 + fine AABB 1행 추가.
4. **반사 소비자**(gdf_reflect.slang:257-337 동일 SH 읽기)는 **1차에선 coarse 유지**(GI만 fine) —
   측정 후 후속 커밋으로 포팅(회귀 격리).
5. **Seam**: `P_GI_VOL_CLIP=1` opt-in(티어 기본 OFF, 측정 후 Apple 편입 결정). OFF = 셰이더/호스트
   모두 현행 경로 바이트 동일(갤러리 앵커 + 콘텐츠 무회귀).

## 3. 게이트 (프로젝트 불변)
- 갤러리 바이트 동일(OFF 경로; ON은 콘텐츠 전용 opt-in).
- **콘텐츠 PT budget**: sunlit ≤30.9 / interior ≤34.65 — **개선 기대, 최소 중립**. 개선 시 하향
  재기준선(F6B 규약).
- **정적 셔머 무회귀**: `CAPTURE_SEQ` 시퀀스 diff (fine 레벨 EMA가 셔머를 늘리면 안 됨).
- 결정론(고정 카메라 fine AABB·시드), `PROFILE_GPU` 비용 보고(update 2×지만 period 4 — ~수십 µs/평균
  프레임 예상, 실측), clippy/fmt, DX≡VK 배치 추가, 상표명 금지.

## 4. 하지 말 것
- WRC escaped-ray 폴백 부활(측정 무효 — main.rs:3103 판정 존중). 볼륨 update spp 상향으로 위장한
  비용 증가(성능트랙 원칙). 측정 없는 티어 기본 ON.

## 5. 1차 증분 검증 결과 (랜딩 커밋)
- 갤러리 앵커 바이트 동일 PASS(OFF 경로 무손상). 기본(OFF) PT budget PASS: sunlit 30.628 / interior 34.153.
- **ON(`P_GI_VOL_CLIP=1`) 가설 측정**: interior masked_avg **34.071 (OFF 대비 −0.08 개선)**, sunlit
  30.631(중립, +0.003). fine box half 7.25 m, 프로브 간격 1.1→**0.45 m**. 개선 방향은 가설대로이나
  크기가 작음(정직) — 티어 기본 편입은 보류, opt-in 유지. 후속(반사 fine 폴스루·edge-fade·재중심)
  측정 후 재평가.
- 정적 셔머 무회귀: interior 4프레임 seq diff OFF 0.042 → ON 0.040/ch(노이즈 플로어 동일).
- sky-vis 4번째 소비자(`sdf_cache_light.slang`)도 동일 coarse-리맵 적용(구현 중 발견 — ON에서
  스카이라이트 오샘플 방지). EMA 자기읽기는 정확 텍셀 센터 유지(공간 확산 방지).
- clippy/fmt clean. DX≡VK Windows 배치에 `P_GI_VOL_CLIP` 콘텐츠 캡처 추가.

## (원) 검증 계획
OFF 바이트 동일 → ON: pt budget 2종(+시각 interior 디테일 비교) → CAPTURE_SEQ 셔머 → PROFILE_GPU →
단일 커밋. 결과에 따라 후속: 반사 소비자 포팅 / edge-fade / 토로이달 재중심 / Apple 티어 편입.
