# GI 월드 irradiance 볼륨 (DDGI-lite radiance cache) — 권위 계획

상위: [lumen-parity-swrt.md](lumen-parity-swrt.md). UE Lumen GI fidelity 조사(메모리 unreal-engine-reference)
결과, 우리 1-bounce GDF GI는 깊은 실내(Sponza nave)에 **GI가 도달하지 못해** 어둡다(멀티바운스 부재). UE는
**월드 radiance cache + radiosity 멀티바운스**로 채운다. 이 트랙은 그 핵심을 **월드 공간 irradiance 볼륨
(DDGI-lite)** 으로 도입 — 밝은 atrium의 빛이 프레임 누적으로 nave까지 전파(멀티바운스)되어 실내를 채운다.

## 설계 (v1)
- **볼륨**: GDF AABB 위 RGB irradiance 3D 텍스처. 기존 albedo 패턴 재사용 = **3개 R32F 볼륨**, ping-pong(read/
  write) = 6 볼륨. 해상도 32³(저주파 diffuse). RHI 변경 없음.
- **update 컴퓨트(`gi_volume.slang`)**: voxel당 1스레드 → world pos. spp개 uniform-sphere 레이를 GDF(clipmap)에
  쏴 hit 셰이딩 = `albedo(hit)·(sun·ndl·shadow + sky_fill + E_read(hit))` — **E_read(hit)=직전 볼륨 샘플=멀티
  바운스**. miss=sky. 평균=E_probe, read 볼륨과 EMA → write 볼륨. (DDGI 핵심: probe가 볼륨 자기참조 → atrium→nave 전파.)
- **라이팅(`pbr.slang`)**: world_pos에서 볼륨 trilinear 샘플 → E. indirect = `albedo·(1-metallic)·E`. 볼륨이
  바인드되면 gdf_gi 대신 사용(콘텐츠), 아니면 기존 gdf_gi. v1=normal-독립 평균(flat fill, 차후 directional/SH).
- **게이트**: 콘텐츠 켜짐, **갤러리 강제 off(앵커)**. `P_GI_VOLUME`. DX≡VK(결정적, ping-pong race-free)·검증.

## 스테이지
- v1 ✅ (2026-06-29): 볼륨(`gi_volume.slang`, 32³ R32F×3 ping-pong)+update+라이팅 통합. `P_GI_VOLUME=1`
  콘텐츠 전용, 갤러리 강제 off(`!gallery_scene`). gdf_gi에 볼륨-샘플 분기(`vol_r!=MAX`)·GI_DENOISE_WARMUP
  64프레임 워밍업.
  - **측정(Sponza nave, CAM_EYE 실내)**: base→volume **14.5/ch(90.6%)** 밝아짐 = 깊은 실내 채워짐(목표 달성).
  - **DX≡VK 0.015 byte ≈ 0.00006/ch** (결정적 ping-pong, ≤0.001 게이트 통과). 갤러리 앵커 0.001 byte(무회귀, 구조상 코드 미접촉).
  - **비용(d3d12 PROFILE_GPU)**: gi_volume update **0.131ms**(32³ probe×spp), gdf_gi는 per-pixel 마치→볼륨 샘플로
    **0.0066ms**로 하락 → DDGI는 fidelity↑ + 픽셀당 마치 제거로 사실상 ~0.13ms 순증(저주파 update 상각). Vulkan 검증 클린.
- **v2 ✅ (2026-06-30): directional SH-L1 probe** — v1의 probe당 무방향 스칼라 RGB 1개를 **밴드0+1 구면
  조화(채널당 4계수, RGB=12 R32F 볼륨/슬롯, ping-pong=24)**로 교체. 색 커튼이 인접 면에 GI 색을 못 번지게
  하던 근본 원인(바닥 한 점이 probe의 **전 구면 평균**을 읽어 옆 커튼 입체각이 평균에서 ~0으로 희석)을 해소:
  리시버가 자기 법선 방향으로 **코사인 가중 반구 평균 radiance E(n)** 재구성(A0/π=1, A1/π=2/3, clamp≥0) →
  등방 필드에선 v1 스칼라 평균과 **정확히 일치**(엄밀한 일반화).
  - **update(`gi_volume.slang`)**: 각 uniform-sphere 샘플의 incoming radiance를 SH 기저로 투영
    (`coeff = 4π/N·Σ L·Y`, Y0=√(1/4π), Y1=√(3/4π)·{y,z,x}), 계수별 EMA. 멀티바운스·EMA read는 hit 법선
    방향 방향성 재구성. **miss=0**(이전 sky 누적 제거: per-pixel 마치와 동일한 단일소스 sky=IBL, 이중계상 방지).
  - **소비(`gdf_gi.slang` vol 분기)**: 12계수를 픽셀 법선으로 평가해 E 출력(이후 기존 denoise/lighting 불변).
  - **★ 푸시 레이아웃 0 변경(DX≡VK 안전)**: `create_volume`이 sampled·storage 바인드리스 인덱스를 연속 할당 →
    12볼륨을 연속 배치하고 `base + channel*4 + coeff`로 주소화, **베이스 인덱스 1개만** 전달. `gi_volume_push`(160B)
    ·`gdf_gi_push`(240B, 이미 Vulkan 256B 한계)가 바이트 레이아웃 불변. 볼륨 총 6→24(전체 ~42 ≤ 64 슬롯, `gi.rs`에
    연속성 `debug_assert`). 콘텐츠 한정·**갤러리 바이트 동일**(SHA 불변)·Metal 결정론적.
  - **측정(sponza_intel, Metal, EV100=11)**: 방향성 GI ~1.9× 밝아짐(DEBUG_VIEW=10 평균 0.30→0.58). 우측 빨강
    커튼 base 바닥의 **GI항 적색초과 R−B +2.66→+4.26**, 컴포짓 바닥 R−B −40.2→−38.4(+1.8 더 빨강); 중립
    나브중앙 바닥(GI≈0)은 바이트 동일 → 전역 틴트 아닌 **순수 방향성** 변화 입증. (after−before) 차분: 커튼을
    **마주보는 수직 벽**에 균일 warm(dR−dB +4.25 peak) 추가 — 평평 바닥은 거의 불변.
  - **물리적 한계(측정으로 확정, 버그 아님)**: 평평 바닥의 반구는 청색조 석재가 지배 + 빨강 커튼은 작은 입체각
    & up 법선에 낮은 코사인 가중 → 바닥 간접광은 **정당하게 약간만 warm**(+4.3). 장거리 AO로 sky 차폐를 키워도
    (`AO_REACH=4`) 커튼 근처가 더 많이 어두워져 bleed 메트릭 +31.8→+22.4로 **하락** → sky 차폐는 바닥 해법 아님.
    강한 색 번짐은 **커튼을 마주보는 면**에 정상 발현. 바닥의 극적 적화는 비물리적(보정 거부, Rule 1).
  - **DX≡VK 보류**: Windows 동결 → Metal 구현·검증만(셰이더는 3백엔드 컴파일 성공). 미커밋 작업트리.
- **v3 ✅ (2026-07-01): 실내 스카이라이트 차폐(skylight occlusion) — 바닥 파랑의 진짜 원인 해소.**
  깨끗한 항-분리 측정(임시 DEBUG_VIEW)으로 **바닥 파랑 = 100% IBL diffuse 스카이라이트**(irradiance 큐브,
  중앙 바닥 컴포짓=diffuse, 정반사≈0, GI≈0)임을 확정 — 이전 "정반사" 진단은 대수 오류였음. 우리는 diffuse
  스카이라이트를 **씬-스케일로 전혀 차폐 안 함**(접촉 `gdf_ao` 0.5m뿐) → 파란 하늘광이 무차폐로 쏟아져
  약한 컬러 바운스를 덮음. **레퍼런스 정렬 수정**: probe가 sky-vis(miss=1/hit=0)를 **SH-L1 스칼라**(4볼륨/슬롯,
  ping-pong=8)로 투영(`gi_volume.slang`); `gdf_gi.slang`가 픽셀 법선으로 **V(n)** 재구성→별도 이미지(빈
  `vol_g/vol_b` 재활용 = gdf_gi 푸시 불변); `pbr.slang`가 IBL **diffuse만** 차폐:
  `irr_occ = irradiance·lerp(V,1,min_occ) + tint·Luminance(irradiance)·(1-vis)` — 차폐분을 **중립 틴트
  누설**(레퍼런스 OcclusionTint/MinOcclusion 형)로 채워 검정 회귀 방지. 콘텐츠 한정(`gi_volume` ON일 때만
  sky-vis 바인드; 갤러리=센티넬 no-op). pbr 푸시만 48→60(센티넬 게이트).
  - **노브**: `P_SKYVIS_TINT`(기본 0.5, 차폐분의 중립 휘도 비율), `P_SKYVIS_MIN_OCC`(기본 0.0, 가시성 바닥;
    **=1.0이면 완전 off**=SH-L1 베이스라인 바이트 동일).
  - **측정(sponza_intel, Metal, EV100=11)**: 바닥 디블루 — 우측 빨강 커튼 바닥 R−B **−38→+7.9(warm!)**,
    중앙 바닥 R−B **−70→+5.9(중립)**; 바닥 영역 평균 dRGB **[−6,−35,−59]**(파랑 제거, 빨강 보존). 석재가
    중립이 되어 컬러 커튼이 도드라짐(UE 실내 룩). **갤러리 바이트 동일**(SHA 불변)·**결정론적**·off-스위치가
    베이스라인 바이트 동일. **DX≡VK 보류**(Windows 동결; pbr 푸시 후행 스칼라 = 레이아웃 안전).
- 차후: 차폐분을 평면 틴트 대신 radiance-cache 바운스로 채우기(완전 Lumen형), 클립맵(대형 월드), 상각 update,
  ImGui 토글, RenderQuality 티어 편입.

## 하지 말 것
- 갤러리 앵커 깨기(볼륨 콘텐츠 한정). DX≡VK 발산(ping-pong로 race 제거). 측정 없이 단정.
