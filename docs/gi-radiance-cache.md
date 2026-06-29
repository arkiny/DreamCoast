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
- 차후: directional(SH/octahedral) probe, 가시성(누설 방지), 클립맵(대형 월드), 상각 update, ImGui 토글, RenderQuality 티어 편입.

## 하지 말 것
- 갤러리 앵커 깨기(볼륨 콘텐츠 한정). DX≡VK 발산(ping-pong로 race 제거). 측정 없이 단정.
