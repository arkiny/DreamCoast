# 핸드오프: AO + 스카이라이팅 개선 (레퍼런스 엔진 기반)

DreamCoast 엔진(Rust/raw-RHI, macOS=Metal, repo: /Users/arkiny/GitRepos/DreamCoast)에서 **앰비언트
오클루전(AO)과 스카이라이팅**을 참고용 언리얼(Lumen/GTAO/DFAO) 기반으로 개선한다. 참고 소스는
/Users/arkiny/GitRepos/UnrealEngine-1 에 있다. **트레이드마크명 금지 — 문서/주석/커밋엔 "reference
engine"으로만 표기한다.**

## 목표 (한 줄)
AO와 스카이라이트를 **스칼라 곱**에서 벗어나 레퍼런스 엔진처럼 **방향성(bent-normal) 기반**으로 통합한다:
AO 패스가 **가려지지 않은 평균 방향(bent normal)**을 산출하고, 스카이라이트(IBL diffuse)를 그 bent
normal을 따라 샘플 + AO 스칼라로 감쇠 → 실내에서 물리적으로 정확한 방향성 스카이라이트 오클루전. 추가로
GTAO의 **멀티바운스 항**과 **스페큘러 오클루전**을 도입한다.

## 착수 전 필수
- **git**: `main`(현재 `5a4e1eb`, 반사 트랙 머지 완료)에서 새 브랜치 `feature/bent-normal-ao-skylight`로
  분기. `main == origin/main`이면 정상. **origin/main이 disjoint로 보이면 `git pull` 금지**(과거 강제
  재작성 이력 — 메모리 참조). 새 브랜치에서 작업.
- **메모리 읽기 (필독)**: `dreamcoast-permesh-df-plan.md`(SH-L1 sky-vis + indoor skylight occlusion 배경 —
  핵심), `dreamcoast-gdf-ao-flicker.md`(GDF AO 깊이 load/store 픽스), `dreamcoast-gi-fidelity-and-macos-perf.md`,
  `dreamcoast-reflection-gi-fix.md`(반사 트랙 — bent normal은 스페큘러 오클루전과도 연결), `dreamcoast-no-trademark-names.md`.
- **게이트**: gallery 골든 `af70c1a5`가 항상 byte-identical
  (`python3 tools/golden-image.py --only gallery --backend metal`). 경로-트레이서 패리티
  (`P8_PATHTRACE=1` 캡처 vs 라스터, `tools/rt-compare.py`)가 라이팅 변경의 정식 성공 지표. DX≡VK Windows 후속.

## 방법론 (이번 반사 세션에서 검증된 방식 — 그대로 따를 것)
**추측하지 말고 UE 소스를 정밀 추출한다.** UE 트리가 방대하므로 **Explore 에이전트**로 정확한 파일/라인/공식을
verbatim으로 뽑아온 뒤 근거 기반으로 구현한다. 이번 반사 세션에서 3개 에이전트로 "거울은 color mip이 아니라
virtual-image reproject로 깨끗해진다"는 결정적 사실을 확인해 헛수고를 막았다. AO/스카이라이트도 동일하게:
에이전트에게 GTAO의 정확한 visibility 적분 + 멀티바운스 다항식, DFAO의 bent-normal 산출식, 스카이라이트가
bent normal로 샘플되는 지점을 file:line + 코드로 뽑아오게 한다.

## 지금까지 구현된 것 (현재 상태 — 개선의 출발점)
- **AO 두 계층 (레퍼런스식 GTAO × DFAO 레이어링)**:
  - `crates/shader/shaders/gdf_ao.slang` — **원거리 distance-field AO**(DFAO 상당), GDF 콘 트레이스.
    `apps/sandbox/src/gi.rs::record_ao`. env `P11_GDF_AO`, 해상도 `P_AO_RES_DIV`(Apple 타일=quarter-res +
    joint-bilateral 업샘플). 깊이 flicker는 과거에 픽스됨(메모리 참조).
  - `crates/shader/shaders/gtao.slang` — **근거리 스크린 AO(HBAO-lite obscurance)**, env `SSAO`
    (+`SSAO_RADIUS/INTENSITY/BIAS/POWER`), 정수-해시 회전으로 DX≡VK 결정적, separable depth-aware blur.
  - 둘은 라이팅에서 `near × far`로 합성(`pbr.slang`), AO는 **DIFFUSE 앰비언트에만** 곱함(스페큘러 반사는
    자체 오클루전을 이미 반영 — 반사 트랙 Fix 4).
- **스카이라이트 (레퍼런스식 indoor skylight occlusion — 이미 존재)**:
  - `pbr.slang::occlude_sky_diffuse(irradiance, sky_vis)`: 방향성 sky-visibility `V∈[0,1]`로 diffuse
    스카이라이트(irradiance cube)를 감쇠. neutral **OcclusionTint**(`skyvis_tint`/`P_SKYVIS_TINT`) +
    **MinOcclusion** 바닥(`skyvis_min_occ`/`P_SKYVIS_MIN_OCC`). `sky_vis=1`이면 정확한 no-op(gallery 앵커).
  - `sky_vis`는 **SH-L1 방향성 sky-visibility**에서 온다(`gi_volume.slang`, `skyvis_index` 이미지). 콘텐츠
    전용(gi_volume on), gallery는 no-op으로 byte-identical.
- **한계 (개선 지점)**: 현재 스카이라이트 오클루전은 **스칼라 V**(방향당 하나) — bent normal을 안 쓴다. 즉
  IBL diffuse를 표면 노멀 `n`에서 샘플하고 스칼라로 곱할 뿐, **가려지지 않은 방향으로 편향된 스카이라이트**를
  못 준다. GTAO는 obscurance만(멀티바운스/스페큘러 오클루전 없음). 반사 스페큘러는 AO를 안 받음(정확하지만
  bent-normal 기반 스페큘러 오클루전은 없음).

## 이번 작업 = bent-normal 기반 AO ⟷ 스카이라이트 통합

### 참조 (레퍼런스 엔진 소스, /Users/arkiny/GitRepos/UnrealEngine-1 — 에이전트로 정확히 추출)
아래는 시작점 힌트일 뿐, **정확한 file:line/공식은 에이전트가 verbatim으로** 가져올 것:
- **GTAO**: `Engine/Shaders/Private/PostProcessAmbientOcclusion.usf` + `Engine/Shaders/Private/GTAO/*` —
  진짜 horizon-based **visibility 적분**(코사인 가중), **멀티바운스 다항식**(albedo로 AO를 되살리는
  `GTAOMultiBounce`: `a·AO³ + b·AO² + c·AO` 형태), **bent normal** 산출.
- **DFAO(bent normal + sky occlusion)**: `Engine/Shaders/Private/DistanceFieldAmbientOcclusion*.usf`,
  `DistanceFieldLightingShared.ush` — 콘 트레이스로 **bent normal + occlusion**을 산출해 스카이라이트에 사용.
- **SkyLight가 bent normal로 샘플되는 지점**: 디퍼드 라이팅/리플렉션 환경에서 diffuse 스카이라이트를 표면
  노멀이 아니라 **bent normal**을 따라 샘플하고 AO로 감쇠하는 코드(핵심 — 이 결합이 목표).
- **스페큘러 오클루전**: bent normal + roughness로 GGX 스페큘러를 horizon 오클루전
  (`GetSpecularOcclusion`류) — 반사에도 적용.
- **Lumen AO**(선택): 별도 AO 대신 스크린-프로브 GI에서 AO를 유도(`LumenScreenProbeGather`) — 우리는
  `screen_probe_*`가 이미 있으니 정합 검토.

### 구현 스케치 (권장 순서, 각 단계 gallery byte-identical + 커밋)
1. **AO 패스가 bent normal 출력**: `gdf_ao.slang`(콘 트레이스)과/또는 `gtao.slang`이 스칼라 AO에 더해
   **가려지지 않은 평균 방향(bent normal, world-space)**을 산출해 별도 채널/이미지로 출력. (DFAO는 콘들의
   가중 평균 방향, GTAO는 horizon 적분의 bent 방향.) 업샘플(`gdf_gi_upsample.slang`)이 bent normal도
   운반하도록(반사 트랙에서 alpha 운반한 패턴 참고).
2. **방향성 스카이라이트 오클루전**: `pbr.slang`에서 diffuse 스카이라이트를 **표면 노멀이 아니라 bent
   normal**을 따라 irradiance cube에서 샘플하고 AO 스칼라로 감쇠 — 현재 `occlude_sky_diffuse`의 스칼라
   V를 대체/보강. content 전용 seam, gallery는 표면-노멀·no-op 유지 → byte-identical.
3. **GTAO 멀티바운스 + 진짜 visibility 적분**: `gtao.slang`을 HBAO-lite obscurance → 레퍼런스 GTAO의
   코사인-가중 visibility + 멀티바운스 다항식(albedo 항)으로 업그레이드. DX≡VK 결정성(정수-해시 회전) 유지.
4. **스페큘러 오클루전**: bent normal + roughness로 반사/프리필터 스페큘러에 horizon 오클루전 적용
   (반사 트랙의 "AO는 스페큘러 안 곱함" 규칙과 상충하지 않게 — bent-normal 기반은 물리적으로 다른 항).
5. **(선택) SH-L1 sky-vis와의 관계 정리**: bent normal이 SH-L1 sky-vis를 대체할지/보완할지 판단
   (bent normal = 더 국소적·정확; SH volume = 저주파·캐시). 이중 계산 회피.

### 게이트 / 검증
- **gallery `af70c1a5` byte-identical 필수** — 모든 신규 경로는 content-only(gi_volume/AO on) seam,
  gallery는 스칼라·표면-노멀·no-op 레거시 유지.
- **경로-트레이서 패리티**가 핵심 지표: AO+스카이라이트는 diffuse 앰비언트의 방향성이라 PT와 비교해야
  물리적 정확도를 안다. `sponza_intel`/`sponza_intel_chromeball`에서 실내 그늘/크레비스의 스카이라이트
  방향성이 PT에 가까워지는지.
- 실내 씬(커튼/기둥 그늘)에서 **평평함→입체감** 개선을 육안 + PT 잔차로 확인. over-occlusion(너무 어두워짐)
  주의 — MinOcclusion/tint로 조절.
- **성능**: bent normal은 AO 패스에 채널 추가(≈저비용). `PROFILE_GPU=1`로 측정. Apple 타일 quarter-res AO
  유지.
- **DX≡VK Windows 파리티**: bent normal(방향)은 결정적 재구성 필요 — GTAO 정수-해시 회전 패턴 준수.

## 규칙 (CLAUDE.md)
근본원인 수정·opt-in seam·기본 byte-identical·3백엔드 파리티(Metal 검증 후 DX≡VK Windows 후속)·최적화 가중치·
단일 진실원(공유 상수/노멀 재구성 일치)·상용 트레이드마크명 금지("reference engine"). 커밋 끝에
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## 이 트랙의 남은 후속 (개선 이후)
- DX≡VK Windows 파리티(이번 라인 Metal 검증만).
- Lumen 방식 통합 AO(스크린-프로브에서 AO 유도)로 별도 GTAO/DFAO 축소 검토.
- 반사 트랙 잔여: `fetch_refl_history` reject 거리 튜닝, 거울 history 짧게(UE는 2프레임+TSR).
