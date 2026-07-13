# F1 시작 프롬프트 — 표면 캐시 가상화 (Surface Cache Virtualization)

새 Claude Code 세션(맥, Metal)에 아래를 그대로 붙여넣어 F1을 시작한다. F1은
[gi-fidelity-roadmap.md](gi-fidelity-roadmap.md)의 **최우선(관측된 최대 갭)** 페이즈다.

---

## 붙여넣을 프롬프트

DreamCoast 엔진(`/Users/arkiny/GitRepos/DreamCoast`, Rust, RHI over Vulkan/D3D12/Metal)에서
**GI 충실도 로드맵 F1 — 표면 캐시 가상화**를 시작한다. 먼저 `git fetch origin && git checkout main
&& git pull`로 최신 main을 받고(브랜치 `feature/pt-auto-exposure`가 아직 미머지면 그 것도 포함해 검토),
새 `feature/*` 브랜치를 파라.

### 왜 (측정된 갭)
`LEVEL=sponza_intel_chromeball` 실행 시 로그: **449 drawables 중 카드 예산(`MAX_CARDS=1024`) 초과로
170만 카드 유지, 279는 coarse 폴백**(`apps/sandbox/src/fuse.rs`). 큰 씬일수록 악화 — 지오가 표면 캐시에서
드롭돼 라이팅 구멍이 생긴다. 로드맵이 F1을 "최우선"으로 못박은 이유.

### 목표
하드 드롭 제거 — 모든 (가시) 지오가 캐시에 참여. 접근(로드맵 F1):
- **(a) 데맨드 레지던시** — 카드 요청 피드백(카메라 프러스텀 + GI/반사가 실제 샘플한 draw)으로 필요한 카드만 상주.
- **(b) 페이지 아틀라스 + LRU 방출** — 고정 슬롯 대신 페이지 풀; 예산 초과 시 최근 미사용 카드 방출·재사용(스트리밍).
- **(c) 카드 mip** — 원거리 draw는 저해상 카드로(비용↓, cone 게더와 정합).
`fuse.rs`의 "demand-driven LRU page streaming = next increment" 주석이 착수점.

### 프로세스 (프로젝트 규칙 — 반드시 준수)
1. **먼저 계획서** `docs/phase-f1-*.md` 작성 → 사용자 승인 → 그 다음 구현. 승인 전 코드 변경 금지.
2. 착수 전 관련 코드를 읽어 계획을 코드 근거로 정초: `apps/sandbox/src/fuse.rs`(카드 할당/`MAX_CARDS`/레지던시),
   `apps/sandbox/src/gdf.rs`(표면 캐시 그리드/relight 배선), `crates/shader/shaders/sdf_cache_*.slang`(캡처/relight/게더),
   `crates/asset/src/sdf_atlas.rs`(아틀라스 팩). 소비자: `gi.rs`/`reflect.rs`.
3. 검증된 단일 커밋으로 랜딩. 부분 단계로 쪼개도 각 단계는 게이트 통과.

### 불변 게이트 (전부 통과해야 함)
- **PT 잔차 개선/중립** — 이게 성공 척도(ground truth). **좋은 소식: 방금 F6 슬라이스로 PT가 실내에서도
  자동노출되도록 고쳐(`AUTO_EXPOSURE` 기본 ON, `docs/phase-f6-pt-reference-usability.md`), 콘텐츠 실내
  잔차를 측정 가능**하다. `--screenshot-clean` raster vs `P8_PATHTRACE=1` 캡처를 `tools/rt-compare.py`로 대조.
  (주의: 커튼 뒤 깊은 차폐면은 PT에서도 near-zero radiance = 물리적 한계, F1 대상 아님.)
- **갤러리 바이트 동일** — 앵커 SHA `65d04ceca2c4…`. `python tools/golden-image.py --only gallery`로 확인.
  가상화는 콘텐츠 전용 seam으로, 갤러리는 불변.
- **결정론** — run-to-run 바이트 동일(페이지 방출 시드 고정). 콘텐츠 config는 strict SHA가 아니라 tolerant.
- **DX≡VK ≤0.001** — Windows RTX2070S에서 별도 검증(현재 동결·Metal 우선). Windows 배치에 추가.
- **단일 소스 / heavy=opt-in·티어 / 상표명 금지** — 로드맵 §5 "하지 말 것" 준수.

### 리스크
재조명이 GI/반사 소비자 공유 → 회귀 검증 필수. 페이지 방출의 결정성(시드 고정)이 핵심.

### 참고 문서
`docs/gi-fidelity-roadmap.md`(§F1, 의존 순서 F1→F2→F4), `docs/reflection-gi-quality.md`(카드 예산 스케일링),
`docs/per-mesh-sdf-direct-sample-plan.md`(방금 깐 per-mesh 아틀라스/셀그리드 인프라 — F1이 재사용).

먼저 위 코드를 읽고 **F1 계획서 초안**을 제시하라(데맨드 레지던시·페이지 LRU·카드 mip를 어떤 단계로
쪼갤지, 각 단계의 게이트/측정 방법 포함). 승인 후 구현한다.

---

## 참고: 직전 세션 상태 (2026-07-13)
- `origin/main` = `4e3fe7c` (aniso 기본16 + PT aniso/데칼 + UI 라스터↔PT 토글, 머지·푸시 완료).
- 브랜치 `feature/pt-auto-exposure` (`73f96d9`): PT/SW-RT tonemap 자동노출(F6 슬라이스), 미머지. F1 시작 전 이
  브랜치 머지/푸시 여부 확인.
- **미결**: DX≡VK Windows 재검증(`docs/windows-verify-anisotropy-default.md` — aniso/PT 경로 driver-dependent 발산).
- **PT 함정**: `LEVEL=sponza_intel_chromeball`에서 커튼-차폐 실내 방향은 PT near-black(물리적 GI reach 한계).
  하늘·햇빛·조명 실내가 보이는 각도에서 잔차 측정할 것.
