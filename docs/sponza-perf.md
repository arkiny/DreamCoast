# Sponza 60fps — 성능 분석 & 컬링 트랙 (권위 계획)

상위: [scalable-gi.md](scalable-gi.md)(GDF GI를 Sponza에 일반화 — 완료). 이 트랙은 그 결과로
**GDF GI가 디폴트인 Sponza가 60fps 미만**인 문제를 푼다. 목표: RTX 2070 SUPER, 데모 앵글, **양 백엔드
≥60fps(≤16.6ms/frame)**, 갤러리 무회귀 + GDF GI 품질 수용 가능 유지.

## ★ 1원칙: 측정 먼저 (추측 금지)

성능은 측정으로 시작한다. `PROFILE_GPU=1`(스크린샷 모드에서 패스별 GPU ms를 로그로 덤프)로 Sponza의
**패스별 비용을 먼저 분해**한 뒤, **가장 큰 비용부터** 공략한다. 측정 전에 "컬링이 답"이라고 가정하지
않는다.

### 두 개의 비용 전선 (가설 — Stage 0에서 검증)
| 전선 | 패스 | 지오메트리 의존? | 컬링으로 줄어드나? |
|---|---|---|---|
| **(A) 지오메트리 제출** | G-buffer fill, shadow map (현재 262k tri **전량** 매 프레임, 컬링 0) | O | **O** (프러스텀/오클루전/커버리지) |
| **(B) GDF SW-RT 스크린/캐시** | gdf_gi(풀스크린×spp×march), gdf_reflect, **surface cache relight(~1024카드×32²≈1M 텍셀/프레임)**, GI 디노이저, 반사 temporal | X (화면/캐시 바운드) | **X** (컬링 무관) |

**정직한 프레이밍**: 사용자가 제안한 프러스텀/오클루전/커버리지 컬링은 **(A)만** 줄인다. **(B)는 컬링으로
안 줄어든다** — 화면/캐시 바운드이기 때문. GDF GI 디폴트에서 (B)가 지배적일 가능성이 높으므로(풀스크린
레이마칭 + 매 프레임 100만 카드 텍셀 재조명), Stage 0 측정이 노력 배분을 결정한다. **컬링 단독으로
60fps가 안 될 수 있음을 전제**한다.

## 스테이지

각 Stage 독립 커밋. 게이트: `PROFILE_GPU` before/after(핵심 지표) + `cargo fmt --all` +
`RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets` + 양 백엔드 스크린샷 → **DX≡VK ≤0.001**
+ Vulkan 검증 클린. **보수적 컬링(A/B/C)은 갤러리 + Sponza 바이트 동일**(보이는 픽셀 불변); **품질 영향
(D)은 RenderQuality 티어 + `tools/rt-compare.py` 잔차 재측정·수용**.

### Stage 0 — 측정 & 귀속 (코드 없음, 측정+계획 정련)
- `PROFILE_GPU=1`로 Sponza 데모 앵글 + 2~3개 앵글, 양 백엔드 패스별 ms + 프레임 총합.
- **CPU vs GPU 바운드** 판정(프레임 총합 vs 패스 합; CPU 제출/드라이버 비용 확인).
- 패스를 (A 지오메트리)/(B GDF-SWRT)/(기타: tonemap, shadow, ibl)로 묶어 **Top-3 비용** 식별.
- 산출물: 본 문서에 측정 표 + 확정 스테이지 순서. **이 표가 이후 모든 우선순위를 지배.**

### Stage A — 프러스텀 컬링 (실제 씬 드로우)
- GPU 컴퓨트 프러스텀 컬: draw-list **per-drawable AABB**(Scalable-GI Stage 0가 이미 CPU 지오/AABB
  제공 — 재사용)를 카메라 프러스텀과 테스트 → **visible 인스턴스 리스트 + indirect draw**로 G-buffer fill.
  shadow는 **광원 프러스텀**으로 별도 컬. `cull.rs`(현재 데모 큐브 그리드)의 reset/cull/indirect 패턴
  일반화.
- 보수적(보이는 오브젝트 절대 누락 X) → **렌더 바이트 동일**. 게이트: 갤러리·Sponza 바이트 동일 +
  제출 삼각형 수/gbuffer·shadow ms 감소 측정.

### Stage B — 오클루전 컬링 (Hi-Z 2-phase)
- 전 프레임 depth로 **Hi-Z 피라미드** 빌드 → drawable AABB를 Hi-Z에 테스트. **2-phase**(① 전 프레임
  visible 드로우 → Hi-Z 재빌드 → ② 디오클루전된 것 추가 드로우)로 팝핑 없이 보수적 유지. 밀집 Sponza
  (기둥·아치 상호 가림)의 **오버드로 절감**. 게이트: 바이트 동일 + 오버드로/gbuffer ms 측정.

### Stage C — 커버리지 버퍼(소프트웨어 오클루전) — 선택(Hi-Z 부족 시)
- 큰 오클루더(벽/바닥)를 저해상 **커버리지/깊이 버퍼**에 소프트 래스터(컴퓨트) → AABB 테스트.
  Frostbite/Intel masked-occlusion 스타일. Hi-Z의 지연/엣지 한계를 보완하는 대안. Stage B 측정 후
  필요할 때만.

### Stage D — GDF SW-RT 비용 절감 (B 전선; GDF 라이팅의 핵심 레버)
- **D1 하프해상 GI+반사**: 트레이스를 1/2 해상(쿼터픽셀)으로 → 기존 à-trous/temporal 디노이저로 업스케일
  (Lumen 스크린 프로브식). ~4× 레이 감소. RenderQuality 티어 게이트.
- **D2 surface cache 갱신 예산**: 매 프레임 100만 카드 텍셀 전량 재조명 금지 — **우선순위/피드백 갱신**
  (화면에 보이는/근접 카드만, 나머지 라운드로빈), persistent radiance. UE Lumen surface-cache feedback.
  MAX_CARDS(메모리 예산)에 더해 **per-frame relight(컴퓨트) 예산**을 캡.
- **D3 march/clipmap 샘플 비용**: 원거리 step 수 감소(cone/LOD), clipmap 레벨 선택 비용 절감, 거리 기반
  early-out.
- 각 항목 RenderQuality{Low,Med,High} 노브로 결속, 품질 PT 잔차로 정직 보고.

### Stage E — 60fps 검증
- 데모 앵글 양 백엔드 프레임 ms **≤16.6(≥60fps)** 달성 확인. GDF GI 품질 vs 현재 비교(수용 가능).
- 보수적 스테이지(A/B/C) 갤러리 바이트 동일; 품질 영향(D)은 RenderQuality 티어 + 잔차 측정. 정직 보고
  (어느 패스가 얼마 줄었는지 표).

## 설계 제약 (CLAUDE.md 5원칙)
1. **근본 원인**: 마이크로 패치 금지. 비용의 근원(풀스크린 레이 수 / 카드 텍셀 수 / 미컬링 드로우)을 줄인다.
2. **측정 주도**: `PROFILE_GPU`가 성공 지표. 모든 before/after를 ms로 보고.
3. **확장성**: 모든 성능 노브(해상 배율, relight 예산, march step, 컬링 토글)를 `quality.rs`
   RenderQuality 티어 한 곳에. 기본=현 품질(Med).
4. **단일 소스**: 컬 AABB는 레지스트리/fuse 한 곳에서(중복 금지).
5. **검증 후 주장**: 양 백엔드 + DX≡VK + 무회귀(보수적) / 잔차(품질) 수치 정직 보고 후 커밋.

## 하지 말 것
- 갤러리 무회귀 위반(보수적 컬링은 바이트 동일; 품질 변경은 티어/측정). DX≡VK 깨기. Vulkan 검증 경고.
- 측정 전 컬링이 답이라 단정. 보이는 오브젝트를 컬해 깜빡임 유발(보수성 위반).
- HW-RT 경로 변경. 새 무거운 의존(승인 필요). 스트리밍(월드) — 범위 외.
- 한 앵글만 빠르게 만들고 일반화 누락(여러 앵글 측정).

## 파일 (예상)
- 수정 `apps/sandbox/src/cull.rs`(데모→씬 드로우 일반화), `deferred.rs`(indirect draw), `gi.rs`/
  `reflect.rs`/`gdf.rs`(하프해상 + 캐시 갱신 예산), `quality.rs`(성능 티어), `main.rs`(배선).
- 신규 `apps/sandbox/src/occlusion.rs`(Hi-Z) / `coverage.rs`(선택), `crates/shader/shaders/
  {cull,hiz,coverage,*_halfres}.slang`.
- 수정 `docs/ROADMAP.md`(이 트랙 추가), 본 문서(Stage 0 측정 표).

## 현재 상태 (착수 전 검증 항목)
- 빌드/clippy/fmt 클린, 갤러리 무회귀 기준선(양 백엔드, `base_dx/base_vk`).
- Scalable-GI 완료: GDF GI가 콘텐츠(Sponza/레벨) **디폴트**(`P11_LEGACY_IBL` escape). 클립맵
  (`P11_GDF_CLIP_LEVELS`, 콘텐츠 기본 4레벨), surface cache(MAX_CARDS=1024), 머티리얼 wrap 샘플러.
- `cull.rs`는 **데모 전용**(큐브 그리드, `P7_CULL`) — 실제 씬 드로우엔 컬링 없음(Stage A가 일반화).
- `PROFILE_GPU=1` 패스별 ms 덤프, `RENDER_QUALITY=low|med|high` 티어, `CAM_EYE`/`CAM_TARGET` 앵글 고정.
