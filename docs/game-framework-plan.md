# 게임 프레임워크 + 던전 크롤러 계획서

상태: **초안 — 승인 대기** (2026-07-31)
목표 게임: **탑다운 액션 던전 크롤러** (고정 앵글 실시간 액션, 첫 클래스 = 전사)
애셋 정책: 전부 절차 생성 또는 AI 생성 (수작업 애셋 없음)

---

## 0. 현황 요약 (실측)

엔진은 렌더링은 깊지만 게임 런타임은 사실상 비어 있다 (`docs/commercial-engine-gap-analysis.md`와 일치).

**있는 것 (게임에 바로 쓸 토대):**
- ECS (`crates/scene/ecs.rs`, sparse-set, spawn/despawn/query2) + 부모-자식 트랜스폼 전파(병렬, 비트-동일)
- glTF 애니메이션 재생 (`animation.rs` — 단일 클립 루프, 보간 3종) + GPU 스키닝/모프/디폼 링
- **런타임 절차 메시 업로드** (`MeshRegistry::upload_geometry` — CPU 슬라이스 → 핸들, 쿡 불필요)
- **CPU SDF 쿼리** (`SdfVolume::sample`이 CPU에 존재, `mesh_sdf.rs` 셀-그리드 브로드페이즈) — 임의 메시 충돌의 씨앗
- 쿡 캐시 (content-hash 키 — **절차 생성 결과도 시드 키로 캐시 가능**)
- 고정 스텝 시뮬 루프(1/60, 보간 알파)가 `App::frame`에 이미 존재
- 멀티뷰(`view.rs`), 레벨 핫스왑, 청크 스트리밍

**없는 것 (전부 신규):**
충돌/피직스, 캐릭터 컨트롤러, 입력 액션 매핑(엣지 검출조차 앱이 수기), 애니 블렌딩/스테이트머신, 게임 UI(imgui 디버그 패널뿐), 오디오, ECS의 resources/events/커맨드버퍼, 게임플레이 데이터를 실을 레벨 포맷.

**구조적 제약 (설계에 반영):**
1. "샌드박스가 곧 엔진" — `apps/sandbox/main.rs` 9.4k줄에 렌더 조립이 전부 들어 있고 크레이트가 아님. 골든 게이트가 샌드박스에 걸려 있으므로 **대수술 대신 lib 노출 + 훅 주입**으로 우회한다 (§2).
2. **움직이는 오브젝트는 GI/SDF/반사에 안 보임** (GDF·표면캐시·TLAS는 셋업 시 1회 빌드). → 던전 지오메트리는 **로드 시 정적 업로드**로 풀 GI를 받고, 캐릭터/몬스터만 동적(직접광+그림자 정상, 간접광 기여만 없음)으로 수용. 여닫이 문은 GI 잔상이 남으므로 v1은 창살문/열리면 사라지는 문으로 회피.
3. 레벨 조명이 `level_lighting`에서 **포인트 라이트 4개 UBO 캡** — 횃불 던전에 치명적. 클러스터드 라이트 경로로 게임 라이트를 태우는 조사 필요 (M3, §5-R1).
4. 디퍼드 경로는 오브젝트당 1드로우(인스턴싱 없음) — 타일당 메시 배치 대신 **청크 단위 병합 메시**로 드로우 수 제어 (§4.1).

---

## 1. 전체 구조

```
crates/game        신규 — 재사용 게임 프레임워크 (엔진 종속, 게임 비종속)
  input/           ActionMap: VK/마우스 → 액션, just_pressed/released, WASD→축벡터
  physics/         그리드 충돌(1차) + SDF 캡슐 쿼리(2차), 레이캐스트(DDA), 원-vs-원/부채꼴
  anim/            AnimStateMachine + 크로스페이드 블렌딩 (scene::animation 확장 위에)
  camera/          팔로우 카메라(탑다운 고정 피치, 스무딩)
  combat/          Health/Team/Hurtbox/DamageEvent 등 공용 컴포넌트·시스템
apps/dungeon       신규 — 게임 본체 (던전 생성, 전사, 몬스터, 게임 플로우, HUD)
apps/sandbox       불변 유지 — lib 타겟 추가로 렌더 파사드 노출 (§2)
crates/scene       소폭 확장 — resources/events/커맨드버퍼/query3, 애니 포즈 샘플링 API
```

원칙: **`apps/sandbox`의 렌더 경로와 골든 게이트는 건드리지 않는다.** 프레임워크 코드는 훅 기본 no-op으로, 게이트 배터리 불변이 각 랜딩의 통과 조건.

## 2. M0 — 프레임워크 seam (엔진 작업, ~1주)

1. **sandbox lib화**: `apps/sandbox`에 `lib.rs` 추가, `App`을 크레이트 외부에서 구동 가능하게. `App::frame`의 시뮬 구간(고정 스텝 내부)과 카메라 결정, UI 드로우에 **`GameHooks` 트레이트** 주입점 3개를 연다:
   - `fn fixed_update(&mut self, world: &mut World, input: &ActionState, dt: f32)`
   - `fn camera(&self, alpha: f32) -> Option<CameraPose>` (None이면 기존 Fly/Orbit)
   - `fn draw_ui(&mut self, ui: &Ui, world: &World)`
   - 훅 미장착 시 현행과 바이트-동일 → **골든 배터리 ALL PASS가 랜딩 게이트.**
2. **입력 액션 매핑** (`game::input`): 프레임 경계 스냅샷으로 `pressed/just_pressed/just_released/axis2d`. 바인딩은 RON 데이터.
3. **ECS 확장** (`crates/scene`): `Resources`(타입맵 싱글턴), `Events<T>`(더블버퍼), `Commands`(지연 spawn/despawn — 병렬 구간 구조 변경 금지 계약 준수), `query3/query4`.
4. `apps/dungeon` 스캐폴드: sandbox lib + GameHooks로 빈 씬 구동.

## 3. M1 — 걷는 던전 (절차 생성 + 충돌, ~1–2주)

1. **던전 생성기** (`apps/dungeon`): 시드 결정적(rand_chacha) 방+복도 생성 → 타일 그리드(Floor/Wall/Door/Entry/Exit). 그리드가 **충돌·경로탐색·지오메트리의 단일 소스**.
2. **지오메트리**: 그리드 → **청크(예: 16×16타일) 단위 병합 메시** greedy meshing → `upload_geometry`. UV는 월드-평면 투영, 머티리얼은 AI 생성 텍스처(base/MR/normal, 기존 `MaterialDesc` 그대로).
3. **GI 편입**: 청크 메시를 정적 씬으로 등록 → 기존 GDF/표면캐시/GI가 그대로 동작. 청크 SDF는 `bake_mesh_sdf`(CPU)로 굽되 **쿡 캐시 키 = 생성기 버전+시드**로 재실행 무료화. 베이크 시간 실측이 이 마일스톤의 리스크 게이트 (§5-R2).
4. **충돌 v1 = 그리드**: 원(캐릭터 반경)-vs-타일 슬라이드, 레이캐스트 = 그리드 DDA. SDF 캡슐 쿼리는 소품용으로 M4 이연. Rapier/Jolt 같은 외부 피직스는 **v1 비도입** (탑다운 액션에 그리드+원 충돌로 충분, 의존성 최소).
5. **전사 플레이스홀더**(캡슐)로 WASD 이동+슬라이드, **탑다운 팔로우 카메라**(고정 피치 ~55°, 위치 스무딩, 카메라 충돌 불필요).

## 4. M2 — 전사와 전투 (~2주)

1. **전사 애셋**: AI 생성 리깅 휴머노이드 glTF + 클립(idle/run/attack×3콤보/hit/death, **전부 제자리(in-place) — 루트모션 미지원 정책**). 기존 glTF skin+anim 임포트·GPU 스키닝 경로 그대로. 애셋은 gitignore + 취득 스크립트(IntelSponza 관례).
2. **애니 블렌딩** (`crates/scene` 확장): 클립→포즈버퍼 샘플링 API 분리, TRS lerp/slerp 크로스페이드. 블렌드트리·IK는 범위 밖. 스테이트머신(`game::anim`)은 데이터(RON) 정의: 상태+전이+페이드 시간.
3. **전투**: 부채꼴 판정(공격 아크 vs 몬스터 원) → `DamageEvent` → Health/사망. 전사 v1 무브셋: 이동, 3타 콤보, 회피 구르기. 클래스는 RON `ClassDef`(스탯+무브셋 테이블)로 데이터화 — 두 번째 클래스의 수용 준비.
4. **몬스터 1종**: FSM(대기→인지→추적→공격→사망), 경로탐색 = 타일 그리드 A*.
5. **HUD v1 = imgui 오버레이**(HP/스태미나/콤보) — 디버그 룩 수용, 전용 `crates/ui`는 명시적 이연.

## 5. M3 — 게임 루프 완성 (~1주)

입구→출구(다음 층 재생성) 진행, 사망/재시작, 포션 픽업, **횃불 조명**(→ **R1 조사: 4-라이트 캡 해제** — 게임 라이트를 클러스터드 경로로; 실패 시 v1은 방향광+GI로 수용), 드로우/틱 예산 실측 퍼프 패스.

### 리스크 원장
- ~~**R1 라이트 캡**~~: **해소(M3)** — `level_lighting`의 4-포인트 절단 제거, UBO 4슬롯을 넘는
  프레임은 자동으로 클러스터드 froxel 경로. `Light::range`(authored, 0 = 컷오프 없음) + 단일 소스
  `point_attenuation`으로 경로 간 셰이딩 불변. 실측: 24 횃손 +0.46 ms @2560×1440(예산 ~1 ms 이내),
  4-라이트 3경로 sha256 동일, 골든 배터리 불변. 잔여 수용 사항: **횃불 그림자 없음**(태양만 캐스터),
  **횃불 빛 GI 미기여**(직접광 전용). 상세 = `docs/clustered-lighting.md` §2 파리티 설계·§3 R1 실측·§4.
- **R2 SDF 베이크 시간**: 생성 던전 CPU 베이크가 로드 시간을 지배할 수 있음 — 청크 분할+시드 쿡캐시로 완화, M1에서 실측.
- **R3 동적 GI 부재**: 캐릭터 간접광 기여 없음 — v1 수용, 재검토는 별도 페이즈.
- **R4 AI 애셋 품질**: 리깅/클립 품질 편차 — glTF 검증 스크립트(본 수/클립 목록/제자리 여부)로 게이트.
- 문서·주석·커밋에 상용 엔진/제품명 금지(기존 RULE), 애셋 라이선스 확인 필수.

## 6. 이연 (본 계획 범위 밖, 순서만 합의)

오디오 크레이트(cpal 믹서+AI SFX) → 전용 게임 UI → SDF 소품 충돌·문(동적 GI 결정 포함) → 두 번째 클래스 → 세이브/로드(ECS 직렬화) → 게임패드.

## 7. 렌더링 트랙과의 관계

라이팅 마무리(F6O 증분2, `docs/lighting-wrapup-plan.md` 승인 대기)와 **독립 병행 가능** — 본 트랙은 렌더 경로 무변경(훅 주입만)이며 골든 게이트가 상호 간섭을 검출한다.
