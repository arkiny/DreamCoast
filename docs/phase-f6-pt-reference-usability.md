# F6 (slice) — Content path-tracer reference usability + residual automation

상위: [gi-fidelity-roadmap.md](gi-fidelity-roadmap.md) §F6 (검증·견고성 인프라). 이 문서는 F1–F5를
**PT 잔차로 검증**하기 위한 선결 작업이다: 지금 콘텐츠 PT는 (a) 실내 뷰에서 노출이 어긋나 사실상 검정이고,
(b) 콘텐츠 잔차 게이트가 자동화돼 있지 않다. 둘 다 고쳐 "측정 도구부터 벼린" 뒤 F1로 간다.

## 문제 진단 (2026-07-13, 코드 근거)

**콘텐츠 PT가 실내/그림자 방향에서 순수 검정(mean 0.00)이었다.** 조사 결과:

1. **라이팅은 정상.** 같은 실내 뷰를 밝은 고정 노출(EV100=6)로 PT 렌더하면 mean 6.2 / 37% lit — 다중바운스
   GI가 실제로 존재한다(MAX_BOUNCES=8, RR_START=3, `rt_common.slang`). 설계 의도대로 "실내는 flat sky
   ambient가 아니라 다중바운스 GI로 채운다"(`sky_common.slang` L97-98)가 성립한다.
2. **노출이 근본.** 고정 `EV100=11`은 햇빛 외부 기준이라, 물리적으로 여러 stop 어두운 실내 GI가 tonemap에서
   검정으로 crush된다(실제 카메라와 동일; 래스터는 flat IBL ambient로 실내를 들어올려 밝게 보임 — 이게 오히려
   비물리적).
3. **자동노출이 PT 경로에서 무효.** `AUTO_EXPOSURE=1`도 PT 실내는 검정(mean 0.00). 원인: PT tonemap이
   자동노출 버퍼를 **무시하고 고정 `self.exposure`를 사용**한다 (`main.rs` L8097-8108, `tm_exposure =
   self.exposure` for `pt_active`). 자동노출 히스토그램은 디퍼드 lit `hdr`을 metering해 GPU `exposure_buf`를
   적응시키지만(`main.rs` L7311, `deferred.rs record_auto_exposure`), PT tonemap은 그 값을 읽지 않는다.
   그래서 AUTO_EXPOSURE가 PT에 아무 영향이 없고, 기본 EV100(≈14)이 11보다도 어두워 검정이 된다.

디퍼드 `hdr`은 PT 모드에서도 쓰인다(PT는 별도 `rt_out` 스토리지 이미지에 렌더하고 tonemap이 소스를 선택 —
`main.rs` L7551-7569). 따라서 자동노출 metering 소스(디퍼드 hdr)는 그대로 유효하다.

## 목표

콘텐츠 PT가 **어느 카메라에서도 자동으로 노출을 맞춰** ground-truth로 사용 가능하게, 그리고 콘텐츠 씬의
raster-vs-PT 잔차를 **회귀 게이트로 자동화**한다.

## 접근

### A) PT 경로 자동노출 연결
- `pt_active && auto_exposure`일 때 PT tonemap이 **적응된 exposure를 사용**하도록 한다. 두 안:
  - **A1 (권장, GPU 무읽기)**: tonemap 패스가 `exposure_buf`(bindless)를 옵션으로 읽어 `tm_exposure` 대신
    곱한다 — 디퍼드 라이팅이 이미 하는 방식과 동일한 소스. push-constant에 `exposure_buf_index` + 플래그 추가.
  - **A2 (대안, CPU readback)**: 프레임 끝에 `exposure_buf[0]`를 CPU로 읽어 다음 프레임 `tm_exposure`에 사용
    (1프레임 지연). 셰이더 무변경이나 readback 지연·동기화 필요.
  - A1 채택 — 지연 없고 라이팅과 단일 소스.
- 헤드리스 캡처(`--screenshot-clean`)에서도 자동노출이 warmup 안에 수렴하도록 확인(현재 warmup 내 metering
  프레임 충분).

### B) 콘텐츠 PT 잔차 자동화 (`golden-image.py`)
- `docs/golden-image-regression.md` L88 스펙대로: 매니페스트 config에 `pt: true` + `residual_budget`
  필드를 추가. 러너가 그 config를 raster + `P8_PATHTRACE=1` 두 번(같은 고정 카메라, 자동노출 ON) 렌더하고
  `rt-compare.py`로 잔차(avg/>8/>32)를 재, **잔차 ≤ 기록된 budget(개선/중립)** 이면 PASS.
- 초기 config: `sponza_intel` 고정 카메라 1–2개(햇빛+실내 각 1). budget은 최초 측정치로 시드.
- PT 캡처 결정성: 고정 spp·시드(`rt_path.slang`의 `pc.frame` 기반 rng)로 run-to-run 재현 확인(콘텐츠 GI
  잔차 floor는 tolerant 허용).

## 게이트 (프로젝트 불변)
- **갤러리 바이트 동일**(`65d04ceca2c4…`): 자동노출은 갤러리 OFF(고정 노출 앵커)라 무영향 — 확인.
- **결정론**: A1은 exposure_buf를 곱만; 자동노출 자체는 이미 기존 기능. PT 자동노출 캡처 run-to-run 재현 확인.
- **DX≡VK**: 자동노출은 기존 크로스백엔드 기능(A1은 tonemap에 곱 하나 추가) — Windows 재검증 배치에 포함.
- **단일 소스**: exposure_buf(디퍼드 라이팅과 동일 버퍼) 재사용, 새 노출 상태 신설 금지.

## 검증
- 실내 PT 뷰가 `AUTO_EXPOSURE=1`로 자동 노출돼 GI가 보인다(검정 아님) — before mean 0.00 → after 유의미.
- 햇빛 뷰는 과노출 없이 유지.
- `golden-image.py`의 새 PT config가 PASS(잔차 ≤ budget), 갤러리 앵커 불변.

## 비목표 / 후속
- PT의 실내 밝기 자체를 올리는 것(더 많은 바운스/가짜 ambient)은 하지 않는다 — 물리 정확성 유지. 노출만 맞춘다.
- 이 슬라이스 후 **F1 표면 캐시 가상화**로 이동(로드맵 최우선). 그때 이 PT 잔차 자동화가 F1 이득의 측정 기반.
