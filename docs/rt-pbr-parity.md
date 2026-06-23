# 패스트레이서 정밀화 — Ground-Truth PBR 계획

> 상위: [phase-8-raytracing.md](phase-8-raytracing.md) (Phase 8 ✅). **상태: 📝 계획 / 사인오프 대상.**
> 후속 트랙으로, Phase 8의 "디퓨즈 GI only" 한계를 해소한다.

## 목표 (사용자 확정)

1. **패스트레이서를 향후 작업의 Ground Truth(레퍼런스)로 만든다 — 최대한 정밀·무편향(unbiased).**
   래스터(디퓨즈+split-sum IBL 근사)가 이 결과를 향해 수렴해야 하며, 패스트레이서가 정답이다.
2. **래스터와 같은 PBR 머티리얼 모델**(glTF metallic-roughness)을 써서 두 렌더러를 직접 비교 가능하게 한다.
   반사·금속/거칠기 응답·정반사가 물리적으로 정확히 나오도록(반사, PBR 반영).

**비-목표(명시)**: 실시간 성능 최적화·디노이즈는 범위 밖(누적 수렴 의존). 굴절/투명·SSS·볼류메트릭은 후속.

## 현재 격차 (패스트레이서 vs 래스터 PBR)

| 항목 | 래스터 `pbr.slang` | 패스트레이서 (현재) | 목표 |
|---|---|---|---|
| BRDF | Cook-Torrance (GGX D / Smith G / Schlick F) | **Lambert 디퓨즈만** | 동일 Cook-Torrance + 정반사 |
| 머티리얼 | base_color·metallic·roughness·ao·emissive (+텍스처) | albedo(rgb)+emissive만 | 전체 metallic-roughness |
| 금속/정반사 | F0=lerp(0.04,albedo,metallic), 정반사 IBL | 없음 | GGX 정반사 로브 → **실반사** |
| 간접광 | split-sum IBL(근사 큐브) | 디퓨즈 바운스 | **경로추적 GI/반사(정답)** |
| 직접광 | 태양(PCF 섀도우)+포인트광 | 태양(섀도우 레이)만 | 태양(디스크)+포인트광 NEE+MIS |
| 노멀 | 화면공간 미분 TBN + 노멀맵 | 보간 지오메트리 노멀 | 탄젠트 노멀맵(텍스처 머티리얼) |
| 텍스처 | base/mr/normal/emissive 샘플 | 없음 | hit에서 UV 보간 후 샘플 |
| 추정기 | — | 코사인 바운스, 고정 4바운스 | **MIS + 러시안룰렛 + 디스크광** |

핵심: 정점 레이아웃 32B에 **UV가 이미 포함**(offset 24)되어 hit에서 텍스처 샘플 가능. 탄젠트는 없음 →
삼각형 3정점(pos+uv)으로 hit에서 계산하거나 별도 버퍼. 인스턴스 테이블만 확장하면 됨(현재 32B 레코드).

## 마일스톤 (각 게이트: build+fmt+clippy `-D warnings` + 두 백엔드(VK≡DX) + Vulkan VUID 0 + 인라인≡파이프라인 + 스크린샷; 인라인/파이프라인 두 경로 동시 갱신)

### G1 — 머티리얼 데이터 패리티
- 인스턴스 테이블 레코드 확장: `{ vtx, idx, base_color(rgb)+? , metallic, roughness, emissive, ao, tex 인덱스[base,mr,normal,emissive] }`.
  현재 32B → 48~64B로. 호스트 패킹(`build_pt_instance_table`) + 셰이더 `Instance` 구조 동시 갱신.
- 샘플 씬·Cornell 양쪽에서 `SceneObject`의 metallic/roughness/텍스처를 그대로 채운다.
- **검증**: hit에서 머티리얼을 읽어 metallic/roughness 디버그 시각화(예: roughness를 그레이로) → 두 백엔드 일치.

### G2 — 마이크로페이싯 BSDF + 중요도 샘플링 (반사 등장)
- 래스터와 **동일한** D(GGX)/G(Smith, height-correlated)/F(Schlick) 평가. `F0=lerp(0.04,base,metallic)`,
  `diffuse=(1-metallic)*base/π`. 에너지 보존(kd=(1-F)(1-metallic)).
- 바운스 샘플링: **GGX VNDF**(visible normal) 정반사 + 코사인 디퓨즈, Fresnel 가중 로브 선택, 정확한 pdf 반환.
- 직접 태양광은 풀 BSDF로 평가(half-vector 정반사 포함). → 크롬/구리 구가 **씬·하늘을 실제로 반사**.
- **검증**: 금속(거칠기 0.08)·거친 금속·유전체 구 비교, 래스터의 IBL 반사와 **시각적으로 일관**(정답은 더 정확).
  인라인≡파이프라인, VK≡DX.

### G3 — 무편향 추정기 정밀화 (Ground Truth 핵심)
- **MIS**(BSDF 샘플 ↔ 광원 NEE)로 태양·포인트광·발광면 결합 — 저분산·무편향(파워 휴리스틱).
- **러시안 룰렛**(N바운스 후 throughput 기반 확률 종료) → 무편향으로 깊은 경로 허용(8~16+ 바운스).
- **태양을 디스크 광원**(유한 입체각)으로 → 물리적 소프트 섀도우. 포인트광 NEE 추가(래스터 패리티).
- 펌웨어 결정성 유지(시드·샘플 순서 고정). 파이어플라이 클램프는 **무편향 위해 기본 off**(옵션).
- **검증**: 분산 감소(수렴) 측정, 소프트 섀도우 가시, 동일 씬에서 노이즈가 균일 누적으로 줄어듦.

### G4 — 텍스처 머티리얼 + 노멀 매핑
- hit에서 보간 UV로 base_color(sRGB)·metallic-roughness(linear)·emissive 텍스처 샘플(기존 바인드리스 텍스처 테이블).
- 탄젠트: 삼각형 3정점(pos+uv)에서 hit-time 계산 → 탄젠트공간 노멀맵 적용(아보카도 모델이 노멀맵 보유).
- **검증**: 아보카도가 패스트레이서에서도 텍스처/노멀 디테일을 보임, 래스터와 일관.

### G5 — 검증 + 문서화
- **나란히 비교 하니스**: 같은 카메라/조명에서 패스트레이서(수렴) vs 래스터 PBR 스크린샷 + 수치 차이.
  패스트레이서를 레퍼런스로, 래스터의 IBL 근사 오차를 문서화(향후 래스터 개선의 기준).
- 인라인≡파이프라인(픽셀 근사), VK≡DX, Vulkan VUID 0 / D3D12 디버그 클린.
- `docs/rt-pbr-parity.md` 검증/한계 채우기, `phase-8-raytracing.md`의 "디퓨즈 only" 한계 → 해소 표기, ROADMAP 메모.

## 설계 메모 / 위험
- **인라인·파이프라인 동시 유지**: 모든 BSDF/샘플링 코드는 두 셰이더(`rt_path.slang`/`rt_pipeline.slang`)에
  동일 적용. 공유 헬퍼를 `bindless.slang` 인접 인클루드(예: `rt_common.slang`)로 묶어 중복/드리프트 방지 검토.
- **푸시상수/페이로드 예산**: 파이프라인 페이로드가 커질 수 있음(현재 max_payload 64B). BSDF 상태는 raygen 루프가
  들고 페이로드는 최소(hit 결과)만 → 64B 내 유지 목표, 필요 시 상향(RTX 2070 여유).
- **결정성·이식성**: VNDF/MIS 도입 시에도 PCG 시드·로브 선택 순서를 양 API 동일하게 고정(Phase 8에서 VK≡DX 유지 실적).
- **스토리지 버퍼 예산**: 인스턴스 테이블 1개 확장이라 영향 적음(~34/64 사용).
- **정답 정의**: "무편향 + 충분 수렴"이 기준. 클램프/디노이즈 같은 편향 도구는 기본 비활성(레퍼런스 무결성).

## 제안 진행 순서
G1 → G2(여기서 반사가 등장, 가장 큰 체감) → G3(무편향 정밀화) → G4(텍스처/노멀) → G5(검증·문서).
G2까지만으로도 "반사·PBR 반영" 1차 목표 충족. G3가 "Ground Truth" 정밀성의 핵심. G4는 텍스처 모델 완성.
