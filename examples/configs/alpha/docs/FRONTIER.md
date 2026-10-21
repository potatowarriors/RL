# alpha RL 프론티어 대장

학습 결과를 바꾸는 결정의 근거로 조사한 프론티어 모델의 실제 설정을 모은다. 규칙은 사용자 공통 원칙 "결정 근거 — 프론티어 조사",
조사 절차와 비교표 양식은 `frontier-research` skill 이다. 결정 자체는 [`RL_PLAN.md`](RL_PLAN.md) 결정 표에 쓰고 여기를 링크한다.

**기록 규칙**: 한 행에 값 하나. 원천은 이번 조사에서 직접 연 링크와 절·표·파일 번호까지 적는다. 공개되지 않은 값은 "공개 안 됨".
조사일이 3개월 지났거나 그 계열에 새 모델이 나왔으면 다시 확인한다.
대상 계열: Alibaba Qwen · DeepSeek · Moonshot Kimi · Z.ai GLM · Xiaomi MiMo (각 계열의 조사 시점 최신) · 기준선 Nemotron 3 Ultra.

## 이미 있는 조사 (다른 문서에 정본)

| 주제 | 정본 | 조사일 |
|---|---|---|
| GDN·선형 attention 재귀 상태 dtype 배포 관행 (Qwen3-Next·3.5~3.8, Nemotron 3, Kimi Linear·GLM-5.3-Flash) | [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) "GDN 재귀 상태가 vLLM 에서 bf16 으로 저장된다" 절 | 2026-10-21 |
| Nemotron 3 Ultra RL 레시피 → alpha 대응 | [`RLVR_READINESS.md`](RLVR_READINESS.md) §2·§4, `../student_rlvr1_alpha.yaml`, NeMo-RL `examples/nemo_gym/nemotron-3-ultra/` | 2026-10-07 |
| Ultra RL 블렌드 구성 (스테이지별 행 수·원천) | [`RL_DATA.md`](RL_DATA.md) | 문서 참조 |
| MiMo-V2.6-RL Code (Harbor) 태스크·설정 | [`study/mimo_code_harbor.md`](study/mimo_code_harbor.md) | 2026-10-21 |

## RL 하이퍼파라미터 (lr · 스케줄 · 배치 · 그룹 · KL · clip · IS · 생성 상한)

| 항목 | 모델 | 값 | 원천 | 조사일 | 규모·구조 비고 |
|---|---|---|---|---|---|
| — | — | 아직 조사 없음 | — | — | — |

## 데이터 · 블렌드 · 필터

| 항목 | 모델 | 값 | 원천 | 조사일 | 비고 |
|---|---|---|---|---|---|
| — | — | 아직 조사 없음 | — | — | — |

## 보상 · 벌점 · 길이 제어

| 항목 | 모델 | 값 | 원천 | 조사일 | 비고 |
|---|---|---|---|---|---|
| — | — | 아직 조사 없음 | — | — | — |

## 단계 설계 · teacher · 증류

| 항목 | 모델 | 값 | 원천 | 조사일 | 비고 |
|---|---|---|---|---|---|
| — | — | 아직 조사 없음 | — | — | — |

## 학습 중 문제와 해결 (속도 · 메모리 · 안정성)

`problem-diagnosis` 의 "프론티어 선행 사례" 결과 중 다시 쓸 만한 것을 쌓는다.

| 문제 | 모델 | 대응 | 원천 | 조사일 |
|---|---|---|---|---|
| — | — | 아직 조사 없음 | — | — |
