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
| 배치·그룹·미니배치 | DAPO (Qwen2.5-32B dense) | 프롬프트 512 × G 16 = 8,192 샘플/롤아웃 step; 미니배치 32 프롬프트(512 샘플) → **롤아웃 step 당 16회 업데이트**; 동적 샘플링 과잉 생성 3배(gen 1,536) | 논문 §4.1 (arXiv 2503.14476); verl v0.4.0 `recipe/dapo/run_dapo_qwen2.5_32b.sh` (train_prompt_bsz 512, gen_prompt_bsz ×3, n 16, mini 32) | 2026-10-22 | 32B dense, 128 GPU(16노드×8). alpha 는 1 업데이트/step (GBS = 프롬프트 × 16) |
| lr·clip·KL·길이 | DAPO | lr 1e-6 고정, 워밍업 10(레시피)/20(논문) step, clip 0.2/0.28, KL 없음, token-mean, 응답 상한 20,480 = 16,384 + 완충 4,096, 완충 벌점 1.0, temperature 1.0 | 같은 원천 | 2026-10-22 | — |
| 배치·그룹·업데이트·lr | MiMo-V2.6 (Pro 1.02T/42B 활성 · Flash 310B/15B 활성 MoE) | 프롬프트 1,568 × G 16 = 25K 궤적/step (2.7~3.7B 학습 토큰/step), 비동기 부분 롤아웃 staleness 4, 토큰별 IS·4개 분리 클립 [0.2, 5.0] 초기값(entropy 로 런타임 조정), prompt-mean 손실, Muon lr 3e-6 고정·워밍업 없음·wd 없음·grad clip 1.0, RL 문맥 1M | 보고서 §4.1·§5.1·§6.4 (arXiv 2610.11959) | 2026-10-22 | 롤아웃당 업데이트 횟수·미니배치 **공개 안 됨**; 총 step·에폭 공개 안 됨 (그림 12 는 30 step 구간, 비용만 $2.6M/$0.9M); GPU 수 공개 안 됨 |

## 데이터 · 블렌드 · 필터

| 항목 | 모델 | 값 | 원천 | 조사일 | 비고 |
|---|---|---|---|---|---|
| 학습 데이터·에폭 | DAPO | DAPO-Math-17K: 고유 프롬프트 약 17.9K(정수 답), HF parquet 은 **1,791,700 행 = 100배 복제**; 레시피 `total_epochs=1` 은 이 복제본 기준 → 고유 프롬프트로는 최대 100 바퀴. 논문 그림 1 x축은 '기울기 업데이트 step'(절대값 본문 미기재), 롤아웃 step 수·에폭 수 공개 안 됨 | HF `BytedTsinghua-SIA/DAPO-Math-17k` (행 수), 논문 §3.5, verl 레시피 README | 2026-10-22 | 동적 샘플링은 0/16·16/16 그룹을 버리고 gen 배치를 반복 (`max_num_gen_batches` 상한) |
| 학습 데이터·에폭 | MiMo-V2.6 | 도메인 비율 agentic·경쟁 코딩 68% · 도구 12% · 디자인 13% · 문맥 3% · 보안 4%; 25개 프로파일 소스, 환경 "수천 개", 공개 환경 4종 약 7K 과제 (§7.2 표 5); 프롬프트 총수·재사용(에폭) **공개 안 됨**; 동적 샘플러가 all-pass/all-fail 그룹 제거 | 보고서 §4.2.2·§5.1·§6.3·§7.2 | 2026-10-22 | 데이터는 저장소 기반 데이터 플레인(메타데이터·페이로드 분리)으로 공급 |

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
