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
| 롤아웃 배치·업데이트 | DeepSeek-R1 / R1-Zero (671B/37B) | 512 × 16 = 8,192; **미니배치 512 (32 × 16) × 16회 업데이트**, 내부 에폭 1; R1 ε=10(사실상 클립 없음), KL 0.001, 참조 모델 400 step 마다 리셋; Zero 10,400 step, R1 2단계 1,700 step; 상한 32K → 65K (step 8.2K 뒤) | arXiv 2501.12948 §2.1·§3.2 | 2026-10-22 | dense 아님(MoE 37B 활성), GPU 수 미기재 |
| 롤아웃 배치·업데이트 | Qwen3-235B-A22B / GSPO (Qwen3-30B-A3B) | Qwen3: 수치 없음("큰 배치·많은 롤아웃", 170 RL step, 3,995 쿼리); GSPO: 롤아웃 배치를 **4 미니배치 → 4회 업데이트**, 시퀀스 비율 클립 3e-4/4e-4, 시퀀스 평균 | arXiv 2505.09388 §4.2; arXiv 2507.18071 §5.1 Eq.5 | 2026-10-22 | Qwen3.8 보고서는 사전학습만 (RL 수치 없음) |
| 롤아웃 배치·업데이트 | DeepSeek-V4 (2026-04-26, arXiv 2606.19348) · V3.2 · V4.1-Flash | **V4: 수치 전부 미기재** — §5.1.1 "GRPO … hyper-parameters closely aligned with our prior research" 가 R1(2501.12948)·V3.2(2512.02556)를 가리킴; §5.2.4 "on-device 미니배치 수는 워크로드에 따라 동적으로 결정"; RL 중 길이 벌점·문맥 창은 effort 별로 다름(값 없음). V3.2 §3.1: "큰 롤아웃 배치를 여러 미니배치로 나눠 수 회 업데이트"(횟수 미기재), 음의 advantage 시퀀스만 정책 괴리 δ 초과 시 마스킹(δ 미기재), K3 보정 비편향 KL, Keep Routing·Keep Sampling Mask, "수천 step", RL 비용 > 사전학습의 10%. V4.1: 비동기 샘플 단위 디스패치, off-policy 비율 상한, staleness 토큰 마스킹, routing replay (수치 없음) | V4 §5.1.1·§5.2.4; V3.2 §3.1·§4.1; V4.1 HF 기술보고서 §5 | 2026-10-22 | DeepSeek 계열에서 업데이트 횟수를 숫자로 밝힌 것은 R1(16회)뿐이고 V4 가 그 설정을 따른다고 명시 |
| 롤아웃 배치·업데이트 | Kimi K3 (2.8T/104B) · K2.5 | 부분 롤아웃: 완료 비율 λ 에서 반복을 끊고 낙오는 다음 반복으로 이월; 토큰별 log-ratio [α, β] 밖은 기울기 0 + τ·(log-ratio)² 정규화 (K2.5 Eq.1, 토큰 평균); 배치·업데이트 수·staleness 수치 없음; 1M 문맥 | Kimi-K3 보고서 §4.1.2·§5.3.1; arXiv 2602.02276 §4.4.2 | 2026-10-22 | — |
| 롤아웃 배치·업데이트 | GLM-5 (744B/40B) · GLM-4.5 · slime | GLM-5 §3.2 'Reasoning RL'(수학·과학·코드·TIR, 이진 검증 보상 — 롤아웃이 아니라 RL 단계 이름): "entirely on-policy with a group size of 32 and a batch size of 32" (배치 단위가 프롬프트인지 궤적인지, 롤아웃당 업데이트 횟수는 **미기재** — 'on-policy' 를 1회로 읽는 것은 추론), IcePop β=2 (비율 [1/β, β] 밖 토큰 마스킹), ε 0.2/0.28, KL 제거, Eq.1 은 샘플·토큰 평균; agentic RL: 완전 비동기, K 업데이트마다 가중치 push 뒤 옵티마이저 리셋, 롤아웃 logprob 대비 토큰 IS + 양측 마스크, staleness > τ 샘플 폐기 (K·ε·τ 미기재); slime 데모(공식 레시피 아님): 8 × 8, GBS 64 → 1회, TIS [0.5, 2], 3,000 롤아웃, 262K | arXiv 2602.15763 §3.2·§3.5·§4.1; arXiv 2508.06471 §3.2·§3.5; THUDM/slime@a4202e0 | 2026-10-22 | GLM-4.5 코드 RL 은 토큰 가중 평균이 시퀀스 평균보다 좋았다고 보고 |
| 롤아웃 배치·업데이트 | Skywork-OR1-32B · Seed1.5-Thinking | Skywork: 1단계 256 × 16, 2단계 160 × 32, 미니배치 = 배치 → **1회 업데이트**("on-policy"), clip 0.2, 토큰 단위, 760 + 370 step, 16K → 24K (Math-7B 는 2회); Seed1.5: 스트리밍 롤아웃(완료 비율 α), 분리 클립, 토큰 단위, 수치 없음 | arXiv 2505.22312 §3.1·§8.1·표 11-12; arXiv 2504.13914 §4.2·§5.2 | 2026-10-22 | — |
| 롤아웃 배치·업데이트 | **Nemotron 3 Super 기술 보고서** (arXiv 2604.12374 — Ultra 가 '비동기 GRPO 와 안정화' 의 출처로 인용) | RL 알고리즘·인프라 절: "In multi-environmental RLVR, we sample 256 prompts per step and generate 16 responses per prompt." · "We train with a batch size of 4096, which corresponds to **a single gradient update per rollout**." · "The trainings used one-step off policy." · "we restrict the inference workers to be at most one step behind the latest model version." · "training can update generation worker weights without waiting for the remaining ongoing rollouts to finish." · "we mask the importance sampling ratio computed from the training and inference logprobs." · 생성 상한 49K → 64K | 2026-10-22 | 1회 업데이트를 **명문화한 유일한 2026 보고서**. alpha 3단계(128 × 16, 나이 1, in-flight, IS 마스킹)와 같은 구조 |
| 롤아웃 배치·업데이트 | Nemotron 3 Ultra **기술 보고서** (research.nvidia.com PDF, 65쪽) | §3.2 (p20): "we use a global batch size of 8192, with each sample generating 16 rollouts. Training begins with a maximum generation length of 48K tokens, which is later increased to 64K"; "largely follows the asynchronous GRPO algorithm with the stability optimizations proposed in NVIDIA (2026)"; §3.6.1 (p31): "During RL and MOPD, we train using a **one-step off-policy asynchronous RL setup**, so rollout generation is overlapped with the policy update"; §3.3.1 (p22) MOPD: 배치 1,024 프롬프트 × 롤아웃 1. 롤아웃당 업데이트 횟수를 숫자로 적은 문장은 없음 — 'global batch 8192 = 512 × 16' 과 '정책 업데이트와 생성이 1 step 어긋나 겹침' 으로 1회를 뒷받침 | 2026-10-22 | 레시피(아래 행)가 이를 `train_global_batch_size = prompts × G`, `force_on_policy_ratio: true` 로 명시 |
| 롤아웃 배치·업데이트 | Nemotron 3 Ultra (NeMo-RL 레시피) | student 512 × 16 = 8,192 = GBS → **1회**; teacher 128 × 16 = 2,048 = GBS → 1회; swe 32 × 16; mopd 1,024 × 1; 전부 `force_on_policy_ratio: true`(비율 1 고정, clip 0.2/0.28 불발), async 나이 1 + in-flight 갱신, 토큰 단위, KL 0 | `examples/nemo_gym/nemotron-3-ultra/*.yaml` | 2026-10-22 | alpha 와 같은 구조 (3단계 = teacher 기하) |

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
