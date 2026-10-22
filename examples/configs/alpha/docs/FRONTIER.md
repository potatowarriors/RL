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
| 배치·그룹·업데이트·lr | MiMo-V2.6 (Pro 1.02T/42B 활성 · Flash 310B/15B 활성 MoE) | 프롬프트 1,568 × G 16 = 25K 궤적/step (2.7~3.7B 학습 토큰/step), 비동기 부분 롤아웃 staleness 4, 토큰별 비율 r = sg[π_train/μ_rollout] (분모는 **롤아웃 시점 확률**, 부분 롤아웃은 재계산 안 함) 에 **마스크** M = 1[A≥0 ∧ ε⁺l ≤ r ≤ ε⁺h ∨ A<0 ∧ ε⁻l ≤ r ≤ ε⁻h] — PPO min(·) 클립도 TIS 절단도 아님; 양·음 advantage 별 4개 경계, 초기값 [0.2, 5.0] (비율 척도), entropy 가 낮으면 양의 경계를 넓히고 음의 경계를 좁힘 (규칙·목표값 미기재); prompt-mean 손실, Muon lr 3e-6 고정·워밍업 없음·wd 없음·grad clip 1.0, RL 문맥 1M | 보고서 §4.1·§5.1·§6.4 (arXiv 2610.11959) | 2026-10-22 | 롤아웃당 업데이트 횟수·미니배치 **공개 안 됨**; 총 step·에폭 공개 안 됨 (그림 12 는 30 step 구간, 비용만 $2.6M/$0.9M); GPU 수 공개 안 됨 |
| staleness 의 이유 | MiMo-V2.6 | **이유 공개 안 됨** — §5.1 "Training uses GRPO with asynchronous partial rollouts at a staleness of 4." (단위 미기재). 보고서가 적은 것은 부분 롤아웃의 목적뿐: §4.1 "Since rollout and grading have a long tail, we use partial rollout … to keep the running batch saturated", "a continuation must rebuild its KV cache after each policy update … a large batch amortizes the re-prefill"; §6.3 "limiting queueing delays that prolong rollout lifetimes and increase staleness" (staleness 는 줄일 비용); §5.5 "Rollout failures arose in the partial-rollout setting: shorter rollouts completed first after startup". 추론(명시 아님): 정책 갱신마다 궤적을 끊고 잇는 구조라 긴 에이전트 궤적이 여러 버전에 걸치며, staleness 는 한 궤적이 걸칠 수 있는 갱신 횟수의 상한 | arXiv 2610.11959 §4.1·§5.1·§5.5·§6.3 (2026-10-22 재열람) | 2026-10-22 | alpha 에는 옮기지 않음: s5 생성은 처리량 병목 (vLLM max_num_seqs 256×8 = 2,048 < target 당 4,048 시퀀스, 수집기가 age 한계에 매 step 닿음, 학습기 대기 2/3) 이라 age 4 의 속도 이득 0, 데이터·π_old 간격이 최대 16 업데이트(약 10배)로 커져 TIS 만으로 보정 → age 1 + 4회 업데이트 유지 (사용자 질의 2026-10-22) |
| 추론(롤아웃) 가속 | Qwen3.8-Flash-Next (125B/6B 활성 + n-gram 임베딩 51B + MTP 4B) | 48층 = 12 × (3 × GDN→MoE → 1 × QSA→MoE), 512 experts 10+1, 문맥 262K; **MTP 1층, multi-step 학습**; 기술 보고서 Table 4: 4-step 투기 디코딩 평균 수락 길이 4.06 (MT-Bench 3.44 · GSM8K 4.19 · MATH 4.29 · HumanEval 4.24 · MBPP 4.12); QSA(블록 예산 2,048 토큰, 압축비 4)는 1M 문맥에서 attention 커널 prefill 7.6× · decode 4.9× (64K 부터 이득); FlashQLA 가 FLA Triton 대비 forward 2~3×; 잔차 상태 FP8 저장. vLLM 공식 레시피(H100/H200 ×8): FP8 ckpt · TEP8 · `--moe-backend triton` · `--gpu-memory-utilization 0.85` · `--max-num-seqs 256` · `--enable-prefix-caching` · `--kv-cache-dtype fp8` · `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` (MTP 메모리 부족이면 3 미만으로). tok/s 수치는 **공개 안 됨** | HF 모델 카드 Qwen/Qwen3.8-Flash-Next · arXiv 2608.30320 §1·§2.1.1·§2.1.2·§2.2 · recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next | 2026-10-22 | alpha(24층 3:1 GDN:GQA, 192 experts top-8+shared, MTP 없음)에 옮길 것: fp8 KV(FA3 설치 확인, R1 게이트 필요) · 유틸 0.85→0.90 · MoE 백엔드 측정; MTP 는 헤드 학습 + 브리지 + 플러그인 + R3 route 가 필요한 별도 과제; QSA·n-gram 임베딩은 구조 변경이라 alpha_v3 후보 |
| 추론(롤아웃) 가속 | Qwen3-Next-80B-A3B | 3:1 GDN:Gated Attention, 512 experts 10+1, 3B 활성; "10 times inference throughput for context over 32K tokens" (vs Qwen3-32B, Base 모델 카드, "depends highly on the implementation"); MTP: "Boosts pretraining model performance and accelerates inference"; vLLM `{"method":"qwen3_next_mtp","num_speculative_tokens":2}` · SGLang `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-num-draft-tokens 4`; 수락률·가속 배수 **공개 안 됨** | HF 모델 카드 Qwen/Qwen3-Next-80B-A3B-Instruct | 2026-10-22 | alpha 와 같은 3:1 하이브리드의 가속 수단이 MTP 임을 확인 (Qwen3.8 도 유지) |
| 추론(롤아웃) 가속 | alpha_v2 s5 실측 (2026-10-22) | 8 엔진 합계 35.8K tok/s (38 h 평균, 엔진당 4.5K), 최근 10 step 32.5K (평균 생성 23.7K 토큰); 시퀀스당 생성 평균 19.9K · p90 31.4K · 상한 48K 도달 1.3%; 엔진: vLLM 0.25.1 · FLASH_ATTN(FA3 설치) · TRITON MoE · CUDA graph · chunked prefill · prefix caching 끔 · KV 풀 34.2 GiB = 290만 토큰 (블록 1,056 토큰, mamba 페이지 = attention 페이지) → 동시 시퀀스 약 110~130 (21~25K 문맥), 48K 면 58 — `max_num_seqs` 256 이 아니라 **KV 용량이 상한**. 디코드 1 step ≈ 27 ms = 가중치 읽기 ~9 ms + KV 읽기 ~9 ms + GDN 상태 ~0.5 ms + 커널 오버헤드 | s5 `rollout_groups.jsonl` · 엔진 로그 (kv_cache_utils·gpu_worker) · `RLVR_READINESS.md` §5.2~5.4 (T2 ngram_gpu −36%·R3 route 누락, FP8 생성 gen_kl 0.0024~0.0031) | 2026-10-22 | 레버 순위: ① fp8 KV + 유틸 0.90 + max_num_seqs 384 (추정 +50~90%, R1 게이트) ② MoE 백엔드 (0~15%, 측정) ③ MTP 헤드 (1.5~2×, 수 주 과제) · 기각: FP8 가중치 · bf16 GDN 상태 · prefix caching · EAGLE3 온라인(packing 비호환) |
| 롤아웃 배치·업데이트 | DeepSeek-R1 / R1-Zero (671B/37B) | 512 × 16 = 8,192; **미니배치 512 (32 × 16) × 16회 업데이트**, 내부 에폭 1; R1 ε=10(사실상 클립 없음), KL 0.001, 참조 모델 400 step 마다 리셋; Zero 10,400 step, R1 2단계 1,700 step; 상한 32K → 65K (step 8.2K 뒤) | arXiv 2501.12948 §2.1·§3.2 | 2026-10-22 | dense 아님(MoE 37B 활성), GPU 수 미기재 |
| 롤아웃 배치·업데이트 | Qwen3-235B-A22B / GSPO (Qwen3-30B-A3B) | Qwen3: 수치 없음("큰 배치·많은 롤아웃", 170 RL step, 3,995 쿼리); GSPO: 롤아웃 배치를 **4 미니배치 → 4회 업데이트**, 시퀀스 비율 클립 3e-4/4e-4, 시퀀스 평균 | arXiv 2505.09388 §4.2; arXiv 2507.18071 §5.1 Eq.5 | 2026-10-22 | Qwen3.8 보고서는 사전학습만 (RL 수치 없음) |
| 롤아웃 배치·업데이트 | DeepSeek-V4 (2026-04-26, arXiv 2606.19348) · V3.2 · V4.1-Flash | **V4: 수치 전부 미기재** — §5.1.1 "GRPO … hyper-parameters closely aligned with our prior research" 가 R1(2501.12948)·V3.2(2512.02556)를 가리킴; §5.2.4 "on-device 미니배치 수는 워크로드에 따라 동적으로 결정"; RL 중 길이 벌점·문맥 창은 effort 별로 다름(값 없음). V3.2 §3.1: "큰 롤아웃 배치를 여러 미니배치로 나눠 수 회 업데이트"(횟수 미기재), 음의 advantage 시퀀스만 정책 괴리 δ 초과 시 마스킹(δ 미기재), K3 보정 비편향 KL, Keep Routing·Keep Sampling Mask, "수천 step", RL 비용 > 사전학습의 10%. V4.1: 비동기 샘플 단위 디스패치, off-policy 비율 상한, staleness 토큰 마스킹, routing replay (수치 없음) | V4 §5.1.1·§5.2.4; V3.2 §3.1·§4.1; V4.1 HF 기술보고서 §5 | 2026-10-22 | DeepSeek 계열에서 업데이트 횟수를 숫자로 밝힌 것은 R1(16회)뿐이고 V4 가 그 설정을 따른다고 명시 |
| 롤아웃 배치·업데이트 | Kimi K3 (2.8T/104B) · K2.5 | 부분 롤아웃: 완료 비율 λ 에서 반복을 끊고 낙오는 다음 반복으로 이월; 토큰별 log-ratio [α, β] 밖은 기울기 0 + τ·(log-ratio)² 정규화 (K2.5 Eq.1, 토큰 평균); 배치·업데이트 수·staleness 수치 없음; 1M 문맥 | Kimi-K3 보고서 §4.1.2·§5.3.1; arXiv 2602.02276 §4.4.2 | 2026-10-22 | — |
| 롤아웃 배치·업데이트 | GLM-5 (744B/40B) · GLM-4.5 · slime | GLM-5 §3.2 'Reasoning RL'(수학·과학·코드·TIR, 이진 검증 보상 — 롤아웃이 아니라 RL 단계 이름): "entirely on-policy with a group size of 32 and a batch size of 32" (배치 단위가 프롬프트인지 궤적인지, 롤아웃당 업데이트 횟수는 **미기재** — 'on-policy' 를 1회로 읽는 것은 추론), IcePop β=2 (비율 [1/β, β] 밖 토큰 마스킹), ε 0.2/0.28, KL 제거, Eq.1 은 샘플·토큰 평균; agentic RL: 완전 비동기, K 업데이트마다 가중치 push 뒤 옵티마이저 리셋, 롤아웃 logprob 대비 토큰 IS + 양측 마스크, staleness > τ 샘플 폐기 (K·ε·τ 미기재); slime 데모(공식 레시피 아님): 8 × 8, GBS 64 → 1회, TIS [0.5, 2], 3,000 롤아웃, 262K | arXiv 2602.15763 §3.2·§3.5·§4.1; arXiv 2508.06471 §3.2·§3.5; THUDM/slime@a4202e0 | 2026-10-22 | GLM-4.5 코드 RL 은 토큰 가중 평균이 시퀀스 평균보다 좋았다고 보고 |
| 롤아웃 배치·업데이트 | Skywork-OR1-32B · Seed1.5-Thinking | Skywork: 1단계 256 × 16, 2단계 160 × 32, 미니배치 = 배치 → **1회 업데이트**("on-policy"), clip 0.2, 토큰 단위, 760 + 370 step, 16K → 24K (Math-7B 는 2회); Seed1.5: 스트리밍 롤아웃(완료 비율 α), 분리 클립, 토큰 단위, 수치 없음 | arXiv 2505.22312 §3.1·§8.1·표 11-12; arXiv 2504.13914 §4.2·§5.2 | 2026-10-22 | — |
| 롤아웃 배치·업데이트 | **Nemotron 3 Super 기술 보고서** (arXiv 2604.12374 — Ultra 가 '비동기 GRPO 와 안정화' 의 출처로 인용) | §3.2.4 Algorithm · §3.2.5 Infrastructure("Async RL Infrastructure"): "In multi-environmental RLVR, we sample 256 prompts per step and generate 16 responses per prompt." · "We train with a batch size of 4096, which corresponds to **a single gradient update per rollout**." · "The trainings used one-step off policy." · "we restrict the inference workers to be at most one step behind the latest model version." · "training can update generation worker weights without waiting for the remaining ongoing rollouts to finish." · "we mask the importance sampling ratio computed from the training and inference logprobs." · 생성 상한 49K → 64K | 2026-10-22 | 1회 업데이트를 **명문화한 유일한 2026 보고서**. alpha 3단계(128 × 16, 나이 1, in-flight, IS 마스킹)와 같은 구조 |
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
