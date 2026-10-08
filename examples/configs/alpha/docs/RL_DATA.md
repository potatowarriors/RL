# alpha RL 데이터 — 자산·준비된 블렌드·주의

RL 데이터의 현행 정본이다. 2026-10-07 이관 출처: Pai `SFT_RL_DATASETS.md` §3, alpha `README.md` "RL 데이터" 절,
`ALPHA_POSTTRAIN_PROGRESS.md` §4. SFT 데이터는 Pai `SFT_RL_DATASETS.md` §2 가 정본이다.
전수 인벤토리·스키마·샘플·라이선스는 [`SPEC_rl_dataset_inventory.md`](SPEC_rl_dataset_inventory.md), GRPO 환경 배선은 [`SPEC_nemo_rl_env_wiring.md`](SPEC_nemo_rl_env_wiring.md) (둘 다 2026-08-13 동결).

## 1. 원천

`/home/work/Datasets/LL_datasets/posttraining/RL/` — Nemotron-Post-Training-v3 RL 26종(62 GB, 블렌드 3종 포함)
+ 자체 `alpha-RL-Identity-Following-v1`(16,510행). 실행 스택은 NeMo RL + NeMo Gym (둘 다 Apache 2.0, 블렌드가 이 스택의 입력 포맷).

### 1.1 훈련 블렌드 3종 (즉시 실행 가능한 레시피 — NeMo Gym 소비 포맷)

행 = 프롬프트 + `agent_ref`(환경/보상) + 검증 메타. 행 수 실측:

| 블렌드 | 파일별 행 수 |
|---|---|
| **Ultra** (재현 대상) | rlvr1 98,424 · rlvr2 99,116 · ifbench 34,649 · rlhf 6,500 · reasoning 5,236 · swe 7,816 · **mopd 85,980** |
| Super (참고) | rlvr1 138,712 · rlvr2 156,278 · rlvr3 107,037 · rlhf 25,171 · swe1 50,661 · swe2 1,444 |
| Nano (참고) | train 93,244 (11 agent 그룹 단일 블렌드) |

구성 상세(agent×dataset×source별 카운트)는 `posttraining/RL/nemotron_blend_recipe.json`.
**주의**: math 일부 행은 DAPO/Skywork 라이선스로 질문·정답이 마스킹 — 각 블렌드 동봉 `fill_placeholders.py`로 복원 필요(원본 HF 데이터셋 자동 다운로드).

### 1.2 RL 환경 데이터셋 26종 분류

| 분류 | 데이터셋 |
|---|---|
| IF 계열 (8) | RL-Instruction-Following-{Structured-Outputs-v2, Citation-Formatting, Free-Form-Formatting, Calendar-v2, MultiTurnChat, Adversarial} + RL-Identity-Following + RL-InverseIFEval |
| Agentic (4) | RL-Agentic-{Function-Calling-Pivot, Conversational-Tool-Use-Pivot, SWE-Pivot(4.8G), Indirect-Prompt-Injection}. **주의(2026-09-22)**: `SWE-Pivot-v1` 은 이름과 달리 단일 스텝 피벗이 아니라 E2E SWE-RL 과제셋(6,436 이슈)이다. 진짜 피벗 행은 `Terminal-Pivot-v1`(31k, 미프로파일)·Conversational-Tool-Use(통과율 동봉). 실사·PivotRL 적용 설계는 [`study/pivotrl_study.md`](study/pivotrl_study.md) §4 |
| Reasoning (5) | RL-Math-v2 · RL-Science-v1 · RL-ARC-AGI-v1 · RL-ReasoningGym-v1 · RLHF-GenRM-v1(5.1G) |
| Safety/기타 (3) | RL-Safety-v1 · RL-QA-Abstention-v1 · RL-litmus-bench-v0.1(평가·모니터링용) |
| 벤치 유래 (4) | RL-SysBench · RL-CFBench · RL-Multichallenge · RL-Multichallenge 계열 |
| 블렌드 (3) | §1.1 |

### 1.3 추가 원천 (2026-09-21 내려받음, 2026-10-08 실측)

| 데이터셋 | 행 | 라이선스 | Gym 형식 | 채점 | 난이도 정보 | Ultra 와의 관계 |
|---|---|---|---|---|---|---|
| `Nemotron-RL-coding-competitive_coding` `opencodereasoning_filtered_25k_train.jsonl` | 23,971 문제 (parquet 16,083 은 부분집합) | 머리말 CC BY-SA 4.0 · 본문 CC BY 4.0 (불일치) | `agent_ref` 추가 → code_gen | unit test (중앙 24개) | 없음 | Ultra code_gen 고유 4,702 문제를 모두 포함 |
| 같은 폴더 `validation.jsonl` | 322 | — | — | — | — | **LiveCodeBench v5 (2024-07~2025-02) — 학습 금지** |
| `Nemotron-RL-knowledge-mcqa` | 617,020 (+ val 68,553) | CC BY 4.0 | `agent_ref` 추가 → mcqa | 정답 문자 (judge 없음) | Qwen3-30B-A3B pass_rate: 1.0 364,853 · (0,1) 129,781 · 없음 122,386 | 겹침 0.2% |
| `Multi-subject-RLVR` (ExamQA) | 573,002 (+ test 6,000) | Apache-2.0 | 변환 필요 (query·label) | 자유형 답 → LLM judge | 없음 | Ultra `reasoning.jsonl` 5,236행의 출처 |
| `Nemotron-RL-instruction_following` | 46,391 | ODC-BY | `agent_ref` 추가 → instruction_following | 프로그램 (48개 id 모두 Gym 등록) | 없음 | Ultra IF 고유 15,166 을 모두 포함 |
| `SWE-Gym` · `SWE-rebench-V2` | 2,438 · 32,079 | MIT · CC BY 4.0 + 저장소별 | 변환 필요 | 컨테이너 | — | 2차 agentic 용 |

기존 원천의 채점 보강 (같은 날 실측): Science-v1 은 ns_tools 90,566 (sandbox) + equivalence judge 60,078 행으로 **전부 judge 가 필요**하다 (물리 80.5%).
ReasoningGym-v1 은 104개 과제 15,000행이고 Ultra 2,089행을 모두 포함한다. Math-v2 는 math_with_judge 3,865 + ns_tools 3,867 행이다.

## 2. 준비된 alpha 블렌드 (2026-08-13)

Ultra RL 블렌드는 **이미 NeMo Gym 실행 형식**(행별 `agent_ref` 라우팅)이고, 레시피 yaml 7종도 트리에 있다
(`examples/nemo_gym/nemotron-3-ultra/`). 수행한 준비:

| 산출물 | 경로 (`/home/work/Datasets/LL_datasets/posttraining/RL/alpha_blends/`) | 내용 |
|---|---|---|
| `ultra_restored/*.jsonl` (7종) | 마스킹 수학 행 복원본 | `fill_placeholders.py`로 rlvr1/rlvr2/mopd 각 6,181행 복원 (DAPO/Skywork 소스) |
| `rlvr1_alpha.jsonl` | 99,113행 | 복원본 + **identity 689행(0.70%)** 주입, 시드 20260813 |
| `rlvr2_alpha.jsonl` | 99,810행 | 복원본 + identity 694행(0.70%), 시드 20260814 |
| `rlvr1_alpha_judgefree.jsonl` (2026-10-07) | 71,730행 | 첫 RLVR 런용 (결정 D1). `rlvr1_alpha` 에서 GenRM·LLM judge·안전 judge·sandbox·nvarc 환경 제외 — `tools/filter_rl_blend.py --preset judge_free`, 통계 `.stats.json`, D1 구조 게이트 OK. identity 행은 GenRM 채점이라 빠진다 |

- 도구: `tools/inject_identity_blend.py`(비율 0.3~1.0% 강제, 시드 고정), `tools/verify_rl_blend.py`(전행 JSON/키/잔여 마스킹 검사 + agent 분포) — 두 블렌드 모두 **구조 검증 통과** (`GATES.md` D1)
- identity 데이터 생성 정본은 Pai `examples/alpha/sdg/identity/README.md` (identity_card 단일 진실 원천).

## 3. 주의

- identity RL 은 **SFT identity 주입 이후에만** 보상 신호가 생긴다 (콜드 정책은 alpha-banana 를 모름).
- `Nemotron-RLHF-GenRM-v1`·`Safety-v1` 은 RL 프롬프트 뱅크가 아니라 RM 학습용이다.
- `Nemotron-RL-ARC-AGI-v1` 은 라이선스 `pending-legal-review` — 블렌드 내 nvarc 행(각 ~2%) 사용 전 법무 확인 필요.

## 4. 미결 (후속 단계로 이관)

- Gym venv 프리페치 (`examples/nemo_gym/prefetch_venvs.py`, 첫 Gym 실행 노드에서)
- Math-v2 복원 → 1차 Code·Math teacher 의 P0-2 로 옮김 (§5.6)
- SWE 196k vs alpha 128k 컨텍스트 상한 결정 (`RL_PLAN.md` §4)
- litmus-bench 모니터링 연결 (활용법 조사 포함)

## 5. 1차 teacher 블렌드안 (2026-10-08, 일부 결정 — `RL_PLAN.md` 결정 18)

teacher 3종(General = chat·IF · STEM · Code·Math)의 블렌드 제안이다. 풀 수치는 Ultra·Super·Nano 블렌드와 §1.3 원천을
프롬프트 해시로 색인해 쟀다 (마스킹 행 제외). alpha 수치는 v1(128K)·E1(64K) 런과 Pai 벤치에서 왔다 (`RLVR_READINESS.md` §5).

### 5.0 결정 현황

| 항목 | 결정 | 일자 |
|---|---|---|
| 시작 ckpt | **agentic 최종 iter2862** — 게이트 P0-1 진행 (`$NRL_ROOT/gates/agentic_iter2862/`) | 사용자 2026-10-08 |
| judge | **Gemma-4-31B-it** (Apache-2.0, bf16 62.6 GB, vLLM 0.25.1 이 `Gemma4ForConditionalGeneration` 지원). 모든 judge 역할에 쓰고, 역할별 적합성은 P0-4 에서 잰다 | 사용자 2026-10-08 |
| P0 | **승인** — 내려받기(수학 복원 원본 · Gemma) · GPU 사전 측정 | 사용자 2026-10-08 |
| 한국어 보강 | **하지 않는다** (시간 없음) | 사용자 2026-10-08 |
| Code·Math teacher 의 MOPD2 재사용 | **보류** — 학습 경향을 보고 정한다 (사용자는 agentic teacher 가 Code·Math 도 개선할 것으로 봄) | 사용자 2026-10-08 |
| 규모 | **B — Ultra teacher 와 같은 샘플 수** (General 640 · STEM 820 · Code·Math 600 스텝, 블렌드 1 에폭). 첫 제안 250 스텝은 작다는 지적 뒤 재제안 (§5.2) | 사용자 2026-10-08 |
| 도구 사용 warm-up teacher | **넣는다** — 1차 teacher 는 4종 (§5.8) | 사용자 2026-10-08 |
| 순서·lr·배치 | **도구 사용 warm-up → STEM → Code·Math → General**, Muon lr 1e-6 · warmup 10 · GBS 1,024 (검증된 RLVR 값) | 사용자 2026-10-08 |
| Code·Math 의 코드 | **B — 미루지 않는다**. 생성 상한 48K 로 능력을 다시 재고, 코드 능력 확인 + 점진적 커리큘럼으로 끌어올린다. **1차 최대 과제** (열린 과제, Code·Math 시작 전까지 해법을 찾는다) | 사용자 2026-10-08 |
| sandbox | **gpu06 `alpha-eval` (DinD)** 에 NeMo-Skills sandbox 를 띄워 ns_tools(python 도구)·Lean 을 쓴다. 구성은 §5.9 | 사용자 2026-10-08 |
| judge 배치 | **A — 롤아웃 노드를 롤아웃 6장 + judge 2장으로 나눈다** (`RL_PLAN.md` 결정 21). 본체 수정은 브랜치 `alpha/judge-split` (`2807ef2be`, 단위 테스트 PASS) — 2노드 스모크 뒤 병합한다. STEM 은 judge 를 포함한 §5.4 비율로 시작한다 | 사용자 2026-10-08 |

### 5.1 원칙

1. **데이터는 충분하고 계산이 모자란다.** 고유 프롬프트는 Ultra 블렌드 33만 행 중 10.2만, Ultra·Super·Nano 합 35.5만이다.
   teacher 1개는 권고 규모(§5.2 B)로도 4.1만~5.3만 프롬프트만 쓴다. 그래서 블렌드는 양이 아니라 **alpha 에게 맞는 난이도**로 고른다.
2. **Ultra 블렌드는 alpha 에게 너무 어렵다.** Ultra 블렌드의 code·math·mcqa·IF 행은 Ultra SFT 가 8회 중 1~7회 맞힌 문제만 담았다.
   alpha 의 보상 > 0 비율은 code_gen 0.7% · math 5.1% 다. 반면 IF 17.7% · mcqa 43.8% · reasoning_gym 41.9% 는 학습 가능한 수준이다.
   Code·Math 는 alpha 로 pass rate 를 먼저 재고 고른다 (P0). STEM·General 은 참조 정책 pass_rate 를 대리 지표로 쓴다.
3. **수학·코드의 실패는 지식보다 종료 실패다.** AIME 응답의 80% 이상이 32K 상한에 닿고, 끝난 응답은 평균 11~12K 다. code 는 64K 에서 88.5% 가 잘린다.
   그래서 Code·Math 는 짧은 상한에서 시작해 늘린다 (24K → 48K).
4. **teacher 블렌드는 자기 도메인만 담는다.** MOPD 에서 teacher 는 자기에게 라우팅된 agent 의 프롬프트에만 logprob 을 준다 (Ultra `mopd.yaml:796-826`).
   MOPD1 은 teacher 학습 프롬프트를 다시 써도 된다 — Ultra MOPD 행의 78.4% 가 앞 단계 프롬프트다. 홀드아웃은 teacher 별 검증셋 512개만 둔다.
5. **벤치 문항은 넣지 않는다.** `coding-competitive_coding/validation.jsonl` 322행은 LiveCodeBench v5 다. P0 에서 LCB·AIME·GPQA 문항과 대조한다.

### 5.2 규모와 시간 (추정)

근거 실측: 학습 3.07만 tok/s/노드 (128K/CP8), 시퀀스당 디코딩 41~65 tok/s, age 2 겹침, GBS 1,024 (64 × 16).
최대 길이가 줄면 CP 를 줄여 학습이 더 빨라질 수 있다 (R5: 64K 4.31만 vs 128K 3.34만 tok/s). 레시피 단계에서 R4 로 확인한다.

| teacher | 생성 상한 | 평균 생성 (근거) | 스텝 시간 (추정) | 하루 프롬프트 |
|---|---|---|---|---|
| STEM | 32K | 7~10K (mcqa 7.5K, GPQA 18.8K) | 6~9분 | 1.0만~1.5만 |
| Code·Math | 24K → 48K | 15~24K (상한 근처) | 9~11분 → 13~16분 | 0.6만~1.0만 |
| General | 16K | 4~5K (IF 3.9K, 구조화 출력 2.3K) | 5~7분 + judge | 1.3만~1.8만 |
| 도구 사용 (§5.8) | 8K | 0.2~1.5K (프롬프트 5~17K 토큰) | 4~5분 | 1.8만~2.3만 |

규모안 (2026-10-08 재제안). 첫 제안 250 스텝은 teacher 3종을 약 1주에 넣으려고 시간 예산에서 역산한 값이다. 데이터나 학습 곡선에서 나온 값이 아니다.

| 안 | teacher 당 스텝 | teacher 당 샘플 | teacher 3종 기간 | 근거 |
|---|---|---|---|---|
| A 첫 제안 | 250~300 | 26만~31만 | 4.1~5.6일 | 시간 예산 역산. Ultra teacher 샘플 수의 30~55% |
| **B 권고** | General 640 · STEM 820 · Code·Math 600 | 61만~84만 | **9.9~13.5일** | Ultra teacher 와 같은 샘플 수 — IFBench 55.4만 + RLHF 10.4만 (General 에 해당) · Reasoning 83.8만 (5,236 × 10 에폭) |
| C 가용 풀 전체 1 에폭 | General 2,590 · STEM 2,870 · Code·Math 1,090 | 110만~290만 | 약 29~41일 | 비권고 — 풀의 상당수가 alpha 에게 너무 쉽거나 어렵고, 같은 원천이 겹친다 |

- B 는 블렌드 크기를 스텝 수에 맞춰 1 에폭으로 만든다: General 41,000 · STEM 52,500 프롬프트. Code·Math 는 P0-3 이 고른 풀을 1~2 에폭 돈다.
- 100 스텝마다 HF 반출과 검증 보상을 보고, 평탄해지면 일찍 멈춘다. 스텝 시간은 첫 teacher 의 20 스텝 실측으로 다시 계산한다.
- P0 (0.5~1일) 와 운영 여유 (+20~30%) 는 따로 더한다. Ultra 형태 전체 일정은 42~44일이었다.

### 5.3 General (chat·IF) — 비율 (프롬프트 수는 §5.2 규모안을 따른다)

| 구성 | agent (채점) | 비중 | 풀 (고유) | 근거 |
|---|---|---|---|---|
| 지시 따르기 | instruction_following (IFEval 검사기) | 35% | 46,391 이상 (§1.3 + Super·Nano) | alpha 보상 > 0 17.7%, 잘림 1.4% |
| 구조화 출력·형식 | structured_outputs v1 · v3 (Ultra pass_rate > 0 만) · citation · freeform (스키마·정규식) | 15% | 9,983 · 3,369 · 1,601 · 1,025 | v1 보상 > 0 37.5%. v3 는 Ultra 도 69% 를 못 풀었다 |
| 대화 품질 (다국어·한국어) | genrm · genrm_reasoning_off (GenRM 쌍대 비교) | 25% | 63,867 · 28,402 | Ultra rlhf teacher. 한국어 프롬프트는 3~10% |
| 복합·다중턴 IF | multichallenge · inverse_if (LLM judge) | 10% | 1,758 · 936 | Ultra ifbench teacher |
| 환각 억제 | abstention (LLM judge) | 8% | 2,502 | 〃 (Ultra 는 43%) |
| 안전 | jailbreak 4종 · jailbreak_detection · over_refusal (safety judge) | 6% | 1,118 · 4,365 · 561 | |
| identity | genrm + alpha principle | 1% | 16,510 | 상한 0.3~1.0% (§3) |

judge 없이 되는 부분은 50% 다. chat 품질 신호는 judge 없이는 없다. calendar 는 모든 런에서 보상 0 이라 원인을 찾을 때까지 뺀다.
도구 계열(tau·swe_pivot·toolcall_schema·IPI·workplace)은 2차 agentic 으로 넘긴다.

### 5.4 STEM — 비율

| 구성 | agent (채점) | 비중 | 풀 (고유) | 근거 |
|---|---|---|---|---|
| 과학·지식 객관식 | mcqa (정답 문자) | 55% | stem_mcqa 37,742 + knowledge-mcqa 129,781 (Qwen3-30B-A3B (0,1)) | alpha 보상 > 0 43.8%, 잘림 4% |
| 과학·지식 주관식 | equivalence_llm_judge (LLM judge) — ExamQA · Science-v1 | 25% | 573,002 · 60,078 | Ultra reasoning teacher 는 이것 100% |
| 논리·퍼즐 | reasoning_gym (104개 과제 검사기) | 15% | 15,000 (+ Super 6,882, 겹침 미측정) | alpha 보상 > 0 41.9% |
| 화학 | rdkit_chemistry (수치 대조) | 5% | 1,167 | |

nvarc(ARC-AGI)는 라이선스 확인 전까지 뺀다. Science-v1 은 CC BY-SA 4.0 이다.
knowledge-mcqa 의 Qwen3-30B-A3B pass_rate 1.0 행(364,853)도 alpha 에게는 (0,1) 일 수 있다 — P0 표본으로 확인한다.

### 5.5 Code·Math — 비율 (24K 상한으로 시작해 48K 로 올린다)

| 구성 | agent (채점) | 비중 | 풀 | 근거 |
|---|---|---|---|---|
| 경쟁 코딩 | code_gen (unit test, Gym CPU 실행) | 50% | OCR-25K 23,971 + Super 9,626 + Nano 10,688 (Nano SFT pass_rate 있음) | alpha 보상 > 0 0.7%, 64K 잘림 88.5% |
| 수학 | math_with_judge (math-verify, judge 끔) | 50% | Ultra 1,591 + 복원 대상 Nano 22,056 · Super 약 12.3K · Math-v2 3,984 (DAPO-17k·Skywork-OR1) | alpha 보상 > 0 5.1%, 64K 잘림 45% |

- **P0-3 스모크 (2026-10-08, iter2862, 생성 상한 24K, 측정 목록 앞 16문제 × 2회, temperature 1.0 · top_p 1.0)** — 본 측정 보류, 사용자 결정 대기:

  | 모드 | 수학 끝냄 | 수학 정답 | 코드 끝냄 | 코드 정답 | 끝낸 응답 평균 |
  |---|---|---|---|---|---|
  | 기본 | 5/16 | 4/16 | 1/16 | 0/16 | 수학 6.8K · 코드 7.5K 토큰 |
  | efficient 마커 | 10/16 | 6/16 | 0/16 | 0/16 | 수학 5.6K 토큰 |

  수학은 끝내면 대부분 맞힌다. 코드는 24K 안에 거의 끝내지 못한다 (이전 64K·128K 런도 잘림 88~90%, 보상 > 0 0.7%). 산출물 `$NRL_ROOT/p0_measure/smoke{,_eff}/`.
- **코드 48K 스모크 (2026-10-08, 사용자 지시 — 결정 20)**: 48문제 = Nano SFT 가 8번 중 7번 맞힌 32 + 참조 없음 16, 기본 모드 4회 · efficient 2회.

  | 모드 · 문제 | 끝냄 | 정답 | 신호 있는 문제 (1회 이상 정답) | 정답 응답 길이 중앙 |
  |---|---|---|---|---|
  | 기본 · Nano-easy 32 | 16% | 6.2% | 6/32 (모두 일부만 정답) | 11.1K |
  | 기본 · 참조 없음 16 | 31% | 4.7% | 2/16 | 16.7K |
  | efficient · Nano-easy 32 | 16% | 4.7% | 2/32 | 1.4K |
  | efficient · 참조 없음 16 | 34% | 3.1% | 1/16 | 8.5K |

  - 해석: 잘린 응답의 끝 2만 자에는 반복 루프가 없다 (107개 중 1개만 반복 비율 > 0.3). 끝나지 않는 탐색(과잉 사고)이다. 틀린 채 끝낸 응답(중앙 26K)이 정답(11K)보다 길다.
  - 비교: Nano 3 RL 은 최대 49K (`examples/nemo_gym/grpo_nanov3.yaml`)에서 이 문제들을 8번 중 7번 풀었다. alpha 는 같은 길이에서 6% 다.
  - efficient 모드는 코드에 효과가 없다. 산출물 `$NRL_ROOT/p0_measure/code48k{,_eff}/`.
- 선별: P0 에서 alpha 로 문제당 4회(24K 상한) 풀어 1~3회 맞힌 문제를 남기고, 0회 문제를 일부 더한다. 고유 10K 이상을 목표로 하고 2에폭을 넘기지 않는다.
- 길이: 24K 상한으로 시작한다. 잘림이 10% 미만이 되면 48K 로 올린다. effort 마커 행은 Ultra 비율(약 3.5%)을 유지한다.
- ns_tools(python sandbox)·lean(Lean sandbox)은 2차 agentic(도구) teacher 로 넘긴다.
- 사전 측정에서 (0,1) 수학 문제가 5K 미만이면 더 쉬운 수학 원천(MATH 계열)을 추가하자고 요청한다.

### 5.6 착수 전 작업 (P0)과 순서

| # | 작업 | 자원 | 비고 |
|---|---|---|---|
| P0-1 | 시작 ckpt 확정 + 게이트 M1·M2·M3 → R1 | 1노드 반나절 | **완료 (2026-10-08)**: iter2862 M1 14,181/14,181 · M2 cos ≥ 0.99980 · M3 1.0228 · M4 cos ≥ 0.99995 · R1 0.0014/0.0013/0.0013 — 모두 PASS (`GATES.md`) |
| P0-2 | 데이터 변환: 수학 마스킹 복원 (HF 내려받기) · 신규 셋 `agent_ref` 부여 · 벤치 대조 · D1 구조 게이트 | CPU | **Code·Math 완료 (2026-10-08)**: 복원 `alpha_blends/{nano,super,math_v2}_restored/` (남은 마스킹 0). 후보 풀 `alpha_blends/teacher_pool/` — 코드 23,971 · 수학 29,406 문제 (원천 99,876 · 59,844 행에서 중복 제거, 벤치 오염 19 · 답 충돌 40 제외), 측정 목록 각 12,000, 6개 파일 D1 통과. 도구 `tools/teacher_pool_{index,build}.py`. 대조한 벤치 AIME25 · HMMT Feb25 · GPQA-D · MMLU-Pro · LCB v5, 못 한 벤치 MATH-500 · AIME24 · HMMT Nov25 · LCB v6 이후 (로컬에 없음). **도구 블렌드 완료**: `teacher_blends/teacher_tool_v1.jsonl` 25,600 + 검증 512 (`tools/build_tool_blend.py`). **STEM·General 의 judge 불필요 부분 완료 (2026-10-08)**: `teacher_stem_judgefree_v1.jsonl` 36,750 (객관식 26,613 · 퍼즐 7,875 · rdkit 2,262 = 고유 1,131 × 2) · `teacher_general_judgefree_v1.jsonl` 20,500 (지시 따르기 14,350 · SO v1 2,768 · SO v3 1,845 · citation 922 · freeform 615) · 각 검증 512 · D1 OK · 오염 제외 STEM 1 · General 4 (IFEval 3 포함). Science-v1 python 도구 행 중 math-verify 채점 가능한 답은 1,409 개뿐이다 (90,566 중 서술형 88,280) — `teacher_stem_scitir_candidates_v1.jsonl`. knowledge-mcqa 변환 코드는 있고 Qwen (0,1) 247,287 문제가 확장 풀로 남았다. 도구 스크립트는 scratchpad (리포 반영은 최종 블렌드 때). 남은 것: judge 부분 (STEM 주관식 · General 50%), effort 마커 재삽입 (약 3.5%), 코드 테스트 수 상한 (p90 106개), General 레시피에 SO v3 · citation · freeform 서버 추가 |
| P0-3 | Code·Math alpha 사전 측정 (Gym `nemo-gym-reward-profiling` 경로, 후보 약 2.4만 × 4회) | GPU 16장 반나절 | 후보는 Nano pass_rate 로 먼저 줄인다 |
| P0-4 | judge 검증 (Gemma 62.6 GB 내려받기 완료, `/home/work/vidsearch/models/gemma-4-31B-it`): Gemma-4-31B-it (사용자 결정) 를 롤아웃 노드 GPU 2장에 올리고 역할별로 잰다 — GenRM-v1 정답 쌍 일치율 · Safety-v1 일치율 · equivalence 판정 · 처리량 | GPU 2장 몇 시간 | Ultra 의 GenRM 은 235B(롤아웃 노드 8장 전부)·550B(불가)라 쓸 수 없다. 안전 판정이 기준에 못 미치면 safety 4B 를 따로 내려받는다 |
| P0-5 | calendar 보상 0 원인 | CPU | **완료 (2026-10-08)**: 모델 능력·어려운 표본이다. 채점기는 정상 — 솔버로 만든 정답 836/836 이 보상 1. 직전 달력 복사로 통과하는 행 111/910 (12.2%), 지금까지 본 calendar 프롬프트는 4개뿐. P0-3 에 후보 1~2K 를 넣고 0 < 정답률 < 1 인 행이 10% 이상이면 General 에 1~2% 로 넣는다 |

순서: P0 → STEM → Code·Math → General. STEM 은 보상 신호가 건강해 teacher 파이프라인을 먼저 검증한다.
도구 사용 teacher (§5.8, 결정 대기) 는 judge·사전 측정이 필요 없어 어느 자리에도 넣을 수 있다.
Code·Math 는 P0-3 결과가 필요하다. General 은 judge 가 가장 많이 필요하다.

### 5.7 MOPD1 라우팅 (예정, MOPD1 설계 때 확정)

| teacher | agent |
|---|---|
| General | IF · structured_outputs v1·v3 · citation · freeform · calendar · multichallenge · inverse_if · abstention · genrm 2종 · jailbreak 계열 · identity · 매핑 없는 agent |
| STEM | mcqa · equivalence_llm_judge · reasoning_gym · rdkit_chemistry |
| Code·Math | code_gen · math_with_judge |
| 도구 사용 (§5.8) | tau · swe_pivot · toolcall_schema · workplace — warm-up teacher, 만들지 않으면 SFT ckpt |

### 5.8 도구 사용 warm-up teacher (사용자 결정 2026-10-08 — 넣는다)

MOPD1 이 세 도메인만 증류하면, 2차 agentic teacher 의 출발점인 MOPD1 에서 도구 사용 능력이 약해질 수 있다.
권고는 1차에 네 번째 teacher 로 넣는 것이다.

| 항목 | 내용 |
|---|---|
| 환경 | judge 가 필요 없는 단일 스텝 도구 호출 — tau 피벗 · swe_pivot · toolcall_schema · workplace_assistant. G3·E1 에서 검증한 경로다 |
| 풀 (고유) | tau 99,105 · toolcall_schema 8,917 · swe_pivot 3,752 (+ SWE-Pivot-v1 50,661) · workplace 1,047 · Function-Calling-Pivot 9,620 |
| alpha 신호 | 보상 > 0: tau 37.2% · swe_pivot 34.9% · toolcall_schema 45.2% (v1) — 학습 가능한 수준이다 |
| 비용 | 생성은 짧다 (145~374 토큰). 학습 시간은 긴 프롬프트가 정한다 (샘플당 tau 5.3K · swe_pivot 17.3K 토큰). 스텝 4~5분, 400 스텝 1.1~1.4일 (추정) |
| MOPD1 에서 | 도구 프롬프트를 MOPD1 에 넣고 이 teacher 에 라우팅한다. 망각을 막고, 단일 스텝 도구 호출을 올린 상태로 2차 agentic teacher 를 시작한다 |
| 대안 (비용 0) | teacher 를 만들지 않고 MOPD1 의 도구 프롬프트를 SFT ckpt 에 라우팅한다. 망각은 막지만 개선은 없다 |
| 근거 | Ultra 도 Student-RLVR 블렌드의 38% 가 단일 스텝 도구 호출이고, MOPD 에서 도구 프롬프트를 general teacher 에 라우팅했다. 원래 계획의 PivotRL (agentic SFT 직후) 과 같은 역할이다 |

### 5.9 sandbox — gpu06 `alpha-eval` (사용자 결정 2026-10-08, **구성 완료 2026-10-08**)

**가동 중**: 컨테이너 `alpha-nemo-skills-sandbox` (이미지 `alpha-nemo-skills-sandbox:da85a88`, 작업 프로세스 32, `NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1`, `--restart unless-stopped`) ·
main1 터널 `$NRL_ROOT/sandbox/tunnel.sh` (끊기면 다시 연다, 로그 `tunnel.log`) → `10.0.37.4:6000`. 측정: Python p50 0.14 s · 동시 32 에서 초당 117건 (main1·sub1 동일) ·
Lean 4 맞는 증명 통과·틀린 증명 실패. 빌드 함정은 `KNOWN_ISSUES.md` 2026-10-08 sandbox 항목.

Backend.AI 노드(main1·sub1)는 docker 를 못 띄운다. Pai 가 SWE-bench·Terminal-Bench 에 쓰는 외부 docker 호스트 gpu06 의 DinD 컨테이너
`alpha-eval` 을 sandbox 로 쓴다. 접속·복구 정본은 Pai `examples/alpha/docs/EVAL_DOCKER_NODE.md` 다.

| 항목 | 내용 |
|---|---|
| 상태 (2026-10-08 확인) | ssh 2홉 접속 정상 · dockerd 29.1.3 overlayfs · `/var/lib/docker` 여유 4.1 TB · 64코어 · 가용 RAM 441 GB · 공용 서버 |
| 띄울 것 | NeMo-Skills sandbox 이미지 — Gym `ns_tools` 요구 커밋 `da85a881` 의 `dockerfiles/Dockerfile.sandbox` (Python·pypy·Lean 4 v4.12 + Mathlib). 공개 이미지가 없어 빌드한다 |
| 연결 | gpu06 앞단 방화벽은 선별 포트만 연다. main1 에서 `ssh -L 0.0.0.0:6000:localhost:6000 alpha-eval` 로 터널을 열고, Gym 서버에 `NEMO_SKILLS_SANDBOX_HOST=10.0.37.4`·`PORT=6000` 을 넘긴다 |
| 쓰는 환경 | ns_tools (python 도구 수학: Ultra 1,569 + Math-v2 3,867, 과학: Science-v1 90,566) · math_formal_lean (Ultra·Super 2,288) |
| 검증 | 터널 너머 실행 지연·동시 처리량 · ns_tools·Lean 각 소량을 P0-3 사전 측정에 넣어 끝까지 돌린다 |
| 미정 | code_gen 은 지금도 sandbox 없이 Gym CPU 에서 모델 코드를 실행한다 (H4). sandbox 로 옮길지는 따로 본다 |

ns_tools·Lean 이 들어오면 블렌드 비율을 다시 낸다 (Code·Math 에 python 도구 수학·Lean, STEM 에 과학 python 도구 문제). 사용자 승인 뒤 반영한다.
