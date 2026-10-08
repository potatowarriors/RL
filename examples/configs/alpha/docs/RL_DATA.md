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

## 5. 1차 teacher 블렌드안 (2026-10-08, 승인 대기 — `RL_PLAN.md` 결정 18)

teacher 3종(General = chat·IF · STEM · Code·Math)의 블렌드 제안이다. 풀 수치는 Ultra·Super·Nano 블렌드와 §1.3 원천을
프롬프트 해시로 색인해 쟀다 (마스킹 행 제외). alpha 수치는 v1(128K)·E1(64K) 런과 Pai 벤치에서 왔다 (`RLVR_READINESS.md` §5).

### 5.1 원칙

1. **데이터는 충분하고 계산이 모자란다.** 고유 프롬프트는 Ultra 블렌드 33만 행 중 10.2만, Ultra·Super·Nano 합 35.5만이다.
   teacher 1개는 250 스텝 × 64 = 16,000 프롬프트만 쓴다 (§5.2). 그래서 블렌드는 양이 아니라 **alpha 에게 맞는 난이도**로 고른다.
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

| teacher | 생성 상한 | 평균 생성 (근거) | 스텝 | 스텝 수 | 기간 |
|---|---|---|---|---|---|
| STEM | 32K | 7~10K (mcqa 7.5K, GPQA 18.8K) | 6~9분 | 250 | 1.0~1.6일 |
| Code·Math | 24K → 48K | 15~24K (상한 근처) | 9~11분 → 13~16분 | 200 + 100 | 2.2~2.7일 |
| General | 16K | 4~5K (IF 3.9K, 구조화 출력 2.3K) | 5~7분 + judge | 250 | 0.9~1.3일 |
| P0 (§5.6) | — | — | — | — | 0.5~1일 |
| **합** | | | | | **4.6~6.6일 무중단** (운영 여유 +20~30%). Ultra 형태는 42~44일 |

검증 보상이 계속 오르면 teacher 마다 400 스텝까지 늘린다 (+0.6~1.5일).

### 5.3 General (chat·IF) — 16,000 프롬프트

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

### 5.4 STEM — 16,000 프롬프트

| 구성 | agent (채점) | 비중 | 풀 (고유) | 근거 |
|---|---|---|---|---|
| 과학·지식 객관식 | mcqa (정답 문자) | 55% | stem_mcqa 37,742 + knowledge-mcqa 129,781 (Qwen3-30B-A3B (0,1)) | alpha 보상 > 0 43.8%, 잘림 4% |
| 과학·지식 주관식 | equivalence_llm_judge (LLM judge) — ExamQA · Science-v1 | 25% | 573,002 · 60,078 | Ultra reasoning teacher 는 이것 100% |
| 논리·퍼즐 | reasoning_gym (104개 과제 검사기) | 15% | 15,000 (+ Super 6,882, 겹침 미측정) | alpha 보상 > 0 41.9% |
| 화학 | rdkit_chemistry (수치 대조) | 5% | 1,167 | |

nvarc(ARC-AGI)는 라이선스 확인 전까지 뺀다. Science-v1 은 CC BY-SA 4.0 이다.
knowledge-mcqa 의 Qwen3-30B-A3B pass_rate 1.0 행(364,853)도 alpha 에게는 (0,1) 일 수 있다 — P0 표본으로 확인한다.

### 5.5 Code·Math — 19,200 프롬프트 (24K 200 스텝 → 48K 100 스텝)

| 구성 | agent (채점) | 비중 | 풀 | 근거 |
|---|---|---|---|---|
| 경쟁 코딩 | code_gen (unit test, Gym CPU 실행) | 50% | OCR-25K 23,971 + Super 9,626 + Nano 10,688 (Nano SFT pass_rate 있음) | alpha 보상 > 0 0.7%, 64K 잘림 88.5% |
| 수학 | math_with_judge (math-verify, judge 끔) | 50% | Ultra 1,591 + 복원 대상 Nano 22,056 · Super 약 12.3K · Math-v2 3,984 (DAPO-17k·Skywork-OR1) | alpha 보상 > 0 5.1%, 64K 잘림 45% |

- 선별: P0 에서 alpha 로 문제당 4회(24K 상한) 풀어 1~3회 맞힌 문제를 남기고, 0회 문제를 일부 더한다. 고유 10K 이상을 목표로 하고 2에폭을 넘기지 않는다.
- 길이: 24K 상한으로 시작한다. 잘림이 10% 미만이 되면 48K 로 올린다. effort 마커 행은 Ultra 비율(약 3.5%)을 유지한다.
- ns_tools(python sandbox)·lean(Lean sandbox)은 2차 agentic(도구) teacher 로 넘긴다.
- 사전 측정에서 (0,1) 수학 문제가 5K 미만이면 더 쉬운 수학 원천(MATH 계열)을 추가하자고 요청한다.

### 5.6 착수 전 작업 (P0)과 순서

| # | 작업 | 자원 | 비고 |
|---|---|---|---|
| P0-1 | 시작 ckpt 확정 + 게이트 M1·M2·M3 → R1 | 1노드 반나절 | agentic iter2862 이면 필요. 게이트 완료는 iter2400 뿐 |
| P0-2 | 데이터 변환: 수학 마스킹 복원 (HF 내려받기) · 신규 셋 `agent_ref` 부여 · 벤치 대조 · D1 구조 게이트 | CPU | |
| P0-3 | Code·Math alpha 사전 측정 (Gym `nemo-gym-reward-profiling` 경로, 후보 약 2.4만 × 4회) | GPU 16장 반나절 | 후보는 Nano pass_rate 로 먼저 줄인다 |
| P0-4 | judge 선정: 후보를 롤아웃 노드 GPU 2장에 올리고 GenRM-v1 정답 쌍·Safety-v1 로 일치율을 잰다 | GPU 2장 몇 시간 | Ultra 의 GenRM 은 235B(롤아웃 노드 8장 전부)·550B(불가)라 쓸 수 없다. 로컬 후보 gpt-oss-120b 등. safety 4B 는 내려받기 필요 |
| P0-5 | calendar 보상 0 원인 | CPU | General 블렌드 포함 여부 |

순서: P0 → STEM → Code·Math → General. STEM 은 보상 신호가 건강해 teacher 파이프라인을 먼저 검증한다.
Code·Math 는 P0-3 결과가 필요하다. General 은 judge 가 가장 많이 필요하다.

### 5.7 MOPD1 라우팅 (예정, MOPD1 설계 때 확정)

| teacher | agent |
|---|---|
| General | IF · structured_outputs v1·v3 · citation · freeform · calendar · multichallenge · inverse_if · abstention · genrm 2종 · jailbreak 계열 · identity · 매핑 없는 agent |
| STEM | mcqa · equivalence_llm_judge · reasoning_gym · rdkit_chemistry |
| Code·Math | code_gen · math_with_judge |
| 미정 | 도구 계열 프롬프트를 MOPD1 에 넣어 SFT ckpt 를 기준 teacher 로 둘지 (agentic 능력 망각 방지) |
