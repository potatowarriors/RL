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
- Math-v2 복원 (reasoning teacher 필요)
- SWE 196k vs alpha 128k 컨텍스트 상한 결정 (`RL_PLAN.md` §4)
- litmus-bench 모니터링 연결 (활용법 조사 포함)
