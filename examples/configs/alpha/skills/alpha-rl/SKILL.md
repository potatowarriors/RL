---
name: alpha-rl
description: "alpha_v2 (15.08B GDN+Attention+MoE) RL 작업의 문서 지도. 작업 유형별로 먼저 읽을 alpha 문서와 함께 볼 upstream NeMo-RL 문서·서브모듈(Megatron-Bridge·Gym) skill 을 알려준다. 이 fork(alpha/post-train)에서 하는 alpha 작업이면 upstream skill 보다 먼저 호출한다. Do NOT use for: upstream NVIDIA-NeMo/RL 기여 작업."
when_to_use: alpha 레시피 작성·수정; GRPO·RLVR 런 실행·감시·분석; 게이트 재실행; AlphaBridge·vLLM 플러그인·Gym 플러그인 수정; upstream 수정 이식(backport); KL·logprob 어긋남 진단; 메모리·처리량 튜닝; RL 데이터·블렌드; alpha 문서·STATUS 갱신; 'alpha', 'alpha_v2', 'RLVR', 'student_rlvr1', 'hfmodel_'.
---

# alpha RL 문서 지도

이 skill 은 **지도**다. 지침은 복사하지 않는다 — 작업 규칙과 upstream skill 적용표는 `.claude/rules/alpha.md`,
모델 불변량·레시피 규약·함정 표는 `examples/configs/alpha/CLAUDE.md` 가 정본이다.
아래 경로는 리포 루트 기준이고, `alpha/` 는 `examples/configs/alpha/` 를 줄인 것이다.

## 먼저

1. `alpha/docs/STATUS.md` 를 읽는다 — 진행 중 런·열린 결정. 다른 세션이 같은 워킹트리에서 일한다.
2. 아래 표에서 작업 행을 찾아 "alpha 정본"을 읽는다.
3. 배경이 필요하면 "upstream" 열의 주제를 `alpha/docs/README.md` "upstream 참조" 표에서 찾아 읽는다.
   upstream 과 alpha 문서가 다르면 alpha 쪽이 이 모델에서 실측한 값이다.

## 작업 유형별

| 작업 | alpha 정본 | upstream ("upstream 참조" 표의 주제) |
|---|---|---|
| 레시피 작성·수정 | `alpha/CLAUDE.md` 레시피 규약 · `alpha/README.md` 레시피 표 · 실행 전 `alpha/tools/check_alpha_recipe.py` ERROR 0 | R3 · Muon · packing·CP · 환경변수 · Ultra 원형 |
| 런 실행·감시 | `alpha/CLAUDE.md` Quick Commands · `alpha/docs/SETUP.md` · `alpha/docs/RLVR_READINESS.md` §5 | async GRPO·refit |
| 게이트 재실행 (새 ckpt·브리지·플러그인 변경) | `alpha/docs/GATES.md` 재실행 조건 | 새 아키텍처·logprob 일관성 |
| KL·logprob 어긋남 진단 | `alpha/docs/GATES.md` R1·G2·G3 · `alpha/tools/analyze_*_logprob_gap.py` · `alpha/docs/KNOWN_ISSUES.md` | 새 아키텍처·logprob 일관성 · R3 |
| 학습 메모리·처리량 | `alpha/docs/GATES.md` R4·R5 · `alpha/docs/RLVR_READINESS.md` §5 | 학습 메모리·장문맥 |
| AlphaBridge 수정 | `.claude/rules/alpha-submodules.md` · `alpha/docs/SPEC_megatron_bridge_surface.md` · `alpha/docs/SPEC_pai_alpha_conversion.md` | 새 아키텍처·logprob 일관성 |
| vLLM 플러그인 수정 | `alpha/vllm_alpha_plugin/` · `.claude/rules/alpha-submodules.md` vLLM 항 | vLLM 커스텀 빌드 |
| Gym 환경·플러그인 | `alpha/docs/SPEC_nemo_rl_env_wiring.md` · `alpha/gym_plugins/` · `alpha/docs/GATES.md` G3 | Gym 연동 |
| RL 데이터·블렌드 | `alpha/docs/RL_DATA.md` · `alpha/docs/SPEC_rl_dataset_inventory.md` | PivotRL 데이터 |
| 체크포인트 반출·재개 | `alpha/tools/export_rl_hf.sh` · `alpha/docs/GATES.md` G5~G7 | HF 반출 · async GRPO·refit |
| upstream 수정 이식 | `.claude/rules/alpha.md` upstream 경계 (문서까지 원형, 별도 커밋, REBASE NOTE) | upstream skill `config-conventions`·`testing`·`error-handling` |
| 단위 테스트 | `alpha/CLAUDE.md` 함정 표 (pytest 가 운영 Ray 클러스터에 붙음) | upstream skill `testing` |
| 단계·설계 결정 | `alpha/docs/RL_PLAN.md` · `alpha/docs/RLVR_READINESS.md` | Ultra 원형 |
| 문서 갱신 | `.claude/rules/alpha.md` 메모리·문서 규칙 · `alpha/docs/README.md` 색인 | — (`docs/` 에 alpha 내용을 쓰지 않는다) |

## 서브모듈 skill

Megatron-Bridge 와 Gym 의 skill 은 각 서브모듈 안의 `.claude/skills/` 에 있다. 그 서브모듈 파일을 Read·Edit 하기 전에는
skill 목록에 나타나지 않는다. 필요하면 `alpha/docs/README.md` "upstream 참조" 표의 경로에서 `SKILL.md` 를 직접 읽는다.
이 skill 들은 각 서브모듈 기여용으로 쓰였다 — 기술 내용만 가져오고, 커밋·문서 규칙은 `.claude/rules/alpha.md` 를 따른다.
