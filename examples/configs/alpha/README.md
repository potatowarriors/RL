# alpha post-training recipes

alpha_v2 (15.08B hybrid GatedDeltaNet+MoE, `model_type: "alpha"`) post-training on NeMo-RL.
Nemotron-3-Ultra 레시피(`ultra-v3` 브랜치의 `examples/configs/ultra/`)를 참조 원형으로 삼는다.

## 구성 (계획)

| 파일 | 단계 | 상태 |
|---|---|---|
| `smoke_grpo_qwen3next.yaml` | 마일스톤 1: stock Qwen3-Next 소형/축소 설정으로 mcore+vLLM 경로 검증 | TODO |
| `student_rlvr1.yaml`, `student_rlvr2.yaml` | RLVR (GRPO, SFT 체크포인트에서 시작) | TODO |
| `ifbench_teacher.yaml` 등 | 전문 teacher RL (2~3개로 축소 예정) | TODO |
| `mopd.yaml` | 멀티 teacher on-policy distillation | TODO |

## 전제

- 학습 백엔드: **Megatron-Core** (Megatron-Bridge에 alpha bridge 필요 — 우리 fork에서 작업)
- 롤아웃: **vLLM** (custom fork에 `AlphaForCausalLM` 등록 — `tools/build-custom-vllm.sh` 사용)
- 모델 입력: Pai-Megatron-Patch의 `evaluate.sh` 경로로 변환된 HF 체크포인트 (`hfmodel_*`)
- SFT는 NeMo-RL이 아닌 기존 Pai-Megatron 스택에서 수행 (프로젝트 결정, 2026-08-13)
- 클러스터: Backend.AI (Slurm 없음) — `ray start` 수동 기동, 환경은 워크스페이스 루트의
  `setup_nemo_rl_env.sh` / `nemo_rl_env.sh` 참조 (드라이버 535 → CUDA 13 forward-compat 필수)

## 토폴로지 (2× H100×8, 노드 간 IB 없음 ~9.1Gbit/s)

- node0: 학습 + 롤아웃 colocate (노드 내 weight sync)
- node1: frozen 서빙 전용 (teacher / GenRM / 평가) — 노드 간에는 시퀀스·로짓만 이동
