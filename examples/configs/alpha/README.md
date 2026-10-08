# alpha post-training recipes

alpha_v2 (15.08B hybrid GatedDeltaNet+MoE, `model_type: "alpha"`) post-training on NeMo-RL.
Nemotron-3-Ultra 레시피(이 리포 `examples/nemo_gym/nemotron-3-ultra/`, 구 `ultra-v3` 브랜치 `examples/configs/ultra/`)를 참조 원형으로 삼는다.

| 보려는 것 | 문서 |
|---|---|
| 지금 상태 · 다음 할 일 · 열린 결정 | [`docs/STATUS.md`](docs/STATUS.md) |
| 지침 (모델 불변량 · 레시피 규약 · 함정 표) | [`CLAUDE.md`](CLAUDE.md) |
| 문서 색인 | [`docs/README.md`](docs/README.md) |
| 검증 게이트 | [`docs/GATES.md`](docs/GATES.md) |

## 레시피

| 파일 | 단계 | 비고 |
|---|---|---|
| `grpo_alpha_smoke.yaml` | 8-GPU 1노드 GRPO 드라이런 + KL 게이트(R1). **alpha RL 기본값 포함** — vLLM GDN 재귀 상태 fp32(`generation.vllm_kwargs.mamba_ssm_cache_dtype`) + R3(`router_replay`) + 멀티턴 경계 `<\|im_end\|>`(`generation.vllm_cfg.turn_end_token_id: 3`). 이후 alpha 레시피는 이 파일을 상속 | 게이트 결과 `docs/GATES.md` R1·R3 |
| `grpo_alpha_smoke_muon.yaml` | 위 + Muon(`dist_muon`, SFT 동역학 정렬: nesterov · extra_scale 0.2 · beta2 0.95 · 필수 off 4개) | 게이트 결과 R1·R2 |
| `student_rlvr1_alpha.yaml` | RLVR 1단계 골격 (GRPO + Gym, 최대 128K, 2노드). Ultra `student_rlvr1` + alpha 기본값 + 위험 가드 | 결정 D1~D7·16·17 반영 (생성 상한 64K·loss 안 마스킹). 게이트 G1~G3·G5~G8 PASS, E1 레버(chunked prefill·age 2) 채택. 1단계 학습 전 차단 과제(GenRM·judge·sandbox·code_gen 긴 응답)와 rollout 속도 작업 진행 중 (`docs/STATUS.md`) |
| `student_rlvr2.yaml` | RLVR 2단계 | 미작성 |
| `ifbench_teacher.yaml` 등 | 전문 teacher RL (2~3개로 축소 예정) | 미작성 |
| `mopd.yaml` | 멀티 teacher on-policy distillation | 미작성 |

## 디렉토리

| 경로 | 내용 |
|---|---|
| `vllm_alpha_plugin/` | vLLM 플러그인 패키지 (커밋 `599b58ac3`, pyproject 의 vllm extra + uv source 로 연결). `vllm.general_plugins` 엔트리포인트로 stock 0.25.1 휠에 `AlphaForCausalLM` 등록. qwen3_next 서브클래스 + 표준 RMSNorm 전면 교체(융합 QK-norm 커널은 zero-centered +1.0 하드코딩이라 비활성화) + FusedMoE DSV3 인자(`apply_routed_scale_to_output=False` 의도적) |
| `runs/launch.sh` | RLVR 런 실행기 (2026-10-08): `launch.sh <campaign> <tag> [override]`. 산출물은 리포 워크스페이스의 `results/alpha/<campaign>/` (gitignored: 로그 디렉토리·드라이버 로그·`runs.log`·`ckpt/`), wandb 는 RL 전용 프로젝트 `alpha-rl` (group = campaign). 기동 전 GPU 점유 검사(`runs/gpu_check.py`)·남은 Gym 서버 정리(`runs/gym_cleanup.py`). 이전 런(v1·E0·E1·v2 구간 A)은 `$NRL_ROOT/runs/` 에 남아 있다. 레시피는 `student_rlvr1_alpha.yaml` 고정이다 |
| `tools/verify_*.py` | 검증 게이트 (`docs/GATES.md` M1·M2·M4·R2·R3·D1·G1·G7). `verify_chat_render_parity.py` 는 CPU 전용 렌더 패리티(R3), `verify_packing_isolation.py` 는 packing 상태 누출(G1), `verify_optimizer_resume.py` 는 재개 전후 옵티마이저 상태 연속성(G7, CPU) |
| `tools/check_alpha_recipe.py` | 레시피 실행 전 검사 (CPU) — R3+router fusion·파서·Ultra token id·GBS 나눗셈·CP 패딩·eos 등 |
| `tools/engine_parity_*.py` | SFT 엔진(Pai)↔RL 엔진(NeMo-RL) forward·gradient 동등성 (M5). `_pai` 는 Pai 환경, `_nemorl` 은 NeMo-RL 워커 venv, `_hf` 는 제3 기준, `_compare` 가 판정 |
| `tools/measure_train_memory.py` | 학습 스텝 메모리·처리량 실측 (R4·R5) — Ray 없이 torchrun 으로 `MegatronPolicyWorkerImpl` 을 직접 만든다. recompute 변형·합성 R3 route·스텝당 마이크로배치 수 |
| `tools/gen_hf_reference_logits.py` | M2·M4 의 HF 참조 로짓 생성 (Pai 환경 전용) |
| `tools/analyze_rollout_logprob_gap.py` | rollout-vs-train logprob 어긋남 분해 (CPU) — R1 진단 |
| `tools/analyze_vllm_request_trace.py` | vLLM 요청별 시각 기록(`NRL_VLLM_REQUEST_TRACE_DIR`) 분석 (CPU) — rollout 긴 꼬리를 대기·prefill·디코딩으로 분해, 긴 요청의 대기 비중·속도, 구간별 동시 실행 수 |
| `tools/analyze_dump_agent_stats.py` | 학습 데이터 덤프(`train_data_step*.jsonl`)의 agent 별 통계 (CPU) — 샘플 수·학습/생성 토큰 비중·잘림(요청별 상한·총 길이)·보상 > 0·advantage 0 비율 (`docs/RLVR_READINESS.md` §5.6) |
| `tools/analyze_gym_logprob_gap.py` | Gym 경로 logprob 어긋남을 시퀀스·환경·호출 구간별로 분해 (CPU) — G3 진단 |
| `tools/teacher_pool_index.py` · `tools/teacher_pool_build.py` | 1차 teacher Code·Math 후보 풀 — 원천 색인 → 문제 단위 중복 제거·정규 행 선택·벤치 오염 제거·사전 측정 목록 (CPU, `docs/RL_DATA.md` §5.6 P0-2) |
| `tools/filter_rl_blend.py` · `tools/measure_blend_prompt_lengths.py` | RL 블렌드 환경 필터(`--preset judge_free`, D1) · 첫 턴 프롬프트 길이 데이터 게이트 (`max_model_len` 초과 행은 런을 멈춘다) |
| `tools/export_rl_hf.sh` | **RL 체크포인트 → Pai 호환 HF 반출은 이것으로만** (G6). 변환 → 메타데이터 시작점 복사 → `compare_hf_weights.py` 대조. 변환기 출력 그대로는 Pai 가 토크나이저를 못 읽는다 |
| `tools/compare_hf_weights.py` · `tools/compare_hf_forward.py` | 반출 HF ↔ 시작점 대조: 텐서·동결·dtype (CPU) · Pai 환경 config·토크나이저·forward + 잡음 바닥 통제 (G6) |
| `tools/export_watch.sh` | 본 런 체크포인트를 N 스텝마다 `export_rl_hf.sh` 로 반출하는 감시 루프 (keep_top_k 가 지우기 전에) |
| `tools/analyze_reward_penalties.py` | Ultra reward_penalties 4종을 롤아웃에 오프라인으로 적용해 발동률·오탐(보상 > 0 이 0 으로 깎이는 수)을 잰다 (CPU) |
| `tools/bench_flashqla.py` | FlashQLA 벤치 3구성: fla-MHA / qla-MHA / qla-네이티브GQA (K1) |
| `tools/inject_identity_blend.py` | RL 블렌드 identity 주입 (`docs/RL_DATA.md` §2) |
| `gym_plugins/responses_api_models/alpha_vllm_model/` | Gym 정책 서버 플러그인 — 도구 정의의 `strict` 를 유지한다 (결정 13). `NEMO_GYM_EXTRA_ROOTS` 로 싣는다 |
| `docs/` | 문서 — 색인 [`docs/README.md`](docs/README.md) |
| `skills/alpha-rl/` | Claude Code skill — 작업 유형별 문서 지도. `.claude/skills/alpha-rl` 링크로 등록, upstream 훅의 "skill 먼저"를 받는다 |

## 이 디렉토리 밖의 alpha 구성 요소

| 구성 요소 | 위치 | 내용 |
|---|---|---|
| AlphaBridge (HF↔mcore) | `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`, fork 브랜치 `alpha/bridge` (`91b88c8f`, `bcc4e415`) | `src/megatron/bridge/models/alpha/` — 문자열 소스 등록(trust-remote-code), 24층 1:1 매핑, qwen3_next 클론 + 델타(표준 RMSNorm·DSV3 라우팅·expert_bias·MTP 제거). 포인터 규칙은 `.claude/rules/alpha-submodules.md` |
| FlashQLA GDN 커널 | `flash-qla==0.1.2` (mcore extra), 스왑은 브리지 fork 의 `alpha/__init__.py` 가 임포트 시 | Qwen 팀 TileLang 커널. 정합성은 fla naive 참조·확장 MHA 경로·15B 실모델 forward 패리티 3중으로 검증 (`docs/GATES.md` K1·M2) |
| vLLM in-tree 구현 (기록용) | `potatowarriors/vllm` 브랜치 `alpha/model` | upstream 기여용. NeMo-RL 통합은 플러그인 사용 |
| RL 데이터 | `/home/work/Datasets/LL_datasets/posttraining/RL/` | `docs/RL_DATA.md` |
| NFS 영속 설치·게이트 산출물 | `/home/work/vidsearch/tools/nemo_rl` (`$NRL_ROOT`) | `docs/SETUP.md` §1 |
