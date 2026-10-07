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
| `grpo_alpha_smoke.yaml` | 8-GPU 1노드 GRPO 드라이런 + KL 게이트(R1). **alpha RL 기본값 포함** — vLLM GDN 재귀 상태 fp32(`generation.vllm_kwargs.mamba_ssm_cache_dtype`) + R3(`router_replay`). 이후 alpha 레시피는 이 파일을 상속 | 게이트 결과 `docs/GATES.md` R1 |
| `grpo_alpha_smoke_muon.yaml` | 위 + Muon(`dist_muon`, SFT 동역학 정렬: nesterov · extra_scale 0.2 · beta2 0.95 · 필수 off 4개) | 게이트 결과 R1·R2 |
| `student_rlvr1.yaml`, `student_rlvr2.yaml` | RLVR (GRPO, SFT 체크포인트에서 시작) | 미작성 |
| `ifbench_teacher.yaml` 등 | 전문 teacher RL (2~3개로 축소 예정) | 미작성 |
| `mopd.yaml` | 멀티 teacher on-policy distillation | 미작성 |

## 디렉토리

| 경로 | 내용 |
|---|---|
| `vllm_alpha_plugin/` | vLLM 플러그인 패키지 (커밋 `599b58ac3`, pyproject 의 vllm extra + uv source 로 연결). `vllm.general_plugins` 엔트리포인트로 stock 0.25.1 휠에 `AlphaForCausalLM` 등록. qwen3_next 서브클래스 + 표준 RMSNorm 전면 교체(융합 QK-norm 커널은 zero-centered +1.0 하드코딩이라 비활성화) + FusedMoE DSV3 인자(`apply_routed_scale_to_output=False` 의도적) |
| `tools/verify_*.py` | 검증 게이트 (`docs/GATES.md` M1·M2·M4·R2·D1) |
| `tools/gen_hf_reference_logits.py` | M2·M4 의 HF 참조 로짓 생성 (Pai 환경 전용) |
| `tools/analyze_rollout_logprob_gap.py` | rollout-vs-train logprob 어긋남 분해 (CPU) — R1 진단 |
| `tools/bench_flashqla.py` | FlashQLA 벤치 3구성: fla-MHA / qla-MHA / qla-네이티브GQA (K1) |
| `tools/inject_identity_blend.py` | RL 블렌드 identity 주입 (`docs/RL_DATA.md` §2) |
| `docs/` | 문서 — 색인 [`docs/README.md`](docs/README.md) |

## 이 디렉토리 밖의 alpha 구성 요소

| 구성 요소 | 위치 | 내용 |
|---|---|---|
| AlphaBridge (HF↔mcore) | `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`, fork 브랜치 `alpha/bridge` (`91b88c8f`, `bcc4e415`) | `src/megatron/bridge/models/alpha/` — 문자열 소스 등록(trust-remote-code), 24층 1:1 매핑, qwen3_next 클론 + 델타(표준 RMSNorm·DSV3 라우팅·expert_bias·MTP 제거). 포인터 규칙은 `.claude/rules/alpha-submodules.md` |
| FlashQLA GDN 커널 | `flash-qla==0.1.2` (mcore extra), 스왑은 브리지 fork 의 `alpha/__init__.py` 가 임포트 시 | Qwen 팀 TileLang 커널. 정합성은 fla naive 참조·확장 MHA 경로·15B 실모델 forward 패리티 3중으로 검증 (`docs/GATES.md` K1·M2) |
| vLLM in-tree 구현 (기록용) | `potatowarriors/vllm` 브랜치 `alpha/model` | upstream 기여용. NeMo-RL 통합은 플러그인 사용 |
| RL 데이터 | `/home/work/Datasets/LL_datasets/posttraining/RL/` | `docs/RL_DATA.md` |
| NFS 영속 설치·게이트 산출물 | `/home/work/vidsearch/tools/nemo_rl` (`$NRL_ROOT`) | `docs/SETUP.md` §1 |
