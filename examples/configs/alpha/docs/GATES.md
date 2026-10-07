# alpha RL 검증 게이트 — 정본

게이트의 이름·기준·명령·결과·재실행 조건을 여기 한 곳에 둔다. 2026-10-07 에 alpha `README.md` 통합 상태표,
`NEMO_RL_SETUP.md` §1·§4.5·§4.6, `ALPHA_POSTTRAIN_PROGRESS.md` §1, Pai `STATUS.md` RL 행에 흩어져 있던 것을 합쳤다.
새 게이트 결과는 이 표의 "결과" 열에 날짜와 함께 추가한다. 해석·사고는 [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) 에 쓴다.

## 재실행 조건

| 바뀐 것 | 다시 돌릴 게이트 |
|---|---|
| 체크포인트 (새 `hfmodel_*`) | M1 · M2 · M3 → R1 |
| AlphaBridge 코드 (Megatron-Bridge fork) | M1 · M2 · M3 (`force_reconvert_from_hf: true`) |
| vLLM 플러그인 코드 | M3 · M4 → R1 |
| 레시피의 생성·라우팅 설정 (`vllm_kwargs`, `router_replay` 등) | R1 |
| 옵티마이저 설정 | R2 → R1 |
| GDN 커널 백엔드 (`ALPHA_GDN_BACKEND`) | K1 · M2 |
| RL 블렌드 파일 | D1 |
| 환경 재구축 (venv·툴킷·lock) | E1 ~ E3 |

## 모델 정합성 (M)

| # | 게이트 | 도구 | 기준 | 결과 |
|---|---|---|---|---|
| M1 | 가중치 라운드트립 HF→mcore→HF | `tools/verify_bridge_roundtrip.py` | 텐서별 `max_diff==0.0` + 양방향 커버리지 (`A_log` 만 문서화된 dtype 예외) | 2026-08-13 base **14,181/14,181** · 2026-10-06 agentic iter2400 **14,181/14,181** |
| M2 | forward 로짓 패리티 (mcore vs Pai 검증 HF 참조) | `tools/verify_forward_parity.py` + `tools/gen_hf_reference_logits.py` | 영/한/887토큰 3종 argmax·top5 일치 + cos ≥ 0.99 | 2026-08-13 cos ≥ 0.99988 · 2026-08-18 FlashQLA 활성 3종 PASS (TileLang 실행 확인) · 2026-10-06 iter2400 argmax 3/3, cos ≥ 0.99984 |
| M3 | mcore→vLLM refit (토큰별 logprob) | upstream `tools/refit_verifier.py` (우리 수정 `1eca2f383`) | mean(exp\|Δ\|) = mult_prob_err < 1.05 | 2026-08-13 mean diff 0.020 / max 0.122 (≈1.02) · 2026-10-06 iter2400 **1.0293** |
| M4 | vLLM 단독 서빙 (디스크 직접 로드) | `tools/verify_vllm_serving_parity.py` | 전체 vocab(163,968) next-token 분포: 3종 argmax 일치 + cos ≥ 0.99 (KL 은 보고) | 2026-08-24 argmax·top5 일치, cos ≥ 0.99997, KL(HF‖vLLM) ≤ 0.0013 |

M2 의 참조 로짓은 **Pai 환경(transformers 4.57)** 에서 만든다 — `modeling_alpha.py` 가 transformers 5.x 와 비호환(OutputRecorder)이다.
M1·M2 는 반드시 **함께** 돌린다. zero-centered RMSNorm 델타는 M1 을 통과하고 M2 에서만 잡힌다 (`KNOWN_ISSUES.md`).

```bash
# 공존 환경(SETUP.md §1)에서는 앞에 $NRL_ROOT/clean_run.sh, uv 는 $NRL_ROOT/bin/uv, 끝에 </dev/null
uv run --locked --extra mcore python examples/configs/alpha/tools/verify_bridge_roundtrip.py --hf-path <hfmodel_00NNNNN>
# M2 참조 로짓 (Pai 환경, 시스템 python):
LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib" \
    python3 examples/configs/alpha/tools/gen_hf_reference_logits.py --hf-path <hfmodel_00NNNNN> --out <ref.pt>
uv run --locked --extra mcore python examples/configs/alpha/tools/verify_forward_parity.py --hf-path <hfmodel_00NNNNN> --ref <ref.pt>
uv run --locked --extra mcore python3 tools/refit_verifier.py --model_name <hfmodel_00NNNNN>
uv run --locked --extra vllm python examples/configs/alpha/tools/verify_vllm_serving_parity.py --hf-path <hfmodel_00NNNNN> --ref <ref.pt>
```

## RL 실행 (R)

| # | 게이트 | 도구 | 기준 | 결과 |
|---|---|---|---|---|
| R1 | GRPO Generation KL (rollout vLLM vs 학습 mcore), 8-GPU 1노드 | `grpo_alpha_smoke.yaml` (Muon 은 `grpo_alpha_smoke_muon.yaml`), 진단 `tools/analyze_rollout_logprob_gap.py` | step 1·2·3 의 `Generation KL Error` < 0.002 | 2026-10-06 iter2400: 기본 0.0042 FAIL → R3 0.0026 → **R3 + GDN 상태 fp32 Adam 0.0015/0.0013/0.0014 PASS · Muon 0.0015/0.0013/0.0013 PASS** |
| R2 | Muon 실적용 (워커와 같은 setup 경로로 옵티마이저 직접 검사) | `tools/verify_muon_optimizer.py` | 4항목 전부 PASS: Muon·Adam 클래스 공존 · 파라미터 분배 · qkv 4-way split · 하이퍼파라미터 레시피 일치 | 2026-10-06 PASS — TensorParallelMuon 15.34B · Adam 0.67B · qkv 4-way · extra_scale 0.2 · nesterov · router 동결(RL 기본) |

R1 의 step 1 은 갱신 전 가중치라 옵티마이저와 무관하고, step 2·3 은 갱신 후 refit 경로까지 본다.
mcore 는 Muon 경로 로그를 기본 설정에서 숨기므로 GRPO 런 로그만으로는 Muon 적용을 증명할 수 없다 → R2 가 필요하다.
게이트 산출물: `$NRL_ROOT/gates/<ckpt>/`.

```bash
cd NeMo-RL && $NRL_ROOT/clean_run.sh $NRL_ROOT/bin/uv run --locked python examples/run_grpo.py \
  --config examples/configs/alpha/grpo_alpha_smoke.yaml policy.model_name=<hfmodel_00NNNNN> logger.log_dir=<dir> </dev/null
python examples/configs/alpha/tools/analyze_rollout_logprob_gap.py <dir>/exp_*/train_data_step*.jsonl   # CPU
$NRL_ROOT/clean_run.sh $W/bin/python -m torch.distributed.run --nproc_per_node 8 \
  examples/configs/alpha/tools/verify_muon_optimizer.py --config examples/configs/alpha/grpo_alpha_smoke_muon.yaml </dev/null
```

## 커널·데이터 (K·D)

| # | 게이트 | 도구 | 기준 | 결과 |
|---|---|---|---|---|
| K1 | FlashQLA GDN 커널 정합성 (fla naive 3자 대조) | `tools/bench_flashqla.py` | fwd/bwd 수치 일치 | 2026-08-18 fwd/bwd cos ≥ 0.9969 · 성능 fwd 1.8~4.2×, fwd+bwd 1.6~3.7× |
| D1 | RL 블렌드 구조 | `tools/verify_rl_blend.py` | 전행 JSON·키·잔여 마스킹 0 + agent 분포 | 2026-08-13 `rlvr1_alpha`·`rlvr2_alpha` 양쪽 OK |

## 환경 (E)

| # | 게이트 | 기준 | 결과 |
|---|---|---|---|
| E1 | GRPO 퀵스타트 (Qwen2.5-1.5B, DTensor+vLLM / **mcore**+vLLM), 1 GPU 2 step | 정상 스텝 · mcore KL ~7e-4 | 2026-08-13 RL 전용 세션: DTensor 57 s/step · mcore 44–50 s/step, KL ~7e-4 · 2026-10-06 공존 설치(alpha/post-train `83976fe0d`, main1) `grpo_math_1B_megatron.yaml` colocated: **KL 0.0007 / 0.0006**, 53.6 / 57.4 s/step, Traceback 0, 종료 후 Ray·GPU 메모리 전부 회수 |
| E2 | venv 빌드 | vllm·mcore extra 각각 성공, 플러그인 등록 | 2026-10-06 vllm venv(deep-ep·deep-gemm 소스 빌드 65 s, torch 2.11.0+cu130, vLLM 0.25.1, `AlphaForCausalLM` 등록 True) · mcore venv 683 s (TE·DeepEP·mamba-ssm·causal-conv1d·fast-hadamard) + TE 재빌드 282 s(#15) |
| E3 | TE fused attn (cuDNN sub-backend 1) | Megatron 워커 import 체인 뒤 fwd/bwd 통과, 매핑 cudart = cu13 만 | 2026-10-06 max\|o−ref\| 0.0080 (fp32 SDPA 대비), grad finite |
| E4 | Gym v0.6.0 | CPU 테스트 · alpha 설정 validate | 2026-10-06 CPU 테스트 69 pass · alpha 설정 4/4 · PivotRL 서버 298 pass |

미검증: 2노드 Ray 스모크, 사설 경로 영속화 구성의 재생성 후 동작, FlashQLA E2E 스텝 이득.
