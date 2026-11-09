# alpha RL 검증 게이트 — 정본

게이트의 이름·기준·명령·결과·재실행 조건을 여기 한 곳에 둔다. 2026-10-07 에 alpha `README.md` 통합 상태표,
`NEMO_RL_SETUP.md` §1·§4.5·§4.6, `ALPHA_POSTTRAIN_PROGRESS.md` §1, Pai `STATUS.md` RL 행에 흩어져 있던 것을 합쳤다.
새 게이트 결과는 이 표의 "결과" 열에 날짜와 함께 추가한다. 해석·사고는 [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) 에 쓴다.
**게이트·스모크를 새로 만들거나 기준·측정 방식을 바꾸면 기준선 실행 전에 적대적 검토를 받는다** — "고장 난 상태에서도 PASS 할 수 있는가"
(사용자 공통 원칙 "문제 진단과 독립 검증", `problem-diagnosis` skill 부록, 2026-10-21). 검토 판정은 해당 게이트 행의 결과 열에 남긴다.

## 재실행 조건

| 바뀐 것 | 다시 돌릴 게이트 |
|---|---|
| 체크포인트 (새 `hfmodel_*`) | M1 · M2 · M3 → R1 |
| RL 반출 체크포인트를 시작점으로 (`hf/hf_step_*`, 사용자 결정 2026-10-19) | M1 · M2 · M4 → **R1** (M3 는 기록만 — R3 없는 10토큰 비교라 RL 로 라우팅이 민감해진 체크포인트에서 1.05 를 넘는다. 판정은 R3 를 켠 R1 과 반출 직전 학습 중 `token_mult_prob_error`·gen KL) |
| AlphaBridge 코드 (Megatron-Bridge fork) | M1 · M2 · M3 (`force_reconvert_from_hf: true`) |
| vLLM 플러그인 코드 | M3 · M4 → R1 |
| 레시피의 생성·라우팅 설정 (`vllm_kwargs`, `router_replay` 등) | R1 |
| 옵티마이저 설정 | R2 → R1 |
| GDN 커널 백엔드 (`ALPHA_GDN_BACKEND`) | K1 · M2 |
| RL 블렌드 파일 | D1 |
| Pai 또는 NeMo-RL 의 mcore·TE 버전, 학습 forward 설정(라우터 dtype·fusion 등) | M5 |
| chat template·토크나이저, Gym 버전·alpha Gym 서버, `openai_server_utils.py`·vLLM 서빙 계층, 레시피 `turn_end_token_id`·`chat_template_kwargs` | R3 |
| 컨텍스트 길이·CP·EP, logprob 설정(`logprob_chunk_size`·`fuse_loss`·`defer_fp32_logits`), 할당자(`PYTORCH_CUDA_ALLOC_CONF`), 옵티마이저 | R4 |
| 환경 재구축 (venv·툴킷·lock) | E1 ~ E3 |
| GDN 커널·packing 코드(mcore GDN·NeMo-RL `data.py` packing)·`ALPHA_GDN_BACKEND` | G1 |
| 레시피 (alpha GRPO 레시피 전부) | `tools/check_alpha_recipe.py` (CPU, ERROR 0 이어야 실행) |
| Gym judge 모델·롤아웃 GPU 수 (`policy.generation.colocated.resources.gpus_per_node`)·`grpo.py` 자원 배치 코드 | J1 |

## 모델 정합성 (M)

| # | 게이트 | 도구 | 기준 | 결과 |
|---|---|---|---|---|
| M1 | 가중치 라운드트립 HF→mcore→HF | `tools/verify_bridge_roundtrip.py` | 텐서별 `max_diff==0.0` + 양방향 커버리지 (`A_log` 만 문서화된 dtype 예외) | 2026-08-13 base **14,181/14,181** · 2026-10-06 agentic iter2400 **14,181/14,181** · 2026-10-08 agentic iter2862 **14,181/14,181** |
| M2 | forward 로짓 패리티 (mcore vs Pai 검증 HF 참조) | `tools/verify_forward_parity.py` + `tools/gen_hf_reference_logits.py` | 영/한/887토큰 3종 argmax·top5 일치 + cos ≥ 0.99 | 2026-08-13 cos ≥ 0.99988 · 2026-08-18 FlashQLA 활성 3종 PASS (TileLang 실행 확인) · 2026-10-06 iter2400 argmax 3/3, cos ≥ 0.99984 · 2026-10-08 iter2862 argmax 3/3, cos ≥ 0.99980 |
| M3 | mcore→vLLM refit (토큰별 logprob) | upstream `tools/refit_verifier.py` (우리 수정 `1eca2f383`) | mean(exp\|Δ\|) = mult_prob_err < 1.05 | 2026-08-13 mean diff 0.020 / max 0.122 (≈1.02) · 2026-10-06 iter2400 **1.0293** · 2026-10-08 iter2862 **1.0228** (mean diff 0.022 / max 0.111) · 2026-10-19 RL 반출 `teacher_codemath2/hf_step_00282` **FAIL 1.07** (mean diff 0.068 / max 0.159, 10토큰 중 8개 0.03~0.16) — 진단 (같은 날 08:43~09:11): 대조군 iter2862 재실행 **1.025** (0.0249, 환경 변화 없음) · step_282 재실행 0.0558 (≈1.057, 재현) · 같은 ckpt 의 R1(R3 켬) **KL 0.0010 × 3 · mult_prob_err 1.018 · \|d\|>1 토큰 0**. 결론: M3(R3 없음, 10토큰)의 초과는 RL 로 라우팅 경계가 민감해진 체크포인트의 성질이고 refit 경로 결함이 아니다 (KNOWN_ISSUES 10-16). 사용자 결정 2026-10-19: RL 반출 시작점은 M3 대신 R1 로 판정 (재실행 조건 표) |
| M4 | vLLM 단독 서빙 (디스크 직접 로드) | `tools/verify_vllm_serving_parity.py` | 전체 vocab(163,968) next-token 분포: 3종 argmax 일치 + cos ≥ 0.99 (KL 은 보고) | 2026-08-24 argmax·top5 일치, cos ≥ 0.99997, KL(HF‖vLLM) ≤ 0.0013 · 2026-10-08 iter2862 argmax 3/3 (top5 1/3), cos ≥ 0.99995, KL ≤ 0.0074 — KL 이 8월보다 크고 원인은 미확인 (GDN 상태를 fp32 로 바꿔도 같은 값). RL 경로 정합은 R1 이 본다 |
| M5 | SFT 엔진(Pai Megatron-LM-251125) ↔ RL 엔진(NeMo-RL mcore) forward·gradient 동등성 | `tools/engine_parity_{pai,nemorl,hf,compare}.py` (+ `engine_parity_common.py`) | Pai↔NeMo-RL 거리가 bf16 노이즈 바닥 이내. 바닥은 두 기준선으로 잰다: HF 를 제3 구현으로 둔 삼각측량, 같은 Pai 엔진에 라우터 dtype 만 바꾼 섭동 런 | 2026-10-07 iter2400 **PASS** (아래 표) |

M2 의 참조 로짓은 **Pai 환경(transformers 4.57)** 에서 만든다 — `modeling_alpha.py` 가 transformers 5.x 와 비호환(OutputRecorder)이다.
M1·M2 는 반드시 **함께** 돌린다. zero-centered RMSNorm 델타는 M1 을 통과하고 M2 에서만 잡힌다 (`KNOWN_ISSUES.md`).

**M5 상세 (2026-10-07, sub1 8-GPU, EP8, 고정 배치 2×2048 토큰, 같은 iter2400 가중치).**
두 엔진은 모델 클래스와 파라미터 이름이 다르다(Pai MambaModel 48층 + megatron_patch GDN, NeMo-RL GPTModel 24층 + mcore GDN).
그래서 텐서를 이름이 아니라 가중치 비트의 정수 해시 지문으로 짝짓는다. 판정은 절대 임계가 아니라 기준선 비교다.
MoE·bf16 에서는 같은 엔진끼리도 gradient cos 가 0.98 근처에서 포화한다 (`KNOWN_ISSUES.md` 2026-10-07 엔진 동등성 항목).

| 지표 | Pai ↔ NeMo-RL | 노이즈 기준선 |
|---|---|---|
| loss | 2.41961 / 2.41911 (상대 차 2.1e-4) | HF 2.41863 · 섭동 런 상대 차 1.1e-3 |
| 토큰 logprob mean\|Δ\| | **0.0673** | HF↔Pai 0.0686 · HF↔NeMo-RL 0.0685 · 섭동 0.0677 — 세 구현이 서로 같은 거리 |
| k3 KL (위치 구간별) | 0.0076 (0.0063~0.0085, 위치에 따라 커지지 않음) | 섭동 0.0079 |
| gradient cos 중앙값 | embedding 0.987 · output 0.998 · experts 0.982 · GDN 0.994~0.995 · shared experts 0.991 | 섭동 0.984 · 0.998 · 0.986 · 0.994 · 0.990 |
| 층별 gradient norm 비율 | 1.000~1.008 (embedding 만 1.023) | 섭동 0.989~0.999 |
| 비교 범위 | 9,459/9,495 텐서, 파라미터의 97.2% | GDN `in_proj`·`conv1d` 36개는 결합 레이아웃이 달라 제외(그 layernorm 은 비교). router 는 RL 동결이라 grad 없음 |
| peak 메모리 | Pai 30.7 GB · NeMo-RL 32.2 GB | — |

M5 의 KL 은 고정 배치 teacher-forced 값이고 R1 의 KL 은 모델 자신의 롤아웃 샘플 값이다. 측정 분포가 달라 두 수치를 직접 비교하지 않는다.
미검증 범위: 시퀀스 2048·CP1 만 봤다. packing·CP 경로의 수치 동등성은 열려 있다 (`STATUS.md`).
산출물·실행 스크립트: `$NRL_ROOT/gates/engine_parity_iter2400/` (`run_{pai,nemorl,hf}_sub1.sh`, `report.json`, 섭동 기준선 `report_pai_vs_pai_router_bf16.json`). 랭크별 로짓 덤프(9.4 GB)는 2026-10-07 판정 기록 뒤 삭제했다 (`report*.json`·로그·스크립트는 남김).

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
| R1 | GRPO Generation KL (rollout vLLM vs 학습 mcore), 8-GPU 1노드 | `grpo_alpha_smoke.yaml` (Muon 은 `grpo_alpha_smoke_muon.yaml`), 진단 `tools/analyze_rollout_logprob_gap.py` | step 1·2·3 의 `Generation KL Error` < 0.002 | 2026-10-06 iter2400: 기본 0.0042 FAIL → R3 0.0026 → **R3 + GDN 상태 fp32 Adam 0.0015/0.0013/0.0014 PASS · Muon 0.0015/0.0013/0.0013 PASS** · 2026-10-08 iter2862 Muon **0.0014/0.0013/0.0013 PASS** · 2026-10-11 `vllm_kwargs.gdn_prefill_backend: triton` + 학습 `NCCL_MAX_NCHANNELS=16` 재실행 (본 런 `main2r3` 의 `ckpt/step_40` 재개 step 41·42·43, 32K 생성·CP8·packing) **0.00177/0.00180/0.00198 PASS** · 마스킹 0/0/0 (이전 step 21~40 은 0.00177~0.00193) · 2026-10-19 RL 반출 `teacher_codemath2/hf_step_00282` (Muon) **0.0010/0.0010/0.0010 PASS** · mult_prob_err 1.018 · 위치별 평탄 · \|d\|>1 토큰 0 |
| R2 | Muon 실적용 (워커와 같은 setup 경로로 옵티마이저 직접 검사) | `tools/verify_muon_optimizer.py` | 4항목 전부 PASS: Muon·Adam 클래스 공존 · 파라미터 분배 · qkv 4-way split · 하이퍼파라미터 레시피 일치 | 2026-10-06 PASS — TensorParallelMuon 15.34B · Adam 0.67B · qkv 4-way · extra_scale 0.2 · nesterov · router 동결(RL 기본) |
| R3 | chat 렌더 패리티 — RL 프롬프트 토큰 ID 가 SFT 변환기(`build_alpha_sft_idxmap.py`)의 학습 토큰 ID 와 정확히 같은가 (CPU) | `tools/verify_chat_render_parity.py` | 경로별 토큰 ID 완전 일치 | 2026-10-07: 네이티브 GRPO **PASS** · Gym 멀티턴 경로 **부분** — 턴 경계 수정 반영, `strict` 유지 미구현 (아래 표) |
| R5 | 학습 처리량·recompute 변형 (`tools/measure_train_memory.py --samples-per-dp N --recompute ...`) | 같은 도구 | 정상 상태(3번째 스텝) 처리량·peak. 판정 게이트가 아니라 기준선 | 2026-10-07 부분 실행: 128K/CP8 full **33.4K tok/s**·peak 57.2 GB, selective 전 변형 OOM (`RLVR_READINESS.md` §5). 나머지는 첫 RLVR 런 뒤 (사용자 지시) |
| R4 | 학습 스텝 메모리 실측 — NeMo-RL 실제 학습 경로(`MegatronPolicyWorkerImpl.train` + `ClippedPGLossFn` + Muon step), 학습 전용 노드 가정(vLLM 없음) | `tools/measure_train_memory.py` | 80 GB 안에 2 스텝 완주 (1 스텝째 Muon momentum 지연 할당 포함) | 2026-10-07 sub1: 1노드 **128K/CP8 OK**, 단 `expandable_segments` 필요 (아래 표) · 2026-10-21 적대적 검토(원본 하네스 `e2a777d86` 대상 보정 시험): **보완 필요** — 10-07 에 드러난 두 결함(없는 `loss` 키 · 워밍업 처리량)을 다시 찾음. 지금 하네스에도 남은 것: 합성 `prev_logprobs`(−U(0,1), ratio ≈ e⁻¹¹ 라 기울기가 거의 0) · reference 모델·logprob 패스 없는 스텝 순서 · torch 할당기 밖 메모리 미측정 — 다음 R4 재실행 전에 보완하고 다시 검토한다 |

R1 의 step 1 은 갱신 전 가중치라 옵티마이저와 무관하고, step 2·3 은 갱신 후 refit 경로까지 본다.
mcore 는 Muon 경로 로그를 기본 설정에서 숨기므로 GRPO 런 로그만으로는 Muon 적용을 증명할 수 없다 → R2 가 필요하다.
게이트 산출물: `$NRL_ROOT/gates/<ckpt>/`.

**R3 상세 (2026-10-07, iter2400 토크나이저, 케이스 12종: 단일 턴 3 · 비도구 멀티턴 2 · 도구 3 · 실제 SFT 행 2 · 실제 RL 블렌드 행 2).**
경로마다 실제 코드를 호출한다(서버·엔진 없이). 멀티턴은 "모델이 SFT 정답 토큰을 그대로 냈다"고 가정하고 Gym 이 다음 요청을 만드는 과정을 재생한다.

| 경로 | MATCH / MISMATCH | 판정 |
|---|---|---|
| P0 SFT 자기 일관성 (`add_generation_prompt`) | 22 / 0 | PASS |
| P1 NeMo-RL 네이티브 GRPO 데이터 프로세서 | 12 / 0 | PASS — 생성 시작점(`<think>`)·정지 토큰 {0, 3} 도 SFT 와 같음 |
| P2 transformers 4.57 · 5.5 · 5.8 교차 전체 렌더 | 22 / 0 | PASS |
| P4 Gym → vLLM + `replace_prefix_tokens` (수정 전, 경계 = EOS 0) | 0 / 40 | FAIL — 2번째 호출부터 전부 (`KNOWN_ISSUES.md` 2026-10-07 턴 경계) |
| P5 위 + `turn_end_token_id=3` (실제 패치 호출) | 8 / 32 | 도구 정의 차이만 남음 |
| P6 위 + 도구 정의 보정 시뮬레이션 (`strict` 복원 · `None` 키 제거) | 36 / 4 | 남은 4건은 c3b(도구 없는 멀티턴 롤아웃의 이전 reasoning 복원) — 구조적, 현 블렌드 영향 없음 |

P6 의 도구 정의 보정은 아직 **시뮬레이션**이다. `strict` 유지(사용자 결정 2026-10-07)를 구현한 뒤 P6 을 실경로로 다시 돌린다.
미검증 범위: vLLM 서버를 띄우지 않았다. SWE·terminus 등 개별 에이전트 하니스의 메시지 조립은 보지 않았다.
산출물: `$NRL_ROOT/gates/render_parity_iter2400/` (`run_full/` 수정 전, `run_patched/` 수정 후, `REPORT.txt`, `strict_survey.py`).

**R4 상세 (2026-10-07, sub1 8×80 GB, EP8 · TP1 · DP = 8/CP, 합성 배치 1 샘플/DP, Muon 레시피, R3 끔, 2 스텝).**
CP>1 은 NeMo-RL 이 packing 을 요구해 `sequence_packing.enabled: true`(`train_mb_tokens` = 길이)로 쟀다.

| 길이 / CP | rank 당 토큰 | 할당자 | chunk | peak alloc | peak reserved | 결과 |
|---|---|---|---|---|---|---|
| 32K / 2 | 16K | 기본 | 없음 | — | — | OOM (Triton) |
| 16K / 2 | 8K | 기본 | 2048 | 50.7 GB | 60.7 GB | OK |
| 64K / 8 | 8K | 기본 | 2048 | 50.7 GB | 60.8 GB | OK |
| 96K / 8 | 12K | 기본 | 2048 | 54.0 GB | 67.6 GB | OK |
| 32K/2 · 64K/4 · 128K/8 | 16K | 기본 | 2048 | — | — | OOM (Triton) |
| 128K / 8 | 16K | **expandable_segments** | 2048 | 57.2 GB | 59.5 GB | **OK** · 2 스텝째 8.7 s |
| 128K / 8 | 16K | **expandable_segments** | 1024 | 53.4 GB | 55.7 GB | **OK** · 2 스텝째 8.5 s |

"chunk" = `policy.logprob_chunk_size`(+ `sequence_packing.fuse_loss: true`, Ultra 레시피 값). 고정 바닥(가중치·grad·Muon 상태)은 27.0 GB 다.
나머지는 RL loss 의 logprob backward 가 차지한다. OOM 원인은 총량이 아니라 단편화다 (`KNOWN_ISSUES.md` 2026-10-07 메모리 항목).
미검증 범위: vLLM 동거(colocated), R3 켠 상태, 실제 데이터 분포, ES 가 refit 시간에 주는 영향, packing+CP 의 KL 수치.
결과 JSON 의 `loss: NaN` 은 하네스 키 오류다(워커 반환 키는 `global_loss`). 그래서 R4 는 loss·grad norm 유한성을 확인하지 않았다 (`KNOWN_ISSUES.md` 2026-10-07, G0 에서 수정).
산출물: `$NRL_ROOT/gates/train_memory/` (`mem_*.json`·`.log`, 런처 `run_mem_sub1.sh`, 스냅샷 분석 `analyze_snapshot.py`).

```bash
cd NeMo-RL && $NRL_ROOT/clean_run.sh $NRL_ROOT/bin/uv run --locked python examples/run_grpo.py \
  --config examples/configs/alpha/grpo_alpha_smoke.yaml policy.model_name=<hfmodel_00NNNNN> logger.log_dir=<dir> </dev/null
python examples/configs/alpha/tools/analyze_rollout_logprob_gap.py <dir>/exp_*/train_data_step*.jsonl   # CPU
$NRL_ROOT/clean_run.sh $W/bin/python -m torch.distributed.run --nproc_per_node 8 \
  examples/configs/alpha/tools/verify_muon_optimizer.py --config examples/configs/alpha/grpo_alpha_smoke_muon.yaml </dev/null
# R3 (CPU, 5개 인터프리터를 단계별로 호출):
CUDA_VISIBLE_DEVICES="" python3 examples/configs/alpha/tools/verify_chat_render_parity.py run --out <dir>   # 종료 0 = 전부 MATCH
# R4 (8 GPU, 조건 하나당 1회). whitelist 밖 환경변수는 clean_run.sh 뒤 /usr/bin/env 로 넘긴다 (원장 #23):
$NRL_ROOT/clean_run.sh /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $W/bin/python -m torch.distributed.run \
  --nproc_per_node 8 examples/configs/alpha/tools/measure_train_memory.py --config examples/configs/alpha/grpo_alpha_smoke_muon.yaml \
  --seq-len 131072 --cp 8 --logprob-chunk-size 2048 --fuse-loss --out mem.json </dev/null
# M5: $NRL_ROOT/gates/engine_parity_iter2400/run_{pai,nemorl,hf}_sub1.sh 실행 뒤
#     <NeMo-RL venv python> examples/configs/alpha/tools/engine_parity_compare.py --pai-dir <pai> --nemorl-dir <nemorl> --out report.json
```
GPU 게이트 런처는 기동 직전에 GPU 점유를 검사한다. 한 장이라도 1 GiB 이상 쓰이면 중단한다 (원장 #25).

## 커널·데이터 (K·D)

| # | 게이트 | 도구 | 기준 | 결과 |
|---|---|---|---|---|
| K1 | FlashQLA GDN 커널 정합성 (fla naive 3자 대조) | `tools/bench_flashqla.py` | fwd/bwd 수치 일치 | 2026-08-18 fwd/bwd cos ≥ 0.9969 · 성능 fwd 1.8~4.2×, fwd+bwd 1.6~3.7× |
| D1 | RL 블렌드 구조 | `tools/verify_rl_blend.py` | 전행 JSON·키·잔여 마스킹 0 + agent 분포 | 2026-08-13 `rlvr1_alpha`·`rlvr2_alpha` 양쪽 OK |

## RLVR 이식 (G) — 계획은 [`RLVR_READINESS.md`](RLVR_READINESS.md) §6

| # | 게이트 | 도구 | 기준 | 결과 |
|---|---|---|---|---|
| G1 | packing 상태 누출 — 같은 길이의 다른 앞 시퀀스 A·A' 뒤에 같은 B 를 묶어 B logprob 비교 (CP1·EP8, `make_sequence_length_divisible_by 16`, 입력 순서 유지 packer) | `tools/verify_packing_isolation.py` | B logprob 이 A 내용과 무관 (비트 동일 또는 잡음 기준선 이내, 앞 64 토큰 집중 없음) | 2026-10-07 iter2400 **PASS** — fla·FlashQLA 모두 4쌍 (A,B) = (4097,2048)·(16,777)·(8191,3001)·(1000,1000) 에서 **비트 동일** |
| G2 | packing+CP 학습 경로의 rollout↔학습 정합 — R1 과 같은 레시피·시드(롤아웃 동일)에서 학습 배치 경로만 바꾼다. R3 route 추적·검증 켬 (`NRL_R3_TRACE*`, `NRL_ROUTER_REPLAY_VALIDATE=1`) | `grpo_alpha_smoke_muon.yaml` + override, `$NRL_ROOT/gates/packing_cp_r1/run_g2.sh`, `tools/analyze_rollout_logprob_gap.py`, `tools/check_r3_trace.py` | step 1·2·3 Generation KL < 0.002 · 위치 구간 평탄 · R3 forward 검증 불일치 0 · CP 토큰 일치 | 2026-10-07 iter2400 4K: **CP1+packing 0.0015/0.0013/0.0013 · CP8+packing 0.0015/0.0013/0.0014 PASS** (R1 Muon 0.0015/0.0013/0.0013). **32K 생성 CP8+packing step 1 0.0016 PASS** — 위치 [2K,4K) 0.00158 → [16K,32K) 0.00166 (+5%) |
| G3 | Gym 경로 정합: Gym HTTP + 도구·추론 파서(`qwen3_xml`·`nemotron_v3`) + `strict` 유지 플러그인 서버 + R3, 1노드 colocated 동기 GRPO, judge 불필요 10개 환경 스모크 138행(도구 72행) | `$NRL_ROOT/gates/gym_smoke/run_g3.sh`, `tools/analyze_gym_logprob_gap.py` | Generation KL < 0.002 · seq mult_prob_error>2 마스킹 0 · invalid tool call·malformed think 0 · 도구 프롬프트에 `<strict>True</strict>` | 2026-10-07 iter2400 **PASS (수정 뒤)** — 첫 판정 KL 0.0043·0.0045, 도구 시퀀스 1/3 마스킹 FAIL → `VLLM_ENFORCE_STRICT_TOOL_CALLING=0` 뒤 **KL 0.0017/0.0016**, 마스킹 0, invalid 0/64, malformed 0/64, strict 렌더 36/36 |
| G5 | 실레시피(`student_rlvr1_alpha.yaml`) 2노드 스모크: async + in-flight + Gym + 분리 토폴로지(롤아웃 sub1 · 학습 main1, NCCL refit over TCP), 128K·CP8·packing·ES·CUDA graph, judge 불필요 블렌드, 16 프롬프트 × 8, 생성 최대 16K, 3 스텝, step 2·3 저장 | `$NRL_ROOT/gates/two_node/{ray_node.sh,run_g5.sh}` | Generation KL < 0.002 · 마스킹 0 · R3 route 누락 경고 0 · 저장 성공 | 2026-10-07 iter2400 **PASS (수정 뒤)** — 첫 실행은 2스텝 비동기 저장에서 멈춤(ES + CUDA IPC) → `async_save: false` 뒤 **KL 0.0018/0.0018/0.0019**, 마스킹 0, route 누락 0, step_2·step_3 저장(각 155 GB) |
| G6 | RL 체크포인트 → HF 반출 → Pai 소비 경로: G5 step_3 를 반출해 시작점 iter2400 과 가중치·메타데이터·Pai forward·Pai 서빙을 대조 | `tools/export_rl_hf.sh` (반출·메타데이터 복원·`compare_hf_weights.py`), `tools/compare_hf_forward.py` (Pai 환경), Pai `forward_sanity.py`, `$NRL_ROOT/gates/export_resume/g6_serve_compare.sh` | 텐서 이름·shape 일치 · 동결 텐서 비트 동일 · dtype 변경 무손실 · 비가중치 파일이 시작점과 같음 · Pai 의 토크나이저·config 해석 동일 · forward 차이가 잡음 바닥 이내 · 서빙 finish 가 stop/tool_calls 이고 `<\|im_end\|>` 에서 정지 | 2026-10-07 G5 step_3 **PASS (수정 뒤)** — 변환기 원본은 Pai 가 토크나이저를 못 읽는다(`TokenizersBackend`) → `export_rl_hf.sh` 가 메타데이터를 복원. 텐서 14,181개 오류 0, 라우터 24/24·expert bias 24/24 비트 동일, `A_log` fp32→bf16 18/18 무손실, forward_sanity ppl 6.48, 서빙 6/6 |
| G7 | 저장 → 재개 (런 연장): 실레시피 2노드로 3스텝(step_2·step_3 저장) → step_3 에서 `max_num_steps` 5 로 재개(step_4·step_5 저장). 옵티마이저 상태 연속성은 연속 구간(2→3·4→5)과 재개 구간(3→4)을 비교 | `$NRL_ROOT/gates/two_node/run_g7.sh`, `tools/verify_optimizer_resume.py` (CPU) | 재개 기동 성공 · 스텝·데이터·replay buffer·lr 이어짐 · KL < 0.002 · 마스킹 0 · 옵티마이저 상태에 초기화·master 재생성·다른 상태 신호 0 · 음성 대조(다른 런 체크포인트) FAIL | 2026-10-07 iter2400 **PASS (수정 뒤)** — 첫 시도는 재개에서 스케줄러 assert (`wd_incr_steps` 640 vs 384) → 레시피 `scheduler.max_steps` 고정 뒤 재개 성공. step 4·5 KL 0.0019/0.0018 (연속 0.0016/0.0018/0.0019), 마스킹 0, lr 3.7e-7·4.6e-7 (warmup 직선), 옵티마이저 상태 초기화 신호 0/62 · 음성 대조 50/62 |
| G8 | v2 재시작 설정 스모크 (실험명 E0): upstream 이식 7건 + 생성 상한 64K + loss 안 시퀀스 마스킹(#4171) + 속도 레버 2개(chunked prefill 16K·`max_trajectory_age_steps` 2)의 기능 확인. 실레시피 2노드, 48행(effort 마커 24행) = 16 프롬프트 × 8 × 3스텝 = 1에폭, 생성 최대 16K | `$NRL_ROOT/runs/rlvr1_alpha_v2/{launch.sh,run_e0.sh}` | rc 0 · 1에폭 끝에서 정상 종료 · KL < 0.002 · loss 안 마스킹 지표 · effort 지표 · refit 일시정지 로그 · R3 fallback 0 | 2026-10-08 iter2400 **PASS (수정 뒤)** — 첫 시도는 기동 직후 `dataloader exhausted while waiting for initial buffer fill` (rc 139): f49e41dda 이식에서 `CyclingDataLoader` 를 빠뜨림 → `8d7bc8832`. 재시도: **KL 0.001989/0.001952/0.001852**, 마스킹 0/128 (3스텝 모두), 궤적 나이 0/1/2, refit 때 일시정지한 생성 배치 0/1/2개, R3 replay forward·backward 2,304/2,304 (fallback 0), logprob 단계 0 s |
| Z1 | `grpo.skip_zero_advantage_samples` 기울기 동일성: 같은 가중치에서 train() 세 번 (lr 0 으로 가중치 고정) — A 전체 배치, B advantage 0 제외 + 재조정 + `lr_scheduler_increment`, C 음성 대조(재조정 없음), A2 전체 배치 반복(잡음 바닥). CP8·packing·R3 합성 route, 16 샘플 × 8K | `tools/verify_zero_adv_skip.py` (torchrun 8 GPU, Ray 없음) | loss 같음 · B 의 기울기 차이가 잡음 바닥의 3배 이내 · lr 전진량 A = B · C 는 바닥의 10배 초과 | 2026-10-08 iter2400 **PASS (판정 기준 수정 뒤)** — loss 3회 비트 동일, 표본 기울기 B 1.70% vs 바닥 1.64%, norm 0.84% vs 0.96%, C 89.7%, lr 전진 A 16 = B 16 (고치기 전 C 8). 첫 판정은 절대 기준 1e-3 이라 backward 비결정성을 실패로 읽었다 |
| J1 | judge 분할 2노드 스모크 (`RL_PLAN.md` 결정 21): 실레시피 `teacher_stem_alpha.yaml` — 롤아웃 노드 = vLLM 롤아웃 6장 + Gym judge (Gemma-4-31B-it TP2, `local_vllm_model`) 2장, 본체 수정 `alpha/judge-split`. 스모크 블렌드 `teacher_stem_smoke_v3.jsonl` 256행 (v2 의 Science-v1 정답 잘림 16행 교체, 결정 25) (객관식 64 · 화학 rdkit 플러그인 64 · 주관식 judge 64 · python 도구 과학 64) = 64 프롬프트 × 16 × 4스텝, 저장 끔, W&B 끔 | `teacher_stem_alpha.yaml` + `grpo.max_num_steps=4`, `ray list placement-groups`, Gym 응답 로그 | rc 0 · 학습 8장이 한 노드, 롤아웃 6장과 judge 2장이 다른 한 노드 · Generation KL < 0.002 · 마스킹 0 · judge 환경과 rdkit 플러그인 보상에 0 과 1 이 모두 있음 · judge 판정 파싱 실패 ≤ 2% · 재판정(`check_full_generation_on_fail`)이 실제로 일어남 — Gym 응답 로그에서 횟수와 0.5 보상 표본을 확인 · ns_tools 도구 호출이 sandbox 에서 실행됨 · 스텝 시간 기록 (롤아웃 8장 런과 비교) | **✅ PASS 2026-10-10** — `j1b` (14:58~16:19, rc 0): 학습 sub1 8장 · 롤아웃 main1 6장 · judge main1 2장 · gen KL 0.00196 · 0.00199 · 0.00193 · 0.00191 · 마스킹 0 · judge · rdkit 보상에 0 과 1 · ns_tools 도구 sandbox 실행 · 스텝 642 · 445 · 573 · 734 s (롤아웃 8장 런과는 데이터가 달라 직접 비교하지 않는다). j1b 는 judge 판정 기록을 남기지 않아 (`KNOWN_ISSUES.md` 2026-10-10) 보충 `j1c` (judge 128 + rdkit 64 행, W&B 오프라인 표, 1스텝 분량 수집 뒤 중단): judge 파싱 실패 0/484 (equivalence) · 0/645 (ns_tools) · 재판정 시도 132 · 277, 0.5 보상 15 · 24 — 그중 잘린 추출 복구 3 · 10, 같은 답의 판정 뒤집힘 12 · 14 (표 중복을 뺀 값, 10-10 정정) → 재판정 끔 (사용자 결정, `RL_PLAN.md` 결정 27). j1c 를 J1 근거로 인정 (사용자 결정 10-10). j1c 에서 rdkit 예/아니오 행 채점 결함을 찾아 고쳤다 (`KNOWN_ISSUES.md`). 이하 사전 기록: 본체 수정은 J1 전에 `alpha/post-train` 에 병합했다 (`8a8544dcc`, 사용자 결정 — FAIL 이면 revert), 병합 트리 검사기 ERROR 0. 10-10 결정 25 뒤 smoke_v3 · v2 로 검사기 ERROR 0 (정답 잘림 검사 포함). 사전 검증: `alpha/judge-split` 단위 테스트 PASS (`2807ef2be`) · 레시피 검사 ERROR 0 · 음성 대조 4건 ERROR (`ab3bfe518`) · rdkit 플러그인 테스트 17건 PASS (`a38ad04bf`) |
| JG1 | General teacher 2노드 스모크 (`RL_PLAN.md` 결정 49~51): 실레시피 `teacher_general_alpha.yaml` — 외부 49B GenRM (`alpha_genrm_model`) + Gemma judge 5종 + 규칙 채점. 스모크 블렌드 `teacher_general_smoke_v1.jsonl` 192행 (17 층 × 약 12) · 검증 `teacher_general_smoke_v1_val.jsonl` 25행 (GenRM 없음) = 64 × 16 × 3 step, 저장 끔, W&B 오프라인 표 (`EXTRA_ENV="WANDB_MODE=offline" WANDB=1` · `logger.wandb.metric_allowlist=null` · `++logger.wandb.log_nemo_gym_full_result_tables=true`), 새 tag | 기동 전: `curl https://gemma4.withai.cj.net:10206/v1/models` 에 genrm · `tools/check_gym_judge_routing.py` (--expect multichallenge·inverse_if·abstention·jailbreak_detection·over_refusal_detection=judge_model, genrm_compare=genrm_model) ERROR 0. 판정: `tools/analyze_general_smoke.py --since <기동 epoch>` (분석기 음성 대조 `tools/negctl_general_smoke.py` 16/16) | P1 rc 0 · 학습 8장 한 노드, 롤아웃 6 + judge 2 다른 노드 · 기동부터 NeMo-RL `SETUP COMPLETE` 까지 60분 안 (Gym 서버 venv 8개 첫 빌드 · NeMo-Gym init 포함, 빌드 시간 미측정 — j1c 는 venv 가 있는 상태에서 Gym init 286 s) · `SETUP COMPLETE` 부터 첫 step 완료까지 60분 안 (step 0 검증 + 첫 step) · 직전 step 완료 뒤 30분 동안 다음 step 완료가 없으면 FAIL (외부 GenRM 무한 재시도 교착 검출 — 교착이면 step 줄이 찍히지 않는다; 근거: STEM j1c 같은 배치 32K step 490 s + 스모크 GenRM 비중 17% 의 비교 약 176건 ÷ 측정 13.6건/분 ≈ 13분 → 약 21분, 30분 ≈ 1.4배. GenRM 은 공유 인스턴스라 외부 부하가 붙으면 거짓 FAIL 가능) · P2 gen KL < 0.002 · 마스킹 ≤ 1% · P3 agent 16 + identity (그룹 기록) · P4 규칙 agent 잘리지 않은 표본 보상 고유값 ≥ 2 · P5a judge 실패 대체 행 0 · P5b 판정 행 ≥ 95% · P5c 엄격 파싱 ≤ 3% · P5d 안전 judge 서버 판정 == 최종 판정 라벨 (단독 라벨 줄 우선) 불일치 ≤ 0.5% · P5e 정책별 verifier 라벨 None ≤ 3% · P5f 정책별 jailbreak 경로가 정책 agent 대상 행의 ≥ 95% 에서 쓰임 · P6a GenRM 무승부 대체 ≤ 2% · P6b 어댑터 파싱 실패 ≤ 3% · P6c 순위 response_1 우세 0.2~0.8 · P6d 동점 ≤ 90% · P6e principle 주입 ≥ 0.5 × 16 × identity 그룹 · P6f 어댑터 파싱 성공 ≥ 0.9 × 16 × GenRM 그룹 | **사전 정의 (2026-11-06)**: 적대적 검토 1차 신뢰 불가 (judge 실패·GenRM 무승부가 0/0 으로 통과 · 검증 멈춤) → 2차 보완 필요 (실제 표 구조 가정 오류 · 안전 라벨 오독) → 3차 보완 필요 경미 (정책별 경로가 조용히 꺼지면 통과 · 반대 라벨 언급 거짓 FAIL · 상한 근거) → 4차 **신뢰 가능** (LOW: 기동 단계 상한 분리 · P5d 판정 단위 — 반영). **❌ FAIL 1/24 — 2026-11-06 jg1 (19:22 → 20:05, rc 0, commit `2971d1f58`)**: P1 ✅ 기동→SETUP 9분 · 첫 step 시작 19:53 · step 간격 4 · 3분 · 학습 sub1 8장, 롤아웃 main1 6장 + Gemma judge main1 · P2 ✅ gen KL 0.00158 · 0.00153 · 0.00169, 마스킹 0 · 0 · 0 (유효 시퀀스 1,024) · P3·P4·P5a·P5b ✅ · P5c ✅ multichallenge 5/512 · abstention 0/202 · 안전 0/2,416, **❌ inverse_if 121/836** · P5d 0/530 · P5e 0/1,886 · P5f 827/827 · P6a 대체 0 · P6b 3/780 · P6c 순위 [2, 52, 349, 325, 44, 5] response_1 우세 0.52 · P6d 동점 0.70 · P6e 주입 625 (하한 208) · P6f 777 ≥ 592. inverse_if 실패는 데이터-파서 불일치: 행 judge 지시문의 약 20% 가 `[JSON]: {"answer_score": …}` 형식을 요구해 Gym inverse_if 가 엄격 파싱을 못 하고 마지막 5줄 낱말 검색으로 대체 (120/120, PASS 46 · FAIL 74). `result` 형식 행은 715/716 (`KNOWN_ISSUES.md` 2026-11-06) |
| MS1 | MOPD1 2노드 스모크 (`RL_PLAN.md` 결정 55): 실레시피 `mopd1_alpha.yaml` 그대로 배치 L1 (학습 sub1 8 · 생성 vLLM main1 4 · teacher main1 4 × 1장, 가상 `num_nodes: 6`, judge 없음 `skip_verification`). 스모크 블렌드 `mopd_blends/mopd1_smoke_v1.jsonl` 192행 (`tools/build_mopd1_smoke_blend.py` — 본 블렌드에서 teacher 비율대로 190 + 렌더 프롬프트 57,261 · 56,836 토큰 swe_pivot 2행 = teacher 1장 메모리 탐침, 도구 슬롯) · 검증 16행. **teacher 슬롯**: General · STEM · 도구 = SFT 2862 를 경로만 바꾼 심볼릭 링크 (`$NRL_ROOT/mopd_smoke/teachers/sft_as_*` — 잡음 바닥, 경로 문자열이 달라 teacher 4개로 나뉘고 변환도 4번 = 본 런 기동 조건) · Code·Math = 실제 s2r 228 (음성 대조). 오버라이드 `grpo.max_num_steps=3 grpo.num_prompts_per_step=64 checkpointing.enabled=false env.should_log_nemo_gym_responses=false` (덤프 켬) · W&B 끔 · 새 tag | `runs/launch.sh mopd1 ms1` · 판정 `tools/analyze_mopd_smoke.py` (덤프 → teacher 슬롯별 차이) · main1 `nvidia-smi` 5초 표본 (GPU 별 최대 메모리) · 호스트 RAM 표본 | **PASS = 전부**: ① 3 step rc 0, 드라이버 로그에 teacher 4개 · `opd_teacher_nodes 4` · policy_nodes 2 ② 배치: main1 GPU 8장 = vLLM 4 + teacher 4 (teacher GPU 마다 ≥ 30 GB), sub1 8장 학습 ③ 잡음 슬롯 3개: |평균 차이| ≤ 0.01 nats/토큰, 평균 |차이| ≤ max(0.05, 3 × 평균 |학생 − 롤아웃|) ④ 실제 슬롯: 평균 |차이| ≥ 5 × 잡음 슬롯 최대, 평균 차이 < 0 ⑤ 긴 행 (총 ≥ 48K) 이 덤프에 있고 teacher advantage 가 0 이 아님 · teacher GPU 최대 메모리 < 76 GB ⑥ gen KL < 0.002 · 로그 오차 마스킹 ≤ 1% ⑦ 기록만 (기준 없음): is_oob_ratio (icepop [0.5, 2]) · teacher_logprob_time · 학생 logprob 패스 시간 · policy_training · exposed_generation · step 시간 · main1 호스트 RAM 최대 · OPD 지표 이름 (W&B allowlist 정리용). 분석기 음성 대조 (합성 덤프): 위치 한 칸 밀림 · 실제 슬롯 = 학생 · 긴 행 없음 · 모르는 agent · 긴 행 점수 0 → 5/5 FAIL, 정상 PASS | **MS1b (2026-11-09 17:14~17:42, commit `d2e0e90b5`, `EXTRA_ENV=UV_OFFLINE=1` — 첫 시도 ms1 은 uv 504 로 GPU 전 rc=2)**: ① 3/3 step 완료 ("Async GRPO training complete!"), teacher 4개 · `policy_nodes:2 + opd_teacher_nodes:4 = total_nodes:6` — **단 rc=1**: 완료 뒤 드라이버가 Ray 에 다시 접속하다 `core_worker_process.cc:89 unexpected system state` 로 죽음 (이전 런의 graceful shutdown 실패는 rc 0 이었다, OPD 에서 새로 생김) ② 배치 PASS — main1 GPU 4~7 = 생성 vLLM (17:16:10 동시 기동, 67~77 GiB), GPU 0~3 = teacher 4개 (17:19:44~17:21:12 teacher 초기화와 같은 간격으로 하나씩 30.7 GiB), sub1 8장 = 학습 (`ms1_layout.jsonl` 152 표본) ③ 잡음 슬롯 평균 차이 −0.0019 · −0.0012 · −0.0017, 평균 |차이| 0.040 · 0.039 · 0.030 (학생−롤아웃 0.025 · 0.027 · 0.019) PASS ④ 실제 슬롯 (s2r 228) 평균 차이 −0.278 · 평균 |차이| 0.514 = 잡음 최대의 12.7배, clip 0.5% PASS ⑤ 긴 행 총 57,324 · 56,921 토큰을 teacher 가 점수 냄 (생성 토큰 499/500 · 416/418 비0), teacher GPU 최대 58.5 GiB (긴 행 슬롯) · 나머지 50.6 GiB PASS ⑥ gen KL 0.00186 · 0.00174 · 0.00188 · 마스킹 0 PASS ⑦ is_oob_ratio (icepop [0.5, 2]) 0.00005 · teacher_logprob_time 8.7 · 9.1 · 2.2 s · 학생 logprob 패스 60.6 · 13.9 · 10.0 s vs 학습 54.8 · 35.9 · 25.8 s (step 2~3 기준 학습의 39%) · weight_sync 28.4~28.9 s · teacher 초기화 419 s (변환 4개) · main1 호스트 RAM 최대 288 GiB · 생성 평균 1.1만 토큰 · 잘림 20~27% · grad_norm 4.9~6.8 (clip 1.0) · OPD 지표 `train/on_policy_distillation/{teacher_student_logprob_gap_mean,adv_mean,adv_std}` · `train/teacher_logprob_time` · `timing/setup/teacher_init_time_s`. 덤프 판정 `tools/analyze_mopd_smoke.py` PASS. **적대적 검토 (Opus xhigh, 2026-11-09): 보완 필요** — 절차: 검토 완료 전에 기동했다 (사용자 지시 "스모크 테스트 진행해" 17:10, 검토 결과 17:45). B1 판정기가 첫 커밋에서 실제 덤프(배치 차원 1)를 못 읽었고 합성 대조가 평평한 형식이라 운영 경로를 시험하지 않았다 · sample_loss_mask 0 행을 섞었다 → 수정, 실제 형식 대조로 재시험. H1 ⑤ "긴 행 advantage ≠ 0" 은 teacher logprob 0 이어도 통과 → 긴 행마다 잡음 기준. H2 ④ 상한 없음 (균등 분포 teacher 가 PASS) → 실제 슬롯 clip ≤ 5% · 평균 ≥ −1. H3 ② `sample_gpu_layout.py` 는 nvidia-smi 호스트 pid 와 Ray 컨테이너 pid 가 달라 역할을 못 가린다 → `runs/check_placement.py` (배치 그룹 bundle → 노드), 학습 노드가 main1 · sub1 어느 쪽이어도 허용. M1 긴 행 57.3K < 최악 65,536 → 외삽 (59.6 − 31.5) × 65,536 / 57,324 + 31.5 ≈ 63.6 GB < 76. M2 실제 General · STEM · 도구 teacher 는 스모크에서 적재되지 않음 → 본 런 첫 step 게이트 MS2. M3 rc=1 은 실제 결함 — ① FAIL, 원인 미확정 (teacher 정리를 `TeacherWorkerGroup.__del__` 에 맡기는 것이 후보, `problem-diagnosis` 필요). **보완 후 재판정**: ③ 잡음 슬롯 평균 |차이| 0.040 [0.036, 0.044] · 0.039 [0.036, 0.042] · 0.030 [0.024, 0.035] PASS · ④ 실제 0.514 [0.497, 0.534] = 12.8배, clip 0.53%, 평균 −0.278 PASS · ⑤ 긴 행 두 개 각각 평균 |차이| 0.030 · 0.013 (잡음 수준) PASS · ② 로그 증거 PASS (`MegatronPolicyWorker[rank=0..7] ip=10.0.37.2` = sub1, vLLM · teacher 는 드라이버 노드 main1) · ⑥ PASS · **① FAIL (rc=1)**. 대조 (실제 덤프 형식, `$CLAUDE_JOB_DIR/tmp/review_ms1/`): 위치 밀림 · 실제 = 학생 · 긴 행 없음 · 모르는 agent · 긴 행 점수 0 · 긴 행 teacher 0 희석 · 균등 분포 teacher → 8/8 FAIL, 정상 PASS |
| MS2 | MOPD1 본 런 첫 step 게이트 (`mopd1/s1`, 결정 55 · MS1 검토 M2): 실제 teacher 넷 (General 640 · STEM 735 · Code·Math 228 · 도구 400) 이 처음 적재되는 자리다. 본 런에 덤프를 켠다 (`env.should_log_nemo_gym_responses=false`, 1,024 행 step 당 약 2.2 GB · 기록 약 36 s — 512 행 1.08 GB · 18 s 의 2배, 결정 56) | `runs/check_placement.py --actors-out <s1_teacher_actors.json>` (step 1 중) · `tools/analyze_mopd_smoke.py` (step 1~3 덤프) `--routing <mopd_blends>/mopd1_v1.routing.yaml --require-long 0 --real-slot <teacher 넷> --slot-floor <General>=0.044 --slot-floor <STEM>=0.042 --slot-floor <도구>=0.035 --slot-floor <Code·Math>=0.044 --driver-log mopd1/s1.log --teacher-actors <s1_teacher_actors.json>` · TensorBoard. 기동은 `EXTRA_ENV="UV_OFFLINE=1 RAY_DEDUP_LOGS_ALLOW_REGEX=successfully.loaded.checkpoint"` — 경로에 숫자가 있으면 Ray 줄 중복 제거가 teacher 적재 줄을 묶어 지운다 (검토 4차 M1, ray_logging `__init__.py:284`). 기동 전: teacher 변환 캐시가 없거나 HF safetensors 보다 새것 (검토 4차 L4, 11-09 18:25 확인 — General·STEM·도구 없음, Code·Math 11-09 17:23 > HF 10-24) | **FAIL 이 하나라도 있으면 멈춘다**: ① 배치 — 학습 한 노드 8 · 생성 다른 노드 4 · teacher 4 × 1 이 생성 노드 (학습 노드는 main1 · sub1 어느 쪽이든) ② teacher 신원 — step 1 에 저장한 teacher 액터 (이름 · pid · 노드) 를 드라이버 로그의 실제 적재 줄 `successfully loaded checkpoint from …mcore_ckpt/model_<경로>` 와 (노드, pid) 로 맞춰, teacher 마다 적재 경로 집합 = {routing 경로} (teacher 끼리 뒤바뀜 · 학생 경로 적재를 잡는다, 검토 2·3차 M3). **신원 증거 부족 (줄·액터 미결합) 도 FAIL = 멈춘다** (검토 3차). 액터는 routing 의 서로 다른 경로를 빠짐·중복 없이 덮어야 하고, MS2 방식은 신원 인자가 필수다 (검토 4차 L1·L2) ③ teacher 손상 — 학생 확신 토큰 (prev ≥ −0.01) 중 차이 < −1 비율 ≤ 5% (s2r 228 실측 0.03% · 균등 분포 ≈ 100%) · clip 경계 비율 ≤ 50% (검토 2차 M1) ④ routing 슬롯 전부 지정 · routing 에 없는 agent 0 · teacher 슬롯마다 생성 토큰 ≥ 2,000 · sample_loss_mask 0 행 ≤ 1% (3 step 합) ⑤ gen KL step 마다 < 0.002 (R1 기준) · OOM · teacher 크래시 없음. **증거 부족 (멈추지 않고 보고)**: 슬롯별 평균 |차이| 부트스트랩 95% 하한 ≤ 그 슬롯 잡음 바닥 (MS1 잡음 슬롯 95% 상한 — 검토 2차 M2) = SFT 와 구분 안 됨. ② 신원이 맞으면 SFT 에 가까운 정상 teacher 로 보고 학습을 잇되 보고한다 (도구 teacher 가 후보: 학습 중 엔트로피 변화 −18%). 전체 평균 차이는 기록만 한다. 음성 대조 (실제 ms1b 덤프 변형, `$CLAUDE_JOB_DIR/tmp/ms1_ctl/`): 위치 밀림 · 실제 = 학생 · 균등 분포 teacher · 실제 행 30% 손상 · 긴 행 teacher 0 · 모르는 agent · 긴 행 없음 → 7/7 FAIL; 슬롯 하나 누락 → FAIL; 신원 (ms1b 로그 변형, `$CLAUDE_JOB_DIR/tmp/ms2_id/`): 정상 · 캐시 줄 추가 · 노드 뒤집힘 → PASS, pid 뒤바꿈 · 다른 경로 적재 · 학생 경로 추가 적재 · 적재 줄 하나 삭제 · 노드 불일치 · 액터 없음 · 액터 파일 없음 · 액터 3개 · 중복 액터 · MS2 신원 인자 누락 → 10/10 FAIL, `[rank=N]` 접두 → PASS (로그 접두 pid = 상태 API pid, 두 노드 실측); MS1 덤프를 MS2 방식으로 보면 SFT 슬롯 셋 = 증거 부족, s2r 228 통과 | **PASS (2026-11-09 21:03, `mopd1/s1` step 1~3, 출력 `results/alpha/mopd1/s1_ms2.txt`)**: ① 배치 PASS — 학습 main1 8 · 생성 sub1 4 · teacher sub1 4×1 ② 신원 4/4 — 적재 줄 5줄 전부 남음 (`runs.log` `extra_env=UV_OFFLINE=1 RAY_DEDUP_LOGS_ALLOW_REGEX=successfully.loaded.checkpoint`), pid·노드·경로 일치 ③ 확신 토큰 손상 0.030~0.102% · clip 0.33~0.58% ④ 미지정 슬롯 0 · 모르는 agent 0 · 슬롯 토큰 최소 140,090 · 마스킹 0/3,072 ⑤ gen KL 0.0018 · 0.0018 · 0.0018 · 크래시 0. 슬롯 평균 |차이| [95% CI]: Code·Math 0.499 [0.494, 0.503] · General 0.458 [0.443, 0.473] · STEM 0.483 [0.474, 0.492] · 도구 0.235 [0.217, 0.255] — 전부 잡음 바닥 0.035~0.044 위 (증거 부족 없음). 행 비율 29.5 / 30.4 / 29.7 / 10.4% (블렌드 30 / 30 / 30 / 10). 게이트 이력: 적대적 검토 4차 "보완 필요" (신원 거짓 PASS 경로 없음 · 거짓 FAIL 1 · 운영 구멍 2) → 반영 `bd27e7b4d` → 재확인 신뢰 가능 (11-09 18:35). 남은 LOW: 분석기는 기동 환경변수를 스스로 확인하지 못한다 (빠뜨리면 거짓 FAIL 쪽) → 판정 기록에 `runs.log` 의 `extra_env=` 줄을 인용한다 · 레시피 agent ⊆ routing 은 검사기가 보지 않는다 (어긋나면 거짓 FAIL 쪽) |

**G8 상세 (2026-10-08, 커밋 `8d7bc8832`)**: 다음을 확인했다.
- logprob 단계 제거: `force_on_policy_ratio` 와 `seq_logprob_error_in_loss` 가 함께 켜지면 prev_logprobs 계산을 건너뛴다 (`nemo_rl/algorithms/opd.py:115`).
  `policy_and_reference_logprobs` 가 3스텝 모두 0.00 s 다 (v1 구간 1 은 스텝당 262~411 s). 마스킹 판정은 학습 forward 안에서 돈다 (`seq_logprob_error_valid_seqs` 128/128).
- chunked prefill: 생성 엔진 8개 모두 `Chunked prefill is enabled with max_num_batched_tokens=16384`. GPU 당 KV 용량은 2,905,088 토큰이다 (v1 의 1,572,864 토큰의 1.85배).
- effort shaping: `mean_length_reward_low` 0.003/0.011/0.022 가 기록됐다. low 응답(14.5K/13.3K/11.7K 토큰)이 high(6.3K/5.9K/4.3K)보다 길다.
  E0 데이터에서 마커가 code_gen 12·math 6·mcqa 6 행에만 붙고, 마커 없는 행은 도구 호출·지시 따르기 위주라서다 (과제 구성 차이, 결함 아님).
- 스텝 시간 165/83/60 s (학습 132/49/26 s, 동기화 28.4 s). 소규모라 속도 판정에는 쓰지 않는다 — 속도는 E1 A/B 로 본다 (`RLVR_READINESS.md` §5.3).
- 런마다 Gym 서버 22개가 남는다 (`KNOWN_ISSUES.md` 2026-10-08). v2 실행기가 다음 기동 전에 정리한다. 종료 때 `NemoGym.shutdown` 재시도 경고는 `ray.kill` 뒤라 무해하다.

**G7 상세 (2026-10-07, 16 프롬프트 × 8, 생성 최대 16K)**: 재개한 런이 이어받은 것은 다음과 같다.
step 카운터(Step 4/5), `consumed_samples` 48 → 80, Gym 작업 인덱스(G7a 마지막 79 → G7b 새 롤아웃 80부터, 중복 없음), lr warmup(3.7e-7·4.6e-7).
replay buffer 32 궤적도 이어받았다. 저장 전에 생성된 궤적을 step 4 학습에 썼고(평균 나이 1.00), R3 route 재생이 그 궤적에 적용됐다 (prev-logprob·train 재생 1,920건씩).
그 궤적의 KL 0.0019 는 저장 전 step 3 과 같다 — 재개한 가중치가 저장 시점 가중치와 같다는 뜻이다.
옵티마이저 상태는 `verify_optimizer_resume.py` 로 봤다 (GDN 층 0·attention 층 3·임베딩·출력·최종 norm 의 83개 키 중 62개 판정).
상태는 0 에서 시작해 lerp 로 갱신된다 (Muon `momentum_buffer.lerp_(g, 0.05)`, Adam β1 0.9·β2 0.95). 그래서 초기화는 "0 에서 한 스텝" 이 되고 아래 기대값이 나온다.

| 상태 (판정 키) | 연속 2→3 | 재개 3→4 | 연속 4→5 | 판정 기준 · 초기화됐을 때 |
|---|---|---|---|---|
| fp32 master 누적량 R 비율 (28) — bf16 아래 갱신의 누적 | 1.60~2.12 | 1.41~1.64 | 1.32~1.50 | ≥ 1 · 한 스텝 크기로 떨어져 < 1 |
| Muon momentum cos (19) | 0.95~0.99 | 0.86~0.95 | 0.94~0.98 | ≥ 0.5 · 다른 상태면 ≈ 0 |
| Muon momentum norm 비율 (19) | 0.95~1.01 | 0.99~1.22 | 0.89~1.04 | ≥ 0.7 · 0.4~0.6 |
| Adam exp_avg_sq 합 비율 (15) | 0.96~1.10 | 0.99~2.03 | 0.95~1.10 | ≥ β2 0.95 (정확한 하한) · ≈ 0.35 |

재개 구간의 Muon cos 가 낮은 것은 step 4 의 gradient 가 컸기 때문이다 (grad norm 0.045 → 0.073, 1.62배).
EMA momentum 에 1.6배 크기의 직교 gradient 가 들어오면 cos 는 0.857 로 계산된다. 실측 최소는 0.862 다.
첫 판정 기준(연속 구간과의 유사도)은 3/77 을 BAD 로 냈다. 두 norm 텐서의 exp_avg_sq 가 ×2.03 으로 **늘었는데**, 이는 초기화와 반대 방향이다.
그래서 판정을 실패 양상별 신호(0 초기화·master 재생성·다른 상태)로 바꿨다. 유사도는 참고(UNUSUAL 3)로 남겼고, Adam exp_avg 는 정상·초기화 범위가 겹쳐 참고만 한다.
음성 대조: 다른 런(G5)의 step_2·step_3 뒤에 G7a 의 step_2 를 이으면 50/62 신호로 FAIL 한다 (momentum 19/19, master 28/28, exp_avg_sq 3/15).
Pai 의 `NCCL_MAX_NCHANNELS=16`(옵티마이저 상태 재개 OOM 회피) 없이도 재개 OOM 은 없었다.
타이밍: G7a 22분(3스텝 + 저장 2회), G7b 18.6분(기동 + 2스텝 + 저장 2회). 산출물: `$NRL_ROOT/gates/two_node/` (`g7a_fresh*`·`g7b_resume*`, `g7_optimizer_resume.json`, `g7_optimizer_negative_control.json`).

**G6 상세 (2026-10-07, G5 step_3 = 3스텝, warmup lr 1e-7~3e-7)**: 학습 행렬은 전부 바뀌었다 (비트 동일 0개, 최대 절대 변화 9.5e-7 = 2^-20, 상대 2e-7~1e-6).
1차원 파라미터(norm·`dt_bias`)는 갱신이 bf16 해상도보다 작아 그대로다. 라우터·expert bias 는 동결(`freeze_moe_router`, bias 갱신 0)대로 비트 동일이다. 같은 크기의 다른 행렬은 전부 바뀌었으니 우연이 아니다.
Pai forward (`compare_hf_forward.py --control-sigmas 0,1e-7,1e-6`, transformers 4.57, GPU 2장)의 차이는 잡음 바닥과 같다.
같은 가중치는 GPU 간 비트 동일이다(σ=0). 그런데 실현 변화가 3스텝 갱신의 1/40 인 무작위 잡음(σ=1e-7)도 도구 대화에서 같은 크기의 차이를 낸다. bf16 + MoE 경계 라우팅이 아주 작은 섭동을 키운다.
같은 시작점도 프로세스를 바꾸면 ppl 이 달라진다 (chat_think 15.14 vs 14.87 — Triton 자동 튜닝 추정).

| 텍스트 (토큰) | ppl 시작점 → 반출 | 반출 평균 \|Δlogp\| · KL · top-1 일치 | 통제 σ=1e-7 (실현 6e-9) 평균 \|Δ\| · KL | 통제 σ=1e-6 (실현 8e-8) 평균 \|Δ\| · KL |
|---|---|---|---|---|
| en_facts (45) | 6.38 → 6.48 | 0.098 · 0.013 · 0.909 | 0 · 0 | 0.068 · 0.024 |
| ko_facts (31) | 9.89 → 10.03 | 0.053 · 0.0042 · 1.000 | 0.021 · 0.0010 | 0.039 · 0.0027 |
| chat_think (55) | 14.87 → 14.44 | 0.073 · 0.012 · 0.963 | 0.033 · 0.0035 | 0.060 · 0.010 |
| chat_tool (350) | 49.97 → 47.75 | 0.441 · 0.137 · 0.883 | 0.379 · 0.135 | 0.401 · 0.157 |

Pai 서빙 (`serve_alpha.sh` TOOLS=1 · `nemotron_v3`, temperature 0, 시작점·반출을 GPU 1장씩): 6/6 요청의 finish 가 같다 (stop 5 · tool_calls 1).
6/6 이 `<|im_end|>`(3)에서 멈췄고, 도구 호출 `get_weather(city=Seoul)` 가 파싱됐다. 4/6 은 토큰까지 같다. fact_on 은 52/101, math_on 은 135/341 토큰에서 갈라졌다 (위 잡음 바닥).
Pai `forward_sanity.py` 는 ppl 6.48 로 PASS 다 (기준 100). 반출은 CPU 에서 2분 17초, 대조는 1분 11초 걸렸다.
산출물: `$NRL_ROOT/gates/export_resume/` (`hf_g5_step3_v2.compare.json`, `g6_pai_forward_control.json`, `g6_serve.json`, 로그). HF 반출본 3개(90 GB)는 2026-10-07 판정 기록 뒤 삭제.

**G5 상세 (2026-10-07)**: 위치별 k3 — step 1 [0,256) 0.00130 → [4K,8K) 0.00197 → [8K,16K) 0.00178. 환경별 마스킹 0 (도구 환경 k3 0.0008~0.0016).
KL 이 스텝마다 조금 오른다(0.00178 → 0.00191) — async 에서 궤적 일부가 한 스텝 전 가중치로 생성되고 3스텝은 생성 토큰이 80만으로 많다. 본 런에서 계속 본다.
step 1 은 갱신 전이라 엔진 차이만 담는다: CUDA graph(0.0018)가 eager(G3 0.0016~0.0017)보다 약간 크다 (M4 확인, 기준 이내).
타이밍 (이 스모크 규모): refit 28.4 s · 롤아웃 대기 `exposed_generation` 3~4 s (1.5%) · prev logprob 82~91 s · 학습 87~90 s. 이 규모에서는 학습측이 병목이다.
드라이버에서 노드 고유 환경변수(`NVIDIA_VISIBLE_DEVICES`·`HOSTNAME`·`BACKENDAI_*`)를 빼고 띄웠다. NCCL·Gloo 는 `eth0` 고정 (두 노드 `eth1` 은 같은 172.18.0.3).
산출물: `$NRL_ROOT/gates/two_node/` (`g5_smoke_b.log`, `g5_smoke_b/exp_001/train_data_step{1,2,3}.jsonl`, `g5_smoke_b/ckpt/step_{2,3}`).

**G3 상세 (2026-10-07, main1·sub1, 32K·CP4·생성 최대 8K, 16 프롬프트 × 4)**: 스모크에서 레시피·코드 결함 5건을 찾아 고쳤다 (`KNOWN_ISSUES.md` 2026-10-07).
① ES + colocated IPC refit 불가 ② 첫 턴 길이 초과 시 런 중단 ③ chunked prefill 끔 + batched tokens < max_model_len 기동 실패
④ NeMo-RL 의 None content `TypeError` (upstream 수정 이식 `12e4b78df`) ⑤ vLLM 의 strict 도구 제약 디코딩 (도구 환경 KL 0.0045).
환경별 보상: toolcall_schema 0.35~0.46 · swe_pivot 0.25~0.30 · mcqa 0.75 — 도구 호출이 파싱돼 채점까지 간다. math·code_gen 은 생성이 전부 8K 상한에서 `</think>` 전에 잘려 0 이다 (스모크 상한 탓).
R3 forward 검증 3,840건 불일치 0. 산출물: `$NRL_ROOT/gates/gym_smoke/` (`g3_dump*`·`g3_final`·`g3_rates`).

**G2 상세 (2026-10-07, 4K, step 1 = 같은 롤아웃 128 샘플·생성 토큰 444,238)**

| 학습 경로 | k3 KL | \|Δ\|>0.5 토큰 | 위치 0–256 / 1k–2k / 3k–4k | R3 forward 검증 |
|---|---|---|---|---|
| R1 (CP1, packing 끔) | 0.00148 | 0.011% | 0.00132 / 0.00148 / 0.00150 | — |
| CP1 + packing | 0.00148 | 0.010% | 0.00131 / 0.00148 / 0.00151 | 5,760건 불일치 0 |
| CP8 + packing (+ chunk 2048·fuse_loss) | 0.00151 | 0.010% | 0.00131 / 0.00152 / 0.00155 | 43,392건 불일치 0 · CP 토큰 일치 16,384행 |

**G2 32K (2026-10-07, sub1, 8 프롬프트 × 8, step 1 생성 토큰 1,073,258, 잘림 25/64)**: k3 KL 0.00162 · \|Δ\|>0.5 0.006% ·
위치 [0,256) 0.00132 · [1K,2K) 0.00157 · [4K,8K) 0.00162 · [8K,16K) 0.00163 · [16K,32K) 0.00166. GDN 상태 bf16 일 때(4K 안에서 2배)와 달리 32K 까지 평탄하다.
분석: `tools/analyze_rollout_logprob_gap.py --max-len 32768` (위치 구간을 길이에 맞춰 늘리는 옵션 추가).

`check_r3_trace.py` 의 producer↔fetch 대조("no rollout_payload_sample")는 비동기 replay buffer·data-plane 경로 전용 기록이라 동기 `grpo_train` 에서는 생기지 않는다 — G5 에서 확인한다.
replay 할당(prev-logprob·train), backward replay(full recompute), forward 검증은 전부 기록·일치했다.

G1 참고 수치: 같은 B 를 단독 packed·unpacked·묶음 안 오프셋으로 바꾸면 B logprob mean\|Δ\| 0.05~0.09, max 1.4~3.0 nat 이다 (일부 쌍은 비트 동일).
입력 내용이 아니라 텐서 모양(패딩 길이)이 바뀌어 GEMM 선택·MoE 경계 라우팅이 달라지는 잡음이다. M5 의 구현 간 잡음 바닥(0.067~0.069)과 같은 크기다.
산출물: `$NRL_ROOT/gates/packing_isolation/` (`run_g1.sh`, `g1_fla.json`·`g1_flashqla.json`·`.log`).

## 환경 (E)

| # | 게이트 | 기준 | 결과 |
|---|---|---|---|
| E1 | GRPO 퀵스타트 (Qwen2.5-1.5B, DTensor+vLLM / **mcore**+vLLM), 1 GPU 2 step | 정상 스텝 · mcore KL ~7e-4 | 2026-08-13 RL 전용 세션: DTensor 57 s/step · mcore 44–50 s/step, KL ~7e-4 · 2026-10-06 공존 설치(alpha/post-train `83976fe0d`, main1) `grpo_math_1B_megatron.yaml` colocated: **KL 0.0007 / 0.0006**, 53.6 / 57.4 s/step, Traceback 0, 종료 후 Ray·GPU 메모리 전부 회수 |
| E2 | venv 빌드 | vllm·mcore extra 각각 성공, 플러그인 등록 | 2026-10-06 vllm venv(deep-ep·deep-gemm 소스 빌드 65 s, torch 2.11.0+cu130, vLLM 0.25.1, `AlphaForCausalLM` 등록 True) · mcore venv 683 s (TE·DeepEP·mamba-ssm·causal-conv1d·fast-hadamard) + TE 재빌드 282 s(#15) |
| E3 | TE fused attn (cuDNN sub-backend 1) | Megatron 워커 import 체인 뒤 fwd/bwd 통과, 매핑 cudart = cu13 만 | 2026-10-06 max\|o−ref\| 0.0080 (fp32 SDPA 대비), grad finite |
| E4 | Gym v0.6.0 | CPU 테스트 · alpha 설정 validate | 2026-10-06 CPU 테스트 69 pass · alpha 설정 4/4 · PivotRL 서버 298 pass |

미검증: 2노드 Ray 스모크, 사설 경로 영속화 구성의 재생성 후 동작, FlashQLA E2E 스텝 이득, packing+CP 경로의 KL·엔진 동등성, `expandable_segments` 의 refit 비용.
