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
| Pai 또는 NeMo-RL 의 mcore·TE 버전, 학습 forward 설정(라우터 dtype·fusion 등) | M5 |
| chat template·토크나이저, Gym 버전·alpha Gym 서버, `openai_server_utils.py`·vLLM 서빙 계층, 레시피 `turn_end_token_id`·`chat_template_kwargs` | R3 |
| 컨텍스트 길이·CP·EP, logprob 설정(`logprob_chunk_size`·`fuse_loss`·`defer_fp32_logits`), 할당자(`PYTORCH_CUDA_ALLOC_CONF`), 옵티마이저 | R4 |
| 환경 재구축 (venv·툴킷·lock) | E1 ~ E3 |
| GDN 커널·packing 코드(mcore GDN·NeMo-RL `data.py` packing)·`ALPHA_GDN_BACKEND` | G1 |
| 레시피 (alpha GRPO 레시피 전부) | `tools/check_alpha_recipe.py` (CPU, ERROR 0 이어야 실행) |

## 모델 정합성 (M)

| # | 게이트 | 도구 | 기준 | 결과 |
|---|---|---|---|---|
| M1 | 가중치 라운드트립 HF→mcore→HF | `tools/verify_bridge_roundtrip.py` | 텐서별 `max_diff==0.0` + 양방향 커버리지 (`A_log` 만 문서화된 dtype 예외) | 2026-08-13 base **14,181/14,181** · 2026-10-06 agentic iter2400 **14,181/14,181** |
| M2 | forward 로짓 패리티 (mcore vs Pai 검증 HF 참조) | `tools/verify_forward_parity.py` + `tools/gen_hf_reference_logits.py` | 영/한/887토큰 3종 argmax·top5 일치 + cos ≥ 0.99 | 2026-08-13 cos ≥ 0.99988 · 2026-08-18 FlashQLA 활성 3종 PASS (TileLang 실행 확인) · 2026-10-06 iter2400 argmax 3/3, cos ≥ 0.99984 |
| M3 | mcore→vLLM refit (토큰별 logprob) | upstream `tools/refit_verifier.py` (우리 수정 `1eca2f383`) | mean(exp\|Δ\|) = mult_prob_err < 1.05 | 2026-08-13 mean diff 0.020 / max 0.122 (≈1.02) · 2026-10-06 iter2400 **1.0293** |
| M4 | vLLM 단독 서빙 (디스크 직접 로드) | `tools/verify_vllm_serving_parity.py` | 전체 vocab(163,968) next-token 분포: 3종 argmax 일치 + cos ≥ 0.99 (KL 은 보고) | 2026-08-24 argmax·top5 일치, cos ≥ 0.99997, KL(HF‖vLLM) ≤ 0.0013 |
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
| R1 | GRPO Generation KL (rollout vLLM vs 학습 mcore), 8-GPU 1노드 | `grpo_alpha_smoke.yaml` (Muon 은 `grpo_alpha_smoke_muon.yaml`), 진단 `tools/analyze_rollout_logprob_gap.py` | step 1·2·3 의 `Generation KL Error` < 0.002 | 2026-10-06 iter2400: 기본 0.0042 FAIL → R3 0.0026 → **R3 + GDN 상태 fp32 Adam 0.0015/0.0013/0.0014 PASS · Muon 0.0015/0.0013/0.0013 PASS** |
| R2 | Muon 실적용 (워커와 같은 setup 경로로 옵티마이저 직접 검사) | `tools/verify_muon_optimizer.py` | 4항목 전부 PASS: Muon·Adam 클래스 공존 · 파라미터 분배 · qkv 4-way split · 하이퍼파라미터 레시피 일치 | 2026-10-06 PASS — TensorParallelMuon 15.34B · Adam 0.67B · qkv 4-way · extra_scale 0.2 · nesterov · router 동결(RL 기본) |
| R3 | chat 렌더 패리티 — RL 프롬프트 토큰 ID 가 SFT 변환기(`build_alpha_sft_idxmap.py`)의 학습 토큰 ID 와 정확히 같은가 (CPU) | `tools/verify_chat_render_parity.py` | 경로별 토큰 ID 완전 일치 | 2026-10-07: 네이티브 GRPO **PASS** · Gym 멀티턴 경로 **부분** — 턴 경계 수정 반영, `strict` 유지 미구현 (아래 표) |
| R5 | 학습 처리량·recompute 변형 (`tools/measure_train_memory.py --samples-per-dp N --recompute ...`) | 같은 도구 | 정상 상태(3번째 스텝) 처리량·peak. 판정 게이트가 아니라 기준선 | 2026-10-07 부분 실행: 128K/CP8 full **33.4K tok/s**·peak 57.2 GB, selective 전 변형 OOM (`RLVR_READINESS.md` §5). 나머지는 첫 RLVR 런 뒤 (사용자 지시) |
| R4 | 학습 스텝 메모리 실측 — NeMo-RL 실제 학습 경로(`MegatronPolicyWorkerImpl.train` + `ClippedPGLossFn` + Muon step), 학습 전용 노드 가정(vLLM 없음) | `tools/measure_train_memory.py` | 80 GB 안에 2 스텝 완주 (1 스텝째 Muon momentum 지연 할당 포함) | 2026-10-07 sub1: 1노드 **128K/CP8 OK**, 단 `expandable_segments` 필요 (아래 표) |

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
