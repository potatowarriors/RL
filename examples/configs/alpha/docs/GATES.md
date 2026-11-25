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
| `alpha_web_search_mcqa` 플러그인 · 설정, 검색 객관식 행 파일 (블렌드), Tavily 키 · 요금제 | WS1 (CPU). `tools/web_search_probe.py` · `wsm/run_wsm_probe.sh` 를 바꾸면 자가 시험 + `DRY=1` 예행 |

## 모델 정합성 (M)

| # | 게이트 | 도구 | 기준 | 결과 |
|---|---|---|---|---|
| M1 | 가중치 라운드트립 HF→mcore→HF | `tools/verify_bridge_roundtrip.py` | 텐서별 `max_diff==0.0` + 양방향 커버리지 (`A_log` 만 문서화된 dtype 예외) | 2026-08-13 base **14,181/14,181** · 2026-10-06 agentic iter2400 **14,181/14,181** · 2026-10-08 agentic iter2862 **14,181/14,181** · 2026-10-19 RL 반출 `teacher_codemath2/hf_step_00282` **PASS** (`$NRL_ROOT/gates/codemath2_step282`, MOPD1 학생 재사용 2026-11-10) · 2026-11-12 RL 반출 `mopd1/hf/hf_step_00120` **PASS** (`$NRL_ROOT/gates/mopd1_step120`) |
| M2 | forward 로짓 패리티 (mcore vs Pai 검증 HF 참조) | `tools/verify_forward_parity.py` + `tools/gen_hf_reference_logits.py` | 영/한/887토큰 3종 argmax·top5 일치 + cos ≥ 0.99 | 2026-08-13 cos ≥ 0.99988 · 2026-08-18 FlashQLA 활성 3종 PASS (TileLang 실행 확인) · 2026-10-06 iter2400 argmax 3/3, cos ≥ 0.99984 · 2026-10-08 iter2862 argmax 3/3, cos ≥ 0.99980 · 2026-10-19 RL 반출 `teacher_codemath2/hf_step_00282` **PASS** (`$NRL_ROOT/gates/codemath2_step282`, MOPD1 학생 재사용 2026-11-10) · 2026-11-12 RL 반출 `mopd1/hf/hf_step_00120` argmax 3/3, cos ≥ 0.99937 **PASS** |
| M3 | mcore→vLLM refit (토큰별 logprob) | upstream `tools/refit_verifier.py` (우리 수정 `1eca2f383`) | mean(exp\|Δ\|) = mult_prob_err < 1.05 | 2026-08-13 mean diff 0.020 / max 0.122 (≈1.02) · 2026-10-06 iter2400 **1.0293** · 2026-10-08 iter2862 **1.0228** (mean diff 0.022 / max 0.111) · 2026-10-19 RL 반출 `teacher_codemath2/hf_step_00282` **FAIL 1.07** (mean diff 0.068 / max 0.159, 10토큰 중 8개 0.03~0.16) — 진단 (같은 날 08:43~09:11): 대조군 iter2862 재실행 **1.025** (0.0249, 환경 변화 없음) · step_282 재실행 0.0558 (≈1.057, 재현) · 같은 ckpt 의 R1(R3 켬) **KL 0.0010 × 3 · mult_prob_err 1.018 · \|d\|>1 토큰 0**. 결론: M3(R3 없음, 10토큰)의 초과는 RL 로 라우팅 경계가 민감해진 체크포인트의 성질이고 refit 경로 결함이 아니다 (KNOWN_ISSUES 10-16). 사용자 결정 2026-10-19: RL 반출 시작점은 M3 대신 R1 로 판정 (재실행 조건 표) |
| M4 | vLLM 단독 서빙 (디스크 직접 로드) | `tools/verify_vllm_serving_parity.py` | 전체 vocab(163,968) next-token 분포: 3종 argmax 일치 + cos ≥ 0.99 (KL 은 보고) | 2026-08-24 argmax·top5 일치, cos ≥ 0.99997, KL(HF‖vLLM) ≤ 0.0013 · 2026-10-08 iter2862 argmax 3/3 (top5 1/3), cos ≥ 0.99995, KL ≤ 0.0074 — KL 이 8월보다 크고 원인은 미확인 (GDN 상태를 fp32 로 바꿔도 같은 값). RL 경로 정합은 R1 이 본다 · 2026-10-19 RL 반출 `teacher_codemath2/hf_step_00282` **PASS** (`$NRL_ROOT/gates/codemath2_step282`, MOPD1 학생 재사용 2026-11-10) · 2026-11-12 RL 반출 `mopd1/hf/hf_step_00120` argmax 3/3, cos ≥ 0.99993, KL ≤ 0.0081 **PASS** |
| M5 | SFT 엔진(Pai Megatron-LM-251125) ↔ RL 엔진(NeMo-RL mcore) forward·gradient 동등성 | `tools/engine_parity_{pai,nemorl,hf,compare}.py` (+ `engine_parity_common.py`) | Pai↔NeMo-RL 거리가 bf16 노이즈 바닥 이내. 바닥은 두 기준선으로 잰다: HF 를 제3 구현으로 둔 삼각측량, 같은 Pai 엔진에 라우터 dtype 만 바꾼 섭동 런 | 2026-10-07 iter2400 **PASS** (아래 표) |

M2 의 참조 로짓은 **Pai 환경(transformers 4.57)** 에서 만든다 — `modeling_alpha.py` 가 transformers 5.x 와 비호환(OutputRecorder)이다.
M1·M2 는 반드시 **함께** 돌린다. zero-centered RMSNorm 델타는 M1 을 통과하고 M2 에서만 잡힌다 (`archive/KNOWN_ISSUES_2026-08_10-07.md`).

**M5 상세 (2026-10-07, sub1 8-GPU, EP8, 고정 배치 2×2048 토큰, 같은 iter2400 가중치).**
두 엔진은 모델 클래스와 파라미터 이름이 다르다(Pai MambaModel 48층 + megatron_patch GDN, NeMo-RL GPTModel 24층 + mcore GDN).
그래서 텐서를 이름이 아니라 가중치 비트의 정수 해시 지문으로 짝짓는다. 판정은 절대 임계가 아니라 기준선 비교다.
MoE·bf16 에서는 같은 엔진끼리도 gradient cos 가 0.98 근처에서 포화한다 (`archive/KNOWN_ISSUES_2026-08_10-07.md` 2026-10-07 엔진 동등성 항목).

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
| R1 | GRPO Generation KL (rollout vLLM vs 학습 mcore), 8-GPU 1노드 | `grpo_alpha_smoke.yaml` (Muon 은 `grpo_alpha_smoke_muon.yaml`), 진단 `tools/analyze_rollout_logprob_gap.py` | step 1·2·3 의 `Generation KL Error` < **0.001** (2026-11-25 사용자 결정 — 그 전 기록은 0.002 기준으로 판정한 것이다. 아래 "기준의 출처") | 2026-10-06 iter2400: 기본 0.0042 FAIL → R3 0.0026 → **R3 + GDN 상태 fp32 Adam 0.0015/0.0013/0.0014 PASS · Muon 0.0015/0.0013/0.0013 PASS** · 2026-10-08 iter2862 Muon **0.0014/0.0013/0.0013 PASS** · 2026-10-11 `vllm_kwargs.gdn_prefill_backend: triton` + 학습 `NCCL_MAX_NCHANNELS=16` 재실행 (본 런 `main2r3` 의 `ckpt/step_40` 재개 step 41·42·43, 32K 생성·CP8·packing) **0.00177/0.00180/0.00198 PASS** · 마스킹 0/0/0 (이전 step 21~40 은 0.00177~0.00193) · 2026-10-19 RL 반출 `teacher_codemath2/hf_step_00282` (Muon) **0.0010/0.0010/0.0010 PASS** · mult_prob_err 1.018 · 위치별 평탄 · \|d\|>1 토큰 0 · 2026-11-12 RL 반출 `mopd1/hf/hf_step_00120` (Muon) **0.0006/0.0007/0.0006 PASS** (M3 기록 mean diff 0.053 / max 0.219) |
| R2 | Muon 실적용 (워커와 같은 setup 경로로 옵티마이저 직접 검사) | `tools/verify_muon_optimizer.py` | 4항목 전부 PASS: Muon·Adam 클래스 공존 · 파라미터 분배 · qkv 4-way split · 하이퍼파라미터 레시피 일치 | 2026-10-06 PASS — TensorParallelMuon 15.34B · Adam 0.67B · qkv 4-way · extra_scale 0.2 · nesterov · router 동결(RL 기본) |
| R3 | chat 렌더 패리티 — RL 프롬프트 토큰 ID 가 SFT 변환기(`build_alpha_sft_idxmap.py`)의 학습 토큰 ID 와 정확히 같은가 (CPU) | `tools/verify_chat_render_parity.py` | 경로별 토큰 ID 완전 일치 | 2026-10-07: 네이티브 GRPO **PASS** · Gym 멀티턴 경로 **부분** — 턴 경계 수정 반영, `strict` 유지 미구현 (아래 표) · **2026-11-25 재실행** (alpha 모델 서버 플러그인 변경, `RL_PLAN.md` 결정 78; `--ckpt results/alpha/mopd1/hf/hf_step_00120`): P0 22/0 · P1 12/0 · P2 22/0 · P4 0/40 · P5 8/32 · P6 36/4 — 10-07 표와 같음 (산출물 `$NRL_ROOT/p0_measure/r3_render_20261125b/`). 도구의 기본 `CKPT` 경로는 스토리지 정리로 없어져 `--ckpt` 를 줘야 한다 (없으면 stage-sft 가 종료 2) |
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
| P4 Gym → vLLM + `replace_prefix_tokens` (수정 전, 경계 = EOS 0) | 0 / 40 | FAIL — 2번째 호출부터 전부 (`archive/KNOWN_ISSUES_2026-08_10-07.md` 2026-10-07 턴 경계) |
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
나머지는 RL loss 의 logprob backward 가 차지한다. OOM 원인은 총량이 아니라 단편화다 (`archive/KNOWN_ISSUES_2026-08_10-07.md` 2026-10-07 메모리 항목).
미검증 범위: vLLM 동거(colocated), R3 켠 상태, 실제 데이터 분포, ES 가 refit 시간에 주는 영향, packing+CP 의 KL 수치.
결과 JSON 의 `loss: NaN` 은 하네스 키 오류다(워커 반환 키는 `global_loss`). 그래서 R4 는 loss·grad norm 유한성을 확인하지 않았다 (`archive/KNOWN_ISSUES_2026-08_10-07.md` 2026-10-07, G0 에서 수정).
산출물: `$NRL_ROOT/gates/train_memory/` (`mem_*.json`·`.log`, 런처 `run_mem_sub1.sh`, 스냅샷 분석 `analyze_snapshot.py`).

```bash
cd NeMo-RL && $NRL_ROOT/clean_run.sh $NRL_ROOT/bin/uv run --locked python examples/run_grpo.py \
  --config examples/configs/alpha/grpo_alpha_smoke.yaml policy.model_name=<hfmodel_00NNNNN> logger.log_dir=<dir> </dev/null
python examples/configs/alpha/tools/analyze_rollout_logprob_gap.py <dir>/exp_*/train_data_step*.jsonl   # CPU
$NRL_ROOT/clean_run.sh $W/bin/python -m torch.distributed.run --nproc_per_node 8 \
  examples/configs/alpha/tools/verify_muon_optimizer.py --config examples/configs/alpha/grpo_alpha_smoke_muon.yaml </dev/null
# R3 (CPU, 5개 인터프리터를 단계별로 호출):
CUDA_VISIBLE_DEVICES="" python3 examples/configs/alpha/tools/verify_chat_render_parity.py run --out <dir> --ckpt <hf 체크포인트>   # 종료 1 이 기준선이다 (P4 · P5 는 수정 전 경로를 일부러 잰다) — 경로별 MATCH / MISMATCH 수를 R3 상세 표와 대조한다
# R4 (8 GPU, 조건 하나당 1회). whitelist 밖 환경변수는 clean_run.sh 뒤 /usr/bin/env 로 넘긴다 (원장 #23):
$NRL_ROOT/clean_run.sh /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $W/bin/python -m torch.distributed.run \
  --nproc_per_node 8 examples/configs/alpha/tools/measure_train_memory.py --config examples/configs/alpha/grpo_alpha_smoke_muon.yaml \
  --seq-len 131072 --cp 8 --logprob-chunk-size 2048 --fuse-loss --out mem.json </dev/null
# M5: $NRL_ROOT/gates/engine_parity_iter2400/run_{pai,nemorl,hf}_sub1.sh 실행 뒤
#     <NeMo-RL venv python> examples/configs/alpha/tools/engine_parity_compare.py --pai-dir <pai> --nemorl-dir <nemorl> --out report.json
```
GPU 게이트 런처는 기동 직전에 GPU 점유를 검사한다. 한 장이라도 1 GiB 이상 쓰이면 중단한다 (원장 #25).

### R1 2×2 — R3 × 학습 라우터 정밀도 (2026-11-25, `$NRL_ROOT/gates/r3_router_20261125/`)

10-06 의 R3 결정 실험은 학습 라우터 fp64 · 고정 (`grpo_math_1B.yaml` 상속), vLLM 플러그인 라우터 fp32 였고, R3 의 효과 (0.0042 → 0.0026) 는 GDN 상태 bf16 에서 쟀다 (그때 로그로 확인) — "R3 끔 + 상태 fp32" 와 "학습 라우터 fp32" 는 재지 않았다. 그 빈칸을 R1 로 메웠다. 공통: 체크포인트 `mopd1/hf/hf_step_00120` · GDN 상태 fp32 · 라우터 고정 · commit `d6422627b`~`10b60024f`.

| 런 | R3 | 학습 라우터 | KL step 1 / 2 / 3 | 그때 기준 0.002 | 새 기준 0.001 |
|---|---|---|---|---|---|
| K0 | 켬 | fp64 (지금 기본값) | 0.00062 / 0.00062 / 0.00058 | PASS | PASS |
| K1 | 켬 | fp32 | 0.00061 / 0.00063 / 0.00057 | PASS | PASS |
| K2 | 끔 | fp64 | 0.00150 / 0.00149 / 0.00139 | PASS | **FAIL** |
| K3 | 끔 | fp32 (Ultra 방식) | 0.00151 / 0.00153 / 0.00135 | PASS | **FAIL** |

- 네 런의 step 1 롤아웃은 128/128 시퀀스가 비트 단위로 같다 (롤아웃측 logprob 도 같다) — 같은 토큰 위에서 학습측만 비교한 값이다. 두 노드 (K2 sub1 · K3 main1) 도 같은 롤아웃을 낸다.
- **R3 를 빼면 KL 이 2.4 배** — 프롬프트 48 개 짝 비교 (`tools/judge_prefix_cache_gate.py --pair`) 로 생성 위치 [0,256) 의 k3 +0.00117 (z 19~20), [0,2048) +0.00095 (z 24), \|d\| > 0.5 토큰 비율 +0.07%p (z 15). 기준은 넘지 않지만 여유가 25% 다.
- **학습 라우터 정밀도는 영향이 없다** — R3 켬: fp32 − fp64 z −0.04 / +0.83 / −1.45, R3 끔: z +0.95 / +0.04 / −0.45. expert 뒤집힘의 원인이 라우터 계산 정밀도가 아니라 입력 hidden state 차이라는 10-06 진단과 일치한다.
- 같은 토큰 위에서도 학습측 logprob 은 런마다 다르다 (K1 대 K0: \|Δ\| > 1e-3 34% · > 0.1 1.4 만 토큰 · 최대 1.0; 평균 KL 은 같다) — 학습측 런 간 비결정의 크기이고, R1-L 의 같은 조건 반복이 이 바닥을 잰다.

**기준의 출처와 변경 (2026-11-25)**: 사용자 결정 2026-11-25 15:36 "기준을 1e-3 으로 낮춰" — R1 · R1-L 의 KL 기준은 이제 **0.001** 이다 (`RL_PLAN.md` 결정 80). 실제 런의 첫 step 판정 (G5 · G7 · G8 · J1 · JG1 · MS1 · MS2 행의 "KL < 0.002") 에도 같은 값을 쓸지는 미결이다 — 11 월 실제 런 (R3 켬) 의 step KL 중앙값이 0.0008 ~ 0.0010 이라 그대로 옮기면 경계에 걸린다 (아래 표). 그전 기준 0.002 의 경위: 이 값의 근거는 기록돼 있지 않다 — 2026-08 준비 문서 (`archive/POSTTRAIN_PREP_2026-08.md`) 에 "Generation KL < 0.002" 로 처음 나오고 유도 과정이 없다. upstream 의 안내는 더 엄하다: `docs/guides/grpo.md` "KL Divergence Error" — "values below 1e-3 considered acceptable. Investigate any metric that shows spikes above this threshold" (출처 논문 *When Speed Kills Stability* 의 경험값 약 1e-3), `docs/guides/dtensor-tp-accuracy.md` — `token_mult_prob_error` 1.05 초과는 프레임워크 결함 의심. 즉 0.002 는 "고장 경보" 로 쓰인 값이고 (10-06 의 0.0042 를 잡았다), "이 아래면 비용이 없다" 는 뜻이 아니다.

| 같은 토큰 위의 비교 (2×2 덤프) | R3 켬 (K0) | R3 끔 (K3) |
|---|---|---|
| Generation KL (k3 평균) | 0.00060 | 0.00145 (upstream 허용 1e-3 초과) |
| `token_mult_prob_error` (평균 exp\|d\|) | 1.0125 | 1.0207 |
| \|d\| > 0.5 토큰 | 0.009% (110 개) | 0.075% (878 개) |
| IS 비율 exp(d) 가 [0.8, 1.25] 밖인 토큰 | 0.295% | 1.188% |
| 시퀀스 logprob 합 오차 Σd 의 표준편차 | 2.28 | 3.70 |
| 생성 위치별 KL [0,256) · [256,1024) · [1024,2048) · [2048,4096) | 0.00055 · 0.00063 · 0.00061 · 0.00060 | 0.00172 · 0.00158 · 0.00145 · 0.00134 (위치 누적 없음) |
| 시퀀스 마스킹 (`seq_logprob_error_threshold` 2) | 0 / 384 | 0 / 384 |

| 11 월 실제 런 (R3 켬, 로그의 step KL — 소수 4 자리) | step 수 | 중앙 | p90 | 최대 | 0.001 초과 step |
|---|---|---|---|---|---|
| `teacher_workplace/w1r` (11-19) | 76 | 0.0010 | 0.0010 | 0.0010 | 0% |
| `teacher_workplace/w1` (11-19) | 38 | 0.0010 | 0.0011 | 0.0011 | 11% |
| `teacher_terminal_pivot_judge/j1` (11-17) | 68 | 0.0009 | 0.0010 | 0.0010 | 0% |
| `teacher_terminal_pivot/p2` (11-15) | 200 | 0.0008 | 0.0009 | 0.0010 | 0% |
| `teacher_swe_pivot/p1` (11-13) | 120 | 0.0008 | 0.0009 | 0.0009 | 0% |
| `mopd1/s3` (11-11) | 120 | 0.0009 | 0.0011 | 0.0013 | 18% |
| `mopd1/s2` (11-10) | 43 | 0.0018 | 0.0018 | 0.0023 | 100% |

### R1-L — prefix cache 적중이 실제로 일어나는 R1 변형 (2026-11-25)

R1 은 프롬프트가 약 50 토큰이라 prefix cache 가 한 번도 적중하지 않는다 (엔진 로그 적중률 0.0% — 10-06 · 11-12 · 11-25 K0 전부; alpha 의 캐시 블록은 GDN 상태 fp32 에서 1,056 토큰이고 align 모드는 블록 경계에서만 상태를 캐시한다). 그래서 캐시 적중 경로와 그 위의 R3 경로 재생은 학습 스택에서 검증된 적이 없다. R1-L 이 그 구멍을 메운다.

| 항목 | 내용 |
|---|---|
| 지키는 것 | ① 캐시 적중 경로 (블록 경계에서 복원한 GDN 상태 · KV 재사용) 가 롤아웃 logprob 을 바꾸지 않는다 ② R3 를 켠 채 캐시된 토큰의 경로 행이 온전히 학습으로 간다 ③ 두 후보 설정 — A (R3 유지) · B (R3 끔 + 학습 라우터 fp32, Ultra 방식) — 의 KL |
| 명령 | `$NRL_ROOT/gates/prefix_cache_<날짜>/run_l.sh withB\|noB` — R1 과 같은 명령 · 레시피 (`grpo_alpha_smoke_muon.yaml`) 에 오버라이드만 더한다 |
| R1 과 다른 점 | 프롬프트 틀 `tools/gate_assets/long_prefix_prompt.txt` (문제 → 긴 참고 자료 → 문제; 프롬프트 3,400 토큰대 = 블록 3 개, 서로 다른 문제의 공유 구간 12 토큰) · 시퀀스 상한 7,168 · 엔진 인자를 학습 레시피와 같게 (chunked prefill · `max_num_batched_tokens` 16384 · `max_num_seqs` 256 · `gdn_prefill_backend` triton) · prefix caching 명시 · `RAY_DEDUP_LOGS=0` |
| 런 | `LA_off` · `LA_on` · `LA_off2` (같은 조건 반복) · `LB_off` · `LB_on` · 음성 대조 `NC_zero` · `NC_shift` · `NC_trunc` (반드시 잡혀야 함) · `NC_state` (검출 한계 참고) |
| 판정 | `tools/judge_prefix_cache_gate.py <gate_dir> --mode withB\|noB` (CPU). 유효성: rc 0 · KL 줄 정확히 3 개이고 덤프 재계산과 일치 · 엔진의 캐시 켬/끔 표지 · **엔진 8 개 각각 step 마다 적중률이 기대값과 "한 블록 덜 적중" 의 중간 이상** · 결함 주입 표지. 통과: KL < 0.001 (R1 기준 — 11-25 15:36 에 0.002 에서 낮춤, 그 시점에 R1-L 결과는 아직 없었다) + 캐시 켬 − 끔의 프롬프트 단위 짝 비교 (48 개) 세 통계가 모두 z < 2.33 — 생성 위치 [0,256) 의 k3 · [0,2048) 의 k3 · \|d\| > 0.5 토큰 비율 + (A) R3 경로 누락 경고 0 줄 + 음성 대조 3 개가 잡힘 |
| 음성 대조 | `tools/gate_assets/fault_inject/sitecustomize.py` — `PYTHONPATH` + `ALPHA_GATE_FAULT` 로 음성 대조 런에서만 설치된 vLLM 을 메모리에서 고친다 (운영 코드 · 플러그인 불변). `state_zero` · `state_scale:1.01` (GDN prefill 이 이어받는 상태) · `routes_trunc:8` · `routes_shift:3168` (`RoutedExpertsManager.get`) |
| 못 보는 것 | 단일 턴이라 캐시 블록이 전부 prefill 산출물이고 깊이가 하나다 (디코드 중에 만든 블록 · 부분 적중 · 연쇄 적중 없음) · sync · eager · colocated (async · CUDA graph · in-flight 갱신 아님) · 캐시 축출 · 슬롯 재사용 뒤의 낡은 경로 행 · async 경로에서 `reset_prefix_cache()` 가 실행 중 요청이 있으면 False 를 돌려주는데 반환값을 아무도 보지 않는다 → 엔진 단독 2 턴 시험과 실제 SWE 런의 첫 step 으로 본다 |
| 적대적 검토 | 2026-11-25 1 차 **보완 필요** — 고장 나도 통과하는 경로 4 개 (모든 요청이 같은 블록을 타 요청 간 뒤섞임이 안 보임 · R3 누락은 오류가 아니라 경고 + 조용한 대체 · 절대 기준 0.002 는 기준선 0.0006 의 3 배까지 통과시키고 앞부분 오차가 희석됨 · Ray 로그 합치기가 엔진별 적중률을 가림) + 작은 것 5 개 → 전부 반영. 2 차 **보완 필요 (판정기만)** — 음성 대조를 캐시 끔 기준선에 대어 채점 · 같은 조건 반복은 step 1 롤아웃이 비트 동일이라 표준오차 0 → NaN → 반영 (민감도는 `LA_on` 대비 · A/A 는 step 2·3 · step 1 은 토큰 동일성 보고 · `state_zero` 를 필수 음성 대조로). 실행기는 2 차에서 수정 사항 없음. 3 차 (판정기만) **보완 필요** — 필수 런이 빠져도 있는 항목만으로 PASS 가 나옴 · 경로만 건드리는 음성 대조의 step 1 토큰 동일성을 보고만 하고 강제하지 않음 · 엔진 표지 줄 수를 세지 않음 → 반영 (`--mode withB\|noB` 의 필수 항목이 빠지면 FAIL · 토큰이 다르면 그 런 무효 · 표지 8 줄 이상). 통계 · NaN 처리 · 적중률 문턱 · 파서는 K 매트릭스 로그로 확인됨 |

**R1-L 결과 (2026-11-25 15:32 ~ 17:10, `$NRL_ROOT/gates/prefix_cache_20261125/`, 체크포인트 `mopd1/hf/hf_step_00120`, commit `cd8c1f55f`, 판정기 `--mode withB` → `judge.txt` · `judge.json`)**

| 런 | R3 | 캐시 | KL step 1 / 2 / 3 | 엔진 8 개 적중률 (step 별 최소 ~ 최대) | 유효성 |
|---|---|---|---|---|---|
| `LA_off` | 켬 | 끔 | 0.00063 / 0.00060 / 0.00057 | 0.0% | 통과 |
| `LA_on` | 켬 | 켬 | 0.00061 / 0.00059 / 0.00058 | 85.7 ~ 92.9% (문턱 약 66%) | 통과 · R3 경로 누락 경고 0 줄 |
| `LA_off2` (같은 조건 반복) | 켬 | 끔 | `LA_off` 와 step 1 토큰 128/128 · 학습측 logprob 까지 동일 | 0.0% | 통과 |
| `LB_off` | 끔 | 끔 | 0.00154 / 0.00150 / 0.00138 | 0.0% | KL 기준 0.001 FAIL |
| `LB_on` | 끔 | 켬 | 0.00148 / 0.00152 / 0.00139 | 87.8 ~ 92.9% | KL 기준 0.001 FAIL |
| `NC_zero` (복원 상태 0) | 켬 | 켬 | 0.0014 / 0.0020 / 0.0014 | — | 결함 주입 확인 |
| `NC_shift` (접두사 expert 번호 +1) | 켬 | 켬 | 0.0385 / 0.0715 / 0.0350 | — | 결함 주입 확인 · step 1 토큰은 `LA_on` 과 128/128 동일 |
| `NC_trunc` (경로 행 끝 8 개 누락) | 켬 | 켬 | 4.170 / 4.168 / 4.349 | — | 결함 주입 확인 · 누락 경고 24 줄 |
| `NC_state` (복원 상태 × 1.01) | 켬 | 켬 | 0.0006 / 0.0006 / 0.0006 | — | 결함 주입 확인 |

| 판정 | 결과 | 근거 (프롬프트 단위 짝 비교, z 는 [0,256) · [0,2048) · \|d\| > 0.5 순) |
|---|---|---|
| A/A 보정 | PASS | step 2·3: z −0.92 / −0.11 / +0.02 |
| **A — R3 유지 + 캐시 켬** | **PASS** | 켬 − 끔: −0.000006 · −0.000014 · −0.000008, z −0.32 / −0.88 / −0.46. 검출 가능한 최소 효과 [0,256) k3 0.00008 |
| B — R3 끔 · 라우터 fp32 + 캐시 켬 | FAIL | 캐시 효과는 없다 (z +0.27 / −0.98 / +0.68) — 절대 기준에서 떨어진다 (R3 를 뺀 효과 z +21.6 / +28.3 / +18.2) |
| 음성 대조 `NC_zero` | 잡힘 | `LA_on` 대비 z +15.6 / +11.1 / +5.5 |
| 음성 대조 `NC_shift` | 잡힘 | z +4.1 / +6.5 / +16.6 |
| 음성 대조 `NC_trunc` | 잡힘 | 누락 경고 24 줄 (KL 4.17 — 대체가 아니라 뒤 토큰의 경로가 통째로 어긋난다) |
| (참고) `NC_state` × 1.01 | 못 잡음 | z +0.34 / −0.89 / −0.97 — 복원 상태의 1% 오차는 이 게이트의 검출 한계 아래다 |

읽는 법: 캐시 켬은 R3 를 켠 채로 MDE 보다 큰 KL 차이를 내지 않고, 캐시된 토큰의 경로 행이 온전히 학습으로 갔다 (누락 0). 검출력이 확인된 것은 "복원 상태가 통째로 틀림" 과 "경로가 틀림 · 빠짐" 수준이다. 캐시를 켜면 같은 seed 라도 샘플이 달라진다 (step 1 공통 접두 평균 67 토큰 — 요청이 스케줄되는 순서가 바뀐다) — 그래서 켬 대 끔은 토큰 동일 비교가 아니라 프롬프트 단위 통계 비교다. 엔진 통계의 적중률은 구간별 값이라 누적 예측 (77 ~ 82%) 보다 높게 찍혔다 (요청당 상한 3,168 / 3,430 ≈ 92%). 못 보는 것 (디코드 중에 만든 블록 · async · in-flight 갱신 · 축출) 은 위 표 그대로다.

### Pai fleet 설정의 fp32 prefix caching 게이트 — FAIL 과 진단 (2026-11-25, `$NRL_ROOT/gates/prefix_cache_20261125/pai_v2b` · `pai_diag`)

Pai `eval_sft/prefix_cache_check.py` (서버 두 대에 같은 토큰을 강제하고 다음 토큰 logprob 을 위치별로 대조; 통과 = 캐시 적중 턴의 mean exp\|Δlogprob\| < 1.05 · top-1 ≥ 14/16 · max < 0.5) 를 GDN 상태 fp32 · 블록 1,056 으로 돌렸다. 검사 스크립트는 transformers 5.x 가 `apply_chat_template(tokenize=True)` 에서 `BatchEncoding` 을 돌려줘 죽으므로 그 한 줄을 푼 사본으로 돌렸다 (원본 미수정, 자명한 도구 오류라 진단 절차에서 제외).

| 비교 (캐시 적중 턴 · 60K 프롬프트) | 위치 | mean exp\|Δlogprob\| [부트스트랩 95%] | top-1 |
|---|---|---|---|
| 끔 ↔ 켬 (첫 실행 — **FAIL**) | 16 | 1.095 [1.039, 1.163] | 16/16 |
| 켬 ↔ 켬 대조군 (첫 실행) | 16 | 1.036 [1.008, 1.071] | 15/16 |
| C 끔 ↔ 켬 | 64 | 1.054 [1.035, 1.075] | 61/64 |
| B 끔 (청크 7392) ↔ 켬 | 64 | 1.057 [1.025, 1.100] | 60/64 |
| A 끔 ↔ 끔 (같은 설정) | 64 | 1.048 [1.026, 1.078], max 0.547 | 61/64 |
| D 끔 (8192) ↔ 끔 (7392) | 64 | 1.062 [1.031, 1.098] | 62/64 |

- **진단 (독립 검증 "확인 — 단, 안전 주장은 범위를 좁힐 것")**: FAIL 의 원인은 게이트 통계다. 임계 1.05 가 따로 띄운 서버 두 대 사이의 잡음 바닥 위에 있고 (캐시를 전혀 쓰지 않는 쌍 A · D 도 같은 크기), 위치 16 개의 표본 오차가 크다 (A 의 64 행에서 16 개를 뽑으면 평균이 1.05 를 넘을 확률 0.38). 서버 하나는 같은 요청에 비트 단위로 같은 값을 낸다 — 차이는 서버 사이에서 난다 (기전 미확정).
- **캐시 적중 경로에 대해 말할 수 있는 것**: 캐시 켠 서버가 낀 쌍 전체에서 캐시 끈 쌍 평균을 뺀 값이 적중 턴 +0.005 [−0.017, +0.023] · 1 턴 +0.002 [−0.007, +0.012] — 서버 간 잡음 위로 드러나는 오차는 없고, 95% 상한은 통계에서 약 +0.02 다. **범위**: 대화 하나 · 적중 위치 하나 (59,136) · 순차 단일 요청 · prefill 직후의 다음 토큰. "캐시가 안전하다" 는 문장은 이 자료로 쓸 수 없다.
- **쓰지 않는 문장 (검증에서 반박)**: "끔 ↔ 켬 이 끔 ↔ 끔 보다 크지 않다" (1 턴에서는 +0.016 [+0.004, +0.031] 로 크다 — 서버 8011 과 짝지은 효과이지 캐시 효과가 아니다) · "D 는 청크 크기만의 효과" (프로세스도 다르다) · "잡음이 fp32 때문에 두 배" (9 월 대비 위치당 잡음 1.6 ~ 3.3 배이나 체크포인트 · 스택도 바뀌었고 9 월에는 캐시 끈 쌍이 없다).
- **게이트 이력의 오류 (Pai 문서 쪽)**: `turn1_long` 은 "캐시 무관" 이 아니다 (켠 서버에서는 이미 캐시에 있다) · 켬 ↔ 켬 대조군은 잡음 바닥이 아니다 (캐시 경로끼리의 비교). 9 월 PASS 는 이 조건으로 넘어오지 않는다.
- **선행 사례**: Pai `SFT_BENCHMARKS.md` §2.5 (9-17, bf16 · 블록 544: 끔 ↔ 켬 1.040 · 대조군 1.045 — 임계 1.05 의 출처) · vLLM 은 하이브리드의 align 모드 캐시를 experimental 로 표시 · R1-L (학습 엔진, 3.4K 프롬프트) 은 캐시 켬 − 끔 KL 차이 없음.
- **다음 (승인 대기)**: 게이트 재정의 — 같은 서버에서 `cache_salt` 로 차갑게 계산한 값과 캐시 적중 값을 대조한다 (서버 간 잡음이 사라진다). 적중 위치가 차가운 요청의 청크 경계와 같을 때만 비트 동일을 기대할 수 있다 (청크 경계 = 7,392 의 배수 + 마지막 1,056 배수; `MAX_BATCHED_TOKENS=1056` 시험 서버면 모든 블록 경계). 디코드 중에 만든 블록 · 결함 주입 음성 대조 포함. 적대적 검토 뒤 실행.

### PC-S — 같은 서버의 "차가운 계산 대 캐시 적중" (2026-11-25 PASS, `$NRL_ROOT/gates/prefix_cache_20261125/pcs` · `run_pcs.sh`)

끔 ↔ 켬 게이트 (위) 를 대신한다 (사용자 승인 11-25). 캐시 켠 서버 한 대가 같은 프롬프트를 두 번 계산한다 — 한 번은 캐시 적중, 한 번은 요청마다 새 `cache_salt` 로 차갑게. 서버 한 대는 같은 요청에 같은 비트를 내므로 **기준은 비트 동일** 이다 (토큰 · logprob · top-5 전체, 요청당 32 토큰). 임계 상수가 없다. 정의의 정본은 `tools/prefix_cache_selfcheck.py` 머리말이다.

- **비트 동일을 기대할 수 있는 조건**: 적중 경로와 차가운 경로가 prefill 을 같은 자리에서 끊을 때뿐이다 (vLLM `_mamba_block_aligned_split`: 계산을 시작한 자리부터 예산을 블록 배수로 내린 청크 + 프롬프트의 마지막 블록 경계에서 강제 절단. 상태는 청크 끝에만 있다). 도구의 `Model` 이 요청마다 적중 위치와 청크 순서를 예측하고, 엔진 카운터가 그 적중 위치를 요청마다 확인한다.
- **서버**: `prod` = fleet 설정 그대로 (청크 예산 8192 = 7 블록) — 끊는 자리가 맞는 적중만 판정. `b1056` = `MAX_BATCHED_TOKENS=1056` — 모든 블록 경계가 청크 끝이라 임의 위치의 적중을 판정. 음성 대조 2 대 = fleet 설정 + 복원 전용 결함 (`restore_scale:1.01` · `restore_bf16`, `gate_assets/fault_inject`).
- **판정 행**: 같은 프롬프트의 차가운 계산 2 회 (다르면 기준 무효) · 적중 대 차가움 (8K ~ 60K, 같은 블록 꼬리 · 예산 배수 경계 · 3 턴 연쇄, 적중마다 2 회 + 원본 경계 재적중) · 생성 중에 만든 경계 블록의 적중 (원래 생성과 이어 붙여 비교; 생성 요청이 차가운 경우와 적중으로 시작한 경우) · 공통 접두사 admission 경로 대 격자 밖 경계의 적중 뒤 7 블록 청크.

| 런 | 판정 행 비트 동일 | 위치 | 차가운 반복 | 요청 (적중 위치 예측 일치) | 결함 발동 |
|---|---|---|---|---|---|
| `prod` (fleet 설정) | **24/24** | 768 | 2/2 동일 | 92/92 (적중 47) | 0 |
| `b1056` | **19/19** | 608 | 1/1 동일 | 67/67 (적중 36) | 0 |
| 음성 대조 × 1.01 | 0/18 (전부 검출, 최소 max\|Δ\| 0.0011) | 519 | 1/1 동일 | 74/74 (적중 39) | 39 = 적중 39, 위치 일치 |
| 음성 대조 bf16 반올림 | 0/18 (전부 검출, 최소 max\|Δ\| 0.013) | 541 | 1/1 동일 | 74/74 (적중 39) | 39 = 적중 39, 위치 일치 |

- **판정하지 않고 보고하는 값 (`prod`, 끊는 자리가 다른 적중 — 운영의 흔한 경우)**: 적중 대 차가움 mean exp\|Δlogprob\| 1.021 (28K) · 1.063 (60K, 2 위치) · 1.031 (admission 경로), max\|Δ\| 0.12 ~ 0.19. 같은 서버 안에서 청크 경계만 달라도 서버 간 잡음 (1.03 ~ 1.06) 과 같은 크기가 난다 — 옛 게이트가 본 차이의 크기와 맞는다. 이 비교에서는 × 1.01 결함이 보이지 않는다 (음성 대조의 같은 행 1.008 ~ 1.015). 생성 경계 뒤의 다음 토큰: 적중 대 차가움 1.012 · 원래 생성 대 차가움 1.007 (16 위치, top-1 16/16).
- **독립 검토 (새 에이전트 Opus · xhigh, 3 회)**: ① 보완 필요 — 같은 블록의 두 번째 적중과 적중으로 시작한 요청이 생성 중 남긴 경계가 판정 밖 (고장 4 종이 통과) → 반복 적중 행 · `decode_hit` · admission 행 · 요청 전수 기록 추가 ② 보완 필요 — admission 이 저장한 상태를 나중에 읽는 행은 attention 블록이 더 긴 청크에서 계산된 사본이라 정상 엔진에서도 다를 수 있다 → 비판정으로 내림 (실측은 32/32 동일) ③ **신뢰 가능 (명시한 범위 안에서)**. CPU 시험 `gate_assets/test_pc_selfcheck.py` 30/30 (정상 1 · 고장과 조작 29).
- **PASS 가 말하지 않는 것**: 캐시 켬 = 캐시 끔 (차가운 경로까지 똑같이 틀리게 하는 결함은 안 보인다) · 격자 밖에서 시작하는 다중 블록 청크 자체의 결함 · 생성 경계 상태를 여러 토큰 prefill 이 읽는 경로 · 동시 요청 · 선점 · 축출 · 채팅 템플릿 경로 · **FlashInfer 가 아닌 GDN prefill 백엔드** (NeMo-RL 학습 엔진은 triton — 학습에 캐시를 켜기 전에 학습 엔진에 같은 검사를 돌린다, 미실행).
- **재실행 조건**: 서빙 venv 의 vLLM · alpha 플러그인 · GDN 상태 dtype · 블록 크기 · 청크 예산 · prefill 백엔드 중 하나가 바뀔 때. 약 6 분 (GPU 2 장).

#### PC-S — 학습 엔진 설정 (2026-11-26 PASS, `gates/prefix_cache_20261125/pcs_train2` · `run_pcs_train.sh`, 사용자 승인 11-26)

학습에 prefix caching 을 켜는 전제다 (fleet 의 PASS 는 FlashInfer 커널 · 다른 torch 빌드라 그대로 넘어오지 않는다). 엔진은 **롤아웃 워커의 venv** (`$NRL_ROOT/venvs/…VllmAsyncGenerationWorker`: vLLM 0.25.1 · torch 2.11.0+cu130 · triton 3.6.0 — 학습 런 `teacher_workplace/w1r.log` 가 쓴 venv) 에서 학습 레시피의 엔진 인자로 띄웠다: triton GDN prefill · 청크 예산 16,384 (15 블록) · fp32 상태 · eager 끔 (cudagraph `FULL_AND_PIECEWISE`) · `processed_logprobs` · 문맥 131,072 · 메모리 0.85 · 라우터 재생용 routed experts 반환. 판정은 fleet 판과 같고, **routed experts (R3) 도 적중 대 차가움으로 비트 대조한다** — 적중이면 캐시 구간의 행은 캐시 블록의 슬롯에서 읽는다. 모든 응답에 행 수 = 프롬프트 + 생성 − 1 을 요구한다 (행 누락 결함).

| 런 | 판정 행 비트 동일 (logprob · 경로) | 위치 · 경로 행 | 차가운 반복 | 요청 (적중 위치 · 행 수 예측 일치) | 결함 발동 |
|---|---|---|---|---|---|
| `prod` (청크 예산 16,384, 프롬프트 최대 127,120) | **24/24 · 24/24** | 768 · 133 만 | 2/2 동일 | 92/92 (적중 47) | 0 |
| `b1056` | **19/19 · 19/19** | 608 · 63 만 | 1/1 동일 | 67/67 (적중 36) | 0 |
| 음성 대조: 복원 상태 × 1.01 | 0/18 · 0/18 (전부 검출, 최소 max\|Δ\| 0.0011) | 497 | 1/1 동일 | 74/74 (적중 39) | 39 = 적중 39, 위치 일치 |
| 음성 대조: 캐시 블록의 경로 값 밀기 | logprob 18/18 동일 · **경로 0/18** (경로 비교로만 검출) | 576 | 1/1 동일 | 74/74 (적중 39) | 39 = 적중 39, 위치 일치 |

- **판정하지 않고 보고하는 값**: 끊는 자리가 다른 적중 대 차가움 mean exp\|Δlogprob\| 1.008 (28K · 60K) · admission 경로 1.031; 생성 경계 뒤 다음 토큰 적중 대 차가움 1.011 · 원래 생성 대 차가움 1.011.
- **독립 검토 (3 회)**: ① 보완 필요 — 실행기가 스모크 엔진의 eager 모드를 옮겨 왔다 (학습 레시피는 eager 끔) · 경로 행 수를 보지 않는다 → 둘 다 고침 ② 신뢰 가능 ③ venv 정정 확인 — 첫 시도 (01:03) 는 옛 독립 venv `venv-vllm` 을 가리켜 import 에서 죽었다 (GPU 미사용). 워커 venv 의 캐시 · 경로 관련 vLLM 소스 13 개 파일은 fleet venv 와 바이트 단위로 같고, NeMo-RL 의 vLLM 패치 6 개는 이 경로와 무관하다. CPU 시험 35/35.
- **PASS 가 말하지 않는 것**: 동시 요청 (행 누락은 동시성에서 생길 수 있다 — 학습에서는 R3 `missing_routes` 통계로 본다) · **가중치 갱신 뒤의 캐시** (`recompute_kv_cache_after_weight_updates: false` 면 갱신 뒤에도 옛 가중치로 만든 KV · 상태 · 경로가 재사용될 수 있다 — S3 계획에서 다룬다) · NeMo-RL 워커의 HTTP 층과 refit (프런트엔드는 vLLM 의 OpenAI 서버, 가중치는 디스크에서 읽었다 — 그 경로의 통계적 확인은 R1-L) · 생성 경계 상태를 여러 토큰 prefill 이 읽는 경로 · 격자 밖에서 시작하는 다중 블록 청크의 공통 결함 · 캐시 켬 = 캐시 끔.
- **재실행 조건**: 워커 venv 의 vLLM · torch · triton · alpha 플러그인 · 엔진 인자 (청크 예산 · 상태 dtype · prefill 백엔드 · eager) 중 하나가 바뀔 때. 약 8 분 (GPU 2 장).

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
| TJ1 | terminal 피벗 judge 판정 품질 (`RL_PLAN.md` 결정 65): 운영 Gym `terminus_judge` verify (스키마 → 완료 일치 → 유사도 0.95 빠른 통과 → judge `rubrics_v4`) 를 프로세스 안에서 불러 라벨 쌍 4,559 개를 채점. 유사도로 갈리지 않는 세트 (순서 뒤집기 · 나누기 · 플래그 · 의미 등가 · heredoc 리다이렉트 · 스크립트 파일 / 2/3 삭제 · 다른 대상 · 다른 동사 · 제어 키 개행 · 본문 잘림 · sed 값 · 실행 생략) + 쉬운 음성 (빈 대기 · 다른 과제 · 직전 반복 · 정답이 대기). 가짜 judge 12 종이 FAIL 하는지 `selftest` 로 강제. 동시성 64 · 256 두 번 → 판정 뒤집힘 | `tools/measure_judge_terminus.py` (`build` · `selftest` · `run` · `flips`), 실행 `$NRL_ROOT/gates/terminus_judge/run_tj1.sh` | 세트별 일치율 (양성 0.75~1.00 · 어려운 음성 0.80~0.95 · 쉬운 음성 0.99) · 파싱 실패 ≤ 2% · 예외 ≤ 1% · 판정문 잘림 0 · 뒤집힘 전체 ≤ 2% · 어려운 세트 · 실제 출력 구간 각 ≤ 5% | **적대적 검토 3회 (Opus 5.5 · xhigh, 2026-11-16)**: v1 보완 필요 (유사도 ≥ 0.7 가짜 judge 가 PASS — BLOCKER) → v2 보완 필요 (제어 키 쌍 라벨이 거꾸로 68/100 — tmux 재현 · 글자 정규화 judge 가 PASS) → v3 보완 필요 (쓰기 · 실행 명령 쌍 없음) → v4 에 검토자가 지정한 다섯 세트 추가. 남은 제한: 의미 등가 양성 78 쌍 · 규칙으로 만든 세트는 전용 정규화로 되돌릴 수 있음. **2026-11-16 실행 (commit `ea09795b3`, 쌍 sha `19ba10cfa5d5`) — 둘 다 FAIL**. Gemma-4-31B-it: 양성 8 세트 전부 1.000 · 어려운 음성 N_drop 1.000 · N_target 1.000 · N_verb 1.000 · N_sed 1.000 · N_wtrunc 0.94 · N_noexec 0.96 · 쉬운 음성 N_wait 1.000 · N_other 0.997 · N_goldwait 0.993 · **N_repeat 0.970 (< 0.99)** · **N_ctrl 0.000 (< 0.80)** · 파싱 실패 0 · 잘림 0 · 뒤집힘 0/4,259 · 11.3~12.0 호출/s. Qwen3.8-27B (사고 끔): 양성 전부 1.000 · N_repeat 0.933 · N_goldwait 0.970 · N_ctrl 0.000 · N_noexec 0.91 · 잘림 7 · 뒤집힘 1.1% (실제 출력 0.3~0.5 구간 8% · 0.5~0.7 구간 14%) · 6.0~8.0 호출/s. 실제 출력의 "같음" 판정률 (유사도 0.1~0.3 · 0.3~0.5 · 0.5~0.7 · 0.7~0.95): Gemma 0.02 · 0.10 · 0.32 · 0.56 / Qwen 0.14 · 0.23 · 0.50 · 0.64. analysis · plan 을 중립 문장으로 바꾼 변형은 원 세트와 같음 (단서 효과 없음). 보고 전용 구멍: 조기 완료 선언 100/100 통과 · sed 치환값만 바꾼 유사도 ≥ 0.95 쌍 100/100 통과 (judge 이전 단계). N_repeat 오탐 9 건을 읽은 결과 약 절반은 직전 턴이 실제로 같은 탐색 (라벨 잡음), 나머지는 "둘 다 환경 탐색" 이라는 관대한 판정. **v5 (결정 66 — alpha 플러그인 `alpha_terminus_judge`: 완료 선언 양방향 · 빠른 통과 임계 1.0; 조기 완료 선언 · sed 근접 쌍을 판정 세트로, 제어 키 세트는 보고 전용, 직전 반복 세트에서 같은 경로 쌍 제외) 2026-11-16 11:31, commit `df7176416`, 쌍 sha `6def9ed03677`, Gemma: PASS** — 양성 8 세트 1.000 · N_drop 1.000 · N_target 1.000 · N_verb 1.000 · N_sed 1.000 · N_fastsed 0.99 · N_wtrunc 0.94 · N_noexec 0.96 · N_wait 1.000 · N_other 0.997 · N_repeat 0.997 · N_goldwait 0.993 · N_earlytc 100/100 judge 호출 0 · 파싱 실패 0 · 예외 0 · 잘림 0 · 뒤집힘 0/4,350 (동시성 64 대 256) · 11.1 호출/s. 플러그인 단위 테스트 11/11 |
| TS1 | terminal 피벗 스모크 — 행 단위 불변식 (`RL_PLAN.md` 결정 68): 학습 경로가 남긴 Gym 전체 결과로 채점기가 의도대로 붙었는지 본다 (judge 부분 장애 · Gym 원본 서버 기동 · 빠른 통과 임계 · 정책이 채점 · 재호출 가드 누락은 평균 보상으로 안 드러난다) | `tools/analyze_terminal_gym_results.py <log_dir>/exp_*/wandb --blend <blend>` — 3 step 스모크를 `WANDB_MODE=offline` + `logger.wandb.metric_allowlist=null ++logger.wandb.log_nemo_gym_full_result_tables=true` 로 띄운 뒤 | I1 judge 장애 · 키 없는 행 0 · I2 조기 완료 선언 ≥ 20 행이 전부 플러그인 서명 (보상 0 · 완료 검사 실패 · judge 판정 0 개) · I3 judge 없는 보상 1 은 유사도 1.0 뿐, judge 대상 행 (≥ 50%) 은 판정 정확히 1 개 · I4 judge 모델 Gemma · I5 판정 파싱 실패 ≤ 2% · I6 재호출 0 · R1 사유 분포가 블렌드 행의 프로파일 분포 ± 0.06 · R2 평균 보상이 기준 × [0.6, 1.4] · 변조 6 종 자가 시험이 겨냥한 불변식에서 FAIL (아니면 종료 코드 2). 재실행: terminal 피벗 레시피 · `alpha_terminus_judge` 플러그인 · judge 모델 · 블렌드 선별 규칙 변경 시 | **적대적 검토 4회 (Opus 5.5 · xhigh, 2026-11-16)**: 보완 필요 (HIGH 3) → 보완 필요 (MEDIUM 4) → 보완 필요 (MEDIUM 2) → 신뢰 가능. **PASS 2026-11-16** `teacher_terminal_pivot_judge/s1` (commit `c6e88724f`, 블렌드 v4): 3,072 행 · 위반 0 · 조기 완료 80 · judge 호출 2,429 · 평균 보상 0.202 대 기준 0.2245 · 사유 분포 최대 차 0.022 · 자가 시험 6/6 · 생성 KL 0.0009~0.0010 |
| WS1 | 검색 객관식 플러그인 CPU 검사 (`RL_PLAN.md` 결정 72 · 73): Gym `google_search` 의 검색을 Tavily 로 바꾼 서버 `alpha_web_search_mcqa` 가 (가) 채점 · 도구 · 세션 통계 · 백엔드 장애 표시 (`instance_config.mask_sample`) 를 의도대로 하는가 (나) 데이터 행과 맞는가 (다) 실제 Tavily 와 실제 웹 페이지로 한 문항이 끝까지 도는가. **GPU 없음 — 정책 모델을 거치는 경로 (도구 호출 파싱 · 롤아웃 · 문맥 길이) 는 이 게이트가 보지 않는다** | 플러그인 venv (`$NRL_ROOT/gates/web_search_mcqa/venv`) 에서 `NEMO_GYM_EXTRA_ROOTS=…/gym_plugins RAY_ADDRESS=local pytest --noconftest tests/test_app.py` · `tools/check_web_search_mcqa.py <행 파일> [--expect-agent-ref alpha_web_search_mcqa_simple_agent] [--live]` (`--live` 는 검색 1회 = 1~2 크레딧) | 단위 테스트 전부 통과 · 데이터 결함 0 (요청 모델 · 도구 이름과 인자 · 정답 글자가 선택지에 있음 · agent_ref) · `--live` 결함 0 (검색 결과 ≥ 1 · browse 본문 ≥ 200 자 · 정답 글자에 보상 1 · 백엔드 장애 0 · 키 미노출) · 음성 대조: 변조 행을 넣으면 종료 코드 1 | **2026-11-20 PASS** — 단위 50/50 (플러그인 보강 뒤 62/62 — 속도 제한 · 크레딧 상한 · usage 누락 · 대조 백엔드 · URL 이름 풀이; `--live` 는 크레딧을 아끼려 다시 돌리지 않았다) · 2,929 행 결함 0 (4지 2,911 · 10지 18, 무작위 선택 기대 보상 0.249) · 변조 5 종 (도구 이름 · 도구 인자 · 선택지에 없는 정답 · 필드 누락 · 다른 agent) 전부 검출 · `--live` 결함 0 (질의 382 자 → 결과 10 건 8,481 자 · 위키 본문 16,273 자 · 응답이 알린 크레딧 1). 원본 `google_search` 는 `\boxed` 없는 답에 verify 가 pydantic 오류를 낸다 (테스트 `test_upstream_verify_raises_…` 로 고정). 독립 검토는 아직 없다 — 2단계 (GPU 소규모 측정) 계획과 함께 받는다 |
| WS2 | 검색 객관식 환경 소규모 측정 (`RL_PLAN.md` 결정 73 2단계): MOPD1 학생으로 48 문항 × 4 를 세 갈래 — tools (실제 검색) · placebo (같은 프롬프트 · 도구 · 턴 상한, 서버만 `search_backend: none`) · notools (도구 정의 없음, 서술용). 재는 것: 롤아웃당 검색 수 · 크레딧, 통과율, 검색이 준 정보의 기여 (주 대조 tools − placebo), 경로 결함 (도구 호출 형식 · 스텝 상한 · 형식만 틀린 0 점 · 백엔드 장애 · 질문 원문 누출) | `$NRL_ROOT/p0_measure/wsm/run_wsm_probe.sh <tag>` (sub1 8장, `DRY=1` 이면 GPU 없이 환경 두 벌만) → `tools/web_search_probe.py analyze` · 갈래마다 몇 행 `smoke-check` 뒤 본 실행 | 유효성 V1~V7 전부 PASS 여야 수치를 쓴다 (행 · 키 완전성 / 갈래 분리 / 세션 통계 = 대화 기록 / 백엔드 장애 0 / 보상 재계산 / 문항 정렬 / 검색 생존 · 크레딧 장부). 판정 (유효할 때만): 네 구간 (tools − placebo · 끝까지 답한 롤아웃만 · 장애 롤아웃을 0 점으로 넣은 것 · tools − notools) 의 문항 단위 부트스트랩 95% 하한이 모두 > 0 이면 "기여". 주 대조만 양이면 "보류", placebo − notools 상한 < 0 이면 "보류", 누출 롤아웃 > 5% 면 "보류 (누출)", 다시 뽑힌 롤아웃 > 5% 면 "보류", 검색 성공 0 이면 "판정 불가", 결과 있는 검색을 한 롤아웃 < 30% 인데 차이가 안 보이면 "판정 불가 (도구 사용 부족)". 사전 등록 상수: 인프라 허용 5% · 누출 5% · 사용률 30% · 검색 응답률 80% · 거절 10%. 모의 검출력: 실제 차이 0.10 → 약 0.4, 0.15 → 약 0.8, 0.20 → 약 0.95 (양성 판정에 엄격). 한계: 빈 결과의 손상과 지시문 · 턴 효과가 비슷하면 가르지 못함 · 스텝 상한 5 에서의 비용 · 효과. 자가 시험 48 경우 | 미실행. 적대적 검토 1차 (2026-11-20, Opus 5.5 · xhigh) **보완 필요** — 두 갈래 (tools 대 notools) 대조가 생성 예산 차이로도 양성 · 크레딧 최악값 계산 오류 · 검색 전면 거절과 usage 누락을 못 봄 · 누출 미통제 등 (요지 `STATUS.md`) → 세 갈래로 고침. 2차 (새 에이전트) **보완 필요** — 빈 결과 placebo 의 거짓 양성 · 누출만의 효과 · 거의 죽은 검색 통과 · `gym env start` 종료가 자식 서버를 남김 (예행 잔류 12 개, 정리함) → 위 판정 규칙 · V7 문턱 · 프로세스 그룹 종료로 고침. 3차 (새 에이전트) **보완 필요** — 누출 롤아웃 사후 제외가 양성을 만듦 · 도구 이름 라우팅 · V7 분모 · 장애 행 제외의 편향 · 낮은 사용률 → 위 규칙으로 고침. CPU 예행 `wsm/dry5` 통과 (환경 두 벌 기동 · 기동 설정 대조 · 종료 뒤 잔류 서버 0 · Tavily 잔여 996 ≥ 최악 980). 4차 (새 에이전트) **보완 필요** — 측정 로직은 신뢰할 만하나 서빙의 GDN prefill 교착 회피가 없어 유료 갈래가 낭비될 수 있음 (미반영). **실행 보류 2026-11-20** — 트랙 방향 재검토 (`STATUS.md`) |
| J1 | judge 분할 2노드 스모크 (`RL_PLAN.md` 결정 21): 실레시피 `teacher_stem_alpha.yaml` — 롤아웃 노드 = vLLM 롤아웃 6장 + Gym judge (Gemma-4-31B-it TP2, `local_vllm_model`) 2장, 본체 수정 `alpha/judge-split`. 스모크 블렌드 `teacher_stem_smoke_v3.jsonl` 256행 (v2 의 Science-v1 정답 잘림 16행 교체, 결정 25) (객관식 64 · 화학 rdkit 플러그인 64 · 주관식 judge 64 · python 도구 과학 64) = 64 프롬프트 × 16 × 4스텝, 저장 끔, W&B 끔 | `teacher_stem_alpha.yaml` + `grpo.max_num_steps=4`, `ray list placement-groups`, Gym 응답 로그 | rc 0 · 학습 8장이 한 노드, 롤아웃 6장과 judge 2장이 다른 한 노드 · Generation KL < 0.002 · 마스킹 0 · judge 환경과 rdkit 플러그인 보상에 0 과 1 이 모두 있음 · judge 판정 파싱 실패 ≤ 2% · 재판정(`check_full_generation_on_fail`)이 실제로 일어남 — Gym 응답 로그에서 횟수와 0.5 보상 표본을 확인 · ns_tools 도구 호출이 sandbox 에서 실행됨 · 스텝 시간 기록 (롤아웃 8장 런과 비교) | **✅ PASS 2026-10-10** — `j1b` (14:58~16:19, rc 0): 학습 sub1 8장 · 롤아웃 main1 6장 · judge main1 2장 · gen KL 0.00196 · 0.00199 · 0.00193 · 0.00191 · 마스킹 0 · judge · rdkit 보상에 0 과 1 · ns_tools 도구 sandbox 실행 · 스텝 642 · 445 · 573 · 734 s (롤아웃 8장 런과는 데이터가 달라 직접 비교하지 않는다). j1b 는 judge 판정 기록을 남기지 않아 (`KNOWN_ISSUES.md` 2026-10-10) 보충 `j1c` (judge 128 + rdkit 64 행, W&B 오프라인 표, 1스텝 분량 수집 뒤 중단): judge 파싱 실패 0/484 (equivalence) · 0/645 (ns_tools) · 재판정 시도 132 · 277, 0.5 보상 15 · 24 — 그중 잘린 추출 복구 3 · 10, 같은 답의 판정 뒤집힘 12 · 14 (표 중복을 뺀 값, 10-10 정정) → 재판정 끔 (사용자 결정, `RL_PLAN.md` 결정 27). j1c 를 J1 근거로 인정 (사용자 결정 10-10). j1c 에서 rdkit 예/아니오 행 채점 결함을 찾아 고쳤다 (`KNOWN_ISSUES.md`). 이하 사전 기록: 본체 수정은 J1 전에 `alpha/post-train` 에 병합했다 (`8a8544dcc`, 사용자 결정 — FAIL 이면 revert), 병합 트리 검사기 ERROR 0. 10-10 결정 25 뒤 smoke_v3 · v2 로 검사기 ERROR 0 (정답 잘림 검사 포함). 사전 검증: `alpha/judge-split` 단위 테스트 PASS (`2807ef2be`) · 레시피 검사 ERROR 0 · 음성 대조 4건 ERROR (`ab3bfe518`) · rdkit 플러그인 테스트 17건 PASS (`a38ad04bf`) |
| JG1 | General teacher 2노드 스모크 (`RL_PLAN.md` 결정 49~51): 실레시피 `teacher_general_alpha.yaml` — 외부 49B GenRM (`alpha_genrm_model`) + Gemma judge 5종 + 규칙 채점. 스모크 블렌드 `teacher_general_smoke_v1.jsonl` 192행 (17 층 × 약 12) · 검증 `teacher_general_smoke_v1_val.jsonl` 25행 (GenRM 없음) = 64 × 16 × 3 step, 저장 끔, W&B 오프라인 표 (`EXTRA_ENV="WANDB_MODE=offline" WANDB=1` · `logger.wandb.metric_allowlist=null` · `++logger.wandb.log_nemo_gym_full_result_tables=true`), 새 tag | 기동 전: `curl https://gemma4.withai.cj.net:10206/v1/models` 에 genrm · `tools/check_gym_judge_routing.py` (--expect multichallenge·inverse_if·abstention·jailbreak_detection·over_refusal_detection=judge_model, genrm_compare=genrm_model) ERROR 0. 판정: `tools/analyze_general_smoke.py --since <기동 epoch>` (분석기 음성 대조 `tools/negctl_general_smoke.py` 16/16) | P1 rc 0 · 학습 8장 한 노드, 롤아웃 6 + judge 2 다른 노드 · 기동부터 NeMo-RL `SETUP COMPLETE` 까지 60분 안 (Gym 서버 venv 8개 첫 빌드 · NeMo-Gym init 포함, 빌드 시간 미측정 — j1c 는 venv 가 있는 상태에서 Gym init 286 s) · `SETUP COMPLETE` 부터 첫 step 완료까지 60분 안 (step 0 검증 + 첫 step) · 직전 step 완료 뒤 30분 동안 다음 step 완료가 없으면 FAIL (외부 GenRM 무한 재시도 교착 검출 — 교착이면 step 줄이 찍히지 않는다; 근거: STEM j1c 같은 배치 32K step 490 s + 스모크 GenRM 비중 17% 의 비교 약 176건 ÷ 측정 13.6건/분 ≈ 13분 → 약 21분, 30분 ≈ 1.4배. GenRM 은 공유 인스턴스라 외부 부하가 붙으면 거짓 FAIL 가능) · P2 gen KL < 0.002 · 마스킹 ≤ 1% · P3 agent 16 + identity (그룹 기록) · P4 규칙 agent 잘리지 않은 표본 보상 고유값 ≥ 2 · P5a judge 실패 대체 행 0 · P5b 판정 행 ≥ 95% · P5c 엄격 파싱 ≤ 3% · P5d 안전 judge 서버 판정 == 최종 판정 라벨 (단독 라벨 줄 우선) 불일치 ≤ 0.5% · P5e 정책별 verifier 라벨 None ≤ 3% · P5f 정책별 jailbreak 경로가 정책 agent 대상 행의 ≥ 95% 에서 쓰임 · P6a GenRM 무승부 대체 ≤ 2% · P6b 어댑터 파싱 실패 ≤ 3% · P6c 순위 response_1 우세 0.2~0.8 · P6d 동점 ≤ 90% · P6e principle 주입 ≥ 0.5 × 16 × identity 그룹 · P6f 어댑터 파싱 성공 ≥ 0.9 × 16 × GenRM 그룹 | **사전 정의 (2026-11-06)**: 적대적 검토 1차 신뢰 불가 (judge 실패·GenRM 무승부가 0/0 으로 통과 · 검증 멈춤) → 2차 보완 필요 (실제 표 구조 가정 오류 · 안전 라벨 오독) → 3차 보완 필요 경미 (정책별 경로가 조용히 꺼지면 통과 · 반대 라벨 언급 거짓 FAIL · 상한 근거) → 4차 **신뢰 가능** (LOW: 기동 단계 상한 분리 · P5d 판정 단위 — 반영). **❌ FAIL 1/24 — 2026-11-06 jg1 (19:22 → 20:05, rc 0, commit `2971d1f58`)**: P1 ✅ 기동→SETUP 9분 · 첫 step 시작 19:53 · step 간격 4 · 3분 · 학습 sub1 8장, 롤아웃 main1 6장 + Gemma judge main1 · P2 ✅ gen KL 0.00158 · 0.00153 · 0.00169, 마스킹 0 · 0 · 0 (유효 시퀀스 1,024) · P3·P4·P5a·P5b ✅ · P5c ✅ multichallenge 5/512 · abstention 0/202 · 안전 0/2,416, **❌ inverse_if 121/836** · P5d 0/530 · P5e 0/1,886 · P5f 827/827 · P6a 대체 0 · P6b 3/780 · P6c 순위 [2, 52, 349, 325, 44, 5] response_1 우세 0.52 · P6d 동점 0.70 · P6e 주입 625 (하한 208) · P6f 777 ≥ 592. inverse_if 실패는 데이터-파서 불일치: 행 judge 지시문의 약 20% 가 `[JSON]: {"answer_score": …}` 형식을 요구해 Gym inverse_if 가 엄격 파싱을 못 하고 마지막 5줄 낱말 검색으로 대체 (120/120, PASS 46 · FAIL 74). `result` 형식 행은 715/716 (`KNOWN_ISSUES.md` 2026-11-06) |
| MS1 | MOPD1 2노드 스모크 (`RL_PLAN.md` 결정 55): 실레시피 `mopd1_alpha.yaml` 그대로 배치 L1 (학습 sub1 8 · 생성 vLLM main1 4 · teacher main1 4 × 1장, 가상 `num_nodes: 6`, judge 없음 `skip_verification`). 스모크 블렌드 `mopd_blends/mopd1_smoke_v1.jsonl` 192행 (`tools/build_mopd1_smoke_blend.py` — 본 블렌드에서 teacher 비율대로 190 + 렌더 프롬프트 57,261 · 56,836 토큰 swe_pivot 2행 = teacher 1장 메모리 탐침, 도구 슬롯) · 검증 16행. **teacher 슬롯**: General · STEM · 도구 = SFT 2862 를 경로만 바꾼 심볼릭 링크 (`$NRL_ROOT/mopd_smoke/teachers/sft_as_*` — 잡음 바닥, 경로 문자열이 달라 teacher 4개로 나뉘고 변환도 4번 = 본 런 기동 조건) · Code·Math = 실제 s2r 228 (음성 대조). 오버라이드 `grpo.max_num_steps=3 grpo.num_prompts_per_step=64 checkpointing.enabled=false env.should_log_nemo_gym_responses=false` (덤프 켬) · W&B 끔 · 새 tag | `runs/launch.sh mopd1 ms1` · 판정 `tools/analyze_mopd_smoke.py` (덤프 → teacher 슬롯별 차이) · main1 `nvidia-smi` 5초 표본 (GPU 별 최대 메모리) · 호스트 RAM 표본 | **PASS = 전부**: ① 3 step rc 0, 드라이버 로그에 teacher 4개 · `opd_teacher_nodes 4` · policy_nodes 2 ② 배치: main1 GPU 8장 = vLLM 4 + teacher 4 (teacher GPU 마다 ≥ 30 GB), sub1 8장 학습 ③ 잡음 슬롯 3개: |평균 차이| ≤ 0.01 nats/토큰, 평균 |차이| ≤ max(0.05, 3 × 평균 |학생 − 롤아웃|) ④ 실제 슬롯: 평균 |차이| ≥ 5 × 잡음 슬롯 최대, 평균 차이 < 0 ⑤ 긴 행 (총 ≥ 48K) 이 덤프에 있고 teacher advantage 가 0 이 아님 · teacher GPU 최대 메모리 < 76 GB ⑥ gen KL < 0.002 · 로그 오차 마스킹 ≤ 1% ⑦ 기록만 (기준 없음): is_oob_ratio (icepop [0.5, 2]) · teacher_logprob_time · 학생 logprob 패스 시간 · policy_training · exposed_generation · step 시간 · main1 호스트 RAM 최대 · OPD 지표 이름 (W&B allowlist 정리용). 분석기 음성 대조 (합성 덤프): 위치 한 칸 밀림 · 실제 슬롯 = 학생 · 긴 행 없음 · 모르는 agent · 긴 행 점수 0 → 5/5 FAIL, 정상 PASS | **MS1b (2026-11-09 17:14~17:42, commit `d2e0e90b5`, `EXTRA_ENV=UV_OFFLINE=1` — 첫 시도 ms1 은 uv 504 로 GPU 전 rc=2)**: ① 3/3 step 완료 ("Async GRPO training complete!"), teacher 4개 · `policy_nodes:2 + opd_teacher_nodes:4 = total_nodes:6` — **단 rc=1**: 완료 뒤 드라이버가 Ray 에 다시 접속하다 `core_worker_process.cc:89 unexpected system state` 로 죽음 (이전 런의 graceful shutdown 실패는 rc 0 이었다, OPD 에서 새로 생김) ② 배치 PASS — main1 GPU 4~7 = 생성 vLLM (17:16:10 동시 기동, 67~77 GiB), GPU 0~3 = teacher 4개 (17:19:44~17:21:12 teacher 초기화와 같은 간격으로 하나씩 30.7 GiB), sub1 8장 = 학습 (`ms1_layout.jsonl` 152 표본) ③ 잡음 슬롯 평균 차이 −0.0019 · −0.0012 · −0.0017, 평균 |차이| 0.040 · 0.039 · 0.030 (학생−롤아웃 0.025 · 0.027 · 0.019) PASS ④ 실제 슬롯 (s2r 228) 평균 차이 −0.278 · 평균 |차이| 0.514 = 잡음 최대의 12.7배, clip 0.5% PASS ⑤ 긴 행 총 57,324 · 56,921 토큰을 teacher 가 점수 냄 (생성 토큰 499/500 · 416/418 비0), teacher GPU 최대 58.5 GiB (긴 행 슬롯) · 나머지 50.6 GiB PASS ⑥ gen KL 0.00186 · 0.00174 · 0.00188 · 마스킹 0 PASS ⑦ is_oob_ratio (icepop [0.5, 2]) 0.00005 · teacher_logprob_time 8.7 · 9.1 · 2.2 s · 학생 logprob 패스 60.6 · 13.9 · 10.0 s vs 학습 54.8 · 35.9 · 25.8 s (step 2~3 기준 학습의 39%) · weight_sync 28.4~28.9 s · teacher 초기화 419 s (변환 4개) · main1 호스트 RAM 최대 288 GiB · 생성 평균 1.1만 토큰 · 잘림 20~27% · grad_norm 4.9~6.8 (clip 1.0) · OPD 지표 `train/on_policy_distillation/{teacher_student_logprob_gap_mean,adv_mean,adv_std}` · `train/teacher_logprob_time` · `timing/setup/teacher_init_time_s`. 덤프 판정 `tools/analyze_mopd_smoke.py` PASS. **적대적 검토 (Opus xhigh, 2026-11-09): 보완 필요** — 절차: 검토 완료 전에 기동했다 (사용자 지시 "스모크 테스트 진행해" 17:10, 검토 결과 17:45). B1 판정기가 첫 커밋에서 실제 덤프(배치 차원 1)를 못 읽었고 합성 대조가 평평한 형식이라 운영 경로를 시험하지 않았다 · sample_loss_mask 0 행을 섞었다 → 수정, 실제 형식 대조로 재시험. H1 ⑤ "긴 행 advantage ≠ 0" 은 teacher logprob 0 이어도 통과 → 긴 행마다 잡음 기준. H2 ④ 상한 없음 (균등 분포 teacher 가 PASS) → 실제 슬롯 clip ≤ 5% · 평균 ≥ −1. H3 ② `sample_gpu_layout.py` 는 nvidia-smi 호스트 pid 와 Ray 컨테이너 pid 가 달라 역할을 못 가린다 → `runs/check_placement.py` (배치 그룹 bundle → 노드), 학습 노드가 main1 · sub1 어느 쪽이어도 허용. M1 긴 행 57.3K < 최악 65,536 → 외삽 (59.6 − 31.5) × 65,536 / 57,324 + 31.5 ≈ 63.6 GB < 76. M2 실제 General · STEM · 도구 teacher 는 스모크에서 적재되지 않음 → 본 런 첫 step 게이트 MS2. M3 rc=1 은 실제 결함 — ① FAIL, 원인 미확정 (teacher 정리를 `TeacherWorkerGroup.__del__` 에 맡기는 것이 후보, `problem-diagnosis` 필요). **보완 후 재판정**: ③ 잡음 슬롯 평균 |차이| 0.040 [0.036, 0.044] · 0.039 [0.036, 0.042] · 0.030 [0.024, 0.035] PASS · ④ 실제 0.514 [0.497, 0.534] = 12.8배, clip 0.53%, 평균 −0.278 PASS · ⑤ 긴 행 두 개 각각 평균 |차이| 0.030 · 0.013 (잡음 수준) PASS · ② 로그 증거 PASS (`MegatronPolicyWorker[rank=0..7] ip=10.0.37.2` = sub1, vLLM · teacher 는 드라이버 노드 main1) · ⑥ PASS · **① FAIL (rc=1)**. 대조 (실제 덤프 형식, `$CLAUDE_JOB_DIR/tmp/review_ms1/`): 위치 밀림 · 실제 = 학생 · 긴 행 없음 · 모르는 agent · 긴 행 점수 0 · 긴 행 teacher 0 희석 · 균등 분포 teacher → 8/8 FAIL, 정상 PASS |
| MS2 | MOPD1 본 런 첫 step 게이트 (`mopd1/s1`, 결정 55 · MS1 검토 M2): 실제 teacher 넷 (General 640 · STEM 735 · Code·Math 228 · 도구 400) 이 처음 적재되는 자리다. 본 런에 덤프를 켠다 (`env.should_log_nemo_gym_responses=false`, 1,024 행 step 당 약 2.2 GB · 기록 약 36 s — 512 행 1.08 GB · 18 s 의 2배, 결정 56) | `runs/check_placement.py --actors-out <s1_teacher_actors.json>` (step 1 중) · `tools/analyze_mopd_smoke.py` (step 1~3 덤프) `--routing <mopd_blends>/mopd1_v1.routing.yaml --require-long 0 --real-slot <teacher 넷> --slot-floor <General>=0.044 --slot-floor <STEM>=0.042 --slot-floor <도구>=0.035 --slot-floor <Code·Math>=0.044 --driver-log mopd1/s1.log --teacher-actors <s1_teacher_actors.json>` · TensorBoard. 기동은 `EXTRA_ENV="UV_OFFLINE=1 RAY_DEDUP_LOGS_ALLOW_REGEX=successfully.loaded.checkpoint"` — 경로에 숫자가 있으면 Ray 줄 중복 제거가 teacher 적재 줄을 묶어 지운다 (검토 4차 M1, ray_logging `__init__.py:284`). 기동 전: teacher 변환 캐시가 없거나 HF safetensors 보다 새것 (검토 4차 L4, 11-09 18:25 확인 — General·STEM·도구 없음, Code·Math 11-09 17:23 > HF 10-24) | **FAIL 이 하나라도 있으면 멈춘다**: ① 배치 — 학습 한 노드 8 · 생성 다른 노드 4 · teacher 4 × 1 이 생성 노드 (학습 노드는 main1 · sub1 어느 쪽이든) ② teacher 신원 — step 1 에 저장한 teacher 액터 (이름 · pid · 노드) 를 드라이버 로그의 실제 적재 줄 `successfully loaded checkpoint from …mcore_ckpt/model_<경로>` 와 (노드, pid) 로 맞춰, teacher 마다 적재 경로 집합 = {routing 경로} (teacher 끼리 뒤바뀜 · 학생 경로 적재를 잡는다, 검토 2·3차 M3). **신원 증거 부족 (줄·액터 미결합) 도 FAIL = 멈춘다** (검토 3차). 액터는 routing 의 서로 다른 경로를 빠짐·중복 없이 덮어야 하고, MS2 방식은 신원 인자가 필수다 (검토 4차 L1·L2) ③ teacher 손상 — 학생 확신 토큰 (prev ≥ −0.01) 중 차이 < −1 비율 ≤ 5% (s2r 228 실측 0.03% · 균등 분포 ≈ 100%) · clip 경계 비율 ≤ 50% (검토 2차 M1) ④ routing 슬롯 전부 지정 · routing 에 없는 agent 0 · teacher 슬롯마다 생성 토큰 ≥ 2,000 · sample_loss_mask 0 행 ≤ 1% (3 step 합) ⑤ gen KL step 마다 < 0.002 (R1 기준) · OOM · teacher 크래시 없음. **증거 부족 (멈추지 않고 보고)**: 슬롯별 평균 |차이| 부트스트랩 95% 하한 ≤ 그 슬롯 잡음 바닥 (MS1 잡음 슬롯 95% 상한 — 검토 2차 M2) = SFT 와 구분 안 됨. ② 신원이 맞으면 SFT 에 가까운 정상 teacher 로 보고 학습을 잇되 보고한다 (도구 teacher 가 후보: 학습 중 엔트로피 변화 −18%). 전체 평균 차이는 기록만 한다. 음성 대조 (실제 ms1b 덤프 변형, `$CLAUDE_JOB_DIR/tmp/ms1_ctl/`): 위치 밀림 · 실제 = 학생 · 균등 분포 teacher · 실제 행 30% 손상 · 긴 행 teacher 0 · 모르는 agent · 긴 행 없음 → 7/7 FAIL; 슬롯 하나 누락 → FAIL; 신원 (ms1b 로그 변형, `$CLAUDE_JOB_DIR/tmp/ms2_id/`): 정상 · 캐시 줄 추가 · 노드 뒤집힘 → PASS, pid 뒤바꿈 · 다른 경로 적재 · 학생 경로 추가 적재 · 적재 줄 하나 삭제 · 노드 불일치 · 액터 없음 · 액터 파일 없음 · 액터 3개 · 중복 액터 · MS2 신원 인자 누락 → 10/10 FAIL, `[rank=N]` 접두 → PASS (로그 접두 pid = 상태 API pid, 두 노드 실측); MS1 덤프를 MS2 방식으로 보면 SFT 슬롯 셋 = 증거 부족, s2r 228 통과 | **`mopd1/s3` (학생 282 · 512 × 120, 결정 58) PASS (2026-11-10 15:32, 출력 `results/alpha/mopd1/s3_ms2.txt`)**: 배치 PASS (학습 sub1 8 · 생성 main1 4 · teacher main1 4×1) · 신원 4/4 (적재 줄 5줄, `extra_env` 에 dedup 허용) · 확신 토큰 손상 0.079~0.284% · clip 0.16~0.76% · 마스킹 0/1,536 · 모르는 agent 0 · gen KL 0.0011 ×3 · 크래시 0 · 슬롯 |차이| [95% CI]: Code·Math 0.270 [0.267, 0.274] · General 0.366 [0.350, 0.381] · STEM 0.386 [0.377, 0.396] · 도구 0.235 [0.215, 0.256] — 전부 잡음 바닥 위. 앞선 판정 — **`mopd1/s2` (시퀀스 평균 손실, 결정 57) PASS (2026-11-09 23:44, 출력 `results/alpha/mopd1/s2_ms2.txt`)**: 배치 PASS (학습 main1 · 생성·teacher sub1) · 신원 4/4 · 확신 토큰 손상 0.029~0.102% · clip 0.32~0.53% · 마스킹 0 · 모르는 agent 0 · gen KL 0.0018 ×3 · 크래시 0 · 슬롯 |차이| Code·Math 0.497 [0.492, 0.501] · General 0.445 [0.428, 0.463] · STEM 0.482 [0.473, 0.492] · 도구 0.228 [0.208, 0.251]. 손실 전환 확인: step 1 train/loss 0.2762 = 덤프 시퀀스 평균 예상 0.2745 (토큰 평균 예상 0.2647); s1 은 0.2671 = 토큰 평균 예상 0.2653 — 두 런 모두 맞는 식과 +0.0017 (같은 상수 차), grad_norm 6.6 → 4.75. 앞선 판정 — `mopd1/s1` (토큰 평균) **PASS (2026-11-09 21:03, `results/alpha/mopd1/s1_ms2.txt`)**: ① 배치 PASS — 학습 main1 8 · 생성 sub1 4 · teacher sub1 4×1 ② 신원 4/4 — 적재 줄 5줄 전부 남음 (`runs.log` `extra_env=UV_OFFLINE=1 RAY_DEDUP_LOGS_ALLOW_REGEX=successfully.loaded.checkpoint`), pid·노드·경로 일치 ③ 확신 토큰 손상 0.030~0.102% · clip 0.33~0.58% ④ 미지정 슬롯 0 · 모르는 agent 0 · 슬롯 토큰 최소 140,090 · 마스킹 0/3,072 ⑤ gen KL 0.0018 · 0.0018 · 0.0018 · 크래시 0. 슬롯 평균 |차이| [95% CI]: Code·Math 0.499 [0.494, 0.503] · General 0.458 [0.443, 0.473] · STEM 0.483 [0.474, 0.492] · 도구 0.235 [0.217, 0.255] — 전부 잡음 바닥 0.035~0.044 위 (증거 부족 없음). 행 비율 29.5 / 30.4 / 29.7 / 10.4% (블렌드 30 / 30 / 30 / 10). 게이트 이력: 적대적 검토 4차 "보완 필요" (신원 거짓 PASS 경로 없음 · 거짓 FAIL 1 · 운영 구멍 2) → 반영 `bd27e7b4d` → 재확인 신뢰 가능 (11-09 18:35). 남은 LOW: 분석기는 기동 환경변수를 스스로 확인하지 못한다 (빠뜨리면 거짓 FAIL 쪽) → 판정 기록에 `runs.log` 의 `extra_env=` 줄을 인용한다 · 레시피 agent ⊆ routing 은 검사기가 보지 않는다 (어긋나면 거짓 FAIL 쪽) |

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

**G3 상세 (2026-10-07, main1·sub1, 32K·CP4·생성 최대 8K, 16 프롬프트 × 4)**: 스모크에서 레시피·코드 결함 5건을 찾아 고쳤다 (`archive/KNOWN_ISSUES_2026-08_10-07.md` 2026-10-07).
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
